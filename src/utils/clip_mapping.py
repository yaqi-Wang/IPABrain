"""Data/provenance utilities for perception-trained brain-to-CLIP heads."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.scheme1 import (
    PROTOCOL, FMRIStandardizer, check_prepared_config, file_sha256,
    data_directory, load_prepared, load_scheme1_config, resolve_path, validate_tag,
)
from src.models.brainvae_factory import build_brainvae
from src.utils.config import brainvae_architecture, load_yaml
from src.utils.mapping_precision import predict_mapping
from src.utils.mapping_objective import mapping_objective
from src.utils.scheme1_runs import training_directory

MAPPING_PROTOCOL = "frozen_brain_mean_to_clip_v1"


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def mapping_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".mapping.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another mapping stage is running in {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_head_source_config(path: str | Path, root: Path):
    """Shared frozen-source/data contract; head-specific objectives are validated separately."""
    config = load_yaml(resolve_path(path, root))
    validate_tag(config["run_name"])
    brain_config, paths = load_scheme1_config(config["brainvae_config"], root)
    original = brain_config["model"].get("architecture") == "ppbe"
    if not original and brain_config["model"].get("semantic_source", "sample") != "mean":
        raise ValueError("These mapping experiments require a mean-supervised BrainVAE config")
    if config.get("brainvae_source", "local") not in {"local", "ppbe_external", "neuroflow_external"}:
        raise ValueError("Unknown BrainVAE source kind")
    if config.get("brainvae_source") == "ppbe_external" and not original:
        raise ValueError("External PPBE source requires the original PPBE architecture")
    for key in ("latent_tokens", "embed_dim"):
        if config["model"][key] != brain_config["model"][key]:
            raise ValueError(f"Mapping / BrainVAE dimension mismatch: {key}")
    if config.get("mean_cache_run"):
        validate_tag(config["mean_cache_run"])
    if config.get('mean_cache_run') and config.get('mean_cache_directory'):
        raise ValueError('Choose mean_cache_run or mean_cache_directory, not both')
    train = config["training"]
    for key in ("epochs", "batch_size", "encode_batch_size", "checkpoint_every_steps"):
        if int(train[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if int(train.get("num_workers", 0)) < 0:
        raise ValueError("num_workers must be nonnegative")
    if int(train.get('prefetch_factor',2)) <= 0:
        raise ValueError('prefetch_factor must be positive')
    if train.get('persistent_workers',False) and int(train.get('num_workers',0)) == 0:
        raise ValueError('persistent_workers requires positive num_workers')
    if int(train.get("micro_batch_size", 0)) < 0:
        raise ValueError("micro_batch_size must be nonnegative")
    if train.get("mixed_precision", "no") not in {"no", "fp16"}:
        raise ValueError("mixed_precision must be no or fp16")
    for key in ("learning_rate", "weight_decay"):
        if not math.isfinite(float(train[key])) or float(train[key]) < 0:
            raise ValueError(f"Invalid training.{key}")
    if not float(train["learning_rate"]) > 0:
        raise ValueError("learning_rate must be positive")
    validation = config["validation"]
    if (int(validation["batch_size"]) <= 0 or int(validation["patience"]) < 0
            or not math.isfinite(float(validation["min_delta"])) or float(validation["min_delta"]) < 0):
        raise ValueError("Invalid mapping validation settings")
    config["evaluation"] = {**brain_config["evaluation"], **config["evaluation"]}
    if config["evaluation"]["aggregation"] != "mean_fmri":
        raise ValueError("Mapping training and default inference use mean_fmri")
    return config, brain_config, paths


def load_mapping_config(path: str | Path, root: Path):
    config, brain_config, paths = load_head_source_config(path, root)
    if config["model"]["type"] not in {"meanbridge", "tokenformer", "adatoken", "tokenformer_dual", "adatoken_refinement"}:
        raise ValueError("Unknown mapping head type")
    if config["validation"]["selection_metric"] != "token_mse":
        raise ValueError("Select mapping heads by internal-validation raw token_mse (min)")
    mapping_objective(config)
    return config, brain_config, paths


def load_brain_source(config: dict, brain_config: dict, root: Path, subject: int,
                      checkpoint_override: str = "") -> dict:
    if config.get("brainvae_source") == "neuroflow_external":
        from src.utils.neuroflow_checkpoint import load_external_neuroflow_source
        return load_external_neuroflow_source(config, brain_config, root, subject, checkpoint_override)
    if config.get("brainvae_source") == "ppbe_external":
        return load_external_ppbe_source(config, brain_config, root, subject, checkpoint_override)
    requested = checkpoint_override or config.get("brainvae_checkpoint", "")
    checkpoint_path = (resolve_path(requested, root) if requested else
                       training_directory(brain_config, root, subject) / "checkpoints/best.pt")
    if not (checkpoint_path.parent.parent / "training_complete.json").is_file():
        raise RuntimeError(f"Source BrainVAE training has not completed: {checkpoint_path.parent.parent}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    protocol = checkpoint.get("protocol_metadata", {})
    if protocol.get("protocol") != PROTOCOL or protocol.get("subject") != subject:
        raise ValueError("Source BrainVAE protocol / subject mismatch")
    checkpoint_config = checkpoint["config"]
    if "architecture" in checkpoint_config:
        checkpoint_config = {**checkpoint_config, "architecture": brainvae_architecture(checkpoint_config)}
    if checkpoint_config != brain_config["model"]:
        raise ValueError("Source BrainVAE model/loss config mismatch; use the checkpoint trained for this method")
    prepared = Path(protocol["prepared_data"])
    manifest, split, prepared_normalizer = load_prepared(prepared)
    check_prepared_config(brain_config, manifest)
    if manifest["subject"] != subject or manifest["data_fingerprint"] != protocol["data_fingerprint"]:
        raise ValueError("Source BrainVAE prepared-data fingerprint mismatch")
    preprocessing_sha = manifest["artifact_sha256"]["preprocessing.npz"]
    if protocol.get("preprocessing_sha256") != preprocessing_sha:
        raise ValueError("Source BrainVAE preprocessing mismatch")
    for key, value in split.items():
        if key not in checkpoint["split"] or not np.array_equal(checkpoint["split"][key].numpy(), value):
            raise ValueError(f"Source BrainVAE image split mismatch: {key}")
    normalizer = FMRIStandardizer.from_checkpoint(checkpoint["preprocessing"])
    if (not np.array_equal(normalizer.mean, prepared_normalizer.mean)
            or not np.array_equal(normalizer.std, prepared_normalizer.std)
            or normalizer.nsd_scale != prepared_normalizer.nsd_scale
            or normalizer.common_divisor != prepared_normalizer.common_divisor
            or normalizer.normalization != prepared_normalizer.normalization):
        raise ValueError("Source BrainVAE normalization differs from prepared data")
    identity = {"protocol": MAPPING_PROTOCOL, "subject": subject,
                "brainvae_checkpoint": str(checkpoint_path.resolve()),
                "brainvae_checkpoint_sha256": file_sha256(checkpoint_path),
                "brainvae_model_config": checkpoint["config"],
                "data_fingerprint": manifest["data_fingerprint"],
                "preprocessing_sha256": preprocessing_sha,
                "source_debug_training": bool(protocol.get("debug_run", False))}
    return {"identity": identity, "weights": checkpoint["model"], "manifest": manifest,
            "split": split, "normalizer": normalizer, "prepared": prepared}


def load_external_ppbe_source(config: dict, brain_config: dict, root: Path, subject: int,
                                 checkpoint_override: str = "") -> dict:
    from src.utils.ppbe_checkpoint import (
        external_model_config, read_ppbe_checkpoint, validate_ppbe_weights,
    )
    specification = config.get("external_source", {})
    if specification.get("subject") != subject:
        raise ValueError("Declare the author checkpoint's matching subject in external_source.subject")
    requested = checkpoint_override or config.get("brainvae_checkpoint", "")
    if not requested:
        raise ValueError("Set brainvae_checkpoint to the downloaded Stage1 BrainVAE")
    checkpoint_path = resolve_path(requested, root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Author Stage1 BrainVAE not downloaded: {checkpoint_path}")
    prepared = data_directory(brain_config, root, subject)
    manifest, split, normalizer = load_prepared(prepared)
    check_prepared_config(brain_config, manifest)
    if manifest["subject"] != subject or normalizer.normalization != "beta_scale":
        raise ValueError("Author BrainVAE requires matching-subject beta_scale prepared data")
    if normalizer.common_divisor != 2000 or normalizer.nsd_scale != 1:
        raise ValueError("The published PPBE input uses beta/2000; do not apply source-train z-scores")
    if manifest.get("source_sample_check", {}).get("status") != "verified":
        raise ValueError("Verify raw NSD beta scale and full XYZ voxel order before loading author weights")
    if manifest.get("voxel_order") != "native_XYZ_mask_C_order":
        raise ValueError("Unexpected author-input voxel ordering")
    weights, metadata = read_ppbe_checkpoint(checkpoint_path)
    if "subject" in metadata and int(metadata["subject"]) != subject:
        raise ValueError("The checkpoint's subject differs from the declared author source")
    config_path = specification.get("author_model_config", "")
    author_config = resolve_path(config_path, root) if config_path else None
    model_config = external_model_config(weights, brain_config["model"], author_config)
    if any(model_config[key] != config["model"][key] for key in ("latent_tokens", "embed_dim")):
        raise ValueError("Author posterior dimensions differ from the requested CLIP mapping head")
    validate_ppbe_weights(weights, model_config)
    identity = {"protocol": MAPPING_PROTOCOL, "source_kind": "ppbe_external", "subject": subject,
                "brainvae_checkpoint": str(checkpoint_path.resolve()),
                "brainvae_checkpoint_sha256": file_sha256(checkpoint_path),
                "brainvae_model_config": model_config, "data_fingerprint": manifest["data_fingerprint"],
                "preprocessing_sha256": manifest["artifact_sha256"]["preprocessing.npz"],
                "source_debug_training": False, "external_source": specification,
                "author_checkpoint_metadata": metadata,
                "author_model_config_sha256": file_sha256(author_config) if author_config else None,
                "encoder_training_data_provenance": specification.get("training_data_provenance", "unknown")}
    return {"identity": identity, "weights": weights, "manifest": manifest, "split": split,
            "normalizer": normalizer, "prepared": prepared}


def frozen_brain(source: dict, device: torch.device):
    model = build_brainvae(source["identity"]["brainvae_model_config"])
    model.load_state_dict(source["weights"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def source_nsd_dataset(source, nsd_root, subject, split, image_indices=None, **kwargs):
    """Use encoder-specific inputs while retaining existing IDs, targets and split."""
    if source['identity'].get('source_kind') == 'neuroflow_external':
        from src.data.neuroflow import NeuroFlowNSDDataset
        return NeuroFlowNSDDataset(nsd_root, subject, split, source['normalizer'], image_indices,
                                   session_statistics=source['neuroflow_statistics'], **kwargs)
    from src.data.scheme1 import Scheme1NSDDataset
    return Scheme1NSDDataset(nsd_root, subject, split, source['normalizer'], image_indices, **kwargs)


def mean_cache_specification(identity, count, shape):
    return {**identity, 'representation':'posterior_mean_before_mapping', 'aggregation':'mean_fmri',
            'shape':[count,*shape], 'dtype':'float32'}


def verified_completed_mean_cache(directory, identity, count, shape):
    """Read-only cache lookup for dry runs; never allocate or touch a shared cache."""
    path, state_path = directory/'posterior_means.npy', directory/'cache_state.json'
    if not state_path.is_file():
        if path.exists():
            raise ValueError('Frozen-mean cache has no provenance')
        return None
    state = json.loads(state_path.read_text())
    if state['identity'] != mean_cache_specification(identity,count,shape):
        raise ValueError('Frozen-mean cache belongs to a different encoder / data protocol')
    if not 0 <= state['next_index'] <= count:
        raise ValueError('Invalid frozen-mean cache progress')
    if state['next_index'] != count:
        return None
    values = np.load(path,mmap_mode='r',allow_pickle=False)
    if values.shape != (count,*shape) or values.dtype != np.float32:
        raise ValueError('Frozen-mean cache shape/dtype mismatch')
    return path


@torch.no_grad()
def cache_source_means(model_factory, dataset, directory: Path, identity: dict,
                       shape: tuple[int, int], batch_size: int, device: torch.device) -> Path:
    """Stream all source-pool means to disk, resuming only the same frozen encoder.

    No image feature enters the encoder. Rows retain original source image order;
    the stored train/validation indices are applied later by CachedMappingDataset.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path, state_path = directory / "posterior_means.npy", directory / "cache_state.json"
    specification = mean_cache_specification(identity, len(dataset), shape)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["identity"] != specification:
            raise ValueError("Frozen-mean cache belongs to a different encoder / data protocol")
    else:
        if path.exists():
            raise ValueError("Frozen-mean cache has no provenance; use a separate run directory")
        state = {"identity": specification, "next_index": 0}
        atomic_json(state, state_path)
    start = int(state["next_index"])
    if not 0 <= start <= len(dataset):
        raise ValueError("Invalid frozen-mean cache progress")
    if path.exists():
        values = np.lib.format.open_memmap(path, mode="r+")
        if list(values.shape) != specification["shape"] or values.dtype != np.float32:
            raise ValueError("Frozen-mean cache shape/dtype mismatch")
    else:
        if start:
            raise FileNotFoundError("Interrupted frozen-mean cache lost its data file")
        values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                         shape=tuple(specification["shape"]))
    if start == len(dataset):
        print(f"[复用冻结均值] {path}", flush=True)
        return path
    model = model_factory().eval().requires_grad_(False)
    loader = DataLoader(Subset(dataset, range(start, len(dataset))), batch_size=batch_size, shuffle=False)
    for batch in loader:
        means = model.encode(batch["fmri"].float().to(device)).mode().float().cpu().numpy()
        if means.shape != (len(batch["fmri"]), *shape) or not np.isfinite(means).all():
            raise FloatingPointError("Invalid frozen BrainVAE means")
        stop = start + len(means)
        values[start:stop] = means
        values.flush()
        state["next_index"] = stop
        atomic_json(state, state_path)
        start = stop
        print(f"[缓存冻结均值] {stop}/{len(dataset)}", flush=True)
    del values, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return path


class CachedMappingDataset(Dataset):
    def __init__(self, mean_path: Path, clip_path: Path, indices: np.ndarray, image_ids=None):
        self.means = np.load(mean_path, mmap_mode="r", allow_pickle=False)
        self.clip = np.load(clip_path, mmap_mode="r", allow_pickle=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.image_ids = None if image_ids is None else np.asarray(image_ids, dtype=np.int64)
        if self.image_ids is not None and len(self.image_ids) != len(self.means):
            raise ValueError("Mapping source image IDs do not align with cached rows")
        if self.means.ndim != 3 or self.clip.shape != self.means.shape:
            raise ValueError("Frozen means and raw CLIP targets must have identical (images,tokens,channels) shapes")
        if (self.indices.ndim != 1 or len(self.indices) == 0
                or len(np.unique(self.indices)) != len(self.indices)
                or np.any(self.indices < 0) or np.any(self.indices >= len(self.means))):
            raise ValueError("Invalid mapping image subset")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = int(self.indices[index])
        item = {"mean": torch.from_numpy(np.array(self.means[row], dtype=np.float32, copy=True)),
                "clip": torch.from_numpy(np.array(self.clip[row], dtype=np.float32, copy=True)),
                "source_row": row}
        if self.image_ids is not None:
            item["image_id"] = int(self.image_ids[row])
        return item


def training_clip_mean(clip: np.ndarray, train_indices: np.ndarray, chunk_size: int = 16) -> torch.Tensor:
    """Constant-token control uses only internal training image identities."""
    total = np.zeros(clip.shape[1:], dtype=np.float64)
    for start in range(0, len(train_indices), chunk_size):
        block = np.asarray(clip[train_indices[start:start + chunk_size]], dtype=np.float32)
        if not np.isfinite(block).all():
            raise FloatingPointError("Non-finite training CLIP targets")
        total += block.sum(axis=0, dtype=np.float64)
    return torch.from_numpy((total / len(train_indices)).astype(np.float32))


def token_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred, true = prediction.float(), target.float()
    flat_pred, flat_true = pred.flatten(1), true.flatten(1)
    matched = F.cosine_similarity(flat_pred, flat_true, dim=-1)
    values = {"token_mse": float((pred - true).square().flatten(1).mean(1).mean()),
              "latent_cosine": float(matched.mean()),
              "token_cosine": float(F.cosine_similarity(pred, true, dim=-1).mean()),
              "norm_ratio": float((flat_pred.norm(dim=1) / flat_true.norm(dim=1).clamp_min(1e-8)).mean())}
    if len(pred) > 1:
        wrong = F.cosine_similarity(flat_pred, flat_true.roll(1, dims=0), dim=-1)
        values["batch_mismatched_cosine"] = float(wrong.mean())
        values["batch_match_minus_mismatch"] = float((matched - wrong).mean())
    return values


@torch.no_grad()
def evaluate_mapper(model, loader, device: torch.device, mean_clip: torch.Tensor,
                    max_steps: int = 0, mixed_precision: str = "no",
                    micro_batch_size: int = 0, objective: dict | None = None) -> dict:
    model.eval()
    controls = ("mapped", "identity", "training_mean_clip")
    totals, counts = {key: {} for key in controls}, {key: {} for key in controls}
    n_items = 0
    joint = objective is not None and objective["type"] == "generation_semantic"
    generation_rows, semantic_rows, target_rows = [], [], []
    sem_total, sem_count = 0.0, 0
    contrast_total, contrast_count = 0.0, 0
    for step, batch in enumerate(loader, start=1):
        mean, target = batch["mean"].to(device, non_blocking=True), batch["clip"].to(device, non_blocking=True)
        if joint:
            from src.utils.mapping_objective import generation_semantic_losses
            from src.utils.mapping_precision import mapping_autocast
            with mapping_autocast(mean.device, mixed_precision):
                outputs = model.forward_train(mean)
            mapped = outputs["generation"].float()
            if len(mean) > 1:
                sem_total += float(generation_semantic_losses(outputs, target, objective)["sem_loss"])*len(mean)
                sem_count += len(mean)
            generation_rows.append(mapped.cpu())
            semantic_rows.append(outputs["semantic"].float().cpu())
            target_rows.append(target.cpu())
        else:
            mapped = predict_mapping(model, mean, mixed_precision, micro_batch_size)
            if objective is not None and objective['type'] == 'generation_contrastive' and len(mean) > 1:
                from src.utils.mapping_objective import generation_contrastive_losses
                loss = generation_contrastive_losses(mapped, target, objective)
                contrast_total += float(loss['contrastive_loss']) * len(mean)
                contrast_count += len(mean)
        predictions = (mapped,
                       mean, mean_clip.to(device).unsqueeze(0).expand_as(mean))
        for name, prediction in zip(controls, predictions):
            metrics = token_metrics(prediction, target)
            for key, value in metrics.items():
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite validation metric: {name}.{key}")
                totals[name][key] = totals[name].get(key, 0.0) + value * len(mean)
                counts[name][key] = counts[name].get(key, 0) + len(mean)
        n_items += len(mean)
        if max_steps and step >= max_steps:
            break
    if not n_items:
        raise ValueError("Empty mapping validation loader")
    result = {"n_items": n_items, **{name: {key: value / counts[name][key]
                                          for key, value in totals[name].items()} for name in controls}}
    if joint:
        from src.utils.token_diagnostics import representation_diagnostics
        result.update({"gen_mse": result["mapped"]["token_mse"],
                       "sem_loss": sem_total/sem_count if sem_count else None,
                       "semantic_loss_n_items": sem_count, "objective": objective})
        if n_items > 1:
            result["generation_diagnostics"] = representation_diagnostics(generation_rows, target_rows, device)
            result["semantic_diagnostics"] = representation_diagnostics(semantic_rows, target_rows, device)
    if objective is not None and objective['type'] == 'generation_contrastive':
        result['contrastive_loss'] = contrast_total / contrast_count if contrast_count else None
        result['contrastive_n_items'] = contrast_count
    return result
