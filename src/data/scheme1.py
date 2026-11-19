"""Leakage-safe data interface for direct BrainVAE cross-state transfer.

The old adapter pipeline is intentionally left unchanged. This interface uses
grouped perception npys, image-level splits, and training-only normalization.
No target image feature is returned by inference datasets.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config import brainvae_architecture, load_config, load_yaml
from src.utils.paths import ProjectPaths

PROTOCOL = "scheme1_perception_only_direct_v1"


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _merge(base: dict, update: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_scheme1_config(path: str | Path, root: Path) -> tuple[dict, ProjectPaths]:
    override = load_yaml(resolve_path(path, root))
    base_path = resolve_path(override.pop("base_config", "configs/brainvae.yaml"), root)
    config = _merge(load_config(base_path), override)
    config.setdefault("data_tag", "v1")
    config["training"].setdefault("seed", 42)
    config["evaluation"].setdefault("encode_batch_size", 8)
    # A paths override must replace the inherited resolved paths mapping too.
    if "paths" in override:
        config["paths_config"] = load_yaml(resolve_path(override["paths"], root))
    normalization = config["preprocessing"]["normalization"]
    architecture = brainvae_architecture(config["model"])
    if "architecture" in config["model"]:
        config["model"]["architecture"] = architecture
    if normalization not in {"source_train", "beta_scale"}:
        raise ValueError("Normalization must be source_train or beta_scale")
    if normalization == "beta_scale" and architecture != "ppbe":
        raise ValueError("beta_scale is reserved for the original PPBE experiment")
    if architecture == "ppbe" and normalization != "beta_scale":
        raise ValueError("Original PPBE requires beta_scale input, without voxel z-scoring")
    return config, ProjectPaths.from_mapping(config["paths_config"], root)


def data_directory(config: dict, root: Path, subject: int) -> Path:
    tag = str(config.get("data_tag", "v1"))
    validate_tag(tag)
    return resolve_path(config["output_root"], root) / f"subj{subject:02d}" / f"data_{tag}"


def validate_tag(tag: str) -> None:
    if not tag or tag in {".", ".."} or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tag):
        raise ValueError("Tags may contain only letters, numbers, underscores and hyphens")


def file_signature(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def digest_json(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_split(image_ids: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(image_ids, dtype=np.int64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("Grouped perception data must have one unique image ID per row")
    if len(ids) < 2 or not 0 < fraction < 1:
        raise ValueError("Need at least two images and 0 < validation fraction < 1")
    n_val = min(len(ids) - 1, max(1, int(round(len(ids) * fraction))))
    order = np.random.default_rng(seed).permutation(len(ids))
    return np.sort(order[n_val:]), np.sort(order[:n_val])


def check_disjoint(source_ids: np.ndarray, test_ids: np.ndarray, imagery_nsd_ids: np.ndarray) -> None:
    for label, ids in (("NSD test", test_ids), ("imagery targets", imagery_nsd_ids)):
        overlap = np.intersect1d(source_ids, ids)
        if len(overlap):
            raise ValueError(f"Source training pool overlaps {label}: {overlap.tolist()}")


class FMRIStandardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, nsd_scale: float, common_divisor: float,
                 normalization: str = "source_train"):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.nsd_scale = float(nsd_scale)
        self.common_divisor = float(common_divisor)
        self.normalization = normalization
        if self.mean.ndim != 1 or self.std.shape != self.mean.shape:
            raise ValueError("Normalization parameters must be one-dimensional voxel arrays")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("Invalid normalization parameters")
        if not np.isfinite([self.nsd_scale, self.common_divisor]).all() or min(self.nsd_scale, self.common_divisor) <= 0:
            raise ValueError("Scaling divisors must be positive and finite")
        if normalization not in {"source_train", "beta_scale"}:
            raise ValueError("Unknown fMRI normalization")
        if normalization == "beta_scale" and (np.any(self.mean != 0) or np.any(self.std != 1)):
            raise ValueError("beta_scale must have identity mean/std; no fitted voxel statistics")

    @classmethod
    def scaled_beta(cls, n_voxels: int, nsd_scale: float, divisor: float):
        return cls(np.zeros(n_voxels, dtype=np.float32), np.ones(n_voxels, dtype=np.float32),
                   nsd_scale, divisor, normalization="beta_scale")

    @property
    def description(self) -> str:
        return ("source_train_frozen" if self.normalization == "source_train" else
                f"beta_divisor_{self.common_divisor:g}_no_zscore")

    @classmethod
    def fit(cls, grouped_fmri: np.ndarray, train_indices: np.ndarray, nsd_scale: float,
            common_divisor: float, epsilon: float = 1e-6, chunk_images: int = 64):
        if grouped_fmri.ndim != 3 or len(train_indices) == 0 or chunk_images <= 0 or epsilon <= 0:
            raise ValueError("Expected nonempty (images, repeats, voxels) training data")
        mean = np.zeros(grouped_fmri.shape[-1], dtype=np.float64)
        m2 = np.zeros_like(mean)
        count = 0
        for start in range(0, len(train_indices), chunk_images):
            block = np.asarray(grouped_fmri[train_indices[start:start + chunk_images]], dtype=np.float64)
            block = block.reshape(-1, block.shape[-1]) * nsd_scale
            if not np.isfinite(block).all():
                raise ValueError("Non-finite training fMRI values")
            block_mean = block.mean(axis=0)
            block_m2 = ((block - block_mean) ** 2).sum(axis=0)
            n = len(block)
            delta = block_mean - mean
            new_count = count + n
            m2 += block_m2 + delta ** 2 * count * n / new_count
            mean += delta * n / new_count
            count = new_count
        std = np.maximum(np.sqrt(m2 / count), epsilon)
        return cls(mean, std, nsd_scale, common_divisor)

    def transform(self, fmri: np.ndarray, source: str) -> np.ndarray:
        values = np.asarray(fmri, dtype=np.float32)
        if values.shape[-1] != len(self.mean) or not np.isfinite(values).all():
            raise ValueError("fMRI voxel count mismatch or non-finite input")
        if source not in {"nsd", "imagery_scaled"}:
            raise ValueError(f"Unknown fMRI source: {source}")
        scale = self.nsd_scale if source == "nsd" else 1.0
        result = (values * scale if self.normalization == "beta_scale" else
                  (values * scale - self.mean) / self.std)
        if not np.isfinite(result).all():
            raise ValueError("Non-finite standardized fMRI")
        return np.array(result, dtype=np.float32, copy=True)

    def arrays(self) -> dict:
        result = {"mean": self.mean, "std": self.std, "nsd_scale": np.asarray(self.nsd_scale),
                  "common_divisor": np.asarray(self.common_divisor)}
        if self.normalization != "source_train":
            result["normalization"] = np.asarray(self.normalization)
        return result

    def checkpoint_state(self) -> dict:
        result = {"mean": torch.from_numpy(self.mean.copy()), "std": torch.from_numpy(self.std.copy()),
                  "nsd_scale": self.nsd_scale, "common_divisor": self.common_divisor}
        if self.normalization != "source_train":
            result["normalization"] = self.normalization
        return result

    @classmethod
    def from_checkpoint(cls, state: dict):
        return cls(state["mean"].cpu().numpy(), state["std"].cpu().numpy(), state["nsd_scale"],
                   state["common_divisor"], state.get("normalization", "source_train"))

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            return cls(data["mean"], data["std"], float(data["nsd_scale"]), float(data["common_divisor"]),
                       str(data["normalization"]) if "normalization" in data else "source_train")


def extract_imagery_xyz(beta_tzyx: np.ndarray, mask_xyz: np.ndarray, divisor: float) -> np.ndarray:
    """Extract voxels in the EXACT XYZ C-order used by perception NIfTI masking."""
    if beta_tzyx.ndim != 4 or tuple(beta_tzyx.shape[1:]) != tuple(mask_xyz.shape[::-1]):
        raise ValueError("Imagery beta axes do not match the native XYZ mask")
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("divisor must be positive")
    coords = np.argwhere(mask_xyz)
    return np.asarray(beta_tzyx[:, coords[:, 2], coords[:, 1], coords[:, 0]], dtype=np.float32) / divisor


def read_imagery_roi_xyz(dataset, mask_xyz: np.ndarray, trial_indices: np.ndarray, divisor: float) -> np.ndarray:
    """Read ROI rows once in time-contiguous HDF5 storage, without loading the brain.

    NSD-Imagery chunks are (720,1,1,1). Re-reading the whole spatial array in
    small trial batches repeatedly reads the same chunks. Instead read all time
    points for each ROI-containing X row, then select trials and voxel columns.
    The output still uses global XYZ mask order, not row iteration order.
    """
    if len(dataset.shape) != 4 or tuple(dataset.shape[1:]) != tuple(mask_xyz.shape[::-1]):
        raise ValueError("Imagery HDF5 axes do not match the native XYZ mask")
    trials = np.asarray(trial_indices, dtype=np.int64)
    if trials.ndim != 1 or np.any(trials < 0) or np.any(trials >= dataset.shape[0]) or not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("Invalid imagery trials or scaling divisor")
    n_voxels = int(mask_xyz.sum())
    positions = np.full(mask_xyz.shape, -1, dtype=np.int32)
    positions[mask_xyz] = np.arange(n_voxels)
    output = np.empty((len(trials), n_voxels), dtype=np.float32)
    for z, y in np.argwhere(mask_xyz.any(axis=0).T):
        x = np.flatnonzero(mask_xyz[:, y, z])
        left, right = int(x[0]), int(x[-1]) + 1
        row = np.asarray(dataset[:, z, y, left:right], dtype=np.float32)
        output[:, positions[x, y, z]] = row[trials][:, x - left] / divisor
    return output


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_prepared(directory: Path) -> tuple[dict, dict, FMRIStandardizer]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Not a scheme 1 data artifact")
    for name, expected in manifest["artifact_sha256"].items():
        if file_sha256(directory / name) != expected:
            raise ValueError(f"Prepared artifact changed: {name}; prepare a new data_tag")
    with np.load(directory / "split.npz", allow_pickle=False) as data:
        split = {key: data[key] for key in data.files}
    check_disjoint(split["source_image_ids"], split["test_image_ids"], split["imagery_nsd_ids"])
    train, val = split["train_indices"], split["val_indices"]
    if not np.array_equal(np.sort(np.concatenate((train, val))), np.arange(len(split["source_image_ids"]))):
        raise ValueError("Training/validation indices must partition the source image pool")
    return manifest, split, FMRIStandardizer.load(directory / "preprocessing.npz")


def check_prepared_config(config: dict, manifest: dict) -> None:
    spec = manifest["input_spec"]
    if (config["preprocessing"] != spec["preprocessing"] or
            config["validation"]["fraction"] != spec["val_fraction"] or
            config["validation"]["split_seed"] != spec["split_seed"]):
        raise ValueError("Current preprocessing / split settings differ from prepared data; choose a new data_tag and prepare again")


def check_source_files(nsd_root: Path, subject: int, manifest: dict, keys: tuple) -> None:
    for key in keys:
        old = manifest["input_spec"]["files"][key]
        # A read-only external cache may use a different source basename from
        # the loader's local symlink. Still verify its resolved path/stat below.
        filename = manifest['input_spec'].get('file_names', {}).get(key, Path(old['path']).name)
        if Path(filename).name != filename:
            raise ValueError('Prepared source filenames must be local basenames')
        current = nsd_root / f"subj{subject:02d}" / filename
        if file_signature(current) != old:
            raise ValueError(f"Source path or file changed: {key}; prepare a new data_tag")


class Scheme1NSDDataset(Dataset):
    def __init__(self, nsd_root: Path, subject: int, split: str, normalizer: FMRIStandardizer,
                 image_indices: np.ndarray | None = None, average_repeats: bool = False,
                 load_targets: bool = True, expected_ids: np.ndarray | None = None):
        if split not in {"train", "test"}:
            raise ValueError("NSD split must be train or test")
        self.subject = subject
        self.split = split
        self.normalizer = normalizer
        directory = nsd_root / f"subj{subject:02d}"
        self.fmri_path = directory / f"nsd_{split}_fmri_scale_sub{subject}.npy"
        self.id_path = directory / f"nsd_{split}_image_id_sub{subject}.npy"
        self.fmri = np.load(self.fmri_path, mmap_mode="r", allow_pickle=False)
        self.image_ids = np.load(self.id_path, allow_pickle=False)
        if self.fmri.ndim != 3 or self.fmri.shape[0] != len(self.image_ids) or self.fmri.shape[-1] != len(normalizer.mean):
            raise ValueError("Expected grouped fMRI aligned with image IDs and normalization voxels")
        if len(np.unique(self.image_ids)) != len(self.image_ids):
            raise ValueError("Grouped fMRI image IDs must be unique")
        if expected_ids is not None and not np.array_equal(self.image_ids, expected_ids):
            raise ValueError("NSD image order changed since data preparation")
        self.indices = np.arange(len(self.image_ids)) if image_indices is None else np.asarray(image_indices, dtype=np.int64)
        if self.indices.ndim != 1 or np.any(self.indices < 0) or np.any(self.indices >= len(self.image_ids)):
            raise ValueError("Invalid image subset indices")
        self.average_repeats = average_repeats
        self.repeats = self.fmri.shape[1]
        self.clip = None
        if load_targets:
            self.clip = np.load(directory / f"nsd_sdxl_clip_{split}_sub{subject}.npy", mmap_mode="r", allow_pickle=False)
            if self.clip.ndim != 3 or len(self.clip) != len(self.image_ids):
                raise ValueError("CLIP file must contain one token array per image, in NSD image_id order")

    def __len__(self):
        return len(self.indices) * (1 if self.average_repeats else self.repeats)

    def image_index(self, index: int) -> int:
        return int(self.indices[index if self.average_repeats else index // self.repeats])

    def __getitem__(self, index: int) -> dict:
        row = self.image_index(index)
        values = self.fmri[row].mean(axis=0) if self.average_repeats else self.fmri[row, index % self.repeats]
        item = {"fmri": torch.from_numpy(self.normalizer.transform(values, "nsd")),
                "index": index, "image_id": int(self.image_ids[row])}
        if self.clip is not None:
            item["clip_vision"] = torch.from_numpy(np.array(self.clip[row], dtype=np.float32, copy=True))
        return item

    def record(self, index: int) -> dict:
        row = self.image_index(index)
        return {"subject_id": self.subject, "image_id": int(self.image_ids[row]),
                "stimulus_id": f"nsd{int(self.image_ids[row]):05d}", "stimulus_type": "natural",
                "task_type": "nsd_test" if self.split == "test" else "perception_source",
                "n_trials": self.repeats if self.average_repeats else 1,
                "source_row": row,
                "repeat_indices": list(range(self.repeats)) if self.average_repeats else [index % self.repeats]}


class Scheme1ImageryDataset(Dataset):
    def __init__(self, path: Path, subject: int, normalizer: FMRIStandardizer, task: str,
                 average_repeats: bool = True, stimulus_types: tuple = ("simple", "complex")):
        if task not in {"vision", "imagery"}:
            raise ValueError("Select vision or imagery explicitly")
        with np.load(path, allow_pickle=False) as archive:
            self.data = {key: archive[key] for key in archive.files}
        if int(self.data["subject_id"]) != subject or float(self.data["common_divisor"]) != normalizer.common_divisor:
            raise ValueError("Imagery subject or raw beta divisor does not match the checkpoint")
        self.subject = subject
        self.task = task
        self.normalizer = normalizer
        selected = np.flatnonzero((self.data["task_type"] == task) & np.isin(self.data["stimulus_type"], stimulus_types))
        groups: dict[tuple, list[int]] = {}
        for row in selected:
            # Task is part of the key: watching and imagining can never be mixed.
            key = (task, str(self.data["stimulus_type"][row]), str(self.data["stimulus_id"][row]))
            groups.setdefault(key, []).append(int(row))
        self.groups = [rows for key, rows in sorted(groups.items())] if average_repeats else [[int(i)] for i in selected]
        if not self.groups:
            raise ValueError(f"No {task} trials selected")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index: int) -> dict:
        # Average brain responses BEFORE the nonlinear encoder, not predicted tokens.
        values = self.data["fmri_scaled"][self.groups[index]].mean(axis=0)
        return {"fmri": torch.from_numpy(self.normalizer.transform(values, "imagery_scaled")), "index": index}

    def record(self, index: int) -> dict:
        rows = self.groups[index]
        first = rows[0]
        return {"subject_id": self.subject, "stimulus_id": str(self.data["stimulus_id"][first]),
                "stimulus_type": str(self.data["stimulus_type"][first]), "task_type": self.task,
                "n_trials": len(rows), "source_trial_ids": self.data["global_trial_idx"][rows].tolist(),
                "runs": self.data["run"][rows].tolist(), "condition_indices": self.data["condition_index"][rows].tolist()}
