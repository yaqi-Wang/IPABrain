"""Frozen BrainVAE -> frozen CLIP mapping head -> frozen SDXL-UnCLIP."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.data.scheme1 import (
    Scheme1ImageryDataset, Scheme1NSDDataset, atomic_torch_save,
    check_source_files, file_sha256, resolve_path, validate_tag,
)
from src.models.clip_mapping import build_clip_mapper
from src.utils.clip_mapping import (
    MAPPING_PROTOCOL, atomic_json, frozen_brain, load_brain_source,
    load_mapping_config, mapping_lock,
)
from src.utils.scheme1_runs import training_directory
from src.utils.mapping_precision import predict_mapping, validate_mapping_precision
from src.utils.mapping_objective import mapping_objective
from src.utils.clip_head_evaluation import head_evaluation_dataset, load_trained_head_source
from decode_brainvae_scheme1 import generate_samples, reconcile_resume_metadata, validate_encoding_metadata
from train_brainvae_scheme1 import choose_device


def validate_mapped_encoding(cached: dict, requested: dict) -> None:
    validate_encoding_metadata(cached, requested)
    keys = ("brainvae_checkpoint_sha256", "mapping_head", "token_representation", "validation_subset_sha256", "mapping_objective")
    changed = [key for key in keys if cached.get(key) != requested.get(key)]
    if cached.get("mapping_precision", "no") != requested.get("mapping_precision", "no"):
        changed.append("mapping_precision")
    if changed:
        raise ValueError(f"Mapped token source differs: {', '.join(changed)}")


@torch.inference_mode()
def encode_mapped_predictions(brain, mapper, dataset, directory: Path, batch_size: int,
                              device, checkpoint_digest: str, max_items: int = 0,
                              resume: bool = False, require_cached: bool = False,
                              mixed_precision: str = "no") -> list[Path]:
    count = min(len(dataset), max_items) if max_items else len(dataset)
    loader = DataLoader(Subset(dataset, range(count)), batch_size=batch_size, shuffle=False)
    token_dir = directory / "tokens"
    if require_cached and not token_dir.is_dir():
        raise FileNotFoundError(f"Finish --encode-only first: missing {token_dir}")
    if not require_cached:
        token_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for number, batch in enumerate(loader):
        records = [dataset.record(index) for index in batch["index"].tolist()]
        path = token_dir / f"chunk{number:05d}.pt"
        expected = (len(records), mapper.latent_tokens, mapper.embed_dim)
        if path.exists():
            if not resume:
                raise FileExistsError(path)
            cached = torch.load(path, map_location="cpu", weights_only=False)
            if (cached["records"] != records or tuple(cached["mean"].shape) != expected
                    or cached.get("mapping_checkpoint_sha256") != checkpoint_digest
                    or cached.get("mapping_precision", "no") != mixed_precision
                    or cached.get("representation") != "mapped_raw_clip_tokens"
                    or not torch.isfinite(cached["mean"]).all()):
                raise ValueError(f"Mapped token cache mismatch: {path}")
        else:
            if require_cached:
                raise FileNotFoundError(f"Missing mapped token chunk: {path}; finish --encode-only first")
            mean = brain.encode(batch["fmri"].float().to(device)).mode()
            mapped = predict_mapping(mapper, mean, mixed_precision).cpu()
            if tuple(mapped.shape) != expected or not torch.isfinite(mapped).all():
                raise FloatingPointError("Invalid mapped CLIP tokens")
            # 'mean' is the existing generator's tensor key, not a claim that
            # the mapped tokens are the original BrainVAE posterior mean.
            atomic_torch_save({"mean": mapped, "records": records,
                               "representation": "mapped_raw_clip_tokens",
                               "mapping_precision": mixed_precision,
                               "mapping_checkpoint_sha256": checkpoint_digest}, path)
        chunks.append(path)
        print(f"[映射编码] {min((number + 1) * batch_size, count)}/{count}", flush=True)
    return chunks


def generate_clip_jobs(jobs, paths, device, unclip, openclip, num_steps, rgb_mode,
                       num_samples, seed, resume):
    """The unchanged UnCLIP generation half, shared by deterministic and sampled tokens."""
    from decode_synthesised_fmri_unclip import load_sdxl_unclip, unclip_decode_batch
    from torchvision.utils import save_image
    engine, suffix, offset = load_sdxl_unclip(paths.generative_models_path, unclip, openclip, device, num_steps)
    target_index = {}
    if paths.raw_nsd_root is not None:
        target_root = paths.raw_nsd_root / 'nsddata/experiments/nsdimagery/rawtargetimages'
        target_index = {path.stem:path for path in target_root.glob('*/*.png')}
    for directory, metadata, chunks in jobs:
        generate_samples(chunks, directory, metadata, engine, suffix, offset,
                         partial(unclip_decode_batch, rgb_mode=rgb_mode), save_image,
                         num_samples, seed, target_index, resume)


def decode(args, config, brain_config, paths, device, checkpoint_path):
    checkpoint, source, protocol = load_trained_head_source(
        checkpoint_path, config, brain_config, ROOT, args.subject, MAPPING_PROTOCOL, args.brainvae_checkpoint)
    if mapping_objective(protocol['config']) != mapping_objective(config):
        raise ValueError('Mapping checkpoint objective differs from requested configuration')
    saved_source = protocol["source_identity"]
    evaluation = config["evaluation"]
    mixed_precision = protocol["config"]["training"].get("mixed_precision", "no")
    validate_mapping_precision(device, mixed_precision)
    batch_size = args.batch_size or int(evaluation["encode_batch_size"])
    num_samples = args.num_samples or int(evaluation["num_samples"])
    num_steps = args.num_steps or int(evaluation["num_steps"])
    seed = args.seed if args.seed is not None else int(evaluation["seed"])
    rgb_mode = args.rgb_mode or evaluation["rgb_mode"]
    unclip = resolve_path(args.unclip_ckpt or evaluation["unclip_checkpoint"], ROOT)
    openclip = resolve_path(args.openclip_path, ROOT) if args.openclip_path else paths.openclip_version
    if min(batch_size, num_samples, num_steps) <= 0:
        raise ValueError("Invalid encoding batch / diffusion sample / step count")
    if not args.encode_only:
        if (config["model"]["latent_tokens"], config["model"]["embed_dim"]) != (256, 1664):
            raise ValueError("UnCLIP requires mapped tokens shaped (N,256,1664)")
        for label, path in (("UnCLIP checkpoint", unclip), ("OpenCLIP checkpoint", openclip)):
            if path is None or not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
        if paths.generative_models_path is None:
            raise ValueError("generative_models_path is required for image reconstruction")
        if device.type != "cuda" and args.device == "auto":
            raise RuntimeError("No GPU for UnCLIP; use --encode-only or explicitly --device cpu")
    brain = frozen_brain(source, device)
    mapper = build_clip_mapper(checkpoint["config"]).to(device).eval().requires_grad_(False)
    mapper.load_state_dict(checkpoint["model"], strict=True)
    del source["weights"]
    partial_suffix = "_partial" if args.max_items else ""
    token_tag = args.token_source_tag or evaluation["token_source_tag"] + partial_suffix
    eval_tag = args.eval_tag or ((token_tag if args.encode_only else evaluation["generation_tag"] + partial_suffix))
    validate_tag(eval_tag)
    validate_tag(token_tag)
    run_dir = checkpoint_path.parent.parent
    eval_root = run_dir / "evaluation" / eval_tag
    token_root = eval_root if args.encode_only else run_dir / "evaluation" / token_tag
    names = ("nsd_test", "vision", "imagery") if args.dataset == "all" else (args.dataset,)
    digest = file_sha256(checkpoint_path)
    jobs = []
    for name in names:
        dataset, subset_metadata = head_evaluation_dataset(
            name, source, paths, args.subject, args.max_items, getattr(args, 'validation_subset', ''))
        directory, encoding_directory = eval_root / name, token_root / name
        metadata = {"protocol": MAPPING_PROTOCOL, "subject": args.subject, "dataset": name,
                    "checkpoint": str(checkpoint_path), "checkpoint_sha256": digest,
                    "brainvae_checkpoint": saved_source["brainvae_checkpoint"],
                    "brainvae_checkpoint_sha256": saved_source["brainvae_checkpoint_sha256"],
                    "mapping_head": config["model"]["type"], "token_representation": "mapped_raw_clip_tokens",
                    "mapping_precision": mixed_precision,
                    "data_fingerprint": protocol["data_fingerprint"], "aggregation": "mean_fmri",
                    "encode_batch_size": batch_size, "num_samples": num_samples, "num_steps": num_steps,
                    "seed": seed, "unclip_checkpoint": str(unclip), "openclip_checkpoint": str(openclip),
                    "rgb_mode": rgb_mode, "stimulus_types": ["natural"] if name in {"nsd_test", "nsd_val"} else ["simple", "complex"],
                    "n_items": min(args.max_items, len(dataset)) if args.max_items else len(dataset),
                    "partial_test": bool(args.max_items) if name != 'nsd_val' else False, "debug_training": protocol["debug_training"],
                    "posterior_output": "mapping_of_posterior_mean", "posterior_variance_conditions_generation": False,
                    "target_usage": "evaluation_only", "normalization": source["normalizer"].description,
                    "raw_nsd_root": str(paths.raw_nsd_root), **subset_metadata}
        if saved_source.get('source_kind') == 'neuroflow_external':
            metadata.update({'brainvae_source': 'neuroflow_external',
                             'normalization': saved_source['encoder_input_normalization'],
                             'encoder_representation': saved_source['encoder_representation'],
                             'neuroflow_preprocessing': saved_source['neuroflow_preprocessing'],
                             'encoder_training_data_provenance': saved_source['encoder_training_data_provenance']})
        if 'objective' in config:
            metadata['mapping_objective'] = mapping_objective(config)
        if token_root != eval_root:
            source_metadata = json.loads((encoding_directory / "metadata.json").read_text(encoding="utf-8"))
            validate_mapped_encoding(source_metadata, metadata)
        if directory.exists():
            if not args.resume:
                raise FileExistsError(f"{directory} exists; use --resume or a separate --eval-tag")
            reconcile_resume_metadata(directory, metadata)
        else:
            directory.mkdir(parents=True, exist_ok=False)
            atomic_json(metadata, directory / "metadata.json")
        chunks = encode_mapped_predictions(brain, mapper, dataset, encoding_directory, batch_size, device,
                                            digest, args.max_items, args.resume or token_root != eval_root,
                                            require_cached=token_root != eval_root,
                                            mixed_precision=mixed_precision)
        atomic_json({**metadata, "encoding_directory": str(encoding_directory),
                     "token_chunks": [str(path) for path in chunks]}, directory / "predictions.json")
        jobs.append((directory, metadata, chunks))
    del brain, mapper, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if args.encode_only:
        print(f"Saved mapped CLIP tokens: {eval_root}", flush=True)
        print("[已完成] stage=mapping_encode", flush=True)
        return
    generate_clip_jobs(jobs, paths, device, unclip, openclip, num_steps, rgb_mode, num_samples, seed, args.resume)
    print(f"Saved mapped-token reconstructions: {eval_root}", flush=True)
    print("[已完成] stage=mapping_decode", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--checkpoint", default="", help="Mapping best.pt override")
    parser.add_argument("--brainvae-checkpoint", default="", help="Must match the head's frozen source")
    parser.add_argument("--dataset", choices=("nsd_test", "nsd_val", "vision", "imagery", "all"), default="all")
    parser.add_argument("--validation-subset", default="", help="Fixed nsd_val image-ID JSON; never a test subset")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-tag", default="")
    parser.add_argument("--token-source-tag", default="", help="Separate mapped-token cache folder; use the same tag for encoding and generation")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rgb-mode", choices=("standard", "legacy"), default=None)
    parser.add_argument("--unclip-ckpt", default="")
    parser.add_argument("--openclip-path", default="")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.subject <= 0 or args.threads <= 0 or args.max_items < 0:
        parser.error("Invalid subject / thread / item count")
    if args.validation_subset and args.dataset != 'nsd_val':
        parser.error('--validation-subset requires --dataset nsd_val')
    if any(value is not None and value <= 0 for value in (args.batch_size, args.num_samples, args.num_steps)):
        parser.error("Encoding batch / sample / step counts must be positive")
    torch.set_num_threads(args.threads)
    config, brain_config, paths = load_mapping_config(args.config, ROOT)
    device = choose_device(args.device)
    checkpoint = resolve_path(args.checkpoint, ROOT) if args.checkpoint else training_directory(config, ROOT, args.subject) / "checkpoints/best.pt"
    with mapping_lock(checkpoint.parent.parent):
        decode(args, config, brain_config, paths, device, checkpoint)


if __name__ == "__main__":
    main()
