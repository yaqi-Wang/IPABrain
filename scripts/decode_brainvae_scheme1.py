"""Frozen BrainVAE mean -> frozen UnCLIP; NSD test / watching / imagery separately."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.data.scheme1 import (PROTOCOL, FMRIStandardizer, Scheme1ImageryDataset, Scheme1NSDDataset, atomic_torch_save,
                              check_source_files, file_sha256, load_prepared, load_scheme1_config,
                              resolve_path, validate_tag)
from src.models.brainvae_factory import build_brainvae
from train_brainvae_scheme1 import choose_device
from src.utils.scheme1_runs import evaluation_name, training_directory


def reconcile_resume_metadata(directory: Path, metadata: dict) -> None:
    """Reuse posterior means when only generation settings changed before decoding."""
    path = directory / "metadata.json"
    previous = json.loads(path.read_text(encoding="utf-8"))
    changed = sorted(key for key in previous.keys() | metadata.keys()
                     if previous.get(key) != metadata.get(key))
    if not changed:
        return
    details = "; ".join(f"{key}: {previous.get(key)!r} -> {metadata.get(key)!r}" for key in changed)
    generation_keys = {"num_samples", "num_steps", "seed", "unclip_seed", "unclip_checkpoint", "openclip_checkpoint", "rgb_mode"}
    if set(changed) - generation_keys:
        raise ValueError(f"Cached posterior settings differ: {details}. Use a separate --eval-tag for different predictions.")
    if any(directory.glob("sample*/images/*.png")) or any(directory.glob("sample*/manifest.json")):
        raise ValueError(f"Generation settings differ after images were saved: {details}. "
                         "Restore the original settings or use a separate --eval-tag.")
    # Generator paths and diffusion settings never enter BrainVAE encoding.
    # Update their provenance only while no generated output can be mixed in.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    print(f"[复用后验缓存] 尚未生成图片，更新生成配置：{details}", flush=True)


def validate_encoding_metadata(cached: dict, requested: dict) -> None:
    """A separate reconstruction folder can read unchanged frozen posterior means."""
    keys = ("protocol", "subject", "dataset", "checkpoint_sha256", "data_fingerprint", "aggregation",
            "encode_batch_size", "stimulus_types", "n_items", "partial_test", "debug_training",
            "posterior_output", "posterior_variance_conditions_generation", "normalization")
    changed = [key for key in keys if cached.get(key) != requested.get(key)]
    if changed:
        raise ValueError(f"Token source does not match this encoding: {', '.join(changed)}")


@torch.inference_mode()
def encode_predictions(model, dataset, directory: Path, batch_size: int, device,
                       max_items: int = 0, resume: bool = False, require_cached: bool = False) -> list[Path]:
    count = min(len(dataset), max_items) if max_items else len(dataset)
    loader = DataLoader(Subset(dataset, range(count)), batch_size=batch_size, shuffle=False)
    chunks = []
    if require_cached:
        if not (directory / "tokens").is_dir():
            raise FileNotFoundError(f"Missing encoded tokens: {directory / 'tokens'}")
    else:
        (directory / "tokens").mkdir(parents=True, exist_ok=True)
    for number, batch in enumerate(loader):
        records = [dataset.record(index) for index in batch["index"].tolist()]
        path = directory / "tokens" / f"chunk{number:05d}.pt"
        if path.exists():
            if not resume:
                raise FileExistsError(path)
            cached = torch.load(path, map_location="cpu", weights_only=False)
            expected_shape = (len(records), model.latent_tokens, model.embed_dim)
            if cached["records"] != records or tuple(cached["mean"].shape) != expected_shape or not torch.isfinite(cached["mean"]).all():
                raise ValueError(f"Cached prediction mismatch: {path}")
        else:
            if require_cached:
                raise FileNotFoundError(f"Missing encoded chunk: {path}; finish the source encoding first")
            posterior = model.encode(batch["fmri"].float().to(device))
            mean = posterior.mode().float().cpu()
            if not torch.isfinite(mean).all():
                raise FloatingPointError("Non-finite predicted mean")
            atomic_torch_save({"mean": mean, "records": records,
                               "mean_logvar": posterior.logvar.flatten(1).mean(dim=1).float().cpu()}, path)
        chunks.append(path)
        print(f"Encoded/cached {min((number + 1) * batch_size, count)}/{count}", flush=True)
    return chunks


def generate_samples(chunks: list[Path], directory: Path, metadata: dict, engine,
                     vector_suffix, offset_noise: float, decode_fn, save_fn,
                     num_samples: int, base_seed: int, target_index: dict[str, Path], resume: bool) -> None:
    """Each diffusion seed has its own manifest; no best-of-N selection."""
    for sample_index in range(num_samples):
        sample_dir = directory / f"sample{sample_index:02d}"
        images_dir = sample_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        items = []
        item_index = 0
        for chunk in chunks:
            payload = torch.load(chunk, map_location="cpu", weights_only=False)
            for offset, record in enumerate(payload["records"]):
                seed = base_seed + sample_index * 100000 + item_index
                image_path = images_dir / f"item{item_index:05d}_{record['stimulus_id']}.png"
                if image_path.exists() and not resume:
                    raise FileExistsError(image_path)
                if not image_path.exists():
                    print(f"[生成] sample={sample_index + 1}/{num_samples} item={item_index + 1} "
                          f"stimulus={record['stimulus_id']} token={chunk.name}[{offset}] seed={seed}", flush=True)
                    generated = decode_fn(payload["mean"][offset:offset + 1], engine,
                                          vector_suffix, offset_noise, seed=seed)
                    temporary = image_path.with_name(image_path.stem + ".tmp.png")
                    save_fn(generated[0].cpu(), temporary)
                    temporary.replace(image_path)
                # Targets are attached only as evaluation metadata AFTER prediction.
                item = {**record, "item_index": item_index, "sample_index": sample_index,
                        "seed": seed, "path": str(image_path.relative_to(sample_dir)),
                        "token_chunk": str(chunk), "token_row": offset}
                if record["stimulus_id"] in target_index:
                    item["target_image"] = str(target_index[record["stimulus_id"]])
                items.append(item)
                item_index += 1
        level = {"mean_fmri": "stimulus_level", "image_level": "image_level"}.get(metadata["aggregation"], "trial_level")
        manifest = {**metadata, "level": level,
                    "sample_index": sample_index, "n_images": len(items), "items": items}
        (sample_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Generated sample {sample_index + 1}/{num_samples}: {len(items)} images", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/brainvae_scheme1.yaml")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--checkpoint", default="", help="Default: this config's parameter-named best.pt; optional explicit scheme 1 checkpoint")
    parser.add_argument("--dataset", choices=("nsd_test", "vision", "imagery", "all"), default="all")
    parser.add_argument("--aggregation", choices=("mean_fmri", "trial"), default=None)
    parser.add_argument("--prepared-data", default="", help="Override location of the matching prepared-data artifact")
    parser.add_argument("--eval-tag", default="")
    parser.add_argument("--token-source-tag", default="", help="Reuse tokens from this evaluation folder without copying or re-encoding")
    parser.add_argument("--rgb-mode", choices=("standard", "legacy"), default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="Default: evaluation.encode_batch_size; generation uses one item for stable seeds")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--unclip-ckpt", default="")
    parser.add_argument("--openclip-path", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-items", type=int, default=0, help="Debug only; mark evaluation as partial")
    parser.add_argument("--include-conceptual", action="store_true", help="Separate qualitative targets; no unique image ground truth")
    parser.add_argument("--encode-only", action="store_true", help="Save frozen posterior means without loading the image generator")
    parser.add_argument("--resume", action="store_true", help="Validate the same experiment settings, reuse tokens and fill missing images")
    args = parser.parse_args()
    if (args.batch_size is not None and args.batch_size <= 0) or args.threads <= 0 or args.max_items < 0:
        raise ValueError("Invalid batch/thread/item count")
    torch.set_num_threads(args.threads)
    config, paths = load_scheme1_config(args.config, ROOT)
    evaluation = config["evaluation"]
    rgb_mode = args.rgb_mode or evaluation.get("rgb_mode", "standard")
    if rgb_mode not in {"standard", "legacy"}:
        raise ValueError("evaluation.rgb_mode must be standard or legacy")
    args.batch_size = args.batch_size if args.batch_size is not None else int(evaluation.get("encode_batch_size", 8))
    if args.batch_size <= 0:
        raise ValueError("encode_batch_size must be positive")
    num_samples = args.num_samples if args.num_samples is not None else int(evaluation["num_samples"])
    num_steps = args.num_steps if args.num_steps is not None else int(evaluation["num_steps"])
    base_seed = args.seed if args.seed is not None else int(evaluation["seed"])
    aggregation = args.aggregation or evaluation["aggregation"]
    if num_samples <= 0 or num_steps <= 0:
        raise ValueError("num_samples and num_steps must be positive")
    checkpoint_path = resolve_path(args.checkpoint, ROOT) if args.checkpoint else training_directory(config, ROOT, args.subject) / "checkpoints/best.pt"
    if not (checkpoint_path.parent.parent / "training_complete.json").is_file():
        raise RuntimeError("Training has not completed; wait for [已完成] stage=train before encoding or decoding")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    protocol = checkpoint.get("protocol_metadata", {})
    if protocol.get("protocol") != PROTOCOL or protocol.get("subject") != args.subject:
        raise ValueError("Checkpoint is not a matching-subject, perception-only scheme 1 model")
    prepared = resolve_path(args.prepared_data, ROOT) if args.prepared_data else Path(protocol["prepared_data"])
    manifest, split, _ = load_prepared(prepared)
    if manifest["subject"] != args.subject or manifest["data_fingerprint"] != protocol["data_fingerprint"]:
        raise ValueError("Prepared data differs from the training checkpoint")
    normalizer = FMRIStandardizer.from_checkpoint(checkpoint["preprocessing"])
    device = choose_device(args.device)
    unclip = resolve_path(args.unclip_ckpt or evaluation["unclip_checkpoint"], ROOT)
    openclip = resolve_path(args.openclip_path, ROOT) if args.openclip_path else paths.openclip_version
    if not args.encode_only:
        for label, path in (("UnCLIP checkpoint", unclip), ("OpenCLIP checkpoint", openclip)):
            if path is None or not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}; provide its actual path or use --encode-only")
        if paths.generative_models_path is None:
            raise ValueError("generative_models_path is required for image reconstruction")
        if device.type != "cuda" and args.device == "auto":
            raise RuntimeError("No GPU available for full UnCLIP generation; use --encode-only or explicitly --device cpu")
        if (checkpoint["config"]["latent_tokens"], checkpoint["config"]["embed_dim"]) != (256, 1664):
            raise ValueError("UnCLIP requires posterior means shaped (N, 256, 1664)")
    model = build_brainvae(checkpoint["config"]).to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"])
    default_tag = evaluation_name(config, ROOT, aggregation=aggregation, num_samples=num_samples,
                                               num_steps=num_steps, seed=base_seed, encode_batch_size=args.batch_size,
                                               max_items=args.max_items, include_conceptual=args.include_conceptual,
                                               unclip_checkpoint=str(unclip), openclip_checkpoint=str(openclip))
    eval_tag = args.eval_tag or (evaluation.get("generation_tag", default_tag) if not args.encode_only else default_tag)
    validate_tag(eval_tag)
    eval_root = checkpoint_path.parent.parent / "evaluation" / eval_tag
    source_tag = args.token_source_tag or (evaluation.get("token_source_tag", "") if not args.encode_only else "")
    if source_tag:
        validate_tag(source_tag)
    token_root = checkpoint_path.parent.parent / "evaluation" / source_tag if source_tag else eval_root
    names = ["nsd_test", "vision", "imagery"] if args.dataset == "all" else [args.dataset]
    checkpoint_digest = file_sha256(checkpoint_path)
    types = ("simple", "complex", "conceptual") if args.include_conceptual else ("simple", "complex")
    jobs = []
    for name in names:
        if name == "nsd_test":
            check_source_files(paths.nsd_root, args.subject, manifest, ("test_fmri", "test_ids"))
            dataset = Scheme1NSDDataset(paths.nsd_root, args.subject, "test", normalizer,
                                       average_repeats=aggregation == "mean_fmri", load_targets=False,
                                       expected_ids=checkpoint["split"]["test_image_ids"].numpy())
        else:
            dataset = Scheme1ImageryDataset(prepared / "imagery_trials.npz", args.subject, normalizer,
                                            name, average_repeats=aggregation == "mean_fmri", stimulus_types=types)
        directory = eval_root / name
        metadata = {"protocol": PROTOCOL, "subject": args.subject, "dataset": name,
                    "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_digest,
                    "data_fingerprint": manifest["data_fingerprint"], "aggregation": aggregation,
                    "encode_batch_size": args.batch_size, "num_samples": num_samples, "num_steps": num_steps,
                    "seed": base_seed, "unclip_checkpoint": str(unclip), "openclip_checkpoint": str(openclip),
                    "rgb_mode": rgb_mode,
                    "stimulus_types": list(types) if name != "nsd_test" else ["natural"],
                    "n_items": min(args.max_items, len(dataset)) if args.max_items else len(dataset),
                    "partial_test": bool(args.max_items), "debug_training": protocol["debug_run"],
                    "posterior_output": "mean_only", "posterior_variance_conditions_generation": False,
                    "target_usage": "evaluation_only", "normalization": normalizer.description,
                    "raw_nsd_root": str(paths.raw_nsd_root)}
        encoding_directory = token_root / name
        if token_root != eval_root:
            source_metadata = json.loads((encoding_directory / "metadata.json").read_text(encoding="utf-8"))
            validate_encoding_metadata(source_metadata, metadata)
            print(f"[复用后验缓存] {name}: {encoding_directory / 'tokens'}", flush=True)
        if directory.exists():
            if not args.resume:
                raise FileExistsError(f"{directory} exists; use --resume or a new --eval-tag")
            reconcile_resume_metadata(directory, metadata)
        else:
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        chunks = encode_predictions(model, dataset, encoding_directory, args.batch_size, device, args.max_items,
                                    args.resume or token_root != eval_root, require_cached=token_root != eval_root)
        token_paths = [str(p.relative_to(directory)) if token_root == eval_root else str(p) for p in chunks]
        (directory / "predictions.json").write_text(json.dumps({**metadata, "encoding_directory": str(encoding_directory),
                                                               "token_chunks": token_paths},
                                                               indent=2, ensure_ascii=False), encoding="utf-8")
        jobs.append((directory, metadata, chunks))
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if args.encode_only:
        print(f"Saved frozen posterior means: {eval_root}", flush=True)
        print("[已完成] stage=encode", flush=True)
        return
    # Heavy diffusion dependencies and image ground truth are absent from encoding.
    from decode_synthesised_fmri_unclip import load_sdxl_unclip, unclip_decode_batch
    from torchvision.utils import save_image

    engine, suffix, offset = load_sdxl_unclip(paths.generative_models_path, unclip, openclip, device, num_steps)
    target_index = {}
    if paths.raw_nsd_root is not None:
        target_root = paths.raw_nsd_root / "nsddata/experiments/nsdimagery/rawtargetimages"
        target_index = {path.stem: path for path in target_root.glob("*/*.png")}
    for directory, metadata, chunks in jobs:
        generate_samples(chunks, directory, metadata, engine, suffix, offset, partial(unclip_decode_batch, rgb_mode=rgb_mode),
                         save_image, num_samples, base_seed, target_index, args.resume)
    print(f"Saved scheme 1 reconstructions: {eval_root}", flush=True)
    print("[已完成] stage=decode", flush=True)


if __name__ == "__main__":
    main()
