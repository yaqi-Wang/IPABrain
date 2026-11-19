"""Compare frozen local/author BrainVAE means directly in UnCLIP, on NSD test only."""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.data.scheme1 import (
    Scheme1NSDDataset, check_prepared_config, check_source_files, data_directory,
    digest_json, file_sha256, load_prepared, load_scheme1_config, resolve_path, validate_tag,
)
from src.utils.clip_mapping import atomic_json, frozen_brain, load_brain_source, mapping_lock
from src.utils.scheme1_runs import training_directory
from decode_brainvae_scheme1 import encode_predictions, generate_samples, reconcile_resume_metadata
from train_brainvae_scheme1 import choose_device

DIRECT_PROTOCOL = "synbrain_direct_mean_nsd_v1"


def test_dataset(paths, subject, manifest, split, normalizer):
    check_source_files(paths.nsd_root, subject, manifest, ("test_fmri", "test_ids"))
    return Scheme1NSDDataset(paths.nsd_root, subject, "test", normalizer,
                            average_repeats=True, load_targets=False,
                            expected_ids=split["test_image_ids"])


def save_test_selection(directory: Path, dataset, manifest: dict, split: dict,
                        subject: int, count: int) -> Path:
    """Pin the same held-out targets for both encoders; never alter the training split."""
    if count <= 0 or count > len(dataset):
        raise ValueError(f"Choose between 1 and {len(dataset)} NSD test targets")
    if dataset.split != "test" or not dataset.average_repeats:
        raise ValueError("Select stimulus-level mean_fmri from NSD test only")
    if not np.array_equal(dataset.image_ids, split["test_image_ids"]):
        raise ValueError("NSD test order differs from the prepared split")
    if np.intersect1d(split["source_image_ids"], split["test_image_ids"]).size:
        raise ValueError("NSD test overlaps the local training/validation source pool")
    selection = {
        "protocol": DIRECT_PROTOCOL, "subject": subject, "dataset": "nsd_test",
        "selection": "first_in_original_nsd_test_order", "n_targets": count,
        "parent_test_targets": len(split["test_image_ids"]),
        "source_pool_targets": len(split["source_image_ids"]),
        "internal_train_targets": len(split["train_indices"]),
        "internal_val_targets": len(split["val_indices"]),
        "data_fingerprint": manifest["data_fingerprint"], "aggregation": "mean_fmri",
        "parent_test_image_ids_sha256": digest_json({"image_ids": split["test_image_ids"].tolist()}),
        "records": [dataset.record(index) for index in range(count)],
    }
    path = directory / f"nsd_test_first{count}.json"
    with mapping_lock(directory):
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != selection:
                raise ValueError(f"Existing NSD selection differs: {path}; do not mix target sets")
        else:
            atomic_json(selection, path)
    return path


def source_specification(args, brain_config):
    # Only the source-loader interface is reused; no mapping head is built or loaded.
    if args.source == "local":
        return {"brainvae_source": "local"}
    return {
        "brainvae_source": "synbrain_external",
        "brainvae_checkpoint": f"checkpoints/synbrain_author/subj{args.subject:02d}/brainvae.pth",
        "external_source": {
            "subject": args.subject, "author_model_config": args.author_config,
            "origin": "https://huggingface.co/MichaelMaiii/SynBrain",
            "training_data_provenance": "author_checkpoint_not_yet_audited",
        },
        "model": {key: brain_config["model"][key] for key in ("latent_tokens", "embed_dim")},
    }


def decode(args, brain_config, paths, output):
    source = load_brain_source(source_specification(args, brain_config), brain_config,
                               ROOT, args.subject, args.checkpoint)
    identity = source["identity"]
    dataset = test_dataset(paths, args.subject, source["manifest"], source["split"], source["normalizer"])
    selection = save_test_selection(output / "evaluation_sets", dataset, source["manifest"],
                                    source["split"], args.subject, args.max_items)
    evaluation = brain_config["evaluation"]
    batch_size = args.batch_size or int(evaluation["encode_batch_size"])
    unclip = resolve_path(args.unclip_ckpt or evaluation["unclip_checkpoint"], ROOT)
    openclip = resolve_path(args.openclip_path, ROOT) if args.openclip_path else paths.openclip_version
    device = choose_device(args.device)
    model_config = identity["brainvae_model_config"]
    if not args.encode_only:
        if (model_config["latent_tokens"], model_config["embed_dim"]) != (256, 1664):
            raise ValueError("UnCLIP requires BrainVAE means shaped (N,256,1664)")
        for label, path in (("UnCLIP checkpoint", unclip), ("OpenCLIP checkpoint", openclip)):
            if path is None or not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
        if paths.generative_models_path is None:
            raise ValueError("generative_models_path is required for reconstruction")
        if device.type != "cuda" and args.device == "auto":
            raise RuntimeError("No GPU for UnCLIP; use --encode-only or explicitly --device cpu")
    run_dir = (training_directory(brain_config, ROOT, args.subject) if args.source == "local" else
               output / "runs/brainvae_synbrain_author")
    tag = args.eval_tag or f"direct_mean_nsd{args.max_items}_s{args.num_samples}"
    validate_tag(tag)
    directory = run_dir / "evaluation" / tag / "nsd_test"
    metadata = {
        "protocol": DIRECT_PROTOCOL, "subject": args.subject, "dataset": "nsd_test",
        "checkpoint": identity["brainvae_checkpoint"],
        "checkpoint_sha256": identity["brainvae_checkpoint_sha256"],
        "source": args.source, "source_identity": identity,
        "data_fingerprint": identity["data_fingerprint"], "aggregation": "mean_fmri",
        "encode_batch_size": batch_size, "num_samples": args.num_samples,
        "num_steps": args.num_steps, "seed": args.seed, "rgb_mode": args.rgb_mode,
        "unclip_checkpoint": str(unclip), "openclip_checkpoint": str(openclip),
        "stimulus_types": ["natural"], "n_items": args.max_items,
        "partial_test": args.max_items < len(dataset),
        "debug_training": identity["source_debug_training"],
        "posterior_output": "mean_only", "posterior_variance_conditions_generation": False,
        "mapping_head": None, "token_representation": "brainvae_posterior_mean",
        "target_usage": "evaluation_only", "normalization": source["normalizer"].description,
        "raw_nsd_root": str(paths.raw_nsd_root), "selection_manifest": str(selection),
        "selection_manifest_sha256": file_sha256(selection),
    }
    if directory.exists():
        if not args.resume:
            raise FileExistsError(f"{directory} exists; use --resume or another --eval-tag")
        reconcile_resume_metadata(directory, metadata)
    else:
        directory.mkdir(parents=True)
        atomic_json(metadata, directory / "metadata.json")
    model = frozen_brain(source, device)
    del source["weights"]
    chunks = encode_predictions(model, dataset, directory, batch_size, device,
                                max_items=args.max_items, resume=args.resume)
    atomic_json({**metadata, "encoding_directory": str(directory),
                 "token_chunks": [str(path.relative_to(directory)) for path in chunks]},
                directory / "predictions.json")
    del model, source
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if args.encode_only:
        print(f"[已完成] source={args.source} stage=encode targets={args.max_items} output={directory}", flush=True)
        return
    from decode_synthesised_fmri_unclip import load_sdxl_unclip, unclip_decode_batch
    from torchvision.utils import save_image

    engine, suffix, offset = load_sdxl_unclip(paths.generative_models_path, unclip, openclip,
                                            device, args.num_steps)
    generate_samples(chunks, directory, metadata, engine, suffix, offset,
                     partial(unclip_decode_batch, rgb_mode=args.rgb_mode), save_image,
                     args.num_samples, args.seed, {}, args.resume)
    print(f"[已完成] source={args.source} stage=decode targets={args.max_items} output={directory}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("local", "author"), default="local")
    parser.add_argument("--config", default="configs/brainvae_scheme1_synbrain.yaml")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--checkpoint", default="", help="Override this source's BrainVAE checkpoint")
    parser.add_argument("--author-config", default="", help="Optional original author model YAML")
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=38)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rgb-mode", choices=("standard", "legacy"), default="standard")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--unclip-ckpt", default="")
    parser.add_argument("--openclip-path", default="")
    parser.add_argument("--eval-tag", default="")
    parser.add_argument("--prepare-only", action="store_true", help="Pin NSD targets without loading any model")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if (min(args.subject, args.max_items, args.num_samples, args.num_steps, args.threads) <= 0
            or (args.batch_size is not None and args.batch_size <= 0)):
        parser.error("Subject, counts, threads and batch size must be positive")
    torch.set_num_threads(args.threads)
    config, paths = load_scheme1_config(args.config, ROOT)
    if config["model"].get("architecture") != "synbrain":
        parser.error("Use the original-compatible SynBrain configuration")
    output = resolve_path(config["output_root"], ROOT) / f"subj{args.subject:02d}"
    if args.prepare_only:
        manifest, split, normalizer = load_prepared(data_directory(config, ROOT, args.subject))
        check_prepared_config(config, manifest)
        if manifest["subject"] != args.subject:
            raise ValueError("Prepared data subject mismatch")
        dataset = test_dataset(paths, args.subject, manifest, split, normalizer)
        selection = save_test_selection(output / "evaluation_sets", dataset, manifest,
                                        split, args.subject, args.max_items)
        print(f"[已完成] stage=prepare_nsd_test targets={args.max_items} selection={selection}", flush=True)
        return
    with mapping_lock(output / "direct_nsd_locks" / args.source):
        decode(args, config, paths, output)


if __name__ == "__main__":
    main()
