"""CLIP similarity and ground-truth-image UnCLIP controls, NOT fMRI reconstruction.

The oracle stage never loads BrainVAE or fMRI. Target-image CLIP is deliberately
used as the condition, with outputs isolated from the zero-shot evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.data.scheme1 import atomic_torch_save, file_signature, load_scheme1_config, resolve_path, validate_tag
from src.utils.scheme1_runs import training_directory

TOKEN_SHAPE = (256, 1664)


def similarity_output_directory(config: dict, subject: int, tag: str) -> Path:
    """Keep legacy reports and new model comparisons in separate stable folders."""
    diagnostics = resolve_path(config["output_root"], ROOT) / f"subj{subject:02d}" / "diagnostics"
    run = training_directory(config, ROOT, subject)
    if run.name != "brainvae":
        diagnostics = diagnostics / run.name
    return diagnostics / f"clip_similarity_{tag}"


def checked_tokens(values) -> torch.Tensor:
    tokens = torch.as_tensor(np.array(values, dtype=np.float32, copy=True))
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != TOKEN_SHAPE:
        raise ValueError(f"Expected (N, 256, 1664) image tokens, got {tuple(tokens.shape)}")
    if not torch.isfinite(tokens).all() or (tokens.flatten(1).norm(dim=1) == 0).any():
        raise ValueError("Non-finite or zero CLIP condition")
    return tokens


def load_nsd_truth(nsd_root: Path, subject: int):
    directory = nsd_root / f"subj{subject:02d}"
    token_path = directory / f"nsd_sdxl_clip_test_sub{subject}.npy"
    id_path = directory / f"nsd_test_image_id_sub{subject}.npy"
    tokens = np.load(token_path, mmap_mode="r", allow_pickle=False)
    ids = np.load(id_path, allow_pickle=False).astype(np.int64)
    if tokens.shape != (len(ids), *TOKEN_SHAPE) or len(np.unique(ids)) != len(ids):
        raise ValueError("NSD test token / image ID mismatch")
    records = [{"subject_id": subject, "image_id": int(image_id),
                "stimulus_id": f"nsd{int(image_id):05d}", "stimulus_type": "natural",
                "task_type": "oracle_image", "source_row": row} for row, image_id in enumerate(ids)]
    return tokens, records, [file_signature(token_path), file_signature(id_path)]


def load_imagery_truth(path: Path, subject: int):
    """Read the repeated NPZ token array row by row; retain only 12 unique images.

    Also check that all cached repetitions of the same stimulus have identical
    image CLIP. Conceptual word-only trials are never treated as image features.
    """
    with np.load(path, allow_pickle=True) as archive:
        ids = archive["stimulus_id"].astype(str)
        types = archive["stimulus_type"].astype(str)
        selected = archive["has_clip_vision"].astype(bool) & np.isin(types, ("simple", "complex"))
        if not np.all(archive["subject_id"] == subject):
            raise ValueError("Imagery CLIP cache belongs to another subject")
    if not selected.any():
        raise ValueError("Imagery cache has no real image CLIP tokens")
    unique = {}
    source_rows = {}
    with zipfile.ZipFile(path) as archive, archive.open("target_clip_vision.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported target token NPY header: {version}")
        if shape != (len(ids), *TOKEN_SHAPE) or fortran or dtype.hasobject:
            raise ValueError("Invalid imagery target token array")
        row_bytes = int(np.prod(TOKEN_SHAPE)) * dtype.itemsize
        for row in range(int(np.flatnonzero(selected)[-1]) + 1):
            raw = stream.read(row_bytes)
            if len(raw) != row_bytes:
                raise ValueError("Truncated imagery token cache")
            if not selected[row]:
                continue
            key = (types[row], ids[row])
            token = np.frombuffer(raw, dtype=dtype).reshape(TOKEN_SHAPE)
            if key in unique:
                if not np.array_equal(unique[key], token):
                    raise ValueError(f"Different target CLIP for repetitions of {ids[row]}")
            else:
                unique[key] = token.copy()
                source_rows[key] = row
    keys = sorted(unique)  # Same ordering as mean_fmri vision / imagery predictions.
    tokens = checked_tokens(np.stack([unique[key] for key in keys]))
    records = [{"subject_id": subject, "stimulus_id": sid, "stimulus_type": kind,
                "task_type": "oracle_image", "source_row": source_rows[(kind, sid)]}
               for kind, sid in keys]
    return tokens, records, [file_signature(path)]


def paired_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, list[float]]:
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("Prediction and target token shapes differ")
    predicted, target = predicted.float(), target.float()
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
        raise ValueError("Non-finite CLIP in similarity evaluation")
    flat_p, flat_t = predicted.flatten(1), target.flatten(1)
    norm_p, norm_t = flat_p.norm(dim=1), flat_t.norm(dim=1)
    if (norm_p == 0).any() or (norm_t == 0).any():
        raise ValueError("Zero-norm CLIP in similarity evaluation")
    return {"cosine_flat": F.cosine_similarity(flat_p, flat_t, dim=1).tolist(),
            "cosine_token_mean": F.cosine_similarity(predicted, target, dim=-1).mean(dim=1).tolist(),
            "mse": ((flat_p - flat_t) ** 2).mean(dim=1).tolist(),
            "pred_norm": norm_p.tolist(), "true_norm": norm_t.tolist(),
            "norm_ratio": (norm_p / norm_t).tolist()}


def evaluate_predictions(directory: Path, truth, truth_records: list[dict], name: str, subject: int):
    """Align by image identity, not by an assumed trial or array row ordering."""
    index = {record["stimulus_id"]: row for row, record in enumerate(truth_records)}
    manifest = json.loads((directory / "predictions.json").read_text(encoding="utf-8"))
    if manifest["subject"] != subject or manifest["dataset"] != name:
        raise ValueError(f"Prediction subject / dataset mismatch: {directory}")
    chunks = [resolve_path(value, directory) for value in manifest["token_chunks"]]
    # Exact average similarity to ALL wrong images, without a large NxN matrix.
    normalized_sum = torch.zeros(int(np.prod(TOKEN_SHAPE)), dtype=torch.float64)
    for start in range(0, len(truth), 8):
        block = checked_tokens(truth[start:start + 8]).flatten(1)
        normalized_sum += F.normalize(block, dim=1).double().sum(dim=0)
    rows, seen = [], set()
    for chunk in chunks:
        payload = torch.load(chunk, map_location="cpu", weights_only=False)
        records = payload["records"]
        if any(record["stimulus_id"] not in index for record in records):
            raise ValueError(f"Prediction target identity not found in true CLIP: {chunk}")
        positions = [index[record["stimulus_id"]] for record in records]
        for record, position in zip(records, positions):
            if record["subject_id"] != subject or record["task_type"] != name:
                raise ValueError(f"Prediction record subject / task mismatch: {chunk}")
            if "image_id" in record and record["image_id"] != truth_records[position]["image_id"]:
                raise ValueError("NSD image ID mismatch")
            if manifest["aggregation"] == "mean_fmri" and record["stimulus_id"] in seen:
                raise ValueError("Repeated stimulus in mean_fmri predictions")
            seen.add(record["stimulus_id"])
        predicted = payload["mean"].float()
        target = checked_tokens(np.stack([np.asarray(truth[position]) for position in positions]))
        metrics = paired_metrics(predicted, target)
        sum_cosines = F.normalize(predicted.flatten(1), dim=1).double() @ normalized_sum
        for offset, record in enumerate(records):
            item = {"dataset": name, "stimulus_id": record["stimulus_id"],
                    "stimulus_type": record["stimulus_type"], "n_trials": record["n_trials"],
                    **{key: values[offset] for key, values in metrics.items()}}
            if len(truth) > 1:
                item["wrong_image_cosine_mean"] = float((sum_cosines[offset] - item["cosine_flat"]) / (len(truth) - 1))
                item["matched_minus_wrong"] = item["cosine_flat"] - item["wrong_image_cosine_mean"]
            rows.append(item)
    if len(rows) != manifest["n_items"]:
        raise ValueError("Cached prediction count differs from its manifest")
    metric_names = [key for key in rows[0] if key not in ("dataset", "stimulus_id", "stimulus_type", "n_trials")]

    def summarize(items):
        return {"n_items": len(items), **{key: {"mean": float(np.mean([row[key] for row in items])),
                                               "std": float(np.std([row[key] for row in items]))}
                                        for key in metric_names}}

    summary = {**summarize(rows), "by_stimulus_type": {
        kind: summarize([row for row in rows if row["stimulus_type"] == kind])
        for kind in sorted({row["stimulus_type"] for row in rows})}}
    print(f"[相似性] {name} n={len(rows)} cosine={summary['cosine_flat']['mean']:.6f} "
          f"norm_ratio={summary['norm_ratio']['mean']:.6f}", flush=True)
    return summary, rows


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def prepare_oracle(directory: Path, tokens, records: list[dict], metadata: dict, resume: bool):
    """Save small, explicit true-CLIP chunks for the existing generation loop."""
    from decode_brainvae_scheme1 import reconcile_resume_metadata

    if directory.exists():
        if not resume:
            raise FileExistsError(f"Use --resume to complete existing control images: {directory}")
        reconcile_resume_metadata(directory, metadata)
    else:
        directory.mkdir(parents=True)
        write_json(directory / "metadata.json", metadata)
    (directory / "tokens").mkdir(exist_ok=True)
    chunks = []
    for start in range(0, len(records), 8):
        path = directory / "tokens" / f"chunk{start // 8:05d}.pt"
        batch = checked_tokens(tokens[start:start + 8])
        batch_records = records[start:start + 8]
        if path.exists():
            cached = torch.load(path, map_location="cpu", weights_only=False)
            if cached["records"] != batch_records or not torch.equal(cached["mean"], batch):
                raise ValueError(f"True CLIP cache differs: {path}")
        else:
            # 'mean' is only the shared generator interface key, not a posterior.
            atomic_torch_save({"mean": batch, "records": batch_records,
                               "conditioning_source": "ground_truth_image_clip"}, path)
        chunks.append(path)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/brainvae_scheme1.yaml")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--stage", choices=("similarity", "oracle"), required=True)
    parser.add_argument("--dataset", choices=("nsd_test", "imagery_targets"), default="imagery_targets",
                        help="Oracle only: imagery_targets is the 12 images shared by vision and imagery")
    parser.add_argument("--max-items", type=int, default=0, help="Oracle only; 0 means all images")
    parser.add_argument("--token-source-tag", default="", help="Similarity only: existing BrainVAE encoding tag")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=38)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Oracle only: validate / cache true tokens without loading UnCLIP")
    args = parser.parse_args()
    if min(args.num_samples, args.num_steps, args.threads) <= 0 or args.max_items < 0:
        parser.error("num-samples, num-steps and threads must be positive; max-items cannot be negative")
    torch.set_num_threads(args.threads)
    config, paths = load_scheme1_config(args.config, ROOT)
    diagnostics = resolve_path(config["output_root"], ROOT) / f"subj{args.subject:02d}" / "diagnostics"
    imagery_path = paths.imagery_root / f"subj{args.subject:02d}" / "imagery_trials.npz"
    if args.stage == "similarity":
        tag = args.token_source_tag or config["evaluation"].get("token_source_tag", "mean_fmri")
        validate_tag(tag)
        prediction_root = training_directory(config, ROOT, args.subject) / "evaluation" / tag
        nsd_truth, nsd_records, nsd_files = load_nsd_truth(paths.nsd_root, args.subject)
        imagery_truth, imagery_records, imagery_files = load_imagery_truth(imagery_path, args.subject)
        summaries, rows = {}, []
        for name in ("nsd_test", "vision", "imagery"):
            truth, records = (nsd_truth, nsd_records) if name == "nsd_test" else (imagery_truth, imagery_records)
            summaries[name], items = evaluate_predictions(prediction_root / name, truth, records, name, args.subject)
            rows.extend(items)
        directory = similarity_output_directory(config, args.subject, tag)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(directory / "summary.json", {"subject": args.subject, "prediction_root": str(prediction_root),
                   "metric": "cosine of flattened posterior mean vs raw image CLIP (256x1664)",
                   "wrong_image_baseline": "average cosine to all other images in the same target set",
                   "target_usage": "evaluation_only", "truth_files": nsd_files + imagery_files,
                   "datasets": summaries})
        print(f"[已完成] stage=similarity output={directory}", flush=True)
        return

    if args.dataset == "nsd_test":
        tokens, records, sources = load_nsd_truth(paths.nsd_root, args.subject)
    else:
        tokens, records, sources = load_imagery_truth(imagery_path, args.subject)
    if args.max_items:
        tokens, records = tokens[:args.max_items], records[:args.max_items]
    label = args.dataset + (f"_first{len(records)}" if args.max_items else "")
    directory = diagnostics / f"true_clip_rgb01_s{args.num_samples}" / label
    unclip = resolve_path(config["evaluation"]["unclip_checkpoint"], ROOT)
    openclip = paths.openclip_version
    metadata = {"protocol": "generator_diagnostic_true_image_clip", "subject": args.subject,
                "dataset": args.dataset, "aggregation": "image_level", "n_items": len(records),
                "conditioning_source": "ground_truth_image_clip", "target_usage": "generation_condition_ORACLE",
                "is_fmri_reconstruction": False, "brainvae_used": False, "low_level_init": False,
                "clip_unit_conversion": "none_raw_image_tokens", "rgb_mode": "standard",
                "source_files": sources, "num_samples": args.num_samples, "num_steps": args.num_steps,
                "seed": args.seed, "unclip_checkpoint": str(unclip), "openclip_checkpoint": str(openclip)}
    print(f"[生成器对照] 输入真实图片 CLIP，不使用 BrainVAE/fMRI；{len(records)} 个目标 x {args.num_samples} 张", flush=True)
    print(f"[输出目录] {directory}", flush=True)
    if not args.prepare_only:
        for path in (unclip, openclip):
            if path is None or not path.is_file():
                raise FileNotFoundError(f"Missing generator checkpoint: {path}")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable here; run on the GPU host or use --prepare-only")
        if paths.generative_models_path is None:
            raise ValueError("Missing generative_models_path")
    chunks = prepare_oracle(directory, tokens, records, metadata, args.resume)
    if args.prepare_only:
        print("[已完成] stage=oracle_prepare（尚未生成图片）", flush=True)
        return
    from decode_brainvae_scheme1 import generate_samples
    from decode_synthesised_fmri_unclip import load_sdxl_unclip, unclip_decode_batch
    from torchvision.utils import save_image

    engine, suffix, offset = load_sdxl_unclip(paths.generative_models_path, unclip, openclip,
                                            torch.device(args.device), args.num_steps)
    targets = {}
    if paths.raw_nsd_root is not None:
        root = paths.raw_nsd_root / "nsddata/experiments/nsdimagery/rawtargetimages"
        targets = {path.stem: path for path in root.glob("*/*.png")}
    generate_samples(chunks, directory, metadata, engine, suffix, offset,
                     partial(unclip_decode_batch, rgb_mode="standard"), save_image,
                     args.num_samples, args.seed, targets, args.resume)
    print(f"[已完成] stage=oracle output={directory}", flush=True)


if __name__ == "__main__":
    main()
