"""Prepare immutable image splits, source-only normalization and raw imagery."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

from _bootstrap import add_project_root

ROOT = add_project_root()

from prepare_imagery_nsd import decode_trials
from prepare_nsd_perception_fmri import build_ppbe_split, load_expdesign
from src.data.scheme1 import (PROTOCOL, FMRIStandardizer, check_disjoint, data_directory,
                              digest_json, file_sha256, file_signature, read_imagery_roi_xyz,
                              image_split, load_prepared, load_scheme1_config)


def verify_source_sample(raw_root: Path, subject: int, source_ids: np.ndarray,
                         grouped_fmri: np.ndarray, mask: np.ndarray, stored_divisor: float) -> dict:
    """Check scale AND the full voxel ordering against a raw perception trial."""
    design = load_expdesign(raw_root / "nsddata/experiments/nsd/nsd_expdesign.mat")
    _, _, lookup, _ = build_ppbe_split(design, subject, sessions=40)
    trial = min(lookup[int(source_ids[0])])
    session = trial // 750 + 1
    frame = trial % 750
    path = raw_root / f"nsddata_betas/ppdata/subj{subject:02d}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{session:02d}.nii.gz"
    raw = np.asarray(nib.load(str(path)).dataobj[..., frame], dtype=np.float32)
    expected = raw[mask] / stored_divisor
    actual = np.asarray(grouped_fmri[0, 0], dtype=np.float32)
    if not np.allclose(actual, expected, rtol=1e-5, atol=1e-6):
        raise ValueError("Existing NSD npy does not match raw XYZ voxel order / stored divisor. "
                         f"Max absolute difference: {float(np.max(np.abs(actual - expected))):.6g}")
    return {"status": "verified", "raw_file": str(path), "trial": trial,
            "max_absolute_difference": float(np.max(np.abs(actual - expected)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/brainvae_scheme1.yaml")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Audit IDs, array headers and source voxel order without writing data.")
    parser.add_argument("--reuse", action="store_true", help="Reuse an identical, complete preparation; never overwrite existing data.")
    parser.add_argument("--reuse-imagery-from", default="", help="Reuse verified raw ROI rows from an existing prepared data directory")
    args = parser.parse_args()
    config, paths = load_scheme1_config(args.config, ROOT)
    prep = config["preprocessing"]
    if paths.raw_nsd_root is None:
        raise ValueError("raw_nsd_root is required to rebuild imagery in canonical voxel order")
    raw_root = paths.raw_nsd_root
    directory = paths.nsd_root / f"subj{args.subject:02d}"
    files = {f"{split}_{kind}": directory / name for split in ("train", "test")
             for kind, name in (("fmri", f"nsd_{split}_fmri_scale_sub{args.subject}.npy"),
                                ("ids", f"nsd_{split}_image_id_sub{args.subject}.npy"),
                                ("clip", f"nsd_sdxl_clip_{split}_sub{args.subject}.npy"))}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source files: " + ", ".join(missing))
    arrays = {key: np.load(path, mmap_mode="r", allow_pickle=False) for key, path in files.items()}
    source_ids, test_ids = arrays["train_ids"], arrays["test_ids"]
    for split in ("train", "test"):
        fmri, ids, clip = (arrays[f"{split}_{key}"] for key in ("fmri", "ids", "clip"))
        if fmri.ndim != 3 or ids.ndim != 1 or len(ids) != len(fmri) or len(np.unique(ids)) != len(ids):
            raise ValueError(f"Invalid grouped fMRI / unique IDs for {split}")
        expected = (len(ids), config["model"]["latent_tokens"], config["model"]["embed_dim"])
        if tuple(clip.shape) != expected:
            raise ValueError(f"{split} CLIP shape {clip.shape} != expected {expected}")
    experiment = raw_root / "nsddata/experiments/nsdimagery"
    _, records = decode_trials(experiment)
    nsd_ids = set()
    for record in records:
        match = re.search(r"_nsd(\d+)$", record["stimulus_id"])
        if match and int(match.group(1)) > 0:
            # Filenames use ONE-based NSD IDs; stored perception IDs are zero-based.
            nsd_ids.add(int(match.group(1)) - 1)
    imagery_nsd_ids = np.asarray(sorted(nsd_ids), dtype=np.int64)
    check_disjoint(source_ids, test_ids, imagery_nsd_ids)
    train_indices, val_indices = image_split(source_ids, float(config["validation"]["fraction"]),
                                            int(config["validation"]["split_seed"]))
    mask_path = raw_root / f"nsddata/ppdata/subj{args.subject:02d}/func1pt8mm/roi/nsdgeneral.nii.gz"
    mask_img = nib.load(str(mask_path))
    mask = mask_img.get_fdata() > 0
    n_voxels = int(mask.sum())
    if any(arrays[f"{split}_fmri"].shape[-1] != n_voxels for split in ("train", "test")):
        raise ValueError("Perception voxel count does not match native nsdgeneral mask")
    beta_path = raw_root / f"nsddata_betas/ppdata/subj{args.subject:02d}/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.hdf5"
    with h5py.File(beta_path, "r") as handle:
        if tuple(handle["betas"].shape) != (len(records), *mask.shape[::-1]):
            raise ValueError("Raw imagery beta dimensions do not match trials / native mask")
    keep_records = [record for record in records if record["task"] == "imagery" or
                    (record["task"] == "vision" and record["stimulus_type"] in {"simple", "complex"})]
    raw_files = [mask_path, beta_path, experiment / "designmatrixGLMsingle.mat",
                 *[experiment / f"{name}_pair_list.mat" for name in ("A", "B", "C")]]
    source = {key: file_signature(path) for key, path in files.items()}
    source.update({str(path): file_signature(path) for path in raw_files})
    input_spec = {"subject": args.subject, "files": source, "preprocessing": prep,
                  "val_fraction": config["validation"]["fraction"], "split_seed": config["validation"]["split_seed"]}
    fingerprint = digest_json(input_spec)
    out_dir = data_directory(config, ROOT, args.subject)
    summary = {"protocol": PROTOCOL, "subject": args.subject, "data_fingerprint": fingerprint,
               "input_spec": input_spec, "n_source_images": len(source_ids),
               "n_train_images": len(train_indices), "n_val_images": len(val_indices),
               "n_test_images": len(test_ids), "n_voxels": n_voxels,
               "imagery_nsd_ids_zero_based": imagery_nsd_ids.tolist(),
               "imagery_trial_counts": dict(Counter(f"{r['task']}/{r['stimulus_type']}" for r in keep_records)),
               "voxel_order": "native_XYZ_mask_C_order",
               "normalization_fit": ("NSD_internal_training_trials_only" if prep["normalization"] == "source_train"
                                     else "none_beta_scale_only"),
               "test_statistics_used": False, "output_dir": str(out_dir)}
    if out_dir.exists() and not args.dry_run:
        if not args.reuse:
            raise FileExistsError(f"{out_dir} exists; use --reuse or choose a new data_tag in the config")
        previous, _, _ = load_prepared(out_dir)
        if previous["data_fingerprint"] != fingerprint:
            raise ValueError("Inputs or settings changed; select a new data_tag instead of reusing artifacts")
        print(json.dumps(previous, indent=2, ensure_ascii=False))
        print("[已完成] stage=prepare（复用已有数据）", flush=True)
        return
    if prep.get("verify_nsd_sample", True):
        summary["source_sample_check"] = verify_source_sample(raw_root, args.subject, source_ids,
                                                              arrays["train_fmri"], mask, float(prep["nsd_stored_divisor"]))
    else:
        summary["source_sample_check"] = {"status": "disabled_not_verified"}
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    divisor = float(prep["common_divisor"])
    if divisor <= 0 or float(prep["nsd_stored_divisor"]) <= 0:
        raise ValueError("Scaling divisors must be positive")
    nsd_scale = float(prep["nsd_stored_divisor"]) / divisor
    if prep["normalization"] == "beta_scale":
        normalizer = FMRIStandardizer.scaled_beta(n_voxels, nsd_scale, divisor)
        print(f"[原版输入] beta/{divisor:g}; 不拟合逐体素均值/标准差", flush=True)
    else:
        print("Fitting normalization using internal training trials only ...", flush=True)
        normalizer = FMRIStandardizer.fit(arrays["train_fmri"], train_indices, nsd_scale, divisor,
                                          float(prep["epsilon"]), int(prep["chunk_images"]))
    indices = np.asarray([r["global_trial_idx"] for r in keep_records], dtype=np.int64)
    if args.reuse_imagery_from:
        previous_dir = Path(args.reuse_imagery_from)
        if not previous_dir.is_absolute():
            previous_dir = ROOT / previous_dir
        previous, previous_split, _ = load_prepared(previous_dir)
        if previous["subject"] != args.subject or previous["input_spec"]["files"] != source:
            raise ValueError("Raw imagery cache subject/source files differ from current data")
        if not np.array_equal(previous_split["source_image_ids"], source_ids):
            raise ValueError("Raw imagery cache source image order differs")
        with np.load(previous_dir / "preprocessing.npz", allow_pickle=False) as old:
            if not np.array_equal(old["voxel_xyz"], np.argwhere(mask)):
                raise ValueError("Raw imagery cache voxel order differs")
        with np.load(previous_dir / "imagery_trials.npz", allow_pickle=False) as old:
            for key in ("global_trial_idx", "stimulus_id", "stimulus_type", "run", "condition_index", "cue_letter", "stim_set"):
                if not np.array_equal(old[key], np.asarray([r[key] for r in keep_records])):
                    raise ValueError(f"Raw imagery cache record mismatch: {key}")
            if not np.array_equal(old["task_type"], np.asarray([r["task"] for r in keep_records])):
                raise ValueError("Raw imagery cache task order differs")
            old_divisor = float(old["common_divisor"])
            if not np.isfinite(old_divisor) or old_divisor <= 0:
                raise ValueError("Invalid old raw imagery divisor")
            fmri_scaled = np.asarray(old["fmri_scaled"], dtype=np.float32) * (old_divisor / divisor)
        summary["raw_imagery_cache_source"] = str(previous_dir.resolve())
        print(f"[复用原始 ROI] {previous_dir}; 缩放 {old_divisor:g}→{divisor:g}", flush=True)
    else:
        print("Extracting raw imagery ROI in canonical XYZ order ...", flush=True)
        with h5py.File(beta_path, "r") as handle:
            fmri_scaled = read_imagery_roi_xyz(handle["betas"], mask, indices, divisor)
    if not np.isfinite(fmri_scaled).all():
        raise ValueError("Non-finite raw imagery fMRI")
    out_dir.mkdir(parents=True, exist_ok=False)
    np.savez(out_dir / "split.npz", source_image_ids=source_ids, test_image_ids=test_ids,
             train_indices=train_indices, val_indices=val_indices, imagery_nsd_ids=imagery_nsd_ids)
    np.savez(out_dir / "preprocessing.npz", **normalizer.arrays(),
             voxel_xyz=np.argwhere(mask).astype(np.int32), mask_affine=mask_img.affine, subject_id=args.subject)
    # This cache contains scaled raw responses, NOT test-fitted z-scores or image features.
    payload = {"fmri_scaled": fmri_scaled, "subject_id": np.asarray(args.subject),
               "common_divisor": np.asarray(divisor)}
    for key in ("stimulus_id", "stimulus_type", "run", "global_trial_idx", "condition_index", "cue_letter", "stim_set"):
        payload[key] = np.asarray([record[key] for record in keep_records])
    payload["task_type"] = np.asarray([record["task"] for record in keep_records])
    np.savez_compressed(out_dir / "imagery_trials.npz", **payload)
    summary["artifact_sha256"] = {name: file_sha256(out_dir / name) for name in
                                   ("split.npz", "preprocessing.npz", "imagery_trials.npz")}
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared scheme 1 data: {out_dir}", flush=True)
    print("[已完成] stage=prepare", flush=True)


if __name__ == "__main__":
    main()
