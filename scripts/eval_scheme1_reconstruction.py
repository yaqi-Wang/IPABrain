"""Score each diffusion seed separately, then average; never select the best seed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_project_root

ROOT = add_project_root()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="One evaluation's nsd_test, vision or imagery directory")
    parser.add_argument("--models", default="", help="Optional cached feature metrics: Alex2,Alex5,Incep,CLIP,Eff,SwAV; empty = PixCorr/SSIM only")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clip-ckpt", default="huggingface/ViT-L-14.pt")
    parser.add_argument("--swav-ckpt", default="huggingface/swav_800ep_pretrain.pth.tar")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from eval_unclip_image_semantics import (compute_low_level, correlation_distance, extract_features,
                                            get_model, get_preprocess, image_tensor_from_numpy,
                                            load_gt_rgb, load_rgb, two_way_identification)
    from src.data.nsd_sdxl_features import find_stimuli_hdf5
    from train_brainvae_scheme1 import choose_device

    directory = Path(args.dataset_dir)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["aggregation"] != "mean_fmri":
        raise ValueError("This evaluator expects stimulus-level mean_fmri outputs, not correlated trial-level candidates")
    manifests = []
    reference = None
    for sample in range(metadata["num_samples"]):
        sample_dir = directory / f"sample{sample:02d}"
        manifest = json.loads((sample_dir / "manifest.json").read_text(encoding="utf-8"))
        if any(manifest.get(key) != metadata.get(key) for key in ("checkpoint_sha256", "data_fingerprint", "dataset", "num_samples")):
            raise ValueError("Sample manifests belong to different experiments")
        items = [item for item in manifest["items"] if item["stimulus_type"] != "conceptual"]
        keys = [(item["task_type"], item["stimulus_id"]) for item in items]
        if not items or len(set(keys)) != len(keys):
            raise ValueError("Need one reconstruction per unique target in each diffusion sample")
        if reference is not None and keys != reference:
            raise ValueError("Target identities or order differ between diffusion samples")
        reference = keys
        for item in items:
            if not (sample_dir / item["path"]).is_file():
                raise FileNotFoundError(sample_dir / item["path"])
        manifests.append((sample_dir, items))
    print(json.dumps({"dataset": metadata["dataset"], "n_samples": len(manifests),
                      "n_targets": len(reference), "partial_test": metadata["partial_test"],
                      "debug_training": metadata["debug_training"]}, indent=2), flush=True)
    if args.dry_run:
        return
    first_items = manifests[0][1]
    if metadata["dataset"] in {"nsd_test", "nsd_val"}:
        hdf5 = find_stimuli_hdf5(Path(metadata["raw_nsd_root"]))
        gt_images = [load_gt_rgb(hdf5, int(item["image_id"]), 256) for item in first_items]
    else:
        gt_images = [load_rgb(Path(item["target_image"]), 256) for item in first_items]
    strata = {"all": np.arange(len(first_items))}
    if metadata["dataset"] not in {"nsd_test", "nsd_val"}:
        for stimulus_type in sorted({item["stimulus_type"] for item in first_items}):
            strata[stimulus_type] = np.asarray([i for i, item in enumerate(first_items) if item["stimulus_type"] == stimulus_type])
    per_seed = []
    for sample_dir, items in manifests:
        pred_images = [load_rgb(sample_dir / item["path"], 256) for item in items]
        metrics = {}
        for stratum, indices in strata.items():
            prefix = "" if stratum == "all" else f"{stratum}_"
            values = compute_low_level([gt_images[i] for i in indices], [pred_images[i] for i in indices])
            metrics.update({prefix + key: value for key, value in values.items() if not key.endswith("_std")})
        per_seed.append(metrics)
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    device = choose_device(args.device)
    for name in names:
        model, layer = get_model(name, device, Path(args.clip_ckpt), Path(args.swav_ckpt))
        preprocess = get_preprocess(name)
        gt_features = extract_features(image_tensor_from_numpy(gt_images), model, preprocess, layer, args.batch_size, device)
        for number, (sample_dir, items) in enumerate(manifests):
            pred_images = [load_rgb(sample_dir / item["path"], 256) for item in items]
            pred_features = extract_features(image_tensor_from_numpy(pred_images), model, preprocess, layer, args.batch_size, device)
            for stratum, indices in strata.items():
                prefix = "" if stratum == "all" else f"{stratum}_"
                if name in {"Alex", "Alex2", "Alex5", "Incep", "CLIP"}:
                    if len(indices) > 1:
                        per_seed[number][f"{prefix}{name}_2way"] = two_way_identification(gt_features[indices], pred_features[indices])
                else:
                    per_seed[number][f"{prefix}{name}_corr_distance"] = correlation_distance(gt_features[indices], pred_features[indices])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {"protocol": metadata["protocol"], "dataset": metadata["dataset"], "subject": metadata["subject"],
              "n_targets": len(first_items), "n_samples": len(manifests), "aggregation": "arithmetic_mean_across_all_diffusion_seeds",
              "partial_test": metadata["partial_test"], "debug_training": metadata["debug_training"],
              "per_seed": per_seed, "metrics": {key: float(np.mean([row[key] for row in per_seed])) for key in per_seed[0]}}
    if "fusion_alpha" in metadata:
        result.update({key: metadata[key] for key in ("fusion_alpha", "fusion_source_sha256")})
    for key in ('validation_subset_sha256','validation_image_ids','mapping_objective',
                'prior_seed','unclip_seed','prior_sample_index','sampling_identity',
                'low_level_variant','low_level_oracle','low_level_identity','img2img_strength',
                'low_level_latent_space','low_level_scale_factor','target_usage',
                'generator_backend','text_condition','text_oracle','oracle','generation'):
        if key in metadata:
            result[key] = metadata[key]
    if metadata.get('generator_backend') == 'versatile_diffusion':
        from src.data.scheme1 import file_sha256
        result['source_metadata_sha256'] = file_sha256(directory / 'metadata.json')
        result['source_manifest_sha256'] = [file_sha256(sample / 'manifest.json') for sample, _ in manifests]
    # Seed dispersion is NOT a confidence interval over independent stimuli.
    result["diffusion_seed_std"] = {key: float(np.std([row[key] for row in per_seed])) for key in per_seed[0]}
    (directory / "image_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print(f"[已完成] stage=evaluate dataset={metadata['dataset']}", flush=True)


if __name__ == "__main__":
    main()
