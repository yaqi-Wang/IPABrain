"""Evaluate UnCLIP reconstructed images against NSD ground-truth stimuli.

This adapts the NeuroFlow-style image metrics to this project's output layout:

  outputs/synthesised_decoding_unclip/subjXX/recon/
    manifest.json
    images/trial*_img*.png

Ground-truth images are read from the NSD stimuli HDF5 by manifest image_id.
The reported metrics cover low-level image fidelity (PixCorr, SSIM and early
AlexNet features) as well as high-level semantics. The script is intentionally
offline-friendly and defaults to local cached pretrained weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    AlexNet_Weights,
    EfficientNet_B1_Weights,
    Inception_V3_Weights,
    alexnet,
    efficientnet_b1,
    inception_v3,
    resnet50,
)
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.data.nsd_sdxl_features import find_stimuli_hdf5
from src.utils.config import load_yaml
from src.utils.paths import ProjectPaths


DEFAULT_CLIP_CKPT = "huggingface/ViT-L-14.pt"
DEFAULT_SWAV_CKPT = "huggingface/swav_800ep_pretrain.pth.tar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate low- and high-level image metrics for UnCLIP reconstructions.")
    parser.add_argument("--paths", default="configs/paths.example.yaml")
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--decoded-dir", default="")
    parser.add_argument("--max-images", type=int, default=0, help="0 = evaluate all images in manifest.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", default="Alex2,Alex5,Incep,CLIP,Eff,SwAV",
                        help="Comma-separated subset from Alex2,Alex5,Incep,CLIP,Eff,SwAV. "
                             "Alex is retained as an alias for Alex5.")
    parser.add_argument("--clip-ckpt", default=DEFAULT_CLIP_CKPT)
    parser.add_argument("--swav-ckpt", default=DEFAULT_SWAV_CKPT)
    parser.add_argument("--save-json", default="")
    parser.add_argument("--skip-low-level", action="store_true",
                        help="Skip PixCorr/SSIM and only compute semantic metrics.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_manifest(decoded_dir: Path, max_images: int) -> list[dict]:
    manifest_path = decoded_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    items = manifest["items"]
    return items[:max_images] if max_images > 0 else items


def load_rgb(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def load_gt_rgb(stimuli_hdf5: Path, image_id: int, size: int) -> np.ndarray:
    with h5py.File(stimuli_hdf5, "r") as f:
        arr = np.asarray(f["imgBrick"][int(image_id)])
    img = Image.fromarray(arr).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img).astype(np.float32) / 255.0


def image_tensor_from_numpy(images: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()


def pixel_corr(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1) - a.mean()
    bf = b.reshape(-1) - b.mean()
    return float((af @ bf) / ((np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-8))


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Gaussian-window SSIM matching the standard reconstruction protocol."""
    a_gray = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    b_gray = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    sigma = 1.5
    truncate = 3.5
    mu_a = sp.ndimage.gaussian_filter(a_gray, sigma=sigma, truncate=truncate, mode="reflect")
    mu_b = sp.ndimage.gaussian_filter(b_gray, sigma=sigma, truncate=truncate, mode="reflect")
    var_a = sp.ndimage.gaussian_filter(a_gray * a_gray, sigma=sigma, truncate=truncate, mode="reflect") - mu_a ** 2
    var_b = sp.ndimage.gaussian_filter(b_gray * b_gray, sigma=sigma, truncate=truncate, mode="reflect") - mu_b ** 2
    cov = sp.ndimage.gaussian_filter(a_gray * b_gray, sigma=sigma, truncate=truncate, mode="reflect") - mu_a * mu_b
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2) + 1e-12
    )
    # skimage excludes the filter-radius border before averaging.
    pad = int(truncate * sigma + 0.5)
    return float(ssim_map[pad:-pad, pad:-pad].mean())


def compute_low_level(gt_images: list[np.ndarray], pred_images: list[np.ndarray]) -> dict[str, float]:
    pix = []
    ssim_vals = []
    for gt, pred in tqdm(list(zip(gt_images, pred_images)), desc="PixCorr/SSIM"):
        pix.append(pixel_corr(gt, pred))
        ssim_vals.append(ssim_gray(gt, pred))
    pix = np.asarray(pix, dtype=np.float32)
    ssim_vals = np.asarray(ssim_vals, dtype=np.float32)
    return {
        "PixCorr": float(pix.mean()),
        "PixCorr_std": float(pix.std()),
        "SSIM": float(ssim_vals.mean()),
        "SSIM_std": float(ssim_vals.std()),
    }


def load_swav(ckpt_path: Path) -> nn.Module:
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    model = resnet50(weights=None)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model


def get_model(model_name: str, device: torch.device, clip_ckpt: Path, swav_ckpt: Path):
    if model_name in {"Alex2", "Alex5", "Alex"}:
        # These match the conventional AlexNet layer-2/layer-5 hooks at
        # features[4] and features[11] used by NSD reconstruction benchmarks.
        layer = "features.4" if model_name == "Alex2" else "features.11"
        model = create_feature_extractor(alexnet(weights=AlexNet_Weights.DEFAULT), return_nodes=[layer])
        return model.to(device).eval().requires_grad_(False), layer
    if model_name == "Incep":
        model = create_feature_extractor(inception_v3(weights=Inception_V3_Weights.DEFAULT), return_nodes=["avgpool"])
        return model.to(device).eval().requires_grad_(False), "avgpool"
    if model_name == "CLIP":
        import clip

        model, _ = clip.load(str(clip_ckpt), device=device)
        model = model.eval().requires_grad_(False)
        return model.encode_image, None
    if model_name == "Eff":
        model = create_feature_extractor(efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT), return_nodes=["avgpool"])
        return model.to(device).eval().requires_grad_(False), "avgpool"
    if model_name == "SwAV":
        model = create_feature_extractor(load_swav(swav_ckpt), return_nodes=["avgpool"])
        return model.to(device).eval().requires_grad_(False), "avgpool"
    raise ValueError(f"Unsupported model: {model_name}")


def get_preprocess(model_name: str):
    if model_name == "Incep":
        return transforms.Compose([
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    if model_name == "CLIP":
        return transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    if model_name in {"Alex", "Alex2", "Alex5", "Eff", "SwAV"}:
        return transforms.Compose([
            transforms.Resize(224 if model_name != "Eff" else 255, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    raise ValueError(f"Unsupported model: {model_name}")


def extract_features(
    images: torch.Tensor,
    model,
    preprocess,
    layer: str | None,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    feats = []
    for start in tqdm(range(0, len(images), batch_size), desc="features"):
        batch = preprocess(images[start:start + batch_size]).to(device)
        with torch.no_grad():
            if layer is None:
                feat = model(batch).float().flatten(1)
            else:
                feat = model(batch)[layer].float().flatten(1)
        feats.append(feat.detach().cpu())
    return torch.cat(feats, dim=0).numpy()


def two_way_identification(gt_feat: np.ndarray, pred_feat: np.ndarray) -> float:
    n = len(gt_feat)
    corr = np.corrcoef(gt_feat, pred_feat)[:n, n:]
    congruent = np.diag(corr)
    success = corr < congruent
    return float(np.mean(np.sum(success, axis=0)) / (n - 1))


def correlation_distance(gt_feat: np.ndarray, pred_feat: np.ndarray) -> float:
    return float(np.array([sp.spatial.distance.correlation(gt_feat[i], pred_feat[i]) for i in range(len(gt_feat))]).mean())


def clip_retrieval(gt_feat: np.ndarray, pred_feat: np.ndarray) -> dict[str, float]:
    gt = gt_feat / (np.linalg.norm(gt_feat, axis=1, keepdims=True) + 1e-8)
    pred = pred_feat / (np.linalg.norm(pred_feat, axis=1, keepdims=True) + 1e-8)
    sim = pred @ gt.T
    true = np.diag(sim)
    rank = (sim > true[:, None]).sum(axis=1)
    return {
        "CLIP_Top1": float((rank < 1).mean()),
        "CLIP_Top5": float((rank < 5).mean()),
        "CLIP_Top10": float((rank < 10).mean()),
        "CLIP_Sim": float(true.mean()),
        "CLIP_Sim_std": float(true.std()),
    }


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.from_mapping(load_yaml(args.paths), ROOT)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    decoded_dir = Path(args.decoded_dir) if args.decoded_dir else (
        paths.output_root / "synthesised_decoding_unclip" / f"subj{args.subject:02d}" / "recon"
    )
    items = load_manifest(decoded_dir, args.max_images)
    if not items:
        raise RuntimeError(f"No items found in {decoded_dir / 'manifest.json'}")

    if paths.raw_nsd_root is None:
        raise ValueError("raw_nsd_root must be set in paths config.")
    stimuli_hdf5 = find_stimuli_hdf5(Path(paths.raw_nsd_root))

    print("decoded_dir:", decoded_dir)
    print("stimuli_hdf5:", stimuli_hdf5)
    print("n_images:", len(items))
    print("device:", device)

    if args.dry_run:
        first = items[0]
        print("[dry-run] first item:", first)
        print("[dry-run] first decoded image:", decoded_dir / first["path"])
        return

    gt_images = []
    pred_images = []
    for item in tqdm(items, desc="loading images"):
        pred_path = decoded_dir / item["path"]
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        gt_images.append(load_gt_rgb(stimuli_hdf5, int(item["image_id"]), size=256))
        pred_images.append(load_rgb(pred_path, size=256))

    results: dict[str, float | int | str] = {
        "subject": args.subject,
        "decoded_dir": str(decoded_dir),
        "n_images": len(items),
    }
    if not args.skip_low_level:
        results.update(compute_low_level(gt_images, pred_images))

    gt_tensor = image_tensor_from_numpy(gt_images)
    pred_tensor = image_tensor_from_numpy(pred_images)
    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]

    for model_name in selected_models:
        print(f"\n=== {model_name} ===", flush=True)
        model, layer = get_model(model_name, device, Path(args.clip_ckpt), Path(args.swav_ckpt))
        preprocess = get_preprocess(model_name)
        gt_feat = extract_features(gt_tensor, model, preprocess, layer, args.batch_size, device)
        pred_feat = extract_features(pred_tensor, model, preprocess, layer, args.batch_size, device)

        if model_name in {"Alex", "Alex2", "Alex5", "Incep", "CLIP"}:
            score = two_way_identification(gt_feat, pred_feat)
            results[f"{model_name}_2way"] = score
            print(f"{model_name} 2-way percent correct: {score * 100:.2f}%")
        else:
            score = correlation_distance(gt_feat, pred_feat)
            results[f"{model_name}_corr_distance"] = score
            print(f"{model_name} correlation distance: {score:.4f}")

        if model_name == "CLIP":
            retrieval = clip_retrieval(gt_feat, pred_feat)
            results.update(retrieval)
            print(
                f"CLIP sim={retrieval['CLIP_Sim']:.4f}, "
                f"Top1={retrieval['CLIP_Top1']:.4f}, "
                f"Top5={retrieval['CLIP_Top5']:.4f}, "
                f"Top10={retrieval['CLIP_Top10']:.4f}"
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== Summary ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print("saved:", out_path)


if __name__ == "__main__":
    main()
