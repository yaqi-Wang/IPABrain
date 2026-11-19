"""Extract OpenCLIP ViT-bigG-14 features for the 18 NSD-Imagery stimuli, then
merge them into the per-subject imagery_trials.npz produced by
prepare_imagery_nsd.py.

The CLIP model and pre-processing must match scripts/extract_nsd_sdxl_clip_vision.py
and scripts/extract_nsd_sdxl_clip_text.py so that imagery latents share a space
with NSD-perception features (required for CLIP retrieval-based priors).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.data.nsd_sdxl_features import OPENCLIP_ARCH
from src.utils.config import load_yaml
from src.utils.paths import ProjectPaths


def add_generative_models_to_path(path: Path) -> None:
    """Allow `from generative_models...` when config points at the package dir."""
    sys.path.append(str(path))
    if path.name == "generative_models":
        sys.path.append(str(path.parent))


# 18 unique stimulus_id strings produced by prepare_imagery_nsd.py.
SET_A_FILES = {
    "bar_000.0deg_450L_43W": "bar_000.0deg_450L_43W.png",
    "bar_045.0deg_450L_43W": "bar_045.0deg_450L_43W.png",
    "bar_090.0deg_450L_43W": "bar_090.0deg_450L_43W.png",
    "bar_135.0deg_450L_43W": "bar_135.0deg_450L_43W.png",
    "crs_000.0deg_450L_43W": "crs_000.0deg_450L_43W.png",
    "crs_045.0deg_450L_43W": "crs_045.0deg_450L_43W.png",
}
SET_B_FILES = {
    "shared0000_nsd00000": "shared0000_nsd00000.png",
    "shared0385_nsd28752": "shared0385_nsd28752.png",
    "shared0413_nsd30857": "shared0413_nsd30857.png",
    "shared0741_nsd53882": "shared0741_nsd53882.png",
    "shared0842_nsd61178": "shared0842_nsd61178.png",
    "shared0907_nsd65873": "shared0907_nsd65873.png",
}
# Set C is conceptual: only words, no images. CLIP-Text is used instead.
SET_C_WORDS = ["banana", "fruit", "mammal", "stripes", "yellow", "zebra"]


def load_image(path: Path, size: int = 224) -> torch.Tensor:
    """Load an image as a (3, H, W) float tensor in [0, 1], matching the
    NSD perception loader's convention (raw RGB / 255, embedder handles its own
    normalization internally)."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def extract_vision_features(
    embedder, files_dir: Path, ordered_ids: list[str], file_map: dict[str, str], device: str
) -> tuple[np.ndarray, np.ndarray]:
    images = torch.stack([load_image(files_dir / file_map[sid]) for sid in ordered_ids])
    with torch.no_grad():
        tokens, pooled = embedder(images.to(device))
    return tokens.detach().cpu().float().numpy(), pooled.detach().cpu().float().numpy()


def extract_text_features(
    text_embedder, words: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        tokens, pooled = text_embedder(words)
    return tokens.detach().cpu().float().numpy(), pooled.detach().cpu().float().numpy()


def merge_into_imagery_trials(
    npz_path: Path,
    vision_token_by_id: dict[str, np.ndarray],
    vision_pool_by_id: dict[str, np.ndarray],
    text_token_by_id: dict[str, np.ndarray],
    text_pool_by_id: dict[str, np.ndarray],
) -> None:
    data = dict(np.load(npz_path, allow_pickle=True))
    stim_ids = data["stimulus_id"].astype(str)
    stim_types = data["stimulus_type"].astype(str)
    n = len(stim_ids)

    sample_token_vision = next(iter(vision_token_by_id.values()))
    token_dim_vision = sample_token_vision.shape  # (T_tokens, D)
    sample_pool_vision = next(iter(vision_pool_by_id.values()))
    pool_dim_vision = sample_pool_vision.shape    # (D,)

    sample_token_text = next(iter(text_token_by_id.values())) if text_token_by_id else None
    sample_pool_text = next(iter(text_pool_by_id.values())) if text_pool_by_id else None
    token_dim_text = sample_token_text.shape if sample_token_text is not None else token_dim_vision
    pool_dim_text = sample_pool_text.shape if sample_pool_text is not None else pool_dim_vision

    target_clip_vision = np.zeros((n, *token_dim_vision), dtype=np.float32)
    target_clip_vision_pool = np.zeros((n, *pool_dim_vision), dtype=np.float32)
    target_clip_text = np.zeros((n, *token_dim_text), dtype=np.float32)
    target_clip_text_pool = np.zeros((n, *pool_dim_text), dtype=np.float32)
    has_clip_vision = np.zeros(n, dtype=bool)
    has_clip_text = np.zeros(n, dtype=bool)

    for i in range(n):
        sid = stim_ids[i]
        stype = stim_types[i]
        if stype in {"simple", "complex"}:
            if sid not in vision_token_by_id:
                raise KeyError(f"Missing vision feature for stimulus_id={sid}")
            target_clip_vision[i] = vision_token_by_id[sid]
            target_clip_vision_pool[i] = vision_pool_by_id[sid]
            has_clip_vision[i] = True
        elif stype == "conceptual":
            if sid not in text_token_by_id:
                raise KeyError(f"Missing text feature for stimulus_id={sid}")
            target_clip_text[i] = text_token_by_id[sid]
            target_clip_text_pool[i] = text_pool_by_id[sid]
            has_clip_text[i] = True
            # Some downstream code expects target_clip_vision to be filled even
            # for conceptual trials (to keep tensors a uniform shape). We use the
            # text pooled embedding broadcast to the vision token shape only if
            # the dims match; otherwise leave zeros and rely on has_clip_vision.

    data["target_clip_vision"] = target_clip_vision
    data["target_clip_vision_pool"] = target_clip_vision_pool
    data["target_clip_text"] = target_clip_text
    data["target_clip_text_pool"] = target_clip_text_pool
    data["has_clip_vision"] = has_clip_vision
    data["has_clip_text"] = has_clip_text

    np.savez_compressed(npz_path, **data)
    print(f"Merged CLIP features into {npz_path}")
    print(f"  target_clip_vision: {target_clip_vision.shape}")
    print(f"  target_clip_text:   {target_clip_text.shape}")
    print(f"  has_clip_vision:    {has_clip_vision.sum()}/{n}")
    print(f"  has_clip_text:      {has_clip_text.sum()}/{n}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract OpenCLIP features for NSD-Imagery stimuli and merge into imagery_trials.npz."
    )
    parser.add_argument("--paths", default="configs/paths.example.yaml")
    parser.add_argument("--subjects", type=str, default="1,2,3,4,5,6,7,8",
                        help="Comma-separated subject ids whose imagery_trials.npz will be updated.")
    parser.add_argument(
        "--raw-data-root", type=str, default="",
        help="Root containing nsddata/. Defaults to raw_nsd_root from paths config.",
    )
    parser.add_argument("--generative-models-path", type=Path, default=None)
    parser.add_argument("--openclip-version", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths.from_mapping(load_yaml(args.paths), ROOT)
    raw_root = Path(args.raw_data_root) if args.raw_data_root else Path(paths.raw_nsd_root)
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)
    rawtarget_dir = raw_root / "nsddata" / "experiments" / "nsdimagery" / "rawtargetimages"
    set_a_dir = rawtarget_dir / "setA"
    set_b_dir = rawtarget_dir / "setB"

    generative_models_path = args.generative_models_path or paths.generative_models_path
    if generative_models_path is None:
        raise ValueError("Set --generative-models-path or generative_models_path in paths config.")
    add_generative_models_to_path(Path(generative_models_path))
    from generative_models.sgm.modules.encoders.modules import (
        FrozenOpenCLIPImageEmbedder,
        FrozenOpenCLIPEmbedder2,
    )

    openclip_version = args.openclip_version or paths.openclip_version
    if openclip_version is None:
        raise ValueError("Set --openclip-version or openclip_version in paths config.")

    subjects = [int(s) for s in args.subjects.split(",") if s.strip()]
    print(f"Will update subjects: {subjects}")
    print(f"  arch:    {OPENCLIP_ARCH}")
    print(f"  version: {openclip_version}")
    if args.dry_run:
        return

    print("Loading vision embedder ...")
    image_embedder = FrozenOpenCLIPImageEmbedder(
        arch=OPENCLIP_ARCH,
        version=str(openclip_version),
        output_tokens=True,
        only_tokens=False,
    ).to(args.device).eval()

    print("Extracting vision features for set A and set B (12 stimuli) ...")
    set_a_ids = sorted(SET_A_FILES.keys())
    set_b_ids = sorted(SET_B_FILES.keys())
    a_tokens, a_pool = extract_vision_features(image_embedder, set_a_dir, set_a_ids, SET_A_FILES, args.device)
    b_tokens, b_pool = extract_vision_features(image_embedder, set_b_dir, set_b_ids, SET_B_FILES, args.device)
    vision_token_by_id = {sid: a_tokens[i] for i, sid in enumerate(set_a_ids)}
    vision_token_by_id.update({sid: b_tokens[i] for i, sid in enumerate(set_b_ids)})
    vision_pool_by_id = {sid: a_pool[i] for i, sid in enumerate(set_a_ids)}
    vision_pool_by_id.update({sid: b_pool[i] for i, sid in enumerate(set_b_ids)})
    del image_embedder
    torch.cuda.empty_cache()

    print("Loading text embedder ...")
    text_embedder = FrozenOpenCLIPEmbedder2(
        arch=OPENCLIP_ARCH,
        version=str(openclip_version),
        device=args.device,
        layer="last",
        always_return_pooled=True,
        legacy=False,
    ).to(args.device).eval()

    print("Extracting text features for set C (6 conceptual words) ...")
    c_tokens, c_pool = extract_text_features(text_embedder, SET_C_WORDS)
    text_token_by_id = {word: c_tokens[i] for i, word in enumerate(SET_C_WORDS)}
    text_pool_by_id = {word: c_pool[i] for i, word in enumerate(SET_C_WORDS)}
    del text_embedder
    torch.cuda.empty_cache()

    for subject in subjects:
        npz_path = paths.imagery_root / f"subj{subject:02d}" / "imagery_trials.npz"
        if not npz_path.exists():
            print(f"[skip] {npz_path} does not exist (run prepare_imagery_nsd.py first)")
            continue
        merge_into_imagery_trials(
            npz_path,
            vision_token_by_id,
            vision_pool_by_id,
            text_token_by_id,
            text_pool_by_id,
        )


if __name__ == "__main__":
    main()
