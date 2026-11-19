from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.data.nsd_sdxl_features import (
    OPENCLIP_ARCH,
    atomic_save_npy,
    build_image_loader,
    find_stimuli_hdf5,
    image_ids_path,
)
from src.utils.config import load_yaml
from src.utils.paths import ProjectPaths


def add_generative_models_to_path(path: Path) -> None:
    """Allow `from generative_models...` when config points at the package dir."""
    sys.path.append(str(path))
    if path.name == "generative_models":
        sys.path.append(str(path.parent))


def extract_split(embedder, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    token_chunks: list[torch.Tensor] = []
    pooled_chunks: list[torch.Tensor] = []
    for batch_idx, images in enumerate(loader):
        print(f"  image batch {batch_idx}")
        with torch.no_grad():
            tokens, pooled = embedder(images.to(device))
        token_chunks.append(tokens.detach().cpu().float())
        pooled_chunks.append(pooled.detach().cpu().float())
    return torch.cat(token_chunks, dim=0).numpy(), torch.cat(pooled_chunks, dim=0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SynBrain-style SDXL/OpenCLIP CLIP-Vision features.")
    parser.add_argument("--paths", default="configs/paths.example.yaml")
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--nsd-root", type=Path, default=None)
    parser.add_argument("--generative-models-path", type=Path, default=None)
    parser.add_argument("--openclip-version", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths.from_mapping(load_yaml(args.paths), ROOT)
    generative_models_path = args.generative_models_path or paths.generative_models_path
    if generative_models_path is None:
        raise ValueError("Set --generative-models-path or generative_models_path/generative-models-path in the paths config.")
    add_generative_models_to_path(Path(generative_models_path))
    from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder

    nsd_root = args.nsd_root or paths.raw_nsd_root
    if nsd_root is None:
        raise ValueError("Set --nsd-root or raw_nsd_root in the paths config.")
    openclip_version = args.openclip_version or paths.openclip_version
    if openclip_version is None:
        raise ValueError("Set --openclip-version or openclip_version in the paths config.")
    subj_dir = paths.nsd_root / f"subj{args.subject:02d}"
    stimuli_hdf5 = find_stimuli_hdf5(nsd_root)

    print(f"Subject {args.subject}")
    print("  arch:", OPENCLIP_ARCH)
    print("  version:", openclip_version)
    print("  stimuli:", stimuli_hdf5)
    if args.dry_run:
        for split in ["train", "test"]:
            ids = np.load(image_ids_path(subj_dir, split, args.subject))
            print(f"  {split} image ids:", ids.shape)
        return

    embedder = FrozenOpenCLIPImageEmbedder(
        arch=OPENCLIP_ARCH,
        version=str(openclip_version),
        output_tokens=True,
        only_tokens=False,
    ).to(args.device).eval()

    for split in ["train", "test"]:
        ids = np.load(image_ids_path(subj_dir, split, args.subject))
        loader = build_image_loader(stimuli_hdf5, ids, args.batch_size)
        tokens, pooled = extract_split(embedder, loader, args.device)
        atomic_save_npy(subj_dir / f"nsd_sdxl_clip_{split}_sub{args.subject}.npy", tokens.astype(np.float32))
        atomic_save_npy(subj_dir / f"nsd_sdxl_clip_pool_{split}_sub{args.subject}.npy", pooled.astype(np.float32))
        print(f"Saved {split} CLIP-Vision tokens {tokens.shape}, pooled {pooled.shape}")


if __name__ == "__main__":
    main()
