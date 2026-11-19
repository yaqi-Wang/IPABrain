from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.data.nsd_sdxl_features import OPENCLIP_ARCH, atomic_save_npy, captions_path, load_texts
from src.utils.config import load_yaml
from src.utils.paths import ProjectPaths


def add_generative_models_to_path(path: Path) -> None:
    """Allow `from generative_models...` when config points at the package dir."""
    sys.path.append(str(path))
    if path.name == "generative_models":
        sys.path.append(str(path.parent))


def extract_text_split(text_embedder, texts_by_image: list[list[str]], batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    token_rows: list[np.ndarray] = []
    pooled_rows: list[np.ndarray] = []
    for start in range(0, len(texts_by_image), batch_size):
        batch_texts = texts_by_image[start : start + batch_size]
        flat: list[str] = []
        counts: list[int] = []
        for texts in batch_texts:
            flat.extend(texts)
            counts.append(len(texts))
        print(f"  text images {start}:{start + len(batch_texts)} captions={len(flat)}")
        with torch.no_grad():
            tokens, pooled = text_embedder(flat)
        tokens_np = tokens.detach().cpu().float().numpy()
        pooled_np = pooled.detach().cpu().float().numpy()
        offset = 0
        for count in counts:
            token_rows.append(tokens_np[offset : offset + count].mean(axis=0))
            pooled_rows.append(pooled_np[offset : offset + count].mean(axis=0))
            offset += count
    return np.stack(token_rows).astype(np.float32), np.stack(pooled_rows).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SDXL/OpenCLIP caption CLIP-Text features.")
    parser.add_argument("--paths", default="configs/paths.example.yaml")
    parser.add_argument("--subject", type=int, required=True)
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
    from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPEmbedder2

    openclip_version = args.openclip_version or paths.openclip_version
    if openclip_version is None:
        raise ValueError("Set --openclip-version or openclip_version in the paths config.")
    subj_dir = paths.nsd_root / f"subj{args.subject:02d}"

    print(f"Subject {args.subject}")
    print("  arch:", OPENCLIP_ARCH)
    print("  version:", openclip_version)
    if args.dry_run:
        for split in ["train", "test"]:
            texts = load_texts(captions_path(subj_dir, split, args.subject))
            print(f"  {split} caption rows:", len(texts))
        return

    text_embedder = FrozenOpenCLIPEmbedder2(
        arch=OPENCLIP_ARCH,
        version=str(openclip_version),
        device=args.device,
        layer="last",
        always_return_pooled=True,
        legacy=False,
    ).to(args.device).eval()

    for split in ["train", "test"]:
        texts = load_texts(captions_path(subj_dir, split, args.subject))
        tokens, pooled = extract_text_split(text_embedder, texts, args.batch_size)
        atomic_save_npy(subj_dir / f"nsd_sdxl_text_token_{split}_sub{args.subject}.npy", tokens)
        atomic_save_npy(subj_dir / f"nsd_sdxl_text_pool_{split}_sub{args.subject}.npy", pooled)
        atomic_save_npy(subj_dir / f"nsd_clip_text_{split}_sub{args.subject}.npy", pooled)
        print(f"Saved {split} CLIP-Text tokens {tokens.shape}, pooled {pooled.shape}")


if __name__ == "__main__":
    main()
