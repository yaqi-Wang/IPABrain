from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.io as sio

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.utils.config import load_yaml
from src.utils.paths import ProjectPaths


# 12 runs of NSD-Imagery, decoded via designmatrixGLMsingle column ranges.
# Confirmed empirically: each cell of designmatrixGLMsingle has 36 cols, of which
# only one 6-col block is active per run; trial counts match (720 total).
RUN_LAYOUT: list[dict] = [
    {"run": 1,  "task": "vision",    "stim_set": "A", "col_offset": 0,  "n_trials": 48},
    {"run": 2,  "task": "attention", "stim_set": "A", "col_offset": 6,  "n_trials": 96},
    {"run": 3,  "task": "imagery",   "stim_set": "A", "col_offset": 18, "n_trials": 48},
    {"run": 4,  "task": "vision",    "stim_set": "B", "col_offset": 0,  "n_trials": 48},
    {"run": 5,  "task": "attention", "stim_set": "B", "col_offset": 6,  "n_trials": 96},
    {"run": 6,  "task": "imagery",   "stim_set": "B", "col_offset": 24, "n_trials": 48},
    {"run": 7,  "task": "vision",    "stim_set": "C", "col_offset": 0,  "n_trials": 48},
    {"run": 8,  "task": "attention", "stim_set": "C", "col_offset": 6,  "n_trials": 96},
    {"run": 9,  "task": "imagery",   "stim_set": "C", "col_offset": 30, "n_trials": 48},
    {"run": 10, "task": "imagery",   "stim_set": "A", "col_offset": 18, "n_trials": 48},
    {"run": 11, "task": "imagery",   "stim_set": "B", "col_offset": 24, "n_trials": 48},
    {"run": 12, "task": "imagery",   "stim_set": "C", "col_offset": 30, "n_trials": 48},
]

# The cue letter ordering inside each 6-col block, in alphabetical order
# (matches condit_list of vis*/img*_dm.mat).
SET_CUE_ORDER: dict[str, list[str]] = {
    "A": ["E", "H", "L", "P", "R", "V"],
    "B": ["B", "C", "D", "K", "T", "W"],
    "C": ["F", "M", "N", "S", "Y", "Z"],
}

STIMULUS_TYPE_BY_SET: dict[str, str] = {
    "A": "simple",
    "B": "complex",
    "C": "conceptual",
}


def load_pair_lists(experiment_dir: Path) -> dict[str, dict[str, str]]:
    """Return mapping: stim_set -> {cue_letter -> stimulus_id_string}."""
    out: dict[str, dict[str, str]] = {}
    for set_name in ("A", "B", "C"):
        mat = sio.loadmat(experiment_dir / f"{set_name}_pair_list.mat")
        cue_to_stim: dict[str, str] = {}
        for row in mat["pair_list"]:
            stim_name = str(row[1][0])
            cue = str(row[2][0])
            stem = stim_name.replace(".png", "")
            cue_to_stim[cue] = stem
        out[set_name] = cue_to_stim
    return out


def decode_trials(experiment_dir: Path) -> tuple[np.ndarray, list[dict]]:
    """Decode 720 trials from designmatrixGLMsingle.mat.

    Returns:
        order: (720,) int64 array, trial index in beta dimension (always 0..719).
        meta:  list of dicts with run, task, stim_set, cue_letter, stimulus_id,
               trial_in_run, frame_index for each trial.
    """
    design_path = experiment_dir / "designmatrixGLMsingle.mat"
    mat = sio.loadmat(design_path)
    cells = mat["stimulus"][0]
    if len(cells) != 12:
        raise ValueError(f"Expected 12 run cells, got {len(cells)}")

    pair_lists = load_pair_lists(experiment_dir)

    meta: list[dict] = []
    global_idx = 0
    for layout, cell in zip(RUN_LAYOUT, cells):
        if int(cell.sum()) != layout["n_trials"]:
            raise ValueError(
                f"Run {layout['run']} expected {layout['n_trials']} trials, "
                f"got {int(cell.sum())}"
            )
        # GLMsingle marks each trial at a single timepoint (one row per trial).
        # Sort by ascending frame_index, then within frame by column to keep
        # deterministic ordering.
        nz_rows, nz_cols = np.nonzero(cell)
        # Stable sort by (row, col)
        order = np.lexsort((nz_cols, nz_rows))
        nz_rows = nz_rows[order]
        nz_cols = nz_cols[order]
        if len(nz_rows) != layout["n_trials"]:
            raise ValueError(
                f"Run {layout['run']}: trial count mismatch after sort"
            )

        cues_in_block = SET_CUE_ORDER[layout["stim_set"]]
        col_offset = layout["col_offset"]
        block_len = 6
        for trial_in_run, (frame_idx, abs_col) in enumerate(zip(nz_rows, nz_cols)):
            local_col = int(abs_col) - col_offset
            if not 0 <= local_col < block_len * (
                3 if layout["task"] == "attention" else (2 if layout["task"] == "vision" else 1)
            ):
                raise ValueError(
                    f"Run {layout['run']} unexpected col {abs_col} (offset {col_offset})"
                )
            cue_index_in_block = local_col % block_len  # 0..5 (handles vision/attention multi-block)
            condition_index = local_col // block_len     # 0=match, 1=mismatch (vision), 2=NaN (attention)
            cue_letter = cues_in_block[cue_index_in_block]
            stim_id = pair_lists[layout["stim_set"]][cue_letter]
            meta.append({
                "global_trial_idx": global_idx,
                "run": layout["run"],
                "task": layout["task"],
                "stim_set": layout["stim_set"],
                "stimulus_type": STIMULUS_TYPE_BY_SET[layout["stim_set"]],
                "cue_letter": cue_letter,
                "stimulus_id": stim_id,
                "frame_index": int(frame_idx),
                "trial_in_run": trial_in_run,
                "condition_index": condition_index,
            })
            global_idx += 1

    if global_idx != 720:
        raise ValueError(f"Expected 720 trials total, decoded {global_idx}")
    return np.arange(720, dtype=np.int64), meta


def load_betas(hdf5_path: Path, scale: float = 300.0) -> np.ndarray:
    """Load NSD-Imagery betas as float32 trials × X × Y × Z.

    NSD stores betas as int16 ×scale (default 300). We divide by scale to
    recover the GLM coefficients in PSC-like units.
    """
    with h5py.File(hdf5_path, "r") as handle:
        betas = handle["betas"][:].astype(np.float32) / float(scale)
    if betas.shape[0] != 720:
        raise ValueError(f"Expected 720 trials, got {betas.shape[0]}")
    return betas


def load_nsdgeneral_mask(mask_path: Path) -> np.ndarray:
    """Load nsdgeneral mask and return as a flat boolean mask aligned with betas.

    NIfTI shape is (X, Y, Z) = (81, 104, 83) but betas are (T, Z, Y, X) =
    (720, 83, 104, 81). We transpose the mask to (Z, Y, X) before flattening.
    """
    mask = nib.load(str(mask_path)).get_fdata()
    mask = (mask > 0).astype(bool)
    mask = np.transpose(mask, (2, 1, 0))  # XYZ -> ZYX, matches betas spatial layout
    return mask


def apply_mask(betas: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply 3D mask to (T, Z, Y, X) betas, returning (T, n_voxels) float32."""
    if betas.shape[1:] != mask.shape:
        raise ValueError(
            f"Mask shape {mask.shape} incompatible with betas spatial dims {betas.shape[1:]}"
        )
    flat = betas.reshape(betas.shape[0], -1)
    mask_flat = mask.reshape(-1)
    return flat[:, mask_flat]


def zscore_per_voxel(fmri: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-voxel z-score across all retained trials.

    Matches the SynBrain / NSD-perception convention: each voxel is centered
    and scaled by its trial-wise mean and std.
    """
    mean = fmri.mean(axis=0, keepdims=True)
    std = fmri.std(axis=0, keepdims=True)
    return ((fmri - mean) / np.maximum(std, eps)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode raw NSD-Imagery betas into a per-subject imagery_trials.npz."
    )
    parser.add_argument("--paths", default="configs/paths.example.yaml")
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument(
        "--raw-data-root",
        type=str,
        default="",
        help="Root containing nsddata/ and nsddata_betas/. Defaults to raw_nsd_root from paths config.",
    )
    parser.add_argument("--include-attention", action="store_true",
                        help="Keep attention trials (288/subject). Default: drop them per the NSD-Imagery paper recommendation.")
    parser.add_argument("--include-conceptual-vision", action="store_true",
                        help="Keep set-C vision trials (48/subject). Default: drop them per the paper recommendation.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths.from_mapping(load_yaml(args.paths), ROOT)
    raw_root = Path(args.raw_data_root) if args.raw_data_root else Path(paths.raw_nsd_root)
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)

    experiment_dir = raw_root / "nsddata" / "experiments" / "nsdimagery"
    betas_dir = (
        raw_root / "nsddata_betas" / "ppdata" / f"subj{args.subject:02d}"
        / "func1pt8mm" / "nsdimagerybetas_fithrf_GLMdenoise_RR"
    )
    mask_path = (
        raw_root / "nsddata" / "ppdata" / f"subj{args.subject:02d}"
        / "func1pt8mm" / "roi" / "nsdgeneral.nii.gz"
    )
    out_dir = paths.imagery_root / f"subj{args.subject:02d}"
    out_path = out_dir / "imagery_trials.npz"

    print(f"Subject {args.subject}")
    print(f"  experiment dir: {experiment_dir}")
    print(f"  betas dir:      {betas_dir}")
    print(f"  mask path:      {mask_path}")
    print(f"  output:         {out_path}")

    if args.dry_run:
        return

    print("Decoding 720 trials from designmatrixGLMsingle.mat ...")
    _, meta = decode_trials(experiment_dir)

    print("Loading nsdgeneral mask ...")
    mask = load_nsdgeneral_mask(mask_path)
    print(f"  mask voxels: {int(mask.sum())}")

    print("Loading betas ...")
    betas = load_betas(betas_dir / "betas_nsdimagery.hdf5")
    print(f"  betas shape: {betas.shape}, dtype: {betas.dtype}")

    print("Applying mask ...")
    fmri_all = apply_mask(betas, mask)
    del betas
    print(f"  fmri shape: {fmri_all.shape}")

    keep = np.ones(720, dtype=bool)
    if not args.include_attention:
        keep &= np.array([m["task"] != "attention" for m in meta])
    if not args.include_conceptual_vision:
        keep &= np.array([
            not (m["task"] == "vision" and m["stim_set"] == "C") for m in meta
        ])
    n_keep = int(keep.sum())
    print(f"  kept trials: {n_keep}/{len(meta)}")

    print("Per-voxel z-score across kept trials ...")
    fmri_kept = zscore_per_voxel(fmri_all[keep])

    kept_meta = [m for m, k in zip(meta, keep) if k]
    n = len(kept_meta)

    payload = {
        "imagery_fmri": fmri_kept,
        "subject_id": np.full(n, args.subject, dtype=np.int64),
        "trial_id": np.arange(n, dtype=np.int64),
        "stimulus_id": np.array([m["stimulus_id"] for m in kept_meta], dtype=object),
        "stimulus_type": np.array([m["stimulus_type"] for m in kept_meta], dtype=object),
        "task_type": np.array([m["task"] for m in kept_meta], dtype=object),
        "stim_set": np.array([m["stim_set"] for m in kept_meta], dtype=object),
        "cue_letter": np.array([m["cue_letter"] for m in kept_meta], dtype=object),
        "run": np.array([m["run"] for m in kept_meta], dtype=np.int64),
        "global_trial_idx": np.array([m["global_trial_idx"] for m in kept_meta], dtype=np.int64),
        "condition_index": np.array([m["condition_index"] for m in kept_meta], dtype=np.int64),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"Wrote {out_path}")
    print("  fmri:", fmri_kept.shape, fmri_kept.dtype)
    print("  trial mix:", {
        f"{m['task']}/{m['stimulus_type']}": int((np.array([
            (mm['task'], mm['stimulus_type']) for mm in kept_meta
        ]) == [m['task'], m['stimulus_type']]).all(axis=1).sum())
        for m in kept_meta[::max(1, n//6)][:6]
    })


if __name__ == "__main__":
    main()
