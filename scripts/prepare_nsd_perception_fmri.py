from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.io as spio

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.utils.config import load_yaml
from src.utils.paths import ProjectPaths


def load_expdesign(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return spio.loadmat(path, squeeze_me=True, struct_as_record=False)


def build_ppbe_split(expdesign: dict, subject: int, sessions: int) -> tuple[list[int], list[int], dict[int, list[int]], dict[int, list[int]]]:
    subjectim = expdesign["subjectim"]
    masterordering = np.asarray(expdesign["masterordering"]).reshape(-1)
    num_trials = sessions * 750
    if num_trials > len(masterordering):
        raise ValueError(f"Requested {num_trials} trials, but masterordering only has {len(masterordering)}")

    train_trials: dict[int, list[int]] = {}
    test_trials: dict[int, list[int]] = {}
    for trial_idx in range(num_trials):
        master_id = int(masterordering[trial_idx])
        nsd_id = int(subjectim[subject - 1, master_id - 1]) - 1
        if master_id > 1000:
            train_trials.setdefault(nsd_id, []).append(trial_idx)
        else:
            test_trials.setdefault(nsd_id, []).append(trial_idx)

    train_image_ids = list(train_trials.keys())
    test_image_ids = list(test_trials.keys())
    return train_image_ids, test_image_ids, train_trials, test_trials


def load_session_betas(betas_dir: Path, mask: np.ndarray, sessions: int) -> tuple[np.ndarray, int]:
    """Load up to `sessions` per-session beta files.

    NSD subjects do not all complete the same number of sessions (subj 1/2/5/7
    completed 40, others completed fewer). If a session file is missing we stop
    early instead of crashing, and return both the stacked array (cropped to
    the actual number of loaded sessions) and that count.
    """
    num_voxels = int(mask.sum())
    fmri = np.empty((sessions * 750, num_voxels), dtype=np.float32)
    loaded_sessions = 0
    for session_idx in range(sessions):
        beta_path = betas_dir / f"betas_session{session_idx + 1:02d}.nii.gz"
        if not beta_path.exists():
            print(
                f"  Session {session_idx + 1:02d} not found at {beta_path}; "
                f"stopping after {loaded_sessions} sessions."
            )
            break
        beta_img = nib.load(str(beta_path)).get_fdata().astype(np.float32)
        betas = beta_img[mask].transpose()
        fmri[session_idx * 750 : (session_idx + 1) * 750] = betas / 2000.0
        loaded_sessions += 1
        print(f"Loaded session {session_idx + 1:02d}: {betas.shape}")
    return fmri[: loaded_sessions * 750], loaded_sessions


def gather_by_image(
    fmri_by_trial: np.ndarray,
    image_ids: list[int],
    trial_lookup: dict[int, list[int]],
    expected_reps: int,
) -> tuple[np.ndarray, list[int]]:
    """Stack per-image, per-rep fMRI into a fixed-shape array.

    NSD is designed so each training image is shown 3 times and each test image
    once, but some subjects do not complete all 40 sessions, leaving a small
    number of images with fewer reps than expected. To keep a fixed-shape
    output we drop those incomplete images here and return both the stacked
    array and the surviving image_id list (in the same order).
    """
    voxels = fmri_by_trial.shape[1]
    kept_ids: list[int] = []
    for image_id in image_ids:
        if len(trial_lookup[image_id]) >= expected_reps:
            kept_ids.append(image_id)
    out = np.empty((len(kept_ids), expected_reps, voxels), dtype=np.float32)
    for i, image_id in enumerate(kept_ids):
        trial_indices = sorted(trial_lookup[image_id])[:expected_reps]
        out[i] = fmri_by_trial[trial_indices]
    return out, kept_ids


def load_captions(annots_path: Path, image_ids: list[int]) -> np.ndarray:
    if not annots_path.exists():
        raise FileNotFoundError(annots_path)
    annots = np.load(annots_path, allow_pickle=True)
    captions = np.empty((len(image_ids), annots.shape[1]), dtype=annots.dtype)
    for i, image_id in enumerate(image_ids):
        captions[i] = annots[int(image_id)]
    return captions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fixed PPBE-style NSD perception fMRI/id/caption files.")
    parser.add_argument("--paths", default="configs/paths.example.yaml")
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--nsd-root", type=Path, default=None, help="Root containing nsddata/ and nsddata_betas/.")
    parser.add_argument("--annots", type=Path, default='data/raw_nsd/annots/COCO_73k_annots_curated.npy', help="COCO_73k_annots_curated.npy.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths.from_mapping(load_yaml(args.paths), ROOT)
    subject = args.subject
    nsd_root = args.nsd_root or paths.raw_nsd_root
    if nsd_root is None:
        raise ValueError("Set --nsd-root or raw_nsd_root in the paths config.")

    expdesign = nsd_root / "nsddata" / "experiments" / "nsd" / "nsd_expdesign.mat"
    ppdata_root = nsd_root / "nsddata" / "ppdata"
    betas_root = nsd_root / "nsddata_betas" / "ppdata"
    roi_dir = ppdata_root / f"subj{subject:02d}" / "func1pt8mm" / "roi"
    betas_dir = betas_root / f"subj{subject:02d}" / "func1pt8mm" / "betas_fithrf_GLMdenoise_RR"
    mask_path = roi_dir / "nsdgeneral.nii.gz"
    out_dir = paths.nsd_root / f"subj{subject:02d}"

    design = load_expdesign(expdesign)
    train_ids, test_ids, train_trials, test_trials = build_ppbe_split(design, subject, args.sessions)
    mask = nib.load(str(mask_path)).get_fdata() > 0

    print(f"Subject {subject}")
    print("  expdesign:", expdesign)
    print("  mask:", mask_path, "voxels:", int(mask.sum()))
    print("  betas:", betas_dir)
    print("  train images:", len(train_ids), "test images:", len(test_ids))
    print("  output:", out_dir)
    if args.dry_run:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    fmri_by_trial, loaded_sessions = load_session_betas(betas_dir, mask, args.sessions)
    if loaded_sessions < args.sessions:
        print(
            f"  Subject {subject} only has {loaded_sessions} sessions on disk "
            f"(requested {args.sessions}); recomputing the train/test split for {loaded_sessions} sessions."
        )
        train_ids, test_ids, train_trials, test_trials = build_ppbe_split(design, subject, loaded_sessions)

    train_fmri, kept_train_ids = gather_by_image(fmri_by_trial, train_ids, train_trials, expected_reps=3)
    test_fmri, kept_test_ids = gather_by_image(fmri_by_trial, test_ids, test_trials, expected_reps=3)

    dropped_train = len(train_ids) - len(kept_train_ids)
    dropped_test = len(test_ids) - len(kept_test_ids)
    if dropped_train or dropped_test:
        print(
            f"  Dropped {dropped_train} train and {dropped_test} test images that lacked"
            f" all 3 repetitions in {loaded_sessions} sessions."
        )

    np.save(out_dir / f"nsd_train_fmri_scale_sub{subject}.npy", train_fmri)
    np.save(out_dir / f"nsd_test_fmri_scale_sub{subject}.npy", test_fmri)
    np.save(out_dir / f"nsd_train_fmri_all_scale_sub{subject}.npy", train_fmri.reshape(-1, train_fmri.shape[-1]))
    np.save(out_dir / f"nsd_test_fmri_all_scale_sub{subject}.npy", test_fmri.reshape(-1, test_fmri.shape[-1]))
    np.save(out_dir / f"nsd_train_image_id_sub{subject}.npy", np.asarray(kept_train_ids, dtype=np.int64))
    np.save(out_dir / f"nsd_test_image_id_sub{subject}.npy", np.asarray(kept_test_ids, dtype=np.int64))

    np.save(out_dir / f"nsd_train_cap_sub{subject}.npy", load_captions(args.annots, kept_train_ids))
    np.save(out_dir / f"nsd_test_cap_sub{subject}.npy", load_captions(args.annots, kept_test_ids))

    print("Saved:")
    print("  train_fmri:", train_fmri.shape)
    print("  test_fmri:", test_fmri.shape)
    print("  train_fmri_all:", (train_fmri.shape[0] * train_fmri.shape[1], train_fmri.shape[2]))
    print("  test_fmri_all:", (test_fmri.shape[0] * test_fmri.shape[1], test_fmri.shape[2]))


if __name__ == "__main__":
    main()
