from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


OPENCLIP_ARCH = "ViT-bigG-14"


class NSDImageDataset(Dataset):
    def __init__(self, stimuli_hdf5: Path, image_ids: np.ndarray) -> None:
        self.stimuli_hdf5 = stimuli_hdf5
        self.image_ids = np.asarray(image_ids, dtype=np.int64)
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return len(self.image_ids)

    def _images(self):
        if self._handle is None:
            self._handle = h5py.File(self.stimuli_hdf5, "r")
        return self._handle["imgBrick"]

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = self._images()[int(self.image_ids[idx])]
        return torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    tmp.replace(path)


def find_stimuli_hdf5(nsd_root: Path) -> Path:
    path = nsd_root / "nsddata_stimuli" / "stimuli" / "nsd" / "nsd_stimuli.hdf5"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def image_ids_path(subj_dir: Path, split: str, subject: int) -> Path:
    return subj_dir / f"nsd_{split}_image_id_sub{subject}.npy"


def captions_path(subj_dir: Path, split: str, subject: int) -> Path:
    return subj_dir / f"nsd_{split}_cap_sub{subject}.npy"


def load_texts(path: Path) -> list[list[str]]:
    captions = np.load(path, allow_pickle=True)
    if captions.ndim == 1:
        captions = captions[:, None]
    out: list[list[str]] = []
    for row in captions:
        texts = [str(item) for item in row if str(item).strip()]
        out.append(texts or [""])
    return out


def build_image_loader(stimuli_hdf5: Path, image_ids: np.ndarray, batch_size: int) -> DataLoader:
    dataset = NSDImageDataset(stimuli_hdf5, image_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

