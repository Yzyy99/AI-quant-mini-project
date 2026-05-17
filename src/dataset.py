from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class NpyTimeSeriesDataset(Dataset):
    """Memory-mapped dataset for X_split.npy / y_split.npy arrays."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        start: Optional[int] = None,
        stop: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.X = np.load(self.data_dir / f"X_{split}.npy", mmap_mode="r")
        self.y = np.load(self.data_dir / f"y_{split}.npy", mmap_mode="r")

        n = len(self.y)
        self.start = 0 if start is None else max(0, start)
        self.stop = n if stop is None else min(n, stop)
        if self.start >= self.stop:
            raise ValueError(f"empty slice for {split}: start={self.start}, stop={self.stop}, n={n}")

    def __len__(self) -> int:
        return self.stop - self.start

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        j = self.start + idx
        x = torch.from_numpy(np.array(self.X[j], dtype=np.float32, copy=True))
        y = torch.tensor(int(self.y[j]), dtype=torch.long)
        return x, y


def split_shape(data_dir: str | Path, split: str) -> tuple[int, int, int]:
    X = np.load(Path(data_dir) / f"X_{split}.npy", mmap_mode="r")
    return tuple(X.shape)
