from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RobustChannelStats:
    median: np.ndarray
    iqr: np.ndarray
    eps: float = 1e-6

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        return ((arr - self.median.reshape(1, 1, -1)) / np.maximum(self.iqr.reshape(1, 1, -1), self.eps)).astype(
            np.float32
        )


def fit_source_normal_stats(tensors: np.ndarray, eps: float = 1e-6) -> RobustChannelStats:
    """Fit channel-wise source-normal robust statistics from [N,T,F,C]."""
    x = np.asarray(tensors, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected [N,T,F,C] tensors, got {x.shape}")
    flat = x.reshape(-1, x.shape[-1])
    median = np.median(flat, axis=0).astype(np.float32)
    q75 = np.percentile(flat, 75, axis=0)
    q25 = np.percentile(flat, 25, axis=0)
    iqr = np.maximum(q75 - q25, eps).astype(np.float32)
    return RobustChannelStats(median=median, iqr=iqr, eps=eps)
