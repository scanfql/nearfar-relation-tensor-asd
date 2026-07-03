from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PatchMaxResult:
    score: float
    top_time: int
    top_freq: int
    channel_energy: np.ndarray


def patchmax_score(residual: np.ndarray, patch: tuple[int, int] = (8, 8), stride: tuple[int, int] | None = None) -> PatchMaxResult:
    """Return the maximum localized residual patch score over [T,F,C]."""
    x = np.abs(np.asarray(residual, dtype=np.float32))
    if x.ndim != 3:
        raise ValueError(f"Expected residual [T,F,C], got {x.shape}")
    pt, pf = patch
    st, sf = stride or patch
    best = (-np.inf, 0, 0, None)
    for t0 in range(0, max(1, x.shape[0] - pt + 1), st):
        for f0 in range(0, max(1, x.shape[1] - pf + 1), sf):
            block = x[t0 : t0 + pt, f0 : f0 + pf, :]
            ch = block.mean(axis=(0, 1))
            score = float(ch.mean())
            if score > best[0]:
                best = (score, t0, f0, ch)
    return PatchMaxResult(score=float(best[0]), top_time=int(best[1]), top_freq=int(best[2]), channel_energy=np.asarray(best[3]))


def channel_share(channel_energy: np.ndarray, channel_indices: list[int]) -> float:
    values = np.asarray(channel_energy, dtype=np.float64)
    denom = float(values.sum())
    if denom <= 1e-12:
        return 0.0
    return float(values[channel_indices].sum() / denom)
