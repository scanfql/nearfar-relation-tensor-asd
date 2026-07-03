from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MPCAModel:
    mean: np.ndarray
    basis_time: np.ndarray
    basis_freq: np.ndarray
    basis_channel: np.ndarray

    def residual(self, tensors: np.ndarray) -> np.ndarray:
        x = np.asarray(tensors, dtype=np.float32)
        single = x.ndim == 3
        if single:
            x = x[None]
        z = x - self.mean[None]
        core = np.einsum("ta,fb,ck,ntfc->nabc", self.basis_time, self.basis_freq, self.basis_channel, z, optimize=True)
        recon = np.einsum("ta,fb,ck,nabc->ntfc", self.basis_time, self.basis_freq, self.basis_channel, core, optimize=True)
        residual = (z - recon).astype(np.float32)
        return residual[0] if single else residual


def fit_mpca(tensors: np.ndarray, ranks: tuple[int, int, int] = (32, 32, 8)) -> MPCAModel:
    """Fit a deterministic MPCA/HOSVD normal subspace from [N,T,F,C]."""
    x = np.asarray(tensors, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected [N,T,F,C], got {x.shape}")
    mean = x.mean(axis=0).astype(np.float32)
    z = x - mean[None]
    n, t, f, c = z.shape
    rt, rf, rc = ranks
    bt = _leading_eigenvectors(z.reshape(n, t, -1), min(rt, t))
    bf = _leading_eigenvectors(np.transpose(z, (0, 2, 1, 3)).reshape(n, f, -1), min(rf, f))
    bc = _leading_eigenvectors(np.transpose(z, (0, 3, 1, 2)).reshape(n, c, -1), min(rc, c))
    return MPCAModel(mean=mean, basis_time=bt, basis_freq=bf, basis_channel=bc)


def _leading_eigenvectors(mode_stack: np.ndarray, rank: int) -> np.ndarray:
    cov = np.einsum("nxi,nyi->xy", mode_stack, mode_stack, optimize=True) / max(mode_stack.shape[0], 1)
    vals, vecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(vals)[::-1]
    return vecs[:, order[:rank]].astype(np.float32)


def hybridtail_score(residual: np.ndarray, source_residuals: np.ndarray | None = None, top_fraction: float = 0.05) -> float:
    """Conservative residual score: top-tail residual mean plus band-local evidence."""
    abs_r = np.abs(np.asarray(residual, dtype=np.float32))
    flat = abs_r.reshape(-1)
    k = max(1, int(round(flat.size * top_fraction)))
    top_tail = float(np.partition(flat, flat.size - k)[-k:].mean())
    bands = np.array_split(np.arange(abs_r.shape[1]), 4)
    band_vals = np.asarray([np.percentile(abs_r[:, b, :], 95) for b in bands], dtype=np.float32)
    band_score = float(band_vals.mean())
    raw = 0.5 * top_tail + 0.5 * band_score
    if source_residuals is None:
        return raw
    ref = np.asarray([hybridtail_score(r, None, top_fraction) for r in source_residuals], dtype=np.float32)
    return float((raw - ref.mean()) / max(ref.std(), 1e-8))
