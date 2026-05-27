#!/usr/bin/env python
"""V39 Frontend Pre-screen: evaluate three candidate 3-view frontends
(modulation, cepstral, transient) for anomaly separation, redundancy,
and gate clearance.  Offline-only: no model training.

Produces:
  results/v39_frontend_prescreen/
    summary.csv
    per_machine.csv
    redundancy.csv
    figures/          (if --save_panels)

  docs/CTAF_V39_FRONTEND_PRESCREEN.md
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa

from src.data.dataset import scan_dataset, filter_items, split_train_test
from src.features.logmel import load_audio, extract_logmel, compute_mfcc_from_logmel
from src.features.deltas import compute_delta, compute_delta_delta
from src.features.edge import compute_edge
from src.features.enhanced import compute_enhanced


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MACHINES = "gearbox,slider,toycar,toytrain,fan"
DEFAULT_FRONTENDS = "modulation_3view,cepstral_3view,transient_3view"


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V39 Frontend Pre-screen Diagnostic")
    p.add_argument("--data_root", required=True, type=str,
                   help="Root of DCASE dataset (containing train/test audio)")
    p.add_argument("--reference_result_dir", required=True, type=str,
                   help="V37 seed panel root, e.g. results/v37_.../seed_13711")
    p.add_argument("--reference_experiment", required=True, type=str,
                   help="Experiment name for NPZ lookup, e.g. full_5view")
    p.add_argument("--reference_score_mode", required=True, type=str,
                   help="Score mode for NPZ lookup, e.g. latent_summary_knn")
    p.add_argument("--machines", default=DEFAULT_MACHINES, type=str,
                   help="Comma-separated machine list")
    p.add_argument("--frontends", default=DEFAULT_FRONTENDS, type=str,
                   help="Comma-separated frontend list")
    p.add_argument("--normal_tail_q_low", default=0.95, type=float,
                   help="Lower quantile for normal-tail band (default 0.95)")
    p.add_argument("--normal_tail_q_high", default=0.99, type=float,
                   help="Upper quantile for normal-tail band (default 0.99)")
    p.add_argument("--pass_robust_d", default=0.30, type=float,
                   help="Gate A threshold on robust_d_missed (default 0.30)")
    p.add_argument("--max_redundancy_r", default=0.85, type=float,
                   help="Gate C threshold on max Pearson r (default 0.85)")
    p.add_argument("--min_pass_machines", default=2, type=int,
                   help="Gate B: minimum machines passing gate A (default 2)")
    p.add_argument("--detected_sanity_min_robust_d", default=0.0, type=float,
                   help="Gate D: robust_d_detected must be > this (default 0.0)")
    p.add_argument("--out_dir", default="results/v39_frontend_prescreen", type=str,
                   help="Output directory for CSV tables")
    p.add_argument("--out_md", default="docs/CTAF_V39_FRONTEND_PRESCREEN.md", type=str,
                   help="Output markdown report path")
    p.add_argument("--save_panels", action="store_true",
                   help="Save per-machine visual example panels")
    p.add_argument("--no_train", action="store_true",
                   help="No training mode (descriptive flag)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GroupData:
    """Carrier for a group of test-sample indices."""
    indices: list
    paths: list
    labels: list
    domains: list

    @property
    def size(self) -> int:
        return len(self.indices)


@dataclass
class FrontendResult:
    frontend: str
    machine: str
    robust_d_missed: float
    robust_d_detected: float
    cohens_d: float
    redundancy_max_r: float
    gate_a: bool  # robust_d_missed >= pass_robust_d
    gate_b: bool  # >= min_pass_machines
    gate_c: bool  # r_max <= max_redundancy_r
    gate_d: bool  # robust_d_detected > detected_sanity_min_robust_d
    n_missed: int = 0
    n_detected: int = 0
    n_normal_tail: int = 0
    n_total_anomalies: int = 0
    flag: str = ""  # edge-case flags


@dataclass
class RedundancyEntry:
    frontend: str
    view_name: str
    max_abs_r: float
    correlated_baseline: str = ""


# ---------------------------------------------------------------------------
# InputNormalizer
# ---------------------------------------------------------------------------

class InputNormalizer:
    """Train-set mean/std normalizer for a frontend's multi-view tensors.

    Replicates TensorNormalizer logic but operates as a standalone utility
    on raw input tensors [V, F, T].

    Attributes
    ----------
    eps : float
    mean : np.ndarray | None   shape [V, 1, 1]
    std  : np.ndarray | None   shape [V, 1, 1]
    fitted : bool
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.fitted = False

    def fit(self, tensors: List[np.ndarray]) -> None:
        """Fit on a list of [V, F, T] tensors."""
        if not tensors:
            return
        stacked = np.stack(tensors, axis=0)  # [N, V, F, T]
        self.mean = stacked.mean(axis=(0, 2, 3), keepdims=True)  # [1, V, 1, 1]
        self.std = stacked.std(axis=(0, 2, 3), keepdims=True)    # [1, V, 1, 1]
        self.mean = self.mean.squeeze(0)  # [V, 1, 1]
        self.std = self.std.squeeze(0)    # [V, 1, 1]
        self.fitted = True

    def transform(self, tensor: np.ndarray) -> np.ndarray:
        """Normalize a single [V, F, T] tensor."""
        if not self.fitted:
            raise RuntimeError("InputNormalizer not fitted")
        return (tensor - self.mean) / (self.std + self.eps)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def align_time(x: np.ndarray, t_ref: int) -> np.ndarray:
    """Crop or edge-pad to reference frame count."""
    if x.shape[1] == t_ref:
        return x.astype(np.float32)
    if x.shape[1] > t_ref:
        return x[:, :t_ref].astype(np.float32)
    pad = t_ref - x.shape[1]
    return np.pad(x, ((0, 0), (0, pad)), mode="edge").astype(np.float32)


def moving_average_time(x: np.ndarray, win: int = 11) -> np.ndarray:
    """Per-row moving average. [F, T] -> [F, T]."""
    if win <= 1:
        return x.astype(np.float32)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), 1, x
    ).astype(np.float32)


def _upsample_freq(x: np.ndarray, n_target: int = 128) -> np.ndarray:
    """NaN-safe linear upsample along frequency axis to n_target bins.

    Key fix for V46:
      - when n_src == 1 (e.g. periodicity map is [1, T]), interpolate-by-line
        is ill-posed; repeat the single row instead of calling a 1-point
        interpolator.
      - use boundary-safe np.interp rather than extrapolating beyond the
        source range.
      - sanitize non-finite inputs/outputs so the frontend is implementation
        valid before Stage-0 scientific evaluation.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"_upsample_freq expects 2D input, got shape {x.shape}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n_src, t = x.shape
    if n_src == n_target:
        return x.astype(np.float32, copy=False)
    if n_src <= 1:
        return np.repeat(x[:1], repeats=n_target, axis=0).astype(np.float32, copy=False)

    src_axis = np.linspace(0.0, 1.0, n_src, dtype=np.float32)
    tgt_axis = np.linspace(0.0, 1.0, n_target, dtype=np.float32)
    out = np.empty((n_target, t), dtype=np.float32)
    for frame in range(t):
        col = np.nan_to_num(x[:, frame], nan=0.0, posinf=0.0, neginf=0.0)
        out[:, frame] = np.interp(tgt_axis, src_axis, col, left=col[0], right=col[-1]).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _compute_periodicity_from_mfcc(mfcc_high: np.ndarray) -> np.ndarray:
    """Compute autocorrelation-based periodicity from high-order MFCC.

    Parameters
    ----------
    mfcc_high : np.ndarray  shape [>=8, T]
        High-order MFCC coefficients, e.g. MFCC[20:].

    Returns
    -------
    periodicity : np.ndarray  shape [T]
        Per-frame periodicity (max autocorrelation lag-1 to lag-8).
    """
    n_coeff, n_frames = mfcc_high.shape
    periodicity = np.zeros(n_frames, dtype=np.float32)
    for t_idx in range(n_frames):
        col = mfcc_high[:, t_idx]  # [n_coeff]
        col_centered = col - np.mean(col)
        denom = np.dot(col_centered, col_centered)
        if denom < 1e-12:
            periodicity[t_idx] = 0.0
            continue
        ac = np.correlate(col_centered, col_centered, mode="full")
        mid = len(ac) // 2
        # lags 1..8 correspond to ac[mid+1 : mid+9]
        end = min(mid + 9, len(ac))
        lags = ac[mid + 1:end]
        periodicity[t_idx] = np.max(lags) / denom
    return periodicity


def compute_robust_d(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[float, float, float, float]:
    """Compute robust Cohen's d between two distance distributions.

    robust_d = (median(A) - median(B)) /
               (0.5*(IQR(A) + IQR(B)) + eps)

    Returns
    -------
    median_a, median_b, iqr_a, iqr_b, robust_d
    """
    if len(dist_a) == 0 or len(dist_b) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    med_a = float(np.median(dist_a))
    med_b = float(np.median(dist_b))
    iqr_a = float(np.subtract(*np.percentile(dist_a, [75, 25])))
    iqr_b = float(np.subtract(*np.percentile(dist_b, [75, 25])))
    denom = 0.5 * (iqr_a + iqr_b) + eps
    if denom <= 0:
        return med_a, med_b, iqr_a, iqr_b, 0.0
    rd = (med_a - med_b) / denom
    return med_a, med_b, iqr_a, iqr_b, float(rd)


def compute_cohens_d(dist_a: np.ndarray, dist_b: np.ndarray,
                     eps: float = 1e-8) -> float:
    """Standard Cohen's d = (mean(A) - mean(B)) / pooled_std."""
    if len(dist_a) == 0 or len(dist_b) == 0:
        return 0.0
    pooled_var = (np.var(dist_a) + np.var(dist_b)) / 2.0
    denom = np.sqrt(max(pooled_var, eps))
    if denom <= 0:
        return 0.0
    return float((np.mean(dist_a) - np.mean(dist_b)) / denom)


# ---------------------------------------------------------------------------
# Frontend builders  (each returns [V=3, F=128, T])
# ---------------------------------------------------------------------------

def build_modulation_3view(
    logmel: np.ndarray,
) -> np.ndarray:
    """logmel_base, modulation_low, modulation_high."""
    mod_low = moving_average_time(logmel, win=11)
    mod_high = logmel - mod_low
    return np.stack([logmel, mod_low, mod_high], axis=0).astype(np.float32)


def build_cepstral_3view(
    logmel: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 40,
) -> np.ndarray:
    """logmel_base, cepstral_low_quefrency, cepstral_periodicity."""
    mfcc = compute_mfcc_from_logmel(logmel, sr=sr, n_mfcc=n_mfcc)
    t_ref = logmel.shape[1]

    # Low-quefrency: MFCC[1:13] upsampled to [128, T]
    low_q = _upsample_freq(mfcc[1:13].astype(np.float32), n_target=128)
    low_q = align_time(low_q, t_ref)

    # Periodicity: autocorrelation-based from MFCC[20:]
    if mfcc.shape[0] < 28:
        # Edge case: not enough MFCCs for MFCC[20:]
        per = np.zeros((1, logmel.shape[1]), dtype=np.float32)
    else:
        per_1d = _compute_periodicity_from_mfcc(mfcc[20:].astype(np.float32))
        per = per_1d[None, :]
    per_up = _upsample_freq(per, n_target=128)
    per_up = align_time(per_up, t_ref)

    return np.stack([logmel, low_q, per_up], axis=0).astype(np.float32)


def build_transient_3view(
    audio: np.ndarray,
    logmel: np.ndarray,
) -> np.ndarray:
    """logmel_base, spectral_flux, transient_energy."""
    sr = 16000
    t_ref = logmel.shape[1]

    # Spectral flux via librosa onset strength
    flux_1d = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=512)
    flux_1d = np.asarray(flux_1d, dtype=np.float32)
    # Broadcast to [128, T]
    flux_map = np.tile(flux_1d[None, :], (128, 1))
    flux_map = align_time(flux_map, t_ref)

    # Transient energy = max(diff(flux), 0)
    flux_diff = np.diff(flux_1d, prepend=flux_1d[0])
    transient_1d = np.maximum(flux_diff, 0.0).astype(np.float32)
    transient_map = np.tile(transient_1d[None, :], (128, 1))
    transient_map = align_time(transient_map, t_ref)

    return np.stack([logmel, flux_map, transient_map], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature extraction cache
# ---------------------------------------------------------------------------

class FeatureCache:
    """Cache for raw audio, logmel, and computed frontend tensors.

    Keyed by (machine, frontend_key, item_path) where frontend_key is
    "logmel" for the base spectrogram or the frontend name for 3-view tensors.
    """

    def __init__(self):
        self._audio: Dict[str, np.ndarray] = {}
        self._logmel: Dict[str, np.ndarray] = {}
        self._tensors: Dict[str, np.ndarray] = {}
        self._baseline: Dict[str, Dict[str, np.ndarray]] = {}  # path -> {view_name: array}

    def _key(self, path: Path) -> str:
        return str(path)

    def get_audio(self, path: Path) -> np.ndarray:
        k = self._key(path)
        if k not in self._audio:
            self._audio[k] = load_audio(path)
        return self._audio[k]

    def get_logmel(self, path: Path) -> np.ndarray:
        k = self._key(path)
        if k not in self._logmel:
            audio = self.get_audio(path)
            self._logmel[k] = extract_logmel(audio)
        return self._logmel[k]

    def get_tensor(self, path: Path, frontend: str) -> np.ndarray:
        """Get [3, 128, T] tensor for a frontend."""
        fk = f"{frontend}__{self._key(path)}"
        if fk not in self._tensors:
            logmel = self.get_logmel(path)
            if frontend == "modulation_3view":
                self._tensors[fk] = build_modulation_3view(logmel)
            elif frontend == "cepstral_3view":
                self._tensors[fk] = build_cepstral_3view(logmel)
            elif frontend == "transient_3view":
                audio = self.get_audio(path)
                self._tensors[fk] = build_transient_3view(audio, logmel)
            else:
                raise ValueError(f"Unknown frontend: {frontend}")
        return self._tensors[fk]

    def get_baseline_views(self, path: Path) -> Dict[str, np.ndarray]:
        """Return dict of {view_name: [F, T]} for the 5 baseline views."""
        k = self._key(path)
        if k not in self._baseline:
            logmel = self.get_logmel(path)
            edge = compute_edge(logmel)
            self._baseline[k] = {
                "logmel": logmel,
                "delta": compute_delta(logmel),
                "delta_delta": compute_delta_delta(logmel),
                "edge": edge,
                "enhanced": compute_enhanced(edge),
            }
        return self._baseline[k]


# ---------------------------------------------------------------------------
# Per-machine analysis
# ---------------------------------------------------------------------------

def _build_groups(
    test_items: list,
    test_scores: np.ndarray,
    y_test: np.ndarray,
    q_low: float,
    q_high: float,
) -> Tuple[GroupData, GroupData, GroupData, float, float]:
    """Classify test items into normal_tail, missed, detected groups.

    Returns
    -------
    normal_tail, missed, detected : GroupData
    q_low_val, q_high_val : float
        Actual quantile thresholds computed from normal test scores.
    """
    normal_scores = test_scores[y_test == 0]
    q_low_val = float(np.quantile(normal_scores, q_low))
    q_high_val = float(np.quantile(normal_scores, q_high))

    normal_idx = np.where(y_test == 0)[0]
    anomaly_idx = np.where(y_test == 1)[0]

    # normal_tail: q_low < score <= q_high
    tail_mask = (test_scores[normal_idx] > q_low_val) & (test_scores[normal_idx] <= q_high_val)
    tail_indices = normal_idx[tail_mask]

    # missed: anomaly score <= q_high
    missed_mask = test_scores[anomaly_idx] <= q_high_val
    missed_indices = anomaly_idx[missed_mask]

    # detected: anomaly score > q_high
    detected_mask = test_scores[anomaly_idx] > q_high_val
    detected_indices = anomaly_idx[detected_mask]

    def _make_group(indices: np.ndarray) -> GroupData:
        idxs = indices.tolist()
        return GroupData(
            indices=idxs,
            paths=[test_items[i].path for i in idxs],
            labels=[int(y_test[i]) for i in idxs],
            domains=[str(test_items[i].domain) for i in idxs],
            # ↑ str() because numpy d_test entries may be numpy.str_
        )

    return (
        _make_group(tail_indices),
        _make_group(missed_indices),
        _make_group(detected_indices),
        q_low_val,
        q_high_val,
    )


def compute_distances(
    items: list,
    normalizer: InputNormalizer,
    frontend: str,
    cache: FeatureCache,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Euclidean distances to normal-tail centroid.

    Uses the normalizer fitted on train-normal data. Returns tuple of
    (distances, flags) where flags is an array of str describing any edge-case
    issues per sample.
    """
    distances = np.zeros(len(items), dtype=np.float64)
    flags = np.array([""] * len(items), dtype=object)

    for i, item in enumerate(items):
        try:
            tensor = cache.get_tensor(item.path, frontend)
            if tensor.ndim != 3:
                distances[i] = np.nan
                flags[i] = "bad_tensor_ndim"
                continue
            normed = normalizer.transform(tensor)
            vec = normed.ravel()
            distances[i] = np.linalg.norm(vec)
        except Exception as e:
            distances[i] = np.nan
            flags[i] = f"extraction_error:{str(e)[:60]}"

    return distances, flags


def analyze_machine(
    machine: str,
    m_items: list,
    ref_dir: Path,
    experiment: str,
    score_mode: str,
    frontends: List[str],
    q_low: float,
    q_high: float,
    cache: FeatureCache,
    save_panels: bool,
    panel_dir: Optional[Path],
) -> Tuple[List[FrontendResult], List[FrontendResult], Dict[str, str]]:
    """Run full analysis for one machine across all frontends.

    Returns
    -------
    results : list[FrontendResult]  per-frontend results
    all_flags : dict[str, str]       flag -> description for this machine
    """
    # --- Load reference scores ---
    npz_path = ref_dir / "scores" / f"{machine}_{experiment}_{score_mode}_scores.npz"
    arr = np.load(npz_path, allow_pickle=True)
    test_scores = arr["test_scores"].astype(np.float64)
    y_test = arr["y_test"].astype(np.int64)
    d_test = arr["d_test"]

    # Build test_items list
    all_test = [x for x in m_items if x.split == "test" and x.label is not None
                and x.domain in ("source", "target")]

    # --- Build groups ---
    normal_tail, missed, detected, q_low_val, q_high_val = _build_groups(
        all_test, test_scores, y_test, q_low, q_high,
    )

    # --- Train-normal (source only) for InputNormalizer ---
    train_normal = [x for x in m_items if x.split == "train"
                    and x.label in (0, None) and x.domain == "source"]

    # --- Per-frontend analysis ---
    results = []
    warnings: Dict[str, str] = {}

    for frontend in frontends:
        flags = []

        # Fit normalizer on train-normal tensors
        train_tensors = []
        for item in train_normal:
            try:
                t = cache.get_tensor(item.path, frontend)
                train_tensors.append(t)
            except Exception:
                continue
        if not train_tensors:
            results.append(FrontendResult(
                frontend=frontend, machine=machine,
                robust_d_missed=0.0, robust_d_detected=0.0, cohens_d=0.0,
                redundancy_max_r=1.0,
                gate_a=False, gate_b=False, gate_c=False, gate_d=False,
                flag="no_train_tensors",
            ))
            warnings["no_train_tensors"] = f"{machine}/{frontend}: no train tensors"
            continue

        normalizer = InputNormalizer()
        normalizer.fit(train_tensors)

        # --- Compute distances for test groups ---
        # We need item-like objects that match the test_items order
        tail_dists, tail_flags = compute_distances(
            [all_test[i] for i in normal_tail.indices], normalizer, frontend, cache,
        )
        missed_dists, missed_flags = compute_distances(
            [all_test[i] for i in missed.indices], normalizer, frontend, cache,
        )
        detected_dists, detected_flags = compute_distances(
            [all_test[i] for i in detected.indices], normalizer, frontend, cache,
        )

        # --- Normal-tail centroid ---
        # Compute centroid from normal-tail flattened vectors
        n_tail = normal_tail.size
        if n_tail > 0:
            tail_vecs = []
            for idx in normal_tail.indices:
                item = all_test[idx]
                try:
                    t = cache.get_tensor(item.path, frontend)
                    normed = normalizer.transform(t)
                    tail_vecs.append(normed.ravel())
                except Exception:
                    continue
            if tail_vecs:
                centroid = np.mean(np.stack(tail_vecs, axis=0), axis=0)
                # Recompute distances relative to centroid
                tail_dists = np.array(
                    [np.linalg.norm(v - centroid) for v in tail_vecs], dtype=np.float64
                )
                # Compute missed distances relative to centroid
                missed_vecs = []
                missed_dists = np.full(missed.size, np.nan, dtype=np.float64)
                for i, idx in enumerate(missed.indices):
                    item = all_test[idx]
                    try:
                        t = cache.get_tensor(item.path, frontend)
                        normed = normalizer.transform(t)
                        vec = normed.ravel()
                        missed_dists[i] = np.linalg.norm(vec - centroid)
                    except Exception:
                        pass
                # Compute detected distances relative to centroid
                detected_vecs = []
                detected_dists = np.full(detected.size, np.nan, dtype=np.float64)
                for i, idx in enumerate(detected.indices):
                    item = all_test[idx]
                    try:
                        t = cache.get_tensor(item.path, frontend)
                        normed = normalizer.transform(t)
                        vec = normed.ravel()
                        detected_dists[i] = np.linalg.norm(vec - centroid)
                    except Exception:
                        pass
            else:
                tail_dists = np.array([])
                missed_dists = np.array([])
                detected_dists = np.array([])
                warnings["empty_normal_tail_centroid"] = (
                    f"{machine}/{frontend}: could not compute normal-tail centroid"
                )

        # Remove NaNs
        tail_dists_clean = tail_dists[~np.isnan(tail_dists)]
        missed_dists_clean = missed_dists[~np.isnan(missed_dists)]
        detected_dists_clean = detected_dists[~np.isnan(detected_dists)]

        # Edge-case flags
        if n_tail == 0:
            flags.append("empty_normal_tail")
        if missed.size == 0:
            flags.append("no_missed_anomalies")
        if detected.size == 0:
            flags.append("no_detected_anomalies")
        if n_tail == 1 or missed.size == 1:
            flags.append("single_sample_group_IQR_zero")
        if any(np.isnan(tail_dists)):
            flags.append("nan_in_tail_distances")
        if any(np.isnan(missed_dists)):
            flags.append("nan_in_missed_distances")

        # --- robust_d computations ---
        if len(missed_dists_clean) == 0 or len(tail_dists_clean) == 0:
            rd_missed = 0.0
            if not flags:
                flags.append("robust_d_undefined_empty_group")
        else:
            _, _, _, _, rd_missed = compute_robust_d(missed_dists_clean, tail_dists_clean)

        if len(detected_dists_clean) == 0 or len(tail_dists_clean) == 0:
            rd_detected = 0.0
        else:
            _, _, _, _, rd_detected = compute_robust_d(detected_dists_clean, tail_dists_clean)

        # Cohen's d (missed vs normal-tail)
        if len(missed_dists_clean) > 0 and len(tail_dists_clean) > 0:
            cd = compute_cohens_d(missed_dists_clean, tail_dists_clean)
        else:
            cd = 0.0

        # Gates (Gate B is evaluated at summary level)
        gate_a = rd_missed >= 0.30
        gate_d = rd_detected > 0.0

        results.append(FrontendResult(
            frontend=frontend, machine=machine,
            robust_d_missed=round(rd_missed, 6),
            robust_d_detected=round(rd_detected, 6),
            cohens_d=round(cd, 6),
            redundancy_max_r=0.0,  # filled later
            gate_a=gate_a,
            gate_b=False,  # filled later
            gate_c=False,  # filled later
            gate_d=gate_d,
            n_missed=int(missed.size),
            n_detected=int(detected.size),
            n_normal_tail=int(n_tail),
            n_total_anomalies=int(np.sum(y_test == 1)),
            flag="; ".join(flags) if flags else "",
        ))

        # Panels
        if save_panels and panel_dir is not None:
            panel_dir.mkdir(parents=True, exist_ok=True)
            _save_panel(
                machine, frontend, all_test,
                normal_tail, missed, detected,
                tail_dists_clean, missed_dists_clean, detected_dists_clean,
                cache, panel_dir,
            )

    return results, warnings


# ---------------------------------------------------------------------------
# Redundancy computation
# ---------------------------------------------------------------------------

def compute_redundancy(
    frontends: List[str],
    machines: List[str],
    m_items_by_machine: Dict[str, list],
    cache: FeatureCache,
) -> List[RedundancyEntry]:
    """Compute Pearson r between each frontend's new views and baseline views.

    Uses one representative sample per machine (first test-normal).

    Returns list of RedundancyEntry per (frontend, new_view_name).
    """
    entries: List[RedundancyEntry] = []
    baseline_names = ["logmel", "delta", "delta_delta", "edge", "enhanced"]

    # Collect cumulative max |r| per frontend per view
    running_max: Dict[str, Dict[str, float]] = {}
    for fe in frontends:
        running_max[fe] = {"view1": 0.0, "view2": 0.0}

    for machine in machines:
        # Pick a representative test-normal item
        test_items = [x for x in m_items_by_machine[machine]
                      if x.split == "test" and x.label == 0]
        if not test_items:
            continue
        rep = test_items[0]

        baseline = cache.get_baseline_views(rep.path)

        for fe in frontends:
            tensor = cache.get_tensor(rep.path, fe)
            # New views are tensor[1] and tensor[2] (index 0 is logmel_base)
            new_views = [
                ("view1", tensor[1].ravel()),
                ("view2", tensor[2].ravel()),
            ]

            for vname, vvec in new_views:
                max_r = 0.0
                correlated = ""
                for bname, barr in baseline.items():
                    # Align dimensions
                    bvec = barr.ravel()
                    if len(vvec) != len(bvec):
                        continue
                    r_val = abs(float(np.corrcoef(vvec, bvec)[0, 1]))
                    if np.isnan(r_val):
                        r_val = 0.0
                    if r_val > max_r:
                        max_r = r_val
                        correlated = bname
                if max_r > running_max[fe][vname]:
                    running_max[fe][vname] = max_r

        # Build display names for new views by frontend
        frontend_view_names = {
            "modulation_3view": ["modulation_low", "modulation_high"],
            "cepstral_3view": ["cepstral_low_quefrency", "cepstral_periodicity"],
            "transient_3view": ["spectral_flux", "transient_energy"],
        }

    for fe in frontends:
        names = frontend_view_names.get(fe, ["view1", "view2"])
        for i, vkey in enumerate(["view1", "view2"]):
            entries.append(RedundancyEntry(
                frontend=fe,
                view_name=names[i],
                max_abs_r=round(running_max[fe][vkey], 6),
            ))

    return entries


# ---------------------------------------------------------------------------
# Panel figures
# ---------------------------------------------------------------------------

def _save_panel(
    machine: str,
    frontend: str,
    all_test: list,
    normal_tail: GroupData,
    missed: GroupData,
    detected: GroupData,
    tail_dists: np.ndarray,
    missed_dists: np.ndarray,
    detected_dists: np.ndarray,
    cache: FeatureCache,
    panel_dir: Path,
):
    """Save a 3-panel figure showing the frontend views for one representative
    sample from each group."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    group_names = ["normal_tail", "missed", "detected"]
    groups = [normal_tail, missed, detected]
    group_dists = [tail_dists, missed_dists, detected_dists]

    for ax_idx, (gname, group, gdists) in enumerate(zip(group_names, groups, group_dists)):
        ax = axes[ax_idx]
        if group.size == 0:
            ax.text(0.5, 0.5, f"{gname}\n(empty)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(gname)
            continue

        # Find the "most typical" sample (closest to median distance) or first
        if len(gdists) > 0:
            median_dist = np.median(gdists)
            sel_idx = int(np.argmin(np.abs(gdists - median_dist)))
        else:
            sel_idx = 0
        item = all_test[group.indices[sel_idx]]

        try:
            tensor = cache.get_tensor(item.path, frontend)
            # Show the second view (new view 1) as representative
            view_to_show = tensor[1]
            ax.imshow(view_to_show, aspect="auto", origin="lower", cmap="magma")
            ax.set_title(f"{gname}\n{item.path.name[:40]}", fontsize=9)
        except Exception:
            ax.text(0.5, 0.5, "extraction\nerror", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(gname)

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"{machine} / {frontend}", fontsize=13)
    fig.tight_layout()
    out_path = panel_dir / f"{machine}_{frontend}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report & CSV generation
# ---------------------------------------------------------------------------

def _write_per_machine_csv(out_dir: Path, results: List[FrontendResult]):
    csv_path = out_dir / "per_machine.csv"
    fields = [
        "machine", "frontend",
        "robust_d_missed", "robust_d_detected", "cohens_d",
        "redundancy_max_r",
        "n_missed", "n_detected", "n_normal_tail", "n_total_anomalies",
        "gate_a", "gate_b", "gate_c", "gate_d", "all_pass",
        "flag",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            all_pass = r.gate_a and r.gate_b and r.gate_c and r.gate_d
            w.writerow({
                "machine": r.machine,
                "frontend": r.frontend,
                "robust_d_missed": r.robust_d_missed,
                "robust_d_detected": r.robust_d_detected,
                "cohens_d": r.cohens_d,
                "redundancy_max_r": r.redundancy_max_r,
                "n_missed": r.n_missed,
                "n_detected": r.n_detected,
                "n_normal_tail": r.n_normal_tail,
                "n_total_anomalies": r.n_total_anomalies,
                "gate_a": r.gate_a,
                "gate_b": r.gate_b,
                "gate_c": r.gate_c,
                "gate_d": r.gate_d,
                "all_pass": all_pass,
                "flag": r.flag,
            })


def _write_summary_csv(out_dir: Path, results: List[FrontendResult]):
    """Aggregate across machines: per-frontend verdict."""
    csv_path = out_dir / "summary.csv"
    frontends_set = sorted(set(r.frontend for r in results))
    fields = [
        "frontend",
        "machines_passing_gate_a", "machines_passing_all",
        "max_redundancy_r", "gate_b_pass", "gate_c_pass",
        "stage0_clearance", "mean_robust_d_missed", "mean_robust_d_detected",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for fe in frontends_set:
            fe_results = [r for r in results if r.frontend == fe]
            gate_a_count = sum(1 for r in fe_results if r.gate_a)
            all_pass_count = sum(
                1 for r in fe_results
                if r.gate_a and r.gate_b and r.gate_c and r.gate_d
            )
            gate_b_pass = gate_a_count >= 2
            r_max = max(r.redundancy_max_r for r in fe_results) if fe_results else 1.0
            gate_c_pass = r_max <= 0.85
            stage0 = gate_b_pass and gate_c_pass and all_pass_count >= 2
            mean_rd_m = (np.mean([r.robust_d_missed for r in fe_results])
                         if fe_results else 0.0)
            mean_rd_d = (np.mean([r.robust_d_detected for r in fe_results])
                         if fe_results else 0.0)
            w.writerow({
                "frontend": fe,
                "machines_passing_gate_a": gate_a_count,
                "machines_passing_all": all_pass_count,
                "max_redundancy_r": round(r_max, 6),
                "gate_b_pass": gate_b_pass,
                "gate_c_pass": gate_c_pass,
                "stage0_clearance": stage0,
                "mean_robust_d_missed": round(mean_rd_m, 6),
                "mean_robust_d_detected": round(mean_rd_d, 6),
            })


def _write_redundancy_csv(out_dir: Path, entries: List[RedundancyEntry]):
    csv_path = out_dir / "redundancy.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frontend", "view_name", "max_abs_r"])
        w.writeheader()
        for e in entries:
            w.writerow({
                "frontend": e.frontend,
                "view_name": e.view_name,
                "max_abs_r": e.max_abs_r,
            })


def _build_md_report(
    args: argparse.Namespace,
    results: List[FrontendResult],
    redundancy: List[RedundancyEntry],
    warnings: Dict[str, str],
    machine_info: Dict[str, dict],
) -> str:
    """Build the CTAF markdown report."""
    lines = [
        "# CTAF V39 Frontend Pre-screen Report",
        "",
        "## 1. Pre-registered Gate Definition",
        "",
        f"All gates were written **before** running this diagnostic. No threshold was tuned per machine.",
        "",
        "| Gate | Condition | Threshold |",
        "|------|-----------|-----------|",
        f"| A | robust_d_missed >= pass_robust_d | >= {args.pass_robust_d} |",
        f"| B | >= min_pass_machines machines pass Gate A | >= {args.min_pass_machines} |",
        f"| C | max_redundancy_r <= max_redundancy_r | <= {args.max_redundancy_r} |",
        f"| D | robust_d_detected > detected_sanity_min | > {args.detected_sanity_min_robust_d} |",
        "",
        "robust_d = (median(group) - median(normal_tail)) / "
        "(0.5*(IQR(group)+IQR(normal_tail)) + 1e-8)",
        "",
        "## 2. Reference Score Source",
        "",
        f"- Result directory: `{args.reference_result_dir}`",
        f"- Experiment: `{args.reference_experiment}`",
        f"- Score mode: `{args.reference_score_mode}`",
        f"- Normal-tail band: q{args.normal_tail_q_low} -- q{args.normal_tail_q_high}",
        "",
        "## 3. Sample Group Statistics",
        "",
        "| Machine | n_normal_tail | n_missed | n_detected | n_total_anomalies | q_low | q_high |",
        "|---------|---------------|----------|------------|-------------------|-------|--------|",
    ]

    for machine, info in sorted(machine_info.items()):
        lines.append(
            f"| {machine} | {info['n_normal_tail']} | {info['n_missed']} | "
            f"{info['n_detected']} | {info['n_total_anomalies']} | "
            f"{info['q_low']:.6f} | {info['q_high']:.6f} |"
        )

    lines.extend([
        "",
        "## 4. Per-Frontend Per-Machine Separation",
        "",
        "| Machine | Frontend | robust_d_missed | robust_d_detected "
        "| cohens_d | Gate A | Gate D | Flags |",
        "|---------|----------|-----------------|-------------------"
        "|----------|--------|--------|-------|",
    ])

    for r in sorted(results, key=lambda x: (x.machine, x.frontend)):
        lines.append(
            f"| {r.machine} | {r.frontend} | {r.robust_d_missed:.4f} | "
            f"{r.robust_d_detected:.4f} | {r.cohens_d:.4f} | "
            f"{'PASS' if r.gate_a else 'FAIL'} | "
            f"{'PASS' if r.gate_d else 'FAIL'} | "
            f"{r.flag or '--'} |"
        )

    lines.extend([
        "",
        "## 5. Redundancy (Pearson r vs 5 baseline views)",
        "",
        "| Frontend | View | max(|r|) |",
        "|----------|------|----------|",
    ])

    for e in redundancy:
        lines.append(f"| {e.frontend} | {e.view_name} | {e.max_abs_r:.4f} |")

    # Gate results table
    lines.extend([
        "",
        "## 6. Gate Results Table",
        "",
        "| Frontend | Machines Pass A | Gate B (>=2) | "
        "max_r | Gate C (<=0.85) | Machines Pass All | Stage 0 |",
        "|----------|-----------------|--------------|-------|----------------|-------------------|---------|",
    ])

    frontends_set = sorted(set(r.frontend for r in results))
    for fe in frontends_set:
        fe_results = [r for r in results if r.frontend == fe]
        gate_a_n = sum(1 for r in fe_results if r.gate_a)
        gate_b_pass = gate_a_n >= args.min_pass_machines
        r_max = max(r.redundancy_max_r for r in fe_results)
        gate_c_pass = r_max <= args.max_redundancy_r
        all_pass_n = sum(
            1 for r in fe_results
            if r.gate_a and r.gate_b and r.gate_c and r.gate_d
        )
        stage0 = gate_b_pass and gate_c_pass and all_pass_n >= args.min_pass_machines
        lines.append(
            f"| {fe} | {gate_a_n}/5 | {'PASS' if gate_b_pass else 'FAIL'} | "
            f"{r_max:.4f} | {'PASS' if gate_c_pass else 'FAIL'} | "
            f"{all_pass_n}/5 | {'PASS' if stage0 else 'FAIL'} |"
        )

    # Visual examples note
    if args.save_panels:
        lines.extend([
            "",
            "## 7. Per-Machine Visual Examples",
            "",
            "See `results/v39_frontend_prescreen/figures/` for example panels.",
        ])

    # Final verdict
    any_pass = any(
        sum(1 for r in results if r.frontend == fe and r.gate_a) >= args.min_pass_machines
        and max(r.redundancy_max_r for r in results if r.frontend == fe) <= args.max_redundancy_r
        for fe in frontends_set
    )

    lines.extend([
        "",
        "## 8. Final Verdict",
        "",
    ])

    stage0_frontends = []
    for fe in frontends_set:
        fe_results = [r for r in results if r.frontend == fe]
        gate_a_n = sum(1 for r in fe_results if r.gate_a)
        gate_b_pass = gate_a_n >= args.min_pass_machines
        r_max = max(r.redundancy_max_r for r in fe_results)
        gate_c_pass = r_max <= args.max_redundancy_r
        all_pass_n = sum(
            1 for r in fe_results
            if r.gate_a and r.gate_b and r.gate_c and r.gate_d
        )
        if gate_b_pass and gate_c_pass and all_pass_n >= args.min_pass_machines:
            stage0_frontends.append(fe)

    if stage0_frontends:
        lines.append("**Stage 0 clearance granted** for the following frontends:")
        for fe in stage0_frontends:
            lines.append(f"- `{fe}`")
        lines.append("")
        lines.append("These frontends satisfy all four pre-registered gates "
                      "and are candidates for V40 train-from-scratch AE evaluation.")
    else:
        lines.extend([
            "**No frontend passed all four gates.**",
            "",
            "## 9. Hard-stop",
            "",
            "Stage 0 clearance was **not granted** for any frontend. "
            "Do not proceed to V40 training. "
            "Re-evaluate frontend design or gate thresholds before continuing.",
        ])

    # Edge-case notes
    if warnings:
        lines.extend([
            "",
            "## Edge-case Notes",
            "",
        ])
        for key, desc in sorted(warnings.items()):
            lines.append(f"- `{key}`: {desc}")

    lines.extend([
        "",
        "---",
        "",
        "This diagnostic is offline-only. No model was trained, no score was tuned. "
        "All gates are pre-registered and applied uniformly across machines.",
    ])

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    ref_dir = Path(args.reference_result_dir)
    out_dir = Path(args.out_dir)
    out_md = Path(args.out_md)
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    frontends = [f.strip() for f in args.frontends.split(",") if f.strip()]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    if args.save_panels:
        panel_dir = out_dir / "figures"
        panel_dir.mkdir(parents=True, exist_ok=True)
    else:
        panel_dir = None

    # Scan dataset
    items = scan_dataset(data_root)

    # Build per-machine item maps
    m_items_by_machine: Dict[str, list] = {}
    for m in machines:
        m_items_by_machine[m] = filter_items(items, m)

    # Shared feature cache
    cache = FeatureCache()

    # --- Per-machine analysis ---
    all_results: List[FrontendResult] = []
    all_warnings: Dict[str, str] = {}
    machine_info: Dict[str, dict] = {}

    for machine in machines:
        m_items = m_items_by_machine[machine]

        # Quick group stats (from reference scores)
        npz_path = ref_dir / "scores" / f"{machine}_{args.reference_experiment}_{args.reference_score_mode}_scores.npz"
        arr = np.load(npz_path, allow_pickle=True)
        test_scores = arr["test_scores"].astype(np.float64)
        y_test = arr["y_test"].astype(np.int64)

        normal_scores = test_scores[y_test == 0]
        q_low_val = float(np.quantile(normal_scores, args.normal_tail_q_low))
        q_high_val = float(np.quantile(normal_scores, args.normal_tail_q_high))

        normal_idx = np.where(y_test == 0)[0]
        anomaly_idx = np.where(y_test == 1)[0]
        n_tail = int(np.sum(
            (test_scores[normal_idx] > q_low_val) &
            (test_scores[normal_idx] <= q_high_val)
        ))
        n_missed = int(np.sum(test_scores[anomaly_idx] <= q_high_val))
        n_detected = int(np.sum(test_scores[anomaly_idx] > q_high_val))
        n_anom = int(np.sum(y_test == 1))

        machine_info[machine] = {
            "n_normal_tail": n_tail,
            "n_missed": n_missed,
            "n_detected": n_detected,
            "n_total_anomalies": n_anom,
            "q_low": q_low_val,
            "q_high": q_high_val,
        }

        results, warn = analyze_machine(
            machine=machine,
            m_items=m_items,
            ref_dir=ref_dir,
            experiment=args.reference_experiment,
            score_mode=args.reference_score_mode,
            frontends=frontends,
            q_low=args.normal_tail_q_low,
            q_high=args.normal_tail_q_high,
            cache=cache,
            save_panels=args.save_panels,
            panel_dir=panel_dir,
        )
        all_results.extend(results)
        all_warnings.update(warn)

    # --- Redundancy ---
    redundancy = compute_redundancy(
        frontends, machines, m_items_by_machine, cache,
    )

    # Fill redundancy_max_r and gate_c into per-machine results
    red_by_fe: Dict[str, float] = {}
    for fe in frontends:
        fe_red = [e.max_abs_r for e in redundancy if e.frontend == fe]
        red_by_fe[fe] = max(fe_red) if fe_red else 1.0

    for r in all_results:
        r.redundancy_max_r = round(red_by_fe.get(r.frontend, 1.0), 6)
        r.gate_c = r.redundancy_max_r <= args.max_redundancy_r
        # Gate B: at the per-machine level, it's checked when gate_a count >= min_pass
        # We compute this at the summary level and propagate
        gate_a_count = sum(
            1 for rr in all_results
            if rr.frontend == r.frontend and rr.gate_a
        )
        r.gate_b = gate_a_count >= args.min_pass_machines

    if not all_results:
        print("No results generated. Check input paths and machines/frontends.")
        return

    # --- Write CSVs ---
    _write_per_machine_csv(out_dir, all_results)
    _write_summary_csv(out_dir, all_results)
    _write_redundancy_csv(out_dir, redundancy)

    # --- Write Markdown report ---
    md_content = _build_md_report(args, all_results, redundancy, all_warnings, machine_info)
    out_md.write_text(md_content)

    # Summary to stdout
    frontends_set = sorted(set(r.frontend for r in all_results))
    print("\nV39 Frontend Pre-screen complete.")
    print(f"Output directory: {out_dir}")
    print(f"Report:          {out_md}")
    print(f"Machines:        {len(machines)}")
    print(f"Frontends:       {len(frontends_set)}")
    for fe in frontends_set:
        fe_results = [r for r in all_results if r.frontend == fe]
        gate_a_n = sum(1 for r in fe_results if r.gate_a)
        print(f"  {fe}: gate_A={gate_a_n}/{len(machines)} "
              f"mean_rd_missed={np.mean([r.robust_d_missed for r in fe_results]):.4f}")


if __name__ == "__main__":
    main()
