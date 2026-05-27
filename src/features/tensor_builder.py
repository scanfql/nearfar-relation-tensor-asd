"""Tensor builder: stack views into multi-view tensor and normalize."""
import numpy as np


def _align_time(x: np.ndarray, t_ref: int) -> np.ndarray:
    """Crop or edge-pad to reference frame count."""
    if x.shape[1] == t_ref:
        return x.astype(np.float32)
    if x.shape[1] > t_ref:
        return x[:, :t_ref].astype(np.float32)
    pad = t_ref - x.shape[1]
    return np.pad(x, ((0, 0), (0, pad)), mode="edge").astype(np.float32)


def _upsample_freq(x: np.ndarray, n_target: int = 128) -> np.ndarray:
    """NaN-safe linear upsample along frequency axis to n_target bins."""
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
    """Compute autocorrelation-based periodicity from high-order MFCC."""
    if mfcc_high.ndim != 2:
        raise ValueError(f"mfcc_high must be 2D, got shape {mfcc_high.shape}")
    t = mfcc_high.shape[1]
    periodicity = np.zeros((t,), dtype=np.float32)
    for t_idx in range(t):
        col = np.nan_to_num(mfcc_high[:, t_idx].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        col_centered = col - np.mean(col)
        denom = float(np.dot(col_centered, col_centered))
        if denom < 1e-12:
            periodicity[t_idx] = 0.0
            continue
        ac = np.correlate(col_centered, col_centered, mode="full")
        mid = len(ac) // 2
        end = min(mid + 9, len(ac))
        lags = ac[mid + 1:end]
        periodicity[t_idx] = np.max(lags) / denom if len(lags) > 0 else 0.0
    return periodicity.astype(np.float32)


def build_tensor(experiment: str, logmel, audio=None) -> np.ndarray:
    """Build multi-view tensor [V, F, T] from log-Mel.

    experiment choices:
      - logmel_only     -> V=1, logmel only
      - dynamic_3view   -> V=3, logmel + delta + delta-delta
      - edge_3view      -> V=3, logmel + edge + enhanced
      - full_5view      -> V=5, all five views
      - multires_7view  -> V=7, short/base/long logmel + 4 base-scale structure views
      - cepstral_3view  -> V=3, logmel + low-quefrency MFCC + periodicity
      - transient_3view -> V=3, logmel + spectral_flux + transient_energy
                           (requires audio=raw waveform)
    """
    from src.features.deltas import compute_delta, compute_delta_delta
    from src.features.edge import compute_edge
    from src.features.enhanced import compute_enhanced
    from src.features.logmel import compute_mfcc_from_logmel

    if experiment == "multires_7view":
        if not isinstance(logmel, dict):
            raise ValueError("multires_7view requires a dict of aligned logmel views")
        base = logmel["logmel_base"]
        edge = compute_edge(base)
        views = [
            logmel["logmel_short"],
            base,
            logmel["logmel_long"],
            compute_delta(base),
            compute_delta_delta(base),
            edge,
            compute_enhanced(edge),
        ]
        return np.stack(views, axis=0).astype(np.float32)

    if experiment == "transient_3view":
        if audio is None:
            raise ValueError("transient_3view requires raw audio")
        import librosa
        t_ref = logmel.shape[1]
        flux_1d = librosa.onset.onset_strength(y=audio, sr=16000, hop_length=512)
        flux_1d = np.asarray(flux_1d, dtype=np.float32)
        flux_map = np.tile(flux_1d[None, :], (128, 1))
        flux_map = _align_time(flux_map, t_ref)
        flux_diff = np.diff(flux_1d, prepend=flux_1d[0])
        transient_1d = np.maximum(flux_diff, 0.0).astype(np.float32)
        transient_map = np.tile(transient_1d[None, :], (128, 1))
        transient_map = _align_time(transient_map, t_ref)
        return np.stack([logmel, flux_map, transient_map], axis=0).astype(np.float32)

    if experiment == "cepstral_3view":
        mfcc = compute_mfcc_from_logmel(logmel, sr=16000, n_mfcc=40)
        t_ref = logmel.shape[1]
        low_q = _upsample_freq(mfcc[1:13].astype(np.float32), n_target=128)
        low_q = _align_time(low_q, t_ref)
        if mfcc.shape[0] < 28:
            per = np.zeros((1, t_ref), dtype=np.float32)
        else:
            per_1d = _compute_periodicity_from_mfcc(mfcc[20:].astype(np.float32))
            per = per_1d[None, :]
        per_up = _upsample_freq(per, n_target=128)
        per_up = _align_time(per_up, t_ref)
        views = [logmel.astype(np.float32), low_q, per_up]
        out = np.stack(views, axis=0).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    views = [logmel]

    if experiment in ("dynamic_3view", "full_5view"):
        views.append(compute_delta(logmel))
        views.append(compute_delta_delta(logmel))

    if experiment in ("edge_3view", "full_5view"):
        edge = compute_edge(logmel)
        views.append(edge)
        views.append(compute_enhanced(edge))

    return np.stack(views, axis=0).astype(np.float32)


class TensorNormalizer:
    """Train-set mean/std normalizer for multi-view tensors.

    Computes per-view statistics from training data (no anomaly labels).
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean = None   # [V, 1, 1]
        self.std = None    # [V, 1, 1]
        self.fitted = False

    def fit(self, tensors: list):
        """Fit on list of [V, F, T] tensors."""
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
            raise RuntimeError("Normalizer not fitted")
        return (tensor - self.mean) / (self.std + self.eps)

    def fit_transform(self, tensors: list) -> list:
        self.fit(tensors)
        return [self.transform(t) for t in tensors]
