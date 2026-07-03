from __future__ import annotations

import numpy as np


DEFAULT_CHANNELS = [
    "near_logmel",
    "far_logmel",
    "signed_near_far_difference",
    "absolute_near_far_difference",
    "log_energy_ratio",
    "inter_channel_level_difference",
    "coherence_proxy",
    "phase_delay_proxy",
    "local_relation_residual",
    "near_temporal_delta",
    "far_temporal_delta",
    "physical_consistency",
]


def compute_logmel(
    wav: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 160,
    n_mels: int = 128,
    target_frames: int = 96,
    target_bins: int = 64,
) -> np.ndarray:
    """Compute a compact log-Mel map with shape [target_frames, target_bins]."""
    try:
        import librosa
    except ImportError as exc:
        raise ImportError("librosa is required for log-Mel extraction") from exc

    y = np.asarray(wav, dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    logmel = np.log(np.maximum(mel, 1e-10)).T.astype(np.float32)
    return resize_time_frequency(logmel, target_frames, target_bins)


def resize_time_frequency(x: np.ndarray, target_frames: int, target_bins: int) -> np.ndarray:
    """Linearly resize a [T, F] map to the compact tensor grid."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected [T,F] map, got {arr.shape}")
    t_old = np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float32)
    t_new = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    tmp = np.empty((target_frames, arr.shape[1]), dtype=np.float32)
    for f in range(arr.shape[1]):
        tmp[:, f] = np.interp(t_new, t_old, arr[:, f]).astype(np.float32)
    f_old = np.linspace(0.0, 1.0, arr.shape[1], dtype=np.float32)
    f_new = np.linspace(0.0, 1.0, target_bins, dtype=np.float32)
    out = np.empty((target_frames, target_bins), dtype=np.float32)
    for t in range(target_frames):
        out[t] = np.interp(f_new, f_old, tmp[t]).astype(np.float32)
    return out


def temporal_delta(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    delta = np.zeros_like(arr)
    delta[1:] = arr[1:] - arr[:-1]
    return delta


def build_relation_tensor(
    near_logmel: np.ndarray,
    far_logmel: np.ndarray,
    eps: float = 1e-6,
    channel_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a 96 x 64 x 12 near/far relation tensor.

    Phase and coherence channels are proxies derived from compact log-Mel maps.
    If complex STFT phase is available, projects can replace these proxy
    channels before MPCA fitting while preserving the same tensor interface.
    """
    near = np.asarray(near_logmel, dtype=np.float32)
    far = np.asarray(far_logmel, dtype=np.float32)
    if near.shape != far.shape:
        raise ValueError(f"near/far shapes must match, got {near.shape} and {far.shape}")
    if near.ndim != 2:
        raise ValueError(f"Expected [T,F] maps, got {near.shape}")

    signed = near - far
    absdiff = np.abs(signed)
    log_ratio = np.log(np.maximum(np.exp(near), eps) / np.maximum(np.exp(far), eps))
    ild = signed / (np.abs(near) + np.abs(far) + eps)
    coherence = 1.0 - absdiff / (np.abs(near) + np.abs(far) + eps)
    phase_delay = temporal_delta(signed)
    local_residual = signed - signed.mean(axis=0, keepdims=True)
    near_delta = temporal_delta(near)
    far_delta = temporal_delta(far)
    consistency = 1.0 / (1.0 + np.abs(near_delta - far_delta) + np.abs(ild))

    values = {
        "near_logmel": near,
        "far_logmel": far,
        "signed_near_far_difference": signed,
        "absolute_near_far_difference": absdiff,
        "log_energy_ratio": log_ratio,
        "inter_channel_level_difference": ild,
        "coherence_proxy": coherence,
        "phase_delay_proxy": phase_delay,
        "local_relation_residual": local_residual,
        "near_temporal_delta": near_delta,
        "far_temporal_delta": far_delta,
        "physical_consistency": consistency,
    }
    names = list(DEFAULT_CHANNELS if channel_names is None else channel_names)
    tensor = np.stack([values[name] for name in names], axis=-1).astype(np.float32)
    return tensor, names
