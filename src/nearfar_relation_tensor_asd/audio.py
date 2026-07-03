from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load audio as channels-first float32 array with shape [C, samples]."""
    wav, file_sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = wav.T
    if sr is not None and int(file_sr) != int(sr):
        try:
            import librosa
        except ImportError as exc:
            raise ImportError("librosa is required for resampling") from exc
        wav = np.stack([librosa.resample(ch, orig_sr=int(file_sr), target_sr=int(sr)) for ch in wav], axis=0)
        file_sr = sr
    return np.asarray(wav, dtype=np.float32), int(file_sr)


def split_near_far(wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return near and far channels from a two-channel waveform.

    The expected layout is [channels, samples]. A [samples, channels] array is
    accepted when the second dimension is clearly the channel dimension.
    """
    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D waveform, got {arr.shape}")
    if arr.shape[0] == 2:
        return arr[0], arr[1]
    if arr.shape[1] == 2:
        return arr[:, 0], arr[:, 1]
    raise ValueError(f"Expected exactly two audio channels, got {arr.shape}")
