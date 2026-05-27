"""log-Mel spectrogram extraction."""
import librosa
import numpy as np


def extract_logmel(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
    eps: float = 1e-6,
) -> np.ndarray:
    """Extract log-Mel spectrogram. Returns [F, T] float32."""
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    return np.log(mel + eps).astype(np.float32)


def extract_multires_logmels(
    audio: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
    eps: float = 1e-6,
) -> dict:
    """Extract fixed short/base/long log-Mel views aligned to one time axis.

    We vary the analysis window while keeping the hop fixed so the three
    resolutions describe the same segment at different time-frequency scales.
    """
    hop_length = 512
    base = extract_logmel(
        audio, sr=sr, n_fft=1024, hop_length=hop_length, n_mels=n_mels, eps=eps
    )
    short = extract_logmel(
        audio, sr=sr, n_fft=512, hop_length=hop_length, n_mels=n_mels, eps=eps
    )
    long = extract_logmel(
        audio, sr=sr, n_fft=2048, hop_length=hop_length, n_mels=n_mels, eps=eps
    )
    t_ref = base.shape[1]
    return {
        "logmel_short": _align_time_axis(short, t_ref),
        "logmel_base": base,
        "logmel_long": _align_time_axis(long, t_ref),
    }


def _align_time_axis(x: np.ndarray, t_ref: int) -> np.ndarray:
    """Crop or edge-pad to the reference number of frames."""
    if x.shape[1] == t_ref:
        return x.astype(np.float32)
    if x.shape[1] > t_ref:
        return x[:, :t_ref].astype(np.float32)
    pad = t_ref - x.shape[1]
    return np.pad(x, ((0, 0), (0, pad)), mode="edge").astype(np.float32)


def compute_mfcc_from_logmel(
    logmel: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 40,
) -> np.ndarray:
    """Compute MFCC from log-Mel spectrogram.

    Uses librosa.feature.mfcc with the log-Mel as the input spectrogram
    (S=logmel). Returns [n_mfcc, T] float32.
    """
    mfcc = librosa.feature.mfcc(S=logmel, sr=sr, n_mfcc=n_mfcc)
    return mfcc.astype(np.float32)


def load_audio(path, sr=16000):
    """Load mono audio at target sample rate."""
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio
