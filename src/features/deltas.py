"""Temporal delta features: first and second differences of log-Mel."""
import numpy as np


def compute_delta(logmel: np.ndarray) -> np.ndarray:
    """First temporal difference. [F, T] -> [F, T]."""
    d = np.diff(logmel, axis=1)
    padded = np.pad(d, ((0, 0), (1, 0)), mode="edge")
    return padded.astype(np.float32)


def compute_delta_delta(logmel: np.ndarray) -> np.ndarray:
    """Second temporal difference. [F, T] -> [F, T]."""
    return compute_delta(compute_delta(logmel))
