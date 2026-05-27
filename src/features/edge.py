"""Edge map: Sobel gradient magnitude on log-Mel spectrogram."""
import numpy as np
from scipy.ndimage import sobel, gaussian_filter


def compute_edge(logmel: np.ndarray, smooth_sigma: float = 0.0) -> np.ndarray:
    """Compute Sobel gradient magnitude on log-Mel.

    Optionally applies Gaussian smoothing first.
    Returns [F, T] float32.
    """
    img = logmel
    if smooth_sigma > 0:
        img = gaussian_filter(img, sigma=smooth_sigma)

    grad_f = sobel(img, axis=0)
    grad_t = sobel(img, axis=1)
    mag = np.sqrt(grad_f ** 2 + grad_t ** 2)
    return mag.astype(np.float32)
