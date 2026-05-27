"""Local enhanced map: threshold edge map to suppress weak background."""
import numpy as np
from scipy.ndimage import uniform_filter


def compute_enhanced(
    edge_map: np.ndarray,
    local_radius: int = 5,
    background_suppress: float = 0.5,
) -> np.ndarray:
    """Apply local thresholding to enhance salient edge structures.

    Keep values above (local_mean + background_suppress * local_std).
    Returns [F, T] float32.
    """
    window = 2 * local_radius + 1
    local_mean = uniform_filter(edge_map, size=window, mode="nearest")
    local_sq_mean = uniform_filter(edge_map * edge_map, size=window, mode="nearest")
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))

    threshold = local_mean + background_suppress * local_std
    enhanced = np.where(edge_map > threshold, edge_map, 0.0)
    return enhanced.astype(np.float32)
