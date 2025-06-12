import numpy as np
from scipy.spatial.distance import directed_hausdorff

def hausdorff_distance(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute the symmetric Hausdorff Distance between two binary masks.

    Args:
        mask1 (np.ndarray): Binary mask (e.g., ground truth), shape (H, W).
        mask2 (np.ndarray): Binary mask (e.g., prediction), shape (H, W).

    Returns:
        float: Symmetric Hausdorff distance between mask edges.
    """
    if mask1.shape != mask2.shape:
        raise ValueError("Both masks must have the same shape")

    # Get foreground (non-zero) coordinates
    coords1 = np.column_stack(np.where(mask1 > 0))
    coords2 = np.column_stack(np.where(mask2 > 0))

    if coords1.size == 0 or coords2.size == 0:
        return np.inf  # Can't compute distance if either mask is empty

    # Directed Hausdorff distances
    forward = directed_hausdorff(coords1, coords2)[0]
    backward = directed_hausdorff(coords2, coords1)[0]

    return max(forward, backward)