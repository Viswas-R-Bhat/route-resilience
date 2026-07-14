"""Phase 2a — binary road mask -> 1-pixel-wide skeleton (centerline).

Order matters (per spec): morphological closing to bridge tiny gaps, remove small
noise blobs, THEN Zhang-Suen skeletonize. Skeletonizing a noisy/thick mask directly
produces spurious branches, so the cleanup comes first.
"""
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import binary_closing


def mask_to_skeleton(binary_mask, min_size=80, close_kernel=5):
    b = binary_mask.astype(bool)
    if close_kernel:
        b = binary_closing(b, structure=np.ones((close_kernel, close_kernel)))
    b = remove_small_objects(b, max_size=min_size - 1)   # skimage >=0.26 API (removes <= max_size)
    skel = skeletonize(b)
    return skel.astype(np.uint8)
