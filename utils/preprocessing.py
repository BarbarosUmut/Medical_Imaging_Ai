from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

DATA_DIR_BRAIN = Path("data/brain")
DATA_DIR_LUNG = Path("data/lung")


def normalize_volume(volume: np.ndarray, nonzero: bool = True) -> np.ndarray:
    """Z-score normalize a volume, optionally computed over nonzero voxels only."""
    volume = volume.astype(np.float32)
    if nonzero:
        mask = volume != 0
        if mask.sum() == 0:
            return volume
        mean = volume[mask].mean()
        std = volume[mask].std()
    else:
        mean = volume.mean()
        std = volume.std()

    if std < 1e-8:
        return volume - mean
    return (volume - mean) / std


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: int = 8,
) -> np.ndarray:
    """CLAHE without cv2.

    Uses skimage.exposure.equalize_adapthist when available; falls back to a
    manual numpy-based tile-wise histogram equalization.

    Args:
        image: 2D float array in [0, 1].
        clip_limit: Contrast limit (normalised to [0, 1] for skimage).
        tile_size: Number of tiles per axis (kernel_size = image_size / tile_size).

    Returns:
        CLAHE-enhanced 2D float array in [0, 1].
    """
    image = np.clip(image, 0.0, 1.0).astype(np.float32)

    try:
        from skimage.exposure import equalize_adapthist

        clip_skimage = clip_limit / 100.0
        return equalize_adapthist(
            image,
            kernel_size=max(image.shape[0] // tile_size, 1),
            clip_limit=clip_skimage,
        ).astype(np.float32)
    except ImportError:
        pass

    return _manual_clahe(image, clip_limit=clip_limit, tile_size=tile_size)


def _manual_clahe(
    image: np.ndarray,
    clip_limit: float,
    tile_size: int,
    n_bins: int = 256,
) -> np.ndarray:
    h, w = image.shape
    th = max(h // tile_size, 1)
    tw = max(w // tile_size, 1)

    img_int = (image * (n_bins - 1)).astype(np.int32).clip(0, n_bins - 1)

    tile_luts: list[list[np.ndarray]] = []
    for r in range(tile_size):
        row_luts = []
        for c in range(tile_size):
            r0, r1 = r * th, min((r + 1) * th, h)
            c0, c1 = c * tw, min((c + 1) * tw, w)
            tile = img_int[r0:r1, c0:c1]
            hist, _ = np.histogram(tile, bins=n_bins, range=(0, n_bins))
            limit = int(clip_limit * tile.size / n_bins)
            excess = np.maximum(hist - limit, 0).sum()
            hist = np.minimum(hist, limit)
            hist += excess // n_bins
            cdf = hist.cumsum().astype(np.float32)
            cdf /= cdf[-1]
            row_luts.append(cdf)
        tile_luts.append(row_luts)

    out = np.zeros_like(image, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            r_f = (i / th) - 0.5
            c_f = (j / tw) - 0.5
            r0 = int(np.floor(r_f))
            c0 = int(np.floor(c_f))
            r1 = r0 + 1
            c1 = c0 + 1
            dr = r_f - r0
            dc = c_f - c0

            r0c = np.clip(r0, 0, tile_size - 1)
            r1c = np.clip(r1, 0, tile_size - 1)
            c0c = np.clip(c0, 0, tile_size - 1)
            c1c = np.clip(c1, 0, tile_size - 1)

            v = img_int[i, j]
            v00 = tile_luts[r0c][c0c][v]
            v01 = tile_luts[r0c][c1c][v]
            v10 = tile_luts[r1c][c0c][v]
            v11 = tile_luts[r1c][c1c][v]

            out[i, j] = (
                (1 - dr) * (1 - dc) * v00
                + (1 - dr) * dc * v01
                + dr * (1 - dc) * v10
                + dr * dc * v11
            )
    return out.clip(0.0, 1.0)


def resize_volume(
    volume: np.ndarray,
    target_shape: tuple,
    order: int = 1,
) -> np.ndarray:
    """Resize a volume to target_shape using scipy.ndimage.zoom."""
    from scipy.ndimage import zoom

    factors = tuple(t / s for t, s in zip(target_shape, volume.shape))
    return zoom(volume.astype(np.float32), factors, order=order)


def window_level(volume: np.ndarray, window: int, level: int) -> np.ndarray:
    """Apply CT window/level to map HU values to [0, 1]."""
    low = level - window / 2.0
    high = level + window / 2.0
    clipped = np.clip(volume.astype(np.float32), low, high)
    return (clipped - low) / (window if window != 0 else 1.0)


def to_one_hot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer label mask to one-hot encoding.

    Args:
        mask: Integer array of shape (...).
        num_classes: Number of classes.

    Returns:
        One-hot array of shape (num_classes, *mask.shape).
    """
    one_hot = np.zeros((num_classes, *mask.shape), dtype=np.float32)
    for c in range(num_classes):
        one_hot[c] = (mask == c).astype(np.float32)
    return one_hot


def extract_patches_3d(
    volume: np.ndarray,
    patch_size: tuple,
    stride: tuple,
) -> list[np.ndarray]:
    """Extract 3D patches from a volume.

    Args:
        volume: 3D array of shape (D, H, W).
        patch_size: Patch size (pd, ph, pw).
        stride: Step size (sd, sh, sw).

    Returns:
        List of patch arrays each of shape patch_size.
    """
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    patches: list[np.ndarray] = []

    d = 0
    while d + pd <= D:
        h = 0
        while h + ph <= H:
            w = 0
            while w + pw <= W:
                patches.append(volume[d : d + pd, h : h + ph, w : w + pw].copy())
                w += sw
            h += sh
        d += sd
    return patches
