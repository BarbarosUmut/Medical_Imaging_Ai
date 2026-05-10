"""
Brain MRI segmentation inference module.

Provides:
  - BrainSegmentor  — load a trained checkpoint and segment NIfTI or DICOM inputs.
  - visualize_segmentation — overlay a colour mask on a greyscale slice.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow importing sibling packages when run from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.brain.unet_2d import build_unet_2d  # noqa: E402
from models.brain.unet_3d import build_unet_3d, get_sliding_window_inferer  # noqa: E402


# ---------------------------------------------------------------------------
# Colour palette for up to 4 classes (background transparent)
# ---------------------------------------------------------------------------

_CLASS_COLOURS = np.array(
    [
        [0, 0, 0],        # 0 — background (black / transparent)
        [255, 0, 0],      # 1 — necrotic core          (red)
        [0, 255, 0],      # 2 — peritumoral oedema     (green)
        [0, 0, 255],      # 3 — enhancing tumour       (blue)
    ],
    dtype=np.uint8,
)


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def visualize_segmentation(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    alpha: float = 0.4,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Overlay a coloured segmentation mask on a greyscale MRI slice.

    Args:
        image_slice: 2-D float array (H, W) — single-channel MRI slice.
        mask_slice : 2-D integer array (H, W) — class indices 0..num_classes-1.
        alpha      : Opacity of the overlay (0 = transparent, 1 = opaque).
        save_path  : If provided, save the RGB image to this path (PNG/JPG).

    Returns:
        RGB numpy array of shape (H, W, 3) with dtype uint8.
    """
    # Normalise image to 0–255.
    img = image_slice.astype(np.float32)
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
    grey_rgb = np.stack([img.astype(np.uint8)] * 3, axis=-1)  # (H, W, 3)

    # Build colour overlay.
    h, w = mask_slice.shape
    colour_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, colour in enumerate(_CLASS_COLOURS):
        colour_mask[mask_slice == cls_idx] = colour

    # Blend: keep greyscale where mask==0, blend elsewhere.
    fg = mask_slice > 0
    result = grey_rgb.copy()
    result[fg] = (
        (1 - alpha) * grey_rgb[fg].astype(np.float32)
        + alpha * colour_mask[fg].astype(np.float32)
    ).astype(np.uint8)

    if save_path is not None:
        try:
            from PIL import Image
            Image.fromarray(result).save(save_path)
        except ImportError:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.imsave(save_path, result)

    return result


# ---------------------------------------------------------------------------
# BrainSegmentor
# ---------------------------------------------------------------------------

class BrainSegmentor:
    """Load a trained brain segmentation checkpoint and run inference.

    Args:
        model_path: Path to a checkpoint saved by MetricsLogger.save_best_model,
                    i.e. a dict with key ``"model_state_dict"``.
        mode      : ``"2d"`` or ``"3d"``.
        device    : ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
    """

    NUM_CLASSES = 4
    MODALITY_NAMES = ["t1", "t1ce", "t2", "flair"]

    def __init__(
        self,
        model_path: str,
        mode: str = "2d",
        device: Optional[str] = None,
    ) -> None:
        if mode not in ("2d", "3d"):
            raise ValueError(f"mode must be '2d' or '3d', got '{mode}'")

        self.mode = mode
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Build model skeleton then load weights.
        self.model: nn.Module = (
            build_unet_2d(in_channels=4, num_classes=self.NUM_CLASSES)
            if mode == "2d"
            else build_unet_3d(in_channels=4, num_classes=self.NUM_CLASSES)
        )
        self.model.to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Sliding-window inferer for 3-D.
        self.inferer = get_sliding_window_inferer() if mode == "3d" else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, input_path: str) -> dict:
        """Segment a brain MRI volume.

        Args:
            input_path: Path to a NIfTI file (``.nii`` / ``.nii.gz``) **or**
                        a directory that contains a DICOM series.

        Returns:
            dict with keys:

            - ``"mask"``             — integer np.ndarray, same H×W×D as input.
            - ``"probabilities"``    — float np.ndarray (num_classes, D, H, W).
            - ``"dice_scores"``      — dict (empty; ground truth not available here).
            - ``"tumor_volume_ml"``  — float, estimated in millilitres.
            - ``"modalities_found"`` — list[str] of modality names detected.
        """
        p = Path(input_path)
        if p.is_dir():
            volume = self._load_dicom_dir(str(p))
            affine = np.eye(4)
            modalities_found = ["dicom_series"]
        else:
            volume, affine = self._load_nifti(str(p))
            modalities_found = self._guess_modalities(str(p))

        tensor = self._preprocess(volume)  # (1, C, [D,] H, W)

        with torch.no_grad():
            if self.mode == "3d" and self.inferer is not None:
                logits = self.inferer(tensor, self.model)
            else:
                logits = self._run_2d_slice_by_slice(tensor)

        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (C, D, H, W)
        mask = probs.argmax(axis=0).astype(np.int32)               # (D, H, W)

        # Voxel volume in ml: use affine diagonal.
        voxel_size_mm3 = float(
            abs(affine[0, 0]) * abs(affine[1, 1]) * abs(affine[2, 2])
        )
        if voxel_size_mm3 == 0:
            voxel_size_mm3 = 1.0
        tumor_voxels = int((mask > 0).sum())
        tumor_volume_ml = tumor_voxels * voxel_size_mm3 / 1000.0

        return {
            "mask": mask,
            "probabilities": probs,
            "dice_scores": {},
            "tumor_volume_ml": tumor_volume_ml,
            "modalities_found": modalities_found,
        }

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_nifti(self, path: str) -> tuple[np.ndarray, Any]:
        """Load a NIfTI file.

        Returns:
            (volume, affine) where volume has shape (D, H, W) or (D, H, W, C).
        """
        import nibabel as nib
        img = nib.load(path)
        volume = img.get_fdata(dtype=np.float32)
        return volume, img.affine

    def _load_dicom_dir(self, dir_path: str) -> np.ndarray:
        """Load a DICOM series from *dir_path* using pydicom.

        Returns:
            3-D float32 numpy array (D, H, W) sorted by InstanceNumber.
        """
        import pydicom

        dcm_files = sorted(
            Path(dir_path).glob("*.dcm"),
            key=lambda f: (
                int(pydicom.dcmread(str(f), stop_before_pixels=True).get("InstanceNumber", 0))
            ),
        )
        if not dcm_files:
            raise FileNotFoundError(f"No .dcm files found in {dir_path}")

        slices = []
        for f in dcm_files:
            ds = pydicom.dcmread(str(f))
            slices.append(ds.pixel_array.astype(np.float32))

        volume = np.stack(slices, axis=0)  # (D, H, W)
        return volume

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, volume: np.ndarray) -> torch.Tensor:
        """Normalise volume and convert to a model-ready tensor.

        Handles both single-channel (D, H, W) and multi-channel
        (D, H, W, C) inputs. For single-channel volumes the same
        normalised slice is replicated across the 4 expected channels.

        Returns:
            Tensor of shape (1, 4, D, H, W) for 3-D mode or
            (D, 4, H, W) treated as a batch of 2-D slices.
        """
        vol = volume.astype(np.float32)

        if vol.ndim == 3:
            # Single-channel volume: (D, H, W) → replicate to (D, H, W, 4)
            vol = np.stack([vol] * 4, axis=-1)

        # vol shape: (D, H, W, C) with C==4
        # Normalise each channel independently (non-zero voxels).
        for c in range(vol.shape[-1]):
            ch = vol[..., c]
            mask = ch != 0
            if mask.any():
                ch[mask] = (ch[mask] - ch[mask].mean()) / (ch[mask].std() + 1e-8)
            vol[..., c] = ch

        # Convert to (C, D, H, W) then add batch dim → (1, C, D, H, W).
        tensor = torch.from_numpy(vol).permute(3, 0, 1, 2).unsqueeze(0)  # (1, 4, D, H, W)
        return tensor.to(self.device)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _run_2d_slice_by_slice(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run 2-D forward passes over each axial slice and reassemble.

        Pads H and W to the nearest multiple of 16 (2^4 strides in the
        2-D AttentionUnet) before inference, then crops back so that
        mismatched skip-connection sizes cannot occur.

        Args:
            tensor: (1, 4, D, H, W)

        Returns:
            Logits of shape (1, num_classes, D, H, W).
        """
        B, C, D, H, W = tensor.shape  # B==1 here

        # Determine padding needed to make H, W divisible by 16.
        stride_factor = 16  # 2^4 downsampling levels in the 2-D model
        pad_h = (stride_factor - H % stride_factor) % stride_factor
        pad_w = (stride_factor - W % stride_factor) % stride_factor

        if pad_h > 0 or pad_w > 0:
            # F.pad last-dims-first: (left_W, right_W, top_H, bottom_H)
            tensor_padded = F.pad(tensor, (0, pad_w, 0, pad_h))
        else:
            tensor_padded = tensor

        slices_out = []
        for d in range(D):
            s = tensor_padded[:, :, d, :, :]  # (1, 4, H_pad, W_pad)
            logits_s = self.model(s)           # (1, num_classes, H_pad, W_pad)
            # Crop back to original spatial size before accumulating.
            if pad_h > 0 or pad_w > 0:
                logits_s = logits_s[:, :, :H, :W]
            slices_out.append(logits_s)

        # Stack along depth: (1, num_classes, D, H, W)
        return torch.stack([s.squeeze(0) for s in slices_out], dim=1).unsqueeze(0)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_dice(
        self, pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6
    ) -> dict:
        """Compute per-class Dice score against a ground-truth mask.

        Args:
            pred  : Integer array of predicted class indices.
            target: Integer array of ground-truth class indices (same shape).
            smooth: Laplace smoothing term.

        Returns:
            Dict with keys ``"dice_class_1"``, ..., ``"dice_class_N"``,
            and ``"dice_mean"``.
        """
        out = {}
        for c in range(1, self.NUM_CLASSES):
            p = (pred == c).astype(np.float32)
            t = (target == c).astype(np.float32)
            intersection = (p * t).sum()
            denom = p.sum() + t.sum()
            out[f"dice_class_{c}"] = float(
                (2.0 * intersection + smooth) / (denom + smooth)
            )
        vals = list(out.values())
        out["dice_mean"] = float(np.mean(vals)) if vals else 0.0
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _guess_modalities(path: str) -> list[str]:
        """Attempt to identify which MRI modality a filename refers to."""
        lower = Path(path).stem.lower()
        found = []
        for mod in ["t1ce", "t1", "t2", "flair"]:  # t1ce before t1
            if mod in lower:
                found.append(mod)
        return found if found else ["unknown"]
