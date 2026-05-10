"""
3D Attention U-Net model for brain MRI tumour segmentation.

Architecture: MONAI AttentionUnet
  - spatial_dims : 3
  - in_channels  : 4  (T1, T1ce, T2, FLAIR)
  - out_channels : 4  (background + 3 tumour sub-regions)
  - channels     : (32, 64, 128, 256)
  - strides      : (2, 2, 2)
  - dropout      : 0.1

Sliding-window inference uses roi_size=(128,128,128), sw_batch_size=2,
overlap=0.5, mode="gaussian".
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import AttentionUnet


class BrainUNet3D(nn.Module):
    """Thin wrapper around MONAI's AttentionUnet configured for 3-D brain segmentation."""

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        strides: tuple[int, ...] = (2, 2, 2),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.model = AttentionUnet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            channels=channels,
            strides=strides,
            dropout=dropout,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_unet_3d(in_channels: int = 4, num_classes: int = 4) -> BrainUNet3D:
    """Return a BrainUNet3D (Attention U-Net) placed on the appropriate device."""
    return BrainUNet3D(in_channels=in_channels, num_classes=num_classes)


def get_sliding_window_inferer(
    roi_size: tuple[int, int, int] = (128, 128, 128),
    sw_batch_size: int = 2,
    overlap: float = 0.5,
    mode: str = "gaussian",
) -> SlidingWindowInferer:
    return SlidingWindowInferer(
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode=mode,
    )
