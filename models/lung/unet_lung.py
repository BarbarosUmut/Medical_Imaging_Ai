"""
U-Net for lung region segmentation from chest X-rays.

Binary segmentation: background vs lung region.

Architecture:
  - Encoder: 4 stages of ConvBlock + MaxPool2d (stride 2)
  - Bottleneck: ConvBlock (no pooling)
  - Decoder: 4 stages of bilinear upsample + skip concatenation + ConvBlock
  - Output: 1×1 Conv2d → final logits

Default feature sizes: [64, 128, 256, 512]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU.

    Args:
        in_channels:  Number of input channels.
        out_channels: Number of output channels (both conv layers produce this).
        kernel_size:  Convolution kernel size (default 3).
        padding:      Padding added to both sides (default 1 → preserves H×W).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class LungUNet(nn.Module):
    """Encoder-decoder U-Net for 2-D lung region segmentation.

    Args:
        in_channels:  Number of input image channels (1 = grayscale, 3 = RGB).
        out_channels: Number of output channels.
                      Use 1 for binary segmentation with BCEWithLogitsLoss,
                      or 2 for two-class with CrossEntropyLoss.
        features:     Channel counts for the four encoder stages.
                      Default: [64, 128, 256, 512].
    """

    _DEFAULT_FEATURES: list[int] = [64, 128, 256, 512]

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list[int] | None = None,
    ) -> None:
        super().__init__()
        if features is None:
            features = self._DEFAULT_FEATURES

        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ----- Encoder -----
        prev_ch = in_channels
        for feat in features:
            self.encoder_blocks.append(ConvBlock(prev_ch, feat))
            prev_ch = feat

        # ----- Bottleneck -----
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # ----- Decoder -----
        # Each decoder stage: upsample + ConvBlock(feat*2 → feat)
        # Input channels = feat (from upsample) + feat (skip) = feat*2
        self.decoder_blocks = nn.ModuleList()
        decoder_features = list(reversed(features))
        bottleneck_ch = features[-1] * 2
        prev_ch = bottleneck_ch
        for feat in decoder_features:
            # up conv halves channels, then concatenate skip → feat*2 total
            self.decoder_blocks.append(ConvBlock(prev_ch + feat, feat))
            prev_ch = feat

        # ----- Output projection -----
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (N, in_channels, H, W).

        Returns:
            Segmentation logits (N, out_channels, H, W).
            Apply sigmoid (out_channels=1) or softmax (out_channels>1)
            externally before computing the final prediction.
        """
        # ----- Encoder -----
        skip_connections: list[torch.Tensor] = []
        for enc in self.encoder_blocks:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        # ----- Bottleneck -----
        x = self.bottleneck(x)

        # ----- Decoder -----
        skip_connections = list(reversed(skip_connections))
        for dec, skip in zip(self.decoder_blocks, skip_connections):
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.output_conv(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_lung_unet(
    in_channels: int = 1,
    out_channels: int = 1,
    features: list[int] | None = None,
) -> nn.Module:
    """Build a LungUNet and move it to the auto-detected device.

    Args:
        in_channels:  1 (grayscale) or 3 (RGB).
        out_channels: 1 (binary BCEWithLogitsLoss) or 2 (CrossEntropyLoss).
        features:     Encoder feature counts. Defaults to [64, 128, 256, 512].

    Returns:
        LungUNet instance on cuda (if available) or cpu.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LungUNet(in_channels=in_channels, out_channels=out_channels, features=features)
    return model.to(device)
