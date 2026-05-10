"""
DenseNet-121 for chest X-ray classification.

Supports two modes:
  - 'covid': 4-class (COVID / Lung_Opacity / Normal / Viral Pneumonia)
             Loss: CrossEntropyLoss  |  output: raw logits, shape (N, 4)
  - 'chexnet': 14-class multi-label (NIH ChestX-ray14)
               Loss: BCELoss         |  output: sigmoid probabilities, shape (N, 14)

The mode is determined solely by num_classes:
  num_classes == 4  → COVID mode (no sigmoid applied in forward)
  num_classes == 14 → CheXNet mode (Sigmoid applied in forward)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------------------------------------------------------------------------
# Class name lists
# ---------------------------------------------------------------------------

CLASS_NAMES_COVID: list[str] = [
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Viral Pneumonia",
]

CLASS_NAMES_CHEXNET: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ChestDenseNet(nn.Module):
    """DenseNet-121 adapted for chest X-ray classification.

    Args:
        num_classes: Number of output classes (4 for COVID, 14 for CheXNet).
        pretrained:  Load ImageNet weights when True.
        dropout:     Dropout probability inserted before the final linear layer.
                     Set to 0.0 to disable.
    """

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # ----- backbone -----
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        # Keep only the feature extractor; discard the original classifier
        self.features = backbone.features
        in_features: int = backbone.classifier.in_features  # 1024

        # ----- classifier head -----
        layers: list[nn.Module] = []
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(in_features, num_classes))

        # CheXNet multi-label mode: sigmoid inside the model so that raw
        # BCELoss can be applied without numerical instability.
        if num_classes == 14:
            layers.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*layers)

        # Adaptive pool to squeeze spatial dims → (N, C)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (N, 3, H, W) — ImageNet-normalised.

        Returns:
            Logits (N, num_classes) for num_classes == 4, or
            sigmoid probabilities (N, 14)  for num_classes == 14.
        """
        feat = self.features(x)       # (N, 1024, H', W')
        feat = F.relu(feat, inplace=True)  # DenseNet convention
        feat = self.pool(feat)            # (N, 1024, 1, 1)
        feat = torch.flatten(feat, 1)     # (N, 1024)
        return self.classifier(feat)

    # ------------------------------------------------------------------
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature maps *before* the classifier (used for Grad-CAM).

        Returns:
            Feature map tensor (N, 1024, H', W') after relu, before pooling.
        """
        feat = self.features(x)
        return F.relu(feat, inplace=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_densenet121(
    num_classes: int = 4,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """Build a ChestDenseNet and move it to the auto-detected device.

    Args:
        num_classes: 4 (COVID) or 14 (CheXNet).
        pretrained:  Use ImageNet weights.
        dropout:     Dropout before classifier head (0.0 = disabled).

    Returns:
        ChestDenseNet instance on cuda (if available) or cpu.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChestDenseNet(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
    return model.to(device)
