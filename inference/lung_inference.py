"""
LungClassifier: chest X-ray classification with Grad-CAM visualisation.

Wraps ChestDenseNet (models/lung/densenet121.py) for single-image inference.

Typical usage
-------------
    from inference.lung_inference import LungClassifier

    clf = LungClassifier(model_path="models/lung/chexnet_best.pth", num_classes=4)
    result = clf.classify("path/to/xray.png")
    print(result["predicted_class"], result["confidence"])

    overlay = clf.overlay_gradcam(np.array(image), result["gradcam_heatmap"])
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when module is run directly
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.lung.densenet121 import (
    CLASS_NAMES_CHEXNET,
    CLASS_NAMES_COVID,
    ChestDenseNet,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

INPUT_SIZE: int = 224  # pixels

# Grad-CAM colour map: blue → green → red (OpenCV COLORMAP_JET equivalent)
_JET_COLOURS = np.array(
    [
        [0, 0, 128],
        [0, 0, 255],
        [0, 128, 255],
        [0, 255, 255],
        [128, 255, 128],
        [255, 255, 0],
        [255, 128, 0],
        [255, 0, 0],
        [128, 0, 0],
    ],
    dtype=np.float32,
)


def _jet_colormap(t: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB colours using a jet-like palette.

    Args:
        t: Array of arbitrary shape with values in [0, 1].

    Returns:
        uint8 RGB array of shape (*t.shape, 3).
    """
    t = np.clip(t, 0.0, 1.0)
    n = len(_JET_COLOURS) - 1
    idx_f = t * n
    idx_lo = np.floor(idx_f).astype(int).clip(0, n - 1)
    idx_hi = (idx_lo + 1).clip(0, n)
    frac = (idx_f - idx_lo)[..., np.newaxis]
    rgb = (1.0 - frac) * _JET_COLOURS[idx_lo] + frac * _JET_COLOURS[idx_hi]
    return rgb.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    """Grad-CAM for DenseNet-121.

    Hooks into *target_layer* (default ``features.denseblock4``) to capture
    forward activations and backward gradients, then produces a spatial
    heatmap of shape (H, W) normalised to [0, 1].

    Args:
        model:        ChestDenseNet (or any nn.Module with an attribute path
                      reachable via ``getattr`` chains).
        target_layer: Dot-separated attribute path from *model* to the layer.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str = "features.denseblock4",
    ) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        # Resolve nested attribute path, e.g. "features.denseblock4"
        layer = model
        for part in target_layer.split("."):
            layer = getattr(layer, part)
        self._target_layer = layer

        self._fwd_hook = self._target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = self._target_layer.register_full_backward_hook(self._save_gradients)

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _save_activations(
        self,
        _module: nn.Module,
        _input: tuple,
        output: torch.Tensor,
    ) -> None:
        self._activations = output.detach()

    def _save_gradients(
        self,
        _module: nn.Module,
        _grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap.

        Args:
            input_tensor: Pre-processed image tensor (1, C, H, W) on the
                          same device as the model.  Grad computation is
                          enabled internally — the tensor does not need
                          ``requires_grad=True``.
            class_idx:    Target class index.  ``None`` → argmax of output.

        Returns:
            Numpy array of shape (H, W) with values in [0, 1].
        """
        self.model.eval()

        # Need gradients for Grad-CAM
        input_tensor = input_tensor.clone().requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)  # (1, num_classes) or (1, num_classes) with sigmoid

        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())

        # Scalar score for the target class
        score = output[0, class_idx]

        self.model.zero_grad()
        score.backward()

        # Grad-CAM formula: ReLU( mean(grads over H,W) * activations )
        grads = self._gradients        # (1, C, H', W')
        acts = self._activations       # (1, C, H', W')

        if grads is None or acts is None:
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h, w), dtype=np.float32)

        weights = grads.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Upsample to input resolution
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()  # (H, W)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
        """Remove forward and backward hooks from the target layer."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ---------------------------------------------------------------------------
# LungClassifier
# ---------------------------------------------------------------------------

class LungClassifier:
    """Chest X-ray classifier wrapping ChestDenseNet.

    Args:
        model_path:  Path to a checkpoint saved by MetricsLogger
                     (dict with key ``model_state_dict``) or a plain
                     ``state_dict`` file.
        num_classes: 4 (COVID mode) or 14 (CheXNet mode).
        device:      ``'cuda'``, ``'cpu'``, or ``None`` (auto-detect).
    """

    def __init__(
        self,
        model_path: str,
        num_classes: int = 4,
        device: str | None = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.num_classes = num_classes
        self.class_names: list[str] = (
            CLASS_NAMES_COVID if num_classes == 4 else CLASS_NAMES_CHEXNET
        )

        # Build model — dropout=0.3 matches the training configuration
        self.model = ChestDenseNet(num_classes=num_classes, pretrained=False, dropout=0.3)
        self._load_weights(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Grad-CAM helper (lazy: initialised once, reused)
        self._gradcam = GradCAM(self.model, target_layer="features.denseblock4")

        # Preprocessing transform (inference — no augmentation)
        self._transform = transforms.Compose(
            [
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_weights(self, model_path: str) -> None:
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        image: str | Path | Image.Image | np.ndarray,
    ) -> torch.Tensor:
        """Convert various input types to a normalised (1, 3, 224, 224) tensor.

        Args:
            image: File path, PIL Image, or numpy array (H, W) / (H, W, C).

        Returns:
            Tensor of shape (1, 3, 224, 224) on self.device.
        """
        if isinstance(image, (str, Path)):
            pil = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                # Grayscale → RGB
                pil = Image.fromarray(image).convert("RGB")
            else:
                pil = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        tensor = self._transform(pil)          # (3, 224, 224)
        return tensor.unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(
        self,
        image_input: str | Path | Image.Image | np.ndarray,
    ) -> dict:
        """Run classification (and Grad-CAM) on a chest X-ray image.

        Args:
            image_input: File path (str / Path), PIL Image, or numpy array.

        Returns:
            dict with keys:
                ``predictions``      — {class_name: probability}
                ``top_findings``     — top-3 class names sorted by probability
                ``predicted_class``  — argmax class name
                ``confidence``       — max probability
                ``gradcam_heatmap``  — numpy (224, 224) float32 in [0, 1]
        """
        tensor = self._preprocess(image_input)  # (1, 3, 224, 224)

        with torch.no_grad():
            raw_output = self.model(tensor)  # (1, num_classes)

        # For COVID (4-class, raw logits): apply softmax to get probabilities.
        # For CheXNet (14-class): model already applies sigmoid → probabilities.
        if self.num_classes == 4:
            probs = torch.softmax(raw_output, dim=1)
        else:
            probs = raw_output  # already sigmoid

        probs_np: np.ndarray = probs.squeeze(0).cpu().numpy()  # (num_classes,)

        predictions: dict[str, float] = {
            name: float(p) for name, p in zip(self.class_names, probs_np)
        }

        sorted_classes = sorted(predictions, key=lambda k: predictions[k], reverse=True)
        predicted_class: str = sorted_classes[0]
        confidence: float = predictions[predicted_class]
        top_findings: list[str] = sorted_classes[:3]

        # Grad-CAM — target the predicted class
        pred_idx = self.class_names.index(predicted_class)
        # Re-run forward through GradCAM (needs grad, so cannot use no_grad block)
        gradcam_heatmap: np.ndarray = self._gradcam.generate(tensor, class_idx=pred_idx)

        return {
            "predictions": predictions,
            "top_findings": top_findings,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "gradcam_heatmap": gradcam_heatmap,
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def overlay_gradcam(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay a Grad-CAM heatmap on the original image.

        Args:
            image:   Original image as numpy array.  Accepted shapes:
                     (H, W) grayscale, (H, W, 1), (H, W, 3) or (H, W, 4).
                     dtype can be uint8 [0, 255] or float [0, 1].
            heatmap: Grad-CAM heatmap (H', W') with values in [0, 1].
            alpha:   Heatmap opacity (0 = original only, 1 = heatmap only).

        Returns:
            Blended RGB image as uint8 numpy array of shape (H, W, 3).
        """
        # Ensure uint8 RGB
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.concatenate([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]

        h, w = image.shape[:2]

        # Resize heatmap to match image spatial dims
        if heatmap.shape != (h, w):
            heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
            heatmap_pil = heatmap_pil.resize((w, h), Image.BILINEAR)
            heatmap = np.array(heatmap_pil, dtype=np.float32) / 255.0

        # Colour map
        coloured = _jet_colormap(heatmap)  # (H, W, 3) uint8

        # Blend
        blended = (
            (1.0 - alpha) * image.astype(np.float32)
            + alpha * coloured.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        return blended
