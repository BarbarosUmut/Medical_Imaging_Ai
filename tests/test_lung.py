"""
Unit tests for lung model components.

All tests use synthetic tensors / PIL Images — no real model weights
or image data required.

Run:
    python -m pytest tests/test_lung.py -v
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.lung.densenet121 import (
    CLASS_NAMES_CHEXNET,
    CLASS_NAMES_COVID,
    ChestDenseNet,
    build_densenet121,
)
from models.lung.unet_lung import ConvBlock, LungUNet, build_lung_unet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cpu_model(num_classes: int = 4, pretrained: bool = False) -> ChestDenseNet:
    """Return a ChestDenseNet on CPU without downloading ImageNet weights."""
    return ChestDenseNet(num_classes=num_classes, pretrained=pretrained)


def _synthetic_xray(batch: int = 1, channels: int = 3, h: int = 224, w: int = 224) -> torch.Tensor:
    return torch.randn(batch, channels, h, w)


# ---------------------------------------------------------------------------
# TestChestDenseNet
# ---------------------------------------------------------------------------

class TestChestDenseNet(unittest.TestCase):
    """Tests for ChestDenseNet (DenseNet-121 wrapper)."""

    # ------------------------------------------------------------------
    # build_densenet121 factory
    # ------------------------------------------------------------------

    def test_build_densenet121_4class_output_shape(self):
        """build_densenet121(num_classes=4) should produce (1, 4) output."""
        model = build_densenet121(num_classes=4, pretrained=False)
        model.eval()
        device = next(model.parameters()).device
        x = _synthetic_xray().to(device)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 4))

    def test_build_densenet121_14class_output_shape(self):
        """build_densenet121(num_classes=14) should produce (1, 14) output."""
        model = build_densenet121(num_classes=14, pretrained=False)
        model.eval()
        device = next(model.parameters()).device
        x = _synthetic_xray().to(device)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 14))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def test_forward_4class(self):
        """Forward pass with 4-class model returns correct shape."""
        model = _cpu_model(num_classes=4)
        model.eval()
        x = _synthetic_xray(batch=2)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 4))

    def test_forward_14class_sigmoid_range(self):
        """14-class model applies Sigmoid — all values must be in (0, 1)."""
        model = _cpu_model(num_classes=14)
        model.eval()
        x = _synthetic_xray(batch=2)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (2, 14))
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0),
                        "14-class outputs should be in [0, 1] after Sigmoid")

    def test_forward_4class_no_sigmoid(self):
        """4-class model should NOT apply Sigmoid — logits may exceed [0,1]."""
        model = _cpu_model(num_classes=4)
        model.eval()
        # Weight the classifier to force large positive logits
        with torch.no_grad():
            for p in model.classifier.parameters():
                nn.init.constant_(p, 10.0)
        x = _synthetic_xray()
        with torch.no_grad():
            out = model(x)
        # At least some logits should be > 1 (no sigmoid was applied)
        self.assertTrue(
            out.max().item() > 1.0,
            "4-class model should return raw logits, not sigmoid probabilities",
        )

    # ------------------------------------------------------------------
    # get_features
    # ------------------------------------------------------------------

    def test_get_features_shape(self):
        """get_features should return (N, 1024, H', W') feature maps."""
        model = _cpu_model(num_classes=4)
        model.eval()
        x = _synthetic_xray()
        with torch.no_grad():
            feat = model.get_features(x)
        # DenseNet-121 produces 1024 feature channels
        self.assertEqual(feat.ndim, 4)
        self.assertEqual(feat.shape[0], 1)
        self.assertEqual(feat.shape[1], 1024)
        # Spatial dims should be smaller than input (no global pool yet)
        self.assertLess(feat.shape[2], 224)
        self.assertLess(feat.shape[3], 224)

    def test_get_features_returns_tensor(self):
        """get_features must return a torch.Tensor."""
        model = _cpu_model(num_classes=4)
        model.eval()
        with torch.no_grad():
            feat = model.get_features(_synthetic_xray())
        self.assertIsInstance(feat, torch.Tensor)

    # ------------------------------------------------------------------
    # Class name lists
    # ------------------------------------------------------------------

    def test_class_names_covid_length(self):
        self.assertEqual(len(CLASS_NAMES_COVID), 4)

    def test_class_names_chexnet_length(self):
        self.assertEqual(len(CLASS_NAMES_CHEXNET), 14)


# ---------------------------------------------------------------------------
# TestLungUNet
# ---------------------------------------------------------------------------

class TestLungUNet(unittest.TestCase):
    """Tests for LungUNet (2-D U-Net)."""

    def test_forward_grayscale(self):
        """U-Net should accept (1, 1, 256, 256) and return same spatial dims."""
        model = LungUNet(in_channels=1, out_channels=1)
        model.eval()
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 1, 256, 256))

    def test_output_shape_matches_input(self):
        """Output H×W should exactly match input H×W."""
        model = LungUNet(in_channels=1, out_channels=1)
        model.eval()
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape[2:], x.shape[2:])

    def test_forward_rgb(self):
        """U-Net with in_channels=3 should accept RGB input."""
        model = LungUNet(in_channels=3, out_channels=1)
        model.eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 1, 256, 256))

    def test_output_channels_2(self):
        """out_channels=2 produces two-channel segmentation map."""
        model = LungUNet(in_channels=1, out_channels=2)
        model.eval()
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 2, 256, 256))

    def test_batch_size_4(self):
        """Model should handle batch size > 1."""
        model = LungUNet(in_channels=1, out_channels=1)
        model.eval()
        x = torch.randn(4, 1, 256, 256)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape[0], 4)

    def test_build_lung_unet_factory(self):
        """Factory function returns a LungUNet instance."""
        model = build_lung_unet(in_channels=1, out_channels=1)
        self.assertIsInstance(model, LungUNet)

    def test_conv_block_output_shape(self):
        """ConvBlock should preserve spatial dimensions."""
        block = ConvBlock(in_channels=3, out_channels=64)
        x = torch.randn(1, 3, 64, 64)
        out = block(x)
        self.assertEqual(out.shape, (1, 64, 64, 64))

    def test_custom_features(self):
        """U-Net with custom feature list should still produce correct output."""
        model = LungUNet(in_channels=1, out_channels=1, features=[16, 32, 64])
        model.eval()
        x = torch.randn(1, 1, 128, 128)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(out.shape, (1, 1, 128, 128))


# ---------------------------------------------------------------------------
# TestGradCAM
# ---------------------------------------------------------------------------

class TestGradCAM(unittest.TestCase):
    """Tests for the GradCAM class using a real (untrained) DenseNet-121."""

    def setUp(self):
        """Create a small pretrained-free model for hook testing."""
        # Import here to avoid circular imports at module level
        from inference.lung_inference import GradCAM

        self.GradCAM = GradCAM
        self.model = ChestDenseNet(num_classes=4, pretrained=False)
        self.model.eval()

    def test_generate_returns_2d_array(self):
        """generate() should return a 2-D numpy array."""
        cam = self.GradCAM(self.model)
        x = _synthetic_xray()
        heatmap = cam.generate(x)
        self.assertEqual(heatmap.ndim, 2)
        cam.remove_hooks()

    def test_generate_output_shape(self):
        """Heatmap spatial dims should match input spatial dims."""
        cam = self.GradCAM(self.model)
        x = _synthetic_xray(h=224, w=224)
        heatmap = cam.generate(x)
        self.assertEqual(heatmap.shape, (224, 224))
        cam.remove_hooks()

    def test_generate_values_in_range(self):
        """All heatmap values should be in [0, 1]."""
        cam = self.GradCAM(self.model)
        x = _synthetic_xray()
        heatmap = cam.generate(x)
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)
        cam.remove_hooks()

    def test_generate_with_explicit_class_idx(self):
        """generate() should accept an explicit class_idx without error."""
        cam = self.GradCAM(self.model)
        x = _synthetic_xray()
        heatmap = cam.generate(x, class_idx=2)
        self.assertEqual(heatmap.shape, (224, 224))
        cam.remove_hooks()

    def test_remove_hooks_does_not_raise(self):
        """remove_hooks() should complete without errors."""
        cam = self.GradCAM(self.model)
        try:
            cam.remove_hooks()
        except Exception as exc:
            self.fail(f"remove_hooks() raised an exception: {exc}")


# ---------------------------------------------------------------------------
# TestLungClassifier
# ---------------------------------------------------------------------------

class TestLungClassifier(unittest.TestCase):
    """Tests for LungClassifier using a mocked model weights file."""

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    def _make_classifier(self, num_classes: int = 4) -> "LungClassifier":  # noqa: F821
        """Return a LungClassifier backed by randomly-initialised weights."""
        from inference.lung_inference import LungClassifier

        # Build a random-weight state_dict and patch torch.load
        dummy_model = ChestDenseNet(num_classes=num_classes, pretrained=False)
        state_dict = dummy_model.state_dict()

        with patch("inference.lung_inference.torch.load", return_value=state_dict):
            clf = LungClassifier(
                model_path="fake/path/model.pth",
                num_classes=num_classes,
                device="cpu",
            )
        return clf

    def _synthetic_pil(self, w: int = 224, h: int = 224) -> Image.Image:
        arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    # ------------------------------------------------------------------
    # classify() — output dict structure
    # ------------------------------------------------------------------

    def test_classify_returns_required_keys(self):
        """classify() output dict must contain all documented keys."""
        clf = self._make_classifier(num_classes=4)
        pil = self._synthetic_pil()
        result = clf.classify(pil)
        required = {"predictions", "top_findings", "predicted_class", "confidence", "gradcam_heatmap"}
        self.assertTrue(required.issubset(result.keys()),
                        f"Missing keys: {required - result.keys()}")

    def test_classify_predictions_length(self):
        """predictions dict should have num_classes entries."""
        clf = self._make_classifier(num_classes=4)
        result = clf.classify(self._synthetic_pil())
        self.assertEqual(len(result["predictions"]), 4)

    def test_classify_14class_predictions_length(self):
        """14-class classifier should return 14 entries in predictions."""
        clf = self._make_classifier(num_classes=14)
        result = clf.classify(self._synthetic_pil())
        self.assertEqual(len(result["predictions"]), 14)

    def test_classify_top_findings_length(self):
        """top_findings should always contain 3 entries."""
        clf = self._make_classifier(num_classes=4)
        result = clf.classify(self._synthetic_pil())
        self.assertEqual(len(result["top_findings"]), 3)

    def test_classify_confidence_in_range(self):
        """Confidence must be in [0, 1]."""
        clf = self._make_classifier(num_classes=4)
        result = clf.classify(self._synthetic_pil())
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_classify_predicted_class_in_class_names(self):
        """predicted_class must be one of the class name list."""
        clf = self._make_classifier(num_classes=4)
        result = clf.classify(self._synthetic_pil())
        self.assertIn(result["predicted_class"], CLASS_NAMES_COVID)

    def test_classify_with_numpy_input(self):
        """classify() should accept a numpy array as image_input."""
        clf = self._make_classifier(num_classes=4)
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = clf.classify(arr)
        self.assertIn("predicted_class", result)

    def test_classify_gradcam_heatmap_shape(self):
        """gradcam_heatmap should be a 2-D numpy array of shape (224, 224)."""
        clf = self._make_classifier(num_classes=4)
        result = clf.classify(self._synthetic_pil())
        heatmap = result["gradcam_heatmap"]
        self.assertIsInstance(heatmap, np.ndarray)
        self.assertEqual(heatmap.ndim, 2)
        self.assertEqual(heatmap.shape, (224, 224))

    # ------------------------------------------------------------------
    # _preprocess
    # ------------------------------------------------------------------

    def test_preprocess_output_shape(self):
        """_preprocess should return a (1, 3, 224, 224) tensor."""
        clf = self._make_classifier(num_classes=4)
        pil = self._synthetic_pil()
        tensor = clf._preprocess(pil)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_preprocess_grayscale_pil(self):
        """_preprocess should convert grayscale PIL to (1, 3, 224, 224)."""
        clf = self._make_classifier(num_classes=4)
        gray = Image.fromarray(np.random.randint(0, 255, (300, 300), dtype=np.uint8))
        tensor = clf._preprocess(gray)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    def test_preprocess_numpy_grayscale(self):
        """_preprocess should handle 2-D numpy arrays."""
        clf = self._make_classifier(num_classes=4)
        arr = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        tensor = clf._preprocess(arr)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))

    # ------------------------------------------------------------------
    # overlay_gradcam
    # ------------------------------------------------------------------

    def test_overlay_gradcam_returns_uint8_rgb(self):
        """overlay_gradcam should return uint8 (H, W, 3)."""
        clf = self._make_classifier(num_classes=4)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        heatmap = np.random.rand(224, 224).astype(np.float32)
        result = clf.overlay_gradcam(img, heatmap)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape[2], 3)

    def test_overlay_gradcam_shape_preserved(self):
        """overlay_gradcam output H×W should match the original image."""
        clf = self._make_classifier(num_classes=4)
        h, w = 300, 400
        img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        heatmap = np.random.rand(224, 224).astype(np.float32)  # different size
        result = clf.overlay_gradcam(img, heatmap)
        self.assertEqual(result.shape[:2], (h, w))

    def test_overlay_gradcam_grayscale_image(self):
        """overlay_gradcam should accept a 2-D grayscale image."""
        clf = self._make_classifier(num_classes=4)
        img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        heatmap = np.random.rand(224, 224).astype(np.float32)
        result = clf.overlay_gradcam(img, heatmap)
        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_overlay_values_in_range(self):
        """overlay_gradcam output values must be in [0, 255]."""
        clf = self._make_classifier(num_classes=4)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        heatmap = np.random.rand(224, 224).astype(np.float32)
        result = clf.overlay_gradcam(img, heatmap)
        self.assertGreaterEqual(int(result.min()), 0)
        self.assertLessEqual(int(result.max()), 255)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
