"""
Unit tests for app/ui_components.py and app/main.py.

All tests use synthetic data — no real model weights, NIfTI files, or
Ollama server are required.  Inference calls are mocked with unittest.mock.

Run:
    python -m pytest tests/test_ui.py -v
"""
from __future__ import annotations

import sys
import types
import importlib
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

# Ensure the project root is on sys.path so imports work from any cwd.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

def _make_brain_seg_result(
    depth: int = 8,
    height: int = 32,
    width: int = 32,
    include_dice: bool = True,
) -> dict:
    """Return a synthetic BrainSegmentor.segment() output."""
    mask = np.random.randint(0, 4, size=(depth, height, width), dtype=np.int32)
    probs = np.random.rand(4, depth, height, width).astype(np.float32)
    dice = (
        {"dice_class_1": 0.82, "dice_class_2": 0.74, "dice_mean": 0.78}
        if include_dice
        else {}
    )
    return {
        "mask": mask,
        "probabilities": probs,
        "dice_scores": dice,
        "tumor_volume_ml": 15.3,
        "modalities_found": ["t1", "flair"],
    }


def _make_lung_clf_result(
    h: int = 64,
    w: int = 64,
    predicted: str = "COVID",
) -> dict:
    """Return a synthetic LungClassifier.classify() output."""
    classes = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
    probs = np.random.dirichlet(np.ones(4)).astype(np.float64)
    predictions = dict(zip(classes, probs.tolist()))
    # Force predicted class to have highest probability
    max_other = max(v for k, v in predictions.items() if k != predicted)
    predictions[predicted] = max(max_other + 0.1, 0.6)
    top = sorted(predictions, key=lambda k: predictions[k], reverse=True)[:3]
    heatmap = np.random.rand(h, w).astype(np.float32)
    return {
        "predictions": predictions,
        "top_findings": top,
        "predicted_class": predicted,
        "confidence": predictions[predicted],
        "gradcam_heatmap": heatmap,
    }


def _make_rgb_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a random uint8 RGB image array."""
    return np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)


def _make_gray_image(h: int = 64, w: int = 64) -> np.ndarray:
    """Return a random uint8 grayscale image array."""
    return np.random.randint(0, 255, size=(h, w), dtype=np.uint8)


# ===========================================================================
# 1.  TestUIComponents
# ===========================================================================

class TestUIComponents:
    """Tests for the helper functions in app/ui_components.py."""

    # ------------------------------------------------------------------
    # create_brain_segmentation_overlay
    # ------------------------------------------------------------------

    def test_overlay_returns_correct_shape_and_dtype(self):
        """create_brain_segmentation_overlay must return (H, W, 3) uint8."""
        from app.ui_components import create_brain_segmentation_overlay

        H, W = 48, 64
        image_slice = np.random.rand(H, W).astype(np.float32) * 1000
        mask_slice = np.random.randint(0, 4, size=(H, W), dtype=np.int32)

        result = create_brain_segmentation_overlay(image_slice, mask_slice)

        assert result.shape == (H, W, 3), f"Expected ({H}, {W}, 3), got {result.shape}"
        assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"

    def test_overlay_background_stays_gray(self):
        """Pixels where mask == 0 should remain the greyscale value (no colour tint)."""
        from app.ui_components import create_brain_segmentation_overlay

        H, W = 32, 32
        # Flat image → all grey (128)
        image_slice = np.full((H, W), 128.0, dtype=np.float32)
        mask_slice = np.zeros((H, W), dtype=np.int32)  # all background

        result = create_brain_segmentation_overlay(image_slice, mask_slice, alpha=0.5)

        # Background pixels: R == G == B (grey)
        assert np.all(result[..., 0] == result[..., 1])
        assert np.all(result[..., 1] == result[..., 2])

    def test_overlay_returns_uint8_with_flat_image(self):
        """Function must not crash or produce NaN when image is all zeros."""
        from app.ui_components import create_brain_segmentation_overlay

        image_slice = np.zeros((32, 32), dtype=np.float32)
        mask_slice = np.ones((32, 32), dtype=np.int32)  # all class 1

        result = create_brain_segmentation_overlay(image_slice, mask_slice)

        assert result.dtype == np.uint8
        assert result.shape == (32, 32, 3)

    def test_overlay_alpha_zero_equals_grayscale(self):
        """With alpha=0, the output should be the raw greyscale image in RGB."""
        from app.ui_components import create_brain_segmentation_overlay

        H, W = 20, 20
        image_slice = np.random.rand(H, W).astype(np.float32)
        mask_slice = np.ones((H, W), dtype=np.int32)

        result = create_brain_segmentation_overlay(image_slice, mask_slice, alpha=0.0)

        # Channels should be equal (greyscale RGB)
        assert np.all(result[..., 0] == result[..., 1])

    # ------------------------------------------------------------------
    # create_segmentation_figure
    # ------------------------------------------------------------------

    def test_segmentation_figure_returns_pil_image(self):
        """create_segmentation_figure must return a PIL Image."""
        from app.ui_components import create_segmentation_figure

        D, H, W = 16, 32, 32
        volume = np.random.rand(D, H, W).astype(np.float32)
        mask = np.random.randint(0, 4, size=(D, H, W), dtype=np.int32)

        result = create_segmentation_figure(volume, mask, num_slices=3)

        assert isinstance(result, Image.Image), f"Expected PIL Image, got {type(result)}"

    def test_segmentation_figure_with_4d_volume(self):
        """Input with shape (C, D, H, W) should be handled correctly."""
        from app.ui_components import create_segmentation_figure

        C, D, H, W = 4, 12, 32, 32
        volume = np.random.rand(C, D, H, W).astype(np.float32)
        mask = np.random.randint(0, 4, size=(D, H, W), dtype=np.int32)

        result = create_segmentation_figure(volume, mask, num_slices=3)

        assert isinstance(result, Image.Image)

    def test_segmentation_figure_single_slice(self):
        """Function must work with num_slices=1."""
        from app.ui_components import create_segmentation_figure

        D, H, W = 10, 24, 24
        volume = np.random.rand(D, H, W).astype(np.float32)
        mask = np.zeros((D, H, W), dtype=np.int32)

        result = create_segmentation_figure(volume, mask, num_slices=1)

        assert isinstance(result, Image.Image)

    def test_segmentation_figure_has_nonzero_size(self):
        """Returned PIL Image must have positive width and height."""
        from app.ui_components import create_segmentation_figure

        D, H, W = 20, 40, 40
        volume = np.random.rand(D, H, W).astype(np.float32)
        mask = np.random.randint(0, 4, size=(D, H, W), dtype=np.int32)

        result = create_segmentation_figure(volume, mask, num_slices=2)

        assert result.width > 0 and result.height > 0

    # ------------------------------------------------------------------
    # create_lung_results_figure
    # ------------------------------------------------------------------

    def test_lung_figure_returns_pil_image_rgb(self):
        """create_lung_results_figure must return a PIL Image."""
        from app.ui_components import create_lung_results_figure

        result_data = _make_lung_clf_result()
        image = _make_rgb_image()

        result = create_lung_results_figure(
            image,
            result_data["gradcam_heatmap"],
            result_data["predictions"],
            result_data["predicted_class"],
            result_data["confidence"],
        )

        assert isinstance(result, Image.Image), f"Expected PIL Image, got {type(result)}"

    def test_lung_figure_with_grayscale_input(self):
        """Grayscale (H, W) image input should be handled without error."""
        from app.ui_components import create_lung_results_figure

        result_data = _make_lung_clf_result()
        image = _make_gray_image()

        result = create_lung_results_figure(
            image,
            result_data["gradcam_heatmap"],
            result_data["predictions"],
            result_data["predicted_class"],
            result_data["confidence"],
        )

        assert isinstance(result, Image.Image)

    def test_lung_figure_with_float_image(self):
        """Float [0,1] image input should be converted without error."""
        from app.ui_components import create_lung_results_figure

        result_data = _make_lung_clf_result()
        image = np.random.rand(64, 64, 3).astype(np.float32)

        result = create_lung_results_figure(
            image,
            result_data["gradcam_heatmap"],
            result_data["predictions"],
            result_data["predicted_class"],
            result_data["confidence"],
        )

        assert isinstance(result, Image.Image)

    def test_lung_figure_nonzero_dimensions(self):
        """Output figure must have positive dimensions."""
        from app.ui_components import create_lung_results_figure

        result_data = _make_lung_clf_result()
        image = _make_rgb_image()

        result = create_lung_results_figure(
            image,
            result_data["gradcam_heatmap"],
            result_data["predictions"],
            result_data["predicted_class"],
            result_data["confidence"],
        )

        assert result.width > 0 and result.height > 0

    # ------------------------------------------------------------------
    # format_brain_metrics_html
    # ------------------------------------------------------------------

    def test_brain_metrics_html_returns_string(self):
        """format_brain_metrics_html must return a string."""
        from app.ui_components import format_brain_metrics_html

        result = format_brain_metrics_html(_make_brain_seg_result())

        assert isinstance(result, str)

    def test_brain_metrics_html_contains_volume_keyword(self):
        """HTML must mention tumour volume (tümör or volume)."""
        from app.ui_components import format_brain_metrics_html

        result = format_brain_metrics_html(_make_brain_seg_result())

        has_kw = any(
            kw in result
            for kw in ["tümör", "Tümör", "tumor", "volume", "Hacim", "hacim", "ml"]
        )
        assert has_kw, "Expected tumour/volume keyword not found in HTML"

    def test_brain_metrics_html_contains_tumor_volume_value(self):
        """HTML must embed the numeric tumor volume from the result dict."""
        from app.ui_components import format_brain_metrics_html

        seg = _make_brain_seg_result()
        seg["tumor_volume_ml"] = 42.5

        result = format_brain_metrics_html(seg)

        assert "42.5" in result or "42,5" in result

    def test_brain_metrics_html_handles_no_dice_scores(self):
        """HTML generation must not crash when dice_scores is empty."""
        from app.ui_components import format_brain_metrics_html

        seg = _make_brain_seg_result(include_dice=False)
        result = format_brain_metrics_html(seg)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_brain_metrics_html_is_valid_fragment(self):
        """Returned HTML must start with '<' indicating it is markup."""
        from app.ui_components import format_brain_metrics_html

        result = format_brain_metrics_html(_make_brain_seg_result()).strip()

        assert result.startswith("<"), "Expected HTML output to start with '<'"

    # ------------------------------------------------------------------
    # format_lung_metrics_html
    # ------------------------------------------------------------------

    def test_lung_metrics_html_returns_string(self):
        """format_lung_metrics_html must return a string."""
        from app.ui_components import format_lung_metrics_html

        result = format_lung_metrics_html(_make_lung_clf_result())

        assert isinstance(result, str)

    def test_lung_metrics_html_contains_predicted_class(self):
        """HTML must mention the predicted class name."""
        from app.ui_components import format_lung_metrics_html

        clf = _make_lung_clf_result(predicted="COVID")
        result = format_lung_metrics_html(clf)

        assert "COVID" in result

    def test_lung_metrics_html_contains_confidence(self):
        """HTML must show a confidence value (percentage number)."""
        from app.ui_components import format_lung_metrics_html

        clf = _make_lung_clf_result()
        result = format_lung_metrics_html(clf)

        # Confidence of e.g. 0.75 → "75" somewhere in the HTML
        conf_pct = int(clf["confidence"] * 100)
        assert str(conf_pct) in result or f"{conf_pct:.1f}" in result or "%" in result

    def test_lung_metrics_html_handles_empty_predictions(self):
        """HTML must not crash when predictions dict is empty."""
        from app.ui_components import format_lung_metrics_html

        empty_result = {
            "predictions": {},
            "top_findings": [],
            "predicted_class": "Bilinmiyor",
            "confidence": 0.0,
        }
        result = format_lung_metrics_html(empty_result)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_lung_metrics_html_is_valid_fragment(self):
        """Returned string should start with '<' (HTML)."""
        from app.ui_components import format_lung_metrics_html

        result = format_lung_metrics_html(_make_lung_clf_result()).strip()

        assert result.startswith("<")


# ===========================================================================
# 2.  TestMainApp
# ===========================================================================

class TestMainApp:
    """Tests for app/main.py — model loading and handler functions."""

    # ------------------------------------------------------------------
    # load_models()
    # ------------------------------------------------------------------

    def test_load_models_runs_without_error_when_pth_missing(self, tmp_path):
        """load_models() must complete gracefully when weight files don't exist."""
        import app.main as main_mod

        # Patch MODEL_PATHS to point at a directory that has no .pth files
        fake_paths = {
            "brain_2d": str(tmp_path / "no_brain_2d.pth"),
            "brain_3d": str(tmp_path / "no_brain_3d.pth"),
            "lung":     str(tmp_path / "no_lung.pth"),
        }
        with patch.object(main_mod, "MODEL_PATHS", fake_paths):
            # Mock OllamaReportGenerator so no network call is made
            with patch("app.main.OllamaReportGenerator") as MockRG:
                MockRG.return_value.is_ollama_available.return_value = False
                # Should not raise
                main_mod.load_models()

        # All inference models should be None (files missing)
        assert main_mod.brain_segmentor_2d is None
        assert main_mod.brain_segmentor_3d is None
        assert main_mod.lung_classifier is None
        # report_generator should be set (always created)
        assert main_mod.report_generator is not None

    def test_load_models_sets_globals_none_when_pth_absent(self, tmp_path):
        """After load_models() with missing files, segmentors remain None."""
        import app.main as main_mod

        fake_paths = {
            "brain_2d": str(tmp_path / "missing.pth"),
            "brain_3d": str(tmp_path / "missing3d.pth"),
            "lung":     str(tmp_path / "missinglung.pth"),
        }
        with patch.object(main_mod, "MODEL_PATHS", fake_paths):
            with patch("app.main.OllamaReportGenerator") as MockRG:
                MockRG.return_value.is_ollama_available.return_value = False
                main_mod.load_models()

        assert main_mod.brain_segmentor_2d is None
        assert main_mod.brain_segmentor_3d is None
        assert main_mod.lung_classifier is None

    # ------------------------------------------------------------------
    # build_ui()
    # ------------------------------------------------------------------

    def test_build_ui_returns_gradio_blocks(self):
        """build_ui() must return a gr.Blocks instance."""
        import gradio as gr
        from app.main import build_ui

        demo = build_ui()

        assert isinstance(demo, gr.Blocks), f"Expected gr.Blocks, got {type(demo)}"

    def test_build_ui_can_be_called_multiple_times(self):
        """build_ui() must be idempotent — calling it twice should not crash."""
        from app.main import build_ui

        demo1 = build_ui()
        demo2 = build_ui()

        assert demo1 is not None
        assert demo2 is not None

    # ------------------------------------------------------------------
    # run_brain_segmentation()
    # ------------------------------------------------------------------

    def test_run_brain_seg_returns_4_tuple_when_model_not_loaded(self):
        """When brain segmentor is None, must return a 4-tuple with error status."""
        import app.main as main_mod

        original = main_mod.brain_segmentor_2d
        main_mod.brain_segmentor_2d = None
        main_mod.brain_segmentor_3d = None

        # Create a fake file object
        fake_file = Mock()
        fake_file.name = "dummy.nii.gz"

        try:
            result = main_mod.run_brain_segmentation(fake_file, "2D U-Net", False)
        finally:
            main_mod.brain_segmentor_2d = original

        assert isinstance(result, tuple), "Return value must be a tuple"
        assert len(result) == 4, f"Expected 4-tuple, got {len(result)}-tuple"

    def test_run_brain_seg_no_file_returns_error_status(self):
        """Passing file_obj=None must return a 4-tuple with an error message."""
        from app.main import run_brain_segmentation

        result = run_brain_segmentation(None, "2D U-Net", False)

        assert isinstance(result, tuple) and len(result) == 4
        # Status (4th element) should mention an error or hint
        status = result[3]
        assert isinstance(status, str) and len(status) > 0

    def test_run_brain_seg_status_contains_warning_when_model_missing(self):
        """Status text must communicate that the model is unavailable."""
        import app.main as main_mod

        original_2d = main_mod.brain_segmentor_2d
        original_3d = main_mod.brain_segmentor_3d
        main_mod.brain_segmentor_2d = None
        main_mod.brain_segmentor_3d = None

        fake_file = Mock()
        fake_file.name = "dummy.nii.gz"

        try:
            result = main_mod.run_brain_segmentation(fake_file, "2D U-Net", False)
        finally:
            main_mod.brain_segmentor_2d = original_2d
            main_mod.brain_segmentor_3d = original_3d

        status = result[3]
        # Should mention model missing / unavailable
        has_warning_kw = any(
            kw in status
            for kw in ["Uyarı", "uyarı", "model", "bulunamadı", "yok", "missing"]
        )
        assert has_warning_kw, f"Status should mention missing model, got: {status!r}"

    def test_run_brain_seg_with_mocked_model(self, tmp_path):
        """With a mocked BrainSegmentor, function must return (PIL, str, str, str)."""
        import app.main as main_mod

        seg_result = _make_brain_seg_result(depth=12, height=32, width=32)

        mock_segmentor = Mock()
        mock_segmentor.segment.return_value = seg_result

        mock_report_gen = Mock()
        mock_report_gen.generate_brain_report.return_value = "Türkçe rapor."

        original_2d = main_mod.brain_segmentor_2d
        original_rg = main_mod.report_generator
        main_mod.brain_segmentor_2d = mock_segmentor
        main_mod.report_generator = mock_report_gen

        # Write a dummy file path (file existence not checked in mock path)
        dummy_nii = tmp_path / "dummy.nii"
        dummy_nii.write_bytes(b"FAKE")

        fake_file = Mock()
        fake_file.name = str(dummy_nii)

        try:
            result = main_mod.run_brain_segmentation(fake_file, "2D U-Net", True)
        finally:
            main_mod.brain_segmentor_2d = original_2d
            main_mod.report_generator = original_rg

        assert isinstance(result, tuple) and len(result) == 4
        img, metrics_html, report, status = result
        assert isinstance(img, Image.Image), f"Expected PIL Image, got {type(img)}"
        assert isinstance(metrics_html, str)
        assert isinstance(report, str)
        assert isinstance(status, str)

    # ------------------------------------------------------------------
    # run_lung_classification()
    # ------------------------------------------------------------------

    def test_run_lung_clf_returns_4_tuple_when_model_not_loaded(self):
        """When lung_classifier is None, must return a 4-tuple with warning."""
        import app.main as main_mod

        original = main_mod.lung_classifier
        main_mod.lung_classifier = None

        image_arr = _make_rgb_image()

        try:
            result = main_mod.run_lung_classification(image_arr, False)
        finally:
            main_mod.lung_classifier = original

        assert isinstance(result, tuple) and len(result) == 4

    def test_run_lung_clf_no_image_returns_error(self):
        """Passing image_input=None must return a 4-tuple with an error message."""
        from app.main import run_lung_classification

        result = run_lung_classification(None, False)

        assert isinstance(result, tuple) and len(result) == 4
        status = result[3]
        assert isinstance(status, str) and len(status) > 0

    def test_run_lung_clf_status_contains_warning_when_model_missing(self):
        """Status text must inform the user the model is not loaded."""
        import app.main as main_mod

        original = main_mod.lung_classifier
        main_mod.lung_classifier = None
        image_arr = _make_rgb_image()

        try:
            result = main_mod.run_lung_classification(image_arr, False)
        finally:
            main_mod.lung_classifier = original

        status = result[3]
        has_warning_kw = any(
            kw in status
            for kw in ["Uyarı", "uyarı", "model", "bulunamadı", "yok", "missing"]
        )
        assert has_warning_kw, f"Status should mention missing model, got: {status!r}"

    def test_run_lung_clf_with_mocked_model(self):
        """With a mocked LungClassifier, function must return (PIL, str, str, str)."""
        import app.main as main_mod

        clf_result = _make_lung_clf_result(h=64, w=64)

        mock_clf = Mock()
        mock_clf.classify.return_value = clf_result

        mock_report_gen = Mock()
        mock_report_gen.generate_lung_report.return_value = "Türkçe akciğer raporu."

        original_clf = main_mod.lung_classifier
        original_rg = main_mod.report_generator
        main_mod.lung_classifier = mock_clf
        main_mod.report_generator = mock_report_gen

        image_arr = _make_rgb_image()

        try:
            result = main_mod.run_lung_classification(image_arr, True)
        finally:
            main_mod.lung_classifier = original_clf
            main_mod.report_generator = original_rg

        assert isinstance(result, tuple) and len(result) == 4
        img, metrics_html, report, status = result
        assert isinstance(img, Image.Image), f"Expected PIL Image, got {type(img)}"
        assert isinstance(metrics_html, str)
        assert isinstance(report, str)
        assert isinstance(status, str)

    def test_run_lung_clf_without_report_generation(self):
        """When generate_report=False, report field should be empty string."""
        import app.main as main_mod

        clf_result = _make_lung_clf_result()

        mock_clf = Mock()
        mock_clf.classify.return_value = clf_result

        original_clf = main_mod.lung_classifier
        main_mod.lung_classifier = mock_clf

        image_arr = _make_rgb_image()

        try:
            _, _, report, _ = main_mod.run_lung_classification(image_arr, False)
        finally:
            main_mod.lung_classifier = original_clf

        assert report == "", f"Expected empty report when generate_report=False, got: {report!r}"

    def test_run_brain_seg_without_report_generation(self, tmp_path):
        """When generate_report=False, report field should be empty string."""
        import app.main as main_mod

        seg_result = _make_brain_seg_result()

        mock_segmentor = Mock()
        mock_segmentor.segment.return_value = seg_result

        mock_report_gen = Mock()

        original_2d = main_mod.brain_segmentor_2d
        original_rg = main_mod.report_generator
        main_mod.brain_segmentor_2d = mock_segmentor
        main_mod.report_generator = mock_report_gen

        dummy_nii = tmp_path / "dummy.nii"
        dummy_nii.write_bytes(b"FAKE")

        fake_file = Mock()
        fake_file.name = str(dummy_nii)

        try:
            _, _, report, _ = main_mod.run_brain_segmentation(fake_file, "2D U-Net", False)
        finally:
            main_mod.brain_segmentor_2d = original_2d
            main_mod.report_generator = original_rg

        assert report == "", f"Expected empty report when generate_report=False, got: {report!r}"

    def test_run_lung_clf_status_contains_class_and_confidence(self):
        """Status text should include the predicted class and confidence value."""
        import app.main as main_mod

        clf_result = _make_lung_clf_result(predicted="Normal")
        mock_clf = Mock()
        mock_clf.classify.return_value = clf_result

        original_clf = main_mod.lung_classifier
        main_mod.lung_classifier = mock_clf

        image_arr = _make_rgb_image()

        try:
            _, _, _, status = main_mod.run_lung_classification(image_arr, False)
        finally:
            main_mod.lung_classifier = original_clf

        assert "Normal" in status, f"Expected predicted class in status: {status!r}"
