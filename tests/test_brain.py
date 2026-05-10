"""
Unit tests for brain MRI segmentation components.

All tests use synthetic data — no real model weights or imaging data required.
Ollama / network calls are not made.

Run:
    python -m pytest tests/test_brain.py -v
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Ensure project root is on path so imports work from any working directory.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from monai.inferers import SlidingWindowInferer

from models.brain.unet_2d import BrainUNet2D, build_unet_2d
from models.brain.unet_3d import BrainUNet3D, build_unet_3d, get_sliding_window_inferer
from inference.brain_inference import BrainSegmentor, visualize_segmentation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")  # force CPU for unit tests


# ---------------------------------------------------------------------------
# 1. TestBrainUNet2D
# ---------------------------------------------------------------------------

class TestBrainUNet2D:

    def test_build_returns_brain_unet2d(self):
        model = build_unet_2d()
        assert isinstance(model, BrainUNet2D)

    def test_output_shape(self, device):
        model = build_unet_2d(in_channels=4, num_classes=4).to(device)
        x = torch.zeros(1, 4, 128, 128, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4, 128, 128), (
            f"Expected (1, 4, 128, 128), got {tuple(out.shape)}"
        )

    def test_forward_pass_runs_without_error(self, device):
        model = build_unet_2d(in_channels=4, num_classes=4).to(device)
        x = torch.randn(1, 4, 128, 128, device=device)
        with torch.no_grad():
            out = model(x)
        assert out is not None

    def test_custom_num_classes(self, device):
        model = build_unet_2d(in_channels=4, num_classes=2).to(device)
        x = torch.zeros(1, 4, 64, 64, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == 2

    def test_batch_size_greater_than_one(self, device):
        model = build_unet_2d().to(device)
        x = torch.zeros(2, 4, 64, 64, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 2


# ---------------------------------------------------------------------------
# 2. TestBrainUNet3D
# ---------------------------------------------------------------------------

class TestBrainUNet3D:

    def test_build_returns_brain_unet3d(self):
        model = build_unet_3d()
        assert isinstance(model, BrainUNet3D)

    def test_output_shape_small_volume(self, device):
        """Use 32³ input — 16³ reduces to 1³ at bottleneck and breaks InstanceNorm."""
        model = build_unet_3d(in_channels=4, num_classes=4).to(device)
        x = torch.zeros(1, 4, 32, 32, 32, device=device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 4, 32, 32, 32), (
            f"Expected (1, 4, 32, 32, 32), got {tuple(out.shape)}"
        )

    def test_forward_pass_runs_without_error(self, device):
        model = build_unet_3d(in_channels=4, num_classes=4).to(device)
        x = torch.randn(1, 4, 32, 32, 32, device=device)
        with torch.no_grad():
            out = model(x)
        assert out is not None

    def test_get_sliding_window_inferer_returns_correct_type(self):
        inferer = get_sliding_window_inferer()
        assert isinstance(inferer, SlidingWindowInferer)

    def test_sliding_window_inferer_default_params(self):
        inferer = get_sliding_window_inferer()
        assert inferer.roi_size == (128, 128, 128)
        assert inferer.sw_batch_size == 4
        assert inferer.overlap == 0.5

    def test_sliding_window_inferer_custom_params(self):
        inferer = get_sliding_window_inferer(
            roi_size=(64, 64, 64), sw_batch_size=2, overlap=0.25, mode="constant"
        )
        assert inferer.roi_size == (64, 64, 64)
        assert inferer.sw_batch_size == 2


# ---------------------------------------------------------------------------
# 3. TestBrainSegmentor
# ---------------------------------------------------------------------------

def _make_mock_segmentor(mode: str = "2d") -> BrainSegmentor:
    """Return a BrainSegmentor with weights loaded from a synthetic checkpoint."""
    if mode == "2d":
        real_model = build_unet_2d(in_channels=4, num_classes=4)
    else:
        real_model = build_unet_3d(in_channels=4, num_classes=4)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        ckpt_path = f.name
        torch.save({"model_state_dict": real_model.state_dict()}, ckpt_path)

    segmentor = BrainSegmentor(model_path=ckpt_path, mode=mode, device="cpu")
    return segmentor


class TestBrainSegmentor:

    # --- _preprocess ---

    def test_preprocess_single_channel_3d(self):
        seg = _make_mock_segmentor(mode="2d")
        vol = np.random.rand(16, 32, 32).astype(np.float32)
        tensor = seg._preprocess(vol)
        # Expected: (1, 4, D, H, W)
        assert tensor.shape == (1, 4, 16, 32, 32)

    def test_preprocess_multi_channel(self):
        seg = _make_mock_segmentor(mode="2d")
        vol = np.random.rand(16, 32, 32, 4).astype(np.float32)
        tensor = seg._preprocess(vol)
        assert tensor.shape == (1, 4, 16, 32, 32)

    def test_preprocess_output_is_cuda_or_cpu_tensor(self):
        seg = _make_mock_segmentor(mode="2d")
        vol = np.random.rand(8, 16, 16).astype(np.float32)
        tensor = seg._preprocess(vol)
        assert isinstance(tensor, torch.Tensor)

    # --- segment with synthetic NIfTI ---

    def test_segment_nifti_returns_expected_keys(self, tmp_path):
        import nibabel as nib

        # _preprocess treats shape as (D, H, W); 2D slices become (H, W).
        # 4 stride-2 layers need H,W >= 32.  Use (D=4, H=64, W=64).
        data = np.random.rand(4, 64, 64).astype(np.float32)
        nii = nib.Nifti1Image(data, affine=np.eye(4))
        nii_path = str(tmp_path / "test_t1.nii.gz")
        nib.save(nii, nii_path)

        seg = _make_mock_segmentor(mode="2d")
        result = seg.segment(nii_path)

        assert "mask" in result
        assert "probabilities" in result
        assert "dice_scores" in result
        assert "tumor_volume_ml" in result
        assert "modalities_found" in result

    def test_segment_nifti_mask_shape(self, tmp_path):
        import nibabel as nib

        data = np.random.rand(4, 64, 64).astype(np.float32)
        nii = nib.Nifti1Image(data, affine=np.eye(4))
        nii_path = str(tmp_path / "test_vol.nii.gz")
        nib.save(nii, nii_path)

        seg = _make_mock_segmentor(mode="2d")
        result = seg.segment(nii_path)

        # mask should be 3-D integer array.
        assert result["mask"].ndim == 3
        assert result["mask"].dtype in (np.int32, np.int64)

    def test_segment_nifti_probabilities_shape(self, tmp_path):
        import nibabel as nib

        D, H, W = 4, 64, 64
        data = np.random.rand(D, H, W).astype(np.float32)
        nii = nib.Nifti1Image(data, affine=np.eye(4))
        nii_path = str(tmp_path / "test_prob.nii.gz")
        nib.save(nii, nii_path)

        seg = _make_mock_segmentor(mode="2d")
        result = seg.segment(nii_path)

        probs = result["probabilities"]
        assert probs.shape[0] == 4, "First dim should be num_classes=4"

    def test_tumor_volume_ml_non_negative(self, tmp_path):
        import nibabel as nib

        data = np.random.rand(4, 64, 64).astype(np.float32)
        nii = nib.Nifti1Image(data, affine=np.eye(4))
        nii_path = str(tmp_path / "test_vol2.nii.gz")
        nib.save(nii, nii_path)

        seg = _make_mock_segmentor(mode="2d")
        result = seg.segment(nii_path)
        assert result["tumor_volume_ml"] >= 0.0

    # --- compute_dice ---

    def test_compute_dice_perfect_prediction(self):
        seg = _make_mock_segmentor(mode="2d")
        pred = np.array([0, 1, 2, 3])
        target = np.array([0, 1, 2, 3])
        dice = seg.compute_dice(pred, target)
        # Perfect overlap → dice should be 1.0 for each foreground class.
        for c in range(1, 4):
            assert dice[f"dice_class_{c}"] == pytest.approx(1.0, abs=1e-3)

    def test_compute_dice_zero_prediction(self):
        seg = _make_mock_segmentor(mode="2d")
        pred = np.zeros(100, dtype=np.int32)
        target = np.ones(100, dtype=np.int32)
        dice = seg.compute_dice(pred, target)
        assert dice["dice_class_1"] < 0.1

    def test_compute_dice_returns_mean(self):
        seg = _make_mock_segmentor(mode="2d")
        pred = np.array([0, 1, 2, 3, 0, 1])
        target = np.array([0, 1, 2, 3, 0, 1])
        dice = seg.compute_dice(pred, target)
        assert "dice_mean" in dice

    # --- visualize_segmentation ---

    def test_visualize_segmentation_output_shape(self):
        image_slice = np.random.rand(64, 64).astype(np.float32)
        mask_slice = np.random.randint(0, 4, (64, 64), dtype=np.int32)
        rgb = visualize_segmentation(image_slice, mask_slice)
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8

    def test_visualize_segmentation_saves_file(self, tmp_path):
        image_slice = np.random.rand(32, 32).astype(np.float32)
        mask_slice = np.zeros((32, 32), dtype=np.int32)
        mask_slice[10:20, 10:20] = 1
        save_path = str(tmp_path / "overlay.png")
        rgb = visualize_segmentation(image_slice, mask_slice, save_path=save_path)
        assert Path(save_path).exists()
        assert rgb is not None

    def test_visualize_segmentation_all_background(self):
        image_slice = np.ones((32, 32), dtype=np.float32)
        mask_slice = np.zeros((32, 32), dtype=np.int32)
        rgb = visualize_segmentation(image_slice, mask_slice)
        assert rgb.shape == (32, 32, 3)


# ---------------------------------------------------------------------------
# 4. TestTrainBrainScript
# ---------------------------------------------------------------------------

class TestTrainBrainScript:

    def test_train_brain_importable(self):
        """The training script can be imported without executing main()."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_brain",
            str(_PROJECT_ROOT / "training" / "train_brain.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod is not None

    def test_build_model_2d_returns_brain_unet2d(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_brain",
            str(_PROJECT_ROOT / "training" / "train_brain.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        model = mod.build_model("2d")
        assert isinstance(model, BrainUNet2D)

    def test_build_model_3d_returns_brain_unet3d(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_brain",
            str(_PROJECT_ROOT / "training" / "train_brain.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        model = mod.build_model("3d")
        assert isinstance(model, BrainUNet3D)

    def test_parse_args_defaults(self):
        """parse_args() returns sensible defaults when no CLI args are given."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_brain",
            str(_PROJECT_ROOT / "training" / "train_brain.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with patch("sys.argv", ["train_brain.py"]):
            args = mod.parse_args()

        assert args.mode == "3d"
        assert args.epochs == 100
        assert args.lr == pytest.approx(1e-4)
        assert args.batch_size is None  # resolved later in main()
        assert args.data_dir == "data/brain"
        assert args.results_dir == "results"

    def test_parse_args_2d_mode(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "train_brain",
            str(_PROJECT_ROOT / "training" / "train_brain.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with patch("sys.argv", ["train_brain.py", "--mode", "2d", "--epochs", "50"]):
            args = mod.parse_args()

        assert args.mode == "2d"
        assert args.epochs == 50
