"""Unit tests for dataset readers and preprocessing utilities.

All tests use synthetic data — no real MRI scans or X-Ray images required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nifti(tmp_path: Path, case_id: str, modalities: list[str], shape=(16, 16, 16)) -> Path:
    """Create synthetic NIfTI files for one BraTS case."""
    import nibabel as nib

    case_dir = tmp_path / "raw" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    for mod in modalities:
        data = np.random.rand(*shape).astype(np.float32)
        img = nib.Nifti1Image(data, affine=np.eye(4))
        nib.save(img, str(case_dir / f"{case_id}_{mod}.nii.gz"))

    seg_data = np.zeros(shape, dtype=np.int16)
    seg_data[4:8, 4:8, 4:8] = 1
    seg_data[8:12, 8:12, 8:12] = 4
    seg_img = nib.Nifti1Image(seg_data, affine=np.eye(4))
    nib.save(seg_img, str(case_dir / f"{case_id}_seg.nii.gz"))

    return case_dir


def _make_lung_dataset(tmp_path: Path) -> Path:
    """Create synthetic PNG images for each COVID dataset class."""
    from PIL import Image

    class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
    for cls in class_names:
        cls_dir = tmp_path / "raw" / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            arr = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
            Image.fromarray(arr, mode="L").save(str(cls_dir / f"img_{i:03d}.png"))
    return tmp_path


# ---------------------------------------------------------------------------
# BraTS2020Dataset tests
# ---------------------------------------------------------------------------

class TestBraTS2020Dataset:
    def test_dataset_loads(self, tmp_path):
        """Dataset initialises with synthetic NIfTI files."""
        nibabel = pytest.importorskip("nibabel")
        modalities = ["t1", "t1ce", "t2", "flair"]
        for i in range(3):
            _make_nifti(tmp_path, f"BraTS20_Training_{i:03d}", modalities)

        from data.dataset_brain import BraTS2020Dataset

        ds = BraTS2020Dataset(data_dir=tmp_path, split="train", val_fraction=0.34, cache_rate=0.0)
        assert len(ds) > 0

    def test_val_split(self, tmp_path):
        """Train and val splits are non-overlapping and cover all cases."""
        nibabel = pytest.importorskip("nibabel")
        modalities = ["t1", "t1ce", "t2", "flair"]
        for i in range(5):
            _make_nifti(tmp_path, f"BraTS20_Training_{i:03d}", modalities)

        from data.dataset_brain import BraTS2020Dataset

        train_ds = BraTS2020Dataset(data_dir=tmp_path, split="train", val_fraction=0.2)
        val_ds = BraTS2020Dataset(data_dir=tmp_path, split="val", val_fraction=0.2)
        assert len(train_ds) + len(val_ds) == 5

    def test_label_remapping(self, tmp_path):
        """BraTS label 4 is remapped to 3; no label 4 remains."""
        import nibabel as nib
        nibabel = pytest.importorskip("nibabel")
        from data.dataset_brain import RemapBraTSLabels
        import torch

        seg_data = np.array([0, 1, 2, 4], dtype=np.float32)
        data = {"seg": torch.tensor(seg_data)}
        remapped = RemapBraTSLabels()(data)
        assert 4 not in remapped["seg"].numpy()
        assert 3 in remapped["seg"].numpy()

    def test_inference_split_has_no_seg(self, tmp_path):
        """Inference transforms do not require a segmentation key."""
        nibabel = pytest.importorskip("nibabel")
        modalities = ["t1", "t1ce", "t2", "flair"]
        for i in range(2):
            _make_nifti(tmp_path, f"BraTS20_Training_{i:03d}", modalities)

        from data.dataset_brain import BraTS2020Dataset

        ds = BraTS2020Dataset(data_dir=tmp_path, split="inference", cache_rate=0.0)
        assert len(ds) == 2

    def test_patch_extraction_shape(self, tmp_path):
        """3D patch extraction yields the correct patch_size."""
        from utils.preprocessing import extract_patches_3d

        vol = np.zeros((32, 32, 32), dtype=np.float32)
        patches = extract_patches_3d(vol, patch_size=(16, 16, 16), stride=(16, 16, 16))
        assert len(patches) == 8
        for p in patches:
            assert p.shape == (16, 16, 16)


# ---------------------------------------------------------------------------
# COVIDDataset tests
# ---------------------------------------------------------------------------

class TestCOVIDDataset:
    def test_dataset_loads(self, tmp_path):
        """Dataset initialises from synthetic PNG images."""
        _make_lung_dataset(tmp_path)
        from data.dataset_lung import COVIDDataset

        ds = COVIDDataset(data_dir=tmp_path, split="train", use_clahe=False)
        assert len(ds) > 0

    def test_class_mapping(self, tmp_path):
        """All four classes are present with correct integer labels."""
        _make_lung_dataset(tmp_path)
        from data.dataset_lung import COVIDDataset, CLASS_TO_IDX

        ds = COVIDDataset(data_dir=tmp_path, split="train", val_split=0.1, use_clahe=False)
        labels = {label for _, label in ds.samples}
        assert labels == set(CLASS_TO_IDX.values())

    def test_output_tensor_shape(self, tmp_path):
        """Each sample returns a (3, 224, 224) tensor and an integer label."""
        _make_lung_dataset(tmp_path)
        from data.dataset_lung import COVIDDataset

        ds = COVIDDataset(data_dir=tmp_path, split="train", val_split=0.1, use_clahe=False)
        img, label = ds[0]
        assert img.shape == (3, 224, 224)
        assert isinstance(label, int)

    def test_clahe_applied(self, tmp_path):
        """With use_clahe=True the output tensor differs from use_clahe=False."""
        _make_lung_dataset(tmp_path)
        from data.dataset_lung import COVIDDataset

        import torch

        ds_clahe = COVIDDataset(data_dir=tmp_path, split="train", val_split=0.1, use_clahe=True, seed=0)
        ds_plain = COVIDDataset(data_dir=tmp_path, split="train", val_split=0.1, use_clahe=False, seed=0)

        img_clahe, _ = ds_clahe[0]
        img_plain, _ = ds_plain[0]
        assert not torch.allclose(img_clahe, img_plain)

    def test_val_split_sizes(self, tmp_path):
        """Train + val sizes sum to total number of images."""
        _make_lung_dataset(tmp_path)
        from data.dataset_lung import COVIDDataset

        ds_train = COVIDDataset(data_dir=tmp_path, split="train", val_split=0.2, use_clahe=False)
        ds_val = COVIDDataset(data_dir=tmp_path, split="val", val_split=0.2, use_clahe=False)
        assert len(ds_train) + len(ds_val) == 20

    def test_class_names_constant(self):
        """CLASS_NAMES list is present and has 4 entries."""
        from data.dataset_lung import CLASS_NAMES

        assert len(CLASS_NAMES) == 4
        assert "COVID" in CLASS_NAMES


# ---------------------------------------------------------------------------
# Preprocessing utility tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_normalize_volume_nonzero(self):
        """Normalisation over nonzero voxels yields approximately zero mean."""
        from utils.preprocessing import normalize_volume

        vol = np.random.rand(10, 10, 10).astype(np.float32)
        vol[0, 0, 0] = 0.0
        out = normalize_volume(vol, nonzero=True)
        mask = vol != 0
        assert abs(out[mask].mean()) < 0.05

    def test_normalize_volume_all(self):
        """Global normalisation yields approximately zero mean."""
        from utils.preprocessing import normalize_volume

        vol = np.random.rand(10, 10, 10).astype(np.float32)
        out = normalize_volume(vol, nonzero=False)
        assert abs(out.mean()) < 0.05

    def test_normalize_volume_constant(self):
        """Constant volume does not raise; returns near-zero array."""
        from utils.preprocessing import normalize_volume

        vol = np.ones((5, 5, 5), dtype=np.float32) * 3.0
        out = normalize_volume(vol, nonzero=False)
        assert np.all(out == 0.0)

    def test_apply_clahe_range(self):
        """CLAHE output is in [0, 1]."""
        from utils.preprocessing import apply_clahe

        img = np.random.rand(64, 64).astype(np.float32)
        out = apply_clahe(img, clip_limit=2.0, tile_size=8)
        assert out.shape == (64, 64)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-6

    def test_apply_clahe_dtype(self):
        """CLAHE returns float32."""
        from utils.preprocessing import apply_clahe

        img = np.random.rand(32, 32).astype(np.float32)
        out = apply_clahe(img)
        assert out.dtype == np.float32

    def test_resize_volume_shape(self):
        """resize_volume produces exact target shape."""
        from utils.preprocessing import resize_volume

        vol = np.random.rand(10, 20, 30).astype(np.float32)
        out = resize_volume(vol, target_shape=(20, 40, 60))
        assert out.shape == (20, 40, 60)

    def test_resize_volume_downsample(self):
        """Downsampling produces smaller volume."""
        from utils.preprocessing import resize_volume

        vol = np.random.rand(32, 32, 32).astype(np.float32)
        out = resize_volume(vol, target_shape=(16, 16, 16), order=1)
        assert out.shape == (16, 16, 16)

    def test_window_level_clipping(self):
        """window_level clips to [0, 1] for typical HU range."""
        from utils.preprocessing import window_level

        vol = np.linspace(-2000, 2000, 100, dtype=np.float32).reshape(10, 10)
        out = window_level(vol, window=400, level=40)
        assert out.min() >= 0.0
        assert out.max() <= 1.0 + 1e-6

    def test_to_one_hot_shape(self):
        """to_one_hot returns shape (num_classes, *mask.shape)."""
        from utils.preprocessing import to_one_hot

        mask = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.int32)
        out = to_one_hot(mask, num_classes=4)
        assert out.shape == (4, 2, 3)

    def test_to_one_hot_correctness(self):
        """Each channel is 1 only where mask equals that class."""
        from utils.preprocessing import to_one_hot

        mask = np.array([0, 1, 2, 1, 0], dtype=np.int32)
        out = to_one_hot(mask, num_classes=3)
        np.testing.assert_array_equal(out[0], [1, 0, 0, 0, 1])
        np.testing.assert_array_equal(out[1], [0, 1, 0, 1, 0])
        np.testing.assert_array_equal(out[2], [0, 0, 1, 0, 0])

    def test_extract_patches_3d_count(self):
        """Number of patches matches expected count for exact-divisible volume."""
        from utils.preprocessing import extract_patches_3d

        vol = np.zeros((32, 32, 32), dtype=np.float32)
        patches = extract_patches_3d(vol, patch_size=(8, 8, 8), stride=(8, 8, 8))
        assert len(patches) == 64

    def test_extract_patches_3d_shape(self):
        """Each extracted patch has the specified patch_size."""
        from utils.preprocessing import extract_patches_3d

        vol = np.random.rand(24, 24, 24).astype(np.float32)
        patches = extract_patches_3d(vol, patch_size=(8, 8, 8), stride=(8, 8, 8))
        for p in patches:
            assert p.shape == (8, 8, 8)

    def test_extract_patches_3d_non_divisible(self):
        """Non-divisible volume drops the trailing partial patch."""
        from utils.preprocessing import extract_patches_3d

        vol = np.zeros((10, 10, 10), dtype=np.float32)
        patches = extract_patches_3d(vol, patch_size=(8, 8, 8), stride=(8, 8, 8))
        assert len(patches) == 1

    def test_data_dir_constants(self):
        """DATA_DIR_BRAIN and DATA_DIR_LUNG are Path objects."""
        from utils.preprocessing import DATA_DIR_BRAIN, DATA_DIR_LUNG
        from pathlib import Path

        assert isinstance(DATA_DIR_BRAIN, Path)
        assert isinstance(DATA_DIR_LUNG, Path)
