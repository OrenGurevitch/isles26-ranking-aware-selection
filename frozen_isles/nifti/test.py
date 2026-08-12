"""Contracts for reading native-space volumes, anchored on the real corrected r032 files where present.

The synthetic cases run everywhere. The real-file cases skip when the ATLAS corrections are not staged,
because the defect they guard against — a float mask whose foreground is 1.00000006 — was found in the
released data and would be unconvincing as a purely invented example.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from frozen_isles.nifti import load_mask, orientation, thick_axis, voxel_volume_ml

CORRECTIONS = Path.home() / "data" / "isles26" / "corrections" / "isles26_train" / "raw"
_FLOAT_MASK = CORRECTIONS / "sub-r032s013_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz"
needs_corrections = pytest.mark.skipif(
    not _FLOAT_MASK.exists(), reason=f"ATLAS corrections not staged at {CORRECTIONS}"
)


def _write(tmp_path: Path, data: np.ndarray, zooms: tuple[float, float, float]) -> Path:
    affine = np.diag([*zooms, 1.0])
    path = tmp_path / "volume.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), path)
    return path


def test_a_float_mask_whose_foreground_is_not_exactly_one_still_loads(tmp_path: Path) -> None:
    """THE bug this module exists for: `mask == 1` would return an empty mask, silently."""
    data = np.zeros((6, 6, 6), np.float64)
    data[1:3, 1:3, 1:3] = 1.00000006
    path = _write(tmp_path, data, (1.0, 1.0, 1.0))

    assert (data == 1).sum() == 0, "the premise: equality against 1 finds nothing"
    assert load_mask(path).sum() == 8


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_every_dtype_the_release_uses_loads_to_the_same_mask(tmp_path: Path, dtype: type) -> None:
    data = np.zeros((5, 5, 5), dtype)
    data[2, 2, 2] = 1
    assert load_mask(_write(tmp_path, data, (1.0, 1.0, 1.0))).sum() == 1


def test_a_multi_label_map_is_refused_rather_than_flattened(tmp_path: Path) -> None:
    data = np.zeros((5, 5, 5), np.uint8)
    data[1, 1, 1] = 1
    data[3, 3, 3] = 4                                          # a second label, not lesion foreground
    with pytest.raises(AssertionError, match="not a binary mask"):
        load_mask(_write(tmp_path, data, (1.0, 1.0, 1.0)))


@pytest.mark.parametrize(
    ("zooms", "expected_ml"),
    [((0.9375, 3.0, 0.9375), 0.9375 * 3.0 * 0.9375 / 1000.0), ((1.0, 1.0, 1.0), 0.001)],
)
def test_voxel_volume_is_in_ml_and_reads_the_subjects_own_spacing(
    tmp_path: Path, zooms: tuple[float, float, float], expected_ml: float
) -> None:
    path = _write(tmp_path, np.zeros((4, 4, 4), np.uint8), zooms)
    assert voxel_volume_ml(path) == pytest.approx(expected_ml)


@pytest.mark.parametrize(
    ("zooms", "expected_axis"), [((0.9375, 3.0, 0.9375), 1), ((5.0, 1.0, 1.0), 0), ((1.0, 1.0, 4.0), 2)]
)
def test_thick_axis_finds_the_coarsely_sampled_direction(
    tmp_path: Path, zooms: tuple[float, float, float], expected_axis: int
) -> None:
    assert thick_axis(_write(tmp_path, np.zeros((4, 4, 4), np.uint8), zooms)) == expected_axis


@needs_corrections
def test_the_real_corrected_masks_load_and_are_non_empty() -> None:
    """All three released r032 masks, across three different dtypes, in one pass."""
    masks = sorted(CORRECTIONS.glob("*label-lesion*.nii.gz"))
    assert len(masks) == 3, f"expected 3 corrected masks, found {[m.name for m in masks]}"
    for path in masks:
        assert load_mask(path).sum() > 0, f"{path.name} loaded empty"


@needs_corrections
def test_the_real_files_are_anisotropic_and_agree_on_geometry() -> None:
    """r032 is the site behind the forum's "Possible R032 A-P spacing issue" thread."""
    for path in sorted(CORRECTIONS.glob("*.nii.gz")):
        assert orientation(path) == "LAS"
        assert thick_axis(path) == 1, f"{path.name}: expected the A-P axis to be the thick one"
        assert voxel_volume_ml(path) == pytest.approx(0.9375 * 3.0 * 0.9375 / 1000.0)


def test_the_binary_check_accepts_and_REJECTS_exactly_what_the_sorted_form_did() -> None:
    """`load_mask`'s check stopped using `np.unique` for speed — pin that it changed nothing.

    `np.unique` sorts the whole volume (~8.7M voxels) and `predict_from_store` calls `load_mask` on
    every training step, where it measured 357 ms against a 1.36 s step. The replacement is an O(n)
    per-voxel predicate. It is only worth having if it is EXACTLY as strict, so this compares it
    against the sorted form it replaced, including the cases the module docstring exists for.
    """
    threshold, tolerance = 0.5, 1e-3

    def sorted_form(values):
        unique = np.unique(values)
        foreground = unique[unique > threshold]
        return bool(np.all(np.abs(foreground - 1.0) < tolerance))

    def per_voxel_form(values):
        foreground = values > threshold
        return bool(np.all(np.abs(values[foreground] - 1.0) < tolerance))

    cases = [
        np.array([0, 1, 0, 1], float),                  # clean binary
        np.array([0.0, 1.00000006, 0.0]),               # sub-r032s013's float64 mask
        np.array([0, 1, 2, 3], float),                  # multi-label — must be refused
        np.array([0, 255, 0], float),                   # 0-255 map — must be refused
        np.array([0.0, 0.6, 0.9, 1.0]),                 # probabilistic — must be refused
        np.zeros(10), np.zeros(0),                      # all background, and empty
        np.array([0.0, 1.0009]), np.array([0.0, 1.0011]),   # either side of the tolerance
        np.array([0, 1], np.uint8), np.array([0, 1], np.uint16),
    ]
    rng = np.random.default_rng(0)
    cases += [rng.choice([0.0, 1.0, 1.00000006, 2.0, 0.7], size=97) for _ in range(50)]

    for values in cases:
        assert sorted_form(values) == per_voxel_form(values), f"divergence on {np.unique(values)[:8]}"
