import csv
from pathlib import Path

import numpy as np
import pytest
from timing_beyond_size import (
    DegenerateInput,
    _ranks,
    partial_spearman,
    reference_volumes_ml,
    spearman,
)


def test_tied_values_get_averaged_ranks() -> None:
    # Dice has a spike at exactly 0 -- 29 of 291 subjects on the 250-epoch arm. Without averaging,
    # tied values take ranks in whatever order the sort happened to produce, and the correlation then
    # depends on the order subjects were read out of the scores JSON.
    assert _ranks(np.array([1.0, 2.0, 2.0, 3.0])).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_a_constant_variable_raises_rather_than_reporting_no_association() -> None:
    # Returning 0.0 here would be a legitimate-looking "no association" for an analysis that is
    # actually undefined, which is the silent-fallback failure this repo forbids.
    with pytest.raises(DegenerateInput):
        spearman(np.ones(10), np.arange(10, dtype=float))


def test_a_continuous_control_is_ranked_whether_passed_as_1d_or_2d() -> None:
    # The site-controlled bootstrap passes a 2-D control block and the volume-only one passes a
    # single column; if those two paths disagreed, the two published intervals would not be
    # comparable -- which is exactly the defect this signature exists to prevent.
    rng = np.random.default_rng(11)
    z = rng.normal(size=200)
    x = 0.4 * z + rng.normal(size=200)
    y = 0.6 * z + 0.3 * x + rng.normal(size=200)
    assert partial_spearman(x, y, z) == pytest.approx(partial_spearman(x, y, z[:, None]))


def test_indicator_columns_are_not_rank_transformed() -> None:
    rng = np.random.default_rng(12)
    z = rng.normal(size=150)
    indicator = (rng.random(150) < 0.5).astype(float)
    x, y = rng.normal(size=150), rng.normal(size=150)
    controls = np.column_stack([z, indicator])
    mask = np.array([True, False])
    # Rank-transforming a 0/1 column is a monotone relabelling, so the result must not move.
    assert partial_spearman(x, y, controls, rank_controls=mask) == pytest.approx(
        partial_spearman(x, y, controls, rank_controls=np.array([True, True]))
    )


def test_correlation_is_invariant_to_input_order() -> None:
    rng = np.random.default_rng(0)
    metric = np.concatenate([np.zeros(30), rng.uniform(0.1, 0.9, 70)])  # a realistic zero spike
    days = rng.uniform(0, 3000, 100)
    shuffle = rng.permutation(100)
    assert spearman(days, metric) == pytest.approx(spearman(days[shuffle], metric[shuffle]))


def test_partial_removes_a_pure_confound() -> None:
    # y depends on z alone; x depends on z alone. The raw association is spurious and the partial
    # association must collapse -- the whole point of controlling for lesion volume.
    rng = np.random.default_rng(1)
    z = rng.normal(size=400)
    x = z + 0.05 * rng.normal(size=400)
    y = z + 0.05 * rng.normal(size=400)
    assert spearman(x, y) > 0.9
    assert abs(partial_spearman(x, y, z)) < 0.3


def test_partial_keeps_an_association_that_is_not_the_confound() -> None:
    rng = np.random.default_rng(2)
    z = rng.normal(size=400)
    x = rng.normal(size=400)
    y = z + 0.8 * x + 0.1 * rng.normal(size=400)
    assert partial_spearman(x, y, z) > 0.5


def test_reference_volume_uses_the_per_subject_zooms(tmp_path: Path) -> None:
    # Not every subject is 1 mm isotropic, so a voxel count is not a volume. A fixed 1 mm assumption
    # would misreport the control this whole analysis rests on.
    inventory = tmp_path / "inventory.csv"
    with inventory.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "center", "lesion_voxels", "shape", "zooms",
                         "orientation", "mask_dtype"])
        writer.writerow(["sub-a", "R001", "1000", "1x1x1", "1.0x1.0x1.0", "RAS", "float32"])
        writer.writerow(["sub-b", "R001", "1000", "1x1x1", "2.0x1.0x0.5", "RAS", "float32"])
    volumes = reference_volumes_ml(inventory)
    assert volumes["sub-a"] == pytest.approx(1.0)
    assert volumes["sub-b"] == pytest.approx(1.0)  # same voxel count, same volume, different spacing


def test_zooms_that_are_not_three_dimensional_are_rejected(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.csv"
    with inventory.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "center", "lesion_voxels", "shape", "zooms",
                         "orientation", "mask_dtype"])
        writer.writerow(["sub-a", "R001", "1000", "1x1", "1.0x1.0", "RAS", "float32"])
    with pytest.raises(AssertionError, match="expected 3 zooms"):
        reference_volumes_ml(inventory)
