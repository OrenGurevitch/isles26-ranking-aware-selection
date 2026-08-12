"""Contracts for the fold construction, including the cohort shapes that break naive stratification."""

import numpy as np
import pytest

from frozen_isles.splits import (
    Subject,
    chronicity_band,
    size_band,
    size_band_edges,
    stratified_folds,
)


def _cohort(n: int = 200, n_centers: int = 8, seed: int = 0) -> list[Subject]:
    rng = np.random.default_rng(seed)
    return [
        Subject(
            subject_id=f"sub-{i:04d}",
            center=f"r{rng.integers(0, n_centers):03d}",
            lesion_voxels=int(rng.lognormal(7.0, 1.5)),
            days_post_stroke=None if rng.random() < 0.3 else int(rng.integers(0, 900)),
        )
        for i in range(n)
    ]


def test_every_subject_lands_in_exactly_one_fold() -> None:
    subjects = _cohort()
    folds = stratified_folds(subjects, n_folds=5)
    flat = [s for fold in folds for s in fold]
    assert sorted(flat) == sorted(s.subject_id for s in subjects)
    assert len(flat) == len(set(flat))


def test_folds_are_deterministic_for_a_seed_and_differ_between_seeds() -> None:
    subjects = _cohort()
    assert stratified_folds(subjects, seed=0) == stratified_folds(subjects, seed=0)
    assert stratified_folds(subjects, seed=0) != stratified_folds(subjects, seed=1)


def test_folds_are_near_equal_in_size() -> None:
    folds = stratified_folds(_cohort(200), n_folds=5)
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= 1, sizes


def test_every_center_is_spread_across_folds_rather_than_concentrated() -> None:
    """The organizers' first stratification axis: site-specific protocols must not define a fold."""
    subjects = _cohort(240, n_centers=6)
    by_id = {s.subject_id: s for s in subjects}
    folds = stratified_folds(subjects, n_folds=4)

    for center in {s.center for s in subjects}:
        per_fold = [sum(by_id[i].center == center for i in fold) for fold in folds]
        # <=1 is what dealing a centre's members into CONSECUTIVE slots guarantees, and it is the
        # contract worth pinning: bucket-wise dealing only managed 3 on this cohort.
        assert max(per_fold) - min(per_fold) <= 1, f"{center} concentrated: {per_fold}"


def test_a_stream_of_single_subject_centers_does_not_pile_into_one_fold() -> None:
    """The failure mode that motivates carrying the deal offset between buckets."""
    subjects = [
        Subject(subject_id=f"sub-{i}", center=f"solo{i}", lesion_voxels=100 + i, days_post_stroke=30)
        for i in range(20)
    ]
    folds = stratified_folds(subjects, n_folds=5)
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= 1, f"singleton centres piled up: {sizes}"


def test_small_lesions_reach_every_fold() -> None:
    """"Benchmarked on challenging small infarcts, not only larger lesions" — their words."""
    subjects = _cohort(200)
    by_id = {s.subject_id: s for s in subjects}
    edges = size_band_edges(subjects, n_bands=3)
    folds = stratified_folds(subjects, n_folds=5, n_size_bands=3)

    for fold in folds:
        smallest = [i for i in fold if size_band(by_id[i].lesion_voxels, edges=edges) == 0]
        assert smallest, "a fold contains no small lesions at all"


@pytest.mark.parametrize(
    ("days", "expected"),
    [(None, "unknown"), (0, "acute"), (13, "acute"), (14, "subacute"), (179, "subacute"),
     (180, "chronic"), (900, "chronic")],
)
def test_chronicity_bands_use_the_organizers_180_day_boundary(days: int | None, expected: str) -> None:
    assert chronicity_band(days) == expected


def test_unknown_timing_is_its_own_stratum_not_folded_into_acute() -> None:
    """CHRONICITY is 1-or-NaN, never 0 — 'not retrievable' is not 'early'."""
    assert chronicity_band(None) != chronicity_band(0)


def test_size_bands_are_quantiles_of_this_cohort_not_fixed_thresholds() -> None:
    subjects = [Subject(f"s{i}", "r001", lesion_voxels=v) for i, v in enumerate([1, 2, 3, 400, 5000])]
    edges = size_band_edges(subjects, n_bands=3)
    assert size_band(1, edges=edges) == 0
    assert size_band(5000, edges=edges) == len(edges)


def test_a_cohort_of_identical_lesion_sizes_still_produces_folds() -> None:
    """Degenerate quantiles collapse the size bands; the deal must survive it."""
    subjects = [Subject(f"s{i}", "r001", lesion_voxels=500, days_post_stroke=10) for i in range(10)]
    folds = stratified_folds(subjects, n_folds=5)
    assert sorted(s for f in folds for s in f) == sorted(s.subject_id for s in subjects)


def test_too_few_subjects_is_refused() -> None:
    with pytest.raises(AssertionError, match="cannot fill"):
        stratified_folds([Subject("a", "r001", 10)], n_folds=5)


def test_duplicate_subject_ids_are_refused() -> None:
    duplicated = [Subject("same", "r001", 10), Subject("same", "r002", 20), Subject("x", "r003", 30)]
    with pytest.raises(AssertionError, match="duplicate subject_id"):
        stratified_folds(duplicated, n_folds=2)
