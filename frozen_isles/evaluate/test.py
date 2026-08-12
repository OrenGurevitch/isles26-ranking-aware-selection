"""Contracts for cohort scoring and model comparison, including the misalignment that must never pass."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from frozen_isles.evaluate import (
    as_subject_list,
    metric_arrays,
    rank,
    score_cohort,
    summarise,
    worst_subjects,
)
from frozen_isles.metrics import SubjectScore


def _write(path: Path, data: np.ndarray, zooms: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, np.diag([*zooms, 1.0])), path)
    return path


def _lesion(shape: tuple[int, int, int] = (12, 12, 12), *, offset: int = 0) -> np.ndarray:
    volume = np.zeros(shape, np.uint8)
    volume[3 + offset:7 + offset, 3:7, 3:7] = 1
    return volume


def _score(dice: float, f1: float = 0.5, avd: float = 1.0, count: int = 0, auc: float = 0.5):
    return SubjectScore(dice=dice, lesion_f1=f1, abs_volume_difference_ml=avd,
                        abs_lesion_count_difference=count, pr_auc=auc)


def test_a_perfect_prediction_scores_perfectly_across_the_cohort(tmp_path: Path) -> None:
    references, predictions = {}, {}
    for i in range(3):
        subject = f"sub-{i:03d}"
        volume = _lesion(offset=i)
        references[subject] = _write(tmp_path / "ref" / f"{subject}_mask.nii.gz", volume)
        predictions[subject] = _write(tmp_path / "pred" / f"{subject}_mask.nii.gz", volume)

    scores = score_cohort(references=references, predictions=predictions)
    assert set(scores) == set(references)
    for score in scores.values():
        assert score.dice == pytest.approx(1.0)
        assert score.abs_volume_difference_ml == pytest.approx(0.0)
        assert score.abs_lesion_count_difference == 0


def test_voxel_volume_comes_from_each_subjects_own_header(tmp_path: Path) -> None:
    """Native space: spacing differs per subject, and r032's 3 mm stands by organizer ruling."""
    reference = _write(tmp_path / "r" / "sub-001_mask.nii.gz", _lesion(), zooms=(0.9375, 3.0, 0.9375))
    prediction = _write(tmp_path / "p" / "sub-001_mask.nii.gz", np.zeros((12, 12, 12), np.uint8),
                        zooms=(0.9375, 3.0, 0.9375))
    scores = score_cohort(references={"sub-001": reference}, predictions={"sub-001": prediction})
    expected_ml = 64 * (0.9375 * 3.0 * 0.9375) / 1000.0        # the 4x4x4 lesion, entirely missed
    assert scores["sub-001"].abs_volume_difference_ml == pytest.approx(expected_ml)


def test_a_subject_without_a_prediction_is_refused(tmp_path: Path) -> None:
    reference = _write(tmp_path / "r" / "sub-001_mask.nii.gz", _lesion())
    with pytest.raises(AssertionError, match="no prediction for"):
        score_cohort(
            references={"sub-001": reference, "sub-002": reference},
            predictions={"sub-001": reference},
        )


def test_models_scoring_different_subjects_are_refused_not_intersected() -> None:
    """The alignment invariant: a per-case rank is meaningless across different cases."""
    a = {"sub-001": _score(0.8), "sub-002": _score(0.7)}
    b = {"sub-001": _score(0.6), "sub-003": _score(0.9)}
    with pytest.raises(AssertionError, match="score different subjects"):
        metric_arrays({"model_a": a, "model_b": b})


def test_metric_arrays_align_models_on_the_same_subject_order() -> None:
    a = {"sub-002": _score(0.2), "sub-001": _score(0.9)}       # deliberately unsorted
    b = {"sub-001": _score(0.5), "sub-002": _score(0.5)}
    models, subjects, arrays = metric_arrays({"model_a": a, "model_b": b})

    assert models == ["model_a", "model_b"]
    assert subjects == ["sub-001", "sub-002"]
    assert arrays["dice"].shape == (2, 2)
    assert arrays["dice"][0].tolist() == [0.9, 0.2]            # model_a, in subject order


def test_ranking_a_cohort_follows_the_challenge_scheme_not_the_mean() -> None:
    """Same disagreement pinned in ranking/test.py, reached through the cohort API."""
    flashy = {f"sub-{i:03d}": _score(0.9 if i < 4 else 0.0) for i in range(5)}
    steady = {f"sub-{i:03d}": _score(0.75) for i in range(5)}
    ranks = rank({"flashy": flashy, "steady": steady})

    assert np.mean([s.dice for s in flashy.values()]) < np.mean([s.dice for s in steady.values()])
    assert ranks["flashy"] < ranks["steady"], "the challenge still ranks flashy first"


def test_worst_subjects_names_where_the_rank_is_lost() -> None:
    a = {"sub-001": _score(0.9), "sub-002": _score(0.9), "sub-003": _score(0.01)}
    b = {"sub-001": _score(0.5), "sub-002": _score(0.5), "sub-003": _score(0.5)}
    assert worst_subjects({"model_a": a, "model_b": b}, model="model_a", k=1) == ["sub-003"]


def test_summarise_reports_medians() -> None:
    scores = {"a": _score(0.1), "b": _score(0.5), "c": _score(0.9)}
    assert summarise(scores)["dice"] == pytest.approx(0.5)


def test_a_single_model_cannot_be_ranked() -> None:
    with pytest.raises(AssertionError, match="at least two models"):
        metric_arrays({"only": {"sub-001": _score(0.5)}})


def test_subject_ids_are_parsed_from_bids_filenames() -> None:
    paths = [Path("sub-r032s013_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz"),
             Path("sub-r001s005_ses-1_space-orig_desc-brain_T1w.nii.gz")]
    mapping = as_subject_list(paths, suffix="mask")
    assert sorted(mapping) == ["sub-r001s005", "sub-r032s013"]


def test_two_files_for_one_subject_are_refused() -> None:
    paths = [Path("sub-001_a_mask.nii.gz"), Path("sub-001_b_mask.nii.gz")]
    with pytest.raises(AssertionError, match="two mask files"):
        as_subject_list(paths, suffix="mask")


def _soft_case(hit: bool) -> tuple[np.ndarray, np.ndarray]:
    """A reference lesion, and a soft map that only exceeds 0.5 when `hit`."""
    reference = np.zeros((10, 10, 10), bool)
    reference[3:6, 3:6, 3:6] = True
    soft = np.full(reference.shape, 0.05, np.float32)
    soft[reference] = 0.9 if hit else 0.3          # confident, or under the default cutoff
    return reference, soft


def test_the_threshold_sweep_finds_a_cutoff_that_0_5_would_have_missed() -> None:
    """The first real fold's failure mode: real ranking signal, worthless at 0.5."""
    from frozen_isles.evaluate import sweep_threshold

    reference, soft = _soft_case(hit=False)
    refs = {"sub-a": reference, "sub-b": reference}
    softs = {"sub-a": soft, "sub-b": soft}
    volumes = {"sub-a": 0.001, "sub-b": 0.001}

    best, by_threshold = sweep_threshold(
        references=refs, soft_predictions=softs, voxel_volumes_ml=volumes,
    )
    assert best < 0.5, f"the lesion sits at 0.3; a sweep must go below 0.5, got {best}"
    assert by_threshold[0.5]["sub-a"].dice == 0.0, "premise: 0.5 finds nothing here"
    assert by_threshold[best]["sub-a"].dice > 0.9, "the chosen threshold should recover the lesion"


def test_the_sweep_refuses_mismatched_subject_sets() -> None:
    from frozen_isles.evaluate import sweep_threshold

    reference, soft = _soft_case(hit=True)
    with pytest.raises(AssertionError, match="different subjects"):
        sweep_threshold(
            references={"sub-a": reference}, soft_predictions={"sub-b": soft},
            voxel_volumes_ml={"sub-a": 0.001},
        )


def test_subject_ids_come_from_filenames_with_and_without_a_bids_suffix() -> None:
    """A file named for the subject alone has no `_`, so splitting on it kept the extension.

    Our own masks always carry `_space-orig_...`, so `"sub-x.nii.gz".split("_")[0]` returning
    `"sub-x.nii.gz"` stayed invisible until nnU-Net's predictions — plain `sub-XXXX.nii.gz` — were
    scored against them and matched zero of 291 references.
    """
    from pathlib import Path

    from frozen_isles.evaluate import as_subject_list

    bids = Path("sub-r001s002_space-orig_label-lesion_desc-T1lesion_mask.nii.gz")
    bare = Path("sub-r001s002.nii.gz")
    assert list(as_subject_list([bids], suffix="reference")) == ["sub-r001s002"]
    assert list(as_subject_list([bare], suffix="prediction")) == ["sub-r001s002"]
    assert as_subject_list([bids], suffix="a").keys() == as_subject_list([bare], suffix="b").keys()


def test_score_cohort_can_score_the_shipped_post_processed_configuration(tmp_path):
    """The container ships min_voxels=25; scoring without it describes a different system.

    Found 2026-08-05: the project's headline row was the min_voxels=0 entry while Methods described a
    25-voxel filter. This pins that the option exists and that it changes the mask-driven metrics
    without touching PR-AUC, which is computed from the untouched probability map.
    """
    import nibabel as nib
    import numpy as np

    from frozen_isles.evaluate import score_cohort

    reference = np.zeros((32, 32, 32), dtype=np.uint8)
    reference[8:20, 8:20, 8:20] = 1                 # one real lesion
    predicted = reference.copy()
    predicted[28:30, 28:30, 28:30] = 1              # an 8-voxel invented cluster, under any filter

    affine = np.eye(4)
    paths = {}
    for name, volume in (("ref", reference), ("pred", predicted)):
        directory = tmp_path / name
        directory.mkdir()
        path = directory / "sub-x0001_mask.nii.gz"
        nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), path)
        paths[name] = {"sub-x0001": path}

    raw = score_cohort(references=paths["ref"], predictions=paths["pred"])
    filtered = score_cohort(references=paths["ref"], predictions=paths["pred"], min_voxels=25)

    assert raw["sub-x0001"].abs_lesion_count_difference == 1, "the invented cluster should count"
    assert filtered["sub-x0001"].abs_lesion_count_difference == 0, "the filter should remove it"
