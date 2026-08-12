from pathlib import Path

import pytest
from pool_fold_validations import pool, subject_of


def _fold(model: Path, fold: int, subjects: list[str]) -> None:
    validation = model / f"fold_{fold}" / "validation"
    validation.mkdir(parents=True)
    for subject in subjects:
        for suffix in (".nii.gz", ".npz", ".pkl"):
            (validation / f"{subject}{suffix}").write_bytes(b"")


def test_disjoint_folds_pool_to_the_whole_cohort(tmp_path: Path) -> None:
    model = tmp_path / "model"
    _fold(model, 0, ["sub-a", "sub-b"])
    _fold(model, 1, ["sub-c"])
    _fold(model, 2, ["sub-d"])
    _fold(model, 3, ["sub-e"])
    _fold(model, 4, ["sub-f"])

    counts = pool(model=model, out=tmp_path / "pooled")

    assert counts["total_subjects"] == 6
    assert counts["files_linked"] == 18                      # 6 subjects x 3 suffixes
    assert {p.name for p in (tmp_path / "pooled").iterdir()} == {
        f"sub-{s}{x}" for s in "abcdef" for x in (".nii.gz", ".npz", ".pkl")
    }


def test_an_overlapping_split_is_refused_rather_than_double_counted(tmp_path: Path) -> None:
    """The reason this script exists. Overlapping folds would inflate the cohort and go unnoticed."""
    model = tmp_path / "model"
    _fold(model, 0, ["sub-a", "sub-shared"])
    _fold(model, 1, ["sub-shared"])
    for fold in (2, 3, 4):
        _fold(model, fold, [f"sub-{fold}"])

    with pytest.raises(AssertionError, match="more than one fold"):
        pool(model=model, out=tmp_path / "pooled")


def test_a_missing_fold_is_refused(tmp_path: Path) -> None:
    model = tmp_path / "model"
    for fold in (0, 1, 2, 3):
        _fold(model, fold, [f"sub-{fold}"])

    with pytest.raises(AssertionError, match="no validation directory for fold 4"):
        pool(model=model, out=tmp_path / "pooled")


def test_rerunning_replaces_stale_links_instead_of_failing(tmp_path: Path) -> None:
    model = tmp_path / "model"
    for fold in range(5):
        _fold(model, fold, [f"sub-{fold}"])
    out = tmp_path / "pooled"

    pool(model=model, out=out)
    counts = pool(model=model, out=out)

    assert counts["total_subjects"] == 5


def test_subject_of_strips_the_double_extension() -> None:
    assert subject_of(Path("sub-r001s002.nii.gz")) == "sub-r001s002"
    assert subject_of(Path("sub-r001s002.npz")) == "sub-r001s002"
