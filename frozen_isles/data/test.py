"""Contracts for reading the ATLAS tree, against a synthetic copy of its real layout.

The layout and the CSV schema here are copied from the extracted release, including the details that
would otherwise be guessed wrong: DAYS_POST_STROKE is a float, and centre comes from the directory
rather than the SITE column.
"""

from pathlib import Path

import pytest

from frozen_isles.data import (
    AtlasSubject,
    discover,
    load_lesion_voxels,
    read_metadata,
    split_subjects,
)

_HEADER = "SESSION_ID,ATLAS2_DATASET,DAYS_POST_STROKE,CHRONICITY,SITE"


def _subject(
    root: Path, center: str, subject_id: str, *, days: str = "182.5", chronicity: str = "1",
    dataset: str = "Training", files: tuple[str, ...] = ("t1w", "mask", "metadata"),
) -> Path:
    anat = root / center / subject_id / "ses-1" / "anat"
    anat.mkdir(parents=True)
    if "t1w" in files:
        (anat / f"{subject_id}_ses-1_space-orig_desc-brain_T1w.nii.gz").write_bytes(b"")
    if "mask" in files:
        (anat / f"{subject_id}_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz").write_bytes(b"")
    if "metadata" in files:
        (anat / f"{subject_id}_ses-1_metadata.csv").write_text(
            f"{_HEADER}\n{subject_id}_ses-1,{dataset},{days},{chronicity},{center}\n"
        )
    return anat


def test_discover_finds_every_subject_with_its_centre_from_the_path(tmp_path: Path) -> None:
    _subject(tmp_path, "R017", "sub-r017s116")
    _subject(tmp_path, "SOOP", "sub-soop001")
    subjects = discover(tmp_path)

    assert [s.subject_id for s in subjects] == ["sub-r017s116", "sub-soop001"]
    assert {s.center for s in subjects} == {"R017", "SOOP"}
    assert subjects[0].t1w.name.endswith("_space-orig_desc-brain_T1w.nii.gz")
    assert subjects[0].mask.name.endswith("_label-lesion_desc-T1lesion_mask.nii.gz")


def test_days_post_stroke_is_read_as_a_float(tmp_path: Path) -> None:
    """The released files carry 182.5 and 1825.0 — parsing as int would raise or truncate."""
    _subject(tmp_path, "R017", "sub-a", days="182.5")
    assert discover(tmp_path)[0].days_post_stroke == pytest.approx(182.5)


@pytest.mark.parametrize("missing", ["", "NA", "NaN", "  "])
def test_unretrievable_timing_becomes_none_not_zero(tmp_path: Path, missing: str) -> None:
    """CHRONICITY is 1-or-NaN and never 0, so "unknown" must not collapse into "acute"."""
    _subject(tmp_path, "R017", "sub-a", days=missing, chronicity=missing)
    subject = discover(tmp_path)[0]
    assert subject.days_post_stroke is None
    assert subject.chronicity is None


def test_the_atlas2_split_is_carried_for_the_contamination_check(tmp_path: Path) -> None:
    """316 of the hidden test set is the ISLES'22 test subset — provenance has to be checkable."""
    _subject(tmp_path, "R017", "sub-a", dataset="Testing")
    assert discover(tmp_path)[0].atlas2_dataset == "Testing"


@pytest.mark.parametrize("absent", ["t1w", "mask", "metadata"])
def test_a_subject_missing_any_file_is_refused_not_skipped(tmp_path: Path, absent: str) -> None:
    """A silently shorter cohort is not noticed until a number is wrong."""
    present = tuple(f for f in ("t1w", "mask", "metadata") if f != absent)
    _subject(tmp_path, "R017", "sub-a", files=present)
    with pytest.raises(AssertionError, match=f"expected one {absent} file"):
        discover(tmp_path)


def test_an_empty_tree_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="no .*anat directories"):
        discover(tmp_path)


def test_metadata_with_more_than_one_row_is_refused(tmp_path: Path) -> None:
    anat = _subject(tmp_path, "R017", "sub-a")
    csv_path = anat / "sub-a_ses-1_metadata.csv"
    csv_path.write_text(f"{_HEADER}\na,Training,1.0,1,R017\nb,Training,2.0,1,R017\n")
    with pytest.raises(AssertionError, match="expected at most one metadata row"):
        read_metadata(csv_path)


def test_split_subjects_joins_the_inventory_by_id(tmp_path: Path) -> None:
    _subject(tmp_path, "R017", "sub-a")
    _subject(tmp_path, "R040", "sub-b")
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("subject_id,lesion_voxels\nsub-a,1200\nsub-b,45\n")

    subjects = split_subjects(discover(tmp_path), inventory_csv=inventory)
    assert {s.subject_id: s.lesion_voxels for s in subjects} == {"sub-a": 1200, "sub-b": 45}
    assert {s.center for s in subjects} == {"R017", "R040"}


def test_a_subject_absent_from_the_inventory_is_refused(tmp_path: Path) -> None:
    _subject(tmp_path, "R017", "sub-a")
    _subject(tmp_path, "R040", "sub-b")
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("subject_id,lesion_voxels\nsub-a,1200\n")
    with pytest.raises(AssertionError, match="no lesion count for"):
        split_subjects(discover(tmp_path), inventory_csv=inventory)


def test_load_lesion_voxels_refuses_an_empty_inventory(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.csv"
    inventory.write_text("subject_id,lesion_voxels\n")
    with pytest.raises(AssertionError, match="has no rows"):
        load_lesion_voxels(inventory)


def test_to_split_subject_carries_the_float_timing(tmp_path: Path) -> None:
    subject = AtlasSubject(
        subject_id="sub-a", center="R017", t1w=tmp_path / "t", mask=tmp_path / "m",
        days_post_stroke=182.5, chronicity=1, atlas2_dataset="Training",
    )
    assert subject.to_split_subject(lesion_voxels=99).days_post_stroke == pytest.approx(182.5)


def test_a_header_only_metadata_file_is_tolerated_not_fatal(tmp_path: Path) -> None:
    """Two of the 1453 released files have no data row; refusing the cohort over them is wrong."""
    anat = _subject(tmp_path, "SOOP", "sub-soop0910")
    (anat / "sub-soop0910_ses-1_metadata.csv").write_text(f"{_HEADER}\n")
    subject = discover(tmp_path)[0]
    assert subject.days_post_stroke is None
    assert subject.chronicity is None
    assert subject.atlas2_dataset is None


def test_columns_are_read_by_name_because_centres_disagree_on_order(tmp_path: Path) -> None:
    """SOOP writes ATLAS2_DATASET first, the R-centres write SESSION_ID first."""
    anat = _subject(tmp_path, "SOOP", "sub-soop0961")
    (anat / "sub-soop0961_ses-1_metadata.csv").write_text(
        "ATLAS2_DATASET,SESSION_ID,DAYS_POST_STROKE,CHRONICITY,SITE\n"
        "ATLAS3,sub-soop0961_ses-1,,1.0,SOOP\n"
    )
    subject = discover(tmp_path)[0]
    assert subject.atlas2_dataset == "ATLAS3", "column order must not swap id and split"
    assert subject.chronicity == 1
    assert subject.days_post_stroke is None
