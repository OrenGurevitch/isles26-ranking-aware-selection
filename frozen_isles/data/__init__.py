"""Reading the ATLAS v3.0 release: where the files are, and what the metadata actually says.

The layout, confirmed against the extracted archive on 2026-07-27:

    ATLAS3_Training_Raw/<CENTER>/<sub-id>/ses-1/anat/
        <sub-id>_ses-1_space-orig_desc-brain_T1w.nii.gz
        <sub-id>_ses-1_space-orig_label-lesion_desc-T1lesion_mask.nii.gz
        <sub-id>_ses-1_metadata.csv

1453 subjects across **55 centres**, every one carrying all three files. The centre is a directory,
so it comes free from the path rather than needing the CSV.

The CSV's columns are `SESSION_ID, ATLAS2_DATASET, DAYS_POST_STROKE, CHRONICITY, SITE`. Two things the
documentation does not tell you and the files do:

- **`DAYS_POST_STROKE` is a FLOAT** — real values include `182.5` and `1825.0`, not integers.
- **`ATLAS2_DATASET` records each subject's ATLAS v2.0 split.** That matters beyond curiosity: 316 of
  the hidden ISLES'26 test set is the ISLES'22 test subset, so any published model that trained on
  ATLAS v2.0's test split is contaminated for this challenge. Ours is not — DINOv3 and V-JEPA never
  saw stroke MRI — but the field is worth carrying so the claim can be checked rather than asserted.

`discover` is cheap: it stats paths and parses small CSVs, touching no image data. Lesion volumes need
every mask read, which is slow on Lustre, so they come from the inventory `scripts/qc_geometry.py`
writes rather than being recomputed on each call.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from frozen_isles.splits import Subject

_T1W_SUFFIX = "_space-orig_desc-brain_T1w.nii.gz"
_MASK_SUFFIX = "_space-orig_label-lesion_desc-T1lesion_mask.nii.gz"
_METADATA_SUFFIX = "_metadata.csv"


@dataclass(frozen=True)
class AtlasSubject:
    subject_id: str
    center: str
    t1w: Path
    mask: Path
    days_post_stroke: float | None
    chronicity: int | None
    atlas2_dataset: str | None

    def to_split_subject(self, *, lesion_voxels: int) -> Subject:
        return Subject(
            subject_id=self.subject_id,
            center=self.center,
            lesion_voxels=lesion_voxels,
            days_post_stroke=self.days_post_stroke,
        )


def _optional_float(raw: str | None) -> float | None:
    """'' , 'NA', 'NaN' and a missing column all mean "not retrievable", which is its own stratum."""
    if raw is None:
        return None
    text = raw.strip()
    if not text or text.upper() in {"NA", "NAN", "N/A", "NONE"}:
        return None
    return float(text)


def read_metadata(path: Path) -> dict[str, str]:
    """The single metadata row, or {} when the file carries only a header.

    Two of the 1453 files (`sub-soop0910`, `sub-soop1701`) have no data row. That is a legitimate
    "not retrievable" rather than corruption, and refusing the whole cohort over 2 subjects would be
    the wrong failure — every field simply becomes None, which is already a stratum. More than one row
    IS refused, because that would mean the file means something other than what this assumes.

    Read by COLUMN NAME, never by position: the centres disagree on column order — SOOP writes
    `ATLAS2_DATASET` first while the R-centres write `SESSION_ID` first — so positional parsing would
    silently swap the session id and the split assignment for every SOOP subject.
    """
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) <= 1, f"{path.name}: expected at most one metadata row, got {len(rows)}"
    if not rows:
        return {}
    return {key: (value or "") for key, value in rows[0].items() if key}


def discover(root: Path) -> list[AtlasSubject]:
    """Every subject under an extracted ATLAS tree, sorted by id. Reads no image data.

    Refuses a subject missing any of its three files rather than skipping it: a silently shorter
    cohort is the kind of thing that is never noticed until a number is wrong.
    """
    anat_dirs = sorted(p for p in root.glob("*/sub-*/ses-*/anat") if p.is_dir())
    assert anat_dirs, f"no */sub-*/ses-*/anat directories under {root}"

    subjects: list[AtlasSubject] = []
    for anat in anat_dirs:
        subject_id = anat.parents[1].name
        center = anat.parents[2].name
        found = {
            name: sorted(anat.glob(f"*{suffix}"))
            for name, suffix in (
                ("t1w", _T1W_SUFFIX), ("mask", _MASK_SUFFIX), ("metadata", _METADATA_SUFFIX)
            )
        }
        for name, paths in found.items():
            assert len(paths) == 1, f"{subject_id}: expected one {name} file, found {len(paths)}"

        metadata = read_metadata(found["metadata"][0])
        chronicity = _optional_float(metadata.get("CHRONICITY"))
        subjects.append(AtlasSubject(
            subject_id=subject_id,
            center=center,
            t1w=found["t1w"][0],
            mask=found["mask"][0],
            days_post_stroke=_optional_float(metadata.get("DAYS_POST_STROKE")),
            chronicity=None if chronicity is None else int(chronicity),
            atlas2_dataset=metadata.get("ATLAS2_DATASET") or None,
        ))

    ids = [s.subject_id for s in subjects]
    assert len(set(ids)) == len(ids), "duplicate subject ids in the tree"

    # Loud but not fatal: a subject with no metadata row still trains, it just carries no timing.
    without_metadata = [s.subject_id for s in subjects if s.days_post_stroke is None and s.chronicity is None]
    if without_metadata:
        print(f"[data] {len(without_metadata)} subject(s) have no usable timing metadata: "
              f"{', '.join(without_metadata[:8])}", flush=True)
    return subjects


def load_lesion_voxels(inventory_csv: Path) -> dict[str, int]:
    """Per-subject lesion voxel counts from the QC inventory, so masks are read once, not per call."""
    with inventory_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = {row["subject_id"]: int(row["lesion_voxels"]) for row in rows}
    assert counts, f"{inventory_csv} has no rows"
    return counts


def split_subjects(subjects: list[AtlasSubject], *, inventory_csv: Path) -> list[Subject]:
    """`frozen_isles.splits.Subject` records, ready for `stratified_folds`."""
    counts = load_lesion_voxels(inventory_csv)
    missing = sorted({s.subject_id for s in subjects} - set(counts))
    assert not missing, f"no lesion count for {len(missing)} subject(s): {missing[:5]}"
    return [s.to_split_subject(lesion_voxels=counts[s.subject_id]) for s in subjects]
