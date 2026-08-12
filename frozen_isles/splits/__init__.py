"""Cross-validation folds stratified the three ways the organizers asked for.

ISLES'26 ships no validation set, deliberately. Their recommendation, quoted from the design document:
stratify by **centre site** ("to prevent optimistic bias from site-specific imaging protocols"), by
**lesion size** ("to ensure models are benchmarked on challenging small infarcts rather than only
larger lesions"), and **temporally** using `days post stroke` or `chronicity`.

This takes values, not paths. Nothing here knows what the ATLAS directory tree looks like, so it was
written and tested before the data was decrypted, and a reader that produces `Subject` records is the
only thing left to add. It also means the folds can be checked against synthetic cohorts with awkward
shapes — one-subject centres, everything the same size — instead of only against whatever the real
data happens to contain.

Centre is the primary axis. Each centre's subjects are ordered by (size band, chronicity band) and
dealt into consecutive fold slots, which spreads that centre across folds to within one subject —
k consecutive slots necessarily hit k distinct folds — while the ordering makes size and chronicity
land on different folds rather than clustering. The deal position carries between centres so total
fold sizes stay balanced.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Subject:
    subject_id: str
    center: str
    lesion_voxels: int
    # FLOAT, not int: the released metadata carries values like 182.5 and 1825.0. None when
    # DAYS_POST_STROKE was not retrievable — the organizers confirmed CHRONICITY is 1-or-NaN, never 0,
    # where 1 means 180+ days post stroke, so "unknown" is its own stratum rather than a zero.
    days_post_stroke: float | None = None


def size_band(lesion_voxels: int, *, edges: Sequence[int]) -> int:
    """Which lesion-size stratum a subject falls in. `edges` are upper bounds, exclusive."""
    return int(np.searchsorted(np.asarray(edges), lesion_voxels, side="right"))


def size_band_edges(subjects: Sequence[Subject], *, n_bands: int = 3) -> list[int]:
    """Quantile cut points over the cohort's own lesion sizes.

    Data-driven rather than fixed thresholds: "small" is only definable against this cohort, and ATLAS
    spans punctiform embolic lesions to massive infarcts. Quantiles guarantee the bands are populated.
    """
    assert n_bands >= 1, f"n_bands must be positive, got {n_bands}"
    volumes = np.array([s.lesion_voxels for s in subjects], dtype=float)
    quantiles = np.linspace(0, 1, n_bands + 1)[1:-1]
    return [int(v) for v in np.quantile(volumes, quantiles)] if quantiles.size else []


def chronicity_band(days_post_stroke: float | None) -> str:
    """acute / subacute / chronic / unknown — the organizers' 180-day boundary, plus a stratum for NaN."""
    if days_post_stroke is None:
        return "unknown"
    if days_post_stroke < 14:
        return "acute"
    return "subacute" if days_post_stroke < 180 else "chronic"


def stratified_folds(
    subjects: Sequence[Subject], *, n_folds: int = 5, n_size_bands: int = 3, seed: int = 0
) -> list[list[str]]:
    """Subject ids per fold, stratified by (centre × lesion-size band × chronicity band).

    Deterministic given `seed`. Every subject appears exactly once across the folds.
    """
    assert n_folds >= 2, f"need at least 2 folds, got {n_folds}"
    assert len(subjects) >= n_folds, f"{len(subjects)} subjects cannot fill {n_folds} folds"
    ids = [s.subject_id for s in subjects]
    assert len(set(ids)) == len(ids), "duplicate subject_id"

    edges = size_band_edges(subjects, n_bands=n_size_bands)
    rng = np.random.default_rng(seed)

    # Group by CENTRE first, then order each centre's subjects by (size band, chronicity band) and deal
    # them into consecutive fold slots. Dealing a centre's members consecutively is what guarantees the
    # centre itself is spread evenly — at most one subject apart across folds — because k consecutive
    # slots hit k distinct folds. Ordering within the centre by size and chronicity means those axes
    # also land on different folds instead of clustering. Bucketing per (centre, size, chronicity) and
    # dealing bucket-by-bucket balances fold SIZES but lets a centre drift: measured spread of 3 folds
    # for one centre on a 240-subject synthetic cohort, which is what this ordering fixes.
    by_center: dict[str, list[Subject]] = {}
    for subject in subjects:
        by_center.setdefault(subject.center, []).append(subject)

    folds: list[list[str]] = [[] for _ in range(n_folds)]
    offset = 0
    for center in sorted(by_center, key=lambda c: (-len(by_center[c]), c)):
        pool = by_center[center]
        # Permute indices rather than the list itself: numpy's shuffle is for arrays, and a stable
        # sort afterwards then leaves the permutation as the tie-break WITHIN each stratum.
        members = [pool[i] for i in rng.permutation(len(pool))]
        members.sort(key=lambda s: (
            size_band(s.lesion_voxels, edges=edges), chronicity_band(s.days_post_stroke)
        ))
        for i, subject in enumerate(members):
            folds[(offset + i) % n_folds].append(subject.subject_id)
        # Carry the position so the next centre starts where this one stopped, keeping total fold
        # sizes balanced instead of every centre restarting at fold 0.
        offset = (offset + len(members)) % n_folds

    assert sum(len(f) for f in folds) == len(subjects)
    return folds
