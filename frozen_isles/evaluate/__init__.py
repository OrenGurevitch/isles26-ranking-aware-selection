"""Scoring a whole cohort, and turning several models' scores into the challenge's ranking.

Two jobs kept apart on purpose. `score_cohort` reads volumes off disk and produces one `SubjectScore`
per subject. `metric_arrays` reshapes several models' scores into the `(n_models, n_cases)` arrays
`frozen_isles.ranking` consumes. Nothing here decides what a good model is — that is the ranking's job,
and it is not mean Dice.

⚠️ **panoptica spawns multiprocessing workers**, so any caller must be a file with an
`if __name__ == "__main__":` guard. From a heredoc or `python -c`, each spawned child re-imports a
`__main__` it cannot find and the process storm hangs — measured at over ten minutes on ONE subject.
`scripts/score_cohort.py` is the guarded entry point.

**Subjects are aligned by id, never by position.** The sister project lost real work to a loader that
globbed files and stacked them: shapes matched regardless of which condition each came from, so it
"worked" and produced pooled statistics over mixed conditions with no error. Here a per-case rank is
meaningless if model A's case 7 is not model B's case 7, so the alignment is explicit and a mismatched
subject set is refused rather than truncated to the intersection.
"""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from frozen_isles.metrics import PrAucConvention, SubjectScore, score_subject
from frozen_isles.nifti import load_mask, voxel_volume_ml
from frozen_isles.postprocess import postprocess
from frozen_isles.ranking import METRIC_POLARITY

# The SubjectScore field behind each ranking metric name. Stated once so a rename cannot silently
# desynchronise the polarity table from the score object.
_FIELD_OF_METRIC = {
    "dice": "dice",
    "lesion_f1": "lesion_f1",
    "pr_auc": "pr_auc",
    "abs_volume_difference_ml": "abs_volume_difference_ml",
    "abs_lesion_count_difference": "abs_lesion_count_difference",
}


def score_cohort(
    *,
    references: Mapping[str, Path],
    predictions: Mapping[str, Path],
    soft_predictions: Mapping[str, Path] | None = None,
    threshold: float = 0.5,
    pr_auc_convention: PrAucConvention = "shipped",
    min_voxels: int = 0,
    closing_radius: int = 0,
) -> dict[str, SubjectScore]:
    """One `SubjectScore` per subject, keyed by subject id.

    `predictions` are binary masks. `soft_predictions` are the probability maps PR-AUC needs; when a
    subject has none, its binary mask stands in — which caps PR-AUC at what a thresholded map can
    achieve, so it is a fallback for development, not for a submission.

    ⚠️ **`min_voxels` and `closing_radius` default to 0, which scores the model's RAW output.** The
    submitted container ships `min_voxels=25`, so a number reported without them describes a different
    system from the one submitted — which is exactly what happened to this project's headline row
    (found 2026-08-05). Pass the shipped settings to score the shipped configuration. The filter is
    applied to the MASK only; PR-AUC comes from the untouched probability map, because post-processing
    cannot change it.

    Voxel volume is read from each subject's OWN reference header. In native space it varies between
    subjects, and for the 37 r032 cases the organizers ruled that the header stands even though its
    3 mm A-P spacing is probably wrong by ~2× — matching the graders beats being anatomically right.
    """
    missing = sorted(set(references) - set(predictions))
    assert not missing, f"no prediction for {len(missing)} subject(s): {missing[:5]}"
    extra = sorted(set(predictions) - set(references))
    assert not extra, f"prediction without a reference for {len(extra)} subject(s): {extra[:5]}"

    scores: dict[str, SubjectScore] = {}
    for subject_id in sorted(references):
        reference = load_mask(references[subject_id])
        binary = load_mask(predictions[subject_id])
        if min_voxels or closing_radius:
            binary = postprocess(binary, min_voxels=min_voxels, closing_radius=closing_radius) > 0
        soft_path = (soft_predictions or {}).get(subject_id)
        soft = binary.astype(np.float32) if soft_path is None else _load_soft(soft_path)
        # ⚠️ The MASK drives the four mask-based metrics and the SOFT MAP drives PR-AUC. Until
        # 2026-08-02 this loaded `binary` and then discarded it whenever a soft map was supplied, so
        # every metric came from re-thresholding the map. That is identical when the mask IS the
        # thresholded map — nnU-Net's exported mask is its own argmax — and silently wrong for a
        # POST-PROCESSED submission, where the mask is closed and filtered and the map is not.
        scores[subject_id] = score_subject(
            reference=reference,
            soft_prediction=soft,
            binary_prediction=binary,
            voxel_volume_ml=voxel_volume_ml(references[subject_id]),
            threshold=threshold,
            pr_auc_convention=pr_auc_convention,
        )
    return scores


def _load_soft(path: Path) -> npt.NDArray[np.float32]:
    import nibabel as nib

    image = nib.load(path)
    assert isinstance(image, nib.Nifti1Image), f"{path.name} is not a NIfTI"
    return np.asanyarray(image.dataobj).astype(np.float32)


def metric_arrays(
    scores_by_model: Mapping[str, Mapping[str, SubjectScore]],
) -> tuple[list[str], list[str], dict[str, npt.NDArray]]:
    """(model order, subject order, {metric: (n_models, n_cases)}) — the ranking's input shape.

    Every model must cover exactly the same subjects. A model that skipped one is refused here rather
    than quietly compared on a smaller set: under rank-then-aggregate a per-case rank across models is
    only meaningful if the case is the same case.
    """
    assert len(scores_by_model) >= 2, "ranking needs at least two models to compare"
    models = sorted(scores_by_model)
    subject_sets = {name: set(scores) for name, scores in scores_by_model.items()}
    reference_set = subject_sets[models[0]]
    for name in models[1:]:
        difference = reference_set.symmetric_difference(subject_sets[name])
        assert not difference, (
            f"{models[0]} and {name} score different subjects "
            f"({len(difference)} differ, e.g. {sorted(difference)[:5]}) — "
            "a per-case rank requires the same cases"
        )

    subjects = sorted(reference_set)
    arrays = {
        metric: np.array(
            [[getattr(scores_by_model[m][s], field) for s in subjects] for m in models], dtype=float
        )
        for metric, field in _FIELD_OF_METRIC.items()
    }
    assert set(arrays) == set(METRIC_POLARITY), "metric names drifted from the polarity table"
    return models, subjects, arrays


def summarise(scores: Mapping[str, SubjectScore]) -> dict[str, float]:
    """Median per metric — what the organizers report, being "less impacted by outliers" than a mean.

    A summary for reading, never for selection: selection goes through `frozen_isles.ranking`.
    """
    return {
        metric: float(np.median([getattr(s, field) for s in scores.values()]))
        for metric, field in _FIELD_OF_METRIC.items()
    }


def worst_subjects(
    scores_by_model: Mapping[str, Mapping[str, SubjectScore]], *, model: str, k: int = 10
) -> list[str]:
    """The subject ids where `model` places worst against the others — where its rank is lost."""
    from frozen_isles.ranking import worst_cases

    models, subjects, arrays = metric_arrays(scores_by_model)
    indices = worst_cases(scores=arrays, model=models.index(model), k=k)
    return [subjects[i] for i in indices]


def rank(scores_by_model: Mapping[str, Mapping[str, SubjectScore]]) -> dict[str, float]:
    """Final rank per model under the challenge's own scheme. Lower is better."""
    from frozen_isles.ranking import rank_then_aggregate

    models, _, arrays = metric_arrays(scores_by_model)
    return dict(zip(models, (float(r) for r in rank_then_aggregate(scores=arrays)), strict=True))


def as_subject_list(paths: Sequence[Path], *, suffix: str) -> dict[str, Path]:
    """Map filenames to subject ids by their `sub-XXXX` prefix.

    ⚠️ The extension is stripped BEFORE splitting on `_`, because a file named for the subject alone
    has no underscore at all and `"sub-r001s002.nii.gz".split("_")[0]` returns the whole filename.
    Our own BIDS-style masks always carry a `_space-orig...` suffix, so this went unnoticed until
    nnU-Net's predictions — plain `sub-XXXX.nii.gz` — keyed on the extension and matched none of the
    291 references, failing a scoring job after its four soft-map exports had already run.
    """
    mapping: dict[str, Path] = {}
    for path in paths:
        stem = path.name
        for extension in (".nii.gz", ".nii"):
            stem = stem.removesuffix(extension)
        subject_id = stem.split("_")[0]
        assert subject_id.startswith("sub-"), f"{path.name} does not start with a sub- id"
        assert subject_id not in mapping, f"two {suffix} files for {subject_id}"
        mapping[subject_id] = path
    return mapping


# The challenge lets us pick the cutoff: "participants submit their best-performing binary masks by
# choosing their preferred cutoff and post-processing strategies". 0.5 is a default, not a given, and
# the first real fold showed why — median Dice 0.000 alongside PR-AUC 0.195, i.e. a soft map ranking
# lesion voxels far above chance while its 0.5 binarisation scored nothing.
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = (
    0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9
)


def sweep_threshold(
    *,
    references: Mapping[str, npt.NDArray],
    soft_predictions: Mapping[str, npt.NDArray],
    voxel_volumes_ml: Mapping[str, float],
    thresholds: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    pr_auc_convention: PrAucConvention = "shipped",
) -> tuple[float, dict[float, dict[str, SubjectScore]]]:
    """Score every threshold on the SAME soft maps and rank them by the challenge's own scheme.

    Returns the winning threshold and every threshold's per-subject scores. Each threshold is a
    competing "model" over identical cases, which is exactly what `rank_then_aggregate` is for — and
    what selecting on mean Dice would get wrong in the usual way.

    ⚠️ Choose the threshold on VALIDATION only. Picking it on the data you then report is test-peeking,
    and the hidden test is scored once.
    """
    assert set(references) == set(soft_predictions), "references and predictions cover different subjects"
    cases = (
        (subject_id, references[subject_id], soft_predictions[subject_id], voxel_volumes_ml[subject_id])
        for subject_id in sorted(references)
    )
    return sweep_threshold_over(cases, thresholds=thresholds, pr_auc_convention=pr_auc_convention)


def sweep_threshold_over(
    cases: Iterable[tuple[str, npt.NDArray, npt.NDArray, float]],
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    pr_auc_convention: PrAucConvention = "shipped",
) -> tuple[float, dict[float, dict[str, SubjectScore]]]:
    """Streaming form: scores each case at every threshold, then drops it.

    291 native-space soft maps are ~12 GB in float32, so holding the cohort in memory to sweep it is
    not an option on any machine we run on. Each case is scored across the whole grid while it is
    loaded, and only the scores are kept.
    """
    assert thresholds, "no thresholds given"
    grid = [float(t) for t in thresholds]
    by_threshold: dict[float, dict[str, SubjectScore]] = {t: {} for t in grid}

    for subject_id, reference, soft, voxel_ml in cases:
        for threshold in grid:
            by_threshold[threshold][subject_id] = score_subject(
                reference=reference, soft_prediction=soft, voxel_volume_ml=voxel_ml,
                threshold=threshold, pr_auc_convention=pr_auc_convention,
            )

    named = {f"t={t:g}": scores for t, scores in by_threshold.items()}
    ranked = rank(named)
    best_name = min(ranked, key=lambda k: ranked[k])
    return float(best_name.removeprefix("t=")), by_threshold
