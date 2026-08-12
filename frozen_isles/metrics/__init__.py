"""The ISLES'26 ranking metrics, computed by the organizers' own code path.

This module deliberately does NOT reimplement the five metrics. `panoptica` and `sklearn` are the
same libraries `utils/eval_utils.py` calls in github.com/ezequieldlrosa/isles26, configured with the
same arguments, so a number produced here is the number the leaderboard produces. A parallel
implementation would be a second source of truth for the one quantity we are optimising.

What this module adds over calling their functions directly: types, a single per-subject result
object, and one evaluator built once (constructing `Panoptica_Evaluator` per subject is pure waste
over 1453 scans).

⚠️ **panoptica spawns multiprocessing workers.** Any script that calls into here must live in a FILE
and be guarded by `if __name__ == "__main__":`. Calling it from `python - <<EOF` or `python -c` makes
each spawned child re-import a `__main__` it cannot find, and the process storm hangs — measured at
over ten minutes for one subject before being killed. This also means nested parallelism needs care
when scoring all 1453 subjects.
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt
from panoptica import (
    ConnectedComponentsInstanceApproximator,
    InputType,
    NaiveThresholdMatching,
    Panoptica_Evaluator,
)
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

# Which of the two conflicting PR-AUC definitions to use — see `pr_auc`. Not a style preference:
# the two produce different numbers and the organizers have shipped both.
PrAucConvention = Literal["shipped", "design"]

# IoU at which a predicted instance is accepted as matching a reference instance. Pinned to the
# organizers' value; changing it changes what "lesion-wise F1" means.
MATCHING_THRESHOLD = 0.25

# Their convention: a metric on two empty masks is a perfect score, not a division by zero.
EMPTY_VALUE = 1.0


@dataclass(frozen=True)
class SubjectScore:
    dice: float
    lesion_f1: float
    abs_volume_difference_ml: float
    abs_lesion_count_difference: int
    pr_auc: float


_evaluator = Panoptica_Evaluator(
    expected_input=InputType.SEMANTIC,
    instance_approximator=ConnectedComponentsInstanceApproximator(),
    instance_matcher=NaiveThresholdMatching(matching_threshold=MATCHING_THRESHOLD),
)


def pr_auc(
    *,
    reference: npt.NDArray,
    soft_prediction: npt.NDArray,
    convention: PrAucConvention = "shipped",
) -> float:
    """Voxel-wise PR-AUC over the CONTINUOUS map — thresholding first forfeits this metric.

    ⚠️ The design document and the shipped `eval_utils.py` DISAGREE, in two ways at once, and which
    one runs on the test set is unresolved (see `docs/CHALLENGE.md` and the organizers' forum thread
    "Design document vs. eval_utils.py"). Both are implemented so the gap can be measured instead of
    assumed away:

    - `"shipped"` — `auc(recall, precision)`, the TRAPEZOIDAL area, and an empty reference scores 1.0
      when the soft map is exactly constant, else 0.0.
    - `"design"` — Average Precision, the NON-interpolated step function the document specifies
      ("consistent with common implementations, e.g. scikit-learn"), and an empty reference is NaN,
      excluded from the aggregation rather than scored.

    Trapezoidal interpolation is optimistic on a precision-recall curve, so these are not two spellings
    of one number.
    """
    gt = np.asarray(reference).astype(np.bool_).ravel()
    pred = np.asarray(soft_prediction).astype(np.float32).ravel()
    assert gt.shape == pred.shape, f"shape mismatch: reference {gt.shape} vs prediction {pred.shape}"

    if not gt.any():
        if convention == "design":
            return float("nan")
        # `pred.size == 0` first: a zero-voxel volume is vacuously constant, and without this guard
        # `pred[0]` raises IndexError inside the scoring path for the one metric where an empty
        # reference is a real case — five training subjects have empty masks.
        is_constant = pred.size == 0 or bool(np.all(pred == pred[0]))
        return EMPTY_VALUE if is_constant else 0.0

    if convention == "design":
        return float(average_precision_score(gt, pred))
    precision, recall, _ = precision_recall_curve(gt, pred)
    return float(auc(recall, precision))


@dataclass(frozen=True)
class InstanceCounts:
    """Matched and unmatched LESIONS, which lesion-F1 collapses into one number.

    lesion-F1 is the harmonic mean of instance precision and recall, so a 0.5 can mean the model
    invents lesions, misses them, or both equally. Those want opposite remedies — a stricter filter
    against false positives, a lower threshold against false negatives — so the split is what makes
    the number actionable.
    """

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        found = self.true_positives + self.false_positives
        return EMPTY_VALUE if found == 0 else self.true_positives / found

    @property
    def recall(self) -> float:
        present = self.true_positives + self.false_negatives
        return EMPTY_VALUE if present == 0 else self.true_positives / present


class InstanceMetrics(NamedTuple):
    """The three instance-derived metrics. A NamedTuple so `a, b, c = ...` still reads correctly.

    ⚠️ Dice is LAST here and FIRST in `SubjectScore`. The orders differ, which is why these are named.
    """

    lesion_f1: float
    abs_lesion_count_difference: int
    dice: float


@dataclass(frozen=True)
class InstanceEvaluation:
    metrics: InstanceMetrics
    counts: InstanceCounts


def evaluate_instances(*, reference: npt.NDArray, prediction: npt.NDArray) -> InstanceEvaluation:
    """One panoptica pass, returning everything derived from it.

    **Call this when you want both the metrics and the TP/FP/FN split.** `instance_metrics` and
    `instance_counts` each run the matching from scratch — measured at ~2.9 s on a 150x170x135 volume
    — so asking for them separately costs twice, about 14 minutes over 291 subjects.
    """
    result, _ = _evaluator.evaluate(
        np.asarray(prediction).astype(int), np.asarray(reference).astype(int), verbose=False
    )["ungrouped"]
    count_difference = abs(result.num_ref_instances - result.num_pred_instances)
    empty = result.num_ref_instances == 0 and result.num_pred_instances == 0
    metrics = InstanceMetrics(
        lesion_f1=EMPTY_VALUE if empty else float(result.rq),
        abs_lesion_count_difference=count_difference,
        dice=EMPTY_VALUE if empty else float(result.global_bin_dsc),
    )
    counts = InstanceCounts(int(result.tp), int(result.fp), int(result.fn))
    return InstanceEvaluation(metrics=metrics, counts=counts)


def instance_metrics(*, reference: npt.NDArray, prediction: npt.NDArray) -> InstanceMetrics:
    return evaluate_instances(reference=reference, prediction=prediction).metrics


def instance_counts(*, reference: npt.NDArray, prediction: npt.NDArray) -> InstanceCounts:
    """The TP/FP/FN behind `lesion_f1`, matched at the same IoU threshold the challenge uses."""
    return evaluate_instances(reference=reference, prediction=prediction).counts


def score_subject(
    *,
    reference: npt.NDArray,
    soft_prediction: npt.NDArray,
    voxel_volume_ml: float,
    threshold: float = 0.5,
    pr_auc_convention: PrAucConvention = "shipped",
    binary_prediction: npt.NDArray | None = None,
) -> SubjectScore:
    """All five ranking metrics for one subject.

    `voxel_volume_ml` is the product of the voxel dimensions in mL, read from the subject's own
    affine — in native space it differs between subjects, so it is never a module constant.

    **`binary_prediction` exists because the challenge takes TWO outputs, not one.** Teams submit a
    binary mask AND a probability map; four of the five metrics come from the mask and PR-AUC alone
    comes from the map. When the mask is simply the thresholded map they are the same thing, which is
    why this defaults to None and the default path is byte-identical to before.

    ⚠️ **They come apart the moment POST-PROCESSING is involved.** Closing gaps and dropping small
    components changes the mask and leaves the probability map untouched, so scoring a post-processed
    submission by re-thresholding its soft map would measure the un-post-processed model — and, worse,
    would make post-processing appear to change PR-AUC, which it cannot.
    """
    assert reference.shape == soft_prediction.shape, (
        f"shape mismatch: reference {reference.shape} vs prediction {soft_prediction.shape}"
    )
    assert voxel_volume_ml > 0, f"voxel_volume_ml must be positive, got {voxel_volume_ml}"

    if binary_prediction is None:
        binary = soft_prediction >= threshold
    else:
        assert binary_prediction.shape == reference.shape, (
            f"shape mismatch: reference {reference.shape} vs mask {binary_prediction.shape}"
        )
        binary = np.asarray(binary_prediction).astype(bool)
    lesion_f1, count_difference, dice = instance_metrics(reference=reference, prediction=binary)
    volume_difference = abs(
        int(np.asarray(reference).astype(bool).sum()) - int(binary.sum())
    ) * voxel_volume_ml
    return SubjectScore(
        dice=dice,
        lesion_f1=lesion_f1,
        abs_volume_difference_ml=volume_difference,
        abs_lesion_count_difference=count_difference,
        pr_auc=pr_auc(
            reference=reference, soft_prediction=soft_prediction, convention=pr_auc_convention
        ),
    )
