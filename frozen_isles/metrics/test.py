"""Differential tests against the organizers' shipped code, plus the edges we plan to exploit.

The point of the differential test: our wrapper is only worth having if it is numerically the code
that ranks us. A test that re-derives our own formula would confirm nothing (frozen-wmh learned this
when a verifier that re-derived the implementation's own arithmetic missed a real max-pool bug).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from frozen_isles import metrics
from frozen_isles.metrics import (
    EMPTY_VALUE,
    evaluate_instances,
    instance_metrics,
    pr_auc,
    score_subject,
)

_REF = Path(__file__).resolve().parents[2] / "references" / "isles26_eval_utils.py"
_spec = importlib.util.spec_from_file_location("isles26_eval_utils", _REF)
assert _spec is not None and _spec.loader is not None
official = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(official)


def _two_lesions_and_a_near_miss() -> tuple[np.ndarray, np.ndarray]:
    """A reference with 2 separated lesions; the prediction hits one, splits the other, invents a third."""
    reference = np.zeros((24, 24, 24), np.uint8)
    reference[4:9, 4:9, 4:9] = 1
    reference[15:20, 15:20, 15:20] = 1
    prediction = np.zeros_like(reference)
    prediction[4:9, 4:9, 4:9] = 1          # clean hit
    prediction[15:17, 15:20, 15:20] = 1    # partial — tests the 0.25 IoU matching edge
    prediction[15:17, 2:5, 15:18] = 1      # a spurious instance, gapped from everything else
    return reference, prediction


@pytest.mark.parametrize("case", ["overlap", "empty_both", "empty_reference", "perfect"])
def test_matches_the_organizers_code_exactly(case: str) -> None:
    if case == "overlap":
        reference, prediction = _two_lesions_and_a_near_miss()
    elif case == "empty_both":
        reference = np.zeros((12, 12, 12), np.uint8)
        prediction = np.zeros_like(reference)
    elif case == "empty_reference":
        reference = np.zeros((12, 12, 12), np.uint8)
        prediction = np.zeros_like(reference)
        prediction[3:6, 3:6, 3:6] = 1
    else:
        reference, _ = _two_lesions_and_a_near_miss()
        prediction = reference.copy()

    ours_f1, ours_count, ours_dice = instance_metrics(reference=reference, prediction=prediction)
    their_f1, their_count, their_dice = official.compute_dice_f1_instance_difference(
        reference, prediction
    )
    assert ours_f1 == pytest.approx(their_f1)
    assert ours_count == their_count
    assert ours_dice == pytest.approx(their_dice)


def test_pr_auc_matches_the_organizers_code_on_a_soft_map() -> None:
    rng = np.random.default_rng(0)
    reference = np.zeros((16, 16, 16), np.uint8)
    reference[5:10, 5:10, 5:10] = 1
    soft = rng.random(reference.shape).astype(np.float32) * 0.4 + reference * 0.5
    assert pr_auc(reference=reference, soft_prediction=soft) == pytest.approx(
        official.compute_pr_auc(reference, soft)
    )


def test_volume_difference_matches_and_is_in_ml() -> None:
    reference = np.zeros((10, 10, 10), np.uint8)
    reference[:2, :2, :2] = 1                                  # 8 voxels
    prediction = np.zeros_like(reference)
    prediction[:3, :2, :2] = 1                                 # 12 voxels
    voxel_ml = 0.001                                           # a 1 mm isotropic voxel in mL
    score = score_subject(
        reference=reference, soft_prediction=prediction.astype(np.float32),
        voxel_volume_ml=voxel_ml,
    )
    assert score.abs_volume_difference_ml == pytest.approx(
        official.compute_absolute_volume_difference(reference, prediction, np.float64(voxel_ml))
    )
    assert score.abs_volume_difference_ml == pytest.approx(4 * voxel_ml)


def test_the_two_pr_auc_conventions_disagree_and_by_how_much() -> None:
    """The design doc says Average Precision; the shipped code takes a trapezoidal area. Not equal.

    If these ever coincide, someone has quietly changed one of them and the CHALLENGE.md conflict
    note is stale — that is what this test is for.
    """
    rng = np.random.default_rng(7)
    reference = np.zeros((16, 16, 16), np.uint8)
    reference[5:9, 5:9, 5:9] = 1
    soft = (rng.random(reference.shape).astype(np.float32) * 0.6 + reference * 0.35).clip(0, 1)

    shipped = pr_auc(reference=reference, soft_prediction=soft, convention="shipped")
    design = pr_auc(reference=reference, soft_prediction=soft, convention="design")
    assert shipped != pytest.approx(design, abs=1e-6)
    # Measured on six seeds of this synthetic case: the gap is ~3e-4, and `shipped` was the LOWER
    # of the two every time. Small here because the curve is dense; a real sparse-lesion PR curve has
    # far fewer operating points, which is where trapezoidal interpolation can diverge much further.
    # The magnitude on real data is unmeasured until we have predictions.
    assert abs(shipped - design) < 0.01


def test_the_design_convention_returns_nan_on_an_empty_reference() -> None:
    """The doc excludes empty-reference cases from the AP aggregation rather than scoring them."""
    reference = np.zeros((8, 8, 8), np.uint8)
    varying = np.full(reference.shape, 1e-9, np.float32)
    varying[0, 0, 0] = 2e-9
    assert np.isnan(pr_auc(reference=reference, soft_prediction=varying, convention="design"))


def test_a_constant_map_is_the_only_way_to_score_on_an_empty_reference() -> None:
    """Their empty branch is all-or-nothing, and it decides how we must emit "no lesion here"."""
    reference = np.zeros((8, 8, 8), np.uint8)
    assert pr_auc(reference=reference, soft_prediction=np.zeros(reference.shape, np.float32)) == EMPTY_VALUE
    almost_zero = np.full(reference.shape, 1e-9, np.float32)
    almost_zero[0, 0, 0] = 2e-9
    assert pr_auc(reference=reference, soft_prediction=almost_zero) == 0.0


def test_fragmenting_one_lesion_is_punished_on_both_instance_metrics() -> None:
    """The design claim behind our full-res-logit head: splitting costs F1 AND the count difference."""
    reference = np.zeros((20, 20, 20), np.uint8)
    reference[5:15, 5:8, 5:8] = 1                              # one elongated lesion
    whole = reference.copy()
    split = reference.copy()
    split[9:11, :, :] = 0                                      # sever it into two components

    f1_whole, count_whole, _ = instance_metrics(reference=reference, prediction=whole)
    f1_split, count_split, _ = instance_metrics(reference=reference, prediction=split)
    assert f1_split < f1_whole
    assert count_split > count_whole


def test_score_subject_rejects_a_shape_mismatch() -> None:
    with pytest.raises(AssertionError, match="shape mismatch"):
        score_subject(
            reference=np.zeros((4, 4, 4), np.uint8),
            soft_prediction=np.zeros((4, 4, 5), np.float32),
            voxel_volume_ml=0.001,
        )


def test_an_empty_reference_pays_1_0_on_three_metrics_when_the_map_is_constant() -> None:
    """The lever, verified against the scoring code rather than against my reading of it.

    The organizers confirmed empty ground truth scores 1. This shows what that is worth: a lesion-free
    case answered with an exactly-constant map takes Dice, lesion-F1 AND PR-AUC at 1.0 simultaneously.
    """
    reference = np.zeros((16, 16, 16), np.uint8)
    constant = np.zeros(reference.shape, np.float32)

    score = score_subject(
        reference=reference.astype(bool), soft_prediction=constant, voxel_volume_ml=0.001
    )
    assert score.dice == 1.0
    assert score.lesion_f1 == 1.0
    assert score.pr_auc == 1.0
    assert score.abs_lesion_count_difference == 0
    assert score.abs_volume_difference_ml == 0.0


def test_the_same_empty_case_scores_zero_if_the_map_merely_looks_empty() -> None:
    """Sub-threshold noise segments to nothing, so Dice and F1 still pay — but PR-AUC collapses."""
    reference = np.zeros((16, 16, 16), np.uint8)
    almost = np.full(reference.shape, 1e-9, np.float32)
    almost[0, 0, 0] = 2e-9                                     # one voxel of dust

    score = score_subject(
        reference=reference.astype(bool), soft_prediction=almost, voxel_volume_ml=0.001
    )
    assert score.dice == 1.0                                   # nothing crosses the threshold
    assert score.lesion_f1 == 1.0
    assert score.pr_auc == 0.0, "PR-AUC is the metric that punishes a non-constant 'empty' map"


def _blob(seed: int, n_lesions: int = 3) -> "np.ndarray":
    """A small synthetic lesion map — enough instances for the matcher to have work to do."""
    generator = np.random.default_rng(seed)
    volume = np.zeros((32, 32, 8), dtype=bool)
    for _ in range(n_lesions):
        x, y, z = generator.integers(4, 26), generator.integers(4, 26), generator.integers(1, 6)
        volume[x - 2:x + 2, y - 2:y + 2, z - 1:z + 1] = True
    return volume


def test_the_shared_evaluator_does_not_carry_state_between_subjects() -> None:
    """`metrics` builds ONE `Panoptica_Evaluator` at import and reuses it for all 1,453 subjects.

    That is a deliberate optimisation — constructing one per subject is pure waste — but it is only
    safe if `evaluate` is stateless. If it accumulated anything, a subject's score would depend on
    what was scored before it: invisible in any log, and enough to make two runs over the same cohort
    disagree because the fold order differed.

    So score one fixed pair three times — cold, after five other subjects, and after those five again
    in the opposite order — and require all three to be identical.
    """
    reference, prediction = _blob(1), _blob(2)
    cold = instance_metrics(reference=reference, prediction=prediction)

    for k in range(5):
        instance_metrics(reference=_blob(10 + k), prediction=_blob(20 + k))
    after_others = instance_metrics(reference=reference, prediction=prediction)

    for k in reversed(range(5)):
        instance_metrics(reference=_blob(20 + k), prediction=_blob(10 + k))
    after_reversed = instance_metrics(reference=reference, prediction=prediction)

    assert cold == after_others == after_reversed, (
        f"the shared evaluator is STATEFUL: {cold} cold, {after_others} after five other subjects, "
        f"{after_reversed} after them reversed — scores depend on call order"
    )


def test_pr_auc_survives_a_zero_voxel_volume() -> None:
    """The empty-reference branch used to index `pred[0]` unguarded.

    No zero-voxel volume exists in ATLAS, so this never fired — but it sat in the scoring path for
    the one metric where an empty reference is a real case (five training subjects have empty masks),
    and an IndexError there would kill a run at validation rather than at load.
    """
    empty = np.zeros(0, dtype=float)
    assert pr_auc(reference=empty, soft_prediction=empty) == EMPTY_VALUE
    assert np.isnan(pr_auc(reference=empty, soft_prediction=empty, convention="design"))


def test_post_processing_changes_the_mask_metrics_and_never_pr_auc() -> None:
    """The challenge takes TWO outputs: a binary mask and a probability map.

    Four metrics come from the mask, PR-AUC alone from the map. They are the same thing only while
    the mask IS the thresholded map — and post-processing breaks that, since closing and size
    filtering change the mask and leave the map untouched. `score_cohort` used to load the mask and
    discard it whenever a map was supplied, which would have scored a post-processed submission as
    though it had not been post-processed, and made post-processing appear to move PR-AUC.
    """
    import numpy as np

    from frozen_isles.metrics import score_subject

    reference = np.zeros((12, 12, 12), dtype=bool)
    reference[4:8, 4:8, 4:8] = True
    soft = np.zeros_like(reference, dtype=np.float32)
    soft[4:8, 4:8, 4:8] = 0.9
    soft[0, 0, 0] = 0.9                      # a one-voxel speck a size filter would drop

    from_map = score_subject(reference=reference, soft_prediction=soft, voxel_volume_ml=1.0)
    despecked = soft >= 0.5
    despecked[0, 0, 0] = False
    with_mask = score_subject(reference=reference, soft_prediction=soft,
                              binary_prediction=despecked, voxel_volume_ml=1.0)

    assert with_mask.abs_lesion_count_difference < from_map.abs_lesion_count_difference, (
        "dropping the speck must improve the count metric — the mask is not driving it"
    )
    assert with_mask.pr_auc == from_map.pr_auc, (
        "PR-AUC comes from the untouched soft map and cannot move when only the mask changes"
    )


def test_instance_counts_splits_lesion_f1_into_its_parts():
    """One matched pair, one reference-only lesion, one prediction-only lesion.

    lesion-F1 reports 0.5 for this case; the split says WHY — one invented lesion and one missed —
    which is the difference between wanting a stricter filter and wanting a lower threshold.
    """
    reference = np.zeros((40, 40, 40), dtype=np.uint8)
    prediction = np.zeros_like(reference)
    reference[5:15, 5:15, 5:15] = 1
    prediction[6:16, 6:16, 6:16] = 1        # overlaps well above the 0.25 matching threshold
    reference[25:30, 25:30, 25:30] = 1      # never predicted -> false negative
    prediction[32:36, 32:36, 32:36] = 1     # no reference -> false positive

    counts = metrics.instance_counts(reference=reference, prediction=prediction)
    assert (counts.true_positives, counts.false_positives, counts.false_negatives) == (1, 1, 1)
    assert counts.precision == 0.5
    assert counts.recall == 0.5


def test_instance_counts_on_empty_masks_does_not_divide_by_zero():
    """The organizers' convention: nothing to find and nothing found is perfect, not undefined."""
    empty = np.zeros((16, 16, 16), dtype=np.uint8)
    counts = metrics.instance_counts(reference=empty, prediction=empty)
    assert counts == metrics.InstanceCounts(0, 0, 0)
    assert counts.precision == metrics.EMPTY_VALUE
    assert counts.recall == metrics.EMPTY_VALUE


def test_one_evaluation_gives_what_two_separate_calls_give() -> None:
    """`evaluate_instances` exists so a caller needing both pays panoptica once instead of twice.

    It is only worth having if it is exactly equivalent; a faster path that quietly differs would be
    worse than the duplication it replaces.
    """
    reference = _blob(10)
    prediction = _blob(12)

    together = evaluate_instances(reference=reference, prediction=prediction)

    assert together.metrics == instance_metrics(reference=reference, prediction=prediction)
    assert together.counts == metrics.instance_counts(reference=reference, prediction=prediction)


def test_instance_metrics_fields_are_named_because_the_order_differs_from_subjectscore() -> None:
    """Dice is last here and first in SubjectScore. Unpacking by position is easy to get backwards."""
    result = instance_metrics(reference=_blob(10), prediction=_blob(10))

    lesion_f1, count_difference, dice = result
    assert (result.lesion_f1, result.abs_lesion_count_difference, result.dice) == (
        lesion_f1, count_difference, dice
    )
