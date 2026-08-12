import numpy as np
import pytest

from frozen_isles.ranking import per_case_ranks, rank_then_aggregate, worst_cases


def _scores(**metrics: list[list[float]]) -> dict[str, np.ndarray]:
    return {name: np.array(rows, dtype=float) for name, rows in metrics.items()}


def test_a_model_that_wins_every_metric_on_every_case_ranks_first() -> None:
    scores = _scores(
        dice=[[0.9, 0.8, 0.7], [0.5, 0.4, 0.3]],
        abs_volume_difference_ml=[[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]],
    )
    final = rank_then_aggregate(scores=scores)
    assert final[0] == pytest.approx(1.0)
    assert final[1] == pytest.approx(2.0)


def test_polarity_is_respected_for_the_lower_is_better_metrics() -> None:
    """Volume and count differences are errors — a smaller one must rank BETTER."""
    scores = _scores(abs_lesion_count_difference=[[0.0, 0.0], [5.0, 5.0]])
    final = rank_then_aggregate(scores=scores)
    assert final[0] < final[1]


def test_a_single_catastrophic_case_costs_one_rank_position_not_the_comparison() -> None:
    """The design claim, pinned in the direction it actually holds.

    The organizers chose rank-then-aggregate because it is "less influenced by outliers in the
    case-level metrics" than aggregate-then-rank. So the outlier is CAPPED: `flashy` collapses to 0.0
    on one case and still wins, because that case can cost it at most one rank position. Selecting on
    mean Dice would have picked `steady`. The two orderings disagree, and the challenge's is the one
    that counts.
    """
    flashy = [0.90, 0.90, 0.90, 0.90, 0.00]
    steady = [0.75, 0.75, 0.75, 0.75, 0.75]
    scores = _scores(dice=[flashy, steady])

    assert np.mean(flashy) < np.mean(steady), "the premise: the disaster sinks flashy's MEAN"
    final = rank_then_aggregate(scores=scores)
    assert final[0] < final[1], "but the challenge still ranks flashy first"


def test_being_narrowly_second_on_every_case_is_the_expensive_failure() -> None:
    """The flip side, and the one that should drive model selection.

    `always_second` loses every case by a hair and is ranked last; `flashy` above wins despite a total
    collapse. Under this scheme uniform competitiveness across ALL cases is worth more than peak
    quality, and a small consistent deficit is worth more attention than a rare disaster.
    """
    winner = [0.50, 0.50, 0.50, 0.50, 0.50]
    always_second = [0.49, 0.49, 0.49, 0.49, 0.49]
    final = rank_then_aggregate(scores=_scores(dice=[winner, always_second]))
    assert final[1] == pytest.approx(2.0), "a 0.01 deficit everywhere is a last place everywhere"


def test_a_missing_result_takes_the_worst_rank_rather_than_being_skipped() -> None:
    """Their rule for a container that fails on a case: worst possible value, not an exclusion."""
    scores = _scores(dice=[[0.9, np.nan], [0.5, 0.5]])
    ranks = per_case_ranks(scores=scores)
    assert ranks[0, 0] == pytest.approx(1.0)      # wins the case it completed
    assert ranks[0, 1] == pytest.approx(2.0)      # worst rank on the one it missed
    assert ranks[1, 1] == pytest.approx(1.0)


def test_ties_share_the_average_rank() -> None:
    scores = _scores(dice=[[0.5], [0.5]])
    assert rank_then_aggregate(scores=scores) == pytest.approx([1.5, 1.5])


def test_every_metric_carries_equal_weight() -> None:
    """Four metrics favour one model, one favours the other; the majority must decide."""
    scores = _scores(
        dice=[[0.9], [0.1]], lesion_f1=[[0.9], [0.1]], pr_auc=[[0.9], [0.1]],
        abs_volume_difference_ml=[[1.0], [9.0]],
        abs_lesion_count_difference=[[9.0], [1.0]],       # the lone dissenter
    )
    final = rank_then_aggregate(scores=scores)
    assert final[0] < final[1]
    assert final[0] == pytest.approx((1 + 1 + 1 + 1 + 2) / 5)


def test_worst_cases_points_at_the_tail_that_costs_the_rank() -> None:
    scores = _scores(dice=[[0.9, 0.9, 0.01, 0.9], [0.5, 0.5, 0.5, 0.5]])
    assert worst_cases(scores=scores, model=0, k=1).tolist() == [2]


def test_an_undeclared_metric_is_refused_rather_than_guessed() -> None:
    with pytest.raises(AssertionError, match="no polarity declared"):
        rank_then_aggregate(scores=_scores(hausdorff_95=[[1.0], [2.0]]))


def test_metrics_of_disagreeing_shape_are_refused() -> None:
    scores = {"dice": np.zeros((2, 3)), "pr_auc": np.zeros((2, 4))}
    with pytest.raises(AssertionError, match="disagree on shape"):
        rank_then_aggregate(scores=scores)
