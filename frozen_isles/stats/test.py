"""Contracts for the bootstrap — above all that pairing follows the subject id, not array position."""

import numpy as np
import pytest

from frozen_isles.metrics import SubjectScore
from frozen_isles.stats import mean_ci, paired_delta, paired_delta_by_subject


def _score(dice: float) -> SubjectScore:
    return SubjectScore(dice=dice, lesion_f1=dice / 2, abs_volume_difference_ml=1.0,
                        abs_lesion_count_difference=1, pr_auc=dice)


def test_a_real_difference_is_significant_and_noise_is_not() -> None:
    """The Δ-CI excluding 0 IS the significance test — it has to separate signal from noise."""
    rng = np.random.default_rng(0)
    baseline = rng.normal(0.50, 0.05, size=200)

    shifted = baseline + 0.08                       # a uniform, real improvement
    assert paired_delta(shifted, baseline).excludes_zero()

    jittered = baseline + rng.normal(0.0, 0.05, size=200)   # same mean, pure noise
    assert not paired_delta(jittered, baseline).excludes_zero()


def test_pairing_follows_the_subject_id_not_the_array_position() -> None:
    """Two runs whose subjects are stored in different orders must give the SAME Δ.

    This is the whole reason `paired_delta_by_subject` exists. Position-pairing two dicts that
    happen to be the same length produces a delta that is confidently wrong, with no error --
    the failure mode that already cost this group once, in a loader that stacked mixed conditions
    because the shapes agreed.
    """
    better = {f"sub-{i}": _score(0.6) for i in range(10)}
    worse = {f"sub-{i}": _score(0.4) for i in range(10)}
    shuffled = dict(reversed(list(worse.items())))

    delta = paired_delta_by_subject(better, worse, "dice")
    assert delta.mean == pytest.approx(0.2)
    assert paired_delta_by_subject(better, shuffled, "dice").mean == pytest.approx(delta.mean)


def test_only_subjects_in_both_runs_are_compared() -> None:
    """A fold that scored a subject the other skipped must not shift the comparison."""
    a = {f"sub-{i}": _score(0.6) for i in range(10)}
    b = {f"sub-{i}": _score(0.5) for i in range(6)}          # 4 subjects missing
    assert paired_delta_by_subject(a, b, "dice").mean == pytest.approx(0.1)


def test_a_constant_sample_keeps_a_degenerate_interval() -> None:
    """24% of ISLES subjects score exactly 0.000, so a constant sample is reachable, not theoretical."""
    ci = mean_ci([0.0] * 20)
    assert (ci.mean, ci.lo, ci.hi) == (0.0, 0.0, 0.0)
    assert not ci.excludes_zero(), "a degenerate interval at 0 must never read as significant"


def test_an_unknown_metric_is_refused() -> None:
    """A typo'd metric name must fail loudly rather than compare something else."""
    a = {f"sub-{i}": _score(0.6) for i in range(4)}
    with pytest.raises(AssertionError, match="not a SubjectScore field"):
        paired_delta_by_subject(a, a, "voxel_dice")
