import numpy as np
import pytest
from postproc_selection_bias import advantage_over_baseline, mean_ranks

BASELINE = "min0"


def _cohort(effects: dict[str, float], n: int = 120, noise: float = 0.25, seed: int = 0):
    """Per-subject scores where each setting shifts every metric by a constant, plus subject noise."""
    rng = np.random.default_rng(seed)
    subjects = [f"sub-{i:03d}" for i in range(n)]
    per_subject_noise = rng.normal(0, noise, n)
    scores = {}
    for setting, effect in effects.items():
        scores[setting] = {}
        for i, subject in enumerate(subjects):
            value = 0.5 + effect + per_subject_noise[i] + rng.normal(0, 0.05)
            scores[setting][subject] = {
                "dice": value, "lesion_f1": value, "pr_auc": value,
                # polarity is inverted for these two, so negate to keep "better" consistent
                "abs_volume_difference_ml": -value, "abs_lesion_count_difference": -value,
            }
    return scores, subjects


def test_a_genuinely_better_setting_beats_the_baseline() -> None:
    scores, subjects = _cohort({BASELINE: 0.0, "min25": 0.30})
    assert advantage_over_baseline(scores, "min25", subjects, BASELINE) > 0.5


def test_the_baseline_has_no_advantage_over_itself() -> None:
    scores, subjects = _cohort({BASELINE: 0.0, "min25": 0.30})
    assert advantage_over_baseline(scores, BASELINE, subjects, BASELINE) == 0.0


def test_selecting_among_pure_noise_inflates_on_the_same_data_but_not_out_of_sample() -> None:
    # Every setting is equivalent to the baseline, so the TRUE advantage is zero. Picking the winner
    # on a set and measuring on that same set must look positive; measuring on held-out subjects must
    # not. This is the property the whole script exists to estimate, and the reason it halves at all.
    settings = {BASELINE: 0.0, "a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
    scores, subjects = _cohort(settings, n=160, seed=3)
    rng = np.random.default_rng(0)
    names = list(settings)
    same, held_out = [], []
    for _ in range(40):
        shuffled = list(subjects)
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        selection, evaluation = sorted(shuffled[:half]), sorted(shuffled[half:])
        picked = names[int(np.argmin(mean_ranks(scores, names, selection)))]
        same.append(advantage_over_baseline(scores, picked, selection, BASELINE))
        held_out.append(advantage_over_baseline(scores, picked, evaluation, BASELINE))
    assert np.mean(same) > 0.02, "selecting on the data must look better than it is"
    assert abs(np.mean(held_out)) < np.mean(same), "held-out estimate must shrink toward the truth"


def test_ranks_average_to_the_middle_when_two_settings_are_identical() -> None:
    scores, subjects = _cohort({BASELINE: 0.0, "clone": 0.0}, seed=7)
    for setting in scores["clone"]:
        scores["clone"][setting] = dict(scores[BASELINE][setting])
    ranks = mean_ranks(scores, [BASELINE, "clone"], subjects)
    assert ranks[0] == pytest.approx(1.5)
    assert ranks[1] == pytest.approx(1.5)
