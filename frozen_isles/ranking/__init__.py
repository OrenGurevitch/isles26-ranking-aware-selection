"""The challenge's own ranking scheme — the objective, which is NOT mean Dice.

From the ISLES'26 design document: "Metrics are calculated for each case, followed by establishing
metric-specific ranks separately for each dataset. A mean rank over all metrics is then obtained to
obtain the team's rank for each case. The final teams' rank is the mean over all case-specific ranks."

So a model is judged by how it places against the others on EVERY case, not by its average score. The
organizers chose this because it is "less influenced by outliers in the case-level metrics" than
aggregate-then-rank.

What that buys, measured in `test.py` rather than assumed: a catastrophic case costs at most ONE rank
position, so a model that collapses on one subject and wins the rest still wins — even where its mean
Dice is the lower of the two. The expensive failure is the opposite shape: losing every case by a hair
ranks last everywhere. **Uniform competitiveness across all cases is worth more than peak quality, and
a small consistent deficit deserves more attention than a rare disaster.** Selecting on mean Dice gets
this backwards in both directions.

Two uses. Against real competitors we cannot rank — their per-case scores are private until the
leaderboard. Between OUR OWN candidate models the same computation is the closest available proxy for
the objective, and it is what model selection should use.
"""

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
from scipy.stats import rankdata

# True = a larger value is better. Straight from the metric definitions in `frozen_isles.metrics`;
# getting a polarity backwards silently inverts the ranking, so it is stated once, here.
METRIC_POLARITY: Mapping[str, bool] = {
    "dice": True,
    "lesion_f1": True,
    "pr_auc": True,
    "abs_volume_difference_ml": False,
    "abs_lesion_count_difference": False,
}


def per_case_ranks(
    *, scores: Mapping[str, npt.NDArray], polarity: Mapping[str, bool] = METRIC_POLARITY
) -> npt.NDArray:
    """Mean-over-metrics rank of each model on each case, shape (n_models, n_cases). Lower is better.

    Each `scores[metric]` is (n_models, n_cases). A NaN takes the worst rank rather than being
    skipped: the design document says missing results on a test case "result in ranks for the
    corresponding metrics to be set to the worst possible value", so a container that dies on one
    subject pays for that subject.

    ⚠️ **That rule is about a MISSING RESULT, and there is one NaN here that is not one.** The design
    document also says an empty ground truth makes AP *undefined*, sets it to NaN, and **excludes it
    from the AP aggregation** — excluded, not scored worst. `metrics.pr_auc` only produces such a NaN
    under `convention="design"`; the default `"shipped"` returns 1.0 or 0.0 and never NaN, so no
    number this project has reported is affected. **But the two conventions exist to be COMPARED, and
    ranking one of them this way compares it wrongly.** Excluding a case per-metric is not expressible
    in this array shape — every metric must cover the same cases for a per-case rank to mean anything
    — so the honest fix is a decision, not an edit here. See `FIXME.md`, the section whose heading
    begins *"C3 ... the `\"design\"` PR-AUC convention returns NaN"*.
    """
    assert scores, "no metrics given"
    missing = set(scores) - set(polarity)
    assert not missing, f"no polarity declared for {sorted(missing)} — a wrong guess inverts the rank"

    shapes = {name: np.asarray(v).shape for name, v in scores.items()}
    assert len(set(shapes.values())) == 1, f"metrics disagree on shape: {shapes}"
    n_models, n_cases = next(iter(shapes.values()))
    assert n_models >= 2, f"ranking needs at least 2 models to compare, got {n_models}"

    stacked = np.empty((len(scores), n_models, n_cases), dtype=float)
    for i, (name, value) in enumerate(scores.items()):
        # Rank ascending on a loss, so "rank 1" is always the best regardless of the metric's polarity.
        loss = -np.asarray(value, dtype=float) if polarity[name] else np.asarray(value, dtype=float)
        # `rankdata` with nan_policy="omit" leaves NaNs as NaN; we then push them to the worst rank.
        ranks = rankdata(loss, method="average", axis=0, nan_policy="omit")
        stacked[i] = np.where(np.isnan(loss), float(n_models), ranks)
    return stacked.mean(axis=0)


def rank_then_aggregate(
    *, scores: Mapping[str, npt.NDArray], polarity: Mapping[str, bool] = METRIC_POLARITY
) -> npt.NDArray:
    """Final rank per model, shape (n_models,) — the mean of its per-case ranks. Lower is better."""
    return per_case_ranks(scores=scores, polarity=polarity).mean(axis=1)


def worst_cases(
    *, scores: Mapping[str, npt.NDArray], model: int, k: int = 10,
    polarity: Mapping[str, bool] = METRIC_POLARITY,
) -> npt.NDArray:
    """Indices of the `k` cases where `model` places worst — where its final rank is actually lost.

    Under rank-then-aggregate every case carries equal weight, so the cheapest improvement is nearly
    always the tail rather than the mean. This says which subjects to open.
    """
    ranks = per_case_ranks(scores=scores, polarity=polarity)[model]
    return np.argsort(-ranks)[:k]
