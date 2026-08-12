"""Does an arm's lead in rank-then-aggregate survive resampling the subjects?

`slurm/nnunet_score.sh` prints a rank ordering. An ordering says who is ahead; it never says whether
being ahead means anything. Under rank-then-aggregate the whole spread across arms is bounded by the
number of arms — with three, every final rank sits in [1, 3] and a 0.23 gap has no intuitive scale —
so the point estimate alone cannot be read.

This resamples the SUBJECTS with replacement, recomputing the full per-case ranking inside each draw.
Pairing is what makes it informative: every arm is re-ranked on the same resampled cohort, so the
shared per-subject difficulty cancels and what remains is the arms disagreeing.

⚠️ **Ranks are recomputed per draw, never averaged from the full-cohort ranking.** Rank-then-aggregate
is not a per-subject statistic that can be resampled directly — a case's rank depends on which arms it
is scored against, so the ranking must be rebuilt inside the draw or the interval is meaningless.

⚠️ A rank here is only a rank AMONG THESE ARMS, on this fold. It says nothing about the leaderboard.

  uv run python scripts/bootstrap_arm_ranking.py scores_topk10_250-da5_250-plain250.json \\
      --against plain250
"""

import argparse
import json
from pathlib import Path

import numpy as np

from frozen_isles.ranking import METRIC_POLARITY, rank_then_aggregate

DRAWS = 2000
RANDOM_SEED = 0
CONFIDENCE = 0.95


def ranking_of(scores: dict[str, dict[str, dict[str, float]]], arms: list[str],
               subjects: list[str]) -> np.ndarray:
    """Final rank per arm over exactly these subjects, in `arms` order."""
    stacked = {
        metric: np.array([[scores[arm][subject][metric] for subject in subjects] for arm in arms])
        for metric in METRIC_POLARITY
    }
    return rank_then_aggregate(scores=stacked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scores", type=Path, help="scores_<arm>-<arm>....json from nnunet_score.sh")
    parser.add_argument("--against", required=True,
                        help="the reference arm every other arm is differenced against")
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    scores = json.loads(args.scores.read_text())
    arms = sorted(scores)
    assert args.against in arms, f"no arm {args.against!r} — have {arms}"
    assert len(arms) >= 2, "ranking needs at least two arms"

    # Every arm must cover the same subjects or the per-case ranks compare different cohorts.
    cohorts = {arm: set(scores[arm]) for arm in arms}
    shared = set.intersection(*cohorts.values())
    for arm in arms:
        assert cohorts[arm] == shared, (
            f"{arm} covers {len(cohorts[arm])} subjects against {len(shared)} shared — these arms "
            f"cannot be ranked against each other"
        )
    subjects = sorted(shared)
    print(f"[data] {len(arms)} arms over {len(subjects)} shared subjects: {', '.join(arms)}")

    observed = ranking_of(scores, arms, subjects)
    reference = arms.index(args.against)
    print("\n=== observed rank-then-aggregate (lower is better) ===")
    for arm, rank in sorted(zip(arms, observed, strict=True), key=lambda pair: pair[1]):
        print(f"  {arm:24s} {rank:.4f}")

    generator = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(len(subjects))
    draws = np.empty((args.draws, len(arms)))
    for draw in range(args.draws):
        resampled = [subjects[i] for i in generator.choice(indices, indices.size, replace=True)]
        draws[draw] = ranking_of(scores, arms, resampled)
        if (draw + 1) % 500 == 0:
            print(f"[bootstrap] {draw + 1}/{args.draws}", flush=True)

    tail = (1 - CONFIDENCE) / 2 * 100
    print(f"\n=== Δ against {args.against} — NEGATIVE means better ({args.draws} paired draws) ===")
    print(f"  {'arm':24s} {'Δ rank':>9s} {'95% CI':>22s}  {'P(better)':>10s}")
    summary = {}
    for position, arm in enumerate(arms):
        if position == reference:
            continue
        delta = draws[:, position] - draws[:, reference]
        low, high = float(np.percentile(delta, tail)), float(np.percentile(delta, 100 - tail))
        better = float((delta < 0).mean())
        point = float(observed[position] - observed[reference])
        summary[arm] = {"delta": point, "ci95": [low, high], "p_better": better}
        verdict = "" if low < 0 < high else "  <- interval excludes 0"
        print(f"  {arm:24s} {point:+9.4f}   [{low:+.4f}, {high:+.4f}]  {better:10.3f}{verdict}")

    print("\n⚠️ An interval containing 0 means this fold cannot separate the arms, however clean the "
          "ordering looks. `P(better)` is the fraction of draws in which the arm beat the reference.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "arms": arms, "reference": args.against, "n_subjects": len(subjects),
            "draws": args.draws, "seed": RANDOM_SEED,
            "observed": dict(zip(arms, observed.tolist(), strict=True)),
            "delta_vs_reference": summary,
        }, indent=2))
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
