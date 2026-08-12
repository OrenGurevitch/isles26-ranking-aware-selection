"""How much of the post-processing gain is selection bias? — split-half, no rescoring.

**This rebuilds a headline result that had no code.** `docs/RESULTS.md` reported a split-half estimate
of the component filter's advantage — 0.0425 against a same-data 0.0509 — from an ad-hoc run whose
script and output were never kept. The number is one of the manuscript's three contributions and the
organizers require code matching the manuscript, so it is reconstructed here.

**The construction, recovered by reproducing the published 0.0509 exactly.** It is two rankings, not
one, and using a single five-way ranking gives 0.1072 instead:

1. **Selection** ranks all five `min_voxels` settings against each other and takes the best.
2. **Measurement** ranks only the winner against no filtering, so the advantage is on the [1, 2] scale
   of a two-model comparison.

Selecting and measuring on the same subjects inflates the advantage. Partitioning the cohort into a
selection half and a disjoint evaluation half removes that, and the gap between the two is the bias.

**No rescoring happens here.** `sweep_postprocess_by_band.py` already scored every setting over
every subject and kept the per-subject values, which is what makes this cheap enough to run inline.

**The estimate is of the SELECTION procedure, not of a particular threshold.** A high agreement rate
would mean the winner is stable; a low one means the procedure picks near-ties and the specific
constant should not be reported as tuned.

    uv run python scripts/postproc_selection_bias.py \\
        --scores /scratch/orengur2/nnunet_isles/scored/postproc_by_band_plain500.json \\
        --out /scratch/orengur2/nnunet_isles/scored/postproc_selection_bias_plain500.json
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from frozen_isles.ranking import METRIC_POLARITY, per_case_ranks

BASELINE = "min0_close0_t0.5"
SPLITS = 200
SEED = 0

PerSubject = Mapping[str, Mapping[str, Mapping[str, float]]]


def mean_ranks(scores: PerSubject, settings: Sequence[str], subjects: Sequence[str]) -> np.ndarray:
    stacked = {
        metric: np.array([[scores[s][subject][metric] for subject in subjects] for s in settings])
        for metric in METRIC_POLARITY
    }
    return per_case_ranks(scores=stacked).mean(axis=1)


def advantage_over_baseline(
    scores: PerSubject, winner: str, subjects: Sequence[str], baseline: str
) -> float:
    """Rank advantage of `winner` over `baseline`, as a TWO-model comparison (lower rank is better)."""
    if winner == baseline:
        return 0.0
    ranks = mean_ranks(scores, [baseline, winner], subjects)
    return float(ranks[0] - ranks[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scores", type=Path, required=True,
                        help="JSON with a per_subject mapping of {setting: {subject: {metric: value}}}")
    parser.add_argument("--baseline", default=BASELINE, help="the no-post-processing setting")
    parser.add_argument("--splits", type=int, default=SPLITS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    raw = json.loads(args.scores.read_text())
    scores: PerSubject = raw.get("per_subject", raw)
    settings = list(scores)
    assert args.baseline in settings, f"baseline {args.baseline} absent; have {settings}"

    subject_sets = {s: set(scores[s]) for s in settings}
    shared = set.intersection(*subject_sets.values())
    assert all(len(v) == len(shared) for v in subject_sets.values()), (
        "settings cover different subjects; a rank over a ragged cohort is not defined"
    )
    subjects = sorted(shared)
    assert len(subjects) >= 4, f"need at least 4 subjects to halve twice, got {len(subjects)}"

    full = mean_ranks(scores, settings, subjects)
    winner = settings[int(np.argmin(full))]
    biased = advantage_over_baseline(scores, winner, subjects, args.baseline)

    print(f"=== {len(subjects)} subjects, {len(settings)} settings, baseline {args.baseline} ===")
    for setting, value in sorted(zip(settings, full, strict=True), key=lambda pair: pair[1]):
        print(f"  {setting:24} {value:.4f}{'   <- selected' if setting == winner else ''}")
    print(f"\nselected and measured on ALL subjects : {biased:+.4f}")

    rng = np.random.default_rng(args.seed)
    honest, agreements = [], 0
    for _ in range(args.splits):
        shuffled = list(subjects)
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        selection, evaluation = sorted(shuffled[:half]), sorted(shuffled[half:])
        picked = settings[int(np.argmin(mean_ranks(scores, settings, selection)))]
        agreements += picked == winner
        honest.append(advantage_over_baseline(scores, picked, evaluation, args.baseline))

    honest_array = np.array(honest)
    low, high = np.percentile(honest_array, [2.5, 97.5])
    mean = float(honest_array.mean())
    penalty = biased - mean
    agreement = agreements / args.splits

    print(f"select on half, measure on the OTHER : {mean:+.4f}  "
          f"2.5-97.5 pct across splits [{low:+.4f}, {high:+.4f}]")
    print("   ⚠️ that interval is the SPREAD OF SINGLE-SPLIT ESTIMATES, each from half the cohort --")
    print("      not a confidence interval. The splits REUSE the same subjects, so they are not")
    print("      independent samples and the spread cannot be divided down by sqrt(splits); a real")
    print("      interval needs a subject-level bootstrap wrapped around this whole procedure.")
    if biased:
        print(f"selection penalty                    : {penalty:+.4f} "
              f"({100 * penalty / biased:.0f}% of the apparent gain)")
    else:
        print("selection penalty                    : undefined, the same-data advantage is exactly 0")
    print(f"\nthe half-selected winner matched the full-cohort winner in {100 * agreement:.0f}% "
          f"of {args.splits} splits")
    if agreement < 0.7:
        print("  ⚠️ below 0.7: the procedure is picking between near-ties, so report "
              "'a small filter helps' rather than presenting the constant as tuned")

    if args.out:
        args.out.write_text(json.dumps({
            "baseline": args.baseline, "settings": settings, "n_subjects": len(subjects),
            "splits": args.splits, "seed": args.seed,
            "full_cohort_rank": dict(zip(settings, full.tolist(), strict=True)),
            "winner": winner,
            "advantage_same_data": biased,
            "advantage_split_half": mean,
            "advantage_split_half_spread_2p5_97p5": [float(low), float(high)],
            "selection_penalty": penalty,
            "winner_agreement_rate": agreement,
        }, indent=1))
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
