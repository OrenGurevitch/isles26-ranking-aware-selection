"""What does an UNSEEN acquisition site cost? — the challenge's stated purpose, measured per centre.

The hidden test *"contains data from completely new sites absent from training"* (`docs/CHALLENGE.md`),
and every other number this project holds comes from a split stratified BY CENTRE, where each held-out
subject shares its centre with the training set. This is the one measurement that removes that.

⚠️ **DO NOT read LOCO fold N against stratified fold 0.** Greedy largest-first packing put each big
centre in a different LOCO fold, and those centres differ enormously in intrinsic difficulty — R009
scores 0.369 with its centre IN training, R047 scores 0.860. A fold-level difference therefore
conflates the generalization penalty with a change in cohort composition, and it would have looked
like a clean derivation.

**The comparison here is PER CENTRE and PAIRED ON SUBJECTS.** For a centre C, take the subjects scored
under BOTH models — that is `stratified fold 0 ∩ C`, which is a subset of C's LOCO held-out set — and
compare the two scores subject by subject. Same subjects, same budget, same plans; the single
difference is whether C's centre appeared in training. That is what
`frozen_isles.stats.paired_delta_by_subject` tests.

  uv run python scripts/loco_penalty.py \\
      --stratified $R/scored/scores_plain100.json \\
      --loco $R/scored/scores_loco_f0.json $R/scored/scores_loco_f1.json ... \\
      --inventory $DATA/inventory.csv

⚠️ Each `--loco` file may contribute SEVERAL centres, and a centre is only usable when enough of its
subjects were scored under both models. `--min-paired` drops the rest rather than reporting a Δ over
three subjects as though it were an estimate.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from frozen_isles.metrics import SubjectScore
from frozen_isles.stats import paired_delta_by_subject

MIN_PAIRED = 10


def load_scores(path: Path) -> dict[str, dict[str, SubjectScore]]:
    raw = json.loads(path.read_text())
    return {model: {subject: SubjectScore(**values) for subject, values in subjects.items()}
            for model, subjects in raw.items()}


def single_model(path: Path, name: str | None = None) -> dict[str, SubjectScore]:
    """One model's per-subject scores, named explicitly whenever the file holds more than one.

    Score files are written per SCORING RUN, so `scores_plain100-plain250.json` carries two budgets.
    Picking one of them implicitly would mean a Δ could silently compare against the wrong budget --
    the confound this whole script exists to avoid -- so an ambiguous file is an error.
    """
    models = load_scores(path)
    if name is not None:
        assert name in models, f"{path} holds {sorted(models)}, not {name!r}"
        return models[name]
    assert len(models) == 1, (
        f"{path} holds {sorted(models)} — name one with --stratified-model, because picking a budget "
        f"implicitly is exactly the confound this comparison exists to avoid"
    )
    return next(iter(models.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stratified", type=Path, required=True,
                        help="scores for the model that HAD every centre in training")
    parser.add_argument("--loco", type=Path, nargs="+", required=True,
                        help="scores for each leave-centres-out fold")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--stratified-model", default=None,
                        help="which model in --stratified, when that file holds several. ⚠️ It must "
                             "be the SAME budget and plans as the LOCO folds, or the Δ measures the "
                             "budget rather than the unseen site")
    parser.add_argument("--metric", default="dice")
    parser.add_argument("--min-paired", type=int, default=MIN_PAIRED)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    centre_of = {row["subject_id"]: row["center"]
                 for row in csv.DictReader(args.inventory.open())}
    seen = single_model(args.stratified, args.stratified_model)

    rows = []
    for loco_path in args.loco:
        unseen = single_model(loco_path)
        shared = sorted(set(seen) & set(unseen))
        assert shared, f"{loco_path} shares no subjects with {args.stratified}"

        by_centre: dict[str, list[str]] = {}
        for subject in shared:
            by_centre.setdefault(centre_of[subject], []).append(subject)

        for centre, subjects in sorted(by_centre.items()):
            if len(subjects) < args.min_paired:
                continue
            # a − b, so a NEGATIVE delta is the cost of not having seen the centre.
            delta = paired_delta_by_subject(unseen, seen, args.metric, subjects=subjects)
            rows.append({
                "centre": centre,
                "fold": loco_path.stem,
                "n_paired": len(subjects),
                "delta": delta.mean,
                "lo": delta.lo,
                "hi": delta.hi,
                "excludes_zero": delta.excludes_zero(),
            })

    assert rows, (
        f"no centre had {args.min_paired}+ subjects scored under both models — either the folds do "
        f"not overlap the stratified held-out set, or --min-paired is too high"
    )

    print(f"\n=== unseen-site penalty on {args.metric}, per centre "
          f"(negative = worse when the centre was NOT in training) ===")
    print(f"{'centre':10s} {'fold':22s} {'n':>4s}  {'Δ':>8s}  {'95% CI':>20s}")
    for row in sorted(rows, key=lambda r: r["delta"]):
        flag = " *" if row["excludes_zero"] else ""
        print(f"{row['centre']:10s} {row['fold']:22s} {row['n_paired']:4d}  {row['delta']:+8.4f}  "
              f"[{row['lo']:+.4f}, {row['hi']:+.4f}]{flag}")
    print("\n* = the 95% CI excludes 0. A centre without a star is not evidence of no penalty; with "
          "these sample sizes it is most often an interval too wide to say either way.")

    # The pooled view: every subject in the cohort, each scored by the one model that never saw its
    # centre. Five folds partition all 1,453 subjects, so this is the whole dataset under
    # unseen-site conditions -- the closest thing we have to a forecast of the hidden test.
    pooled = [getattr(score, args.metric)
              for path in args.loco
              for score in single_model(path).values()]
    finite = [v for v in pooled if v is not None and np.isfinite(v)]
    print(f"\n=== pooled over all {len(finite)} subjects, each scored by the model that never saw "
          f"its centre ===")
    print(f"    median {args.metric}  {float(np.median(finite)):.4f}")
    print(f"    mean   {args.metric}  {float(np.mean(finite)):.4f}")
    print("    ⚠️ Descriptive, NOT a paired test: this is a different subject set from any single "
          "fold-0 number, so it must not be differenced against one. The per-centre table above is "
          "the comparison.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"metric": args.metric, "centres": rows}, indent=2))
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
