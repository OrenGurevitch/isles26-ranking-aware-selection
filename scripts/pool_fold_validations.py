"""Pool the five folds' validation predictions into one directory, so the cohort can be scored once.

Each fold predicts only the subjects it held out, so the five together cover every subject exactly
once — that is what makes the pooled set an honest cross-validation estimate rather than five
disconnected numbers. De la Rosa told participants to report cross-validation metrics from the train
set; this is the input to that.

  R=/scratch/orengur2/nnunet_isles
  D=$R/nnUNet_results_1000ep/Dataset510_ISLES1453
  uv run python scripts/pool_fold_validations.py --expect 1453 \\
      --model $D/nnUNetTrainerDiceTopK10Loss_1000epochs__nnUNetPlans__3d_fullres \\
      --out $R/scored/cv5_topk10_1000

Then score it as a single arm — `slurm/nnunet_score.sh cv5_topk10_1000=<out>`. One arm prints medians
and no ranking, which is correct: a rank is a comparison across models and there is nothing here to
compare against.

**The disjointness check is the point of this script.** Symlinking five directories together is a
one-liner; what a one-liner will not tell you is that two folds predicted the same subject, which
would mean the splits are wrong and every pooled number is quietly measuring the wrong thing.
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

SUFFIXES = (".nii.gz", ".npz", ".pkl")


def subject_of(path: Path) -> str:
    name = path.name
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    raise AssertionError(f"unexpected file in a validation directory: {path}")


def pool(*, model: Path, out: Path, folds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict[str, int]:
    seen: dict[str, int] = {}
    collisions: dict[str, list[int]] = defaultdict(list)
    per_fold: dict[str, int] = {}

    for fold in folds:
        validation = model / f"fold_{fold}" / "validation"
        assert validation.is_dir(), f"no validation directory for fold {fold} at {validation}"
        subjects = {subject_of(p) for p in validation.iterdir() if p.name.endswith(SUFFIXES)}
        assert subjects, f"fold {fold} has a validation directory but no predictions in it"
        per_fold[f"fold_{fold}"] = len(subjects)
        for subject in subjects:
            if subject in seen:
                collisions[subject] += [seen[subject], fold]
            seen[subject] = fold

    assert not collisions, (
        f"{len(collisions)} subject(s) predicted by more than one fold, so the splits overlap and a "
        f"pooled score would double-count them: {dict(list(collisions.items())[:5])}"
    )

    out.mkdir(parents=True, exist_ok=True)
    linked = 0
    for fold in folds:
        for source in (model / f"fold_{fold}" / "validation").iterdir():
            if not source.name.endswith(SUFFIXES):
                continue
            destination = out / source.name
            if destination.is_symlink() or destination.exists():
                destination.unlink()
            os.symlink(source, destination)
            linked += 1

    per_fold["total_subjects"] = len(seen)
    per_fold["files_linked"] = linked
    return per_fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True,
                        help="an nnUNet_results .../TRAINER__PLANS__CONFIG directory holding fold_0..4")
    parser.add_argument("--out", type=Path, required=True, help="directory to fill with symlinks")
    parser.add_argument("--expect", type=int, default=None,
                        help="assert this many subjects in total, e.g. 1453")
    args = parser.parse_args()

    counts = pool(model=args.model, out=args.out)
    for key, value in counts.items():
        print(f"  {key:16s} {value}")
    if args.expect is not None:
        assert counts["total_subjects"] == args.expect, (
            f"pooled {counts['total_subjects']} subjects, expected {args.expect} — a fold is missing "
            "or a split changed"
        )
        print(f"  ✅ covers the whole cohort ({args.expect} subjects), each predicted exactly once")


if __name__ == "__main__":
    main()
