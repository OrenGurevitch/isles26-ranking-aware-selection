"""Score one or more models against reference masks, and rank them the way the challenge will.

  uv run python scripts/score_cohort.py --references REF_DIR --predictions NAME=DIR [NAME=DIR ...]

Each directory holds BIDS-named volumes; subjects are matched by their `sub-XXXX` prefix, never by
position. With two or more models it also prints the rank-then-aggregate standing and each model's
worst cases, because that is where a rank is actually lost.

This is a FILE with a `__main__` guard on purpose: panoptica spawns multiprocessing workers, and
running the same code from `python -c` or a heredoc makes every child re-import a `__main__` it cannot
find — a process storm that ran over ten minutes on one subject before being killed.
"""

import argparse
import json
from pathlib import Path

from frozen_isles.evaluate import (
    as_subject_list,
    rank,
    score_cohort,
    summarise,
    worst_subjects,
)


def _volumes(directory: Path, suffix: str) -> dict[str, Path]:
    paths = sorted(p for p in directory.rglob("*.nii*") if p.is_file())
    assert paths, f"no NIfTI volumes under {directory}"
    return as_subject_list(paths, suffix=suffix)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--references", type=Path, required=True, help="directory of reference masks")
    parser.add_argument("--predictions", nargs="+", required=True, metavar="NAME=DIR",
                        help="one or more NAME=DIR pairs of predicted masks")
    parser.add_argument("--soft", nargs="*", default=[], metavar="NAME=DIR",
                        help="optional NAME=DIR pairs of probability maps; PR-AUC needs these")
    parser.add_argument("--min-voxels", type=int, default=0,
                        help="drop predicted components smaller than this. 0 scores the model's RAW "
                             "output; the submitted container ships 25, so leaving this at 0 reports a "
                             "different system from the one submitted")
    parser.add_argument("--closing-radius", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="write per-subject scores as JSON")
    args = parser.parse_args()

    references = _volumes(args.references, "reference")
    print(f"[refs] {len(references)} subjects from {args.references}", flush=True)
    shipped = "RAW model output" if not (args.min_voxels or args.closing_radius) else (
        f"min_voxels={args.min_voxels}, closing_radius={args.closing_radius}")
    print(f"[config] scoring the {shipped} — the container ships min_voxels=25", flush=True)

    soft_dirs = dict(pair.split("=", 1) for pair in args.soft)
    scores_by_model = {}
    for pair in args.predictions:
        name, directory = pair.split("=", 1)
        # flush=True on every progress line. panoptica takes ~8 s/subject, so an arm is ~40 min of
        # silence, and python block-buffers stdout when it is redirected to a file — which made a
        # healthy run indistinguishable from a hang for 52 minutes (job 207694). Announcing the arm
        # BEFORE scoring it is the point: a line only at the end tells you nothing while you wait.
        minutes = len(references) * 8 // 60
        print(f"\n[{name}] scoring {len(references)} subjects (~8 s each, so ~{minutes} min)",
              flush=True)
        soft = _volumes(Path(soft_dirs[name]), "soft") if name in soft_dirs else None
        if soft is None:
            print(f"[warn] {name}: no probability maps given — PR-AUC is computed from the binary "
                  f"mask and is NOT comparable to a real soft-map score", flush=True)
        scores_by_model[name] = score_cohort(
            references=references, predictions=_volumes(Path(directory), "prediction"),
            soft_predictions=soft,
            min_voxels=args.min_voxels, closing_radius=args.closing_radius,
        )
        medians = summarise(scores_by_model[name])
        print(f"[{name}] medians over {len(scores_by_model[name])} subjects", flush=True)
        for metric, value in medians.items():
            print(f"    {metric:32s} {value:.4f}", flush=True)

    if len(scores_by_model) >= 2:
        print("\n=== rank-then-aggregate (lower is better; this is the objective, not mean Dice) ===",
              flush=True)
        for name, value in sorted(rank(scores_by_model).items(), key=lambda kv: kv[1]):
            print(f"    {name:24s} {value:.4f}")
            worst = worst_subjects(scores_by_model, model=name, k=5)
            print(f"        worst cases: {', '.join(worst)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {name: {s: vars(score) for s, score in scores.items()}
             for name, scores in scores_by_model.items()}, indent=2))
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
