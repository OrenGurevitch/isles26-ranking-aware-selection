"""Is our lesion-F1 held down by INVENTED lesions or MISSED ones? — they want opposite remedies.

lesion-F1 sits at 0.5000 for every nnU-Net arm we have scored, and it is the harmonic mean of instance
precision and recall, so that one number hides which side is binding. The two failures point opposite
ways:

  precision-limited  ->  the model invents lesions   ->  a STRICTER size filter, a HIGHER threshold
  recall-limited     ->  the model misses lesions    ->  a LOWER threshold, a hard-example loss

We already know 33% of predicted clusters are spurious and that they FLOAT rather than over-grow
(`scripts/false_positive_geometry.py`), which predicts precision-limited. This measures it instead of
inferring it, at the same IoU 0.25 matching the challenge uses, so the answer is in the metric's own
terms rather than in a voxel-overlap proxy.

Reports both aggregations, because they answer different questions:

  MICRO   pool TP/FP/FN over the cohort  ->  where the lesions are, so large-lesion subjects dominate
  MACRO   median of per-subject values   ->  what a typical subject looks like, which is what
                                             rank-then-aggregate actually rewards

⚠️ Post-processing is applied first, so this measures the SHIPPED predictions rather than raw output —
`min_voxels=25` already removes small invented clusters, and measuring before it would overstate the
precision problem it exists to fix.

⚠️ Panoptica costs ~8 s/subject, so this is CPU work: ~40 min per arm over 291 subjects.

  uv run python scripts/lesion_precision_recall.py --references .../references_plain500 \\
      --soft .../soft_plain500 --out .../lesion_pr_plain500.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from frozen_isles.evaluate import as_subject_list
from frozen_isles.metrics import instance_counts
from frozen_isles.nifti import load_mask, load_soft
from frozen_isles.postprocess import postprocess

# The shipped post-processing, so this describes the predictions the challenge would actually score.
THRESHOLD = 0.5
MIN_VOXELS = 25
SIZE_BANDS: tuple[tuple[str, int, int], ...] = (
    ("tiny   <1k vox", 1, 1_000),
    ("small  1k-5k", 1_000, 5_000),
    ("medium 5k-20k", 5_000, 20_000),
    ("large  20k+", 20_000, 10**9),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--soft", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--min-voxels", type=int, default=MIN_VOXELS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    references = as_subject_list(sorted(args.references.rglob("*.nii*")), suffix="reference")
    softs = as_subject_list(sorted(args.soft.rglob("*.nii*")), suffix="soft")
    subjects = sorted(set(references) & set(softs))
    assert subjects, "no subjects in common"
    print(f"[data] {len(subjects)} subjects, threshold {args.threshold}, "
          f"min_voxels {args.min_voxels}", flush=True)

    rows = []
    for n, subject in enumerate(subjects, 1):
        truth = load_mask(references[subject]) > 0
        binary = load_soft(softs[subject]) > args.threshold
        if args.min_voxels:
            binary = postprocess(binary, min_voxels=args.min_voxels, closing_radius=0) > 0
        counts = instance_counts(reference=truth, prediction=binary)
        rows.append({
            "subject": subject,
            "reference_voxels": int(truth.sum()),
            "true_positives": counts.true_positives,
            "false_positives": counts.false_positives,
            "false_negatives": counts.false_negatives,
            "precision": counts.precision,
            "recall": counts.recall,
        })
        if n % 25 == 0:
            print(f"[progress] {n}/{len(subjects)}", flush=True)

    # MICRO: pooled over the cohort, so subjects carrying many lesions dominate.
    tp = sum(r["true_positives"] for r in rows)
    fp = sum(r["false_positives"] for r in rows)
    fn = sum(r["false_negatives"] for r in rows)
    micro_precision = tp / (tp + fp) if tp + fp else float("nan")
    micro_recall = tp / (tp + fn) if tp + fn else float("nan")

    print(f"\n=== MICRO — pooled over {len(rows)} subjects ===")
    print(f"  matched lesions (TP)   : {tp:5d}")
    print(f"  invented lesions (FP)  : {fp:5d}")
    print(f"  missed lesions (FN)    : {fn:5d}")
    print(f"  precision              : {micro_precision:.4f}")
    print(f"  recall                 : {micro_recall:.4f}")

    # MACRO: every subject counts once, which is what rank-then-aggregate rewards.
    scored = [r for r in rows if r["reference_voxels"] > 0]
    macro_precision = float(np.median([r["precision"] for r in scored]))
    macro_recall = float(np.median([r["recall"] for r in scored]))
    print(f"\n=== MACRO — median over the {len(scored)} subjects with a lesion ===")
    print(f"  precision              : {macro_precision:.4f}")
    print(f"  recall                 : {macro_recall:.4f}")

    binding = ("PRECISION — the model INVENTS lesions; a stricter filter or higher threshold is the "
               "lever" if micro_precision < micro_recall else
               "RECALL — the model MISSES lesions; a lower threshold or a hard-example loss is the "
               "lever")
    print(f"\n**BINDING CONSTRAINT: {binding}.**")
    print(f"   micro precision {micro_precision:.4f} against recall {micro_recall:.4f}; "
          f"the smaller one is what lesion-F1 is paying for.")

    print("\n=== by reference lesion size — where each failure lives ===")
    print(f"  {'band':16s} {'n':>4s} {'TP':>6s} {'FP':>6s} {'FN':>6s} {'prec':>7s} {'recall':>7s}")
    bands = {}
    for name, low, high in SIZE_BANDS:
        members = [r for r in scored if low <= r["reference_voxels"] < high]
        if not members:
            print(f"  {name:16s} {0:4d}")
            continue
        b_tp = sum(r["true_positives"] for r in members)
        b_fp = sum(r["false_positives"] for r in members)
        b_fn = sum(r["false_negatives"] for r in members)
        b_precision = b_tp / (b_tp + b_fp) if b_tp + b_fp else float("nan")
        b_recall = b_tp / (b_tp + b_fn) if b_tp + b_fn else float("nan")
        bands[name] = {"n": len(members), "tp": b_tp, "fp": b_fp, "fn": b_fn,
                       "precision": b_precision, "recall": b_recall}
        print(f"  {name:16s} {len(members):4d} {b_tp:6d} {b_fp:6d} {b_fn:6d} "
              f"{b_precision:7.4f} {b_recall:7.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "threshold": args.threshold, "min_voxels": args.min_voxels,
            "micro": {"tp": tp, "fp": fp, "fn": fn,
                      "precision": micro_precision, "recall": micro_recall},
            "macro": {"precision": macro_precision, "recall": macro_recall},
            "by_size_band": bands, "per_subject": rows,
        }, indent=2))
        print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
