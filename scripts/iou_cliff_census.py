"""How many subjects sit near the metric's IoU 0.25 cliff — and on which side?

**The cliff is the challenge's own behaviour, not a defect** (`docs/CHALLENGE.md`): `global_bin_dsc`
is computed over MATCHED instances, `NaiveThresholdMatching` matches at IoU ≥ 0.25, and below it Dice
and lesion-F1 both collapse to exactly 0. A case at IoU 0.250 scores 0.4000; the same case at 0.212
scores 0.0000.

That makes the IoU distribution around 0.25 the thing that sizes every remaining lever:

- **Subjects just BELOW** are the prize. Each one pushed over is worth ~0.4 Dice and ~1.0 lesion-F1,
  and under rank-then-aggregate every case weighs the same.
- **Subjects just ABOVE are a RISK nobody has counted.** They are one bad post-processing choice away
  from scoring zero, and a change that helps the median can silently push them off.

**This computes the GLOBAL foreground IoU**, not panoptica's per-instance matching IoU. For a
single-lesion case the two coincide; for a multi-lesion case they do not, so read this as sizing the
population near the cliff rather than as predicting any individual subject's fate.

  uv run python scripts/iou_cliff_census.py --references .../references_plain500 \\
      --soft .../soft_plain500 --out .../iou_census.json
"""

import argparse
import json
from pathlib import Path

from frozen_isles.evaluate import as_subject_list
from frozen_isles.nifti import load_mask, load_soft
from frozen_isles.postprocess import postprocess

MATCHING_THRESHOLD = 0.25
BANDS: tuple[tuple[str, float, float], ...] = (
    ("blank / no overlap", 0.0, 0.001),
    ("0.001-0.10 far below", 0.001, 0.10),
    ("0.10-0.20 below", 0.10, 0.20),
    ("**0.20-0.25 JUST BELOW**", 0.20, 0.25),
    ("**0.25-0.35 JUST ABOVE**", 0.25, 0.35),
    ("0.35-0.50 above", 0.35, 0.50),
    ("0.50+ comfortable", 0.50, 1.01),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--soft", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-voxels", type=int, default=25)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    references = as_subject_list(sorted(args.references.rglob("*.nii*")), suffix="reference")
    softs = as_subject_list(sorted(args.soft.rglob("*.nii*")), suffix="soft")
    subjects = sorted(set(references) & set(softs))
    assert subjects, "no subjects in common"

    rows = []
    for subject in subjects:
        truth = load_mask(references[subject]) > 0
        soft = load_soft(softs[subject])
        binary = soft > args.threshold
        if args.min_voxels:
            binary = postprocess(binary, min_voxels=args.min_voxels, closing_radius=0) > 0
        inter = int((truth & binary).sum())
        union = int((truth | binary).sum())
        rows.append({
            "subject": subject,
            "reference_voxels": int(truth.sum()),
            "predicted_voxels": int(binary.sum()),
            "iou": (inter / union) if union else float("nan"),
        })

    empty_reference = [r for r in rows if r["reference_voxels"] == 0]
    scored = [r for r in rows if r["reference_voxels"] > 0]
    print(f"[data] {len(rows)} subjects — {len(empty_reference)} with an EMPTY reference "
          f"(scored separately; IoU is undefined for them)\n")

    print(f"=== global foreground IoU, threshold {args.threshold}, min_voxels {args.min_voxels} ===")
    for name, low, high in BANDS:
        members = [r for r in scored if low <= r["iou"] < high]
        if not members:
            print(f"    {name:28s}   0")
            continue
        print(f"    {name:28s} {len(members):3d}  ({100 * len(members) / len(scored):4.1f}%)")

    below = [r for r in scored if 0.001 <= r["iou"] < MATCHING_THRESHOLD]
    just_below = [r for r in scored if 0.20 <= r["iou"] < MATCHING_THRESHOLD]
    just_above = [r for r in scored if MATCHING_THRESHOLD <= r["iou"] < 0.35]
    print(f"\n**THE PRIZE**: {len(below)} subjects score ZERO with real overlap; {len(just_below)} of "
          f"them are within 0.05 IoU of clearing the cliff.")
    print(f"**THE RISK** : {len(just_above)} subjects sit within 0.10 IoU ABOVE it — each is one bad "
          f"post-processing choice away from scoring zero.")
    if empty_reference:
        spurious = [r for r in empty_reference if r["predicted_voxels"] > 0]
        print(f"**FREE POINTS**: {len(spurious)} of {len(empty_reference)} empty-reference subjects "
              f"get a non-empty prediction: "
              f"{', '.join(f'{r['subject']} ({r['predicted_voxels']} vox)' for r in spurious)}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"threshold": args.threshold, "min_voxels": args.min_voxels,
                                        "matching_threshold": MATCHING_THRESHOLD,
                                        "per_subject": rows}, indent=2))
        print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
