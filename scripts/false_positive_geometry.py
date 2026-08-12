"""Do our false positives FLOAT, or do they OVER-GROW true lesions? — which remedy even applies.

frozen-wmh ran this and it redirected their whole precision programme: **30% of their predicted
clusters were spurious and only 9.9% sat within 2 voxels of a real lesion**, so their failure was
INVENTING blobs in clean tissue rather than over-growing boundaries. That made centreline Dice the
wrong tool for it — a loss that penalises fragmented structure has no purchase on an isolated blob
with no true structure to be faithful to — and promoted augmentation and a precision-biased loss
instead.

frozen-isles knows its over/under-segmentation COUNTS (`scripts/component_balance.py`: 35% over, 33%
under, median Δ exactly 0 before filtering) but has never measured the SPATIAL relationship. The two
failures want different remedies, and a size filter is currently being applied without knowing which
one we have:

- **Floating** false positives are cheap to remove by size, because invented blobs are small and
  isolated — but removing them does nothing for boundary quality.
- **Over-growth** is not removable by size at all: the spurious voxels are attached to a real lesion,
  so a size filter either keeps them or deletes the lesion with them.

⚠️ **A cluster is "spurious" iff it has ZERO overlap with the reference.** An over-grown lesion is
therefore NOT counted as spurious — it is one matched component that is too big — which is exactly the
distinction this script exists to draw. The two populations are reported separately.

  uv run python scripts/false_positive_geometry.py --references .../references_plain500 \\
      --soft .../soft_plain500 --out .../fp_geometry.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from frozen_isles.evaluate import as_subject_list
from frozen_isles.nifti import load_mask, load_soft
from frozen_isles.postprocess import CONNECTIVITY_26, postprocess

NEAR_VOXELS = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--soft", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-voxels", type=int, default=0,
                        help="0 measures the RAW model. The shipped filter is 25, but the point here "
                             "is what the model does before post-processing hides it")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    references = as_subject_list(sorted(args.references.rglob("*.nii*")), suffix="reference")
    softs = as_subject_list(sorted(args.soft.rglob("*.nii*")), suffix="soft")
    subjects = sorted(set(references) & set(softs))
    assert subjects, "no subjects in common"
    print(f"[data] {len(subjects)} subjects, threshold {args.threshold}, "
          f"min_voxels {args.min_voxels}", flush=True)

    rows, distances, spurious_sizes, matched_sizes = [], [], [], []
    for n, subject in enumerate(subjects, 1):
        truth = load_mask(references[subject]) > 0
        binary = load_soft(softs[subject]) > args.threshold
        if args.min_voxels:
            binary = postprocess(binary, min_voxels=args.min_voxels, closing_radius=0) > 0
        if not truth.any():
            continue                      # distance to "the nearest true lesion" is undefined
        labelled, count = ndimage.label(binary, structure=CONNECTIVITY_26)  # type: ignore[misc]
        if count == 0:
            rows.append({"subject": subject, "clusters": 0, "spurious": 0})
            continue
        # Distance from every voxel to the nearest TRUE lesion voxel, in voxels.
        to_lesion = ndimage.distance_transform_edt(~truth)
        assert isinstance(to_lesion, np.ndarray)
        overlap = ndimage.sum_labels(truth, labelled, index=range(1, count + 1))
        sizes = ndimage.sum_labels(np.ones_like(labelled, dtype=np.float32), labelled,
                                   index=range(1, count + 1))
        nearest = ndimage.minimum(to_lesion, labelled, index=range(1, count + 1))
        spurious = np.asarray(overlap) == 0
        distances.extend(np.asarray(nearest)[spurious].tolist())
        spurious_sizes.extend(np.asarray(sizes)[spurious].tolist())
        matched_sizes.extend(np.asarray(sizes)[~spurious].tolist())
        rows.append({"subject": subject, "clusters": int(count),
                     "spurious": int(spurious.sum())})
        if n % 50 == 0:
            print(f"[progress] {n}/{len(subjects)}", flush=True)

    total = sum(r["clusters"] for r in rows)
    spur = sum(r["spurious"] for r in rows)
    distances = np.array(distances)
    print(f"\n=== predicted clusters over {len(rows)} subjects with a lesion ===")
    print(f"  total predicted clusters : {total}")
    print(f"  SPURIOUS (zero overlap)  : {spur}  ({100 * spur / total:.1f}% of predictions)")
    print(f"  matched                  : {total - spur}")
    if len(distances):
        near = float((distances <= NEAR_VOXELS).mean())
        print("\n=== where the spurious clusters SIT ===")
        print(f"  within {NEAR_VOXELS:.0f} voxels of a true lesion : {100 * near:.1f}%")
        for q in (25, 50, 75, 90):
            print(f"  {q}th percentile distance            : {np.percentile(distances, q):.1f} voxels")
        print(f"\n  median spurious cluster : {np.median(spurious_sizes):.0f} voxels")
        print(f"  median matched cluster  : {np.median(matched_sizes):.0f} voxels")
        verdict = ("FLOAT — invented in clean tissue; a size filter is the applicable remedy"
                   if near < 0.25 else
                   "sit ON or NEAR true lesions; a size filter cannot fix boundary quality")
        print(f"\n**VERDICT: our false positives {verdict}.**")
        print("⚠️ Compare with frozen-wmh's WMH numbers (30% spurious, 9.9% within 2 voxels) only as "
              "a SHAPE. Different lesion, different contrast, different cohort.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "threshold": args.threshold, "min_voxels": args.min_voxels,
            "total_clusters": total, "spurious_clusters": spur,
            "within_2_voxels_fraction": float((distances <= NEAR_VOXELS).mean()) if len(distances) else None,
            "median_spurious_size": float(np.median(spurious_sizes)) if spurious_sizes else None,
            "median_matched_size": float(np.median(matched_sizes)) if matched_sizes else None,
            "per_subject": rows,
        }, indent=2))
        print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
