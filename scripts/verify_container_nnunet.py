"""Does the submission container's nnU-Net path reproduce nnU-Net's OWN prediction?

The risk in `container/inference_nnunet.py` is silent: a wrong axis order or a wrong probability
channel produces a plausible mask and a worse score, with no crash to notice. `predict_single_npy_array`
says so in its own docstring. So the check is not "does it run" — it is whether the class, called
exactly as the container calls it, lands on the mask nnU-Net already wrote for a validation subject.

  ISLES_NNUNET_DIR=$R/nnUNet_results_250ep/Dataset510_ISLES1453/nnUNetTrainer__nnUNetPlans__3d_fullres \\
  uv run --with nnunetv2 python scripts/verify_container_nnunet.py \\
      --images $R/nnUNet_raw/Dataset510_ISLES1453/imagesTr \\
      --validation $R/.../fold_0/validation --subjects 3

Agreement is not expected to be EXACT. nnU-Net's own validation predictions come from a different
entry point with its own resampling and Gaussian accumulation, and mirroring makes the arithmetic
order-dependent. A wrong transpose is not a near-miss though — it moves Dice to near zero — so the
threshold below separates the two cases with room to spare rather than pretending to bit-equality.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from container.inference_nnunet import MODEL_DIR, NnunetPredictor

AGREEMENT_FLOOR = 0.95


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = int(a.sum()) + int(b.sum())
    return 1.0 if total == 0 else 2.0 * float((a & b).sum()) / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True, help="nnUNet_raw imagesTr")
    parser.add_argument("--validation", type=Path, required=True,
                        help="fold_N/validation, holding nnU-Net's own predictions")
    parser.add_argument("--subjects", type=int, default=3)
    parser.add_argument("--only", nargs="*", default=None,
                        help="subject ids instead of the first --subjects. The per-case time depends "
                             "on volume, so the BUDGET question is answered by naming the biggest "
                             "cases here, not by whichever ones sort first")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device} model={MODEL_DIR}")
    predictor = NnunetPredictor()
    predictor.load(device)

    if args.only:
        references = [args.validation / f"{subject}.nii.gz" for subject in args.only]
        missing = [r for r in references if not r.exists()]
        assert not missing, f"not in this fold's validation set: {[r.name for r in missing]}"
    else:
        references = sorted(args.validation.glob("*.nii.gz"))[: args.subjects]
    assert references, f"no predictions in {args.validation}"

    agreements = []
    for reference_path in references:
        subject = reference_path.name.removesuffix(".nii.gz")
        image_path = args.images / f"{subject}_0000.nii.gz"
        assert image_path.exists(), f"no image for {subject} at {image_path}"

        started = time.perf_counter()
        image = sitk.ReadImage(str(image_path))
        probability = predictor.predict(image=image, image_path=image_path, device=device)
        elapsed = time.perf_counter() - started

        assert probability.shape == tuple(image.GetSize()), (
            f"{subject}: got {probability.shape} for an input of {image.GetSize()}"
        )
        ours = probability >= 0.5
        theirs = sitk.GetArrayFromImage(sitk.ReadImage(str(reference_path))).transpose(2, 1, 0) > 0
        agreement = dice(ours, theirs)
        agreements.append(agreement)
        print(f"[{subject}] agreement={agreement:.4f}  ours={int(ours.sum()):,} vox  "
              f"nnU-Net={int(theirs.sum()):,} vox  {elapsed:.1f}s")

    worst = min(agreements)
    print(f"\n[worst] {worst:.4f} over {len(agreements)} subjects (floor {AGREEMENT_FLOOR})")
    assert worst >= AGREEMENT_FLOOR, (
        f"the container path disagrees with nnU-Net's own prediction (Dice {worst:.4f}). A near-zero "
        f"value means the axis order or the probability channel is wrong — do NOT submit this."
    )
    print("[ok] the container reproduces nnU-Net's own segmentation")


if __name__ == "__main__":
    main()
