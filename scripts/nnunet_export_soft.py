"""Turn nnU-Net's validation `.npz` softmax into NIfTI probability maps our scorer can read.

PR-AUC is one of the five ranked metrics and it is defined over the SOFT map. `score_cohort` falls
back to the binary mask when no probabilities are given, which caps PR-AUC at what a thresholded map
can reach — and that would understate nnU-Net in OUR favour, which is a worse comparison than none.

**The axis order is reversed and this script proves its own transpose.** nnU-Net stores
`probabilities` as `(class, z, y, x)` in its internal order while the exported NIfTI is `(x, y, z)`,
so the foreground channel needs `transpose(2, 1, 0)`. Rather than trust that, every case asserts that
thresholding the transposed map at 0.5 reproduces nnU-Net's OWN saved mask — a wrong transpose on a
near-cubic volume would otherwise pass a shape check and silently score a scrambled prediction.

Also writes a `references/` directory of symlinks for exactly the subjects present, because
`score_cohort` requires the reference set and the prediction set to match exactly.

  uv run python scripts/nnunet_export_soft.py \\
      --validation .../fold_0/validation --tree .../ATLAS3_Training_Raw --out .../soft_10ep
"""

import argparse
from pathlib import Path

MASK_SUFFIX = "_space-orig_label-lesion_desc-T1lesion_mask.nii.gz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validation", type=Path, required=True,
                        help="nnU-Net fold_N/validation directory (.npz + .nii.gz per case)")
    parser.add_argument("--tree", type=Path, required=True, help="ATLAS tree, for reference masks")
    parser.add_argument("--out", type=Path, required=True, help="where probability NIfTIs go")
    parser.add_argument("--references", type=Path, default=None,
                        help="also build a symlink directory of matching reference masks")
    args = parser.parse_args()

    import nibabel as nib
    import numpy as np

    args.out.mkdir(parents=True, exist_ok=True)
    if args.references:
        args.references.mkdir(parents=True, exist_ok=True)

    archives = sorted(args.validation.glob("*.npz"))
    assert archives, f"no .npz under {args.validation} — was --npz passed to nnUNetv2_train?"
    print(f"[in] {len(archives)} cases from {args.validation}", flush=True)

    for index, archive in enumerate(archives, start=1):
        subject_id = archive.stem
        exported = args.validation / f"{subject_id}.nii.gz"
        assert exported.is_file(), f"{subject_id}: .npz without its exported mask"
        mask_image = nib.load(str(exported))
        assert isinstance(mask_image, nib.Nifti1Image)

        foreground = np.load(archive)["probabilities"][1].transpose(2, 1, 0)
        assert foreground.shape == mask_image.shape, (
            f"{subject_id}: probabilities {foreground.shape} vs exported mask {mask_image.shape}"
        )
        # The transpose is PROVEN here, not assumed: nnU-Net's own mask is argmax over two classes,
        # which is exactly `foreground > 0.5`.
        agreement = float(((foreground > 0.5) == (np.asanyarray(mask_image.dataobj) > 0.5)).mean())
        assert agreement > 0.9999, (
            f"{subject_id}: thresholded probabilities match nnU-Net's own mask on only "
            f"{agreement:.6f} of voxels — the axis order is wrong, and a shape check cannot see it"
        )
        nib.save(nib.Nifti1Image(foreground.astype(np.float32), mask_image.affine),
                 str(args.out / f"{subject_id}.nii.gz"))

        if args.references:
            found = sorted(args.tree.glob(f"*/{subject_id}/ses-*/anat/*{MASK_SUFFIX}"))
            assert len(found) == 1, f"{subject_id}: {len(found)} reference masks found"
            link = args.references / f"{subject_id}{MASK_SUFFIX}"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(found[0].resolve(strict=True))

        if index % 50 == 0 or index == len(archives):
            print(f"[export] {index}/{len(archives)}", flush=True)

    print(f"[out] {args.out}" + (f"\n[out] {args.references}" if args.references else ""), flush=True)


if __name__ == "__main__":
    main()
