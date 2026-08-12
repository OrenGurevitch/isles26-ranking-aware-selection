"""Copy a trained nnU-Net fold into container/resources_nnunet/ so the image can be built offline.

The submission container has no network, so the model is baked in at build time. Run this before
`docker build -f container/Dockerfile.nnunet`.

The shipped recipe is TopK10 @ 1,000 epochs on all 1,453 subjects, so the staged fold is `all`:

  R=/scratch/orengur2/nnunet_isles/nnUNet_results_1000ep/Dataset510_ISLES1453
  uv run python scripts/stage_nnunet_container_weights.py \\
      --model $R/nnUNetTrainerDiceTopK10Loss_1000epochs__nnUNetPlans__3d_fullres --folds all

Stage only the folds the container will USE. `container/inference_nnunet.py` names them in
`FOLDS`, and a fold present on disk but absent from that tuple is dead weight in the image, while the
reverse is a crash at submission time — so the two are checked against each other here.
"""

import argparse
import json
import shutil
from pathlib import Path

RESOURCES = Path(__file__).resolve().parents[1] / "container" / "resources_nnunet"
REQUIRED_FILES = ("plans.json", "dataset.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True,
                        help="an nnUNet_results .../nnUNetTrainer__PLANS__CONFIG directory")
    parser.add_argument("--folds", nargs="+", default=["all"],
                        help="fold numbers, or 'all' for a model trained on every subject. Must match "
                             "container/inference_nnunet.py's FOLDS")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth")
    args = parser.parse_args()

    from container.inference_nnunet import FOLDS

    folds = [f if f == "all" else int(f) for f in args.folds]
    assert tuple(sorted(map(str, folds))) == tuple(sorted(map(str, FOLDS))), (
        f"staging folds {tuple(folds)} but container/inference_nnunet.py loads {FOLDS} — "
        "a fold the container asks for and the image does not carry is a crash at submission time"
    )

    for name in REQUIRED_FILES:
        assert (args.model / name).exists(), f"{args.model} has no {name}"

    if RESOURCES.exists():
        shutil.rmtree(RESOURCES)                       # a stale fold left behind would ship silently
    RESOURCES.mkdir(parents=True)

    for name in REQUIRED_FILES:
        shutil.copy2(args.model / name, RESOURCES / name)

    for fold in folds:
        source = args.model / f"fold_{fold}" / args.checkpoint
        assert source.exists(), f"no {args.checkpoint} for fold {fold} at {source}"
        destination = RESOURCES / f"fold_{fold}"
        destination.mkdir()
        shutil.copy2(source, destination / args.checkpoint)
        print(f"[fold {fold}] {source} ({source.stat().st_size / 1e6:.0f} MB)")

    plans = json.loads((RESOURCES / "plans.json").read_text())
    print(f"[plans] {plans['plans_name']} configurations={sorted(plans['configurations'])}")
    total_mb = sum(f.stat().st_size for f in RESOURCES.rglob("*") if f.is_file()) / 1e6
    print(f"[done] {RESOURCES} ({total_mb:.0f} MB)\n"
          f"  now: docker build --platform=linux/amd64 -f container/Dockerfile.nnunet -t isles26-nnunet .")


if __name__ == "__main__":
    main()
