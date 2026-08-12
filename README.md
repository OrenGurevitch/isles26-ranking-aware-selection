# Ranking-aware model selection for stroke lesion segmentation

Our entry to [ISLES'26](https://isles-26.grand-challenge.org/) (MICCAI 2026): ischemic stroke lesion
segmentation from native-space T1-weighted MRI, on ATLAS v3.0, N = 1,453.

**Every design choice here was made against the challenge's own ranking rule rather than against
Dice**, and the two disagree often enough to change what gets submitted. The loss we ship scores a
*lower* median Dice than the arm it beats. A component filter that Dice alone would have discarded is
kept. On the binarisation threshold the two criteria prefer opposite directions.

## The method

A self-configuring [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) v2, `3d_fullres`, trained on our own
stratified folds. The submitted configuration is **1,000 epochs with a loss that adds Dice to the
cross-entropy of the hardest 10% of voxels** (`nnUNetTrainerDiceTopK10Loss_1000epochs`), followed by
removal of predicted components below 25 voxels under 26-connectivity. No external data.

Five-fold cross-validation over all 1,453 training subjects, each predicted by the fold that held it
out, at the shipped post-processing:

| Dice | lesion-wise F1 | PR-AUC | \|Δvolume\| | \|Δcount\| |
|---|---|---|---|---|
| 0.7672 | 0.6667 | 0.8833 | 1.39 mL | 1.0 |

All values are medians, matching the challenge evaluation. **No number here is from the hidden test
set.**

## Why the ranking rule matters

ISLES'26 scores each case on five metrics, ranks submissions per case per metric, then averages. Dice
carries one fifth of the criterion. `frozen_isles/ranking` implements that rule, and it is what model
selection uses.

`frozen_isles/metrics` does not reimplement the five metrics. It calls `panoptica` and `scikit-learn`
with the organizers' own configuration, and `frozen_isles/metrics/test.py` asserts numerical identity
against a verbatim copy of their evaluation code kept as `references/isles26_eval_utils.py`.
`panoptica` is pinned to
`1.0.1`, the version in their `requirements.txt`.

## Layout

| | |
|---|---|
| `frozen_isles/metrics` | the five ranked metrics, via the organizers' own libraries |
| `frozen_isles/ranking` | rank-then-aggregate: the objective, not a proxy for it |
| `frozen_isles/stats` | paired subject-level bootstrap behind every interval reported |
| `frozen_isles/evaluate` | scoring a cohort and comparing models |
| `frozen_isles/splits` | folds stratified by centre × lesion size × chronicity |
| `frozen_isles/data` | reading the ATLAS tree and its per-subject metadata |
| `frozen_isles/nifti` | native-space geometry, and masks whose dtypes disagree |
| `frozen_isles/postprocess` | the minimum-component-size filter, applied after thresholding |
| `frozen_isles/gcio` | the Grand Challenge container boundary |
| `container/` | the submission entrypoint and its Dockerfile |
| `scripts/`, `slurm/` | the analyses behind the reported numbers, and the cluster jobs that ran them |

Two naming notes. The Python package is `frozen_isles` because this work began as a frozen-features
experiment; that arm was tested, lost to nnU-Net, and is not part of what is submitted or described
here. `container/probe.py` is a model-free instrument used to check the platform's input and output
contract, kept because `frozen_isles/gcio/test.py` exercises it end to end.

⚠️ The `slurm/` scripts carry the account, paths and walltimes of the cluster they ran on. They are
included as the record of what was actually run, not as a portable pipeline; expect to edit every
`#SBATCH` line. Some source comments cite an internal project record that is not part of this
repository, and are kept because they say where a number came from.

## Reproducing

```bash
uv sync
uv run just check          # lint, typecheck, tests
uv run just verify-metrics # differential test against the organizers' shipped code
```

Training and scoring assume a SLURM cluster and the ISLES'26 release, which is not redistributable;
`slurm/` carries the job scripts with their resource requests. The pipeline is: preprocess
(`slurm/nnunet_prepare.sh`) → train (`slurm/nnunet_train.sh`) → export soft maps and score
(`slurm/nnunet_score.sh`) → pool the folds (`scripts/pool_fold_validations.py`) → the error analyses
(`slurm/analyse_arm.sh`).

⚠️ **`--seed` chooses the FOLD SPLIT**, via `stratified_folds`, not just initialisation. Seed 0 is the
split every reported number uses.

## Building the submission container

🔴 **The trained weights are not in this repository** — a 250 MB checkpoint does not belong in git.
They are attached to this repository's latest Release as `isles26-nnunet-weights.zip`, which unpacks
to exactly the layout the build expects:

```bash
unzip isles26-nnunet-weights.zip -d container/resources_nnunet/
docker build --platform=linux/amd64 -f container/Dockerfile.nnunet -t isles26-nnunet .
```

`docker build` fails without them, by design: the Dockerfile copies the model in, because the container
has no network at inference time.

To rebuild the weights instead of downloading them, train with `slurm/nnunet_train.sh` and then stage
the result:

```bash
uv run python scripts/stage_nnunet_container_weights.py \
    --model <nnUNet_results>/nnUNetTrainerDiceTopK10Loss_1000epochs__nnUNetPlans__3d_fullres \
    --folds all
```

🔴 **The Dockerfile regenerates the trainer class at build time, and that step is load-bearing.**
nnU-Net reads the trainer name out of the checkpoint and imports it by name;
`nnUNetTrainerDiceTopK10Loss_1000epochs` is the cross product of an epoch variant and a loss variant,
which nnU-Net ships separately but not together. A stock install raises at model load, before any case
runs.

Agreement against nnU-Net's own predictions is checked by `scripts/verify_container_nnunet.py`; the
floor is 0.95 and a wrong axis order lands near zero.

## Citation

> Ranking-aware model selection for stroke lesion segmentation. MICCAI 2026, SWITCH+ workshop.

## Licence

Apache 2.0, see `LICENSE`. `references/isles26_eval_utils.py` is the organizers' own evaluation code,
redistributed under its original MIT licence with its notice intact.
