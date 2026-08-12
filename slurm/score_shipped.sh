#!/bin/bash
#SBATCH --job-name=score_shipped
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=def-gmitsis
#SBATCH --output=/scratch/orengur2/isles26/logs/nnunet_%x_%j.out
#
# Score fold 0 in the SHIPPED configuration — threshold 0.5 then `min_voxels=25` — so the reported
# numbers describe the system the container actually submits.
#
#   sbatch slurm/score_shipped.sh
#
# **Why this exists.** `scripts/score_cohort.py` defaulted to no post-processing, so every headline
# number in `docs/RESULTS.md` was the model's RAW output while `container/run.py` ships
# `MIN_VOXELS = 25` and the manuscript's Methods describe removing components under 25 voxels. Found
# 2026-08-05: the reported row was exactly the `min_voxels=0` entry of `postproc_plain500.json`.
#
# The gap falls on the LESION-WISE metrics, which is where the reported number is weakest — at
# `min_voxels=100` on the full fold, lesion-F1 moved 0.5714 → 0.6667 while Dice moved 0.7683 → 0.7645.
#
# CPU ONLY. panoptica is the whole cost, ~8 s/subject, so THREE arms over 291 is ~2 h; the wall is
# 4 h, sized from the slow end as the standing rule requires.
#
# **`topk10_1000` added 2026-08-06, and it is the reason to rerun the whole file rather than append.**
# Job 292882 compared `topk10_1000` against `plain1000` on the RAW output — its own log says
# "[config] scoring the RAW model output — the container ships min_voxels=25". The filter is not
# neutral: at 500 epochs it costs the acute band 0.130 median Dice (`docs/RESULTS.md`). A recipe
# decision taken on a raw-vs-raw comparison is therefore taken on a system nobody ships.
# All three arms must be scored in ONE call, because rank-then-aggregate is only defined across the
# models present in it — scoring them separately gives medians with no ranking.
set -uo pipefail
REPO=${REPO:-$HOME/frozen-isles}
cd "$REPO" || { echo "FATAL: no repo at $REPO" >&2; exit 1; }
S=/scratch/orengur2/nnunet_isles/scored
R=/scratch/orengur2/nnunet_isles
D=Dataset510_ISLES1453

echo "[env] $(date) node=$(hostname)"
uv run python scripts/score_cohort.py --min-voxels 25 --closing-radius 0 \
    --references "$S/references_plain500" \
    --predictions \
        plain500_shipped="$R/nnUNet_results_500ep/$D/nnUNetTrainer_500epochs__nnUNetPlans__3d_fullres/fold_0/validation" \
        plain1000_shipped="$R/nnUNet_results_1000ep/$D/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation" \
        topk10_1000_shipped="$R/nnUNet_results_1000ep/$D/nnUNetTrainerDiceTopK10Loss_1000epochs__nnUNetPlans__3d_fullres/fold_0/validation" \
    --soft plain500_shipped="$S/soft_plain500" plain1000_shipped="$S/soft_plain1000" \
           topk10_1000_shipped="$S/soft_topk10_1000" \
    --out "$S/scores_shipped_min25.json"
# `set -uo pipefail` excludes -e, so without this the script reaches `[done]` and exits 0 after a
# crash, and sacct reports COMPLETED for a run that produced nothing.
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "FATAL: score_cohort.py exited $STATUS" >&2
    exit "$STATUS"
fi
echo "[done] $(date)"
