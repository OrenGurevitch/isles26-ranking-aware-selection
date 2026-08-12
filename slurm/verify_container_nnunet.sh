#!/bin/bash
#SBATCH --job-name=isles_verify_nnunet
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --account=def-gmitsis
#SBATCH --mail-type=FAIL,TIME_LIMIT,END
#SBATCH --output=/scratch/orengur2/isles26/logs/verify_%x_%j.out
#
# Does the SUBMISSION container's nnU-Net class reproduce nnU-Net's own prediction?
#
#   sbatch slurm/verify_container_nnunet.sh [EPOCHS] [SUBJECTS]
#
# ⚠️ THIS IS THE GATE ON SUBMITTING THE nnU-Net ARM. The failure it guards is silent -- a wrong axis
# order segments something, it just segments the wrong thing -- so nothing downstream would catch it
# and we get ONE test-phase submission.
#
# ⚠️ It also prints the per-case wallclock, which is the only real evidence about the 10-minute
# budget we can get before the sanity phase reports it on the actual T4. A GPU here is an A100, so
# treat the number as a floor, never as the T4 time.
set -uo pipefail
EPOCHS=${1:-250}
SUBJECTS=${2:-3}
# Named subjects override the count. sub-soop0468 is 85 M voxels, 12.7x the median, and is the case
# that decides whether the 10-minute budget holds -- so it is timed by NAME, never by sort order.
ONLY=${ONLY:-}
# Which fold's own predictions to check against. `all` is the model trained on every subject -- its
# `validation/` is on data it TRAINED on, so this measures the container's inference path against
# nnU-Net's, which is a plumbing check and NOT a performance number.
FOLD=${FOLD:-0}
PLANS=${PLANS:-nnUNetPlans}
# Which TRAINER, when several share a budget. Two do at 1000 epochs -- nnUNetTrainer (the default
# loss) and nnUNetTrainerDiceTopK10Loss_1000epochs (what we submit) -- so the resolver below found two
# candidates and refused, correctly, but could then verify neither. Set TRAINER to the directory's
# trainer prefix to pick one; leave it unset and the refuse-on-ambiguous behaviour is unchanged.
TRAINER=${TRAINER:-}

REPO=${REPO:-$HOME/frozen-isles}
cd "$REPO" || { echo "FATAL: no repo at $REPO" >&2; exit 1; }
module load httpproxy

R=/scratch/orengur2/nnunet_isles
DATASET=Dataset510_ISLES1453
# ⚠️ The trainer directory carries the EPOCH COUNT in its name (nnUNetTrainer_250epochs__...), so a
# hardcoded `nnUNetTrainer__nnUNetPlans__3d_fullres` resolves for exactly one budget and silently
# fails for every other. Resolve it, and refuse an ambiguous match rather than picking one.
mapfile -t CANDIDATES < <(find "$R/nnUNet_results_${EPOCHS}ep/$DATASET" -maxdepth 1 -type d \
    -name "${TRAINER:+${TRAINER}}*__${PLANS}__3d_fullres" | sort)
[ "${#CANDIDATES[@]}" -eq 1 ] || {
    echo "FATAL: expected one ${TRAINER:+${TRAINER}}*__${PLANS}__3d_fullres under $R/nnUNet_results_${EPOCHS}ep/$DATASET, found ${#CANDIDATES[@]}:" >&2
    printf '  %s\n' "${CANDIDATES[@]}" >&2
    echo "  set TRAINER=<trainer-prefix> to disambiguate" >&2; exit 1; }
MODEL=${CANDIDATES[0]}

echo "[env] $(date) node=$(hostname) epochs=$EPOCHS subjects=$SUBJECTS plans=$PLANS fold=$FOLD"
echo "[model] $MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv

# ⚠️ The trainer the checkpoint names must exist in THIS environment. `make_nnunet_trainers.py` writes
# it into the installed nnunetv2, and `uv run --with` resolves a fresh environment per invocation, so
# generating and using it in two separate `uv run` calls is a race this project has already lost once.
# Generate it here, in the same interpreter that is about to load the model.
ISLES_NNUNET_DIR=$MODEL uv run --with nnunetv2 bash -c '
  python scripts/make_nnunet_trainers.py nnUNetTrainerDiceTopK10Loss:1000 || exit 1
  python scripts/verify_container_nnunet.py \
      --images "'"$R/nnUNet_raw/$DATASET/imagesTr"'" \
      --validation "'"$MODEL/fold_${FOLD}/validation"'" \
      --subjects "'"$SUBJECTS"'" '"${ONLY:+--only $ONLY}"'
'
# `set -uo pipefail` excludes -e, so without this the script reaches `[done]` and exits 0 after the
# verification CRASHED, and sacct reports COMPLETED. That happened on job 677958: nnU-Net raised
# "Could not find requested nnunet trainer" at model load and the job still reported COMPLETED 0:0.
# This is the gate on a one-shot submission -- it must fail loudly.
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "FATAL: verification exited $STATUS -- DO NOT SUBMIT on this build" >&2
    exit "$STATUS"
fi
echo "[done] $(date)"
