#!/bin/bash
#SBATCH --job-name=isles_nnunet_train
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --account=def-gmitsis
#SBATCH --mail-type=FAIL,TIME_LIMIT,END
#SBATCH --output=/scratch/orengur2/isles26/logs/nnunet_%x_%j.out
#
# STAGE 3+4 of docs/NNUNET_BASELINE.md. ARG1 = fold (0-4), ARG2 = configuration.
#
#   EPOCHS=5   sbatch --time=01:00:00 --job-name=nnu_timing slurm/nnunet_train.sh 0 3d_fullres
#   EPOCHS=100 sbatch slurm/nnunet_train.sh 0 3d_fullres
#
# RUN THE 5-EPOCH TIMING JOB FIRST, and size everything else from it. The per-epoch cost here is
# PREDICTED from frozen-wmh's 0.60 GPU-h/fold at 100 epochs, on the argument that nnU-Net sizes its
# patch to a fixed memory budget so per-iteration cost transfers -- that argument has not been
# checked on this data, and sizing a long run from an unchecked prediction is what the standing rule
# forbids.
#
# EPOCHS IS PART OF THE RESULT, NOT A KNOB TO HIDE. nnU-Net's PolyLR anneals to zero over
# `num_epochs`, so a 100-epoch run is a COMPLETE schedule at its own length rather than a truncated
# 1000-epoch one -- but reporting a short run as "nnU-Net's result" is the inadequate-baseline failure
# named in arXiv 2404.09556. Every number this produces carries its epoch count and its GPU-hours.
#
# Results roots are SEPARATE per epoch budget, so a cheap run can never overwrite or be confused
# with an expensive one. frozen-wmh keeps the same separation.
set -uo pipefail
FOLD=${1:?usage: [EPOCHS=100] sbatch slurm/nnunet_train.sh <0-4> <configuration>}
CONFIG=${2:?usage: [EPOCHS=100] sbatch slurm/nnunet_train.sh <0-4> <configuration>}
# `all` trains on every subject and holds nothing out -- the mode for the model that SHIPS, because
# the hidden test set IS the held-out set. Everything else must be one of our five folds.
case "$FOLD" in 0|1|2|3|4|all) ;; *) echo "FATAL: fold must be 0-4 or 'all' (got '$FOLD')" >&2; exit 1;; esac
if [ "$FOLD" = all ]; then
    cat >&2 <<'WARN'
 FOLD=all: this model CANNOT BE SCORED BY US. There is no held-out set left, so it produces no
    comparable number and its `validation/` output is on data it trained on. Train it only from a
    recipe already settled on the folds, and keep reporting the fold-0 numbers.
WARN
fi
EPOCHS="${EPOCHS:-100}"
case "$EPOCHS" in ''|*[!0-9]*) echo "FATAL: EPOCHS must be an integer, got '$EPOCHS'" >&2; exit 1;; esac
# 1000 epochs is nnU-Net's DEFAULT and therefore the PLAIN `nnUNetTrainer` -- there is no
# `nnUNetTrainer_1000epochs` variant, because the base class already is one. Building the name
# unconditionally sent a request for the default budget to a trainer that does not exist, and the
# nearest shipped variant below it is 750. Anyone reaching for "the default" must get 1000.
# TRAINER can be set explicitly to reach nnU-Net's loss, augmentation and sampling variants
# (nnUNetTrainerDiceTopK10Loss, nnUNetTrainerDA5, ...). Those variants carry the BASE trainer's
# 1000-epoch default and ignore EPOCHS, so exploring one at a shorter budget needs a trainer nnU-Net
# does not ship -- `scripts/make_nnunet_trainers.py` generates it, and the preflight below then
# resolves it exactly as nnUNetv2_train will.
#
#   TRAINER=nnUNetTrainerDA5 EPOCHS=250 sbatch slurm/nnunet_train.sh 0 3d_fullres
#
# A generated `X_250epochs` is comparable to `nnUNetTrainer_250epochs` and to NOTHING ELSE: an arm
# at a different budget differs in two things at once. The plain 250-epoch run on Dataset510 is the
# matched control for every 250-epoch arm, and it is already trained.
GENERATE_FROM=""
if [ -n "${TRAINER:-}" ]; then
    case "$TRAINER" in
        *epochs) ;;                        # already carries its own budget
        *) GENERATE_FROM="$TRAINER"; TRAINER="${TRAINER}_${EPOCHS}epochs";;
    esac
    echo "[trainer] explicit: $TRAINER" >&2
elif [ "$EPOCHS" = 1000 ]; then TRAINER="nnUNetTrainer"
else TRAINER="nnUNetTrainer_${EPOCHS}epochs"; fi
# PLANS selects the network family, and must match a planner that has already preprocessed.
#   PLANS=nnUNetResEncUNetLPlans EPOCHS=250 sbatch slurm/nnunet_train.sh 0 3d_fullres
PLANS=${PLANS:-nnUNetPlans}
# DATASET 511 is the leave-centres-out cohort; 510 is the stratified one. Results roots are already
# separated by epoch budget, and the dataset id keeps the two SPLIT STRATEGIES apart inside them.
DATASET_ID=${DATASET_ID:-510}
case "$DATASET_ID" in 510) DATASET_DIR=Dataset510_ISLES1453;; 511) DATASET_DIR=Dataset511_ISLES1453_LOCO;;
  *) echo "FATAL: unknown DATASET_ID $DATASET_ID" >&2; exit 1;; esac

module load httpproxy
REPO=${REPO:-$HOME/frozen-isles}
cd "$REPO" || { echo "FATAL: no repo at $REPO" >&2; exit 1; }

# ONE PERSISTENT ENVIRONMENT for every nnU-Net call in this script.
# `uv run --with nnunetv2` resolves per invocation and CAN rebuild between them: on 2026-08-06 job
# 290413 generated a trainer into one environment, uv then reinstalled 99 packages, and the preflight
# looked for it in a different one — "no trainer named ..." for a file that had just been written.
# `scripts/make_nnunet_trainers.py` writes INTO the installed package, so generation and use must share
# an environment or the whole mechanism is a race.
NNUNET_VENV=${NNUNET_VENV:-$HOME/.venvs/nnunet}
if [ ! -x "$NNUNET_VENV/bin/nnUNetv2_train" ]; then
    echo "[venv] building $NNUNET_VENV once — subsequent jobs reuse it"
    uv venv "$NNUNET_VENV" || { echo "FATAL: uv venv failed" >&2; exit 1; }
    VIRTUAL_ENV="$NNUNET_VENV" uv pip install nnunetv2 \
        || { echo "FATAL: could not install nnunetv2 into $NNUNET_VENV" >&2; exit 1; }
fi
PY_BIN=$NNUNET_VENV/bin/python
echo "[venv] $NNUNET_VENV"

R=/scratch/orengur2/nnunet_isles
export nnUNet_raw=$R/nnUNet_raw
export nnUNet_preprocessed=$R/nnUNet_preprocessed
export nnUNet_results=$R/nnUNet_results_${EPOCHS}ep
mkdir -p "$nnUNet_results"

# STAGE THE PREPROCESSED DATA TO NODE-LOCAL DISK when the archive exists. /scratch fell to 8 MiB/s
# on 2026-08-05 and had not recovered a day later; nnU-Net opens two files per case per iteration, so a
# dataloader reading it directly starves and the GPU idles at 0% (measured: ~10 min/epoch against a
# ~30 s steady state). Copying one archive converts thousands of small reads into one sequential read.
# STAGE=0 disables it. When `slurm/io_probe.sh` reports normal on both axes, this is unnecessary —
# delete the archive and it turns itself off.
ARCHIVE=$R/nnUNet_preprocessed.tar
if [ "${STAGE:-1}" = 1 ] && [ -s "$ARCHIVE" ] && [ -n "${SLURM_TMPDIR:-}" ]; then
    echo "[stage] copying $(stat -c %s "$ARCHIVE") bytes to $SLURM_TMPDIR — one sequential read"
    date
    mkdir -p "$SLURM_TMPDIR/preprocessed"
    tar -xf "$ARCHIVE" -C "$SLURM_TMPDIR/preprocessed" || {
        echo "FATAL: staging failed; NOT falling back to /scratch silently" >&2; exit 1; }
    STAGED=$SLURM_TMPDIR/preprocessed/$DATASET_DIR
    # Verified by COUNT, because a truncated archive untars without error and would train on a subset.
    N=$(ls "$STAGED"/nnUNetPlans_3d_fullres/*.b2nd 2>/dev/null | wc -l)
    [ "$N" -ge 2900 ] || { echo "FATAL: staged only $N .b2nd files, expected ~2906" >&2; exit 1; }
    export nnUNet_preprocessed=$SLURM_TMPDIR/preprocessed
    echo "[stage] done: $N files local at $nnUNet_preprocessed"
    date
else
    echo "[stage] reading /scratch directly (STAGE=${STAGE:-1}, archive $([ -s "$ARCHIVE" ] && echo present || echo absent))"
fi

echo "[env] $(date) node=$(hostname) dataset=$DATASET_ID fold=$FOLD config=$CONFIG plans=$PLANS epochs=$EPOCHS ($((EPOCHS * 250)) steps)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The splits file is the whole basis of the comparison. If it is missing, nnU-Net silently generates
# its own KFold and trains on our held-out subjects -- a number that looks comparable and is not.
SPLITS=$nnUNet_preprocessed/$DATASET_DIR/splits_final.json  # honours the staged root above
[ -s "$SPLITS" ] || { echo "FATAL: $SPLITS missing -- nnU-Net would invent its own folds" >&2; exit 1; }
if [ "$FOLD" = all ]; then
    # `all` ignores splits_final.json by design, so there is no fold to check -- but the file must
    # still be the one we wrote, or a LATER fold run would silently use nnU-Net's own KFold.
    python3 -c "
import json
d=json.load(open('$SPLITS'))
assert len(d)==5, f'expected 5 folds, got {len(d)}'
print('[splits] FOLD=all ignores the split file; it holds our 5 folds and is intact')
" || { echo "FATAL: splits_final.json is not the file we wrote" >&2; exit 1; }
else
    python3 -c "
import json,sys
d=json.load(open('$SPLITS'))
assert len(d)==5, f'expected 5 folds, got {len(d)}'
f=d[$FOLD]; print(f\"[splits] fold $FOLD: {len(f['train'])} train / {len(f['val'])} val\")
assert not (set(f['train']) & set(f['val'])), 'train and val overlap'
" || { echo "FATAL: splits_final.json is not the file we wrote" >&2; exit 1; }
fi

# The generated trainer lives inside the INSTALLED nnunetv2 package. It is regenerated per job because
# the venv can be rebuilt or deleted between jobs, and the script is idempotent. Generation, the
# preflight and the training all use $PY_BIN so they share ONE environment — see the venv block above.
if [ -n "$GENERATE_FROM" ]; then
    "$PY_BIN" scripts/make_nnunet_trainers.py "${GENERATE_FROM}:${EPOCHS}" \
        || { echo "FATAL: could not generate trainer $TRAINER from $GENERATE_FROM" >&2; exit 1; }
fi

# A trainer for an arbitrary epoch count MAY NOT EXIST -- nnU-Net ships a fixed set of length
# variants. An unknown name fails deep inside nnUNetv2_train, AFTER the job has queued, started and
# taken the GPU, so it is checked here first.
"$PY_BIN" -c "
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2, os
cls = recursive_find_python_class(os.path.join(nnunetv2.__path__[0], 'training', 'nnUNetTrainer'),
                                  '$TRAINER', 'nnunetv2.training.nnUNetTrainer')
assert cls is not None, 'no trainer named $TRAINER -- nnU-Net ships a fixed set of epoch variants'
print('[preflight] trainer $TRAINER exists')
" || { echo "FATAL: trainer $TRAINER not found" >&2; exit 1; }

# `--c` resumes, and is passed ONLY when a checkpoint exists: on a fresh directory it asks nnU-Net to
# continue a training that never started.
RESUME=""
[ -f "$nnUNet_results/$DATASET_DIR/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}/checkpoint_latest.pth" ] && RESUME="--c"

# Declare where nnU-Net will write its own line-flushed training log. THIS file is the progress
# source, not the SLURM .out: python block-buffers stdout when redirected, so the .out can sit
# untouched for hours while training runs normally (observed 2026-08-04, jobs 99941 and 98690 at
# epoch 211 and 237 with .out files unwritten since launch). scripts/watch_jobs.py reads this line
# so the path lives in ONE place rather than being rebuilt from the naming convention.
echo "[foldrun] $nnUNet_results/$DATASET_DIR/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}"

# torch.compile is nnU-Net's default and it HUNG two jobs on 2026-08-05, both immediately after the
# first "Current learning rate" line, where the first forward triggers compilation.
# **The likeliest cause is not compile.** `/scratch` throughput collapsed to ~8 MiB/s that evening
# and stayed there, which starves any read — including the cache files compilation writes and reads.
# The same degradation explains the 10-minute epochs seen with compile already disabled.
# An earlier note here blamed node-local storage differing by node family. That was WRONG:
# `scontrol show node` reports identical `milan,a100,nvlink`, 510 GB, 48 CPUs across ng10xxx, ng20xxx
# and ng30xxx, so there was never a hardware basis for it.
# COMPILE=0 stays the default because a run that has to finish should not carry an unresolved suspect.
# **Once staging is confirmed working, COMPILE=1 is the cheap test** of whether compile was ever
# implicated at all — with the data local, an inductor stall would no longer have an IO explanation.
if [ "${COMPILE:-0}" = 1 ]; then
    export nnUNet_compile=true
    echo "[compile] torch.compile ENABLED — nnU-Net's default. ⚠️ Hung two jobs on 2026-08-05" >&2
else
    export nnUNet_compile=false
    echo "[compile] torch.compile disabled (COMPILE=1 to enable)" >&2
fi

echo "=== TRAIN ==="; date
"$NNUNET_VENV/bin/nnUNetv2_train" "$DATASET_ID" "$CONFIG" "$FOLD" -tr "$TRAINER" -p "$PLANS" --npz $RESUME
# `set -uo pipefail` excludes -e, so WITHOUT this check the script would reach `[done]` and exit 0
# after a crash -- sacct then reports COMPLETED for a run that produced nothing. That has happened
# four times on this project; the status field is not evidence.
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "FATAL: nnUNetv2_train exited $STATUS -- training did NOT finish" >&2
    exit "$STATUS"
fi
echo "[done] $(date)"
echo "NEXT: predict fold $FOLD's held-out subjects WITH --save_probabilities, convert them to NIfTI,"
echo "      and score through OUR harness (scripts/score_cohort.py) -- never nnU-Net's own Dice."
