#!/bin/bash
#SBATCH --job-name=isles_nnunet_score
#SBATCH --time=06:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --account=def-gmitsis
#SBATCH --mail-type=FAIL,TIME_LIMIT,END
#SBATCH --output=/scratch/orengur2/isles26/logs/nnunet_%x_%j.out
#
# STAGE 5 of docs/NNUNET_BASELINE.md: score nnU-Net through OUR harness. **CPU ONLY** -- panoptica is
# the cost here, not the GPU.
#
#   sbatch slurm/nnunet_score.sh NAME=/path/to/fold_0/validation [NAME=...]
#   sbatch --export=ALL,MIN_VOXELS=25 slurm/nnunet_score.sh NAME=...
#
# POST-PROCESSING DEFAULTS TO NONE, and the container ships `min_voxels=25`. Scoring raw output
# while describing a filtered system is the 2026-08-05 defect: it understated lesion-F1 by 0.095,
# because a component filter moves the LESION-WISE metrics, which is where our numbers are weakest.
# The default stays 0 so every past invocation still means what its log says; pass `MIN_VOXELS=25`
# for any number that describes the shipped system. `score_cohort.py` prints the setting it used.
#
# Each ARG is NAME=VALIDATION_DIR, given explicitly. It used to derive the path from an epoch count
# and a configuration, with the plans name and dataset hardcoded -- which silently could not address
# the ResEncL runs or the leave-centres-out dataset, the two things we most need to score. An
# explicit path cannot go stale as the matrix of plans x datasets grows.
#
# SCORE EVERY MODEL YOU WANT RANKED IN ONE CALL: rank-then-aggregate is a comparison ACROSS models,
# so scoring them separately yields medians but no ranking.
#
# THIS IS THE WHOLE POINT: the comparison is decided by `frozen_isles.metrics` -- the organizers'
# own libraries, the same code that scores every probe number -- and ranked by rank-then-aggregate,
# never by nnU-Net's own Dice, which uses a different implementation on a different grid.
#
# Soft maps are exported first because PR-AUC is one of the five ranked metrics and is defined on
# the probability map. Falling back to the binary mask caps nnU-Net's PR-AUC and would understate it
# in OUR favour -- a worse comparison than none.
#
# Walltime has TWO costs, and sizing from panoptica alone is what made a 3 h invocation too tight:
#   export     ~13 min per arm  (skipped when the soft maps are already complete -- see below)
#   panoptica  ~8 s per subject, so ~40 min per arm over 291
# Three arms from cold is therefore ~41 min + ~2 h = ~2.7 h, not 2 h. **A user cannot RAISE a walltime**
# (`scontrol update TimeLimit` is denied), so size it long: the 6 h default here is deliberate and
# over-long walltimes cost fairshare nothing, only backfill position.
set -uo pipefail
[ $# -ge 1 ] || { echo "usage: sbatch slurm/nnunet_score.sh <epochs>ep_<config> [...]" >&2; exit 1; }

REPO=${REPO:-$HOME/frozen-isles}
cd "$REPO" || { echo "FATAL: no repo at $REPO" >&2; exit 1; }
DATA=/scratch/orengur2/isles26
R=/scratch/orengur2/nnunet_isles
OUT=$R/scored
MIN_VOXELS=${MIN_VOXELS:-0}
CLOSING_RADIUS=${CLOSING_RADIUS:-0}
mkdir -p "$OUT"

echo "[env] $(date) node=$(hostname) runs=$*"
echo "[env] min_voxels=$MIN_VOXELS closing_radius=$CLOSING_RADIUS"

PRED_ARGS=(); SOFT_ARGS=(); NAMES=(); REFS=""
for spec in "$@"; do
    case "$spec" in *=*) ;; *) echo "FATAL: expected NAME=VALIDATION_DIR, got '$spec'" >&2; exit 1;; esac
    run=${spec%%=*}; VAL=${spec#*=}
    [ -d "$VAL" ] || { echo "FATAL: no validation directory for $run at $VAL" >&2; exit 1; }
    n=$(ls "$VAL"/*.nii.gz 2>/dev/null | wc -l)
    m=$(ls "$VAL"/*.npz 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || { echo "FATAL: $run has no predictions in $VAL" >&2; exit 1; }
    [ "$n" -eq "$m" ] || { echo "FATAL: $run has $n masks but $m probability files -- was --npz passed to training?" >&2; exit 1; }
    echo "[run] $run: $n cases from $VAL"

    SOFT=$OUT/soft_$run
    REFS=$OUT/references_$run
    # Resumable: export costs ~13 min per arm and is deterministic, so a re-run after a walltime
    # timeout must not pay for it twice. The guard is the COUNT matching the validation directory --
    # a partial export from a killed job has fewer files and is redone rather than trusted.
    have_soft=$(ls "$SOFT"/*.nii.gz 2>/dev/null | wc -l)
    have_refs=$(ls "$REFS"/*.nii.gz 2>/dev/null | wc -l)
    if [ "$have_soft" -eq "$n" ] && [ "$have_refs" -eq "$n" ]; then
        echo "=== soft maps for $run already complete ($n cases) — skipping export ==="
    else
        echo "=== exporting soft maps for $run ==="; date
        [ "$have_soft" -gt 0 ] && echo "  (partial export found: $have_soft of $n — redoing)"
        uv run python scripts/nnunet_export_soft.py \
            --validation "$VAL" --tree "$DATA/ATLAS3_Training_Raw" --out "$SOFT" --references "$REFS" \
            || { echo "FATAL: soft export failed for $run" >&2; exit 1; }
    fi
    PRED_ARGS+=("$run=$VAL")
    SOFT_ARGS+=("$run=$SOFT")
    NAMES+=("$run")
done

# Every model must be scored on the SAME subjects, and REFS below is whichever run came last. A
# stratified fold and a leave-centres-out fold hold DIFFERENT cohorts, so ranking them together is a
# category error -- caught here with a readable message rather than downstream as "no prediction for
# 291 subjects".
for name in "${NAMES[@]}"; do
    a=$(ls "$OUT/references_$name" | sort | md5sum)
    b=$(ls "$REFS" | sort | md5sum)
    [ "$a" = "$b" ] || { echo "FATAL: $name covers different subjects than $(basename "$REFS") -- these cohorts cannot be ranked against each other" >&2; exit 1; }
done

echo "=== scoring through frozen_isles.metrics (the organizers' own code) ==="; date
# One --references directory, so every model is scored on the SAME subjects. Scoring a
# stratified-fold model against a leave-centres-out model is a category error: they hold out
# different cohorts, and score_cohort refuses a mismatch rather than comparing across them.
uv run python scripts/score_cohort.py \
    --min-voxels "$MIN_VOXELS" --closing-radius "$CLOSING_RADIUS" \
    --references "$REFS" \
    --predictions "${PRED_ARGS[@]}" \
    --soft "${SOFT_ARGS[@]}" \
    --out "$OUT/scores_$(IFS=-; echo "${NAMES[*]}")_min${MIN_VOXELS}.json"
# `set -uo pipefail` excludes -e, so without this the script reaches `[done]` and exits 0 after a
# crash, and sacct reports COMPLETED for a run that produced nothing.
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "FATAL: score_cohort.py exited $STATUS" >&2
    exit "$STATUS"
fi
echo "[done] $(date)"
