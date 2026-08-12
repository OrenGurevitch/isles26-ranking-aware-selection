#!/bin/bash
#SBATCH --job-name=isles_nnunet_prep
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --account=def-gmitsis
#SBATCH --mail-type=FAIL,TIME_LIMIT,END
#SBATCH --output=/scratch/orengur2/isles26/logs/nnunet_%x_%j.out
#
# STAGE 1+2 of docs/NNUNET_BASELINE.md: convert ATLAS to nnU-Net's layout, then let nnU-Net
# fingerprint, plan and preprocess. **CPU ONLY** -- none of this touches a GPU, so it runs while the
# HP searches hold both A100s, and putting it on a GPU node would idle an accelerator behind a longer
# queue for nothing.
#
#   sbatch slurm/nnunet_prepare.sh
#
# ⚠️ WALLTIME IS A GUESS AND THIS JOB EXISTS PARTLY TO REPLACE IT WITH A MEASUREMENT. frozen-wmh
# recorded "1:06 on 8 CPU cores" for 50 subjects without an unambiguous unit, and ISLES has 29x the
# cases at ~4x the voxels. 8 h at 16 cores is sized from the SLOW reading of that; the log prints
# wall-clock for each stage so the next run can be sized from evidence. Over-requesting a CPU job
# costs backfill position only.
#
# ⚠️ DISK: preprocessed 3d_fullres at ~1 mm is roughly 46 MB/case plus segmentation, so expect ~80 GB
# for one configuration and ~165 GB if `2d` is planned too. /scratch was at 9.1 of 20 TB.
#
# ⚠️ The 353.6 M-voxel outlier (0.5 mm acquisitions) is resampled DOWN to ~44 M, but peak memory is
# set by the pre-resampling array -- hence 64 G rather than the 16 G this would otherwise need.
set -uo pipefail

module load httpproxy
REPO=${REPO:-$HOME/frozen-isles}
cd "$REPO" || { echo "FATAL: no repo at $REPO" >&2; exit 1; }
DATA=/scratch/orengur2/isles26
R=/scratch/orengur2/nnunet_isles          # SEPARATE from frozen-wmh's /scratch/orengur2/nnunet
export nnUNet_raw=$R/nnUNet_raw
export nnUNet_preprocessed=$R/nnUNet_preprocessed
export nnUNet_results=$R/nnUNet_results
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# PLANNER selects the network family. The default `ExperimentPlanner` is plain nnU-Net; the field's
# strong default -- and the public ISLES'26 competitor container -- is ResEncL, a residual-encoder
# U-Net. Each planner writes its OWN plans file and its OWN preprocessed data, so they coexist.
#   PLANNER=nnUNetPlannerResEncL sbatch slurm/nnunet_prepare.sh
PLANNER=${PLANNER:-}
# SPLITS=leave-centres-out builds Dataset511, whose folds hold out WHOLE CENTRES. That is the only
# split that measures what the hidden test measures -- it contains sites absent from training -- and
# it lands 291 held-out / 1,162 training per fold, the same numbers as the stratified split, so the
# two differ in exactly one variable.
SPLITS=${SPLITS:-stratified}
DATASET_ID=510; DATASET_DIR=Dataset510_ISLES1453
[ "$SPLITS" = "leave-centres-out" ] && { DATASET_ID=511; DATASET_DIR=Dataset511_ISLES1453_LOCO; }

echo "[env] $(date) node=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} splits=$SPLITS dataset=$DATASET_ID"

# Idempotent: the conversion is deterministic and slow, so a second planner reuses it rather than
# rewriting 1,453 masks. splits_final.json is ours and must survive either way.
if [ -s "$nnUNet_raw/$DATASET_DIR/dataset.json" ]; then
    echo "=== STAGE 1: already converted, reusing $nnUNet_raw/$DATASET_DIR ==="
else
    echo "=== STAGE 1: convert + our folds ==="; date
    uv run python scripts/nnunet_prepare.py \
        --tree "$DATA/ATLAS3_Training_Raw" --inventory "$DATA/inventory.csv" \
        --raw "$nnUNet_raw" --preprocessed "$nnUNet_preprocessed" --seed 0 --splits "$SPLITS" \
        || { echo "FATAL: conversion failed" >&2; exit 1; }
    date
fi

# ⚠️ `--verify_dataset_integrity` reads every case, so it runs ONCE -- on the first planner. It is what
# catches a geometry or label problem before 1,453 cases are preprocessed on a bad premise; a second
# planner over the same raw data has nothing new to verify. `VERIFY=1` forces it back on.
# ⚠️ splits_final.json is written by stage 1 into $nnUNet_preprocessed/<dataset>. Confirm it SURVIVES
# this step -- if the planner ever clears that directory, nnU-Net silently falls back to its own
# KFold and every downstream number is trained on our held-out subjects.
FINGERPRINT=$nnUNet_preprocessed/$DATASET_DIR/dataset_fingerprint.json
VERIFY=${VERIFY:-}
[ -s "$FINGERPRINT" ] || VERIFY=1
echo "=== STAGE 2: fingerprint + plan + preprocess ${PLANNER:+(planner $PLANNER)} ==="; date
uv run --with nnunetv2 nnUNetv2_plan_and_preprocess \
    -d "$DATASET_ID" -np "${SLURM_CPUS_PER_TASK:-8}" \
    ${PLANNER:+-pl "$PLANNER"} ${VERIFY:+--verify_dataset_integrity} \
    || { echo "FATAL: plan_and_preprocess failed" >&2; exit 1; }
date

SPLITS_FILE=$nnUNet_preprocessed/$DATASET_DIR/splits_final.json
[ -s "$SPLITS_FILE" ] || { echo "FATAL: $SPLITS_FILE is gone -- nnU-Net would invent its own folds" >&2; exit 1; }
echo "[splits] survived: $(python3 -c "import json;d=json.load(open('$SPLITS_FILE'));print(len(d),'folds, fold0',len(d[0]['train']),'train /',len(d[0]['val']),'val')")"

echo "=== WHAT THE PLANNER CHOSE (the input to the cost model) ==="
for p in "$nnUNet_preprocessed"/$DATASET_DIR/*Plans*.json; do
    [ -r "$p" ] && echo "-- $(basename "$p")" && python3 -c "
import json;p=json.load(open('$p'))
print('configurations:', list(p['configurations']))
for name,c in p['configurations'].items():
    print(f\"  {name:12s} patch={c.get('patch_size')} batch={c.get('batch_size')} spacing={[round(s,3) for s in c.get('spacing',[])]}\")"
done
echo "=== DISK ==="; du -sh "$nnUNet_raw" "$nnUNet_preprocessed" 2>/dev/null

echo "[done] $(date)"
echo "NEXT: a 5-EPOCH timing run into a throwaway results root before anything long --"
echo "      docs/NNUNET_BASELINE.md stage 3. Do not submit a real training without that number."
