#!/bin/bash
#SBATCH --job-name=stage_preproc
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --account=def-gmitsis
#SBATCH --output=/scratch/orengur2/isles26/logs/stage_%j.out
#
# Pack the preprocessed dataset into ONE archive, so training jobs can stage it to node-local disk
# with a single sequential read instead of 2,906 opens on a shared filesystem.
#
#   sbatch slurm/stage_preprocessed.sh
#
# **Why.** `/scratch` degraded to 7-17 MiB/s on 2026-08-05 and had not recovered 13 hours later.
# Sequential throughput is only half the problem: a `du -sh` over the 2,906 preprocessed files took
# more than eight minutes, so METADATA operations are crawling too. nnU-Net opens two files per case
# per iteration, which is the shape that produces 10-minute epochs against a 30-second steady state.
#
# The remedy is the one Alliance documents for IO-heavy jobs: touch the shared filesystem once, then
# read from `$SLURM_TMPDIR` (node-local NVMe, 805 GB free on Narval GPU nodes). One archive costs one
# open instead of thousands.
#
# This is a WORKAROUND for a degraded filesystem, not an improvement to keep forever. When
# `slurm/io_probe.sh` reads normal again, training directly from `/scratch` is simpler and the archive
# is one more thing that can go stale. Regenerate it if the preprocessing is ever rerun.
set -uo pipefail
R=/scratch/orengur2/nnunet_isles
SRC=$R/nnUNet_preprocessed
DEST=$R/nnUNet_preprocessed.tar
DATASET=${1:-Dataset510_ISLES1453}

[ -d "$SRC/$DATASET" ] || { echo "FATAL: no $SRC/$DATASET" >&2; exit 1; }
if [ -s "$DEST" ]; then
    echo "[skip] $DEST already exists ($(stat -c %s "$DEST") bytes). Delete it to rebuild."
    exit 0
fi

echo "[env] $(date) node=$(hostname) dataset=$DATASET"
# ONLY what training reads. The dataset directory is ~17.8 GB, of which `nnUNetPlans_2d` is 9.7 GB
# that we never train — every arm here is `3d_fullres`. Packing it would add ~20 min to this job AND to
# every job that stages the archive. Add it back here if a 2d arm is ever run.
CONFIG_DIR=${CONFIG_DIR:-nnUNetPlans_3d_fullres}
echo "=== packing $DATASET/{$CONFIG_DIR, gt_segmentations, *.json} ==="
echo "    (one sequential pass; no compression — the .b2nd payload is already compressed)"
# -C so the archive holds `Dataset510_.../...` and untars straight into an nnUNet_preprocessed root.
tar -cf "$DEST.partial" -C "$SRC" \
    "$DATASET/$CONFIG_DIR" "$DATASET/gt_segmentations" \
    $(cd "$SRC" && ls "$DATASET"/*.json)
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "FATAL: tar exited $STATUS" >&2
    rm -f "$DEST.partial"
    exit "$STATUS"
fi
# Rename only after a clean exit, so a killed job never leaves a half-archive that looks complete —
# the `[skip]` check above would otherwise accept it.
mv "$DEST.partial" "$DEST"
echo "[out] $DEST  ($(stat -c %s "$DEST") bytes)"
echo "[done] $(date)"
