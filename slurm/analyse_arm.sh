#!/bin/bash
#SBATCH --job-name=analyse_arm
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --account=def-gmitsis
#SBATCH --output=/scratch/orengur2/isles26/logs/nnunet_%x_%j.out
#
# Everything we compute per arm from its exported soft maps: the precision/recall split behind
# lesion-F1, and the census of where cases sit relative to the IoU 0.25 matching threshold.
#
# Both feed manuscript claims, so they must be recomputed for whichever arm the paper describes —
# quoting one arm's zero-scoring counts alongside another's headline metrics is the inconsistency
# this script exists to make cheap to avoid.
#
#   sbatch slurm/analyse_arm.sh [ARM]    # ARM defaults to plain500
#
# ⚠️ CPU ONLY. panoptica is the entire cost (~8 s/subject, so ~40 min over 291) and a GPU would sit
# idle behind a longer queue for nothing. Two hours is sized from that, from the slow end.
#
# ⚠️ Runs on the SHIPPED predictions -- threshold 0.5 then `min_voxels=25` -- because the question is
# about what the challenge would score, not about raw model output. Measuring before post-processing
# would overstate the precision problem the filter exists to fix.
set -uo pipefail
ARM=${1:-plain500}
REPO=${REPO:-$HOME/frozen-isles}
cd "$REPO" || { echo "FATAL: no repo at $REPO" >&2; exit 1; }
S=/scratch/orengur2/nnunet_isles/scored
[ -d "$S/references_$ARM" ] || { echo "FATAL: no $S/references_$ARM -- score the arm first" >&2; exit 1; }
[ -d "$S/soft_$ARM" ] || { echo "FATAL: no $S/soft_$ARM -- score the arm first" >&2; exit 1; }

echo "[env] $(date) node=$(hostname) arm=$ARM"
echo "=== lesion-wise precision and recall ==="
uv run python scripts/lesion_precision_recall.py \
    --references "$S/references_$ARM" --soft "$S/soft_$ARM" \
    --out "$S/lesion_pr_$ARM.json"
STATUS=$?
[ "$STATUS" -ne 0 ] && { echo "FATAL: lesion_precision_recall.py exited $STATUS" >&2; exit "$STATUS"; }

echo "=== IoU census against the 0.25 matching threshold ==="
uv run python scripts/iou_cliff_census.py \
    --references "$S/references_$ARM" --soft "$S/soft_$ARM" \
    --out "$S/iou_census_$ARM.json"
# `set -uo pipefail` excludes -e, so without this the script reaches `[done]` and exits 0 after a
# crash, and sacct reports COMPLETED for a run that produced nothing.
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "FATAL: iou_cliff_census.py exited $STATUS" >&2
    exit "$STATUS"
fi
echo "=== false-positive geometry: do the spurious components float or over-grow? ==="
# ⚠️ RUN AT min_voxels=0, UNLIKE the two above. The question is what the RAW model invents; applying
# the filter first would remove the small spurious components this analysis exists to count, and the
# manuscript quotes it as "in the unfiltered output" for exactly that reason. State the setting beside
# any number taken from it -- mixing it with the min_voxels=25 figures above is the error that reached
# the manuscript on 2026-08-06.
uv run python scripts/false_positive_geometry.py --min-voxels 0 \
    --references "$S/references_$ARM" --soft "$S/soft_$ARM" \
    --out "$S/fp_geometry_$ARM.json"
STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    echo "FATAL: false_positive_geometry.py exited $STATUS" >&2
    exit "$STATUS"
fi
echo "[done] $(date)"
