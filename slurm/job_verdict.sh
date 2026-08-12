#!/bin/bash
#SBATCH --job-name=verdict
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --account=def-gmitsis
#SBATCH --output=/scratch/orengur2/isles26/logs/verdict_%j.out
#
# Decide whether a finished job actually produced what it claimed, ON THE CLUSTER.
#
#   sbatch --dependency=afterany:<JOBID> slurm/job_verdict.sh <JOBID> [EXPECTED_ARTIFACT ...]
#
# 🔴 **Why this runs here and not on the laptop.** On 2026-08-10 a local pace monitor watched an
# overnight job, the MacBook entered Clamshell Sleep at 23:56:43, and every `ssh` after that returned
# nothing. It printed `state='UNKNOWN'` 37 times over nine hours and never fired its terminal branch.
# The job had finished at 02:23. Five earlier monitor failures were all about what the watcher
# COMPUTED; that one was about WHERE IT RAN, and no amount of better arithmetic fixes it.
# `afterany` fires on COMPLETED, FAILED, TIMEOUT and CANCELLED alike, so the verdict is written
# whatever happens.
#
# ⚠️ **It checks ARTIFACTS, not status.** `COMPLETED` has lied here repeatedly: job 677958 reported
# COMPLETED 0:0 while nnU-Net raised at model load, and job 207694 reported FAILED with valid output.
# A verdict that reads sacct and stops would inherit exactly that.
set -uo pipefail
TARGET=${1:?usage: sbatch --dependency=afterany:JOBID slurm/job_verdict.sh JOBID [artifact ...]}
shift
ARTIFACTS=("$@")

LOGDIR=/scratch/orengur2/isles26/logs
VERDICTS=$LOGDIR/verdicts
mkdir -p "$VERDICTS"
OUT=$VERDICTS/${TARGET}.txt

{
    echo "verdict for job $TARGET, written $(date) by verdict job ${SLURM_JOB_ID:-?}"
    echo

    STATE=$(sacct -j "$TARGET" -X -n -P -o State | head -1)
    ELAPSED=$(sacct -j "$TARGET" -X -n -P -o Elapsed | head -1)
    EXITCODE=$(sacct -j "$TARGET" -X -n -P -o ExitCode | head -1)
    echo "sacct: state=$STATE elapsed=$ELAPSED exit=$EXITCODE"

    # The log name carries the job name, which varies, so glob on the id rather than hardcode it.
    LOG=$(ls -t "$LOGDIR"/*_"${TARGET}".out 2>/dev/null | head -1)
    FAILURES=0

    if [ -z "$LOG" ]; then
        echo "🔴 no log found matching $LOGDIR/*_${TARGET}.out"
        FAILURES=$((FAILURES + 1))
    else
        echo "log: $LOG"
        ERRS=$(grep -icE 'traceback|fatal|error|out of memory|killed' "$LOG")
        echo "error-shaped lines in log: $ERRS"
        [ "$ERRS" -gt 0 ] && { echo "--- matching lines ---"; grep -inE 'traceback|fatal|error|out of memory|killed' "$LOG" | head -20; FAILURES=$((FAILURES + 1)); }
        echo "--- last 5 log lines ---"
        tail -5 "$LOG"
    fi

    echo
    for a in "${ARTIFACTS[@]}"; do
        if [ ! -e "$a" ]; then
            echo "🔴 MISSING artifact: $a"
            FAILURES=$((FAILURES + 1))
        elif [ ! -s "$a" ]; then
            echo "🔴 EMPTY artifact: $a"
            FAILURES=$((FAILURES + 1))
        else
            echo "✅ $(stat -c '%s bytes  %y' "$a")  $a"
        fi
    done

    echo
    case "$STATE" in
        COMPLETED) ;;
        *) echo "🔴 terminal state is $STATE, not COMPLETED"; FAILURES=$((FAILURES + 1)) ;;
    esac

    if [ "$FAILURES" -eq 0 ]; then
        echo "VERDICT: PASS — state COMPLETED, no error-shaped log lines, every expected artifact present and non-empty"
    else
        echo "VERDICT: FAIL — $FAILURES problem(s) above. Read the log before believing any downstream number."
    fi
} 2>&1 | tee "$OUT"

echo "[verdict written] $OUT"
