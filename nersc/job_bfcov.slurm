#!/bin/bash
# job_bfcov.slurm — second pass (best-fit covariance) for the sets
# <first>-<last>, iteration <it>, 16 trees in parallel on one node.
#
#   source <install>/activate_bbsure.sh      (the job inherits the environment)
#   cd <root>
#   sbatch -A <project> <repo>/nersc/job_bfcov.slurm <first set> <last set> <it> [skies]
# skies: "rand" (default), "d9s4" or "d9s4 rand"
#
# Only the sets whose previous pass is concluded; the ones already done are
# skipped, interrupted ones restart from the fits already done.
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 05:00:00
#SBATCH -J bbsure_bfcov
#SBATCH -o logs/slurm_bfcov_%j.out

set -uo pipefail
FIRST="$1"; LAST="$2"; IT="$3"; KINDS="${4:-rand}"
ROOT="${SLURM_SUBMIT_DIR}"
REPO="$(python -c 'import bbsure, os; print(os.path.dirname(os.path.dirname(bbsure.__file__)))')"
NPAR=16

export BBPOWER_NPROC=8
export PROFILE_NPROC=8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "job ${SLURM_JOB_ID}: bfcov${IT} for sets ${FIRST}-${LAST} (${KINDS}), $(date '+%F %T')"
for i in $(seq "$FIRST" "$LAST"); do
    for kind in $KINDS; do
        echo "$i $kind"
    done
done | xargs -P "$NPAR" -n 2 bash -c \
    'bash "'"$REPO"'/nersc/run_bfcov.sh" "'"$ROOT"'" "$0" "$1" "'"$IT"'" \
     && echo "set $0 $1 bfcov'"$IT"': done" \
     || echo "set $0 $1 bfcov'"$IT"': FAILED or skipped (see logs)"'
echo "job ${SLURM_JOB_ID}: finished $(date '+%F %T')"
