#!/bin/bash
# job_validation.slurm — one Perlmutter CPU node, 16 trees in parallel
# (8 MPI processes each for PolyChord: 16 x 8 = 128 cores).
#
#   source <install>/activate_bbsure.sh      (the job inherits the environment)
#   cd <root>
#   sbatch -A <project> <repo>/nersc/job_validation.slurm <first set> <last set>
# e.g. sbatch -A <project> $BBSURE/nersc/job_validation.slurm 0 7   (8 sets x 2 skies)
#
# Sets already concluded are skipped, interrupted ones restart reusing the
# fits already done: if the job runs out of time, just resubmit it.
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 05:00:00
#SBATCH -J bbsure_val
#SBATCH -o logs/slurm_%j.out

set -uo pipefail
FIRST="$1"; LAST="$2"
ROOT="${SLURM_SUBMIT_DIR}"
REPO="$(python -c 'import bbsure, os; print(os.path.dirname(os.path.dirname(bbsure.__file__)))')"
# trees in parallel (16 on a full node; e.g. NPAR=8 on half a node of the
# shared queue: sbatch -q shared -c 128 --export=ALL,NPAR=8 ...)
NPAR=${NPAR:-16}
# skies to run (e.g. KINDS=d9s4 for the d9s4 sets only)
KINDS=${KINDS:-d9s4 rand}

export BBPOWER_NPROC=8           # emcee pool processes per tree
# profile points fitted in parallel per tree (one ML each, one core each)
export PROFILE_NPROC=8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "job ${SLURM_JOB_ID}: sets ${FIRST}-${LAST} (${KINDS}), ${NPAR} trees at a time, $(date '+%F %T')"
for i in $(seq "$FIRST" "$LAST"); do
    for kind in $KINDS; do
        echo "$i $kind"
    done
done | xargs -P "$NPAR" -n 2 bash -c \
    'bash "'"$REPO"'/nersc/run_one.sh" "'"$ROOT"'" "$0" "$1" \
     && echo "set $0 $1: done" || echo "set $0 $1: FAILED (see logs)"'
echo "job ${SLURM_JOB_ID}: finished $(date '+%F %T')"
