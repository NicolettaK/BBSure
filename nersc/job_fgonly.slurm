#!/bin/bash
# job_fgonly.slurm — Asimov foreground-only trees (nersc/run_fgonly.sh) of
# the 9 skies (d9, d10, d12 x s4, s5, s7; d9s4 is the control), all in parallel on one Perlmutter CPU node
# (debug queue: 30 minutes).
#
#   source <install>/activate_bbsure.sh      (the job inherits the environment)
#   cd <root of a validation run>            (with mcm_cache, cov, cov_ref)
#   sbatch -A <project> <repo>/nersc/job_fgonly.slurm [skies...]
#
# Skies already concluded are skipped, interrupted ones restart reusing the
# fits already done: if the job runs out of time, just resubmit it.
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J bbsure_fgonly
#SBATCH -o logs/slurm_fgonly_%j.out

set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR}"
REPO="$(python -c 'import bbsure, os; print(os.path.dirname(os.path.dirname(bbsure.__file__)))')"
SKIES="${*:-d9s4 d9s5 d9s7 d10s4 d10s5 d10s7 d12s4 d12s5 d12s7}"
NPAR=$(echo $SKIES | wc -w)
# 9 trees x 14 processes = 126 of the 128 cores
export BBPOWER_NPROC=${BBPOWER_NPROC:-14}
export BBSURE_NAUTILUS_PROCS=${BBSURE_NAUTILUS_PROCS:-$BBPOWER_NPROC}
export PROFILE_NPROC=${PROFILE_NPROC:-14}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "job ${SLURM_JOB_ID}: fgonly ${SKIES}, ${BBPOWER_NPROC} processes each, $(date '+%F %T')"
for sky in $SKIES; do echo "$sky"; done | xargs -P "$NPAR" -n 1 bash -c \
    'bash "'"$REPO"'/nersc/run_fgonly.sh" "'"$ROOT"'" "$0" \
     && echo "fgonly $0: done" || echo "fgonly $0: FAILED (see logs)"'
echo "job ${SLURM_JOB_ID}: finished $(date '+%F %T')"
