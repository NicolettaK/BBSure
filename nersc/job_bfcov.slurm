#!/bin/bash
# job_bfcov.slurm — secondo passaggio (covarianza del best fit) per i set
# <primo>-<ultimo>, iterazione <it>, 16 tree in parallelo su un nodo.
#
#   cd /pscratch/sd/k/krach/SObs/BBpower_Bsure/PySM_rnd2
#   sbatch code/job_bfcov.slurm <primo set> <ultimo set> <it> [cieli]
# cieli: "rand" (default), "d9s4" o "d9s4 rand"
# es. sbatch code/job_bfcov.slurm 0 15 1        (16 tree random)
#
# Fa solo i set il cui passaggio precedente e' concluso; quelli gia' fatti
# vengono saltati, quelli interrotti ripartono dai fit gia' fatti.
#SBATCH -A mp107b
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 05:00:00
#SBATCH -J bbpower_bfcov
#SBATCH -o logs/slurm_bfcov_%j.out

set -uo pipefail
FIRST="$1"; LAST="$2"; IT="$3"; KINDS="${4:-rand}"
ROOT="${SLURM_SUBMIT_DIR}"
NPAR=16

set +u
source /global/homes/k/krach/usr/activate_bbpower.sh
set -u
export BBPOWER_NPROC=8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "job ${SLURM_JOB_ID}: bfcov${IT} for sets ${FIRST}-${LAST} (${KINDS}), $(date '+%F %T')"
for i in $(seq "$FIRST" "$LAST"); do
    for kind in $KINDS; do
        echo "$i $kind"
    done
done | xargs -P "$NPAR" -n 2 bash -c \
    'bash "'"$ROOT"'/code/run_bfcov.sh" "'"$ROOT"'" "$0" "$1" "'"$IT"'" \
     && echo "set $0 $1 bfcov'"$IT"': done" \
     || echo "set $0 $1 bfcov'"$IT"': FAILED or skipped (see logs)"'
echo "job ${SLURM_JOB_ID}: finished $(date '+%F %T')"
