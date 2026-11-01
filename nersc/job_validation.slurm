#!/bin/bash
# job_validation.slurm — un nodo CPU di Perlmutter, 16 tree in parallelo
# (8 processi MPI ciascuno per PolyChord: 16 x 8 = 128 core).
#
#   cd /pscratch/sd/k/krach/SObs/BBpower_Bsure/PySM_rnd
#   sbatch code/job_validation.slurm <primo set> <ultimo set>
# es. pilota: sbatch code/job_validation.slurm 0 7   (8 set x 2 cieli = 16 tree)
# Terzo argomento opzionale: tag del config (es. _l130 -> tree con l_max 130,
# in runs/set_*/tree_l130/, riusando mappe e spettri).
#
# I set gia' conclusi vengono saltati, quelli interrotti ripartono riusando i
# fit fatti: se il job finisce il tempo basta risottometterlo.
#SBATCH -A mp107b
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 05:00:00
#SBATCH -J bbpower_val
#SBATCH -o logs/slurm_%j.out

set -uo pipefail
FIRST="$1"; LAST="$2"; TAG="${3:-}"
ROOT="${SLURM_SUBMIT_DIR}"
NPAR=16

set +u
source /global/homes/k/krach/usr/activate_bbpower.sh
set -u
export BBPOWER_NPROC=8           # processi del pool di emcee per tree
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "job ${SLURM_JOB_ID}: sets ${FIRST}-${LAST}${TAG}, ${NPAR} trees at a time, $(date '+%F %T')"
for i in $(seq "$FIRST" "$LAST"); do
    for kind in d9s4 rand; do
        echo "$i $kind"
    done
done | xargs -P "$NPAR" -n 2 bash -c \
    'bash "'"$ROOT"'/code/run_one.sh" "'"$ROOT"'" "$0" "$1" "'"$TAG"'" \
     && echo "set $0 $1: done" || echo "set $0 $1: FAILED (see logs)"'
echo "job ${SLURM_JOB_ID}: finished $(date '+%F %T')"
