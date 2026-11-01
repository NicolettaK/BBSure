#!/bin/bash
# job_profile_tree.slurm — schema "profilo" (Res_wr*, profile_tree.py) su tree
# gia' conclusi in <src> (PySM_rnd2), risultati nella cartella da cui si
# sottomette (PySM_rnd3), un processo per set, in parallelo su un nodo.
#
#   cd /pscratch/sd/k/krach/SObs/BBpower_Bsure/PySM_rnd3
#   sbatch code/job_profile_tree.slurm <src_root> "<i> <kind>" ["<i> <kind>" ...]
# es. sbatch code/job_profile_tree.slurm $SCRATCH/SObs/BBpower_Bsure/PySM_rnd2 "47 rand" "47 d9s4"
#SBATCH -A mp107b
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -J bbpower_proftree
#SBATCH -o logs/slurm_proftree_%j.out

set -uo pipefail
SRC="$1"; shift
OUT="${SLURM_SUBMIT_DIR}"

set +u
source /global/homes/k/krach/usr/activate_bbpower.sh
set -u
export BBPOWER_NPROC=8
# punti dei profili in parallelo per set (fit ML da un core ciascuno): con 4
# set, 4 x 24 = 96 processi sui 128 core del nodo
export PROFILE_NPROC=$(( 96 / $# ))
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "job ${SLURM_JOB_ID}: profile scheme for $*, $(date '+%F %T')"
for s in "$@"; do
    set -- $s
    python "${OUT}/code/profile_tree.py" "$SRC" "$OUT" "$1" "$2" \
        > "${OUT}/logs/profile_tree_${1}_${2}.log" 2>&1 \
        && echo "set $1 $2: done" || echo "set $1 $2: FAILED (see logs)" &
done
wait
echo "job ${SLURM_JOB_ID}: finished $(date '+%F %T')"
