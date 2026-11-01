#!/usr/bin/env bash
# install_nersc.sh — conda environment for BBPower + BBSure on Perlmutter.
#
# Usage (from a login node, inside the BBSure repository):
#   bash nersc/install_nersc.sh /global/common/software/<project>/$USER/bbsure
# The argument is where to put the conda environment, BBPower and
# PolyChordLite (NERSC recommends /global/common/software for environments:
# faster to load on the compute nodes). Each step prints what it does; if
# one fails the script stops and can be restarted from there.

set -euo pipefail

PREFIX="${1:?usage: bash nersc/install_nersc.sh <install dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
ENV_DIR="${PREFIX}/envs/bbsure"
BBPOWER_URL="${BBPOWER_URL:-https://github.com/NicolettaK/BBPower.git}"
BBPOWER_BRANCH="${BBPOWER_BRANCH:-Bsure}"
mkdir -p "$PREFIX"

echo "==> [1/7] modules"
module load python                 # NERSC conda
module load PrgEnv-gnu cray-mpich  # GNU compilers + Cray MPI (cc, CC, ftn)

echo "==> [2/7] conda environment in ${ENV_DIR}"
if [[ ! -d "$ENV_DIR" ]]; then
    conda env create -p "$ENV_DIR" -f "${HERE}/environment_nersc.yml"
else
    # environment already there (also from a failed attempt): complete it
    conda env update -p "$ENV_DIR" -f "${HERE}/environment_nersc.yml"
fi
set +u; conda activate "$ENV_DIR"; set -u

echo "==> [3/7] mpi4py against Cray MPICH"
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir \
    --no-binary=mpi4py mpi4py

echo "==> [4/7] PolyChordLite with MPI (Cray compiler wrappers)"
if [[ ! -d "${PREFIX}/PolyChordLite" ]]; then
    git clone https://github.com/PolyChord/PolyChordLite.git \
        "${PREFIX}/PolyChordLite"
fi
cd "${PREFIX}/PolyChordLite"
make clean > /dev/null 2>&1 || true
export MPI=1 FC=ftn CC=cc CXX=CC LD=ftn
make -e libchord.so
pip install --force-reinstall --no-deps --no-build-isolation .
cd "$HERE"

echo "==> [5/7] BBPower (${BBPOWER_URL}, branch ${BBPOWER_BRANCH})"
if [[ ! -d "${PREFIX}/BBPower" ]]; then
    git clone -b "$BBPOWER_BRANCH" "$BBPOWER_URL" "${PREFIX}/BBPower"
fi

echo "==> [6/7] BBSure (editable install of ${REPO})"
pip install --no-deps -e "$REPO"

echo "==> [7/7] activation file"
cat > "${PREFIX}/activate_bbsure.sh" <<EOT
# source ${PREFIX}/activate_bbsure.sh
module load python PrgEnv-gnu cray-mpich
conda activate ${ENV_DIR}
export PYTHONPATH=${PREFIX}/BBPower:\${PYTHONPATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# data directory of BBSure (bandpasses, beams, masks, fiducial, pysm_cache)
export BBSURE_DATA=\${BBSURE_DATA:-${REPO}/data}
# PySM templates already on CFS (no downloads from the compute nodes)
export PYSM_LOCAL_DATA=/global/cfs/cdirs/cmb/www/pysm-data
EOT

echo
echo "Done. To use it:"
echo "  source ${PREFIX}/activate_bbsure.sh"
echo "Then check the installation on a compute node:"
echo "  salloc -N 1 -C cpu -q interactive -t 00:15:00 -A <project>"
echo "  source ${PREFIX}/activate_bbsure.sh"
echo "  python ${HERE}/test_install.py && srun -n 4 python ${HERE}/test_install.py --mpi"
