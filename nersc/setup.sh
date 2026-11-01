#!/usr/bin/env bash
# setup.sh — prepares the directory of a validation run (ONCE, login node).
#   bash nersc/setup.sh <root> <covariance tarball>
#
# 1. extracts the spectra of the covariance sims in <root>/cov
#    (cells_all_splits_sim*.fits, from run_spectra.sh --cov-maps)
# 2. checks the PySM maps of the 9 skies at nside 128 (computed from 512 with
#    the harmonic downgrade, in $BBSURE_DATA/pysm_cache: PySM is not run here)
# 3. creates the plan (plan/sets.csv) and the true r (truth/r_true.csv)
# The environment (conda, BBPower and BBSure importable) must be active.
set -euo pipefail

ROOT="$1"; COV_TAR="$2"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${BBSURE_DATA:-${REPO}/data}"
mkdir -p "$ROOT"

echo "==> [1/3] covariance sims"
mkdir -p "${ROOT}/cov"
tar -xf "$COV_TAR" -C "${ROOT}/cov"
ls "${ROOT}"/cov/cells_all_splits_sim*.fits | sort -V > "${ROOT}/cov/cells_all_sims.txt"
echo "    $(wc -l < "${ROOT}/cov/cells_all_sims.txt") sims"

echo "==> [2/3] PySM maps of the 9 skies (nside 128 from 512, in ${DATA}/pysm_cache)"
for d in d9 d10 d12; do
    for s in s4 s5 s7; do
        f="${DATA}/pysm_cache/${d}${s}_nside128_from512.npy"
        [[ -f "$f" ]] || { echo "Missing $f"; exit 1; }
    done
done
echo "    ok"

echo "==> [3/3] plan"
python "${REPO}/scripts/make_plan.py" "$ROOT"
echo
echo "Ready. Do NOT open ${ROOT}/truth until the end."
