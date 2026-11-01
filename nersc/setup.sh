#!/usr/bin/env bash
# setup.sh — prepara la cartella della validazione (UNA volta, nodo di login).
#   source <install>/activate_bbpower.sh
#   bash setup.sh <root> <cov tarball>
# es. bash setup.sh /pscratch/sd/k/krach/SObs/BBpower_Bsure/PySM_rnd2 /pscratch/sd/k/krach/SObs/BBpower_Bsure/cov_d1s1_hits_pg_n128_noee.tar
#
# 1. copia codice e dati in <root>, sostituisce i percorsi nei config
# 2. estrae gli spettri delle 500 sims della covarianza in <root>/cov
# 3. controlla che ci siano le mappe PySM dei 9 cieli a nside 128 (calcolate
#    sul Mac da 512 con downgrade armonico: al NERSC non serve PySM)
# 4. crea il piano (plan/sets.csv) e gli r veri (truth/r_true.csv)
set -euo pipefail

ROOT="$1"; COV_TAR="$2"
PKG="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT"

echo "==> [1/4] code and data in ${ROOT}"
cp -r "${PKG}/code" "${PKG}/data" "$ROOT"/
for f in "${ROOT}"/code/config/base_config_*.yml; do
    sed -i "s#@ROOT@#${ROOT}#g" "$f"
done
for b in LF1 LF2 MF1 MF2 UHF1 UHF2; do
    echo "${ROOT}/data/beams/beam_${b}.txt"
done > "${ROOT}/data/beams_list.txt"
for b in LF1 LF2 MF1 MF2 UHF1 UHF2; do
    echo "${ROOT}/data/bandpasses/${b}.txt"
done > "${ROOT}/data/bpass_list.txt"

echo "==> [2/4] covariance sims"
mkdir -p "${ROOT}/cov"
tar -xf "$COV_TAR" -C "${ROOT}/cov"
ls "${ROOT}"/cov/cells_all_splits_sim*.fits | sort -V > "${ROOT}/cov/cells_all_sims.txt"
echo "    $(wc -l < "${ROOT}/cov/cells_all_sims.txt") sims"

echo "==> [3/4] PySM maps of the 9 skies (nside 128 from 512, precomputed in data/pysm_cache)"
for d in d9 d10 d12; do
    for s in s4 s5 s7; do
        f="${ROOT}/data/pysm_cache/${d}${s}_nside128_from512.npy"
        [[ -f "$f" ]] || { echo "Missing $f"; exit 1; }
    done
done
echo "    ok"

echo "==> [4/4] plan"
python "${ROOT}/code/make_plan.py" "$ROOT"
echo
echo "Ready. Do NOT open ${ROOT}/truth until the end."
