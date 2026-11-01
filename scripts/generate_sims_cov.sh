#!/usr/bin/env bash
# generate_sims_cov.sh — Gaussian sims for the covariance of the PySM data
#
# Usage: bash scripts/generate_sims_cov.sh --sims-dir <dir> --fg-params <yml>
#            [--first 5002] [--n 500] [--hits <hits map>] [--nside 128]
#            [--noise baseline_optimistic|pessimistic_goal] [--no-kludge]
#            [--nproc 1] [--no-ee]
#
# Foregrounds: Gaussian, with the amplitudes/slopes/SEDs fitted on PySM by
# fit_fg_spectra.py; noise scaled with the (galactic) hits map, like the data.
# Existing sims are skipped. Writes <sims-dir>/sims_list.txt, the input of
# run_spectra.sh --cov-maps. As used for the validation:
#   bash scripts/generate_sims_cov.sh --nside 128 --sims-dir sims_cov_n128 \
#       --fg-params configs/fg_params_d1s1_hits_n128.yml \
#       --hits data/masks/norm_nHits_SA_35FOV_G_nside128.fits \
#       --noise pessimistic_goal --no-kludge --no-ee --nproc 8
# The environment (conda, BBSure importable) must already be active.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"

FIRST=5002
N=500
FG_PARAMS=""
HITS="${BBSURE_DATA:-${REPO}/data}/masks/norm_nHits_SA_35FOV_G_nside128.fits"
NSIDE=128
SIMS_DIR=""
NOISE=baseline_optimistic
KLUDGE=""
NOEE=""
NPROC=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --first)     FIRST="$2"; shift 2 ;;
        --n)         N="$2"; shift 2 ;;
        --fg-params) FG_PARAMS="$2"; shift 2 ;;
        --hits)      HITS="$2"; shift 2 ;;
        --nside)     NSIDE="$2"; shift 2 ;;
        --sims-dir)  SIMS_DIR="$2"; shift 2 ;;
        --noise)     NOISE="$2"; shift 2 ;;
        --no-kludge) KLUDGE="--no-kludge"; shift ;;
        --nproc)     NPROC="$2"; shift 2 ;;
        --no-ee)     NOEE="--no-ee"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$SIMS_DIR" && -n "$FG_PARAMS" ]] || { echo "--sims-dir and --fg-params are required"; exit 1; }
LAST=$((FIRST + N - 1))
mkdir -p "${SIMS_DIR}"
SIMS_DIR="$(cd "${SIMS_DIR}" && pwd)"
export OMP_NUM_THREADS=1

SIMS_LIST="${SIMS_DIR}/sims_list.txt"
mkdir -p "${SIMS_DIR}"

echo "==> Gaussian sims s${FIRST}..s${LAST} in ${SIMS_DIR} (nside ${NSIDE}, ${NPROC} at a time)"
echo "    foregrounds: ${FG_PARAMS}"
echo "    hits:        ${HITS}"
echo "    noise:       ${NOISE} ${KLUDGE} ${NOEE}"
export REPO FG_PARAMS HITS NSIDE SIMS_DIR NOISE KLUDGE NOEE
seq "${FIRST}" "${LAST}" | xargs -P "${NPROC}" -n 1 bash -c '
    outdir="${SIMS_DIR}/s$0"
    [ -f "${outdir}/obs_split4of4.fits.gz" ] && exit 0
    python "${REPO}/scripts/generate_maps.py" --seed "$0" \
        --nside "${NSIDE}" --nsplits 4 --r 0.0 --fg-params "${FG_PARAMS}" \
        --hits-noise "${HITS}" --noise "${NOISE}" ${KLUDGE} ${NOEE} \
        --output-dir "${outdir}" > /dev/null || echo "seed $0 FAILED"'

> "${SIMS_LIST}"
for seed in $(seq "${FIRST}" "${LAST}"); do
    echo "${SIMS_DIR}/s${seed}" >> "${SIMS_LIST}"
done
echo "==> ${SIMS_LIST}: $(wc -l < "${SIMS_LIST}") sims"
