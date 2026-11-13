#!/usr/bin/env bash
# run_fgonly.sh — decision tree and profile scheme on an Asimov data set of
# one sky: PySM foregrounds alone (no CMB, no noise realization) + the mean
# lensing spectrum (A_lens = 1), with the covariance, the noise spectra and
# the H&L fiducial of a validation run. Shows which model the tree picks and
# the bias of r on average for that sky.
#   bash nersc/run_fgonly.sh <root of a validation run> <sky, e.g. d10s5>
# Output in <root>/fgonly/<sky>/ (output/: spectra, tree/: fits and
# conclusion). Interrupted runs restart reusing the fits already done.
# The noise spectra (the offset of the H&L likelihood) are the mean of the
# sets of the run, <root>/fgonly/cells_noise_mean.fits (made by
# job_fgonly.slurm): the maps have no noise, their own estimate is ~0.
set -euo pipefail
ROOT="$1"; SKY="$2"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${BBSURE_DATA:-${REPO}/data}"
RUN="${ROOT}/fgonly/${SKY}"
LOG="${ROOT}/logs/fgonly_${SKY}.log"
mkdir -p "$RUN" "${ROOT}/logs"
if [[ -f "${RUN}/tree/conclusion.json" ]]; then
    echo "fgonly ${SKY}: already concluded"
    exit 0
fi
{
echo "==> fgonly ${SKY}: started $(date '+%F %T')"
if [[ ! -f "${RUN}/output/cells_coadded.fits" ]]; then
    python "${REPO}/scripts/generate_maps.py" --seed 1 --nside 128 --r 0 \
        --pysm "$SKY" --pysm-nside-in 512 --no-ee --no-cmb --no-noise \
        --hits-noise "${DATA}/masks/norm_nHits_SA_35FOV_G_nside128.fits" \
        --noise pessimistic_goal --no-kludge \
        --output-dir "${RUN}/maps" > /dev/null
    echo "maps done"
    mkdir -p "${RUN}/output"
    cp "${ROOT}"/mcm_cache/mcm_*.fits "${RUN}/output/"
    head -2 "${ROOT}/cov/cells_all_sims.txt" > "${RUN}/output/cells_2sims.txt"
    bash "${REPO}/scripts/run_spectra.sh" --maps "${RUN}/maps" \
        --out "${RUN}/output" --cov-cells "${RUN}/output/cells_2sims.txt"
    python "${REPO}/scripts/fix_cov.py" put "${ROOT}/cov_ref" \
        "${RUN}/output/cells_coadded.fits" "${RUN}/output/cells_noise.fits"
    python "${REPO}/scripts/add_lensing.py" "${RUN}/output/cells_coadded.fits"
    echo "spectra done"
    rm -rf "${RUN}/maps"
fi
if [[ ! -f "${RUN}/output/.noise_mean" ]]; then
    # mean noise of the run (also for spectra made before this fix: their
    # fits used the noise estimated from the maps, ~0, and are redone)
    cp "${ROOT}/fgonly/cells_noise_mean.fits" "${RUN}/output/cells_noise.fits"
    rm -rf "${RUN}/tree"
    touch "${RUN}/output/.noise_mean"
    echo "noise: mean of the sets of the run"
fi
sed -e "s#@REPO@#${REPO}#g" -e "s#@DATA@#${DATA}#g" -e "s#@RUN@#${RUN}#g" \
    "${REPO}/configs/tree_config_template.yml" > "${RUN}/tree_config.yml"
python "${REPO}/scripts/run_tree.py" --config "${RUN}/tree_config.yml"
echo "==> fgonly ${SKY}: finished $(date '+%F %T')"
} > "$LOG" 2>&1
