#!/usr/bin/env bash
# run_one.sh — one set of the validation: maps, spectra, decision tree and
# profile scheme.
#   bash nersc/run_one.sh <root> <index> <d9s4|rand>
# Output in <root>/runs/set_<index>_<kind>/ (only the index in the name: r
# stays hidden). If the tree is already concluded it does nothing; if it had
# stopped, it restarts reusing the fits already done.
# Variant (environment), e.g. the single-moment tree on the same spectra:
#   TREE_TEMPLATE  tree config template in configs/ (default
#                  tree_config_template.yml; e.g. tree_config_single.yml)
#   TREE_DIR       tree directory in the set (default tree; e.g. tree_single)
set -euo pipefail

ROOT="$1"; I="$2"; KIND="$3"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${BBSURE_DATA:-${REPO}/data}"
NAME=$(printf "set_%03d_%s" "$I" "$KIND")
RUN="${ROOT}/runs/${NAME}"
TEMPLATE="${TREE_TEMPLATE:-tree_config_template.yml}"
TDIR="${TREE_DIR:-tree}"
SFX=$([[ "$TDIR" == tree ]] && echo "" || echo "_${TDIR}")
LOG="${ROOT}/logs/${NAME}${SFX}.log"
mkdir -p "$RUN" "${ROOT}/logs"

if [[ -f "${RUN}/${TDIR}/conclusion.json" ]]; then
    echo "${NAME}${SFX}: already concluded"
    exit 0
fi

# Lock: the same set must not run in two jobs at once (they would write in
# the same runs, e.g. the same emcee checkpoint). The file holds the ID of
# the owner job; if that job is no longer in the queue the set is taken over.
LOCK="${RUN}/.lock${SFX}"
ME="${SLURM_JOB_ID:-pid$$}"
if ! ( set -o noclobber; echo "$ME" > "$LOCK" ) 2> /dev/null; then
    OWNER=$(cat "$LOCK" 2> /dev/null || true)
    ALIVE=""
    if [[ "$OWNER" == pid* ]]; then
        kill -0 "${OWNER#pid}" 2> /dev/null && ALIVE=1
    elif [[ -n "$OWNER" ]]; then
        [[ -n "$(squeue -h -j "$OWNER" 2> /dev/null)" ]] && ALIVE=1
    fi
    if [[ -n "$ALIVE" && "$OWNER" != "$ME" ]]; then
        echo "${NAME}: running in job ${OWNER}, skipped"
        exit 0
    fi
    echo "$ME" > "$LOCK"
fi
trap 'rm -f "$LOCK"' EXIT

# seed and sky from the plan; r from the truth file (the tree does not see
# it). \r removed: plans written by the older make_plan.py end their lines
# with \r\n, which left a \r on the sky name (the PySM cache was never found)
read -r SEED SKY_RAND < <(tr -d '\r' < "${ROOT}/plan/sets.csv" | \
                          awk -F, -v i="$I" '$1 == i {print $2, $3}')
R=$(tr -d '\r' < "${ROOT}/truth/r_true.csv" | \
    awk -F, -v i="$I" '$1 == i {print $2}')
SKY=$([[ "$KIND" == d9s4 ]] && echo d9s4 || echo "$SKY_RAND")

{
echo "==> ${NAME}${SFX}: sky ${SKY}, seed ${SEED}, template ${TEMPLATE}, started $(date '+%F %T')"
# spectra made elsewhere (e.g. on a laptop) and copied gzipped
for f in cells_coadded cells_noise; do
    if [[ ! -f "${RUN}/output/${f}.fits" && -f "${RUN}/output/${f}.fits.gz" ]]; then
        gunzip "${RUN}/output/${f}.fits.gz"
    fi
done
if [[ ! -f "${RUN}/output/cells_coadded.fits" ]]; then
    # the output of the generator contains r: it does not go in the log
    # nside 128: PySM (cache in pysm_cache) brought from 512 to 128 in
    # harmonic space; signal without E modes (no E -> B leakage)
    python "${REPO}/scripts/generate_maps.py" --seed "$SEED" --nside 128 \
        --r "$R" --pysm "$SKY" --pysm-nside-in 512 --no-ee \
        --hits-noise "${DATA}/masks/norm_nHits_SA_35FOV_G_nside128.fits" \
        --noise pessimistic_goal --no-kludge \
        --output-dir "${RUN}/maps" > /dev/null
    echo "maps done"
    if [[ -d "${ROOT}/mcm_cache" && -d "${ROOT}/cov_ref" ]]; then
        # what does not depend on the data, computed once (as on the laptop,
        # PySM_tests/BBSure_s9000/make_set.sh): the NaMaster coupling
        # matrices (reused by BBPowerSpecter) and the covariance of the 500
        # sims (scripts/fix_cov.py save); the Summarizer reads only 2 sims
        mkdir -p "${RUN}/output"
        cp "${ROOT}"/mcm_cache/mcm_*.fits "${RUN}/output/"
        head -2 "${ROOT}/cov/cells_all_sims.txt" > "${RUN}/output/cells_2sims.txt"
        bash "${REPO}/scripts/run_spectra.sh" --maps "${RUN}/maps" \
            --out "${RUN}/output" --cov-cells "${RUN}/output/cells_2sims.txt"
        python "${REPO}/scripts/fix_cov.py" put "${ROOT}/cov_ref" \
            "${RUN}/output/cells_coadded.fits" "${RUN}/output/cells_noise.fits"
    else
        bash "${REPO}/scripts/run_spectra.sh" --maps "${RUN}/maps" \
            --out "${RUN}/output" --cov-cells "${ROOT}/cov/cells_all_sims.txt"
    fi
    echo "spectra done"
    rm -rf "${RUN}/maps"
fi
sed -e "s#@REPO@#${REPO}#g" -e "s#@DATA@#${DATA}#g" -e "s#@RUN@#${RUN}#g" \
    -e "s#runs_dir: .*#runs_dir:      \"${RUN}/${TDIR}\"#" \
    "${REPO}/configs/${TEMPLATE}" > "${RUN}/tree_config${SFX}.yml"
python "${REPO}/scripts/run_tree.py" --config "${RUN}/tree_config${SFX}.yml"
echo "==> ${NAME}${SFX}: finished $(date '+%F %T')"
} > "$LOG" 2>&1
