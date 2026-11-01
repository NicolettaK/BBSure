#!/usr/bin/env bash
# run_one.sh — one set of the validation: maps, spectra, decision tree and
# profile scheme.
#   bash nersc/run_one.sh <root> <index> <d9s4|rand>
# Output in <root>/runs/set_<index>_<kind>/ (only the index in the name: r
# stays hidden). If the tree is already concluded it does nothing; if it had
# stopped, it restarts reusing the fits already done.
set -euo pipefail

ROOT="$1"; I="$2"; KIND="$3"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${BBSURE_DATA:-${REPO}/data}"
NAME=$(printf "set_%03d_%s" "$I" "$KIND")
RUN="${ROOT}/runs/${NAME}"
LOG="${ROOT}/logs/${NAME}.log"
mkdir -p "$RUN" "${ROOT}/logs"

if [[ -f "${RUN}/tree/conclusion.json" ]]; then
    echo "${NAME}: already concluded"
    exit 0
fi

# Lock: the same set must not run in two jobs at once (they would write in
# the same runs, e.g. the same emcee checkpoint). The file holds the ID of
# the owner job; if that job is no longer in the queue the set is taken over.
LOCK="${RUN}/.lock"
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

# seed and sky from the plan; r from the truth file (the tree does not see it)
read -r SEED SKY_RAND < <(awk -F, -v i="$I" '$1 == i {print $2, $3}' \
                          "${ROOT}/plan/sets.csv")
R=$(awk -F, -v i="$I" '$1 == i {print $2}' "${ROOT}/truth/r_true.csv")
SKY=$([[ "$KIND" == d9s4 ]] && echo d9s4 || echo "$SKY_RAND")

{
echo "==> ${NAME}: sky ${SKY}, seed ${SEED}, started $(date '+%F %T')"
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
    bash "${REPO}/scripts/run_spectra.sh" --maps "${RUN}/maps" \
        --out "${RUN}/output" --cov-cells "${ROOT}/cov/cells_all_sims.txt"
    rm -rf "${RUN}/maps"
fi
sed -e "s#@REPO@#${REPO}#g" -e "s#@DATA@#${DATA}#g" -e "s#@RUN@#${RUN}#g" \
    "${REPO}/configs/tree_config_template.yml" > "${RUN}/tree_config.yml"
python "${REPO}/scripts/run_tree.py" --config "${RUN}/tree_config.yml"
echo "==> ${NAME}: finished $(date '+%F %T')"
} > "$LOG" 2>&1
