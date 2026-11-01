#!/usr/bin/env bash
# run_bfcov.sh — second pass of a set: covariance and HL fiducial from the
# best fit of the previous tree (bbsure/covariance.py), then the tree again.
#   bash nersc/run_bfcov.sh <root> <index> <d9s4|rand> <iteration: 1, 2, ...>
# Iteration n: starts from the tree "tree" (n = 1) or "tree_bfcov<n-1>",
# writes runs/set_<i>_<kind>/bfcov<n>/ (covariance, model) and tree_bfcov<n>/.
# Uses the same data spectra (output/): no maps to redo.
set -euo pipefail

ROOT="$1"; I="$2"; KIND="$3"; IT="$4"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${BBSURE_DATA:-${REPO}/data}"
NAME=$(printf "set_%03d_%s" "$I" "$KIND")
RUN="${ROOT}/runs/${NAME}"
PREV=$([[ "$IT" == 1 ]] && echo tree || echo "tree_bfcov$((IT - 1))")
LOG="${ROOT}/logs/${NAME}_bfcov${IT}.log"

if [[ ! -f "${RUN}/${PREV}/conclusion.json" ]]; then
    echo "${NAME}: ${PREV} not concluded, skipped"
    exit 1
fi
if [[ -f "${RUN}/tree_bfcov${IT}/conclusion.json" ]]; then
    echo "${NAME}: bfcov${IT} already concluded"
    exit 0
fi

# Lock (as in run_one.sh): the same set must not run in two jobs
LOCK="${RUN}/.lock_bfcov${IT}"
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
        echo "${NAME} bfcov${IT}: running in job ${OWNER}, skipped"
        exit 0
    fi
    echo "$ME" > "$LOCK"
fi
trap 'rm -f "$LOCK"' EXIT

{
echo "==> ${NAME} bfcov${IT}: started $(date '+%F %T')"
if [[ ! -f "${RUN}/bfcov${IT}/cells_cov_bf.fits" ]]; then
    python "${REPO}/scripts/make_bf_cov.py" --tree "${RUN}/${PREV}" \
        --cells "${RUN}/output" \
        --fiducial "${DATA}/cls_fid_d1s1_hits_pg_n128.fits" \
        --out "${RUN}/bfcov${IT}"
fi
sed -e "s#@REPO@#${REPO}#g" -e "s#@DATA@#${DATA}#g" -e "s#@RUN@#${RUN}#g" \
    -e "s#cells_fiducial: .*#cells_fiducial: \"${RUN}/bfcov${IT}/cells_model_bf.fits\"\n    cells_coadded_cov: \"${RUN}/bfcov${IT}/cells_cov_bf.fits\"#" \
    -e "s#runs_dir: .*#runs_dir:      \"${RUN}/tree_bfcov${IT}\"#" \
    "${REPO}/configs/tree_config_template.yml" > "${RUN}/tree_config_bfcov${IT}.yml"
python "${REPO}/scripts/run_tree.py" --config "${RUN}/tree_config_bfcov${IT}.yml"
echo "==> ${NAME} bfcov${IT}: finished $(date '+%F %T')"
} > "$LOG" 2>&1
