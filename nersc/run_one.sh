#!/usr/bin/env bash
# run_one.sh — un set della validazione: mappe, spettri, decision tree.
#   bash run_one.sh <root> <index> <d9s4|rand> [tag]
# Output in <root>/runs/set_<index>_<kind>/ (solo l'indice nel nome: r resta
# nascosto). Se il tree e' gia' concluso non fa niente; se si era fermato
# riparte riusando i fit gia' fatti.
# Con un tag (es. _l130) usa config/tree_config_template<tag>.yml e scrive il
# tree in tree<tag>/: mappe e spettri gia' fatti vengono riusati.
set -euo pipefail

ROOT="$1"; I="$2"; KIND="$3"; TAG="${4:-}"
NAME=$(printf "set_%03d_%s" "$I" "$KIND")
RUN="${ROOT}/runs/${NAME}"
LOG="${ROOT}/logs/${NAME}${TAG}.log"
mkdir -p "$RUN" "${ROOT}/logs"

if [[ -f "${RUN}/tree${TAG}/conclusion.json" ]]; then
    echo "${NAME}${TAG}: already concluded"
    exit 0
fi

# Lucchetto: lo stesso set non deve girare in due job insieme (scriverebbero
# negli stessi run, es. lo stesso checkpoint di emcee). Il file contiene l'ID
# del job proprietario; se quel job non e' piu' in coda il set si riprende.
LOCK="${RUN}/.lock${TAG}"
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
        echo "${NAME}${TAG}: running in job ${OWNER}, skipped"
        exit 0
    fi
    echo "$ME" > "$LOCK"
fi
trap 'rm -f "$LOCK"' EXIT

# seed e cielo dal piano; r dal file della verita' (il tree non lo vede)
read -r SEED SKY_RAND < <(awk -F, -v i="$I" '$1 == i {print $2, $3}' \
                          "${ROOT}/plan/sets.csv")
R=$(awk -F, -v i="$I" '$1 == i {print $2}' "${ROOT}/truth/r_true.csv")
SKY=$([[ "$KIND" == d9s4 ]] && echo d9s4 || echo "$SKY_RAND")

{
echo "==> ${NAME}${TAG}: sky ${SKY}, seed ${SEED}, started $(date '+%F %T')"
if [[ ! -f "${RUN}/output/cells_coadded.fits" ]]; then
    # l'output del generatore contiene r: non va nel log
    # nside 128: PySM (cache in data/pysm_cache) portato da 512 a 128 nello
    # spazio armonico; segnale senza modi E (niente dispersione E -> B)
    python "${ROOT}/code/generate_maps.py" --seed "$SEED" --nside 128 \
        --r "$R" --pysm "$SKY" --pysm-nside-in 512 --no-ee \
        --hits-noise "${ROOT}/data/masks/norm_nHits_SA_35FOV_G_nside128.fits" \
        --noise pessimistic_goal --no-kludge \
        --output-dir "${RUN}/maps" > /dev/null
    echo "maps done"
    bash "${ROOT}/code/run_spectra.sh" "$ROOT" "${RUN}/maps" "${RUN}/output"
    rm -rf "${RUN}/maps"
fi
sed -e "s#@ROOT@#${ROOT}#g" -e "s#@RUN@#${RUN}#g" \
    "${ROOT}/code/config/tree_config_template${TAG}.yml" \
    > "${RUN}/tree_config${TAG}.yml"
python "${ROOT}/code/decision_tree.py" --config "${RUN}/tree_config${TAG}.yml"
echo "==> ${NAME}${TAG}: finished $(date '+%F %T')"
} > "$LOG" 2>&1
