#!/usr/bin/env bash
# run_spectra.sh — data vector of one data set (BBPowerSpecter +
# BBPowerSummarizer), with the covariance from simulations.
#
#   bash scripts/run_spectra.sh --maps <dir with obs_split*of4.fits.gz> \
#        --out <dir> (--cov-cells <list> | --cov-maps <list>) \
#        [--mask <apodized mask>] [--config <spectra config>]
#
#   --cov-cells  list of the spectra of the covariance sims already computed
#                (cells_all_sims.txt, e.g. from a previous --cov-maps run)
#   --cov-maps   list of the directories of the covariance sims (maps): their
#                spectra are computed here and listed in <out>/cells_all_sims.txt
#   --mask       default data/masks/mask_hits_C2_7deg_nside128.fits.gz
#   --config     default configs/base_config_spectra.yml (@DATA@ is filled in)
#
# The environment (conda, BBPower importable) must already be active. Data
# files from BBSURE_DATA or the data/ directory of the repository.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${BBSURE_DATA:-${REPO}/data}"
MAPS=""; OUT=""; COV_CELLS=""; COV_MAPS=""
MASK="${DATA}/masks/mask_hits_C2_7deg_nside128.fits.gz"
CONFIG="${REPO}/configs/base_config_spectra.yml"
while [[ $# -gt 0 ]]; do
    case $1 in
        --maps)      MAPS="$2"; shift 2 ;;
        --out)       OUT="$2"; shift 2 ;;
        --cov-cells) COV_CELLS="$2"; shift 2 ;;
        --cov-maps)  COV_MAPS="$2"; shift 2 ;;
        --mask)      MASK="$2"; shift 2 ;;
        --config)    CONFIG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -n "$MAPS" && -n "$OUT" ]] || { echo "--maps and --out are required"; exit 1; }
[[ -n "$COV_CELLS" || -n "$COV_MAPS" ]] || { echo "one of --cov-cells / --cov-maps is required"; exit 1; }

# absolute paths: they end up in cells_all_sims.txt, read from other directories
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
MAPS="$(cd "$MAPS" && pwd)"

sed "s#@DATA@#${DATA}#g" "$CONFIG" > "${OUT}/config_spectra.yml"
for b in LF1 LF2 MF1 MF2 UHF1 UHF2; do
    echo "${DATA}/bandpasses/${b}.txt"
done > "${OUT}/bpass_list.txt"
for b in LF1 LF2 MF1 MF2 UHF1 UHF2; do
    echo "${DATA}/beams/beam_${b}.txt"
done > "${OUT}/beams_list.txt"

SPLITS_LIST="${OUT}/splits_list.txt"
> "$SPLITS_LIST"
for i in 1 2 3 4; do
    f="${MAPS}/obs_split${i}of4.fits.gz"
    [[ -f "$f" ]] || { echo "Missing $f"; exit 1; }
    echo "$f" >> "$SPLITS_LIST"
done

# with --cov-cells Specter runs on the data only (empty sims list) and
# Summarizer takes the spectra of the sims from the list
if [[ -n "$COV_CELLS" ]]; then
    SPECTER_SIMS="${OUT}/sims_list_empty.txt"
    > "$SPECTER_SIMS"
    CELLS_SIMS="$COV_CELLS"
else
    SPECTER_SIMS="$COV_MAPS"
    CELLS_SIMS="${OUT}/cells_all_sims.txt"
fi

python -m bbpower BBPowerSpecter \
    --splits_list="$SPLITS_LIST" \
    --masks_apodized="$MASK" \
    --bandpasses_list="${OUT}/bpass_list.txt" \
    --sims_list="$SPECTER_SIMS" \
    --beams_list="${OUT}/beams_list.txt" \
    --cells_all_splits="${OUT}/cells_all_splits.fits" \
    --cells_all_sims="${OUT}/cells_all_sims.txt" \
    --mcm="${OUT}/mcm.dum" \
    --config="${OUT}/config_spectra.yml"

python -m bbpower BBPowerSummarizer \
    --splits_list="$SPLITS_LIST" \
    --bandpasses_list="${OUT}/bpass_list.txt" \
    --cells_all_splits="${OUT}/cells_all_splits.fits" \
    --cells_all_sims="$CELLS_SIMS" \
    --cells_coadded_total="${OUT}/cells_coadded_total.fits" \
    --cells_coadded="${OUT}/cells_coadded.fits" \
    --cells_noise="${OUT}/cells_noise.fits" \
    --cells_null="${OUT}/cells_null.fits" \
    --config="${OUT}/config_spectra.yml"

# large files the tree does not use
rm -f "${OUT}/cells_null.fits" "${OUT}"/mcm_*.fits
