#!/usr/bin/env bash
# run_spectra.sh — data vector di un set (BBPowerSpecter + BBPowerSummarizer),
# con la covarianza delle 500 sims d1s1 gia' calcolate (cov/).
#   bash run_spectra.sh <root> <data dir con obs_split*of4.fits.gz> <out dir>
# L'ambiente (conda, PYTHONPATH di BBPower) deve essere gia' attivo.
set -euo pipefail

ROOT="$1"; DATA_DIR="$2"; OUT="$3"
CONFIG="${ROOT}/code/config/base_config_spectra.yml"
mkdir -p "$OUT"

SPLITS_LIST="${OUT}/splits_list.txt"
> "$SPLITS_LIST"
for i in 1 2 3 4; do
    f="${DATA_DIR}/obs_split${i}of4.fits.gz"
    [[ -f "$f" ]] || { echo "Missing $f"; exit 1; }
    echo "$f" >> "$SPLITS_LIST"
done
> "${OUT}/sims_list_empty.txt"

python -m bbpower BBPowerSpecter \
    --splits_list="$SPLITS_LIST" \
    --masks_apodized="${ROOT}/data/masks/mask_hits_C2_7deg_nside128.fits.gz" \
    --bandpasses_list="${ROOT}/data/bpass_list.txt" \
    --sims_list="${OUT}/sims_list_empty.txt" \
    --beams_list="${ROOT}/data/beams_list.txt" \
    --cells_all_splits="${OUT}/cells_all_splits.fits" \
    --cells_all_sims="${OUT}/cells_all_sims.txt" \
    --mcm="${OUT}/mcm.dum" \
    --config="$CONFIG"

python -m bbpower BBPowerSummarizer \
    --splits_list="$SPLITS_LIST" \
    --bandpasses_list="${ROOT}/data/bpass_list.txt" \
    --cells_all_splits="${OUT}/cells_all_splits.fits" \
    --cells_all_sims="${ROOT}/cov/cells_all_sims.txt" \
    --cells_coadded_total="${OUT}/cells_coadded_total.fits" \
    --cells_coadded="${OUT}/cells_coadded.fits" \
    --cells_noise="${OUT}/cells_noise.fits" \
    --cells_null="${OUT}/cells_null.fits" \
    --config="$CONFIG"

# file grandi che il tree non usa
rm -f "${OUT}/cells_null.fits" "${OUT}"/mcm_*.fits
