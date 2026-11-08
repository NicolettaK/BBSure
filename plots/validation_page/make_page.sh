#!/usr/bin/env bash
# Results page of a validation run, from its root (copied from NERSC):
#   runs/set_*/tree/{decision_log.json,conclusion*.json,profile/result.json}
#   results_unblinded.csv, profile_results.csv (scripts/collect.py)
#   corners/set_*.png (optional, corner plots)
#
#   bash plots/validation_page/make_page.sh <root> <run text> <out dir>
#
# <run text>: title, byline, setup and main results of the run (see
# runs/aic_2026-11.html). Writes <out dir>/index.html (standalone page) with
# flowchart.png, figures/, paths/ and corners/ next to it, ready for a web
# server (e.g. the NERSC portal). Run from anywhere, in the bbpower env.
set -euo pipefail
ROOT="$(cd "$1" && pwd)"; TEXT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
OUT="$3"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
mkdir -p "$OUT/figures" "$OUT/paths" "$OUT/corners"

echo "==> tables"
python "$HERE/tables.py" "$ROOT"

echo "==> summary plots"
TMP="$(mktemp -d)"
python plots/make_profile_table.py "$ROOT/results_unblinded.csv" \
    "$ROOT/profile_results.csv" "$TMP/pvt.pdf" --title "$(basename "$ROOT")" \
    | grep -E "^d9s4|^rand" || true
python plots/plot_rtrue_vs_rest.py "$TMP/pvt.csv" --out "$OUT/figures/rtrue_vs_rest.png" > /dev/null
for k in d9s4 rand; do
    python plots/plot_bias_profile.py "$TMP/pvt.csv" --kind "$k" --out "$OUT/figures/bias_$k.png" > /dev/null
done

echo "==> r set by set"
P="$HERE/plot_r_sets.py"
python "$P" "$ROOT/tree_full.csv" --kind d9s4 --label "Bayesian (tree)" --out "$TMP/r_sets_d9s4_tree.pdf" > /dev/null
python "$P" "$ROOT/profile_as_tree.csv" --kind d9s4 --label "profile scheme" --out "$TMP/r_sets_d9s4_profile.pdf" > /dev/null
python "$P" "$ROOT/tree_full.csv" --kind rand --ref "$ROOT/tree_full.csv" --label "Bayesian (tree)" --out "$TMP/r_sets_rand_vs_d9s4_tree.pdf" > /dev/null
python "$P" "$ROOT/profile_as_tree.csv" --kind rand --ref "$ROOT/profile_as_tree.csv" --label "profile scheme" --out "$TMP/r_sets_rand_vs_d9s4_profile.pdf" > /dev/null
cp "$TMP"/r_sets_*.png "$OUT/figures/"

echo "==> path of every set"
python "$HERE/paths.py" "$ROOT" "$OUT/paths"

echo "==> corner plots and flowchart"
if ls "$ROOT"/corners/*.png > /dev/null 2>&1; then cp "$ROOT"/corners/*.png "$OUT/corners/"; fi
cp docs/flowcharts/step_by_step/tree_steps.png "$OUT/flowchart.png"

echo "==> page"
python "$HERE/build_page.py" "$ROOT" "$TEXT" "$OUT/index.html" --standalone
rm -rf "$TMP"
echo "written $OUT/index.html ($(ls "$OUT/paths" | wc -l | tr -d ' ') paths, $(ls "$OUT/corners" | wc -l | tr -d ' ') corner plots)"
