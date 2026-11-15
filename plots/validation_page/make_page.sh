#!/usr/bin/env bash
# Results page of a validation run, from its root (copied from NERSC):
#   runs/set_*/tree/{decision_log.json,conclusion*.json,profile/result.json}
#   results_unblinded.csv, profile_results.csv (scripts/collect.py; for a
#   blind run results.csv instead of results_unblinded.csv: page without
#   true r and sky)
#   corners/set_*.png (optional, corner plots)
#
#   bash plots/validation_page/make_page.sh <root> <run text> <out dir>
#
# <run text>: title, byline, setup and main results of the run (see
# run_text_example.html). Writes <out dir>/index.html (standalone page) with
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

TMP="$(mktemp -d)"; mkdir -p "$TMP/review"
P="$HERE/plot_r_sets.py"
if [[ ! -f "$ROOT/results_unblinded.csv" ]]; then
    # blind run (collect.py without --unblind): only the plots that do not
    # need the true r or the sky
    echo "==> blind run: r set by set and r = 0 flag"
    python "$P" "$ROOT/tree_full.csv" --kind rand --label "Bayesian (tree)" --out "$TMP/r_sets_tree.pdf" > /dev/null
    python "$P" "$ROOT/profile_as_tree.csv" --kind rand --label "profile scheme" --out "$TMP/r_sets_profile.pdf" > /dev/null
    cp "$TMP"/r_sets_*.png "$OUT/figures/"
    python "$HERE/plot_r0flag.py" "$ROOT" "$OUT/figures/r0flag_vs_rest.png" > /dev/null
else
echo "==> summary plots"
python plots/make_profile_table.py "$ROOT/results_unblinded.csv" \
    "$ROOT/profile_results.csv" "$TMP/pvt.pdf" --title "$(basename "$ROOT")" \
    | grep -E "^d9s4|^rand" || true
python plots/plot_rtrue_vs_rest.py "$TMP/pvt.csv" --out "$OUT/figures/rtrue_vs_rest.png" > /dev/null
# kinds of sets in the run (d9s4 and / or rand)
KINDS=$(python -c "import pandas as pd; print(' '.join(sorted(set(pd.read_csv('$ROOT/results_unblinded.csv').kind))))")
for k in $KINDS; do
    python plots/plot_bias_profile.py "$TMP/pvt.csv" --kind "$k" --out "$OUT/figures/bias_$k.png" > /dev/null
done

echo "==> r set by set"
if [[ " $KINDS " == *" d9s4 "* ]]; then
python "$P" "$ROOT/tree_full.csv" --kind d9s4 --label "Bayesian (tree)" --out "$TMP/r_sets_d9s4_tree.pdf" > /dev/null
python "$P" "$ROOT/profile_as_tree.csv" --kind d9s4 --label "profile scheme" --out "$TMP/r_sets_d9s4_profile.pdf" > /dev/null
python "$P" "$ROOT/tree_full.csv" --kind rand --ref "$ROOT/first_fit.csv" --label "Bayesian (tree)" --out "$TMP/r_sets_rand_vs_d9s4_tree.pdf" > /dev/null
python "$P" "$ROOT/profile_as_tree.csv" --kind rand --ref "$ROOT/first_fit.csv" --label "profile scheme" --out "$TMP/r_sets_rand_vs_d9s4_profile.pdf" > /dev/null
else
# random skies only (e.g. a blind run after the unblinding): no d9s4 reference
python "$P" "$ROOT/tree_full.csv" --kind rand --label "Bayesian (tree)" --out "$TMP/r_sets_rand_tree.pdf" > /dev/null
python "$P" "$ROOT/profile_as_tree.csv" --kind rand --label "profile scheme" --out "$TMP/r_sets_rand_profile.pdf" > /dev/null
fi
cp "$TMP"/r_sets_*.png "$OUT/figures/"
fi
if [[ -f "$ROOT/review.csv" ]]; then
    # review of the sets fixed in the page (exported from the page): the
    # plots of r with its bands (random skies)
    echo "==> review of the sets (review.csv): plots with its bands"
    python "$P" "$ROOT/tree_full.csv" --kind rand --label "Bayesian (tree)" --review "$ROOT/review.csv" --out "$TMP/review/r_sets_tree_review.pdf" > /dev/null
    python "$P" "$ROOT/profile_as_tree.csv" --kind rand --label "profile scheme" --review "$ROOT/review.csv" --out "$TMP/review/r_sets_profile_review.pdf" > /dev/null
    cp "$TMP"/review/*.png "$OUT/figures/"
    python "$HERE/plot_r0flag.py" "$ROOT" "$OUT/figures/r0flag_vs_rest_review.png" --review "$ROOT/review.csv" > /dev/null
fi

echo "==> path of every set"
python "$HERE/paths.py" "$ROOT" "$OUT/paths"

echo "==> corner plots and flowchart"
if ls "$ROOT"/corners/*.png > /dev/null 2>&1; then cp "$ROOT"/corners/*.png "$OUT/corners/"; fi
cp docs/flowcharts/step_by_step/tree_steps.png "$OUT/flowchart.png"

echo "==> page"
python "$HERE/build_page.py" "$ROOT" "$TEXT" "$OUT/index.html" --standalone
rm -rf "$TMP"
echo "written $OUT/index.html ($(ls "$OUT/paths" | wc -l | tr -d ' ') paths, $(ls "$OUT/corners" | wc -l | tr -d ' ') corner plots)"
