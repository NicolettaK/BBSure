#!/bin/bash
# Renders tree_steps.html (formulas with KaTeX from the CDN, so it needs the
# network) to tree_steps.png with headless Chrome.
#   bash mkpng.sh [height in px, default 2580]
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$CHROME" --headless --disable-gpu --hide-scrollbars --virtual-time-budget=8000 \
    --force-device-scale-factor=1 --window-size=2220,${1:-2580} \
    --screenshot="$PWD/tree_steps.png" "file://$PWD/tree_steps.html" 2>/dev/null
echo "written $PWD/tree_steps.png"
