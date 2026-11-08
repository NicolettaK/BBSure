"""
Path of every set through the flowchart (plots/plot_tree_path.py), in
parallel, as PNG reduced to 96 colours (enough to keep the flag badges
apart).

  python plots/validation_page/paths.py <root> <out dir> [--nproc 8]
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PLOT = os.path.join(HERE, '..', 'plot_tree_path.py')

ap = argparse.ArgumentParser()
ap.add_argument('root')
ap.add_argument('out')
ap.add_argument('--nproc', type=int, default=8)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tmp = tempfile.mkdtemp()


def one(tree):
    s = os.path.basename(os.path.dirname(tree))
    pdf = os.path.join(tmp, s + '.pdf')
    r = subprocess.run([sys.executable, PLOT, tree, '--out', pdf,
                        '--title', s.replace('_', ' ')],
                       capture_output=True, text=True)
    if r.returncode:
        return s, r.stderr[-300:]
    Image.open(pdf[:-4] + '.png').convert('RGB').quantize(
        colors=96, method=Image.Quantize.MEDIANCUT).save(
        os.path.join(a.out, s + '.png'), optimize=True)
    return s, None


trees = sorted(glob.glob(os.path.join(a.root, 'runs', 'set_*', 'tree')))
with ThreadPoolExecutor(a.nproc) as ex:
    for s, err in ex.map(one, trees):
        if err:
            print('FAILED', s, err)
print(f"{len(trees)} paths -> {a.out}")
