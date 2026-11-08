"""
Degeneracy between the moment amplitudes and r, for a few sets. For each
set: posterior of r (tree, model with the moment at r free, profile scheme)
and the planes r - A^d_{10,10}, r - A^s_{10,10}.

  python plots/plot_degeneracy.py <src> <table.csv> <out.png> 76:rand:M_d+s-... 97:rand:M_d+d2+s-...

src: root of the run (runs/set_<i>_<kind>/tree and tree/profile);
table.csv: the csv written by make_profile_table.py (true r and sky).
"""
import json
import os
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import corner  # noqa: E402

from bbsure.config import free_param_names  # noqa: E402
from bbsure.runner import equal_weights_file  # noqa: E402

src, table, out, specs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
C_B, C_P, C_X = '#2a6fdb', '#e8590c', '#2f9e44'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15})
truth = {r['index'] + r['kind']: (float(r['r_true']), r['sky']) for r in
         __import__('csv').DictReader(open(table))}


def pc(d):
    cfg = yaml.safe_load(open(os.path.join(d, 'config.yml')))
    names = free_param_names(cfg['BBCompSep'])
    s = np.loadtxt(equal_weights_file(d), ndmin=2)[:, 2:]
    return names, s


fig, axes = plt.subplots(len(specs), 3, figsize=(18, 5.2 * len(specs)),
                         squeeze=False)
kw = dict(bins=30, plot_datapoints=False, plot_density=False,
          levels=(0.393, 0.865), smooth=1.0, no_fill_contours=True)
for row, spec in enumerate(specs):
    i, kind, mx = spec.split(':')
    I = f'{int(i):03d}'
    tree = os.path.join(src, 'runs', f'set_{I}_{kind}', 'tree')
    prof = os.path.join(tree, 'profile')
    c = json.load(open(os.path.join(tree, 'conclusion.json')))
    nb, sb = pc(os.path.join(tree, f"{c['model']}_PC"))
    nx, sx = pc(os.path.join(tree, f'{mx}_PC'))
    res = json.load(open(os.path.join(prof, 'result.json')))
    pz = np.load(os.path.join(prof, f"profile_{res['model_final']}.npz"))
    fine = np.linspace(pz['r'][0], pz['r'][-1], 2001)
    d = np.interp(fine, pz['r'], pz['chi2'])
    post = np.exp(-(d - d.min()) / 2)
    rt, sky = truth[str(int(i)) + kind]
    rb, rx = 1e3 * sb[:, nb.index('r_tensor')], 1e3 * sx[:, nx.index('r_tensor')]
    lo = min(np.percentile(rb, 0.5), np.percentile(rx, 0.5), 1e3 * rt) - 2
    hi = max(np.percentile(rb, 99.5), np.percentile(rx, 99.5), 1e3 * rt) + 2
    ax = axes[row, 0]
    ax.hist(rb, bins=40, range=(lo, hi), density=True, histtype='step',
            color=C_B, lw=2, label='tree model (no new moment)')
    ax.hist(rx, bins=40, range=(lo, hi), density=True, histtype='step',
            color=C_X, lw=2, label='+ moment, r free')
    ax.plot(1e3 * fine, post / np.trapezoid(post, 1e3 * fine), color=C_P,
            lw=2, label='profile scheme')
    ax.axvline(1e3 * rt, color='k', lw=1.5, label='true r')
    ax.set_xlim(lo, hi)
    ax.set_yticks([])
    ax.set_xlabel(r'$r\ [10^{-3}]$')
    ax.set_title(f'set {i} {kind} ({sky})', loc='left', fontsize=16)
    if row == 0:
        ax.legend(frameon=False, fontsize=12, loc='upper left')
    for k, (p, lab) in enumerate([('A_d_w10w10', r'$A^{d}_{10,10}$ (dust)'),
                                  ('A_s_w10w10', r'$A^{s}_{10,10}$ (synch)')]):
        ax = axes[row, k + 1]
        for n, s, col in ((nb, sb, C_B), (nx, sx, C_X)):
            if p in n:
                corner.hist2d(1e3 * s[:, n.index('r_tensor')], s[:, n.index(p)],
                              ax=ax, color=col, **kw)
        ax.axvline(1e3 * rt, color='k', lw=1.5)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(r'$r\ [10^{-3}]$')
        ax.set_ylabel(lab)
fig.tight_layout()
fig.savefig(out, dpi=110)
print('written', out)
