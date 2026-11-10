"""
Estimated r (median +- sigma) vs true r, Bayesian part of the tree and
profile scheme, for the two sky types (rows) and the two estimates
(columns).

  python plots/plot_rtrue_vs_rest.py profile_vs_tree.csv --out rtrue_vs_rest.pdf

The csv is the one written by make_profile_table.py. Colour = conclusion of
that estimate (detection / upper limit); line: weighted fit
r_estimated = a + b r_true.
"""
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('csv')
ap.add_argument('--out', required=True)
a = ap.parse_args()

rows = list(csv.DictReader(open(a.csv)))
INK = '#5a5a5a'
C_DET, C_UL = '#2a6fdb', '#d9822b'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.edgecolor': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK})
kinds = [k for k in ('d9s4', 'rand') if any(r['kind'] == k for r in rows)]
fig, axes = plt.subplots(len(kinds), 2, figsize=(12, 5.2 * len(kinds)),
                         squeeze=False, sharex=True, sharey=True)
# x up to the largest true r, y over the estimates (+- sigma) of both
# methods: no empty corner when the estimates go above the true r
xmax = 1e3 * max(float(r['r_true']) for r in rows) * 1.05 + 1
est = [(float(r[f'r_med_{t}']), float(r[f'r_std_{t}']))
       for r in rows for t in ('bayes', 'profile')]
ymin = min(-2., 1e3 * min(m - s for m, s in est) - 1)
ymax = 1e3 * max(m + s for m, s in est) * 1.03 + 1
lim = max(xmax, ymax)
for i, kind in enumerate(kinds):
    sel = [r for r in rows if r['kind'] == kind]
    t = 1e3 * np.array([float(r['r_true']) for r in sel])
    for j, (tag, lab) in enumerate([('bayes', 'Bayesian (tree)'),
                                    ('profile', 'profile scheme')]):
        ax = axes[i, j]
        m = 1e3 * np.array([float(r[f'r_med_{tag}']) for r in sel])
        s = 1e3 * np.array([float(r[f'r_std_{tag}']) for r in sel])
        det = np.array([r[f'concl_{tag}'] == 'r_detection' for r in sel])
        for mask, col, name in ((det, C_DET, 'r detection'),
                                (~det, C_UL, 'upper limit')):
            ax.errorbar(t[mask], m[mask], yerr=s[mask], fmt='o', ms=4,
                        color=col, ecolor=col, elinewidth=0.8, alpha=0.8,
                        label=f'{name} ({mask.sum()})')
        ax.plot([-5, lim], [-5, lim], color='#1b1b1b', lw=1, zorder=0,
                label='estimated = true')
        # weighted fit m = a + b t
        w = 1 / s ** 2
        A = np.vstack([np.ones_like(t), t]).T
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        a0, b0 = cov @ (A.T @ (w * m))
        xx = np.linspace(0, lim, 50)
        ax.plot(xx, a0 + b0 * xx, color='#c0392b', lw=1.5, ls='--',
                label=f'fit: a = {a0:+.2f} ± {np.sqrt(cov[0, 0]):.2f}, '
                      f'b = {b0:.2f} ± {np.sqrt(cov[1, 1]):.2f}')
        bias = (m - t) / s
        ax.set_title(f'{kind}, {lab}: {len(sel)} sets — bias '
                     f'{bias.mean():+.2f} ± {bias.std(ddof=1) / np.sqrt(len(bias)):.2f}',
                     fontsize=10.5, color='#1b1b1b', loc='left')
        ax.set_xlim(-1, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(color='#e6e6e6', lw=0.6)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc='upper left')
        if i == len(kinds) - 1:
            ax.set_xlabel(r'$r_{\rm true}\ [10^{-3}]$')
        if j == 0:
            ax.set_ylabel(r'$r$ estimated (median $\pm\,\sigma$) $[10^{-3}]$')
fig.text(0.99, 0.005, 'colour: conclusion of that estimate; fit weighted '
         'with 1/σ²', ha='right', fontsize=8, color=INK)
fig.tight_layout()
fig.savefig(a.out, bbox_inches='tight')
fig.savefig(a.out.rsplit('.', 1)[0] + '.png', dpi=130, bbox_inches='tight')
print('written', a.out)
