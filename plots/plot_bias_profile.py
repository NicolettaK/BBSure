"""
Histograms of the bias (r_med - r_true)/sigma(r): profile scheme (bars) vs
Bayesian part of the tree (outline), for one sky type; all the sets, upper
limits and detections (conclusion of the profile scheme).

  python plots/plot_bias_profile.py profile_vs_tree.csv --kind rand \
      --out bias_profile_rand.pdf

The csv is the one written by make_profile_table.py (bias_bayes,
bias_profile, concl_profile).
"""
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('csv')
ap.add_argument('--kind', default='rand')
ap.add_argument('--out', required=True)
a = ap.parse_args()

rows = [r for r in csv.DictReader(open(a.csv)) if r['kind'] == a.kind]

INK = '#5a5a5a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.edgecolor': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK})
fig, axes = plt.subplots(1, 3, figsize=(19, 4.8), sharey=True)
bins = np.arange(-4, 4.01, 0.5)
x = np.linspace(-4, 4, 400)
# colour of the bars and lighter version for the fitted Gaussian
for ax, (concl, lab, col, dark) in zip(axes, [
        (None, 'all', '#3a9e6f', '#86d9b0'),
        ('upper_limit', 'upper limits', '#d9822b', '#f5bd82'),
        ('r_detection', 'r detections', '#2a6fdb', '#8db6f7')]):
    sel = [r for r in rows if concl is None or r['concl_profile'] == concl]
    if len(sel) < 2:
        ax.set_title(f'{a.kind}, {lab}: {len(sel)} sets', loc='left')
        continue
    b0 = np.array([float(r['bias_bayes']) for r in sel])
    b0 = b0[np.isfinite(b0)]   # no r of the tree at max_iterations
    b1 = np.array([float(r['bias_profile']) for r in sel])
    ax.hist(b1, bins=bins, color=col, edgecolor='white', lw=1.5,
            label=f'profile scheme: mean {b1.mean():+.2f}, '
                  f'std {b1.std(ddof=1):.2f}')
    ax.hist(b0, bins=bins, histtype='step', color='#1b1b1b', lw=1.4,
            label=f'Bayesian (tree): mean {b0.mean():+.2f}, '
                  f'std {b0.std(ddof=1):.2f}'
                  + (f' ({len(sel) - len(b0)} without r)'
                     if len(b0) < len(sel) else ''))
    norm = len(sel) * (bins[1] - bins[0])
    ax.plot(x, norm * stats.norm.pdf(x), color='#1b1b1b', lw=2,
            label=r'$\mathcal{N}(0, 1)$')
    m, sd = b1.mean(), b1.std(ddof=1)
    # thin dark edge: the light line shows also over the bars
    ax.plot(x, norm * stats.norm.pdf(x, m, sd), color=dark, lw=2.6,
            ls='--', label=rf'$\mathcal{{N}}({m:+.2f}, {sd:.2f})$ (profile)',
            path_effects=[pe.Stroke(linewidth=4, foreground='#1b1b1b'),
                          pe.Normal()])
    m0, sd0 = b0.mean(), b0.std(ddof=1)
    ax.plot(x, norm * stats.norm.pdf(x, m0, sd0), color='#1b1b1b', lw=1.8,
            ls=':', label=rf'$\mathcal{{N}}({m0:+.2f}, {sd0:.2f})$ (Bayesian)')
    ax.axvline(0, color='#bbbbbb', lw=0.8, zorder=0)
    # number of sets per bar (profile scheme) and, in brackets, the expected
    cnt = np.histogram(b1, bins=bins)[0]
    top = np.maximum(cnt, np.histogram(b0, bins=bins)[0])
    exp = len(sel) * np.diff(stats.norm.cdf(bins))
    for k in range(len(cnt)):
        if cnt[k] == 0 and exp[k] < 0.05:
            continue
        e = f'{exp[k]:.1f}' if exp[k] >= 0.1 else f'{exp[k]:.2f}'
        ax.text(0.5 * (bins[k] + bins[k + 1]), top[k] + 0.2,
                f'{cnt[k]}\n({e})', ha='center', va='bottom', fontsize=7.5,
                color='#1b1b1b', linespacing=1.0)
    err = b1.std(ddof=1) / np.sqrt(len(b1))
    ax.set_title(f'{a.kind}, {lab}: {len(sel)} sets — bias {b0.mean():+.2f} '
                 f'→ {b1.mean():+.2f} $\\pm$ {err:.2f}',
                 fontsize=10, color='#1b1b1b', loc='left')
    ax.set_xlabel(r'bias = $(r_{\rm med} - r_{\rm true})\,/\,\sigma(r)$')
    ax.set_xlim(-4, 4)
    ax.grid(axis='y', color='#e6e6e6', lw=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
axes[0].set_ylabel('sets')
ymax = max(ax.get_ylim()[1] for ax in axes)
axes[0].set_ylim(0, ymax * 1.45)
fig.text(0.99, 0.005, 'per bar: sets, profile scheme (expected from N(0,1)); '
         'upper limit / detection = conclusion of the profile scheme',
         ha='right', fontsize=8, color=INK)
fig.tight_layout()
fig.savefig(a.out, bbox_inches='tight')
fig.savefig(a.out.rsplit('.', 1)[0] + '.png', dpi=140, bbox_inches='tight')
print('written', a.out)
