"""
Histogram of the paired difference (r_rand - r_d9s4)/sigma_rand on the same
sets (same CMB and noise, only the foregrounds change): profile scheme
(bars) vs Bayesian part of the tree (outline); all, upper limits and
detections (conclusion of the profile scheme on the random sky).

  python plots/plot_paired_bias.py profile_results.csv --out paired_bias.pdf

Tree: r_bayes_tree, sigma_bayes_tree; profile: r_med_pl, r_std_pl.
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
ap.add_argument('--out', required=True)
a = ap.parse_args()

rows = {}
for r in csv.DictReader(open(a.csv)):
    if r['status'] == 'done':
        rows[(int(r['index']), r['kind'])] = r
idx = sorted(i for i, k in rows if k == 'rand' and (i, 'd9s4') in rows)


def paired(rkey, skey):
    return np.array([(float(rows[i, 'rand'][rkey]) - float(rows[i, 'd9s4'][rkey]))
                     / float(rows[i, 'rand'][skey]) for i in idx])


B0 = paired('r_bayes_tree', 'sigma_bayes_tree')
B1 = paired('r_med_pl', 'r_std_pl')
concl = np.array([rows[i, 'rand']['conclusion'] for i in idx])

INK = '#5a5a5a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.edgecolor': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK})
fig, axes = plt.subplots(1, 3, figsize=(19, 4.8), sharey=True)
bins = np.arange(-4, 4.01, 0.5)
x = np.linspace(-4, 4, 400)
for ax, (cc, lab, col, dark) in zip(axes, [
        (None, 'all', '#3a9e6f', '#86d9b0'),
        ('upper_limit', 'upper limits', '#d9822b', '#f5bd82'),
        ('r_detection', 'r detections', '#2a6fdb', '#8db6f7')]):
    sel = np.ones(len(idx), bool) if cc is None else concl == cc
    b0, b1 = B0[sel], B1[sel]
    n = sel.sum()
    ax.hist(b1, bins=bins, color=col, edgecolor='white', lw=1.5,
            label=f'profile scheme: mean {b1.mean():+.2f}, '
                  f'std {b1.std(ddof=1):.2f}')
    ax.hist(b0, bins=bins, histtype='step', color='#1b1b1b', lw=1.4,
            label=f'Bayesian (tree): mean {b0.mean():+.2f}, '
                  f'std {b0.std(ddof=1):.2f}')
    norm = n * (bins[1] - bins[0])
    ax.plot(x, norm * stats.norm.pdf(x), color='#1b1b1b', lw=2,
            label=r'$\mathcal{N}(0, 1)$')
    m, sd = b1.mean(), b1.std(ddof=1)
    ax.plot(x, norm * stats.norm.pdf(x, m, sd), color=dark, lw=2.6,
            ls='--', label=rf'$\mathcal{{N}}({m:+.2f}, {sd:.2f})$ (profile)',
            path_effects=[pe.Stroke(linewidth=4, foreground='#1b1b1b'),
                          pe.Normal()])
    m0, sd0 = b0.mean(), b0.std(ddof=1)
    ax.plot(x, norm * stats.norm.pdf(x, m0, sd0), color='#1b1b1b', lw=1.8,
            ls=':', label=rf'$\mathcal{{N}}({m0:+.2f}, {sd0:.2f})$ (Bayesian)')
    ax.axvline(0, color='#bbbbbb', lw=0.8, zorder=0)
    # number of pairs per bar (profile scheme)
    cnt = np.histogram(b1, bins=bins)[0]
    top = np.maximum(cnt, np.histogram(b0, bins=bins)[0])
    for k in range(len(cnt)):
        if cnt[k]:
            ax.text(0.5 * (bins[k] + bins[k + 1]), top[k] + 0.2, f'{cnt[k]}',
                    ha='center', va='bottom', fontsize=7.5, color='#1b1b1b')
    rt = np.sqrt(n)
    ax.set_title(f'{lab}: {n} pairs — {b0.mean():+.2f} $\\pm$ '
                 f'{b0.std(ddof=1) / rt:.2f} (tree) → {b1.mean():+.2f} '
                 f'$\\pm$ {b1.std(ddof=1) / rt:.2f} (profile)',
                 fontsize=10, color='#1b1b1b', loc='left')
    ax.set_xlabel(r'$(r_{\rm random} - r_{\rm d9s4})\,/\,\sigma_{\rm random}$')
    ax.set_xlim(-4, 4)
    ax.grid(axis='y', color='#e6e6e6', lw=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
axes[0].set_ylabel('pairs')
ymax = max(ax.get_ylim()[1] for ax in axes)
axes[0].set_ylim(0, ymax * 1.45)
fig.text(0.99, 0.005, 'same CMB + noise in each pair, only the foregrounds '
         'change; upper limit / detection = conclusion of the profile scheme '
         'on the random sky; per bar: pairs, profile scheme',
         ha='right', fontsize=8, color=INK)
fig.tight_layout()
fig.savefig(a.out, bbox_inches='tight')
fig.savefig(a.out.rsplit('.', 1)[0] + '.png', dpi=140, bbox_inches='tight')
print('written', a.out)
