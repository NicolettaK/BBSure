"""
Differenza accoppiata (r_rand - r_d9s4)/sigma_rand in funzione di r vero, per la
parte bayesiana del tree e per lo schema profilo. Punti colorati per tipo di
cielo random (dust e synch entrambi complessi o no); medie in bin di r vero.

  python plot_paired_vs_r.py plots/validation/profile_scheme_v3/profile_vs_tree.csv \
      --out plots/validation/profile_scheme_v3/paired_vs_r.pdf

Il csv e' quello scritto da make_profile_table.py.
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

rows = {(int(r['index']), r['kind']): r for r in csv.DictReader(open(a.csv))}
idx = sorted(i for i, k in rows if k == 'rand' and (i, 'd9s4') in rows)
rt = np.array([float(rows[i, 'rand']['r_true']) for i in idx])
sky = [rows[i, 'rand']['sky'] for i in idx]
# entrambi complessi: dust d10/d12 e synch s5/s7
both = np.array([s[:3] in ('d10', 'd12') and s[-2:] in ('s5', 's7')
                 for s in sky])


def paired(rkey, skey):
    return np.array([(float(rows[i, 'rand'][rkey]) - float(rows[i, 'd9s4'][rkey]))
                     / float(rows[i, 'rand'][skey]) for i in idx])


INK = '#5a5a5a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.edgecolor': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK})
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
edges = np.array([0, 0.003, 0.008, 0.015, 0.0301])
for ax, (lab, d) in zip(axes, [
        ('Bayesian (tree)', paired('r_med_bayes', 'r_std_bayes')),
        ('profile scheme', paired('r_med_profile', 'r_std_profile'))]):
    ax.axhline(0, color='#bbbbbb', lw=0.8, zorder=0)
    ax.scatter(1e3 * rt[~both], d[~both], s=28, color='#9a9a9a',
               edgecolor='white', lw=0.6,
               label=f'other skies ({(~both).sum()}): mean '
                     f'{d[~both].mean():+.2f}')
    ax.scatter(1e3 * rt[both], d[both], s=28, color='#C2410C',
               edgecolor='white', lw=0.6,
               label=f'dust d10/d12 and synch s5/s7 ({both.sum()}): mean '
                     f'{d[both].mean():+.2f}')
    # media in bin di r vero, tutte le coppie
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (rt >= lo) & (rt < hi)
        m, e = d[s].mean(), d[s].std(ddof=1) / np.sqrt(s.sum())
        ax.errorbar(1e3 * 0.5 * (lo + hi), m, yerr=e, xerr=1e3 * 0.5 * (hi - lo),
                    color='#14213D', lw=1.8, capsize=0, zorder=5)
        ax.text(1e3 * 0.5 * (lo + hi), m + e + 0.08, f'{m:+.2f}',
                ha='center', va='bottom', fontsize=8.5, color='#14213D')
    ax.plot([], [], color='#14213D', lw=1.8,
            label='mean in bins of true r, all pairs')
    ax.set_title(f'{lab}: {d.mean():+.2f} $\\pm$ '
                 f'{d.std(ddof=1) / np.sqrt(len(d)):.2f} over {len(d)} pairs',
                 fontsize=10, color='#1b1b1b', loc='left')
    ax.set_xlabel(r'true $r$ $[10^{-3}]$')
    ax.set_xlim(-0.6, 1e3 * edges[-1])
    ax.grid(color='#eeeeee', lw=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
axes[0].set_ylabel(r'$(r_{\rm random} - r_{\rm d9s4})\,/\,\sigma_{\rm random}$')
axes[0].set_ylim(-3, 4)
fig.text(0.99, 0.005, 'same CMB + noise in each pair, only the foregrounds '
         'change; colour = random sky type', ha='right', fontsize=8, color=INK)
fig.tight_layout()
fig.savefig(a.out, bbox_inches='tight')
fig.savefig(a.out.rsplit('.', 1)[0] + '.png', dpi=140, bbox_inches='tight')
print('written', a.out)
