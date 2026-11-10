"""
r set by set in a validation run: true r (star), median and 68% interval
(detections: filled, upper limits: open) and the 95% upper limit (bar with a
downward arrow). Red circle and band: wrong conclusion (true r above r_95, or
detection with true r = 0); yellow: detection with true r < 3e-3. With --ref
(first_fit.csv of tables.py) also the d9s4 median of the same set (same
CMB + noise) from the first fit of the tree, without moments (M_none_r).

  python plots/validation_page/plot_r_sets.py <root>/tree_full.csv \
      --kind d9s4 --label "Bayesian (tree)" --out r_sets_d9s4_tree.pdf
"""
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('csv')
ap.add_argument('--kind', default='d9s4')
ap.add_argument('--out', required=True)
ap.add_argument('--title', default=None)
ap.add_argument('--label', default='')
ap.add_argument('--ref', default=None, help='csv whose d9s4 median is shown next to the true r '
                '(first_fit.csv: fit without moments)')
a = ap.parse_args()

rows = sorted((r for r in csv.DictReader(open(a.csv))
               if r['kind'] == a.kind and r['status'] == 'done'),
              key=lambda r: int(r['index']))
f = lambda r, k: float(r[k]) if r.get(k) not in (None, '') else np.nan
i = np.array([int(r['index']) for r in rows])
rt = np.array([f(r, 'r_true') for r in rows]) * 1e3
rm = np.array([f(r, 'r_med') for r in rows]) * 1e3
sd = np.array([f(r, 'r_std') for r in rows]) * 1e3
lo = np.array([f(r, 'r_lo1') for r in rows]) * 1e3
hi = np.array([f(r, 'r_hi1') for r in rows]) * 1e3
r95 = np.array([f(r, 'r_95') for r in rows]) * 1e3
det = np.array([r['conclusion'] == 'r_detection' for r in rows])
STAR = 'true r'
REFL = 'd9s4 of the same set (same CMB + noise), fit without moments: median'
rd = None
if a.ref:
    ref = {int(r['index']): float(r['r_med']) for r in csv.DictReader(open(a.ref))
           if r['kind'] == 'd9s4' and r['status'] == 'done'}
    rd = np.array([ref.get(k, np.nan) for k in i]) * 1e3

C_DET, C_UL, C_TRUE, INK = '#2a6fdb', '#d9822b', '#1b1b1b', '#5a5a5a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.edgecolor': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK, 'ytick.color': INK})
# width grows with the number of sets (room for the sky labels; the legend
# stays within the axes)
fig, ax = plt.subplots(figsize=(max(15, 0.11 * len(rows)), 5.2))
ax.axhline(0, color='#bbbbbb', lw=0.8, zorder=0)
ax.grid(axis='y', color='#e6e6e6', lw=0.6, zorder=0)

# detection: median and central 68% interval
ax.errorbar(i[det], rm[det], yerr=[rm[det] - lo[det], hi[det] - rm[det]],
            fmt='o', ms=4, color=C_DET, ecolor=C_DET, elinewidth=1.2,
            capsize=0, zorder=3, label='r detection: median, 68% interval')
# upper limit: median, 68% interval and r_95
ul = ~det
ax.errorbar(i[ul], rm[ul], yerr=[rm[ul] - lo[ul], hi[ul] - rm[ul]], fmt='o', ms=4, mfc='white',
            color=C_UL, ecolor=C_UL, elinewidth=1.2, capsize=0, zorder=3,
            label='upper limit: median, 68% interval')
ax.scatter(i[ul], r95[ul], marker='_', s=110, lw=1.8, color=C_UL, zorder=3,
           label='upper limit: $r_{95}$')
for x, y in zip(i[ul], r95[ul]):
    ax.annotate('', xy=(x, y - 1.6), xytext=(x, y),
                arrowprops=dict(arrowstyle='-|>', color=C_UL, lw=1.0,
                                mutation_scale=7), zorder=3)
ax.scatter(i, rt, marker='*', s=70, color=C_TRUE, edgecolor='white', lw=0.4,
           zorder=4, label=STAR)
# wrong conclusions: upper limit below the true r, detection with true r = 0
bad = (ul & (rt > r95)) | (det & (rt == 0))
if bad.any():
    ax.scatter(i[bad], rt[bad], marker='o', s=230, facecolor='none',
               edgecolor='#d62728', lw=1.6, zorder=5,
               label='wrong: true r above $r_{95}$, or detection with r = 0')
    for x in i[bad]:
        ax.axvspan(x - 0.5, x + 0.5, color='#d62728', alpha=0.08, zorder=0)
# detection with a small true r (< 3e-3), not already wrong: yellow, as in the table
low = det & (rt < 3) & ~bad
if low.any():
    ax.scatter(i[low], rt[low], marker='o', s=230, facecolor='none',
               edgecolor='#d4a106', lw=1.6, zorder=5,
               label=r'detection with true r < $3\times10^{-3}$')
    for x in i[low]:
        ax.axvspan(x - 0.5, x + 0.5, color='#f2c94c', alpha=0.18, zorder=0)
if rd is not None:
    # d9s4 median of the same set, shifted a little to the right
    ax.scatter(i + 0.3, rd, marker='D', s=20, color='#2f9e44', edgecolor='white',
               lw=0.4, zorder=4, label=REFL)

# sky of each set (only if it changes from set to set, e.g. random skies)
skies = [r['sky'] for r in rows]
if len(set(skies)) > 1:
    for x, sk in zip(i, skies):
        ax.text(x, -0.02, sk, rotation=90, ha='center', va='top', fontsize=6,
                color=INK, transform=ax.get_xaxis_transform())
    ax.tick_params(axis='x', pad=26)
ax.set_xlim(i.min() - 1, i.max() + 1)
ax.set_xlabel('set')
ax.set_ylabel(r'$r \times 10^{3}$')
pull = (rm - rt) / sd   # bias in units of sigma
n_cov = np.sum(det & (rt >= lo) & (rt <= hi)), np.sum(ul & (rt <= r95))
extra = ''
if rd is not None:
    pp = (rm - rd) / sd
    extra = (f"; paired bias $(r - r_{{\\rm d9s4,\\,M_{{0,r}}}})/\\sigma$ "
             f"{np.nanmean(pp):+.2f}")
ax.set_title(a.title or
             f"{a.kind}, {a.label}: {len(rows)} sets — {det.sum()} detections "
             f"(true r in 68%: {n_cov[0]}/{det.sum()}), {ul.sum()} upper "
             f"limits (true r < r$_{{95}}$: {n_cov[1]}/{ul.sum()}) — "
             f"bias {np.nanmean(pull):+.2f} $\\pm$ {np.nanstd(pull):.2f}" + extra,
             fontsize=11, color='#1b1b1b', loc='left', pad=48)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
h, l = ax.get_legend_handles_labels()
order = [l.index(x) for x in ([STAR] + ([REFL] if rd is not None else []) + ['r detection: median, 68% interval',
                              'upper limit: median, 68% interval',
                              'upper limit: $r_{95}$'] + [x for x in l if x.startswith('wrong') or x.startswith('detection with')])]
ax.legend([h[k] for k in order], [l[k] for k in order], loc='lower left',
          bbox_to_anchor=(0, 1.0), frameon=False, ncol=4, fontsize=9)
fig.tight_layout()
fig.savefig(a.out, bbox_inches='tight')
fig.savefig(a.out.rsplit('.', 1)[0] + '.png', dpi=160, bbox_inches='tight')
print('written', a.out)
