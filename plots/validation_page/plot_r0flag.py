"""
Sets in increasing estimated r (best fit r* of the model chosen by the
profile scheme, with its 68% interval and the set number), coloured by the
r=0 flag of the page (the highest grade among the M_1,0 of Res_r0 and M_0,0,
as in build_page.py); circle: r detection, triangle: upper limit (profile
scheme). Needs no true r: used for blind runs.

  python plots/validation_page/plot_r0flag.py <root> <out.png> [--kind rand]
"""
import argparse, csv, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ap = argparse.ArgumentParser()
ap.add_argument('root'); ap.add_argument('out'); ap.add_argument('--kind', default='rand')
ap.add_argument('--review', default=None, help='csv of the page: red band, conclusion not '
                'trusted; yellow band, conclusion trusted but model not')
a = ap.parse_args()
R = a.root
RANK = {'': 0, '✗✗': 0.5, '✗': 1, '✓': 2, '✓✓': 3}
def grade(x, e):
    b = max(1., 2 * e if e == e else 0.)
    if x > 0: return '✓✓' if x >= b else '✓'
    return '✗' if x > -b else '✗✗'
def r0g(name):
    t = f'{R}/runs/{name}/tree'
    ev = json.load(open(f'{t}/decision_log.json')); ct = json.load(open(f'{t}/conclusion_tree.json'))
    gs = [grade(e['lnB'], e['lnB_err']) for e in ev if e['node'] == 'pref'
          and e.get('question', '').startswith('Is M_1,0 (') and 'lnB' in e and not e.get('method')]
    if ct.get('lnB_rfree_vs_r0') is not None:
        gs.append(grade(-ct['lnB_rfree_vs_r0'], ct['lnB_err']))
    return max(gs, key=lambda g: RANK[g], default='')
p = pd.read_csv(f'{R}/profile_results.csv'); p = p[(p.status == 'done') & (p.kind == a.kind)]
rows = []
for _, r in p.iterrows():
    n = f"set_{int(r['index']):03d}_{a.kind}"
    rows.append(dict(i=int(r['index']), rs=1e3 * r.r_star_final, lo=1e3 * r.r_lo1_pl, hi=1e3 * r.r_hi1_pl,
                     c=r.conclusion, g=r0g(n)))
d = sorted(rows, key=lambda r: (r['rs'], r['i']))
col = {'✓✓': '#2c5f8e', '✓': '#86afd6', '✗': '#f1a7c5', '✗✗': '#c42a6e'}
lab = {'✓✓': 'r = 0 preferred by a lot  (✓✓)', '✓': 'r = 0 preferred by a little  (✓)',
       '✗': 'r = 0 disfavoured by a little  (✗)', '✗✗': 'r = 0 disfavoured by a lot  (✗✗)'}
ink, soft, edge = '#2f3640', '#8a919c', '#4a4a4a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'axes.edgecolor': soft, 'xtick.color': soft, 'ytick.color': soft,
                     'axes.labelcolor': ink, 'ytick.left': False})
fig, ax = plt.subplots(figsize=(15, 5.4))
ax.set_facecolor('#f7f7f5')
for j, r in enumerate(d):
    ax.plot([j, j], [r['lo'], r['hi']], color=col[r['g']], lw=1.6, alpha=0.8, solid_capstyle='round', zorder=1)
for g in col:
    for conc, mk, sz in (('r_detection', 'o', 85), ('upper_limit', 'v', 95)):
        jj = [j for j, r in enumerate(d) if r['g'] == g and r['c'] == conc]
        ax.scatter(jj, [d[j]['rs'] for j in jj], s=sz, marker=mk, c=col[g], edgecolors=edge, linewidths=0.8, zorder=3)
for j, r in enumerate(d):
    ax.text(j, r['hi'] + 0.6, f"{r['i']:03d}", rotation=90, ha='center', va='bottom', fontsize=6.5, color=ink)
nb = nm = 0
if a.review:
    rv = {r['set']: r for r in csv.DictReader(open(a.review))}
    for j, r in enumerate(d):
        q = rv.get(f"set_{r['i']:03d}_{a.kind}")
        if q is None:
            continue
        if q['conclusion_trusted'] != '1':
            ax.axvspan(j - 0.5, j + 0.5, color='#dc2832', alpha=0.16, zorder=0, lw=0); nb += 1
        elif q['model_trusted'] != '1':
            ax.axvspan(j - 0.5, j + 0.5, color='#f2c94c', alpha=0.35, zorder=0, lw=0); nm += 1
ax.axhline(0, color=soft, lw=0.8, zorder=0)
ax.grid(axis='y', color='white', lw=1.4); ax.set_axisbelow(True)
ax.set_xlim(-2, len(d) + 1); ax.set_ylabel('r  [10$^{-3}$]', fontsize=11)
ax.set_xlabel('sets, in increasing estimated r', fontsize=11)
n = {g: sum(r['g'] == g for r in d) for g in col}
ax.text(0, 1.03, 'r = 0 flag', transform=ax.transAxes, fontsize=13, fontweight='bold', color=ink, va='bottom')
ax.text(0.085, 1.03, '   best fit of r of the chosen model (profile scheme) and its 68% interval;  '
        'conclusion of the profile scheme', transform=ax.transAxes, fontsize=10.5, color=soft, va='bottom')
sym = [Line2D([], [], marker='o', ls='', mfc='#c4c8ce', mec=edge, mew=0.8, ms=10, label='circle: r detection'),
       Line2D([], [], marker='v', ls='', mfc='#c4c8ce', mec=edge, mew=0.8, ms=10, label='triangle: upper limit')]
cs = [Line2D([], [], marker='o', ls='', mfc=col[g], mec=edge, mew=0.8, ms=11, label=f'{lab[g]}: {n[g]}') for g in col]
if a.review:
    sym += [Patch(color='#dc2832', alpha=0.16, label=f'band: conclusion not trusted ({nb})'),
            Patch(color='#f2c94c', alpha=0.35, label=f'band: conclusion trusted, model not ({nm})')]
fig.legend(handles=sym + cs, loc='upper center', ncol=4 if a.review else 3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0),
           columnspacing=2.4, handletextpad=0.6, labelcolor=ink)
fig.tight_layout(rect=(0, 0, 1, 0.86))
fig.savefig(a.out, dpi=140, facecolor='white')
print(n, 'no grade:', sum(r['g'] == '' for r in d))
