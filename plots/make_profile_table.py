"""
Summary table of the tree (Bayesian part) vs the profile scheme, set by set.

  python plots/make_profile_table.py results_unblinded.csv profile_results.csv out.pdf \
      [--title "..."] [--note "..."] [--kind rand]

results_unblinded.csv: unblinded csv of the validation (scripts/collect.py
tree --unblind: true r, sky, model and r of the tree); profile_results.csv:
scripts/collect.py profile. Only the sets concluded in both. Also writes
<out>.csv and prints a summary (mean bias, detections, coverage).

Bayesian: median and sigma of the PolyChord posterior of the model chosen by
the tree, and r_95 for the upper limits. Profile: median and sigma of the
posterior of r from the profile (prior of the tree), and r_95 with r >= 0
for the upper limits.
Rows in pairs: for each set the two skies (d9s4 and random) one below the
other; grey = profile scheme not concluded yet.
Colours: red = wrong conclusion of the profile scheme, green = right where
the tree was wrong, blue = conclusion different from the tree's, yellow =
|bias| > 2 in the profile scheme. Right conclusion: upper limit with
true r <= r_95, detection with true r > 0.
"""
import argparse
import csv
import os
import re

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from bbsure.models import amp_name, group_spectra  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('tree_csv')
ap.add_argument('profile_csv')
ap.add_argument('out')
ap.add_argument('--title', default='Tree (Bayesian) vs profile scheme')
ap.add_argument('--note', default='')
ap.add_argument('--kind', default=None, help='only this kind (rand, d9s4)')
ap.add_argument('--png', default=None, help='also save the first page as png')
ap.add_argument('--r95', default=None,
                help='csv with the tree upper limits recomputed with r >= 0 '
                     '(columns index, kind, r_95_pos; only for trees run '
                     'before r_95 was computed with r >= 0)')
a = ap.parse_args()
plt.rcParams['font.family'] = 'DejaVu Sans'

SPEC = {'w00w10': r'\omega_0^{X}\!\times\omega_1^{\beta_X}',
        'w10w10': r'\omega_1^{\beta_X}\!\times\omega_1^{\beta_X}',
        'w00w01': r'\omega_0^{d}\!\times\omega_1^{T_d}',
        'w01w01': r'\omega_1^{T_d}\!\times\omega_1^{T_d}',
        'w10w01': r'\omega_1^{\beta_d}\!\times\omega_1^{T_d}',
        'w00w20': r'\omega_0^{X}\!\times\omega_2^{\beta_X}',
        'w10w20': r'\omega_1^{\beta_X}\!\times\omega_2^{\beta_X}',
        'w20w20': r'\omega_2^{\beta_X}\!\times\omega_2^{\beta_X}',
        'w10w00': r'\omega_1^{\beta_d}\!\times\omega_0^{s}'}


def model_label(m):
    """M_d+s-d_w00w10-s_w00w10_r -> moment spectra LEFT, in LaTeX."""
    body = re.sub(r'^M_|_(r|0)$', '', m)
    parts = body.split('-')
    if parts[0] == 'none':
        return 'no moments'
    off = {'A_' + p for p in parts[1:]}
    out = []
    for _, pref, spec in group_spectra(set(parts[0].split('+'))):
        if amp_name(pref, spec) in off:
            continue
        comp = {'d': 'd', 's': 's', 'ds': 'd'}[pref]
        tex = SPEC[spec].replace('X', comp)
        if pref == 'ds' and spec == 'w10w10':
            tex = r'\omega_1^{\beta_d}\!\times\omega_1^{\beta_s}'
        elif pref == 'ds' and spec == 'w00w10':
            tex = r'\omega_0^{d}\!\times\omega_1^{\beta_s}'
        out.append('$' + tex + '$')
    return ' + '.join(out) if out else 'no moments'


def r_text(med, sd):
    return f'{med * 1e3:.1f} ± {sd * 1e3:.1f}'


def ul_text(concl, r95):
    return f'< {r95 * 1e3:.1f}' if concl == 'upper_limit' else ''


def chi2_text(chi2, dof, pte):
    """reduced chi2 (PTE); empty if missing (e.g. csv without chi2 at r*)."""
    try:
        return f'{float(chi2) / float(dof):.2f} ({float(pte):.2f})'
    except (TypeError, ValueError, ZeroDivisionError):
        return ''


def dof_tree(T, P):
    """Degrees of freedom of the fit of the tree (r free): those at r* minus
    one if the model is the same, otherwise from the PTE (inversion)."""
    if P is not None and P.get('dof_rstar'):
        if P['model_final'] == T['model']:
            return int(P['dof_rstar']) - 1
        d0 = int(P['dof_rstar'])
        cand = range(d0 - 30, d0 + 30)
    else:
        cand = range(200, 800)
    try:
        c, pte = float(T['chi2']), float(T['pte'])
    except (TypeError, ValueError):
        return None
    return min(cand, key=lambda d: abs(stats.chi2.sf(c, d) - pte))


def ok(concl, t, r95):
    return t <= r95 if concl == 'upper_limit' else t > 0


tree = {(r['index'], r['kind']): r for r in csv.DictReader(open(a.tree_csv))
        if r['status'] == 'done'}
if a.r95:
    # upper limit of the tree recomputed with r >= 0 (as the profile scheme)
    for r in csv.DictReader(open(a.r95)):
        k = (r['index'], r['kind'])
        if k in tree and tree[k]['conclusion'] == 'upper_limit':
            tree[k]['r_95'] = r['r_95_pos']
prof = {(r['index'], r['kind']): r for r in csv.DictReader(open(a.profile_csv))
        if r['status'] == 'done'}
# in pairs (same set, the two skies one below the other): a set whose
# profile scheme is not concluded yet shows only the Bayesian part
idx = sorted({int(k[0]) for k in prof if k in tree})
keys = [(str(i), kd) for i in idx for kd in sorted({k[1] for k in tree})
        if (str(i), kd) in tree and (a.kind is None or kd == a.kind)]

rows, colors, recs = [], [], []
for k in keys:
    T = tree[k]
    t = float(T['r_true'])
    ct = T['conclusion']
    mt, st = float(T['r_med']), float(T['r_std'])
    rt95 = float(T['r_95']) if T['r_95'] else np.nan
    bt = (mt - t) / st
    if k not in prof:
        rows.append([k[0], T['sky'], f'{t * 1e3:.1f}', model_label(T['model']),
                     'UL' if ct == 'upper_limit' else 'det.', r_text(mt, st),
                     ul_text(ct, rt95), f'{bt:+.1f}',
                     chi2_text(T.get('chi2'), dof_tree(T, None), T.get('pte')),
                     'not concluded',
                     '', '', '', '', '', ''])
        colors.append('#f2f2f2')
        continue
    P = prof[k]
    cp = P['conclusion']
    mp, sp = float(P['r_med_pl']), float(P['r_std_pl'])
    rp95 = float(P['r_95_pl0'])
    bp = (mp - t) / sp
    okt, okp = ok(ct, t, rt95), ok(cp, t, rp95)
    same = P['model_final'] == T['model']
    rows.append([k[0], T['sky'], f'{t * 1e3:.1f}',
                 model_label(T['model']), 'UL' if ct == 'upper_limit'
                 else 'det.', r_text(mt, st), ul_text(ct, rt95), f'{bt:+.1f}',
                 chi2_text(T.get('chi2'), dof_tree(T, P), T.get('pte')),
                 '=' if same else model_label(P['model_final']),
                 'UL' if cp == 'upper_limit' else 'det.',
                 r_text(mp, sp), ul_text(cp, rp95), f'{bp:+.1f}',
                 chi2_text(P.get('chi2_at_rstar'), P.get('dof_rstar'),
                           P.get('pte_rstar')),
                 'yes' if okp else 'NO'])
    colors.append('#fde2e1' if not okp else '#dff3e2' if not okt else
                  '#e3ecfa' if cp != ct else
                  '#fff4d6' if abs(bp) > 2 else 'white')
    recs.append(dict(index=k[0], kind=k[1], sky=T['sky'], r_true=t,
                     model_bayes=T['model'], concl_bayes=ct, r_med_bayes=mt,
                     r_std_bayes=st, r_95_bayes=rt95, bias_bayes=bt,
                     model_profile=P['model_final'], concl_profile=cp,
                     r_star=float(P['r_star_final']), r_med_profile=mp,
                     r_std_profile=sp, r_95_profile=rp95, bias_profile=bp,
                     ok_bayes=okt, ok_profile=okp, start=P['start'],
                     rejected=P['rejected'], chi2_bayes=T.get('chi2'),
                     pte_bayes=T.get('pte'), dof_bayes=dof_tree(T, P),
                     chi2_profile=P.get('chi2_at_rstar'),
                     dof_profile=P.get('dof_rstar'),
                     pte_profile=P.get('pte_rstar')))

with open(os.path.splitext(a.out)[0] + '.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(recs[0]))
    w.writeheader()
    w.writerows(recs)

# summary per sky type
summary = []
for kind in sorted({r['kind'] for r in recs}):
    R = [r for r in recs if r['kind'] == kind]
    n = len(R)
    sem = (lambda x: x.std(ddof=1) / np.sqrt(n)) if n > 1 else (lambda x: 0.)
    bb = np.array([r['bias_bayes'] for r in R])
    bp = np.array([r['bias_profile'] for r in R])
    s = (f"{kind}: N = {n}; Bayesian bias {bb.mean():+.2f} ± "
         f"{sem(bb):.2f}, profile {bp.mean():+.2f} ± "
         f"{sem(bp):.2f} (paired "
         f"{(bp - bb).mean():+.2f}); wrong conclusions "
         f"{sum(not r['ok_bayes'] for r in R)} -> "
         f"{sum(not r['ok_profile'] for r in R)}; conclusion changed in "
         f"{sum(r['concl_bayes'] != r['concl_profile'] for r in R)}, "
         f"model in "
         f"{sum(r['model_bayes'] != r['model_profile'] for r in R)}")
    summary.append(s)
    print(s)

head = ['set', 'sky', 'true r', 'model (Bayes.)', 'concl.', 'r Bayes.',
        'r < (95%)', 'bias', 'χ²_red (PTE)', 'model (profile)', 'concl.',
        'r profile', 'r < (95%)', 'bias', 'χ²_red (PTE)', 'ok?']
per_page = 18
with PdfPages(a.out) as pdf:
    for k0 in range(0, len(rows), per_page):
        fig = plt.figure(figsize=(11.69, 8.27))
        if k0 == 0:
            fig.text(0.03, 0.955, a.title, fontsize=13, weight='bold')
            fig.text(0.03, 0.932, "r in units of 10$^{-3}$: median $\\pm\\sigma$ of the posterior "
                     "(also for the upper limits), r < (95%) only for the "
                     "upper limits; bias = (median - true r)/$\\sigma$. Model (profile) '=': "
                     "same as the tree's.", fontsize=8, color='#555')
            fig.text(0.03, 0.915, "Red: wrong conclusion (profile); "
                     "green: right where the tree was wrong; blue: "
                     "conclusion different from the tree; yellow: |bias| > 2; grey: not "
                     "concluded. "
                     + a.note, fontsize=8, color='#555')
            fig.text(0.03, 0.898, "$\\chi^2_{\\rm red}$ (PTE): $\\chi^2$/degrees of "
                     "freedom at the ML best fit of the model (Bayesian: r free; "
                     "profile: r = r*), without Gaussian priors; degrees of freedom "
                     "= data - free parameters.", fontsize=8, color='#555')
            for i, s in enumerate(summary):
                fig.text(0.03, 0.878 - 0.017 * i, s, fontsize=8)
        ax = fig.add_axes([0.02, 0.03, 0.96, 0.82])
        ax.axis('off')
        page = [list(r) for r in rows[k0:k0 + per_page]]
        for i in range(1, len(page)):
            if keys[k0 + i][0] == keys[k0 + i - 1][0]:
                page[i][0] = ''
        tb = ax.table(cellText=page, colLabels=head,
                      loc='upper left', cellLoc='center',
                      colWidths=[0.03, 0.04, 0.04, 0.15, 0.036, 0.074,
                                 0.054, 0.036, 0.064, 0.15, 0.036, 0.074,
                                 0.054, 0.036, 0.064, 0.03])
        tb.auto_set_font_size(False)
        tb.set_fontsize(7.5)
        for (i, j), cl in tb.get_celld().items():
            cl.set_edgecolor('#d9d9d9')
            cl.set_height(0.045)
            if i == 0:
                cl.set_facecolor('#eef1f5')
                cl.get_text().set_weight('bold')
            else:
                cl.set_facecolor(colors[k0 + i - 1])
                if j in (3, 9) and '$' in cl.get_text().get_text():
                    cl.get_text().set_fontsize(6.5)
        pdf.savefig(fig)
        if k0 == 0 and a.png:
            fig.savefig(a.png, dpi=110)
        plt.close(fig)
print('written', a.out, 'and', os.path.splitext(a.out)[0] + '.csv',
      len(rows), 'rows')
