"""
Corner plot di un set: posterior bayesiano del tree (PolyChord del modello
finale, tutti i parametri, r compreso) e posterior dello schema profilo
(emcee del modello finale con r fissato a r*), sovrapposti. Sulla colonna di
r: per il profilo il posterior di r dal profilo (prior del tree) sulla
diagonale e la linea r = r* nei pannelli 2D; r vero in nero.

  python plot_corner.py <tree_dir> <profile_set_dir> --r-true 0.0111 \
      --title "set 47 rand (d12s5)" --out corner_047_rand.pdf

tree_dir: runs/set_<i>_<kind>/tree (PySM_rnd2); profile_set_dir:
set_<i>_<kind> dello schema profilo (PySM_rnd3: result.json,
profile_<modello>.npz, post_<modello>_rstar/emcee.npz).
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import corner  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_tree as dt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('tree')
ap.add_argument('prof')
ap.add_argument('--r-true', type=float, default=None)
ap.add_argument('--extra-pc', default=None,
                help='another PolyChord run of the tree (<model>_PC dir), '
                     'e.g. the M_1,r with the moment and r free')
ap.add_argument('--title', default='')
ap.add_argument('--out', required=True)
a = ap.parse_args()

C_B, C_P, C_X = '#2a6fdb', '#e8590c', '#2f9e44'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

# ── posterior bayesiano del tree ──────────────────────────────────────────
_c = os.path.join(a.tree, 'conclusion_tree.json')
concl = json.load(open(_c if os.path.isfile(_c) else
                       os.path.join(a.tree, 'conclusion.json')))
M0 = concl['model']
cfg = yaml.safe_load(open(os.path.join(a.tree, f'{M0}_PC', 'config.yml')))
names_b = dt.free_param_names(cfg['BBCompSep'])
samp_b = np.loadtxt(os.path.join(a.tree, f'{M0}_PC', 'polychord',
                                 'pch_equal_weights.txt'), ndmin=2)[:, 2:]

# ── schema profilo: emcee a r = r*, profilo di r ──────────────────────────
res = json.load(open(os.path.join(a.prof, 'result.json')))
M1, r_star = res['model_final'], res['r_star_final']
z = np.load(os.path.join(a.prof, f'post_{M1}_rstar', 'emcee.npz'))
chain = z['chain']
names_p = [str(n) for n in z['names']]
samp_p = chain[:, chain.shape[1] // 2:, :].reshape(-1, chain.shape[-1])
pz = np.load(os.path.join(a.prof, f'profile_{M1}.npz'))
fine = np.linspace(pz['r'][0], pz['r'][-1], 2001)
d = np.interp(fine, pz['r'], pz['chi2'])
d -= d.min()
post_r = np.exp(-d / 2)


# ── opzionale: un altro modello del tree (r libero) ────────────────────────
names_x = samp_x = None
if a.extra_pc:
    MX = os.path.basename(os.path.normpath(a.extra_pc))[:-3]
    cx = yaml.safe_load(open(os.path.join(a.extra_pc, 'config.yml')))
    names_x = dt.free_param_names(cx['BBCompSep'])
    samp_x = np.loadtxt(os.path.join(a.extra_pc, 'polychord',
                                     'pch_equal_weights.txt'), ndmin=2)[:, 2:]


def label(n):
    """Nome del parametro in LaTeX."""
    if n == 'r_tensor':
        return '$r$'
    m = re.match(r'(A|alpha)_(d|s|ds)_w(\d)(\d)w(\d)(\d)$', n)
    if m:
        sym = 'A' if m[1] == 'A' else r'\alpha'
        return f'${sym}^{{{m[2]}}}_{{{m[3]}{m[4]},{m[5]}{m[6]}}}$'
    fixed = {'beta_d': r'$\beta_d$', 'beta_s': r'$\beta_s$',
             'temp_d': r'$T_d$', 'epsilon_ds': r'$\epsilon_{ds}$',
             'amp_d_bb': r'$A_d$', 'amp_s_bb': r'$A_s$',
             'alpha_d_bb': r'$\alpha_d$', 'alpha_s_bb': r'$\alpha_s$',
             'A_lens': r'$A_{\rm lens}$'}
    return fixed.get(n, n.replace('_', r'\_'))


# insiemi di campioni: (nome, colore, nomi, campioni); il profilo ha r fisso
SETS = [('bayes', C_B, names_b, samp_b), ('prof', C_P, names_p, samp_p)]
if samp_x is not None:
    SETS.append(('extra', C_X, names_x, samp_x))


def get(S, p):
    return S[3][:, S[2].index(p)] if p in S[2] else None


# parametri: r, poi quelli del tree, poi quelli in piu' degli altri modelli
params = ['r_tensor'] + [n for n in names_b if n != 'r_tensor']
for S in SETS[1:]:
    params += [n for n in S[2] if n not in params]
n = len(params)
rng = {}
for p in params:
    v = np.concatenate([x for x in (get(S, p) for S in SETS) if x is not None])
    lo, hi = np.percentile(v, [0.5, 99.5])
    if p == 'r_tensor':
        lo, hi = min(lo, fine[post_r > 1e-3].min()), max(hi, fine[post_r > 1e-3].max())
        if a.r_true is not None:
            lo, hi = min(lo, a.r_true), max(hi, a.r_true)
    pad = 0.05 * (hi - lo)
    rng[p] = (lo - pad, hi + pad)

fig, axes = plt.subplots(n, n, figsize=(1.55 * n + 1, 1.55 * n + 1))
kw = dict(bins=30, plot_datapoints=False, plot_density=False,
          levels=(0.393, 0.865), smooth=1.0)
for i, pi in enumerate(params):
    for j, pj in enumerate(params):
        ax = axes[i, j]
        if j > i:
            ax.axis('off')
            continue
        rj, ri = rng[pj], rng[pi]
        if i == j:
            for S in SETS:
                if S[0] == 'prof' and pj == 'r_tensor':
                    ax.plot(fine, post_r / np.trapezoid(post_r, fine),
                            color=C_P, lw=1.3)
                elif get(S, pj) is not None:
                    ax.hist(get(S, pj), bins=40, range=rj, density=True,
                            histtype='step', color=S[1], lw=1.3)
            ax.set_yticks([])
        else:
            for S in SETS:
                xj, xi = get(S, pj), get(S, pi)
                if xj is not None and xi is not None:
                    corner.hist2d(xj, xi, ax=ax, range=[rj, ri], color=S[1],
                                  **kw)
            if pj == 'r_tensor':
                ax.axvline(r_star, color=C_P, lw=1, ls='--')
            ax.set_ylim(ri)
        if pj == 'r_tensor' and a.r_true is not None:
            ax.axvline(a.r_true, color='k', lw=1)
        ax.set_xlim(rj)
        ax.tick_params(labelsize=7)
        if i < n - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(label(pj), fontsize=11)
            for t in ax.get_xticklabels():
                t.set_rotation(45)
        if j > 0 or i == 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel(label(pi), fontsize=11)
fig.subplots_adjust(hspace=0.06, wspace=0.06, left=0.07, bottom=0.07,
                    right=0.98, top=0.95)
rb = samp_b[:, names_b.index('r_tensor')]
lines = [(f'Bayesian (tree): {M0}  —  r = {np.median(rb) * 1e3:.1f} ± '
          f'{rb.std() * 1e3:.1f} e-3', C_B),
         (f"profile scheme: {M1}  —  r* = {r_star * 1e3:.1f} e-3, r = "
          f"{res['r_med_pl'] * 1e3:.1f} ± {res['r_std_pl'] * 1e3:.1f} e-3 "
          '(other parameters: emcee at r = r*)', C_P)]
if samp_x is not None:
    rx = samp_x[:, names_x.index('r_tensor')]
    lines.append((f'tree, {MX} (r free)  —  r = {np.median(rx) * 1e3:.1f} ± '
                  f'{rx.std() * 1e3:.1f} e-3', C_X))
if a.r_true is not None:
    lines.append((f'true r = {a.r_true * 1e3:.1f} e-3 (black line)', 'k'))
lines.append(('contours: 68% and 95%; dashed orange: r = r*', '#555'))
fig.text(0.42, 0.93, a.title, fontsize=14, weight='bold')
for k, (t, c) in enumerate(lines):
    fig.text(0.42, 0.90 - 0.025 * k, t, fontsize=10, color=c)
fig.savefig(a.out)
fig.savefig(os.path.splitext(a.out)[0] + '.png', dpi=90)
print('written', a.out)
