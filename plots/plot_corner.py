"""
Corner plot of one set: Bayesian posterior of the tree (PolyChord of the
final model, all the parameters, r included) and posterior of the profile
scheme (emcee of the final model with r fixed at r*), overlaid. In the r
column: for the profile, the posterior of r from the profile (prior of the
tree) on the diagonal and the line r = r* in the 2D panels; true r in black.

  python plots/plot_corner.py <tree_dir> <profile_dir> --r-true 0.0111 \
      --title "set 47 rand (d12s5)" --out corner_047_rand.pdf

tree_dir: runs/set_<i>_<kind>/tree; profile_dir: the output directory of
the profile scheme (tree/profile: result.json, profile_<model>.npz,
post_<model>_rstar/emcee.npz).
"""
import argparse
import json
import os
import re

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import corner  # noqa: E402

from bbsure.config import free_param_names  # noqa: E402
from bbsure.runner import equal_weights_file, evidence_run_dir  # noqa: E402

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

# ── Bayesian posterior of the tree ────────────────────────────────────────
_c = os.path.join(a.tree, 'conclusion_tree.json')
concl = json.load(open(_c if os.path.isfile(_c) else
                       os.path.join(a.tree, 'conclusion.json')))
M0 = concl['model']
cfg = yaml.safe_load(open(os.path.join(evidence_run_dir(a.tree, M0),
                                       'config.yml')))
names_b = free_param_names(cfg['BBCompSep'])
samp_b = np.loadtxt(equal_weights_file(evidence_run_dir(a.tree, M0)),
                    ndmin=2)[:, 2:]

# ── profile scheme: emcee at r = r*, profile of r ─────────────────────────
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


# ── optional: another model of the tree (r free) ──────────────────────────
names_x = samp_x = None
if a.extra_pc:
    MX = os.path.basename(os.path.normpath(a.extra_pc))[:-3]
    cx = yaml.safe_load(open(os.path.join(a.extra_pc, 'config.yml')))
    names_x = free_param_names(cx['BBCompSep'])
    samp_x = np.loadtxt(equal_weights_file(a.extra_pc), ndmin=2)[:, 2:]


def label(n):
    """Name of the parameter in LaTeX."""
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


# sample sets: (name, colour, names, samples); the profile has r fixed
SETS = [('bayes', C_B, names_b, samp_b), ('prof', C_P, names_p, samp_p)]
if samp_x is not None:
    SETS.append(('extra', C_X, names_x, samp_x))


def get(S, p):
    return S[3][:, S[2].index(p)] if p in S[2] else None


# parameters: r, then those of the tree, then the extra ones of the other models
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
