"""
Fit the PySM foregrounds with the Gaussian model of the simulations, to
build Gaussian covariance sims that match the PySM data.

For dust (at nu0_d = 353 GHz) and synchrotron (at nu0_s = 23 GHz),
monochromatic PySM maps in uK_CMB are analysed with NaMaster in the chosen
apodized mask (B purification, bandpowers of data/bpw_edges.txt), and

    D_ell^XX = A_XX * (ell / 80)^alpha,   XX = EE, BB

is fitted over [lmin, lmax] (weights ~ number of modes per bin). The mean
spectral parameters (beta_d, T_d, beta_s) are computed inside the mask,
both with uniform weights and weighted by the polarized intensity P^2
(which is what sets the effective SED of the Gaussian model).

Writes a yaml with the FG_PARAMS keys of bbsure/sims/sky.py, usable with
`generate_maps.py --fg-params`. alpha_d / alpha_s are taken from BB (the
generator uses a single slope for EE and BB); the EE slopes are reported.

With --nside-in N (e.g. 512) PySM runs at N and the maps are downgraded to
--nside in harmonic space (sky.harmonic_downgrade_qu), as generate_maps.py
--pysm-nside-in does for the data; without it PySM runs directly at --nside
(pixel-space ud_grade: pixel window and aliasing in the foreground maps).

Usage: python scripts/fit_fg_spectra.py [--model d1s1]
           [--mask mask_hits_C2_7deg_nside128] [--nside 128] [--nside-in 512]
           [--edges bpw_edges_nside128.txt] [--lmin 30] [--lmax 250]
           --out configs/fg_params_d1s1_hits_n128.yml
"""
import argparse
import os
import re
import numpy as np
import healpy as hp
import pymaster as nmt
import yaml
import pysm3
import pysm3.units as u

from bbsure.sims import sky as gm

p = argparse.ArgumentParser()
p.add_argument('--model', default='d1s1')
p.add_argument('--mask', default='mask_apo_C2_7deg')
p.add_argument('--nside', type=int, default=64)
p.add_argument('--nside-in', type=int, default=0)
p.add_argument('--edges', default='bpw_edges.txt')
p.add_argument('--lmin', type=float, default=30)
p.add_argument('--lmax', type=float, default=190)
p.add_argument('--out', required=True)
args = p.parse_args()
out = args.out

nside = args.nside
mask = hp.read_map(os.path.join(gm.DATA_DIR, 'masks', f'{args.mask}.fits.gz'))
presets = re.findall(r'[a-z]+\d+', args.model)
sky = pysm3.Sky(nside=args.nside_in or nside, preset_strings=presets)
dust = [c for c in sky.components if hasattr(c, 'mbb_index')][0]
synch = [c for c in sky.components if hasattr(c, 'pl_index')][0]

edges = np.loadtxt(os.path.join(gm.DATA_DIR, args.edges)).astype(int)
edges = edges[edges <= 3 * nside]
b = nmt.NmtBin.from_edges(edges[:-1], edges[1:])
leff = b.get_effective_ells()
sel = (leff >= args.lmin) & (leff <= args.lmax)
w = (2 * leff + 1) * np.diff(edges)                  # ~ modes per bin
dl = leff * (leff + 1) / (2 * np.pi)


def qu_cmb(comp, nu0):
    m = comp.get_emission(nu0 * u.GHz)
    m = m.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(nu0 * u.GHz))
    if args.nside_in:
        return tuple(gm.harmonic_downgrade_qu(m[1].value, m[2].value, nside))
    return m[1].value, m[2].value


def at_nside(x):
    """Spectral parameter map at the analysis nside (pixel average)."""
    x = np.asarray(x)
    return hp.ud_grade(x, nside) if x.ndim and x.size != hp.nside2npix(nside) \
        else x


def fit_pl(d):
    """Weighted least squares of log D = log A + alpha log(l/80)."""
    x, y = np.log(leff[sel] / 80.), np.log(d[sel])
    W = np.diag(w[sel])
    X = np.vstack([np.ones_like(x), x]).T
    coef = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    resid = y - X @ coef
    return float(np.exp(coef[0])), float(coef[1]), float(np.sqrt(np.average(resid**2, weights=w[sel])))


res = {}
spectra = {}
for name, comp, nu0 in [('d', dust, gm.FG_PARAMS['nu0_d']),
                        ('s', synch, gm.FG_PARAMS['nu0_s'])]:
    Q, U = qu_cmb(comp, nu0)
    f = nmt.NmtField(mask, [Q, U], spin=2, purify_b=True)
    cl = nmt.compute_full_master(f, f, b)
    ee, bb = cl[0] * dl, cl[3] * dl
    spectra[name] = (ee, bb)
    for xx, d in [('EE', ee), ('BB', bb)]:
        A, al, rms = fit_pl(d)
        res[f'{name}_{xx}'] = (A, al, rms)
        print(f"{'dust' if name == 'd' else 'synch'} {xx} at {nu0:.0f} GHz: "
              f"A = {A:.4g} uK_CMB^2, alpha = {al:+.3f}  (rms log-residual {rms:.3f})")

# mean spectral parameters inside the mask
Qd, Ud = qu_cmb(dust, gm.FG_PARAMS['nu0_d'])
Qs, Us = qu_cmb(synch, gm.FG_PARAMS['nu0_s'])
wd, ws = mask * (Qd**2 + Ud**2), mask * (Qs**2 + Us**2)
beta_d = at_nside(dust.mbb_index.value)
T_d = at_nside(dust.mbb_temperature.value)
beta_s = at_nside(synch.pl_index.value)
means = {
    'beta_d': (np.average(beta_d, weights=mask), np.average(beta_d, weights=wd)),
    'T_d':    (np.average(T_d, weights=mask), np.average(T_d, weights=wd)),
    'beta_s': (np.average(beta_s, weights=mask), np.average(beta_s, weights=ws)),
}
for k, (m1, m2) in means.items():
    print(f"{k:7s}: mask mean {m1:.4f}   P^2-weighted mean {m2:.4f}")
print(f"std inside mask: beta_d {np.sqrt(np.cov(beta_d, aweights=mask)):.4f}, "
      f"T_d {np.sqrt(np.cov(T_d, aweights=mask)):.3f} K, "
      f"beta_s {np.sqrt(np.cov(beta_s, aweights=mask)):.4f}")

params = {
    'beta_d':  means['beta_d'][1],
    'T_d':     means['T_d'][1],
    'nu0_d':   gm.FG_PARAMS['nu0_d'],
    'A_d_EE':  res['d_EE'][0],
    'A_d_BB':  res['d_BB'][0],
    'alpha_d': res['d_BB'][1],
    'beta_s':  means['beta_s'][1],
    'nu0_s':   gm.FG_PARAMS['nu0_s'],
    'A_s_EE':  res['s_EE'][0],
    'A_s_BB':  res['s_BB'][0],
    'alpha_s': res['s_BB'][1],
}
params = {k: round(float(v), 6) for k, v in params.items()}
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as fo:
    fo.write(f"# Gaussian foreground parameters fitted on PySM {args.model}, mask {args.mask},\n"
             f"# nside {nside}" + (f" (PySM at {args.nside_in}, harmonic downgrade)" if args.nside_in else "") + ",\n"
             f"# ell in [{args.lmin:.0f}, {args.lmax:.0f}], made by fit_fg_spectra.py.\n"
             f"# alpha from BB; EE slopes: dust {res['d_EE'][1]:+.3f}, synch {res['s_EE'][1]:+.3f}.\n"
             f"# beta_d, T_d, beta_s: P^2-weighted means inside the mask.\n")
    yaml.safe_dump(params, fo, sort_keys=False)
np.savez(out.replace('.yml', '_spectra.npz'), leff=leff, sel=sel,
         dust_ee=spectra['d'][0], dust_bb=spectra['d'][1],
         synch_ee=spectra['s'][0], synch_bb=spectra['s'][1])
print(f"written {out}")
