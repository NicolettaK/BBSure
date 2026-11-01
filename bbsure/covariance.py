"""
Best-fit covariance for a data set: the covariance of the sims (fiducial
d1s1, r = 0) plus the Gaussian (Knox) correction for the difference between
the best-fit spectrum of the tree and the fiducial of the sims:

    M_bf = M_sims + K(S_bf + N) - K(S_fid + N)

K is the Knox covariance of the cross-spectra at each bandpower b,

    K[(ij), (kl)] = (C_ik C_jl + C_il C_jk) / nu_b ,

with C = S + N 6x6 matrices between the frequencies (N = cells_noise). The
effective number of modes nu_b is calibrated on the sims: median over (ij)
of (C0_ii C0_jj + C0_ij^2) / M_sims[(ij),(ij)]. The part that depends on the
mask and on the noise stays that of the sims; only the missing cosmic
variance is added (e.g. the tensor part with r != 0).

The best fit is that of the final model of the tree (conclusion.json): it
does not use the true r.

Writes in <out>:
  cells_model_bf.fits  best-fit spectrum (also the fiducial for HL)
  cells_cov_bf.fits    data sacc with the new covariance

Usage: python scripts/make_bf_cov.py --tree <runs_dir of the tree>
           --cells <spectra dir> --fiducial <cls_fid_...fits> --out <dir>
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import sacc
import yaml

from .config import free_param_names, iter_param_entries


def best_fit(tree_dir):
    """Final model and best fit (ML) from the decision log of the tree."""
    concl = json.load(open(os.path.join(tree_dir, 'conclusion.json')))
    model = concl['model']
    ev = json.load(open(os.path.join(tree_dir, 'decision_log.json')))
    fits = [e for e in ev if e['node'] == 'fit' and e.get('model') == model
            and e.get('best')]
    if not fits:
        raise RuntimeError(f"no best fit for {model} in {tree_dir}")
    return model, fits[-1]['best']


def model_spectra(tree_dir, model, best, cells, fiducial, out):
    """Spectrum of the model at the best fit (BBCompSep predicted_spectra)."""
    cfg = yaml.safe_load(open(os.path.join(tree_dir, f'{model}_PC',
                                           'config.yml')))
    cs = cfg['BBCompSep']
    cs['sampler'] = 'predicted_spectra'
    cs['predict_at_minimum'] = False
    cs['predict_to_sacc'] = True
    entries = iter_param_entries(cs)
    for k, v in best.items():
        entries[k][1:] = ['fixed', [float(v)]]
    assert free_param_names(cs) == [], free_param_names(cs)
    tmp = tempfile.mkdtemp()
    cfg_path = os.path.join(tmp, 'config.yml')
    yaml.safe_dump(cfg, open(cfg_path, 'w'), sort_keys=False)
    cmd = [sys.executable, '-m', 'bbpower', 'BBCompSep',
           f"--cells_coadded={cells}/cells_coadded.fits",
           f"--cells_noise={cells}/cells_noise.fits",
           f"--cells_fiducial={fiducial}",
           f"--cells_coadded_cov={cells}/cells_coadded.fits",
           f"--output_dir={tmp}", f"--config_copy={tmp}/cc.yml",
           f"--config={cfg_path}"]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    fout = os.path.join(out, 'cells_model_bf.fits')
    shutil.copy(os.path.join(tmp, 'cells_model.fits'), fout)
    shutil.rmtree(tmp)
    return fout


def bb_matrices(s, tracers, idx):
    """BB spectra of the sacc s as matrices [nbpw, nfreq, nfreq]."""
    nf = len(tracers)
    nb = len(s.mean[idx[(tracers[0], tracers[0])]])
    C = np.zeros([nb, nf, nf])
    for i, t1 in enumerate(tracers):
        for j, t2 in enumerate(tracers):
            key = (t1, t2) if (t1, t2) in idx else (t2, t1)
            C[:, i, j] = s.mean[idx[key]]
    return C


def bestfit_covariance(tree, cells, fiducial, out):
    """Writes cells_model_bf.fits and cells_cov_bf.fits in `out` (see the
    module docstring) and returns their paths."""
    os.makedirs(out, exist_ok=True)

    model, best = best_fit(tree)
    print(f"best fit of {model}: r = {best.get('r_tensor', 0.):.4g}")
    f_bf = model_spectra(tree, model, best, cells, fiducial, out)

    cs = yaml.safe_load(open(os.path.join(tree, f'{model}_PC',
                                          'config.yml')))['BBCompSep']
    lmin, lmax = cs['l_min'], cs['l_max']

    def load(f):
        """BB only, with the same ell cuts as the fit."""
        s = sacc.Sacc.load_fits(f)
        for c in ('cl_ee', 'cl_eb', 'cl_be'):
            s.remove_selection(c)
        s.remove_selection(ell__gt=lmax)
        s.remove_selection(ell__lt=lmin)
        return s

    s_noi = load(os.path.join(cells, 'cells_noise.fits'))
    s_fid = load(fiducial)
    s_bf = load(f_bf)
    # full data sacc: the new covariance must be written in its numbering
    # (BBCompSep checks that it matches the one of the data)
    s_out = sacc.Sacc.load_fits(os.path.join(cells, 'cells_coadded.fits'))
    tracers = sorted(s_out.tracers.keys())
    pairs = s_out.get_tracer_combinations('cl_bb')
    rows = {}
    for p in pairs:
        ind = s_out.indices('cl_bb', p)
        ell = np.array([s_out.data[k]['ell'] for k in ind])
        rows[p] = ind[(ell >= lmin) & (ell <= lmax)]
    Mfull = s_out.covariance.covmat.copy()

    def index(s):
        return {p: s.indices('cl_bb', p) for p in pairs}

    N = bb_matrices(s_noi, tracers, index(s_noi))
    C0 = bb_matrices(s_fid, tracers, index(s_fid)) + N
    C1 = bb_matrices(s_bf, tracers, index(s_bf)) + N
    nb = C0.shape[0]
    assert all(len(rows[p]) == nb for p in pairs), (nb, len(rows[pairs[0]]))

    tpos = {t: i for i, t in enumerate(tracers)}
    ij = [(tpos[p[0]], tpos[p[1]]) for p in pairs]
    allrows = np.concatenate([rows[p] for p in pairs])
    M = Mfull[np.ix_(allrows, allrows)]
    Mnew_full = Mfull.copy()
    nu = np.zeros(nb)
    for b in range(nb):
        r = np.array([rows[p][b] for p in pairs])

        def knox(C):
            K = np.zeros([len(pairs), len(pairs)])
            for a, (i, j) in enumerate(ij):
                for c, (k, l) in enumerate(ij):
                    K[a, c] = C[b, i, k] * C[b, j, l] + C[b, i, l] * C[b, j, k]
            return K
        K0, K1 = knox(C0), knox(C1)
        nu[b] = np.median(np.diag(K0) / np.diag(Mfull[np.ix_(r, r)]))
        Mnew_full[np.ix_(r, r)] += (K1 - K0) / nu[b]

    Mnew = Mnew_full[np.ix_(allrows, allrows)]
    ev_new, ev_old = np.linalg.eigvalsh(Mnew), np.linalg.eigvalsh(M)
    ratio = np.diag(Mnew) / np.diag(M)
    print(f"effective modes per bandpower: {np.round(nu, 1)}")
    print(f"diag(M_bf)/diag(M_sims): min {ratio.min():.3f}, median "
          f"{np.median(ratio):.3f}, max {ratio.max():.3f}; smallest "
          f"eigenvalue ratio {ev_new.min() / ev_old.min():.3f}")
    if ev_new.min() <= 0:
        raise RuntimeError("best-fit covariance is not positive definite")

    s_out.add_covariance(Mnew_full, overwrite=True)
    fcov = os.path.join(out, 'cells_cov_bf.fits')
    s_out.save_fits(fcov, overwrite=True)
    print(f"written {fcov} and {f_bf}")
    return f_bf, fcov
