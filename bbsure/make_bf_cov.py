"""
Covarianza del best fit per un set: la covarianza delle sims (fiduciale d1s1,
r = 0) piu' la correzione gaussiana (Knox) per la differenza tra lo spettro
del best fit del tree e il fiduciale delle sims:

    M_bf = M_sims + K(S_bf + N) - K(S_fid + N)

K e' la covarianza di Knox dei cross-spettri a ogni bandpower b,

    K[(ij), (kl)] = (C_ik C_jl + C_il C_jk) / nu_b ,

con C = S + N matrici 6x6 tra le frequenze (N = cells_noise). Il numero di
modi effettivo nu_b e' calibrato sulle sims: mediana su (ij) di
(C0_ii C0_jj + C0_ij^2) / M_sims[(ij),(ij)]. La parte che dipende dalla
maschera e dal rumore resta quella delle sims; si aggiunge solo la varianza
cosmica che manca (es. la parte tensoriale con r != 0).

Il best fit e' quello del modello finale del tree (conclusion.json): non usa
gli r veri.

Scrive in <out>:
  cells_model_bf.fits  spettro del best fit (anche fiduciale per HL)
  cells_cov_bf.fits    sacc dei dati con la covarianza nuova

Uso: python make_bf_cov.py --tree <runs_dir del tree> --cells <dir spettri>
         --fiducial <cls_fid_d1s1_hits.fits> --out <dir>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import sacc
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decision_tree as dt  # noqa: E402


def best_fit(tree_dir):
    """Modello finale e best fit (ML) dal decision log del tree."""
    concl = json.load(open(os.path.join(tree_dir, 'conclusion.json')))
    model = concl['model']
    ev = json.load(open(os.path.join(tree_dir, 'decision_log.json')))
    fits = [e for e in ev if e['node'] == 'fit' and e.get('model') == model
            and e.get('best')]
    if not fits:
        raise RuntimeError(f"no best fit for {model} in {tree_dir}")
    return model, fits[-1]['best']


def model_spectra(tree_dir, model, best, cells, fiducial, out):
    """Spettro del modello al best fit (BBCompSep predicted_spectra)."""
    cfg = yaml.safe_load(open(os.path.join(tree_dir, f'{model}_PC',
                                           'config.yml')))
    cs = cfg['BBCompSep']
    cs['sampler'] = 'predicted_spectra'
    cs['predict_at_minimum'] = False
    cs['predict_to_sacc'] = True
    entries = dt.iter_param_entries(cs)
    for k, v in best.items():
        entries[k][1:] = ['fixed', [float(v)]]
    assert dt.free_param_names(cs) == [], dt.free_param_names(cs)
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
    """Spettri BB del sacc s come matrici [nbpw, nfreq, nfreq]."""
    nf = len(tracers)
    nb = len(s.mean[idx[(tracers[0], tracers[0])]])
    C = np.zeros([nb, nf, nf])
    for i, t1 in enumerate(tracers):
        for j, t2 in enumerate(tracers):
            key = (t1, t2) if (t1, t2) in idx else (t2, t1)
            C[:, i, j] = s.mean[idx[key]]
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', required=True)
    ap.add_argument('--cells', required=True)
    ap.add_argument('--fiducial', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, best = best_fit(args.tree)
    print(f"best fit of {model}: r = {best.get('r_tensor', 0.):.4g}")
    f_bf = model_spectra(args.tree, model, best, args.cells, args.fiducial,
                         args.out)

    cs = yaml.safe_load(open(os.path.join(args.tree, f'{model}_PC',
                                          'config.yml')))['BBCompSep']
    lmin, lmax = cs['l_min'], cs['l_max']

    def load(f):
        """Solo BB, con gli stessi tagli in ell del fit."""
        s = sacc.Sacc.load_fits(f)
        for c in ('cl_ee', 'cl_eb', 'cl_be'):
            s.remove_selection(c)
        s.remove_selection(ell__gt=lmax)
        s.remove_selection(ell__lt=lmin)
        return s

    s_noi = load(os.path.join(args.cells, 'cells_noise.fits'))
    s_fid = load(args.fiducial)
    s_bf = load(f_bf)
    # sacc completo dei dati: la covarianza nuova va scritta nella sua
    # numerazione (BBCompSep controlla che coincida con quella dei dati)
    s_out = sacc.Sacc.load_fits(os.path.join(args.cells, 'cells_coadded.fits'))
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
    fcov = os.path.join(args.out, 'cells_cov_bf.fits')
    s_out.save_fits(fcov, overwrite=True)
    print(f"written {fcov} and {f_bf}")


if __name__ == '__main__':
    main()
