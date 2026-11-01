"""
Ramo "profilo" del tree, provato a posteriori su un tree gia' concluso
(pensato per le detection: r grande e nessun M_1,r preferito).

  python profile_tree.py <src_root> <out_root> <index> <d9s4|rand>
         [--lnb-res 0] [--lnb-full 0] [--n 37] [--max-it 4]

1. M_0 = modello finale del tree; primo candidato C = ultimo M_1,r provato
   dal tree (solo i suoi gruppi nuovi, aggiunti a M_0). Se il tree non ha
   trovato nessun momento nuovo (niente rivelato in Res_r0): nessun candidato,
   niente Res_wr*, si va direttamente al punto 5 con il profilo di M_0,r.
2. Profilo di r di M_0 + C -> r_k (minimo; i punti ML non convergiti si
   rifittano dal vicino).
3. Ogni modello attorno al proprio best fit ML, ognuno con il suo r (M_0 + C
   a r = r_k, M_0 al minimo del suo profilo), liberi solo gli spettri dei suoi
   momenti (ampiezze, pendenze) e gli STESSI beta nei due modelli (quelli dei
   gruppi di M_0 + C: es. dust vs dust + synch -> beta_d e beta_s liberi in
   entrambi); ln B > lnb_full -> C entra nel modello.
   Se C e' respinto ci si ferma (niente Res_wr*) -> punto 5.
4. Res_wr* (solo se C e' entrato, o se si parte da M_1): ogni gruppo g si
   testa al suo r, r_g = minimo del profilo di (modello corrente + g):
   residui del modello corrente rifittato a r = r_g, gruppi non ancora nel modello e non gia' respinti come
   candidati (anche ordini superiori). Screening HPD 68%, poi evidenza
   contro il nullo (stesso punto, ampiezze del gruppo a zero, liberi solo i
   beta del gruppo): passa se ln B > lnb_res. Il migliore diventa il nuovo
   candidato -> punto 2 (r**, ...). Nessuno -> stop.
5. Risultato dal profilo dell'ultimo modello (se non e' entrato niente,
   quello del vecchio M_0,r): best fit, 68%, UL 95% e posterior di r con il
   profilo come likelihood: prior del tree su r (_pl, per il bias) e con
   r >= 0 (_pl0).
6. Altri parametri: posterior bayesiano (emcee) del modello finale con r
   fissato al minimo del suo profilo (condizionato a r = r*), tutto il resto
   libero -> mediana, 16-84%, sigma in result.json (params_at_rstar).
7. Decisione: 0 dentro l'intervallo centrale a 3 sigma del posterior dal
   profilo (prior del tree) -> upper limit (r_95 con r >= 0), altrimenti r
   detection (mediana e 68%), come il test a 3 sigma del tree.

Se nel tree M_1,0 era preferito su M_0,r, si parte da M_1 (r libero), il
modello migliore trovato: niente confronto del candidato, Res_wr* al minimo
del profilo di M_1, e se non entra altro il risultato e' il profilo di M_1.

Legge spettri e config da <src_root>/runs/set_<i>_<kind>/, scrive in
<out_root>/set_<i>_<kind>/ (fit, decision_log, result.json).
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
import os
import re
import sys

import numpy as np
import yaml
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decision_tree as dt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('src', nargs='?')
ap.add_argument('out', nargs='?')
ap.add_argument('index', type=int, nargs='?')
ap.add_argument('kind', nargs='?')
# dentro la pipeline (decision_tree.py con run: profile_scheme: true)
ap.add_argument('--config', help='tree_config of the tree')
ap.add_argument('--tree-dir', help='runs_dir of the concluded tree')
ap.add_argument('--out', dest='out_dir', help='output directory')
ap.add_argument('--lnb-res', type=float, default=0.0)
ap.add_argument('--lnb-full', type=float, default=0.0)
ap.add_argument('--n', type=int, default=37)
ap.add_argument('--max-it', type=int, default=4)
ap.add_argument('--nproc', type=int,
                default=int(os.environ.get('PROFILE_NPROC', 8)),
                help='profile points fitted in parallel (one ML each)')
ap.add_argument('--no-stop', action='store_true',
                help='diagnostic: after a rejected candidate still run '
                     'Res_wr* on all the other groups')
ap.add_argument('--ignore-m10', action='store_true',
                help='test: ignore M_1,0 preferred in the tree (detection path)')
a = ap.parse_args()

if a.config:
    name = os.path.basename(os.path.normpath(a.tree_dir))
    tree, out, cfg_path = a.tree_dir, a.out_dir, a.config
else:
    name = f'set_{a.index:03d}_{a.kind}'
    src = os.path.join(a.src, 'runs', name)
    tree = os.path.join(src, 'tree')
    out = os.path.join(a.out, name)
    cfg_path = os.path.join(src, 'tree_config.yml')
os.makedirs(out, exist_ok=True)

tc = yaml.safe_load(open(cfg_path))
tc['paths']['runs_dir'] = out
log = dt.DecisionLog(out)
F = dt.Fitter(dt.BBCompSepRunner(tc, log),
              hpd_frac=tc['thresholds'].get('hpd_frac', 0.68))
R = F.R
mc = tc['moments']
GROUPS = mc.get('res_r0', ['d', 's', 'dT', 'd2', 's2', 'ds1', 'ds2'])
REQ = mc.get('requires', {})

# ── modello finale e candidato dal tree ───────────────────────────────────
# conclusion_tree.json: scritto dalla pipeline (conclusion.json diventa poi
# quella finale dello schema)
_c = os.path.join(tree, 'conclusion_tree.json')
concl = json.load(open(_c if os.path.isfile(_c) else
                       os.path.join(tree, 'conclusion.json')))
ev = json.load(open(os.path.join(tree, 'decision_log.json')))
M0 = dt.model_from_label(concl['model'])
cands = [e['model'] for e in ev if e['node'] == 'fit' and 'M_1,r' in
         e.get('question', '') and e.get('model', '').endswith('_r')]
M1c = dt.model_from_label(cands[-1]) if cands else None
if M1c is None or M1c.moments <= M0.moments:
    # il tree non ha trovato nessun momento nuovo da provare (niente rivelato
    # in Res_r0): nessun candidato, il risultato e' il profilo di M_0,r
    M1c = None
else:
    # il candidato puo' essere stato provato su un modello precedente: si
    # aggiungono al modello finale solo i suoi gruppi nuovi (con la stessa
    # potatura), cosi' M_0 e' annidato in M_0 + C
    newg = M1c.moments - M0.moments
    amps_new = {dt.amp_name(p, sp) for _, p, sp in dt.group_spectra(newg)}
    M1c = dt.Model(M0.moments | newg, True,
                   M0.off | (M1c.off & amps_new))
cfg = yaml.safe_load(open(os.path.join(tree, f'{M0.label}_PC', 'config.yml')))
names = dt.free_param_names(cfg['BBCompSep'])
samp = np.loadtxt(os.path.join(tree, f'{M0.label}_PC', 'polychord',
                               'pch_equal_weights.txt'), ndmin=2)[:, 2:]
r_b = samp[:, names.index('r_tensor')]


def post_summary(names, samples):
    """Mediana, 16-84% e sigma di ogni parametro."""
    q = np.percentile(samples, [15.865, 50., 84.135], axis=0)
    return {n: dict(med=float(q[1, k]), lo1=float(q[0, k]),
                    hi1=float(q[2, k]), std=float(np.std(samples[:, k])))
            for k, n in enumerate(names)}


# parte bayesiana (tree): best fit ML e posterior (PolyChord) di M_0,r
_z = np.load(os.path.join(tree, f'{M0.label}_ML', 'chi2.npz'))
BAYES = dict(model=M0.label,
             best_fit={str(n): float(v) for n, v in
                       zip(_z['names'], np.atleast_1d(_z['params']))},
             chi2_ml=float(_z['chi2']), posterior=post_summary(names, samp))
med, sig = float(np.median(r_b)), float(np.std(r_b))
GRID = np.linspace(med - 6 * sig, med + 3 * sig, a.n)
log.info(f"==> {name}: final model {M0.label}, Bayesian r = {med:.4f} +- "
         f"{sig:.4f}; candidate: {M1c.label if M1c else 'none'}")


STEP = GRID[1] - GRID[0]
EDGE_DCHI2 = 16.    # ai bordi il profilo deve salire almeno di tanto (4 sigma):
                    # serve per gli intervalli a 3 sigma del posterior


def profile(M, tag):
    """Profilo chi2(r) di M; ritorna (griglia fine, chi2 interpolato, r_min).
    Un fit ML non convergito puo' solo stare sopra il profilo vero: i punti
    piu' alti di 0.5 rispetto alla media dei vicini (o con chi2 non finito) si
    rifittano partendo dal best fit del vicino piu' basso (fino a 3 passate) e
    si tiene il minimo. Se il minimo e' vicino a un bordo o il chi2 ai bordi
    non sale di almeno EDGE_DCHI2, la griglia si estende da quel lato."""
    m = M.with_r(False)
    grid = list(GRID)
    fits = {}

    def fit(r, centers=None, sfx=''):
        k = int(round((r - GRID[0]) / STEP))
        return R.ml(m, f'prof_{tag}_{k:+03d}{sfx}', fixed={'r_tensor': float(r)},
                    centers=centers)

    def batch(jobs):
        """Fit ML indipendenti in parallelo: [(r, centers, sfx)] -> fits."""
        with ThreadPoolExecutor(max_workers=a.nproc) as ex:
            return list(ex.map(lambda j: fit(*j), jobs))

    for r, f in zip(grid, batch([(r, None, '') for r in grid])):
        fits[r] = f
    # punti con chi2 infinito (partenza fuori dai vincoli sui momenti): si
    # riparte con le ampiezze a zero tranne gli auto-spettri del primo ordine
    # (come base_at); se non resta nessun punto valido il modello non e'
    # fittabile
    bad = [r for r in fits if not np.isfinite(fits[r].chi2)]
    if bad:
        start = {x: 0. for x in M.amplitudes()
                 if not re.match(r'A_[ds]_(w10w10|w01w01)$', x)}
        for r, f in zip(bad, batch([(r, start, '_ML0') for r in bad])):
            if np.isfinite(f.chi2):
                fits[r] = f
    if not any(np.isfinite(f.chi2) for f in fits.values()):
        raise dt.NoValidStart(f'no valid ML start for the profile of '
                              f'{M.label}')
    # un intero tratto di fit non convergiti (plateau) non si vede come
    # "bump" rispetto ai vicini: si rifittano tutti i punti partendo dal
    # best fit del minimo globale e si tiene il migliore, finche' migliora
    for s in range(3):
        rb = min((r for r in fits if np.isfinite(fits[r].chi2)),
                 key=lambda r: fits[r].chi2)
        jobs = [(r, fits[rb].best, f'_s{s}') for r in sorted(fits) if r != rb]
        n_imp = 0
        for (r, _, _), f in zip(jobs, batch(jobs)):
            if f.chi2 < fits[r].chi2 - 0.5 or not np.isfinite(fits[r].chi2):
                log.info(f"    profile point r = {r:.4f}: refit from the "
                         f"minimum, chi2 {fits[r].chi2:.2f} -> {f.chi2:.2f}")
                fits[r] = f
                n_imp += 1
        if not n_imp:
            break
    for ext in range(10):
        g = np.array(sorted(fits))
        chi2 = np.array([fits[r].chi2 for r in g])
        for p in range(3):
            bad = [i for i in range(len(g)) if not np.isfinite(chi2[i]) or
                   (0 < i < len(g) - 1 and chi2[i] - 0.5 * (chi2[i - 1] +
                                                            chi2[i + 1]) > 0.5)]
            if not bad:
                break
            jobs, idx = [], []
            for i in bad:
                nb = [j for j in (i - 1, i + 1) if 0 <= j < len(g)
                      and np.isfinite(chi2[j])]
                if not nb:
                    continue
                j = min(nb, key=lambda j: chi2[j])
                jobs.append((g[i], fits[g[j]].best, f'_p{p}'))
                idx.append(i)
            for i, f in zip(idx, batch(jobs)):
                log.info(f"    profile point r = {g[i]:.4f}: refit from the "
                         f"neighbour, chi2 {chi2[i]:.2f} -> {f.chi2:.2f}")
                if f.chi2 < chi2[i] or not np.isfinite(chi2[i]):
                    chi2[i], fits[g[i]] = f.chi2, f
        ok = np.isfinite(chi2)
        g, chi2 = g[ok], chi2[ok]
        imin = int(np.argmin(chi2))
        d = chi2 - chi2[imin]
        lo = imin < 3 or d[0] < EDGE_DCHI2
        hi = imin > len(g) - 4 or d[-1] < EDGE_DCHI2
        if not (lo or hi):
            break
        new = ([g[0] - STEP * (k + 1) for k in range(4)] if lo else []) + \
              ([g[-1] + STEP * (k + 1) for k in range(4)] if hi else [])
        log.info(f"    profile of {M.label}: extending the grid "
                 f"({'low' if lo else ''}{' high' if hi else ''})")
        jobs = [(r, fits[g[0] if r < g[0] else g[-1]].best, '') for r in new]
        for r, f in zip(new, batch(jobs)):
            fits[r] = f
    else:
        log.info(f"    WARNING: profile of {M.label} still open at the edges")
    np.savez(os.path.join(out, f'profile_{M.label}.npz'), r=g, chi2=chi2)
    PROF_FITS[M.label] = {r: fits[r] for r in g}
    fine = np.linspace(g[0], g[-1], 20001)
    cf = interp1d(g, chi2, kind='cubic')(fine)
    return fine, cf, float(fine[np.argmin(cf)])


PROF = {}          # label -> (griglia fine, chi2 interpolato)
PROF_FITS = {}     # label -> {r della griglia: fit ML}


# prior su r del tree (tophat) per il posterior con il profilo come likelihood
_rp = next(v for k, v in dt.iter_param_entries(R.base['BBCompSep']).items()
           if k == 'r_tensor')
R_PRIOR = (float(_rp[2][0]), float(_rp[2][-1]))


def summary(fine, cf):
    """Profilo: best fit, Delta chi2 = 1 e 2.71. Posterior con il profilo
    come likelihood ('pl'): prior del tree su r, e con r >= 0 ('pl0')."""
    d = cf - cf.min()
    i = int(np.argmin(cf))
    out = dict(r_best=float(fine[i]),
               r_lo1=float(fine[:i + 1][d[:i + 1] <= 1].min()),
               r_hi1=float(fine[i:][d[i:] <= 1].max()),
               r_ul95_profile=float(fine[i:][d[i:] <= 2.71].max()),
               dchi2_r0=float(np.interp(0., fine, d))
               if fine[0] <= 0. <= fine[-1] else float('nan'))
    for tag, lo in (('pl', R_PRIOR[0]), ('pl0', 0.0)):
        w = np.exp(-d / 2) * (fine >= lo) * (fine <= R_PRIOR[1])
        w /= w.sum()
        c = np.cumsum(w)
        q = lambda p: float(fine[np.searchsorted(c, p)])
        mean = float(np.sum(w * fine))
        out.update({f'r_med_{tag}': q(0.5), f'r_lo1_{tag}': q(0.15865),
                    f'r_hi1_{tag}': q(0.84135), f'r_95_{tag}': q(0.95),
                    f'r_lo3_{tag}': q(0.00135), f'r_hi3_{tag}': q(0.99865),
                    f'r_std_{tag}': float(np.sqrt(np.sum(w * (fine - mean)
                                                         ** 2)))})
    return out


def base_at(M, r, tag, centers=None):
    """ML di M con r fissato: FitResult con r_tensor nel best fit. Se il
    minimizer parte fuori dalla regione permessa dai vincoli sui momenti
    (chi2 infinito), riprova con le ampiezze a zero tranne gli auto-spettri
    del primo ordine (come screen_fit del tree)."""
    m = M.with_r(False)
    if centers is None and M.label in PROF_FITS:
        # punto di partenza: il fit del profilo piu' vicino a r (il minimizer
        # dal punto di partenza di default puo' fermarsi lontano dal minimo)
        pf = PROF_FITS[M.label]
        centers = pf[min(pf, key=lambda x: abs(x - r))].best
    res = R.ml(m, tag, fixed={'r_tensor': r}, centers=centers)
    if not np.isfinite(res.chi2):
        start = dict(centers or {})
        start.update({x: 0. for x in M.amplitudes()
                      if not re.match(r'A_[ds]_(w10w10|w01w01)$', x)})
        res = R.ml(m, tag + '_ML0', fixed={'r_tensor': r}, centers=start)
    if not np.isfinite(res.chi2):
        raise dt.NoValidStart(f'no valid ML start for {M.label} at r = {r}')
    res.best = dict(res.best, r_tensor=r)
    return res


def local_logz(M, base, free, tag):
    """Evidenza di M attorno al best fit `base` (r fissato): liberi solo i
    parametri in `free`; se nessuno e' libero, -chi2/2."""
    fixed = {k: v for k, v in base.best.items() if k not in free}
    cs = dt.build_config(R.base, M.with_r(False), 'polychord',
                         fixed)['BBCompSep']
    if not dt.free_param_names(cs):
        return -0.5 * base.chi2
    return R.polychord(M.with_r(False), tag, fixed=fixed,
                       centers=base.best)[0]


def null_logz(base, group, tag):
    """Evidenza del nullo: base, ampiezze del gruppo a zero, liberi solo i
    parametri spettrali del gruppo presenti nel modello."""
    free = set(dt.MOMENT_GROUPS[dt.cand_group(group)]['free'])
    fixed = {k: v for k, v in base.best.items() if k not in free}
    cs = dt.build_config(R.base, base.model, 'polychord', fixed)['BBCompSep']
    if not dt.free_param_names(cs):
        return -0.5 * base.chi2
    logZ, _, _, _, _ = R.polychord(base.model, tag, fixed=fixed,
                                   centers=base.best)
    return logZ




def prof(M, tag):
    if M.label not in PROF:
        fine, cf, _ = profile(M, tag)
        PROF[M.label] = (fine, cf)
    fine, cf = PROF[M.label]
    return fine, cf, float(fine[np.argmin(cf)])


def new_group(M1, M):
    """Gruppo/i di momenti di M1 non in M (per l'etichetta)."""
    return '+'.join(sorted(M1.moments - M.moments)) or 'restore'


# se nel tree M_1,0 era preferito su M_0,r, il modello migliore trovato e' M_1
# (stesso modello con r libero): si parte da li', senza candidato, e Res_wr*
# si fa al minimo del suo profilo; se non entra altro, il risultato e' il suo
m10 = [e for e in ev if e['node'] == 'pref' and 'M_1,0' in
       e.get('question', '') and e.get('answer') == 'Yes']
if a.ignore_m10:
    m10 = []
cur, cand = M0, M1c
rejected = set()          # gruppi gia' respinti come candidati
n_it = a.max_it
if not m10 and M1c is None:
    log.node('start', 'no new moment found by the tree (nothing detected in '
             'Res_r0): no candidate, result from the profile of M_0,r',
             M0.label)
    n_it = 0
if m10:
    cur = dt.model_from_label(m10[-1]['question'].split('(')[1]
                              .split(')')[0]).with_r(True)
    cand = None
    log.node('adopt', f'M_1,0 was preferred in the tree (ln B = '
             f'{m10[-1].get("lnB", float("nan")):+.2f}): start from M_1',
             cur.label)
for it in range(1, n_it + 1):
    if cand is None:
        fine, cf, r_k = prof(cur, f'it{it}')
        log.info(f"\n==> iteration {it}: model {cur.label}, no candidate, "
                 f"r_{it} = {r_k:.4f} (minimum of its profile)")
        log.node('profile', f'profile of {cur.label} -> r_{it}', f'{r_k:.4f}',
                 r_bayes_tree=med, sigma_bayes_tree=sig, **summary(fine, cf))
        try:
            base = base_at(cur, r_k, f'ml_{cur.label}_r{it}')
        except dt.NoValidStart as err:
            log.node('resid', 'Res_wr*: no valid ML start at r_k', 'stop',
                     reason=str(err))
            break
        return_to_res = True
    else:
        return_to_res = False
    if not return_to_res:
        g_c = new_group(cand, cur)
        # 2. profilo del modello corrente + candidato -> r_k
        try:
            fine, cf, r_k = prof(cand, f'it{it}')
        except dt.NoValidStart as err:
            # candidato non fittabile (vincoli sui momenti): respinto
            rejected.add(g_c)
            log.node('profile', f'profile of {cand.label}', 'unfittable',
                     reason=str(err))
            log.node('stop', f'candidate {g_c} cannot be fitted: stop, no '
                     'search for further moments', cur.label)
            break
        log.info(f"\n==> iteration {it}: model {cur.label}, candidate {g_c} "
                 f"({cand.label}), r_{it} = {r_k:.4f}")
        log.node('profile', f'profile of {cand.label} -> r_{it}', f'{r_k:.4f}',
                 r_bayes_tree=med, sigma_bayes_tree=sig, **summary(fine, cf))
        # 3. confronto: ogni modello attorno al proprio best fit ML, ognuno con il
        #    suo r; liberi solo i suoi momenti (ampiezze, pendenze) e i loro beta
        # il modello corrente al suo meglio: r al minimo del suo profilo
        _, _, r_cur = prof(cur, f'cur{it}')

        def moment_spectra(M):
            """Ampiezze e pendenze degli spettri dei momenti di M."""
            return {f(p, sp) for _, p, sp in M.spectra()
                    for f in (dt.amp_name, dt.slope_name)}

        # gli STESSI beta liberi nei due modelli (quelli dei gruppi del
        # candidato, che contiene il modello corrente): la differenza e' solo
        # negli spettri dei momenti, il fattore di Occam dei beta si cancella
        betas = set().union(*(dt.MOMENT_GROUPS[g]['free']
                              for g in cand.moments)) if cand.moments else set()
        free1 = moment_spectra(cand) | betas
        free0 = moment_spectra(cur) | betas
        b0 = b1 = None
        try:
            b0 = base_at(cur, r_cur, f'ml_{cur.label}_rcur{it}')
            b1 = base_at(cand, r_k, f'ml_{cand.label}_r{it}')
            lz1 = local_logz(cand, b1, free1, f'loc_{cand.label}_r{it}')
            lz0 = local_logz(cur, b0, free0, f'loc_{cur.label}_r{it}')
            lnb_full = lz1 - lz0
        except dt.NoValidStart as err:
            log.node('pref', f'{cand.label} at r_{it}', 'unfittable',
                     reason=str(err))
            lnb_full, lz1, lz0 = -np.inf, None, None
        ok = lnb_full > a.lnb_full
        log.node('pref', f'Each model at its own ML point ({cand.label}: r = '
                 f'{r_k:.4f}; {cur.label}: r = {r_cur:.4f}), is {cand.label} '
                 f'preferred?', 'Yes' if ok else 'No', lnB=lnb_full,
                 r_cand=r_k, r_cur=r_cur, logZ_cand=lz1,
                 logZ_cur=lz0, free_cand=sorted(free1),
                 free_cur=sorted(free0),
                 chi2_ml_cur=b0.chi2 if b0 else None,
                 chi2_ml_cand=b1.chi2 if b1 else None)
        if ok:
            cur, base = cand, b1
            log.node('adopt', f'{g_c} enters the model', cur.label)
        else:
            # candidato respinto: ci si ferma (niente Res_wr*), il risultato e'
            # il profilo del modello corrente
            rejected.add(g_c)
            if not a.no_stop:
                log.node('stop', f'candidate {g_c} rejected: stop, no search '
                         'for further moments', cur.label)
                break
            # diagnostica: Res_wr* sul modello corrente rifittato a r = r_k
            try:
                base = base_at(cur, r_k, f'ml_{cur.label}_r{it}')
            except dt.NoValidStart as err:
                log.node('resid', 'Res_wr*: no valid ML start at r_k',
                         'stop', reason=str(err))
                break
    # 4. Res_wr*: ogni gruppo g si testa al SUO r: r_g = minimo del profilo
    #    di (modello corrente + g); residui del modello corrente rifittato a
    #    r = r_g; screening HPD, poi evidenza contro il nullo nello stesso punto
    groups = [g for g in dt.candidate_groups(cur.with_r(False), GROUPS,
                                             extra_requires=REQ)
              if g not in rejected]
    passed = []
    for g in groups:
        Mg = cur.apply(g)
        try:
            fg_, cg_, r_g = prof(Mg, f'g{it}_{g}')
            base_g = base_at(cur, r_g, f'ml_{cur.label}_r{it}_{g}')
            rr = F.residual_test(base_g, g)
        except (dt.NoValidStart, ValueError) as err:
            log.node('resid_fit', f'Res_wr*: {g}', 'unfittable',
                     reason=str(err))
            continue
        where = f'Res_wr* (r = r_{g} = {r_g:.4f}, minimum of the profile of ' \
                f'{Mg.label}): moment {g} on the residuals of {cur.label}'
        if not rr.detected:
            log.node('resid_fit', where, 'not detected (HPD)', r_g=r_g,
                     hpd68={k: list(v) for k, v in rr.hpd.items()})
            continue
        lz0 = null_logz(base_g, g, f'null_{it}_{g}')
        lnb = rr.logZ - lz0
        okr = lnb > a.lnb_res
        log.node('resid_fit', where, 'passes' if okr else 'rejected',
                 r_g=r_g, lnB_vs_null=lnb, logZ=rr.logZ, logZ_null=lz0,
                 hpd68={k: list(v) for k, v in rr.hpd.items()})
        if okr:
            passed.append((lnb, g, rr))
    if not passed:
        log.node('resid', 'Res_wr*: any new moment?', 'No')
        break
    lnb, g, rr = max(passed, key=lambda x: x[0])
    cand = cur.apply(g).without(rr.off)
    log.node('resid', 'Res_wr*: any new moment?', 'Yes',
             new_candidate=cand.label, lnB_vs_null=lnb)
else:
    if n_it:
        log.node('stop', 'max iterations reached', 'stop', max_it=a.max_it)

# ── risultato: profilo dell'ultimo modello (il vecchio M_0,r se non e'
#    entrato niente), con il profilo come likelihood ───────────────────────
fine, cf, r_fin = prof(cur, 'final')
# altri parametri: posterior bayesiano del modello finale condizionato a r = r*
# (minimo del suo profilo), tutto il resto libero
bf = base_at(cur, r_fin, f'ml_{cur.label}_final')
pn, ps = R.emcee(cur.with_r(False), f'post_{cur.label}_rstar',
                 fixed={'r_tensor': r_fin}, centers=bf.best)
params = post_summary(pn, ps)
log.node('post', f'posterior of the other parameters of {cur.label} at r = '
         f'r* = {r_fin:.4f} (emcee)', 'ok',
         **{f'{n}_med': v['med'] for n, v in params.items()})
res = dict(rule='profile scheme (Res_wr*)', model_tree=M0.label,
           model_final=cur.label,
           first_candidate=M1c.label if M1c else None,
           start=('M_1 (M_1,0 preferred)' if m10 else
                  'M_0,r + candidate' if M1c else 'M_0,r, no candidate'),
           rejected=sorted(rejected),
           r_bayes_tree=med, sigma_bayes_tree=sig, **summary(fine, cf),
           r_star_final=r_fin, best_fit_at_rstar=dict(bf.best),
           chi2_at_rstar=float(bf.chi2), params_at_rstar=params,
           bayes=BAYES)
# decisione finale, come nel tree ma con il posterior dal profilo (prior del
# tree su r): 0 dentro l'intervallo centrale a 3 sigma -> upper limit (r_95
# con r >= 0), altrimenti detection (mediana e 68%)
ul = res['r_lo3_pl'] <= 0.0 <= res['r_hi3_pl']
res['conclusion'] = 'upper_limit' if ul else 'r_detection'
log.node('r3sigma', 'Is r compatible with zero at 3 sigma (profile '
         'likelihood posterior)?', 'Yes' if ul else 'No',
         r_lo3=res['r_lo3_pl'], r_hi3=res['r_hi3_pl'])
log.node('END', ('upper limit on r: r < %.4f (95%%)' % res['r_95_pl0']) if ul
         else ('r detection: r = %.4f [%.4f, %.4f] (68%%)'
               % (res['r_med_pl'], res['r_lo1_pl'], res['r_hi1_pl'])),
         res['conclusion'],
         **{k: v for k, v in res.items()
            if k not in ('params_at_rstar', 'best_fit_at_rstar', 'bayes')})
json.dump(res, open(os.path.join(out, 'result.json'), 'w'), indent=2)
