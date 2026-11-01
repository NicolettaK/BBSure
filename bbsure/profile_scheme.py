"""
Profile scheme: run on a concluded tree, in place of the frequentist branch
(designed for the detections: large r and no M_1,r preferred).

1. M_0 = final model of the tree; first candidate C = last M_1,r tried by
   the tree (only its new groups, added to M_0). If the tree found no new
   moment (nothing detected in Res_r0): no candidate, no Res_wr*, straight
   to point 5 with the profile of M_0,r.
2. Profile of r of M_0 + C -> r_k (minimum; ML points that did not converge
   are refitted from the neighbour).
3. Each model around its own ML best fit, each with its own r (M_0 + C at
   r = r_k, M_0 at the minimum of its profile), free only the spectra of its
   moments (amplitudes, slopes) and the SAME betas in the two models (those
   of the groups of M_0 + C: e.g. dust vs dust + synch -> beta_d and beta_s
   free in both); ln B > lnb_full -> C enters the model.
   If C is rejected the scheme stops (no Res_wr*) -> point 5.
4. Res_wr* (only if C entered, or when starting from M_1): each group g is
   tested at its own r, r_g = minimum of the profile of (current model + g):
   residuals of the current model refitted at r = r_g, groups not yet in the
   model and not already rejected as candidates (higher orders too). HPD 68%
   screening, then evidence against the null (same point, amplitudes of the
   group at zero, free only the betas of the group): passes if
   ln B > lnb_res. The best one becomes the new candidate -> point 2 (r**,
   ...). None -> stop.
5. Result from the profile of the last model (if nothing entered, the one of
   the old M_0,r): best fit, 68%, 95% UL and posterior of r with the profile
   as the likelihood: prior of the tree on r (_pl, for the bias) and with
   r >= 0 (_pl0).
6. Other parameters: Bayesian posterior (emcee) of the final model with r
   fixed at the minimum of its profile (conditioned on r = r*), everything
   else free -> median, 16-84%, sigma in result.json (params_at_rstar).
7. Decision: 0 inside the central 3 sigma interval of the posterior from the
   profile (prior of the tree) -> upper limit (r_95 with r >= 0), otherwise
   r detection (median and 68%), like the 3 sigma test of the tree.

If M_1,0 was preferred over M_0,r in the tree, the scheme starts from M_1
(r free), the best model found: no comparison of the candidate, Res_wr* at
the minimum of the profile of M_1, and if nothing else enters the result is
the profile of M_1.

Output in `out`: the fits, decision_log.{txt,json}, profile_<model>.npz,
result.json.
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml
from scipy.interpolate import interp1d

from .config import build_config, free_param_names, iter_param_entries
from .fitter import Fitter, NoValidStart
from .log import DecisionLog
from .models import (MOMENT_GROUPS, Model, amp_name, cand_group,
                     candidate_groups, group_spectra, model_from_label,
                     slope_name)
from .runner import BBCompSepRunner

# at the edges the profile must rise at least this much (4 sigma): needed
# for the 3 sigma intervals of the posterior
EDGE_DCHI2 = 16.


def post_summary(names, samples):
    """Median, 16-84% and sigma of each parameter."""
    q = np.percentile(samples, [15.865, 50., 84.135], axis=0)
    return {n: dict(med=float(q[1, k]), lo1=float(q[0, k]),
                    hi1=float(q[2, k]), std=float(np.std(samples[:, k])))
            for k, n in enumerate(names)}


def new_group(M1, M):
    """Moment group(s) of M1 not in M (for the label)."""
    return '+'.join(sorted(M1.moments - M.moments)) or 'restore'


def moment_spectra(M):
    """Amplitudes and slopes of the moment spectra of M."""
    return {f(p, sp) for _, p, sp in M.spectra()
            for f in (amp_name, slope_name)}


class ProfileScheme:
    """
    tc       : tree config of the tree (its runs_dir is replaced by `out`)
    tree_dir : runs_dir of the concluded tree
    out      : output directory
    n        : points of the initial profile grid
    nproc    : profile points fitted in parallel (one ML each)
    lnb_res, lnb_full : thresholds of Res_wr* and of the candidate comparison
    max_it   : maximum number of iterations
    no_stop  : diagnostic: after a rejected candidate still run Res_wr* on
               all the other groups
    ignore_m10 : test: ignore M_1,0 preferred in the tree (detection path)
    name     : name of the data set in the log (default: basename of out)
    """

    def __init__(self, tc, tree_dir, out, n=37, nproc=8, lnb_res=0.,
                 lnb_full=0., max_it=4, no_stop=False, ignore_m10=False,
                 name=None):
        os.makedirs(out, exist_ok=True)
        tc['paths']['runs_dir'] = out
        self.out = out
        self.nproc = nproc
        self.lnb_res, self.lnb_full = lnb_res, lnb_full
        self.max_it = max_it
        self.no_stop, self.ignore_m10 = no_stop, ignore_m10
        self.log = DecisionLog(out)
        self.F = Fitter(BBCompSepRunner(tc, self.log),
                        hpd_frac=tc['thresholds'].get('hpd_frac', 0.68))
        self.R = self.F.R
        mc = tc['moments']
        self.groups = mc.get('res_r0', ['d', 's', 'dT', 'd2', 's2', 'ds1',
                                        'ds2'])
        self.req = mc.get('requires', {})
        self.PROF = {}          # label -> (fine grid, interpolated chi2)
        self.PROF_FITS = {}     # label -> {r of the grid: ML fit}

        # ── final model and candidate from the tree ───────────────────────
        # conclusion_tree.json: written by run_tree.py (conclusion.json then
        # becomes the final one of the scheme)
        c = os.path.join(tree_dir, 'conclusion_tree.json')
        concl = json.load(open(c if os.path.isfile(c) else
                               os.path.join(tree_dir, 'conclusion.json')))
        self.ev = json.load(open(os.path.join(tree_dir, 'decision_log.json')))
        M0 = model_from_label(concl['model'])
        cands = [e['model'] for e in self.ev if e['node'] == 'fit' and
                 'M_1,r' in e.get('question', '') and
                 e.get('model', '').endswith('_r')]
        M1c = model_from_label(cands[-1]) if cands else None
        if M1c is None or M1c.moments <= M0.moments:
            # the tree found no new moment to try (nothing detected in
            # Res_r0): no candidate, the result is the profile of M_0,r
            M1c = None
        else:
            # the candidate may have been tried on an earlier model: only its
            # new groups are added to the final model (with the same pruning),
            # so that M_0 is nested in M_0 + C
            newg = M1c.moments - M0.moments
            amps_new = {amp_name(p, sp) for _, p, sp in group_spectra(newg)}
            M1c = Model(M0.moments | newg, True,
                        M0.off | (M1c.off & amps_new))
        self.M0, self.M1c = M0, M1c
        cfg = yaml.safe_load(open(os.path.join(tree_dir, f'{M0.label}_PC',
                                               'config.yml')))
        names = free_param_names(cfg['BBCompSep'])
        samp = np.loadtxt(os.path.join(tree_dir, f'{M0.label}_PC',
                                       'polychord', 'pch_equal_weights.txt'),
                          ndmin=2)[:, 2:]
        r_b = samp[:, names.index('r_tensor')]

        # Bayesian part (tree): ML best fit and posterior (PolyChord) of M_0,r
        z = np.load(os.path.join(tree_dir, f'{M0.label}_ML', 'chi2.npz'))
        self.bayes = dict(model=M0.label,
                          best_fit={str(n): float(v) for n, v in
                                    zip(z['names'], np.atleast_1d(z['params']))},
                          chi2_ml=float(z['chi2']),
                          posterior=post_summary(names, samp))
        self.med, self.sig = float(np.median(r_b)), float(np.std(r_b))
        self.grid = np.linspace(self.med - 6 * self.sig,
                                self.med + 3 * self.sig, n)
        self.step = self.grid[1] - self.grid[0]
        name = name or os.path.basename(os.path.normpath(out))
        self.log.info(f"==> {name}: final model {M0.label}, Bayesian r = "
                      f"{self.med:.4f} +- {self.sig:.4f}; candidate: "
                      f"{M1c.label if M1c else 'none'}")

        # prior of the tree on r (tophat) for the posterior with the profile
        # as the likelihood
        rp = next(v for k, v in
                  iter_param_entries(self.R.base['BBCompSep']).items()
                  if k == 'r_tensor')
        self.r_prior = (float(rp[2][0]), float(rp[2][-1]))

    # ── profiles ──────────────────────────────────────────────────────────
    def profile(self, M, tag):
        """Profile chi2(r) of M; returns (fine grid, interpolated chi2,
        r_min). An ML fit that did not converge can only lie above the true
        profile: the points higher by 0.5 than the mean of their neighbours
        (or with non-finite chi2) are refitted starting from the best fit of
        the lowest neighbour (up to 3 passes) and the minimum is kept. If the
        minimum is close to an edge or the chi2 at the edges does not rise by
        at least EDGE_DCHI2, the grid is extended on that side."""
        R, log = self.R, self.log
        GRID, STEP = self.grid, self.step
        m = M.with_r(False)
        grid = list(GRID)
        fits = {}

        def fit(r, centers=None, sfx=''):
            k = int(round((r - GRID[0]) / STEP))
            return R.ml(m, f'prof_{tag}_{k:+03d}{sfx}',
                        fixed={'r_tensor': float(r)}, centers=centers)

        def batch(jobs):
            """Independent ML fits in parallel: [(r, centers, sfx)] -> fits."""
            with ThreadPoolExecutor(max_workers=self.nproc) as ex:
                return list(ex.map(lambda j: fit(*j), jobs))

        for r, f in zip(grid, batch([(r, None, '') for r in grid])):
            fits[r] = f
        # points with infinite chi2 (start outside the moment constraints):
        # restart with the amplitudes at zero except the first-order
        # auto-spectra (as in base_at); if no valid point is left the model
        # cannot be fitted
        bad = [r for r in fits if not np.isfinite(fits[r].chi2)]
        if bad:
            start = {x: 0. for x in M.amplitudes()
                     if not re.match(r'A_[ds]_(w10w10|w01w01)$', x)}
            for r, f in zip(bad, batch([(r, start, '_ML0') for r in bad])):
                if np.isfinite(f.chi2):
                    fits[r] = f
        if not any(np.isfinite(f.chi2) for f in fits.values()):
            raise NoValidStart(f'no valid ML start for the profile of '
                               f'{M.label}')
        # a whole stretch of non-converged fits (plateau) does not show up as
        # a "bump" with respect to the neighbours: all the points are refitted
        # starting from the best fit of the global minimum and the best is
        # kept, as long as it improves
        for s in range(3):
            rb = min((r for r in fits if np.isfinite(fits[r].chi2)),
                     key=lambda r: fits[r].chi2)
            jobs = [(r, fits[rb].best, f'_s{s}') for r in sorted(fits)
                    if r != rb]
            n_imp = 0
            for (r, _, _), f in zip(jobs, batch(jobs)):
                if f.chi2 < fits[r].chi2 - 0.5 or not np.isfinite(fits[r].chi2):
                    log.info(f"    profile point r = {r:.4f}: refit from the "
                             f"minimum, chi2 {fits[r].chi2:.2f} -> "
                             f"{f.chi2:.2f}")
                    fits[r] = f
                    n_imp += 1
            if not n_imp:
                break
        for ext in range(10):
            g = np.array(sorted(fits))
            chi2 = np.array([fits[r].chi2 for r in g])
            for p in range(3):
                bad = [i for i in range(len(g)) if not np.isfinite(chi2[i]) or
                       (0 < i < len(g) - 1 and
                        chi2[i] - 0.5 * (chi2[i - 1] + chi2[i + 1]) > 0.5)]
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
                    log.info(f"    profile point r = {g[i]:.4f}: refit from "
                             f"the neighbour, chi2 {chi2[i]:.2f} -> "
                             f"{f.chi2:.2f}")
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
            jobs = [(r, fits[g[0] if r < g[0] else g[-1]].best, '')
                    for r in new]
            for r, f in zip(new, batch(jobs)):
                fits[r] = f
        else:
            log.info(f"    WARNING: profile of {M.label} still open at the "
                     "edges")
        np.savez(os.path.join(self.out, f'profile_{M.label}.npz'), r=g,
                 chi2=chi2)
        self.PROF_FITS[M.label] = {r: fits[r] for r in g}
        fine = np.linspace(g[0], g[-1], 20001)
        cf = interp1d(g, chi2, kind='cubic')(fine)
        return fine, cf, float(fine[np.argmin(cf)])

    def prof(self, M, tag):
        """Profile of M, computed once per model."""
        if M.label not in self.PROF:
            fine, cf, _ = self.profile(M, tag)
            self.PROF[M.label] = (fine, cf)
        fine, cf = self.PROF[M.label]
        return fine, cf, float(fine[np.argmin(cf)])

    def summary(self, fine, cf):
        """Profile: best fit, Delta chi2 = 1 and 2.71. Posterior with the
        profile as the likelihood ('pl'): prior of the tree on r, and with
        r >= 0 ('pl0')."""
        d = cf - cf.min()
        i = int(np.argmin(cf))
        out = dict(r_best=float(fine[i]),
                   r_lo1=float(fine[:i + 1][d[:i + 1] <= 1].min()),
                   r_hi1=float(fine[i:][d[i:] <= 1].max()),
                   r_ul95_profile=float(fine[i:][d[i:] <= 2.71].max()),
                   dchi2_r0=float(np.interp(0., fine, d))
                   if fine[0] <= 0. <= fine[-1] else float('nan'))
        for tag, lo in (('pl', self.r_prior[0]), ('pl0', 0.0)):
            w = np.exp(-d / 2) * (fine >= lo) * (fine <= self.r_prior[1])
            w /= w.sum()
            c = np.cumsum(w)

            def q(p):
                return float(fine[np.searchsorted(c, p)])
            mean = float(np.sum(w * fine))
            out.update({f'r_med_{tag}': q(0.5), f'r_lo1_{tag}': q(0.15865),
                        f'r_hi1_{tag}': q(0.84135), f'r_95_{tag}': q(0.95),
                        f'r_lo3_{tag}': q(0.00135),
                        f'r_hi3_{tag}': q(0.99865),
                        f'r_std_{tag}': float(np.sqrt(np.sum(
                            w * (fine - mean) ** 2)))})
        return out

    # ── fits at fixed r and local evidences ───────────────────────────────
    def base_at(self, M, r, tag, centers=None):
        """ML of M with r fixed: FitResult with r_tensor in the best fit. If
        the minimizer starts outside the region allowed by the moment
        constraints (infinite chi2), it retries with the amplitudes at zero
        except the first-order auto-spectra (as screen_fit of the tree)."""
        m = M.with_r(False)
        if centers is None and M.label in self.PROF_FITS:
            # starting point: the profile fit closest to r (the minimizer from
            # the default starting point may stop far from the minimum)
            pf = self.PROF_FITS[M.label]
            centers = pf[min(pf, key=lambda x: abs(x - r))].best
        res = self.R.ml(m, tag, fixed={'r_tensor': r}, centers=centers)
        if not np.isfinite(res.chi2):
            start = dict(centers or {})
            start.update({x: 0. for x in M.amplitudes()
                          if not re.match(r'A_[ds]_(w10w10|w01w01)$', x)})
            res = self.R.ml(m, tag + '_ML0', fixed={'r_tensor': r},
                            centers=start)
        if not np.isfinite(res.chi2):
            raise NoValidStart(f'no valid ML start for {M.label} at r = {r}')
        res.best = dict(res.best, r_tensor=r)
        return res

    def local_logz(self, M, base, free, tag):
        """Evidence of M around the best fit `base` (r fixed): free only the
        parameters in `free`; if none is free, -chi2/2."""
        fixed = {k: v for k, v in base.best.items() if k not in free}
        cs = build_config(self.R.base, M.with_r(False), 'polychord',
                          fixed)['BBCompSep']
        if not free_param_names(cs):
            return -0.5 * base.chi2
        return self.R.polychord(M.with_r(False), tag, fixed=fixed,
                                centers=base.best)[0]

    def null_logz(self, base, group, tag):
        """Evidence of the null: base, amplitudes of the group at zero, free
        only the spectral parameters of the group present in the model."""
        free = set(MOMENT_GROUPS[cand_group(group)]['free'])
        fixed = {k: v for k, v in base.best.items() if k not in free}
        cs = build_config(self.R.base, base.model, 'polychord',
                          fixed)['BBCompSep']
        if not free_param_names(cs):
            return -0.5 * base.chi2
        logZ, _, _, _, _ = self.R.polychord(base.model, tag, fixed=fixed,
                                            centers=base.best)
        return logZ

    # ── the scheme ────────────────────────────────────────────────────────
    def run(self):
        """Runs the scheme, writes result.json in `out` and returns it."""
        log, F, R = self.log, self.F, self.R
        M0, M1c, med, sig = self.M0, self.M1c, self.med, self.sig
        prof, summary, base_at = self.prof, self.summary, self.base_at
        # if M_1,0 was preferred over M_0,r in the tree, the best model found
        # is M_1 (same model with r free): start from there, without
        # candidate, and Res_wr* is done at the minimum of its profile; if
        # nothing else enters, the result is its profile
        m10 = [e for e in self.ev if e['node'] == 'pref' and 'M_1,0' in
               e.get('question', '') and e.get('answer') == 'Yes']
        if self.ignore_m10:
            m10 = []
        cur, cand = M0, M1c
        rejected = set()          # groups already rejected as candidates
        n_it = self.max_it
        if not m10 and M1c is None:
            log.node('start', 'no new moment found by the tree (nothing '
                     'detected in Res_r0): no candidate, result from the '
                     'profile of M_0,r', M0.label)
            n_it = 0
        if m10:
            cur = model_from_label(m10[-1]['question'].split('(')[1]
                                   .split(')')[0]).with_r(True)
            cand = None
            log.node('adopt', f'M_1,0 was preferred in the tree (ln B = '
                     f'{m10[-1].get("lnB", float("nan")):+.2f}): start from '
                     'M_1', cur.label)
        for it in range(1, n_it + 1):
            if cand is None:
                fine, cf, r_k = prof(cur, f'it{it}')
                log.info(f"\n==> iteration {it}: model {cur.label}, no "
                         f"candidate, r_{it} = {r_k:.4f} (minimum of its "
                         "profile)")
                log.node('profile', f'profile of {cur.label} -> r_{it}',
                         f'{r_k:.4f}', r_bayes_tree=med, sigma_bayes_tree=sig,
                         **summary(fine, cf))
                try:
                    base = base_at(cur, r_k, f'ml_{cur.label}_r{it}')
                except NoValidStart as err:
                    log.node('resid', 'Res_wr*: no valid ML start at r_k',
                             'stop', reason=str(err))
                    break
                return_to_res = True
            else:
                return_to_res = False
            if not return_to_res:
                g_c = new_group(cand, cur)
                # 2. profile of the current model + candidate -> r_k
                try:
                    fine, cf, r_k = prof(cand, f'it{it}')
                except NoValidStart as err:
                    # candidate that cannot be fitted (moment constraints):
                    # rejected
                    rejected.add(g_c)
                    log.node('profile', f'profile of {cand.label}',
                             'unfittable', reason=str(err))
                    log.node('stop', f'candidate {g_c} cannot be fitted: '
                             'stop, no search for further moments', cur.label)
                    break
                log.info(f"\n==> iteration {it}: model {cur.label}, "
                         f"candidate {g_c} ({cand.label}), r_{it} = "
                         f"{r_k:.4f}")
                log.node('profile', f'profile of {cand.label} -> r_{it}',
                         f'{r_k:.4f}', r_bayes_tree=med, sigma_bayes_tree=sig,
                         **summary(fine, cf))
                # 3. comparison: each model around its own ML best fit, each
                #    with its own r; free only its moments (amplitudes,
                #    slopes) and their betas. The current model at its best:
                #    r at the minimum of its profile
                _, _, r_cur = prof(cur, f'cur{it}')

                # the SAME betas free in the two models (those of the groups
                # of the candidate, which contains the current model): the
                # difference is only in the moment spectra, the Occam factor
                # of the betas cancels
                betas = set().union(*(MOMENT_GROUPS[g]['free']
                                      for g in cand.moments)) \
                    if cand.moments else set()
                free1 = moment_spectra(cand) | betas
                free0 = moment_spectra(cur) | betas
                b0 = b1 = None
                try:
                    b0 = base_at(cur, r_cur, f'ml_{cur.label}_rcur{it}')
                    b1 = base_at(cand, r_k, f'ml_{cand.label}_r{it}')
                    lz1 = self.local_logz(cand, b1, free1,
                                          f'loc_{cand.label}_r{it}')
                    lz0 = self.local_logz(cur, b0, free0,
                                          f'loc_{cur.label}_r{it}')
                    lnb_full = lz1 - lz0
                except NoValidStart as err:
                    log.node('pref', f'{cand.label} at r_{it}', 'unfittable',
                             reason=str(err))
                    lnb_full, lz1, lz0 = -np.inf, None, None
                ok = lnb_full > self.lnb_full
                log.node('pref', f'Each model at its own ML point '
                         f'({cand.label}: r = {r_k:.4f}; {cur.label}: r = '
                         f'{r_cur:.4f}), is {cand.label} preferred?',
                         'Yes' if ok else 'No', lnB=lnb_full,
                         r_cand=r_k, r_cur=r_cur, logZ_cand=lz1,
                         logZ_cur=lz0, free_cand=sorted(free1),
                         free_cur=sorted(free0),
                         chi2_ml_cur=b0.chi2 if b0 else None,
                         chi2_ml_cand=b1.chi2 if b1 else None)
                if ok:
                    cur, base = cand, b1
                    log.node('adopt', f'{g_c} enters the model', cur.label)
                else:
                    # candidate rejected: stop (no Res_wr*), the result is
                    # the profile of the current model
                    rejected.add(g_c)
                    if not self.no_stop:
                        log.node('stop', f'candidate {g_c} rejected: stop, '
                                 'no search for further moments', cur.label)
                        break
                    # diagnostic: Res_wr* on the current model refitted at
                    # r = r_k
                    try:
                        base = base_at(cur, r_k, f'ml_{cur.label}_r{it}')
                    except NoValidStart as err:
                        log.node('resid', 'Res_wr*: no valid ML start at r_k',
                                 'stop', reason=str(err))
                        break
            # 4. Res_wr*: each group g is tested at ITS OWN r: r_g = minimum
            #    of the profile of (current model + g); residuals of the
            #    current model refitted at r = r_g; HPD screening, then
            #    evidence against the null at the same point
            groups = [g for g in candidate_groups(cur.with_r(False),
                                                  self.groups,
                                                  extra_requires=self.req)
                      if g not in rejected]
            passed = []
            for g in groups:
                Mg = cur.apply(g)
                try:
                    fg_, cg_, r_g = prof(Mg, f'g{it}_{g}')
                    base_g = base_at(cur, r_g, f'ml_{cur.label}_r{it}_{g}')
                    rr = F.residual_test(base_g, g)
                except (NoValidStart, ValueError) as err:
                    log.node('resid_fit', f'Res_wr*: {g}', 'unfittable',
                             reason=str(err))
                    continue
                where = f'Res_wr* (r = r_{g} = {r_g:.4f}, minimum of the ' \
                        f'profile of {Mg.label}): moment {g} on the ' \
                        f'residuals of {cur.label}'
                if not rr.detected:
                    log.node('resid_fit', where, 'not detected (HPD)',
                             r_g=r_g,
                             hpd68={k: list(v) for k, v in rr.hpd.items()})
                    continue
                lz0 = self.null_logz(base_g, g, f'null_{it}_{g}')
                lnb = rr.logZ - lz0
                okr = lnb > self.lnb_res
                log.node('resid_fit', where, 'passes' if okr else 'rejected',
                         r_g=r_g, lnB_vs_null=lnb, logZ=rr.logZ,
                         logZ_null=lz0,
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
                log.node('stop', 'max iterations reached', 'stop',
                         max_it=self.max_it)

        # ── result: profile of the last model (the old M_0,r if nothing
        #    entered), with the profile as the likelihood ────────────────────
        fine, cf, r_fin = prof(cur, 'final')
        # other parameters: Bayesian posterior of the final model conditioned
        # on r = r* (minimum of its profile), everything else free
        bf = base_at(cur, r_fin, f'ml_{cur.label}_final')
        pn, ps = R.emcee(cur.with_r(False), f'post_{cur.label}_rstar',
                         fixed={'r_tensor': r_fin}, centers=bf.best)
        params = post_summary(pn, ps)
        log.node('post', f'posterior of the other parameters of {cur.label} '
                 f'at r = r* = {r_fin:.4f} (emcee)', 'ok',
                 **{f'{n}_med': v['med'] for n, v in params.items()})
        res = dict(rule='profile scheme (Res_wr*)', model_tree=M0.label,
                   model_final=cur.label,
                   first_candidate=M1c.label if M1c else None,
                   start=('M_1 (M_1,0 preferred)' if m10 else
                          'M_0,r + candidate' if M1c else
                          'M_0,r, no candidate'),
                   rejected=sorted(rejected),
                   r_bayes_tree=med, sigma_bayes_tree=sig,
                   **summary(fine, cf),
                   r_star_final=r_fin, best_fit_at_rstar=dict(bf.best),
                   chi2_at_rstar=float(bf.chi2), params_at_rstar=params,
                   bayes=self.bayes)
        # final decision, as in the tree but with the posterior from the
        # profile (prior of the tree on r): 0 inside the central 3 sigma
        # interval -> upper limit (r_95 with r >= 0), otherwise detection
        # (median and 68%)
        ul = res['r_lo3_pl'] <= 0.0 <= res['r_hi3_pl']
        res['conclusion'] = 'upper_limit' if ul else 'r_detection'
        log.node('r3sigma', 'Is r compatible with zero at 3 sigma (profile '
                 'likelihood posterior)?', 'Yes' if ul else 'No',
                 r_lo3=res['r_lo3_pl'], r_hi3=res['r_hi3_pl'])
        log.node('END', ('upper limit on r: r < %.4f (95%%)'
                         % res['r_95_pl0']) if ul
                 else ('r detection: r = %.4f [%.4f, %.4f] (68%%)'
                       % (res['r_med_pl'], res['r_lo1_pl'], res['r_hi1_pl'])),
                 res['conclusion'],
                 **{k: v for k, v in res.items()
                    if k not in ('params_at_rstar', 'best_fit_at_rstar',
                                 'bayes')})
        json.dump(res, open(os.path.join(self.out, 'result.json'), 'w'),
                  indent=2)
        return res
