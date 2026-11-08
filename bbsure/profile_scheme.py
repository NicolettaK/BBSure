"""
Profile scheme: run on a concluded tree. The moments are searched with r = 0
(Res_r0 of the tree, where they have the most room); here the candidates are
judged with every model at its own r, the minimum of its profile likelihood,
instead of with r marginalised.

1. M* = final model of the tree (M_0,r). Candidates: M* +
   every combination of the families added to M_0,r in the last Res_r0 of
   the tree (M_1 included), respecting the requirements of the families
   (e.g. ds1 only with s); none if the tree concluded before Res_r0 or
   nothing was detected there.
   With profile_scheme: candidates: false in the tree config the candidates
   are not compared: only the profile of M* (steps 4-6).
2. Profiles in r: M* -> r* = its minimum; M* + g -> r*_g for every candidate.
   A profile: r fixed on a grid, all the other parameters re-fitted (ML).
3. Comparison of all the models, each at its own r (M* at r*, M* + g at
   r*_g): local evidence of each, around its own ML best fit at that r, with
   its moment spectra and its spectral parameters free (beta_d and beta_s in
   all the models, T_d only in those with the dT moments). The model with
   the highest evidence is the final one (M* if no candidate beats it). If
   the two best are within seed_band errors, their local evidences are
   repeated with other seeds (mean), up to seed_max_runs runs each.
4. Result from the profile of the final model: best fit, 68%, 95% UL and
   posterior of r with the profile as the likelihood: prior of the tree on r
   (_pl, for the bias) and with r >= 0 (_pl0).
5. Other parameters: Bayesian posterior (emcee) of the final model with r
   fixed at the minimum of its profile (conditioned on r = r_min), everything
   else free -> median, 16-84%, sigma in result.json (params_at_rstar).
6. Decision: 0 inside the central 3 sigma interval of the posterior from the
   profile (prior of the tree) -> upper limit (r_95 with r >= 0), otherwise
   r detection (median and 68%).

Output in `out`: the fits, decision_log.{txt,json}, profile_<model>.npz,
result.json.
"""
import itertools
import json
import os
import re
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml
from scipy.interpolate import interp1d

from .config import build_config, free_param_names, iter_param_entries
from .fitter import Fitter, NoValidStart
from .log import DecisionLog
from .models import (MOMENT_GROUPS, amp_name,
                     model_from_label, requirements_met, slope_name)
from .runner import BBCompSepRunner, equal_weights_file

# at the edges the profile must rise at least this much (4 sigma): needed
# for the 3 sigma intervals of the posterior
EDGE_DCHI2 = 16.


def post_summary(names, samples):
    """Median, 16-84% and sigma of each parameter."""
    q = np.percentile(samples, [15.865, 50., 84.135], axis=0)
    return {n: dict(med=float(q[1, k]), lo1=float(q[0, k]),
                    hi1=float(q[2, k]), std=float(np.std(samples[:, k])))
            for k, n in enumerate(names)}


def moment_spectra(M):
    """Amplitudes and slopes of the moment spectra of M."""
    return {f(p, sp) for _, p, sp in M.spectra()
            for f in (amp_name, slope_name)}


# spectral parameters free in every local evidence: beta_d and beta_s (dust
# and synchrotron are in all the models), plus those of the model's own
# moment families (T_d only with dT)
BETAS = {'beta_d', 'beta_s'}


def _ml_job(job):
    """One ML fit of a profile point (in a worker process): job = (runner,
    model, tag, fixed, centers)."""
    R, m, tag, fixed, centers = job
    return R.ml(m, tag, fixed=fixed, centers=centers)


def last_res_r0(ev):
    """Index of the 'resid' node of the last Res_r0 of the tree (None if
    there is none)."""
    last = None
    for i, e in enumerate(ev):
        if e['node'] == 'resid' and e['question'].startswith('Res_r0'):
            last = i
    return last


def add_combinations(M0, fams):
    """
    M0 + every combination of the families fams = [(group, off)] (each with
    the spectra of `off` removed): [(label, model)], label e.g. 'd2+s'.
    """
    fams = [(g, off) for g, off in fams if g not in M0.moments]
    out = []
    for k in range(1, len(fams) + 1):
        for combo in itertools.combinations(fams, k):
            M = M0
            for g, off in combo:
                M = M.apply(g).without(off)
            out.append(('+'.join(g for g, _ in combo), M))
    return out


def chain_candidates(ev):
    """
    Families added to M_0,r in the last Res_r0 of the tree (node
    'r0chain'), with their removed spectra: [(group, off)]. Empty if there
    is none or if the tree adopted a new model after it.
    """
    idx = [i for i, e in enumerate(ev) if e['node'] == 'r0chain']
    if not idx or any(e['node'] == 'replace' for e in ev[idx[-1]:]):
        return []
    e = ev[idx[-1]]
    return list(zip(e['families'], e['pruned']))


def res_r0_candidates(ev):
    """
    Families detected in the last Res_r0 of the tree, with the spectra the
    residual screening removed: [(group, off)]. Empty if there is no Res_r0
    or if the tree adopted a new model after it. (Trees of the previous
    versions, e.g. to choose the sets to rerun: scripts/prepare_rerun.py.)
    """
    last = last_res_r0(ev)
    if last is None or any(e['node'] == 'replace' for e in ev[last:]):
        return []
    e = ev[last]
    pruned = e.get('pruned') or [[] for _ in e.get('detected', [])]
    return list(zip(e.get('detected', []), pruned))


class ProfileScheme:
    """
    tc       : tree config of the tree (its runs_dir is replaced by `out`)
    tree_dir : runs_dir of the concluded tree
    out      : output directory
    n        : points of the initial profile grid
    nproc    : profile points fitted in parallel (one ML each)
    name     : name of the data set in the log (default: basename of out)
    """

    def __init__(self, tc, tree_dir, out, n=37, nproc=8, name=None):
        os.makedirs(out, exist_ok=True)
        tc['paths']['runs_dir'] = out
        self.out = out
        self.nproc = nproc
        self.log = DecisionLog(out)
        self.F = Fitter(BBCompSepRunner(tc, self.log),
                        hpd_frac=tc['thresholds'].get('hpd_frac', 0.68))
        self.R = self.F.R
        th = tc['thresholds']
        # close comparisons: |ln B| < seed_band * error between the two best
        # models -> more PolyChord runs (other seeds), up to seed_max_runs
        self.seed_band = th.get('seed_band', 2.0)
        # the two best within lnB_flag (inconclusive evidence): flagged
        self.lnB_flag = th.get('lnB_flag', 1.0)
        self.seed_max_runs = th.get('seed_max_runs', 1)
        # final model with PTE < pte_flag (at r = r_min): flagged
        self.pte_flag = th.get('pte_flag', 0.05)
        self.PROF = {}          # label -> (fine grid, interpolated chi2)
        self.PROF_FITS = {}     # label -> {r of the grid: ML fit}

        # ── final model and candidates from the tree ──────────────────────
        # conclusion_tree.json: written by run_tree.py (conclusion.json then
        # becomes the final one of the scheme)
        c = os.path.join(tree_dir, 'conclusion_tree.json')
        concl = json.load(open(c if os.path.isfile(c) else
                               os.path.join(tree_dir, 'conclusion.json')))
        ev = json.load(open(os.path.join(tree_dir, 'decision_log.json')))
        M0 = model_from_label(concl['model'])
        self.M0 = M0
        # candidates: M_0,r + every combination of the families added in
        # Res_r0 (node 'r0chain') that respects the requirements (e.g. ds1
        # only with s), both when M_1,0 was preferred and when not
        self.m10 = any(e['node'] == 'm10' for e in ev)
        req = tc.get('moments', {}).get('requires')
        self.cands = [(lab, M) for lab, M in
                      add_combinations(M0, chain_candidates(ev))
                      if requirements_met(M, req)]
        # profile_scheme: candidates: false -> only the profile of the final
        # model of the tree (no comparison with the candidates)
        self.use_cands = (tc.get('profile_scheme') or {}).get('candidates',
                                                              True)
        self.skipped = [] if self.use_cands else [g for g, _ in self.cands]
        if not self.use_cands:
            self.cands = []
        cfg = yaml.safe_load(open(os.path.join(tree_dir, f'{M0.label}_PC',
                                               'config.yml')))
        names = free_param_names(cfg['BBCompSep'])
        samp = np.loadtxt(equal_weights_file(
            os.path.join(tree_dir, f'{M0.label}_PC')), ndmin=2)[:, 2:]
        r_b = samp[:, names.index('r_tensor')]

        # Bayesian part (tree): ML best fit and posterior (PolyChord or
        # nautilus) of M_0,r
        z = np.load(os.path.join(tree_dir, f'{M0.label}_ML', 'chi2.npz'))
        self.bayes = dict(model=M0.label,
                          conclusion=concl.get('conclusion'),
                          best_fit={str(n): float(v) for n, v in
                                    zip(z['names'], np.atleast_1d(z['params']))},
                          chi2_ml=float(z['chi2']),
                          posterior=post_summary(names, samp))
        self.med, self.sig = float(np.median(r_b)), float(np.std(r_b))
        self.grid = np.linspace(self.med - 6 * self.sig,
                                self.med + 3 * self.sig, n)
        self.step = self.grid[1] - self.grid[0]
        name = name or os.path.basename(os.path.normpath(out))
        self.log.info(f"==> {name}: final model of the tree {M0.label}, "
                      f"Bayesian r = {self.med:.4f} +- {self.sig:.4f}; "
                      f"candidates: {[g for g, _ in self.cands] or 'none'}")

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

        def args(r, centers=None, sfx=''):
            k = int(round((r - GRID[0]) / STEP))
            return (R, m, f'prof_{tag}_{k:+03d}{sfx}',
                    {'r_tensor': float(r)}, centers)

        def batch(jobs):
            """Independent ML fits in parallel: [(r, centers, sfx)] -> fits.
            In separate processes (the in-process BBCompSep fits are
            serialized by a lock within one process); fits already done are
            read from their run directory."""
            todo = [args(*j) for j in jobs]
            # macOS: fork after the Objective-C runtime is initialised
            # crashes, unless OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
            mac_unsafe = (sys.platform == 'darwin' and os.environ.get(
                'OBJC_DISABLE_INITIALIZE_FORK_SAFETY') != 'YES')
            if self.nproc <= 1 or len(todo) <= 1 or mac_unsafe:
                return [_ml_job(t) for t in todo]
            ctx = multiprocessing.get_context('fork')
            with ProcessPoolExecutor(max_workers=min(self.nproc, len(todo)),
                                     mp_context=ctx) as ex:
                return list(ex.map(_ml_job, todo))

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
        parameters in `free` (T_d too if it is in `free`, also without the
        dT moments); if none is free, -chi2/2 (error 0).
        Returns (ln Z, error)."""
        fixed = {k: v for k, v in base.best.items() if k not in free}
        free_temp = 'temp_d' in free
        cs = build_config(self.R.base, M.with_r(False), 'polychord',
                          fixed, free_temp=free_temp)['BBCompSep']
        if not free_param_names(cs):
            return -0.5 * base.chi2, 0.
        lz, lzerr = self.R.polychord(M.with_r(False), tag, fixed=fixed,
                                     centers=base.best,
                                     free_temp=free_temp)[:2]
        return lz, lzerr

    def settle_close(self, comparison, runs):
        """
        If the two best models are within seed_band errors of each other,
        their local evidences are repeated with other seeds (first the one
        with fewer runs; ln Z = mean of the runs, error = sqrt(sum err^2) /
        N) until they separate or seed_max_runs is reached. Only reduces the
        numerical error of PolyChord.
        """
        while len(comparison) > 1:
            top = sorted(comparison, key=lambda k: -comparison[k]['logZ'])[:2]
            a, b = comparison[top[0]], comparison[top[1]]
            err = float(np.hypot(a['logZerr'], b['logZerr']))
            if err == 0 or a['logZ'] - b['logZ'] >= self.seed_band * err:
                return
            n = {k: len(runs[k][3]) for k in top}
            k = min(top, key=lambda k: n[k])
            if n[k] >= self.seed_max_runs:
                return
            M, base, free, lzs, errs = runs[k]
            self.log.info(f"    ln B = {a['logZ'] - b['logZ']:.2f} +- "
                          f"{err:.2f} between {top[0]} and {top[1]}: "
                          f"PolyChord rerun {n[k] + 1}/{self.seed_max_runs}"
                          f" of {k}")
            lz, lzerr = self.local_logz(M, base, free,
                                        f'loc_{M.label}_rep{n[k] + 1}')
            lzs.append(lz)
            errs.append(lzerr)
            comparison[k].update(
                logZ=float(np.mean(lzs)),
                logZerr=float(np.sqrt(np.sum(np.square(errs))) / len(lzs)),
                logZ_runs=list(lzs))

    def close_pair(self, comparison):
        """The two best models within lnB_flag (inconclusive evidence) or
        still within seed_band errors (after the repetitions):
        [dict(chosen, other, lnB, lnB_err)] or []."""
        if len(comparison) < 2:
            return []
        top = sorted(comparison, key=lambda k: -comparison[k]['logZ'])[:2]
        a, b = comparison[top[0]], comparison[top[1]]
        lb = a['logZ'] - b['logZ']
        err = float(np.hypot(a['logZerr'], b['logZerr']))
        if lb >= max(self.lnB_flag, self.seed_band * err):
            return []
        self.log.info(f"    [undecided] {top[0]} chosen over {top[1]} with "
                      f"ln B = {lb:.2f} +- {err:.2f}")
        return [dict(kind='best two models', chosen=top[0], other=top[1],
                     lnB=lb, lnB_err=err, threshold=0.)]

    # ── the scheme ────────────────────────────────────────────────────────
    def run(self):
        """Runs the scheme, writes result.json in `out` and returns it."""
        log, R = self.log, self.R
        M0, med, sig = self.M0, self.med, self.sig

        # 2-3. every model at its own r, local evidences, the best one wins
        final, comparison, runs, undecided = M0, {}, {}, []
        if self.cands:
            models = [M0] + [M for _, M in self.cands]
            for M in models:
                try:
                    fine, cf, r_m = self.prof(M, M.label)
                    base = self.base_at(M, r_m, f'ml_{M.label}_rmin')
                    free = moment_spectra(M) | BETAS | set().union(
                        *(MOMENT_GROUPS[g]['free'] for g in M.moments))
                    lz, lzerr = self.local_logz(M, base, free,
                                                f'loc_{M.label}')
                except NoValidStart as err:
                    log.node('compare', f'{M.label}', 'unfittable',
                             reason=str(err))
                    continue
                runs[M.label] = (M, base, free, [lz], [lzerr])
                comparison[M.label] = dict(r_min=r_m, logZ=lz,
                                           logZerr=lzerr, chi2_ml=base.chi2)
                log.node('compare', f'{M.label} at its own r = {r_m:.4f}',
                         f'ln Z = {lz:.2f}', r_min=r_m, logZ=lz,
                         logZerr=lzerr, chi2_ml=base.chi2, free=sorted(free),
                         **self.summary(fine, cf))
            self.settle_close(comparison, runs)
            undecided = self.close_pair(comparison)
            best = max(comparison, key=lambda k: comparison[k]['logZ'])
            final = next(M for M in models if M.label == best)
            log.node('final', 'model with the highest evidence (each at its '
                     'own r)', final.label,
                     lnB_vs_tree={k: v['logZ'] - comparison[M0.label]['logZ']
                                  for k, v in comparison.items()}
                     if M0.label in comparison else None)
        elif self.skipped:
            log.node('final', 'candidates not compared (profile_scheme: '
                     'candidates: false): the final model is the one of the '
                     'tree', M0.label, skipped=self.skipped)
        else:
            log.node('final', 'no candidates from Res_r0: the final model is '
                     'the one of the tree', M0.label)

        # 4. result: profile of the final model, as the likelihood of r
        fine, cf, r_fin = self.prof(final, final.label)
        # 5. other parameters: Bayesian posterior of the final model
        # conditioned on r = r_min (minimum of its profile)
        bf = self.base_at(final, r_fin, f'ml_{final.label}_final')
        pn, ps = R.emcee(final.with_r(False), f'post_{final.label}_rstar',
                         fixed={'r_tensor': r_fin}, centers=bf.best)
        params = post_summary(pn, ps)
        log.node('post', f'posterior of the other parameters of {final.label}'
                 f' at r = r_min = {r_fin:.4f} (emcee)', 'ok',
                 **{f'{n}_med': v['med'] for n, v in params.items()})
        res = dict(rule='profile scheme (every model at its own r)',
                   model_tree=M0.label, model_final=final.label,
                   candidates=[g for g, _ in self.cands],
                   comparison=comparison,
                   start=('M_1,0 preferred in the tree' if self.m10
                          else 'M_1,0 not preferred in the tree')
                   if self.cands else ('candidates not compared'
                                       if self.skipped else 'no candidates'),
                   candidates_skipped=self.skipped,
                   rejected=sorted(g for g, M in self.cands
                                   if M.label != final.label),
                   r_bayes_tree=med, sigma_bayes_tree=sig,
                   **self.summary(fine, cf),
                   r_star_final=r_fin, best_fit_at_rstar=dict(bf.best),
                   chi2_at_rstar=float(bf.chi2), params_at_rstar=params,
                   undecided=bool(undecided), undecided_choices=undecided,
                   pte_at_rstar=bf.pte,
                   low_pte=bool(bf.pte < self.pte_flag),
                   bayes=self.bayes)
        # 6. decision, as in the tree but with the posterior from the profile
        # (prior of the tree on r): 0 inside the central 3 sigma interval ->
        # upper limit (r_95 with r >= 0), otherwise detection (median, 68%)
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
                                 'bayes', 'comparison')})
        json.dump(res, open(os.path.join(self.out, 'result.json'), 'w'),
                  indent=2)
        return res
