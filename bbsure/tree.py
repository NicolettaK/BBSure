"""
SO cross-cl pipeline: decision tree (no blinding version), Bayesian part.

Implements the flowchart on top of BBPower's BBCompSep, with the moments in
the formalism of Vacher et al. 2022: moment maps w_nm = A dbeta^n dp^m
(p = 1/T) and free cross spectra A (ell/80)^alpha. The data vector
(cells_coadded, cells_noise) is produced before, once, with run_spectra.sh:
here only the model changes.

Notation
--------
    M_{0,r}   current model, r free              (at the start: no moments)
    M_{0,0}   current model, r fixed to 0
    M_{1,r}   model with one more moment group, r free
    M_{1,0}   the same with r fixed to 0
    M*_{.,r}  profile likelihood in r of the model

Moment groups: see models.py.

Flowchart
---------
 1. Fit of M_0,r. Good chi2 (PTE > pte_min) and r compatible with 0 at 1
    sigma -> upper limit.
 2. Res_wr: residual test with everything fixed at the best fit of M_0,r, r
    included; candidates are the res_wr groups (d, s) not yet in the model,
    each with its spectral parameter free. Detected = zero outside the 68%
    HPD interval of at least one amplitude (screening with emcee, then
    PolyChord on the detected ones: confirmation and evidence).
      - none detected -> 3.
      - in order of evidence: full fit of M_0,r + group and comparison with
        M_0,r (ln B > lnB_preferred). The first one that passes becomes
        M_0,r -> 1.
      - none passes -> 3 (with r = 0, families not detected or not
        preferred with r free can be).
 Pruning before every comparison with M_0,r: (a) screening (ML + emcee):
 the new spectra not detected (zero inside the 68% HPD of their amplitude)
 are removed and the screening is redone; (b) full fit (ML + PolyChord) of
 the screened model and pruning of ALL its moment spectra, old and new
 (point 4); only this pruned model is compared with M_0,r, and the
 candidate is rejected if nothing new is left. The removed spectra do not
 come back as candidates in Res_r0 (restore-<group> only with moments:
 res_r0_restore: true).
 3. Res_r0: like Res_wr but on the residuals of M_0,0 (r = 0), candidates
    the res_r0 groups (d, s, dT, d2, s2, ds1, ds2). None detected -> 3 sigma
    test. The best one (evidence) -> M_1,r: preferred over M_0,r -> replaces
    M_0,r, back to 1. Otherwise M_1,0: preferred over M_0,r -> frequentist
    branch (profile likelihood), otherwise 3 sigma test. With
    run: profile_scheme: true there is no frequentist branch: the tree goes
    to the 3 sigma test and then the profile scheme starts
    (profile_scheme.py) from M_1.
    Res_wr is repeated as long as the replacements come from Res_wr (e.g.
    after M + d, s is tested); after a replacement from Res_r0 the tree
    always goes directly to Res_r0.
 4. Pruning (of every candidate before its comparison with M_0,r, and of
    the final model): the spectra with zero inside the HPD of their
    amplitude are removed and the model refitted;
    the pruned model is kept if ln B(pruned vs full) > lnB_prune (-2.5).
    Repeated until only detected amplitudes are left.
 If after the pruning the new M_0,r is a model already used as M_0,r
 (cycle, e.g. restore-s adopted and then pruned again) -> 3 sigma test.
 5. 3 sigma test on r of M_0,r (or upper limit from point 1) on the pruned
    model: upper limit or r detection.
 Borderline comparisons: if |ln B - threshold| < seed_band * error, the two
 PolyChord fits are repeated with another (random) seed and ln Z averaged,
 up to seed_max_runs runs per fit. This only reduces the numerical error,
 not the fluctuation of the realization.

Output (in runs_dir)
--------------------
  decision_log.json   every node of the tree with question, answer, numbers
  decision_log.txt    the same in readable form
  <tag>/              one directory per BBCompSep run (config + output)
"""
import os

import numpy as np
from scipy import stats

from .fitter import NoValidStart
from .log import DecisionLog
from .models import Model, candidate_groups, model_from_label
from .stats import FitResult, detection, lnB, profile_interval


class DecisionTree:

    def __init__(self, tc, fitter, log: DecisionLog):
        self.tc = tc
        self.F = fitter
        self.log = log
        th = tc['thresholds']
        self.pte_min = th['pte_min']
        self.lnB_pref = th['lnB_preferred']
        # pruning: the model without the undetected spectra is kept if
        # ln B(pruned vs full) > lnB_prune
        self.lnB_prune = th.get('lnB_prune', -2.5)
        self.hpd_frac = th.get('hpd_frac', 0.68)
        # borderline comparisons: |ln B - threshold| < seed_band * error ->
        # more PolyChord runs (different seeds), up to seed_max_runs per fit
        self.seed_band = th.get('seed_band', 2.0)
        self.seed_max_runs = th.get('seed_max_runs', 1)
        self.alpha = th['lrt_alpha']
        mc = tc['moments']
        self.res_wr = mc.get('res_wr', ['d', 's'])
        self.res_r0 = mc.get('res_r0', ['d', 's', 'dT', 'd2', 's2', 'ds1',
                                        'ds2'])
        self.extra_requires = mc.get('requires', {})
        # restore-<group> in Res_r0: off by default. The pruned spectra have
        # already been judged with r free; with r = 0 they look detected but
        # refitted with r free they are undetected again (d10s5 seed 6002)
        self.restore = mc.get('res_r0_restore', False)
        pr = tc['profile']
        self.r_grid = np.linspace(pr['r_min'], pr['r_max'], pr['n_r'])
        self.max_it = tc['run']['max_iterations']
        self.freq_replaces = tc['run'].get('frequentist_replaces_baseline',
                                           True)
        # profile scheme (profile_scheme.py) instead of the frequentist branch:
        # the tree ends with the 3 sigma test, then the scheme is launched
        self.profile_scheme = tc['run'].get('profile_scheme', False)

    # ── entry point ───────────────────────────────────────────────────────
    def run(self):
        M0 = Model(frozenset(), r_free=True)        # "run BBPower nominal"
        # "what if" test: start from a given model (label, e.g.
        # M_d+s-d_w00w10-s_w00w10_r) instead of the one without moments
        start = self.tc['run'].get('initial_model')
        if start:
            M0 = model_from_label(start)
        use_wr = True       # False after a replacement coming from Res_r0
        visited = set()     # models already used as M_0,r
        self.log.info(f"==> Initial model M_0,r = {M0.label}")
        for it in range(1, self.max_it + 1):
            self.log.info(f"\n==> Iteration {it}: M_0,r = {M0.label}")
            visited.add(M0)
            fit0 = self.F.full_fit(M0)
            self.log.node('fit', f'fit of {M0.label}', 'ok',
                          **fit0.summary())

            chi2_ok = fit0.pte > self.pte_min
            self.log.node('chi2', 'Is chi2 good?', 'Yes' if chi2_ok else 'No',
                          chi2=fit0.chi2, dof=fit0.dof, pte=fit0.pte,
                          pte_min=self.pte_min)
            if chi2_ok:
                ok1, rs = fit0.r_compatible_with_zero(1)
                self.log.node('r1sigma', 'Is r compatible with zero at 1 '
                              'sigma?', 'Yes' if ok1 else 'No',
                              r_med=rs['r_med'], r_lo1=rs['r_lo1'],
                              r_hi1=rs['r_hi1'])
                if ok1:
                    M0, fit0 = self.prune(M0, fit0)
                    return self.conclude_upper_limit(M0, fit0)

            outcome, M1, branch = self.search_moments(M0, fit0, use_wr)
            if outcome == 'replace':
                self.log.node('replace', f'M_0,r = M_1,r (from {branch})',
                              M1.label)
                # the new M_0,r was already pruned before its comparison
                M0, fit0 = M1, self.F.full_fit(M1)
                use_wr = use_wr and branch == 'Res_wr'
                # cycle: after the pruning the new M_0,r is a model already
                # used (e.g. restore-s adopted and then pruned again) -> the
                # search leads nowhere else, 3 sigma test
                cycle = M0 in visited
                self.log.node('cycle', f'Is the new M_0,r ({M0.label}) a '
                              'model already used as M_0,r?',
                              'Yes' if cycle else 'No')
                if cycle:
                    return self.three_sigma(M0, fit0)
                continue
            # all the other exits lead to the 3 sigma test on M_0,r
            M0, fit0 = self.prune(M0, fit0)
            return self.three_sigma(M0, fit0)

        self.log.node('stop', 'max_iterations reached', 'stop',
                      max_iterations=self.max_it)
        M0, fit0 = self.prune(M0, self.F.full_fit(M0))
        return dict(conclusion='max_iterations', model=M0.label)

    # ── residual tests ────────────────────────────────────────────────────
    def residual_scan(self, base: FitResult, groups, name, which):
        """
        Residual test for each group; returns the detected ones sorted by
        decreasing evidence.
        """
        found = []
        for g in groups:
            rr = self.F.residual_test(base, g)
            hpd = {a: list(v) for a, v in rr.hpd.items()}
            self.log.node('resid_fit', f'{name}: moment {g} on the residuals '
                          f'of {base.model.label} ({which})',
                          'detected' if rr.detected else 'not detected',
                          stage=rr.stage, logZ=rr.logZ, logZerr=rr.logZerr,
                          pruned=list(rr.off), hpd68=hpd)
            if rr.detected:
                found.append(rr)
        found.sort(key=lambda rr: -rr.logZ)
        self.log.node('resid', f'{name}: is there any moment in the '
                      'residuals?', 'Yes' if found else 'No',
                      detected=[rr.group for rr in found],
                      logZ=[rr.logZ for rr in found])
        return found

    def lnB_decided(self, fit1: FitResult, fit0: FitResult, threshold):
        """
        ln B(fit1 vs fit0); if it is borderline with respect to the
        threshold, PolyChord is repeated for both (alternating, first the one
        with fewer runs) until it leaves the band or seed_max_runs is reached.
        """
        lb, lberr = lnB(fit1, fit0)
        while abs(lb - threshold) < self.seed_band * lberr:
            nr = [max(1, len(f.logZ_runs)) for f in (fit1, fit0)]
            if min(nr) >= self.seed_max_runs:
                break
            f = fit1 if nr[0] <= nr[1] else fit0
            self.log.info(f"    ln B = {lb:.2f} +- {lberr:.2f} too close to "
                          f"{threshold:g}: PolyChord rerun "
                          f"{min(nr) + 1}/{self.seed_max_runs} of "
                          f"{f.model.label} with a new seed")
            self.F.repeat_pc(f)
            lb, lberr = lnB(fit1, fit0)
        return lb, lberr

    def compare(self, fit1: FitResult, fit0: FitResult, name):
        lb, lberr = self.lnB_decided(fit1, fit0, self.lnB_pref)
        pref = lb > self.lnB_pref
        self.log.node('pref', f'Is {name} ({fit1.model.label}) preferred '
                      f'over M_0,r ({fit0.model.label})?',
                      'Yes' if pref else 'No', lnB=lb, lnB_err=lberr,
                      pc_runs=[len(fit1.logZ_runs), len(fit0.logZ_runs)],
                      lnB_preferred=self.lnB_pref)
        return pref

    def compare_pruned(self, M1: Model, fit0: FitResult, name,
                       centers=None):
        """
        Pruning before the comparison. Screening (ML + emcee, no evidence):
        the new spectra (not in M_0,r) with zero inside the HPD of their
        amplitude are removed and the screening is redone, until the
        remaining ones are all detected; if none is left the candidate is
        rejected. Then the full fit (ML + PolyChord) and the pruning of all
        the moment spectra of the model (prune); only the pruned model is
        compared with M_0,r, rejected if no new spectrum is left.
        Returns (preferred, fit or None).
        """
        old = set(fit0.model.amplitudes())
        while True:
            fit1 = self.F.screen_fit(M1, centers=centers)
            self.log.node('fit', f'screening fit (ML + emcee) of {name} = '
                          f'{M1.label}', 'ok', **fit1.summary())
            new = [a for a in M1.amplitudes() if a not in old]
            if not new:
                return False, None
            _, hpd = detection(fit1.sample_names, fit1.samples, new,
                               self.hpd_frac, self.F.prior_bounds(new))
            weak = [a for a in new if hpd[a][0] <= 0.0 <= hpd[a][1]]
            self.log.node('prune_check', f'Undetected new moment amplitudes '
                          f'in {M1.label}?',
                          'Yes' if weak else 'No', undetected=weak,
                          hpd68={a: list(v) for a, v in hpd.items()})
            if not weak:
                break
            if len(weak) == len(new):
                self.log.node('pruned_out', f'{name}: no new moment '
                              'detected, candidate rejected', 'rejected',
                              model=M1.label)
                return False, None
            M1, centers = M1.without(weak), fit1.best
        fit1 = self.F.full_fit(M1, centers=fit1.best)
        self.log.node('fit', f'fit of {name} = {M1.label}', 'ok',
                      **fit1.summary())
        M1, fit1 = self.prune(M1, fit1)
        if not set(M1.amplitudes()) - old:
            self.log.node('pruned_out', f'{name}: no new moment left after '
                          'the pruning, candidate rejected', 'rejected',
                          model=M1.label)
            return False, None
        return self.compare(fit1, fit0, name), fit1

    # ── moment search ─────────────────────────────────────────────────────
    def search_moments(self, M0: Model, fit0: FitResult, use_wr=True):
        """
        Returns ('replace', M1, branch) if a model with moments must be
        adopted, otherwise ('to_3sigma', None, branch).
        """
        # ── Res_wr: everything fixed at the best fit of M_0,r, r included ──
        groups = candidate_groups(M0, self.res_wr,
                                  extra_requires=self.extra_requires) \
            if use_wr else []
        if not use_wr:
            self.log.info("    (Res_wr skipped: the current model comes "
                          "from Res_r0)")
        found = self.residual_scan(fit0, groups, 'Res_wr',
                                   'all fixed at the best fit') if groups \
            else []
        if found:
            for rr in found:
                M1 = M0.apply(rr.group).without(rr.off)
                try:
                    pref, fit1 = self.compare_pruned(
                        M1, fit0, 'M_1,r', centers={**fit0.best, **rr.best})
                except NoValidStart as err:
                    self.log.node('unfittable', f'Res_wr: {rr.group} cannot '
                                  'be fitted with the moment constraints',
                                  'skipped', reason=str(err))
                    continue
                if pref:
                    return 'replace', fit1.model, 'Res_wr'
            # no candidate passes: Res_r0, where with r = 0 the families
            # can be detected or preferred (and the higher orders tested)
            self.log.info("    (Res_wr candidates not preferred: going to "
                          "Res_r0)")

        # ── Res_r0: residuals of M_0,0 (r = 0) ────────────────────────────
        M00 = M0.with_r(False)
        fit00 = self.F.ml_fit(M00, centers=fit0.best)
        self.log.node('fit', f'best fit of {M00.label} (r = 0) for the '
                      'residuals', 'ok', chi2=fit00.chi2, pte=fit00.pte)
        groups = candidate_groups(M00, self.res_r0, restore=self.restore,
                                  extra_requires=self.extra_requires)
        found = self.residual_scan(fit00, groups, 'Res_r0', 'r fixed at 0') \
            if groups else []
        if not found:
            return 'to_3sigma', None, 'Res_r0'

        # candidates in order of evidence; one that cannot be fitted (no
        # starting point allowed by the moment constraints) is skipped and
        # the next one is tried
        for rr in found:
            M1r = M0.apply(rr.group).without(rr.off)
            try:
                pref, fit1r = self.compare_pruned(
                    M1r, fit0, 'M_1,r', centers={**fit0.best, **rr.best})
            except NoValidStart as err:
                self.log.node('unfittable', f'Res_r0: {rr.group} cannot be '
                              'fitted with the moment constraints, next '
                              'candidate', 'skipped', reason=str(err))
                continue
            break
        else:
            return 'to_3sigma', None, 'Res_r0'
        if pref:
            return 'replace', fit1r.model, 'Res_r0'

        M10 = M1r.with_r(False)
        pref, fit10 = self.compare_pruned(
            M10, fit0, 'M_1,0', centers={**fit00.best, **rr.best})
        if not pref:
            return 'to_3sigma', None, 'Res_r0'
        if self.profile_scheme:
            # the profile scheme restarts from M_1 (same model, r free)
            self.log.node('to_profile', 'M_1,0 preferred: the profile scheme '
                          'will start from M_1 (no frequentist branch here)',
                          fit10.model.with_r(True).label)
            return 'to_3sigma', None, 'Res_r0'
        # profile: the same model (pruned as M_1,0) with r free
        M1r = fit10.model.with_r(True)
        fit1r = self.F.full_fit(M1r, centers=fit10.best)

        # ── frequentist branch: profile likelihood ────────────────────────
        out, M1 = self.profile_branch(M0, fit0, M1r, fit1r)
        return out, M1, 'Res_r0'

    def profile_branch(self, M0, fit0, M1r, fit1r):
        self.log.info("    (frequentist branch: profile likelihood in r; "
                      "see the note in the flowchart)")
        p0 = self.F.profile(M0, self.r_grid, centers=fit0.best)
        p1 = self.F.profile(M1r, self.r_grid, centers=fit1r.best)
        np.savez(os.path.join(self.tc['paths']['runs_dir'],
                              f'profile_{M0.label}_vs_{M1r.label}.npz'),
                 r=self.r_grid, chi2_M0=p0, chi2_M1=p1)
        dchi2 = float(np.min(p0) - np.min(p1))
        ddof = M1r.n_moment_params() - M0.n_moment_params()
        pval = float(stats.chi2.sf(max(dchi2, 0.), ddof))
        r0 = profile_interval(self.r_grid, p0)
        r1 = profile_interval(self.r_grid, p1)
        pref = pval < self.alpha
        self.log.node('pref_profile', f'Is M*_1,r ({M1r.label}) preferred '
                      f'over M*_0,r ({M0.label})?', 'Yes' if pref else 'No',
                      dchi2=dchi2, ddof=ddof, p_value=pval,
                      lrt_alpha=self.alpha,
                      r_best_M0=r0[0], r_lo_M0=r0[1], r_hi_M0=r0[2],
                      r_best_M1=r1[0], r_lo_M1=r1[1], r_hi_M1=r1[2])
        if pref:
            self.log.node('mstar', 'M* = M*_1,r', M1r.label)
            if self.freq_replaces:
                return 'replace', M1r
            self.log.info("    frequentist_replaces_baseline=false: "
                          "M_0,r stays the baseline")
            return 'to_3sigma', None
        self.log.node('mstar', 'M* = M*_0,r', M0.label)
        return 'to_3sigma', None

    # ── pruning ───────────────────────────────────────────────────────────
    def prune(self, M0, fit0):
        """
        Removes from the final model the undetected moment spectra (zero
        inside the HPD of their amplitude in the posterior of the full fit),
        refits and keeps the pruned model if it is not disfavoured
        (ln B > lnB_prune). Repeated until only detected amplitudes are left.
        """
        while M0.amplitudes():
            amps = M0.amplitudes()
            det, hpd = detection(fit0.sample_names, fit0.samples, amps,
                                 self.hpd_frac, self.F.prior_bounds(amps))
            weak = [a for a in amps if hpd[a][0] <= 0.0 <= hpd[a][1]]
            self.log.node('prune_check', f'Are all moment amplitudes of '
                          f'{M0.label} detected?', 'No' if weak else 'Yes',
                          undetected=weak,
                          hpd68={a: list(v) for a, v in hpd.items()})
            if not weak:
                break
            M1 = M0.without(weak)
            fit1 = self.F.full_fit(M1, centers=fit0.best)
            self.log.node('fit', f'fit of the pruned model {M1.label}', 'ok',
                          **fit1.summary())
            lb, lberr = self.lnB_decided(fit1, fit0, self.lnB_prune)
            keep = lb > self.lnB_prune
            self.log.node('prune', f'Is the pruned model {M1.label} not '
                          f'disfavoured w.r.t. {M0.label}?',
                          'Yes' if keep else 'No', lnB=lb, lnB_err=lberr,
                          pc_runs=[len(fit1.logZ_runs),
                                   len(fit0.logZ_runs)],
                          lnB_prune=self.lnB_prune)
            if not keep:
                break
            M0, fit0 = M1, fit1
        return M0, fit0

    # ── exits ─────────────────────────────────────────────────────────────
    def three_sigma(self, M0, fit0):
        ok3, rs = fit0.r_compatible_with_zero(3)
        self.log.node('r3sigma', 'Is r for M_0,r compatible with zero at '
                      '3 sigma?', 'Yes' if ok3 else 'No',
                      r_med=rs['r_med'], r_lo3=rs['r_lo3'],
                      r_hi3=rs['r_hi3'])
        if ok3:
            return self.conclude_upper_limit(M0, fit0)
        return self.conclude_detection(M0, fit0)

    def _fit_r0(self, M0, fit0):
        M00 = M0.with_r(False)
        fit00 = self.F.full_fit(M00, centers=fit0.best)
        lb, lberr = lnB(fit0, fit00)
        return fit00, lb, lberr

    def conclude_upper_limit(self, M0, fit0):
        fit00, lb, lberr = self._fit_r0(M0, fit0)          # M_0,0
        rs = fit0.r_summary()
        res = dict(conclusion='upper_limit', model=M0.label,
                   r_95=rs['r_95'], r_med=rs['r_med'],
                   lnB_rfree_vs_r0=lb, lnB_err=lberr,
                   chi2_M00=fit00.chi2, pte_M00=fit00.pte)
        self.log.node('END', 'put an upper limit on r', 'upper limit', **res)
        return res

    def conclude_detection(self, M0, fit0):
        fit00, lb, lberr = self._fit_r0(M0, fit0)
        rs = fit0.r_summary()
        res = dict(conclusion='r_detection', model=M0.label,
                   r_med=rs['r_med'], r_lo1=rs['r_lo1'], r_hi1=rs['r_hi1'],
                   r_snr=rs['r_mean'] / rs['r_std'],
                   lnB_rfree_vs_r0=lb, lnB_err=lberr)
        self.log.node('END', 'You cannot get rid of your r detection!!! Be '
                      'careful and wait before organizing a press '
                      'conference!!!', 'r detection', **res)
        return res
