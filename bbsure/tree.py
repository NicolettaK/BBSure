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
    each with its spectral parameter free. moments: residual_test: lrt,
    aic or bic: two ML fits (with and without the group), k = amplitudes of
    the group; lrt: detected if p < lrt_p (chi2 with k degrees of freedom;
    0.3173 = 1 sigma); aic / bic: detected if Delta chi2 > 2k / k ln N
    (N = size of the data vector: the AIC / BIC decreases); candidates
    ordered by p (lrt) or by Delta AIC / BIC = penalty - Delta chi2. With
    residual_test: bayes: detected = zero outside the 68% HPD interval of
    at least one amplitude (screening with emcee, then
    PolyChord on the detected ones: confirmation and evidence, which sets
    the order).
      - none detected -> 3.
      - in order of significance: full fit of M_0,r + group and comparison with
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
    test. Each round (M_1 = M_0,r at the first one):
      a) every detected family added to M_1 with r = 0 and its new spectra
         pruned with r = 0 by the screening (ML + emcee; the spectra
         already in M_1 stay); a family pruned to nothing is not tested
         again; the others ordered by Delta AIC / BIC = penalty - (chi2 of M_1
         with r = 0 - chi2 of M_1,0), k = new spectra left;
      b) for each family in this order: M_1,r = M_1,0 with r free, pruned
         as in Res_wr (r free), against M_0,r: preferred -> replaces M_0,r,
         back to 1; otherwise M_1,0 with the full pruning with r = 0
         (screening + full fit) against M_0,r: preferred -> stop, M_1 to
         the profile scheme; otherwise the next family;
      c) none preferred: Res_r0 on the residuals of the M_1,0 of the first
         family of the list, M_1 = that M_1,0 and the next round; nothing
         left in the residuals -> stop. The families tested and not
         detected, or pruned to nothing with r = 0, in a round of the chain
         are not tested again in the next rounds.
    In both cases the Bayesian result is the 3 sigma test on M_0,r; the
    profile scheme compares M_0,r, M_1 and the models in between.
    The families added to M_0,r (node 'r0chain') are the candidates of the
    profile scheme (profile_scheme.py) in both cases, each model at its own
    r.
    Res_wr is repeated as long as the replacements come from Res_wr (e.g.
    after M + d, s is tested); after a replacement from Res_r0 the tree
    always goes directly to Res_r0.
 4. Pruning (of every candidate before its comparison with M_0,r, and of
    the final model): the spectra with zero inside the HPD of their
    amplitude are removed and the model refitted;
    the pruned model is kept if ln B(pruned vs full) > lnB_prune (0).
    Repeated until only detected amplitudes are left.
 If after the pruning the new M_0,r is a model already used as M_0,r
 (cycle, e.g. restore-s adopted and then pruned again) -> 3 sigma test.
 5. 3 sigma test on r of M_0,r (or upper limit from point 1) on the pruned
    model: upper limit or r detection.
 Borderline comparisons: if |ln B - threshold| < seed_band * error, the two
 PolyChord fits are repeated with another (random) seed and ln Z averaged,
 up to seed_max_runs runs per fit. This only reduces the numerical error,
 not the fluctuation of the realization. A choice is flagged as undecided if
 |ln B - threshold| < lnB_flag (1: inconclusive on the Jeffreys scale of
 Trotta 2008) or still < seed_band * error after the repetitions.

Output (in runs_dir)
--------------------
  decision_log.json   every node of the tree with question, answer, numbers
  decision_log.txt    the same in readable form
  <tag>/              one directory per BBCompSep run (config + output)
"""
import numpy as np

from .fitter import NoValidStart
from .log import DecisionLog
from .models import (Model, candidate_groups, group_amplitudes,
                     model_from_label)
from .stats import FitResult, detection, lnB


class DecisionTree:

    def __init__(self, tc, fitter, log: DecisionLog):
        self.tc = tc
        self.F = fitter
        self.log = log
        th = tc['thresholds']
        self.pte_min = th['pte_min']
        # final model with PTE < pte_flag: flagged in the conclusion (the
        # best model available does not describe the data well)
        self.pte_flag = th.get('pte_flag', 0.05)
        self.lnB_pref = th['lnB_preferred']
        # pruning: the model without the undetected spectra is kept if
        # ln B(pruned vs full) > lnB_prune
        self.lnB_prune = th.get('lnB_prune', 0.)
        self.hpd_frac = th.get('hpd_frac', 0.68)
        # borderline comparisons: |ln B - threshold| < seed_band * error ->
        # more PolyChord runs (different seeds), up to seed_max_runs per fit
        self.seed_band = th.get('seed_band', 2.0)
        self.seed_max_runs = th.get('seed_max_runs', 1)
        # choices with |ln B - threshold| < lnB_flag are flagged as
        # undecided (inconclusive evidence), whatever the numerical error
        self.lnB_flag = th.get('lnB_flag', 1.0)
        # comparisons of a candidate with M_0,r first with the ML fits:
        # ln B ~ -Delta chi2 / 2 - occam_per_param * Delta k; below the
        # threshold - lnB_approx_skip -> rejected without PolyChord;
        # otherwise PolyChord (None: always PolyChord). Acceptances always go
        # through PolyChord (the accepted model needs its evidence and
        # posterior); not used for the pruning (a stuck ML fit of the pruned
        # model would remove a spectrum without check)
        self.approx_skip = th.get('lnB_approx_skip', None)
        self.occam = th.get('occam_per_param', 1.0)
        mc = tc['moments']
        self.res_wr = mc.get('res_wr', ['d', 's'])
        self.res_r0 = mc.get('res_r0', ['d', 's', 'dT', 'd2', 's2', 'ds1',
                                        'ds2'])
        self.extra_requires = mc.get('requires', {})
        # restore-<group> in Res_r0: off by default. The pruned spectra have
        # already been judged with r free; with r = 0 they look detected but
        # refitted with r free they are undetected again (d10s5 seed 6002)
        self.restore = mc.get('res_r0_restore', False)
        # optional cap on the iterations (none by default: the cycle check
        # below ends the loop, the models are finite)
        self.max_it = tc['run'].get('max_iterations')
        # comparisons still within seed_band errors of the threshold after
        # the PolyChord repetitions: reported in the conclusion
        self.undecided = []
        # PolyChord of M_0,0 at the end, only for ln B(r free vs r = 0) in
        # the conclusion (off: one PolyChord run less per tree)
        self.fit_r0_at_end = tc['run'].get('fit_r0_at_end', True)

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
        it = 0
        while self.max_it is None or it < self.max_it:
            it += 1
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
            if outcome == 'to_profile':
                # M_1,0 preferred over M_0,r: the Bayesian result is still
                # that of M_0,r (3 sigma test); the profile scheme compares
                # M_0,r, M_1 and the models in between
                self.log.node('m10', 'M_1,0 preferred over M_0,r: 3 sigma '
                              'test on M_0,r, M_1 to the profile scheme',
                              M1.label)
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
        return dict(conclusion='max_iterations', model=M0.label,
                    **self.flags(fit0))

    # ── residual tests ────────────────────────────────────────────────────
    def residual_scan(self, base: FitResult, groups, name, which):
        """
        Residual test for each group; returns the detected ones sorted by
        decreasing evidence (or significance, with the likelihood-ratio
        test).
        """
        found = []
        # groups tested and not detected (excluded from the next rounds of
        # the Res_r0 chain)
        self.last_undetected = []
        for g in groups:
            try:
                rr = self.F.residual_test(base, g)
            except NoValidStart as err:
                self.log.node('unfittable', f'{name}: {g} cannot be fitted '
                              'with the moment constraints, skipped',
                              'skipped', reason=str(err))
                continue
            hpd = {a: list(v) for a, v in rr.hpd.items()}
            self.log.node('resid_fit', f'{name}: moment {g} on the residuals '
                          f'of {base.model.label} ({which})',
                          'detected' if rr.detected else 'not detected',
                          stage=rr.stage, logZ=rr.logZ, logZerr=rr.logZerr,
                          dchi2=rr.dchi2, pvalue=rr.pvalue, dchi2_thr=rr.thr,
                          pruned=list(rr.off), hpd68=hpd)
            if rr.detected:
                found.append(rr)
            else:
                self.last_undetected.append(g)
        found.sort(key=lambda rr: rr.rank)
        self.log.node('resid', f'{name}: is there any moment in the '
                      'residuals?', 'Yes' if found else 'No',
                      base=base.model.label,
                      detected=[rr.group for rr in found],
                      pruned=[list(rr.off) for rr in found],
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

    def note_undecided(self, kind, chosen, other, lb, lberr, threshold):
        """Records a choice taken with |ln B - threshold| < lnB_flag
        (inconclusive evidence) or within seed_band errors of the threshold
        (also after the repetitions): flagged in the results."""
        if abs(lb - threshold) >= max(self.lnB_flag, self.seed_band * lberr):
            return
        d = dict(kind=kind, chosen=chosen, other=other, lnB=lb,
                 lnB_err=lberr, threshold=threshold)
        self.undecided.append(d)
        self.log.info(f"    [undecided] {kind}: {chosen} chosen over {other}"
                      f" with ln B = {lb:.2f} +- {lberr:.2f}")

    def compare(self, fit1: FitResult, fit0: FitResult, name):
        lb, lberr = self.lnB_decided(fit1, fit0, self.lnB_pref)
        pref = lb > self.lnB_pref
        m1, m0 = fit1.model.label, fit0.model.label
        self.note_undecided(f'{name} vs M_0,r', m1 if pref else m0,
                            m0 if pref else m1, lb, lberr, self.lnB_pref)
        self.log.node('pref', f'Is {name} ({fit1.model.label}) preferred '
                      f'over M_0,r ({fit0.model.label})?',
                      'Yes' if pref else 'No', lnB=lb, lnB_err=lberr,
                      pc_runs=[len(fit1.logZ_runs), len(fit0.logZ_runs)],
                      lnB_preferred=self.lnB_pref)
        return pref

    def lnB_approx(self, fit1: FitResult, fit0: FitResult):
        """
        ln B(fit1 vs fit0) from the ML fits: -Delta chi2 / 2 minus an Occam
        penalty of occam_per_param per extra free parameter. Calibrated on
        the 490 comparisons of the seed-8000 validation (1.0 per parameter,
        scatter about 2 around the PolyChord ln B; the BIC penalty,
        ln N / 2 ~ 3 per parameter, is too strong with our priors).
        """
        return (-0.5 * (fit1.chi2 - fit0.chi2)
                - self.occam * (fit1.nfree - fit0.nfree))

    def rejected_by_approx(self, fit1: FitResult, fit0: FitResult, name):
        """
        True (and the comparison logged as not preferred) if the ML
        estimate already rejects fit1 against M_0,r by more than
        lnB_approx_skip: no PolyChord needed. Never used to accept.
        """
        if self.approx_skip is None:
            return False
        lb = self.lnB_approx(fit1, fit0)
        if lb >= self.lnB_pref - self.approx_skip:
            return False
        self.log.node('pref', f'Is {name} ({fit1.model.label}) preferred '
                      f'over M_0,r ({fit0.model.label})?', 'No', lnB=lb,
                      lnB_err=float('nan'), method='ML estimate',
                      lnB_preferred=self.lnB_pref,
                      lnB_approx_skip=self.approx_skip,
                      occam_per_param=self.occam)
        return True

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
        if self.rejected_by_approx(fit1, fit0, name):
            return False, None
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

    def prune_r0(self, M10: Model, old, centers=None):
        """
        Pruning with r = 0 of the new spectra of M_1,0 (those not in old):
        screening (ML + emcee) until the new spectra left are all detected.
        The spectra in old are never removed. No PolyChord here: the
        evidence of M_1,0 is computed only if it is compared with M_0,r
        (M_1,r not preferred).
        Returns (M_1,0, screening fit) or None if no new spectrum is left.
        """
        old = set(old)
        while True:
            fit = self.F.screen_fit(M10, centers=centers)
            self.log.node('fit', f'screening fit (ML + emcee) of M_1,0 = '
                          f'{M10.label}', 'ok', **fit.summary())
            new = [a for a in M10.amplitudes() if a not in old]
            _, hpd = detection(fit.sample_names, fit.samples, new,
                               self.hpd_frac, self.F.prior_bounds(new))
            weak = [a for a in new if hpd[a][0] <= 0.0 <= hpd[a][1]]
            self.log.node('prune_check', f'Undetected new moment amplitudes '
                          f'in {M10.label} (r = 0)?',
                          'Yes' if weak else 'No', undetected=weak,
                          hpd68={a: list(v) for a, v in hpd.items()})
            if not weak:
                break
            if len(weak) == len(new):
                self.log.node('pruned_out', 'M_1,0: no new moment detected '
                              'with r = 0, family rejected', 'rejected',
                              model=M10.label)
                return None
            M10, centers = M10.without(weak), fit.best
        return M10, fit

    # ── moment search ─────────────────────────────────────────────────────
    def search_moments(self, M0: Model, fit0: FitResult, use_wr=True):
        """
        Returns ('replace', M1, branch) if a model with moments must be
        adopted, ('to_profile', M1, 'Res_r0') if M_1,0 is preferred over
        M_0,r, otherwise ('to_3sigma', None, branch).
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
        # families tested and not detected: not tested again in the next
        # rounds of this chain
        excluded = set(self.last_undetected)

        # rounds (M_1 = M_0,r at the first one): every detected family
        # pruned with r = 0 and ordered by Delta AIC / BIC (r0_candidates); each
        # tried as M_1,r, then M_1,0 (try_family); none preferred -> Res_r0
        # on the residuals of the M_1,0 of the first family, next round
        M1, chained = M0, False
        while True:
            old = set(M1.amplitudes())
            first = None
            for g, M10, scr in self.r0_candidates(M1, found, fit00, excluded):
                out, M10, fit10 = self.try_family(M0, fit0, M10, scr, old)
                if out == 'replace':
                    return 'replace', M10, 'Res_r0'
                if out == 'm10':
                    M1 = M10.with_r(True)
                    self.log_chain(M0, M1, True)
                    return 'to_profile', M1, 'Res_r0'
                if out == 'out':
                    excluded.add(g)
                elif first is None:
                    first = (M10, fit10)
            if first is None:
                break
            M10, fit10 = first
            M1, chained = M10.with_r(True), True
            groups = [g for g in candidate_groups(
                M10, self.res_r0, extra_requires=self.extra_requires)
                if g not in excluded]
            found = self.residual_scan(fit10, groups, 'Res_r0',
                                       'r fixed at 0') if groups else []
            if not groups:
                self.log.info("    (Res_r0: no family left to test on the "
                              "residuals of M_1,0)")
            else:
                excluded |= set(self.last_undetected)
            if not found:
                break
            fit00 = fit10
        if chained:
            self.log_chain(M0, M1, False)
        return 'to_3sigma', None, 'Res_r0'

    def r0_candidates(self, M1, found, fit00, excluded):
        """
        Every detected family added to M_1 with r = 0 and its new spectra
        pruned with r = 0 by the screening (prune_r0); the families pruned
        to nothing go in excluded. Returns [(family, M_1,0, screening fit)]
        ordered by Delta AIC / BIC = penalty - (chi2 of M_1 with r = 0 -
        chi2 of M_1,0), penalty 2k / k ln N, k = new spectra left (lowest
        first).
        """
        old = set(M1.amplitudes())
        cands = []
        for rr in found:
            cand = M1.with_r(False).apply(rr.group).without(rr.off)
            try:
                got = self.prune_r0(cand, old,
                                    centers={**fit00.best, **rr.best})
            except NoValidStart as err:
                self.log.node('unfittable', f'Res_r0: {rr.group} cannot be '
                              'fitted with the moment constraints, next '
                              'candidate', 'skipped', reason=str(err))
                continue
            if got is None:
                # pruned to nothing with r = 0: the moments can only grow
                # along the chain, so it would be pruned again
                excluded.add(rr.group)
                continue
            M10, scr = got
            k = len(set(M10.amplitudes()) - old)
            dbic = self.F.ic_penalty(k, fit00) - (fit00.chi2 - scr.chi2)
            cands.append((dbic, rr.group, M10, scr))
        cands.sort(key=lambda c: c[0])
        if cands:
            self.log.node('r0order', 'Res_r0: families pruned with r = 0, in '
                          f'order of Delta {self.F.ic_name()}',
                          ', '.join(f'{g} ({b:+.1f})' for b, g, _, _ in cands),
                          order=[[g, b, M.label] for b, g, M, _ in cands])
        return [(g, M10, scr) for _, g, M10, scr in cands]

    def try_family(self, M0, fit0, M10, scr, old):
        """
        One family of the Res_r0 list: M_1,r = M_1,0 with r free, pruned as
        in Res_wr, against M_0,r; if not preferred, M_1,0 with the full
        pruning with r = 0 (screening done, then full fit) against M_0,r.
        Returns (outcome, model, fit): 'replace' (M_1,r adopted), 'm10'
        (M_1,0 preferred), 'lost' (neither) or 'out' (M_1,0 pruned to
        nothing or not fittable).
        """
        M1r = M10.with_r(True)
        try:
            pref, fit1r = self.compare_pruned(
                M1r, fit0, 'M_1,r', centers={**fit0.best, **scr.best})
        except NoValidStart as err:
            self.log.node('unfittable', f'M_1,r = {M1r.label} cannot be '
                          'fitted with the moment constraints', 'skipped',
                          reason=str(err))
            pref = False
        if pref:
            return 'replace', fit1r.model, fit1r
        if self.rejected_by_approx(scr, fit0, 'M_1,0'):
            return 'lost', M10, scr
        try:
            fit10 = self.F.full_fit(M10, centers=scr.best)
        except NoValidStart as err:
            self.log.node('unfittable', f'M_1,0 = {M10.label} cannot be '
                          'fitted with the moment constraints', 'skipped',
                          reason=str(err))
            return 'out', M10, None
        self.log.node('fit', f'fit of M_1,0 = {M10.label}', 'ok',
                      **fit10.summary())
        M10, fit10 = self.prune(M10, fit10, keep_amps=old)
        if not set(M10.amplitudes()) - old:
            self.log.node('pruned_out', 'M_1,0: no new moment left after the '
                          'pruning with r = 0', 'rejected', model=M10.label)
            return 'out', M10, fit10
        if self.compare(fit10, fit0, 'M_1,0'):
            return 'm10', M10, fit10
        return 'lost', M10, fit10

    def log_chain(self, M0, M1, preferred):
        """Families added to M_0,r in Res_r0 (each with its removed
        spectra): the candidates of the profile scheme."""
        fams = sorted(M1.moments - M0.moments)
        self.log.node('r0chain', f'Res_r0: families added to M_0,r '
                      f'({M0.label}) up to M_1 = {M1.label}; is M_1,0 '
                      'preferred?', 'Yes' if preferred else 'No',
                      base=M0.label, m1=M1.label, families=fams,
                      pruned=[sorted(set(group_amplitudes(g)) & set(M1.off))
                              for g in fams])

    # ── pruning ───────────────────────────────────────────────────────────
    def prune(self, M0, fit0, keep_amps=()):
        """
        Removes from the final model the undetected moment spectra (zero
        inside the HPD of their amplitude in the posterior of the full fit),
        refits and keeps the pruned model if it is preferred
        (ln B > lnB_prune). Repeated until only detected amplitudes are left.
        The amplitudes in keep_amps are never removed.
        """
        while [a for a in M0.amplitudes() if a not in keep_amps]:
            amps = [a for a in M0.amplitudes() if a not in keep_amps]
            det, hpd = detection(fit0.sample_names, fit0.samples, amps,
                                 self.hpd_frac, self.F.prior_bounds(amps))
            weak = [a for a in amps if hpd[a][0] <= 0.0 <= hpd[a][1]]
            r0 = '' if M0.r_free else ' (r = 0)'
            self.log.node('prune_check', f'Are all moment amplitudes of '
                          f'{M0.label} detected{r0}?', 'No' if weak else 'Yes',
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
            self.note_undecided('pruning', M1.label if keep else M0.label,
                                M0.label if keep else M1.label, lb, lberr,
                                self.lnB_prune)
            self.log.node('prune', f'Is the pruned model {M1.label} preferred '
                          f'over {M0.label}?',
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
        """
        With run: fit_r0_at_end: true (default): full fit (ML + PolyChord)
        of M_0,0 (the final model with r = 0) and ln B(r free vs r = 0),
        reported in the conclusion. No decision uses it; it costs one
        PolyChord run per tree.
        """
        if not self.fit_r0_at_end:
            return {}
        fit00 = self.F.full_fit(M0.with_r(False), centers=fit0.best)
        lb, lberr = lnB(fit0, fit00)
        return dict(lnB_rfree_vs_r0=lb, lnB_err=lberr,
                    chi2_M00=fit00.chi2, pte_M00=fit00.pte)

    def flags(self, fit0):
        return dict(undecided=bool(self.undecided),
                    undecided_choices=list(self.undecided),
                    pte=fit0.pte, low_pte=bool(fit0.pte < self.pte_flag))

    def conclude_upper_limit(self, M0, fit0):
        rs = fit0.r_summary()
        res = dict(conclusion='upper_limit', model=M0.label,
                   r_95=rs['r_95'], r_med=rs['r_med'],
                   **self._fit_r0(M0, fit0), **self.flags(fit0))
        self.log.node('END', 'put an upper limit on r', 'upper limit', **res)
        return res

    def conclude_detection(self, M0, fit0):
        rs = fit0.r_summary()
        res = dict(conclusion='r_detection', model=M0.label,
                   r_med=rs['r_med'], r_lo1=rs['r_lo1'], r_hi1=rs['r_hi1'],
                   r_snr=rs['r_mean'] / rs['r_std'],
                   **self._fit_r0(M0, fit0), **self.flags(fit0))
        self.log.node('END', 'You cannot get rid of your r detection!!! Be '
                      'careful and wait before organizing a press '
                      'conference!!!', 'r detection', **res)
        return res
