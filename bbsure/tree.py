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

Moment groups: see models.py (one spectrum each).

Flowchart
---------
 1. Fit of M_0,r. Good chi2 (PTE > pte_min) and r compatible with 0 at 1
    sigma -> upper limit.
 2. Res_wr: residual test with everything fixed at the best fit of M_0,r, r
    included; candidates are the res_wr groups (e.g. d11, s11, d01, s01)
    not yet in the model whose required groups are there, each with its
    spectral parameter free. Two ML fits (with and without the group),
    k = new amplitudes (1, 2 with the beta_d x T_d cross spectrum):
    aic / bic: detected if Delta chi2 > 2k / k ln N (N = size of the data
    vector); lrt: if p < lrt_p (chi2 with k degrees of freedom); ordered by
    Delta AIC / BIC = penalty - Delta chi2 (lrt: by p).
      - none detected -> 3.
      - in order: full fit (ML + evidence) of M_0,r + group against M_0,r
        (ln B > lnB_preferred). The first one that passes becomes M_0,r
        -> 1.
      - none passes -> 3.
 3. Res_r0: the same on the residuals of M_0,0 (r = 0), candidates the
    res_r0 groups. None detected -> 3 sigma test. Each round (M_1 = M_0,r
    at the first one):
      a) every detected group added to M_1 with r = 0 (ML fit), ordered by
         Delta AIC / BIC = penalty - (chi2 of M_1 with r = 0 - chi2 of
         M_1,0);
      b) for each group in this order: M_1,r = M_1,0 with r free against
         M_0,r: preferred -> replaces M_0,r, back to 1; otherwise M_1,0
         against M_0,r: preferred -> stop, M_1 to the profile scheme;
         otherwise the next group;
      c) none preferred: Res_r0 on the residuals of the M_1,0 of the first
         group of the list, M_1 = that M_1,0 and the next round; nothing
         left in the residuals -> stop. The groups tested and not detected
         in a round are not tested again in the next rounds.
    In both cases the Bayesian result is the 3 sigma test on M_0,r. The
    groups added to M_0,r (node 'r0chain') are the candidates of the
    profile scheme (profile_scheme.py), each model at its own r.
    Res_wr is repeated as long as the replacements come from Res_wr; after
    a replacement from Res_r0 the tree always goes directly to Res_r0.
 4. 3 sigma test on r of M_0,r (or upper limit from point 1): upper limit
    or r detection.
 Comparisons with M_0,r first from the ML fits: ln B ~ -Delta chi2 / 2 -
 occam_per_param * Delta k; rejected without the evidence run if below
 lnB_preferred - lnB_approx_skip (never used to accept).
 Borderline comparisons: if |ln B - threshold| < seed_band * error, the two
 evidence runs are repeated with another (random) seed and ln Z averaged,
 up to seed_max_runs runs per fit. A choice is flagged as undecided if
 |ln B - threshold| < lnB_flag (1: inconclusive on the Jeffreys scale of
 Trotta 2008) or still < seed_band * error after the repetitions.

Output (in runs_dir)
--------------------
  decision_log.json   every node of the tree with question, answer, numbers
  decision_log.txt    the same in readable form
  <tag>/              one directory per BBCompSep run (config + output)
"""
from .fitter import NoValidStart
from .log import DecisionLog
from .models import Model, candidate_groups, model_from_label
from .stats import FitResult, lnB


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
        # through the evidence run (the accepted model needs its evidence
        # and posterior)
        self.approx_skip = th.get('lnB_approx_skip', None)
        self.occam = th.get('occam_per_param', 1.0)
        mc = tc['moments']
        self.res_wr = mc['res_wr']
        self.res_r0 = mc['res_r0']
        self.extra_requires = mc.get('requires', {})
        self.max_it = tc['run']['max_iterations']
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
        # M_d11+s11_r) instead of the one without moments
        start = self.tc['run'].get('initial_model')
        if start:
            M0 = model_from_label(start)
        use_wr = True       # False after a replacement coming from Res_r0
        self.log.info(f"==> Initial model M_0,r = {M0.label}")
        for it in range(1, self.max_it + 1):
            self.log.info(f"\n==> Iteration {it}: M_0,r = {M0.label}")
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
                M0, fit0 = M1, self.F.full_fit(M1)
                use_wr = use_wr and branch == 'Res_wr'
                continue
            # all the other exits lead to the 3 sigma test on M_0,r
            return self.three_sigma(M0, fit0)

        self.log.node('stop', 'max_iterations reached', 'stop',
                      max_iterations=self.max_it)
        fit0 = self.F.full_fit(M0)
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
            self.log.node('resid_fit', f'{name}: moment {g} on the residuals '
                          f'of {base.model.label} ({which})',
                          'detected' if rr.detected else 'not detected',
                          stage=rr.stage,
                          dchi2=rr.dchi2, pvalue=rr.pvalue, dchi2_thr=rr.thr)
            if rr.detected:
                found.append(rr)
            else:
                self.last_undetected.append(g)
        found.sort(key=lambda rr: rr.rank)
        self.log.node('resid', f'{name}: is there any moment in the '
                      'residuals?', 'Yes' if found else 'No',
                      base=base.model.label,
                      detected=[rr.group for rr in found])
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

    def compare_candidate(self, M1: Model, fit0: FitResult, name,
                          centers=None):
        """
        Candidate M1 against M_0,r: ML fit (rejected without the evidence
        run if the ML estimate of ln B is clearly below the threshold), then
        full fit (ML + evidence) and comparison. Returns (preferred, fit or
        None).
        """
        ml = self.F.ml_start(M1, centers=centers)
        self.log.node('fit', f'ML fit of {name} = {M1.label}', 'ok',
                      **ml.summary())
        if self.rejected_by_approx(ml, fit0, name):
            return False, None
        fit1 = self.F.full_fit(M1, centers=ml.best)
        self.log.node('fit', f'fit of {name} = {M1.label}', 'ok',
                      **fit1.summary())
        return self.compare(fit1, fit0, name), fit1

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
                M1 = M0.plus(rr.group)
                try:
                    pref, fit1 = self.compare_candidate(
                        M1, fit0, 'M_1,r', centers={**fit0.best, **rr.best})
                except NoValidStart as err:
                    self.log.node('unfittable', f'Res_wr: {rr.group} cannot '
                                  'be fitted with the moment constraints',
                                  'skipped', reason=str(err))
                    continue
                if pref:
                    return 'replace', fit1.model, 'Res_wr'
            # no candidate passes: Res_r0, where with r = 0 the moments
            # can be detected or preferred (and the higher orders tested)
            self.log.info("    (Res_wr candidates not preferred: going to "
                          "Res_r0)")

        # ── Res_r0: residuals of M_0,0 (r = 0) ────────────────────────────
        M00 = M0.with_r(False)
        fit00 = self.F.ml_fit(M00, centers=fit0.best)
        self.log.node('fit', f'best fit of {M00.label} (r = 0) for the '
                      'residuals', 'ok', chi2=fit00.chi2, pte=fit00.pte)
        groups = candidate_groups(M00, self.res_r0,
                                  extra_requires=self.extra_requires)
        found = self.residual_scan(fit00, groups, 'Res_r0', 'r fixed at 0') \
            if groups else []
        if not found:
            return 'to_3sigma', None, 'Res_r0'
        # groups tested and not detected: not tested again in the next
        # rounds of this chain
        excluded = set(self.last_undetected)

        # rounds (M_1 = M_0,r at the first one): every detected group added
        # with r = 0 and ordered by Delta AIC / BIC (r0_candidates); each
        # tried as M_1,r, then M_1,0 (try_group); none preferred -> Res_r0
        # on the residuals of the M_1,0 of the first group, next round
        M1, chained = M0, False
        while True:
            first = None
            for g, M10, ml10 in self.r0_candidates(M1, found, fit00):
                out, M10, fit10 = self.try_group(M0, fit0, M10, ml10)
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
                self.log.info("    (Res_r0: no group left to test on the "
                              "residuals of M_1,0)")
            else:
                excluded |= set(self.last_undetected)
            if not found:
                break
            fit00 = fit10
        if chained:
            self.log_chain(M0, M1, False)
        return 'to_3sigma', None, 'Res_r0'

    def r0_candidates(self, M1, found, fit00):
        """
        Every detected group added to M_1 with r = 0 (ML fit). Returns
        [(group, M_1,0, ML fit)] ordered by Delta AIC / BIC = penalty -
        (chi2 of M_1 with r = 0 - chi2 of M_1,0), penalty 2k / k ln N,
        k = new amplitudes (lowest first).
        """
        old = set(M1.amplitudes())
        cands = []
        for rr in found:
            M10 = M1.with_r(False).plus(rr.group)
            try:
                ml = self.F.ml_start(M10, centers={**fit00.best, **rr.best})
            except NoValidStart as err:
                self.log.node('unfittable', f'Res_r0: {rr.group} cannot be '
                              'fitted with the moment constraints, next '
                              'candidate', 'skipped', reason=str(err))
                continue
            self.log.node('fit', f'ML fit of M_1,0 = {M10.label}', 'ok',
                          **ml.summary())
            k = len(set(M10.amplitudes()) - old)
            dic = self.F.ic_penalty(k, fit00) - (fit00.chi2 - ml.chi2)
            cands.append((dic, rr.group, M10, ml))
        cands.sort(key=lambda c: c[0])
        if cands:
            self.log.node('r0order', 'Res_r0: groups added with r = 0, in '
                          f'order of Delta {self.F.ic_name()}',
                          ', '.join(f'{g} ({b:+.1f})' for b, g, _, _ in cands),
                          order=[[g, b, M.label] for b, g, M, _ in cands])
        return [(g, M10, ml) for _, g, M10, ml in cands]

    def try_group(self, M0, fit0, M10, ml10):
        """
        One group of the Res_r0 list: M_1,r = M_1,0 with r free against
        M_0,r; if not preferred, M_1,0 against M_0,r. Returns (outcome,
        model, fit): 'replace' (M_1,r adopted), 'm10' (M_1,0 preferred),
        'lost' (neither) or 'out' (M_1,0 not fittable).
        """
        M1r = M10.with_r(True)
        try:
            pref, fit1r = self.compare_candidate(
                M1r, fit0, 'M_1,r', centers={**fit0.best, **ml10.best})
        except NoValidStart as err:
            self.log.node('unfittable', f'M_1,r = {M1r.label} cannot be '
                          'fitted with the moment constraints', 'skipped',
                          reason=str(err))
            pref = False
        if pref:
            return 'replace', fit1r.model, fit1r
        if self.rejected_by_approx(ml10, fit0, 'M_1,0'):
            return 'lost', M10, ml10
        try:
            fit10 = self.F.full_fit(M10, centers=ml10.best)
        except NoValidStart as err:
            self.log.node('unfittable', f'M_1,0 = {M10.label} cannot be '
                          'fitted with the moment constraints', 'skipped',
                          reason=str(err))
            return 'out', M10, None
        self.log.node('fit', f'fit of M_1,0 = {M10.label}', 'ok',
                      **fit10.summary())
        if self.compare(fit10, fit0, 'M_1,0'):
            return 'm10', M10, fit10
        return 'lost', M10, fit10

    def log_chain(self, M0, M1, preferred):
        """Groups added to M_0,r in Res_r0: the candidates of the profile
        scheme."""
        fams = sorted(M1.moments - M0.moments)
        self.log.node('r0chain', f'Res_r0: groups added to M_0,r '
                      f'({M0.label}) up to M_1 = {M1.label}; is M_1,0 '
                      'preferred?', 'Yes' if preferred else 'No',
                      base=M0.label, m1=M1.label, families=fams)

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
