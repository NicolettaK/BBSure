"""
High-level fit operations used by the tree and by the profile scheme: full
fit (ML + PolyChord), screening fit (ML + emcee), residual test of a moment
group.
"""
import copy
import re

import numpy as np
from scipy.stats import chi2 as chi2_dist

from .config import iter_param_entries
from .models import MOMENT_GROUPS, Model, cand_group
from .runner import BBCompSepRunner, NoValidStart  # noqa: F401
from .stats import FitResult, ResidualResult, detection


class Fitter:
    """
    High-level operations used by the tree. A MockFitter with the same
    interface is used in tests/test_decision_tree.py.
    """

    def __init__(self, runner: BBCompSepRunner, hpd_frac=0.68,
                 residual_test='lrt', lrt_p=0.3173):
        self.R = runner
        self.hpd_frac = hpd_frac
        # residual test: 'lrt' (likelihood ratio from two ML fits, detected
        # if p < lrt_p) or 'bayes' (emcee HPD, PolyChord evidence)
        self.residual_test_kind = residual_test
        self.lrt_p = lrt_p
        self._cache = {}

    def full_fit(self, model: Model, centers=None) -> FitResult:
        """ML (best fit, chi2) + PolyChord (evidence, posterior of r)."""
        key = ('full', model)
        if key in self._cache:
            return self._cache[key]
        res = copy.copy(self.ml_fit(model, centers=centers))
        logZ, logZerr, names, samp, m2lnl = self.R.polychord(
            model, f"{model.label}_PC", centers=res.best)
        if not np.isfinite(res.chi2):
            # the minimizer started outside the region allowed by the moment
            # constraints (lnL = -inf) and stayed there: restart from the
            # PolyChord sample with the highest likelihood
            start = dict(zip(names, map(float, samp[int(np.argmin(m2lnl))])))
            ml = self.R.ml(model, f"{model.label}_ML2", centers=start)
            if np.isfinite(ml.chi2):
                res.best, res.chi2 = ml.best, ml.chi2
            else:
                # -2lnL from PolyChord: already without the priors
                res.best, res.chi2 = start, float(np.min(m2lnl))
        res.logZ, res.logZerr = logZ, logZerr
        res.logZ_runs, res.logZerr_runs = [logZ], [logZerr]
        res.sample_names, res.samples = names, samp
        if 'r_tensor' in names:
            res.r_samples = samp[:, names.index('r_tensor')]
        self._cache[key] = res
        return res

    def repeat_pc(self, res: FitResult):
        """
        Another PolyChord run of res.model with a different seed. Updates res
        in place: logZ = mean of the runs, error = sqrt(sum err^2) / N,
        samples concatenated.
        """
        k = len(res.logZ_runs) + 1
        logZ, logZerr, names, samp, _ = self.R.polychord(
            res.model, f"{res.model.label}_PC_rep{k}", centers=res.best)
        assert names == res.sample_names
        res.logZ_runs.append(logZ)
        res.logZerr_runs.append(logZerr)
        n = len(res.logZ_runs)
        res.logZ = float(np.mean(res.logZ_runs))
        res.logZerr = float(np.sqrt(np.sum(np.square(res.logZerr_runs))) / n)
        res.samples = np.vstack([res.samples, samp])
        if 'r_tensor' in names:
            res.r_samples = res.samples[:, names.index('r_tensor')]
        return res

    def prior_bounds(self, params):
        """Prior edges {param: (lo, hi)} from the base config."""
        entries = iter_param_entries(self.R.base['BBCompSep'])
        return {a: (float(entries[a][2][0]), float(entries[a][2][-1]))
                for a in params}

    def screen_fit(self, model: Model, centers=None) -> FitResult:
        """
        ML + emcee (no evidence): only to see which amplitudes are detected
        before the immediate pruning.
        """
        key = ('screen', model)
        if key in self._cache:
            return self._cache[key]
        res = copy.copy(self.ml_fit(model, centers=centers))
        if not np.isfinite(res.chi2):
            # start outside the region allowed by the moment constraints
            # (lnL = -inf everywhere around): restart with all the amplitudes
            # at zero except the first-order auto-spectra (w10w10, w01w01),
            # which stay where they are: positivity and hierarchical bound
            # are satisfied (the zero of the tophats becomes 1e-6 of the
            # width, so first orders at zero would violate them too)
            start = dict(centers or {})
            start.update({a: 0. for a in model.amplitudes()
                          if not re.match(r'A_[ds]_(w10w10|w01w01)$', a)})
            res = copy.copy(self.R.ml(model, f"{model.label}_ML0",
                                      centers=start))
            self._cache[('ml', model)] = res
        if not np.isfinite(res.chi2):
            raise NoValidStart(f"no valid starting point for {model.label}")
        names, samp = self.R.emcee(model, f"{model.label}_MC",
                                   centers=res.best)
        res.sample_names, res.samples = names, samp
        if 'r_tensor' in names:
            res.r_samples = samp[:, names.index('r_tensor')]
        self._cache[key] = res
        return res

    def ml_fit(self, model: Model, centers=None) -> FitResult:
        key = ('ml', model)
        if key not in self._cache:
            self._cache[key] = self.R.ml(model, f"{model.label}_ML",
                                         centers=centers)
        return self._cache[key]

    def residual_test(self, base: FitResult, group) -> ResidualResult:
        """
        Test of the moment group `group` on the residuals of `base` (or, with
        'restore-<group>', of the spectra of that group removed by the
        pruning).

        Free: the spectra of the group (amplitudes and slopes, plus the cross
        spectra with the groups already there) and the spectral parameters of
        the group (MOMENT_GROUPS[...]['free']). Everything else is fixed at
        the best fit of `base`, r included (0 if base has r fixed).

        Screening with emcee: detected if zero is outside the HPD of at least
        one amplitude. If detected, PolyChord: evidence and confirmation of
        the detection with its samples.
        """
        m = base.model.apply(group).with_r(False)
        free = set(MOMENT_GROUPS[cand_group(group)]['free'])
        fixed = {k: v for k, v in base.best.items() if k not in free}
        fixed.setdefault('r_tensor', 0.0)
        old = set(base.model.amplitudes())
        amps = [a for a in m.amplitudes() if a not in old]
        tag = f"resid_{base.model.label}_plus_{group}"
        bounds = self.prior_bounds(amps)

        if self.residual_test_kind == 'lrt':
            return self._residual_lrt(base, group, m, fixed, amps, tag)
        names, samp = self.R.emcee(m, tag + "_MC", fixed=fixed,
                                   centers=base.best)
        det, hpd = detection(names, samp, amps, self.hpd_frac, bounds)
        if not det:
            return ResidualResult(group, False, hpd, stage='emcee')
        # immediate pruning: PolyChord only with the spectra detected by emcee
        weak = tuple(a for a in amps if hpd[a][0] <= 0.0 <= hpd[a][1])
        if weak:
            m = m.without(weak)
            amps = [a for a in amps if a not in weak]
            tag += ''.join(f"-{a[2:]}" for a in weak)
        logZ, logZerr, names, samp, m2lnl = self.R.polychord(
            m, tag + "_PC", fixed=fixed, centers=base.best)
        det, hpd_pc = detection(names, samp, amps, self.hpd_frac,
                                self.prior_bounds(amps))
        hpd.update(hpd_pc)
        i = int(np.argmin(m2lnl))
        best = dict(zip(names, map(float, samp[i])))
        return ResidualResult(group, det, hpd, logZ, logZerr, best,
                              stage='polychord', off=weak)

    def _residual_lrt(self, base, group, m, fixed, amps, tag):
        """
        Likelihood-ratio residual test: ML fit of the group on the residuals
        (everything else fixed as in the Bayesian test) against the base
        best fit; Delta chi2 with k = number of new amplitudes degrees of
        freedom, detected if p < lrt_p (0.3173: 1 sigma, the analogue of
        zero outside the 68% HPD). No pruning of single spectra: the group
        enters whole. Candidates ordered by log p.
        """
        res = self.R.ml(m, tag + "_ML", fixed=fixed, centers=base.best)
        if not np.isfinite(res.chi2):
            # start outside the moment constraints: amplitudes at zero
            # except the first-order auto-spectra (as in screen_fit)
            start = dict(base.best)
            start.update({a: 0. for a in amps
                          if not re.match(r'A_[ds]_(w10w10|w01w01)$', a)})
            res = self.R.ml(m, tag + "_ML0", fixed=fixed, centers=start)
        if not np.isfinite(res.chi2):
            raise NoValidStart(f"no valid starting point for {tag}")
        # nested models: a negative gain is a minimizer that did not move
        dchi2 = max(base.chi2 - res.chi2, 0.)
        k = len(amps)
        logp = float(chi2_dist.logsf(dchi2, k))
        return ResidualResult(group, bool(np.exp(logp) < self.lrt_p), {},
                              best=dict(res.best), stage='lrt',
                              dchi2=dchi2, pvalue=float(np.exp(logp)),
                              rank=logp)
