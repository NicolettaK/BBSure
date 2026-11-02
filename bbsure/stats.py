"""
Fit results and the statistics used by the tree: chi2 and PTE, summary of
the posterior of r, Bayes factors, HPD intervals and detection of moment
amplitudes.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from .models import Model


@dataclass
class FitResult:
    model: Model
    tag: str
    names: List[str] = field(default_factory=list)
    best: Dict[str, float] = field(default_factory=dict)  # best fit (ML)
    chi2: float = np.nan        # -2 lnL at the best fit (without priors)
    ndata: int = 0
    logZ: float = np.nan
    logZerr: float = np.nan
    r_samples: Optional[np.ndarray] = None   # equal-weight posterior of r
    sample_names: List[str] = field(default_factory=list)
    samples: Optional[np.ndarray] = None     # equal-weight posterior
    # one element per PolyChord run (different seeds): logZ is the mean
    logZ_runs: List[float] = field(default_factory=list)
    logZerr_runs: List[float] = field(default_factory=list)

    @property
    def nfree(self):
        return len(self.names)

    @property
    def dof(self):
        return self.ndata - self.nfree

    @property
    def pte(self):
        return float(stats.chi2.sf(self.chi2, self.dof))

    def r_summary(self):
        if self.r_samples is None or len(self.r_samples) == 0:
            return {}
        s = self.r_samples
        q = np.percentile(s, [0.135, 2.275, 15.865, 50., 84.135, 97.725,
                              99.865, 95.])
        # upper limit: 95% of the posterior with r >= 0 (the prior of the
        # tree extends to r < 0 so as not to distort median and bias, but a
        # negative r_95 has no physical meaning); r_95_full = 95% with the
        # full prior
        pos = s[s >= 0.]
        r95 = float(np.percentile(pos, 95.)) if len(pos) else 0.
        return dict(r_med=q[3], r_lo1=q[2], r_hi1=q[4], r_lo2=q[1],
                    r_hi2=q[5], r_lo3=q[0], r_hi3=q[6], r_95=r95,
                    r_95_full=q[7],
                    r_mean=float(np.mean(s)), r_std=float(np.std(s)),
                    n_samples=len(s))

    def r_compatible_with_zero(self, nsigma):
        """0 inside the central nsigma interval of the posterior of r."""
        rs = self.r_summary()
        lo, hi = rs[f'r_lo{nsigma}'], rs[f'r_hi{nsigma}']
        return (lo <= 0.0 <= hi), rs

    def summary(self):
        d = dict(model=self.model.label, tag=self.tag, nfree=self.nfree,
                 chi2=self.chi2, dof=self.dof,
                 chi2_red=self.chi2 / self.dof if self.dof > 0 else np.nan,
                 pte=self.pte if self.dof > 0 else np.nan,
                 logZ=self.logZ, logZerr=self.logZerr,
                 n_pc_runs=max(1, len(self.logZ_runs)),
                 best=self.best)
        d.update(self.r_summary())
        return d


@dataclass
class ResidualResult:
    """Residual test of a moment group."""
    group: str
    detected: bool
    hpd: Dict[str, tuple]               # amplitude -> HPD interval
    logZ: float = np.nan                # only if detected (PolyChord)
    logZerr: float = np.nan
    best: Dict[str, float] = field(default_factory=dict)  # max likelihood
    stage: str = ''                     # 'emcee' or 'polychord'
    off: tuple = ()                     # spectra of the group not detected
                                        # by the screening, removed before PC


def lnB(a: FitResult, b: FitResult):
    """ln B_ab = lnZ_a - lnZ_b, with error."""
    return a.logZ - b.logZ, float(np.hypot(a.logZerr, b.logZerr))


def hpd_interval(x, frac=0.68, bounds=None):
    """
    Highest density (shortest) interval containing frac.
    If less than 1% of the samples are left below (above) the interval and
    the extreme sample is closer to the prior edge than 5% of the width of
    the interval, the posterior is squeezed against the edge: with
    bounds = (lo, hi) the interval is extended to the edge (the samples never
    touch the edge exactly, e.g. an amplitude >= 0 compatible with zero, and
    the sample noise shifts the start of the interval slightly).
    """
    xs = np.sort(np.asarray(x))
    n = len(xs)
    k = max(int(np.ceil(frac * n)), 1)
    widths = xs[k - 1:] - xs[:n - k + 1]
    i = int(np.argmin(widths))
    lo, hi = float(xs[i]), float(xs[i + k - 1])
    if bounds is not None:
        edge, tol = 0.01 * n, 0.05 * (hi - lo)
        if i < edge and xs[0] - bounds[0] < tol:
            lo = float(bounds[0])
        if n - (i + k) < edge and bounds[1] - xs[-1] < tol:
            hi = float(bounds[1])
    return lo, hi


def detection(names, samples, amps, frac=0.68, bounds=None):
    """
    HPD of each amplitude in `amps`; detected if zero is outside the HPD of
    at least one. bounds: {amplitude: (lo, hi)} of the prior.
    """
    bounds = bounds or {}
    hpd = {a: hpd_interval(samples[:, names.index(a)], frac, bounds.get(a))
           for a in amps}
    det = any(not (lo <= 0.0 <= hi) for lo, hi in hpd.values())
    return det, hpd
