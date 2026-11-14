"""
Phase 0: frequency-split screening that sets the starting model of the tree.

The data are split in two halves, each fitted with BBCompSep on a subset of
the bands (config option `bands`), r free:
  LF+MF  (e.g. band1-4): only the synchrotron moments are searched, the dust
         has no moments;
  MF+UHF (e.g. band3-6): only the dust moments are searched, the
         synchrotron has no moments.
In each half the search starts from the model without moments and adds one
moment group at a time: every group of the half whose required groups are
already there is fitted (ML + evidence run); the one with the highest
evidence is kept if ln B(with vs without) > lnB_preferred, and the next
round starts from it; otherwise the half stops. No residual test and no
"r compatible with 0" shortcut: in a half r is poorly constrained, only the
moments matter, and a moment hidden in the full data by the other component
can have a small Delta chi2 in the half (d12s5, LF+MF: beta_s x beta_s,
Delta chi2 = 1.1) and still be preferred by the evidence.
The tree on the full data then starts from M_0,r = the groups kept in the
two halves (run: initial_model).

Config (tree config):
  phase0:
      enabled: true
      parallel: true      (the two halves at the same time)
      splits:
          - {name: LFMF,  bands: [band1, band2, band3, band4],
             groups: [s11, s01]}
          - {name: MFUHF, bands: [band3, band4, band5, band6],
             groups: [d11, d01]}
First order only (the higher orders are searched by the tree on the full
data), and no T_d moments on MF+UHF: there they are degenerate with beta_d
(the evidence picked T_d x T_d over beta_d x beta_d in all the d10 / d12
skies). Candidates clearly rejected by the ML estimate of ln B
(lnB_approx_skip) get no evidence run.
Output: the fits of each half in <runs_dir>/phase0_<name>/ (result.json:
its model); in the decision log of the tree the nodes phase0_fit,
phase0_pref, phase0_split and phase0_start.
"""
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor

from .fitter import Fitter, NoValidStart
from .log import json_default
from .models import Model, candidate_groups, model_from_label
from .runner import BBCompSepRunner
from .stats import lnB


class Phase0:
    def __init__(self, tc, log):
        self.tc = tc
        self.log = log
        self.cfg = tc['phase0']
        th = tc['thresholds']
        self.lnB_pref = th['lnB_preferred']
        self.lnB_flag = th.get('lnB_flag', 1.0)
        self.extra_requires = tc.get('moments', {}).get('requires', {})
        self.max_rounds = self.cfg.get('max_rounds', 5)
        # candidates clearly worse than the current model already from the
        # ML fits (ln B ~ -Delta chi2 / 2 - occam_per_param * Delta k below
        # lnB_preferred - lnB_approx_skip): no evidence run, as in the tree
        self.approx_skip = th.get('lnB_approx_skip', None)
        self.occam = th.get('occam_per_param', 1.0)
        # the two halves at the same time, each with half of the processes
        self.parallel = self.cfg.get('parallel', True)
        # choices with |ln B - threshold| < lnB_flag: flagged as undecided
        # (added to those of the tree on the full data)
        self.undecided = []

    def fitter(self, split, d):
        """Fitter on the bands of the split, runs in d."""
        tc = copy.deepcopy(self.tc)
        tc['paths']['runs_dir'] = d
        R = BBCompSepRunner(tc, self.log)
        R.base['BBCompSep']['bands'] = list(split['bands'])
        if self.parallel and len(self.cfg['splits']) > 1:
            n = max(1, R.nautilus_procs() // len(self.cfg['splits']))
            R.nautilus_procs = lambda: n
        return Fitter(R)

    def run_split(self, split):
        """Groups kept in one half."""
        name = split['name']
        d = os.path.join(self.tc['paths']['runs_dir'], f"phase0_{name}")
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, 'result.json')
        if os.path.isfile(out):
            res = json.load(open(out))
            self.undecided += res['undecided']
            self.log_split(split, res)
            return set(model_from_label(res['model']).moments)
        F = self.fitter(split, d)
        M = Model()
        fit0 = F.full_fit(M)
        self.log_fit(name, fit0)
        und = []
        for _ in range(self.max_rounds):
            cands = candidate_groups(M, split['groups'], self.extra_requires)
            fits = []
            for g in cands:
                try:
                    ml = F.ml_start(M.plus(g), centers=fit0.best)
                    if self.rejected_by_approx(name, ml, fit0):
                        continue
                    f = F.full_fit(M.plus(g), centers=ml.best)
                except NoValidStart as err:
                    self.log.node('unfittable', f'phase 0 {name}: {g} cannot '
                                  'be fitted with the moment constraints',
                                  'skipped', reason=str(err))
                    continue
                self.log_fit(name, f)
                fits.append(f)
            if not fits:
                break
            best = max(fits, key=lambda f: f.logZ)
            lb, lberr = lnB(best, fit0)
            pref = lb > self.lnB_pref
            if abs(lb - self.lnB_pref) < self.lnB_flag:
                und.append(dict(kind=f'phase 0 {name}',
                                chosen=(best if pref else fit0).model.label,
                                other=(fit0 if pref else best).model.label,
                                lnB=lb, lnB_err=lberr,
                                threshold=self.lnB_pref))
            self.log.node('phase0_pref', f'phase 0 {name}: is the best '
                          f'candidate ({best.model.label}) preferred over '
                          f'{M.label}?', 'Yes' if pref else 'No', lnB=lb,
                          lnB_err=lberr, split=name,
                          candidates={f.model.label: f.logZ for f in fits},
                          lnB_preferred=self.lnB_pref)
            if not pref:
                break
            M, fit0 = best.model, best
        res = dict(model=M.label, bands=split['bands'], undecided=und,
                   r_med=fit0.r_summary()['r_med'], logZ=fit0.logZ)
        with open(out, 'w') as f:
            json.dump(res, f, indent=2, default=json_default)
        self.undecided += und
        self.log_split(split, res)
        return set(M.moments)

    def rejected_by_approx(self, name, ml, fit0):
        """True (logged) if the ML estimate of ln B(candidate vs current)
        is below lnB_preferred - lnB_approx_skip."""
        if self.approx_skip is None:
            return False
        lb = (-0.5 * (ml.chi2 - fit0.chi2)
              - self.occam * (ml.nfree - fit0.nfree))
        if lb >= self.lnB_pref - self.approx_skip:
            return False
        self.log.node('phase0_pref', f'phase 0 {name}: is {ml.model.label} '
                      f'preferred over {fit0.model.label}?', 'No', lnB=lb,
                      method='ML estimate', split=name,
                      lnB_approx_skip=self.approx_skip)
        return True

    def log_fit(self, name, f):
        s = f.summary()
        self.log.node('phase0_fit', f'phase 0 {name}: fit of '
                      f'{f.model.label}', 'ok', split=name,
                      **{k: s[k] for k in ('model', 'chi2', 'logZ', 'logZerr',
                                           'r_med', 'r_std') if k in s})

    def log_split(self, split, res):
        self.log.node('phase0_split', f"phase 0 {split['name']} (bands "
                      f"{', '.join(split['bands'])}): model kept",
                      res['model'], split=split['name'],
                      undecided=bool(res['undecided']))

    def run(self):
        """Starting model of the tree on the full data (r free)."""
        self.log.info("==> Phase 0: moments searched on the two frequency "
                      "halves")
        groups = set()
        for split in self.cfg['splits']:
            self.log.info(f"==> Phase 0, {split['name']}: bands "
                          f"{split['bands']}, groups {split['groups']}")
        if self.parallel:
            with ThreadPoolExecutor(len(self.cfg['splits'])) as ex:
                for gs in ex.map(self.run_split, self.cfg['splits']):
                    groups |= gs
        else:
            for split in self.cfg['splits']:
                groups |= self.run_split(split)
        M = Model(frozenset(groups), r_free=True)
        self.log.node('phase0_start', 'phase 0: starting model of the tree',
                      M.label, groups=sorted(groups))
        return M
