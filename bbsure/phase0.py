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
      splits:
          - {name: LFMF,  bands: [band1, band2, band3, band4],
             groups: [s11, s01, s22, s12, s02]}
          - {name: MFUHF, bands: [band3, band4, band5, band6],
             groups: [d11, d01, dT11, dT01, d22, d12, d02]}
Output: the fits of each half in <runs_dir>/phase0_<name>/ (result.json:
its model); in the decision log of the tree the nodes phase0_fit,
phase0_pref, phase0_split and phase0_start.
"""
import copy
import json
import os

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
        # choices with |ln B - threshold| < lnB_flag: flagged as undecided
        # (added to those of the tree on the full data)
        self.undecided = []

    def fitter(self, split, d):
        """Fitter on the bands of the split, runs in d."""
        tc = copy.deepcopy(self.tc)
        tc['paths']['runs_dir'] = d
        R = BBCompSepRunner(tc, self.log)
        R.base['BBCompSep']['bands'] = list(split['bands'])
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
                    f = F.full_fit(M.plus(g))
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
            groups |= self.run_split(split)
        M = Model(frozenset(groups), r_free=True)
        self.log.node('phase0_start', 'phase 0: starting model of the tree',
                      M.label, groups=sorted(groups))
        return M
