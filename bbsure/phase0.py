"""
Phase 0: frequency-split screening that sets the starting model of the tree.

The data are split in two halves, each fitted with BBCompSep on a subset of
the bands (config option `bands`):
  LF+MF  (e.g. band1-4): only the synchrotron moments are searched, the dust
         has no moments;
  MF+UHF (e.g. band3-6): only the dust moments are searched, the
         synchrotron has no moments.
In each half the Bayesian tree runs as on the full data (tree.py: residual
tests with r free and with r = 0, evidence comparisons, 3 sigma test), with
the moment groups restricted to those of the half and no profile scheme.
The tree on the full data then starts from M_0,r = the groups of the final
models of the two halves (run: initial_model).

Config (tree config):
  phase0:
      enabled: true
      splits:
          - {name: LFMF,  bands: [band1, band2, band3, band4],
             groups: [s11, s01, s22, s12, s02]}
          - {name: MFUHF, bands: [band3, band4, band5, band6],
             groups: [d11, d01, dT11, dT01, d22, d12, d02]}
Output: each half in <runs_dir>/phase0_<name>/ (fits, decision_log,
conclusion.json); in the decision log of the full tree the nodes
phase0_split (final model of each half) and phase0_start.
"""
import copy
import json
import os

from .fitter import Fitter
from .log import DecisionLog, json_default
from .models import Model, model_from_label
from .runner import BBCompSepRunner
from .tree import DecisionTree


class Phase0:
    def __init__(self, tc, log):
        self.tc = tc
        self.log = log
        self.cfg = tc['phase0']
        # choices flagged as undecided in the halves (added to those of the
        # tree on the full data)
        self.undecided = []

    def run_split(self, split):
        """Bayesian tree on the bands of one half: its final model."""
        name = split['name']
        tc = copy.deepcopy(self.tc)
        d = os.path.join(self.tc['paths']['runs_dir'], f"phase0_{name}")
        tc['paths']['runs_dir'] = d
        mc = tc['moments']
        mc['res_wr'] = [g for g in mc['res_wr'] if g in split['groups']]
        mc['res_r0'] = [g for g in mc['res_r0'] if g in split['groups']]
        tc['run']['initial_model'] = None
        tc['run']['fit_r0_at_end'] = False
        os.makedirs(d, exist_ok=True)
        f = os.path.join(d, 'conclusion.json')
        if os.path.isfile(f):
            res = json.load(open(f))
        else:
            log = DecisionLog(d)
            log.info(f"==> Phase 0, {name}: bands {split['bands']}, "
                     f"groups {split['groups']}")
            R = BBCompSepRunner(tc, log)
            R.base['BBCompSep']['bands'] = list(split['bands'])
            F = Fitter(R, residual_test=mc.get('residual_test', 'aic'),
                       lrt_p=mc.get('lrt_p', 0.3173))
            res = DecisionTree(tc, F, log).run()
            with open(f, 'w') as out:
                json.dump(res, out, indent=2, default=json_default)
        for u in res.get('undecided_choices', []):
            self.undecided.append(dict(u, kind=f"phase 0 {name}: {u['kind']}"))
        self.log.node('phase0_split', f'phase 0 {name} (bands '
                      f"{', '.join(split['bands'])}): final model of the "
                      'Bayesian tree', res['model'], split=name,
                      conclusion=res['conclusion'], r_med=res.get('r_med'),
                      undecided=res.get('undecided', False))
        return set(model_from_label(res['model']).moments)

    def run(self):
        """Starting model of the tree on the full data (r free)."""
        self.log.info("==> Phase 0: Bayesian tree on the two frequency "
                      "halves")
        groups = set()
        for split in self.cfg['splits']:
            groups |= self.run_split(split)
        M = Model(frozenset(groups), r_free=True)
        self.log.node('phase0_start', 'phase 0: starting model of the tree',
                      M.label, groups=sorted(groups))
        return M
