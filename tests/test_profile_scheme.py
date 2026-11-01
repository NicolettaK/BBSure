#!/usr/bin/env python
"""
Test of the logic of the profile scheme without running BBCompSep.

A MockRunner replaces BBCompSepRunner: the chi2 of each model is a parabola
in r, chi2(r) = chi2_min + ((r - r_min) / 0.003)^2, the evidences and the
residual detections are given per scenario. A fake concluded tree (the
files the scheme reads) is written in a temporary directory, and the
scheme must follow the right path and take r from the profile of the
right final model.

Usage:  python tests/test_profile_scheme.py
"""
import json
import os
import tempfile

import numpy as np
import yaml

from bbsure.config import build_config, free_param_names
from bbsure.models import model_from_label
from bbsure.paths import load_yaml
from bbsure.profile_scheme import ProfileScheme
from bbsure.stats import FitResult

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = load_yaml(os.path.join(REPO, 'configs', 'base_config_tree.yml'))
NDATA = 455
SIG_R = 0.003


class MockRunner:
    """
    scen: dict with
      r_min[label]     minimum of the profile of each model (r-free label;
                       default r_default)
      chi2_min[label]  chi2 at that minimum (default 450)
      logz_loc[label]  local evidence of a model in the candidate comparison
      resid[group]     (logZ of the group, logZ of the null) if detected in
                       Res_wr* (groups not listed: not detected)
    """

    def __init__(self, scen):
        self.s = scen
        self.base = BASE
        self.calls = []

    def _names(self, model, sampler, fixed=None):
        cfg = build_config(self.base, model, sampler, fixed)
        return free_param_names(cfg['BBCompSep'])

    def ml(self, model, tag, fixed=None, centers=None):
        lab = model.with_r(True).label
        r = (fixed or {}).get('r_tensor', 0.)
        chi2 = (self.s.get('chi2_min', {}).get(lab, 450.)
                + ((r - self.s['r_min'].get(lab, self.s.get('r_default'))) /
                   SIG_R) ** 2)
        names = self._names(model, 'maximum_likelihood', fixed)
        self.calls.append(('ml', lab, round(r, 6)))
        return FitResult(model, tag, names, {n: 0.1 for n in names}, chi2,
                         NDATA)

    def polychord(self, model, tag, fixed=None, centers=None):
        lab = model.with_r(True).label
        names = self._names(model, 'polychord', fixed)
        samp = np.full([400, len(names)], 0.1)
        if tag.startswith('loc_'):
            logZ = self.s['logz_loc'][lab]
        elif tag.startswith('null_'):
            g = tag.split('_')[-1]
            logZ = self.s['resid'][g][1]
        else:                          # residual test of a group
            g = tag.split('_plus_')[1].split('_')[0].split('-')[0]
            logZ = self.s['resid'][g][0]
            for k, n in enumerate(names):
                if n.startswith('A_'):
                    samp[:, k] = 1. + 0.1 * np.random.randn(len(samp))
        self.calls.append(('pc', tag))
        return logZ, 0.1, names, samp, np.full(len(samp), 450.)

    def emcee(self, model, tag, fixed=None, centers=None):
        names = self._names(model, 'emcee', fixed)
        samp = 0.1 + 0.01 * np.random.randn(2000, len(names))
        if '_plus_' in tag:            # residual screening
            g = tag.split('_plus_')[1].split('_')[0]
            det = g in self.s.get('resid', {})
            for k, n in enumerate(names):
                if n.startswith('A_'):
                    samp[:, k] = (1. if det else 0.) + 0.1 * np.random.randn(
                        len(samp))
        self.calls.append(('mc', tag))
        return names, samp


def fake_tree(d, model, cand=None, m10=None):
    """Files of a concluded tree read by the scheme: final model `model`,
    last M_1,r tried `cand`, M_1,0 preferred `m10`."""
    M = model_from_label(model)
    ev = []
    if cand:
        ev.append(dict(node='fit', question=f'fit of M_1,r = {cand}',
                       answer='ok', model=cand))
    if m10:
        ev.append(dict(node='pref', question=f'Is M_1,0 ({m10}) preferred '
                       f'over M_0,r ({model})?', answer='Yes', lnB=3.))
    json.dump(ev, open(os.path.join(d, 'decision_log.json'), 'w'))
    json.dump(dict(conclusion='r_detection', model=model),
              open(os.path.join(d, 'conclusion_tree.json'), 'w'))
    pc = os.path.join(d, f'{model}_PC')
    os.makedirs(os.path.join(pc, 'polychord'))
    cfg = build_config(BASE, M, 'polychord')
    yaml.safe_dump(cfg, open(os.path.join(pc, 'config.yml'), 'w'))
    names = free_param_names(cfg['BBCompSep'])
    samp = np.full([1000, 2 + len(names)], 0.1)
    samp[:, 2 + names.index('r_tensor')] = 0.012 + SIG_R * np.random.randn(1000)
    np.savetxt(os.path.join(pc, 'polychord', 'pch_equal_weights.txt'), samp)
    ml = os.path.join(d, f'{model}_ML')
    os.makedirs(ml)
    np.savez(os.path.join(ml, 'chi2.npz'), names=names,
             params=np.full(len(names), 0.1), chi2=450., ndof=NDATA)


def run(scen, model, cand=None, m10=None):
    tree = tempfile.mkdtemp()
    fake_tree(tree, model, cand, m10)
    tc = dict(paths=dict(runs_dir=tree, base_config='unused'),
              thresholds=dict(hpd_frac=0.68),
              moments=dict(res_r0=['d', 's', 'dT', 'd2', 's2', 'ds1', 'ds2'],
                           requires={'ds1': ['s']}))
    ps = ProfileScheme.__new__(ProfileScheme)
    # ProfileScheme with the mock runner instead of BBCompSepRunner
    import bbsure.profile_scheme as psm
    orig = psm.BBCompSepRunner
    psm.BBCompSepRunner = lambda tc, log: MockRunner(scen)
    try:
        ps.__init__(tc, tree, os.path.join(tree, 'profile'), n=21, nproc=1)
    finally:
        psm.BBCompSepRunner = orig
    res = ps.run()
    path = [(e['node'], e['answer']) for e in ps.log.events]
    return res, path, ps.R


def check(name, cond):
    print(f"  {'OK ' if cond else 'FAIL'} {name}")
    if not cond:
        raise SystemExit(1)


def close(a, b, tol=2e-4):
    return abs(a - b) < tol


def main():
    np.random.seed(3)
    print("Profile scheme scenarios:")

    # 1. no candidate from the tree -> result from the profile of M_0,r
    res, path, R = run({'r_min': {'M_d_r': 0.010}}, 'M_d_r')
    check("no candidate: result from the profile of M_0,r, no Res_wr*",
          res['model_final'] == 'M_d_r' and res['start'] ==
          'M_0,r, no candidate' and close(res['r_star_final'], 0.010)
          and not any(c[0] == 'mc' and '_plus_' in c[1] for c in R.calls))
    check("  r detection at 0.010 +- 0.003",
          res['conclusion'] == 'r_detection'
          and close(res['r_med_pl'], 0.010) and close(res['r_std_pl'], SIG_R))

    # 2. candidate s rejected (ln B < 0) -> stop, profile of M_0,r
    scen = {'r_min': {'M_d_r': 0.010, 'M_d+s_r': 0.004},
            'logz_loc': {'M_d_r': -10., 'M_d+s_r': -11.}}
    res, path, R = run(scen, 'M_d_r', cand='M_d+s_r')
    check("candidate rejected: stop, final model M_0,r",
          res['model_final'] == 'M_d_r' and res['rejected'] == ['s']
          and ('stop', 'M_d_r') in path and close(res['r_star_final'], 0.010))

    # 3. candidate s accepted, then nothing in Res_wr* -> profile of M+s
    scen['logz_loc'] = {'M_d_r': -10., 'M_d+s_r': -8.}
    scen['r_default'] = 0.004
    res, path, R = run(scen, 'M_d_r', cand='M_d+s_r')
    check("candidate accepted: final model M_0,r + s, r from its profile",
          res['model_final'] == 'M_d+s_r' and ('adopt', 'M_d+s_r') in path
          and close(res['r_star_final'], 0.004)
          and ('resid', 'No') in path)
    check("  upper limit (0.004 is within 3 sigma of 0)",
          res['conclusion'] == 'upper_limit' and res['r_95_pl0'] > 0.004)

    # 4. candidate accepted, then dT passes Res_wr* (ln B vs null > 0) and is
    #    the next candidate, accepted too
    scen = {'r_min': {'M_d_r': 0.010, 'M_d+s_r': 0.006, 'M_d+dT+s_r': 0.005},
            'r_default': 0.006,
            'logz_loc': {'M_d_r': -10., 'M_d+s_r': -8., 'M_d+dT+s_r': -7.},
            'resid': {'dT': (-5., -6.)}}
    res, path, R = run(scen, 'M_d_r', cand='M_d+s_r')
    check("Res_wr*: dT detected and better than the null -> new candidate "
          "-> adopted",
          res['model_final'] == 'M_d+dT+s_r'
          and ('adopt', 'M_d+dT+s_r') in path
          and close(res['r_star_final'], 0.005))
    check("  each group tested at its own r (minimum of model + group)",
          ('ml', 'M_d+s_r', 0.005) in R.calls)

    # 5. M_1,0 preferred in the tree -> start from M_1, no comparison
    res, path, R = run({'r_min': {'M_d_r': 0.010, 'M_d+s_r': 0.003},
                        'r_default': 0.003},
                       'M_d_r', cand='M_d+s_r', m10='M_d+s_0')
    check("M_1,0 preferred in the tree: start from M_1 without comparison",
          res['start'] == 'M_1 (M_1,0 preferred)'
          and res['model_final'] == 'M_d+s_r'
          and not any(c[0] == 'pc' and c[1].startswith('loc_')
                      for c in R.calls)
          and close(res['r_star_final'], 0.003))
    print("\nAll tests passed.")


if __name__ == '__main__':
    main()
