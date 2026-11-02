#!/usr/bin/env python
"""
Test of the logic of the profile scheme without running BBCompSep.

A MockRunner replaces BBCompSepRunner: the chi2 of each model is a parabola
in r, chi2(r) = chi2_min + ((r - r_min) / 0.003)^2, and the local evidences
are given per scenario. A fake concluded tree (the files the scheme reads,
with the families detected in its last Res_r0) is written in a temporary
directory, and the scheme must compare the right models at their own r and
take r from the profile of the winner.

Usage:  python tests/test_profile_scheme.py
"""
import json
import os
import tempfile

import numpy as np
import yaml

import bbsure.profile_scheme as psm
from bbsure.config import build_config, free_param_names
from bbsure.models import model_from_label
from bbsure.paths import load_yaml
from bbsure.stats import FitResult

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = load_yaml(os.path.join(REPO, 'configs', 'base_config_tree.yml'))
NDATA = 455
SIG_R = 0.003


class MockRunner:
    """
    scen: dict with
      r_min[label]   minimum of the profile of each model (r-free label)
      logz[label]    local evidence of each model at its own r (error 0.1)
    """

    def __init__(self, scen):
        self.s = scen
        self.base = BASE
        self.calls = []

    def _names(self, model, sampler, fixed=None, free_temp=False):
        cfg = build_config(self.base, model, sampler, fixed,
                           free_temp=free_temp)
        return free_param_names(cfg['BBCompSep'])

    def ml(self, model, tag, fixed=None, centers=None):
        lab = model.with_r(True).label
        r = (fixed or {}).get('r_tensor', 0.)
        chi2 = 450. + ((r - self.s['r_min'][lab]) / SIG_R) ** 2
        names = self._names(model, 'maximum_likelihood', fixed)
        self.calls.append(('ml', lab, round(r, 6)))
        return FitResult(model, tag, names, {n: 0.1 for n in names}, chi2,
                         NDATA)

    def polychord(self, model, tag, fixed=None, centers=None,
                  free_temp=False):
        lab = model.with_r(True).label
        names = self._names(model, 'polychord', fixed, free_temp)
        self.calls.append(('pc', lab, tuple(sorted(names)), tag))
        samp = np.full([400, len(names)], 0.1)
        return self.s['logz'][lab], 0.1, names, samp, np.full(400, 450.)

    def emcee(self, model, tag, fixed=None, centers=None):
        names = self._names(model, 'emcee', fixed)
        return names, 0.1 + 0.01 * np.random.randn(2000, len(names))


def fake_tree(d, model, res_r0=None, replace_after=False, m10=False,
              base=None):
    """Files of a concluded tree read by the scheme: final model `model`,
    last Res_r0 with res_r0 = [(group, pruned spectra)] detected."""
    M = model_from_label(model)
    ev = []
    if res_r0 is not None:
        ev.append(dict(node='resid', question='Res_r0: is there any moment '
                       'in the residuals?', answer='Yes' if res_r0 else 'No',
                       base=base or M.with_r(False).label,
                       detected=[g for g, _ in res_r0],
                       pruned=[list(off) for _, off in res_r0]))
    if m10:
        ev.append(dict(node='m10', question='M_1,0 preferred over M_0,r: '
                       'upper limit with M_1,r', answer=model))
    if replace_after:
        ev.append(dict(node='replace', question='M_0,r = M_1,r (from Res_r0)',
                       answer=model))
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


def run(scen, model, res_r0=None, replace_after=False, seed_max_runs=1,
        m10=False, base=None):
    tree = tempfile.mkdtemp()
    fake_tree(tree, model, res_r0, replace_after, m10, base)
    tc = dict(paths=dict(runs_dir=tree, base_config='unused'),
              thresholds=dict(hpd_frac=0.68, seed_band=2.0,
                              seed_max_runs=seed_max_runs))
    orig = psm.BBCompSepRunner
    psm.BBCompSepRunner = lambda tc, log: MockRunner(scen)
    try:
        ps = psm.ProfileScheme(tc, tree, os.path.join(tree, 'profile'),
                               n=21, nproc=1)
    finally:
        psm.BBCompSepRunner = orig
    res = ps.run()
    return res, ps.R


def check(name, cond):
    print(f"  {'OK ' if cond else 'FAIL'} {name}")
    if not cond:
        raise SystemExit(1)


def close(a, b, tol=2e-4):
    return abs(a - b) < tol


def main():
    np.random.seed(3)
    print("Profile scheme scenarios:")

    # 1. the tree concluded before Res_r0: no candidates, profile of M*
    res, R = run({'r_min': {'M_d_r': 0.010}}, 'M_d_r')
    check("no Res_r0: no candidates, result from the profile of M*",
          res['model_final'] == 'M_d_r' and res['candidates'] == []
          and close(res['r_star_final'], 0.010)
          and not any(c[0] == 'pc' for c in R.calls))
    check("  r detection at 0.010 +- 0.003",
          res['conclusion'] == 'r_detection'
          and close(res['r_med_pl'], 0.010) and close(res['r_std_pl'], SIG_R))

    # 2. Res_r0 followed by an adoption: its families are not candidates
    res, R = run({'r_min': {'M_d_r': 0.010}}, 'M_d_r', [('s', [])],
                 replace_after=True)
    check("Res_r0 before an adoption: no candidates", res['candidates'] == [])

    # 3. candidates d and s; M* has the highest evidence -> final M*
    scen = {'r_min': {'M_none_r': 0.010, 'M_d_r': 0.006, 'M_s_r': 0.004,
                      'M_d+s_r': 0.003},
            'logz': {'M_none_r': -10., 'M_d_r': -11., 'M_s_r': -12.,
                     'M_d+s_r': -13.}}
    res, R = run(scen, 'M_none_r', [('d', []), ('s', [])])
    check("candidates d, s, d+s; M* wins -> final M*, all rejected",
          res['model_final'] == 'M_none_r'
          and res['rejected'] == ['d', 'd+s', 's']
          and close(res['r_star_final'], 0.010))
    check("  every model at its own r",
          ('ml', 'M_d_r', 0.006) in R.calls
          and ('ml', 'M_s_r', 0.004) in R.calls
          and ('ml', 'M_none_r', 0.01) in R.calls)

    # 4. s has the highest evidence -> final M* + s, r from its profile
    scen['logz'] = {'M_none_r': -10., 'M_d_r': -9., 'M_s_r': -8.,
                    'M_d+s_r': -8.5}
    res, R = run(scen, 'M_none_r', [('d', []), ('s', [])])
    check("s wins -> final M* + s, r from its profile",
          res['model_final'] == 'M_s_r' and res['rejected'] == ['d', 'd+s']
          and close(res['r_star_final'], 0.004)
          and res['conclusion'] == 'upper_limit')

    # 5. spectral parameters in the local evidences: beta_d and beta_s in
    #    all the models, T_d only in the one with the dT moments
    scen = {'r_min': {'M_none_r': 0.010, 'M_d_r': 0.006, 'M_dT_r': 0.008,
                      'M_d+dT_r': 0.005},
            'logz': {'M_none_r': -10., 'M_d_r': -11., 'M_dT_r': -12.,
                     'M_d+dT_r': -13.}}
    res, R = run(scen, 'M_none_r', [('d', []), ('dT', [])])
    free = {c[1]: set(c[2]) for c in R.calls if c[0] == 'pc'}
    check("  beta_d, beta_s free in all; T_d only with dT",
          len(free) == 4
          and all({'beta_d', 'beta_s'} <= f for f in free.values())
          and 'temp_d' in free['M_dT_r'] and 'temp_d' not in free['M_d_r']
          and 'temp_d' not in free['M_none_r']
          and 'r_tensor' not in set().union(*free.values()))

    # 6. the spectra pruned by the residual screening stay out
    scen = {'r_min': {'M_d_r': 0.010, 'M_d+s-s_w00w10_r': 0.005},
            'logz': {'M_d_r': -10., 'M_d+s-s_w00w10_r': -9.}}
    res, R = run(scen, 'M_d_r', [('s', ['A_s_w00w10'])])
    check("candidate with a pruned spectrum: M* + s without it",
          res['model_final'] == 'M_d+s-s_w00w10_r')

    # 7. the two best models within 2 errors (ln B = 0.1 +- 0.14): their
    #    local evidences are repeated with other seeds, up to 3 runs each
    scen = {'r_min': {'M_d_r': 0.010, 'M_d+s_r': 0.006},
            'logz': {'M_d_r': -10., 'M_d+s_r': -10.1}}
    res, R = run(scen, 'M_d_r', [('s', [])], seed_max_runs=3)
    reps = [c[3] for c in R.calls if c[0] == 'pc' and '_rep' in c[3]]
    check("close evidences: PolyChord repeated with other seeds (2 + 2)",
          len(reps) == 4 and res['comparison']['M_d_r']['logZ_runs']
          == [-10.] * 3)
    res, R = run(scen, 'M_d_r', [('s', [])])
    check("  no repetitions with seed_max_runs = 1",
          not any('_rep' in c[3] for c in R.calls if c[0] == 'pc'))

    # 8. the tree chose M_1 = M_0,r + d + s (M_0,r = M_none) because M_1,0
    #    was preferred: every model in between, down to M_none, is tried
    scen = {'r_min': {'M_d+s_r': 0.002, 'M_d_r': 0.008, 'M_s_r': 0.006,
                      'M_none_r': 0.012},
            'logz': {'M_d+s_r': -10., 'M_d_r': -9., 'M_s_r': -11.,
                     'M_none_r': -12.}}
    res, R = run(scen, 'M_d+s_r', [('d', []), ('s', [])], m10=True,
                 base='M_none_0')
    check("M_1,0 in the tree: models between M_1 and M_0,r, M_d wins",
          sorted(res['candidates']) == ['-d', '-d+s', '-s']
          and res['model_final'] == 'M_d_r'
          and res['start'].startswith('families removed')
          and close(res['r_star_final'], 0.008))

    # 9. additions: all the combinations of the Res_r0 families
    scen = {'r_min': {'M_none_r': 0.010, 'M_d_r': 0.008, 'M_s_r': 0.007,
                      'M_d+s_r': 0.004},
            'logz': {'M_none_r': -10., 'M_d_r': -11., 'M_s_r': -11.,
                     'M_d+s_r': -9.}}
    res, R = run(scen, 'M_none_r', [('d', []), ('s', [])])
    check("additions: d, s and d+s tried; d+s wins",
          res['candidates'] == ['d', 's', 'd+s']
          and res['model_final'] == 'M_d+s_r')
    print("\nAll tests passed.")


if __name__ == '__main__':
    main()
