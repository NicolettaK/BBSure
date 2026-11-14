#!/usr/bin/env python
"""
Test of the logic of the profile scheme without running BBCompSep.

A MockRunner replaces BBCompSepRunner: the chi2 of each model is a parabola
in r, chi2(r) = chi2_min + ((r - r_min) / 0.003)^2, and the local evidences
are given per scenario. A fake concluded tree (the files the scheme reads,
with the groups detected in its last Res_r0) is written in a temporary
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


def fake_tree(d, model, chain=None, replace_after=False, m10=False):
    """Files of a concluded tree read by the scheme: M_0,r `model`, groups
    added in the last Res_r0 chain; m10: M_1,0 preferred."""
    M = model_from_label(model)
    ev = []
    if chain is not None:
        ev.append(dict(node='r0chain', question='Res_r0: groups added to '
                       'M_0,r; is M_1,0 preferred?',
                       answer='Yes' if m10 else 'No', base=model,
                       families=list(chain)))
    if m10:
        ev.append(dict(node='m10', question='M_1,0 preferred over M_0,r',
                       answer=model))
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


def run(scen, model, chain=None, replace_after=False, seed_max_runs=1,
        m10=False, candidates=True):
    tree = tempfile.mkdtemp()
    fake_tree(tree, model, chain, replace_after, m10)
    tc = dict(paths=dict(runs_dir=tree, base_config='unused'),
              thresholds=dict(seed_band=2.0, seed_max_runs=seed_max_runs),
              moments=dict(requires={}),
              profile_scheme=dict(candidates=candidates))
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
    res, R = run({'r_min': {'M_d11_r': 0.010}}, 'M_d11_r')
    check("no Res_r0: no candidates, result from the profile of M*",
          res['model_final'] == 'M_d11_r' and res['candidates'] == []
          and close(res['r_star_final'], 0.010)
          and not any(c[0] == 'pc' for c in R.calls))
    check("  r detection at 0.010 +- 0.003",
          res['conclusion'] == 'r_detection'
          and close(res['r_med_pl'], 0.010) and close(res['r_std_pl'], SIG_R))

    # 2. Res_r0 followed by an adoption: its groups are not candidates
    res, R = run({'r_min': {'M_d11_r': 0.010}}, 'M_d11_r', ['s11'],
                 replace_after=True)
    check("Res_r0 before an adoption: no candidates", res['candidates'] == [])

    # 3. candidates d11 and s11; M* has the highest evidence -> final M*
    scen = {'r_min': {'M_none_r': 0.010, 'M_d11_r': 0.006, 'M_s11_r': 0.004,
                      'M_d11+s11_r': 0.003},
            'logz': {'M_none_r': -10., 'M_d11_r': -11., 'M_s11_r': -12.,
                     'M_d11+s11_r': -13.}}
    res, R = run(scen, 'M_none_r', ['d11', 's11'])
    check("candidates d11, s11, d11+s11; M* wins -> final M*, all rejected",
          res['model_final'] == 'M_none_r'
          and res['rejected'] == ['d11', 'd11+s11', 's11']
          and close(res['r_star_final'], 0.010))
    at = lambda lab, r: any(c[0] == 'ml' and c[1] == lab
                            and abs(c[2] - r) < 2e-5 for c in R.calls)
    check("  every model at its own r",
          at('M_d11_r', 0.006) and at('M_s11_r', 0.004)
          and at('M_none_r', 0.01))

    # 3b. same candidates with profile_scheme: candidates: false: no
    #     comparison, only the profile of M*
    scen_s = dict(scen, logz=dict(scen['logz'], **{'M_s11_r': -5.}))
    res, R = run(scen_s, 'M_none_r', ['d11', 's11'],
                 candidates=False)
    check("candidates: false -> only M*, candidates listed as skipped",
          res['model_final'] == 'M_none_r' and res['candidates'] == []
          and res['candidates_skipped'] == ['d11', 's11', 'd11+s11']
          and res['start'] == 'candidates not compared'
          and close(res['r_star_final'], 0.010))
    check("  no profile or local evidence of the candidates",
          not any(c[1] in ('M_d11_r', 'M_s11_r', 'M_d11+s11_r') for c in R.calls
                  if len(c) > 1))

    # 4. s11 has the highest evidence -> final M* + s11, r from its profile
    scen['logz'] = {'M_none_r': -10., 'M_d11_r': -9., 'M_s11_r': -8.,
                    'M_d11+s11_r': -8.5}
    res, R = run(scen, 'M_none_r', ['d11', 's11'])
    check("s11 wins -> final M* + s11, r from its profile",
          res['model_final'] == 'M_s11_r' and res['rejected'] == ['d11', 'd11+s11']
          and close(res['r_star_final'], 0.004)
          and res['conclusion'] == 'upper_limit')

    # 5. spectral parameters in the local evidences: beta_d and beta_s in
    #    all the models, T_d only in the ones with dT11
    scen = {'r_min': {'M_none_r': 0.010, 'M_d11_r': 0.006, 'M_dT11_r': 0.008,
                      'M_d11+dT11_r': 0.005},
            'logz': {'M_none_r': -10., 'M_d11_r': -11., 'M_dT11_r': -12.,
                     'M_d11+dT11_r': -13.}}
    res, R = run(scen, 'M_none_r', ['d11', 'dT11'])
    free = {c[1]: set(c[2]) for c in R.calls if c[0] == 'pc'}
    check("  beta_d, beta_s free in all; T_d only with dT11",
          len(free) == 4
          and all({'beta_d', 'beta_s'} <= f for f in free.values())
          and 'temp_d' in free['M_dT11_r'] and 'temp_d' not in free['M_d11_r']
          and 'temp_d' not in free['M_none_r']
          and 'r_tensor' not in set().union(*free.values()))

    # 6. the two best models within 2 errors (ln B = 0.1 +- 0.14): their
    #    local evidences are repeated with other seeds, up to 3 runs each
    scen = {'r_min': {'M_d11_r': 0.010, 'M_d11+s11_r': 0.006},
            'logz': {'M_d11_r': -10., 'M_d11+s11_r': -10.1}}
    res, R = run(scen, 'M_d11_r', ['s11'], seed_max_runs=3)
    reps = [c[3] for c in R.calls if c[0] == 'pc' and '_rep' in c[3]]
    check("close evidences: evidence runs repeated with other seeds (2 + 2)",
          len(reps) == 4 and res['comparison']['M_d11_r']['logZ_runs']
          == [-10.] * 3)
    res, R = run(scen, 'M_d11_r', ['s11'])
    check("  no repetitions with seed_max_runs = 1",
          not any('_rep' in c[3] for c in R.calls if c[0] == 'pc'))
    check("  the two best still within 2 errors: choice flagged",
          res['undecided'] and res['undecided_choices'][0]['chosen']
          == 'M_d11_r' and res['undecided_choices'][0]['other'] == 'M_d11+s11_r')

    # 7. M_1,0 preferred in the tree: M_0,r = M_none, M_1 =
    #    M_d11+s11; the same candidates (every model in between and M_1)
    scen = {'r_min': {'M_d11+s11_r': 0.002, 'M_d11_r': 0.008, 'M_s11_r': 0.006,
                      'M_none_r': 0.012},
            'logz': {'M_d11+s11_r': -10., 'M_d11_r': -9., 'M_s11_r': -11.,
                     'M_none_r': -12.}}
    res, R = run(scen, 'M_none_r', ['d11', 's11'], m10=True)
    check("M_1,0 preferred in the tree: d11, s11, d11+s11 from M_0,r, M_d11 wins",
          res['candidates'] == ['d11', 's11', 'd11+s11']
          and res['model_final'] == 'M_d11_r'
          and res['start'] == 'M_1,0 preferred in the tree'
          and close(res['r_star_final'], 0.008))

    # 8. M_1,0 never preferred (3 sigma test in the tree): the same
    #    combinations of the groups of the chain
    scen = {'r_min': {'M_none_r': 0.010, 'M_d11_r': 0.008, 'M_s11_r': 0.007,
                      'M_d11+s11_r': 0.004},
            'logz': {'M_none_r': -10., 'M_d11_r': -11., 'M_s11_r': -11.,
                     'M_d11+s11_r': -9.}}
    res, R = run(scen, 'M_none_r', ['d11', 's11'])
    check("M_1,0 not preferred: d11, s11 and d11+s11 tried; d11+s11 wins",
          res['candidates'] == ['d11', 's11', 'd11+s11']
          and res['model_final'] == 'M_d11+s11_r'
          and res['start'] == 'M_1,0 not preferred in the tree')

    # 9. chain s11, then ds01 (requires s11): ds01 alone is not a candidate
    scen = {'r_min': {'M_d11_r': 0.010, 'M_d11+s11_r': 0.007,
                      'M_d11+ds01+s11_r': 0.004},
            'logz': {'M_d11_r': -10., 'M_d11+s11_r': -11., 'M_d11+ds01+s11_r': -12.}}
    res, R = run(scen, 'M_d11_r', ['s11', 'ds01'])
    check("requirements: s11 and s11+ds01, not ds01 alone",
          res['candidates'] == ['s11', 's11+ds01']
          and res['model_final'] == 'M_d11_r')
    print("\nAll tests passed.")


if __name__ == '__main__':
    main()
