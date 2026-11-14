#!/usr/bin/env python
"""
Test of the logic of the decision tree without running BBCompSep.

A MockFitter returns made-up chi2 / evidences / posteriors of r / residual
tests for each scenario, and the tree must follow the right branch of the
flowchart. It also checks the moment groups and their requirements, the
residual tests and that the generated configs are consistent with BBPower's
ParameterManager (if importable).

Usage:  python tests/test_decision_tree.py
"""
import os
import tempfile

import numpy as np

from types import SimpleNamespace

from bbsure import config, fitter, log, models, stats, tree
from bbsure.paths import load_yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# all the names of the tree modules in one namespace
dt = SimpleNamespace(**{k: v for m in (models, stats, config, fitter, log,
                                       tree)
                        for k, v in vars(m).items() if not k.startswith('__')})

NDATA = 315
rng = np.random.default_rng(1)
RES_WR = ['d11', 's11', 'd01', 's01']
RES_R0 = ['d11', 's11', 'dT11', 'd01', 's01', 'dT01', 'd22', 's22', 'd12',
          's12', 'd02', 's02', 'ds01', 'ds10', 'ds11']


class MockFitter:
    """
    scen: dict with
      chi2[label]   chi2 of the best fit of each model (default: good)
      logZ[label]   evidence (default: -chi2/2 - 0.5*nfree)
      r[label]      (mean, sigma) of the posterior of r
      resid[(base_label, group)]  Delta chi2 of the residual test (groups
                    not listed: 0, not detected)
      logZerr[label]  error of the evidence (default 0.1)
      reruns[label]   logZ of the following runs with other seeds
      unfittable    labels whose ML fit has no valid starting point
    """

    def __init__(self, scen):
        self.s = scen
        self.calls = []
        self.residual_test_kind = scen.get('ic', 'aic')

    ic_penalty = dt.Fitter.ic_penalty
    ic_name = dt.Fitter.ic_name

    def _fit(self, model):
        lab = model.label
        nfree = 8 + model.n_moment_params() + (1 if model.r_free else 0)
        chi2 = self.s.get('chi2', {}).get(lab, NDATA - nfree)
        logZ = self.s.get('logZ', {}).get(lab, -0.5 * chi2 - 0.5 * nfree)
        names = [f"p{i}" for i in range(nfree)]
        if model.r_free:
            names[0] = 'r_tensor'
        err = self.s.get('logZerr', {}).get(lab, 0.1)
        f = dt.FitResult(model, lab, names, {n: 0.1 for n in names}, chi2,
                         NDATA, logZ, err, logZ_runs=[logZ],
                         logZerr_runs=[err])
        if model.r_free:
            mu, sig = self.s.get('r', {}).get(lab, (0.0, 0.003))
            f.r_samples = rng.normal(mu, sig, 20000)
        return f

    def repeat_pc(self, res):
        lab = res.model.label
        self.calls.append(('rerun', lab))
        k = len(res.logZ_runs) - 1
        more = self.s.get('reruns', {}).get(lab, [])
        res.logZ_runs.append(more[k] if k < len(more) else res.logZ_runs[0])
        res.logZerr_runs.append(res.logZerr_runs[0])
        n = len(res.logZ_runs)
        res.logZ = float(np.mean(res.logZ_runs))
        res.logZerr = float(np.sqrt(np.sum(np.square(res.logZerr_runs))) / n)
        return res

    def full_fit(self, model, centers=None):
        self.calls.append(('full', model.label))
        return self._fit(model)

    def ml_fit(self, model, centers=None):
        self.calls.append(('ml', model.label))
        return self._fit(model)

    def ml_start(self, model, centers=None):
        self.calls.append(('ml', model.label))
        if model.label in self.s.get('unfittable', ()):
            raise dt.NoValidStart(model.label)
        return self._fit(model)

    def residual_test(self, base, group):
        self.calls.append(('resid', base.model.label, group))
        d = self.s.get('resid', {}).get((base.model.label, group), 0.)
        k = len(set(base.model.plus(group).amplitudes())
                - set(base.model.amplitudes()))
        thr = self.ic_penalty(k, base)
        return dt.ResidualResult(group, d > thr, dchi2=d, thr=thr,
                                 rank=thr - d)


def tree_cfg(runs_dir, seed_max_runs=1, lnB_preferred=0., approx_skip=None,
             initial_model=None):
    return dict(paths=dict(runs_dir=runs_dir),
                thresholds=dict(pte_min=0.01, lnB_preferred=lnB_preferred,
                                seed_band=2.0, seed_max_runs=seed_max_runs,
                                lnB_approx_skip=approx_skip),
                moments=dict(res_wr=RES_WR, res_r0=RES_R0),
                run=dict(max_iterations=6, fit_r0_at_end=False,
                         initial_model=initial_model))


def run(scen, **kw):
    d = tempfile.mkdtemp()
    lg = dt.DecisionLog(d, verbose=False)
    F = MockFitter(scen)
    res = dt.DecisionTree(tree_cfg(d, **kw), F, lg).run()
    path = [(e['node'], e['answer']) for e in lg.events]
    return res, path, F, lg.events


M0 = 'M_none_r'
R2 = (0.006, 0.003)            # r at 2 sigma: the search starts
BAD = -1000.                   # very low evidence


def check(name, cond):
    print(f"  {'OK ' if cond else 'FAIL'} {name}")
    if not cond:
        raise SystemExit(1)


def resid_calls(F, base):
    return [c[2] for c in F.calls if c[0] == 'resid' and c[1] == base]


def main():
    print("Flowchart scenarios:")

    res, path, _, _ = run({})
    check("chi2 ok + r ok at 1 sigma -> upper limit",
          res['conclusion'] == 'upper_limit' and res['model'] == M0
          and ('r1sigma', 'Yes') in path and not res['low_pte'])

    from scipy import stats as st
    res, path, _, _ = run({'chi2': {M0: float(st.chi2.isf(0.02, NDATA - 9))}})
    check("PTE 0.02: accepted as good, but low PTE flagged",
          ('chi2', 'Yes') in path and res['low_pte'])

    res, path, F, _ = run({'r': {M0: R2}})
    check("nothing in the residuals -> 3 sigma -> upper limit; only the "
          "groups whose requirements are met are tested",
          res['conclusion'] == 'upper_limit' and ('r3sigma', 'Yes') in path
          and resid_calls(F, M0) == ['d11', 's11']
          and resid_calls(F, 'M_none_0') == ['d11', 's11', 'dT11'])

    res, path, _, _ = run({'r': {M0: (0.015, 0.003)}})
    check("r at 5 sigma, no moments -> r detection",
          res['conclusion'] == 'r_detection')

    # Res_wr: d11 detected and preferred -> new M_0,r; then its requirements
    # open d01, d22, ds10 ... and nothing else is found
    res, path, F, _ = run({'r': {M0: R2, 'M_d11_r': R2},
                           'resid': {(M0, 'd11'): 10.},
                           'logZ': {'M_d11_r': 0.}})
    check("Res_wr: d11 preferred -> M_0,r = M_d11_r",
          ('replace', 'M_d11_r') in path and res['model'] == 'M_d11_r')
    check("  next Res_wr: s11 and d01 (d01 needs d11)",
          resid_calls(F, 'M_d11_r') == ['s11', 'd01'])
    check("  next Res_r0: second order and cross terms of d11 open up",
          resid_calls(F, 'M_d11_0') == ['s11', 'dT11', 'd01', 'd22', 'ds10'])
    check("  no pruning, no emcee screening",
          not any(n.startswith('prune') for n, _ in path)
          and all(c[0] in ('ml', 'full', 'resid') for c in F.calls))

    # Res_wr candidate not preferred -> Res_r0: s11 detected, M_1,r preferred
    # -> replaces M_0,r; the next iteration goes straight to Res_r0
    res, path, F, _ = run({'r': {M0: R2, 'M_s11_r': R2},
                           'resid': {(M0, 'd11'): 5., ('M_none_0', 's11'): 5.},
                           'logZ': {'M_d11_r': BAD, 'M_s11_r': 0.}})
    check("Res_wr not preferred -> Res_r0: M_1,r preferred -> replace",
          ('replace', 'M_s11_r') in path and res['model'] == 'M_s11_r')
    check("  after a replacement from Res_r0 no Res_wr",
          resid_calls(F, 'M_s11_r') == [])

    # Res_r0: M_1,r not preferred, M_1,0 preferred -> stop, M_1 to the
    # profile scheme, 3 sigma test on M_0,r
    res, path, F, ev = run({'r': {M0: R2},
                            'resid': {('M_none_0', 's11'): 5.},
                            'logZ': {'M_s11_r': BAD, 'M_s11_0': 0.}})
    chain = [e for e in ev if e['node'] == 'r0chain']
    check("Res_r0: M_1,0 preferred -> m10, 3 sigma test on M_none_r",
          ('m10', 'M_s11_r') in path and res['model'] == M0
          and chain and chain[-1]['families'] == ['s11']
          and chain[-1]['answer'] == 'Yes')

    # chain: first round s11 (d11, dT11 not detected), nothing preferred ->
    # residuals of M_s11_0, where d11 and dT11 are not retested; s22 found,
    # M_s11+s22_0 preferred
    res, path, F, ev = run({'r': {M0: R2},
                            'resid': {('M_none_0', 's11'): 5.,
                                      ('M_s11_0', 's22'): 5.},
                            'logZ': {'M_s11_r': BAD, 'M_s11_0': BAD,
                                     'M_s11+s22_r': BAD,
                                     'M_s11+s22_0': 0.}})
    chain = [e for e in ev if e['node'] == 'r0chain']
    check("Res_r0 chain: second round on the residuals of M_s11_0",
          ('m10', 'M_s11+s22_r') in path
          and chain[-1]['families'] == ['s11', 's22'])
    check("  groups not detected in the first round are not retested",
          not {'d11', 'dT11'} & set(resid_calls(F, 'M_s11_0'))
          and {'s01', 's22'} <= set(resid_calls(F, 'M_s11_0')))

    # chain with nothing preferred -> r0chain No, 3 sigma
    res, path, F, ev = run({'r': {M0: R2},
                            'resid': {('M_none_0', 's11'): 5.},
                            'logZ': {'M_s11_r': BAD, 'M_s11_0': BAD}})
    check("Res_r0: nothing preferred -> 3 sigma test on M_0,r",
          res['model'] == M0 and ('r3sigma', 'Yes') in path
          and not any(n == 'm10' for n, _ in path))

    # order in Res_r0 by Delta AIC with the ML fits with r = 0: s11 (larger
    # chi2 gain) before d11
    res, path, F, ev = run({'r': {M0: R2},
                            'resid': {('M_none_0', 'd11'): 3.,
                                      ('M_none_0', 's11'): 3.},
                            'chi2': {'M_none_0': 310., 'M_d11_0': 306.,
                                     'M_s11_0': 300.},
                            'logZ': {'M_d11_r': BAD, 'M_s11_r': BAD,
                                     'M_d11_0': BAD, 'M_s11_0': BAD}})
    order = [e for e in ev if e['node'] == 'r0order'][0]['order']
    check("Res_r0: order by Delta AIC (s11 -8, d11 -2)",
          [o[0] for o in order] == ['s11', 'd11']
          and abs(order[0][1] + 8.) < 1e-9)

    # ML estimate of ln B clearly below the threshold -> no evidence run
    res, path, F, ev = run({'r': {M0: R2}, 'resid': {(M0, 'd11'): 5.},
                            'chi2': {'M_d11_r': 400.}}, approx_skip=5.)
    pref = [e for e in ev if e['node'] == 'pref']
    check("ML estimate clearly below the threshold -> rejected without the "
          "evidence run", pref and pref[0].get('method') == 'ML estimate'
          and ('full', 'M_d11_r') not in F.calls)

    # borderline: |ln B| < 2 errors -> reruns; |ln B| < 1 -> undecided flag
    res, path, F, _ = run({'r': {M0: R2, 'M_d11_r': R2},
                           'resid': {(M0, 'd11'): 5.},
                           'logZ': {M0: 0., 'M_d11_r': 0.1},
                           'logZerr': {M0: 0.1, 'M_d11_r': 0.1},
                           'reruns': {'M_d11_r': [0.5, 0.5]}},
                          seed_max_runs=3)
    check("borderline ln B: evidence runs repeated, choice flagged ≈",
          ('rerun', 'M_d11_r') in F.calls and res['undecided']
          and res['undecided_choices'][0]['kind'] == 'M_1,r vs M_0,r')

    # unfittable candidate: skipped
    res, path, F, _ = run({'r': {M0: R2}, 'resid': {(M0, 'd11'): 5.},
                           'unfittable': ['M_d11_r']})
    check("candidate with no valid starting point -> skipped",
          ('unfittable', 'skipped') in path
          and res['conclusion'] == 'upper_limit')

    # start from a given model
    res, path, F, _ = run({'r': {'M_d11+s11_r': R2}},
                          initial_model='M_d11+s11_r')
    check("run: initial_model: the tree starts from M_d11+s11_r",
          F.calls[0] == ('full', 'M_d11+s11_r')
          and resid_calls(F, 'M_d11+s11_r') == ['d01', 's01'])

    # always preferred -> stops at max_iterations
    always = {lab: 100. * i for i, lab in enumerate(
        ['M_d11_r', 'M_d01+d11_r', 'M_d01+d11+s11_r',
         'M_d01+d11+s01+s11_r'], 1)}
    resid = {(M0, 'd11'): 5., ('M_d11_r', 'd01'): 5.,
             ('M_d01+d11_r', 's11'): 5., ('M_d01+d11+s11_r', 's01'): 5.}
    res, path, F, _ = run({'r': {k: R2 for k in [M0] + list(always)},
                           'resid': resid, 'logZ': always})
    check("one spectrum at a time up to M_d01+d11+s01+s11_r",
          [a for n, a in path if n == 'replace']
          == ['M_d11_r', 'M_d01+d11_r', 'M_d01+d11+s11_r',
              'M_d01+d11+s01+s11_r'])

    print("\nMoment groups:")
    test_models()
    print("\nResidual tests:")
    test_residuals()
    print("\nGenerated BBCompSep configs:")
    test_configs()
    print("\nNested ML fits:")
    test_nested_ml()
    print("\nAll tests passed.")


def test_models():
    m = dt.model_from_label('M_d01+d11+s11_0')
    check("labels: M_d01+d11+s11_0 <-> Model",
          m.label == 'M_d01+d11+s11_0' and not m.r_free
          and m.moments == {'d01', 'd11', 's11'})
    try:
        dt.model_from_label('M_d+s-d_w00w10_r')
        ok = False
    except ValueError:
        ok = True
    check("  family labels of the old trees are rejected", ok)
    check("one spectrum per group",
          all(len(G['spectra']) == 1 for G in dt.MOMENT_GROUPS.values()))
    check("requirements: d01 needs d11, d12 needs d11 and d22, ds11 needs "
          "d11 and s11",
          dt.requirements('d01') == ['d11']
          and set(dt.requirements('d12')) == {'d11', 'd22'}
          and set(dt.requirements('ds11')) == {'d11', 's11'})
    none = dt.Model()
    check("  candidates of M_none: only groups without requirements",
          dt.candidate_groups(none, RES_R0) == ['d11', 's11', 'dT11'])
    check("  requirements_met",
          dt.requirements_met(dt.Model(frozenset({'d11', 'd01'})))
          and not dt.requirements_met(dt.Model(frozenset({'d01'}))))
    check("beta_d x T_d only with d11 and dT11",
          ('component_1', 'd', 'w10w01') in dt.group_spectra({'d11', 'dT11'})
          and ('component_1', 'd', 'w10w01')
          not in dt.group_spectra({'dT11'})
          and dt.group_amplitudes('dT11') == ['A_d_w01w01', 'A_d_w10w01'])


class FakeRunner:
    base = load_yaml(os.path.join(REPO, 'configs', 'base_config_tree.yml'))

    def __init__(self, chi2):
        self.chi2, self.calls = chi2, []

    def ml(self, model, tag, fixed=None, centers=None):
        self.calls.append(tag)
        return dt.FitResult(model, tag, ['A'], {'A': 0.5}, self.chi2[tag],
                            NDATA)


def test_residuals():
    base = dt.FitResult(dt.Model(frozenset(), False), 'b', ['p'], {'p': 0.},
                        400., NDATA)
    tag = 'resid_M_none_0_plus_'
    F = dt.Fitter(FakeRunner({tag + 'd11_ML': 398.1, tag + 's11_ML': 397.9}),
                  residual_test='aic')
    d, s_ = (F.residual_test(base, g) for g in ('d11', 's11'))
    check("AIC, k = 1: Delta chi2 = 1.9 < 2 not detected, 2.1 > 2 detected",
          not d.detected and s_.detected and abs(d.thr - 2.) < 1e-9)
    base_d = dt.FitResult(dt.Model(frozenset({'d11'}), False), 'b', ['p'],
                          {'p': 0.}, 400., NDATA)
    tag = 'resid_M_d11_0_plus_'
    F = dt.Fitter(FakeRunner({tag + 'dT11_ML': 396.5}), residual_test='aic')
    r = F.residual_test(base_d, 'dT11')
    check("  dT11 with d11 there: k = 2 (T_d x T_d and beta_d x T_d), "
          "threshold 4", not r.detected and abs(r.thr - 4.) < 1e-9)
    F = dt.Fitter(FakeRunner({'resid_M_none_0_plus_d11_ML': 400. - 8.}),
                  residual_test='bic')
    r = F.residual_test(base, 'd11')
    check("BIC: threshold ln N", r.detected
          and abs(r.thr - np.log(NDATA)) < 1e-9)
    F = dt.Fitter(FakeRunner({'resid_M_none_0_plus_d11_ML': 399.}),
                  residual_test='lrt', lrt_p=0.3173)
    r = F.residual_test(base, 'd11')
    check("LRT: Delta chi2 = 1, k = 1: p = 0.317 -> not detected",
          not r.detected and abs(r.pvalue - 0.3173) < 1e-3)
    try:
        dt.Fitter(FakeRunner({}), residual_test='bayes')
        ok = False
    except ValueError:
        ok = True
    check("residual_test: bayes no longer exists", ok)


def test_configs():
    base = load_yaml(os.path.join(REPO, 'configs', 'base_config_tree.yml'))
    try:
        from bbpower.param_manager import ParameterManager
    except Exception:
        ParameterManager = None
        print("  (bbpower not importable: skipping the ParameterManager "
              "check)")

    for mom in [frozenset(), {'d11'}, {'s11'}, {'dT11'}, {'d11', 'dT11'},
                {'d11', 'd01', 'd22'}, {'d11', 's11', 'ds11'},
                {'s11', 'ds01'}, {'s11', 's22', 's12', 's02'}]:
        for rf in [True, False]:
            m = dt.Model(frozenset(mom), rf)
            cfg = dt.build_config(base, m, 'maximum_likelihood')
            cs = cfg['BBCompSep']
            names = dt.free_param_names(cs)
            amps = sorted(n for n in names if n.startswith('A_'))
            want = sorted(dt.amp_name(p, s) for _, p, s in
                          dt.group_spectra(mom))
            check(f"{m.label}: free amplitudes {amps}",
                  ('r_tensor' in names) == rf and amps == want
                  and ('temp_d' in names) == bool({'dT11', 'dT01'} & mom)
                  and all(('alpha' + a[1:]) in names for a in amps))
            if ParameterManager:
                pm = ParameterManager(cs)
                check("  same order as ParameterManager",
                      pm.p_free_names == names)

    # residuals: everything fixed except the group and its beta
    m = dt.Model(frozenset({'d11'}), False)
    fixed = {n: 0.5 for n in ['amp_d_bb', 'alpha_d_bb', 'beta_s',
                              'epsilon_ds', 'amp_s_bb', 'alpha_s_bb']}
    fixed['r_tensor'] = 0.003
    cs = dt.build_config(base, m, 'emcee', fixed=fixed)['BBCompSep']
    free = dt.free_param_names(cs)
    check(f"residuals: only the group and its beta free ({free})",
          sorted(free) == sorted(['beta_d', 'A_d_w10w10', 'alpha_d_w10w10'])
          and cs['cmb_model']['params']['r_tensor'][2] == [0.003])
    check("  the base config is not modified",
          base['BBCompSep']['cmb_model']['params']['r_tensor'][1] == 'tophat')


def test_nested_ml():
    """A fit worse than the model it contains is redone from its best fit."""
    small = dt.Model(frozenset({'d11'}), True)
    big = small.plus('s11')

    class R:
        def __init__(self):
            self.tags = []

        def ml(self, model, tag, fixed=None, centers=None):
            self.tags.append(tag)
            if model == small:
                chi2 = 520.
            elif tag.endswith('_ML'):
                chi2 = 600.                  # stuck minimizer
            else:
                chi2 = (515. if centers and centers.get('A_s_w10w10') == 0.
                        else 600.)
            return dt.FitResult(model, tag, ['p'], {'p': 1.}, chi2, NDATA)

    r = R()
    F = dt.Fitter(r)
    F.ml_fit(small)
    res = F.ml_fit(big)
    check("stuck fit of the bigger model redone from the smaller one",
          res.chi2 == 515. and res.tag.endswith('_MLn1'))
    r2 = R()
    F2 = dt.Fitter(r2)
    F2.ml_fit(big)
    check("  no smaller model fitted: nothing redone",
          r2.tags == [big.label + '_ML'])
    check("  contains(): r = 0 model inside the r-free one, not the opposite",
          dt.contains(big, big.with_r(False))
          and not dt.contains(big.with_r(False), big)
          and dt.contains(big, small) and not dt.contains(small, big))


if __name__ == '__main__':
    main()
