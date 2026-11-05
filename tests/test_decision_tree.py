#!/usr/bin/env python
"""
Test of the logic of the decision tree without running BBCompSep.

A MockFitter returns made-up chi2 / evidences / posteriors of r / residual
tests for each scenario, and the tree must follow the right branch of the
flowchart. It also checks that the generated configs are consistent with
BBPower's ParameterManager (if importable).

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


class MockFitter:
    """
    scen: dict with
      chi2[label]  chi2 of the best fit of each model (default: good)
      logZ[label]  evidence (default: -chi2/2 - 0.5*nfree)
      r[label]     (mean, sigma) of the posterior of r
      resid[(base_label, group)]  logZ of the residual test if detected
                   (groups not listed: not detected)
      prof_dchi2   difference between the minima of the profiles (M0 - M1)
      amps[label]  {amplitude: (mean, sigma)} of the posterior of the moment
                   amplitudes (default (1, 0.1): detected)
      amps_full[label]  same, for the full fit (PolyChord) only (default:
                   amps)
      logZerr[label]  PolyChord error (default 0.1)
      reruns[label]   logZ of the following runs with other seeds
                      (default: logZ)
    """

    def __init__(self, scen):
        self.s = scen
        self.calls = []

    def _fit(self, model, full=False):
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
        amps = model.amplitudes()
        spec = self.s.get('amps', {}).get(lab, {})
        if full:
            spec = self.s.get('amps_full', {}).get(lab, spec)
        f.sample_names = amps
        f.samples = np.array([rng.normal(*spec.get(a, (1., 0.1)), 20000)
                              for a in amps]).T.reshape(20000, len(amps))
        return f

    def prior_bounds(self, params):
        return {a: (-100., 100.) for a in params}

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
        return self._fit(model, full=True)

    def ml_fit(self, model, centers=None):
        self.calls.append(('ml', model.label))
        return self._fit(model)

    def screen_fit(self, model, centers=None):
        self.calls.append(('screen', model.label))
        if model.label in self.s.get('unfittable', ()):
            raise dt.NoValidStart(model.label)
        return self._fit(model)

    def residual_test(self, base, group):
        self.calls.append(('resid', base.model.label, group))
        lz = self.s.get('resid', {}).get((base.model.label, group))
        if lz is None:
            return dt.ResidualResult(group, False, {'A': (-1., 1.)},
                                     stage='emcee')
        off = tuple(self.s.get('resid_off', {}).get((base.model.label, group),
                                                    ()))
        return dt.ResidualResult(group, True, {'A': (0.5, 1.)}, lz, 0.1,
                                 {}, stage='polychord', off=off)



def tree_cfg(runs_dir, seed_max_runs=1, lnB_preferred=2.5):
    return dict(paths=dict(runs_dir=runs_dir),
                thresholds=dict(pte_min=0.01, lnB_preferred=lnB_preferred,
                                seed_band=2.0, seed_max_runs=seed_max_runs),
                moments=dict(res_wr=['d', 's'],
                             res_r0=['d', 's', 'dT', 'd2', 's2', 'ds1',
                                     'ds2']),
                run=dict(max_iterations=6))


def run(scen, seed_max_runs=1, lnB_preferred=2.5):
    d = tempfile.mkdtemp()
    log = dt.DecisionLog(d, verbose=False)
    F = MockFitter(scen)
    res = dt.DecisionTree(tree_cfg(d, seed_max_runs, lnB_preferred), F,
                          log).run()
    path = [(e['node'], e['answer']) for e in log.events]
    return res, path, F


M0 = 'M_none_r'
BAD = NDATA - 9 + 120          # chi2 with PTE ~ 0
GOOD = -0.5 * (NDATA - 11) + 20.  # high evidence for the models with moments


def log_events(scen, **kw):
    d = tempfile.mkdtemp()
    log = dt.DecisionLog(d, verbose=False)
    dt.DecisionTree(tree_cfg(d, **kw), MockFitter(scen), log).run()
    return log.events


def check(name, cond):
    print(f"  {'OK ' if cond else 'FAIL'} {name}")
    if not cond:
        raise SystemExit(1)


def resid_calls(F, base):
    return [c[2] for c in F.calls if c[0] == 'resid' and c[1] == base]


def main():
    print("Flowchart scenarios:")

    # 1. good chi2, r ~ 0 -> upper limit right away
    res, path, _ = run({})
    check("chi2 ok + r ok at 1 sigma -> upper limit",
          res['conclusion'] == 'upper_limit' and res['model'] == M0
          and ('r1sigma', 'Yes') in path)

    check("  good chi2: no low-PTE flag", not res['low_pte'])

    # 1b. PTE 0.02: chi2 accepted (> pte_min 0.01) but flagged (< 0.05)
    from scipy import stats as st
    res, path, _ = run({'chi2': {M0: float(st.chi2.isf(0.02, NDATA - 9))}})
    check("PTE 0.02: accepted as good, but low PTE flagged",
          ('chi2', 'Yes') in path and res['low_pte']
          and abs(res['pte'] - 0.02) < 1e-6)

    # 2. r at 2 sigma, nothing in Res_wr nor in Res_r0 -> 3 sigma ok
    res, path, F = run({'r': {M0: (0.006, 0.003)}})
    check("nothing in Res_wr and Res_r0 -> 3 sigma -> upper limit",
          res['conclusion'] == 'upper_limit' and ('r3sigma', 'Yes') in path
          and resid_calls(F, M0) == ['d', 's']
          and resid_calls(F, 'M_none_0') == ['d', 's', 'dT', 'ds1'])

    # 3. r at 5 sigma without moments -> detection
    res, path, _ = run({'r': {M0: (0.015, 0.003)}})
    check("r at 5 sigma, no moments -> r detection",
          res['conclusion'] == 'r_detection')

    # 4. Res_wr: d and s detected, d wins (evidence), M_d_r preferred -> loop
    scen = {'chi2': {M0: BAD},
            'resid': {(M0, 'd'): -100., (M0, 's'): -120.},
            'logZ': {'M_d_r': GOOD}}
    res, path, F = run(scen)
    fulls = [c[1] for c in F.calls if c[0] == 'full']
    check("Res_wr: best detected d -> M_d_r adopted -> upper limit",
          res['model'] == 'M_d_r' and ('replace', 'M_d_r') in path
          and fulls[1] == 'M_d_r' and 'M_s_r' not in fulls)

    # 4b. same with a bad chi2 of M_d_r: the second round tests s in
    #     Res_wr, then (s not detected) Res_r0
    scen['chi2'] = {M0: BAD, 'M_d_r': BAD}
    res, path, F = run(scen)
    check("  second iteration: Res_wr tests only s, then Res_r0",
          resid_calls(F, 'M_d_r') == ['s']
          and resid_calls(F, 'M_d_0') == ['s', 'dT', 'd2', 'ds1'])

    # 4c. s adopted in the second round from Res_wr: in the third Res_wr
    #     has no candidates and the tree goes to Res_r0
    scen = {'chi2': {M0: BAD, 'M_d_r': BAD, 'M_d+s_r': BAD},
            'resid': {(M0, 'd'): -100., ('M_d_r', 's'): -90.},
            'logZ': {'M_d_r': GOOD, 'M_d+s_r': GOOD + 10.}}
    res, path, F = run(scen)
    check("  third iteration: d and s in, Res_r0 on M_d+s_0",
          ('replace', 'M_d+s_r') in path
          and resid_calls(F, 'M_d+s_r') == []
          and resid_calls(F, 'M_d+s_0') == ['dT', 'd2', 's2', 'ds1',
                                                 'ds2'])

    # 5. Res_wr: d detected but not preferred, s detected and preferred
    scen = {'chi2': {M0: BAD},
            'resid': {(M0, 'd'): -100., (M0, 's'): -120.},
            'logZ': {M0: -0.5 * BAD, 'M_d_r': -0.5 * BAD - 5.,
                     'M_s_r': GOOD}}
    res, path, F = run(scen)
    fulls = [c[1] for c in F.calls if c[0] == 'full']
    check("Res_wr: first not preferred, second one tried and adopted",
          fulls[1:3] == ['M_d_r', 'M_s_r'] and ('replace', 'M_s_r') in path)

    # 6. Res_wr: detected but none preferred -> Res_r0, also when M_0,r has
    #    no moments yet: with r = 0, s (not detected with r free) is
    #    detected and M_s_r preferred
    scen = {'chi2': {M0: BAD},
            'resid': {(M0, 'd'): -100., ('M_none_0', 's'): -90.},
            'logZ': {M0: -0.5 * BAD, 'M_d_r': -0.5 * BAD - 5.,
                     'M_s_r': -0.5 * BAD + 5.}}
    res, path, F = run(scen)
    check("Res_wr: detected but none preferred -> Res_r0 (also without "
          "moments in M_0,r), where s is found",
          resid_calls(F, 'M_none_0')[:2] == ['d', 's']
          and ('replace', 'M_s_r') in path)

    # 7. Res_wr empty, Res_r0: dT detected, M_dT_r preferred -> loop;
    #    after a replacement from Res_r0 Res_wr is skipped
    scen = {'chi2': {M0: BAD, 'M_dT_r': BAD},
            'resid': {('M_none_0', 'dT'): -90., ('M_none_0', 'd'): -95.,
                      ('M_dT_0', 'd'): -80.},
            'logZ': {'M_dT_r': GOOD, 'M_d+dT_r': GOOD + 10.}}
    res, path, F = run(scen)
    check("Res_r0: best detected (dT) adopted",
          ('replace', 'M_dT_r') in path)
    check("  next iteration skips Res_wr and goes to Res_r0",
          resid_calls(F, 'M_dT_r') == []
          and resid_calls(F, 'M_dT_0') == ['d', 's', 'ds1']
          and ('replace', 'M_d+dT_r') in path)

    # 8. Res_r0: d2 only after d
    scen = {'chi2': {M0: BAD, 'M_d_r': BAD},
            'resid': {(M0, 'd'): -100., ('M_d_0', 'd2'): -80.},
            'logZ': {'M_d_r': GOOD, 'M_d+d2_r': GOOD + 10.}}
    res, path, F = run(scen)
    check("d2 tested only when d is in the model",
          'd2' not in resid_calls(F, 'M_none_0')
          and resid_calls(F, 'M_d_0') == ['s', 'dT', 'd2', 'ds1']
          and ('replace', 'M_d+d2_r') in path)

    # 9. Res_r0: d and s detected; only d (highest evidence) as M_1,r, not
    #    preferred; M_1,0 = M_d_0 not preferred; nothing in the residuals of
    #    M_d_0 -> 3 sigma on M_0,r; the chain (d) logged for the profile
    scen = {'r': {M0: (0.012, 0.003)},
            'resid': {('M_none_0', 'd'): -100., ('M_none_0', 's'): -110.},
            'logZ': {M0: -150., 'M_d_r': -150.5, 'M_d_0': -152.}}
    res, path, F = run(scen)
    ev = log_events(scen)
    chain = [e for e in ev if e['node'] == 'r0chain'][-1]
    check("Res_r0: only the first family as M_1,r, then M_1,0, then 3 sigma",
          path[-2][0] == 'r3sigma' and res['model'] == M0
          and ('full', 'M_d_0') in F.calls
          and ('screen', 'M_s_r') not in F.calls
          and resid_calls(F, 'M_d_0') == ['s', 'd2'])
    check("  chain of families in the log for the profile scheme",
          chain['answer'] == 'No' and chain['families'] == ['d']
          and chain['base'] == M0)

    # 9b. s found in the residuals of M_d_0 -> M_1,0 = M_d+s,0 (new spectra
    #     pruned with r = 0), M_1,r not preferred, M_1,0 preferred -> stop;
    #     Bayesian result: 3 sigma test on M_0,r
    scen['resid'][('M_d_0', 's')] = -105.
    scen['resid_off'] = {('M_d_0', 's'): ['A_s_w00w10']}
    DS0 = 'M_d+s-s_w00w10_0'
    scen['logZ'][DS0] = -140.
    res, path, F = run(scen)
    chain = [e for e in log_events(scen) if e['node'] == 'r0chain'][-1]
    check("second round: M_d+s,0 preferred -> M_1 = M_d+s, 3 sigma on M_0,r",
          ('m10', 'M_d+s-s_w00w10_r') in path
          and path[-2][0] == 'r3sigma' and res['model'] == M0
          and res['conclusion'] == 'r_detection')
    check("  second round pruned with r = 0, M_1,r compared first, chain "
          "d, s with the residual screening",
          ('screen', DS0) in F.calls
          and ('screen', 'M_d+s-s_w00w10_r') in F.calls
          and [e[1] for e in path if e[0] == 'pref'][-2:] == ['No', 'Yes']
          and chain['families'] == ['d', 's']
          and chain['pruned'] == [[], ['A_s_w00w10']])

    # 9c. M_d_0 already preferred -> M_1 = M_d, no second round
    scen['logZ']['M_d_0'] = -145.
    res, path, F = run(scen)
    check("  M_1,0 of the first family preferred: no second round",
          ('m10', 'M_d_r') in path and path[-2][0] == 'r3sigma'
          and resid_calls(F, 'M_d_0') == [])

    # 9e. the first family (d) pruned to nothing with r = 0 -> the next one
    #     (s) is tried; M_s,0 preferred
    scen = {'r': {M0: (0.012, 0.003)},
            'resid': {('M_none_0', 'd'): -100., ('M_none_0', 's'): -110.},
            'logZ': {M0: -150., 'M_s_0': -140.},
            'amps': {'M_d_0': {'A_d_w00w10': (0.1, 1.), 'A_d_w10w10': (0.1, 1.)}}}
    res, path, F = run(scen)
    check("Res_r0: family pruned to nothing with r = 0 -> next family",
          ('pruned_out', 'rejected') in path and ('full', 'M_d_0') not in F.calls
          and ('screen', 'M_s_0') in F.calls and ('m10', 'M_s_r') in path)

    # 9f. the case of the flowchart question: M_1,r pruned to nothing with r
    #     free (r absorbs the moment), but M_1,0 (pruned with r = 0) is
    #     preferred over M_0,r -> profile scheme
    scen = {'r': {M0: (0.012, 0.003)},
            'resid': {('M_none_0', 'd'): -100.},
            'logZ': {M0: -150., 'M_d_0': -140.},
            'amps': {'M_d_r': {'A_d_w00w10': (0.1, 1.), 'A_d_w10w10': (0.1, 1.)}}}
    res, path, F = run(scen)
    pr = [e for e in path if e[0] in ('pruned_out', 'pref')]
    check("M_1,r rejected with r free, M_1,0 preferred -> profile scheme",
          pr == [('pruned_out', 'rejected'), ('pref', 'Yes')]
          and ('m10', 'M_d_r') in path and path[-2][0] == 'r3sigma')

    # 9g. r = 0 pruning removes one spectrum: M_1,0 and M_1,r without it
    D0 = 'M_d-d_w00w10_0'
    scen = {'r': {M0: (0.012, 0.003)},
            'resid': {('M_none_0', 'd'): -100.},
            'logZ': {M0: -150., D0: -140.},
            'amps': {'M_d_0': {'A_d_w00w10': (0.1, 1.)}}}
    res, path, F = run(scen)
    check("r = 0 pruning of one spectrum: M_1,0 and M_1,r without it",
          ('full', D0) in F.calls and ('screen', 'M_d-d_w00w10_r') in F.calls
          and ('full', 'M_d_0') not in F.calls
          and ('m10', 'M_d-d_w00w10_r') in path)

    # 9h. s not detected in the first round of the chain: not tested again
    #     on the residuals of M_d_0 (even if it would be detected there)
    scen = {'r': {M0: (0.012, 0.003)},
            'resid': {('M_none_0', 'd'): -100., ('M_d_0', 's'): -90.},
            'logZ': {M0: -150., 'M_d_r': -150.5, 'M_d_0': -152.}}
    res, path, F = run(scen)
    check("family not detected in a round is not retested in the next ones",
          {'s', 'dT'} <= set(resid_calls(F, 'M_none_0'))
          and not {'s', 'dT'} & set(resid_calls(F, 'M_d_0'))
          and 'd2' in resid_calls(F, 'M_d_0')
          and path[-2][0] == 'r3sigma')

    # 9i. d pruned to nothing with r = 0 in the first round, s taken
    #     instead and not preferred: d not tested again on the residuals of
    #     M_s_0 (even if it would be detected there)
    scen = {'r': {M0: (0.012, 0.003)},
            'resid': {('M_none_0', 'd'): -100., ('M_none_0', 's'): -110.,
                      ('M_s_0', 'd'): -90., ('M_s_0', 'ds1'): -95.},
            'logZ': {M0: -150., 'M_s_r': -150.5, 'M_s_0': -152.},
            'amps': {'M_d_0': {'A_d_w00w10': (0.1, 1.), 'A_d_w10w10': (0.1, 1.)}}}
    res, path, F = run(scen)
    check("family pruned to nothing with r = 0 is not retested",
          ('pruned_out', 'rejected') in path and ('screen', 'M_s_0') in F.calls
          and 'd' not in resid_calls(F, 'M_s_0'))

    # 9d. the first family preferred as M_1,r -> adopted, s never tried
    scen = {'r': {M0: (0.012, 0.003)},
            'resid': {('M_none_0', 'd'): -100., ('M_none_0', 's'): -110.},
            'logZ': {M0: -150., 'M_d_r': -140.}}
    res, path, F = run(scen)
    check("Res_r0: first family preferred as M_1,r -> adopted",
          ('replace', 'M_d_r') in path and ('screen', 'M_s_r') not in F.calls
          and ('screen', 'M_s_0') not in F.calls)

    # 13. immediate pruning: A_d_w00w10 not detected in the full fit of
    #     M_d_r -> the pruned model is compared directly, and adopted
    scen = {'chi2': {M0: BAD},
            'resid': {(M0, 'd'): -100.},
            'logZ': {'M_d_r': GOOD, 'M_d-d_w00w10_r': GOOD + 1.},
            'amps': {'M_d_r': {'A_d_w00w10': (0.1, 1.)}}}
    res, path, F = run(scen)
    check("immediate pruning: pruned model compared and adopted",
          ('screen', 'M_d_r') in F.calls and ('full', 'M_d_r') not in F.calls
          and ('prune_check', 'Yes') in path
          and ('replace', 'M_d-d_w00w10_r') in path
          and res['model'] == 'M_d-d_w00w10_r')

    # 14. the full model would win, the pruned one does not -> not adopted
    scen['logZ'] = {M0: -0.5 * BAD, 'M_d_r': GOOD,
                    'M_d-d_w00w10_r': -0.5 * BAD - 5.}
    res, path, F = run(scen)
    check("immediate pruning: only the pruned model counts",
          not any(n == 'replace' for n, _ in path)
          and res['model'] == 'M_none_r')

    # 15. comparison with pruning: M_d_r not preferred, but without its
    #     undetected amplitude yes
    scen = {'chi2': {M0: BAD},
            'resid': {(M0, 'd'): -100.},
            'logZ': {M0: -0.5 * BAD, 'M_d_r': -0.5 * BAD + 1.,
                     'M_d-d_w00w10_r': GOOD},
            'amps': {'M_d_r': {'A_d_w00w10': (0.1, 1.)}}}
    res, path, F = run(scen)
    check("comparison with pruning: pruned M_1,r adopted",
          ('replace', 'M_d-d_w00w10_r') in path
          and res['model'] == 'M_d-d_w00w10_r')

    # 16. all the new amplitudes undetected -> no pruning, 3 sigma
    scen['amps'] = {'M_d_r': {'A_d_w00w10': (0.1, 1.),
                              'A_d_w10w10': (0.1, 1.)}}
    res, path, F = run(scen)
    check("  all new amplitudes undetected -> no pruned refit, 3 sigma",
          ('full', 'M_d-d_w00w10_r') not in F.calls
          and path[-2][0] == 'r3sigma')

    # 17. spectrum removed by the pruning in Res_wr: not a candidate again in Res_r0
    scen = {'chi2': {M0: BAD, 'M_d-d_w00w10_r': BAD},
            'resid': {(M0, 'd'): -100.,
                      ('M_d-d_w00w10_0', 'restore-d'): -80.},
            'logZ': {M0: -0.5 * BAD, 'M_d_r': -0.5 * BAD + 1.,
                     'M_d-d_w00w10_r': GOOD},
            'amps': {'M_d_r': {'A_d_w00w10': (0.1, 1.)}}}
    res, path, F = run(scen)
    check("pruned spectrum NOT retested in Res_r0 (no restore by default)",
          'restore-d' not in resid_calls(F, 'M_d-d_w00w10_0')
          and F.calls.count(('screen', 'M_d_r')) == 1)

    # 18. the new M_0,r is pruned right away: in the next round the
    #     comparisons are with the pruned model
    scen = {'chi2': {M0: BAD, 'M_d_r': BAD, 'M_d-d_w00w10_r': BAD},
            'resid': {(M0, 'd'): -100.},
            'logZ': {'M_d_r': GOOD, 'M_d-d_w00w10_r': GOOD + 3.},
            'amps': {'M_d_r': {'A_d_w00w10': (0.1, 1.)}}}
    res, path, F = run(scen)
    check("adopted model pruned right away, next iteration on the pruned one",
          resid_calls(F, 'M_d-d_w00w10_r') == ['s']
          and resid_calls(F, 'M_d-d_w00w10_0')[:1] == ['s'])

    # 18b. pruning of the full fit BEFORE the comparison: the screening
    #      detects both spectra of d, the posterior of the full fit not
    #      A_d_w00w10 -> the pruned model is compared and adopted, and not
    #      pruned again after the adoption (set 22 rand of the validation)
    X = -0.5 * BAD
    scen = {'chi2': {M0: BAD},
            'resid': {(M0, 'd'): -100.},
            'logZ': {M0: X, 'M_d_r': X + 6., 'M_d-d_w00w10_r': X + 7.},
            'amps_full': {'M_d_r': {'A_d_w00w10': (0.1, 1.)}}}
    res, path, F = run(scen)
    i_rep = [n for n, _ in path].index('replace')
    check("full-fit pruning before the comparison: pruned model compared "
          "and adopted",
          ('pref', 'Yes') in path and ('replace', 'M_d-d_w00w10_r') in path
          and ('prune', 'Yes') in path[:i_rep]
          and not any(n == 'prune' for n, _ in path[i_rep:]))

    # 18c. the full fit loses against M_0,r (ln B = -0.5) but the pruned
    #      model wins (ln B = +0.5) -> adopted (with the pruning after the
    #      adoption it was never compared)
    scen['logZ'] = {M0: X, 'M_d_r': X - 0.5, 'M_d-d_w00w10_r': X + 0.5}
    res, path, F = run(scen, lnB_preferred=0.)
    check("  the pruned model, not the full one, is compared with M_0,r",
          ('replace', 'M_d-d_w00w10_r') in path)

    # 18c'. the pruned model is worse than the full one (ln B < 0): the
    #       pruning is undone and the full model is compared
    scen['logZ'] = {M0: X, 'M_d_r': X + 3., 'M_d-d_w00w10_r': X + 2.}
    res, path, F = run(scen)
    check("  pruned model worse than the full one -> pruning undone",
          ('prune', 'No') in path and ('replace', 'M_d_r') in path)

    # 18d. the posterior of the full fit leaves no new spectrum -> candidate
    #      rejected (sets 36 and 81 d9s4 of the validation)
    scen = {'chi2': {M0: BAD},
            'resid': {(M0, 's'): -100.},
            'logZ': {M0: X, 'M_s_r': X + 0.5,
                     'M_s-s_w00w10-s_w10w10_r': X + 0.8},
            'amps_full': {'M_s_r': {'A_s_w00w10': (0.1, 1.),
                                    'A_s_w10w10': (0.1, 1.)}}}
    res, path, F = run(scen, lnB_preferred=0.)
    check("  no new spectrum left after the pruning -> candidate rejected",
          ('pruned_out', 'rejected') in path
          and not any(n == 'replace' for n, _ in path))

    # 19. second round: s detected in Res_wr but not preferred; the model
    #     already has d -> to Res_r0 (not to the 3 sigma test)
    scen = {'chi2': {M0: BAD, 'M_d_r': BAD},
            'resid': {(M0, 'd'): -100., ('M_d_r', 's'): -90.},
            'logZ': {'M_d_r': GOOD, 'M_d+s_r': GOOD - 5.}}
    res, path, F = run(scen)
    check("Res_wr candidates not preferred with moments already in -> Res_r0",
          ('full', 'M_d+s_r') in F.calls
          and resid_calls(F, 'M_d_0') == ['s', 'dT', 'd2', 'ds1'])

    # 20. borderline comparison (ln B = 2.6 +- 0.57 with threshold 2.5):
    #     PolyChord repeated with other seeds on both models; the mean drops
    #     to ln B = 1.85 -> d not adopted. Without repetitions it would be.
    scen = {'chi2': {M0: BAD, 'M_d_r': BAD},
            'resid': {(M0, 'd'): -100.},
            'logZ': {M0: -200., 'M_d_r': -197.4},
            'logZerr': {M0: 0.4, 'M_d_r': 0.4},
            'reruns': {'M_d_r': [-198.4, -198.2, -198.6]}}
    res, path, F = run(scen, seed_max_runs=4)
    reruns = [c[1] for c in F.calls if c[0] == 'rerun']
    pref = [a for n, a in path if n == 'pref']
    # (stops when |ln B - 2.5| = 0.65 exceeds 2 x 0.31: 4 + 3 runs)
    check("borderline lnB -> reruns alternated until out of the band",
          reruns.count('M_d_r') == 3 and reruns.count(M0) == 2
          and pref[0] == 'No')
    res, path, F = run(scen, seed_max_runs=1)
    pref = [a for n, a in path if n == 'pref']
    check("seed_max_runs = 1: no reruns, single-run decision",
          not any(c[0] == 'rerun' for c in F.calls) and pref[0] == 'Yes')
    check("  still within 2 errors: choice flagged in the conclusion",
          res['undecided'] and res['undecided_choices'][0]['chosen']
          == 'M_d_r' and res['undecided_choices'][0]['other'] == M0)
    # far from the threshold: no repetition
    scen['logZ']['M_d_r'] = -190.
    res, path, F = run(scen, seed_max_runs=4)
    check("clear lnB -> no reruns",
          not any(c[0] == 'rerun' for c in F.calls))
    check("  clear lnB -> no undecided flag", not res['undecided'])

    # 21. group s with all its spectra pruned: does not satisfy the
    #     requirements of s2 / ds1 / ds2, but can still be restored
    m = dt.Model(frozenset({'d', 's'}), True,
                 frozenset({'A_d_w00w10', 'A_s_w00w10', 'A_s_w10w10'}))
    c = dt.candidate_groups(m, ['dT', 'd2', 's2', 'ds1', 'ds2'],
                            restore=True, extra_requires={'ds1': ['s']})
    check("fully pruned group does not satisfy requirements",
          'd2' in c and not {'s2', 'ds1', 'ds2'} & set(c)
          and 'restore-s' in c)

    # 22. case d10s5 r = 0.03: s detected in the residuals but, pruned, no
    #     spectrum of s is left -> rejected (Res_wr and Res_r0), no cycle,
    #     3 sigma test on the pruned d
    Ms = 'M_d+s-d_w00w10_r'
    Mp = 'M_d+s-d_w00w10-s_w00w10_r'
    scen = {'chi2': {M0: BAD},
            'resid': {(M0, 'd'): -100., ('M_d-d_w00w10_r', 's'): -90.,
                      ('M_d-d_w00w10_0', 's'): -80.},
            'logZ': {'M_d_r': GOOD, 'M_d-d_w00w10_r': GOOD + 1.,
                     Ms: GOOD + 1.5, Mp: GOOD + 0.5},
            'r': {m: (0.03, 0.005) for m in
                  ['M_d_r', 'M_d-d_w00w10_r', Ms, Mp]},
            'amps': {'M_d_r': {'A_d_w00w10': (0.1, 1.)},
                     Ms: {'A_s_w00w10': (0.1, 1.), 'A_s_w10w10': (1., 0.1)},
                     Mp: {'A_s_w10w10': (0.1, 1.)}}}
    res, path, F = run(scen, lnB_preferred=0.)
    check("new group pruned to nothing -> rejected, no cycle, 3 sigma",
          path.count(('pruned_out', 'rejected')) == 2
          and ('cycle', 'Yes') not in path
          and res['conclusion'] == 'r_detection'
          and res['model'] == 'M_d-d_w00w10_r')

    # 23. Res_r0: the best candidate (s2) cannot be fitted -> the next one
    #     (d2) is tried
    scen = {'chi2': {M0: BAD, 'M_d+s_r': BAD},
            'resid': {(M0, 'd'): -100., ('M_d_r', 's'): -95.,
                      ('M_d+s_0', 's2'): -80., ('M_d+s_0', 'd2'): -85.},
            'logZ': {'M_d_r': GOOD, 'M_d+s_r': GOOD + 1.},
            'r': {'M_d_r': (0.03, 0.005), 'M_d+s_r': (0.03, 0.005)},
            'unfittable': {'M_d+s+s2_0'}}
    res, path, F = run(scen, lnB_preferred=0.)
    check("Res_r0: unfittable best candidate skipped, next one tried",
          ('unfittable', 'skipped') in path
          and ('screen', 'M_d+d2+s_0') in F.calls)

    # 24. pruning in the residual test: A_d_w00w10 not detected by the
    #     screening -> the candidate starts without it (no screening of the
    #     model with A_d_w00w10)
    scen = {'chi2': {M0: BAD},
            'resid': {(M0, 'd'): -100.},
            'resid_off': {(M0, 'd'): ['A_d_w00w10']},
            'logZ': {'M_d-d_w00w10_r': GOOD}}
    res, path, F = run(scen)
    check("residual test pruning: candidate starts already pruned",
          ('screen', 'M_d_r') not in F.calls
          and ('screen', 'M_d-d_w00w10_r') in F.calls
          and res['model'] == 'M_d-d_w00w10_r')

    print("\nLikelihood-ratio residual test:")
    test_lrt()
    print("\nGenerated BBCompSep configs:")
    test_configs()
    print("\nHPD and detection:")
    test_hpd()
    print("\nNested ML fits:")
    test_nested_ml()
    print("\nAll tests passed.")


def test_lrt():
    """Fitter with residual_test: lrt, on a runner that returns given chi2:
    detection from Delta chi2 with k = amplitudes, order by log p."""
    class R:
        base = load_yaml(os.path.join(REPO, 'configs',
                                      'base_config_tree.yml'))

        def __init__(self, chi2):
            self.chi2, self.calls = chi2, []

        def ml(self, model, tag, fixed=None, centers=None):
            self.calls.append(tag)
            return dt.FitResult(model, tag, ['A'], {'A': 0.5},
                                self.chi2[tag], NDATA)

    base = dt.FitResult(dt.Model(frozenset(), False), 'b', ['p'], {'p': 0.},
                        400., NDATA)
    tag = 'resid_M_none_0_plus_'
    r = R({tag + 'd_ML': 397., tag + 's_ML': 399., tag + 'dT_ML': 390.})
    F = dt.Fitter(r, residual_test='lrt', lrt_p=0.3173)
    d, s_, dT = (F.residual_test(base, g) for g in ('d', 's', 'dT'))
    check("Delta chi2 = 3 with 2 amplitudes: p = 0.22 < 0.32 -> detected",
          d.detected and abs(d.pvalue - np.exp(-1.5)) < 1e-9
          and d.stage == 'lrt' and d.off == ())
    check("  Delta chi2 = 1: p = 0.61 -> not detected",
          not s_.detected and abs(s_.dchi2 - 1.) < 1e-12)
    check("  order by log p (dT, Delta chi2 = 10, before d), only ML fits",
          sorted([d, dT], key=lambda x: x.rank)[0].group == 'dT'
          and all(t.endswith('_ML') for t in r.calls))


def test_configs():
    base = load_yaml(os.path.join(REPO, 'configs', 'base_config_tree.yml'))
    try:
        from bbpower.param_manager import ParameterManager
    except Exception:
        ParameterManager = None
        print("  (bbpower not importable: skipping the ParameterManager "
              "check)")

    for mom in [frozenset(), {'d'}, {'s'}, {'dT'}, {'d', 'dT'},
                {'d', 'd2'}, {'d', 's', 's2'}, {'ds1'}, {'d', 's', 'ds2'}]:
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
                  and ('temp_d' in names) == ('dT' in mom)
                  and all(('alpha' + a[1:]) in names for a in amps))
            if ParameterManager:
                pm = ParameterManager(cs)
                check("  same order as ParameterManager",
                      pm.p_free_names == names)

    m = dt.Model(frozenset({'d'}), True, frozenset({'A_d_w00w10'}))
    names = dt.free_param_names(dt.build_config(base, m, 'polychord')
                                ['BBCompSep'])
    check(f"pruned model {m.label}: A_d_w00w10 and its slope fixed",
          'A_d_w10w10' in names and 'A_d_w00w10' not in names
          and 'alpha_d_w00w10' not in names)
    check("  unpruned configs unchanged (runs reused)",
          dt.build_config(base, dt.Model(frozenset({'d'})), 'polychord')
          == dt.build_config(base, dt.Model(frozenset({'d'}), True,
                                            frozenset()), 'polychord'))

    check("cross spectrum w10w01 only with d and dT",
          ('component_1', 'd', 'w10w01') in dt.group_spectra({'d', 'dT'})
          and ('component_1', 'd', 'w10w01') not in dt.group_spectra({'dT'}))

    # residuals: everything fixed except the group and its beta
    m = dt.Model(frozenset({'d'}), False)
    fixed = {n: 0.5 for n in ['amp_d_bb', 'alpha_d_bb', 'beta_s',
                              'epsilon_ds', 'amp_s_bb', 'alpha_s_bb']}
    fixed['r_tensor'] = 0.003
    cs = dt.build_config(base, m, 'emcee', fixed=fixed)['BBCompSep']
    free = dt.free_param_names(cs)
    check(f"residuals: only the group and its beta free ({free})",
          sorted(free) == sorted(['beta_d', 'A_d_w00w10', 'A_d_w10w10',
                                  'alpha_d_w00w10', 'alpha_d_w10w10'])
          and cs['cmb_model']['params']['r_tensor'][2] == [0.003])
    check("  the base config is not modified",
          base['BBCompSep']['cmb_model']['params']['r_tensor'][1] == 'tophat')


def test_hpd():
    rng = np.random.default_rng(2)     # indipendente dagli scenari
    x = rng.normal(1., 0.5, 200000)
    lo, hi = dt.hpd_interval(x)
    check(f"HPD of a Gaussian = central 68% ({lo:.3f}, {hi:.3f})",
          abs(lo - 0.5) < 0.02 and abs(hi - 1.5) < 0.02)
    y = np.abs(rng.normal(0., 1., 200000))
    lo, _ = dt.hpd_interval(y, bounds=(0., 10.))
    check(f"HPD of a half-Gaussian with prior >= 0 starts at 0 ({lo})",
          lo == 0.)
    z = 3. + np.abs(rng.normal(0., 1., 200000))
    lo, _ = dt.hpd_interval(z, bounds=(0., 10.))
    check(f"  but not if the posterior is away from the edge ({lo:.3f})",
          lo > 2.9)
    names = ['A', 'B']
    s = np.vstack([y, rng.normal(3., 1., len(y))]).T
    b = {'A': (0., 10.), 'B': (-10., 10.)}
    check("detection: B away from zero -> detected, A alone not",
          dt.detection(names, s, ['A', 'B'], bounds=b)[0]
          and not dt.detection(names, s, ['A'], bounds=b)[0])


def test_nested_ml():
    """A fit worse than the model it contains is redone from its best fit."""
    small = dt.Model(frozenset({'d'}), True)
    big = small.plus('s')

    class FakeRunner:
        def __init__(self):
            self.tags = []

        def ml(self, model, tag, fixed=None, centers=None):
            self.tags.append(tag)
            if model == small:
                chi2 = 520.
            elif tag.endswith('_ML'):
                chi2 = 600.                  # stuck minimizer
            else:
                # restart from the best fit of the smaller model
                chi2 = 515. if centers and centers.get('A_s_w10w10') == 0. else 600.
            return dt.FitResult(model, tag, ['p'], {'p': 1.}, chi2, NDATA)

    R = FakeRunner()
    F = dt.Fitter(R)
    F.ml_fit(small)
    res = F.ml_fit(big)
    check("stuck fit of the bigger model redone from the smaller one",
          res.chi2 == 515. and res.tag.endswith('_MLn1'))
    R2 = FakeRunner()
    F2 = dt.Fitter(R2)
    F2.ml_fit(big)
    check("  no smaller model fitted: nothing redone", R2.tags == [big.label + '_ML'])
    check("  contains(): r = 0 model inside the r-free one, not the opposite",
          dt.contains(big, big.with_r(False)) and not dt.contains(big.with_r(False), big)
          and dt.contains(big, small) and not dt.contains(small, big))


if __name__ == '__main__':
    main()
