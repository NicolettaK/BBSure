#!/usr/bin/env python
"""
Test della logica del decision tree v2 senza lanciare BBCompSep.

Una MockFitter restituisce chi2 / evidenze / posterior di r / test dei
residui inventati per ogni scenario, e si controlla che il tree percorra il
ramo giusto del flowchart. Controlla anche che i config generati siano
coerenti con ParameterManager di BBPower (se importabile).
La versione per il tree v1 e' test_decision_tree_v1.py.

Uso:  python test_decision_tree.py
"""
import os
import sys
import tempfile

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decision_tree as dt  # noqa: E402

NDATA = 315
rng = np.random.default_rng(1)


class MockFitter:
    """
    scen: dict con
      chi2[label]  chi2 del best fit per ogni modello (default: buono)
      logZ[label]  evidenza (default: -chi2/2 - 0.5*nfree)
      r[label]     (media, sigma) del posterior di r
      resid[(base_label, gruppo)]  logZ del test dei residui se rivelato
                   (gruppi assenti: non rivelati)
      prof_dchi2   differenza tra minimi dei profili (M0 - M1)
      amps[label]  {ampiezza: (media, sigma)} del posterior delle ampiezze
                   dei momenti (default (1, 0.1): rivelate)
      logZerr[label]  errore di PolyChord (default 0.1)
      reruns[label]   logZ dei run successivi con altri semi (default: logZ)
    """

    def __init__(self, scen):
        self.s = scen
        self.calls = []

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
        amps = model.amplitudes()
        spec = self.s.get('amps', {}).get(lab, {})
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
        return self._fit(model)

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

    def profile(self, model, r_grid, centers=None):
        self.calls.append(('profile', model.label))
        base = 100. if not model.moments else 100. - self.s.get(
            'prof_dchi2', 0.)
        return base + ((r_grid - 0.01) / 0.005) ** 2


def tree_cfg(runs_dir, freq_replaces=True, seed_max_runs=1,
             lnB_preferred=2.5, freq_branch='profile'):
    return dict(paths=dict(runs_dir=runs_dir),
                thresholds=dict(pte_min=0.01, lnB_preferred=lnB_preferred,
                                lrt_alpha=0.05, seed_band=2.0,
                                seed_max_runs=seed_max_runs),
                moments=dict(res_wr=['d', 's'],
                             res_r0=['d', 's', 'dT', 'd2', 's2', 'ds1',
                                     'ds2']),
                profile=dict(r_min=-0.01, r_max=0.04, n_r=21),
                run=dict(max_iterations=6,
                         frequentist_replaces_baseline=freq_replaces,
                         frequentist_branch=freq_branch))


def run(scen, freq_replaces=True, seed_max_runs=1, lnB_preferred=2.5,
        freq_branch='profile'):
    d = tempfile.mkdtemp()
    log = dt.DecisionLog(d, verbose=False)
    F = MockFitter(scen)
    res = dt.DecisionTree(tree_cfg(d, freq_replaces, seed_max_runs,
                                   lnB_preferred, freq_branch), F, log).run()
    path = [(e['node'], e['answer']) for e in log.events]
    return res, path, F


M0 = 'M_none_r'
BAD = NDATA - 9 + 120          # chi2 con PTE ~ 0
GOOD = -0.5 * (NDATA - 11) + 20.  # evidenza alta per i modelli con momenti


def check(name, cond):
    print(f"  {'OK ' if cond else 'FAIL'} {name}")
    if not cond:
        raise SystemExit(1)


def resid_calls(F, base):
    return [c[2] for c in F.calls if c[0] == 'resid' and c[1] == base]


def main():
    print("Flowchart scenarios:")

    # 1. chi2 buono, r ~ 0 -> upper limit subito
    res, path, _ = run({})
    check("chi2 ok + r ok at 1 sigma -> upper limit",
          res['conclusion'] == 'upper_limit' and res['model'] == M0
          and ('r1sigma', 'Yes') in path)

    # 2. r a 2 sigma, niente in Res_wr ne' in Res_r0 -> 3 sigma ok
    res, path, F = run({'r': {M0: (0.006, 0.003)}})
    check("nothing in Res_wr and Res_r0 -> 3 sigma -> upper limit",
          res['conclusion'] == 'upper_limit' and ('r3sigma', 'Yes') in path
          and resid_calls(F, M0) == ['d', 's']
          and resid_calls(F, 'M_none_0') == ['d', 's', 'dT', 'ds1'])

    # 3. r a 5 sigma senza momenti -> detection
    res, path, _ = run({'r': {M0: (0.015, 0.003)}})
    check("r at 5 sigma, no moments -> r detection",
          res['conclusion'] == 'r_detection')

    # 4. Res_wr: d e s rivelati, vince d (evidenza), M_d_r preferito -> loop
    scen = {'chi2': {M0: BAD},
            'resid': {(M0, 'd'): -100., (M0, 's'): -120.},
            'logZ': {'M_d_r': GOOD}}
    res, path, F = run(scen)
    fulls = [c[1] for c in F.calls if c[0] == 'full']
    check("Res_wr: best detected d -> M_d_r adopted -> upper limit",
          res['model'] == 'M_d_r' and ('replace', 'M_d_r') in path
          and fulls[1] == 'M_d_r' and 'M_s_r' not in fulls)

    # 4b. stesso con chi2 di M_d_r cattivo: il secondo giro testa s in
    #     Res_wr, poi (s non rivelato) Res_r0
    scen['chi2'] = {M0: BAD, 'M_d_r': BAD}
    res, path, F = run(scen)
    check("  second iteration: Res_wr tests only s, then Res_r0",
          resid_calls(F, 'M_d_r') == ['s']
          and resid_calls(F, 'M_d_0') == ['s', 'dT', 'd2', 'ds1'])

    # 4c. s adottato al secondo giro da Res_wr: al terzo Res_wr non ha
    #     candidati e si va a Res_r0
    scen = {'chi2': {M0: BAD, 'M_d_r': BAD, 'M_d+s_r': BAD},
            'resid': {(M0, 'd'): -100., ('M_d_r', 's'): -90.},
            'logZ': {'M_d_r': GOOD, 'M_d+s_r': GOOD + 10.}}
    res, path, F = run(scen)
    check("  third iteration: d and s in, Res_r0 on M_d+s_0",
          ('replace', 'M_d+s_r') in path
          and resid_calls(F, 'M_d+s_r') == []
          and resid_calls(F, 'M_d+s_0') == ['dT', 'd2', 's2', 'ds1',
                                                 'ds2'])

    # 5. Res_wr: d rivelato ma non preferito, s rivelato e preferito
    scen = {'chi2': {M0: BAD},
            'resid': {(M0, 'd'): -100., (M0, 's'): -120.},
            'logZ': {M0: -0.5 * BAD, 'M_d_r': -0.5 * BAD - 5.,
                     'M_s_r': GOOD}}
    res, path, F = run(scen)
    fulls = [c[1] for c in F.calls if c[0] == 'full']
    check("Res_wr: first not preferred, second one tried and adopted",
          fulls[1:3] == ['M_d_r', 'M_s_r'] and ('replace', 'M_s_r') in path)

    # 6. Res_wr: rivelati ma nessuno preferito -> 3 sigma, niente Res_r0
    scen = {'chi2': {M0: BAD},
            'resid': {(M0, 'd'): -100.},
            'logZ': {M0: -0.5 * BAD, 'M_d_r': -0.5 * BAD - 5.}}
    res, path, F = run(scen)
    check("Res_wr: detected but none preferred -> 3 sigma (no Res_r0)",
          path[-2][0] == 'r3sigma' and resid_calls(F, 'M_none_0') == [])

    # 7. Res_wr vuoto, Res_r0: dT rivelato, M_dT_r preferito -> loop;
    #    dopo una sostituzione da Res_r0 si salta Res_wr
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

    # 8. Res_r0: d2 solo dopo d
    scen = {'chi2': {M0: BAD, 'M_d_r': BAD},
            'resid': {(M0, 'd'): -100., ('M_d_0', 'd2'): -80.},
            'logZ': {'M_d_r': GOOD, 'M_d+d2_r': GOOD + 10.}}
    res, path, F = run(scen)
    check("d2 tested only when d is in the model",
          'd2' not in resid_calls(F, 'M_none_0')
          and resid_calls(F, 'M_d_0') == ['s', 'dT', 'd2', 'ds1']
          and ('replace', 'M_d+d2_r') in path)

    # 9. Res_r0: M_1,r non preferito, M_1,0 si', profilo preferisce M1
    base_scen = {'r': {M0: (0.012, 0.003)},
                 'resid': {('M_none_0', 'd'): -100.},
                 'logZ': {M0: -150., 'M_d_r': -149.5, 'M_d_0': -140.}}
    scen = dict(base_scen, prof_dchi2=20.)
    res, path, F = run(scen)
    check("frequentist branch: M*_1,r wins -> loop on M_d_r",
          ('pref_profile', 'Yes') in path and ('mstar', 'M_d_r') in path
          and ('replace', 'M_d_r') in path and ('profile', M0) in F.calls)

    # 10. stesso con frequentist_replaces_baseline = false -> 3 sigma
    res, path, _ = run(scen, freq_replaces=False)
    check("frequentist branch with replaces=false -> 3 sigma on M_0,r",
          ('mstar', 'M_d_r') in path and path[-2][0] == 'r3sigma'
          and res['model'] == M0)

    # 11. profilo non preferisce M1 -> 3 sigma -> detection
    scen = dict(base_scen, prof_dchi2=0.5)
    res, path, _ = run(scen)
    check("frequentist branch: M*_0,r -> 3 sigma -> r detection",
          ('mstar', M0) in path and res['conclusion'] == 'r_detection')

    # 12. M_1,0 non preferito -> 3 sigma senza profilo
    scen = dict(base_scen)
    scen['logZ'] = {M0: -150., 'M_d_r': -149.5, 'M_d_0': -149.}
    res, path, F = run(scen)
    check("M_1,0 not preferred -> 3 sigma (no profile)",
          not any(c[0] == 'profile' for c in F.calls)
          and path[-2][0] == 'r3sigma')

    # 12b. ramo bounded_r: M_1,0 preferito -> M_0 e M_1 con r in [0, r_95];
    #      M_1 preferito -> sostituisce M_0,r; altrimenti 3 sigma
    scen = dict(base_scen)
    scen['logZ'] = dict(base_scen['logZ'], M_none_rb=-150., M_d_rb=-140.)
    res, path, F = run(scen, freq_branch='bounded_r')
    check("bounded_r branch: M_1 wins with r in [0, r_95] -> loop on M_d_r",
          ('full', 'M_none_rb') in F.calls and ('full', 'M_d_rb') in F.calls
          and ('pref_bounded', 'Yes') in path and ('replace', 'M_d_r') in path
          and not any(c[0] == 'profile' for c in F.calls))
    scen['logZ'] = dict(base_scen['logZ'], M_none_rb=-150., M_d_rb=-151.)
    res, path, F = run(scen, freq_branch='bounded_r')
    check("bounded_r branch: M_0 wins -> 3 sigma on M_0,r",
          ('pref_bounded', 'No') in path and path[-2][0] == 'r3sigma'
          and res['model'] == M0)
    cfg = dt.build_config(
        {'BBCompSep': {'fg_model': {}, 'cmb_model': {'params': {
            'r_tensor': ['r_tensor', 'tophat', [-0.02, 0.0, 0.06]]}}}},
        dt.Model().bounded(0.01), 'polychord')
    check("bounded model: r prior tophat [0, r_max] in the config",
          cfg['BBCompSep']['cmb_model']['params']['r_tensor'][1:]
          == ['tophat', [0.0, 0.005, 0.01]])

    # 13. potatura immediata: A_d_w00w10 non rivelata nel fit completo di
    #     M_d_r -> si confronta direttamente il modello potato, che viene
    #     adottato
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

    # 14. il modello completo vincerebbe, ma quello potato no -> non adottato
    scen['logZ'] = {M0: -0.5 * BAD, 'M_d_r': GOOD,
                    'M_d-d_w00w10_r': -0.5 * BAD - 5.}
    res, path, F = run(scen)
    check("immediate pruning: only the pruned model counts",
          not any(n == 'replace' for n, _ in path)
          and res['model'] == 'M_none_r')

    # 15. confronto con pruning: M_d_r non preferito, ma senza la sua
    #     ampiezza non rivelata si'
    scen = {'chi2': {M0: BAD},
            'resid': {(M0, 'd'): -100.},
            'logZ': {M0: -0.5 * BAD, 'M_d_r': -0.5 * BAD + 1.,
                     'M_d-d_w00w10_r': GOOD},
            'amps': {'M_d_r': {'A_d_w00w10': (0.1, 1.)}}}
    res, path, F = run(scen)
    check("comparison with pruning: pruned M_1,r adopted",
          ('replace', 'M_d-d_w00w10_r') in path
          and res['model'] == 'M_d-d_w00w10_r')

    # 16. tutte le ampiezze nuove non rivelate -> niente pruning, 3 sigma
    scen['amps'] = {'M_d_r': {'A_d_w00w10': (0.1, 1.),
                              'A_d_w10w10': (0.1, 1.)}}
    res, path, F = run(scen)
    check("  all new amplitudes undetected -> no pruned refit, 3 sigma",
          ('full', 'M_d-d_w00w10_r') not in F.calls
          and path[-2][0] == 'r3sigma')

    # 17. spettro tolto dal pruning in Res_wr: non torna candidato in Res_r0
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

    # 18. il nuovo M_0,r viene potato subito: al giro dopo i confronti sono
    #     con il modello potato
    scen = {'chi2': {M0: BAD, 'M_d_r': BAD, 'M_d-d_w00w10_r': BAD},
            'resid': {(M0, 'd'): -100.},
            'logZ': {'M_d_r': GOOD, 'M_d-d_w00w10_r': GOOD + 3.},
            'amps': {'M_d_r': {'A_d_w00w10': (0.1, 1.)}}}
    res, path, F = run(scen)
    check("adopted model pruned right away, next iteration on the pruned one",
          resid_calls(F, 'M_d-d_w00w10_r') == ['s']
          and resid_calls(F, 'M_d-d_w00w10_0')[:1] == ['s'])

    # 19. secondo giro: s rivelato in Res_wr ma non preferito; il modello ha
    #     gia' d -> si va a Res_r0 (non al test a 3 sigma)
    scen = {'chi2': {M0: BAD, 'M_d_r': BAD},
            'resid': {(M0, 'd'): -100., ('M_d_r', 's'): -90.},
            'logZ': {'M_d_r': GOOD, 'M_d+s_r': GOOD - 5.}}
    res, path, F = run(scen)
    check("Res_wr candidates not preferred with moments already in -> Res_r0",
          ('full', 'M_d+s_r') in F.calls
          and resid_calls(F, 'M_d_0') == ['s', 'dT', 'd2', 'ds1'])

    # 20. confronto in bilico (ln B = 2.6 +- 0.57 con soglia 2.5): PolyChord
    #     ripetuto con altri semi su entrambi i modelli; la media scende a
    #     ln B = 1.85 -> d non adottato. Senza ripetizioni sarebbe adottato.
    scen = {'chi2': {M0: BAD, 'M_d_r': BAD},
            'resid': {(M0, 'd'): -100.},
            'logZ': {M0: -200., 'M_d_r': -197.4},
            'logZerr': {M0: 0.4, 'M_d_r': 0.4},
            'reruns': {'M_d_r': [-198.4, -198.2, -198.6]}}
    res, path, F = run(scen, seed_max_runs=4)
    reruns = [c[1] for c in F.calls if c[0] == 'rerun']
    pref = [a for n, a in path if n == 'pref']
    # (si ferma quando |ln B - 2.5| = 0.65 supera 2 x 0.31: 4 + 3 run)
    check("borderline lnB -> reruns alternated until out of the band",
          reruns.count('M_d_r') == 3 and reruns.count(M0) == 2
          and pref[0] == 'No')
    res, path, F = run(scen, seed_max_runs=1)
    pref = [a for n, a in path if n == 'pref']
    check("seed_max_runs = 1: no reruns, single-run decision",
          not any(c[0] == 'rerun' for c in F.calls) and pref[0] == 'Yes')
    # lontano dalla soglia: nessuna ripetizione
    scen['logZ']['M_d_r'] = -190.
    res, path, F = run(scen, seed_max_runs=4)
    check("clear lnB -> no reruns",
          not any(c[0] == 'rerun' for c in F.calls))

    # 21. gruppo s con tutti gli spettri potati: non soddisfa i requisiti
    #     di s2 / ds1 / ds2, ma resta ripristinabile
    m = dt.Model(frozenset({'d', 's'}), True,
                 frozenset({'A_d_w00w10', 'A_s_w00w10', 'A_s_w10w10'}))
    c = dt.candidate_groups(m, ['dT', 'd2', 's2', 'ds1', 'ds2'],
                            restore=True, extra_requires={'ds1': ['s']})
    check("fully pruned group does not satisfy requirements",
          'd2' in c and not {'s2', 'ds1', 'ds2'} & set(c)
          and 'restore-s' in c)

    # 22. caso d10s5 r = 0.03: s rivelato nei residui ma, potato, non resta
    #     nessuno spettro di s -> scartato (Res_wr e Res_r0), niente ciclo,
    #     test a 3 sigma su d potato
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

    # 23. Res_r0: il candidato migliore (s2) non si riesce a fittare -> si
    #     passa al successivo (d2)
    scen = {'chi2': {M0: BAD, 'M_d+s_r': BAD},
            'resid': {(M0, 'd'): -100., ('M_d_r', 's'): -95.,
                      ('M_d+s_0', 's2'): -80., ('M_d+s_0', 'd2'): -85.},
            'logZ': {'M_d_r': GOOD, 'M_d+s_r': GOOD + 1.},
            'r': {'M_d_r': (0.03, 0.005), 'M_d+s_r': (0.03, 0.005)},
            'unfittable': {'M_d+s+s2_r'}}
    res, path, F = run(scen, lnB_preferred=0.)
    check("Res_r0: unfittable best candidate skipped, next one tried",
          ('unfittable', 'skipped') in path
          and ('screen', 'M_d+d2+s_r') in F.calls)

    # 24. potatura nel test dei residui: A_d_w00w10 non rivelata dallo
    #     screening -> il candidato parte gia' senza (niente screening del
    #     modello con A_d_w00w10)
    scen = {'chi2': {M0: BAD},
            'resid': {(M0, 'd'): -100.},
            'resid_off': {(M0, 'd'): ['A_d_w00w10']},
            'logZ': {'M_d-d_w00w10_r': GOOD}}
    res, path, F = run(scen)
    check("residual test pruning: candidate starts already pruned",
          ('screen', 'M_d_r') not in F.calls
          and ('screen', 'M_d-d_w00w10_r') in F.calls
          and res['model'] == 'M_d-d_w00w10_r')

    print("\nGenerated BBCompSep configs:")
    test_configs()
    print("\nHPD and detection:")
    test_hpd()
    print("\nAll tests passed.")


def test_configs():
    base = yaml.safe_load(open(os.path.join(HERE, 'config',
                                            'base_config_free.yml')))
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

    # residui: tutto fissato tranne il gruppo e il suo beta
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


if __name__ == '__main__':
    main()
