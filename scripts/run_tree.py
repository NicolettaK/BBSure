#!/usr/bin/env python
"""
Runs the decision tree on one data set and, with run: profile_scheme: true,
the profile scheme on the concluded tree.

  python scripts/run_tree.py --config tree_config.yml [--phase0]

--phase0 (or phase0: enabled: true in the config), after this analysis (A):
the frequency-split screening (bbsure/phase0.py); if it keeps a moment
group that is not in the final model of A, a second analysis (B: tree +
profile scheme) starts from the model of the phase 0 alone, in
<runs_dir>_p0. A stays the main result; conclusion.json gets a 'phase0'
block with B (model, r, conclusion) and the flags model_differs /
conclusion_changed.

Output in paths: runs_dir: decision_log.{txt,json}, conclusion.json (and
conclusion_tree.json + profile/ with the profile scheme), one directory per
BBCompSep run.
"""
import argparse
import copy
import json
import os
import subprocess
import sys

import yaml

from bbsure import BBCompSepRunner, DecisionLog, DecisionTree, Fitter
from bbsure.log import fmt, json_default
from bbsure.models import model_from_label
from bbsure.phase0 import Phase0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--phase0', action='store_true',
                    help='frequency-split screening after the analysis; '
                         'second analysis from its model if it finds more')
    args = ap.parse_args()

    tc = yaml.safe_load(open(args.config))
    runs_dir = tc['paths']['runs_dir']
    os.makedirs(runs_dir, exist_ok=True)
    log = DecisionLog(runs_dir)
    final = analyse(tc, args.config, runs_dir, log)
    if args.phase0 or (tc.get('phase0') or {}).get('enabled', False):
        final['phase0'] = phase0_check(tc, args.config, runs_dir, log, final)
    # conclusion.json only at the very end (nersc/run_one.sh skips the sets
    # that have it)
    with open(os.path.join(runs_dir, 'conclusion.json'), 'w') as f:
        json.dump(final, f, indent=2, default=json_default)


def phase0_check(tc, config, runs_dir, log, A):
    """Phase 0 after analysis A; analysis B from its model if it keeps a
    group that is not in the final model of A. Returns the 'phase0' block
    of conclusion.json."""
    ph = Phase0(tc, log)
    S = ph.run()
    final_A = model_from_label(A['model'])
    extra = sorted(S.moments - final_A.moments)
    out = dict(start=S.label, extra_groups=extra, run_b=bool(extra),
               undecided=ph.undecided)
    if not extra:
        log.node('phase0_check', f'phase 0 ({S.label}): anything not in the '
                 f"final model of the analysis ({A['model']})?", 'No')
        return out
    log.node('phase0_check', f'phase 0 ({S.label}): anything not in the '
             f"final model of the analysis ({A['model']})?", 'Yes',
             extra=extra)
    dB = runs_dir.rstrip('/') + '_p0'
    os.makedirs(dB, exist_ok=True)
    tcB = copy.deepcopy(tc)
    tcB['paths']['runs_dir'] = dB
    tcB['run']['initial_model'] = S.label
    logB = DecisionLog(dB)
    logB.info(f"==> Analysis B: tree from the model of the phase 0, "
              f"{S.label}")
    B = analyse(tcB, config, dB, logB, undecided=ph.undecided)
    with open(os.path.join(dB, 'conclusion.json'), 'w') as f:
        json.dump(B, f, indent=2, default=json_default)
    keys = ('conclusion', 'model', 'r_med', 'r_lo1', 'r_hi1', 'r_std',
            'r_95')
    out.update(b={k: B.get(k) for k in keys}, b_dir=dB,
               model_differs=B['model'] != A['model'],
               conclusion_changed=B['conclusion'] != A['conclusion'])
    log.info("\n==> ANALYSIS B (from the phase 0): "
             + ', '.join(f"{k} {fmt(B.get(k))}" for k in keys))
    return out


def analyse(tc, config, runs_dir, log, undecided=()):
    """Tree (+ profile scheme with run: profile_scheme: true) in runs_dir:
    the final result (dict of conclusion.json)."""
    mc = tc.get('moments', {})
    fitter = Fitter(BBCompSepRunner(tc, log),
                    residual_test=mc.get('residual_test', 'aic'),
                    lrt_p=mc.get('lrt_p', 0.3173))
    dt = DecisionTree(tc, fitter, log)
    dt.undecided.extend(undecided)
    result = dt.run()
    log.info("\n==> CONCLUSION")
    for k, v in result.items():
        log.info(f"    {k:<18} {fmt(v)}")
    if not tc['run'].get('profile_scheme', False):
        return result

    # ── profile scheme on the concluded tree (run_profile.py) ─────────────
    with open(os.path.join(runs_dir, 'conclusion_tree.json'), 'w') as f:
        json.dump(result, f, indent=2, default=json_default)
    ps = tc.get('profile_scheme', {})
    pdir = os.path.join(runs_dir, 'profile')
    cmd = [sys.executable,
           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'run_profile.py'),
           '--config', os.path.abspath(config), '--tree-dir', runs_dir,
           '--out', pdir]
    for k in ('n', 'nproc'):
        if k in ps:
            cmd += ['--' + k.replace('_', '-'), str(ps[k])]
    log.info("\n==> Profile scheme: " + ' '.join(cmd[1:]))
    subprocess.run(cmd, check=True)
    prof = json.load(open(os.path.join(pdir, 'result.json')))
    ul = prof['conclusion'] == 'upper_limit'
    rkeys = ('r_med', 'r_lo1', 'r_hi1', 'r_95', 'r_lo3', 'r_hi3', 'r_std')
    final = dict(
        conclusion=prof['conclusion'], model=prof['model_final'],
        r_med=prof['r_med_pl'], r_lo1=prof['r_lo1_pl'],
        r_hi1=prof['r_hi1_pl'], r_std=prof['r_std_pl'],
        r_95=prof['r_95_pl0'] if ul else None, source='profile scheme',
        # choices taken with ln B within seed_band errors (also after the
        # PolyChord repetitions), in the tree and in the profile scheme
        undecided_tree=result.get('undecided', False),
        undecided_profile=prof.get('undecided', False),
        undecided_choices=dict(tree=result.get('undecided_choices', []),
                               profile=prof.get('undecided_choices', [])),
        # final model with PTE < pte_flag: it does not describe the data
        # well (e.g. a foreground outside the moment groups)
        low_pte_tree=result.get('low_pte', False),
        low_pte_profile=prof.get('low_pte', False),
        # Bayesian part: model chosen by the tree, ML best fit, posterior
        # (PolyChord) of all the parameters, conclusion of the tree
        bayesian=dict(prof['bayes'], conclusion=result.get('conclusion')),
        # profile part: final model, posterior of r from the profile (prior
        # of the tree; r_95 with r >= 0), best fit at r = r*, posterior of
        # the other parameters conditioned on r = r* (emcee)
        profiled=dict(model=prof['model_final'], r_star=prof['r_star_final'],
                      r_posterior={k: prof[k + '_pl'] for k in rkeys},
                      r_95_rpos=prof['r_95_pl0'],
                      r_profile=dict(best=prof['r_best'], lo1=prof['r_lo1'],
                                     hi1=prof['r_hi1'],
                                     ul95=prof['r_ul95_profile'],
                                     dchi2_r0=prof['dchi2_r0']),
                      best_fit_at_rstar=prof['best_fit_at_rstar'],
                      chi2_at_rstar=prof['chi2_at_rstar'],
                      posterior_at_rstar=prof['params_at_rstar'],
                      conclusion=prof['conclusion'], start=prof['start'],
                      rejected=prof['rejected']),
        tree=result)
    b = final['bayesian']
    log.info("\n==> BAYESIAN PART (tree): " + b['model'] +
             f", chi2 = {b['chi2_ml']:.1f}, conclusion {b['conclusion']}")
    for n, v in b['posterior'].items():
        log.info(f"    {n:<18} best fit {fmt(b['best_fit'].get(n))}   "
                 f"posterior {fmt(v['med'])} [{fmt(v['lo1'])}, "
                 f"{fmt(v['hi1'])}]")
    p_ = final['profiled']
    log.info("\n==> PROFILED PART: " + p_['model'] +
             f", r* = {fmt(p_['r_star'])}, chi2 = {p_['chi2_at_rstar']:.1f}")
    rp = p_['r_posterior']
    log.info(f"    {'r_tensor':<18} posterior from the profile "
             f"{fmt(rp['r_med'])} [{fmt(rp['r_lo1'])}, {fmt(rp['r_hi1'])}]"
             f", r_95 (r >= 0) {fmt(p_['r_95_rpos'])}")
    for n, v in p_['posterior_at_rstar'].items():
        log.info(f"    {n:<18} best fit {fmt(p_['best_fit_at_rstar'].get(n))}"
                 f"   posterior | r* {fmt(v['med'])} [{fmt(v['lo1'])}, "
                 f"{fmt(v['hi1'])}]")
    log.info("\n==> FINAL CONCLUSION (profile scheme)")
    for k in ('conclusion', 'model', 'r_med', 'r_lo1', 'r_hi1', 'r_95'):
        log.info(f"    {k:<18} {fmt(final[k])}")
    return final


if __name__ == '__main__':
    main()
