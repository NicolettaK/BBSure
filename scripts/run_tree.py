#!/usr/bin/env python
"""
Runs the decision tree on one data set and, with run: profile_scheme: true,
the profile scheme on the concluded tree.

  python scripts/run_tree.py --config tree_config.yml

Output in paths: runs_dir: decision_log.{txt,json}, conclusion.json (and
conclusion_tree.json + profile/ with the profile scheme), one directory per
BBCompSep run.
"""
import argparse
import json
import os
import subprocess
import sys

import yaml

from bbsure import BBCompSepRunner, DecisionLog, DecisionTree, Fitter
from bbsure.log import fmt, json_default


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    tc = yaml.safe_load(open(args.config))
    os.makedirs(tc['paths']['runs_dir'], exist_ok=True)
    log = DecisionLog(tc['paths']['runs_dir'])
    fitter = Fitter(BBCompSepRunner(tc, log),
                    hpd_frac=tc['thresholds'].get('hpd_frac', 0.68))
    result = DecisionTree(tc, fitter, log).run()
    log.info("\n==> CONCLUSION")
    for k, v in result.items():
        log.info(f"    {k:<18} {fmt(v)}")
    runs_dir = tc['paths']['runs_dir']
    # conclusion.json only at the very end (nersc/run_one.sh skips the sets
    # that have it): with the profile scheme the tree goes in
    # conclusion_tree.json, so an interrupted scheme is restarted
    if not tc['run'].get('profile_scheme', False):
        with open(os.path.join(runs_dir, 'conclusion.json'), 'w') as f:
            json.dump(result, f, indent=2, default=json_default)
        return

    # ── profile scheme on the concluded tree (run_profile.py) ─────────────
    with open(os.path.join(runs_dir, 'conclusion_tree.json'), 'w') as f:
        json.dump(result, f, indent=2, default=json_default)
    ps = tc.get('profile_scheme', {})
    pdir = os.path.join(runs_dir, 'profile')
    cmd = [sys.executable,
           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'run_profile.py'),
           '--config', os.path.abspath(args.config), '--tree-dir', runs_dir,
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
    with open(os.path.join(runs_dir, 'conclusion.json'), 'w') as f:
        json.dump(final, f, indent=2, default=json_default)


if __name__ == '__main__':
    main()
