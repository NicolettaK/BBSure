"""
Collects the results of a validation run in csv tables.

collect_tree    : conclusions of the trees, <root>/runs/set_<i>_<kind>/<tree>/
                  (blind, or with the true r and the statistics: unblind)
collect_profile : result.json of the profile scheme,
                  <root>/runs/set_<i>_<kind>/tree/profile/ (profile scheme
                  run by run_tree.py) or <root>/set_<i>_<kind>/ (run alone)
"""
import csv
import glob
import json
import os

import numpy as np
from scipy import stats

TREE_COLS = ['index', 'kind', 'sky', 'status', 'conclusion', 'model',
             'r_med', 'r_std', 'r_lo1', 'r_hi1', 'r_95', 'lnB_rfree_vs_r0',
             'chi2', 'pte', 'low_pte', 'undecided', 'undecided_choices']
UNBLIND_COLS = ['r_true', 'pull', 'in_1sigma', 'in_95']
PROFILE_COLS = ['index', 'kind', 'status', 'conclusion', 'model_tree',
                'model_final', 'start', 'rejected', 'r_bayes_tree',
                'sigma_bayes_tree', 'r_star_final', 'r_med_pl', 'r_lo1_pl',
                'r_hi1_pl', 'r_std_pl', 'r_lo3_pl', 'r_hi3_pl', 'r_95_pl0',
                'dchi2_r0', 'chi2_at_rstar', 'dof_rstar', 'pte_rstar',
                'low_pte', 'undecided', 'undecided_choices']


def choices_text(choices):
    """Undecided choices in one csv cell: 'chosen>other (lnB+-err); ...'."""
    return '; '.join(f"{c['chosen']}>{c['other']} ({c['lnB']:.2f}+-"
                     f"{c['lnB_err']:.2f})" for c in choices or [])


def collect_tree(root, unblind=False, tree='tree'):
    """
    One row per set: conclusion, final model, r of the tree (median, sigma,
    68%, r_95), chi2 and PTE of the final model. With unblind, also the true
    r (<root>/truth/r_true.csv), the pull and the coverage, and a summary
    printed per sky type. tree: name of the tree directory (e.g.
    tree_bfcov1 for the second pass with the best-fit covariance).
    Writes <root>/results[_unblinded][_<tree>].csv and returns its path.
    """
    sky = {int(r['index']): r['sky_rand']
           for r in csv.DictReader(open(os.path.join(root, 'plan',
                                                     'sets.csv')))}
    rows = []
    for d in sorted(glob.glob(os.path.join(root, 'runs', 'set_*'))):
        _, i, kind = os.path.basename(d).split('_')
        i = int(i)
        row = dict(index=i, kind=kind,
                   sky='d9s4' if kind == 'd9s4' else sky[i],
                   status='running/failed')
        # the tree alone: conclusion_tree.json when the profile scheme
        # followed it (conclusion.json is then the one of the scheme)
        f = os.path.join(d, tree, 'conclusion_tree.json')
        if not os.path.isfile(f):
            f = os.path.join(d, tree, 'conclusion.json')
        if os.path.isfile(f):
            c = json.load(open(f))
            row.update(status='done', conclusion=c['conclusion'],
                       model=c['model'], r_med=c.get('r_med'),
                       r_lo1=c.get('r_lo1'), r_hi1=c.get('r_hi1'),
                       r_95=c.get('r_95'),
                       lnB_rfree_vs_r0=c.get('lnB_rfree_vs_r0'),
                       undecided=int(bool(c.get('undecided'))),
                       low_pte=int(bool(c.get('low_pte'))),
                       undecided_choices=choices_text(
                           c.get('undecided_choices')))
            ev = json.load(open(os.path.join(d, tree, 'decision_log.json')))
            fits = [e for e in ev if e['node'] == 'fit' and
                    e.get('model') == c['model'] and 'r_std' in e]
            if fits:
                row.update(r_std=fits[-1]['r_std'], chi2=fits[-1]['chi2'],
                           pte=fits[-1]['pte'])
        rows.append(row)

    keys = list(TREE_COLS)
    done = [r for r in rows if r['status'] == 'done']
    print(f"{len(done)}/{len(rows)} trees concluded; "
          f"{sum(r['conclusion'] == 'r_detection' for r in done)} r "
          f"detections, "
          f"{sum(r['conclusion'] == 'upper_limit' for r in done)} upper "
          f"limits")

    if unblind:
        truth = {int(r['index']): float(r['r'])
                 for r in csv.DictReader(open(os.path.join(root, 'truth',
                                                           'r_true.csv')))}
        keys += UNBLIND_COLS
        for r in done:
            rt = truth[r['index']]
            r['r_true'] = rt
            # no r when the tree stops at max_iterations
            if r.get('r_std') and r.get('r_med') is not None:
                r['pull'] = (r['r_med'] - rt) / r['r_std']
            if r.get('r_lo1') is not None:
                r['in_1sigma'] = int(r['r_lo1'] <= rt <= r['r_hi1'])
            if r.get('r_95') is not None:
                r['in_95'] = int(rt <= r['r_95'])
        for kind in ('d9s4', 'rand'):
            sel = [r for r in done if r['kind'] == kind]
            pulls = np.array([r['pull'] for r in sel if 'pull' in r])
            det0 = [r for r in sel if r['r_true'] == 0.
                    and r['conclusion'] == 'r_detection']
            print(f"  {kind}: {len(sel)} trees, pull mean {pulls.mean():+.2f}"
                  f" std {pulls.std():.2f}, false detections at r = 0: "
                  f"{len(det0)}/{sum(r['r_true'] == 0. for r in sel)}")

    suffix = '' if tree == 'tree' else '_' + tree
    out = os.path.join(root, ('results_unblinded' if unblind else 'results')
                       + suffix + '.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r['index'], r['kind'])))
    print('written', out)
    return out


def collect_profile(root, out):
    """
    One row per set of the profile scheme (see the module docstring): model
    of the tree and final model, Bayesian r of the tree, posterior of r from
    the profile (prior of the tree), r_95 with r >= 0, conclusion, chi2 of
    the final model at the best fit with r = r* (without Gaussian priors),
    degrees of freedom (data - free parameters, r excluded) and PTE. Sets
    without result.json (not finished or failed) have status = missing.
    Writes the csv `out`.
    """
    with open(out, 'w', newline='') as fo:
        w = csv.writer(fo)
        w.writerow(PROFILE_COLS)
        runs = os.path.join(root, 'runs')
        dirs = (sorted(glob.glob(os.path.join(runs, 'set_*_*')))
                if os.path.isdir(runs) else
                sorted(glob.glob(os.path.join(root, 'set_*_*'))))
        for d in dirs:
            _, i, kind = os.path.basename(d).split('_', 2)
            if os.path.isdir(runs):
                d = os.path.join(d, 'tree', 'profile')
            f = os.path.join(d, 'result.json')
            if not os.path.isfile(f):
                w.writerow([int(i), kind, 'missing']
                           + [''] * (len(PROFILE_COLS) - 3))
                continue
            r = json.load(open(f))
            r.update(index=int(i), kind=kind, status='done',
                     rejected='+'.join(r.get('rejected', [])),
                     undecided=int(bool(r.get('undecided'))),
                     low_pte=int(bool(r.get('low_pte'))),
                     undecided_choices=choices_text(
                         r.get('undecided_choices')))
            # number of data points from the final ML fit; free parameters =
            # best fit without r
            z = os.path.join(d, f"ml_{r['model_final']}_final", 'chi2.npz')
            if 'chi2_at_rstar' in r and os.path.isfile(z):
                dof = (int(np.load(z)['ndof'])
                       - (len(r['best_fit_at_rstar']) - 1))
                r.update(dof_rstar=dof,
                         pte_rstar=float(stats.chi2.sf(r['chi2_at_rstar'],
                                                       dof)))
            w.writerow(['' if r.get(c) is None else r.get(c)
                        for c in PROFILE_COLS])
    print('written', out)
