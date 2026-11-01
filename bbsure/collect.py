"""
Raccoglie le conclusioni della validazione.

  python collect.py <root>             tabella alla cieca (senza gli r veri)
  python collect.py <root> --unblind   aggiunge gli r veri e le statistiche

Scrive <root>/results.csv (e results_unblinded.csv con --unblind).
"""
import csv
import glob
import json
import os
import sys

import numpy as np

root = sys.argv[1]
unblind = '--unblind' in sys.argv
# --tree tree_bfcov1: risultati del secondo passaggio (default: tree)
tree = (sys.argv[sys.argv.index('--tree') + 1] if '--tree' in sys.argv
        else 'tree')
sky = {int(r['index']): r['sky_rand']
       for r in csv.DictReader(open(os.path.join(root, 'plan', 'sets.csv')))}

rows = []
for d in sorted(glob.glob(os.path.join(root, 'runs', 'set_*'))):
    _, i, kind = os.path.basename(d).split('_')
    i = int(i)
    row = dict(index=i, kind=kind, sky='d9s4' if kind == 'd9s4' else sky[i],
               status='running/failed')
    f = os.path.join(d, tree, 'conclusion.json')
    if os.path.isfile(f):
        c = json.load(open(f))
        row.update(status='done', conclusion=c['conclusion'],
                   model=c['model'], r_med=c.get('r_med'),
                   r_lo1=c.get('r_lo1'), r_hi1=c.get('r_hi1'),
                   r_95=c.get('r_95'),
                   lnB_rfree_vs_r0=c.get('lnB_rfree_vs_r0'))
        ev = json.load(open(os.path.join(d, tree, 'decision_log.json')))
        fits = [e for e in ev if e['node'] == 'fit' and
                e.get('model') == c['model'] and 'r_std' in e]
        if fits:
            row.update(r_std=fits[-1]['r_std'], chi2=fits[-1]['chi2'],
                       pte=fits[-1]['pte'])
    rows.append(row)

keys = ['index', 'kind', 'sky', 'status', 'conclusion', 'model', 'r_med',
        'r_std', 'r_lo1', 'r_hi1', 'r_95', 'lnB_rfree_vs_r0', 'chi2', 'pte']
done = [r for r in rows if r['status'] == 'done']
print(f"{len(done)}/{len(rows)} trees concluded; "
      f"{sum(r['conclusion'] == 'r_detection' for r in done)} r detections, "
      f"{sum(r['conclusion'] == 'upper_limit' for r in done)} upper limits")

if unblind:
    truth = {int(r['index']): float(r['r'])
             for r in csv.DictReader(open(os.path.join(root, 'truth',
                                                       'r_true.csv')))}
    keys += ['r_true', 'pull', 'in_1sigma', 'in_95']
    for r in done:
        rt = truth[r['index']]
        r['r_true'] = rt
        if r.get('r_std'):
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
        print(f"  {kind}: {len(sel)} trees, pull mean {pulls.mean():+.2f} "
              f"std {pulls.std():.2f}, false detections at r = 0: "
              f"{len(det0)}/{sum(r['r_true'] == 0. for r in sel)}")

suffix = '' if tree == 'tree' else '_' + tree
out = os.path.join(root, ('results_unblinded' if unblind else 'results')
                   + suffix + '.csv')
with open(out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r['index'], r['kind'])))
print('written', out)
