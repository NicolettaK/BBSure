"""
Raccoglie i result.json dello schema profilo (PySM_rnd3/set_*/) in un csv.

  python code/collect_profile.py [root] > profile_results.csv

Una riga per set: modello del tree e finale, r bayesiano del tree, posterior
di r dal profilo (prior del tree), r_95 con r >= 0, conclusione, chi2 del
modello finale al best fit con r = r* (senza prior gaussiani), gradi di
liberta' (dati - parametri liberi, r escluso) e PTE. I set senza result.json
(non finiti o falliti) escono con status = missing.
"""
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

root = sys.argv[1] if len(sys.argv) > 1 else '.'
cols = ['index', 'kind', 'status', 'conclusion', 'model_tree', 'model_final',
        'start', 'rejected', 'r_bayes_tree', 'sigma_bayes_tree', 'r_star_final',
        'r_med_pl', 'r_lo1_pl', 'r_hi1_pl', 'r_std_pl', 'r_lo3_pl', 'r_hi3_pl',
        'r_95_pl0', 'dchi2_r0', 'chi2_at_rstar', 'dof_rstar', 'pte_rstar']
w = csv.writer(sys.stdout)
w.writerow(cols)
for d in sorted(glob.glob(os.path.join(root, 'set_*_*'))):
    _, i, kind = os.path.basename(d).split('_', 2)
    f = os.path.join(d, 'result.json')
    if not os.path.isfile(f):
        w.writerow([int(i), kind, 'missing'] + [''] * (len(cols) - 3))
        continue
    r = json.load(open(f))
    r.update(index=int(i), kind=kind, status='done',
             rejected='+'.join(r.get('rejected', [])))
    # numero di dati dal fit ML finale; parametri liberi = best fit senza r
    z = os.path.join(d, f"ml_{r['model_final']}_final", 'chi2.npz')
    if 'chi2_at_rstar' in r and os.path.isfile(z):
        dof = int(np.load(z)['ndof']) - (len(r['best_fit_at_rstar']) - 1)
        r.update(dof_rstar=dof,
                 pte_rstar=float(stats.chi2.sf(r['chi2_at_rstar'], dof)))
    w.writerow(['' if r.get(c) is None else r.get(c) for c in cols])
