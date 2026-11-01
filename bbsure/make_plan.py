"""
Piano della validazione: 100 realizzazioni di CMB + rumore (seed 7000-7099).

Ogni realizzazione ha il suo r:
  - 10 realizzazioni (a caso) con r = 0;
  - le altre 90 con r log-uniforme in [1e-3, 0.03].
Ogni realizzazione si analizza due volte: con d9s4 e con un cielo PySM scelto
a caso tra d9/d10/d12 x s4/s5/s7 (escluso d9s4).

Scrive:
  plan/sets.csv     index, seed, sky_rand        (si puo' guardare)
  truth/r_true.csv  index, r                     (NON aprire fino alla fine)

Uso: python make_plan.py <root>
"""
import csv
import os
import sys

import numpy as np

N, SEED0, N_R0 = 100, 7000, 10
R_MIN, R_MAX = 1e-3, 0.03
SKIES = [f'{d}{s}' for d in ('d9', 'd10', 'd12') for s in ('s4', 's5', 's7')
         if f'{d}{s}' != 'd9s4']

root = sys.argv[1]
rng = np.random.default_rng(20261024)
r = 10 ** rng.uniform(np.log10(R_MIN), np.log10(R_MAX), N)
r[rng.choice(N, N_R0, replace=False)] = 0.
sky = rng.choice(SKIES, N)

os.makedirs(os.path.join(root, 'plan'), exist_ok=True)
os.makedirs(os.path.join(root, 'truth'), exist_ok=True)
with open(os.path.join(root, 'plan', 'sets.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['index', 'seed', 'sky_rand'])
    for i in range(N):
        w.writerow([i, SEED0 + i, sky[i]])
with open(os.path.join(root, 'truth', 'r_true.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['index', 'r'])
    for i in range(N):
        w.writerow([i, f'{r[i]:.6e}'])
print(f"written plan/sets.csv and truth/r_true.csv ({N} sets, "
      f"{N_R0} with r = 0); random skies: "
      + ', '.join(f'{s} x{int((sky == s).sum())}' for s in SKIES))
