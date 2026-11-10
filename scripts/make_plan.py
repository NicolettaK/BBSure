"""
Plan of the validation: 100 realizations of CMB + noise (seeds seed0 ...
seed0 + 99; default 7000-7099, the 200 sets of the first validation).

Each realization has its own r:
  - 10 realizations (at random) with r = 0;
  - the other 90 with r log-uniform in [r_min, 0.03] (r_min: 1e-3 by
    default, as in the first validations).
Each realization is analysed twice: with d9s4 and with a PySM sky chosen at
random among d9/d10/d12 x s4/s5/s7 (d9s4 excluded).

Writes:
  plan/sets.csv     index, seed, sky_rand        (can be looked at)
  truth/r_true.csv  index, r                     (do NOT open until the end)

Usage: python scripts/make_plan.py <root> [--seed0 7000] [--rng-seed 20261024]
                                         [--r-min 1e-3] [--n 100] [--n-r0 10]
                                         [--first-index 0] [--skies ...]
  --seed0     seed of the first CMB + noise realization (the covariance sims
              use 5002-5501: do not overlap them)
  --rng-seed  seed of the draws of r, of the sets with r = 0 and of the
              random skies
  --r-min     lower edge of the log-uniform r (same rng sequence: with the
              same rng-seed only the values of r change, not the r = 0 sets
              nor the skies)
  --n, --n-r0 number of realizations and how many of them with r = 0
  --first-index  index of the first set (seed = seed0 + index), e.g. 100 to
              add sets 100-119 to a run in a new root
  --skies     skies drawn for the random analysis (default: the 8 PySM
              skies above), e.g. d10x2s4 d10x2s5 d10x2s7
"""
import argparse
import csv
import os

import numpy as np

N, N_R0 = 100, 10
R_MIN, R_MAX = 1e-3, 0.03
SKIES = [f'{d}{s}' for d in ('d9', 'd10', 'd12') for s in ('s4', 's5', 's7')
         if f'{d}{s}' != 'd9s4']

ap = argparse.ArgumentParser()
ap.add_argument('root')
ap.add_argument('--seed0', type=int, default=7000)
ap.add_argument('--rng-seed', type=int, default=20261024)
ap.add_argument('--r-min', type=float, default=R_MIN)
ap.add_argument('--n', type=int, default=N)
ap.add_argument('--n-r0', type=int, default=N_R0)
ap.add_argument('--first-index', type=int, default=0)
ap.add_argument('--skies', nargs='+', default=SKIES)
a = ap.parse_args()
root, SEED0, R_MIN = a.root, a.seed0, a.r_min
N, N_R0, I0, SKIES = a.n, a.n_r0, a.first_index, a.skies
rng = np.random.default_rng(a.rng_seed)
r = 10 ** rng.uniform(np.log10(R_MIN), np.log10(R_MAX), N)
r[rng.choice(N, N_R0, replace=False)] = 0.
sky = rng.choice(SKIES, N)

os.makedirs(os.path.join(root, 'plan'), exist_ok=True)
os.makedirs(os.path.join(root, 'truth'), exist_ok=True)
# lines ending with \n only (the csv default \r\n leaves a \r on the last
# field when the files are read with awk, e.g. the sky name)
with open(os.path.join(root, 'plan', 'sets.csv'), 'w', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerow(['index', 'seed', 'sky_rand'])
    for i in range(N):
        w.writerow([I0 + i, SEED0 + I0 + i, sky[i]])
with open(os.path.join(root, 'truth', 'r_true.csv'), 'w', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerow(['index', 'r'])
    for i in range(N):
        w.writerow([I0 + i, f'{r[i]:.6e}'])
print(f"written plan/sets.csv and truth/r_true.csv ({N} sets, indices "
      f"{I0}-{I0 + N - 1}, seeds {SEED0 + I0}-{SEED0 + I0 + N - 1}, "
      f"{N_R0} with r = 0, the others log-uniform in [{R_MIN:g}, {R_MAX:g}]); "
      f"random skies: "
      + ', '.join(f'{s} x{int((sky == s).sum())}' for s in SKIES))
