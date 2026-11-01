"""
Fiducial BB spectra for the Hamimeche-Lewis likelihood: the BBPower model
(CMB with r = 0, A_lens = 1, plus power-law dust and synchrotron, no
moments) evaluated at the foreground parameters used to generate the
Gaussian covariance sims, so that the fiducial is the mean of those sims.

Writes <out> (sacc, same tracers and bandpowers as the spectra in
`--cells`).

Usage: python scripts/make_fiducial.py
           --fg-params configs/fg_params_d1s1_hits_n128.yml
           --base-config configs/base_config_tree.yml --cells <spectra dir>
           --out data/cls_fid_d1s1_hits_pg_n128.fits
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import yaml

from bbsure.config import build_config, free_param_names, iter_param_entries
from bbsure.models import Model
from bbsure.paths import load_yaml

ap = argparse.ArgumentParser()
ap.add_argument('--fg-params', required=True)
ap.add_argument('--base-config', required=True)
ap.add_argument('--cells', required=True,
                help='directory with cells_coadded.fits and cells_noise.fits')
ap.add_argument('--fiducial-in', default=None,
                help='any fiducial sacc with the same tracers, needed by '
                     'BBCompSep to start (default: cells_coadded.fits)')
ap.add_argument('--out', required=True)
args = ap.parse_args()

fg = yaml.safe_load(open(args.fg_params))
base = load_yaml(args.base_config)
cfg = build_config(base, Model(frozenset(), r_free=False),
                   'predicted_spectra')
cs = cfg['BBCompSep']
cs['predict_at_minimum'] = False
cs['predict_to_sacc'] = True
fixed = {'r_tensor': 0., 'beta_d': fg['beta_d'], 'temp_d': fg['T_d'],
         'amp_d_bb': fg['A_d_BB'], 'alpha_d_bb': fg['alpha_d'],
         'beta_s': fg['beta_s'], 'amp_s_bb': fg['A_s_BB'],
         'alpha_s_bb': fg['alpha_s'], 'epsilon_ds': 0.}
entries = iter_param_entries(cs)
for k, v in fixed.items():
    entries[k][1:] = ['fixed', [float(v)]]
assert free_param_names(cs) == [], free_param_names(cs)

tmp = tempfile.mkdtemp()
cfg_path = os.path.join(tmp, 'config.yml')
yaml.safe_dump(cfg, open(cfg_path, 'w'), sort_keys=False)
cells = os.path.abspath(args.cells)
fid_in = args.fiducial_in or f"{cells}/cells_coadded.fits"
cmd = [sys.executable, '-m', 'bbpower', 'BBCompSep',
       f"--cells_coadded={cells}/cells_coadded.fits",
       f"--cells_noise={cells}/cells_noise.fits",
       f"--cells_fiducial={fid_in}",
       f"--cells_coadded_cov={cells}/cells_coadded.fits",
       f"--output_dir={tmp}", f"--config_copy={tmp}/cc.yml",
       f"--config={cfg_path}"]
subprocess.run(cmd, check=True)
out = args.out
shutil.copy(os.path.join(tmp, 'cells_model.fits'), out)
print(f"written {out}")
