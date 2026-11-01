"""
Fiducial BB spectra for the Hamimeche-Lewis likelihood: the BBPower model
(CMB with r = 0, A_lens = 1, plus power-law dust and synchrotron, no
moments) evaluated at the foreground parameters used to generate the
Gaussian covariance sims, so that the fiducial is the mean of those sims.

Writes data/cls_fid_<name>.fits (sacc, same tracers and bandpowers as the
data of `--cells`).

Usage: python make_fiducial.py [--fg-params config/fg_params_d1s1_hits.yml]
           [--cells output_d10s5_hits_s6001] [--name d1s1_hits]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import yaml

import decision_tree as dt

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--fg-params', default='config/fg_params_d1s1_hits.yml')
ap.add_argument('--base-config', default='config/base_config_free_tight.yml')
ap.add_argument('--cells', default='output_d10s5_hits_s6001')
ap.add_argument('--name', default='d1s1_hits')
args = ap.parse_args()

fg = yaml.safe_load(open(args.fg_params))
base = yaml.safe_load(open(args.base_config))
cfg = dt.build_config(base, dt.Model(frozenset(), r_free=False),
                      'predicted_spectra')
cs = cfg['BBCompSep']
cs['predict_at_minimum'] = False
cs['predict_to_sacc'] = True
fixed = {'r_tensor': 0., 'beta_d': fg['beta_d'], 'temp_d': fg['T_d'],
         'amp_d_bb': fg['A_d_BB'], 'alpha_d_bb': fg['alpha_d'],
         'beta_s': fg['beta_s'], 'amp_s_bb': fg['A_s_BB'],
         'alpha_s_bb': fg['alpha_s'], 'epsilon_ds': 0.}
entries = dt.iter_param_entries(cs)
for k, v in fixed.items():
    entries[k][1:] = ['fixed', [float(v)]]
assert dt.free_param_names(cs) == [], dt.free_param_names(cs)

tmp = tempfile.mkdtemp()
cfg_path = os.path.join(tmp, 'config.yml')
yaml.safe_dump(cfg, open(cfg_path, 'w'), sort_keys=False)
cells = os.path.join(HERE, args.cells)
cmd = [sys.executable, '-m', 'bbpower', 'BBCompSep',
       f"--cells_coadded={cells}/cells_coadded.fits",
       f"--cells_noise={cells}/cells_noise.fits",
       f"--cells_fiducial={HERE}/data/cls_fid.fits",
       f"--cells_coadded_cov={cells}/cells_coadded.fits",
       f"--output_dir={tmp}", f"--config_copy={tmp}/cc.yml",
       f"--config={cfg_path}"]
subprocess.run(cmd, check=True)
out = os.path.join(HERE, 'data', f'cls_fid_{args.name}.fits')
shutil.copy(os.path.join(tmp, 'cells_model.fits'), out)
print(f"written {out}")
