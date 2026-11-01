"""
Hits-weighted analysis mask for SO SAT.

    w(n) = apodize_C2(hits > 0, apo_deg) * hits(n)

with hits normalized to max 1 and set to 0 below --hmin (the rotation
to galactic coordinates interpolates and leaves tiny values at the
edges). The hits map must be in GALACTIC coordinates, like PySM:
data/masks/norm_nHits_SA_35FOV_G.fits is the SO map (equatorial, COORDSYS=C)
rotated with healpy.Rotator(coord=['C', 'G']). The
weight is ~ the inverse noise variance of the hits-scaled noise of
generate_maps.py (--hits-noise), so the spectra are inverse-variance
weighted inside the footprint.

Usage: python scripts/make_mask.py [--apo 7]
           [--hits norm_nHits_SA_35FOV_G_nside128.fits] [--mask-dir data/masks]
Writes <mask-dir>/mask_hits_C2_<apo>deg.fits.gz (at nside 64; at other nside
the name ends in _nside<N>, e.g. mask_hits_C2_7deg_nside128.fits.gz)
"""
import argparse
import os
import numpy as np
import healpy as hp
import pymaster as nmt

from bbsure.sims.sky import DATA_DIR

p = argparse.ArgumentParser()
p.add_argument('--apo', type=float, default=7.0)
p.add_argument('--hits', default='norm_nHits_SA_35FOV_G.fits')
p.add_argument('--hmin', type=float, default=1e-3)
p.add_argument('--mask-dir', default=os.path.join(DATA_DIR, 'masks'))
args = p.parse_args()
MASK_DIR = args.mask_dir

hits = hp.read_map(os.path.join(MASK_DIR, args.hits))
hits = hits / hits.max()
hits[hits < args.hmin] = 0.
apo = nmt.mask_apodization((hits > 0).astype(float), args.apo, apotype='C2')
w = apo * hits

nside = hp.get_nside(hits)
suffix = '' if nside == 64 else f'_nside{nside}'
out = os.path.join(MASK_DIR, f'mask_hits_C2_{args.apo:g}deg{suffix}.fits.gz')
hp.write_map(out, w, overwrite=True, dtype=np.float32)
fsky_w2 = np.mean(w**2)**2 / np.mean(w**4)
print(f"written {out}")
print(f"  <w> = {np.mean(w):.4f}, fsky (<w^2>^2/<w^4>) = {fsky_w2:.4f}, "
      f"footprint fraction = {np.mean(w > 0):.4f}")
