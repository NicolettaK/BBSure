"""
Covariance of the data vector from the 500 covariance sims, computed once:
  python fix_cov.py save <cells_coadded.fits> <cells_noise.fits> <dir>
saves it (from a set summarized with all the sims); then, for a set
summarized with only 2 sims (the covariance depends only on the sims, not on
the data),
  python fix_cov.py put <dir> <cells_coadded.fits> <cells_noise.fits>
replaces the covariance of its two files with the saved one.
"""
import sys
import numpy as np
import sacc

mode = sys.argv[1]
if mode == 'save':
    for f, k in zip(sys.argv[2:4], ('coadded', 'noise')):
        s = sacc.Sacc.load_fits(f)
        np.save(f"{sys.argv[4]}/cov_{k}.npy", s.covariance.covmat)
        np.save(f"{sys.argv[4]}/n_{k}.npy", len(s.mean))
elif mode == 'put':
    for f, k in zip(sys.argv[3:5], ('coadded', 'noise')):
        s = sacc.Sacc.load_fits(f)
        cov = np.load(f"{sys.argv[2]}/cov_{k}.npy")
        assert len(s.mean) == int(np.load(f"{sys.argv[2]}/n_{k}.npy")) == len(cov)
        s.add_covariance(cov, overwrite=True)
        s.save_fits(f, overwrite=True)
