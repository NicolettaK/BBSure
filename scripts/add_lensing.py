#!/usr/bin/env python
"""
Adds the expected lensing BB spectrum (A_lens = 1, r = 0: the BB column of
camb_lens_nobb.dat, the lensing template of the model) to every BB
cross-spectrum of a data file, binned with its bandpower windows (which take
C_ell to the binned D_ell of the data). With maps made with --no-cmb
--no-noise this gives an Asimov data set: foregrounds + mean lensing, no
CMB or noise realization (no noise bias in the cross-split spectra).

  python scripts/add_lensing.py <cells_coadded.fits>     (overwritten)
"""
import os
import sys

import numpy as np
import sacc


def main():
    fname = sys.argv[1]
    data = os.environ.get('BBSURE_DATA',
                          os.path.join(os.path.dirname(__file__), '..', 'data'))
    cam = np.loadtxt(os.path.join(data, 'camb', 'camb_lens_nobb.dat'))
    s = sacc.Sacc.load_fits(fname)
    ell = cam[:, 0].astype(int)
    lmax = ell.max()
    cl = np.zeros(lmax + 1)
    ok = ell >= 2
    cl[ell[ok]] = cam[ok, 3] * 2 * np.pi / (ell[ok] * (ell[ok] + 1))
    n = 0
    for t1, t2 in s.get_tracer_combinations():
        ind = s.indices('cl_bb', (t1, t2))
        if len(ind) == 0:
            continue
        w = s.get_bandpower_windows(ind)
        lens = w.weight.T @ cl[w.values]
        for i, v in zip(ind, lens):
            s.data[i].value += v
        n += 1
    s.save_fits(fname, overwrite=True)
    print(f"lensing (A_lens = 1) added to {n} BB spectra of {fname}")


if __name__ == '__main__':
    main()
