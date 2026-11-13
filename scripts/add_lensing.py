#!/usr/bin/env python
"""
Adds the expected lensing BB spectrum (A_lens = 1, r = 0: the BB column of
camb_lens_nobb.dat, the lensing template of the model) to every BB
cross-spectrum of a data file, binned with its bandpower windows (which take
C_ell to the binned D_ell of the data). With maps made with --no-cmb
--no-noise this gives an Asimov data set: foregrounds + mean lensing, no
CMB or noise realization (no noise bias in the cross-split spectra).

  python scripts/add_lensing.py <cells_coadded.fits>     (overwritten)
  python scripts/add_lensing.py <cells_coadded.fits> --tensor-r 0.01
      adds r x the tensor BB spectrum (camb_lens_r1 - camb_lens_nobb)
      instead of the lensing
"""
import os
import sys

import numpy as np
import sacc


def main():
    fname = sys.argv[1]
    r = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == '--tensor-r' else None
    data = os.environ.get('BBSURE_DATA',
                          os.path.join(os.path.dirname(__file__), '..', 'data'))
    cam = np.loadtxt(os.path.join(data, 'camb', 'camb_lens_nobb.dat'))
    s = sacc.Sacc.load_fits(fname)
    ell = cam[:, 0].astype(int)
    lmax = ell.max()
    cl = np.zeros(lmax + 1)
    ok = ell >= 2
    dl = cam[ok, 3]
    if r is not None:
        r1 = np.loadtxt(os.path.join(data, 'camb', 'camb_lens_r1.dat'))
        assert np.array_equal(r1[:, 0].astype(int), ell)
        dl = r * (r1[ok, 3] - cam[ok, 3])
    cl[ell[ok]] = dl * 2 * np.pi / (ell[ok] * (ell[ok] + 1))
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
    what = f"tensor (r = {r})" if r is not None else "lensing (A_lens = 1)"
    print(f"{what} added to {n} BB spectra of {fname}")


if __name__ == '__main__':
    main()
