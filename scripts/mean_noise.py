#!/usr/bin/env python
"""
Replaces the noise spectra of a data set with the mean of the noise spectra
of other sets (same binning and tracers), e.g. for the Asimov
foreground-only data (maps without noise: their own noise estimate is ~0,
which breaks the H&L likelihood, whose offset is the noise). The covariance
of the target file is kept.

  python scripts/mean_noise.py <target cells_noise.fits> <cells_noise.fits of other sets...>
"""
import sys

import numpy as np
import sacc


def main():
    target, sources = sys.argv[1], sys.argv[2:]
    t = sacc.Sacc.load_fits(target)
    mean = np.zeros(len(t.mean))
    for f in sources:
        s = sacc.Sacc.load_fits(f)
        assert len(s.mean) == len(mean), f
        mean += s.mean
    mean /= len(sources)
    for d, v in zip(t.data, mean):
        d.value = v
    t.save_fits(target, overwrite=True)
    print(f"noise of {target}: mean of {len(sources)} sets")


if __name__ == '__main__':
    main()
