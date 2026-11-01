#!/usr/bin/env python
"""
Best-fit covariance for a data set (see bbsure/covariance.py).

  python scripts/make_bf_cov.py --tree <runs_dir of the tree>
      --cells <spectra dir> --fiducial <cls_fid_...fits> --out <dir>
"""
import argparse

from bbsure.covariance import bestfit_covariance


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--tree', required=True)
    ap.add_argument('--cells', required=True)
    ap.add_argument('--fiducial', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    bestfit_covariance(args.tree, args.cells, args.fiducial, args.out)


if __name__ == '__main__':
    main()
