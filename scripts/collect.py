#!/usr/bin/env python
"""
Collects the results of a validation run (see bbsure/collect.py).

  python scripts/collect.py tree <root>              blind table of the trees
  python scripts/collect.py tree <root> --unblind    with the true r (ONLY at the end)
  python scripts/collect.py profile <root> --out profile_results.csv
"""
import argparse

from bbsure.collect import collect_profile, collect_tree


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('what', choices=['tree', 'profile'])
    ap.add_argument('root')
    ap.add_argument('--unblind', action='store_true')
    ap.add_argument('--tree', default='tree',
                    help='tree directory (e.g. tree_bfcov1 for the second '
                         'pass with the best-fit covariance)')
    ap.add_argument('--out', default=None,
                    help='output csv of the profile scheme')
    a = ap.parse_args()
    if a.what == 'tree':
        collect_tree(a.root, unblind=a.unblind, tree=a.tree)
    else:
        if not a.out:
            ap.error('profile needs --out')
        collect_profile(a.root, a.out, tree=a.tree)


if __name__ == '__main__':
    main()
