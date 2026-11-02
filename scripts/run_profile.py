#!/usr/bin/env python
"""
Profile scheme on a concluded tree (see bbsure/profile_scheme.py).

  python scripts/run_profile.py --config tree_config.yml --tree-dir <runs_dir>
         --out <dir> [--n 37]

or, on the layout of the validation (nersc/):

  python scripts/run_profile.py <src_root> <out_root> <index> <d9s4|rand>

which reads <src_root>/runs/set_<i>_<kind>/{tree_config.yml,tree/} and
writes in <out_root>/set_<i>_<kind>/.
"""
import argparse
import os

import yaml

from bbsure.profile_scheme import ProfileScheme


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('src', nargs='?')
    ap.add_argument('out', nargs='?')
    ap.add_argument('index', type=int, nargs='?')
    ap.add_argument('kind', nargs='?')
    # inside the pipeline (run_tree.py with run: profile_scheme: true)
    ap.add_argument('--config', help='tree_config of the tree')
    ap.add_argument('--tree-dir', help='runs_dir of the concluded tree')
    ap.add_argument('--out', dest='out_dir', help='output directory')
    ap.add_argument('--n', type=int, default=37,
                    help='points of the initial profile grid')
    ap.add_argument('--nproc', type=int,
                    default=int(os.environ.get('PROFILE_NPROC', 8)),
                    help='profile points fitted in parallel (one ML each)')
    a = ap.parse_args()

    if a.config:
        name = os.path.basename(os.path.normpath(a.tree_dir))
        tree, out, cfg_path = a.tree_dir, a.out_dir, a.config
    else:
        name = f'set_{a.index:03d}_{a.kind}'
        src = os.path.join(a.src, 'runs', name)
        tree = os.path.join(src, 'tree')
        out = os.path.join(a.out, name)
        cfg_path = os.path.join(src, 'tree_config.yml')

    tc = yaml.safe_load(open(cfg_path))
    ProfileScheme(tc, tree, out, n=a.n, nproc=a.nproc, name=name).run()


if __name__ == '__main__':
    main()
