#!/usr/bin/env python
"""
Profile scheme on a concluded tree (see bbsure/profile_scheme.py).

  python scripts/run_profile.py --config tree_config.yml --tree-dir <runs_dir>
         --out <dir> [--lnb-res 0] [--lnb-full 0] [--n 37] [--max-it 4]

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
    ap.add_argument('--lnb-res', type=float, default=0.0)
    ap.add_argument('--lnb-full', type=float, default=0.0)
    ap.add_argument('--n', type=int, default=37)
    ap.add_argument('--max-it', type=int, default=4)
    ap.add_argument('--nproc', type=int,
                    default=int(os.environ.get('PROFILE_NPROC', 8)),
                    help='profile points fitted in parallel (one ML each)')
    ap.add_argument('--no-stop', action='store_true',
                    help='diagnostic: after a rejected candidate still run '
                         'Res_wr* on all the other groups')
    ap.add_argument('--ignore-m10', action='store_true',
                    help='test: ignore M_1,0 preferred in the tree '
                         '(detection path)')
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
    ProfileScheme(tc, tree, out, n=a.n, nproc=a.nproc, lnb_res=a.lnb_res,
                  lnb_full=a.lnb_full, max_it=a.max_it, no_stop=a.no_stop,
                  ignore_m10=a.ignore_m10, name=name).run()


if __name__ == '__main__':
    main()
