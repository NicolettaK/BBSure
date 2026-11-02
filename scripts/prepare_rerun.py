#!/usr/bin/env python
"""
New validation root from a concluded one, to rerun with a new version of the
tree only the sets it can change, reusing all the fits already done.

  python scripts/prepare_rerun.py <old root> <new root> [--all] [--dry-run]

Every set of <old root>/runs is copied in <new root>/runs (the spectra in
output/ as a symbolic link). The sets whose last Res_r0 detected at least
one family (the only ones where M_1,0 and the combinations of the profile
scheme enter) are reset: conclusion files and decision logs of tree and
profile scheme removed, the fits kept (same config -> reused). The others
keep their conclusion and are skipped by nersc/run_one.sh. With --all every
set is reset. plan/, truth/ and cov/ are linked; truth is not read.
"""
import argparse
import json
import os
import shutil

from bbsure.profile_scheme import res_r0_candidates

RESET = ['conclusion.json', 'conclusion_tree.json', 'decision_log.json',
         'decision_log.txt', 'tree_path.pdf', 'tree_path.png']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('old')
    ap.add_argument('new')
    ap.add_argument('--all', action='store_true',
                    help='reset every set, not only those with Res_r0 '
                         'detections')
    ap.add_argument('--dry-run', action='store_true',
                    help='only print the sets that would be rerun')
    a = ap.parse_args()
    old, new = os.path.abspath(a.old), os.path.abspath(a.new)

    sets = sorted(d for d in os.listdir(os.path.join(old, 'runs'))
                  if d.startswith('set_'))
    rerun = []
    for s in sets:
        f = os.path.join(old, 'runs', s, 'tree', 'decision_log.json')
        if not os.path.isfile(f):
            print(f"{s}: no decision log, rerun")
            rerun.append(s)
            continue
        if a.all or res_r0_candidates(json.load(open(f))):
            rerun.append(s)
    print(f"{len(rerun)}/{len(sets)} sets to rerun: {' '.join(rerun)}")
    if a.dry_run:
        return

    os.makedirs(os.path.join(new, 'runs'), exist_ok=True)
    os.makedirs(os.path.join(new, 'logs'), exist_ok=True)
    for d in ('plan', 'truth', 'cov'):
        if not os.path.exists(os.path.join(new, d)):
            os.symlink(os.path.join(old, d), os.path.join(new, d))
    for s in sets:
        src, dst = os.path.join(old, 'runs', s), os.path.join(new, 'runs', s)
        if os.path.exists(dst):
            print(f"{s}: already in {new}, left as it is")
            continue
        os.makedirs(dst)
        os.symlink(os.path.join(src, 'output'), os.path.join(dst, 'output'))
        shutil.copytree(os.path.join(src, 'tree'), os.path.join(dst, 'tree'),
                        symlinks=True)
        if s in rerun:
            for sub in ('', 'profile'):
                for f in RESET + ['result.json']:
                    p = os.path.join(dst, 'tree', sub, f)
                    if os.path.isfile(p):
                        os.remove(p)
    print(f"ready: {new}")


if __name__ == '__main__':
    main()
