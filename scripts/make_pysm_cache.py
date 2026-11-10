"""
PySM maps of the validation skies at nside 128 without E modes, for
generate_maps.py --pysm <sky> --pysm-nside-in 512 --no-ee: PySM at nside 512
(<sky>_nside512.npy, reused if already there), then harmonic downgrade to
128 with E removed (<sky>_nside128_from512_noE.npy), in
$BBSURE_DATA/pysm_cache. Run once (login node, ~minutes per sky).

  python scripts/make_pysm_cache.py [d9s4 d10s5 ...]   (default: the 9 skies)
"""
import sys

from bbsure.sims.sky import get_pysm_fg

skies = sys.argv[1:] or [d + s for d in ('d9', 'd10', 'd12')
                         for s in ('s4', 's5', 's7')]
for sky in skies:
    print(f"==> {sky}", flush=True)
    get_pysm_fg(sky, 128, 512, no_e=True)
print("done")
