#!/usr/bin/env python
"""
Simulated SO SAT maps of one realization (see bbsure/sims/maps.py).

Validation data (PySM foregrounds, noise following the hits):
  python scripts/generate_maps.py --seed 7047 --nside 128 --r 0.02 \\
      --pysm d10s5 --pysm-nside-in 512 --noise pessimistic_goal \\
      --no-kludge --hits-noise data/masks/norm_nHits_SA_35FOV_G_nside128.fits \\
      --no-ee --output-dir sims/s7047_d10s5

Gaussian covariance sims: --fg-params configs/fg_params_d1s1_hits_n128.yml
instead of --pysm. Gaussian sky with a dust moment: --amp-beta 0.5.
"""
import argparse

from bbsure.sims.maps import generate_maps
from bbsure.sims.sky import load_fg_params


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--seed',       type=int,   default=5001)
    parser.add_argument('--nside',      type=int,   default=64)
    parser.add_argument('--nsplits',    type=int,   default=4)
    parser.add_argument('--r',          type=float, default=0.0)
    parser.add_argument('--output-dir', type=str,   required=True)

    grp2 = parser.add_argument_group(
        'Spectral parameter maps (Gaussian foregrounds only)',
        'C_ell = 1e-6 * amp * (ell/80)^gamma for ell >= 2; amp = 0 -> no map.')
    grp2.add_argument('--amp-beta',   type=float, default=0.0,
                      help='amplitude of the dust delta-beta spectrum')
    grp2.add_argument('--gamma-beta', type=float, default=-3.0,
                      help='slope of the dust delta-beta spectrum')
    grp2.add_argument('--beta-seed',  type=int,   default=None,
                      help='seed of the delta-beta map (default seed+100000)')
    grp2.add_argument('--amp-beta-s',   type=float, default=0.0,
                      help='as --amp-beta for the synchrotron')
    grp2.add_argument('--gamma-beta-s', type=float, default=-3.0,
                      help='as --gamma-beta for the synchrotron')
    grp2.add_argument('--beta-seed-s',  type=int,   default=None,
                      help='seed of the delta-beta_s map (default seed+200000)')
    grp2.add_argument('--amp-u',   type=float, default=0.0,
                      help='map of ut = T_d*delta(1/T) of the dust: amplitude')
    grp2.add_argument('--gamma-u', type=float, default=-3.0,
                      help='slope of the ut spectrum')
    grp2.add_argument('--u-seed',  type=int,   default=None,
                      help='seed of the delta-ut map (default seed+300000)')
    grp2.add_argument('--rho-bu',  type=float, default=0.0,
                      help='correlation of delta-ut with the dust delta-beta '
                           '(C^bu = rho sqrt(C^bb C^uu))')

    grp3 = parser.add_argument_group('Foregrounds and noise')
    grp3.add_argument('--pysm', type=str, default=None,
                      help="PySM preset strings for dust and synchrotron, "
                           "e.g. d1s1 (replaces the Gaussian foregrounds)")
    grp3.add_argument('--pysm-nside-in', type=int, default=0,
                      help='run PySM at this nside and downgrade in harmonic '
                           'space (band-limited, no pixel window, like CMB '
                           'and noise); 0 = PySM directly at --nside')
    grp3.add_argument('--fg-params', type=str, default=None,
                      help='yaml file overriding the Gaussian foreground '
                           'parameters (amplitudes, slopes and SEDs)')
    grp3.add_argument('--noise', default='baseline_optimistic',
                      choices=['baseline_optimistic', 'pessimistic_goal'],
                      help='SO V3 SAT noise: sensitivity and 1/f mode')
    grp3.add_argument('--hits-noise', type=str, default=None,
                      help='hits map: scale the noise as 1/sqrt(hits), '
                           'conserving the total observing time')
    grp3.add_argument('--no-kludge', action='store_true',
                      help='include_kludge=False in noise_calc (use with '
                           '--hits-noise: the hits map already gives the '
                           'non-uniform depth)')
    grp3.add_argument('--no-ee', action='store_true',
                      help='remove the E modes of the signal (CMB + '
                           'foregrounds) before adding noise')
    grp3.add_argument('--no-cmb', action='store_true',
                      help='test: maps without CMB')
    grp3.add_argument('--no-noise', action='store_true',
                      help='test: maps without noise')

    args = parser.parse_args()
    if args.fg_params:
        load_fg_params(args.fg_params)

    generate_maps(
        seed          = args.seed,
        nside         = args.nside,
        output_dir    = args.output_dir,
        nsplits       = args.nsplits,
        r             = args.r,
        amp_beta      = args.amp_beta,
        gamma_beta    = args.gamma_beta,
        beta_seed     = args.beta_seed,
        amp_beta_s    = args.amp_beta_s,
        gamma_beta_s  = args.gamma_beta_s,
        beta_seed_s   = args.beta_seed_s,
        amp_u         = args.amp_u,
        gamma_u       = args.gamma_u,
        u_seed        = args.u_seed,
        pysm          = args.pysm,
        pysm_nside_in = args.pysm_nside_in,
        hits_noise    = args.hits_noise,
        rho_bu        = args.rho_bu,
        sens_mode     = 2 if args.noise == 'pessimistic_goal' else 1,
        one_over_f    = 0 if args.noise == 'pessimistic_goal' else 1,
        kludge        = not args.no_kludge,
        no_ee         = args.no_ee,
        no_cmb        = args.no_cmb,
        no_noise      = args.no_noise,
    )


if __name__ == '__main__':
    main()
