"""
Simulated SO SAT observations: CMB + foregrounds (Gaussian, optionally with
spatially varying spectral parameters, or PySM) + SO V3 noise, smoothed with
the SAT beams, written as one map file per split.
"""
import os

import healpy as hp
import numpy as np

from . import noise_calc as nc
from .sky import (BAND_NAMES, get_beta_cls, get_cmb_spectra,
                  get_dust_beta_derivs, get_dust_u_derivs, get_fg_spectra,
                  get_pysm_fg, get_seds, get_synch_beta_derivs)


def generate_maps(seed, nside, output_dir, nsplits=4, r=0.,
                  amp_beta=0., gamma_beta=-3., beta_seed=None,
                  amp_beta_s=0., gamma_beta_s=-3., beta_seed_s=None,
                  amp_u=0., gamma_u=-3., u_seed=None, pysm=None, pysm_nside_in=0,
                  hits_noise=None, rho_bu=0., sens_mode=1, one_over_f=1,
                  kludge=True, no_ee=False, no_cmb=False, no_noise=False):
    """
    Generates and saves the observed maps of all the splits
    (obs_split<i>of<nsplits>.fits.gz in output_dir).

    With pysm (e.g. 'd1s1') dust and synchrotron are the PySM maps
    (get_pysm_fg) instead of the Gaussian foregrounds; the moments do not
    apply. CMB and noise stay those of the seed (same random numbers as in
    the Gaussian case).

    With amp_beta > 0 a Gaussian map delta-beta(n), uncorrelated with the
    dust, with the spectrum of the BBPower moments (get_beta_cls), is added
    and the SED is applied pixel by pixel at second order:
        d_nu(n) = d(n) * [S0 + dbeta(n) * S1 + 1/2 * dbeta(n)^2 * S2]
    which is the model of Azzoni et al. 2021 (1x1 and 0x2 terms). The
    delta-beta map uses a separate RNG (beta_seed, default seed + 100000):
    CMB, dust, synch and noise stay identical to the amp_beta = 0 case with
    the same seed.
    amp_beta_s / gamma_beta_s do the same for the synchrotron (independent
    delta-beta_s map, default seed + 200000).
    amp_u / gamma_u add a map of ut = T_d * delta(1/T) for the dust,
    independent of delta-beta (default seed + 300000), or correlated with it
    with rho_bu != 0 (C^bu = rho sqrt(C^bb C^uu)), with the SED at second
    order in (beta, ut), mixed term delta-beta * delta-ut included.

    Noise: SO V3 SAT, sens_mode 1 baseline / 2 goal, one_over_f 1 optimistic
    / 0 pessimistic; kludge: V3 factor for the non-uniform hits (to remove
    when the noise already follows the hits map, hits_noise, otherwise it
    counts twice). no_ee: test, signal (CMB + foregrounds) without E modes.
    The E modes are never generated: EE = 0 in the synfast spectra (the B
    modes and all the following random numbers stay those of the case with
    E) and the PySM maps lose their E at nside_in, in the harmonic downgrade.
    Removing E from the maps at nside (map2alm up to 3 nside - 1) left part
    of it in B (~0.3% of the CMB BB at ell ~ 80, r ~ +4e-4). Only with the
    delta-beta / delta-u maps (products in pixel space, which make E from B)
    the E modes are still removed at the end, with that approximation.

    no_cmb / no_noise: test, maps without CMB / without noise (the random
    numbers are drawn anyway, so the other components do not change).
    """
    np.random.seed(seed)
    lmax = 3 * nside - 1
    npix = hp.nside2npix(nside)
    cl0 = np.zeros(lmax + 1)
    nfreq = len(BAND_NAMES)

    print(f"  seed={seed}, nside={nside}, lmax={lmax}, r={r}")

    clee_cmb, clbb_cmb = get_cmb_spectra(lmax, r=r)
    cldee, cldbb, clsee, clsbb = get_fg_spectra(lmax)
    if no_ee:
        clee_cmb, cldee, clsee = cl0, cl0, cl0

    _, Qcmb, Ucmb = hp.synfast([cl0, clee_cmb, clbb_cmb, cl0, cl0, cl0],
                               nside, new=True)
    if no_cmb:
        Qcmb, Ucmb = 0. * Qcmb, 0. * Ucmb
        print("  no CMB (--no-cmb)")
    _, Qdust, Udust = hp.synfast([cl0, cldee, cldbb, cl0, cl0, cl0],
                                 nside, new=True)
    _, Qsynch, Usynch = hp.synfast([cl0, clsee, clsbb, cl0, cl0, cl0],
                                   nside, new=True)

    map_comp = np.array([[Qcmb,   Ucmb],
                         [Qdust,  Udust],
                         [Qsynch, Usynch]])

    seds = get_seds()
    print(f"  SED dust:  {np.round(seds[1], 4)}")
    print(f"  SED synch: {np.round(seds[2], 4)}")

    map_freq = np.einsum('ij,ikl->jkl', seds, map_comp)

    if pysm:
        if amp_beta > 0 or amp_beta_s > 0 or amp_u > 0:
            raise ValueError("moments cannot be combined with --pysm")
        # CMB (SED = 1 in uK_CMB) + PySM foregrounds
        map_freq = map_comp[0][None, :, :] + get_pysm_fg(pysm, nside,
                                                         pysm_nside_in,
                                                         no_e=no_ee)
        print(f"  foregrounds: PySM {pysm}")

    # delta-beta maps: (component, amp, gamma, seed, derivatives, file)
    beta_maps = [
        (1, amp_beta, gamma_beta,
         seed + 100000 if beta_seed is None else beta_seed,
         get_dust_beta_derivs, 'dbeta.fits.gz'),
        (2, amp_beta_s, gamma_beta_s,
         seed + 200000 if beta_seed_s is None else beta_seed_s,
         get_synch_beta_derivs, 'dbeta_s.fits.gz'),
    ]
    dbeta_dust = None
    alm_beta_dust, cl_beta_dust = None, None
    for i_c, amp, gamma, bseed, derivs, fname in beta_maps:
        if amp <= 0:
            continue
        cl_beta = get_beta_cls(lmax, amp, gamma)
        state = np.random.get_state()
        np.random.seed(bseed)
        # synalm + alm2map == synfast (same random numbers), but keeps the alm
        alm = hp.synalm(cl_beta, new=True)
        dbeta = hp.alm2map(alm, nside)
        np.random.set_state(state)
        S1, S2 = derivs()
        if i_c == 1:
            dbeta_dust, alm_beta_dust, cl_beta_dust = dbeta, alm, cl_beta
        print(f"  delta-beta map ({'dust' if i_c == 1 else 'synch'}): "
              f"amp={amp}, gamma={gamma}, rms={np.std(dbeta):.4f}, "
              f"expected sigma2={np.sum((2*np.arange(lmax+1)+1)*cl_beta)/(4*np.pi):.3e}")
        # first- and second-order term, pixel by pixel (Q and U)
        for i_f in range(nfreq):
            map_freq[i_f] += map_comp[i_c] * (dbeta * S1[i_f]
                                              + 0.5 * dbeta**2 * S2[i_f])
        os.makedirs(output_dir, exist_ok=True)
        hp.write_map(os.path.join(output_dir, fname), dbeta,
                     overwrite=True, dtype=np.float32)

    # dust temperature map: ut = T_d * delta(1/T); with rho_bu != 0
    # correlated with the dust delta-beta: C^bu = rho sqrt(C^bb C^uu)
    if amp_u > 0:
        cl_u = get_beta_cls(lmax, amp_u, gamma_u)
        state = np.random.get_state()
        np.random.seed(seed + 300000 if u_seed is None else u_seed)
        if rho_bu == 0:
            dut = hp.synfast(cl_u, nside, new=True)
        else:
            if alm_beta_dust is None:
                raise ValueError("--rho-bu needs the dust delta-beta map "
                                 "(--amp-beta)")
            ratio = np.zeros(lmax + 1)
            ok = cl_beta_dust > 0
            ratio[ok] = np.sqrt(cl_u[ok] / cl_beta_dust[ok])
            alm_g = hp.synalm((1 - rho_bu**2) * cl_u, new=True)
            alm_u = hp.almxfl(alm_beta_dust, rho_bu * ratio) + alm_g
            dut = hp.alm2map(alm_u, nside)
            print(f"  delta-u correlated with delta-beta: rho = {rho_bu}, "
                  f"map correlation = {np.corrcoef(dut, dbeta_dust)[0, 1]:.3f}")
        np.random.set_state(state)
        S1u, S2u, Sbu = get_dust_u_derivs()
        print(f"  delta-u map (dust): amp={amp_u}, gamma={gamma_u}, "
              f"rms={np.std(dut):.4f} (~ rms dT/T), "
              f"expected sigma2={np.sum((2*np.arange(lmax+1)+1)*cl_u)/(4*np.pi):.3e}")
        for i_f in range(nfreq):
            corr = dut * S1u[i_f] + 0.5 * dut**2 * S2u[i_f]
            if dbeta_dust is not None:
                corr = corr + dbeta_dust * dut * Sbu[i_f]
            map_freq[i_f] += map_comp[1] * corr
        os.makedirs(output_dir, exist_ok=True)
        hp.write_map(os.path.join(output_dir, 'du.fits.gz'), dut,
                     overwrite=True, dtype=np.float32)

    fwhm_arr = nc.Simons_Observatory_V3_SA_beam_FWHM()
    for i_f, fwhm_arcmin in enumerate(fwhm_arr):
        fwhm_rad = fwhm_arcmin * np.pi / 180. / 60.
        for i_p in [0, 1]:
            map_freq[i_f, i_p, :] = hp.smoothing(map_freq[i_f, i_p, :],
                                                 fwhm=fwhm_rad)

    fsky = 0.1
    nell_arr = np.zeros([nfreq, lmax + 1])
    _, nell_arr[:, 2:], _ = nc.Simons_Observatory_V3_SA_noise(
        sens_mode, one_over_f, 1, fsky, lmax + 1, 1, include_beam=False,
        include_kludge=kludge)
    print(f"  noise: sensitivity_mode={sens_mode}, one_over_f_mode="
          f"{one_over_f}, include_kludge={kludge}")

    # Noise following the hits: sigma(n) = sigma_0 * sqrt(<h> / (fsky h(n))),
    # with sigma_0 the SO level for fsky. Conserves the total observing time
    # (sum 1/sigma^2 as in the uniform case over fsky); zero outside the
    # footprint.
    noise_scale = np.ones(npix)
    if hits_noise:
        h = hp.ud_grade(hp.read_map(hits_noise), nside)
        h = h / h.max()
        h[h < 1e-3] = 0.          # edges of the C -> G rotation (make_mask.py)
        noise_scale = np.zeros(npix)
        good = h > 0
        noise_scale[good] = np.sqrt(np.mean(h) / (fsky * h[good]))
        print(f"  noise scaled with hits ({hits_noise}): <h> = "
              f"{np.mean(h):.4f}, sigma/sigma_0 in "
              f"[{noise_scale[good].min():.2f}, {noise_scale[good].max():.1f}]")

    if no_ee and (amp_beta > 0 or amp_beta_s > 0 or amp_u > 0):
        # Test: signal (CMB + foregrounds) without E modes, to exclude the
        # E -> B leakage of the mask. The noise stays as it is. The E modes
        # are not generated (see above); here only those made from B by the
        # delta-beta / delta-u products, removed at nside (approximate).
        for i_f in range(nfreq):
            alm = hp.map2alm([np.zeros(npix), map_freq[i_f, 0],
                              map_freq[i_f, 1]], lmax=lmax, pol=True)
            alm[1] *= 0.
            map_freq[i_f] = hp.alm2map(alm, nside, lmax=lmax, pol=True)[1:]
        print("  E modes of the delta-beta / delta-u terms removed at "
              "nside (--no-ee)")
    elif no_ee:
        print("  signal without E modes (--no-ee: EE = 0 in the input, "
              "PySM E removed at nside_in)")

    os.makedirs(output_dir, exist_ok=True)

    for i_s in range(nsplits):
        map_noise = np.zeros([nfreq, 2, npix])
        for i_f in range(nfreq):
            _, mpq, mpu = hp.synfast(
                [cl0, nell_arr[i_f], nell_arr[i_f], cl0, cl0, cl0],
                nside, new=True)
            map_noise[i_f, 0, :] = mpq * np.sqrt(nsplits) * noise_scale
            map_noise[i_f, 1, :] = mpu * np.sqrt(nsplits) * noise_scale
        if no_noise:
            map_noise[:] = 0.

        obs = (map_freq + map_noise).reshape([nfreq * 2, npix])
        fname = os.path.join(output_dir,
                             f'obs_split{i_s+1}of{nsplits}.fits.gz')
        hp.write_map(fname, obs, overwrite=True, dtype=np.float32)
        print(f"    written: {fname}")

    print(f"  Seed {seed} done.")
