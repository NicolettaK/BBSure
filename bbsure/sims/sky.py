"""
Sky model of the simulations: SEDs in K_RJ, SO SAT bandpasses, CMB and
Gaussian foreground power spectra, SED derivatives for the moment maps and
PySM foregrounds integrated over the bandpasses.

Data files (bandpasses, CAMB spectra, PySM cache) are read from DATA_DIR
(bbsure/paths.py).
"""
import os
import re

import healpy as hp
import numpy as np

from ..paths import DATA_DIR
CAMB_NOBB = os.path.join(DATA_DIR, 'camb', 'camb_lens_nobb.dat')
CAMB_R1 = os.path.join(DATA_DIR, 'camb', 'camb_lens_r1.dat')
BPASS_DIR = os.path.join(DATA_DIR, 'bandpasses')

BAND_NAMES = ['LF1', 'LF2', 'MF1', 'MF2', 'UHF1', 'UHF2']

# ── Physical constants ────────────────────────────────────────────────────
H_OVER_KB_CMB = 0.017608676067   # h/(kB * TCMB)  [1/GHz]
H_OVER_KB = 0.04799243           # h/kB            [K/GHz]

# ── Fiducial parameters of the Gaussian foregrounds ───────────────────────
# (overridden by load_fg_params, e.g. with the output of fit_fg_spectra.py)
FG_PARAMS = {
    'beta_d':  1.54,
    'T_d':     20.,
    'nu0_d':   353.,
    'A_d_EE':  10.,
    'A_d_BB':   5.,
    'alpha_d': -0.42,
    'beta_s':  -3.0,
    'nu0_s':   23.,
    'A_s_EE':   3.,
    'A_s_BB':   1.5,
    'alpha_s': -0.6,
}


def load_fg_params(fname):
    """Updates FG_PARAMS from a yaml file (e.g. output of
    fit_fg_spectra.py)."""
    import yaml
    new = yaml.safe_load(open(fname))
    unknown = set(new) - set(FG_PARAMS)
    if unknown:
        raise KeyError(f"Unknown foreground parameters in {fname}: {unknown}")
    FG_PARAMS.update({k: float(v) for k, v in new.items()})
    print(f"  foreground parameters from {fname}: {FG_PARAMS}")


# ═══════════════════════════════════════════════════════════════════════════
# SEDs in K_RJ and their derivatives
# ═══════════════════════════════════════════════════════════════════════════
# Dust: modified black body normalized at nu0,
#     f0(nu; beta, T) = (nu/nu0)^(1+beta) (e^x0 - 1) / (e^x - 1),
#     x = (h/kB) nu u, x0 = (h/kB) nu0 u, u = 1/T.
# Logarithmic derivatives in beta and in u = 1/T:
#     A(nu)  = d_beta ln f0 = ln(nu/nu0)
#     Bu(nu) = d_u ln f0    = [g(x0) - g(x)] / u,      g(x) = x e^x / (e^x - 1)
#     Cu(nu) = d_u^2 ln f0  = [phi(x) - phi(x0)] / u^2, phi(x) = x^2 e^x / (e^x - 1)^2

def _g(x):
    """g(x) = x e^x / (e^x - 1); for x -> 0, g(x) -> 1 + x/2 + x^2/12."""
    ex = np.exp(np.clip(x, -500, 500))
    with np.errstate(invalid='ignore', divide='ignore'):
        result = x * ex / np.expm1(np.clip(x, -500, 500))
    small = np.abs(x) < 1e-4
    result = np.where(small, 1.0 + x/2.0 + x**2/12.0, result)
    return result


def _phi(x):
    """phi(x) = x^2 e^x / (e^x - 1)^2; for x -> 0, phi(x) -> 1 - x^2/12."""
    em1 = np.expm1(np.clip(x, -500, 500))
    ex = np.exp(np.clip(x, -500, 500))
    with np.errstate(invalid='ignore', divide='ignore'):
        result = x**2 * ex / em1**2
    small = np.abs(x) < 1e-4
    result = np.where(small, 1.0 - x**2/12.0, result)
    return result


def fcmb(nu):
    """CMB factor in K_RJ: f(nu) = e^x (x/(e^x-1))^2, x = h nu/(kB TCMB)."""
    x = H_OVER_KB_CMB * nu
    ex = np.exp(x)
    return ex * (x / (ex - 1)) ** 2


def sed_dust_KRJ(nu, beta_d, T_d, nu0):
    """Dust MBB in K_RJ, normalized to 1 at nu0."""
    x0 = H_OVER_KB * nu0 / T_d
    x = H_OVER_KB * nu / T_d
    return (nu / nu0) ** (1.0 + beta_d) * np.expm1(x0) / np.expm1(x)


def sed_synch_KRJ(nu, beta_s, nu0):
    """Synchrotron SED (power law) in K_RJ, normalized at nu0."""
    return (nu / nu0) ** beta_s


def _dlnf_dbeta(nu, nu0):
    """A = d ln f0 / d beta = ln(nu/nu0)   [dimensionless]"""
    return np.log(nu / nu0)


def _dlnf_du(nu, nu0, T_d):
    """Bu = d ln f0 / d(1/T) = [g(x0) - g(x)] / u (note the sign: opposite
    to the derivative in T)."""
    u = 1.0 / T_d
    x0 = H_OVER_KB * nu0 * u
    x = H_OVER_KB * nu * u
    return (_g(x0) - _g(x)) / u


def _d2lnf_du2(nu, nu0, T_d):
    """Cu = d^2 ln f0 / d(1/T)^2 = [phi(x) - phi(x0)] / u^2."""
    u = 1.0 / T_d
    x0 = H_OVER_KB * nu0 * u
    x = H_OVER_KB * nu * u
    return (_phi(x) - _phi(x0)) / u**2


# ═══════════════════════════════════════════════════════════════════════════
# Bandpasses
# ═══════════════════════════════════════════════════════════════════════════

def load_bandpass(band_name):
    fname = os.path.join(BPASS_DIR, f"{band_name}.txt")
    nu, bnu = np.loadtxt(fname, unpack=True)
    dnu = np.zeros_like(nu)
    dnu[1:] = np.diff(nu)
    dnu[0] = dnu[1]
    return nu, bnu, dnu


def convolve_sed_bbpower(nu, bnu, dnu, sed_KRJ_arr):
    """SED integrated over the bandpass with the BBPower convention."""
    nu2 = nu ** 2
    cmb_norm = np.sum(fcmb(nu) * bnu * dnu * nu2)
    return np.sum(sed_KRJ_arr * bnu * dnu * nu2) / cmb_norm


def get_seds():
    """
    SED matrix [3, nfreq] with the BBPower convention: row 0 = CMB (= 1),
    row 1 = dust, row 2 = synchrotron.
    """
    p = FG_PARAMS
    nu0_d = p['nu0_d']
    nu0_s = p['nu0_s']
    cmb_nd = fcmb(nu0_d)
    cmb_ns = fcmb(nu0_s)

    nfreq = len(BAND_NAMES)
    seds = np.zeros((3, nfreq))

    for i, band in enumerate(BAND_NAMES):
        nu, bnu, dnu = load_bandpass(band)

        seds[0, i] = 1.0

        sed_d = sed_dust_KRJ(nu, p['beta_d'], p['T_d'], nu0_d)
        seds[1, i] = convolve_sed_bbpower(nu, bnu, dnu, sed_d) * cmb_nd

        sed_s = sed_synch_KRJ(nu, p['beta_s'], nu0_s)
        seds[2, i] = convolve_sed_bbpower(nu, bnu, dnu, sed_s) * cmb_ns

    return seds


def get_dust_beta_derivs():
    """
    Derivatives in beta of the dust SED (fiducial MBB) integrated over the
    bandpasses:
        S_k[i] = < f0 * A^k >_bandpass  (k = 1, 2),   A = ln(nu/nu0)
    Same convention as BBPower (integrate_seds_der, order=k).
    """
    p = FG_PARAMS
    nu0_d = p['nu0_d']
    cmb_nd = fcmb(nu0_d)
    S1 = np.zeros(len(BAND_NAMES))
    S2 = np.zeros(len(BAND_NAMES))
    for i, band in enumerate(BAND_NAMES):
        nu, bnu, dnu = load_bandpass(band)
        f0 = sed_dust_KRJ(nu, p['beta_d'], p['T_d'], nu0_d)
        A = _dlnf_dbeta(nu, nu0_d)
        S1[i] = convolve_sed_bbpower(nu, bnu, dnu, f0 * A) * cmb_nd
        S2[i] = convolve_sed_bbpower(nu, bnu, dnu, f0 * A**2) * cmb_nd
    return S1, S2


def get_dust_u_derivs():
    """
    Derivatives of the dust SED in the dimensionless temperature variable
    ut = T_d * delta(1/T)  (~ -delta T / T), integrated over the bandpasses:
        S1 = < f0 * Bu/T >,  S2 = < f0 * (Bu^2 + Cu)/T^2 >,
        Sbu = < f0 * A * Bu/T >   (mixed term delta-beta * delta-ut)
    """
    p = FG_PARAMS
    nu0_d, T_d = p['nu0_d'], p['T_d']
    cmb_nd = fcmb(nu0_d)
    S1, S2, Sbu = (np.zeros(len(BAND_NAMES)) for _ in range(3))
    for i, band in enumerate(BAND_NAMES):
        nu, bnu, dnu = load_bandpass(band)
        f0 = sed_dust_KRJ(nu, p['beta_d'], T_d, nu0_d)
        A = _dlnf_dbeta(nu, nu0_d)
        # d/d(ut) = (1/T_d) d/du, with u = 1/T
        Bu = _dlnf_du(nu, nu0_d, T_d) / T_d
        Cu = _d2lnf_du2(nu, nu0_d, T_d) / T_d**2
        S1[i] = convolve_sed_bbpower(nu, bnu, dnu, f0 * Bu) * cmb_nd
        S2[i] = convolve_sed_bbpower(nu, bnu, dnu, f0 * (Bu**2 + Cu)) * cmb_nd
        Sbu[i] = convolve_sed_bbpower(nu, bnu, dnu, f0 * A * Bu) * cmb_nd
    return S1, S2, Sbu


def get_synch_beta_derivs():
    """As get_dust_beta_derivs for the synchrotron: A = ln(nu/nu0_s)."""
    p = FG_PARAMS
    nu0_s = p['nu0_s']
    cmb_ns = fcmb(nu0_s)
    S1 = np.zeros(len(BAND_NAMES))
    S2 = np.zeros(len(BAND_NAMES))
    for i, band in enumerate(BAND_NAMES):
        nu, bnu, dnu = load_bandpass(band)
        f0 = sed_synch_KRJ(nu, p['beta_s'], nu0_s)
        A = _dlnf_dbeta(nu, nu0_s)
        S1[i] = convolve_sed_bbpower(nu, bnu, dnu, f0 * A) * cmb_ns
        S2[i] = convolve_sed_bbpower(nu, bnu, dnu, f0 * A**2) * cmb_ns
    return S1, S2


# ═══════════════════════════════════════════════════════════════════════════
# Power spectra
# ═══════════════════════════════════════════════════════════════════════════

def get_beta_cls(lmax, amp, gamma):
    """
    Spectrum of a delta-beta (or delta-ut) map, same form as BBPower
    (compsep.bcls):
        C_ell = 1e-6 * amp * (ell/80)^gamma,  ell >= 2
    """
    ell = np.arange(lmax + 1)
    cl = np.zeros(lmax + 1)
    cl[2:] = 1e-6 * amp * (ell[2:] / 80.) ** gamma
    return cl


def _dl2cl(lmax):
    ell = np.arange(lmax + 1)
    conv = np.zeros(lmax + 1)
    conv[1:] = 2. * np.pi / (ell[1:] * (ell[1:] + 1))
    return conv


def read_camb(fname, lmax):
    ell_in, dtt, dee, dbb, dte = np.loadtxt(fname, unpack=True)
    ell_out = np.arange(lmax + 1)
    dlee = np.interp(ell_out, ell_in, dee, left=0., right=0.)
    dlbb = np.interp(ell_out, ell_in, dbb, left=0., right=0.)
    return dlee, dlbb


def get_cmb_spectra(lmax, r=0.):
    conv = _dl2cl(lmax)
    dlee_r0, dlbb_r0 = read_camb(CAMB_NOBB, lmax)
    dlee_r1, dlbb_r1 = read_camb(CAMB_R1, lmax)
    # camb_lens_r1.dat includes lensing too: the tensor part is
    # (r1 - nobb), as in BBPower
    clee = (dlee_r0 + r * (dlee_r1 - dlee_r0)) * conv
    clbb = (dlbb_r0 + r * (dlbb_r1 - dlbb_r0)) * conv
    return clee, clbb


def get_fg_spectra(lmax):
    p = FG_PARAMS
    conv = _dl2cl(lmax)
    ell = np.arange(lmax + 1)
    ell0 = 80.

    with np.errstate(divide='ignore', invalid='ignore'):
        def pl(A, alpha):
            return np.where(ell > 0, A * (ell / ell0) ** alpha * conv, 0.)

        cldee = pl(p['A_d_EE'], p['alpha_d'])
        cldbb = pl(p['A_d_BB'], p['alpha_d'])
        clsee = pl(p['A_s_EE'], p['alpha_s'])
        clsbb = pl(p['A_s_BB'], p['alpha_s'])

    return cldee, cldbb, clsee, clsbb


# ═══════════════════════════════════════════════════════════════════════════
# PySM foregrounds
# ═══════════════════════════════════════════════════════════════════════════

def harmonic_downgrade_qu(Q, U, nside):
    """Q, U at a higher nside -> nside in harmonic space: alm up to
    3 nside - 1, divided by the pixel window of the input nside, then a map
    at nside without window (band-limited like the synfast CMB and noise)."""
    lmax = 3 * nside - 1
    pw = hp.pixwin(hp.get_nside(Q), pol=True, lmax=lmax)[1]   # E/B
    pw[:2] = 1.
    alm = hp.map2alm([np.zeros_like(Q), Q, U], lmax=lmax, pol=True)
    for a in alm[1:]:
        hp.almxfl(a, 1. / pw, inplace=True)
    return hp.alm2map(alm, nside, lmax=lmax, pol=True)[1:]


def get_pysm_fg(model, nside, nside_in=0):
    """
    Foreground Q, U per band from PySM [nfreq, 2, npix] in uK_CMB, integrated
    over the SO bandpasses (same convention as BBPower). model is a string
    of presets, e.g. 'd1s1'. The maps are deterministic: saved in
    DATA_DIR/pysm_cache/ and reused.

    nside_in = 0: PySM directly at nside (PySM brings the templates to nside
    with hp.ud_grade, i.e. pixel averaging: the foregrounds get the pixel
    window and aliasing, unlike the synfast CMB and noise).
    nside_in > nside: PySM at nside_in, then harmonic downgrade (see
    harmonic_downgrade_qu), band-limited like the other components.
    """
    cache_dir = os.path.join(DATA_DIR, 'pysm_cache')
    if nside_in:
        fname = os.path.join(cache_dir,
                             f'{model}_nside{nside}_from{nside_in}.npy')
        if os.path.isfile(fname):
            return np.load(fname)
        hi = get_pysm_fg(model, nside_in)
        print(f"  PySM: harmonic downgrade nside {nside_in} -> {nside} "
              f"(lmax {3 * nside - 1}, pixel window of {nside_in} removed)")
        fg = np.array([harmonic_downgrade_qu(hi[i, 0], hi[i, 1], nside)
                       for i in range(len(BAND_NAMES))])
        np.save(fname, fg)
        return fg
    fname = os.path.join(cache_dir, f'{model}_nside{nside}.npy')
    if os.path.isfile(fname):
        return np.load(fname)

    import pysm3
    import pysm3.units as u
    presets = re.findall(r'[a-z]+\d+', model)
    print(f"  PySM: building {presets} at nside={nside} (cached in {fname})")
    # d12 (MKD, 6 layers): the preset only points to the nside 2048 files
    # (~5.8 GB); the same layers exist at nside 512 (~360 MB), enough here.
    extra = []
    if 'd12' in presets:
        presets.remove('d12')
        extra.append(pysm3.ModifiedBlackBodyLayers(
            map_layers='mkd_dust/512/thermaldust_ampl{layer}.fits',
            freq_ref='353 GHz',
            map_mbb_index='mkd_dust/512/thermaldust_specind{layer}.fits',
            map_mbb_temperature='mkd_dust/512/thermaldust_temp{layer}.fits',
            nside=nside, num_layers=6, color_correction=0.911,
            unit_layers='MJy/sr', unit_mbb_temperature='K'))
    sky = pysm3.Sky(nside=nside, preset_strings=presets or None,
                    component_objects=extra or None)
    fg = np.zeros([len(BAND_NAMES), 2, hp.nside2npix(nside)])
    for i, band in enumerate(BAND_NAMES):
        nu, bnu = np.loadtxt(os.path.join(BPASS_DIR, f"{band}.txt"),
                             unpack=True)
        conv = pysm3.utils.bandpass_unit_conversion(nu * u.GHz, bnu,
                                                    u.uK_CMB)
        m = sky.get_emission(nu * u.GHz, bnu) * conv
        fg[i] = m[1:3].value
    os.makedirs(cache_dir, exist_ok=True)
    np.save(fname, fg)
    return fg
