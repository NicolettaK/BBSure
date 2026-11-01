"""
Genera mappe SO simulate con SED della dust estesa tramite Moment Expansion.

È una generalizzazione di generate_maps.py di first_test: quando tutti i
parametri dei moments sono zero, il risultato è identico all'originale.

─────────────────────────────────────────────────────────────────
MOMENTO EXPANSION PER LA DUST (in unità K_RJ)
─────────────────────────────────────────────────────────────────

La SED di base (MBB in K_RJ, normalizzata a nu0) è:

    f0(nu; beta, T) = (nu/nu0)^(1+beta) * (e^x0 - 1) / (e^x - 1)

con  x = (h/kB) * nu / T = H_OVER_KB * nu * u,   u = 1/T
     x0 = (h/kB) * nu0 / T = H_OVER_KB * nu0 * u

Variabile di espansione: u = 1/T  (come in funclib.py e arXiv:2411.11649).
Questa scelta converge meglio di T perché la MBB dipende da nu/T
direttamente, e la serie in 1/T ha derivate più semplici.

Quando i parametri spettrali (beta, 1/T) variano pixel per pixel, il
segnale medio su un patch è:

    <f(nu)> ≈ f0(nu) * [1
        + omega_beta   * A(nu)                    # 1° ordine: shift in beta
        + omega_u      * Bu(nu)                   # 1° ordine: shift in 1/T
        + (1/2) * sigma2_beta  * A(nu)^2          # 2° ordine: var beta
        + sigma_beta_u * A(nu) * Bu(nu)           # 2° ordine: cov beta,1/T
        + (1/2) * sigma2_u * (Bu(nu)^2 + Cu(nu)) # 2° ordine: var 1/T
    ]

Le derivate logaritmiche rispetto a u = 1/T sono:

    A(nu)  = d_beta ln f0 = ln(nu/nu0)

    Bu(nu) = d_u ln f0 = [g(x0) - g(x)] / u
             con g(x) = x * e^x / (e^x - 1)

    Cu(nu) = d_u^2 ln f0 = [phi(x) - phi(x0)] / u^2
             con phi(x) = x^2 * e^x / (e^x - 1)^2

I moments (cumulanti di P(beta, 1/T)) sono:
    omega_beta   = <Delta beta>            (shift del valor medio di beta)
    omega_u      = <Delta(1/T)>    [1/K]  (shift del valor medio di 1/T)
    sigma2_beta  = <(Delta beta)^2>        (varianza di beta)
    sigma2_u     = <(Delta(1/T))^2> [1/K^2] (varianza di 1/T)
    sigma_beta_u = <Delta beta * Delta(1/T)> [1/K] (covarianza)

─────────────────────────────────────────────────────────────────

Uso:
    python generate_maps_moments.py --seed 5001 --nside 64 \\
        --sigma2-beta 0.1 --sigma2-u 1e-5 --sigma-beta-u -1e-3

Per la MBB semplice (nessun momento):
    python generate_maps_moments.py --seed 5001 --nside 64
"""

import sys
import os
import argparse
from dataclasses import dataclass, field, asdict
import numpy as np
import healpy as hp

# ── Percorsi ──────────────────────────────────────────────────────────────────
WORK_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(WORK_DIR, 'data')
if not os.path.isdir(DATA_DIR):     # pacchetto della validazione: <root>/data
    DATA_DIR = os.path.join(os.path.dirname(WORK_DIR), 'data')

sys.path.insert(0, WORK_DIR)
import noise_calc as nc

CAMB_NOBB = os.path.join(DATA_DIR, 'camb', 'camb_lens_nobb.dat')
CAMB_R1   = os.path.join(DATA_DIR, 'camb', 'camb_lens_r1.dat')
BPASS_DIR = os.path.join(DATA_DIR, 'bandpasses')

BAND_NAMES = ['LF1', 'LF2', 'MF1', 'MF2', 'UHF1', 'UHF2']

# ── Costanti fisiche ──────────────────────────────────────────────────────────
H_OVER_KB_CMB = 0.017608676067   # h/(kB * TCMB)  [1/GHz]
H_OVER_KB     = 0.04799243       # h/kB            [K/GHz]

# ── Parametri fiduciali ───────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclass per i parametri dei moments della dust
# (espansione in u = 1/T, come funclib.py e arXiv:2411.11649)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DustMoments:
    """
    Parametri della moment expansion della SED dust.

    Variabile di espansione: u = 1/T.

    Tutti a zero → MBB semplice, identica a generate_maps.py originale.
    """
    omega_beta   : float = 0.0  # <Delta beta>             shift val. medio beta
    omega_u      : float = 0.0  # <Delta(1/T)>   [1/K]    shift val. medio 1/T
    sigma2_beta  : float = 0.0  # <(Delta beta)^2>         varianza di beta
    sigma2_u     : float = 0.0  # <(Delta(1/T))^2> [1/K^2] varianza di 1/T
    sigma_beta_u : float = 0.0  # <Delta beta * Delta(1/T)> [1/K] covarianza

    @property
    def all_zero(self):
        return all(v == 0.0 for v in asdict(self).values())

    def __str__(self):
        if self.all_zero:
            return "DustMoments(plain MBB, all moments = 0)"
        parts = []
        for k, v in asdict(self).items():
            if v != 0.0:
                parts.append(f"{k}={v}")
        return f"DustMoments({', '.join(parts)})"


# ═══════════════════════════════════════════════════════════════════════════════
# Funzioni matematiche di supporto
# ═══════════════════════════════════════════════════════════════════════════════

def _g(x):
    """g(x) = x * e^x / (e^x - 1).

    Usata in Bu: Bu = [g(x0) - g(x)] / u.
    Per x→0: g(x) → 1 + x/2 + x^2/12 + ...
    """
    ex = np.exp(np.clip(x, -500, 500))
    with np.errstate(invalid='ignore', divide='ignore'):
        result = x * ex / np.expm1(np.clip(x, -500, 500))
    small = np.abs(x) < 1e-4
    result = np.where(small, 1.0 + x/2.0 + x**2/12.0, result)
    return result


def _phi(x):
    """phi(x) = x^2 * e^x / (e^x - 1)^2.

    Usata in Cu: Cu = [phi(x) - phi(x0)] / u^2.
    Nota: phi(x) = x * g(x) / (e^x - 1) * (e^x - 1)... meglio valutarla
    direttamente.
    Per x→0: phi(x) → 1 - x^2/12 + ...  [stesso limite di (x/sinh(x/2)/2)^2]
    """
    em1 = np.expm1(np.clip(x, -500, 500))
    ex  = np.exp(np.clip(x, -500, 500))
    with np.errstate(invalid='ignore', divide='ignore'):
        result = x**2 * ex / em1**2
    small = np.abs(x) < 1e-4
    result = np.where(small, 1.0 - x**2/12.0, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SED in K_RJ
# ═══════════════════════════════════════════════════════════════════════════════

def fcmb(nu):
    """Fattore CMB in K_RJ: f(nu) = e^x (x/(e^x-1))^2, x = h*nu/(kB*TCMB)."""
    x  = H_OVER_KB_CMB * nu
    ex = np.exp(x)
    return ex * (x / (ex - 1)) ** 2


def sed_dust_KRJ_base(nu, beta_d, T_d, nu0):
    """
    SED dust MBB di base in K_RJ, normalizzata a 1 in nu0.

        f0(nu) = (nu/nu0)^(1+beta) * (e^x0 - 1) / (e^x - 1)

    Identica a sed_dust_KRJ di generate_maps.py.
    """
    x0 = H_OVER_KB * nu0 / T_d
    x  = H_OVER_KB * nu  / T_d
    return (nu / nu0) ** (1.0 + beta_d) * np.expm1(x0) / np.expm1(x)


def _dlnf_dbeta(nu, nu0):
    """A = d ln f0 / d beta = ln(nu/nu0)   [adimensionale]"""
    return np.log(nu / nu0)


def _dlnf_du(nu, nu0, T_d):
    """
    Bu = d ln f0 / d(1/T) = [g(x0) - g(x)] / u = T * [g(x0) - g(x)]

    con u = 1/T,  x = H_OVER_KB * nu * u,  g(x) = x*e^x/(e^x-1).

    Nota il segno: g(x0) - g(x), opposto rispetto alla derivata in T.
    """
    u  = 1.0 / T_d
    x0 = H_OVER_KB * nu0 * u
    x  = H_OVER_KB * nu  * u
    return (_g(x0) - _g(x)) / u


def _d2lnf_du2(nu, nu0, T_d):
    """
    Cu = d^2 ln f0 / d(1/T)^2 = [phi(x) - phi(x0)] / u^2

    con phi(x) = x^2 * e^x / (e^x - 1)^2.

    Nota il segno: phi(x) - phi(x0)  (x alla frequenza nu, x0 a nu0).
    """
    u  = 1.0 / T_d
    x0 = H_OVER_KB * nu0 * u
    x  = H_OVER_KB * nu  * u
    return (_phi(x) - _phi(x0)) / u**2


def sed_dust_KRJ_moments(nu, beta_d, T_d, nu0, moments: DustMoments):
    """
    SED dust in K_RJ con moment expansion in u=1/T attorno a (beta_d, T_d).

    Se moments.all_zero → identica a sed_dust_KRJ_base (MBB semplice).

    <f(nu)> ≈ f0 * [1 + omega_beta*A + omega_u*Bu
                      + 1/2*sigma2_beta*A^2
                      + sigma_beta_u*A*Bu
                      + 1/2*sigma2_u*(Bu^2 + Cu)]

    Parametri
    ---------
    nu      : array-like  - frequenze in GHz
    beta_d  : float       - indice spettrale pivot
    T_d     : float       - temperatura pivot [K]
    nu0     : float       - frequenza pivot [GHz]
    moments : DustMoments - parametri dei moments in u=1/T

    Ritorna
    -------
    array della stessa forma di nu
    """
    nu = np.asarray(nu, dtype=float)
    f0 = sed_dust_KRJ_base(nu, beta_d, T_d, nu0)

    if moments.all_zero:
        return f0

    A  = _dlnf_dbeta(nu, nu0)         # d_beta  ln f0  [adim]
    Bu = _dlnf_du(nu, nu0, T_d)      # d_u     ln f0  [K]
    Cu = _d2lnf_du2(nu, nu0, T_d)   # d_u^2   ln f0  [K^2]

    corr = (
          moments.omega_beta   * A
        + moments.omega_u      * Bu
        + 0.5 * moments.sigma2_beta  * A**2
        + moments.sigma_beta_u * A  * Bu
        + 0.5 * moments.sigma2_u    * (Bu**2 + Cu)
    )

    return f0 * (1.0 + corr)


def sed_synch_KRJ(nu, beta_s, nu0):
    """SED synchrotron (legge di potenza) in K_RJ, normalizzata a nu0."""
    return (nu / nu0) ** beta_s


# ═══════════════════════════════════════════════════════════════════════════════
# Gestione bandpass (identica a generate_maps.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_bandpass(band_name):
    fname = os.path.join(BPASS_DIR, f"{band_name}.txt")
    nu, bnu = np.loadtxt(fname, unpack=True)
    dnu = np.zeros_like(nu)
    dnu[1:] = np.diff(nu)
    dnu[0]  = dnu[1]
    return nu, bnu, dnu


def convolve_sed_bbpower(nu, bnu, dnu, sed_KRJ_arr):
    """Integrazione SED sulla bandpass con la formula BBPower."""
    nu2      = nu ** 2
    cmb_norm = np.sum(fcmb(nu) * bnu * dnu * nu2)
    return np.sum(sed_KRJ_arr * bnu * dnu * nu2) / cmb_norm


# ═══════════════════════════════════════════════════════════════════════════════
# Matrice SED con moments
# ═══════════════════════════════════════════════════════════════════════════════

def get_seds(dust_moments: DustMoments = None):
    """
    Matrice SED [3, nfreq] con convenzione BBPower.

    Riga 0 = CMB (=1), riga 1 = dust (con moments in u=1/T), riga 2 = synch.

    Se dust_moments è None o all_zero → identica a generate_maps.py originale.
    """
    if dust_moments is None:
        dust_moments = DustMoments()

    p      = FG_PARAMS
    nu0_d  = p['nu0_d']
    nu0_s  = p['nu0_s']
    cmb_nd = fcmb(nu0_d)
    cmb_ns = fcmb(nu0_s)

    nfreq = len(BAND_NAMES)
    seds  = np.zeros((3, nfreq))

    for i, band in enumerate(BAND_NAMES):
        nu, bnu, dnu = load_bandpass(band)

        seds[0, i] = 1.0

        sed_d = sed_dust_KRJ_moments(nu, p['beta_d'], p['T_d'], nu0_d,
                                     dust_moments)
        seds[1, i] = convolve_sed_bbpower(nu, bnu, dnu, sed_d) * cmb_nd

        sed_s = sed_synch_KRJ(nu, p['beta_s'], nu0_s)
        seds[2, i] = convolve_sed_bbpower(nu, bnu, dnu, sed_s) * cmb_ns

    return seds


# ═══════════════════════════════════════════════════════════════════════════════
# Spettri CMB e foreground (identici a generate_maps.py)
# ═══════════════════════════════════════════════════════════════════════════════

def get_dust_beta_derivs():
    """
    Derivate in beta della SED dust (MBB fiduciale) convolute sulla bandpass:
        S_k[i] = < f0 * A^k >_bandpass  (k = 1, 2),   A = ln(nu/nu0)
    Stessa convenzione di BBPower (integrate_seds_der, order=k).
    """
    p      = FG_PARAMS
    nu0_d  = p['nu0_d']
    cmb_nd = fcmb(nu0_d)
    S1 = np.zeros(len(BAND_NAMES))
    S2 = np.zeros(len(BAND_NAMES))
    for i, band in enumerate(BAND_NAMES):
        nu, bnu, dnu = load_bandpass(band)
        f0 = sed_dust_KRJ_base(nu, p['beta_d'], p['T_d'], nu0_d)
        A  = _dlnf_dbeta(nu, nu0_d)
        S1[i] = convolve_sed_bbpower(nu, bnu, dnu, f0 * A) * cmb_nd
        S2[i] = convolve_sed_bbpower(nu, bnu, dnu, f0 * A**2) * cmb_nd
    return S1, S2


def get_dust_u_derivs():
    """
    Derivate della SED dust nella variabile di temperatura adimensionale
    ut = T_d * delta(1/T)  (~ -delta T / T), convolute sulla bandpass:
        S1 = < f0 * Bu/T >,  S2 = < f0 * (Bu^2 + Cu)/T^2 >,
        Sbu = < f0 * A * Bu/T >   (termine misto delta-beta * delta-ut)
    Stessa convenzione di BBPower (integrate_seds_der_u).
    """
    p      = FG_PARAMS
    nu0_d, T_d = p['nu0_d'], p['T_d']
    cmb_nd = fcmb(nu0_d)
    S1, S2, Sbu = (np.zeros(len(BAND_NAMES)) for _ in range(3))
    for i, band in enumerate(BAND_NAMES):
        nu, bnu, dnu = load_bandpass(band)
        f0 = sed_dust_KRJ_base(nu, p['beta_d'], T_d, nu0_d)
        A  = _dlnf_dbeta(nu, nu0_d)
        # d/d(ut) = (1/T_d) d/du, con u = 1/T
        Bu = _dlnf_du(nu, nu0_d, T_d) / T_d
        Cu = _d2lnf_du2(nu, nu0_d, T_d) / T_d**2
        S1[i]  = convolve_sed_bbpower(nu, bnu, dnu, f0 * Bu) * cmb_nd
        S2[i]  = convolve_sed_bbpower(nu, bnu, dnu, f0 * (Bu**2 + Cu)) * cmb_nd
        Sbu[i] = convolve_sed_bbpower(nu, bnu, dnu, f0 * A * Bu) * cmb_nd
    return S1, S2, Sbu


def get_synch_beta_derivs():
    """Come get_dust_beta_derivs per il sincrotrone: A = ln(nu/nu0_s)."""
    p      = FG_PARAMS
    nu0_s  = p['nu0_s']
    cmb_ns = fcmb(nu0_s)
    S1 = np.zeros(len(BAND_NAMES))
    S2 = np.zeros(len(BAND_NAMES))
    for i, band in enumerate(BAND_NAMES):
        nu, bnu, dnu = load_bandpass(band)
        f0 = sed_synch_KRJ(nu, p['beta_s'], nu0_s)
        A  = _dlnf_dbeta(nu, nu0_s)
        S1[i] = convolve_sed_bbpower(nu, bnu, dnu, f0 * A) * cmb_ns
        S2[i] = convolve_sed_bbpower(nu, bnu, dnu, f0 * A**2) * cmb_ns
    return S1, S2


def get_beta_cls(lmax, amp, gamma):
    """
    Spettro della mappa di delta-beta, stessa forma di BBPower (compsep.bcls):
        C_ell^beta = 1e-6 * amp * (ell/80)^gamma,  ell >= 2
    """
    ell = np.arange(lmax + 1)
    cl  = np.zeros(lmax + 1)
    cl[2:] = 1e-6 * amp * (ell[2:] / 80.) ** gamma
    return cl


def _dl2cl(lmax):
    ell  = np.arange(lmax + 1)
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
    dlee_r1, dlbb_r1 = read_camb(CAMB_R1,   lmax)
    # camb_lens_r1.dat contiene anche il lensing: la parte tensoriale e'
    # (r1 - nobb), come in BBPower (prima: r * r1, cioe' r * lensing in piu')
    clee = (dlee_r0 + r * (dlee_r1 - dlee_r0)) * conv
    clbb = (dlbb_r0 + r * (dlbb_r1 - dlbb_r0)) * conv
    return clee, clbb


def get_fg_spectra(lmax):
    p    = FG_PARAMS
    conv = _dl2cl(lmax)
    ell  = np.arange(lmax + 1)
    ell0 = 80.

    with np.errstate(divide='ignore', invalid='ignore'):
        def pl(A, alpha):
            return np.where(ell > 0, A * (ell / ell0) ** alpha * conv, 0.)

        cldee = pl(p['A_d_EE'], p['alpha_d'])
        cldbb = pl(p['A_d_BB'], p['alpha_d'])
        clsee = pl(p['A_s_EE'], p['alpha_s'])
        clsbb = pl(p['A_s_BB'], p['alpha_s'])

    return cldee, cldbb, clsee, clsbb


# ═══════════════════════════════════════════════════════════════════════════════
# Generazione mappe
# ═══════════════════════════════════════════════════════════════════════════════

def harmonic_downgrade_qu(Q, U, nside):
    """Q, U a nside piu' alto -> nside nello spazio armonico: alm fino a
    3 nside - 1, divise per la finestra del pixel di partenza, poi mappa a
    nside senza finestra (band-limited come CMB e rumore di synfast)."""
    lmax = 3 * nside - 1
    pw = hp.pixwin(hp.get_nside(Q), pol=True, lmax=lmax)[1]   # E/B
    pw[:2] = 1.
    alm = hp.map2alm([np.zeros_like(Q), Q, U], lmax=lmax, pol=True)
    for a in alm[1:]:
        hp.almxfl(a, 1. / pw, inplace=True)
    return hp.alm2map(alm, nside, lmax=lmax, pol=True)[1:]


def get_pysm_fg(model, nside, nside_in=0):
    """
    Foreground Q, U per banda da PySM [nfreq, 2, npix] in uK_CMB, integrati
    sui bandpass di SO (stessa convenzione di BBPower, vedi
    check_pysm_bandpass.py). model e' una stringa di preset, es. 'd1s1'.
    Le mappe sono deterministiche: salvate in data/pysm_cache/ e riusate.

    nside_in = 0: PySM direttamente a nside (PySM porta i template a nside
    con hp.ud_grade, cioe' media dei pixel: i foreground prendono la finestra
    del pixel e l'aliasing, a differenza di CMB e rumore fatti con synfast).
    nside_in > nside: PySM a nside_in, poi downgrade armonico: map2alm fino a
    3 nside - 1, divisione per la finestra del pixel di nside_in, alm2map a
    nside senza finestra (band-limited come gli altri componenti).
    """
    import re
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


def load_fg_params(fname):
    """Aggiorna FG_PARAMS da un file yaml (es. output di fit_fg_spectra.py)."""
    import yaml
    new = yaml.safe_load(open(fname))
    unknown = set(new) - set(FG_PARAMS)
    if unknown:
        raise KeyError(f"Unknown foreground parameters in {fname}: {unknown}")
    FG_PARAMS.update({k: float(v) for k, v in new.items()})
    print(f"  foreground parameters from {fname}: {FG_PARAMS}")


def generate_maps(seed, nside, output_dir, nsplits=4, r=0.,
                  dust_moments: DustMoments = None,
                  amp_beta=0., gamma_beta=-3., beta_seed=None,
                  amp_beta_s=0., gamma_beta_s=-3., beta_seed_s=None,
                  amp_u=0., gamma_u=-3., u_seed=None, pysm=None, pysm_nside_in=0,
                  hits_noise=None, rho_bu=0., sens_mode=1, one_over_f=1,
                  kludge=True, no_ee=False):
    """
    Genera e salva le mappe osservate per tutti gli split.

    Con pysm (es. 'd1s1') dust e sincrotrone sono le mappe di PySM
    (get_pysm_fg) invece dei foreground gaussiani; i momenti non si
    applicano. CMB e rumore restano quelli del seed (stessi numeri casuali
    del caso gaussiano).

    Rispetto a generate_maps.py originale, accetta dust_moments per usare
    una SED della dust con moment expansion in u=1/T al posto della MBB.

    Con amp_beta > 0 aggiunge una mappa gaussiana delta-beta(n), scorrelata
    dalla dust, con lo spettro dei momenti di BBPower (get_beta_cls), e
    applica la SED pixel per pixel al secondo ordine:
        d_nu(n) = d(n) * [S0 + dbeta(n) * S1 + 1/2 * dbeta(n)^2 * S2]
    che è il modello di BBPower (termini 1x1 e 0x2). La mappa delta-beta
    usa un RNG separato (beta_seed, default seed + 100000): CMB, dust,
    synch e rumore restano identici al caso amp_beta = 0 con lo stesso seed.
    amp_beta_s / gamma_beta_s fanno lo stesso per il sincrotrone (mappa
    delta-beta_s indipendente, default seed + 200000).
    amp_u / gamma_u aggiungono una mappa di ut = T_d * delta(1/T) per la
    dust, indipendente da delta-beta (default seed + 300000), con la SED al
    secondo ordine in (beta, ut), termine misto delta-beta * delta-ut incluso.
    """
    if dust_moments is None:
        dust_moments = DustMoments()

    np.random.seed(seed)
    lmax  = 3 * nside - 1
    npix  = hp.nside2npix(nside)
    cl0   = np.zeros(lmax + 1)
    nfreq = len(BAND_NAMES)

    print(f"  seed={seed}, nside={nside}, lmax={lmax}, r={r}")
    print(f"  dust: {dust_moments}")

    clee_cmb, clbb_cmb         = get_cmb_spectra(lmax, r=r)
    cldee, cldbb, clsee, clsbb = get_fg_spectra(lmax)

    _, Qcmb,   Ucmb   = hp.synfast([cl0, clee_cmb, clbb_cmb, cl0, cl0, cl0],
                                    nside, new=True)
    _, Qdust,  Udust  = hp.synfast([cl0, cldee, cldbb, cl0, cl0, cl0],
                                    nside, new=True)
    _, Qsynch, Usynch = hp.synfast([cl0, clsee, clsbb, cl0, cl0, cl0],
                                    nside, new=True)

    map_comp = np.array([[Qcmb,   Ucmb  ],
                          [Qdust,  Udust ],
                          [Qsynch, Usynch]])

    seds = get_seds(dust_moments)
    print(f"  SED dust:  {np.round(seds[1], 4)}")
    print(f"  SED synch: {np.round(seds[2], 4)}")

    map_freq = np.einsum('ij,ikl->jkl', seds, map_comp)

    if pysm:
        if amp_beta > 0 or amp_beta_s > 0 or amp_u > 0 or \
                not dust_moments.all_zero:
            raise ValueError("moments cannot be combined with --pysm")
        # CMB (SED = 1 in uK_CMB) + foreground di PySM
        map_freq = map_comp[0][None, :, :] + get_pysm_fg(pysm, nside, pysm_nside_in)
        print(f"  foregrounds: PySM {pysm}")

    # mappe di delta-beta: (componente, amp, gamma, seed, derivate, file)
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
        # synalm + alm2map == synfast (stessi numeri casuali), ma tiene le alm
        alm = hp.synalm(cl_beta, new=True)
        dbeta = hp.alm2map(alm, nside)
        np.random.set_state(state)
        S1, S2 = derivs()
        if i_c == 1:
            dbeta_dust, alm_beta_dust, cl_beta_dust = dbeta, alm, cl_beta
        print(f"  delta-beta map ({'dust' if i_c == 1 else 'synch'}): "
              f"amp={amp}, gamma={gamma}, rms={np.std(dbeta):.4f}, "
              f"expected sigma2={np.sum((2*np.arange(lmax+1)+1)*cl_beta)/(4*np.pi):.3e}")
        # termine al primo e secondo ordine, pixel per pixel (Q e U)
        for i_f in range(nfreq):
            map_freq[i_f] += map_comp[i_c] * (dbeta * S1[i_f]
                                              + 0.5 * dbeta**2 * S2[i_f])
        os.makedirs(output_dir, exist_ok=True)
        hp.write_map(os.path.join(output_dir, fname), dbeta,
                     overwrite=True, dtype=np.float32)

    # mappa di temperatura della dust: ut = T_d * delta(1/T); con rho_bu != 0
    # correlata con delta-beta della dust: C^bu = rho sqrt(C^bb C^uu)
    if amp_u > 0:
        cl_u = get_beta_cls(lmax, amp_u, gamma_u)
        state = np.random.get_state()
        np.random.seed(seed + 300000 if u_seed is None else u_seed)
        if rho_bu == 0:
            dut = hp.synfast(cl_u, nside, new=True)
        else:
            if alm_beta_dust is None:
                raise ValueError("--rho-bu needs the dust delta-beta map (--amp-beta)")
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
    # sens_mode 1 baseline / 2 goal; one_over_f 1 ottimistico / 0 pessimistico;
    # kludge: fattore di V3 per gli hit non uniformi (da togliere quando il
    # rumore segue gia' la mappa degli hit, altrimenti conta due volte)
    _, nell_arr[:, 2:], _ = nc.Simons_Observatory_V3_SA_noise(
        sens_mode, one_over_f, 1, fsky, lmax + 1, 1, include_beam=False,
        include_kludge=kludge)
    print(f"  noise: sensitivity_mode={sens_mode}, one_over_f_mode="
          f"{one_over_f}, include_kludge={kludge}")

    # Rumore che segue gli hit: sigma(n) = sigma_0 * sqrt(<h> / (fsky h(n))),
    # con sigma_0 il livello di SO per fsky. Conserva il tempo totale di
    # osservazione (sum 1/sigma^2 come nel caso uniforme su fsky); zero
    # fuori dall'impronta.
    noise_scale = np.ones(npix)
    if hits_noise:
        h = hp.ud_grade(hp.read_map(hits_noise), nside)
        h = h / h.max()
        h[h < 1e-3] = 0.          # bordi della rotazione C -> G (make_masks.py)
        noise_scale = np.zeros(npix)
        good = h > 0
        noise_scale[good] = np.sqrt(np.mean(h) / (fsky * h[good]))
        print(f"  noise scaled with hits ({hits_noise}): <h> = {np.mean(h):.4f}, "
              f"sigma/sigma_0 in [{noise_scale[good].min():.2f}, "
              f"{noise_scale[good].max():.1f}]")

    if no_ee:
        # Test: segnale (CMB + foreground) senza modi E, per escludere la
        # dispersione E -> B della maschera. Il rumore resta com'e'.
        for i_f in range(nfreq):
            alm = hp.map2alm([np.zeros(npix), map_freq[i_f, 0],
                              map_freq[i_f, 1]], lmax=lmax, pol=True)
            alm[1] *= 0.
            map_freq[i_f] = hp.alm2map(alm, nside, lmax=lmax, pol=True)[1:]
        print("  signal E modes removed (--no-ee)")

    os.makedirs(output_dir, exist_ok=True)

    for i_s in range(nsplits):
        map_noise = np.zeros([nfreq, 2, npix])
        for i_f in range(nfreq):
            _, mpq, mpu = hp.synfast(
                [cl0, nell_arr[i_f], nell_arr[i_f], cl0, cl0, cl0],
                nside, new=True)
            map_noise[i_f, 0, :] = mpq * np.sqrt(nsplits) * noise_scale
            map_noise[i_f, 1, :] = mpu * np.sqrt(nsplits) * noise_scale

        obs   = (map_freq + map_noise).reshape([nfreq * 2, npix])
        fname = os.path.join(output_dir,
                             f'obs_split{i_s+1}of{nsplits}.fits.gz')
        hp.write_map(fname, obs, overwrite=True, dtype=np.float32)
        print(f"    Salvata: {fname}")

    print(f"  Seed {seed} completato.")


# ═══════════════════════════════════════════════════════════════════════════════
# Entrypoint CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Genera mappe SO con SED dust estesa da moment expansion '
                    'in u=1/T (compatibile con funclib.py / arXiv:2411.11649).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--seed',       type=int,   default=5001)
    parser.add_argument('--nside',      type=int,   default=64)
    parser.add_argument('--nsplits',    type=int,   default=4)
    parser.add_argument('--r',          type=float, default=0.0)
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(WORK_DIR, 'sims', 's5001'))

    grp = parser.add_argument_group(
        'Dust moments  (espansione in u = 1/T)',
        '0 su tutti → MBB semplice, identico a generate_maps.py originale. '
        'Le unità di omega_u, sigma2_u, sigma_beta_u sono [1/K], [1/K^2], [1/K].')
    grp.add_argument('--omega-beta',   type=float, default=0.0,
                     help='<Delta beta>: shift val. medio beta')
    grp.add_argument('--omega-u',      type=float, default=0.0,
                     help='<Delta(1/T)> [1/K]: shift val. medio 1/T')
    grp.add_argument('--sigma2-beta',  type=float, default=0.0,
                     help='<(Delta beta)^2>: varianza di beta')
    grp.add_argument('--sigma2-u',     type=float, default=0.0,
                     help='<(Delta(1/T))^2> [1/K^2]: varianza di 1/T')
    grp.add_argument('--sigma-beta-u', type=float, default=0.0,
                     help='<Delta beta * Delta(1/T)> [1/K]: covarianza')

    grp2 = parser.add_argument_group(
        'Mappa di beta (modello dei momenti di BBPower)',
        'C_ell^beta = 1e-6 * amp * (ell/80)^gamma per ell >= 2; '
        'amp = 0 → nessuna mappa.')
    grp2.add_argument('--amp-beta',   type=float, default=0.0,
                      help='ampiezza dello spettro di delta-beta (amp_d_beta)')
    grp2.add_argument('--gamma-beta', type=float, default=-3.0,
                      help='pendenza dello spettro di delta-beta (gamma_d_beta)')
    grp2.add_argument('--beta-seed',  type=int,   default=None,
                      help='seed della mappa delta-beta (default seed+100000)')
    grp2.add_argument('--amp-beta-s',   type=float, default=0.0,
                      help='come --amp-beta per il sincrotrone (amp_s_beta)')
    grp2.add_argument('--gamma-beta-s', type=float, default=-3.0,
                      help='come --gamma-beta per il sincrotrone (gamma_s_beta)')
    grp2.add_argument('--beta-seed-s',  type=int,   default=None,
                      help='seed della mappa delta-beta_s (default seed+200000)')
    grp2.add_argument('--amp-u',   type=float, default=0.0,
                      help='mappa di ut = T_d*delta(1/T) della dust (amp_d_u)')
    grp2.add_argument('--gamma-u', type=float, default=-3.0,
                      help='pendenza dello spettro di ut (gamma_d_u)')
    grp2.add_argument('--u-seed',  type=int,   default=None,
                      help='seed della mappa delta-ut (default seed+300000)')
    grp2.add_argument('--rho-bu',  type=float, default=0.0,
                      help='correlation of delta-ut with the dust delta-beta '
                           '(C^bu = rho sqrt(C^bb C^uu))')

    grp3 = parser.add_argument_group('Foregrounds')
    grp3.add_argument('--pysm', type=str, default=None,
                      help="PySM preset strings for dust and synchrotron, "
                           "e.g. d1s1 (replaces the Gaussian foregrounds)")
    grp3.add_argument('--pysm-nside-in', type=int, default=0,
                      help='run PySM at this nside and downgrade in harmonic '
                           'space (band-limited, no pixel window, like CMB '
                           'and noise); 0 = PySM directly at --nside '
                           '(pixel-space ud_grade, old behaviour)')
    grp3.add_argument('--noise', default='baseline_optimistic',
                      choices=['baseline_optimistic', 'pessimistic_goal'],
                      help='SO V3 SAT noise: sensitivity and 1/f mode')
    grp3.add_argument('--no-ee', action='store_true',
                      help='test: remove the E modes of the signal (CMB + '
                           'foregrounds) before adding noise')
    grp3.add_argument('--no-kludge', action='store_true',
                      help='include_kludge=False in noise_calc (use with '
                           '--hits-noise: the hits map already gives the '
                           'non-uniform depth)')
    grp3.add_argument('--hits-noise', type=str, default=None,
                      help='hits map: scale the noise as 1/sqrt(hits), '
                           'conserving the total observing time')
    grp3.add_argument('--fg-params', type=str, default=None,
                      help='yaml file overriding FG_PARAMS (Gaussian '
                           'foreground amplitudes, slopes and SEDs)')

    args = parser.parse_args()
    if args.fg_params:
        load_fg_params(args.fg_params)

    dust_moments = DustMoments(
        omega_beta   = args.omega_beta,
        omega_u      = args.omega_u,
        sigma2_beta  = args.sigma2_beta,
        sigma2_u     = args.sigma2_u,
        sigma_beta_u = args.sigma_beta_u,
    )

    generate_maps(
        seed         = args.seed,
        nside        = args.nside,
        output_dir   = args.output_dir,
        nsplits      = args.nsplits,
        r            = args.r,
        dust_moments = dust_moments,
        amp_beta     = args.amp_beta,
        gamma_beta   = args.gamma_beta,
        beta_seed    = args.beta_seed,
        amp_beta_s   = args.amp_beta_s,
        gamma_beta_s = args.gamma_beta_s,
        beta_seed_s  = args.beta_seed_s,
        amp_u        = args.amp_u,
        gamma_u      = args.gamma_u,
        u_seed       = args.u_seed,
        pysm         = args.pysm,
        pysm_nside_in = args.pysm_nside_in,
        hits_noise   = args.hits_noise,
        rho_bu       = args.rho_bu,
        sens_mode    = 2 if args.noise == 'pessimistic_goal' else 1,
        one_over_f   = 0 if args.noise == 'pessimistic_goal' else 1,
        kludge       = not args.no_kludge,
        no_ee        = args.no_ee,
    )
