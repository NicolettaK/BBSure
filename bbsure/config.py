"""
BBCompSep configs: from the base config and a Model to the config of one
run (moments on/off, r fixed, fixed parameters, starting points).
"""
import copy

import numpy as np

from .models import (MOMENT_GROUPS, TEMP_PARAM, Model, amp_name,
                     group_spectra, slope_name)


def iter_param_entries(cs):
    """
    {parameter name: config list} for all the BBCompSep parameters (the list
    is the yaml one: [label, prior, values]).
    """
    out = {}
    for k, v in cs['cmb_model']['params'].items():
        out[k] = v
    fg = cs['fg_model']
    for c in sorted(k for k in fg if k.startswith('component_')):
        comp = fg[c]
        for tag in ['sed_parameters', 'cross', 'decorr', 'moments']:
            for k, v in (comp.get(tag) or {}).items():
                out[k] = v
        for _, d in (comp.get('cl_parameters') or {}).items():
            for k, v in d.items():
                out[k] = v
    return out


def free_param_names(cs):
    """Same order as bbpower.param_manager.ParameterManager."""
    names = []

    def add(d):
        for k in sorted(d.keys()):
            if d[k][1] != 'fixed':
                names.append(k)

    add(cs['cmb_model']['params'])
    fg = cs['fg_model']
    for c in sorted(k for k in fg if k.startswith('component_')):
        comp = fg[c]
        for tag in ['sed_parameters', 'cross', 'decorr']:
            if comp.get(tag):
                add(comp[tag])
        for cl_name, d in (comp.get('cl_parameters') or {}).items():
            if all(p in cs['pol_channels'] for p in cl_name):
                add(d)
        if comp.get('moments') and fg.get('use_moments'):
            add(comp['moments'])
    return names


def gaussian_prior_chi2(cs, best):
    """-2 ln(prior) of the Gaussian priors, to remove from the BBCompSep
    chi2."""
    entries = iter_param_entries(cs)
    c2 = 0.
    for n, v in best.items():
        p = entries[n]
        if str(p[1]).lower() == 'gaussian':
            mu, sig = float(p[2][0]), float(p[2][1])
            c2 += ((v - mu) / sig) ** 2
    return c2


def _center(p):
    """Central value of a config parameter (tophat or fixed)."""
    return float(p[2][1]) if p[1] == 'tophat' else float(p[2][0])


def build_config(base, model: Model, sampler, fixed=None, centers=None,
                 settings=None, free_temp=False):
    """
    Variant of the base config for a given model.

    Moment spectra of the groups that are off: amplitude fixed to 0 and slope
    to its central value. T_d fixed to its central value without dT (unless
    free_temp).

    fixed    : {param: value} to fix (e.g. blocks at the best fit)
    centers  : {param: value} new central values of the tophats (starting
               point of the minimizer and of emcee)
    settings : additional BBCompSep keys (nlive, nwalkers, ...)
    """
    cfg = copy.deepcopy(base)
    cs = cfg['BBCompSep']
    cs['sampler'] = sampler
    if settings:
        cs.update(settings)

    fg = cs['fg_model']
    fg['use_moments'] = len(model.moments) > 0
    entries = iter_param_entries(cs)

    on = set(model.spectra())
    for c, pref, spec in group_spectra(set(MOMENT_GROUPS)):
        if (c, pref, spec) in on:
            continue
        a, al = amp_name(pref, spec), slope_name(pref, spec)
        if a in entries:
            entries[a][1:] = ['fixed', [0.0]]
        if al in entries:
            entries[al][1:] = ['fixed', [_center(entries[al])]]

    tname, tgroup = TEMP_PARAM
    if tname in entries and tgroup not in model.moments and not free_temp:
        entries[tname][1:] = ['fixed', [_center(entries[tname])]]

    if not model.r_free:
        entries['r_tensor'][1:] = ['fixed', [0.0]]

    for n, v in (centers or {}).items():
        p = entries.get(n)
        if p is not None and p[1] == 'tophat':
            lo, hi = float(p[2][0]), float(p[2][2])
            eps = 1e-6 * (hi - lo)
            p[2][1] = float(np.clip(v, lo + eps, hi - eps))

    for n, v in (fixed or {}).items():
        entries[n][1:] = ['fixed', [float(v)]]

    return cfg
