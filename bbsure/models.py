"""
Models of the decision tree: which moment spectra are switched on, and
whether r is free.

Moment groups (MOMENT_GROUPS): one spectrum each ("single moments"), the
tree adds one spectrum at a time. <c><i><j> is the spectrum w<i>0 x w<j>0
of component c (moment maps w_nm = A dbeta^n dp^m, p = 1/T_d):

    d11  beta_d x beta_d          s11  beta_s x beta_s
    d01  A_d x beta_d             s01  A_s x beta_s
    d22  beta_d^2 x beta_d^2      s22  beta_s^2 x beta_s^2
    d12  beta_d x beta_d^2        s12  beta_s x beta_s^2
    d02  A_d x beta_d^2           s02  A_s x beta_s^2
    dT11 T_d x T_d (w01 x w01)    dT01 A_d x T_d (w00 x w01)
    ds01 A_d x beta_s             ds10 beta_d x A_s
    ds11 beta_d x beta_s
    beta_d x T_d (w10 x w01): on when d11 and dT11 are both there.

Requirements, from the moment constraints: a cross spectrum needs both
auto-spectra (positivity; A x A is always there), a second-order
auto-spectrum needs the first-order one (hierarchical bound).
Each group frees its spectral parameter (beta_d, beta_s or T_d) in the
residual test; T_d is free only with dT11 / dT01.

Labels: M_<groups>_<r|0>, e.g. M_none_r, M_d11+s11_0.
"""
from dataclasses import dataclass
from typing import FrozenSet, List

# group -> config component, parameter prefix, spectrum (amplitude
# A_<pref>_<spectrum>, slope alpha_<pref>_<spectrum>), spectral parameters
# freed with the group in the residual tests, required groups
MOMENT_GROUPS = {}
for _c, _comp, _free in (('d', 'component_1', ['beta_d']),
                         ('s', 'component_2', ['beta_s'])):
    MOMENT_GROUPS.update({
        f'{_c}11': dict(comp=_comp, pref=_c, spectra=['w10w10'], free=_free),
        f'{_c}01': dict(comp=_comp, pref=_c, spectra=['w00w10'], free=_free,
                        requires=f'{_c}11'),
        f'{_c}22': dict(comp=_comp, pref=_c, spectra=['w20w20'], free=_free,
                        requires=f'{_c}11'),
        f'{_c}12': dict(comp=_comp, pref=_c, spectra=['w10w20'], free=_free,
                        requires=[f'{_c}11', f'{_c}22']),
        f'{_c}02': dict(comp=_comp, pref=_c, spectra=['w00w20'], free=_free,
                        requires=f'{_c}22'),
    })
MOMENT_GROUPS.update({
    'dT11': dict(comp='component_1', pref='d', spectra=['w01w01'],
                 free=['temp_d']),
    'dT01': dict(comp='component_1', pref='d', spectra=['w00w01'],
                 free=['temp_d'], requires='dT11'),
    # dust-synch cross terms (in the dust block, role ..._x_component_2)
    'ds01': dict(comp='component_1', pref='ds', spectra=['w00w10'],
                 free=['beta_d', 'beta_s'], requires='s11'),
    'ds10': dict(comp='component_1', pref='ds', spectra=['w10w00'],
                 free=['beta_d', 'beta_s'], requires='d11'),
    'ds11': dict(comp='component_1', pref='ds', spectra=['w10w10'],
                 free=['beta_d', 'beta_s'], requires=['d11', 's11']),
})

# cross spectra between two groups: on only if both are there
MOMENT_CROSS = {('d11', 'dT11'): dict(comp='component_1', pref='d',
                                      spectra=['w10w01'])}

# dust temperature: free only with a dT group, otherwise fixed to the
# central value of the base config
TEMP_PARAM = ('temp_d', ('dT11', 'dT01'))


def amp_name(pref, spec):
    return f'A_{pref}_{spec}'


def slope_name(pref, spec):
    return f'alpha_{pref}_{spec}'


def group_spectra(groups):
    """Spectra (component, prefix, spectrum) on for a set of groups."""
    out = []
    for g in sorted(groups):
        G = MOMENT_GROUPS[g]
        out += [(G['comp'], G['pref'], s) for s in G['spectra']]
    for pair, X in MOMENT_CROSS.items():
        if set(pair) <= set(groups):
            out += [(X['comp'], X['pref'], s) for s in X['spectra']]
    return out


def group_amplitudes(g):
    """Amplitudes belonging to group g (the cross spectra of MOMENT_CROSS
    belong to the second group of the pair)."""
    G = MOMENT_GROUPS[g]
    out = [amp_name(G['pref'], s) for s in G['spectra']]
    for pair, X in MOMENT_CROSS.items():
        if pair[1] == g:
            out += [amp_name(X['pref'], s) for s in X['spectra']]
    return out


@dataclass(frozen=True)
class Model:
    """A model of the tree: which moment groups are on, is r free?"""
    moments: FrozenSet[str] = frozenset()
    r_free: bool = True

    @property
    def label(self):
        mom = '+'.join(sorted(self.moments)) if self.moments else 'none'
        return f"M_{mom}_{'r' if self.r_free else '0'}"

    def with_r(self, r_free):
        return Model(self.moments, r_free)

    def plus(self, group):
        return Model(self.moments | {group}, self.r_free)

    def spectra(self):
        """Spectra that are on (component, prefix, spectrum)."""
        return group_spectra(self.moments)

    def amplitudes(self):
        return [amp_name(p, s) for _, p, s in self.spectra()]

    def n_moment_params(self):
        return 2 * len(self.spectra())


def model_from_label(lab):
    """M_d11+s11_r -> Model (inverse of Model.label)."""
    body, r = lab[2:].rsplit('_', 1)
    mom = frozenset() if body == 'none' else frozenset(body.split('+'))
    unknown = mom - set(MOMENT_GROUPS)
    if unknown or r not in ('r', '0'):
        raise ValueError(f"{lab}: not a single-moment model label "
                         f"(unknown groups {sorted(unknown)})")
    return Model(mom, r == 'r')


def requirements(group, extra_requires=None):
    """Groups required by `group` (MOMENT_GROUPS + extra_requires)."""
    req = MOMENT_GROUPS[group].get('requires') or []
    req = [req] if isinstance(req, str) else list(req)
    return req + list((extra_requires or {}).get(group, []))


def requirements_met(M, extra_requires=None):
    """Every group of M has the groups it requires."""
    return all(set(requirements(g, extra_requires)) <= M.moments
               for g in M.moments)


def candidate_groups(current: Model, allowed,
                     extra_requires=None) -> List[str]:
    """Groups of `allowed` not yet in the model whose required groups are
    already there."""
    return [g for g in allowed if g not in current.moments
            and set(requirements(g, extra_requires)) <= current.moments]
