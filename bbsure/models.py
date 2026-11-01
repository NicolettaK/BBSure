"""
Models of the decision tree: which groups of moment spectra are switched on,
and whether r is free.

Moment groups (MOMENT_GROUPS)
-----------------------------
    d    1st order in beta_d: w00 x w10, w10 x w10            (beta_d free)
    s    1st order in beta_s: same for synchrotron            (beta_s free)
    dT   1st order in p = 1/T_d: w00 x w01, w01 x w01         (T_d free)
         (+ w10 x w01 if d is there too)
    d2   2nd order in beta_d: w00 x w20, w10 x w20, w20 x w20 (only with d)
    s2   2nd order in beta_s: same                            (only with s)
    ds1  dust-synch cross: A_d x w1^beta_s, w1^beta_d x A_s
                                                    (beta_d and beta_s free)
    ds2  cross w1^beta_d x w1^beta_s                      (only with d and s)

Labels: M_<groups>[-<pruned spectra>]_<r|0>, e.g. M_none_r, M_d+s_0,
M_d+s-d_w00w10_r.
"""
from dataclasses import dataclass
from typing import FrozenSet, List

# group -> config component, parameter prefix, moment spectra
# (A_<pref>_<spectrum>, alpha_<pref>_<spectrum>), spectral parameters freed
# with the group in the residual tests, required group
MOMENT_GROUPS = {
    'd':  dict(comp='component_1', pref='d', spectra=['w00w10', 'w10w10'],
               free=['beta_d']),
    's':  dict(comp='component_2', pref='s', spectra=['w00w10', 'w10w10'],
               free=['beta_s']),
    'dT': dict(comp='component_1', pref='d', spectra=['w00w01', 'w01w01'],
               free=['temp_d']),
    'd2': dict(comp='component_1', pref='d',
               spectra=['w00w20', 'w10w20', 'w20w20'], free=['beta_d'],
               requires='d'),
    's2': dict(comp='component_2', pref='s',
               spectra=['w00w20', 'w10w20', 'w20w20'], free=['beta_s'],
               requires='s'),
    # dust-synch cross terms (in the dust block, role ..._x_component_2):
    # ds1 = A_d x w1^beta_s (w00w10) and w1^beta_d x A_s (w10w00);
    # ds2 = w1^beta_d x w1^beta_s, only with d and s (|C| <= sqrt(C^d_11 C^s_11))
    'ds1': dict(comp='component_1', pref='ds', spectra=['w00w10', 'w10w00'],
                free=['beta_d', 'beta_s']),
    'ds2': dict(comp='component_1', pref='ds', spectra=['w10w10'],
                free=['beta_d', 'beta_s'], requires=['d', 's']),
}

# cross spectra between two groups: on only if both are there
MOMENT_CROSS = {('d', 'dT'): dict(comp='component_1', pref='d',
                                  spectra=['w10w01'])}

# dust temperature: free only with the dT group, otherwise fixed to the
# central value of the base config
TEMP_PARAM = ('temp_d', 'dT')

# Res_r0 candidates that switch back on spectra removed by the pruning
RESTORE = 'restore-'


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


def cand_group(cand):
    """Group of a candidate ('d' or 'restore-d' -> 'd')."""
    return cand[len(RESTORE):] if cand.startswith(RESTORE) else cand


@dataclass(frozen=True)
class Model:
    """
    A model of the tree: which moment groups are on, is r free?
    off: amplitudes of spectra of the groups that are on, removed by the
    pruning (A = 0).
    """
    moments: FrozenSet[str] = frozenset()
    r_free: bool = True
    off: FrozenSet[str] = frozenset()

    @property
    def label(self):
        mom = '+'.join(sorted(self.moments)) if self.moments else 'none'
        r = 'r' if self.r_free else '0'
        cut = ''.join(f"-{a[2:]}" for a in sorted(self.off))
        return f"M_{mom}{cut}_{r}"

    def with_r(self, r_free):
        return Model(self.moments, r_free, self.off)

    def plus(self, group):
        return Model(self.moments | {group}, self.r_free, self.off)

    def without(self, amps):
        return Model(self.moments, self.r_free, self.off | set(amps))

    def apply(self, cand):
        """
        Model with the candidate `cand`: a new group ('d', 's', ...) or
        'restore-<group>', which switches back on the spectra of that group
        removed by the pruning.
        """
        if cand.startswith(RESTORE):
            g = cand[len(RESTORE):]
            return Model(self.moments, self.r_free,
                         self.off - set(group_amplitudes(g)))
        return self.plus(cand)

    def spectra(self):
        """Spectra that are on (component, prefix, spectrum)."""
        return [x for x in group_spectra(self.moments)
                if amp_name(x[1], x[2]) not in self.off]

    def amplitudes(self):
        return [amp_name(p, s) for _, p, s in self.spectra()]

    def n_moment_params(self):
        return 2 * len(self.spectra())


def model_from_label(lab):
    """M_d+s-d_w00w10-s_w00w10_r -> Model (inverse of Model.label)."""
    body, r = lab[2:].rsplit('_', 1)
    parts = body.split('-')
    mom = frozenset() if parts[0] == 'none' else frozenset(parts[0].split('+'))
    m = Model(mom, r == 'r', frozenset('A_' + p for p in parts[1:]))
    assert m.label == lab, (m.label, lab)
    return m


def candidate_groups(current: Model, allowed, restore=False,
                     extra_requires=None) -> List[str]:
    """
    Groups not yet in the model whose required group is already there;
    with restore=True also 'restore-<group>' for the groups of the model with
    spectra removed by the pruning. extra_requires: additional
    {group: [groups]} (e.g. with the positivity ds1 requires s, the
    auto-spectrum of w1^beta_s).
    """
    # a group with all its spectra removed by the pruning does not satisfy
    # the requirements (e.g. s2 without the spectra of s: with the
    # hierarchical bound and the positivity the likelihood is -inf
    # everywhere)
    active = {g for g in current.moments
              if set(group_amplitudes(g)) - set(current.off)}
    out = []
    for g in allowed:
        req = MOMENT_GROUPS[g].get('requires') or []
        req = [req] if isinstance(req, str) else list(req)
        req += list((extra_requires or {}).get(g, []))
        if g not in current.moments and set(req) <= active:
            out.append(g)
    if restore:
        for g in sorted(current.moments):
            if set(group_amplitudes(g)) & set(current.off):
                out.append(RESTORE + g)
    return out
