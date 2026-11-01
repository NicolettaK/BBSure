#!/usr/bin/env python
"""
SO cross-cl pipeline: decisional tree (no blinding version), v2.

Implementa il flowchart sopra BBCompSep di BBPower (branch moments-free),
con i momenti nel formalismo di Vacher et al.: mappe dei momenti
w_nm = A dbeta^n dp^m (p = 1/T) e spettri incrociati liberi
A (ell/80)^alpha. Il data vector (cells_coadded, cells_noise) va prodotto
prima, UNA volta, con run_spectra.sh: qui cambia solo il modello.
La versione precedente (momenti sigma/gamma di BBPower) e' decision_tree_v1.py.

Notazione
---------
    M_{0,r}   modello corrente, r libero         (all'inizio: senza momenti)
    M_{0,0}   modello corrente, r fissato a 0
    M_{1,r}   modello con un gruppo di momenti in piu', r libero
    M_{1,0}   lo stesso con r fissato a 0
    M*_{.,r}  profile likelihood in r del modello

Gruppi di momenti (MOMENT_GROUPS)
---------------------------------
    d    1o ordine in beta_d: w00 x w10, w10 x w10            (beta_d libero)
    s    1o ordine in beta_s: idem per il sincrotrone          (beta_s libero)
    dT   1o ordine in p = 1/T_d: w00 x w01, w01 x w01          (T_d libero)
         (+ w10 x w01 se c'e' anche d)
    d2   2o ordine in beta_d: w00 x w20, w10 x w20, w20 x w20  (solo con d)
    s2   2o ordine in beta_s: idem                             (solo con s)
    ds1  incrociati dust-synch: A_d x w1^beta_s, w1^beta_d x A_s
                                                    (beta_d e beta_s liberi)
    ds2  incrociato w1^beta_d x w1^beta_s                  (solo con d e s)

Flowchart
---------
 1. Fit di M_0,r. chi2 buono (PTE > pte_min) e r compatibile con 0 a 1 sigma
    -> upper limit.
 2. Res_wr: test dei residui con tutto fissato al best fit di M_0,r, r
    compreso; candidati i gruppi res_wr (d, s) non ancora nel modello, ognuno
    con il suo parametro spettrale libero. Rivelato = lo zero e' fuori
    dall'intervallo HPD al 68% di almeno una ampiezza (screening con emcee,
    poi PolyChord sui rivelati: conferma ed evidenza).
      - nessuno rivelato -> 3.
      - in ordine di evidenza: fit completo di M_0,r + gruppo e confronto con
        M_0,r (ln B > lnB_preferred). Il primo che passa diventa M_0,r -> 1.
      - nessuno passa -> test a 3 sigma se il modello non ha ancora momenti,
        altrimenti -> 3.
 Potatura immediata: prima di ogni confronto con M_0,r si tolgono gli spettri
 nuovi non rivelati (zero nell'HPD della loro ampiezza, screening con ML +
 emcee) e si rifa' lo screening; solo il modello potato si fitta con
 PolyChord e si confronta (scartato se non resta niente di nuovo). Gli
 spettri tolti non tornano candidati in Res_r0 (restore-<gruppo> solo con
 moments: res_r0_restore: true).
 3. Res_r0: come Res_wr ma sui residui di M_0,0 (r = 0), candidati i gruppi
    res_r0 (d, s, dT, d2, s2). Nessuno rivelato -> test a 3 sigma. Il
    migliore (evidenza) -> M_1,r: preferito su M_0,r -> sostituisce M_0,r,
    torna a 1. Altrimenti M_1,0: preferito su M_0,r -> ramo frequentista
    (con run: profile_scheme: true niente ramo frequentista: il tree va al
    test a 3 sigma e poi parte lo schema profilo, profile_tree.py, che
    riparte da M_1; output in <runs_dir>/profile, conclusion.json finale con
    la parte bayesiana e quella dal profilo)
    (profile likelihood), altrimenti test a 3 sigma.
    Res_wr si ripete finche' le sostituzioni vengono da Res_wr (es. dopo
    M + d si testa s); dopo una sostituzione venuta da Res_r0 si va sempre
    direttamente a Res_r0.
 4. Pruning (anche di ogni nuovo M_0,r appena adottato): si tolgono gli spettri con lo zero dentro
    l'HPD della loro ampiezza e si rifitta; il modello potato si tiene se
    ln B(potato vs completo) > lnB_prune (-2.5). Si ripete finche' restano
    solo ampiezze rivelate.
 Se dopo il pruning il nuovo M_0,r e' un modello gia' usato come M_0,r
 (ciclo, es. restore-s adottato e poi ripotato) -> test a 3 sigma.
 5. Test a 3 sigma su r di M_0,r (o upper limit dal punto 1) sul modello
    potato: upper limit oppure r detection.
 Confronti in bilico: se |ln B - soglia| < seed_band * errore, i due fit
 PolyChord si ripetono con un altro seme (seed casuale) e si media ln Z,
 fino a seed_max_runs run per fit. Riduce solo l'errore numerico, non la
 fluttuazione della realizzazione.

Output (in runs_dir)
--------------------
  decision_log.json   ogni nodo del tree con domanda, risposta e numeri
  decision_log.txt    stessa cosa in forma leggibile
  <tag>/              una cartella per ogni run di BBCompSep (config + output)

Uso
---
  python decision_tree.py --config config/tree_config.yml
"""

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional

import numpy as np
import yaml
from scipy import stats

# ═══════════════════════════════════════════════════════════════════════════
# Modelli
# ═══════════════════════════════════════════════════════════════════════════

# gruppo -> componente del config, prefisso dei parametri, spettri dei momenti
# (A_<pref>_<spettro>, alpha_<pref>_<spettro>), parametri spettrali liberati
# con il gruppo nei test dei residui, gruppo richiesto
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
    # incrociati dust-synch (nel blocco della dust, ruolo ..._x_component_2):
    # ds1 = A_d x w1^beta_s (w00w10) e w1^beta_d x A_s (w10w00);
    # ds2 = w1^beta_d x w1^beta_s, solo con d e s (|C| <= sqrt(C^d_11 C^s_11))
    'ds1': dict(comp='component_1', pref='ds', spectra=['w00w10', 'w10w00'],
                free=['beta_d', 'beta_s']),
    'ds2': dict(comp='component_1', pref='ds', spectra=['w10w10'],
                free=['beta_d', 'beta_s'], requires=['d', 's']),
}

# spettri incrociati tra due gruppi: accesi solo se ci sono entrambi
MOMENT_CROSS = {('d', 'dT'): dict(comp='component_1', pref='d',
                                  spectra=['w10w01'])}

# temperatura della dust: libera solo con il gruppo dT, altrimenti fissata
# al valore centrale del base config
TEMP_PARAM = ('temp_d', 'dT')


def group_spectra(groups):
    """Spettri (componente, prefisso, spettro) accesi per un insieme di gruppi."""
    out = []
    for g in sorted(groups):
        G = MOMENT_GROUPS[g]
        out += [(G['comp'], G['pref'], s) for s in G['spectra']]
    for pair, X in MOMENT_CROSS.items():
        if set(pair) <= set(groups):
            out += [(X['comp'], X['pref'], s) for s in X['spectra']]
    return out


# candidati di Res_r0 che riaccendono spettri tolti dal pruning
RESTORE = 'restore-'


def group_amplitudes(g):
    """Ampiezze che appartengono al gruppo g (gli spettri incrociati di
    MOMENT_CROSS appartengono al secondo gruppo della coppia)."""
    G = MOMENT_GROUPS[g]
    out = [amp_name(G['pref'], s) for s in G['spectra']]
    for pair, X in MOMENT_CROSS.items():
        if pair[1] == g:
            out += [amp_name(X['pref'], s) for s in X['spectra']]
    return out


def cand_group(cand):
    """Gruppo di un candidato ('d' o 'restore-d' -> 'd')."""
    return cand[len(RESTORE):] if cand.startswith(RESTORE) else cand


def amp_name(pref, spec):
    return f'A_{pref}_{spec}'


def slope_name(pref, spec):
    return f'alpha_{pref}_{spec}'


@dataclass(frozen=True)
class Model:
    """
    Un modello del tree: quali gruppi di momenti sono accesi, r libero?
    off: ampiezze di spettri dei gruppi accesi tolte dal pruning (A = 0).
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
        Modello con il candidato `cand`: un gruppo nuovo ('d', 's', ...) o
        'restore-<gruppo>', che riaccende gli spettri di quel gruppo tolti
        dal pruning.
        """
        if cand.startswith(RESTORE):
            g = cand[len(RESTORE):]
            return Model(self.moments, self.r_free,
                         self.off - set(group_amplitudes(g)))
        return self.plus(cand)

    def spectra(self):
        """Spettri accesi (componente, prefisso, spettro)."""
        return [x for x in group_spectra(self.moments)
                if amp_name(x[1], x[2]) not in self.off]

    def amplitudes(self):
        return [amp_name(p, s) for _, p, s in self.spectra()]

    def n_moment_params(self):
        return 2 * len(self.spectra())


def model_from_label(lab):
    """M_d+s-d_w00w10-s_w00w10_r -> Model (inverso di Model.label)."""
    body, r = lab[2:].rsplit('_', 1)
    parts = body.split('-')
    mom = frozenset() if parts[0] == 'none' else frozenset(parts[0].split('+'))
    m = Model(mom, r == 'r', frozenset('A_' + p for p in parts[1:]))
    assert m.label == lab, (m.label, lab)
    return m


def candidate_groups(current: Model, allowed, restore=False,
                     extra_requires=None) -> List[str]:
    """
    Gruppi non ancora nel modello e con il gruppo richiesto gia' presente;
    con restore=True anche 'restore-<gruppo>' per i gruppi del modello con
    spettri tolti dal pruning. extra_requires: {gruppo: [gruppi]} in piu'
    (es. con la positivita' ds1 richiede s, l'auto-spettro di w1^beta_s).
    """
    # un gruppo con tutti gli spettri tolti dal pruning non soddisfa i
    # requisiti (es. s2 senza spettri di s: con il prior gerarchico e la
    # positivita' la likelihood e' -inf ovunque)
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


# ═══════════════════════════════════════════════════════════════════════════
# Risultati dei fit
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FitResult:
    model: Model
    tag: str
    names: List[str] = field(default_factory=list)
    best: Dict[str, float] = field(default_factory=dict)  # best fit (ML)
    chi2: float = np.nan        # -2 lnL al best fit (senza prior)
    ndata: int = 0
    logZ: float = np.nan
    logZerr: float = np.nan
    r_samples: Optional[np.ndarray] = None   # posterior equal-weight di r
    sample_names: List[str] = field(default_factory=list)
    samples: Optional[np.ndarray] = None     # posterior equal-weight
    # un elemento per run di PolyChord (semi diversi): logZ e' la media
    logZ_runs: List[float] = field(default_factory=list)
    logZerr_runs: List[float] = field(default_factory=list)

    @property
    def nfree(self):
        return len(self.names)

    @property
    def dof(self):
        return self.ndata - self.nfree

    @property
    def pte(self):
        return float(stats.chi2.sf(self.chi2, self.dof))

    def r_summary(self):
        if self.r_samples is None or len(self.r_samples) == 0:
            return {}
        s = self.r_samples
        q = np.percentile(s, [0.135, 2.275, 15.865, 50., 84.135, 97.725,
                              99.865, 95.])
        # upper limit: 95% del posterior con r >= 0 (il prior del tree arriva
        # a r < 0 per non distorcere mediana e bias, ma r_95 negativo non ha
        # senso fisico); r_95_full = 95% con il prior completo (come prima)
        pos = s[s >= 0.]
        r95 = float(np.percentile(pos, 95.)) if len(pos) else 0.
        return dict(r_med=q[3], r_lo1=q[2], r_hi1=q[4], r_lo2=q[1],
                    r_hi2=q[5], r_lo3=q[0], r_hi3=q[6], r_95=r95,
                    r_95_full=q[7],
                    r_mean=float(np.mean(s)), r_std=float(np.std(s)),
                    n_samples=len(s))

    def r_compatible_with_zero(self, nsigma):
        """0 dentro l'intervallo centrale a nsigma del posterior di r."""
        rs = self.r_summary()
        lo, hi = rs[f'r_lo{nsigma}'], rs[f'r_hi{nsigma}']
        return (lo <= 0.0 <= hi), rs

    def summary(self):
        d = dict(model=self.model.label, tag=self.tag, nfree=self.nfree,
                 chi2=self.chi2, dof=self.dof,
                 chi2_red=self.chi2 / self.dof if self.dof > 0 else np.nan,
                 pte=self.pte if self.dof > 0 else np.nan,
                 logZ=self.logZ, logZerr=self.logZerr,
                 n_pc_runs=max(1, len(self.logZ_runs)),
                 best=self.best)
        d.update(self.r_summary())
        return d


@dataclass
class ResidualResult:
    """Test dei residui di un gruppo di momenti."""
    group: str
    detected: bool
    hpd: Dict[str, tuple]               # ampiezza -> intervallo HPD
    logZ: float = np.nan                # solo se rivelato (PolyChord)
    logZerr: float = np.nan
    best: Dict[str, float] = field(default_factory=dict)  # max likelihood
    stage: str = ''                     # 'emcee' o 'polychord'
    off: tuple = ()                     # spettri del gruppo non rivelati
                                        # dallo screening, tolti prima di PC


def lnB(a: FitResult, b: FitResult):
    """ln B_ab = lnZ_a - lnZ_b, con errore."""
    return a.logZ - b.logZ, float(np.hypot(a.logZerr, b.logZerr))


def hpd_interval(x, frac=0.68, bounds=None):
    """
    Intervallo di massima densita' (il piu' corto) che contiene frac.
    Se sotto (sopra) l'intervallo resta meno dell'1% dei campioni e il
    campione estremo dista dal bordo del prior meno del 5% della larghezza
    dell'intervallo, il posterior e' schiacciato contro il bordo: con
    bounds = (lo, hi)
    l'intervallo viene esteso fino al bordo (i campioni non toccano mai
    esattamente il bordo, es. un'ampiezza >= 0 compatibile con zero, e il
    rumore dei campioni sposta di poco l'inizio dell'intervallo).
    """
    xs = np.sort(np.asarray(x))
    n = len(xs)
    k = max(int(np.ceil(frac * n)), 1)
    widths = xs[k - 1:] - xs[:n - k + 1]
    i = int(np.argmin(widths))
    lo, hi = float(xs[i]), float(xs[i + k - 1])
    if bounds is not None:
        edge, tol = 0.01 * n, 0.05 * (hi - lo)
        if i < edge and xs[0] - bounds[0] < tol:
            lo = float(bounds[0])
        if n - (i + k) < edge and bounds[1] - xs[-1] < tol:
            hi = float(bounds[1])
    return lo, hi


def detection(names, samples, amps, frac=0.68, bounds=None):
    """
    HPD di ogni ampiezza in `amps`; rivelato se lo zero e' fuori dall'HPD di
    almeno una. bounds: {ampiezza: (lo, hi)} del prior.
    """
    bounds = bounds or {}
    hpd = {a: hpd_interval(samples[:, names.index(a)], frac, bounds.get(a))
           for a in amps}
    det = any(not (lo <= 0.0 <= hi) for lo, hi in hpd.values())
    return det, hpd


# ═══════════════════════════════════════════════════════════════════════════
# Gestione dei config BBCompSep
# ═══════════════════════════════════════════════════════════════════════════

def iter_param_entries(cs):
    """
    Restituisce {nome_parametro: lista_config} per tutti i parametri di
    BBCompSep (la lista e' quella del yaml: [label, prior, valori]).
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
    """Stesso ordine di bbpower.param_manager.ParameterManager."""
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
    """Termine -2 ln(prior) gaussiano, da togliere al chi2 di BBCompSep."""
    entries = iter_param_entries(cs)
    c2 = 0.
    for n, v in best.items():
        p = entries[n]
        if str(p[1]).lower() == 'gaussian':
            mu, sig = float(p[2][0]), float(p[2][1])
            c2 += ((v - mu) / sig) ** 2
    return c2


def _center(p):
    """Valore centrale di un parametro del config (tophat o fixed)."""
    return float(p[2][1]) if p[1] == 'tophat' else float(p[2][0])


def build_config(base, model: Model, sampler, fixed=None, centers=None,
                 settings=None):
    """
    Variante del config base per un dato modello.

    Spettri dei momenti dei gruppi non accesi: ampiezza fissata a 0 e
    pendenza al valore centrale. T_d fissata al valore centrale senza dT.

    fixed    : {param: valore} da fissare (es. blocchi al best fit)
    centers  : {param: valore} nuovi valori centrali dei tophat (punto di
               partenza del minimizer e di emcee)
    settings : chiavi aggiuntive di BBCompSep (nlive, nwalkers, ...)
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
    if tname in entries and tgroup not in model.moments:
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


# ═══════════════════════════════════════════════════════════════════════════
# Esecuzione di BBCompSep
# ═══════════════════════════════════════════════════════════════════════════

# ── BBCompSep nello stesso processo (fit veloci) ──────────────────────────
# Lanciare `python -m bbpower` costa l'avvio di Python, gli import e la lettura
# degli spettri a ogni fit (~4 s sul Mac, ~50 s al NERSC), contro ~1 s di fit
# ML. ML, single_point e predicted_spectra girano qui dentro, con i file sacc
# letti una volta sola; PolyChord (MPI) ed emcee (Pool) restano processi a
# parte. Un lucchetto li serializza (stdout e' globale; il fit e' legato al
# GIL, i thread non darebbero comunque parallelismo).
INPROC_SAMPLERS = ('maximum_likelihood', 'single_point', 'predicted_spectra')
_INPROC_LOCK = threading.Lock()
_SACC_CACHE = {}


def _run_inproc(args, log_path):
    """BBCompSep con gli argomenti della riga di comando `args`; output in
    log_path. Ritorna True se e' andato a buon fine."""
    import contextlib
    import copy
    import traceback
    import sacc
    from bbpower.compsep import BBCompSep
    if not getattr(sacc.Sacc, '_tree_cached', False):
        orig = sacc.Sacc.load_fits.__func__

        def load_cached(cls, fname, *a, **k):
            key = (os.path.abspath(fname), os.path.getmtime(fname))
            if key not in _SACC_CACHE:
                _SACC_CACHE[key] = orig(cls, fname, *a, **k)
            return copy.deepcopy(_SACC_CACHE[key])
        sacc.Sacc.load_fits = classmethod(load_cached)
        sacc.Sacc._tree_cached = True
    args = dict(args)
    for x in BBCompSep.config_options:
        args.setdefault(x, None)
    with _INPROC_LOCK, open(log_path, 'w') as out, \
            contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            BBCompSep(args).run()
            return True
        except Exception:
            traceback.print_exc()
            return False


class BBCompSepRunner:
    """Lancia `python -m bbpower BBCompSep` con un config generato."""

    def __init__(self, tc, logger):
        self.tc = tc
        self.log = logger
        p = tc['paths']
        self.base = yaml.safe_load(open(p['base_config']))
        self.runs_dir = p['runs_dir']
        self.reuse = tc['run'].get('reuse_runs', True)
        # fit veloci nello stesso processo (vedi _run_inproc)
        self.inproc = tc['run'].get('inproc', True)
        self.pc_cfg = tc['polychord']
        self.mc_cfg = tc.get('emcee', {})
        os.makedirs(self.runs_dir, exist_ok=True)

    def pc_settings(self, ndim):
        """
        nlive e nrepeat di PolyChord in proporzione al numero di parametri
        liberi (regole pratiche: nlive ~ 25 ndim, num_repeats ~ 2-5 ndim).
        Senza le chiavi *_per_dim usa i valori fissi nlive / nrepeat.
        """
        c = self.pc_cfg
        if 'nlive_per_dim' not in c:
            return dict(nlive=c['nlive'], nrepeat=c['nrepeat'])
        return dict(nlive=max(c['nlive_min'], c['nlive_per_dim'] * ndim),
                    nrepeat=max(c['nrepeat_min'],
                                c['nrepeat_per_dim'] * ndim))

    # ── basso livello ─────────────────────────────────────────────────────
    def _run(self, tag, cfg, result_file):
        d = os.path.join(self.runs_dir, tag)
        os.makedirs(os.path.join(d, 'polychord'), exist_ok=True)
        txt = yaml.safe_dump(cfg, sort_keys=False)
        h = hashlib.md5(txt.encode()).hexdigest()
        cfg_path = os.path.join(d, 'config.yml')
        hash_path = os.path.join(d, 'config.md5')
        res_path = os.path.join(d, result_file)

        if (self.reuse and os.path.isfile(res_path)
                and os.path.isfile(hash_path)
                and open(hash_path).read().strip() == h):
            self.log.info(f"    [reuse] {tag}")
            return d

        # checkpoint di emcee di un run interrotto (es. job ucciso dal limite
        # di tempo): BBPower riprenderebbe da un file troncato e fallisce
        stale = os.path.join(d, 'emcee.npz.h5')
        if os.path.isfile(stale):
            os.remove(stale)

        with open(cfg_path, 'w') as f:
            f.write(txt)
        p = self.tc['paths']
        cmd = [sys.executable, '-m', 'bbpower', 'BBCompSep',
               f"--cells_coadded={p['cells_coadded']}",
               f"--cells_noise={p['cells_noise']}",
               f"--cells_fiducial={p['cells_fiducial']}",
               # covarianza: di default quella del sacc dei dati (sims);
               # con paths: cells_coadded_cov quella del best fit
               f"--cells_coadded_cov={p.get('cells_coadded_cov', p['cells_coadded'])}",
               f"--output_dir={d}",
               f"--config_copy={os.path.join(d, 'config_copy.yml')}",
               f"--config={cfg_path}"]
        sampler = cfg['BBCompSep']['sampler']
        if self.inproc and sampler in INPROC_SAMPLERS:
            self.log.info(f"    [run] {tag} ({sampler}, in-process)")
            t0 = time.time()
            log_path = os.path.join(d, 'stdout.log')
            args = {a[2:].split('=', 1)[0]: a.split('=', 1)[1]
                    for a in cmd[4:]}
            if not _run_inproc(args, log_path) or not os.path.isfile(res_path):
                raise RuntimeError(f"BBCompSep failed for {tag}: see "
                                   f"{log_path}")
            with open(hash_path, 'w') as f:
                f.write(h)
            self.log.info(f"          done in {time.time() - t0:.1f} s")
            return d
        # PolyChord su piu' processi MPI (1 master + nproc-1 per la
        # likelihood); ML ed emcee restano seriali (emcee usa un Pool)
        nproc = int(self.pc_cfg.get('mpi_procs', 1))
        nlaunch = 0
        if sampler == 'polychord' and nproc > 1:
            # comando MPI: di default il mpirun dell'ambiente; sul cluster
            # es. polychord: mpi_launcher: "srun -n {n} --cpu-bind=cores"
            launcher = self.pc_cfg.get('mpi_launcher')
            if launcher:
                pre = launcher.format(n=nproc).split()
            else:
                pre = [os.path.join(os.path.dirname(sys.executable),
                                    'mpirun'), '-np', str(nproc)]
            nlaunch = len(pre)
            cmd = pre + cmd
        self.log.info(f"    [run] {tag} ({sampler}"
                      f"{f', {nproc} MPI procs' if sampler == 'polychord' and nproc > 1 else ''})")
        t0 = time.time()
        log_path = os.path.join(d, 'stdout.log')
        with open(log_path, 'w') as out:
            ret = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)
        if (ret.returncode != 0 and sampler == 'polychord' and nproc > 1
                and 'Finished sampling' not in open(log_path).read()):
            # MPI non parte (PMIx, senza rete) o crolla a meta' (cambio di
            # rete, stop del Mac): PolyChord in seriale
            os.replace(log_path, log_path + '.mpi_failed')
            self.log.info(f"    [warn] {tag}: MPI run failed (network "
                          "change?), running PolyChord serially")
            with open(log_path, 'w') as out:
                ret = subprocess.run(cmd[nlaunch:], stdout=out,
                                     stderr=subprocess.STDOUT)
        if (ret.returncode != 0 and sampler == 'polychord'
                and os.path.isfile(res_path)
                and 'Finished sampling' in open(log_path).read()):
            # PolyChord ha finito ma MPI e' uscito male in chiusura (es. rete
            # cambiata dopo uno stop del Mac): i risultati sono completi
            self.log.info(f"    [warn] {tag}: MPI exited with code "
                          f"{ret.returncode} after sampling finished; "
                          "results kept")
        elif ret.returncode != 0 or not os.path.isfile(res_path):
            raise RuntimeError(f"BBCompSep failed for {tag}: see {log_path}")
        with open(hash_path, 'w') as f:
            f.write(h)
        self.log.info(f"          done in {time.time() - t0:.0f} s")
        return d

    def ml(self, model, tag, fixed=None, centers=None):
        """Maximum likelihood: best fit e chi2 (senza prior gaussiani)."""
        cfg = build_config(self.base, model, 'maximum_likelihood',
                           fixed, centers)
        d = self._run(tag, cfg, 'chi2.npz')
        z = np.load(os.path.join(d, 'chi2.npz'))
        names = [str(n) for n in z['names']]
        expected = free_param_names(cfg['BBCompSep'])
        assert names == expected, f"unexpected parameter order: {names}"
        best = dict(zip(names, map(float, np.atleast_1d(z['params']))))
        chi2 = float(z['chi2']) - gaussian_prior_chi2(cfg['BBCompSep'], best)
        return FitResult(model, tag, names, best, chi2, int(z['ndof']))

    def polychord(self, model, tag, fixed=None, centers=None):
        """
        Evidenza e posterior (equal weights) con PolyChord (seme casuale:
        un tag diverso, es. _rep2, da' un run indipendente).
        Ritorna logZ, logZerr, nomi, campioni [nsamp, npar], -2lnL.
        """
        cfg = build_config(self.base, model, 'polychord', fixed, centers)
        names = free_param_names(cfg['BBCompSep'])
        cfg['BBCompSep'].update(self.pc_settings(len(names)))
        d = self._run(tag, cfg, os.path.join('polychord', 'pch.stats'))
        logZ, logZerr = parse_polychord_stats(
            os.path.join(d, 'polychord', 'pch.stats'))
        samp = np.loadtxt(os.path.join(d, 'polychord',
                                       'pch_equal_weights.txt'), ndmin=2)
        # colonne: peso, -2lnL, parametri
        return logZ, logZerr, names, samp[:, 2:], samp[:, 1]

    def emcee(self, model, tag, fixed=None, centers=None):
        """
        Posterior con emcee (screening della rivelazione). Ritorna nomi e
        campioni dopo il burn-in (frazione burn_frac della catena scartata).
        """
        c = self.mc_cfg
        cfg = build_config(self.base, model, 'emcee', fixed, centers)
        names = free_param_names(cfg['BBCompSep'])
        # emcee (red-blue move) vuole almeno 2 walker per parametro
        cfg['BBCompSep'].update(
            nwalkers=max(int(c.get('nwalkers', 32)), 2 * len(names) + 2),
            n_iters=int(c.get('n_iters', 2000)))
        d = self._run(tag, cfg, 'emcee.npz')
        chain = np.load(os.path.join(d, 'emcee.npz'))['chain']
        burn = int(chain.shape[1] * c.get('burn_frac', 0.5))
        return names, chain[:, burn:, :].reshape(-1, chain.shape[-1])


def parse_polychord_stats(fname):
    txt = open(fname).read()
    m = re.search(r'log\(Z\)\s*=\s*([-+0-9.eE]+)\s*\+/-\s*([-+0-9.eE]+)', txt)
    if not m:
        raise RuntimeError(f"log(Z) not found in {fname}")
    return float(m.group(1)), float(m.group(2))


class NoValidStart(RuntimeError):
    """Nessun punto con likelihood finita (vincoli sui momenti)."""


class Fitter:
    """
    Operazioni di alto livello usate dal tree. Una MockFitter con la stessa
    interfaccia viene usata in test_decision_tree.py.
    """

    def __init__(self, runner: BBCompSepRunner, hpd_frac=0.68):
        self.R = runner
        self.hpd_frac = hpd_frac
        self._cache = {}

    def full_fit(self, model: Model, centers=None) -> FitResult:
        """ML (best fit, chi2) + PolyChord (evidenza, posterior di r)."""
        key = ('full', model)
        if key in self._cache:
            return self._cache[key]
        res = copy.copy(self.ml_fit(model, centers=centers))
        logZ, logZerr, names, samp, m2lnl = self.R.polychord(
            model, f"{model.label}_PC", centers=res.best)
        if not np.isfinite(res.chi2):
            # il minimizer e' partito fuori dalla regione permessa dai
            # vincoli sui momenti (lnL = -inf) ed e' rimasto li': si riparte
            # dal campione PolyChord con la likelihood piu' alta
            start = dict(zip(names, map(float, samp[int(np.argmin(m2lnl))])))
            ml = self.R.ml(model, f"{model.label}_ML2", centers=start)
            if np.isfinite(ml.chi2):
                res.best, res.chi2 = ml.best, ml.chi2
            else:
                # -2lnL di PolyChord: gia' senza i prior
                res.best, res.chi2 = start, float(np.min(m2lnl))
        res.logZ, res.logZerr = logZ, logZerr
        res.logZ_runs, res.logZerr_runs = [logZ], [logZerr]
        res.sample_names, res.samples = names, samp
        if 'r_tensor' in names:
            res.r_samples = samp[:, names.index('r_tensor')]
        self._cache[key] = res
        return res

    def repeat_pc(self, res: FitResult):
        """
        Un altro run PolyChord di res.model con un seme diverso. Aggiorna res
        sul posto: logZ = media dei run, errore = sqrt(sum err^2) / N,
        campioni concatenati.
        """
        k = len(res.logZ_runs) + 1
        logZ, logZerr, names, samp, _ = self.R.polychord(
            res.model, f"{res.model.label}_PC_rep{k}", centers=res.best)
        assert names == res.sample_names
        res.logZ_runs.append(logZ)
        res.logZerr_runs.append(logZerr)
        n = len(res.logZ_runs)
        res.logZ = float(np.mean(res.logZ_runs))
        res.logZerr = float(np.sqrt(np.sum(np.square(res.logZerr_runs))) / n)
        res.samples = np.vstack([res.samples, samp])
        if 'r_tensor' in names:
            res.r_samples = res.samples[:, names.index('r_tensor')]
        return res

    def prior_bounds(self, params):
        """Bordi dei prior {param: (lo, hi)} dal config base."""
        entries = iter_param_entries(self.R.base['BBCompSep'])
        return {a: (float(entries[a][2][0]), float(entries[a][2][-1]))
                for a in params}

    def screen_fit(self, model: Model, centers=None) -> FitResult:
        """
        ML + emcee (senza evidenza): serve solo a vedere quali ampiezze sono
        rivelate prima della potatura immediata.
        """
        key = ('screen', model)
        if key in self._cache:
            return self._cache[key]
        res = copy.copy(self.ml_fit(model, centers=centers))
        if not np.isfinite(res.chi2):
            # partenza fuori dalla regione permessa dai vincoli sui momenti
            # (lnL = -inf ovunque intorno): si riparte con tutte le ampiezze
            # a zero tranne gli auto-spettri del primo ordine (w10w10,
            # w01w01), che restano dove sono: positivita' e prior gerarchico
            # sono soddisfatti (lo zero dei tophat diventa 1e-6 della
            # larghezza, quindi anche i primi ordini a zero li violerebbero)
            start = dict(centers or {})
            start.update({a: 0. for a in model.amplitudes()
                          if not re.match(r'A_[ds]_(w10w10|w01w01)$', a)})
            res = copy.copy(self.R.ml(model, f"{model.label}_ML0",
                                      centers=start))
            self._cache[('ml', model)] = res
        if not np.isfinite(res.chi2):
            raise NoValidStart(f"no valid starting point for {model.label}")
        names, samp = self.R.emcee(model, f"{model.label}_MC",
                                   centers=res.best)
        res.sample_names, res.samples = names, samp
        if 'r_tensor' in names:
            res.r_samples = samp[:, names.index('r_tensor')]
        self._cache[key] = res
        return res

    def ml_fit(self, model: Model, centers=None) -> FitResult:
        key = ('ml', model)
        if key not in self._cache:
            self._cache[key] = self.R.ml(model, f"{model.label}_ML",
                                         centers=centers)
        return self._cache[key]

    def residual_test(self, base: FitResult, group) -> ResidualResult:
        """
        Test del gruppo di momenti `group` sui residui di `base` (o, con
        'restore-<gruppo>', degli spettri di quel gruppo tolti dal pruning).

        Liberi: gli spettri del gruppo (ampiezze e pendenze, piu' gli spettri
        incrociati con i gruppi gia' presenti) e i parametri spettrali del
        gruppo (MOMENT_GROUPS[...]['free']). Tutto il resto e' bloccato al
        best fit di `base`, r compreso (0 se base ha r fissato).

        Screening con emcee: rivelato se lo zero e' fuori dall'HPD di almeno
        una ampiezza. Se rivelato, PolyChord: evidenza e conferma della
        rivelazione con i suoi campioni.
        """
        m = base.model.apply(group).with_r(False)
        free = set(MOMENT_GROUPS[cand_group(group)]['free'])
        fixed = {k: v for k, v in base.best.items() if k not in free}
        fixed.setdefault('r_tensor', 0.0)
        old = set(base.model.amplitudes())
        amps = [a for a in m.amplitudes() if a not in old]
        tag = f"resid_{base.model.label}_plus_{group}"
        bounds = self.prior_bounds(amps)

        names, samp = self.R.emcee(m, tag + "_MC", fixed=fixed,
                                   centers=base.best)
        det, hpd = detection(names, samp, amps, self.hpd_frac, bounds)
        if not det:
            return ResidualResult(group, False, hpd, stage='emcee')
        # potatura subito: PolyChord solo con gli spettri rivelati da emcee
        weak = tuple(a for a in amps if hpd[a][0] <= 0.0 <= hpd[a][1])
        if weak:
            m = m.without(weak)
            amps = [a for a in amps if a not in weak]
            tag += ''.join(f"-{a[2:]}" for a in weak)
        logZ, logZerr, names, samp, m2lnl = self.R.polychord(
            m, tag + "_PC", fixed=fixed, centers=base.best)
        det, hpd_pc = detection(names, samp, amps, self.hpd_frac,
                                self.prior_bounds(amps))
        hpd.update(hpd_pc)
        i = int(np.argmin(m2lnl))
        best = dict(zip(names, map(float, samp[i])))
        return ResidualResult(group, det, hpd, logZ, logZerr, best,
                              stage='polychord', off=weak)

    def profile(self, model: Model, r_grid, centers=None):
        """Profile likelihood: chi2_min(r) minimizzando sugli altri param."""
        m = model.with_r(False)
        chi2 = []
        for i, r in enumerate(r_grid):
            res = self.R.ml(m, f"prof_{model.label}_{i:02d}",
                            fixed={'r_tensor': float(r)}, centers=centers)
            chi2.append(res.chi2)
        return np.array(chi2)


# ═══════════════════════════════════════════════════════════════════════════
# Log delle decisioni
# ═══════════════════════════════════════════════════════════════════════════

class DecisionLog:
    def __init__(self, runs_dir, verbose=True):
        self.events = []
        self.json_path = os.path.join(runs_dir, 'decision_log.json')
        self.txt_path = os.path.join(runs_dir, 'decision_log.txt')
        self.verbose = verbose
        self._lines = []

    def info(self, msg):
        if self.verbose:
            print(msg, flush=True)
        self._lines.append(msg)
        self._flush()

    def node(self, node, question, answer, **numbers):
        ev = dict(step=len(self.events) + 1, node=node, question=question,
                  answer=answer, **numbers)
        self.events.append(ev)
        nums = ', '.join(f"{k}={_fmt(v)}" for k, v in numbers.items()
                         if not isinstance(v, (dict, list)))
        self.info(f"[{ev['step']:02d}] {node}: {question} -> {answer}"
                  + (f"   ({nums})" if nums else ''))

    def _flush(self):
        try:
            with open(self.json_path, 'w') as f:
                json.dump(self.events, f, indent=2, default=_json_default)
            with open(self.txt_path, 'w') as f:
                f.write('\n'.join(self._lines) + '\n')
        except OSError:
            pass


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


# ═══════════════════════════════════════════════════════════════════════════
# Il decision tree
# ═══════════════════════════════════════════════════════════════════════════

class DecisionTree:

    def __init__(self, tc, fitter, log: DecisionLog):
        self.tc = tc
        self.F = fitter
        self.log = log
        th = tc['thresholds']
        self.pte_min = th['pte_min']
        self.lnB_pref = th['lnB_preferred']
        # pruning: il modello senza gli spettri non rivelati si tiene se
        # ln B(potato vs completo) > lnB_prune
        self.lnB_prune = th.get('lnB_prune', -2.5)
        self.hpd_frac = th.get('hpd_frac', 0.68)
        # confronti in bilico: |ln B - soglia| < seed_band * errore -> altri
        # run PolyChord (semi diversi), fino a seed_max_runs per fit
        self.seed_band = th.get('seed_band', 2.0)
        self.seed_max_runs = th.get('seed_max_runs', 1)
        self.alpha = th['lrt_alpha']
        mc = tc['moments']
        self.res_wr = mc.get('res_wr', ['d', 's'])
        self.res_r0 = mc.get('res_r0', ['d', 's', 'dT', 'd2', 's2', 'ds1',
                                        'ds2'])
        self.extra_requires = mc.get('requires', {})
        # restore-<gruppo> in Res_r0: spento di default. Gli spettri potati
        # sono gia' stati giudicati con r libero; con r = 0 sembrano rivelati
        # ma rifittati con r libero tornano non rivelati (d10s5 seed 6002)
        self.restore = mc.get('res_r0_restore', False)
        pr = tc['profile']
        self.r_grid = np.linspace(pr['r_min'], pr['r_max'], pr['n_r'])
        self.max_it = tc['run']['max_iterations']
        self.freq_replaces = tc['run'].get('frequentist_replaces_baseline',
                                           True)
        # schema "profilo" (profile_tree.py) al posto del ramo frequentista:
        # il tree si chiude con il test a 3 sigma, poi main() lancia lo schema
        self.profile_scheme = tc['run'].get('profile_scheme', False)

    # ── punto d'ingresso ──────────────────────────────────────────────────
    def run(self):
        M0 = Model(frozenset(), r_free=True)        # "run BBPower nominal"
        # prova "e se": si parte da un modello dato (etichetta, es.
        # M_d+s-d_w00w10-s_w00w10_r) invece che da quello senza momenti
        start = self.tc['run'].get('initial_model')
        if start:
            M0 = model_from_label(start)
        use_wr = True       # False dopo una sostituzione venuta da Res_r0
        visited = set()     # modelli gia' usati come M_0,r
        self.log.info(f"==> Initial model M_0,r = {M0.label}")
        for it in range(1, self.max_it + 1):
            self.log.info(f"\n==> Iteration {it}: M_0,r = {M0.label}")
            visited.add(M0)
            fit0 = self.F.full_fit(M0)
            self.log.node('fit', f'fit of {M0.label}', 'ok',
                          **fit0.summary())

            chi2_ok = fit0.pte > self.pte_min
            self.log.node('chi2', 'Is chi2 good?', 'Yes' if chi2_ok else 'No',
                          chi2=fit0.chi2, dof=fit0.dof, pte=fit0.pte,
                          pte_min=self.pte_min)
            if chi2_ok:
                ok1, rs = fit0.r_compatible_with_zero(1)
                self.log.node('r1sigma', 'Is r compatible with zero at 1 '
                              'sigma?', 'Yes' if ok1 else 'No',
                              r_med=rs['r_med'], r_lo1=rs['r_lo1'],
                              r_hi1=rs['r_hi1'])
                if ok1:
                    M0, fit0 = self.prune(M0, fit0)
                    return self.conclude_upper_limit(M0, fit0)

            outcome, M1, branch = self.search_moments(M0, fit0, use_wr)
            if outcome == 'replace':
                self.log.node('replace', f'M_0,r = M_1,r (from {branch})',
                              M1.label)
                # il nuovo M_0,r viene subito potato: i confronti successivi
                # sono con il modello senza ampiezze non rivelate
                M0, fit0 = self.prune(M1, self.F.full_fit(M1))
                use_wr = use_wr and branch == 'Res_wr'
                # ciclo: dopo il pruning il nuovo M_0,r e' un modello gia'
                # usato (es. restore-s adottato e poi ripotato) -> la
                # ricerca non porta altro, test a 3 sigma
                cycle = M0 in visited
                self.log.node('cycle', f'Is the new M_0,r ({M0.label}) a '
                              'model already used as M_0,r?',
                              'Yes' if cycle else 'No')
                if cycle:
                    return self.three_sigma(M0, fit0)
                continue
            # tutte le altre uscite portano al test a 3 sigma su M_0,r
            M0, fit0 = self.prune(M0, fit0)
            return self.three_sigma(M0, fit0)

        self.log.node('stop', 'max_iterations reached', 'stop',
                      max_iterations=self.max_it)
        M0, fit0 = self.prune(M0, self.F.full_fit(M0))
        return dict(conclusion='max_iterations', model=M0.label)

    # ── test dei residui ──────────────────────────────────────────────────
    def residual_scan(self, base: FitResult, groups, name, which):
        """
        Test dei residui per ogni gruppo; ritorna i rivelati ordinati per
        evidenza decrescente.
        """
        found = []
        for g in groups:
            rr = self.F.residual_test(base, g)
            hpd = {a: list(v) for a, v in rr.hpd.items()}
            self.log.node('resid_fit', f'{name}: moment {g} on the residuals '
                          f'of {base.model.label} ({which})',
                          'detected' if rr.detected else 'not detected',
                          stage=rr.stage, logZ=rr.logZ, logZerr=rr.logZerr,
                          pruned=list(rr.off), hpd68=hpd)
            if rr.detected:
                found.append(rr)
        found.sort(key=lambda rr: -rr.logZ)
        self.log.node('resid', f'{name}: is there any moment in the '
                      'residuals?', 'Yes' if found else 'No',
                      detected=[rr.group for rr in found],
                      logZ=[rr.logZ for rr in found])
        return found

    def lnB_decided(self, fit1: FitResult, fit0: FitResult, threshold):
        """
        ln B(fit1 vs fit0); se e' in bilico rispetto alla soglia ripete
        PolyChord per entrambi (alternando, prima quello con meno run)
        finche' esce dalla banda o si arriva a seed_max_runs.
        """
        lb, lberr = lnB(fit1, fit0)
        while abs(lb - threshold) < self.seed_band * lberr:
            nr = [max(1, len(f.logZ_runs)) for f in (fit1, fit0)]
            if min(nr) >= self.seed_max_runs:
                break
            f = fit1 if nr[0] <= nr[1] else fit0
            self.log.info(f"    ln B = {lb:.2f} +- {lberr:.2f} too close to "
                          f"{threshold:g}: PolyChord rerun "
                          f"{min(nr) + 1}/{self.seed_max_runs} of "
                          f"{f.model.label} with a new seed")
            self.F.repeat_pc(f)
            lb, lberr = lnB(fit1, fit0)
        return lb, lberr

    def compare(self, fit1: FitResult, fit0: FitResult, name):
        lb, lberr = self.lnB_decided(fit1, fit0, self.lnB_pref)
        pref = lb > self.lnB_pref
        self.log.node('pref', f'Is {name} ({fit1.model.label}) preferred '
                      f'over M_0,r ({fit0.model.label})?',
                      'Yes' if pref else 'No', lnB=lb, lnB_err=lberr,
                      pc_runs=[len(fit1.logZ_runs), len(fit0.logZ_runs)],
                      lnB_preferred=self.lnB_pref)
        return pref

    def compare_pruned(self, M1: Model, fit0: FitResult, name,
                       centers=None):
        """
        Potatura immediata. Screening (ML + emcee, senza evidenza): si
        tolgono gli spettri nuovi (non in M_0,r) con lo zero dentro l'HPD
        della loro ampiezza e si rifa' lo screening, finche' quelli rimasti
        sono tutti rivelati; se non ne resta nessuno il candidato e'
        scartato. Solo il modello potato si fitta con PolyChord e si
        confronta con M_0,r. Ritorna (preferito, fit o None).
        """
        old = set(fit0.model.amplitudes())
        while True:
            fit1 = self.F.screen_fit(M1, centers=centers)
            self.log.node('fit', f'screening fit (ML + emcee) of {name} = '
                          f'{M1.label}', 'ok', **fit1.summary())
            new = [a for a in M1.amplitudes() if a not in old]
            if not new:
                return False, None
            _, hpd = detection(fit1.sample_names, fit1.samples, new,
                               self.hpd_frac, self.F.prior_bounds(new))
            weak = [a for a in new if hpd[a][0] <= 0.0 <= hpd[a][1]]
            self.log.node('prune_check', f'Undetected new moment amplitudes '
                          f'in {M1.label}?',
                          'Yes' if weak else 'No', undetected=weak,
                          hpd68={a: list(v) for a, v in hpd.items()})
            if not weak:
                break
            if len(weak) == len(new):
                self.log.node('pruned_out', f'{name}: no new moment '
                              'detected, candidate rejected', 'rejected',
                              model=M1.label)
                return False, None
            M1, centers = M1.without(weak), fit1.best
        fit1 = self.F.full_fit(M1, centers=fit1.best)
        self.log.node('fit', f'fit of {name} = {M1.label}', 'ok',
                      **fit1.summary())
        return self.compare(fit1, fit0, name), fit1

    # ── ricerca di momenti ────────────────────────────────────────────────
    def search_moments(self, M0: Model, fit0: FitResult, use_wr=True):
        """
        Ritorna ('replace', M1, ramo) se un modello con momenti va adottato,
        altrimenti ('to_3sigma', None, ramo).
        """
        # ── Res_wr: tutto fissato al best fit di M_0,r, r compreso ────────
        groups = candidate_groups(M0, self.res_wr,
                                  extra_requires=self.extra_requires) \
            if use_wr else []
        if not use_wr:
            self.log.info("    (Res_wr skipped: the current model comes "
                          "from Res_r0)")
        found = self.residual_scan(fit0, groups, 'Res_wr',
                                   'all fixed at the best fit') if groups \
            else []
        if found:
            for rr in found:
                M1 = M0.apply(rr.group).without(rr.off)
                try:
                    pref, fit1 = self.compare_pruned(
                        M1, fit0, 'M_1,r', centers={**fit0.best, **rr.best})
                except NoValidStart as err:
                    self.log.node('unfittable', f'Res_wr: {rr.group} cannot '
                                  'be fitted with the moment constraints',
                                  'skipped', reason=str(err))
                    continue
                if pref:
                    return 'replace', fit1.model, 'Res_wr'
            # nessun candidato passa: test a 3 sigma solo se il modello non ha
            # ancora momenti (senza primo ordine non ci sono neanche gli
            # altri); altrimenti si cercano gli altri termini con r = 0
            if not M0.moments:
                return 'to_3sigma', None, 'Res_wr'
            self.log.info("    (Res_wr candidates not preferred and the model "
                          "already has moments: going to Res_r0)")

        # ── Res_r0: residui di M_0,0 (r = 0) ──────────────────────────────
        M00 = M0.with_r(False)
        fit00 = self.F.ml_fit(M00, centers=fit0.best)
        self.log.node('fit', f'best fit of {M00.label} (r = 0) for the '
                      'residuals', 'ok', chi2=fit00.chi2, pte=fit00.pte)
        groups = candidate_groups(M00, self.res_r0, restore=self.restore,
                                  extra_requires=self.extra_requires)
        found = self.residual_scan(fit00, groups, 'Res_r0', 'r fixed at 0') \
            if groups else []
        if not found:
            return 'to_3sigma', None, 'Res_r0'

        # candidati in ordine di evidenza; uno che non si riesce a fittare
        # (nessun punto di partenza permesso dai vincoli sui momenti) si
        # salta e si passa al successivo
        for rr in found:
            M1r = M0.apply(rr.group).without(rr.off)
            try:
                pref, fit1r = self.compare_pruned(
                    M1r, fit0, 'M_1,r', centers={**fit0.best, **rr.best})
            except NoValidStart as err:
                self.log.node('unfittable', f'Res_r0: {rr.group} cannot be '
                              'fitted with the moment constraints, next '
                              'candidate', 'skipped', reason=str(err))
                continue
            break
        else:
            return 'to_3sigma', None, 'Res_r0'
        if pref:
            return 'replace', fit1r.model, 'Res_r0'

        M10 = M1r.with_r(False)
        pref, fit10 = self.compare_pruned(
            M10, fit0, 'M_1,0', centers={**fit00.best, **rr.best})
        if not pref:
            return 'to_3sigma', None, 'Res_r0'
        if self.profile_scheme:
            # lo schema profilo riparte da M_1 (stesso modello, r libero)
            self.log.node('to_profile', 'M_1,0 preferred: the profile scheme '
                          'will start from M_1 (no frequentist branch here)',
                          fit10.model.with_r(True).label)
            return 'to_3sigma', None, 'Res_r0'
        # profilo: lo stesso modello (potato come M_1,0) con r libero
        M1r = fit10.model.with_r(True)
        fit1r = self.F.full_fit(M1r, centers=fit10.best)

        # ── ramo frequentista: profile likelihood ─────────────────────────
        out, M1 = self.profile_branch(M0, fit0, M1r, fit1r)
        return out, M1, 'Res_r0'

    def profile_branch(self, M0, fit0, M1r, fit1r):
        self.log.info("    (frequentist branch: profile likelihood in r; "
                      "see the note in the flowchart)")
        p0 = self.F.profile(M0, self.r_grid, centers=fit0.best)
        p1 = self.F.profile(M1r, self.r_grid, centers=fit1r.best)
        np.savez(os.path.join(self.tc['paths']['runs_dir'],
                              f'profile_{M0.label}_vs_{M1r.label}.npz'),
                 r=self.r_grid, chi2_M0=p0, chi2_M1=p1)
        dchi2 = float(np.min(p0) - np.min(p1))
        ddof = M1r.n_moment_params() - M0.n_moment_params()
        pval = float(stats.chi2.sf(max(dchi2, 0.), ddof))
        r0 = _profile_interval(self.r_grid, p0)
        r1 = _profile_interval(self.r_grid, p1)
        pref = pval < self.alpha
        self.log.node('pref_profile', f'Is M*_1,r ({M1r.label}) preferred '
                      f'over M*_0,r ({M0.label})?', 'Yes' if pref else 'No',
                      dchi2=dchi2, ddof=ddof, p_value=pval,
                      lrt_alpha=self.alpha,
                      r_best_M0=r0[0], r_lo_M0=r0[1], r_hi_M0=r0[2],
                      r_best_M1=r1[0], r_lo_M1=r1[1], r_hi_M1=r1[2])
        if pref:
            self.log.node('mstar', 'M* = M*_1,r', M1r.label)
            if self.freq_replaces:
                return 'replace', M1r
            self.log.info("    frequentist_replaces_baseline=false: "
                          "M_0,r stays the baseline")
            return 'to_3sigma', None
        self.log.node('mstar', 'M* = M*_0,r', M0.label)
        return 'to_3sigma', None

    # ── pruning ───────────────────────────────────────────────────────────
    def prune(self, M0, fit0):
        """
        Toglie dal modello finale gli spettri dei momenti non rivelati (zero
        dentro l'HPD della loro ampiezza nel posterior del fit completo),
        rifitta e tiene il modello potato se non e' sfavorito
        (ln B > lnB_prune). Ripete finche' restano solo ampiezze rivelate.
        """
        while M0.amplitudes():
            amps = M0.amplitudes()
            det, hpd = detection(fit0.sample_names, fit0.samples, amps,
                                 self.hpd_frac, self.F.prior_bounds(amps))
            weak = [a for a in amps if hpd[a][0] <= 0.0 <= hpd[a][1]]
            self.log.node('prune_check', f'Are all moment amplitudes of '
                          f'{M0.label} detected?', 'No' if weak else 'Yes',
                          undetected=weak,
                          hpd68={a: list(v) for a, v in hpd.items()})
            if not weak:
                break
            M1 = M0.without(weak)
            fit1 = self.F.full_fit(M1, centers=fit0.best)
            self.log.node('fit', f'fit of the pruned model {M1.label}', 'ok',
                          **fit1.summary())
            lb, lberr = self.lnB_decided(fit1, fit0, self.lnB_prune)
            keep = lb > self.lnB_prune
            self.log.node('prune', f'Is the pruned model {M1.label} not '
                          f'disfavoured w.r.t. {M0.label}?',
                          'Yes' if keep else 'No', lnB=lb, lnB_err=lberr,
                          pc_runs=[len(fit1.logZ_runs),
                                   len(fit0.logZ_runs)],
                          lnB_prune=self.lnB_prune)
            if not keep:
                break
            M0, fit0 = M1, fit1
        return M0, fit0

    # ── uscite ────────────────────────────────────────────────────────────
    def three_sigma(self, M0, fit0):
        ok3, rs = fit0.r_compatible_with_zero(3)
        self.log.node('r3sigma', 'Is r for M_0,r compatible with zero at '
                      '3 sigma?', 'Yes' if ok3 else 'No',
                      r_med=rs['r_med'], r_lo3=rs['r_lo3'],
                      r_hi3=rs['r_hi3'])
        if ok3:
            return self.conclude_upper_limit(M0, fit0)
        return self.conclude_detection(M0, fit0)

    def _fit_r0(self, M0, fit0):
        M00 = M0.with_r(False)
        fit00 = self.F.full_fit(M00, centers=fit0.best)
        lb, lberr = lnB(fit0, fit00)
        return fit00, lb, lberr

    def conclude_upper_limit(self, M0, fit0):
        fit00, lb, lberr = self._fit_r0(M0, fit0)          # M_0,0
        rs = fit0.r_summary()
        res = dict(conclusion='upper_limit', model=M0.label,
                   r_95=rs['r_95'], r_med=rs['r_med'],
                   lnB_rfree_vs_r0=lb, lnB_err=lberr,
                   chi2_M00=fit00.chi2, pte_M00=fit00.pte)
        self.log.node('END', 'put an upper limit on r', 'upper limit', **res)
        return res

    def conclude_detection(self, M0, fit0):
        fit00, lb, lberr = self._fit_r0(M0, fit0)
        rs = fit0.r_summary()
        res = dict(conclusion='r_detection', model=M0.label,
                   r_med=rs['r_med'], r_lo1=rs['r_lo1'], r_hi1=rs['r_hi1'],
                   r_snr=rs['r_mean'] / rs['r_std'],
                   lnB_rfree_vs_r0=lb, lnB_err=lberr)
        self.log.node('END', 'You cannot get rid of your r detection!!! Be '
                      'careful and wait before organizing a press '
                      'conference!!!', 'r detection', **res)
        return res


def _profile_interval(r, chi2):
    """Best fit e intervallo Delta chi2 = 1 (interpolato) dal profilo."""
    i = int(np.argmin(chi2))
    d = chi2 - chi2[i]
    fine = np.linspace(r[0], r[-1], 2001)
    df = np.interp(fine, r, d)
    inside = fine[df <= 1.0]
    return float(r[i]), float(inside.min()), float(inside.max())


# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    tc = yaml.safe_load(open(args.config))
    os.makedirs(tc['paths']['runs_dir'], exist_ok=True)
    log = DecisionLog(tc['paths']['runs_dir'])
    fitter = Fitter(BBCompSepRunner(tc, log),
                    hpd_frac=tc['thresholds'].get('hpd_frac', 0.68))
    result = DecisionTree(tc, fitter, log).run()
    log.info("\n==> CONCLUSION")
    for k, v in result.items():
        log.info(f"    {k:<18} {_fmt(v)}")
    runs_dir = tc['paths']['runs_dir']
    with open(os.path.join(runs_dir, 'conclusion.json'), 'w') as f:
        json.dump(result, f, indent=2, default=_json_default)
    if not tc['run'].get('profile_scheme', False):
        return

    # ── schema "profilo" sul tree concluso (profile_tree.py) ──────────────
    with open(os.path.join(runs_dir, 'conclusion_tree.json'), 'w') as f:
        json.dump(result, f, indent=2, default=_json_default)
    ps = tc.get('profile_scheme', {})
    pdir = os.path.join(runs_dir, 'profile')
    cmd = [sys.executable,
           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'profile_tree.py'),
           '--config', os.path.abspath(args.config), '--tree-dir', runs_dir,
           '--out', pdir]
    for k in ('lnb_res', 'lnb_full', 'n', 'max_it', 'nproc'):
        if k in ps:
            cmd += ['--' + k.replace('_', '-'), str(ps[k])]
    log.info("\n==> Profile scheme (Res_wr*): " + ' '.join(cmd[1:]))
    subprocess.run(cmd, check=True)
    prof = json.load(open(os.path.join(pdir, 'result.json')))
    ul = prof['conclusion'] == 'upper_limit'
    rkeys = ('r_med', 'r_lo1', 'r_hi1', 'r_95', 'r_lo3', 'r_hi3', 'r_std')
    final = dict(
        conclusion=prof['conclusion'], model=prof['model_final'],
        r_med=prof['r_med_pl'], r_lo1=prof['r_lo1_pl'],
        r_hi1=prof['r_hi1_pl'], r_std=prof['r_std_pl'],
        r_95=prof['r_95_pl0'] if ul else None, source='profile scheme',
        # parte bayesiana: modello scelto dal tree, best fit ML, posterior
        # (PolyChord) di tutti i parametri, conclusione del tree
        bayesian=dict(prof['bayes'], conclusion=result.get('conclusion')),
        # parte profilo: modello finale, posterior di r dal profilo (prior
        # del tree; r_95 con r >= 0), best fit a r = r*, posterior degli
        # altri parametri condizionati a r = r* (emcee)
        profiled=dict(model=prof['model_final'], r_star=prof['r_star_final'],
                      r_posterior={k: prof[k + '_pl'] for k in rkeys},
                      r_95_rpos=prof['r_95_pl0'],
                      r_profile=dict(best=prof['r_best'], lo1=prof['r_lo1'],
                                     hi1=prof['r_hi1'],
                                     ul95=prof['r_ul95_profile'],
                                     dchi2_r0=prof['dchi2_r0']),
                      best_fit_at_rstar=prof['best_fit_at_rstar'],
                      chi2_at_rstar=prof['chi2_at_rstar'],
                      posterior_at_rstar=prof['params_at_rstar'],
                      conclusion=prof['conclusion'], start=prof['start'],
                      rejected=prof['rejected']),
        tree=result)
    b = final['bayesian']
    log.info("\n==> BAYESIAN PART (tree): " + b['model'] +
             f", chi2 = {b['chi2_ml']:.1f}, conclusion {b['conclusion']}")
    for n, v in b['posterior'].items():
        log.info(f"    {n:<18} best fit {_fmt(b['best_fit'].get(n))}   "
                 f"posterior {_fmt(v['med'])} [{_fmt(v['lo1'])}, "
                 f"{_fmt(v['hi1'])}]")
    p_ = final['profiled']
    log.info("\n==> PROFILED PART: " + p_['model'] +
             f", r* = {_fmt(p_['r_star'])}, chi2 = {p_['chi2_at_rstar']:.1f}")
    rp = p_['r_posterior']
    log.info(f"    {'r_tensor':<18} posterior from the profile "
             f"{_fmt(rp['r_med'])} [{_fmt(rp['r_lo1'])}, {_fmt(rp['r_hi1'])}]"
             f", r_95 (r >= 0) {_fmt(p_['r_95_rpos'])}")
    for n, v in p_['posterior_at_rstar'].items():
        log.info(f"    {n:<18} best fit {_fmt(p_['best_fit_at_rstar'].get(n))}"
                 f"   posterior | r* {_fmt(v['med'])} [{_fmt(v['lo1'])}, "
                 f"{_fmt(v['hi1'])}]")
    log.info("\n==> FINAL CONCLUSION (profile scheme)")
    for k in ('conclusion', 'model', 'r_med', 'r_lo1', 'r_hi1', 'r_95'):
        log.info(f"    {k:<18} {_fmt(final[k])}")
    with open(os.path.join(runs_dir, 'conclusion.json'), 'w') as f:
        json.dump(final, f, indent=2, default=_json_default)


if __name__ == '__main__':
    main()
