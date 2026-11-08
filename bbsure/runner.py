"""
Running BBCompSep: maximum likelihood, evidence and posterior (PolyChord,
optionally with MPI, or nautilus), emcee and single_point, with a generated
config. Each run has its own
directory (config.yml, config.md5, stdout.log, output); runs with the same
config are reused.
"""
import copy
import hashlib
import os
import re
import subprocess
import sys
import threading
import time

import numpy as np
import yaml

from .config import (build_config, free_param_names, gaussian_prior_chi2,
                     iter_param_entries)
from .paths import load_yaml
from .stats import FitResult


class NoValidStart(RuntimeError):
    """No point with finite likelihood (constraints on the moments)."""


class NoLivePoints(NoValidStart):
    """PolyChord could not generate its live points (nautilus: allowed
    prior fraction below 1e-3): the region of the
    prior allowed by the moment constraints is too small."""

# ── BBCompSep in the same process (fast fits) ────────────────────────────
# Launching `python -m bbpower` costs the Python start-up, the imports and
# the reading of the spectra at every fit (~4 s on a laptop, ~50 s at NERSC),
# against ~1 s for an ML fit. ML, single_point and predicted_spectra run in
# here, with the sacc files read only once; PolyChord (MPI) and emcee (Pool)
# stay separate processes. A lock serializes them (stdout is global; the fit
# is bound to the GIL, threads would not run in parallel anyway).
INPROC_SAMPLERS = ('maximum_likelihood', 'single_point', 'predicted_spectra')
_INPROC_LOCK = threading.Lock()
_SACC_CACHE = {}


def _run_inproc(args, log_path):
    """BBCompSep with the command-line arguments `args`; output in log_path.
    Returns True on success."""
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


# samplers whose result does not depend on the centre of the tophats (only
# a starting point: PolyChord samples the prior between the edges, emcee
# starts its walkers there): a run with the same config up to the centres
# is reused
CENTRE_FREE_SAMPLERS = ('polychord', 'nautilus', 'emcee')


def without_centres(cfg):
    """The config with the centres of the tophat priors removed."""
    cfg = copy.deepcopy(cfg)
    for p in iter_param_entries(cfg['BBCompSep']).values():
        if len(p) > 2 and str(p[1]).lower() == 'tophat':
            p[2][1] = None
    return cfg


def equal_weights_file(run_dir):
    """Equal-weight posterior of an evidence run (the `<model>_PC` runs,
    PolyChord or nautilus): columns 1, -2 ln L, parameters."""
    f = os.path.join(run_dir, 'nautilus', 'equal_weights.txt')
    if os.path.isfile(f):
        return f
    return os.path.join(run_dir, 'polychord', 'pch_equal_weights.txt')


def parse_polychord_stats(fname):
    txt = open(fname).read()
    m = re.search(r'log\(Z\)\s*=\s*([-+0-9.eE]+)\s*\+/-\s*([-+0-9.eE]+)', txt)
    if not m:
        raise RuntimeError(f"log(Z) not found in {fname}")
    return float(m.group(1)), float(m.group(2))


class BBCompSepRunner:
    """Runs `python -m bbpower BBCompSep` with a generated config."""

    def __init__(self, tc, logger):
        self.tc = tc
        self.log = logger
        p = tc['paths']
        self.base = load_yaml(p['base_config'])
        self.runs_dir = p['runs_dir']
        self.reuse = tc['run'].get('reuse_runs', True)
        # fast fits in the same process (see _run_inproc)
        self.inproc = tc['run'].get('inproc', True)
        self.pc_cfg = tc['polychord']
        self.mc_cfg = tc.get('emcee', {})
        os.makedirs(self.runs_dir, exist_ok=True)

    def pc_settings(self, ndim):
        """
        PolyChord nlive and nrepeat proportional to the number of free
        parameters (rules of thumb: nlive ~ 25 ndim, num_repeats ~ 2-5 ndim).
        Without the *_per_dim keys, the fixed values nlive / nrepeat.
        """
        c = self.pc_cfg
        if 'nlive_per_dim' not in c:
            return dict(nlive=c['nlive'], nrepeat=c['nrepeat'])
        return dict(nlive=max(c['nlive_min'], c['nlive_per_dim'] * ndim),
                    nrepeat=max(c['nrepeat_min'],
                                c['nrepeat_per_dim'] * ndim))

    @property
    def evidence_sampler(self):
        """Sampler of the evidence runs: polychord (default) or nautilus."""
        return self.pc_cfg.get('sampler', 'polychord')

    def nautilus_settings(self):
        """nautilus block of the BBCompSep config (all but procs and
        prior_cache, which do not change the result)."""
        return {k: v for k, v in (self.pc_cfg.get('nautilus') or {}).items()
                if k not in ('procs', 'prior_cache')}

    def nautilus_procs(self):
        """Pool processes of a nautilus run: BBSURE_NAUTILUS_PROCS (set by
        the NERSC job), else polychord: nautilus: procs, else mpi_procs."""
        env = os.environ.get('BBSURE_NAUTILUS_PROCS')
        if env:
            return int(env)
        return int((self.pc_cfg.get('nautilus') or {}).get(
            'procs', self.pc_cfg.get('mpi_procs', 1)))

    # ── low level ─────────────────────────────────────────────────────────
    def _run(self, tag, cfg, result_file):
        d = os.path.join(self.runs_dir, tag)
        os.makedirs(os.path.join(d, 'polychord'), exist_ok=True)
        txt = yaml.safe_dump(cfg, sort_keys=False)
        h = hashlib.md5(txt.encode()).hexdigest()
        cfg_path = os.path.join(d, 'config.yml')
        hash_path = os.path.join(d, 'config.md5')
        res_path = os.path.join(d, result_file)

        if (self.reuse and os.path.isfile(res_path)
                and os.path.isfile(hash_path)):
            if open(hash_path).read().strip() == h:
                self.log.info(f"    [reuse] {tag}")
                return d
            # PolyChord / emcee: the same run with another starting point
            # (e.g. a best fit that differs in the last digits) is reused
            sampler = cfg['BBCompSep']['sampler']
            if (sampler in CENTRE_FREE_SAMPLERS and os.path.isfile(cfg_path)
                    and without_centres(yaml.safe_load(open(cfg_path)))
                    == without_centres(cfg)):
                self.log.info(f"    [reuse] {tag} (same config up to the "
                              "starting point)")
                return d

        # emcee checkpoint of an interrupted run (e.g. job killed by the time
        # limit): BBPower would resume from a truncated file and fail
        stale = os.path.join(d, 'emcee.npz.h5')
        if os.path.isfile(stale):
            os.remove(stale)
        # the hash marks a completed run: removed now, written again only at
        # the end, so that an interrupted run is never reused
        if os.path.isfile(hash_path):
            os.remove(hash_path)

        with open(cfg_path, 'w') as f:
            f.write(txt)
        p = self.tc['paths']
        cmd = [sys.executable, '-m', 'bbpower', 'BBCompSep',
               f"--cells_coadded={p['cells_coadded']}",
               f"--cells_noise={p['cells_noise']}",
               f"--cells_fiducial={p['cells_fiducial']}",
               # covariance: by default the one in the data sacc (sims);
               # with paths: cells_coadded_cov the best-fit one
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
        # PolyChord on several MPI processes (1 master + nproc-1 for the
        # likelihood); ML and emcee stay serial (emcee uses a Pool)
        nproc = int(self.pc_cfg.get('mpi_procs', 1))
        nlaunch = 0
        if sampler == 'polychord' and nproc > 1:
            # MPI command: by default the mpirun of the environment; on a
            # cluster e.g. polychord: mpi_launcher: "srun -n {n} --cpu-bind=cores"
            launcher = self.pc_cfg.get('mpi_launcher')
            if launcher:
                pre = launcher.format(n=nproc).split()
            else:
                pre = [os.path.join(os.path.dirname(sys.executable),
                                    'mpirun'), '-np', str(nproc)]
            nlaunch = len(pre)
            cmd = pre + cmd
        env = None
        if sampler == 'nautilus':
            # nautilus: one process with a pool (BBPOWER_NPROC)
            env = dict(os.environ, BBPOWER_NPROC=str(self.nautilus_procs()))
            # allowed prior fraction of the moment constraints, shared by
            # the same model in every set (see BBPower _prior_fraction_cache)
            cache = (self.pc_cfg.get('nautilus') or {}).get('prior_cache')
            if cache:
                env['BBPOWER_PRIOR_CACHE'] = os.path.abspath(cache)
        if sampler == 'polychord' and nproc > 1:
            procs = f", {nproc} MPI procs"
        elif env:
            procs = f", {self.nautilus_procs()} procs"
        else:
            procs = ''
        self.log.info(f"    [run] {tag} ({sampler}{procs})")
        t0 = time.time()
        log_path = os.path.join(d, 'stdout.log')
        stuck = os.path.join(d, 'no_live_points')
        if sampler in ('polychord', 'nautilus') and os.path.isfile(stuck) \
                and open(stuck).read() == h:
            raise NoLivePoints(f"{tag}: no live points (already found)")
        with open(log_path, 'w') as out:
            if sampler == 'polychord':
                ret = self._polychord_proc(cmd, out, log_path)
            else:
                ret = subprocess.run(cmd, stdout=out,
                                     stderr=subprocess.STDOUT, env=env)
        if (sampler == 'nautilus' and ret.returncode != 0 and
                'allowed prior fraction too small' in open(log_path).read()):
            # the analogue of PolyChord without live points
            with open(stuck, 'w') as f:
                f.write(h)
            raise NoLivePoints(f"{tag}: allowed prior fraction below 1e-3 "
                               "(moment constraints)")
        if ret is None:
            # the live points were never all generated: the prior volume
            # allowed by the moment constraints is too small
            with open(stuck, 'w') as f:
                f.write(h)
            raise NoLivePoints(f"{tag}: live points not generated in "
                               f"{self.pc_cfg.get('start_timeout', 1800)} s "
                               "(moment constraints)")
        if (ret.returncode != 0 and sampler == 'polychord' and nproc > 1
                and 'Finished sampling' not in open(log_path).read()):
            # MPI does not start (PMIx, no network) or crashes half way
            # (network change, laptop sleep): PolyChord serially
            os.replace(log_path, log_path + '.mpi_failed')
            self.log.info(f"    [warn] {tag}: MPI run failed (network "
                          "change?), running PolyChord serially")
            with open(log_path, 'w') as out:
                ret = subprocess.run(cmd[nlaunch:], stdout=out,
                                     stderr=subprocess.STDOUT)
        if (ret.returncode != 0 and sampler == 'polychord'
                and os.path.isfile(res_path)
                and 'Finished sampling' in open(log_path).read()):
            # PolyChord finished but MPI exited badly on shutdown (e.g.
            # network changed after a laptop sleep): the results are complete
            self.log.info(f"    [warn] {tag}: MPI exited with code "
                          f"{ret.returncode} after sampling finished; "
                          "results kept")
        elif ret.returncode != 0 or not os.path.isfile(res_path):
            raise RuntimeError(f"BBCompSep failed for {tag}: see {log_path}")
        with open(hash_path, 'w') as f:
            f.write(h)
        self.log.info(f"          done in {time.time() - t0:.0f} s")
        return d

    def _polychord_proc(self, cmd, out, log_path):
        """
        PolyChord as a subprocess (returns the finished Popen, with its
        returncode); killed (returns None) if the live points
        are not all generated within polychord: start_timeout seconds
        (default 1800): with the moment constraints almost all the prior
        samples can be rejected, and the generation never ends.
        """
        timeout = float(self.pc_cfg.get('start_timeout', 1800))
        p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        t0 = time.time()
        while p.poll() is None:
            time.sleep(10)
            if time.time() - t0 < timeout:
                continue
            out.flush()
            if 'all live points generated' in open(log_path).read():
                p.wait()
                return p
            p.terminate()
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            return None
        return p

    def ml(self, model, tag, fixed=None, centers=None):
        """Maximum likelihood: best fit and chi2 (without Gaussian priors)."""
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

    def polychord(self, model, tag, fixed=None, centers=None,
                  free_temp=False):
        """
        Evidence and posterior (equal weights) with PolyChord, or nautilus
        with polychord: sampler: nautilus (random seed: a different tag,
        e.g. _rep2, gives an independent run). With moment constraints the
        evidence is on the prior restricted to the allowed region (both).
        free_temp: T_d free also without the dT moments.
        Returns logZ, logZerr, names, samples [nsamp, npar], -2lnL.
        """
        sampler = self.evidence_sampler
        cfg = build_config(self.base, model, sampler, fixed, centers,
                           free_temp=free_temp)
        names = free_param_names(cfg['BBCompSep'])
        if sampler == 'nautilus':
            cfg['BBCompSep']['nautilus'] = self.nautilus_settings()
            d = self._run(tag, cfg, os.path.join('nautilus', 'results.npz'))
            z = np.load(os.path.join(d, 'nautilus', 'results.npz'))
            logZ, logZerr = float(z['log_z']), float(z['log_z_err'])
        else:
            cfg['BBCompSep'].update(self.pc_settings(len(names)))
            d = self._run(tag, cfg, os.path.join('polychord', 'pch.stats'))
            logZ, logZerr = parse_polychord_stats(
                os.path.join(d, 'polychord', 'pch.stats'))
        samp = np.loadtxt(equal_weights_file(d), ndmin=2)
        # columns: weight, -2lnL, parameters
        return logZ, logZerr, names, samp[:, 2:], samp[:, 1]

    def emcee(self, model, tag, fixed=None, centers=None):
        """
        Posterior with emcee (detection screening). Returns names and samples
        after the burn-in (fraction burn_frac of the chain discarded).
        """
        c = self.mc_cfg
        cfg = build_config(self.base, model, 'emcee', fixed, centers)
        names = free_param_names(cfg['BBCompSep'])
        # emcee (red-blue move) needs at least 2 walkers per parameter
        cfg['BBCompSep'].update(
            nwalkers=max(int(c.get('nwalkers', 32)), 2 * len(names) + 2),
            n_iters=int(c.get('n_iters', 2000)))
        d = self._run(tag, cfg, 'emcee.npz')
        chain = np.load(os.path.join(d, 'emcee.npz'))['chain']
        burn = int(chain.shape[1] * c.get('burn_frac', 0.5))
        return names, chain[:, burn:, :].reshape(-1, chain.shape[-1])
