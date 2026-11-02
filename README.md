# BBSure

Decision tree for the foreground modelling of BB cross-spectra analyses
with [BBPower](https://github.com/simonsobs/BBPower), with a
profile-likelihood scheme on the tensor-to-scalar ratio r.

Starting from the nominal model without moments, the tree decides, node by
node, whether the data need moments of the foreground SEDs (moment
expansion in the formalism of
[Vacher et al. 2022](https://arxiv.org/abs/2111.07742), with a power law in
ell for each moment spectrum), until it reaches an upper limit on r or an r
detection. The moments are searched with r = 0, where they have the most
room; the profile scheme then compares the candidates found there with every
model at its own value of r (the minimum of its profile likelihood), to
reduce the degeneracy between moments and r, and gives the final posterior
of r from the profile likelihood.

The flowchart is `docs/flowcharts/decision_tree` (tree and profile scheme);
`profile_scheme_nersc` is the profile scheme of the 200-set validation run.

## Installation

BBSure needs the BBPower fork with the moment expansion:

```bash
git clone -b Bsure https://github.com/NicolettaK/BBPower.git
export PYTHONPATH=$PWD/BBPower:$PYTHONPATH
git clone https://github.com/NicolettaK/BBSure.git
pip install -e BBSure
```

BBPower's own dependencies (bbpipe, fgbuster, sacc, NaMaster, healpy,
emcee) are needed, plus PolyChord (`pypolychord`, compiled with MPI for
parallel runs) and, for the simulations, PySM 3. `nersc/environment_nersc.yml`
lists tested versions; `nersc/install_nersc.sh` installs everything on
Perlmutter.

Data files (SO SAT bandpasses and beams, CAMB spectra, masks, HL fiducial)
are in `data/`; another directory can be used with the environment
variable `BBSURE_DATA`. Configs refer to it as `@DATA@`.

## Pipeline

| step | script |
|---|---|
| 1. analysis mask, weighted with the SO hits (galactic coordinates) | `scripts/make_mask.py` |
| 2. Gaussian foreground model fitted on PySM d1s1 | `scripts/fit_fg_spectra.py` |
| 3. Gaussian sims for the covariance and their spectra | `scripts/generate_sims_cov.sh`, `scripts/run_spectra.sh --cov-maps` |
| 4. fiducial spectra for the Hamimeche-Lewis likelihood | `scripts/make_fiducial.py` |
| 5. simulated maps of a data set (CMB + PySM or Gaussian foregrounds + noise) | `scripts/generate_maps.py` |
| 6. data vector (BBPowerSpecter + BBPowerSummarizer) | `scripts/run_spectra.sh --cov-cells` |
| 7. decision tree + profile scheme | `scripts/run_tree.py --config <tree config>` |
| 8. tables of a validation run | `scripts/collect.py` |
| 9. figures | `plots/` |

Steps 1-4 are done once; the validation repeats 5-7 for each data set
(`scripts/make_plan.py`, `nersc/`). `scripts/make_bf_cov.py` gives the
optional second pass with the covariance computed at the best fit of the
tree.

The tree config (`configs/tree_config_template.yml`) sets the thresholds
(PTE, ln B, HPD fraction), the moment groups tested, PolyChord and emcee
settings, and `run: profile_scheme: true` to run the profile scheme after
the tree (with `false`, the tree alone). The BBPower config
(`configs/base_config_tree.yml`) holds the model and the priors of all the parameters, moments included.

Output of a tree (in `runs_dir`): `decision_log.{txt,json}` (every node
with question, answer and numbers), `conclusion.json` (final result:
Bayesian part and profile part), `conclusion_tree.json` (conclusion of the
tree alone), one directory per BBCompSep run (runs with the same config are
reused), `profile/` (profile scheme, `result.json`).

## Code

| module | content |
|---|---|
| `bbsure/models.py` | moment groups (`d`, `s`, `dT`, `d2`, `s2`, `ds1`, `ds2`), `Model` and its labels |
| `bbsure/config.py` | BBCompSep config of a model |
| `bbsure/runner.py` | runs BBCompSep (ML, PolyChord with MPI, emcee) |
| `bbsure/fitter.py` | full fits, screening, residual tests, profiles |
| `bbsure/stats.py` | fit results, PTE, Bayes factors, HPD intervals |
| `bbsure/tree.py` | the decision tree |
| `bbsure/profile_scheme.py` | the profile scheme |
| `bbsure/sims/` | simulated maps: sky model, PySM, SO V3 noise |
| `bbsure/covariance.py` | best-fit covariance |
| `bbsure/collect.py` | tables of a validation run |

## Tests

```bash
python tests/test_decision_tree.py     # tree logic with a mock fitter
python tests/test_profile_scheme.py    # profile scheme with a mock runner
```

They do not run BBCompSep (seconds); `test_decision_tree.py` also checks
the generated configs against BBPower's `ParameterManager` if BBPower is
importable.

## License

BSD 3-Clause, as BBPower (see `LICENSE`).
