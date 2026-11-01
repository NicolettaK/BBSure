# Running BBSure on Perlmutter (NERSC)

## Installation (once)

From a login node, inside a clone of BBSure:

```bash
bash nersc/install_nersc.sh /global/common/software/<project>/$USER/bbsure
```

It creates a conda environment (`environment_nersc.yml`), compiles mpi4py and
PolyChordLite against Cray MPICH, clones BBPower (fork with the moment
expansion, branch `Bsure`), installs BBSure in editable mode and writes
`activate_bbsure.sh`. Check it on a compute node:

```bash
salloc -N 1 -C cpu -q interactive -t 00:15:00 -A <project>
source /global/common/software/<project>/$USER/bbsure/activate_bbsure.sh
python nersc/test_install.py
srun -n 4 python nersc/test_install.py --mpi
```

The PySM maps of the 9 skies at nside 128 (`<d><s>_nside128_from512.npy`)
must be in `$BBSURE_DATA/pysm_cache/`: they are produced by
`scripts/generate_maps.py --pysm <sky> --pysm-nside-in 512` (PySM at nside
512, then harmonic downgrade), e.g. on a laptop, and copied there.

## Validation run

Layout of a run directory `<root>`:

| | |
|---|---|
| `cov/` | spectra of the covariance sims (`cells_all_splits_sim*.fits`, `cells_all_sims.txt`) |
| `plan/sets.csv` | index, seed, random sky |
| `truth/r_true.csv` | true r (do not open until the end) |
| `runs/set_<i>_<d9s4\|rand>/` | spectra (`output/`), tree and profile scheme (`tree/`, `tree/profile/`) |
| `logs/` | one log per set and per job |

1. Once, on a login node (the covariance tarball holds the spectra of the
   Gaussian covariance sims, see `scripts/generate_sims_cov.sh` and
   `scripts/run_spectra.sh --cov-maps`):

   ```bash
   source <install>/activate_bbsure.sh
   bash nersc/setup.sh <root> <covariance tarball>
   ```

2. Launch (one node runs 16 trees: 8 sets x 2 skies). The job inherits the
   environment of the shell that submits it:

   ```bash
   cd <root>
   sbatch -A <project> <repo>/nersc/job_validation.slurm 0 7
   ```

   Concluded sets are skipped and interrupted ones restart from the fits
   already done: if the job runs out of time, resubmit the same command.

3. During and after:

   ```bash
   squeue -u $USER
   tail logs/set_003_rand.log
   python <repo>/scripts/collect.py tree <root>             # blind table
   python <repo>/scripts/collect.py tree <root> --unblind   # ONLY at the end
   python <repo>/scripts/collect.py profile <root> --out profile_results.csv
   ```

Second pass with the best-fit covariance (`bbsure/covariance.py`):
`sbatch -A <project> <repo>/nersc/job_bfcov.slurm <first> <last> 1`.
