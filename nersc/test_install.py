"""
Check of the installation on Perlmutter.

  python nersc/test_install.py          imports of all the packages + PySM
  srun -n 4 python nersc/test_install.py --mpi
                                        PolyChord with MPI on a 2D Gaussian
"""
import sys

if '--mpi' not in sys.argv:
    import numpy, scipy, healpy, pymaster, sacc, emcee, pyshtools  # noqa
    import fitsio, yaml, corner, numdifftools, bbpipe              # noqa
    import pypolychord                                             # noqa
    import bbpower.compsep, bbpower.power_specter                  # noqa
    import bbpower.power_summarizer                                # noqa
    import bbsure.tree, bbsure.profile_scheme, bbsure.sims.maps    # noqa
    import pysm3
    import pysm3.units as u
    sky = pysm3.Sky(nside=16, preset_strings=['d9', 's4'])
    m = sky.get_emission(90 * u.GHz)
    print(f"imports ok; PySM d9s4 at 90 GHz: map shape {m.shape}")
    sys.exit(0)

import numpy as np
from mpi4py import MPI
import pypolychord
from pypolychord.settings import PolyChordSettings
from pypolychord.priors import UniformPrior

comm = MPI.COMM_WORLD


def like(theta):
    return float(-0.5 * np.sum(theta ** 2) - np.log(2 * np.pi)), [0.]


def prior(h):
    return UniformPrior(-10., 10.)(h)


s = PolyChordSettings(2, 1)
s.nlive, s.feedback, s.base_dir = 100, 0, 'pc_test'
out = pypolychord.run_polychord(like, 2, 1, s, prior)
if comm.rank == 0:
    # true Z = 1 / 400 (uniform prior over 20 x 20) -> ln Z = -5.99
    print(f"PolyChord with {comm.size} MPI processes: ln Z = "
          f"{out.logZ:.2f} +- {out.logZerr:.2f} (expected -5.99)")
