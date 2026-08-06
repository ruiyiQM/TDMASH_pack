"""TDMASH: energy-scaled real-time TDDFT for ab initio surface hopping dynamics.

The package propagates a ground-state and a ΔSCF excited-state Kohn–Sham
determinant simultaneously along a nuclear trajectory using an energy-scaled
time-dependent Kohn–Sham equation, extracts nonadiabatic couplings from the
propagated states in a Dynamical Transition Orbital gauge, and drives the
nuclei with the Mapping Approach to Surface Hopping.

Typical use::

    from tdmash import TDRKSSimulation, MASH

    sim1 = TDRKSSimulation(mf_ground, dt=0.4, propagation_method='accelerated')
    sim2 = TDRKSSimulation(mf_excited, dt=0.4, propagation_method='accelerated')
    dynamics = MASH(sim1, sim2, ...)
    dynamics.run()
"""

__version__ = "0.1.0"

from .TDRHF import TDRHFSimulation, lowdin
from .TDRKS import TDRKSSimulation
from .TDUHF import TDUHFSimulation
from .TDUKS import TDUKSSimulation
from .TDOverlap import TDOverlap
from .Spin import Spin
from .FD import Finite_Difference_NAC
from .MASH import MASH
from .Ehrenfest import FixedPathDynamics
from .MOM import mom_occ_, singlet_occ_
from .Anderson import AndersonAcceleration
from .Wigner import WignerSampler

__all__ = [
    "TDRHFSimulation",
    "TDRKSSimulation",
    "TDUHFSimulation",
    "TDUKSSimulation",
    "TDOverlap",
    "Spin",
    "Finite_Difference_NAC",
    "MASH",
    "FixedPathDynamics",
    "mom_occ_",
    "singlet_occ_",
    "AndersonAcceleration",
    "WignerSampler",
    "lowdin",
    "__version__",
]
