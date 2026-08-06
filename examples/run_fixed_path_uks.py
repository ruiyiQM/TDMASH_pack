
from pyscf import gto,scf ,dft 
from Ehrenfest2 import MASH

from matplotlib import pyplot as plt
from TDUKS import TDUKSSimulation
import numpy as np


mol = gto.M(
  atom='''
C        0.21748627      0.03041187      0.12785250
C        -0.18721957     -0.02224824     2.63987900
H        1.97520305      -0.03844242     -0.73638100
H        -1.43020513     -0.17863585     -1.11986023
H        -2.20993578     0.13003341      3.13549369
H        1.30459025      -0.01015813     3.87199466
  ''',
  basis='6-31g*',
  unit = "Bohr"
    )






    # Simulation parameters
dt = 0.1# Time step in atomic units
steps = 1000# Number of steps

mf1 = scf.UHF(mol)
mf1.convergence = 1e-16
mf1.kernel()
  
#mf1.mo_occ[0,7] = 0
#mf1.mo_occ[0,8] = 1


mf2 = scf.UHF(mol)
mf2.convergence = 1e-16
mf2.kernel()
  
mo0 = mf2.mo_coeff
occ = mf2.mo_occ

  # Assign initial occupation pattern
occ[0][7] = 0  # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
occ[0][8] = 1  # it is still a singlet state
  
dm_u = mf2.make_rdm1(mo0, occ)
  # Apply mom occupation principle
mf2 = scf.addons.mom_occ(mf2, mo0, occ)
mf2.conv_tol = 1e-16  # Increase the convergence threshold
mf2.max_cycle = 500  # Increase the maximum number of SCF cycles
  # Start new SCF with new density matrix
mf2.scf(dm_u)



sim1 = TDUKSSimulation(mf1, dt=dt, nuclei_dt=dt,  steps=600, field_type='none',  ehrenfest= False  , output_interval = -1, convergence_threshold=1e-5, mu=1.0)
sim2 = TDUKSSimulation(mf2, dt=dt, nuclei_dt=dt,  steps=600, field_type='none',  ehrenfest= False,   output_interval = -1, convergence_threshold=1e-5, mu=1.0)


# Initialize and run the MASH simulation
mash_simulation = MASH(
    sim1 =sim1,
    sim2 =sim2,
    dt=dt,
    nuclei_dt=dt,
    steps=steps,
    output_interval=-1,
    mu_1=1,
    mu_2=10,
    energy_filename="NHNH_UKS_structure_1.npz",
    apply_scf= False
    )
mash_simulation.run()
mash_simulation.plot_results()



