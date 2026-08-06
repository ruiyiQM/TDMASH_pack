
from pyscf import gto,scf ,dft 
from MASH2 import MASH
import numpy as np

from matplotlib import pyplot as plt
from MOM import mom_occ_

from TDRKS import TDRKSSimulation
from TDRHF import TDRHFSimulation
mol = gto.M(atom='''
N          -7.84651        1.28271       -0.25720
N          -5.30782        3.39826       -0.51257
H          -8.45063        2.38905       -2.53646
H          -4.55615        2.29563        1.14696
''', basis='6-31g*', unit="Bohr")
mol.build()
initial_velocity = np.array([
  [-1.13505183e-05, 2.47249587e-04, -4.42711954e-04],
  [-1.49838229e-04, 3.89535031e-04, 1.68332456e-04],
  [2.58394889e-03, 5.01774265e-05, 2.90318782e-03],
  [2.06828037e-03, -3.75446805e-04, -1.87823386e-03]
])



    # Simulation parameters
dt = 0.4 # Time step in atomic units
steps = 2000
mf1 = dft.RKS(mol,xc='lda')
#mf1 = scf.RHF(mol)
#mf1 = dft.RKS(mol,xc='lda')
mf1.convergence = 1e-16
mf1.kernel()
  
#mf1.mo_occ[0,7] = 0
#mf1.mo_occ[0,8] = 1


mf2 = dft.RKS(mol,xc='lda')
#mf2 = scf.RHF(mol)
#mf2 = dft.RKS(mol,xc='lda')
mf2.convergence = 1e-16


mo0 = mf1.mo_coeff

occ =np.zeros((2, mf1.mo_coeff.shape[1]))
occ[0, mf1.mo_occ > 0] = 1  # Set occupied orbitals to 1
occ[1, mf1.mo_occ > 0] = 1  # Set occupied orbitals to 1 for beta spin

occ[0,7] = 0  # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
occ[0,8] = 1  # it is still a singlet state
#print (occ)
mf2.init_guess = mf1.mo_coeff
#dm_u = mf2.make_rdm1(mo0, occ)
  # Apply mom occupation principle
mf2= mom_occ_(mf2, mf1.mo_coeff.copy(), occ)
mf2.conv_tol = 1e-16  # Increase the convergence threshold
mf2.max_cycle = 500  # Increase the maximum number of SCF cycles
  # Start new SCF with new density matrix
mf2.kernel()

sim1 = TDRKSSimulation(mf1, dt=dt, nuclei_dt=dt, steps=600, field_type='none',  ehrenfest= False,  output_interval = -1, convergence_threshold=1e-5, mu=80, propagation_method='accelerated',imaginary_time= False  ,count= True )
sim2 = TDRKSSimulation(mf2, dt=dt, nuclei_dt=dt, steps=600, field_type='none',  ehrenfest= False,  output_interval = -1, convergence_threshold=1e-5, mu=1, imaginary_time= False  )


# Initialize and run the MASH simulation
mash_simulation = MASH(
    sim1 =sim1,
    sim2 =sim2,
    dt=dt,
    nuclei_dt=dt,
    steps=steps,
    output_interval=-1,
    mu_1=1,
    mu_2=1,
    energy_filename="debug_MASH.npz",
    initial_velocity=initial_velocity,
    apply_scf=False,
    initial_spin=[0,0,-1]
    #output_filename="trajectory_RKS.xyz"
    )
mash_simulation.run()

#np.savez("NHNH_RKS_test_mu_0d1_structure_1_final_debug_overlap.npz", mo_coeff2=mash_simulation.sim2.mo_coeff, mo_coeff1=mash_simulation.sim1.mo_coeff, positions= mash_simulation.sim1.mol.atom_coords())
mash_simulation.plot_results()



