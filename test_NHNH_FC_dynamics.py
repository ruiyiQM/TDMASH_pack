
from pyscf import gto,scf ,dft 
from Ehrenfest import MASH 
import numpy as np

from matplotlib import pyplot as plt
from MOM import mom_occ_

from TDRKS import TDRKSSimulation
mol = gto.M(atom='''
   N  -0.000000   0.090422   0.007506   
   N  -0.000000  -0.090422   1.242501  
   H   0.000000   1.121059  -0.123926    
   H   0.000000  -1.121060   1.373920
''', basis='6-31g*')
mol.build()




    # Simulation parameters
dt = 1# Time step in atomic units
steps = 100
mf1 = dft.RKS(mol,xc='m06')
mf1.convergence = 1e-16
mf1.kernel()
  
#mf1.mo_occ[0,7] = 0
#mf1.mo_occ[0,8] = 1


mf2 = dft.RKS(mol,xc='m06')
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

sim1 = TDRKSSimulation(mf1, dt=dt, nuclei_dt=dt,  steps=600, field_type='none',  ehrenfest= False,  output_interval = -1, convergence_threshold=1e-5, mu=1, imaginary_time= False )
sim2 = TDRKSSimulation(mf2, dt=dt, nuclei_dt=dt,  steps=600, field_type='none',  ehrenfest= False,  output_interval = -1, convergence_threshold=1e-5, mu=1, imaginary_time= False )


# Initialize and run the MASH simulation
mash_simulation = MASH(
    sim1 =sim1,
    sim2 =sim2,
    dt=dt,
    nuclei_dt=dt,
    steps=steps,
    output_interval=1,
    mu_1=1,
    mu_2=1,
    energy_filename="test_FC_dynamics.npz",
    apply_scf= False,
    initial_spin=[0,0,-1]

    )
mash_simulation.run()

#np.savez("NHNH_RKS_test_mu_0d1_structure_1_final_debug_overlap.npz", mo_coeff2=mash_simulation.sim2.mo_coeff, mo_coeff1=mash_simulation.sim1.mo_coeff, positions= mash_simulation.sim1.mol.atom_coords())
mash_simulation.plot_results()



