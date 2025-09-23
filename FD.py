import copy
import numpy as np
from TDOverlap import TDOverlap
import scipy
from numpy.linalg import svd
from TDRKS import TDRKSSimulation

class Finite_Difference_NAC: 
    def __init__(self, sim1, sim2,  displacement=0.01, steps=1):

        self.sim1_update = TDRKSSimulation(sim1.mf, dt=sim1.dt, nuclei_dt=sim1.nuclei_dt, steps=steps, field_type='none', ehrenfest=False, output_interval=-1, convergence_threshold=1e-5, mu=sim1.mu, imaginary_time=sim1.imaginary_time)
        self.sim2_update = TDRKSSimulation(sim2.mf, dt=sim2.dt, nuclei_dt=sim2.nuclei_dt, steps=steps, field_type='none', ehrenfest=False, output_interval=-1, convergence_threshold=1e-5, mu=sim2.mu, imaginary_time=sim2.imaginary_time)
        #self.sim1_update.previous_mol = self.sim1.mol.copy()
        #self.sim2_update.previous_mol = self.sim2.mol.copy()
        #self.sim1_update.convergence_threshold = 1e-8
        #self.sim2_update.convergence_threshold = 1e-8
        self.steps = steps
        self.displacement = displacement  # Displacement in Bohr

    

    def compute_nac(self, mo_coeff1 , mo_coeff2, positions):
        self.positions = positions.copy()
        nac = np.zeros((len(self.positions), 3), dtype=np.complex128)

        for i in range(len(self.positions)):   
            for j in range(3):
                
                self.sim1_update.update_mol(self.positions)
                self.sim2_update.update_mol(self.positions)
                self.sim1_update.mo_coeff = mo_coeff1.copy()
                self.sim2_update.mo_coeff = mo_coeff2.copy()
                tdoverlap_class = TDOverlap(self.sim1_update, self.sim2_update)
                displaced_positions_plus = self.positions.copy()
                displaced_positions_plus[i, j] += self.displacement
                
                # Update the molecular geometries in the simulations
                #tdoverlap_class.update_mo_coeff(self.sim1_update, self.sim2_update)

                self.sim1_update.update_mol(displaced_positions_plus)
                self.sim2_update.update_mol(displaced_positions_plus)
                for k in range(self.sim1_update.steps):
                    self.sim1_update.update_electronic_wavefunction(real_time=False)
                for k in range(self.sim2_update.steps):
                    self.sim2_update.update_electronic_wavefunction(real_time=False)

                tdoverlap_class.update_mo_coeff(self.sim1_update, self.sim2_update)
                tdoverlap = tdoverlap_class.update_overlap()[0]
                U, s, Vh = svd(tdoverlap)

                phase = U @ Vh 
                # Central difference approximation
                nac[i, j] = (scipy.linalg.logm(phase)[0,1]) / (self.displacement/(self.sim1_update.dt/self.sim1_update.mu))
               

        return nac   # Factor of 1/2 for NAC definition



if __name__ == "__main__":
    from pyscf import gto, dft
    from TDRKS import TDRKSSimulation
    from MASH import MASH
    from MOM import mom_occ_
    import numpy as np

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
    dt = 1.0# Time step in atomic units
    steps = 5
    mf1 = dft.RKS(mol,xc='lda')
    mf1.convergence = 1e-16
    mf1.kernel()
    
    #mf1.mo_occ[0,7] = 0
    #mf1.mo_occ[0,8] = 1


    mf2 = dft.RKS(mol,xc='lda')
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


    

    sim1 = TDRKSSimulation(mf1, dt=dt, nuclei_dt=dt,  steps=2, field_type='none',  ehrenfest= False,  output_interval = -1, convergence_threshold=1e-5, mu=1, imaginary_time= True )
    sim2 = TDRKSSimulation(mf2, dt=dt, nuclei_dt=dt,  steps=2, field_type='none',  ehrenfest= False,  output_interval = -1, convergence_threshold=1e-5, mu=1, imaginary_time= True )



    FD = Finite_Difference_NAC(sim1, sim2, steps=1 , displacement=0.005)

    print (FD.compute_nac(sim1.mo_coeff, sim2.mo_coeff, mol.atom_coords(unit="Bohr")))



