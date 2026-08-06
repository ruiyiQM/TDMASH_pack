import copy
import numpy as np
from TDOverlap import TDOverlap
import scipy
from numpy.linalg import svd
from TDRKS import TDRKSSimulation
from PT2 import PTCN_TDRHFSimulation
class Finite_Difference_NAC: 
    def __init__(self, sim1, sim2,  displacement=0.01, steps=1):
        

        self.sim1_forward = TDRKSSimulation(sim1.mf, dt=sim1.dt, nuclei_dt=sim1.nuclei_dt, steps=steps, field_type='none', ehrenfest=False, output_interval=-1, convergence_threshold=1e-6,  mu=sim1.mu, imaginary_time=sim1.imaginary_time , propagation_method='accelerated')
        self.sim2_forward =  TDRKSSimulation(sim2.mf, dt=sim2.dt, nuclei_dt=sim2.nuclei_dt, steps=steps, field_type='none', ehrenfest=False, output_interval=-1, convergence_threshold=1e-6, mu=sim2.mu, imaginary_time=sim2.imaginary_time , propagation_method='accelerated')
        self.sim1_backward = TDRKSSimulation(sim1.mf, dt=-sim1.dt, nuclei_dt=sim1.nuclei_dt, steps=steps, field_type='none', ehrenfest=False, output_interval=-1, convergence_threshold=1e-6, mu=sim1.mu, imaginary_time=sim1.imaginary_time , propagation_method='accelerated')
        self.sim2_backward = TDRKSSimulation(sim2.mf, dt=-sim2.dt, nuclei_dt=sim2.nuclei_dt, steps=steps, field_type='none', ehrenfest=False, output_interval=-1, convergence_threshold=1e-6, mu=sim2.mu, imaginary_time=sim2.imaginary_time , propagation_method='accelerated' )
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

                self.sim1_forward.update_mol(self.positions)
                self.sim2_forward.update_mol(self.positions)
                self.sim1_forward.mo_coeff = mo_coeff1.copy()
                self.sim2_forward.mo_coeff = mo_coeff2.copy()


                self.sim1_backward.update_mol(self.positions)
                self.sim2_backward.update_mol(self.positions)
                self.sim1_backward.mo_coeff = mo_coeff1.copy()
                self.sim2_backward.mo_coeff = mo_coeff2.copy()      
                                
                displaced_positions_plus = self.positions.copy()
                displaced_positions_plus[i, j] += self.displacement

                displaced_positions_half_plus = self.positions.copy()
                displaced_positions_half_plus[i, j] += self.displacement * 0.5

                displaced_positions_minus = self.positions.copy()
                displaced_positions_minus[i, j] -= self.displacement
                
                displaced_positions_half_minus = self.positions.copy()
                displaced_positions_half_minus[i, j] -= self.displacement * 0.5


                self.sim1_forward.update_mol(displaced_positions_half_plus)
                self.sim2_forward.update_mol(displaced_positions_half_plus)
                self.sim1_forward.update_electronic_wavefunction(apply_scf=False, dt = self.sim1_forward.dt, mu=self.sim1_forward.mu) 
                self.sim2_forward.update_electronic_wavefunction(apply_scf=False, dt = self.sim2_forward.dt, mu=self.sim2_forward.mu)
                self.sim1_forward.update_mol(displaced_positions_plus)
                self.sim2_forward.update_mol(displaced_positions_plus)


                self.sim1_backward.update_mol(displaced_positions_half_minus)
                self.sim2_backward.update_mol(displaced_positions_half_minus)
                self.sim1_backward.update_electronic_wavefunction(apply_scf=False, dt = self.sim1_backward.dt, mu=self.sim1_backward.mu) 
                self.sim2_backward.update_electronic_wavefunction(apply_scf=False, dt = self.sim2_backward.dt, mu=self.sim2_backward.mu)
                self.sim1_backward.update_mol(displaced_positions_minus)
                self.sim2_backward.update_mol(displaced_positions_minus)

                tdoverlap_class = TDOverlap(self.sim1_backward, self.sim2_backward)
                tdoverlap_class.update_mo_coeff(self.sim1_forward, self.sim2_forward)
                tdoverlap = tdoverlap_class.update_overlap()
                #print (tdoverlap)
                #print (np.angle(tdoverlap[0,0],deg=True), np.angle(tdoverlap[0,1],deg=True), np.angle(tdoverlap[1,0],deg=True), np.angle(tdoverlap[1,1],deg=True))

                U, s, Vh = svd(tdoverlap)

                phase = U @ Vh 
                #print (phase)
                nac[i, j] = (scipy.linalg.logm(phase).real[0,1]) / (2*self.displacement)

        norm = np.dot(nac.flatten(), nac.flatten())**0.5
        return nac /norm  # Factor of 1/2 for NAC definition



if __name__ == "__main__":
    from pyscf import gto, dft , scf
    from TDRKS import TDRKSSimulation
    from MASH import MASH
    from MOM import mom_occ_
    from TDRHF import TDRHFSimulation
    import numpy as np

    mol = gto.M (atom='''
    N       -0.005705     0.091476     0.001669
    N       -0.005705    -0.101830     1.247086
    H        0.081005     1.153029    -0.129897
    H        0.081006    -1.006810     1.398653
    ''', basis='6-31g*',verbose = 0)




        # Simulation parameters
    dt = 0.01# Time step in atomic units
    steps = 5
    #mf1 = dft.RKS(mol,xc='lda')
    mf1 = scf.RHF(mol)
    mf1.convergence = 1e-16
    mf1.kernel()
    
    #mf1.mo_occ[0,7] = 0
    #mf1.mo_occ[0,8] = 1


    mf2 = scf.RHF(mol)
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


    

    sim1 = TDRHFSimulation(mf1, dt=dt, nuclei_dt=dt,  steps=2, field_type='none',  ehrenfest= False,  output_interval = -1, convergence_threshold=1e-8, mu=1, imaginary_time= False )
    sim2 = TDRHFSimulation(mf2, dt=dt, nuclei_dt=dt,  steps=2, field_type='none',  ehrenfest= False,  output_interval = -1, convergence_threshold=1e-8, mu=1, imaginary_time= False )
    #sim1.mo_coeff = 1j* sim1.mo_coeff
    #sim2.mo_coeff = 1j* sim2.mo_coeff
    FD = Finite_Difference_NAC(sim1, sim2, steps=1 , displacement=0.001)

    print (FD.compute_nac(sim1.mo_coeff, sim2.mo_coeff, mol.atom_coords(unit="Bohr")))



