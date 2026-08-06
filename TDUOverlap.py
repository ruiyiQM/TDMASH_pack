from numpy.linalg import det
import numpy as np
from pyscf import gto 
from numpy.linalg import det, svd 


class TDOverlap():
    def __init__(self, sim):
        """
        Initialize the Overlap class with two simulations.

        Args:
            sim1 (TDRKSSimulation): First simulation object.
            sim2 (TDROKSSimulation): Second simulation object.
        """


        self.mo_coeff = sim.mo_coeff.copy()
        self.mol = sim.mol.copy()
        self.previous_mol = None 
        self.previous_mo_coeff = None

    def update_mo_coeff(self, sim):

        self.previous_mo_coeff = self.mo_coeff.copy()
        self.previous_mol = self.mol.copy()

        self.mo_coeff =  sim.mo_coeff.copy()
        self.mol = sim.mol.copy()
        self.overlap = gto.intor_cross('int1e_ovlp', self.previous_mol, self.mol)

    def update_overlap(self):

        overlap_alpha = self.previous_mo_coeff[0][:, :].conj().T @ self.overlap @ self.mo_coeff[0][:, :]
        overlap_beta = self.previous_mo_coeff[1][:, :].conj().T @ self.overlap @ self.mo_coeff[1][:, :] 
        #print (overlap_alpha.shape, overlap_beta.shape)
        return det(overlap_alpha) * det(overlap_beta)



    
if __name__ == "__main__":

    from pyscf import gto, dft

    from TDUKS import TDUKSSimulation
    mol = gto.Mole(atom =
    '''
    N -4.207472 1.315189 0.093245
    N -2.931316 1.315040 -0.093291
    H -4.871892 1.243618 -0.692130
    H -2.266939 1.386608 0.692134
    ''', basis = "6-31g*")
    mol.build()
    mf = dft.UKS(mol, xc='lda')
    mf.kernel()
   
    tdrks = TDUKSSimulation(mf, dt=1, nuclei_dt=1, steps=1, field_type='none', ehrenfest=False, output_interval=-1, convergence_threshold=1e-6, mu=1)
    
    td_overlap = TDOverlap(tdrks)
    tdrks.run()

    td_overlap.update_mo_coeff(tdrks)
    overlap = td_overlap.update_overlap()
    print("Overlap:", overlap)

