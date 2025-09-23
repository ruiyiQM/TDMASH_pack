from numpy.linalg import det
import numpy as np
from pyscf import gto 
from numpy.linalg import det, svd 

def update_overlap(self):



    overlap_matrix = np.zeros((2, 2), dtype=complex)

    overlap_matrix[0, 0] = compute_overlap(self.previous_mo_coeff1, self.mo_coeff1, self.overlap, self.mo_occ1, self.mo_occ1)
    overlap_matrix[1, 1] = compute_overlap(self.previous_mo_coeff2, self.mo_coeff2, self.overlap, self.mo_occ2, self.mo_occ2)
    overlap_matrix[0, 1] = compute_overlap(self.previous_mo_coeff1, self.mo_coeff2, self.overlap, self.mo_occ1, self.mo_occ2)
    overlap_matrix[1, 0] = compute_overlap(self.previous_mo_coeff2, self.mo_coeff1, self.overlap, self.mo_occ2, self.mo_occ1)
    
    overlap_matrix_2 = np.zeros((2, 2), dtype=complex)
    overlap_matrix_2[0, 0] = compute_overlap(self.mo_coeff1, self.mo_coeff1, self.overlap_same, self.mo_occ1, self.mo_occ1)
    overlap_matrix_2[1, 1] = compute_overlap(self.mo_coeff2, self.mo_coeff2, self.overlap_same, self.mo_occ2, self.mo_occ2)
    overlap_matrix_2[0, 1] = compute_overlap(self.mo_coeff1, self.mo_coeff2, self.overlap_same, self.mo_occ1, self.mo_occ2)
    overlap_matrix_2[1, 0] = compute_overlap(self.mo_coeff2, self.mo_coeff1, self.overlap_same, self.mo_occ2, self.mo_occ1)
    return [overlap_matrix, overlap_matrix_2]

def update_mo_coeff(self, sim1, sim2):



    self.previous_mo_coeff1 = self.mo_coeff1.copy()
    self.previous_mo_coeff2 = self.mo_coeff2.copy()

    self.mo_coeff1 =  sim1.mo_coeff.copy()
    self.mo_coeff2 =  sim2.mo_coeff.copy()

    self.overlap = gto.intor_cross('int1e_ovlp', sim1.previous_mol, sim1.mol)

    self.overlap_same = sim1.overlap.copy()

def update_mo_coeff_v2(self,sim1, sim2):

    
    self.previous_mo_coeff1 = self.mo_coeff1.copy()
    self.previous_mo_coeff2 = self.mo_coeff2.copy()

    overlap_matrix = sim1.mo_coeff[:, sim1.mo_occ == 2].conj().T @ sim1.overlap @ sim2.mo_coeff[:, sim2.mo_occ == 1]

    U, s, Vh = svd(overlap_matrix)

    self.mo_coeff1 = sim1.mo_coeff.copy()

    self.mo_coeff2 = sim2.mo_coeff.copy()

    self.mo_coeff2[:, sim2.mo_occ == 1] = sim2.mo_coeff[:, sim2.mo_occ == 1] @ Vh.conj().T 

    self.overlap = gto.intor_cross('int1e_ovlp', sim1.previous_mol, sim1.mol)


def update_mo_coeff_v3(self,sim1, sim2):

    
    self.previous_mo_coeff1 = self.mo_coeff1.copy()
    self.previous_mo_coeff2 = self.mo_coeff2.copy()

    

    overlap_matrix = sim1.mo_coeff[:, sim1.mo_occ == 2].conj().T @ sim1.overlap @ sim2.mo_coeff[:, sim2.mo_occ == 1]

    tensor = overlap_matrix.conj().T @ overlap_matrix

    evals, evecs = np.linalg.eig(tensor)


    order = np.argsort(np.real(evals))[::-1]
    evecs = evecs[:,order]  # Sort eigenvectors by eigenvalues in descending order
    evals = evals[order]  # Sort eigenvalues in descending order
    
    self.mo_coeff1 = sim1.mo_coeff.copy()
    self.mo_coeff2 = sim2.mo_coeff.copy()

    self.mo_coeff2[:, sim2.mo_occ == 1] = sim2.mo_coeff[:, sim2.mo_occ == 1] @ evecs

    if (sim1.previous_mol is not None):
        self.overlap = gto.intor_cross('int1e_ovlp', sim1.previous_mol, sim1.mol)
    else:
        self.overlap = sim1.overlap.copy()
    
    self.overlap_same = sim1.overlap.copy()



def compute_overlap(mo_coeff1, mo_coeff2, overlap, mo_occ1, mo_occ2):

    open_shell_1 = (np.sum(mo_occ1==1)>0)
    open_shell_2 = (np.sum(mo_occ2==1)>0)

    if not open_shell_1 and not open_shell_2:
        mo_coeff1_occ = mo_coeff1[:,mo_occ1 == 2]
        mo_coeff2_occ = mo_coeff2[:,mo_occ2 == 2]
        overlap_matrix = mo_coeff1_occ.conj().T @ overlap @ mo_coeff2_occ
        return det(overlap_matrix)**2
    elif open_shell_1 and not open_shell_2:
        mo_coeff1_occ = mo_coeff1[:,mo_occ1 == 2]
        mo_coeff1_alpha_occ = mo_coeff1[:,mo_occ1 == 1]
        mo_coeff2_occ = mo_coeff2[:,mo_occ2 == 2]

        mo_coeff1_alpha = np.hstack([mo_coeff1_occ, mo_coeff1_alpha_occ[:,0].reshape(-1,1)])
        mo_coeff1_beta  = np.hstack([mo_coeff1_occ, mo_coeff1_alpha_occ[:,1].reshape(-1,1)])
        
    
        overlap_matrix_alpha = mo_coeff1_alpha.conj().T @ overlap @ mo_coeff2_occ
        overlap_matrix_beta = mo_coeff1_beta.conj().T @ overlap @  mo_coeff2_occ
        return det(overlap_matrix_alpha) * det(overlap_matrix_beta)
    
    elif not open_shell_1 and open_shell_2:
        mo_coeff1_occ = mo_coeff1[:,mo_occ1 == 2]
        mo_coeff2_occ = mo_coeff2[:,mo_occ2 == 2]
        mo_coeff2_alpha_occ = mo_coeff2[:,mo_occ2 == 1]

        mo_coeff2_alpha = np.hstack([mo_coeff2_occ, mo_coeff2_alpha_occ[:,0].reshape(-1,1)])
        mo_coeff2_beta  = np.hstack([mo_coeff2_occ, mo_coeff2_alpha_occ[:,1].reshape(-1,1)])
        overlap_matrix_alpha = mo_coeff1_occ.conj().T @ overlap @ mo_coeff2_alpha   
        overlap_matrix_beta = mo_coeff1_occ.conj().T @ overlap @  mo_coeff2_beta
        return det(overlap_matrix_alpha) * det(overlap_matrix_beta)
    
    elif open_shell_1 and open_shell_2:
        mo_coeff1_occ = mo_coeff1[:,mo_occ1 == 2]
        mo_coeff2_occ = mo_coeff2[:,mo_occ2 == 2]
        mo_coeff1_alpha_occ = mo_coeff1[:,mo_occ1 == 1]
        mo_coeff2_alpha_occ = mo_coeff2[:,mo_occ2 == 1]

        mo_coeff1_alpha = np.hstack([mo_coeff1_occ, mo_coeff1_alpha_occ[:,0].reshape(-1,1)])
        mo_coeff1_beta  = np.hstack([mo_coeff1_occ, mo_coeff1_alpha_occ[:,1].reshape(-1,1)])
        mo_coeff2_alpha = np.hstack([mo_coeff2_occ, mo_coeff2_alpha_occ[:,0].reshape(-1,1)])
        mo_coeff2_beta  = np.hstack([mo_coeff2_occ, mo_coeff2_alpha_occ[:,1].reshape(-1,1)])
    
        overlap_matrix_alpha = mo_coeff1_alpha.conj().T @ overlap @ mo_coeff2_alpha
        overlap_matrix_beta = mo_coeff1_beta.conj().T @ overlap @  mo_coeff2_beta

        overlap_matrix2_alpha = mo_coeff1_alpha.conj().T @ overlap @ mo_coeff2_beta
        overlap_matrix2_beta = mo_coeff1_beta.conj().T @ overlap @  mo_coeff2_alpha 
        return det(overlap_matrix_alpha) * det(overlap_matrix_beta) + det(overlap_matrix2_alpha) * det(overlap_matrix2_beta)


class TDOverlap():
    def __init__(self, sim1, sim2 ):
        """
        Initialize the Overlap class with two simulations.

        Args:
            sim1 (TDRKSSimulation): First simulation object.
            sim2 (TDROKSSimulation): Second simulation object.
        """


        self.mo_occ1 = sim1.mo_occ
        self.mo_occ2 = sim2.mo_occ
        self.init_mo_coeff(sim1, sim2)

    def init_mo_coeff(self, sim1, sim2):

        overlap_matrix = sim1.mo_coeff[:, sim1.mo_occ == 2].conj().T @ sim1.overlap @ sim2.mo_coeff[:, sim2.mo_occ == 1]

        tensor = overlap_matrix.conj().T @ overlap_matrix

        evals, evecs = np.linalg.eig(tensor)


        order = np.argsort(np.real(evals))[::-1]
        evecs = evecs[:,order]  # Sort eigenvectors by eigenvalues in descending order
        evals = evals[order]  # Sort eigenvalues in descending order
    
        self.mo_coeff1 = sim1.mo_coeff.copy()
        self.mo_coeff2 = sim2.mo_coeff.copy()

        self.mo_coeff2[:, sim2.mo_occ == 1] = sim2.mo_coeff[:, sim2.mo_occ == 1] @ evecs

            # Alternatively, you can raise an error:
            # raise ValueError(f"Unsupported overlap_type: {self.type}")
    
    update_mo_coeff = update_mo_coeff_v3

    update_overlap = update_overlap


    
if __name__ == "__main__":

    from pyscf import gto, dft
    from ROKS import ROKSExcitedStateSCF
    from TDROKS import TDROKSSimulation
    from TDRKS import TDRKSSimulation
    mol = gto.Mole(atom =
    '''
    N -4.207472 1.315189 0.093245
    N -2.931316 1.315040 -0.093291
    H -4.871892 1.243618 -0.692130
    H -2.266939 1.386608 0.692134
    ''', basis = "6-31g*")
    mol.build()
    mf = dft.RKS(mol, xc='lda')
    mf.kernel()
   
    tdrks = TDRKSSimulation(mf, dt=1, nuclei_dt=1, steps=1000, field_type='none', ehrenfest=False, output_interval=-1, convergence_threshold=1e-6, mu=1)
    tdrks.update_overlap()

    roks = ROKSExcitedStateSCF(mf, max_cycle=500, level_shift=0.0, diis_damping=0.0)
    
    roks.diis_space = 12

    roks.run()

    tdroks = TDROKSSimulation(roks, dt=1, nuclei_dt=1, steps=1000, field_type='none', ehrenfest=False, output_interval=-1, convergence_threshold=1e-4, mu=1)
    tdroks.update_overlap()
    

    overlap = TDOverlap(tdrks, tdroks)
    overlap.update_mo_coeff(tdrks, tdroks)
    overlap_matrix = overlap.update_overlap()
    print("Overlap Matrix:" , overlap_matrix)
