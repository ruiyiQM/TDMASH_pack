from numpy.linalg import det
import numpy as np
from pyscf import gto 
from numpy.linalg import det, svd 

def update_overlap(self):
    """Time-overlap matrix O_MN = <Phi_M(t) | Phi_N(t + dt)> of the two states.

    Rows use the coefficients stored at the previous call (previous geometry),
    columns the current ones; ``self.overlap`` is the AO cross-overlap between
    the two geometries.  This is the matrix whose unitary part gives the
    time-derivative coupling used by ``Spin.propogate_spin``.
    """
    overlap_matrix = np.zeros((2, 2), dtype=complex)
    overlap_matrix[0, 0] = compute_overlap(self.previous_mo_coeff1, self.mo_coeff1, self.overlap, self.mo_occ1, self.mo_occ1)
    overlap_matrix[1, 1] = compute_overlap(self.previous_mo_coeff2, self.mo_coeff2, self.overlap, self.mo_occ2, self.mo_occ2)
    overlap_matrix[0, 1] = compute_overlap(self.previous_mo_coeff1, self.mo_coeff2, self.overlap, self.mo_occ1, self.mo_occ2)
    overlap_matrix[1, 0] = compute_overlap(self.previous_mo_coeff2, self.mo_coeff1, self.overlap, self.mo_occ2, self.mo_occ1)
    return overlap_matrix


def update_overlap_same_time(self):
    """Same-time overlap <Phi_M(t) | Phi_N(t)> (diagnostic of nonorthogonality only).

    Not a coupling: it is Hermitian and positive, so its unitary part is the
    identity and it would give zero time-derivative coupling.
    """
    overlap_matrix_2 = np.zeros((2, 2), dtype=complex)
    overlap_matrix_2[0, 0] = compute_overlap(self.mo_coeff1, self.mo_coeff1, self.overlap_same, self.mo_occ1, self.mo_occ1)
    overlap_matrix_2[1, 1] = compute_overlap(self.mo_coeff2, self.mo_coeff2, self.overlap_same, self.mo_occ2, self.mo_occ2)
    overlap_matrix_2[0, 1] = compute_overlap(self.mo_coeff1, self.mo_coeff2, self.overlap_same, self.mo_occ1, self.mo_occ2)
    overlap_matrix_2[1, 0] = compute_overlap(self.mo_coeff2, self.mo_coeff1, self.overlap_same, self.mo_occ2, self.mo_occ1)
    return overlap_matrix_2


def update_mo_coeff(self, sim1, sim2,  previous_mol = None):



    self.previous_mo_coeff1 = self.mo_coeff1.copy()
    self.previous_mo_coeff2 = self.mo_coeff2.copy()

    self.mo_coeff1 =  sim1.mo_coeff.copy()
    self.mo_coeff2 =  sim2.mo_coeff.copy()

    if (previous_mol  is not None):
        self.overlap = gto.intor_cross('int1e_ovlp', previous_mol, sim1.mol)
    else:
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


def align_orbital_phases(coeff, reference, overlap):
    """Choose phases so corresponding reference overlaps are real positive."""
    diagonal = np.einsum('ij,ij->j', reference.conj(), overlap @ coeff)
    return coeff * np.exp(-1j * np.angle(diagonal))


def dto_coefficients(sim1, sim2):
    active = sim2.mo_occ == 1
    overlap = sim1.mo_coeff[:, sim1.mo_occ == 2].conj().T @ sim1.overlap @ sim2.mo_coeff[:, active]
    _, eigenvectors = np.linalg.eigh(overlap.conj().T @ overlap)
    coeff = sim2.mo_coeff.copy()
    coeff[:, active] = sim2.mo_coeff[:, active] @ eigenvectors[:, ::-1]
    return sim1.mo_coeff.copy(), coeff


def update_mo_coeff_v3(self, sim1, sim2, previous_mol=None, reference=None):
    self.previous_mo_coeff1 = self.mo_coeff1.copy()
    self.previous_mo_coeff2 = self.mo_coeff2.copy()
    self.overlap = gto.intor_cross(
        'int1e_ovlp', self.mol if previous_mol is None else previous_mol, sim1.mol)
    self.mo_coeff1, self.mo_coeff2 = dto_coefficients(sim1, sim2)
    if reference is None:
        active = sim2.mo_occ == 1
        self.mo_coeff2[:, active] = align_orbital_phases(
            self.mo_coeff2[:, active], self.previous_mo_coeff2[:, active], self.overlap)
    else:
        # Spatial differences use one central gauge for both displaced endpoints.
        cross = gto.intor_cross('int1e_ovlp', reference.mol, sim1.mol)
        self.mo_coeff1 = align_orbital_phases(self.mo_coeff1, reference.mo_coeff1, cross)
        self.mo_coeff2 = align_orbital_phases(self.mo_coeff2, reference.mo_coeff2, cross)
    self.mol = sim1.mol.copy()
    self.overlap_same = sim1.overlap.copy()

def update_mo_coeff_v4(self,sim1, sim2, previous_mol = None):

    
    self.previous_mo_coeff1 = self.mo_coeff1.copy()
    self.previous_mo_coeff2 = self.mo_coeff2.copy()

    

    overlap_matrix = sim1.mo_coeff[:, sim1.mo_occ == 2].conj().T @ sim1.overlap @ sim2.mo_coeff[:, sim2.mo_occ == 1]

    tensor = overlap_matrix.conj().T @ overlap_matrix

    evals, evecs = np.linalg.eig(tensor)


    order = np.argsort(np.real(evals))[::-1]
    evecs = evecs[:,order]  # Sort eigenvectors by eigenvalues in descending order
    evals = evals[order]  # Sort eigenvalues in descending order
    
    self.mo_coeff1 = sim1.overlap_half@ sim1.mo_coeff.copy()
    self.mo_coeff2 = sim1.overlap_half@sim2.mo_coeff.copy()

    self.mo_coeff2[:, sim2.mo_occ == 1] = sim1.overlap_half @ sim2.mo_coeff[:, sim2.mo_occ == 1] @ evecs

    self.overlap = np.identity(sim1.mo_coeff.shape[0])
    
    self.overlap_same = np.identity(sim1.mo_coeff.shape[0])
    

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
        # Singlet (|h lb> + |l hb>)/sqrt(2): both determinants give the same
        # overlap with the closed-shell state, hence the factor sqrt(2).
        return np.sqrt(2.0) * det(overlap_matrix_alpha) * det(overlap_matrix_beta)
    
    elif not open_shell_1 and open_shell_2:
        mo_coeff1_occ = mo_coeff1[:,mo_occ1 == 2]
        mo_coeff2_occ = mo_coeff2[:,mo_occ2 == 2]
        mo_coeff2_alpha_occ = mo_coeff2[:,mo_occ2 == 1]

        mo_coeff2_alpha = np.hstack([mo_coeff2_occ, mo_coeff2_alpha_occ[:,0].reshape(-1,1)])
        mo_coeff2_beta  = np.hstack([mo_coeff2_occ, mo_coeff2_alpha_occ[:,1].reshape(-1,1)])
        overlap_matrix_alpha = mo_coeff1_occ.conj().T @ overlap @ mo_coeff2_alpha   
        overlap_matrix_beta = mo_coeff1_occ.conj().T @ overlap @  mo_coeff2_beta
        # Singlet normalization, see the open/closed case above.
        return np.sqrt(2.0) * det(overlap_matrix_alpha) * det(overlap_matrix_beta)
    
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
    def __init__(self, sim1, sim2, reference=None):
        """
        Initialize the Overlap class with two simulations.

        Args:
            sim1 (TDRKSSimulation): First simulation object.
            sim2 (TDROKSSimulation): Second simulation object.
        """


        self.mo_occ1 = sim1.mo_occ
        self.mo_occ2 = sim2.mo_occ
        self.init_mo_coeff(sim1, sim2, reference=reference)

    def init_mo_coeff(self, sim1, sim2, reference=None):
        self.mo_coeff1, self.mo_coeff2 = dto_coefficients(sim1, sim2)
        self.mol = sim1.mol.copy()
        if reference is None:
            active = sim2.mo_occ == 1
            self.mo_coeff2[:, active] = align_orbital_phases(
                self.mo_coeff2[:, active], sim2.mo_coeff[:, active], sim1.overlap)
        else:
            cross = gto.intor_cross('int1e_ovlp', reference.mol, sim1.mol)
            self.mo_coeff1 = align_orbital_phases(self.mo_coeff1, reference.mo_coeff1, cross)
            self.mo_coeff2 = align_orbital_phases(self.mo_coeff2, reference.mo_coeff2, cross)

    # DTO gauge + cross-geometry AO overlap (the scheme used for the paper).
    update_mo_coeff = update_mo_coeff_v3

    update_overlap = update_overlap

    update_overlap_same_time = update_overlap_same_time
