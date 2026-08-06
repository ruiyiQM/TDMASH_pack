from pyscf import gto, scf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from TDRHF import TDRHFSimulation

class TDUHFSimulation(TDRHFSimulation):
    """
    Time-Dependent Unrestricted Hartree-Fock (TDUHF) Simulation.
    Inherits from TDRHFSimulation and overrides methods for spin-separated calculations.
    """

    def initialize_molecular_orbitals(self, n_virtual):
        """Initializes molecular orbitals and occupation numbers for UHF."""
        self.n_virtual = n_virtual
        
        # mf.mo_coeff and mf.mo_occ for UHF have shape (2, ...) for alpha/beta spins
        if n_virtual == 0:
            # Select only MOs that have an occupation number > 1e-3
            occupied_mask_alpha = self.mf.mo_occ[0] > 1e-3 
            occupied_mask_beta = self.mf.mo_occ[1] > 1e-3
            
            mo_coeff_alpha = self.mf.mo_coeff[0][:, occupied_mask_alpha].astype(complex)
            mo_coeff_beta = self.mf.mo_coeff[1][:, occupied_mask_beta].astype(complex)
            
            mo_occ_alpha = self.mf.mo_occ[0][occupied_mask_alpha].astype(complex)
            mo_occ_beta = self.mf.mo_occ[1][occupied_mask_beta].astype(complex)
            
            # Store as numpy array of shape (2, NAO, NMO_spin)
            # Note: alpha and beta might have different numbers of occupied orbitals if not careful,
            # but standard numpy arrays need same shape. 
            # If they differ, prefer using a list or ensuring equal size by padding (complex).
            # Assuming they are equal for typical closed-shell starting points or handled by user.
            # For general UHF, best to keep as object array or list if sizes differ.
            # Here we assume standard cases or that the user knows they might differ.
            # SAFEGUARD: if shapes differ, we might need to use a list or object array.
            # For now, following the original pattern which implies they can be stacked:
            self.mo_coeff = np.array([mo_coeff_alpha, mo_coeff_beta])
            self.mo_occ = np.array([mo_occ_alpha, mo_occ_beta])
        else:
            raise ValueError("n_virtual must be 0 for current TDUHF implementation.")

        self.prev_mo_coeff = self.mo_coeff.copy()

    def make_rdm1(self, mo_coeff, mo_occ):
        """
        Makes the UHF total density matrix from MO coefficients and occupations.
        Returns a (2, NAO, NAO) array for [alpha, beta] density matrices.
        """
        dm_alpha = mo_coeff[0] @ np.diag(mo_occ[0]) @ mo_coeff[0].conj().T
        dm_beta = mo_coeff[1] @ np.diag(mo_occ[1]) @ mo_coeff[1].conj().T
        return np.asarray([dm_alpha, dm_beta])

    def get_fock_matrix(self, dm_total=None, E_t=0):
        """
        Calculates the UHF Fock matrix (alpha and beta components).
        """
        if dm_total is None:
            dm_total = self.make_rdm1(self.mo_coeff, self.mo_occ)
        
        # Use cached H_core
        h_core = self.h_core

        # UHF V_eff (Coulomb - Exchange for each spin)
        # PySCF get_veff for UHF with dm_total shape (2, NAO, NAO) returns (2, NAO, NAO) V_eff
        self.veff = self.mf.get_veff(self.mol, dm_total.real, dm_last=self.dm_last.real, vhf_last=self.veff_last)
        
        self.veff_last = self.veff.copy()
        # External field interaction term
        field_component_map = {'x': 0, 'y': 1, 'z': 2}
        field_idx = field_component_map.get(self.field_direction.lower(), 2)
        dipole_integrals = self.mol.intor('cint1e_r_sph', comp=3) 
        field_term = E_t * dipole_integrals[field_idx]

        # Total Fock matrix for alpha and beta
        F_total_alpha = self.h_core + self.veff[0] - field_term
        F_total_beta  = self.h_core + self.veff[1] - field_term
        
        return np.asarray([F_total_alpha, F_total_beta]).astype(complex)

    def energy_tot(self, dm_total=None):
        """
        Compute the total electronic energy for UHF.
        """
        if dm_total is None:
            dm_total = self.make_rdm1(self.mo_coeff, self.mo_occ)
            
        h_core = self.h_core
        # PySCF get_veff for UHF returns (V_eff_alpha, V_eff_beta)
        vhf = self.mf.get_veff(self.mol, dm_total.real) 
        
        # Energy = Tr(H_core * D_tot) + 0.5 * Tr(V_eff_alpha * D_alpha + V_eff_beta * D_beta)
        dm_sum = dm_total[0] + dm_total[1]
        e1 = np.einsum('ij,ji->', h_core.real, dm_sum.real)
        e_coul = 0.5 * (np.einsum('ij,ji->', vhf[0].real, dm_total[0].real) + 
                        np.einsum('ij,ji->', vhf[1].real, dm_total[1].real))
        
        electronic_energy = (e1 + e_coul).real
        return electronic_energy + self.mol.energy_nuc()

    def predictor_corrector_propagation(self, F_current, E_t, dt=None):
        """
        Performs one step of predictor-corrector propagation for UHF.
        Propagates alpha and beta orbitals separately.
        """
        self.prev_mo_coeff = self.mo_coeff.copy()
        if dt is None: dt = self.dt

        # Predictor step
        local_F_scaled = F_current * self.mu
        F_t_scaled = local_F_scaled.copy()

        # Alpha propagation matrices
        A_pred_alpha = self.overlap + 0.5j * self.dt * F_t_scaled[0]
        B_pred_alpha = self.overlap - 0.5j * self.dt * F_t_scaled[0]
        # Beta propagation matrices
        A_pred_beta = self.overlap + 0.5j * self.dt * F_t_scaled[1]
        B_pred_beta = self.overlap - 0.5j * self.dt * F_t_scaled[1]
        
        mo_coeff_predict_iter = np.zeros_like(self.mo_coeff, dtype=complex)
        mo_coeff_predict_iter[0] = np.linalg.solve(A_pred_alpha, B_pred_alpha @ self.mo_coeff[0])
        mo_coeff_predict_iter[1] = np.linalg.solve(A_pred_beta, B_pred_beta @ self.mo_coeff[1])

        # Corrector iterations
        for iter_count in range(self.max_iterations):
            dm_predict = self.make_rdm1(mo_coeff=mo_coeff_predict_iter, mo_occ=self.mo_occ)
            E_t_plus_dt = self.electric_field(self.t + self.dt)
            F_predict_raw = self.get_fock_matrix(dm_predict, E_t_plus_dt)
            self.F = F_predict_raw.copy()
            F_predict_scaled = F_predict_raw * self.mu
            
            F_average_scaled = 0.5 * (local_F_scaled + F_predict_scaled)

            # Corrector step for alpha/beta
            A_corr_alpha = self.overlap + 0.5j * self.dt * F_average_scaled[0]
            B_corr_alpha = self.overlap - 0.5j * self.dt * F_average_scaled[0]
            A_corr_beta = self.overlap + 0.5j * self.dt * F_average_scaled[1]
            B_corr_beta = self.overlap - 0.5j * self.dt * F_average_scaled[1]
            
            mo_coeff_corrected = np.zeros_like(self.mo_coeff, dtype=complex)
            mo_coeff_corrected[0] = np.linalg.solve(A_corr_alpha, B_corr_alpha @ self.mo_coeff[0])
            mo_coeff_corrected[1] = np.linalg.solve(A_corr_beta, B_corr_beta @ self.mo_coeff[1])

            #if iter_count > 0:
            mo_diff = np.max(np.abs(mo_coeff_corrected - mo_coeff_predict_iter))
            print(f"  Iteration: {iter_count}, MO diff: {mo_diff:.2e}")
            if mo_diff < self.convergence_threshold:
                mo_coeff_predict_iter = mo_coeff_corrected
                break
            
            mo_coeff_predict_iter = mo_coeff_corrected.copy()
            # F_t_scaled = F_predict_scaled.copy() # Standard P-C doesn't usually update F(t), kept as commented in original TDUHF

            if iter_count == self.max_iterations - 1:
                print(f"Warning: Predictor-corrector did not converge in {self.max_iterations} iterations at step {int(self.t/self.dt)}")

        self.mo_coeff = mo_coeff_predict_iter.copy()

    def apply_elec_nuc_coupling(self, F_matrix):
        """Applies Non-Adiabatic Coupling (NAC) term for Ehrenfest dynamics to the UHF Fock matrix."""
        ipovlp = self.mol.intor('int1e_ipovlp', comp=3)
        aoslices = self.mol.aoslice_by_atom()
        
        nac_term_matrix = np.zeros_like(F_matrix[0], dtype=complex)

        for ia in range(self.mol.natm):
            p0, p1, _, _ = aoslices[ia]
            v_ia = self.velocities[ia]
            nac_term_matrix[p0:p1] += 1j * np.einsum('sij,s->ij', ipovlp[:, p0:p1], v_ia)
            
        # Apply to both alpha and beta Fock matrices
        F_matrix[0] -= nac_term_matrix.conj().T
        F_matrix[1] -= nac_term_matrix.conj().T
        return F_matrix 

    def orthogonalize_mo_coeff(self, mo_coeff=None):
        """Orthogonalizes alpha and beta molecular orbital coefficients separately."""
        if mo_coeff is None:
            mo_coeff = self.mo_coeff.copy()
            
        # Orthogonalize alpha
        overlap_alpha = mo_coeff[0].conj().T @ self.overlap @ mo_coeff[0]
        try:
            S_inv_sqrt_alpha = scipy.linalg.inv(scipy.linalg.sqrtm(overlap_alpha))
            mo_coeff[0] = mo_coeff[0] @ S_inv_sqrt_alpha
        except np.linalg.LinAlgError:
            # Fallback using eigh for alpha
            evals, evecs = scipy.linalg.eigh(overlap_alpha)
            evals_sqrt_inv = np.zeros_like(evals)
            mask = evals > 1e-12
            evals_sqrt_inv[mask] = 1.0 / np.sqrt(evals[mask])
            S_inv_sqrt = evecs @ np.diag(evals_sqrt_inv) @ evecs.conj().T
            mo_coeff[0] = mo_coeff[0] @ S_inv_sqrt

        # Orthogonalize beta
        overlap_beta = mo_coeff[1].conj().T @ self.overlap @ mo_coeff[1]
        try:
             S_inv_sqrt_beta = scipy.linalg.inv(scipy.linalg.sqrtm(overlap_beta))
             mo_coeff[1] = mo_coeff[1] @ S_inv_sqrt_beta
        except np.linalg.LinAlgError:
            # Fallback using eigh for beta
            evals, evecs = scipy.linalg.eigh(overlap_beta)
            evals_sqrt_inv = np.zeros_like(evals)
            mask = evals > 1e-12
            evals_sqrt_inv[mask] = 1.0 / np.sqrt(evals[mask])
            S_inv_sqrt = evecs @ np.diag(evals_sqrt_inv) @ evecs.conj().T
            mo_coeff[1] = mo_coeff[1] @ S_inv_sqrt

        return mo_coeff

    def energy_grad(self, F_matrix,  dm_total=None):
        

        """
        Compute the gradient of the total RHF energy with respect to nuclear coordinates.
        F_rhf is the current RHF Fock matrix.
        dm0_rhf is the current RHF density matrix.
        """
        if dm_total is None:
            dm_total= self.make_rdm1(
                mo_coeff=self.mo_coeff,
                mo_occ=self.mo_occ
            )
        # Get RHF gradient object from the current mf (which has updated mol if needed)
        mf_grad = self.mf.nuc_grad_method() 
        mol = self.mol # Current molecule object

        hcore_deriv_generator = mf_grad.hcore_generator(mol) # Generator for dHcore/dRa
        s1_deriv_ao = mf_grad.get_ovlp(mol) # dS/dRa, list of (3, nao, nao) per atom
                                            # Or (3, nao, nao) - check PySCF
                                            # PySCF: list of length natm, each item (3, N, N)
        
        # Derivative of RHF effective potential V_eff = J - 0.5K
        # mf_grad.get_veff(mol, dm0_rhf.real) returns (dveff/dx, dveff/dy, dveff/dz)
        # where each component is (natm, nao, nao)

        vhf_deriv_ao = mf_grad.get_veff(mol, dm_total.real) # Shape (3, nao, nao)
        #print (vhf_deriv_ao)
        # Energy-weighted density matrix term for Pulay force
        # corr = S^-1 @ F @ D (S^-1 F D is related to energy-weighted DM P_E S)
        corr = self.overlap_inv @ F_matrix[0] @ dm_total[0] + self.overlap_inv @ F_matrix[1] @ dm_total[1]

        atmlst = range(mol.natm)
        de = np.zeros((len(atmlst), 3))

        aoslices = mol.aoslice_by_atom()


        for k_atom_idx, ia_atom_glob_idx in enumerate(atmlst):
            
            #_, _, p0, p1 = aoslices[ia_atom_glob_idx]

            p0, p1 = aoslices [ia_atom_glob_idx,2:]

            h1ao_deriv_atom_k = hcore_deriv_generator(ia_atom_glob_idx) # (3, nao, nao)

            de[k_atom_idx] += np.einsum('xij,sij->x', h1ao_deriv_atom_k, dm_total.real)    

            # Contribution from derivative of V_eff
            # dVeff/d(R_k_atom_idx) has shape (3, nao, nao)

            de[k_atom_idx] += np.einsum('sxij,sij->x', vhf_deriv_ao[:,:, p0:p1] , dm_total[:,p0:p1].real)* 2.0 # Factor of 2 for double counting in Coulomb term
           
            # Pulay force contribution from dS/dR   
            # s1_deriv_ao is a list, s1_deriv_ao[k_atom_idx] is (3, nao, nao) for atom k_atom_idx
            # Term is -Tr[(corr + corr.conj().T) @ dS/dRa_k]
            # Note: PySCF grad objects often handle complex numbers internally or expect real inputs.
            # The original UHF code used .real for einsum results.
            # corr is complex. s1_deriv_atom_k is real.
            # (corr + corr.H) is Hermitian.
            pulay_term_matrix = corr + corr.conj().T 
            de[k_atom_idx] -= np.einsum('xij,ij->x', s1_deriv_ao[:,p0:p1], pulay_term_matrix[p0:p1]).real 
            # For any extra terms specific to the gradient method (e.g., for specific functionals in DFT)
            # For HF, this is usually zero unless there are other perturbations.
            de[k_atom_idx] += mf_grad.extra_force(ia_atom_glob_idx, locals()) # locals() might not be safe here. Pass specific needed vars.
            # For standard HF, this term is often not needed or is handled by other parts.

        de += mf_grad.grad_nuc(mol) # Nuclear repulsion gradient

        return de


    def calculate_dipole(self, dm_total):
        """
        Calculates the total dipole moment for UHF.
        Overrides TDRHF to handle (2, NAO, NAO) density matrix.
        """
        dipole_integrals = self.mol.intor('cint1e_r_sph', comp=3)
        dm_sum = dm_total[0] + dm_total[1]
        dip_x = np.trace(dm_sum.real @ dipole_integrals[0])
        dip_y = np.trace(dm_sum.real @ dipole_integrals[1])
        dip_z = np.trace(dm_sum.real @ dipole_integrals[2])
        return np.array([dip_x, dip_y, dip_z]).real

    def update_mol(self, positions=None):
        """Updates the PySCF molecule object and rebuilds UHF mean-field."""
        if positions is None:
            positions = self.positions

        atom_coords_str = [f"{self.mol.atom_symbol(i)} {p[0]} {p[1]} {p[2]}" for i, p in enumerate(positions)]
        
        new_mol = gto.M(atom="\n".join(atom_coords_str),
                        basis=self.mol.basis,
                        unit="Bohr",
                        charge=self.mol.charge,
                        spin=self.mol.spin,
                        cart=self.mol.cart)
        if self.mol.ecp is not None:
            new_mol.ecp = self.mol.ecp
        new_mol.build()
        self.mol = new_mol

        # Propagate MOs to new geometry (using Lowdin overlap)
        # mo_coeff is (2, NAO, NMO), overlap_half is (NAO, NAO).
        # Need to apply overlap_half to each spin component.
        self.mo_coeff[0] = self.overlap_half @ self.mo_coeff[0]
        self.mo_coeff[1] = self.overlap_half @ self.mo_coeff[1]

        if not self.with_df:
            self.mf = scf.UHF(self.mol)
        else:
            self.mf = scf.UHF(self.mol).density_fit()
            
        self.update_overlap()

        # Transform back using new overlap
        self.mo_coeff[0] = self.overlap_half_inv @ self.mo_coeff[0]
        self.mo_coeff[1] = self.overlap_half_inv @ self.mo_coeff[1]

        # Recalculate and cache the core Hamiltonian for the new geometry
        self.h_core = self.mf.get_hcore(self.mol)

        # Reset cached density and potential
        self.dm_last = 0
        self.veff_last = 0

    def energy_constraint(self, dm_total=None, fock_matrix=None):
        """
        Compute energy constraint residue for UHF (max of alpha/beta residues).
        """
        if dm_total is None:
            dm_total = self.make_rdm1(self.mo_coeff, self.mo_occ)
        if fock_matrix is None:
            fock_matrix = self.get_fock_matrix(dm_total)

        # Residue = F D S - S D F for each spin
        residue_alpha = fock_matrix[0] @ dm_total[0] @ self.overlap 
        residue_alpha -= residue_alpha.conj().T

        residue_beta = fock_matrix[1] @ dm_total[1] @ self.overlap
        residue_beta -= residue_beta.conj().T
        
        return np.max(np.abs(residue_alpha)), np.max(np.abs(residue_beta))


if __name__ == "__main__":
    # Example usage
    mol = gto.M(
        atom='''
        H 0.000000 0.000000 0.000000
        F 0.000000 0.000000 1.5
        ''',
        basis='6-31g',
        spin=0 # Explicitly specify spin if needed, UHF can still run on singlet
    )

    # Run ground state UHF
    mf = scf.UHF(mol)
    mf.convergence = 1e-12
    mf.kernel()

    # Run TDUHF simulation
    simulation_uhf = TDUHFSimulation(mf, dt=0.5, nuclei_dt=0.5, steps=200, 
                                     field_type='none', ehrenfest=True, 
                                     output_interval=10, convergence_threshold=1e-8)
    simulation_uhf.run()

    # Basic plotting
    times = np.array(simulation_uhf.times)
    total_energies = np.array(simulation_uhf.total_energies)
    
    plt.figure()
    plt.plot(times, total_energies - total_energies[0], label='Total Energy Deviation')
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Energy (Hartree)")
    plt.title("TDUHF Ehrenfest Dynamics")
    plt.legend()
    plt.show()