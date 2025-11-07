from pyscf import gto, scf, dft
import numpy as np
import matplotlib.pyplot as plt
import scipy
from TDUHF import TDUHFSimulation

def energy_tot(self, dm_total=None):
    """
    Compute the total electronic energy for RHF.
    Includes nuclear repulsion energy.
    """
    if dm_total is None:
        dm_total = self.make_rdm1(
            mo_coeff=self.mo_coeff,
            mo_occ=self.mo_occ
        )
        
    dm_total = np.asarray(dm_total)  # Ensure C-contiguous array for PySCF compatibility
    # Use PySCF's RHF energy calculation for robustness
    # mf.energy_elec(dm=dm_total.real) returns (e_elec, e_nuc_from_hf_approx)
    # We need total electronic energy + explicit nuclear repulsion
    h_core = self.mf.get_hcore(self.mol) # Needs to be current H_core if mol moves
    # Electronic energy: Tr(D * (H_core + 0.5 * V_eff))
    # V_eff = J(D) - 0.5 K(D) for RHF (using D_total)
     # PySCF get_veff(dm_total.real) for RHF returns J(D_total) - 0.5 K(D_total)
    vhf = self.mf.get_veff(self.mol, dm_total.real) # V_eff for RHF
    #vhf =self.veff 

    e1 = np.einsum('ij,xji->', h_core.real, dm_total.real)
    ecoul = vhf.ecoul.real
    exc = vhf.exc.real
    e2 = ecoul + exc
        
    electronic_energy = (e1 + e2).real

    return electronic_energy + self.mol.energy_nuc()  # Add nuclear repulsion energy



def update_mol(self, positions=None):
    """Updates the PySCF molecule object with current nuclear positions and rebuilds mf."""
        # Create a new molecule object with the updated geometry
        # This is safer than trying to update self.mol._atom in place for PySCF.

    if positions is None:
        positions = self.positions

    atom_coords_str = []
    for i in range(self.mol.natm):
        symbol = self.mol.atom_symbol(i)
        x, y, z = positions[i]
        atom_coords_str.append(f"{symbol} {x} {y} {z}")
        
    new_mol = gto.M(atom="\n".join(atom_coords_str),
                    basis=self.mol.basis,
                    unit="Bohr",
                    charge=self.mol.charge,
                    spin=self.mol.spin, # For RHF, spin should be 0 (singlet)
                    cart=self.mol.cart) # Preserve cartesian/spherical setting
        
    new_mol.build()
    self.previous_mol = self.mol.copy()
    self.mol = new_mol

    self.xc = self.mf.xc  # Preserve the exchange-correlation functional
    self.mf = dft.UKS(self.mol, xc=self.xc)  # Reinitialize with the new molecule
    self.mo_coeff[0] = self.overlap_half @self.mo_coeff[0]  # Preserve MOs
    self.mo_coeff[1] = self.overlap_half @self.mo_coeff[1]  # Preserve MOs
    self.update_overlap() # Update overlap matrix S and its inverses S^-1, S^-1/2
    self.mo_coeff[0] = self.overlap_half_inv @ self.mo_coeff[0]  # Preserve MOs
    self.mo_coeff[1] = self.overlap_half_inv @ self.mo_coeff[1]  # Preserve MOs

    self.h_core = self.mf.get_hcore(self.mol)

    # Reset cached density and potential since geometry has changed
    self.dm_last = 0
    self.veff_last = 0
    # Important: If mf had specific convergence, DIIS settings, etc., they are reset here.
    # For robust Ehrenfest, one might need to carry over more from the original mf setup
    # or re-converge SCF if that's part of the model (not typical for pure Ehrenfest propagation).
    # For this code, we assume propagation relies on the MOs, and mf is mainly for integrals.


def apply_scf(self):

    """
    Apply the SCF procedure to the mean-field object.

    Returns:
        float: Total energy after SCF.
    """
    from MOM import mom_occ_
    dm = self.make_rdm1(self.mo_coeff, self.mo_occ).real 

    rhf = self.mf.__class__(self.mol, xc=self.mf.xc)
    rhf.verbose = 4
    rhf = mom_occ_(rhf, occorb=self.mo_coeff, setocc=self.mo_occ)
    # rhf.max_cycle = 5
    rhf.scf(dm)
    self.mf = rhf
    self.initialize_molecular_orbitals(n_virtual=0)
    return self.energy_tot()


class TDUKSSimulation(TDUHFSimulation):

    __init__ = TDUHFSimulation.__init__

    energy_tot = energy_tot
    update_mol = update_mol
    apply_scf = apply_scf

    
    

    




if __name__ == "__main__":
    mol = gto.M(
        atom='''
        H 0.000000 0.000000 0.000000
        F 0.000000 0.000000 1.500000
        ''',
        basis='6-31g',
        unit="Bohr"
    )


    mf = dft.UKS(mol, xc='lda')
    mf.convergence = 1e-16
    mf.kernel()

    mf_grad = mf.nuc_grad_method()
    mf_grad.kernel()


    mf.mo_occ[0,4] = 0
    mf.mo_occ[0,5] = 1



    simulation_uhf = TDUKSSimulation(mf, dt=1, nuclei_dt=1,  steps=1500, field_type='none',  ehrenfest=False ,  output_interval = -1)

    simulation_uhf.run()


    plt.plot(simulation_uhf.potential_energies - simulation_uhf.potential_energies[0], label='Potential Energy')
    #plt.plot(simulation_uhf.kinetic_energies-simulation_uhf.kinetic_energies[0], label='Kinetic Energy')
    #plt.plot(simulation_uhf.total_energies-simulation_uhf.total_energies[0], label='Total Energy')
    plt.legend()

    plt.show()
