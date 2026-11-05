from pyscf import gto, scf, dft
import numpy as np
import matplotlib.pyplot as plt
import scipy
from .TDRHF import TDRHFSimulation
from .MOM import singlet_occ_

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
        
    # Use PySCF's RHF energy calculation for robustness
    # mf.energy_elec(dm=dm_total.real) returns (e_elec, e_nuc_from_hf_approx)
    # We need total electronic energy + explicit nuclear repulsion
    h_core = self.mf.get_hcore(self.mol) # Needs to be current H_core if mol moves
    # Electronic energy: Tr(D * (H_core + 0.5 * V_eff))
    # V_eff = J(D) - 0.5 K(D) for RHF (using D_total)
     # PySCF get_veff(dm_total.real) for RHF returns J(D_total) - 0.5 K(D_total)
    vhf = self.mf.get_veff(self.mol, dm_total.real) # V_eff for RHF
    #vhf =self.veff 

    e1 = np.einsum('ij,ji->', h_core.real, dm_total.real)
    ecoul = vhf.ecoul.real
    exc = vhf.exc.real
    e2 = ecoul + exc
        
    electronic_energy = (e1 + e2).real

    return electronic_energy + self.mol.energy_nuc()  # Add nuclear repulsion energy



def apply_scf(self):

        """
        Apply the SCF procedure to the mean-field object.

        Returns:
            float: Total energy after SCF.
        """
        from .MOM import mom_occ_

        if (self.mo_occ is not None) and (np.sum(self.mo_occ == 1) > 0):
            rhf = dft.RKS(self.mol, xc = self.mf.xc)
            rhf = singlet_occ_(rhf)
        else:
            rhf = dft.RKS(self.mol, xc = self.mf.xc).newton()
        rhf.verbose = 4

        rhf.kernel()

        self.mf = rhf
        self.initialize_molecular_orbitals(n_virtual=0)
        return self.energy_tot()
    

class TDRKSSimulation(TDRHFSimulation):

    __init__ = TDRHFSimulation.__init__

    energy_tot = energy_tot
    update_mol = TDRHFSimulation.update_mol
    apply_scf = apply_scf

    
    

    




if __name__ == "__main__":
    mol = gto.M(
        atom='''
        H 0.000000 0.000000 0.000000
        F 0.000000 0.000000 0.78000
        ''',
        basis='6-31g',
        
        
    )


    mf = dft.RKS(mol, xc='lda')
    mf.convergence = 1e-16
    
    mf.kernel()

    mf_grad = mf.nuc_grad_method()
    mf_grad.kernel()


    #mf.mo_occ[4] = 1
    #mf.mo_occ[6] = 1



    simulation_uhf = TDRKSSimulation(mf, dt=1, nuclei_dt=1,  steps=1000, field_type='none',  ehrenfest=True,  output_interval = -1,  convergence_threshold=1e-4)

    simulation_uhf.run()

    #simulation_uhf.dipoles = np.array(simulation_uhf.dipoles)
    #plt.plot( simulation_uhf.dipoles[:,2], label='Dipole X')


    #plt.plot(simulation_uhf.potential_energies - simulation_uhf.potential_energies[0], label='Potential Energy')
    #plt.plot(simulation_uhf.kinetic_energies-simulation_uhf.kinetic_energies[0], label='Kinetic Energy')
    plt.plot(simulation_uhf.total_energies-simulation_uhf.total_energies[0], label='Total Energy')
    plt.legend()

    plt.show()
