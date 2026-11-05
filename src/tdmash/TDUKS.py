from pyscf import gto, scf, dft
import numpy as np
import matplotlib.pyplot as plt
import scipy
from .TDUHF import TDUHFSimulation

def energy_tot(self, dm_total=None):
    """Return the unrestricted KS energy, including complex exchange density."""
    if dm_total is None:
        dm_total = self.make_rdm1(
            mo_coeff=self.mo_coeff,
            mo_occ=self.mo_occ
        )
        
    dm_total = np.asarray(dm_total)
    h_core = self.mf.get_hcore(self.mol)
    vhf = self.mf.get_veff(self.mol, dm_total)

    e1 = np.einsum('ij,xji->', h_core.real, dm_total.real)
    ecoul = vhf.ecoul.real
    exc = vhf.exc.real
    e2 = ecoul + exc
        
    electronic_energy = (e1 + e2).real

    return electronic_energy + self.mol.energy_nuc()  # Add nuclear repulsion energy



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
    from .MOM import mom_occ_
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
    update_mol = TDUHFSimulation.update_mol
    apply_scf = apply_scf

    
    

    




if __name__ == "__main__":
    mol = gto.M(
        atom='''
        H 0.000000 0.000000 0.000000
        F 0.000000 0.000000 0.800000
        ''',
        basis='6-31g',
        unit="Bohr"
    )


    mf = dft.UKS(mol, xc='lda')
    mf.convergence = 1e-16
    mf.kernel()

    mf_grad = mf.nuc_grad_method()
    mf_grad.kernel()


    #mf.mo_occ[0,4] = 0
    #mf.mo_occ[0,5] = 1



    simulation_uhf = TDUKSSimulation(mf, dt=1, nuclei_dt=1,  steps=1500, field_type='none',  ehrenfest=False ,  output_interval = -1)

    simulation_uhf.run()


    plt.plot(simulation_uhf.potential_energies - simulation_uhf.potential_energies[0], label='Potential Energy')
    #plt.plot(simulation_uhf.kinetic_energies-simulation_uhf.kinetic_energies[0], label='Kinetic Energy')
    #plt.plot(simulation_uhf.total_energies-simulation_uhf.total_energies[0], label='Total Energy')
    plt.legend()

    plt.show()
