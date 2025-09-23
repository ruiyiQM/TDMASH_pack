import numpy 
from functools import reduce
from pyscf import gto, scf, lib

def mom_occ_(mf, occorb, setocc):
    '''Use maximum overlap method to determine occupation number for each orbital in every
    iteration.'''
    from pyscf.lib import logger
    import numpy 
    log = logger.Logger(mf.stdout, mf.verbose)

    if occorb.ndim == 2:
        coef_occ_a = occorb[:, setocc[0]>0]
        coef_occ_b = occorb[:, setocc[1]>0]
    elif occorb.ndim == 3:
        coef_occ_a = occorb[0][:, setocc[0]>0]
        coef_occ_b = occorb[1][:, setocc[1]>0]

    def get_occ(mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = mf.mo_energy
        if mo_coeff is None: mo_coeff = mf.mo_coeff

        if occorb.ndim == 2:
            mo_coeff = numpy.array([mo_coeff, mo_coeff])
        mo_occ = numpy.zeros((2,mo_coeff.shape[1]))
        nocc_a = int(numpy.sum(setocc[0]))
        nocc_b = int(numpy.sum(setocc[1]))

        s_a = reduce(numpy.dot, (coef_occ_a.conj().T, mf.get_ovlp(), mo_coeff[0]))
        s_b = reduce(numpy.dot, (coef_occ_b.conj().T, mf.get_ovlp(), mo_coeff[1]))
            #choose a subset of mo_coeff, which maximizes <old|now>
        #print (s_a.shape)
        idx_a = numpy.argsort(numpy.einsum('ij,ij->j', s_a, s_a.conj()))[::-1]
        
        idx_b = numpy.argsort(numpy.einsum('ij,ij->j', s_b, s_b.conj()))[::-1]

        mo_occ[0,idx_a[:nocc_a]] = 1.
        mo_occ[1,idx_b[:nocc_b]] = 1.


        log.debug(' New alpha occ pattern: %s', mo_occ[0])
        log.debug(' New beta occ pattern: %s', mo_occ[1])
        if occorb.ndim == 2:
            mo_occ = mo_occ[0, :] + mo_occ[1, :]
        return mo_occ


    mf.get_occ = get_occ
    return mf


def mom_occ_closed_shell_(mf, occorb, setocc):
    '''Use maximum overlap method to determine occupation number for each orbital in every
    iteration.'''
    from pyscf.lib import logger
    import numpy 
    log = logger.Logger(mf.stdout, mf.verbose)



    def get_occ(mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = mf.mo_energy
        if mo_coeff is None: mo_coeff = mf.mo_coeff

        mo_occ = numpy.zeros((mo_coeff.shape[1]))
        nocc = sum(setocc ==2)

        overlap = reduce(numpy.dot, (occorb[:,setocc==2].conj().T, mf.get_ovlp(), mo_coeff))

        #choose a subset of mo_coeff, which maximizes <old|now>
        #print (s_a.shape)
        idx = numpy.argsort(numpy.einsum('ij,ij->j', overlap, overlap.conj()))[::-1]
        

        mo_occ[idx[:nocc]] = 2.

        log.debug(' New occ pattern: %s', mo_occ)
        print (mo_occ)

        return mo_occ


    mf.get_occ = get_occ

    return mf

def singlet_occ_(mf):
    '''Use maximum overlap method to determine occupation number for each orbital in every
    iteration.'''
    from pyscf.lib import logger
    import numpy 
    log = logger.Logger(mf.stdout, mf.verbose)


    def get_occ(mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = mf.mo_energy
        if mo_coeff is None: mo_coeff = mf.mo_coeff

        nelec = mf.mol.nelec[0] + mf.mol.nelec[1]

        if nelec%2 == 0:
            mo_occ =  numpy.zeros((mo_coeff.shape[1]))

            nocc = nelec // 2
            mo_occ[:nocc-1] = 2.0
            mo_occ[nocc-1] = 1.0   
            mo_occ[nocc] = 1.0
        return mo_occ

    mf.get_occ = get_occ
    return mf
# --- Example: Calculate a core-excited state of O2 ---

if __name__ == '__main__':

    import numpy as np
    from pyscf import gto, dft
    # 1. Define the molecule
    mol = gto.Mole()
    mol.atom = '''
    H    0.7493682    0.0000000    0.4424329
    O    0.0000000    0.0000000   -0.1653507
    H   -0.7493682    0.0000000    0.4424329
    '''
    mol.basis = '6-31g*'
    mol.spin = 0  # Triplet ground state
    mol.build(verbose=4)

    # 2. First, calculate the ground state to get a set of reference orbitals
    # We use UHF (Unrestricted Hartree-Fock)
    mf_gs = dft.RKS(mol,xc='pbe')
    print("--- Running Ground State Calculation ---")
    mf_gs.kernel()
    print(f"Ground state converged: {mf_gs.converged}")
    print(f"Ground state energy: {mf_gs.e_tot}\n")

    # 3. Define the occupation for the desired *excited state*.
    # In the ground state of O2 (triplet), the occupations are:
    # Alpha: [1,1,1,1,1,1,1,1,0,0,0,0,0,0] (8 electrons)
    # Beta:  [1,1,1,1,1,1,0,0,0,0,0,0,0,0] (6 electrons)
    # Let's target an excited state by moving a beta electron from HOMO to LUMO.
    # The ground state HOMO is orbital 5, LUMO is orbital 6 for beta spin.
    
    occ_gs = mf_gs.mo_occ

    occ_excited = np.zeros((2, occ_gs.shape[0]))
    
    occ_excited[:,occ_gs==2]= 1 
    

    # Create the excited state occupation pattern
    occ_excited[0,4] = 0.
    occ_excited[0,6] = 1.
    
    print("--- Preparing Excited State Calculation (with MOM) ---")
    print("Target alpha occupations:", occ_excited[0])
    print("Target beta occupations: ", occ_excited[1])
    
    # 4. Set up a new SCF calculation for the excited state
    mf_excited = dft.RKS(mol,xc='pbe')

    # Use the ground state orbitals as the initial guess to get the right "character"
    mf_excited.init_guess = mf_gs.mo_coeff

    print (occ_excited)
    # 5. Apply the mom_occ_ function to the new SCF object.
    # We give it the ground state MOs (`mf_gs.mo_coeff`) as the reference orbitals
    # and the excited occupation pattern (`occ_excited`) as the target.
    mf_excited = mom_occ_(mf_excited, occorb=mf_gs.mo_coeff.copy(), setocc=occ_excited)
    
    # 6. Run the MOM-guided SCF calculation
    # We add `conv_tol=1e-7` for a tighter convergence threshold.
    print("\n--- Running MOM-guided Excited State Calculation ---")

    energy_excited = mf_excited.kernel()

    print("\n--- Results ---")
    print(f"Excited state converged: {mf_excited.converged}")
    print(f"Final excited state energy: {energy_excited}")
    print(f"Ground state energy was:  {mf_gs.e_tot}")


    
