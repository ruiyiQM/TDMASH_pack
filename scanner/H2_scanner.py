import numpy as np
from pyscf import gto, scf, dft
import matplotlib.pyplot as plt

def scan_pes(distances, basis='sto-3g', method='RHF', xc_functional=None):
    """
    Scans the Potential Energy Surface (PES) for the H2 molecule.

    This function iterates through a list of bond distances, and for each
    distance, it sets up an H2 molecule, runs a quantum chemistry calculation
    (either RHF or RKS), and stores the resulting total energy.

    Args:
        distances (list or np.ndarray): A list of H-H bond distances in Angstroms.
        basis (str): The basis set to use for the calculation (e.g., 'sto-3g', '6-31g').
        method (str): The calculation method, either 'RHF' or 'RKS'.
        xc_functional (str, optional): The exchange-correlation functional to use
                                       if the method is 'RKS' (e.g., 'pbe0').
                                       Defaults to None.

    Returns:
        tuple: A tuple containing two numpy arrays:
               - The bond distances used for the scan.
               - The calculated total energies in Hartrees.
    """
    energies = []
    print(f"\nStarting PES scan for {method} with basis {basis}...")
    if method.upper() == 'RKS':
        print(f"Using XC functional: {xc_functional}")

    for d in distances:
        # Define the H2 molecule with the current bond distance.
        # The first atom is at the origin, the second is along the z-axis.
        mol = gto.M(
            atom=f'N 0 0 0; N 0 0 {d}',
            basis=basis,
            unit='angstrom'
        )

        # Select the calculation method based on the input
        if method.upper() == 'RHF':
            mf = scf.RHF(mol)
        elif method.upper() == 'RKS':
            if xc_functional is None:
                raise ValueError("An xc_functional must be provided for the RKS method.")
            mf = dft.RKS(mol)
            mf.xc = xc_functional
        else:
            raise ValueError(f"Method '{method}' is not supported. Use 'RHF' or 'RKS'.")

        # Run the Self-Consistent Field (SCF) calculation.
        # We set verbose to 0 to keep the output clean during the scan.
        mf.verbose = 0
        energy = mf.kernel()
        energies.append(energy)
        print(f"  Distance: {d:.3f} Å, Total Energy: {energy:.6f} Hartree")

    print("Scan complete.")
    return np.array(distances), np.array(energies)

if __name__ == '__main__':
    # Define the range of bond distances to scan (in Angstroms).
    # np.linspace creates an array of evenly spaced numbers over a specified interval.
    bond_distances = np.linspace(0.5, 6.5, 60) # 60 points from 0.5 to 3.5 Å

    # --- Run the RHF Scan ---
    rhf_distances, rhf_energies = scan_pes(bond_distances, basis='6-31g', method='RHF')

    # --- Run the RKS Scan using the PBE0 functional ---
    rks_distances, rks_energies = scan_pes(bond_distances, basis='6-31g', method='RKS', xc_functional='pbe0')

    # --- Plotting the results using Matplotlib ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))

    # Plot RHF results
    plt.plot(rhf_distances, rhf_energies, 'o-', label='RHF/6-31g', markersize=4)

    # Plot RKS results
    plt.plot(rks_distances, rks_energies, 's--', label='RKS(PBE0)/6-31g', markersize=4)

    # Find and mark the minimum energy point for the RHF curve
    min_rhf_energy = np.min(rhf_energies)
    min_rhf_dist = rhf_distances[np.argmin(rhf_energies)]
    plt.axvline(x=min_rhf_dist, color='gray', linestyle=':', label=f'RHF min at {min_rhf_dist:.3f} Å')


    # Adding titles and labels for clarity
    plt.title('H₂ Potential Energy Surface', fontsize=16)
    plt.xlabel('Bond Distance (Å)', fontsize=12)
    plt.ylabel('Total Energy (Hartree)', fontsize=12)
    plt.legend(fontsize=10)

    # Display the plot
    plt.tight_layout()
    plt.show()
