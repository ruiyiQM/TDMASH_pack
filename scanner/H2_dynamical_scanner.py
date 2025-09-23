import numpy as np
from pyscf import gto, scf, dft
import matplotlib.pyplot as plt

# Import the simulation classes from your existing files
from TDRHF import TDRHFSimulation
from TDRKS import TDRKSSimulation

def run_pseudo_dynamics_scan(sim_class, initial_mf, start_dist, end_dist, num_steps, basis):
    """
    Performs a pseudo-dynamical scan by slowly changing the geometry.

    This function simulates the stretching of the H2 bond by manually updating
    the nuclear positions over a series of steps. At each step, it uses the
    provided time-dependent simulation class to calculate the total energy.

    Args:
        sim_class (class): The simulation class to use (TDRHFSimulation or TDRKSSimulation).
        initial_mf (pyscf.scf.hf.RHF): A converged PySCF mean-field object for the initial geometry.
        start_dist (float): The initial H-H bond distance in Angstroms.
        end_dist (float): The final H-H bond distance in Angstroms.
        num_steps (int): The number of steps for the scan.
        basis (str): The basis set used for the calculation.

    Returns:
        tuple: A tuple containing two numpy arrays:
               - The bond distances used for the scan.
               - The calculated total energies in Hartrees.
    """
    # Initialize the simulation object. We set dt and steps to dummy values
    # as we will control the propagation manually.
    # 'ehrenfest=False' ensures the nuclei don't move based on calculated forces.
    sim = sim_class(initial_mf, dt=1.0, steps=1, ehrenfest=False,imaginary_time= True )

    # Generate the list of bond distances for our trajectory
    distances = np.linspace(start_dist, end_dist, num_steps)
    energies = []

    method_name = "RHF" if sim_class == TDRHFSimulation else f"RKS({initial_mf.xc})"
    print(f"\nStarting pseudo-dynamics scan for {method_name} with basis {basis}...")

    # Loop through each distance in our pre-defined path
    for i, dist in enumerate(distances):
        # Define the new atomic positions for the current step.
        # Atom 1 is at (-d/2), Atom 2 is at (d/2) along the z-axis.
        # The coordinates are in Bohr units for PySCF's internal processing.
        angstrom_to_bohr = 1.88973
        half_dist_bohr = (dist / 2.0) * angstrom_to_bohr
        new_positions = np.array([
            [0.0, 0.0, -half_dist_bohr],
            [0.0, 0.0,  half_dist_bohr]
        ])

        # Update the simulation object with the new molecular geometry.
        # This method recalculates necessary integrals for the new positions.
        sim.update_mol(positions=new_positions)

        # Propagate the electronic wavefunction for one small step and get the energy.
        # Since no external field is applied, this gives us the ground state
        # energy at the current geometry.
        current_energy = sim.update_electronic_wavefunction(calculate_force=False)
        energies.append(current_energy)

        # Print progress
        if (i + 1) % 10 == 0 or i == num_steps - 1:
            print(f"  Step {i+1}/{num_steps}: Distance = {dist:.3f} Å, Energy = {current_energy:.6f} Hartree")

    print("Scan complete.")
    return np.array(distances), np.array(energies)

if __name__ == '__main__':
    # --- Simulation Parameters ---
    start_distance = 0.5 # Starting bond distance in Angstroms
    end_distance = 6.5    # Ending bond distance in Angstroms
    scan_steps = 100     # Number of geometry steps
    basis_set = '6-31g'   # Basis set

    # --- Initial Setup ---
    # We need to create an initial molecule and run a standard SCF calculation
    # to get the starting molecular orbitals for our time propagation.
    print("Performing initial SCF calculation...")
    initial_mol = gto.M(
        atom=f'N 0 0 {-start_distance/2}; N 0 0 {start_distance/2}',
        basis=basis_set,
        unit='angstrom'
    )

    # Initial RHF calculation
    mf_rhf = scf.RHF(initial_mol).run()

    # Initial RKS calculation
    mf_rks = dft.RKS(initial_mol)
    mf_rks.xc = 'pbe0'
    mf_rks.run()
    print("Initial calculation complete.")

    # --- Run the Pseudo-Dynamical Scans ---
    # Run the scan using the TDRHF class
    rhf_distances, rhf_energies = run_pseudo_dynamics_scan(
        TDRHFSimulation, mf_rhf, start_distance, end_distance, scan_steps, basis_set
    )

    # Run the scan using the TDRKS class
    rks_distances, rks_energies = run_pseudo_dynamics_scan(
        TDRKSSimulation, mf_rks, start_distance, end_distance, scan_steps, basis_set
    )

    # --- Plotting the Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))

    # Plot the PES curves from the pseudo-dynamics
    plt.plot(rhf_distances, rhf_energies, 'o-', label=f'TDRHF/{basis_set}', markersize=4)
    plt.plot(rks_distances, rks_energies, 's--', label=f'TDRKS({mf_rks.xc})/{basis_set}', markersize=4)

    # Adding titles and labels
    plt.title('H₂ PES via Pseudo-Dynamical Scan', fontsize=16)
    plt.xlabel('Bond Distance (Å)', fontsize=12)
    plt.ylabel('Total Energy (Hartree)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
