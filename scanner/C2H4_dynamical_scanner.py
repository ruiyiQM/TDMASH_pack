import numpy as np
from pyscf import gto, scf, dft
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Import the simulation classes from your existing files
# Make sure TDRHF.py and TDRKS.py are in the same directory
from TDRHF import TDRHFSimulation
from TDRKS import TDRKSSimulation

def get_ethylene_geometry(angle_deg=0.0):
    """
    Generates the atomic coordinates for ethylene at a given C=C torsion angle.

    Args:
        angle_deg (float): The torsion angle in degrees for rotating one CH2 group.

    Returns:
        list: A list of tuples, where each tuple contains the atom symbol and its
              (x, y, z) coordinates in Angstroms.
    """
    # Standard literature values for ethylene geometry
    d_CC = 1.339  # Angstroms
    d_CH = 1.087  # Angstroms
    a_HCC = 121.3  # Degrees

    # Convert angles to radians for numpy calculations
    angle_rad = np.deg2rad(angle_deg)
    hcc_rad = np.deg2rad(a_HCC)

    # --- Define the initial planar geometry (in the xy-plane, C=C along x-axis) ---
    # Carbon atoms
    c1_pos = np.array([-d_CC / 2, 0.0, 0.0])
    c2_pos = np.array([ d_CC / 2, 0.0, 0.0])

    # Hydrogens attached to the first carbon (C1) - these will not rotate
    h1_pos = c1_pos + np.array([-d_CH * np.cos(np.pi - hcc_rad),  d_CH * np.sin(np.pi - hcc_rad), 0.0])
    h2_pos = c1_pos + np.array([-d_CH * np.cos(np.pi - hcc_rad), -d_CH * np.sin(np.pi - hcc_rad), 0.0])

    # Hydrogens attached to the second carbon (C2) - these will be rotated
    h3_initial_pos = c2_pos + np.array([d_CH * np.cos(np.pi - hcc_rad),  d_CH * np.sin(np.pi - hcc_rad), 0.0])
    h4_initial_pos = c2_pos + np.array([d_CH * np.cos(np.pi - hcc_rad), -d_CH * np.sin(np.pi - hcc_rad), 0.0])

    # --- Rotate the second CH2 group around the C=C bond (x-axis) ---
    # Create a rotation object for rotation around the x-axis
    rotation = R.from_rotvec(angle_rad * np.array([1, 0, 0]))

    # Translate the H atoms to be rotated so their carbon is at the origin,
    # then rotate, and translate back.
    h3_pos = rotation.apply(h3_initial_pos - c2_pos) + c2_pos
    h4_pos = rotation.apply(h4_initial_pos - c2_pos) + c2_pos

    # Combine all atoms into a list for PySCF
    atoms = [
        ('C', c1_pos), ('C', c2_pos),
        ('H', h1_pos), ('H', h2_pos),
        ('H', h3_pos), ('H', h4_pos)
    ]
    return atoms

def run_rotational_scan(sim_class, initial_mf, start_angle, end_angle, num_steps, basis):
    """
    Performs a pseudo-dynamical scan by slowly rotating one CH2 group of ethylene.
    """
    # Initialize the simulation object. dt and steps are dummy values.
    sim = sim_class(initial_mf, dt=1.0, steps=1, ehrenfest=False)

    # Generate the list of angles for our trajectory
    angles = np.linspace(start_angle, end_angle, num_steps)
    energies = []
    angstrom_to_bohr = 1.8897261246257702

    method_name = "RHF" if sim_class == TDRHFSimulation else f"RKS({initial_mf.xc})"
    print(f"\nStarting rotational scan for {method_name} with basis {basis}...")

    for i, angle in enumerate(angles):
        # Get the geometry for the current angle and convert to Bohr
        geometry_ang = get_ethylene_geometry(angle)
        positions_bohr = np.array([pos for _, pos in geometry_ang]) * angstrom_to_bohr

        # Update the simulation with the new molecular geometry
        sim.update_mol(positions=positions_bohr)

        # Calculate the energy at the current geometry
        current_energy = sim.update_electronic_wavefunction(calculate_force=False)
        energies.append(current_energy)

        # Print progress
        if (i + 1) % 10 == 0 or i == num_steps - 1:
            print(f"  Step {i+1}/{num_steps}: Angle = {angle:5.1f}°, Energy = {current_energy:.6f} Hartree")

    print("Scan complete.")
    return np.array(angles), np.array(energies)

if __name__ == '__main__':
    # --- Simulation Parameters ---
    start_angle_deg = 6
    end_angle_deg = 120
    scan_steps = 120 # e.g., steps of 2 degrees
    basis_set = '6-31g'  # Using a smaller basis for speed

    # --- Initial Setup ---
    # Create the initial planar molecule (angle=0) for the first SCF calculation.
    print("Performing initial SCF calculation for planar ethylene...")
    initial_geometry_list = get_ethylene_geometry(angle_deg=0)
    # Format the geometry for PySCF input
    pyscf_atom_string = "".join([f"{symbol} {x:.8f} {y:.8f} {z:.8f};" for symbol, (x, y, z) in initial_geometry_list])

    initial_mol = gto.M(
        atom=pyscf_atom_string,
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

    # --- Run the Rotational Scans ---
    rhf_angles, rhf_energies = run_rotational_scan(
        TDRHFSimulation, mf_rhf, start_angle_deg, end_angle_deg, scan_steps, basis_set
    )
    #rks_angles, rks_energies = run_rotational_scan(
    #    TDRKSSimulation, mf_rks, start_angle_deg, end_angle_deg, scan_steps, basis_set
    #)

    # --- Plotting the Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))

    # Convert energies from Hartree to kcal/mol relative to the minimum energy
    kcal_per_mol_conversion = 627.509
    rhf_energies_rel = (rhf_energies - np.min(rhf_energies)) * kcal_per_mol_conversion
    #rks_energies_rel = (rks_energies - np.min(rks_energies)) * kcal_per_mol_conversion

    plt.plot(rhf_angles, rhf_energies_rel, 'o-', label=f'TDRHF/{basis_set}', markersize=4)
    #plt.plot(rks_angles, rks_energies_rel, 's--', label=f'TDRKS({mf_rks.xc})/{basis_set}', markersize=4)

    # Find and show the energy barrier at 90 degrees
    #barrier_idx = np.abs(rks_angles - 90.0).argmin()
    #barrier_height = rks_energies_rel[barrier_idx]
    #plt.axvline(x=90, color='gray', linestyle=':', label=f'Barrier at 90° ({barrier_height:.1f} kcal/mol)')


    plt.title('C₂H₄ Rotational Energy Barrier', fontsize=16)
    plt.xlabel('Torsion Angle (Degrees)', fontsize=12)
    plt.ylabel('Relative Energy (kcal/mol)', fontsize=12)
    plt.xticks(np.arange(start_angle_deg, end_angle_deg + 1, 30))
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
