import os
import numpy as np
from pyscf import gto, dft

# --- Assumed custom library imports ---
# These modules were in your original script. Make sure they are available in your Python environment.
try:
    from tdmash.MASH import MASH
    from tdmash.MOM import mom_occ_
    from tdmash.TDRKS import TDRKSSimulation
except ImportError as e:
    print(f"Warning: Could not import custom modules (MASH, MOM, TDRKS). {e}")
    print("The script will fail if these are not in your PYTHONPATH.")
    # Define dummy classes/functions to allow the script to be parsed without errors
    class MASH: pass
    class TDRKSSimulation: pass
    def mom_occ_(mf, mo, occ): return mf

# --- Helper function to read data ---

def coords_to_atom_string(atom_symbols, coords):
    """Combines atom symbols and coordinates into a string for PySCF."""
    return "\n".join([f"{symbol} {x} {y} {z}" for symbol, (x, y, z) in zip(atom_symbols, coords)])

# --- Core Simulation Logic ---

def run_simulation(atom_string, initial_velocity, initial_spin, output_dir):
    """
    Runs a single MASH dynamics simulation with the given initial conditions.
    This function contains the core logic from your original script.
    """
    print(f"--- Starting Simulation ---")

    # Simulation parameters
    dt = 0.5  # Time step in atomic units
    steps = 1600

    # Build the molecule object
    mol = gto.M(atom=atom_string, basis='6-31g*', unit="Bohr")
    mol.build()

    # --- Ground State SCF ---
    mf1 = dft.RKS(mol, xc='lda')
    mf1.conv_tol = 1e-12
    mf1.kernel()

    # --- Excited State SCF using MOM ---
    mf2 = dft.RKS(mol, xc='lda')
    mf2.conv_tol = 1e-12

    # Define occupation for the excited state (HOMO -> LUMO)
    occ = np.zeros((2, mf1.mo_coeff.shape[1]))
    occ[0, mf1.mo_occ > 0] = 1  # Alpha spin occupations
    occ[1, mf1.mo_occ > 0] = 1  # Beta spin occupations
    
    # Create the single excitation
    homo_index = np.where(mf1.mo_occ > 0)[0][-1]
    lumo_index = homo_index + 1
    occ[0, homo_index] = 0
    occ[0, lumo_index] = 1

    # Apply MOM to find the excited state
    mf2 = mom_occ_(mf2, mf1.mo_coeff.copy(), occ)
    mf2.max_cycle = 500
    mf2.kernel()

    # --- Setup TDRKS Simulations for MASH ---
    sim1 = TDRKSSimulation(mf1, dt=dt, nuclei_dt=dt, steps=600, field_type='none', ehrenfest=False, output_interval=-1, convergence_threshold=1e-5, mu=0.5, imaginary_time=True)
    sim2 = TDRKSSimulation(mf2, dt=dt, nuclei_dt=dt, steps=600, field_type='none', ehrenfest=False, output_interval=-1, convergence_threshold=1e-5, mu=0.5, imaginary_time=True)

    # --- Initialize and run the MASH simulation ---
    energy_output_path = os.path.join(output_dir, "mash_output.npz")
    
    mash_simulation = MASH(
        sim1=sim1,
        sim2=sim2,
        dt=dt,
        nuclei_dt=dt,
        steps=steps,
        output_interval=1,
        mu_1=1,
        mu_2=1,
        energy_filename=energy_output_path,
        initial_velocity=initial_velocity,
        apply_scf=False,
        initial_spin=initial_spin
    )
    mash_simulation.run()

    # --- Save final results and plot ---
    final_data_path = os.path.join(output_dir, "final_state.npz")
    np.savez(
        final_data_path,
        mo_coeff2=mash_simulation.sim2.mo_coeff,
        mo_coeff1=mash_simulation.sim1.mo_coeff,
        positions=mash_simulation.sim1.mol.atom_coords()
    )
    
    mash_simulation.plot_results()
    
    print(f"--- Finished Simulation ---")
    print(f"Results saved in: {output_dir}\n")

# --- Main Execution Block ---

if __name__ == "__main__":
    # --- Configuration ---
    # This script is designed to be run inside a specific simulation folder.
    # It reads its inputs from the current directory.
    ATOM_SYMBOLS = ['N', 'N', 'H', 'H']
    
    try:
        # Load the data for this specific run
        current_coords = np.load("coords.npy")
        current_velocities = np.load("velocities.npy")
        current_spin = np.load("spin.npy").tolist() # Convert to list for MASH
        
        # Convert coordinates to the format PySCF needs
        atom_string = coords_to_atom_string(ATOM_SYMBOLS, current_coords)

        # Run the simulation, saving output in the current directory (".")
        run_simulation(atom_string, current_velocities, current_spin, ".")

    except FileNotFoundError as e:
        print("Error: Could not find input files (coords.npy, velocities.npy, spin.npy).")
        print("Make sure you are running this script from within a prepared simulation folder.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during the simulation.")
        print(f"Details: {e}")
