import os
import numpy as np
import datetime
import shutil
import sys

# --- Main Execution Block ---

if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Update these symbols to match the atoms in your coordinates file.
    # The order must be the same as the order of atoms in the .npy array.
    ATOM_SYMBOLS = ['N', 'N', 'H', 'H']
    
    # Define paths for the input numpy files
    initial_conditions_dir = "initial_conditions"
    initial_conditions_dir = "./"
    coords_file = os.path.join(initial_conditions_dir, "wigner_coords_100.npy")
    velocities_file = os.path.join(initial_conditions_dir, "wigner_velocities_100.npy")
    spin_file = os.path.join(initial_conditions_dir, "wigner_spin_100.npy")
    
    # Name of the script that will be copied into each simulation folder
    simulation_script_name = "run_single.py"
    # Name for the shared folder where all results will be saved
    shared_results_folder = "all_results"

    # --- Script Start ---
    # Create a unique parent directory for this batch of simulations using a timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_base_dir = f"simulation_batch_{timestamp}"
    print(f"Preparing simulation batch in base directory: '{output_base_dir}'")

    try:
        # Load all the initial condition data at once
        all_coords = np.load(coords_file)
        all_velocities = np.load(velocities_file)
        all_spins = np.load(spin_file)['spin']

        num_structures = all_coords.shape[0]
        num_structures = 2 
        print(f"Found {num_structures} structures to prepare.")


        if len(ATOM_SYMBOLS) != all_coords.shape[1]:
            print(f"Error: The number of atom symbols ({len(ATOM_SYMBOLS)}) does not match the number of atoms in the coordinate file ({all_coords.shape[1]}).")
            sys.exit(1)
        if not os.path.exists(simulation_script_name):
            print(f"Error: The simulation script '{simulation_script_name}' was not found in the current directory.")
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: Could not find an input file. Make sure all required files exist.")
        print(f"Details: {e}")
        sys.exit(1)
    except KeyError:
        print(f"Error: Could not find the default key 'arr_0' in '{spin_file}'.")
        print("Please ensure the .npz file was saved correctly, for example using 'np.savez(filename, array)'.")
        sys.exit(1)

    # Create the base directory and the shared results folder
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, shared_results_folder), exist_ok=True)

    # Loop through each structure and create its dedicated folder and files
    for i in range(num_structures):
        structure_name = f"structure_{i+1}"
        output_dir = os.path.join(output_base_dir, structure_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Creating folder for {structure_name}...")

        try:
            # Save the specific initial conditions for this single run
            np.save(os.path.join(output_dir, "coords.npy"), all_coords[i])
            np.save(os.path.join(output_dir, "velocities.npy"), all_velocities[i])
            np.save(os.path.join(output_dir, "spin.npy"), all_spins[i])

            # Copy the main simulation script into the directory
            shutil.copy(simulation_script_name, output_dir)
            
            # Create a specific run.sh for this folder that passes the structure name
            # and the relative path to the shared results folder.
            submission_script_path = os.path.join(output_dir, "run.sh")
            with open(submission_script_path, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write("# This script runs the simulation and saves results to a shared folder.\n")
                # Pass structure name and relative path to results folder as arguments
                f.write(f"python {simulation_script_name} {structure_name} ../{shared_results_folder}\n")
            
            # Make the submission script executable
            #os.chmod(submission_script_path, 0o755)

        except Exception as e:
            print(f"An unexpected error occurred while preparing {structure_name}. Skipping.")
            print(f"Details: {e}")
    
    print(f"\nSetup complete. {num_structures} simulation folders have been created in '{output_base_dir}'.")
    print(f"All results will be saved in '{os.path.join(output_base_dir, shared_results_folder)}'.")