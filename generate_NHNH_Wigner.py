from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize
import numpy as np
# Make sure Wigner.py is in the same directory or in your Python path
from Wigner import WignerSampler

# --- 1. System Setup and Optimization ---
# This part is run only once.
print("Setting up the molecule...")
# Using the optimized geometry from your previous script as the starting point
# is a good practice, but we'll re-optimize here to keep the script self-contained.
mol = gto.M(atom='''
N 0 0 0
N 0 0 1.25
H 0 1 0
H 0 -1 1.25
''', basis='6-31g*')


mol.build()

print("Optimizing geometry...")
# Use a DFT functional to optimize the molecule
mf = dft.RKS(mol, xc='m06l')
mol_eq = optimize(mf, maxsteps=100)
print("Geometry optimization complete.")

# --- 2. Hessian Calculation ---
# The Hessian is calculated at the equilibrium geometry.
print("Calculating Hessian matrix...")
mf1 = dft.RKS(mol_eq, xc='m06l')
mf1.kernel()
# .Hessian() returns a Hessian object, .kernel() computes the Hessian matrix
h = mf1.Hessian().kernel()
# The result from pyscf is a (natm, natm, 3, 3) array.
# We need to reshape it to a (3*natm, 3*natm) matrix.
hessian = h.transpose(0, 2, 1, 3).reshape(mol.natm * 3, mol.natm * 3)
print("Hessian calculation complete.")

# --- 3. Wigner Sampler Initialization ---
# Convert masses from amu to atomic units (electron mass)
# 1 amu = 1822.88839 m_e
masses_au = np.array(mol_eq.atom_mass_list()) * 1822.88839
eq_coords_bohr = mol_eq.atom_coords(unit="Bohr")

# Initialize the Wigner Sampler
sampler = WignerSampler(hessian, masses_au, eq_coords_bohr)

# --- 4. Generate 100 Structures ---
num_structures = 100
temperature_k = 300.0

# Lists to store the generated data
all_coords = []
all_velocities = []

print(f"\nGenerating {num_structures} structures at {temperature_k} K...")
for i in range(num_structures):
    # Sample new coordinates and velocities from the Wigner distribution
    new_coords, new_velocities = sampler.sample_wigner(temperature_k)
    print (new_coords)
    # Reshape to (natm, 3) and store
    all_coords.append(new_coords.reshape(mol.natm, 3))
    all_velocities.append(new_velocities.reshape(mol.natm, 3))
    
    # Print progress
    if (i + 1) % 10 == 0:
        print(f"  ... generated structure {i + 1}/{num_structures}")

print("Generation complete.")

# --- 5. Save the Results ---

# A. Save coordinates and velocities to NumPy binary files for easy loading
np.save('wigner_coords_100.npy', np.array(all_coords))
np.save('wigner_velocities_100.npy', np.array(all_velocities))
print("\nSaved coordinates to wigner_coords_100.npy")
print("Saved velocities to wigner_velocities_100.npy")


# B. Save coordinates to a single XYZ file for easy visualization
# Conversion factor: 1 Bohr = 0.529177 Angstrom
bohr_to_angstrom = 0.529177

xyz_filename = 'wigner_structures_100.xyz'
with open(xyz_filename, 'w') as f:
    for i, coords_bohr in enumerate(all_coords):
        # Convert coordinates to Angstrom
        coords_angstrom = coords_bohr * bohr_to_angstrom
        
        # Write the header for this frame
        f.write(f"{mol.natm}\n")
        f.write(f"Structure {i + 1} from Wigner sampling at {temperature_k} K\n")
        
        # Write each atom's data
        for atom_idx, atom_coords in enumerate(coords_angstrom):
            atom_symbol = mol_eq.atom_symbol(atom_idx)
            x, y, z = atom_coords
            f.write(f"{atom_symbol:4s} {x:12.6f} {y:12.6f} {z:12.6f}\n")

print(f"Saved all 100 structures in XYZ format to {xyz_filename}")

# --- 6. Analyze Bond Distances ---
print("\nAnalyzing bond distances...")
nn_distances = []
nh_distances = []

# Get indices for N and H atoms to make the code more general
n_indices = [i for i, atom in enumerate(mol_eq.atom) if atom[0] == 'N']
h_indices = [i for i, atom in enumerate(mol_eq.atom) if atom[0] == 'H']

for coords_bohr in all_coords:
    # Calculate N-N distance (assumes only two N atoms)
    if len(n_indices) == 2:
        n1_coord = coords_bohr[n_indices[0]]
        n2_coord = coords_bohr[n_indices[1]]
        nn_dist = np.linalg.norm(n1_coord - n2_coord)
        nn_distances.append(nn_dist)
    
    # For each N, find the nearest H and record that distance
    for n_idx in n_indices:
        n_coord = coords_bohr[n_idx]
        min_dist_to_h = float('inf')
        for h_idx in h_indices:
            h_coord = coords_bohr[h_idx]
            dist = np.linalg.norm(n_coord - h_coord)
            if dist < min_dist_to_h:
                min_dist_to_h = dist
        # Add the shortest N-H distance for this N atom to our list
        nh_distances.append(min_dist_to_h)

# Convert lists to numpy arrays for statistics
nn_distances_ang = np.array(nn_distances) * bohr_to_angstrom
nh_distances_ang = np.array(nh_distances) * bohr_to_angstrom

# Calculate and print statistics
print("\n--- Bond Distance Statistics (Angstrom) ---")
if nn_distances_ang.size > 0:
    print(f"N-N Distance:")
    print(f"  Average:  {np.mean(nn_distances_ang):.4f} Å")
    print(f"  Variance: {np.var(nn_distances_ang)**0.5:.6f} Å")

if nh_distances_ang.size > 0:
    print(f"\nClosest N-H Distance (for each N):")
    print(f"  Average:  {np.mean(nh_distances_ang):.4f} Å")
    print(f"  Variance: {np.var(nh_distances_ang)**0.5:.6f} Å")