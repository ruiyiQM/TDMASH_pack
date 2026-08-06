import numpy as np

class WignerSampler:
    """
    A class to generate molecular structures and velocities based on the
    Wigner distribution for a given temperature.

    It performs normal mode analysis on the provided Hessian and masses,
    then samples initial conditions (positions and momenta) from the Wigner
    distribution in normal mode coordinates, and transforms them back to
    Cartesian coordinates.
    """

    # Physical constants in atomic units (Hartree, Bohr, electron mass, atomic time unit)
    # k_B (Boltzmann constant) in Hartree/Kelvin
    KB_AU_PER_K = 3.1668114e-6
    # hbar (reduced Planck constant) in atomic units is 1.0

    def __init__(self, hessian_matrix, masses, equilibrium_coordinates):
        """
        Initializes the WignerSampler.

        Args:
            hessian_matrix (np.ndarray): The 3N x 3N Hessian matrix (second derivatives
                                         of energy with respect to Cartesian coordinates).
                                         Units are typically Hartree/Bohr^2.
            masses (np.ndarray): A 1D array of atomic masses for each atom in the molecule.
                                 Must be in atomic mass units (amu).
            equilibrium_coordinates (np.ndarray): A 2D array (N x 3) of equilibrium atomic
                                                  coordinates. Units are typically Bohr.
        """
        if not isinstance(hessian_matrix, np.ndarray) or hessian_matrix.ndim != 2:
            raise ValueError("hessian_matrix must be a 2D numpy array.")
        if hessian_matrix.shape[0] != hessian_matrix.shape[1]:
            raise ValueError("hessian_matrix must be a square matrix.")
        if not isinstance(masses, np.ndarray) or masses.ndim != 1:
            raise ValueError("masses must be a 1D numpy array.")
        if not isinstance(equilibrium_coordinates, np.ndarray) or equilibrium_coordinates.ndim != 2:
            raise ValueError("equilibrium_coordinates must be a 2D numpy array.")

        num_atoms = len(masses)
        if hessian_matrix.shape[0] != 3 * num_atoms:
            raise ValueError(
                f"Hessian matrix dimensions ({hessian_matrix.shape[0]}) do not match "
                f"3 * number of atoms ({3 * num_atoms})."
            )
        if equilibrium_coordinates.shape[0] != num_atoms or equilibrium_coordinates.shape[1] != 3:
            raise ValueError(
                f"Equilibrium coordinates shape ({equilibrium_coordinates.shape}) "
                f"is not consistent with number of atoms ({num_atoms})."
            )

        self.hessian = hessian_matrix
        # Expand masses to a 3N array for mass-weighting
        self.masses_3n = np.repeat(masses, 3)
        self.equilibrium_coordinates = equilibrium_coordinates.flatten() # Flatten to 1D for calculations

        # Perform normal mode analysis upon initialization
        self._perform_normal_mode_analysis()

    def _mass_weight_hessian(self):
        """
        Mass-weights the Hessian matrix: H'_ij = H_ij / sqrt(m_i * m_j).
        """
        mass_matrix_inv_sqrt = np.diag(1.0 / np.sqrt(self.masses_3n))
        self.mass_weighted_hessian = (
            mass_matrix_inv_sqrt @ self.hessian @ mass_matrix_inv_sqrt
        )

    def _diagonalize_hessian(self):
        """
        Diagonalizes the mass-weighted Hessian to obtain eigenvalues (frequencies^2)
        and eigenvectors (normal modes).
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.mass_weighted_hessian)

        # Sort eigenvalues and eigenvectors in ascending order of eigenvalues
        idx = eigenvalues.argsort()
        self.eigenvalues = eigenvalues[idx]
        self.normal_modes = eigenvectors[:, idx] # Each column is a normal mode

        # Calculate angular frequencies (omega = sqrt(lambda))
        # Handle small negative eigenvalues (due to numerical precision) by setting them to zero
        # Real frequencies will be sqrt(positive_eigenvalue)
        self.angular_frequencies = np.sqrt(np.maximum(self.eigenvalues, 0.0))

        # Check for zero/near-zero frequencies corresponding to translations/rotations
        # A common threshold for identifying near-zero frequencies in a.u. for vibrations
        # is typically around 1e-4 to 1e-5 Hartree^0.5 / (amu^0.5 * Bohr) or similar.
        # We'll identify the lowest frequencies as translational/rotational modes
        # (usually 5 for linear, 6 for non-linear molecules).
        # We'll set their corresponding 'sampling' values to zero later.

        # Normalize normal modes if necessary (already orthonormal for eigh)
        # self.normal_modes = self.normal_modes / np.linalg.norm(self.normal_modes, axis=0)

    def _perform_normal_mode_analysis(self):
        """
        Performs the complete normal mode analysis.
        """
        print("Performing normal mode analysis...")
        self._mass_weight_hessian()
        self._diagonalize_hessian()
        print("Normal mode analysis complete.")
        print(f"Calculated angular frequencies (in atomic units): {self.angular_frequencies}")


    def sample_wigner(self, temperature_kelvin):
        """
        Generates a new structure and corresponding velocities by sampling from the
        Wigner distribution for the given temperature.

        Args:
            temperature_kelvin (float): The temperature in Kelvin.

        Returns:
            tuple: A tuple containing:
                   - np.ndarray: The new 3N x 1 structure coordinates (flattened).
                                 Units are Bohr.
                   - np.ndarray: The 3N x 1 atomic velocities (flattened).
                                 Units are Bohr/atomic_time_unit.
        """
        if temperature_kelvin < 0:
            raise ValueError("Temperature cannot be negative.")

        # Initialize arrays for normal mode positions and momenta
        q_modes = np.zeros_like(self.angular_frequencies)
        p_modes = np.zeros_like(self.angular_frequencies)

        #print(f"\nSampling Wigner distribution at {temperature_kelvin} K...")

        # Constants for Wigner sampling
        kB_T_au = self.KB_AU_PER_K * temperature_kelvin # k_B * T in Hartree

        # Iterate over each normal mode
        for i, omega_i in enumerate(self.angular_frequencies):
            # For near-zero frequencies (translations/rotations), do not sample from Wigner.
            # Set their contribution to zero to avoid numerical issues (coth(0) is inf)
            # A typical threshold for a 'zero' frequency in a.u. could be around 1e-6
            # or based on the number of expected zero modes (e.g., lowest 6).
            # We'll use a pragmatic approach: if omega_i is very small, treat as zero.
            if omega_i < 1e-4:  # Threshold for near-zero frequency (in atomic units)
                q_modes[i] = 0.0
                p_modes[i] = 0.0
                print(f"  Mode {i}: Skipping Wigner sampling for near-zero frequency ({omega_i:.2e} a.u.).")
                continue

            # Calculate beta_i = hbar * omega_i / (2 * k_B * T)
            # hbar is 1.0 in atomic units
            beta_i = omega_i / (2.0 * kB_T_au)

            # Calculate the cotangens hyperbolicus (coth(x) = (exp(x) + exp(-x)) / (exp(x) - exp(-x)))
            # Handle cases where beta_i is extremely small (high T) to avoid overflow in exp,
            # coth(x) approx 1/x for small x.
            if beta_i < 1e-4: # Small beta_i (high temperature limit)
                coth_beta_i = 1.0 / beta_i
            else:
                coth_beta_i = 1.0 / np.tanh(beta_i)


            # Variances for Wigner distribution (hbar = 1.0 in a.u.)
            sigma_qi_squared = (1.0 / (2.0 * omega_i)) * coth_beta_i
            sigma_pi_squared = (omega_i / 2.0) * coth_beta_i

            # Sample q_i and p_i from a Gaussian distribution with calculated variances
            q_modes[i] = np.random.normal(0.0, np.sqrt(sigma_qi_squared))
            p_modes[i] = np.random.normal(0.0, np.sqrt(sigma_pi_squared))
            # print(f"  Mode {i} (omega={omega_i:.4f}): sampled q={q_modes[i]:.4f}, p={p_modes[i]:.4f}")


        # Transform normal mode displacements (q_modes) to Cartesian displacements (delta_r)
        # delta_r = V * q_modes (where V are normal modes, columns are modes)
        # Note: self.normal_modes are mass-weighted normal modes.
        # To get Cartesian displacements from mass-weighted normal modes, we need to multiply
        # by 1/sqrt(masses_3n).
        delta_r = self.normal_modes @ q_modes / np.sqrt(self.masses_3n)
        new_structure_cartesian = self.equilibrium_coordinates + delta_r

        # Transform normal mode momenta (p_modes) to Cartesian momenta (p_cartesian)
        # p_cartesian = V * p_modes * sqrt(masses_3n)
        # This is because p_i (normal mode momentum) = P_i / sqrt(mu_i) where P_i is mass-weighted.
        # No, it's simpler: p_cartesian = mass_matrix_sqrt @ normal_modes @ p_modes
        # mass_matrix_sqrt = np.diag(np.sqrt(self.masses_3n))
        # p_cartesian = mass_matrix_sqrt @ self.normal_modes @ p_modes
        # Simplified:
        p_cartesian = self.normal_modes @ p_modes * np.sqrt(self.masses_3n)

        # Calculate Cartesian velocities
        velocities_cartesian = p_cartesian / self.masses_3n

        #print("Wigner sampling complete.")
        return new_structure_cartesian, velocities_cartesian

# Example Usage:
if __name__ == "__main__":
    # --- Example: A simple diatomic molecule (e.g., H2) ---
    # Using arbitrary small numbers for Hessian for demonstration
    # Real Hessian matrices are symmetric and positive semi-definite for stable structures.

    # Number of atoms
    num_atoms = 2

    # Masses in atomic mass units (amu)
    # Using 1.008 for H, 15.999 for O, 12.011 for C
    # Let's simulate H2:
    masses = np.array([1.008, 1.008])

    # Equilibrium coordinates in Bohr (e.g., H-H bond length 1.4 Bohr)
    equilibrium_coords = np.array([
        [0.0, 0.0, 0.0],
        [1.4, 0.0, 0.0]
    ])

    # A simple 6x6 Hessian matrix for a diatomic molecule
    # For a bond along x-axis, K_xx would be dominant for stretching.
    # Off-diagonal terms for bending/rotation might be near zero for a perfect linear molecule.
    # This is a simplified example, a real Hessian would come from quantum chemistry software.
    # Let's imagine a strong bond constant ~0.2 Hartree/Bohr^2 for stretching mode
    # Units: Hartree/Bohr^2
    hessian = np.zeros((3 * num_atoms, 3 * num_atoms))

    # Approximate force constant for H-H bond (e.g., 0.2 Hartree/Bohr^2)
    # This roughly corresponds to a stretching vibration
    k_stretch = 0.2
    # Simple diagonal approximation for demonstration,
    # for x-coordinates of atom 1 and atom 2, assuming bond along x.
    hessian[0, 0] = k_stretch
    hessian[0, 3] = -k_stretch
    hessian[3, 0] = -k_stretch
    hessian[3, 3] = k_stretch

    # Try creating an instance of the sampler
    try:
        sampler = WignerSampler(hessian, masses, equilibrium_coords)

        # Sample at a given temperature (e.g., 300 K)
        temperature = 300.0 # Kelvin
        new_coords, new_velocities = sampler.sample_wigner(temperature)

        print("\n--- Results for H2 example ---")
        print("Original Equilibrium Coordinates (Bohr):")
        print(equilibrium_coords)
        print("\nNew Sampled Coordinates (Bohr):")
        print(new_coords.reshape(num_atoms, 3))
        print("\nSampled Velocities (Bohr/atomic_time_unit):")
        print(new_velocities.reshape(num_atoms, 3))

        # Check total kinetic energy and potential energy (simple check, not rigorous)
        # KE = 0.5 * sum(m_i * v_i^2)
        total_kinetic_energy = 0.5 * np.sum(sampler.masses_3n * new_velocities**2)
        print(f"\nTotal Kinetic Energy of sampled configuration (Hartree): {total_kinetic_energy:.6f}")

        # You can perform this multiple times to get different initial conditions
        # for a molecular dynamics simulation.
        # new_coords_2, new_velocities_2 = sampler.sample_wigner(temperature)

    except ValueError as e:
        print(f"Error during initialization or sampling: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # --- Example: A more complex molecule (e.g., H2O) with more realistic Hessian ---
    # This part would typically come from a quantum chemistry calculation output.
    print("\n" + "="*50)
    print("--- Example for H2O (simplified, hypothetical Hessian) ---")
    num_atoms_h2o = 3
    masses_h2o = np.array([1.008, 15.999, 1.008]) # H, O, H

    # Hypothetical equilibrium coordinates for H2O (in Bohr)
    # This is an approximate geometry, actual values would be optimized.
    eq_coords_h2o = np.array([
        [ 0.000, -1.450,  0.000],  # H
        [ 0.000,  0.000,  0.000],  # O
        [ 0.000,  1.450,  0.000]   # H
    ])

    # Hypothetical Hessian for H2O (9x9 matrix).
    # This is highly simplified and not chemically accurate,
    # just to demonstrate a larger matrix.
    # Real Hessian would have non-zero off-diagonal terms for bending modes etc.
    # Set up some diagonal elements resembling bond stretches and bends.
    hessian_h2o = np.zeros((3 * num_atoms_h2o, 3 * num_atoms_h2o))
    # Approximate stretching force constants for O-H bonds (e.g., 0.3 Hartree/Bohr^2)
    k_oh_stretch = 0.3
    # Approximated bending force constant (e.g., 0.05 Hartree/Bohr^2/rad^2)
    # The actual contribution to Cartesian Hessian is more complex.
    # For demonstration, let's just put some values on the diagonal and some off-diagonal
    # to make it non-trivial.
    for i in range(3 * num_atoms_h2o):
        hessian_h2o[i, i] = 0.1 # General diagonal terms

    # "Bond stretch" elements (simplified direct mapping to cartesian)
    # H1-O along y (index 1 for H1-y, index 4 for O-y)
    hessian_h2o[1, 1] = k_oh_stretch
    hessian_h2o[1, 4] = -k_oh_stretch
    hessian_h2o[4, 1] = -k_oh_stretch

    # H2-O along y (index 7 for H2-y, index 4 for O-y)
    hessian_h2o[7, 7] = k_oh_stretch
    hessian_h2o[7, 4] = -k_oh_stretch
    hessian_h2o[4, 7] = -k_oh_stretch
    hessian_h2o[4, 4] += 2 * k_oh_stretch # Oxygen connected to two hydrogens

    # Add some small off-diagonal terms for coupling (hypothetical)
    hessian_h2o[0, 2] = 0.01
    hessian_h2o[2, 0] = 0.01
    hessian_h2o[5, 8] = -0.005
    hessian_h2o[8, 5] = -0.005

    # Ensure the Hessian is perfectly symmetric for eigh
    hessian_h2o = (hessian_h2o + hessian_h2o.T) / 2.0


    try:
        sampler_h2o = WignerSampler(hessian_h2o, masses_h2o, eq_coords_h2o)
        new_coords_h2o, new_velocities_h2o = sampler_h2o.sample_wigner(300.0)

        print("\n--- Results for H2O example ---")
        print("Original Equilibrium Coordinates (Bohr):")
        print(eq_coords_h2o)
        print("\nNew Sampled Coordinates (Bohr):")
        print(new_coords_h2o.reshape(num_atoms_h2o, 3))
        print("\nSampled Velocities (Bohr/atomic_time_unit):")
        print(new_velocities_h2o.reshape(num_atoms_h2o, 3))

        total_kinetic_energy_h2o = 0.5 * np.sum(sampler_h2o.masses_3n * new_velocities_h2o**2)
        print(f"\nTotal Kinetic Energy of sampled configuration (Hartree): {total_kinetic_energy_h2o:.6f}")

    except ValueError as e:
        print(f"Error during H2O example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred in H2O example: {e}")
