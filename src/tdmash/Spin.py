import numpy as np
import scipy

def coeffs_to_bloch(coeff):
    """
    Converts state coefficients [alpha, beta] to a Bloch vector [x, y, z].

    The Bloch vector components are derived from the coefficients as:
    x = 2 * Re(α*β)
    y = 2 * Im(α*β)
    z = |α|² - |β|²

    Args:
        alpha (complex): The complex coefficient for the |0⟩ state.
        beta (complex): The complex coefficient for the |1⟩ state.

    Returns:
        np.ndarray: A numpy array representing the Bloch vector [x, y, z].
                    Returns None if the coefficients are not normalized.
    """
    alpha, beta = coeff
    # Verify that the state is normalized (|α|² + |β|² = 1)
    norm = np.abs(alpha)**2 + np.abs(beta)**2
    if not np.isclose(norm, 1.0):
        print(f"Error: The provided coefficients are not normalized. |α|² + |β|² = {norm:.4f}")
        return None

    # Calculate Bloch vector components
    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)
    z = np.abs(alpha)**2 - np.abs(beta)**2

    return np.array([x, y, z])

def bloch_to_coeffs(bloch_vector):
    """
    Converts a Bloch vector [x, y, z] to the state coefficients [alpha, beta].

    The state of a qubit can be written as |ψ⟩ = α|0⟩ + β|1⟩.
    The coefficients are parameterized by two angles, θ (theta) and φ (phi):
    α = cos(θ/2)
    β = e^(iφ) * sin(θ/2)

    The Bloch vector components are related to these angles as:
    x = sin(θ)cos(φ)
    y = sin(θ)sin(φ)
    z = cos(θ)

    Args:
        bloch_vector (list or np.ndarray): A list or numpy array of 3 numbers
                                           representing the [x, y, z] coordinates
                                           of the vector on the Bloch sphere.
                                           The vector should have a length <= 1.

    Returns:
        tuple: A tuple containing the two complex coefficients (alpha, beta).
               Returns (None, None) if the input vector is invalid.
    """
    # Ensure the input is a numpy array for vector operations
    vec = np.array(bloch_vector, dtype=float)

    # Validate input vector
    if vec.shape != (3,):
        print("Error: Input must be a 3-element vector [x, y, z].")
        return None, None

    # Calculate the length (radius) of the Bloch vector.
    # For pure states, r=1. For mixed states, r<1.
    r = np.linalg.norm(vec)

    if r > 1.000001:  # Use a small tolerance for floating point errors
        print(f"Warning: The norm of the vector ({r:.4f}) is greater than 1.")
        print("Normalizing the vector to the surface of the Bloch sphere.")
        vec = vec / r
        r = 1.0

    x, y, z = vec

    # Handle the special cases of the poles (states |0⟩ and |1⟩)
    # to avoid issues with arctan2(0, 0)
    if np.isclose(r, 0):  # Maximally mixed state
        # The phase phi is undefined. We can choose a convention, e.g., phi=0.
        # The state is 0.5*I, but the coefficients are not uniquely defined
        # in the same way. A common representation is an equal superposition.
        print("Warning: Vector is at the origin (maximally mixed state). Phase is undefined.")
        theta = np.pi / 2  # Corresponds to z=0
        phi = 0
    elif np.isclose(z, 1.0):  # State |0⟩
        theta = 0
        phi = 0  # phi is undefined here, but its value doesn't matter
    elif np.isclose(z, -1.0):  # State |1⟩
        theta = np.pi
        phi = 0  # phi is undefined here, but its value doesn't matter
    else:
        # Calculate theta from z = cos(θ)
        theta = np.arccos(z)
        # Calculate phi from y/x = tan(φ)
        phi = np.arctan2(y, x)

    # Calculate the coefficients using the derived angles
    alpha = np.cos(theta / 2)
    beta = np.exp(1j * phi) * np.sin(theta / 2)

    return np.asarray([alpha, beta])


class Spin:
    def __init__(self, initial_spin):
        #self.initial_spin = initial_spin.copy()
        self.current_spin = initial_spin.copy()
        #self.initial_coefficent = bloch_to_coeffs(initial_spin)
        self.coefficent = bloch_to_coeffs(initial_spin)
        self.S_value = [] # S value for each step

        #print (self.coefficent)
        
    def determine_active_state_index(self):
        """
        Determines the active state based on the current spin vector.
        Returns a tuple of coefficients (alpha, beta) representing the state.
        """
        return 0 if self.current_spin[2] > 0 else 1 



    def propogate_spin(self, dt, energy_last_step, energy_this_step,  overlap): 
        
        H_last_step = np.diag(energy_last_step)
        H_this_step = np.diag(energy_this_step)

        H_last_step = H_last_step - np.eye(2) * H_last_step[0,0]
        H_this_step = H_this_step - np.eye(2) * H_this_step[0,0]
        
        print (H_last_step, H_this_step)
        
        self.coefficent = scipy.linalg.expm(-1j * dt/2 * H_last_step) @ self.coefficent
        U,s,Vh = np.linalg.svd(overlap)
        T= scipy.linalg.logm(U@Vh)
        self.coefficent = scipy.linalg.expm(-T) @ self.coefficent  
        self.coefficent = scipy.linalg.expm(-1j * dt/2 * H_this_step) @ self.coefficent
        #self.coefficent = scipy.linalg.expm(-1j * dt/2 * (H_last_step + H_this_step) - T ) @ self.coefficent
        #print (self.coefficent)
        
        self.current_spin = coeffs_to_bloch(self.coefficent)
        self.S_value.append(self.current_spin.copy())
        print (self.current_spin)

    def reach_hopping_point(self):
        if (len(self.S_value) < 2):
            # Not enough history to determine if a hop is needed
            return False

        else:
            return ((self.S_value[-1][2]  * self.S_value[-2][2] )< 0)

    def rescale_momentum_on_hop_v0(self, p_current,  energy_this_step, mass): 
        """
        Rescales or reflects momentum if a hop is attempted.
        """

        V_initial_hop = energy_this_step[0] if self.S_value[-2][2]> 0 else energy_this_step[1]
        V_target_hop = energy_this_step[0] if self.S_value[-1][2]> 0 else energy_this_step[1]
        delta_V = V_target_hop - V_initial_hop
                #p_tilde_cuerrent = p_current / np.sqrt(mass)
        p_tilde_cuerrent = np.einsum("i,ij->ij", 1./np.sqrt(mass), p_current   ) 
        E_kin = 0.5 * np.sum(p_tilde_cuerrent.flatten()**2)
        if (E_kin < delta_V):
            # Hop is rejected
            # hop_rejected = True
            self.current_spin[2] = -1* self.current_spin[2]
            self.coefficent = bloch_to_coeffs(self.current_spin).copy()
            self.S_value[-1] = self.current_spin.copy()

            return -p_current
        else:
            # Hop is accepted
            p_out = p_tilde_cuerrent * np.sqrt((E_kin - delta_V) / E_kin)
            return np.einsum("i,ij->ij", np.sqrt(mass), p_out)
        
    def rescale_momentum_on_hop_v1(self, p_current, nac_vec, energy_this_step, mass):


        V_initial_hop = energy_this_step[0] if self.S_value[-2][2]> 0 else energy_this_step[1]
        V_target_hop = energy_this_step[0] if self.S_value[-1][2]> 0 else energy_this_step[1]
        delta_V = V_target_hop - V_initial_hop
        p_tilde_cuerrent = np.einsum("i,ij->ij", 1./np.sqrt(mass), p_current   ) 
        nac_vec_real = self.current_spin[0] * nac_vec.real + self.current_spin[1] * nac_vec.imag
        nac_tilde = np.einsum("i,ij->ij", 1./np.sqrt(mass), nac_vec_real)  

        rescaled_velocity_norm = np.dot(p_tilde_cuerrent.flatten(), nac_tilde.flatten())
        rescaled_nac_norm = np.dot(nac_tilde.flatten(), nac_tilde.flatten())
        E_kin = 0.5 * np.abs( rescaled_velocity_norm )**2 / rescaled_nac_norm

        if (E_kin < delta_V):
            # Hop is rejected
            # hop_rejected = True
            self.current_spin[2] = -1* self.current_spin[2]
            self.S_value[-1] = self.current_spin.copy()
            p_tilde_cuerrent = p_tilde_cuerrent - 2 * rescaled_velocity_norm / rescaled_nac_norm * nac_tilde
            p_out = np.einsum("i,ij->ij", np.sqrt(mass), p_tilde_cuerrent)
            return p_out
        else:
            # Hop is accepted
            p_tilde_cuerrent = p_tilde_cuerrent + (np.sqrt((E_kin - delta_V)/ E_kin ) -1) * rescaled_velocity_norm / rescaled_nac_norm * nac_tilde
            p_out = np.einsum("i,ij->ij", np.sqrt(mass), p_tilde_cuerrent)
            return p_out 

    def rescale_momentum_on_hop_v2(self, p_current, nac_vec_dirt,  energy_this_step, mass):


        V_initial_hop = energy_this_step[0] if self.S_value[-2][2]> 0 else energy_this_step[1]
        V_target_hop = energy_this_step[0] if self.S_value[-1][2]> 0 else energy_this_step[1]
        delta_V = V_target_hop - V_initial_hop
        print ("Delta V:", delta_V)
        p_tilde_cuerrent = np.einsum("i,ij->ij", 1./np.sqrt(mass), p_current   ) 
        v_current = np.einsum("i,ij->ij", 1./mass, p_current   ) 
        #nac_vec_real 
        nac_tilde = np.einsum("i,ij->ij", 1./np.sqrt(mass),  nac_vec_dirt.real   ) 

        rescaled_velocity_norm = np.dot(p_tilde_cuerrent.flatten(), nac_vec_dirt.real.flatten())
        massless_rescaled_velocity_norm = np.dot(v_current.flatten(), nac_vec_dirt.real.flatten())
        print ("Rescaled velocity norm:", rescaled_velocity_norm )
        E_kin = 0.5 * np.abs( rescaled_velocity_norm )**2

        if (E_kin < delta_V):
            # Hop is rejected
            # hop_rejected = True
            self.current_spin[2] = -1* self.current_spin[2]
            self.coefficent = bloch_to_coeffs(self.current_spin).copy()
            self.S_value[-1] = self.current_spin.copy()
            p_tilde_cuerrent = p_tilde_cuerrent - 2 * massless_rescaled_velocity_norm * nac_tilde
            p_out = np.einsum("i,ij->ij", np.sqrt(mass), p_tilde_cuerrent)
            return p_out
        else:
            # Hop is accepted
            p_tilde_cuerrent= p_tilde_cuerrent + (np.sqrt((E_kin - delta_V)/ E_kin ) -1)  * massless_rescaled_velocity_norm * nac_vec_dirt.real
            p_out = np.einsum("i,ij->ij", np.sqrt(mass), p_tilde_cuerrent)
            return p_out 

    def rescale_momentum_on_hop_v3(self, p_current, nac_vec, energy_this_step, mass):
        """
        Implements momentum rescaling for MASH (unSMASH) dynamics.
        Based on Equations (13a)-(13g) from the supplementary material.
        
        Args:
            p_current: Current momentum array (N_atoms, 3)
            nac_vec: Non-adiabatic coupling vector (NACV) (N_atoms, 3)
            energy_this_step: Adiabatic energies [V0, V1]
            mass: Array of atomic masses (N_atoms,)
        """
        # Identify the initial state (n) and target state (j)
        # In MASH, Sz > 0 corresponds to being on state n [cite: 497]
        if self.S_value[-2][2] > 0:
            n, j = 0, 1
        else:
            n, j = 1, 0
            
        # delta_V_nj = V_n - V_j as used in Equation (13a) [cite: 566, 569]
        delta_V_nj = energy_this_step[n] - energy_this_step[j]
        # The energy gap to overcome for a successful hop is V_j - V_n
        energy_gap_to_overcome = -delta_V_nj 

        # 1. Mass-weighting the vectors 
        # p_tilde = p / sqrt(m)
        p_tilde= np.einsum("i,ij->ij", 1./np.sqrt(mass), p_current   ) 
        #p_tilde = p_current / np.sqrt(mass[:, np.newaxis])
        # d_tilde = d / sqrt(m)
        d_tilde = np.einsum("i,ij->ij", 1./np.sqrt(mass), nac_vec.real   )
        #d_tilde = nac_vec.real / np.sqrt(mass[:, np.newaxis])

        # 2. Calculate projections along the NACV 
        dot_pd = np.sum(p_tilde * d_tilde)
        dot_dd = np.sum(d_tilde * d_tilde)
        
        # Guard against zero coupling
        if dot_dd < 1e-12:
            return p_current

        # 3. Calculate kinetic energy along the derivative coupling vector 
        # E_kin_d = 0.5 * (p_tilde · d_tilde)^2 / (d_tilde · d_tilde)
        E_kin_d = 0.5 * (dot_pd**2) / dot_dd

        # 4. Check for frustrated vs successful hop [cite: 564, 582]
        if E_kin_d < energy_gap_to_overcome:
            # --- Frustrated Hop (Rejected) ---
            # Reverse mass-weighted momentum along the derivative coupling vector 
            p_tilde_new = p_tilde - 2 * d_tilde * (dot_pd / dot_dd)
            
            # Correct the electronic variables: Flip Sz back [cite: 590]
            # Solving EOM shows Sz never changes sign at a frustrated hop [cite: 587]
            self.current_spin[2] = -self.current_spin[2]
            self.coefficent = bloch_to_coeffs(self.current_spin).copy()
            self.S_value[-1] = self.current_spin.copy()
            # Revert to standard momentum: p = p_tilde * sqrt(m)
            print ("Frustrated hop detected. Reversing momentum along NACV and flipping spin. E_kin_d: ", E_kin_d, " | Energy gap to overcome: ", energy_gap_to_overcome)
            return np.einsum("i,ij->ij", np.sqrt(mass), p_tilde_new)
        else:
            # --- Successful Hop ---
            # Rescale mass-weighted momentum along the NACV [cite: 565, 566]
            rescale_factor = np.sqrt((E_kin_d + delta_V_nj) / E_kin_d) - 1.0
            p_tilde_new = p_tilde + rescale_factor * d_tilde * (dot_pd / dot_dd)
            
            # Active surface is updated (handled by index detection in the main loop) [cite: 568]
            print ("Successful hop detected. Rescaling momentum along NACV. E_kin_d: ", E_kin_d, " | Energy gap to overcome: ", energy_gap_to_overcome)
            # Revert to standard momentum: p = p_tilde * sqrt(m)
            return np.einsum("i,ij->ij", np.sqrt(mass), p_tilde_new)
        
    def rescale_momentum_on_hop_v4(self, p_current, nac_vec, energy_this_step, mass):
        """
        Implements momentum rescaling for MASH (unSMASH) dynamics.
        Based on Equations (13a)-(13g) from the supplementary material.
        
        Args:
            p_current: Current momentum array (N_atoms, 3)
            nac_vec: Non-adiabatic coupling vector (NACV) (N_atoms, 3)
            energy_this_step: Adiabatic energies [V0, V1]
            mass: Array of atomic masses (N_atoms,)
        """
        # Identify the initial state (n) and target state (j)
        # In MASH, Sz > 0 corresponds to being on state n [cite: 497]
        if self.S_value[-2][2] > 0:
            n, j = 0, 1
        else:
            n, j = 1, 0
            
        # delta_V_nj = V_n - V_j as used in Equation (13a) [cite: 566, 569]
        delta_V_nj = energy_this_step[n] - energy_this_step[j]
        # The energy gap to overcome for a successful hop is V_j - V_n
        energy_gap_to_overcome = -delta_V_nj 

        # 1. Mass-weighting the vectors 
        # p_tilde = p / sqrt(m)
        p_tilde= np.einsum("i,ij->ij", 1./np.sqrt(mass), p_current   ) 
        #p_tilde = p_current / np.sqrt(mass[:, np.newaxis])
        # d_tilde = d / sqrt(m)
        d_tilde = np.einsum("i,ij->ij", 1./np.sqrt(mass), nac_vec.real   )
        #d_tilde = nac_vec.real / np.sqrt(mass[:, np.newaxis])

        # 2. Calculate projections along the NACV 
        dot_pd = np.sum(p_tilde * d_tilde)
        dot_dd = np.sum(d_tilde * d_tilde)
        
        # Guard against zero coupling
        if dot_dd < 1e-12:
            return p_current

        # 3. Calculate kinetic energy along the derivative coupling vector 
        # E_kin_d = 0.5 * (p_tilde · d_tilde)^2 / (d_tilde · d_tilde)
        E_kin_d = 0.5 * (dot_pd**2) / dot_dd

        # 4. Check for frustrated vs successful hop [cite: 564, 582]
        if E_kin_d < energy_gap_to_overcome:
            # --- Frustrated Hop (Rejected) ---
            # Reverse mass-weighted momentum along the derivative coupling vector 
            #p_tilde_new = p_tilde - 2 * d_tilde * (dot_pd / dot_dd)
            
            # Correct the electronic variables: Flip Sz back [cite: 590]
            # Solving EOM shows Sz never changes sign at a frustrated hop [cite: 587]
            self.current_spin = -self.current_spin
            self.coefficent = bloch_to_coeffs(self.current_spin).copy()
            self.S_value[-1] = self.current_spin.copy()
            # Revert to standard momentum: p = p_tilde * sqrt(m)
            print ("Frustrated hop detected. Reversing momentum along NACV and flipping spin. E_kin_d: ", E_kin_d, " | Energy gap to overcome: ", energy_gap_to_overcome)
            return np.einsum("i,ij->ij", np.sqrt(mass), p_tilde)
        else:
            # --- Successful Hop ---
            # Rescale mass-weighted momentum along the NACV [cite: 565, 566]
            rescale_factor = np.sqrt((E_kin_d + delta_V_nj) / E_kin_d) - 1.0
            p_tilde_new = p_tilde + rescale_factor * d_tilde * (dot_pd / dot_dd)
            
            # Active surface is updated (handled by index detection in the main loop) [cite: 568]
            print ("Successful hop detected. Rescaling momentum along NACV. E_kin_d: ", E_kin_d, " | Energy gap to overcome: ", energy_gap_to_overcome)
            # Revert to standard momentum: p = p_tilde * sqrt(m)
            return np.einsum("i,ij->ij", np.sqrt(mass), p_tilde_new)