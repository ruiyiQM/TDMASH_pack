import numpy as np
from pyscf import gto, dft, scf
import matplotlib.pyplot as plt

from .TDRKS import TDRKSSimulation
from .TDRHF import TDRHFSimulation
from .TDUHF import TDUHFSimulation
from .TDUKS import TDUKSSimulation
from .TDUOverlap import TDOverlap
# --- Define Propagation Methods ---
# These functions will be "monkey-patched" onto the TDRKSSimulation class.

def run_ehrenfest_midpoint_decoupled(self, alpha =1 , apply_scf = False ):
    """
    Main simulation loop for Ehrenfest dynamics with decoupled time steps,
    using the "nuclear-position-coupled midpoint Fock integrator"
    concept from Li et al., J. Chem. Phys. 123, 084106 (2005).
    
    This makes the electronic propagation symmetric w.r.t. the
    Velocity Verlet nuclear step, improving energy conservation.
    """
    if not self.ehrenfest:
        raise ValueError("run_ehrenfest_midpoint_decoupled is only for Ehrenfest dynamics.")

    # Determine how many electronic steps per nuclear step
    dt_N = self.nuclei_dt
    dt_e = self.dt
    num_elec_steps = dt_N / dt_e
    if not np.isclose(num_elec_steps, np.round(num_elec_steps)):
        raise ValueError(f"nuclei_dt ({dt_N}) must be an integer multiple of dt ({dt_e}).")
    elec_steps_per_nuc_step = int(np.round(num_elec_steps))

    print(f"Starting midpoint-decoupled Ehrenfest dynamics (Li et al. method):")
    print(f"  Nuclear steps: {self.steps}")
    print(f"  Nuclear dt (dt_N): {dt_N} a.u.")
    print(f"  Electronic dt (dt_e): {dt_e} a.u.")
    print(f"  {elec_steps_per_nuc_step} electronic steps per nuclear step.")

    with open(self.output_filename, "w") as traj_file:
        # --- Initial state (t=0) ---
        self.t = 0.0
        self.write_xyz(traj_file, 0) # Write initial geometry

        # Calculate initial force and energy
        dm_initial = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
        # self.F is updated inside calculate_forces
        self.calculate_forces() # Calculates F(t=0)

        # Log initial energy (t=0)
        self.times.append(self.t)
        potential_energy = self.energy_tot(dm_total=dm_initial)
        kinetic_energy = 0.5 * np.sum(self.mass * np.linalg.norm(self.velocities, axis=1)**2)
        total_energy = potential_energy + kinetic_energy
        
        self.potential_energies.append(potential_energy)
        self.kinetic_energies.append(kinetic_energy)
        self.total_energies.append(total_energy)
        dipole_moment = self.calculate_dipole(dm_initial)
        self.dipoles.append(dipole_moment)

        print(f"Step: 0 (Initial), Time: {self.t:.2f} a.u., PE: {potential_energy:.6f}, KE: {kinetic_energy:.6f}, Total E: {total_energy:.6f}")

        # --- Main loop (nuclear steps) ---
        for nuc_step in range(self.steps):
            
            # --- Step a: Verlet Part 1 (half velocity, half position) ---
            # v(t + dt_N/2) = v(t) + 0.5 * F(t) * dt_N / m
            self.velocities += 0.5 * dt_N * np.einsum('i,ij->ij', 1. / self.mass, self.force)
            
            # R_mid = R(t + dt_N/2) = R(t) + v(t + dt_N/2) * (dt_N / 2)
            R_mid = self.positions + self.velocities * (dt_N / 2.0)
            
            # --- Step b: Update Mol to Midpoint ---
            # Update molecule geometry, h_core, overlap to *midpoint* R_mid
            self.update_mol(positions=R_mid) 

            # --- Step c: Inner Loop (Electronic Propagation) ---
            # Propagate electrons from t to t + dt_N under the *fixed midpoint geometry*

            #for elec_step in range(0,):
            for elec_step in range(elec_steps_per_nuc_step ):
                # Get Fock matrix at current electronic time t
                # H_core and overlap are fixed at H(R_mid)
                dm_current_elec = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
                E_t_current_elec = self.electric_field(self.t)
                F_current_elec = self.get_fock_matrix(dm_current_elec, E_t_current_elec)
                
                # Propagate electronics by one small 'dt_e' step
                self.propagate_electrons(F_current_elec, E_t_current_elec, dt=dt_e * alpha) 
                # self.mo_coeff = self.orthogonalize_mo_coeff(self.mo_coeff)
                
                # Advance electronic time
                self.t += dt_e
            
            # --- Step d: Verlet Part 1 (remaining half position) ---
            # R_new = R(t + dt_N) = R_mid + v(t + dt_N/2) * (dt_N / 2)
            R_new = R_mid + self.velocities * (dt_N / 2.0)
            self.positions = R_new

            # --- Step e: Update Mol to Final Point & Get New Force ---
            # Must update mol to R_new to calculate F(t + dt_N)
            self.update_mol(positions=R_new) 
            
            dm_final = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
            # self.F is updated inside calculate_forces
            self.calculate_forces() # Calculates F(t + dt_N)
            
            # --- Step f: Verlet Part 2 (update velocity) ---
            # v(t + dt_N) = v(t + dt_N/2) + 0.5 * F(t + dt_N) * dt_N / m
            self.velocities += 0.5 * dt_N * np.einsum('i,ij->ij', 1. / self.mass, self.force)
            
            # --- Step g: Log data ---
            self.times.append(self.t)
            potential_energy = self.energy_tot(dm_total=dm_final)
            kinetic_energy = 0.5 * np.sum(self.mass * np.linalg.norm(self.velocities, axis=1)**2)
            total_energy = potential_energy + kinetic_energy
            
            self.potential_energies.append(potential_energy)
            self.kinetic_energies.append(kinetic_energy)
            self.total_energies.append(total_energy)
            dipole_moment = self.calculate_dipole(dm_final)
            self.dipoles.append(dipole_moment)

            if (nuc_step + 1) % 10 == 0:
                print(f"Nuc Step: {nuc_step + 1}, Time: {self.t:.2f} a.u., PE: {potential_energy:.6f}, KE: {kinetic_energy:.6f}, Total E: {total_energy:.6f}")

            if self.output_interval > 0 and (nuc_step + 1) % self.output_interval == 0:
                self.write_xyz(traj_file, nuc_step + 1)

    if self.energy_filename:
        energy_data = {
            "time": np.array(self.times),
            "potential_energy": np.array(self.potential_energies),
            "dipole_moment": np.array(self.dipoles),
            "kinetic_energy": np.array(self.kinetic_energies),
            "total_energy": np.array(self.total_energies)
        }
        np.savez(self.energy_filename, **energy_data)
        print(f"Energy and dipole data saved to {self.energy_filename}")

def run_ehrenfest_midpoint_decoupled_UKS(self, alpha =1 ):
    """
    Main simulation loop for Ehrenfest dynamics with decoupled time steps,
    using the "nuclear-position-coupled midpoint Fock integrator"
    concept from Li et al., J. Chem. Phys. 123, 084106 (2005).
    
    This makes the electronic propagation symmetric w.r.t. the
    Velocity Verlet nuclear step, improving energy conservation.
    """
    if not self.ehrenfest:
        raise ValueError("run_ehrenfest_midpoint_decoupled is only for Ehrenfest dynamics.")

    # Determine how many electronic steps per nuclear step
    dt_N = self.nuclei_dt
    dt_e = self.dt
    num_elec_steps = dt_N / dt_e
    if not np.isclose(num_elec_steps, np.round(num_elec_steps)):
        raise ValueError(f"nuclei_dt ({dt_N}) must be an integer multiple of dt ({dt_e}).")
    elec_steps_per_nuc_step = int(np.round(num_elec_steps))

    overlap_output = [] # To store overlap values for analysis
    overlap_class = TDOverlap(self)
    #print (overlap_class.mo_coeff.shape)

    print(f"Starting midpoint-decoupled Ehrenfest dynamics (Li et al. method):")
    print(f"  Nuclear steps: {self.steps}")
    print(f"  Nuclear dt (dt_N): {dt_N} a.u.")
    print(f"  Electronic dt (dt_e): {dt_e} a.u.")
    print(f"  {elec_steps_per_nuc_step} electronic steps per nuclear step.")

    with open(self.output_filename, "w") as traj_file:
        # --- Initial state (t=0) ---
        self.t = 0.0
        self.write_xyz(traj_file, 0) # Write initial geometry

        # Calculate initial force and energy
        dm_initial = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
        # self.F is updated inside calculate_forces
        self.calculate_forces() # Calculates F(t=0)

        # Log initial energy (t=0)
        self.times.append(self.t)
        potential_energy = self.energy_tot(dm_total=dm_initial)
        kinetic_energy = 0.5 * np.sum(self.mass * np.linalg.norm(self.velocities, axis=1)**2)
        total_energy = potential_energy + kinetic_energy
        
        self.potential_energies.append(potential_energy)
        self.kinetic_energies.append(kinetic_energy)
        self.total_energies.append(total_energy)
        dipole_moment = self.calculate_dipole(dm_initial)
        self.dipoles.append(dipole_moment)

        print(f"Step: 0 (Initial), Time: {self.t:.2f} a.u., PE: {potential_energy:.6f}, KE: {kinetic_energy:.6f}, Total E: {total_energy:.6f}")

        # --- Main loop (nuclear steps) ---
        for nuc_step in range(self.steps):
            
            # --- Step a: Verlet Part 1 (half velocity, half position) ---
            # v(t + dt_N/2) = v(t) + 0.5 * F(t) * dt_N / m
            self.velocities += 0.5 * dt_N * np.einsum('i,ij->ij', 1. / self.mass, self.force)
            
            # R_mid = R(t + dt_N/2) = R(t) + v(t + dt_N/2) * (dt_N / 2)
            R_mid = self.positions + self.velocities * (dt_N / 2.0)
            
            # --- Step b: Update Mol to Midpoint ---
            # Update molecule geometry, h_core, overlap to *midpoint* R_mid
            self.update_mol(positions=R_mid) 

            # --- Step c: Inner Loop (Electronic Propagation) ---
            # Propagate electrons from t to t + dt_N under the *fixed midpoint geometry*

            #for elec_step in range(0,):
            for elec_step in range(elec_steps_per_nuc_step ):
                # Get Fock matrix at current electronic time t
                # H_core and overlap are fixed at H(R_mid)
                dm_current_elec = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
                E_t_current_elec = self.electric_field(self.t)
                F_current_elec = self.get_fock_matrix(dm_current_elec, E_t_current_elec)
                
                # Propagate electronics by one small 'dt_e' step
                self.propagate_electrons(F_current_elec, E_t_current_elec, dt=dt_e * alpha) 
                # self.mo_coeff = self.orthogonalize_mo_coeff(self.mo_coeff)
                
                # Advance electronic time
                self.t += dt_e
            overlap_class.update_mo_coeff(self)
            overlap_output.append(overlap_class.update_overlap())
            # --- Step d: Verlet Part 1 (remaining half position) ---
            # R_new = R(t + dt_N) = R_mid + v(t + dt_N/2) * (dt_N / 2)
            R_new = R_mid + self.velocities * (dt_N / 2.0)
            self.positions = R_new

            # --- Step e: Update Mol to Final Point & Get New Force ---
            # Must update mol to R_new to calculate F(t + dt_N)
            self.update_mol(positions=R_new) 
            
            dm_final = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
            # self.F is updated inside calculate_forces
            self.calculate_forces() # Calculates F(t + dt_N)
            
            # --- Step f: Verlet Part 2 (update velocity) ---
            # v(t + dt_N) = v(t + dt_N/2) + 0.5 * F(t + dt_N) * dt_N / m
            self.velocities += 0.5 * dt_N * np.einsum('i,ij->ij', 1. / self.mass, self.force)
            
            # --- Step g: Log data ---
            self.times.append(self.t)
            potential_energy = self.energy_tot(dm_total=dm_final)
            kinetic_energy = 0.5 * np.sum(self.mass * np.linalg.norm(self.velocities, axis=1)**2)
            total_energy = potential_energy + kinetic_energy
            
            self.potential_energies.append(potential_energy)
            self.kinetic_energies.append(kinetic_energy)
            self.total_energies.append(total_energy)
            dipole_moment = self.calculate_dipole(dm_final)
            self.dipoles.append(dipole_moment)

            if (nuc_step + 1) % 10 == 0:
                print(f"Nuc Step: {nuc_step + 1}, Time: {self.t:.2f} a.u., PE: {potential_energy:.6f}, KE: {kinetic_energy:.6f}, Total E: {total_energy:.6f}")

            if self.output_interval > 0 and (nuc_step + 1) % self.output_interval == 0:
                self.write_xyz(traj_file, nuc_step + 1)

    if self.energy_filename:
        energy_data = {
            "time": np.array(self.times),
            "potential_energy": np.array(self.potential_energies),
            "dipole_moment": np.array(self.dipoles),
            "kinetic_energy": np.array(self.kinetic_energies),
            "total_energy": np.array(self.total_energies),
            "overlap": np.array(overlap_output)
        }
        np.savez(self.energy_filename, **energy_data)
        print(f"Energy and dipole data saved to {self.energy_filename}")

def run_ehrenfest_gear_decoupled(self, alpha =1 ):
    """
    Runs Ehrenfest dynamics using the high-order, multi-step
    predictor-corrector nuclear propagator from the user-provided image.
    
    The electronic propagation is still performed at the fixed midpoint
    nuclear geometry.
    """
    if not self.ehrenfest:
        raise ValueError("run_ehrenfest_gear_decoupled is only for Ehrenfest dynamics.")

    dt_N = self.nuclei_dt
    dt_e = self.dt
    num_elec_steps = dt_N / dt_e
    if not np.isclose(num_elec_steps, np.round(num_elec_steps)):
        raise ValueError(f"nuclei_dt ({dt_N}) must be an integer multiple of dt ({dt_e}).")
    elec_steps_per_nuc_step = int(np.round(num_elec_steps))

    print(f"Starting high-order Gear-decoupled Ehrenfest dynamics:")
    print(f"  Nuclear steps: {self.steps}")
    print(f"  Nuclear dt (dt_N): {dt_N} a.u.")
    print(f"  Electronic dt (dt_e): {dt_e} a.u.")
    print(f"  {elec_steps_per_nuc_step} electronic steps per nuclear step.")

    with open(self.output_filename, "w") as traj_file:
        # --- Initial state (t=0) ---
        self.t = 0.0
        self.write_xyz(traj_file, 0) 

        # Calculate initial force F(t=0) and dm(t=0)
        dm_t = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
        self.calculate_forces() # self.force = F(t=0)

        # Log initial energy (t=0)
        self.times.append(self.t)
        potential_energy = self.energy_tot(dm_total=dm_t)
        kinetic_energy = 0.5 * np.sum(self.mass * np.linalg.norm(self.velocities, axis=1)**2)
        total_energy = potential_energy + kinetic_energy
        
        self.potential_energies.append(potential_energy)
        self.kinetic_energies.append(kinetic_energy)
        self.total_energies.append(total_energy)
        self.dipoles.append(self.calculate_dipole(dm_t))

        print(f"Step: 0 (Initial), Time: {self.t:.2f} a.u., PE: {potential_energy:.6f}, KE: {kinetic_energy:.6f}, Total E: {total_energy:.6f}")

        # --- Main loop (nuclear steps) ---
        for nuc_step in range(self.steps):
            
            # --- Step 1: Propagate Nuclei to t + dt_N/2 (P-C 1) ---
            
            # Get state at time t
            R_t = self.positions.copy()
            v_t = self.velocities.copy()
            force_t = self.force.copy() # This is F(R(t), n(t))
            a_t = np.einsum('i,ij->ij', 1. / self.mass, force_t)
            # dm_t is from the previous step

            # Predict R(t+dt/2) and v(t+dt/4)
            R_mid_pred = R_t + v_t * (dt_N/2.0) + 0.5 * a_t * (dt_N/2.0)**2
            v_quarter = v_t + a_t * (dt_N/4.0)
            
            # Calculate force at R_mid_pred, but using *old* electronic state dm_t
            self.update_mol(positions=R_mid_pred)
            F_mid_pred = self.get_fock_matrix(dm_t, self.electric_field(self.t + dt_N/2.0))
            force_mid_pred = -self.energy_grad(F_mid_pred, dm_t)
            a_mid_pred = np.einsum('i,ij->ij', 1. / self.mass, force_mid_pred)
            
            # Correct v(t+dt/2)
            v_mid = v_quarter + a_mid_pred * (dt_N/4.0)
            R_mid = R_mid_pred # This is the final midpoint position
            
            # --- Step 2: Propagate Electrons (Inner Loop) ---
            
            # Ensure Mol is at the fixed midpoint geometry
            self.update_mol(positions=R_mid) 
            
            # Propagate n(t) to n(t+dt_N) at fixed R_mid
            for elec_step in range(elec_steps_per_nuc_step * alpha ):
                dm_current_elec = self.make_rdm1(self.mo_coeff, self.mo_occ)
                E_t_current_elec = self.electric_field(self.t)
                F_current_elec = self.get_fock_matrix(dm_current_elec, E_t_current_elec)
                
                self.propagate_electrons(F_current_elec, E_t_current_elec, dt=dt_e) 
                self.mo_coeff = self.orthogonalize_mo_coeff(self.mo_coeff)
                self.t += dt_e
            
            # dm_final is now n(t+dt_N)
            dm_final = self.make_rdm1(self.mo_coeff, self.mo_occ)

            # --- Step 3: Propagate Nuclei to t + dt_N (P-C 2) ---
            
            # Calculate force at R_mid, but using *new* electronic state dm_final
            # (mol is still at R_mid geometry)
            E_mid_new_elec = self.electric_field(self.t - dt_N/2.0) # self.t is t+dt_N
            F_mid_new_elec = self.get_fock_matrix(dm_final, E_mid_new_elec)
            force_mid_new_elec = -self.energy_grad(F_mid_new_elec, dm_final)
            a_mid_new_elec = np.einsum('i,ij->ij', 1. / self.mass, force_mid_new_elec)

            # Predict R(t+dt_N) and v(t+3dt/4)
            R_final_pred = R_mid + v_mid * (dt_N/2.0) + 0.5 * a_mid_new_elec * (dt_N/2.0)**2
            v_three_quarter = v_mid + a_mid_new_elec * (dt_N/4.0)

            # Calculate force at R_final_pred, using *new* electronic state dm_final
            self.update_mol(positions=R_final_pred)
            E_final_elec = self.electric_field(self.t)
            F_final_pred = self.get_fock_matrix(dm_final, E_final_elec)
            force_final_pred = -self.energy_grad(F_final_pred, dm_final)
            a_final_pred = np.einsum('i,ij->ij', 1. / self.mass, force_final_pred)

            # Correct v(t+dt_N)
            v_final = v_three_quarter + a_final_pred * (dt_N/4.0)
            R_final = R_final_pred # This is the final position

            # --- Step 4: Update State and Log ---
            
            self.positions = R_final
            self.velocities = v_final
            self.force = force_final_pred # F(R(t+dt_N), n(t+dt_N))
            dm_t = dm_final # Store for next loop
            
            # Log data
            self.times.append(self.t)
            potential_energy = self.energy_tot(dm_total=dm_final)
            kinetic_energy = 0.5 * np.sum(self.mass * np.linalg.norm(self.velocities, axis=1)**2)
            total_energy = potential_energy + kinetic_energy
            
            self.potential_energies.append(potential_energy)
            self.kinetic_energies.append(kinetic_energy)
            self.total_energies.append(total_energy)
            self.dipoles.append(self.calculate_dipole(dm_final))

            if (nuc_step + 1) % 10 == 0:
                print(f"Nuc Step: {nuc_step + 1}, Time: {self.t:.2f} a.u., PE: {potential_energy:.6f}, KE: {kinetic_energy:.6f}, Total E: {total_energy:.6f}")

            if self.output_interval > 0 and (nuc_step + 1) % self.output_interval == 0:
                self.write_xyz(traj_file, nuc_step + 1)

    if self.energy_filename:
        energy_data = {
            "time": np.array(self.times),
            "potential_energy": np.array(self.potential_energies),
            "dipole_moment": np.array(self.dipoles),
            "kinetic_energy": np.array(self.kinetic_energies),
            "total_energy": np.array(self.total_energies)
        }
        np.savez(self.energy_filename, **energy_data)
        print(f"Energy and dipole data saved to {self.energy_filename}")


# --- Main Execution Block ---

if __name__ == "__main__":
    
    # --- Monkey-Patching Methods ---
    # This attaches the functions defined above to the class we imported
    # This is done to satisfy the "keep files separate" request.
    TDRKSSimulation.run_ehrenfest_gear_decoupled = run_ehrenfest_gear_decoupled
    TDRKSSimulation.run_ehrenfest_midpoint_decoupled = run_ehrenfest_midpoint_decoupled
    TDRHFSimulation.run_ehrenfest_gear_decoupled = run_ehrenfest_gear_decoupled
    TDRHFSimulation.run_ehrenfest_midpoint_decoupled = run_ehrenfest_midpoint_decoupled
    TDUHFSimulation.run_ehrenfest_gear_decoupled = run_ehrenfest_gear_decoupled
    TDUHFSimulation.run_ehrenfest_midpoint_decoupled = run_ehrenfest_midpoint_decoupled
    TDUKSSimulation.run_ehrenfest_gear_decoupled = run_ehrenfest_gear_decoupled
    TDUKSSimulation.run_ehrenfest_midpoint_decoupled = run_ehrenfest_midpoint_decoupled
    # 1. Setup Molecule
    mol = gto.M(
        atom='''
        H 0.000000 0.000000 0.000000
        F 0.000000 0.000000 0.8000
        ''',
        basis='6-31g',
        spin=0 # Explicitly specify spin if needed, UHF can still run on singlet
    )

    
    # 2. Run ground-state SCF (using RKS)
    mf = scf.RHF(mol)
    mf.kernel()

    # 3. Setup Simulation
    simulation_rks = TDRHFSimulation(
        mf, 
        dt=1.0,                # Electronic time step (dt_e)
        nuclei_dt=1.0,         # Nuclear time step (dt_N)
        steps=200,            # Number of *nuclear* steps
        #propagation_method= "exponential_midpoint",  # Electronic propagation method
        propagation_method='accelerated', # Electronic propagation method
        #propagation_method="parallel_transport_crank_nicolson",
        #propagation_method="predictor_corrector_diis", # Electronic propagation method
        field_type='none',  
        ehrenfest=True,       
        convergence_threshold=1e-4, 
        output_interval=10,    # Write XYZ every 10 *nuclear* steps
        energy_filename="decoupled_energies_midpoint_dt_0d1_0d5.npz",
        output_filename="decoupled_trajectory.xyz")
    
    # 4. Run the simulation using one of the patched methods
    simulation_rks.run_ehrenfest_midpoint_decoupled(alpha =1)
    # simulation_rks.run_ehrenfest_gear_decoupled() # Uncomment to run the other method

    # 5. Plot results
    plt.figure(figsize=(10, 6))
    
    # Plot total energy change
    plt.plot(simulation_rks.times, np.array(simulation_rks.total_energies) - simulation_rks.total_energies[0], label='Total Energy Change')
    
    # Optional: Plot individual components
    #plt.plot(simulation_rks.times, np.array(simulation_rks.potential_energies) - simulation_rks.potential_energies[0], label='Potential Energy Change')
    # plt.plot(simulation_rks.times, simulation_rks.kinetic_energies, label='Kinetic Energy')
    
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Energy Change (Hartree)")
    plt.title("Energy Conservation in Decoupled Ehrenfest Dynamics of H2")
    plt.legend()
    plt.grid(True)
    plt.show()