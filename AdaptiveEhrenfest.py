import numpy as np
import copy
from Ehrenfest2 import MASH
from TDRHF import TDRHFSimulation
from pyscf import gto, scf
from MOM import mom_occ_
from Spin import coeffs_to_bloch

class AdaptiveMASH(MASH): 
    def __init__(self, sim1, sim2, dt, total_time, 
                 adaptive_n_history=5, 
                 adaptive_threshold=0.1, 
                 mu_scaling_factor=1.0, 
                 dt_scaling_factor=0.1,
                 **kwargs):
        """
        Adaptive MASH.
        """
        super().__init__(sim1, sim2, dt, steps=0, **kwargs)
        
        self.stop_time = total_time
        self.n_history = adaptive_n_history
        self.massey_threshold = adaptive_threshold
        self.mu_factor = mu_scaling_factor
        self.dt_factor = dt_scaling_factor
        
        self.base_mu_1 = self.mu_1
        self.base_mu_2 = self.mu_2
        self.base_nuclei_dt = self.nuclei_dt
        self.base_elec_dt = self.dt
        
        self.is_high_accuracy = False
        self.adaptive_log = []
        self.snapshot = None
        self.time = [] 

    def save_snapshot(self):
        """Deep copy of the current state."""
        self.snapshot = {
            't': self.t,
            'positions': self.positions.copy(),
            'velocities': self.velocities.copy(),
            'force': self.force.copy(),
            'sim1_mo': self.sim1.mo_coeff.copy(),
            'sim2_mo': self.sim2.mo_coeff.copy(),
            'sim1_occ': self.sim1.mo_occ.copy(),
            'sim2_occ': self.sim2.mo_occ.copy(),
            'prev_mol': self.previous_mol.copy() if self.previous_mol else None,
            'spin_coeff': self.spin.coefficent.copy()
        }

    def restore_snapshot_and_reinit(self):
        """Restores state and strictly clears history to prevent drift."""
        if self.snapshot is None:
            raise RuntimeError("No snapshot to restore.")
            
        snap = self.snapshot
        self.t = snap['t']
        self.positions = snap['positions'].copy()
        self.velocities = snap['velocities'].copy()
        self.force = snap['force'].copy()
        
        self.previous_mol = snap['prev_mol'].copy() if snap['prev_mol'] else None
        
        # Restore Spin
        self.spin.coefficent = snap['spin_coeff'].copy()
        self.spin.current_spin = coeffs_to_bloch(self.spin.coefficent)

        # Clear TDRHF History
        self.sim1.dm_last = 0
        self.sim1.veff_last = 0
        self.sim2.dm_last = 0
        self.sim2.veff_last = 0
        
        # Force re-initialization
        self.sim1.update_mol(self.positions)
        self.sim2.update_mol(self.positions)
        self.sim1.mo_coeff = snap['sim1_mo'].copy()
        self.sim2.mo_coeff = snap['sim2_mo'].copy()
        self.sim1.mo_occ = snap['sim1_occ'].copy()
        self.sim2.mo_occ = snap['sim2_occ'].copy()

    def apply_parameters(self, high_accuracy=False):
        """Sets Mu and Dt and clears history."""
        if high_accuracy:
            self.mu_1 = self.base_mu_1 * self.mu_factor
            self.mu_2 = self.base_mu_2 * self.mu_factor
            self.nuclei_dt = self.base_nuclei_dt * self.dt_factor
            self.dt = self.base_elec_dt * self.dt_factor 
            self.sim1.propagation_method = 'accelerated'
            self.sim2.propagation_method = 'accelerated'

        else:
            self.mu_1 = self.base_mu_1
            self.mu_2 = self.base_mu_2
            self.nuclei_dt = self.base_nuclei_dt
            self.dt = self.base_elec_dt
            self.sim1.propagation_method = 'predictor_corrector'
            self.sim2.propagation_method = 'predictor_corrector'

        self.sim1.update_mu(self.mu_1)
        self.sim2.update_mu(self.mu_2)
        self.sim1.dt = self.dt 
        self.sim2.dt = self.dt
        
        # Clear history when parameters change
        self.sim1.dm_last = 0
        self.sim1.veff_last = 0
        self.sim2.dm_last = 0
        self.sim2.veff_last = 0
        
        self.is_high_accuracy = high_accuracy

    def calculate_massey_metric(self, pot1, pot2, tdoverlap_data, block_duration):
        """
        Metric calculated over the entire block duration.
        """
        gap = np.abs(pot1 - pot2)
        # Ensure we are operating on the matrix
        overlap_matrix = tdoverlap_data[0] if isinstance(tdoverlap_data, list) else tdoverlap_data
        
        # Off-diagonal overlap magnitude
        coupling = np.abs(overlap_matrix[0, 1]) 
        epsilon = 1e-9
        return (coupling / block_duration) / (gap + epsilon)

    def run_block(self, energy_initial_ref):
        """
        Runs `n_history` steps.
        Calculates Overlap/Spin and returns Data ONLY at the end of the block.
        """
        energy_initial = energy_initial_ref.copy()
        block_duration = 0.0
        
        active_idx = self.spin.determine_active_state_index() 

        # --- 1. Integration Loop (Nuclei + Electrons only) ---
        # NO data saving inside this loop
        for _ in range(self.n_history):
            # Velocity Verlet Integration
            self.velocities += 0.5 * self.nuclei_dt * np.einsum('i,ij->ij', 1. / self.mass, self.force)
            R_mid = self.positions + self.velocities * (self.nuclei_dt / 2.0)
            
            self.sim1.update_mol(R_mid)
            self.sim2.update_mol(R_mid)
            
            self.sim1.update_electronic_wavefunction(calculate_force=False, dt=self.dt, mu=self.mu_1)
            self.sim2.update_electronic_wavefunction(calculate_force=False, dt=self.dt, mu=self.mu_2)
            
            self.positions = R_mid + self.velocities * (self.nuclei_dt / 2.0)
            
            self.sim1.update_mol(self.positions)
            self.sim2.update_mol(self.positions)
            
            # Note: We update potentials here just for the loop, but we only return the final ones
            pot1 = self.sim1.energy_tot()
            self.sim2.calculate_forces()
            self.force = self.sim2.force
            pot2 = self.sim2.energy_tot()
            
            self.t += self.nuclei_dt
            block_duration += self.nuclei_dt
            
            self.velocities += 0.5 * self.nuclei_dt * np.einsum('i,ij->ij', 1. / self.mass, self.force)

        # --- 2. End of Block Calculations ---
        
        energy_final = np.asarray([pot1, pot2])

        # Calculate Overlap ONCE (Start of Block -> End of Block)
        self.tdoverlap_class.update_mo_coeff(self.sim1, self.sim2, previous_mol=self.previous_mol)
        tdoverlap = self.tdoverlap_class.update_overlap()
        
        # Calculate Metric for the block
        block_metric = self.calculate_massey_metric(pot1, pot2, tdoverlap, block_duration)
        
        # Propagate Spin ONCE (Start of Block -> End of Block)
        self.spin.propogate_spin(block_duration, energy_initial, energy_final, tdoverlap)
        
        # Update previous_mol for the *next* block
        self.previous_mol = self.sim1.mol.copy()

        # Calculate final properties to return
        kin = 0.5 * np.sum(self.mass * np.linalg.norm(self.velocities, axis=1)**2)
        tot = energy_final[active_idx] + kin

        # Pack data - Single points representing the end of the block
        buffer = {
            'time': self.t, 
            'pot1': pot1, 
            'pot2': pot2,
            'kin': kin, 
            'tot': tot, 
            'tdov': tdoverlap,
            'xyz': self.positions.copy()
        }
        
        return block_metric, buffer, energy_final

    def run(self):
        if self.output_interval > 0:
            f_xyz = open(self.output_filename, "w+")
        
        # Save Initial State (Time 0)
        self.potentials_1.append(self.sim1.energy_tot())
        self.potentials_2.append(self.sim2.energy_tot())
        _, self.force = self.sim2.update_electronic_wavefunction(calculate_force=True, dt=0.0)
        self.previous_mol = self.sim1.mol.copy()
        self.time.append(0.0) # Initial time
        
        # Write initial XYZ
        if self.output_interval > 0:
            self._write_xyz_buffer(f_xyz, 0, self.positions, 0.0)
        
        print(f"Starting Adaptive MASH (Block Size: {self.n_history})")
        
        current_energy_vec = np.asarray([self.sim1.energy_tot(), self.sim2.energy_tot()])
        accepted_blocks_count = 0 

        while self.t < self.stop_time:
            self.save_snapshot()
            
            metric, buffer, next_energy_vec = self.run_block(current_energy_vec)
            accept_block = False
            switch_down_next = False
            
            print(f" t={self.t:.4f} | Metric={metric:.4f} | Mode={'HIGH' if self.is_high_accuracy else 'NORM'}")
            
            if not self.is_high_accuracy:
                if metric > self.massey_threshold:
                    print(f"   >>> REJECT. Switching to HIGH.")
                    self.restore_snapshot_and_reinit() 
                    self.apply_parameters(high_accuracy=True)
                    self.adaptive_log.append((self.t, "SWITCH_TO_HIGH"))
                    continue 
                else:
                    accept_block = True
            else:
                if metric < 0.1 * self.massey_threshold:
                    print(f"   >>> ACCEPT. Switching to NORMAL for next block.")
                    accept_block = True
                    switch_down_next = True
                else:
                    accept_block = True

            if accept_block:
                # Save data points (Single point per accepted block)
                self.time.append(buffer['time'])
                self.potentials_1.append(buffer['pot1'])
                self.potentials_2.append(buffer['pot2'])
                self.kinetic_energies.append(buffer['kin'])
                self.total_energies.append(buffer['tot'])
                self.tdoverlap_output.append(buffer['tdov'])
                
                accepted_blocks_count += 1
                
                # Write XYZ for this accepted block
                if self.output_interval > 0:
                     self._write_xyz_buffer(f_xyz, accepted_blocks_count, buffer['xyz'], buffer['time'])

                current_energy_vec = next_energy_vec

                if self.is_high_accuracy and switch_down_next:
                    # Apply parameters for NEXT block 
                    self.apply_parameters(high_accuracy=False)
                    self.adaptive_log.append((self.t, "SWITCH_TO_NORMAL"))

        if self.output_interval > 0:
            f_xyz.close()
        self.save_results()

    def _write_xyz_buffer(self, file, step, positions, time):
        positions_ang = positions * 0.529177
        file.write(f"{len(self.mol.atom_charges())}\n")
        file.write(f"Step {step} Time {time:.4f}\n")
        for i, (atom, pos) in enumerate(zip(self.mol.atom_charges(), positions_ang)):
            symbol = self.mol.atom_symbol(i)
            file.write(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

    def save_results(self):
        if self.energy_filename is not None:
            np.savez(self.energy_filename, 
                     time=self.time,
                     potential_energy_1=self.potentials_1, 
                     potential_energy_2=self.potentials_2, 
                     kinetic_energy=self.kinetic_energies, 
                     total_energy=self.total_energies,
                     tdoverlap=self.tdoverlap_output,
                     adaptive_log=self.adaptive_log,
                     spin_history=self.spin.S_value)

if __name__ == "__main__":
    # Setup Molecule
    mol = gto.M(atom='''
    N          -7.84651        1.28271       -0.25720
    N          -5.30782        3.39826       -0.51257
    H          -8.45063        2.38905       -2.53646
    H          -4.55615        2.29563        1.14696
    ''', basis='6-31g*', unit="Bohr")
    mol.build()
    initial_velocity = np.array([
   [-1.13505183e-05, 2.47249587e-04, -4.42711954e-04],
   [-1.49838229e-04, 3.89535031e-04, 1.68332456e-04],
   [2.58394889e-03, 5.01774265e-05, 2.90318782e-03],
   [2.06828037e-03, -3.75446805e-04, -1.87823386e-03]
    ])  
    
    mf1 = scf.RHF(mol)
    mf1.kernel()
    mf2 = scf.RHF(mol)
    mf2.kernel()
    
    mo0 = mf2.mo_coeff
    occ =np.zeros((2, mf1.mo_coeff.shape[1]))
    occ[0, mf1.mo_occ > 0] = 1 
    occ[1, mf1.mo_occ > 0] = 1 
    occ[0,7] = 0  
    occ[0,8] = 1 
    mf2 = mom_occ_(mf2, mo0, occ)
    mf2.kernel()

    sim1 = TDRHFSimulation(mf1, dt=1.0, nuclei_dt=1.0, ehrenfest=True,convergence_threshold=1e-4)
    sim2 = TDRHFSimulation(mf2, dt=1.0, nuclei_dt=1.0, ehrenfest=True,convergence_threshold=1e-4)

    # Initialize ADAPTIVE MASH
    adaptive_sim = AdaptiveMASH(
        sim1, sim2, 
        dt=1.0, 
        total_time=600.0,
        mu_1=1.0, mu_2=1.0,
        adaptive_n_history=5,       
        adaptive_threshold=0.010,    
        mu_scaling_factor=40.0,      
        dt_scaling_factor=0.20,      
        initial_velocity=initial_velocity,
        energy_filename="adaptive_mash_results_mu_40_dt_0d1_history_5_Anderson_debug.npz"
    )   
    
    adaptive_sim.run()