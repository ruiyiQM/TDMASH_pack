from pyscf import gto, scf
import numpy as np
import matplotlib.pyplot as plt
from TDRHF import TDRHFSimulation

from TDOverlap import TDOverlap
from Spin import Spin 
from FD import Finite_Difference_NAC
class MASH:
    def __init__(self, sim1, sim2, dt, steps, update_mol_dt=None, nuclei_dt=None, energy_filename=None, overlap_filename=None, eigenvalue_filename=None, output_interval=-1, output_filename="trajectory.xyz", initial_velocity=None, mu_1=1.0, mu_2=1.0, output_consrtaint_filename=None, apply_scf=False, initial_spin=np.asarray([0,0,-1])):

        self.dt = dt
        self.t = 0.0

        self.mol = sim1.mol 
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.steps = steps
        self.update_mol_dt = update_mol_dt
        self.initial_velocity = initial_velocity
        self.mass = 1822.88839 * np.array(self.mol.atom_mass_list()) 
        
        if initial_velocity is not None:
            self.velocities = np.array(initial_velocity)
        else:
            self.velocities = np.zeros((len(self.mol.atom_coords()), 3))

        self.positions = np.array(self.mol.atom_coords(unit="Bohr"))

        if nuclei_dt is None:
            self.nuclei_dt = dt
        else:
            self.nuclei_dt = nuclei_dt

        self.sim1 = sim1
        self.sim2 = sim2

        self.output_interval = output_interval
        self.output_filename = output_filename
        self.energy_filename = energy_filename
        self.overlap_filename = overlap_filename
        self.eigenvalue_filename = eigenvalue_filename
        self.constraint_filename = output_consrtaint_filename

        self.potentials_1 = []
        self.potentials_2 = []
        self.kinetic_energies = []
        self.total_energies = []
        
        self.spin = Spin(initial_spin) 
        self.tdoverlap_class = TDOverlap(self.sim1, self.sim2)
        self.fd = Finite_Difference_NAC(self.sim1, self.sim2)
        self.overlap_output = []
        self.tdoverlap_output = []
        self.constraint_values = []
        self.apply_scf = apply_scf
        self.previous_mol = None

    def run(self):
        if self.output_interval > 0:
            f = open(self.output_filename, "w+")

        # --- Initialization ---
        # 1. Calculate initial forces at t=0
        # We need the electronic state energies and forces before starting the loop
        pot1_init, force1_init = self.sim1.update_electronic_wavefunction(calculate_force=True, dt=0.0)
        pot2_init, force2_init = self.sim2.update_electronic_wavefunction(calculate_force=True, dt=0.0)
        
        # Determine initial active state force
        active_state_index = self.spin.determine_active_state_index()
        self.force = [force1_init, force2_init][active_state_index]
        
        self.potentials_1.append(pot1_init)
        self.potentials_2.append(pot2_init)
        energy_initial = np.asarray([pot1_init, pot2_init])

        stop_time = self.dt * self.steps
        time_list = []

        while self.t < stop_time:
            # --- Velocity Verlet Step 1: Half-step Velocity ---
            # v(t + dt/2) = v(t) + 0.5 * a(t) * dt
            self.velocities += 0.5 * self.nuclei_dt * np.einsum('i,ij->ij', 1. / self.mass, self.force)

            # --- Velocity Verlet Step 2: Full-step Position ---
            # R(t + dt) = R(t) + v(t + dt/2) * dt
            self.positions += self.velocities * self.nuclei_dt
            
            # Update geometries in simulations
            self.sim1.update_mol(self.positions)
            self.sim2.update_mol(self.positions)

            # --- Step 3: Electronic and Spin Propagation ---
            # Propagate electrons at the NEW geometry
            pot1_final, force1_final = self.sim1.update_electronic_wavefunction(calculate_force=True, dt=self.dt, mu = self.mu_1, apply_scf=self.apply_scf)
            pot2_final, force2_final = self.sim2.update_electronic_wavefunction(calculate_force=True, dt=self.dt, mu = self.mu_2, apply_scf=self.apply_scf)
            energy_final = np.asarray([pot1_final, pot2_final])
            
            # Calculate overlap and propagate the spin state
            self.tdoverlap_class.update_mo_coeff(self.sim1, self.sim2, previous_mol=self.previous_mol)
            tdoverlap = self.tdoverlap_class.update_overlap()
            self.spin.propogate_spin(self.nuclei_dt, energy_initial, energy_final, tdoverlap)
            
            # Determine new active state index for the new forces
            active_state_index = self.spin.determine_active_state_index()
            
            # --- Velocity Verlet Step 4: Final Half-step Velocity ---
            # Update force to F(t + dt) based on current active state
            self.force = [force1_final, force2_final][active_state_index]
            # v(t + dt) = v(t + dt/2) + 0.5 * a(t + dt) * dt
            self.velocities += 0.5 * self.nuclei_dt * np.einsum('i,ij->ij', 1. / self.mass, self.force)

            # --- Step 5: MASH Hopping (Momentum Rescaling) ---
            if self.spin.reach_hopping_point():
                print ("Hop detected at time {:.2f} fs".format(self.t * 0.02418884))  # Convert from atomic units to femtoseconds 
                nac = self.fd.compute_nac(self.sim1.mo_coeff, self.sim2.mo_coeff, self.positions)
                momentum_current = np.einsum('i,ij->ij', self.mass, self.velocities)
                rescaled_momentum = self.spin.rescale_momentum_on_hop_v4(momentum_current, nac, energy_final, self.mass)
                self.velocities = np.einsum('i,ij->ij', 1. / self.mass, rescaled_momentum)
                print(self.velocities, self.spin.current_spin, self.spin.coefficent)

            # --- Logging and Preparation for next step ---
            self.t += self.nuclei_dt
            time_list.append(self.t)
            self.potentials_1.append(pot1_final)
            self.potentials_2.append(pot2_final)
            self.tdoverlap_output.append(tdoverlap)
            
            self.previous_mol = self.sim1.mol.copy()
            energy_initial = energy_final.copy() # Current final becomes next initial

            kinetic_energy = 0.5 * np.sum(self.mass * np.linalg.norm(self.velocities, axis=1)**2)
            total_energy = energy_final[active_state_index] + kinetic_energy
            self.kinetic_energies.append(kinetic_energy)
            self.total_energies.append(total_energy)
            
            print(f"Time: {self.t:.2f}, Active State: {active_state_index}, Total Energy: {total_energy:.6f}")

            step_idx = int(round(self.t / self.dt))
            if self.output_interval > 0 and step_idx % self.output_interval == 0:
                self.write_xyz(f, step_idx)


        if self.output_interval > 0:
            f.close()


        if self.energy_filename is not None:
            np.savez(self.energy_filename, time =time_list, potential_energy_1=self.potentials_1, potential_energy_2=self.potentials_2, kinetic_energy=self.kinetic_energies, overlap = self.overlap_output , tdoverlap = self.tdoverlap_output, total_energy=self.total_energies, constraint_values=self.constraint_values, spin = self.spin.S_value)
        if self.overlap_filename is not None:
            np.savez(self.overlap_filename, overlap_alpha=self.overlap_alpha, overlap_beta=self.overlap_beta, overlaps=self.overlaps)
        if self.eigenvalue_filename is not None:
            np.savez(self.eigenvalue_filename, eigenvalues_1=np.asarray(self.sim1.eigenvalues_alpha), eigenvalues_2=np.asarray(self.sim2.eigenvalues_alpha))

    def write_xyz(self, file, step):
        positions_angstrom = self.positions * 0.529177  # Convert from Bohr to Angstrom
        file.write(f"{len(self.mol.atom_charges())}\n")
        file.write(f"Step {step}\n")
        for i, (atom, pos) in enumerate(zip(self.mol.atom_charges(), positions_angstrom)):
            symbol = self.mol.atom_symbol(i)
            file.write(f"{symbol} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

    def plot_results(self):
        plt.figure()
        plt.plot(np.arange(len(self.potentials_1)), self.potentials_1, label='Potential Energy 1')
        plt.plot(np.arange(len(self.potentials_2)), self.potentials_2, label='Potential Energy 2')

        plt.plot(np.arange(len(self.total_energies)), self.total_energies, label='Total Energy')
        plt.xlabel('Time step')
        plt.ylabel('Potential Energy')
        plt.legend()
        plt.show()


if __name__ == "__main__":

    mol = gto.M(
        atom='''
        H  0 0 0
        H  0 0 0.5
        ''',
        basis='6-311g*'
    )


    mf1 = scf.RHF(mol)
    mf1.convergence = 1e-16
    mf1.kernel()


    mf2 = scf.RHF(mol)
    mf2.convergence = 1e-16
    mf2.kernel()

    mo0 = mf2.mo_coeff
    occ = mf2.mo_occ

    
    # Assign initial occupation pattern
    occ[0] = 1  # this excited state is originated from HOMO(alpha) -> LUMO(alpha)
    occ[1] = 1  # it is still a singlet state

    dm_u = mf2.make_rdm1(mo0, occ)
  # Apply mom occupation principle
    mf2 = scf.addons.mom_occ(mf2, mo0, occ)
    mf2.conv_tol = 1e-16  # Increase the convergence threshold
    mf2.max_cycle = 500  # Increase the maximum number of SCF cycles
  # Start new SCF with new density matrix
    mf2.scf(dm_u)

    sim1 = TDRHFSimulation(mf1,dt=0.5,nuclei_dt=0.5,ehrenfest=True)
    sim2 = TDRHFSimulation(mf2,dt=0.5,nuclei_dt=0.5,ehrenfest=True)


    # Initialize and run the simulation
    two_trajectory = MASH(sim1, sim2, dt=0.5,nuclei_dt=0.5, steps=800, output_interval=-1, mu_1 =0.25, mu_2=  1)
    two_trajectory.run()
    two_trajectory.plot_results()
