from pyscf import gto, scf
import numpy as np
import matplotlib.pyplot as plt
from .TDRHF import TDRHFSimulation

from .TDOverlap import TDOverlap
from .Spin import Spin 
from .FD import Finite_Difference_NAC

class FixedPathDynamics:
    def __init__(self, sim1, sim2, dt, steps, update_mol_dt=None, nuclei_dt=None, energy_filename=None, overlap_filename=None, eigenvalue_filename=None, output_interval=-1, output_filename="trajectory.xyz", initial_velocity=None, mu_1=1.0, mu_2=1.0, output_consrtaint_filename=None, apply_scf=False, initial_spin=np.asarray([0,0,-1])):

        self.dt = dt
        self.t = 0.0

        self.mol = sim1.mol  # Assuming both simulations use the same molecule
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.steps = steps
        self.update_mol_dt = update_mol_dt
        self.initial_velocity = initial_velocity
        self.mass = 1822.88839 * np.array(self.mol.atom_mass_list())  # Masses of nuclei in a.u.
        if initial_velocity is not None:
            self.velocities = np.array(initial_velocity)  # Initial velocities of nuclei
        else:
            self.velocities = np.zeros((len(self.mol.atom_coords()), 3))  # Initialize velocities to zero

        self.force = np.zeros(self.velocities.shape)  # Initialize force array
        self.positions = np.array(self.mol.atom_coords(unit="Bohr"))  # Initial positions of nuclei

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
        self.overlap_alpha = []
        self.overlap_beta = []
        self.overlaps = []
        
        self.spin = Spin(initial_spin)  # Initialize Spin object with initial spin state

        self.adiabatic_energy = []
        self.constraint_values = []

        self.tdoverlap_class = TDOverlap(self.sim1, self.sim2)
        self.overlap_output = []
        self.tdoverlap_output = []

        self.apply_scf = apply_scf
        self.previous_mol = sim1.mol.copy()
        self.density_matrix_output = []
        self.overlap_matrix_output = []
        self.apply_scf = apply_scf

    def run(self):
        if self.output_interval > 0:
            f = open(self.output_filename, "w+")

        # Log initial potentials
        self.potentials_1.append(self.sim1.energy_tot())
        self.potentials_2.append(self.sim2.energy_tot())

        # --- Initial Force Calculation ---
        # Calculate F(t=0) before the loop starts for correct Velocity Verlet.
        # Using dt=0.0 to calculate forces without propagating electrons.
        _, self.force = self.sim2.update_electronic_wavefunction(calculate_force=True, dt=0.0, apply_scf= self.apply_scf)

        self.time = []

        print(f"Starting midpoint-decoupled Ehrenfest/MASH dynamics.")
        print(f"Nuclear dt: {self.nuclei_dt}, Electronic dt: {self.dt}")
        energy_initial = np.asarray([self.sim1.energy_tot(), self.sim2.energy_tot()])
        
        for current_step in range(1, self.steps + 1):
            # Store potential energy at start of step (t) for spin propagation
            #self.potentials_1.append(self.sim1.energy_tot())
            #self.potentials_2.append(self.sim2.energy_tot())
            # --- Step 1: Velocity Verlet Half-Step (Velocity) ---
            # v(t + dt_N/2) = v(t) + 0.5 * F(t)/m * dt_N
            self.velocities += 0.5 * self.nuclei_dt * np.einsum('i,ij->ij', 1. / self.mass, self.force)

            # --- Step 2: Velocity Verlet Half-Step (Position to Midpoint) ---
            # R_mid = R(t) + v(t + dt_N/2) * (dt_N / 2)
            R_mid = self.positions + self.velocities * (self.nuclei_dt / 2.0)

            # --- Step 3: Update Molecules to Midpoint Geometry ---
            self.sim1.update_mol(R_mid)
            self.sim2.update_mol(R_mid)

            # --- Step 4: Electronic Propagation at Fixed Midpoint Geometry ---
            # Propagate from t to t + dt_N.
            # If electronic dt is smaller than nuclei dt, take multiple steps.
            self.sim1.update_electronic_wavefunction(calculate_force=False, dt=self.dt, mu=self.mu_1, apply_scf= self.apply_scf)
            self.sim2.update_electronic_wavefunction(calculate_force=False, dt=self.dt, mu=self.mu_2, apply_scf= self.apply_scf)

            # --- Step 5: Velocity Verlet Remaining Position Step ---
            # R(t + dt_N) = R_mid + v(t + dt_N/2) * (dt_N / 2)
            self.positions = R_mid + self.velocities * (self.nuclei_dt / 2.0)

            # --- Step 6: Update Molecules to Final Position ---
            self.sim1.update_mol(self.positions)
            self.sim2.update_mol(self.positions)

            # --- Step 7: Calculate New Forces at Final Position ---
            # F(t + dt_N) is calculated at R(t + dt_N) with rho(t + dt_N).
            # We use dt=0.0 to update observables/forces without further propagation.
            pot1_final = self.sim1.energy_tot()
            # Note: Keeping original logic where force is derived specifically from sim2.
            self.sim2.calculate_forces()
            self.force = self.sim2.force
            pot2_final = self.sim2.energy_tot()
            energy_final = np.asarray([pot1_final, pot2_final])

            # --- Step 8: Spin Propagation ---
            # Use current step count for output interval
            # Propagate spin over the full time step [t, t + dt_N]
            #if self.output_interval > 0 and current_step % self.output_interval == 0:
            self.tdoverlap_class.update_mo_coeff(self.sim1, self.sim2, previous_mol=self.previous_mol)
            tdoverlap = self.tdoverlap_class.update_overlap()
            self.spin.propogate_spin(self.nuclei_dt, energy_initial, energy_final, tdoverlap)

            # Determine active state (can be used to select force if standard MASH)
            active_state_index = self.spin.determine_active_state_index()
            energy_initial = energy_final.copy()
            self.previous_mol = self.sim1.mol.copy()

            # --- Step 9: Velocity Verlet Final Velocity Update ---
            # v(t + dt_N) = v(t + dt_N/2) + 0.5 * F(t + dt_N)/m * dt_N
            self.velocities += 0.5 * self.nuclei_dt * np.einsum('i,ij->ij', 1. / self.mass, self.force)
            
            # --- Logging and Output ---
            self.t = current_step * self.nuclei_dt
            self.time.append(self.t)
            self.potentials_1.append(pot1_final)
            self.potentials_2.append(pot2_final)
            self.tdoverlap_output.append(tdoverlap)

            kinetic_energy = 0.5 * np.sum(self.mass * np.linalg.norm(self.velocities, axis=1)**2)
            # Total energy based on active state determined by spin
            total_energy = energy_final[active_state_index] + kinetic_energy
            self.kinetic_energies.append(kinetic_energy)
            self.total_energies.append(total_energy)
            
            print(f"Time: {self.t:.2f}, Active State: {active_state_index}, Total Energy: {total_energy:.6f}")


            if self.output_interval > 0 and current_step % self.output_interval == 0:
                self.write_xyz(f, current_step)
                self.density_matrix_output.append(self.sim1.make_rdm1( mo_coeff = self.sim1.mo_coeff, mo_occ = self.sim1.mo_occ))
                self.overlap_matrix_output.append(self.sim1.overlap)

        if self.output_interval > 0:
            f.close()

        if self.energy_filename is not None:
            np.savez(self.energy_filename, time=self.time, potential_energy_1=self.potentials_1, potential_energy_2=self.potentials_2, kinetic_energy=self.kinetic_energies, overlap=self.overlap_output, tdoverlap=self.tdoverlap_output, total_energy=self.total_energies, constraint_values=self.constraint_values, spin=self.spin.S_value , density_matrix = self.density_matrix_output, overlap_matrix = self.overlap_matrix_output)
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
        plt.plot(self.time, self.potentials_1[1:], label='Potential Energy 1')
        #plt.plot(self.time, self.potentials_2[1:], label='Potential Energy 2')
        #plt.plot(self.time, self.total_energies, label='Total Energy')
        plt.xlabel('Time step')
        plt.ylabel('Energy (Hartree)')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # ... (Same main block as original Ehrenfest.py) ...
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
    occ[0] = 1  # HOMO(alpha) -> LUMO(alpha) excited state
    occ[1] = 1  # still singlet

    dm_u = mf2.make_rdm1(mo0, occ)
    # Apply mom occupation principle
    mf2 = scf.addons.mom_occ(mf2, mo0, occ)
    mf2.conv_tol = 1e-16
    mf2.max_cycle = 500
    # Start new SCF with new density matrix
    mf2.scf(dm_u)

    sim1 = TDRHFSimulation(mf1, dt=0.5, nuclei_dt=0.5, ehrenfest=True)
    sim2 = TDRHFSimulation(mf2, dt=0.5, nuclei_dt=0.5, ehrenfest=True)

    # Initialize and run the simulation
    two_trajectory = FixedPathDynamics(sim1, sim2, dt=0.5, nuclei_dt=0.5, steps=800, output_interval=-1, mu_1=0.25, mu_2=1)
    two_trajectory.run()
    two_trajectory.plot_results()
