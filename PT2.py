import numpy as np
import scipy
from TDRHF import TDRHFSimulation
from pyscf import gto, scf

class PTCN_TDRHFSimulation(TDRHFSimulation):
    """
    Parallel Transport Crank-Nicolson (PT-CN) propagator (Li, Lin et al. style).
    Designed for moving basis sets (Ehrenfest dynamics) in an Atomic Orbital (AO) basis.
    Ensures unitary propagation and gauge consistency by aligning MOs across steps.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store the molecule and overlap from the previous step to compute cross-overlaps
        self.mol_old = self.mol
        self.convergence_threshold = kwargs.get('convergence_threshold', 1e-5)
        self.overlap_old = self.overlap.copy()

    def get_cross_overlap(self):
        """
        Compute the cross-overlap matrix between the new and old basis sets:
        M_uv = <chi_u(t+dt) | chi_v(t)>
        This term connects the Hilbert spaces of the moving nuclei.
        """
        return gto.intor_cross('int1e_ovlp', self.mol, self.mol_old)

    def _parallel_transport_align(self, C_new, C_old, S_new):
        """
        Parallel Transport Alignment (SVD-based):
        Finds a unitary matrix W that minimizes the distance between C_new and C_old
        under the metric of the new overlap matrix S_new.
        This removes dynamical phases and ensures smooth orbital evolution.
        """
        # Calculate MO-basis overlap: M = C_old^H * S_new * C_new
        M_mo = C_old.conj().T @ S_new @ C_new
        U, _, Vh = np.linalg.svd(M_mo)
        # Optimal rotation W = V * U^H
        W = Vh.conj().T @ U.conj().T
        return C_new @ W

    def pt_crank_nicolson_step(self, F_t, dt):
        """
        Self-consistent Predictor-Corrector PT-CN step.
        Iterates until the Fock matrix at t+dt is consistent with the density at t+dt.
        """
        C_t = self.mo_coeff.copy()
        S_tp1 = self.overlap      # S(t+dt)
        S_t = self.overlap_old    # S(t)
        M = self.get_cross_overlap() # <AO(t+dt)|AO(t)>

        # Pre-compute S(t)^-1 to map the Fock matrix into the transport equation
        S_t_inv = np.linalg.inv(S_t)
        
        # Scale the starting Fock matrix
        local_F_t = F_t * self.mu

        # --- Predictor Step ---
        # Initial guess: F(t+dt) approx F(t)
        F_tp1_guess = local_F_t.copy()
        C_tp1_iter = C_t.copy() # Placeholder for convergence check

        # --- Predictor-Corrector (Self-Consistency) Loop ---
        for iter_count in range(self.max_iterations):
            # 1. Average Fock matrix for the Crank-Nicolson interval
            F_avg = 0.5 * (local_F_t + F_tp1_guess)
            
            # 2. Map the average Fock matrix to handle moving basis functions:
            # F_mapped = F_avg * S(t)^-1 * M
            F_mapped = F_avg @ S_t_inv @ M
            
            # 3. Set up the CN linear system: (S_tp1 + i*dt/2 * F_avg) * C_tp1 = (M - i*dt/2 * F_mapped) * C_t
            lhs = S_tp1 + 0.5j * dt * F_avg
            rhs_vec = (M - 0.5j * dt * F_mapped) @ C_t
            
            # 4. Solve for the new coefficients
            C_tp1_new = np.linalg.solve(lhs, rhs_vec)

            # 5. Apply Parallel Transport Alignment to remove gauge artifacts
            # This is critical to keep the iterative process stable
            C_tp1_new = self._parallel_transport_align(C_tp1_new, C_t, S_tp1)
            print(f"PT-CN Iteration {iter_count}: Max MO coeff diff = {np.max(np.abs(C_tp1_new - C_tp1_iter)):.3e}")
            # 6. Convergence check (Maximum difference in MO coefficients)
            diff = np.max(np.abs(C_tp1_new - C_tp1_iter))
            if  diff < self.convergence_threshold:
                # Converged!
    
                self.mo_coeff = C_tp1_new
                # Update stored Fock matrix for the simulation state
                self.F = F_tp1_guess / self.mu 
                return C_tp1_new

            # 7. Update guess for the next iteration
            C_tp1_iter = C_tp1_new.copy()
            dm_new = self.make_rdm1(C_tp1_new, self.mo_occ)
            E_tp1 = self.electric_field(self.t + dt)
            # Recompute Fock matrix at t+dt with current density estimate
            F_tp1_guess = self.get_fock_matrix(dm_new, E_tp1) * self.mu

        print(f"Warning: PT-CN did not converge within {self.max_iterations} iterations (diff={diff:.2e})")
        self.mo_coeff = C_tp1_iter
        return C_tp1_iter

    def update_mol(self, positions=None):
        """
        Update the molecular geometry and store the current state as 'old' 
        before moving to the new configuration.
        """
        self.mol_old = self.mol
        self.overlap_old = self.overlap.copy()
        super().update_mol(positions)

    def propagate_electrons(self, F_current, E_t, dt=None):
        """
        Electronic propagation wrapper using the PT-CN step.
        """
        if dt is None: dt = self.dt
        self.mo_coeff = self.pt_crank_nicolson_step(F_current, dt)

    def run(self):
        print("Starting Self-Consistent PT-CN (Parallel Transport Crank-Nicolson) Simulation...")
        super().run()

if __name__ == "__main__":
    # Test script for Nitrogen Fluoride (HF) system
    from Verlet import run_ehrenfest_midpoint_decoupled
    from matplotlib import pyplot as plt
    mol = gto.M(atom='H 0 0 0; F 0 0 0.8', basis='6-31g')
    mf = scf.RHF(mol).run()
    PTCN_TDRHFSimulation.ehrenfest_integrator = run_ehrenfest_midpoint_decoupled
    # Initialize simulation with Ehrenfest dynamics enabled
    sim = PTCN_TDRHFSimulation(
        mf, dt=1, steps=100,
        ehrenfest=True,
        propagation_method="pt_cn",
        mu=1.0, # Adjust mu as needed for adiabaticity
        convergence_threshold=1e-5
    )
    
    #sim.run()
    sim.ehrenfest_integrator()
    plt.plot(sim.total_energies, label='Total Energy')
    #plt.show()
    #sim.run()
    print("Final MO Coefficients (First 2x2):\n", sim.mo_coeff[:2, :2])