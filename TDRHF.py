from pyscf import gto, scf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from DIIS import DIIS
from Anderson import AndersonAcceleration
from MOM import singlet_occ_

def lowdin(s):
    """
    Perform Lowdin orthogonalization.

    Args:
        s (np.ndarray): Overlap matrix.

    Returns:
        tuple: Lowdin orthogonalization matrices (half-inverse, half, inverse).
    """
    e, v = scipy.linalg.eigh(s)
    idx = e > 1e-15
    sqrt_e = np.sqrt(e[idx])
    inv_sqrt_e = 1 / sqrt_e
    return (
        np.dot(v[:, idx] * inv_sqrt_e, v[:, idx].conj().T),
        np.dot(v[:, idx] * sqrt_e, v[:, idx].conj().T),
        np.dot(v[:, idx] * 1.0/e[idx], v[:, idx].conj().T),
    )



def linear_interpolated_crank_nicolson_accelerated(self, F_current, E_t, dt=None, num_substeps=5):
        """
        Propagates from t to t+dt using Linear Interpolation of the Fock matrix 
        over small substeps (LiCN).
        
        Accelerated by:
        1. Time-Domain Extrapolation (Predictor)
        2. Anderson Mixing on Fock Matrix (Corrector)
        
        Convergence Criteria: 
        - Difference in MO coefficients between iterations (mo_diff).
        """
        self.prev_mo_coeff = self.mo_coeff.copy()
        if dt is None: dt = self.dt
        
        # Constants
        local_F_start = F_current * self.mu  # F(t)
        sub_dt = dt / num_substeps
        
        # --- 1. PREDICTOR STEP (Time-Domain Extrapolation) ---
        # Guess F(t+dt) based on velocity of F change.
        #if hasattr(self, 'F_prev_step') and self.F_prev_step is not None:
            # F_guess = F(t) + (F(t) - F(t-dt))
        #    F_guess_next = 2.0 * local_F_start - self.F_prev_step
        #else:
        F_guess_next = local_F_start.copy()

        # Initialize Anderson Accelerator
        accelerator = AndersonAcceleration(history_size=6, regularization=1e-5)
        
        # Initialize previous iteration coefficients for convergence check
        mo_coeff_iter_prev = self.mo_coeff.copy()

        # --- Helper: Run the full substep chain ---
        # Takes the TARGET Fock matrix (F_end_val) and propagates C(t) -> C(t+dt)
        def propagate_substeps(F_end_val):
            C_curr = self.mo_coeff.copy() # Start from C(t)
            
            for k in range(num_substeps):
                # Calculate interpolation ratio for the MIDPOINT of the current substep
                alpha = (k + 0.5) / num_substeps
                
                # Linearly interpolate Fock matrix
                F_sub_avg = (1.0 - alpha) * local_F_start + alpha * F_end_val
                
                # Crank-Nicolson Step for this substep
                rhs_op = self.overlap - 0.5j * sub_dt * F_sub_avg
                lhs_op = self.overlap + 0.5j * sub_dt * F_sub_avg
                
                # Solve linear system
                C_curr = np.linalg.solve(lhs_op, rhs_op @ C_curr)
            
            return C_curr

        # --- 2. CORRECTOR LOOP (Self-Consistency) ---
        for iter_count in range(self.max_iterations):
            
            # A. Propagate using current guess for F(t+dt)
            mo_coeff_next = propagate_substeps(F_guess_next)
            
            # B. Convergence Check (MO Coefficients)
            # Compare current MOs with MOs from the previous iteration
            error_mo = mo_coeff_next - mo_coeff_iter_prev
            mo_diff = np.max(np.abs(error_mo))
            print (f" LiCN-Accelerated iter {iter_count}, max MO diff: {mo_diff:.2e}")
            
            if iter_count > 0 and mo_diff < self.convergence_threshold:
                # Store history for the NEXT time step's predictor
                self.F_prev_step = local_F_start.copy() 
                
                # Update system state
                self.mo_coeff = mo_coeff_next
                
                # Update self.F for the final state (consistency)
                dm_final = self.make_rdm1(mo_coeff_next, self.mo_occ)
                E_next_final = self.electric_field(self.t + self.dt)
                self.F = self.get_fock_matrix(dm_final, E_next_final)
                
                # print(f"LiCN-Accelerated converged in {iter_count} iterations. MO diff: {mo_diff:.2e}")
                return
            
            # Update 'prev' for the next comparison
            mo_coeff_iter_prev = mo_coeff_next.copy()

            # C. Build the 'Output' Fock Matrix for Anderson Mixing
            dm_next = self.make_rdm1(mo_coeff=mo_coeff_next, mo_occ=self.mo_occ)
            E_next_val = self.electric_field(self.t + self.dt)
            F_next_raw = self.get_fock_matrix(dm_next, E_next_val)
            F_out_next = F_next_raw * self.mu # Normalize
            
            # D. Anderson Mixing (Mix F to generate next guess)
            F_guess_next = accelerator.compute(F_guess_next, F_out_next)

            if iter_count == self.max_iterations - 1:
                print(f"Warning: LiCN-Accelerated did not converge. MO Diff: {mo_diff:.2e}")
                self.F_prev_step = local_F_start.copy() 
                self.mo_coeff = mo_coeff_next
                self.F = F_next_raw.copy()
            
def predictor_corrector_accelerated(self, F_current, E_t, dt=None):
        """
        P-C propagation with Time-Domain Extrapolation and Anderson Acceleration.
        Convergence criteria: Change in MO coefficients.
        """
        self.prev_mo_coeff = self.mo_coeff.copy()
        if dt is None: dt = self.dt
        
        local_F_start = F_current * self.mu # Normalize input F

        F_guess_next = local_F_start.copy()

        # Initialize Accelerator
        accelerator = AndersonAcceleration(history_size=6, regularization=1e-5)
        
        # Initialize previous iteration coefficients for convergence check
        # Initially set to current time step coefficients
        mo_coeff_iter_prev = self.mo_coeff.copy()

        # Helper to solve Crank-Nicolson for a given F_avg
        def solve_cn(F_avg, C_in):
            # (S + i*dt/2 * F_avg) C_next = (S - i*dt/2 * F_avg) C_in
            lhs = self.overlap + 0.5j * dt * F_avg
            rhs = self.overlap - 0.5j * dt * F_avg
            rhs_vec = rhs @ C_in
            return np.linalg.solve(lhs, rhs_vec)

        for iter_count in range(self.max_iterations):
            # A. Midpoint approximation: F_mid = 0.5 * (F(t) + F_guess(t+dt))
            F_average_scaled = 0.5 * (local_F_start + F_guess_next)
            
            # B. Propagate to get Candidate MOs
            mo_coeff_next = solve_cn(F_average_scaled, self.mo_coeff)
            
            # --- C. Convergence Check (MO Coefficients) ---
            # Compare current MOs with MOs from the previous iteration
            error_mo = mo_coeff_next - mo_coeff_iter_prev
            mo_diff = np.max(np.abs(error_mo))
            if (self.count and iter_count > 0):
                print (f" P-C iteration {iter_count}, max MO diff: {mo_diff:.2e}")
            # Check convergence (skip first iter to allow predictor to correct once)
            if iter_count > 0 and mo_diff < self.convergence_threshold:
                # Update system state
                self.mo_coeff = mo_coeff_next
                self.F_prev_step = local_F_start.copy() # Store F(t) for next step's extrapolation
                
                # We need to build the final F to store it in self.F, 
                # even though we didn't use it for the check this specific time.
                dm_final = self.make_rdm1(mo_coeff_next, self.mo_occ)
                E_next_final = self.electric_field(self.t + self.dt)
                self.F = self.get_fock_matrix(dm_final, E_next_final)
                
                return

            # Update 'prev' for the next comparison
            mo_coeff_iter_prev = mo_coeff_next.copy()

            # D. Build new Fock F_out(t+dt) for Anderson Mixing
            dm_next = self.make_rdm1(mo_coeff_next, self.mo_occ)
            E_next_val = self.electric_field(self.t + self.dt)
            F_next_raw = self.get_fock_matrix(dm_next, E_next_val)
            F_out_next = F_next_raw * self.mu # Normalize
            
            # E. Anderson Acceleration / Mixing (On Fock Matrix)
            # We mix F because mixing C destroys orthogonality
            F_guess_next = accelerator.compute(F_guess_next, F_out_next)

            if iter_count == self.max_iterations - 1:
                print(f"Warning: PC-Accelerated did not converge. MO Diff: {mo_diff:.2e}")
                self.mo_coeff = mo_coeff_next
                self.F_prev_step = local_F_start.copy()
                self.F = F_next_raw.copy()

def linear_interpolated_crank_nicolson_diis(self, F_current, E_t, dt=None, num_substeps=5):
        """
        Performs propagation by linearly interpolating F between t and t+dt.
        Uses DIIS to accelerate the self-consistency of F(t+dt).
        
        Args:
            F_current: Fock matrix at time t (F_start).
            num_substeps (int): Number of uniform substeps for the CN integration.
        """
        self.prev_mo_coeff = self.mo_coeff.copy()
        if dt is None: dt = self.dt

        # Constants for the propagation
        local_F_start = F_current * self.mu # F(t)
        sub_dt = dt / num_substeps
        
        # --- Helper: Run the full substep chain ---
        # This function takes F_end (the target Fock matrix) and propagates 
        # from t to t+dt using linear interpolation.
        def run_substep_propagation(F_end_val):
            C_curr = self.mo_coeff.copy() # Start from C(t)
            
            for k in range(num_substeps):
                # Calculate interpolation ratio for the midpoint of the current substep
                # Time at midpoint of substep k is: t + k*sub_dt + 0.5*sub_dt
                # Fraction of total dt is: (k + 0.5) / num_substeps
                tau_ratio = (k + 0.5) / num_substeps
                
                # Linearly interpolate Fock matrix: F(tau) ~ (1-tau)F_start + tau*F_end
                F_sub_avg = (1.0 - tau_ratio) * local_F_start + tau_ratio * F_end_val
                
                # Crank-Nicolson Step for this substep
                # (S + i*dt/2 * F_sub) @ C_{k+1} = (S - i*dt/2 * F_sub) @ C_k
                # Note: using sub_dt
                rhs_op = self.overlap - 0.5j * sub_dt * F_sub_avg
                lhs_op = self.overlap + 0.5j * sub_dt * F_sub_avg
                
                # Solve linear system
                C_curr = np.linalg.solve(lhs_op, rhs_op @ C_curr)
            
            return C_curr

        # --- Predictor (Initial Guess) ---
        # Initial guess: Assume F(t+dt) is roughly F(t)
        # We run one propagation pass assuming F is constant to get a good starting C(t+dt)
        mo_coeff_predict_iter = run_substep_propagation(local_F_start)
        
        # Initialize DIIS
        diis = DIIS(subspace_size=8)

        # --- DIIS Loop ---
        for iter_count in range(1, self.max_iterations):
            # 1. Build F(t+dt) based on current prediction of MOs
            dm_predict = self.make_rdm1(mo_coeff=mo_coeff_predict_iter, mo_occ=self.mo_occ)
            E_t_plus_dt = self.electric_field(self.t + self.dt)
            F_predict_raw = self.get_fock_matrix(dm_predict, E_t_plus_dt)
            F_end_target = F_predict_raw * self.mu

            # 2. Corrector Step: Propagate strictly from t -> t+dt using the new F_end_target
            mo_coeff_corrected = run_substep_propagation(F_end_target)

            # 3. Calculate Error
            error = mo_coeff_corrected - mo_coeff_predict_iter
            mo_diff = np.max(np.abs(error))
            
            print(f" Linear-CN (DIIS) iter {iter_count}, diff: {mo_diff:.2e}")

            if iter_count > 0 and mo_diff < self.convergence_threshold:
                mo_coeff_predict_iter = mo_coeff_corrected
                self.F = F_predict_raw.copy() # Update cached Fock
                break
            
            # 4. DIIS Update and Extrapolation
            diis.update(mo_coeff_corrected, error)
            
            if iter_count >= diis.min_subspace_size:
                mo_extrapolated = diis.extrapolate()
                # CRITICAL: DIIS extrapolation destroys orthogonality. Fix it.
                mo_coeff_predict_iter = self.orthogonalize_mo_coeff(mo_extrapolated)
            else:
                mo_coeff_predict_iter = mo_coeff_corrected

            if iter_count == self.max_iterations - 1:
                 print(f"Warning: Linear-CN (DIIS) did not converge. Diff: {mo_diff:.2e}")

        self.mo_coeff = mo_coeff_predict_iter.copy()

def predictor_corrector_propagation_diis(self, F_current, E_t, dt=None):
        """Performs one step of P-C propagation with stabilized DIIS."""
        self.prev_mo_coeff = self.mo_coeff.copy()
        if dt is None: dt = self.dt

        local_F_scaled = F_current * self.mu
        F_t_scaled = local_F_scaled.copy()
        
        # Predictor (step 0)
        A_pred = self.overlap + 0.5j * dt * F_t_scaled
        B_pred = self.overlap - 0.5j * dt * F_t_scaled
        mo_coeff_predict_iter = np.linalg.solve(A_pred, B_pred @ self.mo_coeff)

        # 初始化 DIIS
        diis = DIIS(subspace_size=16) # 可以尝试调整 subspace_size

        for iter_count in range(1,self.max_iterations):
            # 1. 根据当前的预测 C(t+dt) 构建新的 Fock 矩阵
            dm_predict = self.make_rdm1(mo_coeff=mo_coeff_predict_iter, mo_occ=self.mo_occ)
            E_t_plus_dt = self.electric_field(self.t + self.dt)
            F_predict_raw = self.get_fock_matrix(dm_predict, E_t_plus_dt)
            
            F_predict_scaled = F_predict_raw * self.mu
            F_average_scaled = 0.5 * local_F_scaled + 0.5 * F_predict_scaled
            
            # 2. Corrector 步，得到“修正后”的系数
            A_corr = self.overlap + 0.5j * dt * F_average_scaled
            B_corr = self.overlap - 0.5j * dt * F_average_scaled
            mo_coeff_corrected = np.linalg.solve(A_corr, B_corr @ self.mo_coeff)

            # 3. 计算误差：输入与输出的差异
            error = mo_coeff_corrected - mo_coeff_predict_iter
            mo_diff = np.max(np.abs(error))
            print (f" P-C (DIIS) iteration {iter_count}, max diff: {mo_diff:.2e}")
            if iter_count > 0 and mo_diff < self.convergence_threshold:
                mo_coeff_predict_iter = mo_coeff_corrected
                break
            
            # 4. DIIS 更新与外推
            # 我们希望找到一组组合，使得它们的 error 最小。
            # 这里的参数传入 mo_coeff_corrected，表示我们信任校正后的结果
            diis.update(mo_coeff_corrected, error)
            
            if iter_count >= diis.min_subspace_size:
                mo_extrapolated = diis.extrapolate()
                #CRITICAL: DIIS 外推破坏了正交性，必须重新正交化！
                mo_coeff_predict_iter = self.orthogonalize_mo_coeff(mo_extrapolated)
            else:
                # 在 DIIS 子空间填满前，直接使用校正后的结果
                mo_coeff_predict_iter = mo_coeff_corrected

            # 保存最后一个计算的 F，以备不时之需
            self.F = F_predict_raw.copy()

            if iter_count == self.max_iterations - 1:
                 print(f"Warning: P-C (DIIS) did not converge. Diff: {mo_diff:.2e}")

        self.mo_coeff = mo_coeff_predict_iter.copy()

def predictor_corrector_propagation_diis2(self, F_current, E_t, dt=None):
        """
        P-C propagation with DIIS acceleration on the FOCK matrix.
        """
        self.prev_mo_coeff = self.mo_coeff.copy()
        if dt is None: dt = self.dt
        
        local_F_start = F_current / self.mu

        # --- Predictor Step ---
        # 初始猜测：F(t+dt) 约等于 F(t)
        # 你也可以用 F(t) + (F(t) - F(t-dt)) 来做更好的猜测
        F_guess_next = local_F_start.copy()

        # 初始化 DIIS，用于混合 Fock 矩阵
        diis = DIIS(subspace_size=8, min_subspace_size=2, regularization=1e-5)

        for iter_count in range(self.max_iterations):
            # 1. 构建平均 Fock 矩阵 (Crank-Nicolson 需要)
            # F_avg = 0.5 * (F(t) + F_guess(t+dt))
            F_average_scaled = 0.5 * (local_F_start + F_guess_next)
            
            # 2. 传播一步得到新的 MO 系数
            A_corr = self.overlap + 0.5j * dt * F_average_scaled
            B_corr = self.overlap - 0.5j * dt * F_average_scaled
            # 使用 solve 求解线性方程组 A * C_next = B * C_current
            mo_coeff_next = np.linalg.solve(A_corr, B_corr @ self.mo_coeff)

            # 3. 基于新的 MO 系数构建新的 Fock 矩阵 F_out(t+dt)
            dm_next = self.make_rdm1(mo_coeff_next, self.mo_occ)
            E_next_val = self.electric_field(self.t + self.dt)
            F_next_raw = self.get_fock_matrix(dm_next, E_next_val)
            F_out_next = F_next_raw / self.mu
            
            # 4. 计算误差：F_out - F_guess
            error = F_out_next - F_guess_next
            fock_diff = np.max(np.abs(error))

            # 5. 收敛检查
            if iter_count > 0 and fock_diff < self.convergence_threshold:
                self.mo_coeff = mo_coeff_next # 保存收敛后的 MO 系数
                self.F = F_next_raw.copy()    # 保存收敛后的 Fock 矩阵
                # print(f"Converged in {iter_count} iterations. Diff: {fock_diff:.2e}")
                return

            # 6. DIIS 更新
            diis.update(F_guess_next, error)
            
            # 7. 生成下一个猜测
            # 如果 DIIS 子空间太小，或者误差太大，可以混合一定比例的 F_out (damping)
            if iter_count < 2:
                 # 初期使用简单的线性混合 (Damping)，防止 DIIS 过早介入导致不稳定
                 damping_factor = 0.5 
                 F_guess_next = (1.0 - damping_factor) * F_guess_next + damping_factor * F_out_next
            else:
                 # 使用 DIIS 外推
                 F_extrapolated = diis.extrapolate()
                 # 可选：为了更稳定，可以将外推结果与最新结果再做一次轻微混合
                 # F_guess_next = 0.8 * F_extrapolated + 0.2 * F_out_next
                 F_guess_next = F_extrapolated

            if iter_count == self.max_iterations - 1:
                print(f"Warning: P-C (Fock DIIS) did not converge. Diff: {fock_diff:.2e}")
                # 即使没收敛，也更新为最后一次计算的结果
                self.mo_coeff = mo_coeff_next
                self.F = F_next_raw.copy()

def magnus4_propagation(self, F_current, E_t, dt=None):
        """
        Performs 4th-order Magnus propagation using Gauss-Legendre quadrature points.
        Estimates F(tau1) and F(tau2) by linear interpolation between F(t) and F(t+dt)
        within the predictor-corrector loop.
        """
        self.prev_mo_coeff = self.mo_coeff.copy()
        if dt is None: dt = self.dt

        # Gauss-Legendre coefficients
        sqrt3 = np.sqrt(3)
        c1 = 0.5 - sqrt3 / 6.0  # approx 0.211
        c2 = 0.5 + sqrt3 / 6.0  # approx 0.789

        # Pre-fetch orthogonalization matrices
        X = self.overlap_half_inv
        X_inv = self.overlap_half

        local_F_start = F_current * self.mu

        # --- Helper to build Omega and propagate ---
        def apply_magnus4(F_start_val, F_end_val, C_start):
            # 1. Interpolate F at Gauss points (in AO basis)
            # F(tau) ~ (1 - c) * F(t) + c * F(t+dt)
            F1 = (1.0 - c1) * F_start_val + c1 * F_end_val
            F2 = (1.0 - c2) * F_start_val + c2 * F_end_val

            # 2. Transform to orthogonal basis
            F1_orth = X.conj().T @ F1 @ X
            F2_orth = X.conj().T @ F2 @ X

            # 3. Construct Magnus generator Omega (in orthogonal basis)
            # Term 1: Average Hamiltonian part (like midpoint)
            Omega1 = -1j * (dt / 2.0) * (F1_orth + F2_orth)
            
            # Term 2: Commutator part (handles non-commutativity, gives 4th order)
            # [F2, F1] = F2 @ F1 - F1 @ F2
            Commutator = F2_orth @ F1_orth - F1_orth @ F2_orth
            Omega2 = (sqrt3 * dt**2 / 12.0) * Commutator

            Omega = Omega1 + Omega2

            # 4. Exponentiate and transform back
            U_orth = scipy.linalg.expm(Omega)
            return X @ U_orth @ X_inv @ C_start

        # --- Predictor Step ---
        # Initially assume F(t+dt) = F(t), so F1 = F2 = F(t). 
        # The commutator term will be zero initially.
        mo_coeff_predict = apply_magnus4(local_F_start, local_F_start, self.mo_coeff)

        # --- Corrector Loop ---
        for iter_count in range(self.max_iterations):
            # 1. Build new estimate for F(t+dt)
            dm_predict = self.make_rdm1(mo_coeff_predict, self.mo_occ)
            E_next = self.electric_field(self.t + self.dt)
            F_end_raw = self.get_fock_matrix(dm_predict, E_next)
            F_end_scaled = F_end_raw / self.mu

            # 2. Re-propagate using interpolated F1, F2 based on new F(t+dt)
            mo_coeff_corrected = apply_magnus4(local_F_start, F_end_scaled, self.mo_coeff)
            print (f" Magnus4 iteration {iter_count}, max diff: {np.max(np.abs(mo_coeff_corrected - mo_coeff_predict))}")
            # 3. Convergence Check
            mo_diff = np.max(np.abs(mo_coeff_corrected - mo_coeff_predict))
            if iter_count > 0 and mo_diff < self.convergence_threshold:
                mo_coeff_predict = mo_coeff_corrected
                break
            
            mo_coeff_predict = mo_coeff_corrected
            self.F = F_end_raw.copy()

            if iter_count == self.max_iterations - 1:
                print(f"Warning: Magnus4 propagation did not converge. Diff: {mo_diff:.2e}")

        self.mo_coeff = mo_coeff_predict.copy()

def apply_scf(self):

        """
        Apply the SCF procedure to the mean-field object.

        Returns:
            float: Total energy after SCF.
        """
        from MOM import mom_occ_

        if (self.mo_occ is not None) and (np.sum(self.mo_occ == 1) > 0):
            rhf = self.mf.__class__(self.mol)
            rhf = singlet_occ_(rhf)
        else:
            rhf = self.mf.__class__(self.mol ).newton()
        rhf.verbose = 4

        rhf.kernel()

        self.mf = rhf
        self.initialize_molecular_orbitals(n_virtual=0)
        return self.energy_tot()

def apply_scf_v2(self):

        """
        Apply the SCF procedure to the mean-field object.

        Returns:
            float: Total energy after SCF.
        """
        from MOM import singlet_occ_



        dm = self.make_rdm1(self.mo_coeff, self.mo_occ).real
        rhf = self.mf.__class__(self.mol, xc = self.mf.xc )
        if (np.sum(self.mo_occ == 1) > 0):
            rhf = singlet_occ_(rhf)
        # rhf.max_cycle = 5
        rhf.kernel()

        if (rhf.converged == False):
            rhf.level_shift = 0.1
            rhf.kernel()
        #self.mf = rhf
        #self.initialize_molecular_orbitals(n_virtual=0)
        return rhf.energy_tot()

def linear_interpolated_crank_nicolson(self, F_current, E_t, dt=None, num_substeps=4):
        """
        Performs propagation by linearly interpolating F between t and t+dt,
        and applying Crank-Nicolson propagator over small uniform substeps.
        
        Args:
            F_current: Fock matrix at time t.
            E_t: Electric field at time t (unused directly, kept for signature).
            dt: Time step.
            num_substeps (int): Number of uniform substeps.
        """
        self.prev_mo_coeff = self.mo_coeff.copy()
        if dt is None: dt = self.dt

        local_F_start = F_current / self.mu
        sub_dt = dt / num_substeps

        # --- Helper: Apply substepped CN with linear F interpolation ---
        def apply_linear_cn(F_start_val, F_end_val, C_start):
            C_curr = C_start.copy()
            
            for k in range(num_substeps):
                # Calculate interpolation ratios for start (tau_k) and end (tau_{k+1}) of substep
                ratio_k = k / num_substeps
                ratio_kp1 = (k + 1) / num_substeps
                
                # Interpolate F at substep boundaries
                F_k = (1.0 - ratio_k) * F_start_val + ratio_k * F_end_val
                F_kp1 = (1.0 - ratio_kp1) * F_start_val + ratio_kp1 * F_end_val
                
                # Average F for this substep (standard CN approximation for the substep)
                F_avg_sub = 0.5 * (F_k + F_kp1)
                
                # Crank-Nicolson step: (S + i*dt/2*F) * C_next = (S - i*dt/2*F) * C_curr
                # A @ C_next = B @ C_curr
                A = self.overlap + 0.5j * sub_dt * F_avg_sub
                B = self.overlap - 0.5j * sub_dt * F_avg_sub
                
                # Solve linear system for C_next
                C_curr = np.linalg.solve(A, B @ C_curr)
                
            return C_curr

        # --- Predictor Step ---
        # Assume F(t+dt) approx F(t) initially
        mo_coeff_predict = apply_linear_cn(local_F_start, local_F_start, self.mo_coeff)

        # --- Corrector Loop ---
        for iter_count in range(self.max_iterations):
            # 1. Build F(t+dt) estimate
            dm_predict = self.make_rdm1(mo_coeff_predict, self.mo_occ)
            E_next = self.electric_field(self.t + self.dt)
            F_end_raw = self.get_fock_matrix(dm_predict, E_next)
            F_end_scaled = F_end_raw / self.mu

            # 2. Repropagate with better F(t+dt) estimate
            mo_coeff_corrected = apply_linear_cn(local_F_start, F_end_scaled, self.mo_coeff)
            print (f" LiCN iteration {iter_count}, max diff: {np.max(np.abs(mo_coeff_corrected - mo_coeff_predict))}")
            # 3. Convergence Check
            mo_diff = np.max(np.abs(mo_coeff_corrected - mo_coeff_predict))
            if iter_count > 0 and mo_diff < self.convergence_threshold:
                mo_coeff_predict = mo_coeff_corrected
                break
            
            mo_coeff_predict = mo_coeff_corrected
            self.F = F_end_raw.copy()

            if iter_count == self.max_iterations - 1:
                print(f"Warning: Linear CN propagation did not converge. Diff: {mo_diff:.2e}")

        self.mo_coeff = mo_coeff_predict.copy()

def etrs_propagation(self, F_current, E_t, dt=None):
        """
        Performs one step of Enforced Time-Reversal Symmetry (ETRS) propagation.
        Uses split-operator: U = exp(-i * dt/2 * F(t+dt)) * exp(-i * dt/2 * F(t))
        Optimized by pre-calculating the first half-step and using explicit diagonalization.
        """
        self.prev_mo_coeff = self.mo_coeff.copy()
        if dt is None: dt = self.dt

        # Pre-fetch orthogonalization matrices
        X = self.overlap_half_inv
        X_inv = self.overlap_half

        # --- Helper: Apply exp(-i * tau * F) using diagonalization ---
        # distinct from general expm, eigh is faster/more stable for Hermitian F
        def apply_half_step(F_eff, C_in, tau):
            # 1. Transform Fock to orthogonal basis: F' = X^dagger @ F @ X
            F_orth = X.conj().T @ F_eff @ X
            
            # 2. Diagonalize F_orth (It is Hermitian)
            evals, evecs = scipy.linalg.eigh(F_orth)
            
            # 3. Form propagator in diagonal basis: exp(-i * tau * eigenvalues)
            # U_orth = evecs @ diag(exp(...)) @ evecs.dagger
            # We can optimize applying it to C:
            # C_out = X @ (U_orth @ (X_inv @ C_in))
            
            # Transform input C to orthogonal basis
            C_orth_in = X_inv @ C_in
            
            # Apply diagonal propagator
            # U_orth @ C_orth_in  = evecs @ (np.diag(phases) @ (evecs.conj().T @ C_orth_in))
            phases = np.exp(-1j * tau * evals)
            C_orth_out = evecs @ (phases[:, None] * (evecs.conj().T @ C_orth_in))
            
            # Transform back to AO basis
            return X @ C_orth_out

        # --- 1. Pre-calculation (Fixed during iterations) ---
        # Calculate the half-step propagation from t to t+dt/2 using F(t)
        # C_mid = exp(-i * dt/2 * F(t)) * C(t)
        local_F_scaled = F_current / self.mu
        C_mid_fixed = apply_half_step(local_F_scaled, self.mo_coeff, dt / 2.0)

        # --- 2. Predictor Step ---
        # Initial guess for F(t+dt) can be F(t)
        # C_predict = exp(-i * dt/2 * F(t)) * C_mid_fixed = exp(-i*dt*F(t)) * C(t)
        # Re-using C_mid_fixed for efficiency in first guess
        mo_coeff_predict_iter = apply_half_step(local_F_scaled, C_mid_fixed, dt / 2.0)

        # --- 3. Corrector Loop ---
        for iter_count in range(self.max_iterations):
            # a. Build F(t+dt) using predicted density
            dm_predict = self.make_rdm1(mo_coeff=mo_coeff_predict_iter, mo_occ=self.mo_occ)
            E_t_plus_dt = self.electric_field(self.t + self.dt)
            F_predict_raw = self.get_fock_matrix(dm_predict, E_t_plus_dt)
            F_predict_scaled = F_predict_raw / self.mu

            # b. Apply second half-step using F(t+dt)
            # C(t+dt) = exp(-i * dt/2 * F(t+dt)) * C_mid_fixed
            mo_coeff_corrected = apply_half_step(F_predict_scaled, C_mid_fixed, dt / 2.0)

            # c. Convergence Check
            mo_diff = np.max(np.abs(mo_coeff_corrected - mo_coeff_predict_iter))
            if mo_diff < self.convergence_threshold:
                mo_coeff_predict_iter = mo_coeff_corrected
                break
            
            mo_coeff_predict_iter = mo_coeff_corrected
            self.F = F_predict_raw.copy() # Save for consistency

            if iter_count == self.max_iterations - 1:
                print(f"Warning: ETRS did not converge. Final diff: {mo_diff:.2e}")

        self.mo_coeff = mo_coeff_predict_iter.copy()

def exponential_midpoint_propagation(self, F_current, E_t, dt=None):
        """
        Performs one step of exponential midpoint propagation.
        Uses unitary propagator U = exp(-i * dt * F_mid) in an orthogonal basis.
        """
        self.prev_mo_coeff = self.mo_coeff.copy()
        if dt is None: dt = self.dt

        # Pre-fetch orthogonalization matrices
        # X = S^{-1/2}, X_inv = S^{1/2}
        X = self.overlap_half_inv
        X_inv = self.overlap_half

        local_F_scaled = F_current * self.mu
        #if self.prev_F is not None:
        #        F_predictor = (2*F_current- self.overlap_half @ self.prev_F @ self.overlap_half) * self.mu
        #else:
        #        F_predictor = local_F_scaled.copy()
        # Helper to perform the unitary step: C(t+dt) = X @ exp(-i*dt*F') @ X_inv @ C(t)
        def apply_propagator(F_eff, C_start, time_step):
            # Transform Fock to orthogonal basis: F' = S^{-1/2} @ F @ S^{-1/2}
            F_orth = X.conj().T @ F_eff @ X
            # Calculate unitary propagator in orthogonal basis
            U_orth = scipy.linalg.expm(-1j * time_step * F_orth)
            # Transform back to AO basis
            return X @ U_orth @ X_inv @ C_start

        # --- Predictor Step ---
        # Predict C(t+dt) using F(t)
        mo_coeff_predict_iter = apply_propagator(local_F_scaled, self.mo_coeff, dt)

        # --- Corrector Loop ---
        for iter_count in range(self.max_iterations):
            # 1. Build F(t+dt) using predicted density
            dm_predict = self.make_rdm1(mo_coeff=mo_coeff_predict_iter, mo_occ=self.mo_occ)
            E_t_plus_dt = self.electric_field(self.t + self.dt)
            F_predict_raw = self.get_fock_matrix(dm_predict, E_t_plus_dt)
            
            # 2. Calculate Average Fock matrix (Midpoint approximation)
            F_predict_scaled = F_predict_raw * self.mu
            F_average_scaled = 0.5 * (local_F_scaled + F_predict_scaled)

            # 3. Correct C(t+dt) using F_average
            mo_coeff_corrected = apply_propagator(F_average_scaled, self.mo_coeff, dt)

            # 4. Convergence Check
            if iter_count > 0:
                mo_diff = np.max(np.abs(mo_coeff_corrected - mo_coeff_predict_iter))
                print(f"  Iteration: {iter_count}, MO diff: {mo_diff:.2e}") # Optional debug
                if mo_diff < self.convergence_threshold:
                    mo_coeff_predict_iter = mo_coeff_corrected
                    break
            
            mo_coeff_predict_iter = mo_coeff_corrected
            # Save the last predicted F for consistency if needed elsewhere
            self.F = F_predict_raw.copy() 

            if iter_count == self.max_iterations - 1:
                print(f"Warning: Exponential midpoint did not converge. Final diff: {mo_diff:.2e}")

        self.mo_coeff = mo_coeff_predict_iter.copy()

def linear_interpolated_exponential_propagation(self, F_current, E_t, dt=None, num_substeps=10):
        """
        Performs propagation by assuming F varies linearly from t to t+dt,
        and applying exponential propagators over small uniform substeps.
        U(t+dt, t) ~ Prod_{k=N-1 to 0} exp(-i * (dt/N) * F(tau_k))
        
        Args:
            F_current: Fock matrix at time t.
            E_t: Electric field at time t (unused directly here as F_current includes it, but kept for signature consistency).
            dt: Time step.
            num_substeps (int): Number of uniform substeps to divide dt into.
        """
        self.prev_mo_coeff = self.mo_coeff.copy()
        if dt is None: dt = self.dt

        # Pre-fetch orthogonalization matrices
        # X = S^{-1/2} (transforms F to orthogonal basis)
        # X_inv = S^{1/2} (transforms C to orthogonal basis)
        X = self.overlap_half_inv      
        X_inv = self.overlap_half      

        local_F_start = F_current * self.mu
        sub_dt = dt / num_substeps

        # --- Helper Function: Propagate using linear interpolation ---
        def apply_linear_propagator(F_start, F_end, C_start):
            # 1. Transform initial MOs to orthogonal basis: C_orth = S^{1/2} @ C_AO
            C_orth = X_inv @ C_start
            
            # 2. Sequential propagation over substeps
            for k in range(num_substeps):
                # Calculate interpolation ratio for the midpoint of current substep
                # k=0 puts us at 0.5 * sub_dt. Ratio = (0.5 * sub_dt) / dt = 0.5 / N
                tau_ratio = (k + 0.5) / num_substeps
                
                # Linearly interpolate Fock matrix at this substep
                F_sub = (1.0 - tau_ratio) * F_start + tau_ratio * F_end
                
                # Transform interpolated F to orthogonal basis: F' = S^{-1/2} @ F @ S^{-1/2}
                F_orth_sub = X.conj().T @ F_sub @ X
                
                # Apply unitary propagator for this substep: C_orth = exp(-i * sub_dt * F') @ C_orth
                U_sub = scipy.linalg.expm(-1j * sub_dt * F_orth_sub)
                C_orth = U_sub @ C_orth
                
            # 3. Transform final MOs back to AO basis: C_AO = S^{-1/2} @ C_orth
            return X @ C_orth

        # --- Predictor Step ---
        # Initial guess: F(t+dt) is approximately F(t)
        mo_coeff_predict = apply_linear_propagator(local_F_start, local_F_start, self.mo_coeff)

        # --- Corrector Loop ---
        for iter_count in range(1,self.max_iterations):
            # 1. Construct F(t+dt) based on predicted MOs
            dm_predict = self.make_rdm1(mo_coeff_predict, self.mo_occ)
            E_next = self.electric_field(self.t + self.dt)
            F_end_raw = self.get_fock_matrix(dm_predict, E_next)
            F_end_scaled = F_end_raw * self.mu

            # 2. Re-propagate using the new estimate of F(t+dt) for interpolation
            mo_coeff_corrected = apply_linear_propagator(local_F_start, F_end_scaled, self.mo_coeff)
            print (f"  Iteration: {iter_count}, MO diff: {np.max(np.abs(mo_coeff_corrected - mo_coeff_predict)):.2e}") # Optional debug
            # 3. Convergence Check
            mo_diff = np.max(np.abs(mo_coeff_corrected - mo_coeff_predict))
            if mo_diff < self.convergence_threshold:
                mo_coeff_predict = mo_coeff_corrected
                break
            
            mo_coeff_predict = mo_coeff_corrected
            self.F = F_end_raw.copy() # Cache F(t+dt) for consistency

            if iter_count == self.max_iterations - 1:
                print(f"Warning: Linear interpolated propagation did not converge. Diff: {mo_diff:.2e}")

        self.mo_coeff = mo_coeff_predict.copy()

class TDRHFSimulation:
    def __init__(self, mf, dt, steps=0, E0=0, field_type='none', omega=0, tau=0, t0=0, propagation_method='predictor_corrector',
                 field_direction='z', ehrenfest=False, nuclei_dt=None, initial_velocity=None, update_mol_dt=None,
                 energy_filename=None, output_interval=-1, output_filename="trajectory_rhf.xyz", mu=1.0,
                 n_virtual=0, convergence_threshold=1e-6, max_iterations=40, imaginary_time=False, imaginary_time_dt= None , count= False):
        """
        Initialize the Time-Dependent Restricted Hartree-Fock Simulation.
        """
        self.mf = mf
        self.mol = mf.mol
        self.previous_mol = None 
        self.dt = dt
        self.steps = steps
        self.E0 = E0
        self.field_type = field_type
        self.t0 = t0
        self.omega = omega
        self.tau = tau
        self.propagation_method = propagation_method
        self.field_direction = field_direction
        self.ehrenfest = ehrenfest
        self.nuclei_dt = nuclei_dt or dt
        self.update_mol_dt = update_mol_dt or dt
        self.t = 0.0
        self.mu = mu
        self.convergence_threshold = convergence_threshold
        self.energy_filename = energy_filename
        self.output_interval = output_interval
        self.output_filename = output_filename
        self.max_iterations = max_iterations

        # MODIFICATION: Calculate and cache the initial core Hamiltonian
        self.h_core = self.mf.get_hcore(self.mol)

        self.initialize_molecular_orbitals(n_virtual)
        self.initialize_dynamics_data(initial_velocity)

        self.overlap = self.mol.intor('int1e_ovlp').astype(complex)
        self.overlap_half_inv, self.overlap_half, self.overlap_inv = lowdin(self.overlap)

        self.imaginary_time = imaginary_time
        self.imaginary_time_dt = imaginary_time_dt
        self.F = None
        self.prev_F = None

        self.dm_last = 0
        self.veff_last = 0
        self.with_df = hasattr(mf, 'with_df')

        self.count = count

    def initialize_molecular_orbitals(self, n_virtual):
        """Initializes molecular orbitals and occupation numbers for RHF."""
        self.n_virtual = n_virtual

        if n_virtual == 0:
            occupied_mask = self.mf.mo_occ > 1e-5
            self.mo_coeff = self.mf.mo_coeff[:, occupied_mask].astype(complex)
            self.mo_occ = self.mf.mo_occ[occupied_mask]
        else:
            nocc = np.sum(self.mf.mo_occ > 1e-3)
            num_mos_to_select = nocc + n_virtual
            if num_mos_to_select > self.mf.mo_coeff.shape[1]:
                num_mos_to_select = self.mf.mo_coeff.shape[1]
                print(f"Warning: Requested {nocc + n_virtual} MOs, but only {self.mf.mo_coeff.shape[1]} are available. Using all.")
            self.mo_coeff = self.mf.mo_coeff[:, :num_mos_to_select].astype(complex)
            self.mo_occ = np.zeros(num_mos_to_select, dtype=complex)
            initial_occupied_occ = self.mf.mo_occ[self.mf.mo_occ > 1e-3]
            self.mo_occ[:nocc] = initial_occupied_occ[:nocc]
        self.prev_mo_coeff = self.mo_coeff.copy()

    def initialize_dynamics_data(self, init_velocity):
        """Initializes lists to store time-dependent data."""
        self.times = []
        self.dipoles = []
        self.potential_energies = []
        self.kinetic_energies = []
        self.total_energies = []
        if self.ehrenfest:
            self.positions = self.mol.atom_coords(unit='Bohr')
            self.velocities = init_velocity if init_velocity is not None else np.zeros_like(self.positions)
            if self.velocities.shape != self.positions.shape:
                raise ValueError("Initial velocities shape mismatch with positions.")
            self.force = np.zeros_like(self.positions)
            self.mass = 1822.88839 * np.array(self.mol.atom_mass_list())

    def electric_field(self, t):
        """Calculates the external electric field at time t."""
        if self.field_type == 'gaussian':
            # This implements Eq. (C1) from JCTC 2018, 14, 5645-5652
            # E(t) = E_max * exp(-(t-t0)^2 / (2*sigma^2)) * sin(omega*(t-t0))
            # Note: Our self.tau is sigma (the Gaussian width)
            # The paper's omega is self.omega, t0 is self.t0, E_max is self.E0
            envelope = np.exp(-((t - self.t0) ** 2) / (2 * self.tau ** 2))
            carrier = np.sin(self.omega * (t - self.t0))
            return self.E0 * envelope * carrier
        elif self.field_type == 'sine':
            # Simple continuous sine wave
            return self.E0 * np.sin(self.omega * t)
        
        elif self.field_type == 'delta_kick':
            # This is a numerical approximation of a delta kick
            # E(t) = E0 for one time step 'dt' centered at t0
            # Note: We must ensure t0 is set correctly, e.g., t0 = dt / 2.0
            if abs(t - self.t0) < self.dt / 2.0:
                # The "strength" of the kick is P = E0 * dt.
                # Here we return the field strength E0.
                return self.E0 
            else:
                return 0.0
        
        return 0.0 # 'none' or other types

    def propagate_electrons(self, F_current, E_t, dt=None):
        """Propagates electronic wavefunction using the specified method."""
        if dt is None:
            dt = self.dt
            
        if self.propagation_method == 'predictor_corrector':
            self.predictor_corrector_propagation(F_current, E_t, dt=dt)
        elif self.propagation_method == 'predictor_corrector_diis':
            self.predictor_corrector_propagation_diis(F_current, E_t, dt=dt)
            
        elif self.propagation_method == 'exponential_midpoint':
            self.exponential_midpoint_propagation(F_current, E_t, dt=dt)
        elif self.propagation_method == 'linear_interpolated':
            self.linear_interpolated_exponential_propagation(F_current, E_t, dt=dt, num_substeps=5)
        elif self.propagation_method == 'linear_interpolated_crank_nicolson_diis':
            self.linear_interpolated_crank_nicolson_diis(F_current, E_t, dt=dt, num_substeps=1)
        elif self.propagation_method == 'magnus4':
             self.magnus4_propagation(F_current, E_t, dt=dt)
        elif self.propagation_method == 'accelerated':
             self.predictor_corrector_accelerated(F_current, E_t, dt=dt)
        elif self.propagation_method == 'linear_interpolated_cn_accelerated':
        # You can tune num_substeps here (e.g., 5 or 10)
            self.linear_interpolated_crank_nicolson_accelerated(F_current, E_t, dt=dt, num_substeps=10)
        elif self.propagation_method == 'pt_cn_lowdin':
            self.pt_cn_lowdin_propagation(F_current, E_t, dt=dt)
        else:
            raise ValueError(f"Unknown propagation method: {self.propagation_method}")

    def predictor_corrector_propagation(self, F_current, E_t, dt=None):
        """Performs one step of predictor-corrector propagation for RHF."""
        self.prev_mo_coeff = self.mo_coeff.copy()
        if dt is None: dt = self.dt

        # Predictor
        local_F_scaled = F_current * self.mu
        #local_F_scaled -= self.overlap @ self.make_rdm1(self.mo_coeff, self.mo_occ).real /2. @ local_F_scaled
        #if self.ehrenfest:
        #    local_F_scaled = self.apply_elec_nuc_coupling(local_F_scaled)
        F_t_scaled = local_F_scaled.copy()
        #F_t_scaled -= self.overlap @ self.make_rdm1(self.mo_coeff, self.mo_occ) /2. @ local_F_scaled
        A_pred = self.overlap + 0.5j * dt * F_t_scaled
        B_pred = self.overlap - 0.5j * dt * F_t_scaled
        mo_coeff_predict_iter = np.linalg.solve(A_pred, B_pred @ self.mo_coeff)


        # Corrector
        for iter_count in range(self.max_iterations):
            dm_predict = self.make_rdm1(mo_coeff=mo_coeff_predict_iter, mo_occ=self.mo_occ)
            E_t_plus_dt = self.electric_field(self.t + self.dt)
            F_predict_raw = self.get_fock_matrix(dm_predict, E_t_plus_dt)
            self.F = F_predict_raw.copy()
            F_predict_scaled = F_predict_raw * self.mu
            #F_predict_scaled -= self.overlap @ dm_predict.real/2. @ F_predict_scaled
            #if self.ehrenfest:
            #   F_predict_scaled = self.apply_elec_nuc_coupling(F_predict_scaled)
            
            F_average_scaled = 0.5 * local_F_scaled + 0.5 * F_predict_scaled
            A_corr = self.overlap + 0.5j * dt * F_average_scaled
            B_corr = self.overlap - 0.5j * dt * F_average_scaled
            mo_coeff_corrected = np.linalg.solve(A_corr, B_corr @ self.mo_coeff)

            mo_diff = np.max(np.abs(mo_coeff_corrected - mo_coeff_predict_iter))
            if self.count:
                print(f"  Iteration: {iter_count}, MO diff: {mo_diff:.2e}")
            if mo_diff < self.convergence_threshold:
                mo_coeff_predict_iter = mo_coeff_corrected
                break
            mo_coeff_predict_iter = mo_coeff_corrected
            F_t_scaled = F_predict_scaled.copy()
            if iter_count == self.max_iterations - 1:
                print(f"Warning: Predictor-corrector did not converge in {self.max_iterations} iterations.")
        self.mo_coeff = mo_coeff_predict_iter.copy()

    def get_fock_matrix(self, dm_total=None, E_t=0):
        """Calculates the RHF Fock matrix."""
        if dm_total is None:
            dm_total = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)

        # MODIFICATION: Use the cached core Hamiltonian instead of recalculating it.
        h_core = self.h_core

        # Use last dm and veff to speed up calculation
        self.veff = self.mf.get_veff(self.mol, dm_total.real, dm_last=self.dm_last.real, vhf_last=self.veff_last)
        
        # Update the cached matrices for the next step
        self.dm_last = dm_total.copy()
        self.veff_last = self.veff.copy()

        # External field term
        field_component_map = {'x': 0, 'y': 1, 'z': 2}
        field_idx = field_component_map.get(self.field_direction.lower(), 2)
        dipole_integrals = self.mol.intor('cint1e_r_sph', comp=3)
        field_term = E_t * dipole_integrals[field_idx]

        # Total Fock matrix
        F_total = h_core + self.veff - field_term
        return F_total.astype(complex)

    def make_rdm1(self, mo_coeff, mo_occ):
        """Makes the RHF total density matrix from MO coefficients and occupations."""
        return mo_coeff @ np.diag(mo_occ) @ mo_coeff.conj().T

    def energy_tot(self, dm_total=None):
        """Compute the total electronic energy for RHF."""
        if dm_total is None:
            dm_total = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
        
        # MODIFICATION: Use the cached core Hamiltonian instead of recalculating it.
        h_core = self.h_core
        
        vhf = self.mf.get_veff(self.mol, dm_total.real)

        e1 = np.einsum('ij,ji->', h_core.real, dm_total.real)
        e_coul = 0.5 * np.einsum('ij,ji->', vhf.real, dm_total.conj()).real
        electronic_energy = (e1 + e_coul).real
        return electronic_energy + self.mol.energy_nuc()

    def update_mol(self, positions=None):
        """Updates the PySCF molecule object with current nuclear positions and rebuilds mf."""
        if positions is None:
            positions = self.positions

        atom_coords_str = [f"{self.mol.atom_symbol(i)} {p[0]} {p[1]} {p[2]}" for i, p in enumerate(positions)]
        

        new_mol = gto.M(atom="\n".join(atom_coords_str),
                        basis=self.mol.basis,
                        unit="Bohr",
                        charge=self.mol.charge,
                        spin=self.mol.spin,
                        cart=self.mol.cart)
        if self.mol.ecp is not None:
            new_mol.ecp = self.mol.ecp
        new_mol.build()
        self.mol = new_mol

        #if (self.mo_coeff.shape[0] == 0):
        self.mo_coeff = self.overlap_half @ self.mo_coeff

        if (self.with_df is False):
            self.mf = scf.RHF(self.mol)
        else:
            self.mf = scf.RHF(self.mol).density_fit()
        self.update_overlap()
        #self.mo_coeff = self.orthogonalize_mo_coeff(self.mo_coeff)
        self.mo_coeff = self.overlap_half_inv @ self.mo_coeff

        # MODIFICATION: Recalculate and cache the core Hamiltonian for the new geometry
        self.h_core = self.mf.get_hcore(self.mol)

        # Reset cached density and potential since geometry has changed
        self.dm_last = 0
        self.veff_last = 0

    def run(self):
        """Main simulation loop."""
        with open(self.output_filename, "w") as traj_file:
            if self.ehrenfest:
                self.write_xyz(traj_file, -1)

            # Initial state calculations
            self.t = 0.0
            E_t_initial = self.electric_field(self.t)
            dm_initial = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
            self.F = self.get_fock_matrix(dm_initial, E_t_initial)
            if self.ehrenfest:
                self.calculate_forces()

            for step in range(self.steps):
                self.times.append(self.t)

                if self.ehrenfest:
                    # Update nuclear positions (Velocity Verlet first half)
                    self.velocities += 0.5 * self.nuclei_dt * np.einsum('i,ij->ij', 1. / self.mass, self.force)
                    self.positions += self.velocities * self.nuclei_dt
                    self.update_mol()

                # Calculate properties at time t
                dm_current = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
                dipole_moment = self.calculate_dipole(dm_current)
                self.dipoles.append(dipole_moment)
                
                # Propagate electrons
                E_t_current = self.electric_field(self.t)
                F_current_for_prop = self.get_fock_matrix(dm_current, E_t_current)
                self.propagate_electrons(F_current_for_prop, E_t_current)
                #self.mo_coeff = self.orthogonalize_mo_coeff(self.mo_coeff)

                # Calculate forces and update velocities (Velocity Verlet second half)
                if self.ehrenfest:
                    dm_next_step = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
                    E_t_next_step = self.electric_field(self.t + self.dt)
                    self.F = self.get_fock_matrix(dm_next_step, E_t_next_step)
                    self.calculate_forces()
                    self.velocities += 0.5 * self.nuclei_dt * np.einsum('i,ij->ij', 1. / self.mass, self.force)
                    
                    # Calculate energies
                    potential_energy = self.energy_tot(dm_total=dm_next_step)
                    kinetic_energy = 0.5 * np.sum(self.mass * np.linalg.norm(self.velocities, axis=1)**2)
                    total_energy = potential_energy + kinetic_energy
                    self.potential_energies.append(potential_energy)
                    self.kinetic_energies.append(kinetic_energy)
                    self.total_energies.append(total_energy)
                    if step % 10 == 0:
                        print(f"Step: {step}, Time: {self.t:.2f} a.u., PE: {potential_energy:.6f}, KE: {kinetic_energy:.6f}, Total E: {total_energy:.6f}")
                else:
                    dm_next_step = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
                    potential_energy = self.energy_tot(dm_total=dm_next_step)
                    self.potential_energies.append(potential_energy)
                    print(f"Step: {step}, Time: {self.t:.2f} a.u., Potential Energy: {potential_energy:.6f}")

                if self.ehrenfest and self.output_interval > 0 and step % self.output_interval == 0:
                    self.write_xyz(traj_file, step)
                
                self.t += self.dt

        if self.energy_filename:
            energy_data = {
                "time": np.array(self.times),
                "potential_energy": np.array(self.potential_energies),
                "dipole_moment": np.array(self.dipoles)
            }
            if self.ehrenfest:
                energy_data["kinetic_energy"] = np.array(self.kinetic_energies)
                energy_data["total_energy"] = np.array(self.total_energies)
            np.savez(self.energy_filename, **energy_data)
            print(f"Energy and dipole data saved to {self.energy_filename}")


    # Unchanged utility methods from here...
    def imaginary_time_propagation(self, F_current, dt= None):
        """
        Performs one step of predictor-corrector propagation for RHF.
        F_current is the Fock matrix at the beginning of the step.
        E_t is the electric field at the beginning of the step.
        """
        self.prev_mo_coeff = self.mo_coeff.copy()
        if (dt is None ): dt = 0.1 # Use the default dt if not provided
        F_t_scaled = F_current.copy()
        A_pred = self.overlap + 0.5 * dt * F_t_scaled
        B_pred = self.overlap - 0.5 * dt * F_t_scaled
        mo_coeff_predict_iter = np.linalg.solve(A_pred, B_pred @ self.mo_coeff)
        mo_coeff_predict_iter = self.orthogonalize_mo_coeff(mo_coeff_predict_iter)
        self.mo_coeff = mo_coeff_predict_iter.copy()

    def imaginary_time_propagation_v2(self, F_current, E_0, dt= None):
        """
        Performs one step of predictor-corrector propagation for RHF.
        F_current is the Fock matrix at the beginning of the step.
        E_t is the electric field at the beginning of the step.
        """
        self.prev_mo_coeff = self.mo_coeff.copy()
        if (dt is None ): dt = 10 # Use the default dt if not provided
        F_t_scaled =  E_0 * self.overlap - F_current
        A_pred = F_t_scaled
        B_pred = F_t_scaled - dt * self.overlap
        mo_coeff_predict_iter = np.linalg.solve(A_pred, B_pred @ self.mo_coeff)
        mo_coeff_predict_iter = self.orthogonalize_mo_coeff(mo_coeff_predict_iter)
        self.mo_coeff = mo_coeff_predict_iter.copy()

    def add_penalty(self, F_matrix, penalty_factor=0.05):
        P_occ = np.dot(self.mo_coeff[:, self.mo_occ > 0], self.mo_coeff[:, self.mo_occ > 0].conj().T)
        penalty = penalty_factor * (F_matrix @ (self.overlap_inv-  P_occ) @ F_matrix  )
        F_matrix -= penalty
        return F_matrix

    def apply_elec_nuc_coupling(self, F_matrix):
        """Applies Non-Adiabatic Coupling (NAC) term for Ehrenfest dynamics to the Fock matrix."""
        ipovlp = self.mol.intor('int1e_ipovlp', comp=3)
        aoslices = self.mol.aoslice_by_atom()
        nac_term_matrix = np.zeros_like(F_matrix, dtype=complex)
        for ia in range(self.mol.natm):
            p0, p1, _, _ = aoslices[ia]
            v_ia = self.velocities[ia]
            nac_term_matrix[p0:p1] += 1j * np.einsum('sij,s->ij', ipovlp[:, p0:p1], v_ia)
        return F_matrix - nac_term_matrix.conj().T

    def orthogonalize_mo_coeff(self, mo_coeff=None):
        """Orthogonalizes molecular orbital coefficients using Lowdin orthogonalization."""
        if mo_coeff is  None:
            mo_coeff = self.mo_coeff.copy()
        if False:
        #if (np.sum(self.mo_occ ==1) >0):
            open_shell_index = self.mo_occ == 1
            closed_shell_index = self.mo_occ == 2
            overlap_in_mo_basis_closed = mo_coeff[:, closed_shell_index].conj().T @ self.overlap @ mo_coeff[:, closed_shell_index]
            overlap_in_mo_basis_open =   mo_coeff[:, open_shell_index].conj().T @ self.overlap @  mo_coeff[:, open_shell_index]
            evals_closed, evecs_closed = scipy.linalg.eigh(overlap_in_mo_basis_closed)
            evals_open, evecs_open = scipy.linalg.eigh(overlap_in_mo_basis_open)
            evals_sqrt_inv_closed = np.zeros_like(evals_closed)
            evals_sqrt_inv_open = np.zeros_like(evals_open)
            valid_evals_closed = evals_closed > 1e-12
            valid_evals_open = evals_open > 1e-12
            evals_sqrt_inv_closed[valid_evals_closed] = 1.0 / np.sqrt(evals_closed[valid_evals_closed])
            evals_sqrt_inv_open[valid_evals_open] = 1.0 / np.sqrt(evals_open[valid_evals_open])
            S_inv_sqrt_mo_closed = evecs_closed @ np.diag(evals_sqrt_inv_closed) @ evecs_closed.conj().T
            S_inv_sqrt_mo_open = evecs_open @ np.diag(evals_sqrt_inv_open) @ evecs_open.conj().T
            mo_coeff[:, closed_shell_index] = mo_coeff[:, closed_shell_index] @ S_inv_sqrt_mo_closed
            mo_coeff[:, open_shell_index] =  mo_coeff[:, open_shell_index] @ S_inv_sqrt_mo_open
            return mo_coeff
        else:
            overlap_in_mo_basis = mo_coeff.conj().T @ self.overlap @ mo_coeff
            try:
                S_inv_sqrt_mo = scipy.linalg.inv(scipy.linalg.sqrtm(overlap_in_mo_basis))
                mo_coeff[:] = mo_coeff @ S_inv_sqrt_mo
            except np.linalg.LinAlgError:
                print("Warning: Lowdin orthogonalization of MOs failed. Using pseudoinverse.")
                try:
                    evals, evecs = scipy.linalg.eigh(overlap_in_mo_basis)
                    evals_sqrt_inv = np.zeros_like(evals)
                    valid_evals = evals > 1e-12
                    evals_sqrt_inv[valid_evals] = 1.0 / np.sqrt(evals[valid_evals])
                    S_inv_sqrt_mo = evecs @ np.diag(evals_sqrt_inv) @ evecs.conj().T
                    mo_coeff[:] = mo_coeff @ S_inv_sqrt_mo
                except Exception as e:
                    print(f"Error during MO orthogonalization fallback: {e}.")
            return mo_coeff

    def calculate_dipole(self, dm_total):
        """Calculates the total dipole moment for RHF."""
        dipole_integrals = self.mol.intor('cint1e_r_sph', comp=3)
        dip_x = np.trace(dm_total.real @ dipole_integrals[0])
        dip_y = np.trace(dm_total.real @ dipole_integrals[1])
        dip_z = np.trace(dm_total.real @ dipole_integrals[2])
        return np.array([dip_x, dip_y, dip_z]).real

    def energy_grad(self, F_rhf, dm0_rhf=None):
        if dm0_rhf is None:
            dm0_rhf = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
        
        mf_grad = self.mf.nuc_grad_method()
        mol = self.mol
        hcore_deriv_generator = mf_grad.hcore_generator(mol)
        s1_deriv_ao = mf_grad.get_ovlp(mol)
        vhf_deriv_ao = mf_grad.get_veff(mol, dm0_rhf.real)
        corr = self.overlap_inv @ F_rhf @ dm0_rhf
        atmlst = range(mol.natm)
        de = np.zeros((len(atmlst), 3))
        aoslices = mol.aoslice_by_atom()
        for k_atom_idx, ia_atom_glob_idx in enumerate(atmlst):
            p0, p1 = aoslices [ia_atom_glob_idx,2:]
            h1ao_deriv_atom_k = hcore_deriv_generator(ia_atom_glob_idx)
            de[k_atom_idx] += np.einsum('xij,ij->x', h1ao_deriv_atom_k, dm0_rhf.real)
            de[k_atom_idx] += np.einsum('xij,ij->x', vhf_deriv_ao[:, p0:p1] , dm0_rhf[p0:p1].real) *2
            pulay_term_matrix = corr + corr.conj().T
            de[k_atom_idx] -= np.einsum('xij,ij->x', s1_deriv_ao[:,p0:p1], pulay_term_matrix[p0:p1]).real
            #de[k_atom_idx] += mf_grad.extra_force(ia_atom_glob_idx, locals())
        de += mf_grad.grad_nuc(mol)
        return de

    def energy_constraint(self, dm_total=None, fock_matrix=None):
        """Compute an energy-like constraint for the system."""
        if dm_total is None:
            dm_total = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
        if fock_matrix is None:
            fock_matrix = self.get_fock_matrix(dm_total)
        residue = fock_matrix @ dm_total @ self.overlap
        residue -= residue.conj().T
        return np.linalg.norm(residue)

    def update_electronic_wavefunction(self, mu = None, calculate_force = False, dt= None, apply_scf=False):
        if (apply_scf ):
            potential_energy_next_step = self.apply_scf()
        else:
            if (mu is None):
                mu = self.mu
            self.update_mu(mu)
            if (dt is None): dt = self.dt
            dm_current= self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
            E_t_current = self.electric_field(self.t)
            F_current_for_prop = self.get_fock_matrix(dm_current, E_t_current)
            self.propagate_electrons(F_current_for_prop, E_t_current, dt= dt )
                #self.mo_coeff= self.orthogonalize_mo_coeff(self.mo_coeff)

            if (self.imaginary_time):
                dm_next_step = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
                E_t_current = 0.0
                F_next_step = self.get_fock_matrix(dm_next_step, E_t_current)
                self.F = F_next_step
                self.imaginary_time_propagation(F_next_step, E_t_current)

            dm_next_step = self.make_rdm1(mo_coeff=self.mo_coeff, mo_occ=self.mo_occ)
            F_next_step = self.get_fock_matrix(dm_next_step)
            self.prev_F =  self.overlap_half_inv @ F_next_step @ self.overlap_half_inv
            self.F = F_next_step
            self.dm_total = dm_next_step
            potential_energy_next_step = self.energy_tot(dm_total=dm_next_step)  
        if (not calculate_force):
            return potential_energy_next_step
        else:
            self.calculate_forces()
            return potential_energy_next_step, self.force.copy()

    def update_mu(self, mu):
        """Update the scaling factor mu for the Fock matrix."""
        self.mu = mu
        print(f"Updated mu to {self.mu}.")

    def write_xyz(self, file_handle, step):
        """Writes current nuclear coordinates to an XYZ file."""
        positions_angstrom = self.positions * 0.529177210903
        file_handle.write(f"{len(self.mol.atom_charges())}\n")
        file_handle.write(f"Step {step}, Time {self.t:.4f} a.u.\n")
        for i, pos_ang in enumerate(positions_angstrom):
            symbol = self.mol.atom_symbol(i)
            file_handle.write(f"{symbol} {pos_ang[0]:.8f} {pos_ang[1]:.8f} {pos_ang[2]:.8f}\n")
        file_handle.flush()

    def update_overlap(self):
        """Updates overlap matrix and its inverses if geometry changes."""
        self.overlap = self.mol.intor('int1e_ovlp').astype(complex)
        self.overlap_half_inv_previous = self.overlap_half_inv.copy()
        self.overlap_half_previous = self.overlap_half.copy()

        self.overlap_half_inv, self.overlap_half, self.overlap_inv = lowdin(self.overlap)

    def calculate_forces(self):
        """Calculates nuclear forces using the current electronic state."""
        dm_for_force = self.make_rdm1(self.mo_coeff, self.mo_occ)
        self.F = self.get_fock_matrix(dm_for_force)
        self.force = -self.energy_grad(self.F, dm_for_force)
        #mf_grad = self.mf.nuc_grad_method()
        #self.force = -mf_grad.kernel(self.make_rdm1(self.mo_coeff, self.mo_occ).real)
    def update_dt(self,new_dt):
        """Updates the electronic time step."""
        self.dt = new_dt

    def update_mu(self,new_mu):
        """Updates the mu parameter."""
        self.mu = new_mu
    
    def reset_mo_coeff(self):
        self.mo_coeff = self.prev_mo_coeff.copy()
    
    def reinitialize_simulation(self,mo_coeff, positions=None):
        """Reinitializes the simulation with new MO coefficients and optional nuclear positions."""
        self.mo_coeff = mo_coeff.copy()
        if positions is not None:
            self.positions = positions.copy()
            self.update_mol(self.positions)
        
    apply_scf = apply_scf
    apply_scf_v2 = apply_scf_v2
    exponential_midpoint_propagation = exponential_midpoint_propagation
    linear_interpolated_exponential_propagation = linear_interpolated_exponential_propagation 
    linear_interpolated_crank_nicolson = linear_interpolated_crank_nicolson
    magnus4_propagation = magnus4_propagation
    predictor_corrector_propagation_diis = predictor_corrector_propagation_diis
    linear_interpolated_crank_nicolson_diis = linear_interpolated_crank_nicolson_diis

    predictor_corrector_accelerated = predictor_corrector_accelerated
    linear_interpolated_crank_nicolson_accelerated = linear_interpolated_crank_nicolson_accelerated



if __name__ == "__main__":
    mol = gto.M(
        atom='''
        H 0.0 0.0  0.0 
        F 0.0 0.0  1.0
        ''',
        basis='6-31g'
        #basis='bfd-vtz',
        #ecp = 'bfd-pp',
    )
    
    # 2. 运行基态 SCF
    mf = scf.RHF(mol)
    mf.kernel()
    #mf.mo_occ[4]=0
    #mf.mo_occ[5]=2

    simulation_rhf = TDRHFSimulation(mf, dt=1, nuclei_dt=1,  steps=100, propagation_method="linear_interpolated_cn_accelerated",field_type='none',  mu=10,ehrenfest= True , E0= 0.0,  convergence_threshold=1e-7, output_interval = 10)
    simulation_rhf.run()
    #dipoles = np.array(simulation_rhf.dipoles)
    #plt.plot(simulation_rhf.times, dipoles[:,2])
    #plt.xlabel("Time (a.u.)")
    #plt.ylabel("Dipole Moment (a.u.)")
    #plt.title("Dipole Moment along z-axis in Ehrenfest Dynamics of H2")
    #plt.grid(True)
    #plt.show()
    
    #plt.figure(figsize=(10, 6))
    #plt.plot(simulation_rhf.times, np.array(simulation_rhf.potential_energies) - simulation_rhf.potential_energies[0], label='Potential Energy')
    #plt.plot(simulation_rhf.times, simulation_rhf.kinetic_energies, label='Kinetic Energy')
    plt.plot(simulation_rhf.times, np.array(simulation_rhf.total_energies) - simulation_rhf.total_energies[0], label='Total Energy')
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Energy Change (Hartree)")
    plt.title("Energy Conservation in Ehrenfest Dynamics of H2")
    #plt.legend()
    #plt.grid(True)
    plt.show()