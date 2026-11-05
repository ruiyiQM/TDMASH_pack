"""Parallel-transport Crank--Nicolson propagation at a fixed nuclear geometry."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, eigh, lu_factor, lu_solve, solve_triangular

from .Anderson import AndersonAcceleration


def _preconditioner(S, F, C_start, mu, a):
    """Freeze one spin's Hamiltonian for the grouped occupied-energy solve."""
    if C_start.shape[1] == 0:
        return lambda residual: residual.copy()
    values, vectors = eigh(C_start.conj().T @ S @ C_start)
    reference = C_start @ ((vectors * values**-0.5) @ vectors.conj().T)
    occupied_ham = reference.conj().T @ (mu * F) @ reference
    occupied_ham = 0.5 * (occupied_ham + occupied_ham.conj().T)
    energies, rotation = eigh(occupied_ham)
    groups = [(0, len(energies))]
    if len(energies) > 1 and np.max(np.diff(energies)) > 1e-12 * max(1, np.max(np.abs(energies))):
        split = np.argmax(np.diff(energies)) + 1
        groups = [(0, split), (split, len(energies))]
    while len(groups) < 4:
        spreads = [abs(a) * (energies[end - 1] - energies[start]) for start, end in groups]
        index = int(np.argmax(spreads))
        if spreads[index] <= 1:
            break
        start, end = groups[index]
        split = start + np.argmax(np.diff(energies[start:end])) + 1
        groups[index:index + 1] = [(start, split), (split, end)]
    factors = [lu_factor(S + 1j * a * (mu * F - np.mean(energies[start:end]) * S))
               for start, end in groups]

    def precondition(residual):
        normal = reference.conj().T @ S @ residual
        transformed = rotation.conj().T @ normal @ rotation
        # The occupied Jacobian is real-linear and couples a block to its adjoint.
        hermitian = (transformed + transformed.conj().T) / (
            1 + 1j * a * (energies[:, None] - energies[None, :]))
        occupied = normal + 1j * a * (rotation @ hermitian @ rotation.conj().T) @ occupied_ham
        tangent_rhs = S @ (residual - reference @ normal) @ rotation
        tangent = np.empty_like(tangent_rhs)
        for (start, end), factor in zip(groups, factors):
            tangent[:, start:end] = lu_solve(factor, tangent_rhs[:, start:end])
        tangent -= reference @ (reference.conj().T @ S @ tangent)
        return reference @ occupied + tangent @ rotation.conj().T
    return precondition


def pt_crank_nicolson(self, F_current, E_t, dt=None, unrestricted=False):
    """Solve PT-CN for a closed shell or two coupled unrestricted spin blocks.

    The joint raw AO residual is checked before final orthonormalization.
    Both spin Focks use the same trial alpha/beta density. The caller owns time.
    """
    dt = self.dt if dt is None else dt
    if unrestricted:
        if len(self.mo_coeff) != 2 or len(self.mo_occ) != 2:
            raise ValueError("Unrestricted pt_cn requires alpha and beta blocks.")
        initial = [np.asarray(C).copy() for C in self.mo_coeff]
        occupations = [np.asarray(occ) for occ in self.mo_occ]
        required_occupation = 1
    else:
        initial = [self.mo_coeff.copy()]
        occupations = [self.mo_occ]
        required_occupation = 2
    if self.n_virtual != 0 or any(
            C.ndim != 2 or occ.ndim != 1 or C.shape[1] != len(occ)
            or not np.all(occ == required_occupation)
            for C, occ in zip(initial, occupations)):
        raise ValueError("pt_cn requires closed-shell RHF/RKS or integer-occupied UHF/UKS, with n_virtual=0.")
    nao = initial[0].shape[0]
    if any(C.shape[0] != nao for C in initial) or not any(C.shape[1] for C in initial):
        raise ValueError("pt_cn requires a common AO dimension and at least one occupied orbital.")
    if self.imaginary_time:
        raise ValueError("pt_cn does not support imaginary-time propagation.")
    if not np.isfinite(dt) or not np.isfinite(self.mu) or self.mu <= 0:
        raise ValueError("pt_cn requires a finite dt and a finite positive mu.")
    if (not np.isfinite(self.convergence_threshold) or self.convergence_threshold <= 0
            or not isinstance(self.max_iterations, (int, np.integer)) or self.max_iterations < 1):
        raise ValueError("pt_cn requires a positive tolerance and iteration limit.")
    S = self.overlap
    if (S.shape != (nao, nao) or not np.isfinite(S).all()
            or not np.allclose(S, S.conj().T, rtol=0, atol=1e-10)):
        raise ValueError("pt_cn requires a finite Hermitian overlap matrix.")
    for C in initial:
        if not np.isfinite(C).all() or not np.allclose(
                C.conj().T @ S @ C, np.eye(C.shape[1]), rtol=0, atol=1e-8):
            raise ValueError("pt_cn requires S-orthonormal initial orbitals in each spin block.")

    def fock_blocks(value, exception):
        F = np.asarray(value)
        shape = (2, nao, nao) if unrestricted else (nao, nao)
        if (F.shape != shape or not np.isfinite(F).all()
                or not np.allclose(F, F.conj().swapaxes(-1, -2), rtol=0, atol=1e-10)):
            raise exception("PT-CN requires finite Hermitian Fock matrices with the expected spin shape.")
        return list(F) if unrestricted else [F]

    def orbitals(blocks):
        if not unrestricted:
            return blocks[0]
        return np.stack(blocks) if blocks[0].shape == blocks[1].shape else [C.copy() for C in blocks]

    starts = fock_blocks(F_current, ValueError)
    metric_factor = cho_factor(S)
    if dt == 0:
        return

    def action(C, F):
        if not np.isfinite(C).all():
            raise RuntimeError("PT-CN trial orbitals are not finite.")
        HC = self.mu * F @ C
        return cho_solve(metric_factor, HC) - C @ (C.conj().T @ HC)

    a = 0.5 * dt
    rhs = [C - 1j * a * action(C, F) for C, F in zip(initial, starts)]
    preconditioners = [_preconditioner(S, F, C, self.mu, a) for C, F in zip(initial, starts)]
    sizes = np.cumsum([0] + [C.size for C in initial])

    def flatten(blocks):
        return np.concatenate([C.ravel() for C in blocks])

    def unflatten(vector):
        return [vector[sizes[i]:sizes[i + 1]].reshape(C.shape) for i, C in enumerate(initial)]

    accelerator = AndersonAcceleration(history_size=6, regularization=1e-12)
    blocks = [C.copy() for C in initial]
    E_next = self.electric_field(self.t + dt)
    cache = {key: getattr(self, key) for key in ("dm_last", "veff_last", "veff") if hasattr(self, key)}
    had_veff = hasattr(self, "veff")
    try:
        for iteration in range(1, self.max_iterations + 1):
            F = fock_blocks(self.get_fock_matrix(self.make_rdm1(orbitals(blocks), self.mo_occ), E_next), RuntimeError)
            residuals = [R - C - 1j * a * action(C, H) for R, C, H in zip(rhs, blocks, F)]
            spin_errors = [float(np.linalg.norm(R)) for R in residuals]
            error = np.linalg.norm(spin_errors)
            if not np.isfinite(error):
                raise RuntimeError("PT-CN residual is not finite; electronic state was not advanced.")
            if self.count:
                print(f"PT-CN iteration {iteration}, residual: {error:.3e}")
            if error <= self.convergence_threshold:
                break
            current = flatten(blocks)
            candidate = flatten([C + precondition(R) for C, R, precondition in zip(blocks, residuals, preconditioners)])
            if unrestricted:
                # Relative regularization keeps small coupled spin updates from being damped away.
                accelerator.regularization = 1e-12 * np.linalg.norm(candidate - current)**2
            # Real block mixing respects the real-linear dependence on C and C*.
            mixed = accelerator.compute(
                np.stack((current.real, current.imag)), np.stack((candidate.real, candidate.imag)))
            blocks = unflatten(mixed[0] + 1j * mixed[1])
        else:
            raise RuntimeError(
                f"PT-CN did not converge in {self.max_iterations} iterations "
                f"(residual={error:.3e}); electronic state was not advanced.")

        for i, C in enumerate(blocks):
            if C.shape[1]:
                root = cholesky(C.conj().T @ S @ C, lower=True)
                blocks[i] = solve_triangular(root, C.conj().T, lower=True).conj().T
        accepted = orbitals(blocks)
        F_final = self.get_fock_matrix(self.make_rdm1(accepted, self.mo_occ), E_next)
        fock_blocks(F_final, RuntimeError)
    except Exception:
        for key, value in cache.items():
            setattr(self, key, value)
        if not had_veff and hasattr(self, "veff"):
            del self.veff
        raise

    self.prev_mo_coeff = orbitals(initial)
    self.mo_coeff = accepted
    self.F = F_final
    self.pt_cn_residual = error
    self.pt_cn_iterations = iteration
    if unrestricted:
        self.pt_cn_spin_residuals = tuple(spin_errors)
