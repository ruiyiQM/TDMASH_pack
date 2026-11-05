"""PT-CN checks against independent finite-dimensional quantum dynamics."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from tdmash.TDRHF import TDRHFSimulation, lowdin


def model_simulation(method="pt_cn", nocc=1, nonlinear=False):
    """Use the production dispatch with a small nonorthogonal AO Hamiltonian."""
    sim = TDRHFSimulation.__new__(TDRHFSimulation)
    sim.overlap = np.array([[1.2, 0.1, 0.0], [0.1, 0.9, 0.05], [0.0, 0.05, 1.1]], complex)
    sim.overlap_half_inv, sim.overlap_half, sim.overlap_inv = lowdin(sim.overlap)
    H = np.array([[-0.7, 0.15, 0.03j], [0.15, 0.2, 0.12], [-0.03j, 0.12, 0.8]])
    sim.h_model = sim.overlap_half @ H @ sim.overlap_half
    sim.mo_coeff = sim.overlap_half_inv[:, :nocc].copy()
    sim.prev_mo_coeff = sim.mo_coeff.copy()
    sim.mo_occ = np.full(nocc, 2.0)
    sim.n_virtual = 0
    sim.ehrenfest = False
    sim.imaginary_time = False
    sim.propagation_method = method
    sim.dt = 0.1
    sim.t = 0.0
    sim.mu = 1.0
    sim.count = False
    sim.convergence_threshold = 1e-12
    sim.max_iterations = 100
    sim.dm_last = 0
    sim.veff_last = 0
    sim.F = None
    sim.electric_field = lambda t: 0.0

    def fock(dm, field=0.0):
        sim.dm_last = dm.copy()
        sim.veff_last = np.zeros_like(dm)
        sim.veff = sim.veff_last.copy()
        potential = 0.1 * sim.overlap @ np.diag(dm.diagonal().real) @ sim.overlap if nonlinear else 0
        return sim.h_model * (1 + field) + potential

    sim.get_fock_matrix = fock
    return sim


def advance(sim, dt, stop):
    for _ in range(round(stop / dt)):
        field = sim.electric_field(sim.t)
        F = sim.get_fock_matrix(sim.make_rdm1(sim.mo_coeff, sim.mo_occ), field)
        sim.propagate_electrons(F, field, dt=dt)
        sim.t += dt
    return sim.make_rdm1(sim.mo_coeff, sim.mo_occ)


class PTCNTests(unittest.TestCase):
    def test_density_converges_at_second_order(self):
        for mu in (1.0, 4.0):
            errors = []
            for dt in (0.1, 0.05, 0.025):
                sim = model_simulation(nocc=2)
                sim.mu = mu
                U = expm(-1j * mu * sim.overlap_inv @ sim.h_model * 0.4)
                exact = U @ sim.make_rdm1(sim.mo_coeff, sim.mo_occ) @ U.conj().T
                actual = advance(sim, dt, 0.4)
                errors.append(np.linalg.norm(actual - exact))
                assert_allclose(sim.mo_coeff.conj().T @ sim.overlap @ sim.mo_coeff, np.eye(2), atol=2e-13)
                self.assertLessEqual(sim.pt_cn_residual, sim.convergence_threshold)
                self.assertAlmostEqual(np.trace(actual @ sim.overlap).real, 4.0, places=12)
            with self.subTest(mu=mu, errors=errors):
                self.assertGreater(errors[0] / errors[1], 3.7)
                self.assertGreater(errors[1] / errors[2], 3.7)

    def test_stationary_occupied_subspace_has_no_fast_phase(self):
        sim = model_simulation(nocc=2)
        H_orth = sim.overlap_half_inv @ sim.h_model @ sim.overlap_half_inv
        _, vectors = np.linalg.eigh(H_orth)
        sim.mo_coeff = sim.overlap_half_inv @ vectors[:, :2]
        initial = sim.mo_coeff.copy()
        sim.mu = 80.0
        advance(sim, 2.0, 2.0)
        assert_allclose(sim.mo_coeff, initial, atol=2e-13)
        self.assertEqual(sim.pt_cn_iterations, 1)

    def test_occupied_rotation_preserves_density(self):
        a = model_simulation(nocc=2)
        b = model_simulation(nocc=2)
        rotation = expm(1j * np.array([[0.2, 0.7j], [-0.7j, -0.3]]))
        b.mo_coeff = b.mo_coeff @ rotation
        assert_allclose(advance(a, 0.05, 0.4), advance(b, 0.05, 0.4), atol=2e-10)

    def test_time_dependent_field_uses_override_dt(self):
        sim = model_simulation()
        seen = []
        sim.electric_field = lambda t: seen.append(t) or t
        sim.dt = 9.0
        U = expm(-1j * sim.overlap_inv @ sim.h_model * (0.2 + 0.5 * 0.2**2))
        exact = U @ sim.make_rdm1(sim.mo_coeff, sim.mo_occ) @ U.conj().T
        actual = advance(sim, 0.01, 0.2)
        assert_allclose(actual, exact, atol=2e-6)
        self.assertAlmostEqual(max(seen), 0.2)

    def test_nonlinear_density_matches_schrodinger_reference(self):
        sim = model_simulation(nonlinear=True)
        initial = sim.mo_coeff.copy()

        def derivative(t, flat):
            C = flat.reshape(initial.shape)
            D = sim.make_rdm1(C, sim.mo_occ)
            F = sim.h_model + 0.1 * sim.overlap @ np.diag(D.diagonal().real) @ sim.overlap
            return (-1j * sim.overlap_inv @ F @ C).ravel()

        reference = solve_ivp(derivative, (0, 0.2), initial.ravel(), rtol=1e-12, atol=1e-14)
        self.assertTrue(reference.success)
        C_ref = reference.y[:, -1].reshape(initial.shape)
        assert_allclose(advance(sim, 0.01, 0.2), sim.make_rdm1(C_ref, sim.mo_occ), atol=2e-6)
        assert_allclose(sim.F, sim.get_fock_matrix(sim.make_rdm1(sim.mo_coeff, sim.mo_occ)), atol=1e-14)

    def test_forward_then_backward_preserves_density(self):
        sim = model_simulation()
        initial = sim.make_rdm1(sim.mo_coeff, sim.mo_occ)
        advance(sim, 0.02, 0.2)
        advance(sim, -0.02, -0.2)
        assert_allclose(sim.make_rdm1(sim.mo_coeff, sim.mo_occ), initial, atol=2e-9)

    def test_zero_dt_is_noop(self):
        sim = model_simulation()
        initial = sim.mo_coeff.copy()
        sim.get_fock_matrix = lambda *args: self.fail("Zero dt must not rebuild the Fock matrix.")
        sim.propagate_electrons(sim.h_model, 0, dt=0.0)
        assert_array_equal(sim.mo_coeff, initial)
        self.assertIsNone(sim.F)

    def test_failure_preserves_electronic_state_and_cache(self):
        for failure in ("iteration_limit", "nonfinite", "nonhermitian", "backend_error"):
            with self.subTest(failure=failure):
                sim = model_simulation()
                initial = sim.mo_coeff.copy()
                previous = sim.prev_mo_coeff.copy()
                sim.F = sim.h_model.copy()
                old_F = sim.F.copy()
                sim.dm_last = initial.copy()
                sim.veff_last = np.ones_like(initial)
                old_dm, old_veff = sim.dm_last.copy(), sim.veff_last.copy()
                if failure == "iteration_limit":
                    sim.max_iterations = 1
                elif failure == "nonfinite":
                    sim.get_fock_matrix = lambda *args: np.full((3, 3), np.nan)
                elif failure == "nonhermitian":
                    F = sim.h_model.copy()
                    F[0, 1] += 0.1
                    sim.get_fock_matrix = lambda *args: F
                else:
                    def broken(*args):
                        sim.dm_last = 42
                        raise RuntimeError("Backend failed.")
                    sim.get_fock_matrix = broken
                with self.assertRaises(RuntimeError):
                    sim.propagate_electrons(sim.h_model, 0)
                assert_array_equal(sim.mo_coeff, initial)
                assert_array_equal(sim.prev_mo_coeff, previous)
                assert_array_equal(sim.F, old_F)
                assert_array_equal(sim.dm_last, old_dm)
                assert_array_equal(sim.veff_last, old_veff)
                self.assertFalse(hasattr(sim, "veff"))
                self.assertEqual(sim.t, 0.0)

    def test_unsupported_states_and_invalid_parameters_are_rejected(self):
        cases = [
            ("mo_occ", np.array([1.])), ("mo_occ", np.array([0.5])),
            ("mo_coeff", np.zeros((2, 3, 1))), ("n_virtual", 1),
            ("imaginary_time", True), ("mu", np.nan), ("mu", 0),
            ("dt", np.nan), ("convergence_threshold", 0), ("max_iterations", 0),
            ("overlap", np.diag([1., -1., 1.])),
        ]
        for key, value in cases:
            with self.subTest(parameter=key, value=value):
                sim = model_simulation()
                setattr(sim, key, value)
                with self.assertRaises((ValueError, np.linalg.LinAlgError)):
                    sim.propagate_electrons(sim.h_model, 0)

    def test_nonhermitian_hamiltonian_is_rejected(self):
        sim = model_simulation()
        F = sim.h_model.copy()
        F[0, 1] += 0.1
        with self.assertRaisesRegex(ValueError, "Hermitian"):
            sim.propagate_electrons(F, 0)


if __name__ == "__main__":
    unittest.main()
