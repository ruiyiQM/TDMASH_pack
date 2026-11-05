"""Unrestricted Lowdin forces and scaled single-state Ehrenfest regression."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from pyscf import dft, gto, lib, scf
from scipy.linalg import expm
from scipy.integrate import solve_ivp

from tdmash import TDUHFSimulation, TDUKSSimulation


class UnrestrictedMovingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = lib.num_threads()
        cls.addClassCleanup(lib.num_threads, cls.old_threads)
        lib.num_threads(1)
        cls.backends = []
        for method, simulation in ((scf.UHF, TDUHFSimulation), (dft.UKS, TDUKSSimulation)):
            mol = gto.M(atom='O 0 0 0; H 1.4 0 .9; H -1.2 0 1.1', unit='Bohr', basis='sto-3g', verbose=0)
            mf = method(mol)
            cls.addClassCleanup(mf._chkfile.close)
            mf.chkfile = None
            if method is dft.UKS:
                mf.xc, mf.grids.level, mf.grids.prune, mf.small_rho_cutoff = 'lda,vwn', 0, None, 0
            mf.conv_tol = 1e-12
            mf.kernel()
            if not mf.converged:
                raise RuntimeError('Unrestricted moving-basis reference did not converge.')
            cls.backends.append((simulation, mf))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def simulation(self, index, dt=.05, steps=2, mu=1, method='pt_cn', nuclei_dt=None):
        simulation, mf = self.backends[index]
        return simulation(mf, dt=dt, nuclei_dt=nuclei_dt, steps=steps, mu=mu, ehrenfest=True,
                          propagation_method=method, convergence_threshold=1e-11, max_iterations=200,
                          initial_velocity=np.array([[0., 0., 0.], [8e-4, 0., 2e-4], [-5e-4, 1e-4, -3e-4]]),
                          output_filename=str(Path(self.directory.name) / 'trajectory.xyz'))

    def perturb(self, sim):
        for spin in (0, 1):
            generator = np.zeros((sim.mf.mo_coeff.shape[1],) * 2, complex)
            generator[spin, 5 + spin] = .07 + .025j * (spin + 1)
            generator[5 + spin, spin] = -generator[spin, 5 + spin].conjugate()
            sim.mo_coeff[spin] = (sim.mf.mo_coeff[spin] @ expm(generator))[:, :5]

    def test_transport_roundtrip_preserves_each_spin_and_backend_settings(self):
        for index in (0, 1):
            sim = self.simulation(index)
            self.perturb(sim)
            original, mol, mf = sim.mo_coeff.copy(), sim.mol, sim.mf
            positions = sim.positions.copy()
            orthogonal = [sim.overlap_half @ C for C in original]
            for shift in (.02, -.01, 0):
                moved = positions.copy()
                moved[1, 0] += shift
                sim.update_mol(moved)
                for C, X in zip(sim.mo_coeff, orthogonal):
                    assert_allclose(sim.overlap_half @ C, X, atol=4e-13)
                    assert_allclose(C.conj().T @ sim.overlap @ C, np.eye(5), atol=5e-13)
                self.assertIsNone(sim.F)
            assert_allclose(sim.mo_coeff, original, atol=5e-13)
            self.assertIs(mf.mol, mol)
            assert_array_equal(mol.atom_coords(), positions)
            self.assertEqual(sim.mf.conv_tol, mf.conv_tol)
            if index == 1:
                self.assertEqual(sim.mf.grids.level, 0)
                self.assertIsNone(sim.mf.grids.prune)
                self.assertIsNot(sim.mf.grids, mf.grids)

    def test_spin_summed_lowdin_force_matches_off_scf_energy_derivatives(self):
        for index in (0, 1):
            with self.subTest(backend=index):
                sim = self.simulation(index)
                self.perturb(sim)
                positions = sim.positions.copy()
                X = [sim.overlap_half @ C for C in sim.mo_coeff]
                dm = sim.make_rdm1(sim.mo_coeff, sim.mo_occ)
                F = sim.get_fock_matrix(dm)
                actual = sim.energy_grad(F, dm, lowdin=True)
                legacy = sim.energy_grad(F, dm)
                expected = np.zeros_like(positions)
                for atom in range(3):
                    for axis in range(3):
                        values = []
                        for sign in (1, -1):
                            displaced = positions.copy()
                            displaced[atom, axis] += sign * 1e-4
                            sim.update_mol(displaced)
                            sim.mo_coeff = np.array([sim.overlap_half_inv @ C for C in X])
                            values.append(sim.energy_tot())
                        expected[atom, axis] = (values[0] - values[1]) / 2e-4
                assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
                self.assertGreater(np.linalg.norm(legacy - expected), 1e-4)

    def test_lowdin_force_matches_native_scf_gradient_at_stationarity(self):
        for index in (0, 1):
            sim = self.simulation(index)
            gradient = sim.mf.nuc_grad_method()
            if hasattr(gradient, 'grid_response'):
                gradient.grid_response = True
            expected = gradient.kernel()
            sim.calculate_forces(lowdin=True)
            assert_allclose(-sim.force, expected, atol=2e-6)

    def test_complex_fock_and_energy_match_native_backend_including_exchange(self):
        for index in (0, 1):
            sim = self.simulation(index)
            self.perturb(sim)
            dm = sim.make_rdm1(sim.mo_coeff, sim.mo_occ)
            expected = sim.mf.get_hcore() + sim.mf.get_veff(dm=dm)
            assert_allclose(sim.get_fock_matrix(dm), expected, atol=2e-12)
            assert_allclose(sim.energy_tot(dm), sim.mf.energy_tot(dm=dm), atol=2e-11)
            self.assertGreater(np.linalg.norm(dm.imag), 1e-3)

    def test_excited_cn_and_pt_match_independent_coupled_schrodinger_ode(self):
        duration = .2
        for index in (0, 1):
            reference = self.simulation(index)
            reference.mo_coeff[0, :, -1] = reference.mf.mo_coeff[0, :, 5]
            C0 = reference.mo_coeff.copy()
            X0 = reference.overlap_half @ C0
            nR = reference.positions.size
            y0 = np.concatenate((reference.positions.ravel(), reference.velocities.ravel(), X0.ravel()))

            def derivative(t, vector):
                R = vector[:nR].real.reshape(-1, 3)
                V = vector[nR:2*nR].real.reshape(-1, 3)
                X = vector[2*nR:].reshape(X0.shape)
                reference.update_mol(R)
                Q = reference.overlap_half_inv
                reference.mo_coeff = Q @ X
                dm = reference.make_rdm1(reference.mo_coeff, reference.mo_occ)
                F = reference.get_fock_matrix(dm)
                force = -reference.energy_grad(F, dm, lowdin=True)
                dX = -1j * (Q @ F @ Q) @ X
                return np.concatenate((V.ravel(), (force / reference.mass[:, None]).ravel(), dX.ravel()))

            solution = solve_ivp(derivative, (0, duration), y0.astype(complex), method='DOP853', rtol=1e-11, atol=1e-13)
            self.assertTrue(solution.success)
            expected_R = solution.y[:nR, -1].real.reshape(-1, 3)
            expected_X = solution.y[2*nR:, -1].reshape(X0.shape)
            expected_density = expected_X @ expected_X.conj().swapaxes(-1, -2)
            for method in ('accelerated', 'pt_cn'):
                errors = []
                for dt in (.02, .01, .005):
                    sim = self.simulation(index, dt=dt, steps=round(duration / dt), method=method)
                    sim.mo_coeff = C0.copy()
                    sim.run_ehrenfest_midpoint_decoupled()
                    X = sim.overlap_half @ sim.mo_coeff
                    errors.append(float(np.linalg.norm(X @ X.conj().swapaxes(-1, -2) - expected_density)))
                    assert_allclose(sim.positions, expected_R, atol=2e-7)
                with self.subTest(backend=index, method=method, errors=errors):
                    self.assertGreater(errors[0] / errors[1], 3.5)
                    self.assertGreater(errors[1] / errors[2], 3.5)
                    self.assertLess(errors[-1], 2e-5)

    def test_hybrid_complex_exchange_force_matches_native_energy_derivative(self):
        for xc in ('pbe0', 'camb3lyp'):
            sim = self.simulation(1)
            sim.mf = sim.mf.copy()
            sim.mf.xc = xc
            self.perturb(sim)
            dm = sim.make_rdm1(sim.mo_coeff, sim.mo_occ)
            F = sim.get_fock_matrix(dm)
            assert_allclose(F, sim.mf.get_hcore() + sim.mf.get_veff(dm=dm), atol=2e-12)
            assert_allclose(sim.energy_tot(dm), sim.mf.energy_tot(dm=dm), atol=2e-11)
            gradient = sim.energy_grad(F, dm, lowdin=True)
            R, X = sim.positions.copy(), sim.overlap_half @ sim.mo_coeff
            energies = []
            for displacement in (1e-4, -1e-4):
                moved = R.copy()
                moved[1, 0] += displacement
                sim.update_mol(moved)
                sim.mo_coeff = sim.overlap_half_inv @ X
                density = sim.make_rdm1(sim.mo_coeff, sim.mo_occ)
                energies.append(sim.mf.energy_tot(dm=density))
            with self.subTest(xc=xc):
                assert_allclose(gradient[1, 0], (energies[0] - energies[1]) / 2e-4, atol=4e-6)

    def test_scaled_driver_keeps_synchronized_histories_and_joint_rollback(self):
        for index in (0, 1):
            for method in ('pt_cn', 'accelerated'):
                with self.subTest(backend=index, method=method):
                    sim = self.simulation(index, mu=4, method=method)
                    sim.run_ehrenfest_midpoint_decoupled()
                    assert_allclose(sim.times, [0, .05, .1], atol=1e-15)
                    self.assertLess(np.ptp(sim.total_energies), 2e-6)
                    assert_allclose(sim.positions_history[-1], sim.mol.atom_coords(), atol=1e-14)
                    for C in sim.mo_coeff:
                        assert_allclose(C.conj().T @ sim.overlap @ C, np.eye(5), atol=1e-12)
                    failed = self.simulation(index, mu=4, method=method)
                    C, R, V, mf = failed.mo_coeff.copy(), failed.positions.copy(), failed.velocities.copy(), failed.mf
                    failed.max_iterations = 1
                    with self.assertRaisesRegex(RuntimeError, 'converge'):
                        failed.run_ehrenfest_midpoint_decoupled()
                    assert_array_equal(failed.mo_coeff, C)
                    assert_array_equal(failed.positions, R)
                    assert_array_equal(failed.velocities, V)
                    self.assertIs(failed.mf, mf)
                    self.assertEqual(len(failed.times), 1)
                    self.assertEqual(failed.t, 0)

    def test_scaled_nuclear_trajectory_converges_at_second_order(self):
        for electronic_steps in (1, 2):
            results = []
            for dt_N in (.2, .1, .05, .0125):
                sim = self.simulation(0, dt=dt_N / electronic_steps, nuclei_dt=dt_N,
                                      steps=round(.8 / dt_N), mu=4)
                # Resolve a valence excitation rather than an unresolved oxygen-core oscillation.
                for spin in (0, 1):
                    generator = np.zeros((7, 7), complex)
                    generator[4, 5 + spin] = .07 + .025j
                    generator[5 + spin, 4] = -generator[4, 5 + spin].conjugate()
                    sim.mo_coeff[spin] = (sim.mf.mo_coeff[spin] @ expm(generator))[:, :5]
                sim.run_ehrenfest_midpoint_decoupled()
                results.append(sim.positions.copy())
            errors = [np.linalg.norm(R - results[-1]) for R in results[:-1]]
            self.assertGreater(errors[0] / errors[1], 3.3, errors)
            self.assertGreater(errors[1] / errors[2], 3.3, errors)

    def test_electronic_substeps_keep_midpoint_geometry_and_physical_time(self):
        for index in (0, 1):
            for method in ('pt_cn', 'accelerated'):
                sim = self.simulation(index, mu=4, method=method, nuclei_dt=.1)
                propagate = sim.propagate_electrons
                calls = []

                def observed(F, E, dt=None):
                    if dt:
                        calls.append((sim.t, dt, sim.positions.copy()))
                    return propagate(F, E, dt=dt)

                with patch.object(sim, 'propagate_electrons', side_effect=observed):
                    sim.run_ehrenfest_midpoint_decoupled()
                assert_allclose(sim.times, [0, .1, .2], atol=1e-15)
                assert_allclose([c[:2] for c in calls], [[0, .05], [.05, .05], [.1, .05], [.15, .05]], atol=1e-15)
                for pair in (calls[:2], calls[2:]):
                    assert_array_equal(pair[0][2], pair[1][2])
                assert_allclose(sim.positions_history[-1], sim.mol.atom_coords(), atol=1e-14)
                self.assertLess(np.ptp(sim.total_energies), 2e-6)

    def test_second_electronic_substep_failure_restores_the_whole_nuclear_step(self):
        for index in (0, 1):
            sim = self.simulation(index, mu=4, nuclei_dt=.1)
            initial = (sim.mo_coeff.copy(), sim.positions.copy(), sim.velocities.copy(), sim.mf)
            propagate = sim.propagate_electrons
            calls = []

            def fail_second(F, E, dt=None):
                if dt:
                    calls.append(dt)
                    if len(calls) == 2:
                        raise RuntimeError('Injected second electronic substep failure.')
                return propagate(F, E, dt=dt)

            with patch.object(sim, 'propagate_electrons', side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, 'second electronic substep'):
                    sim.run_ehrenfest_midpoint_decoupled()
            self.assertEqual(len(calls), 2)
            for value, expected in zip((sim.mo_coeff, sim.positions, sim.velocities), initial[:3]):
                assert_array_equal(value, expected)
            self.assertIs(sim.mf, initial[3])
            self.assertEqual(sim.t, 0)
            self.assertEqual(len(sim.times), 1)

    def test_nonintegral_or_invalid_nuclear_steps_are_rejected(self):
        for dt_N in (.025, .075, 0, -1, np.inf, np.nan):
            sim = self.simulation(0)
            sim.nuclei_dt = dt_N
            with self.assertRaises(ValueError):
                sim.run_ehrenfest_midpoint_decoupled()
            self.assertFalse(Path(sim.output_filename).exists())

    def test_unequal_populations_and_empty_spin_follow_a_translating_atom(self):
        for method, simulation in ((scf.UHF, TDUHFSimulation), (dft.UKS, TDUKSSimulation)):
            for atom in ('Li', 'H'):
                mol = gto.M(atom=f'{atom} 0 0 0', basis='6-31g', spin=1, verbose=0)
                mf = method(mol)
                self.addCleanup(mf._chkfile.close)
                mf.chkfile = None
                if method is dft.UKS:
                    mf.xc, mf.grids.level = 'lda,vwn', 0
                mf.conv_tol = 1e-11
                mf.kernel()
                self.assertTrue(mf.converged)
                velocity = np.array([[1e-4, 0., 0.]])
                sim = simulation(mf, dt=.1, nuclei_dt=.1, steps=2, ehrenfest=True,
                                 propagation_method='pt_cn', initial_velocity=velocity.copy(),
                                 convergence_threshold=1e-10, max_iterations=200,
                                 output_filename=str(Path(self.directory.name) / 'atom.xyz'))
                sim.run()
                assert_allclose(sim.positions, velocity * .2, atol=1e-11)
                for s, n in enumerate(mol.nelec):
                    C = sim.mo_coeff[s]
                    self.assertEqual(C.shape[1], n)
                    assert_allclose(C.conj().T @ sim.overlap @ C, np.eye(n), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
