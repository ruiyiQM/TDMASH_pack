"""Moving-basis, force and symmetric Ehrenfest regression checks."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from pyscf import dft, gto, lib, scf
from scipy.linalg import expm

from tdmash import TDRHFSimulation, TDRKSSimulation


class MovingBasisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = lib.num_threads()
        lib.num_threads(1)
        cls.mean_fields = []
        for method in (scf.RHF, dft.RKS):
            mol = gto.M(atom="O 0 0 0; H 1.4 0 0.9; H -1.2 0 1.1", unit="Bohr",
                        basis="sto-3g", verbose=0)
            mf = method(mol)
            if isinstance(mf, dft.rks.KohnShamDFT):
                mf.xc = "lda,vwn"
                mf.grids.level = 0
                mf.grids.prune = None
                mf.small_rho_cutoff = 0
            mf.conv_tol = 1e-12
            mf.kernel()
            if not mf.converged:
                raise RuntimeError("Moving-basis regression SCF did not converge.")
            cls.mean_fields.append(mf)

    @classmethod
    def tearDownClass(cls):
        lib.num_threads(cls.old_threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def simulation(self, index=0, dt=0.1, steps=3, method="pt_cn"):
        cls = (TDRHFSimulation, TDRKSSimulation)[index]
        return cls(self.mean_fields[index], dt=dt, nuclei_dt=dt, steps=steps, ehrenfest=True,
                   propagation_method=method, convergence_threshold=1e-11, max_iterations=120,
                   initial_velocity=np.array([[0., 0., 0.], [8e-4, 0., 2e-4], [-5e-4, 1e-4, -3e-4]]),
                   output_filename=f"{self.directory.name}/trajectory.xyz")

    def perturb(self, sim):
        mf = sim.mf
        nocc = sim.mo_coeff.shape[1]
        generator = np.zeros((mf.mo_coeff.shape[1],) * 2, complex)
        generator[0, nocc] = 0.12 + 0.04j
        generator[nocc, 0] = -generator[0, nocc].conjugate()
        sim.mo_coeff = mf.mo_coeff @ expm(generator)[:, :nocc]

    def test_transport_roundtrip_preserves_orthogonal_orbitals_and_backend(self):
        for index in (0, 1):
            with self.subTest(backend=index):
                sim = self.simulation(index)
                self.perturb(sim)
                original_mf = sim.mf
                original_mol = sim.mol
                original = sim.mo_coeff.copy()
                positions = sim.positions.copy()
                X = sim.overlap_half @ original
                for shift in (0.03, -0.02, 0.0):
                    moved = positions.copy()
                    moved[1, 0] += shift
                    sim.update_mol(moved)
                    assert_allclose(sim.overlap_half @ sim.mo_coeff, X, atol=3e-13)
                    assert_allclose(sim.mo_coeff.conj().T @ sim.overlap @ sim.mo_coeff, np.eye(5), atol=3e-13)
                    D = sim.make_rdm1(sim.mo_coeff, sim.mo_occ)
                    self.assertAlmostEqual(np.trace(D @ sim.overlap).real, 10.0, places=11)
                    self.assertIsNone(sim.F)
                assert_allclose(sim.mo_coeff, original, atol=3e-13)
                assert_allclose(original_mol.atom_coords(), positions, atol=1e-14)
                self.assertIs(original_mf.mol, original_mol)
                self.assertEqual(sim.mf.conv_tol, original_mf.conv_tol)
                if index == 1:
                    self.assertEqual(sim.mf.grids.level, 0)
                    self.assertIsNone(sim.mf.grids.prune)
                    self.assertIsNot(sim.mf.grids, original_mf.grids)
                    self.assertIsNotNone(original_mf.grids.coords)

    def test_lowdin_force_matches_energy_finite_difference_off_scf(self):
        for index in (0, 1):
            with self.subTest(backend=index):
                sim = self.simulation(index)
                self.perturb(sim)
                positions = sim.positions.copy()
                X = sim.overlap_half @ sim.mo_coeff
                D = sim.make_rdm1(sim.mo_coeff, sim.mo_occ)
                F = sim.get_fock_matrix(D)
                analytical = sim.energy_grad(F, D, lowdin=True)
                legacy = sim.energy_grad(F, D)
                numerical = np.zeros_like(positions)
                eps = 1e-4
                for atom in range(sim.mol.natm):
                    for axis in range(3):
                        energies = []
                        for sign in (1, -1):
                            displaced = positions.copy()
                            displaced[atom, axis] += sign * eps
                            sim.update_mol(displaced)
                            values, vectors = np.linalg.eigh(sim.overlap)
                            Q = (vectors * values**-0.5) @ vectors.conj().T
                            sim.mo_coeff = Q @ X
                            energies.append(sim.energy_tot())
                        numerical[atom, axis] = (energies[0] - energies[1]) / (2 * eps)
                assert_allclose(analytical, numerical, atol=2e-6, rtol=2e-6)
                self.assertGreater(np.linalg.norm(legacy - numerical), 1e-4)

    def test_geometry_failure_does_not_modify_state(self):
        sim = self.simulation()
        C, positions, mol, mf = sim.mo_coeff.copy(), sim.positions.copy(), sim.mol, sim.mf
        for invalid in (np.zeros((2, 3)), np.full((3, 3), np.nan), np.zeros((3, 3))):
            with self.subTest(geometry=invalid):
                with self.assertRaises(ValueError):
                    sim.update_mol(invalid)
                assert_array_equal(sim.mo_coeff, C)
                assert_array_equal(sim.positions, positions)
                self.assertIs(sim.mol, mol)
                self.assertIs(sim.mf, mf)

    def test_synchronized_endpoint_history_for_both_backends_and_methods(self):
        for index in (0, 1):
            for method in ("pt_cn", "accelerated", "predictor_corrector"):
                with self.subTest(backend=index, method=method):
                    sim = self.simulation(index, method=method)
                    if method == 'pt_cn':
                        sim.run()
                    else:
                        sim.run_ehrenfest_midpoint_decoupled()
                    assert_allclose(sim.times, [0, .1, .2, .3], atol=1e-15)
                    self.assertEqual(len(sim.total_energies), 4)
                    assert_allclose(sim.positions_history[-1], sim.mol.atom_coords(), atol=1e-14)
                    assert_allclose(sim.forces_history[-1], sim.force, atol=1e-14)
                    self.assertLess(np.ptp(sim.total_energies), 2e-6)
                    assert_allclose(sim.mo_coeff.conj().T @ sim.overlap @ sim.mo_coeff, np.eye(5), atol=1e-12)

    def test_failed_electronic_step_rolls_back_nuclear_state(self):
        for method in ('pt_cn', 'accelerated'):
            with self.subTest(method=method):
                sim = self.simulation(method=method)
                C, R, V = sim.mo_coeff.copy(), sim.positions.copy(), sim.velocities.copy()
                sim.max_iterations = 1
                with self.assertRaisesRegex(RuntimeError, 'converge'):
                    sim.run_ehrenfest_midpoint_decoupled()
                assert_array_equal(sim.mo_coeff, C)
                assert_array_equal(sim.positions, R)
                assert_array_equal(sim.velocities, V)
                assert_allclose(sim.mol.atom_coords(), R, atol=1e-14)
                self.assertEqual(sim.t, 0)
                self.assertEqual(len(sim.times), 1)

    def test_unsupported_driver_inputs_fail_before_writing_output(self):
        for key, value in (("mu", -1), ("nuclei_dt", .15), ("dt", -1),
                           ("field_type", "sine"), ("with_df", True),
                           ("imaginary_time", True), ("steps", -1),
                           ("mo_occ", np.ones(5))):
            with self.subTest(parameter=key):
                sim = self.simulation()
                setattr(sim, key, value)
                with self.assertRaises(ValueError):
                    sim.run_ehrenfest_midpoint_decoupled()
                self.assertEqual(sim.t, 0)
                self.assertFalse(Path(sim.output_filename).exists())

    def test_short_trajectory_converges_with_dt(self):
        results = []
        for dt in (.4, .2, .1, .025):
            sim = self.simulation(dt=dt, steps=round(1.6 / dt))
            sim.run_ehrenfest_midpoint_decoupled()
            results.append(np.asarray(sim.positions))
        errors = [np.linalg.norm(R - results[-1]) for R in results[:-1]]
        self.assertGreater(errors[0] / errors[1], 3.5, errors)
        self.assertGreater(errors[1] / errors[2], 3.5, errors)


if __name__ == "__main__":
    unittest.main()
