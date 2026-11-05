"""Regression coverage for electronic dispatch and real PySCF backends."""

import contextlib
import io
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose
from pyscf import dft, gto, lib, scf
from scipy.linalg import expm

from tdmash import TDRHFSimulation, TDRKSSimulation
from test_pt_cn import advance, model_simulation


class ExistingPropagationTests(unittest.TestCase):
    def test_existing_cn_methods_preserve_metric_and_match_exact_density(self):
        for method in ("predictor_corrector", "accelerated"):
            with self.subTest(method=method):
                sim = model_simulation(method=method, nocc=2)
                U = expm(-1j * sim.overlap_inv @ sim.h_model * 0.1)
                exact = U @ sim.make_rdm1(sim.mo_coeff, sim.mo_occ) @ U.conj().T
                actual = advance(sim, 0.005, 0.1)
                assert_allclose(actual, exact, atol=3e-7)
                assert_allclose(sim.mo_coeff.conj().T @ sim.overlap @ sim.mo_coeff, np.eye(2), atol=1e-13)

    def test_unknown_method_remains_an_error(self):
        sim = model_simulation(method="unknown")
        with self.assertRaisesRegex(ValueError, "Unknown propagation method"):
            sim.propagate_electrons(sim.h_model, 0)


class PySCFPropagationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = lib.num_threads()
        lib.num_threads(1)
        cls.mean_fields = []
        for method in (scf.RHF, dft.RKS):
            mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
            mf = method(mol)
            if isinstance(mf, dft.rks.KohnShamDFT):
                mf.xc = "lda,vwn"
                mf.grids.level = 0
            mf.conv_tol = 1e-12
            mf.kernel()
            if not mf.converged:
                raise RuntimeError("Regression SCF did not converge.")
            cls.mean_fields.append(mf)

    @classmethod
    def tearDownClass(cls):
        lib.num_threads(cls.original_threads)

    def test_rhf_and_rks_stationary_states(self):
        for simulation, mf in zip((TDRHFSimulation, TDRKSSimulation), self.mean_fields):
            with self.subTest(simulation=simulation.__name__):
                sim = simulation(mf, dt=0.1, propagation_method="pt_cn", convergence_threshold=1e-11)
                initial = sim.make_rdm1(sim.mo_coeff, sim.mo_occ)
                energy = sim.energy_tot()
                assert_allclose(advance(sim, 0.1, 0.3), initial, atol=2e-10)
                self.assertAlmostEqual(sim.energy_tot(), energy, places=11)

    def test_kicked_rhf_and_rks_match_fine_cn(self):
        for simulation, mf in zip((TDRHFSimulation, TDRKSSimulation), self.mean_fields):
            with self.subTest(simulation=simulation.__name__):
                pt = simulation(mf, dt=0.02, propagation_method="pt_cn", convergence_threshold=1e-11)
                cn = simulation(mf, dt=0.002, propagation_method="accelerated", convergence_threshold=1e-11)
                C = (np.cos(0.1) * mf.mo_coeff[:, :1] + 1j * np.sin(0.1) * mf.mo_coeff[:, 1:2])
                pt.mo_coeff = C.copy()
                cn.mo_coeff = C.copy()
                assert_allclose(advance(pt, 0.02, 0.1), advance(cn, 0.002, 0.1), atol=2e-6)
                assert_allclose(pt.mo_coeff.conj().T @ pt.overlap @ pt.mo_coeff, np.eye(1), atol=1e-13)

    def test_fixed_nuclei_driver_keeps_endpoint_field_and_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            sim = TDRKSSimulation(
                self.mean_fields[1], dt=0.01, steps=3, propagation_method="pt_cn",
                field_type="sine", E0=0.02, omega=0.5, convergence_threshold=1e-11,
                output_filename=f"{directory}/trajectory.xyz")
            with contextlib.redirect_stdout(io.StringIO()):
                sim.run()
            self.assertAlmostEqual(sim.t, 0.03)
            self.assertEqual(len(sim.potential_energies), 3)
            D = sim.make_rdm1(sim.mo_coeff, sim.mo_occ)
            assert_allclose(sim.F, sim.get_fock_matrix(D, sim.electric_field(sim.t)), atol=1e-14)


if __name__ == "__main__":
    unittest.main()
