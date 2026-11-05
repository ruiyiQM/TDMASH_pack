"""Independent checks of effective-step invariance and non-Aufbau RPA."""

from pathlib import Path
import sys
import unittest

import numpy as np
from pyscf import gto, lib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import validate_pt_cn_theory4 as theory
from test_pt_unrestricted import spin_model


class TheoryFourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = lib.num_threads()
        cls.addClassCleanup(lib.num_threads, previous)
        lib.num_threads(1)

    def test_actual_nonlinear_solver_depends_on_product_alpha_dt(self):
        with self.assertRaises(ValueError):
            theory.fetch_inputs(Path(__file__).resolve().parents[1])
        rows, states = [], []
        for alpha in theory.ALPHAS:
            sim = spin_model((2, 2), nonlinear=True)
            sim.mu, sim.dt = alpha, .5 / alpha
            sim.propagation_method = 'pt_cn'
            sim.max_iterations, sim.convergence_threshold = 120, 1e-11
            row, state = theory.solve_step(sim)
            self.assertTrue(row['completed'])
            rows.append(row);states.append(state)
        result = theory.scaling_check(rows, states)
        self.assertTrue(result['same_outcome'])
        self.assertLess(result['shared_trial_equation_max_difference'], 1e-12)
        self.assertLess(result['max_accepted_density_difference'], 1e-10)

    def test_anti_aligned_actions_do_not_imply_stationarity(self):
        tau, omega, start = 1e5, 1., np.array([[.01 + 0j]])
        factor = (1 - .5j * tau * omega) / (1 + .5j * tau * omega)
        result = theory.action_metrics(np.eye(1), start, factor * start)
        self.assertAlmostEqual(result['norm_ratio'], 1.)
        self.assertLess(result['anti_action_relative'], 1e-4)
        self.assertGreater(result['end_norm'], 1e-3)
        self.assertGreater(abs(result['phase_deg']), 179.)

    def test_complex_rpa_quartet_has_null_indefinite_norm(self):
        A, B = np.diag([1., -1.]), np.array([[0., .2], [.2, 0.]])
        values, _, norms, record = theory.rpa_spectrum(A, B)
        self.assertEqual(record['growing_count'], 2)
        self.assertAlmostEqual(record['max_growth_rate'], .2)
        self.assertLess(max(abs(norms[values.imag > .1])), 1e-12)
        _, _, _, uncoupled = theory.rpa_spectrum(A, np.zeros_like(B))
        self.assertEqual(uncoupled['growing_count'], 0)
        self.assertEqual(uncoupled['negative_norm_positive_real_frequency_count'], 1)

    def test_nonaufbau_ab_matches_fock_derivatives_and_permutations(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr', basis='sto-3g', verbose=0)
        mf = theory.study.mean_field(mol, 'uks')
        mf.kernel()
        self.assertTrue(mf.converged)
        mf.mo_occ[0] = [0., 1.]
        mf = theory.semicanonical(mf)
        A, B, labels = theory.full_ab(mf)
        self.assertTrue(any(item['gap'] < 0 for item in labels))
        result = theory.verify_ab(mf, A, B, labels, permutation=True)
        self.assertLess(result['max_finite_difference_error'], 2e-6)
        self.assertLess(result['permutation_error'], 1e-10)


if __name__ == '__main__':
    unittest.main()
