"""Coupled unrestricted PT-CN checks against independent spin dynamics."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from pyscf import dft, gto, lib, scf
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from tdmash import TDUHFSimulation, TDUKSSimulation
from tdmash.MOM import mom_occ_
from test_pt_cn import advance, model_simulation


def spin_blocks(blocks):
    return np.stack(blocks) if blocks[0].shape == blocks[1].shape else blocks


def spin_model(populations=(2, 1), nonlinear=False):
    base = model_simulation()
    sim = TDUHFSimulation.__new__(TDUHFSimulation)
    sim.__dict__.update(base.__dict__)
    sim.h_model = np.stack((base.h_model, base.h_model.conj() + .2 * base.overlap))
    sim.mo_coeff = spin_blocks([sim.overlap_half_inv[:, :n].copy() for n in populations])
    sim.prev_mo_coeff = spin_blocks([C.copy() for C in sim.mo_coeff])
    sim.mo_occ = spin_blocks([np.ones(n) for n in populations])
    sim.model_coupling = .2 if nonlinear else 0

    def fock(dm, field=0):
        sim.dm_last = dm.copy()
        sim.veff = np.array([sim.model_coupling * sim.overlap @ (dm[s].real + .7 * dm[1 - s].real) @ sim.overlap
                             for s in (0, 1)])
        sim.veff_last = sim.veff.copy()
        return sim.h_model * (1 + field) + sim.veff

    sim.get_fock_matrix = fock
    return sim


def determinant_fidelity(left, right, metric):
    overlap = np.prod([np.linalg.det(a.conj().T @ metric @ b) for a, b in zip(left, right)])
    return float(abs(overlap)**2)


class UnrestrictedModelTests(unittest.TestCase):
    def test_equal_spin_blocks_reduce_to_the_closed_shell_density(self):
        restricted = model_simulation(nocc=2)
        unrestricted = spin_model((2, 2))
        unrestricted.h_model = np.stack([restricted.h_model] * 2)
        expected = advance(restricted, .01, .2)
        actual = advance(unrestricted, .01, .2)
        assert_allclose(actual[0] + actual[1], expected, atol=1e-10)

    def test_spin_densities_converge_at_second_order_with_mu_scaling(self):
        for mu in (1., 10., 80.):
            errors = []
            for scaled_dt in (.1, .05, .025):
                sim = spin_model()
                sim.mu = mu
                exact = [expm(-1j * sim.overlap_inv @ H * .4) @ C for H, C in zip(sim.h_model, sim.mo_coeff)]
                expected = sim.make_rdm1(exact, sim.mo_occ)
                actual = advance(sim, scaled_dt / mu, .4 / mu)
                errors.append(float(np.linalg.norm(actual - expected)))
                self.assertGreater(determinant_fidelity(exact, sim.mo_coeff, sim.overlap), 1 - 1e-5)
                for spin, n in enumerate((2, 1)):
                    self.assertAlmostEqual(np.trace(actual[spin] @ sim.overlap).real, n, places=12)
                    C = sim.mo_coeff[spin]
                    assert_allclose(C.conj().T @ sim.overlap @ C, np.eye(n), atol=2e-13)
                self.assertLessEqual(sim.pt_cn_residual, sim.convergence_threshold)
                self.assertLessEqual(max(sim.pt_cn_spin_residuals), sim.convergence_threshold)
            with self.subTest(mu=mu, errors=errors):
                self.assertGreater(errors[0] / errors[1], 3.7)
                self.assertGreater(errors[1] / errors[2], 3.7)

    def test_joint_nonlinear_spin_solution_matches_independent_schrodinger_ode(self):
        sim = spin_model(nonlinear=True)
        initial = [C.copy() for C in sim.mo_coeff]
        sizes = np.cumsum([0] + [C.size for C in initial])

        def unpack(flat):
            return [flat[sizes[s]:sizes[s + 1]].reshape(initial[s].shape) for s in (0, 1)]

        def derivative(t, flat):
            blocks = unpack(flat)
            dm = np.array([C @ C.conj().T for C in blocks])
            F = [sim.h_model[s] + .2 * sim.overlap @ (dm[s].real + .7 * dm[1 - s].real) @ sim.overlap
                 for s in (0, 1)]
            return np.concatenate([(-1j * sim.overlap_inv @ H @ C).ravel() for H, C in zip(F, blocks)])

        reference = solve_ivp(derivative, (0, .2), np.concatenate([C.ravel() for C in initial]),
                              method='DOP853', rtol=1e-12, atol=1e-14)
        self.assertTrue(reference.success)
        exact = unpack(reference.y[:, -1])
        assert_allclose(advance(sim, .005, .2), sim.make_rdm1(exact, sim.mo_occ), atol=2e-6)
        self.assertGreater(determinant_fidelity(exact, sim.mo_coeff, sim.overlap), 1 - 1e-10)

    def test_independent_spin_rotations_preserve_each_density(self):
        a, b = spin_model((2, 2), True), spin_model((2, 2), True)
        for s, angle in enumerate((.4, -.7)):
            rotation = expm(1j * np.array([[.2, angle * 1j], [-angle * 1j, -.3]]))
            b.mo_coeff[s] = b.mo_coeff[s] @ rotation
        assert_allclose(advance(a, .01, .2), advance(b, .01, .2), atol=2e-9)
        self.assertGreater(determinant_fidelity(a.mo_coeff, b.mo_coeff, a.overlap), 1 - 1e-12)

    def test_empty_spin_and_both_population_orders(self):
        for populations in ((1, 0), (0, 1), (2, 1), (1, 2)):
            with self.subTest(populations=populations):
                sim = spin_model(populations)
                exact = [expm(-1j * sim.overlap_inv @ H * .1) @ C for H, C in zip(sim.h_model, sim.mo_coeff)]
                assert_allclose(advance(sim, .005, .1), sim.make_rdm1(exact, sim.mo_occ), atol=1e-6)
                for s, n in enumerate(populations):
                    self.assertEqual(sim.mo_coeff[s].shape, (3, n))
                    self.assertAlmostEqual(np.trace(sim.make_rdm1(sim.mo_coeff, sim.mo_occ)[s] @ sim.overlap).real,
                                           n, places=12)

    def test_stationary_spin_subspaces_remove_fast_phases(self):
        sim = spin_model()
        sim.mu = 80.
        for s, n in enumerate((2, 1)):
            _, vectors = np.linalg.eigh(sim.overlap_half_inv @ sim.h_model[s] @ sim.overlap_half_inv)
            sim.mo_coeff[s] = sim.overlap_half_inv @ vectors[:, :n]
        initial = [C.copy() for C in sim.mo_coeff]
        advance(sim, 2., 2.)
        for a, b in zip(sim.mo_coeff, initial):
            assert_allclose(a, b, atol=2e-13)
        self.assertEqual(sim.pt_cn_iterations, 1)

    def test_physical_endpoint_field_zero_and_backward_steps(self):
        sim = spin_model()
        sim.dt, sim.mu = 7., 10.
        sim.electric_field = lambda t: t
        initial = [C.copy() for C in sim.mo_coeff]
        sim.propagate_electrons(sim.h_model, 0, dt=0)
        for C, old in zip(sim.mo_coeff, initial):
            assert_array_equal(C, old)
        exact = [expm(-1j * 10 * sim.overlap_inv @ H * (.02 + .5 * .02**2)) @ C
                 for H, C in zip(sim.h_model, initial)]
        assert_allclose(advance(sim, .0005, .02), sim.make_rdm1(exact, sim.mo_occ), atol=2e-6)
        assert_allclose(sim.F, sim.get_fock_matrix(sim.make_rdm1(sim.mo_coeff, sim.mo_occ), .02), atol=1e-14)
        reverse = spin_model()
        initial_density = reverse.make_rdm1(reverse.mo_coeff, reverse.mo_occ)
        advance(reverse, .005, .1)
        advance(reverse, -.005, -.1)
        assert_allclose(reverse.make_rdm1(reverse.mo_coeff, reverse.mo_occ), initial_density, atol=2e-9)

    def test_failure_restores_both_spins_and_fock_caches(self):
        for failure in ('limit', 'beta_nonfinite', 'backend', 'final_fock'):
            with self.subTest(failure=failure):
                sim = spin_model(nonlinear=True)
                old = [C.copy() for C in sim.mo_coeff]
                previous = [C.copy() for C in sim.prev_mo_coeff]
                sim.F = sim.h_model.copy()
                old_F = sim.F.copy()
                fock = sim.get_fock_matrix
                if failure == 'limit':
                    sim.max_iterations = 1
                elif failure == 'beta_nonfinite':
                    def broken(dm, field):
                        value = fock(dm, field)
                        value[1, 0, 0] = np.nan
                        return value
                    sim.get_fock_matrix = broken
                elif failure == 'backend':
                    def broken(dm, field):
                        fock(dm, field)
                        raise RuntimeError('Injected coupled backend failure.')
                    sim.get_fock_matrix = broken
                else:
                    calls = [0]
                    sim.convergence_threshold = 1
                    def broken(dm, field):
                        value = fock(dm, field)
                        calls[0] += 1
                        if calls[0] == 2:
                            value[1, 0, 0] = np.nan
                        return value
                    sim.get_fock_matrix = broken
                with self.assertRaises(RuntimeError):
                    sim.propagate_electrons(sim.h_model, 0)
                for actual, expected in zip(sim.mo_coeff, old):
                    assert_array_equal(actual, expected)
                for actual, expected in zip(sim.prev_mo_coeff, previous):
                    assert_array_equal(actual, expected)
                assert_array_equal(sim.F, old_F)
                self.assertEqual(sim.dm_last, 0)
                self.assertEqual(sim.veff_last, 0)
                self.assertFalse(hasattr(sim, 'veff'))
                self.assertEqual(sim.t, 0)

    def test_invalid_spin_inputs_are_rejected(self):
        for parameter, value in (('mo_occ', [np.array([1., .5]), np.ones(1)]),
                                 ('mo_occ', [np.ones(1), np.ones(1)]),
                                 ('mo_occ', [np.ones(2)]), ('n_virtual', 1), ('imaginary_time', True)):
            sim = spin_model()
            setattr(sim, parameter, value)
            with self.subTest(parameter=parameter), self.assertRaises(ValueError):
                sim.propagate_electrons(sim.h_model, 0)
        sim = spin_model()
        with self.assertRaisesRegex(ValueError, 'Fock'):
            sim.propagate_electrons(sim.h_model[0], 0)


GEOMETRY = ('N -7.84651 1.28271 -0.25720; N -5.30782 3.39826 -0.51257; '
            'H -8.45063 2.38905 -2.53646; H -4.55615 2.29563 1.14696')


class UnrestrictedPySCFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = lib.num_threads()
        lib.num_threads(1)
        cls.backends = []
        for method, simulation in ((scf.UHF, TDUHFSimulation), (dft.UKS, TDUKSSimulation)):
            mol = gto.M(atom=GEOMETRY, unit='Bohr', basis='sto-3g', verbose=0)

            def mean_field():
                mf = method(mol)
                cls.addClassCleanup(mf._chkfile.close)
                mf.chkfile = None
                if method is dft.UKS:
                    mf.xc, mf.grids.level, mf.grids.prune, mf.small_rho_cutoff = 'lda,vwn', 0, None, 0
                mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-11, 1e-7, 200
                return mf

            ground = mean_field()
            ground.kernel()

            def refine(mf):
                # Resolve small orbital gradients instead of DIIS stopping at its linear-dependence floor.
                refined = mf.newton()
                refined.ah_lindep, refined.ah_conv_tol = 1e-22, 1e-14
                refined.kernel(mf.mo_coeff, mf.mo_occ)
                if not refined.converged or np.linalg.norm(refined.get_grad(refined.mo_coeff, refined.mo_occ)) > 1e-7:
                    raise RuntimeError('Unrestricted SCF reference failed its orbital-gradient check.')
                return refined

            ground = refine(ground)
            occupations = ground.mo_occ.copy()
            nocc = mol.nelectron // 2
            occupations[0, nocc - 1], occupations[0, nocc] = 0, 1
            excited = mom_occ_(mean_field(), np.asarray(ground.mo_coeff).copy(), occupations)
            excited.kernel(dm0=ground.make_rdm1(mo_occ=occupations))
            excited = refine(excited)
            if not ground.converged or not excited.converged:
                raise RuntimeError('Unrestricted trans-diimide ground/Delta-SCF did not converge.')
            if not np.allclose(excited.mo_occ.sum(axis=1), [8, 8]):
                raise RuntimeError('Unrestricted MOM lost the target spin populations.')
            if excited.e_tot <= ground.e_tot + .001 or np.linalg.norm(excited.make_rdm1() - ground.make_rdm1()) < .05:
                raise RuntimeError('Unrestricted Delta-SCF collapsed to the ground-state reference.')
            cls.backends.append((simulation, ground, excited))

    @classmethod
    def tearDownClass(cls):
        lib.num_threads(cls.old_threads)

    def simulation(self, simulation, mf, method='pt_cn', dt=.0005, mu=10):
        return simulation(mf, dt=dt, mu=mu, propagation_method=method,
                          convergence_threshold=1e-11, max_iterations=200)

    def test_existing_accelerated_two_state_baseline_and_scaled_pt_stationarity(self):
        for simulation, ground, excited in self.backends:
            for mf in (ground, excited):
                with self.subTest(backend=simulation.__name__, state='ground' if mf is ground else 'excited'):
                    cn = self.simulation(simulation, mf, 'accelerated', .0005, 1)
                    initial = cn.make_rdm1(cn.mo_coeff, cn.mo_occ)
                    self.assertAlmostEqual(cn.energy_tot(), mf.e_tot, places=9)
                    assert_allclose(advance(cn, .0005, .002), initial, atol=3e-8)
                    self.assertTrue(cn.pc_converged)
                    for mu in (1., 10., 80.):
                        with self.subTest(mu=mu):
                            # Large mu*dt is scanned separately rather than assumed converged.
                            dt = min(.2, 2 / mu)
                            sim = self.simulation(simulation, mf, dt=dt, mu=mu)
                            assert_allclose(advance(sim, dt, 2 * dt), initial, atol=2e-5)
                            self.assertAlmostEqual(sim.energy_tot(), mf.e_tot, places=8)
                            assert_array_equal(sim.mo_occ, [mf.mo_occ[s, mf.mo_occ[s] > 0] for s in (0, 1)])

    def test_perturbed_determinant_matches_coupled_ode_and_fine_accelerated(self):
        for simulation, _, mf in self.backends:
            with self.subTest(backend=simulation.__name__):
                pt = self.simulation(simulation, mf)
                cn = self.simulation(simulation, mf, 'accelerated', .00005)
                initial = []
                for s in (0, 1):
                    generator = np.zeros((mf.mo_coeff.shape[1],) * 2, complex)
                    generator[6, 9], generator[9, 6] = .02j, .02j
                    generator[7, 10], generator[10, 7] = .015, -.015
                    initial.append((mf.mo_coeff[s] @ expm(generator))[:, mf.mo_occ[s] > 0])
                initial = np.array(initial)
                pt.mo_coeff, cn.mo_coeff = initial.copy(), initial.copy()
                occupations = pt.mo_occ.copy()
                core = mf.get_hcore()

                def derivative(t, flat):
                    C = flat.reshape(initial.shape)
                    dm = np.array([(C[s] * occupations[s]) @ C[s].conj().T for s in (0, 1)])
                    F = core + mf.get_veff(mf.mol, dm)
                    return np.array([-1j * 10 * pt.overlap_inv @ F[s] @ C[s] for s in (0, 1)]).ravel()

                reference = solve_ivp(derivative, (0, .004), initial.ravel(), method='DOP853', rtol=1e-11, atol=1e-13)
                self.assertTrue(reference.success)
                exact = reference.y[:, -1].reshape(initial.shape)
                expected = pt.make_rdm1(exact, occupations)
                assert_allclose(advance(pt, .0005, .004), expected, atol=2e-6)
                assert_allclose(advance(cn, .00005, .004), expected, atol=2e-7)
                self.assertTrue(cn.pc_converged)
                self.assertGreater(determinant_fidelity(exact, pt.mo_coeff, pt.overlap), 1 - 1e-9)
                self.assertLess(abs(pt.energy_tot() - cn.energy_tot()), 2e-7)
                self.assertLessEqual(pt.pt_cn_residual, pt.convergence_threshold)

    def test_mu_time_rescaling_preserves_each_spin_density(self):
        for simulation, _, mf in self.backends:
            states = []
            for mu in (1., 10., 80.):
                sim = self.simulation(simulation, mf, dt=.01 / mu, mu=mu)
                for s in (0, 1):
                    generator = np.zeros((mf.mo_coeff.shape[1],) * 2)
                    generator[7, 9], generator[9, 7] = .02, -.02
                    sim.mo_coeff[s] = (mf.mo_coeff[s] @ expm(generator))[:, mf.mo_occ[s] > 0]
                states.append(advance(sim, .01 / mu, .04 / mu))
            assert_allclose(states[0], states[1], atol=1e-10)
            assert_allclose(states[0], states[2], atol=1e-10)

    def test_open_spin_initialization_and_fixed_nuclei_driver(self):
        for method, simulation in ((scf.UHF, TDUHFSimulation), (dft.UKS, TDUKSSimulation)):
            for atom in ('Li', 'H'):
                with self.subTest(backend=simulation.__name__, atom=atom):
                    mol = gto.M(atom=f'{atom} 0 0 0', basis='6-31g', spin=1, verbose=0)
                    mf = method(mol)
                    self.addCleanup(mf._chkfile.close)
                    mf.chkfile = None
                    if method is dft.UKS:
                        mf.xc, mf.grids.level = 'lda,vwn', 0
                    mf.conv_tol = 1e-11
                    mf.kernel()
                    self.assertTrue(mf.converged)
                    with self.assertRaisesRegex(ValueError, 'Unequal spin populations'):
                        self.simulation(simulation, mf, method='accelerated')
                    with tempfile.TemporaryDirectory() as directory:
                        sim = self.simulation(simulation, mf, dt=.01, mu=1)
                        sim.steps = 2
                        sim.output_filename = str(Path(directory) / 'trajectory.xyz')
                        with contextlib.redirect_stdout(io.StringIO()):
                            sim.run()
                        self.assertAlmostEqual(sim.t, .02)
                        for s, n in enumerate(mol.nelec):
                            self.assertEqual(sim.mo_coeff[s].shape[1], n)
                            self.assertAlmostEqual(np.trace(sim.make_rdm1(sim.mo_coeff, sim.mo_occ)[s] @ sim.overlap).real,
                                                   n, places=11)
                    fractional = mf.mo_occ.copy()
                    fractional[0, -1] = 1e-5
                    with mock.patch.object(mf, 'mo_occ', fractional):
                        with self.assertRaisesRegex(ValueError, 'integer spin occupations'):
                            self.simulation(simulation, mf)

    def test_fock_cache_tracks_the_density_used_for_the_previous_potential(self):
        simulation, mf, _ = self.backends[0]
        sim = self.simulation(simulation, mf)
        coefficients = sim.mo_coeff.copy()
        for s in (0, 1):
            virtual = mf.mo_coeff[s, :, np.flatnonzero(mf.mo_occ[s] == 0)[0]]
            coefficients[s, :, 0] = np.cos(.02) * coefficients[s, :, 0] + 1j * np.sin(.02) * virtual
        dm = sim.make_rdm1(coefficients, sim.mo_occ)
        potential = sim.mf.get_veff
        seen = []

        def checked(mol, density, dm_last=0, vhf_last=0):
            seen.append(np.asarray(dm_last).copy())
            return potential(mol, density, dm_last=dm_last, vhf_last=vhf_last)

        with mock.patch.object(sim.mf, 'get_veff', checked):
            sim.get_fock_matrix(dm)
            cached = sim.get_fock_matrix(dm * .99)
        assert_allclose(seen[1], dm, atol=0)
        assert_allclose(sim.dm_last, dm * .99, atol=0)
        rebuilt = sim.h_core + potential(sim.mol, dm * .99)
        assert_allclose(cached, rebuilt, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
