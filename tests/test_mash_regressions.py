"""Run with: python -m unittest discover -s tests -v."""
import contextlib
import importlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from pyscf import gto

from tdmash.Ehrenfest import FixedPathDynamics
from tdmash.FD import Finite_Difference_NAC, ZeroNACError
from tdmash.MASH import MASH
from tdmash.Spin import Spin, bloch_to_coeffs
from tdmash.TDOverlap import TDOverlap

overlap_module = importlib.import_module("tdmash.TDOverlap")
mash_module = importlib.import_module("tdmash.MASH")


class ToyMol:
    def __init__(self, positions=None):
        self.positions = np.zeros((1, 3)) if positions is None else positions.copy()

    def copy(self):
        return ToyMol(self.positions)

    def atom_coords(self, unit="Bohr"):
        return self.positions.copy()

    def atom_mass_list(self):
        return [1.0]

    def atom_charges(self):
        return [1]

    def atom_symbol(self, index):
        return "H"


class ToySimulation:
    """Two AO orbitals with a known, nonzero spatial rotation."""
    phase_rng = None

    def __init__(self, mf, dt=0.4, nuclei_dt=0.4, mu=1.0,
                 imaginary_time=False, propagation_method="accelerated", **kwargs):
        self.mf, self.mol = mf, mf.mol.copy()
        self.mo_coeff, self.mo_occ = mf.mo_coeff.copy(), mf.mo_occ.copy()
        self.dt, self.nuclei_dt, self.mu = dt, nuclei_dt, mu
        self.imaginary_time = imaginary_time
        self.propagation_method = propagation_method
        self.overlap = np.eye(2)
        self.force = np.zeros((1, 3))

    def update_mol(self, positions):
        angle = np.array([0.3, 0.5, -0.2]) @ (positions - self.mol.positions)[0]
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]])
        self.mo_coeff = rotation @ self.mo_coeff
        self.mol = ToyMol(positions)

    def update_electronic_wavefunction(self, **kwargs):
        if self.phase_rng is not None:
            self.mo_coeff *= np.exp(1j * self.phase_rng.uniform(-np.pi, np.pi, len(self.mo_occ)))
        return self.energy_tot(), self.force.copy()

    def energy_tot(self):
        return 0.0

    def calculate_forces(self):
        pass

    def make_rdm1(self, mo_coeff, mo_occ):
        return (mo_coeff * mo_occ) @ mo_coeff.conj().T


def toy_pair():
    ground = SimpleNamespace(mol=ToyMol(), mo_coeff=np.eye(2, dtype=complex)[:, :1],
                             mo_occ=np.array([2]))
    excited = SimpleNamespace(mol=ToyMol(), mo_coeff=np.eye(2, dtype=complex),
                              mo_occ=np.array([1, 1]))
    return ToySimulation(ground), ToySimulation(excited)


class OverlapTests(unittest.TestCase):
    def test_zero_fd_nac_raises_specific_value_error(self):
        sim1, sim2 = toy_pair()
        fd = Finite_Difference_NAC(sim1, sim2)
        overlap = SimpleNamespace(update_mo_coeff=lambda *a, **kw: None,
                                  update_overlap=lambda: np.eye(2))
        with patch.object(importlib.import_module("tdmash.FD"), "TDOverlap", return_value=overlap):
            with self.assertRaises(ZeroNACError) as error:
                fd.compute_nac(sim1.mo_coeff, sim2.mo_coeff, np.zeros((1, 3)))
        self.assertIsInstance(error.exception, ValueError)

    def test_fd_configuration_is_independent_of_trajectory(self):
        sim1, sim2 = toy_pair()
        sim1.mu = sim2.mu = 80.0
        sim1.imaginary_time = sim2.imaginary_time = True
        sim1.propagation_method = sim2.propagation_method = "predictor_corrector"
        fd = Finite_Difference_NAC(sim1, sim2)
        for child, dt in [(fd.sim1_forward, 4.), (fd.sim2_forward, 4.),
                          (fd.sim1_backward, -4.), (fd.sim2_backward, -4.)]:
            self.assertIs(type(child), ToySimulation)
            self.assertFalse(child.imaginary_time)
            self.assertEqual(child.propagation_method, "accelerated")
            self.assertEqual(child.dt, dt)
            self.assertEqual(child.mu, 1.)

    def test_first_cross_geometry_overlap_and_phase_alignment(self):
        mol0 = gto.M(atom="H 0 0 0; H 0 0 0.8", basis="sto-3g", verbose=0)
        mol1 = gto.M(atom="H 0 0 0; H 0 0 1.3", basis="sto-3g", verbose=0)
        sim1, sim2 = toy_pair()
        sim1.mol = sim2.mol = mol0
        sim1.overlap = mol0.intor("int1e_ovlp")
        overlap = TDOverlap(sim1, sim2)
        before = overlap.mo_coeff2.copy()
        sim1.mol = sim2.mol = mol1
        sim1.overlap = mol1.intor("int1e_ovlp")
        sim2.mo_coeff *= np.exp(1j * np.array([1.3, -0.7]))
        overlap.update_mo_coeff(sim1, sim2)
        expected = gto.intor_cross("int1e_ovlp", mol0, mol1)
        assert_allclose(overlap.overlap, expected)
        self.assertGreater(np.linalg.norm(expected - sim1.overlap), 0.01)
        diagonal = np.diag(before.conj().T @ expected @ overlap.mo_coeff2)
        assert_allclose(diagonal.imag, 0, atol=1e-14)
        self.assertTrue(np.all(diagonal.real > 0))

    def test_fd_axis_survives_random_displaced_orbital_and_eigenvector_phases(self):
        sim1, sim2 = toy_pair()
        fd = Finite_Difference_NAC(sim1, sim2)
        eigh = np.linalg.eigh
        rng = np.random.default_rng(57)

        def phased_eigh(matrix):
            values, vectors = eigh(matrix)
            return values, vectors * np.exp(1j * rng.uniform(-np.pi, np.pi, len(values)))

        with patch.object(overlap_module.gto, "intor_cross", return_value=np.eye(2)):
            baseline = fd.compute_nac(sim1.mo_coeff, sim2.mo_coeff, np.zeros((1, 3)))
            self.assertGreater(abs(baseline[0, 0]), 0.1)
            try:
                ToySimulation.phase_rng = rng
                with patch.object(overlap_module.np.linalg, "eigh", side_effect=phased_eigh):
                    for _ in range(10):
                        actual = fd.compute_nac(sim1.mo_coeff, sim2.mo_coeff, np.zeros((1, 3)))
                        assert_allclose(actual, baseline, atol=2e-12)
            finally:
                ToySimulation.phase_rng = None


class MomentumTests(unittest.TestCase):
    def test_allowed_and_frustrated_hops_conserve_energy(self):
        mass = np.array([2.0, 5.0])
        nac = np.array([[1., 0., 0.], [0., 0., 0.]])
        for initial, energies, momentum, accepted in [
            (0, [0., 0.3], [[2., 0., 0.], [0., 1., 0.]], True),
            (1, [0., 0.3], [[-2., 0., 0.], [0., 1., 0.]], True),
            (1, [0., 0.3], [[0., 1., 0.], [0., 2., 0.]], True),
            (1, [0., 0.0], [[0., 1., 0.], [0., 2., 0.]], True),
            (0, [0., 0.3], [[0., 1., 0.], [0., 2., 0.]], False),
        ]:
            p = np.array(momentum)
            spin = Spin(np.array([0.6, 0., 0.8 if initial == 0 else -0.8]))
            spin.current_spin *= -1
            spin.coefficent = bloch_to_coeffs(spin.current_spin)
            spin.S_value.append(spin.current_spin.copy())
            before = spin.current_spin.copy()
            with contextlib.redirect_stdout(io.StringIO()), np.errstate(all="raise"):
                actual = spin.rescale_momentum_on_hop_v4(p, nac, energies, mass, initial_state=initial)
            final = 1 - initial if accepted else initial
            self.assertEqual(spin.determine_active_state_index(), final)
            assert_allclose(np.sum(actual**2 / mass[:, None]) / 2 + energies[final],
                            np.sum(p**2 / mass[:, None]) / 2 + energies[initial], atol=2e-15)
            assert_allclose(actual[:, 1:], p[:, 1:])
            assert_allclose(spin.S_value[-1], spin.current_spin)
            assert_allclose(spin.current_spin, before if accepted else -before)
            with contextlib.redirect_stdout(io.StringIO()):
                flipped = spin.rescale_momentum_on_hop_v4(p, -nac, energies, mass, initial_state=initial)
            assert_allclose(actual, flipped, atol=2e-15)


class PropagationTests(unittest.TestCase):
    def test_accelerated_cn_solves_small_nonlinear_fock_residual(self):
        module = importlib.import_module("tdmash.TDRHF")
        for dt in (4.0, -4.0):
            initial = np.array([[np.cos(1e-3)], [np.sin(1e-3)]], dtype=complex)
            sim = SimpleNamespace(mo_coeff=initial.copy(), mo_occ=np.array([2.]),
                                  dt=dt, t=0., mu=1., overlap=np.eye(2),
                                  max_iterations=40, count=False, convergence_threshold=1e-9)
            sim.electric_field = lambda t: 0.
            sim.make_rdm1 = lambda coeff, occ: 2 * coeff @ coeff.conj().T
            sim.get_fock_matrix = lambda dm, field: np.diag([-1., -0.1]) + 0.6 * dm.real
            start = sim.get_fock_matrix(sim.make_rdm1(initial, sim.mo_occ), 0.)
            with contextlib.redirect_stdout(io.StringIO()) as log:
                module.predictor_corrector_accelerated(sim, start, 0., dt=dt)
            self.assertNotIn("did not converge", log.getvalue())
            final = sim.get_fock_matrix(sim.make_rdm1(sim.mo_coeff, sim.mo_occ), 0.)
            average = (start + final) / 2
            residual = ((sim.overlap + 0.5j * dt * average) @ sim.mo_coeff
                        - (sim.overlap - 0.5j * dt * average) @ initial)
            assert_allclose(residual, 0., atol=2e-9)
            assert_allclose(sim.mo_coeff.conj().T @ sim.mo_coeff, np.eye(1), atol=2e-15)


class DriverTests(unittest.TestCase):
    def make_hopping_driver(self, output, initial=1, velocity=0., steps=1, attempt_step=1):
        sim1, sim2 = toy_pair()
        sim2.energy_tot = lambda: 0.3
        overlap = SimpleNamespace(update_mo_coeff=lambda *a, **kw: None,
                                  update_overlap=lambda: np.eye(2))
        with patch.object(mash_module, "TDOverlap", return_value=overlap), \
             patch.object(mash_module, "Finite_Difference_NAC"):
            run = MASH(sim1, sim2, dt=0.4, steps=steps, energy_filename=str(output),
                       initial_velocity=np.array([[velocity, 0., 0.]]),
                       initial_spin=np.array([0.6, 0., -0.8 if initial else 0.8]))
        run.fd.compute_nac.return_value = np.array([[1., 0., 0.]])

        def propagate(*args):
            if len(run.spin.S_value) + 1 >= attempt_step:
                run.spin.current_spin = np.array([0.6, 0., 0.8 if initial else -0.8])
                run.spin.coefficent = bloch_to_coeffs(run.spin.current_spin)
            run.spin.S_value.append(run.spin.current_spin.copy())

        run.spin.propogate_spin = propagate
        return run

    def test_zero_nac_on_first_step_saves_empty_incomplete_trajectory(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "mash_output.npz"
            run = self.make_hopping_driver(output)
            run.fd.compute_nac.side_effect = ZeroNACError("The NACV is zero")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ZeroNACError):
                run.run()
            with np.load(output, allow_pickle=False) as data:
                self.assertFalse(data["completed"].item())
                self.assertEqual(data["completed_steps"].item(), 0)
                self.assertEqual(data["failed_step"].item(), 1)
                self.assertEqual(len(data["time"]), 0)
                self.assertEqual(len(data["spin"]), 0)
                self.assertEqual(len(data["potential_energy_1"]), 1)
                self.assertEqual(data["hop_events"][0]["outcome"], "zero_nac")

    def test_hop_events_are_saved_with_actual_outcomes(self):
        for initial, velocity, outcome in [(1, 0., "accepted"), (0, 0.03, "accepted"),
                                           (0, 0., "frustrated")]:
            with self.subTest(initial=initial, outcome=outcome), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / "mash_output.npz"
                run = self.make_hopping_driver(output, initial=initial, velocity=velocity)
                with contextlib.redirect_stdout(io.StringIO()):
                    run.run()
                with np.load(output, allow_pickle=False) as data:
                    events = data["hop_events"]
                    self.assertEqual(len(events), 1)
                    event = events[0]
                    self.assertEqual(event["step"], 1)
                    self.assertEqual(event["time"], 0.4)
                    self.assertEqual(event["from_state"], initial)
                    self.assertEqual(event["to_state"], 1 - initial)
                    self.assertEqual(event["outcome"], outcome)
                    assert_allclose(event["kinetic_energy_parallel"], 0.5 * run.mass[0] * velocity**2)
                    assert_allclose(event["energy_gap"], 0.3 if initial == 0 else -0.3)
                    self.assertTrue(data["completed"].item())
                    self.assertEqual(data["completed_steps"].item(), 1)
                    self.assertEqual(data["failure_reason"].item(), "")

    def test_zero_nac_preserves_only_completed_steps_and_checker_flags_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            complete = root / "structure_0"
            complete.mkdir()
            good = self.make_hopping_driver(complete / "mash_output.npz", steps=2)
            with contextlib.redirect_stdout(io.StringIO()):
                good.run()
            script = Path(__file__).resolve().parents[1] / "examples/check_mash_runs.py"
            command = [sys.executable, str(script), str(root), "--steps", "2", "--expected-runs"]
            checked = subprocess.run(command + ["1"], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn("structure_0 ok 2 1 0 0", checked.stdout)

            failed = root / "structure_1"
            failed.mkdir()
            output = failed / "mash_output.npz"
            run = self.make_hopping_driver(output, steps=2, attempt_step=2)
            run.fd.compute_nac.side_effect = ZeroNACError("The NACV is zero")
            run.output_interval = 1
            run.output_filename = str(failed / "trajectory.xyz")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ZeroNACError):
                run.run()
            with np.load(output, allow_pickle=False) as data:
                self.assertFalse(data["completed"].item())
                self.assertEqual(data["completed_steps"].item(), 1)
                self.assertEqual(data["requested_steps"].item(), 2)
                self.assertEqual(data["failure_reason"].item(), "zero_nac")
                self.assertEqual(data["failed_step"].item(), 2)
                self.assertEqual(data["hop_events"][0]["outcome"], "zero_nac")
                self.assertEqual(data["hop_events"][0]["time"], 0.8)
                self.assertTrue(np.isnan(data["hop_events"][0]["kinetic_energy_parallel"]))
                for key in ("time", "spin", "kinetic_energy", "total_energy", "tdoverlap"):
                    self.assertEqual(len(data[key]), 1, key)
                self.assertEqual(len(data["potential_energy_1"]), 2)
            (root / "structure_2").mkdir()
            checked = subprocess.run(command + ["3"], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 1)
            self.assertIn("structure_1 zero_nac 1 0 0 1", checked.stdout)
            self.assertIn("structure_2 invalid/missing", checked.stdout)

    def test_saved_times_and_output_follow_nuclear_steps(self):
        for driver, module in [(MASH, "tdmash.MASH"), (FixedPathDynamics, "tdmash.Ehrenfest")]:
            sim1, sim2 = toy_pair()
            overlap = SimpleNamespace(update_mo_coeff=lambda *a, **kw: None,
                                      update_overlap=lambda: np.eye(2))
            with tempfile.TemporaryDirectory() as folder, \
                 patch.object(importlib.import_module(module), "TDOverlap", return_value=overlap), \
                 patch.object(mash_module, "Finite_Difference_NAC"), \
                 contextlib.redirect_stdout(io.StringIO()):
                xyz, output = Path(folder) / "trajectory.xyz", Path(folder) / "output.npz"
                run = driver(sim1, sim2, dt=0.2, nuclei_dt=0.5, steps=5,
                             energy_filename=str(output), output_interval=2, output_filename=str(xyz))
                run.spin.propogate_spin = lambda *args: run.spin.S_value.append(run.spin.current_spin.copy())
                run.run()
                with np.load(output) as data:
                    assert_allclose(data["time"], [0.5, 1., 1.5, 2., 2.5])
                    self.assertEqual(len(data["spin"]), 5)
                    self.assertEqual(len(data["potential_energy_1"]), 6)
                    if driver is MASH:
                        self.assertEqual(data["hop_events"].shape, (0,))
                        self.assertIn("outcome", data["hop_events"].dtype.names)
                self.assertEqual([line for line in xyz.read_text().splitlines() if line.startswith("Step")],
                                 ["Step 2", "Step 4"])

    def test_integer_step_counts_times_and_first_geometry(self):
        for driver, module in [(MASH, "tdmash.MASH"), (FixedPathDynamics, "tdmash.Ehrenfest")]:
            for dt, steps in [(0.2, 20000), (0.4, 10000), (0.8, 5000)]:
                with self.subTest(driver=driver.__name__, dt=dt):
                    sim1, sim2 = toy_pair()
                    first_geometry = []
                    fake_overlap = SimpleNamespace(
                        update_mo_coeff=lambda a, b, previous_mol: first_geometry.append(previous_mol.positions.copy())
                        if not first_geometry else None,
                        update_overlap=lambda: np.eye(2))
                    with patch.object(importlib.import_module(module), "TDOverlap", return_value=fake_overlap), \
                         patch.object(mash_module, "Finite_Difference_NAC"), \
                         contextlib.redirect_stdout(io.StringIO()):
                        run = driver(sim1, sim2, dt=dt, steps=steps,
                                     initial_velocity=np.array([[0.01, 0., 0.]]))
                        run.spin.propogate_spin = lambda *args: None
                        run.run()
                    self.assertEqual(len(run.total_energies), steps)
                    self.assertEqual(len(run.tdoverlap_output), steps)
                    self.assertEqual(len(run.potentials_1), steps + 1)
                    self.assertEqual(run.t, 4000.0)
                    assert_allclose(first_geometry[0], np.zeros((1, 3)))
                    if driver is FixedPathDynamics:
                        assert_allclose(run.time, np.arange(1, steps + 1) * dt)


if __name__ == "__main__":
    unittest.main()
