# Electronic propagation regression tests

The tests use Python's standard `unittest` runner and the package dependencies.
CTest is optional; it does not build a compiled TDMASH component.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
cmake -S . -B build -DPython3_EXECUTABLE="$(command -v python)"
ctest --test-dir build --output-on-failure
```

`test_pt_cn.py` checks the production propagation dispatch against exact matrix
exponentials and an independent Schrödinger integration. It covers nonorthogonal
metrics, complex orbitals, multiple occupied states, second-order density
convergence, energy scaling, stationary occupied subspaces, occupied rotations,
time-dependent fields, backward and zero steps, nonlinear self-consistency,
failure rollback, and unsupported inputs.

`test_propagation_regression.py` checks the existing CN methods against exact
density evolution, unknown-method rejection, and PT-CN with real PySCF RHF/RKS
backends. The molecular cases use H2/STO-3G and compare a perturbed occupied
state with fine-step CN. Numerical thresholds are fixed assertions rather than
reference files generated from the implementation under test.

`test_moving_basis.py` checks Lowdin transport and its inverse, preservation of
backend settings, nonstationary RHF/RKS forces against energy finite differences,
synchronized Velocity Verlet endpoints, timestep convergence, and whole-step
rollback. The same symmetric driver is exercised with PT-CN and both existing CN
schemes. The timestep comparator and Euler/BOMD workflow self-checks are also
registered with CTest. The latter checks native PySCF RHF/RKS BOMD frame times
and matching isotope masses.

`benchmark_pt_cn_bomd.py self-check` checks complete timing pairs, rejects missing
measurements, and runs both RHF/RKS methods with identical output fields and
initial conditions. It is registered as `paired_timing_self_check` in CTest.
The production benchmark uses four AB/BA/BA/AB pairs per backend/timestep,
validates trajectories against the frozen study, and keeps raw arrays on Euler.

`test_pt_unrestricted.py` adds 14 test methods for coupled UHF/UKS PT-CN.
These cover separate spin charges and orthogonality, second-order convergence,
independent spin rotations, reduction to a closed shell, nonlinear spin coupling,
empty and unequal spin populations, physical field times, backward/zero steps,
joint rollback and the density/potential cache. Real PySCF cases use ground and
MOM Delta-SCF trans-diimide determinants, independently integrated Schrodinger
references, fine accelerated CN, mu rescaling and open-spin Li/H atoms.
SCF references must converge and pass an independent orbital-gradient check.
The suite is registered as `unrestricted_pt_cn` in CTest.

`test_unrestricted_moving_basis.py` adds twelve methods for shared spin transport,
backend preservation, off-SCF Lowdin forces against energy finite differences,
agreement with native stationary SCF gradients, scaled symmetric dynamics,
whole-step rollback, valence-excitation timestep convergence and translating
open-spin atoms. Electronic substeps also check physical times, a fixed midpoint,
second-order refinement, invalid ratios and rollback after a later substep fails.
Native complex-density Fock/energy checks cover UHF and UKS. Hybrid and
range-separated hybrid forces are checked against native energy derivatives.
An excited determinant also compares CN and PT-CN with an independent DOP853
integration of the coupled Schrodinger/nuclear equations, including refinement.
It is registered as `unrestricted_moving_basis`. The ethylene
workflow self-check also checks the alpha/dt manifest, thermal velocities with
COM/rotation removed, MOM invariance under orbital reordering, diagnostic JSON
serialization and a small excited-state trajectory with electronic substeps.

`PYTHONPATH=src:scripts python scripts/compare_c2h4_prefix.py --self-check`
checks shared-time prefix metrics and the nuclear contribution to total dipoles.
The `cn_reference_workflow_self_check` CTest checks the physical-alpha CN/PT-CN
trajectory comparison, reference refinement matrix and gauge-invariant densities.
It also checks optional common-grid sampling and rejects invalid intervals;
the original shared-node budget remains required alongside that extra check.
Active acceptance uses trajectory, total dipole and energy. Force and spin
density remain diagnostics with the legacy strict verdict reported in parallel.
Budget checks reject accidentally omitted trajectory, dipole or energy limits.
`mom_predictor_workflow_self_check` checks checkpoint completion, CN/PT orbital
prediction along prescribed nuclear endpoints, fixed-reference MOM correction
and rejection of an ambiguous ground-state reference.
The continuation self-check compares an excited-state restart with native PySCF
NVE, rejects failed frames and rejects a converged ground-state collapse.
The CN/PT timing self-check selects only candidates that passed the complete
short-trajectory budget with a validated CN reference and checks integer-step
coverage of the refined candidate grid in the ten-au timing window.
The long-extension self-check verifies source/state identity, preserves reused
case IDs and rejects a different reference grid. Mock scheduler submissions
check that only new cases run and analysis waits for every unresolved source
array, including nested reuse, together with the new array.

Run `scripts/validate_pt_cn_timestep.py` separately for a larger-timestep
comparison against the independent density/nuclear ODE reference. Its JSON
protocol specifies the accuracy budgets before any case runs, and failed or
inaccurate cases remain in the report.

These are short correctness regressions. Unrestricted moving-nucleus coverage
is limited to the tested single-state paths. The separate Euler ethylene study
assesses longer excited-state trajectories and large scaling factors.
Unrestricted two-state DTO/NAC and hopping remain outside this validation.

The twelve direct CTest workflow scripts also import four helpers:
`compare_c2h4_prefix.py`, `refine_pt_cn_bomd_reference.py`,
`validate_pt_cn_theory4.py` and `probe_c2h4_pt_solver.py`.
Keep all sixteen Python scripts and the three referenced ethylene/CN/Cayley
Slurm templates in a frozen checkout. The Slurm templates require an exported
`TDMASH_PYTHON`; remote collection uses SSH configuration or
`TDMASH_SSH_IDENTITY`. Collector self-checks mock SSH and cover both identity
selection paths without contacting Euler.
