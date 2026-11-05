# TDMASH

Energy-scaled real-time TDDFT for ab initio surface hopping dynamics.

This repository contains the reference implementation used in

> R. Zhou and J. O. Richardson, *Energy-Scaled Real-Time TDDFT for Ab Initio
> Surface Hopping Dynamics* (submitted).

The code propagates two Kohn–Sham determinants — the ground state and a
ΔSCF excited state — simultaneously along a nuclear trajectory using an
energy-scaled time-dependent Kohn–Sham equation,

    i ∂ψ_n/∂t = α H_KS ψ_n ,

in which the scaling factor `α > 1` widens the separation between the
electronic and nuclear timescales so that each determinant follows its own
adiabatic branch without a static SCF calculation at every step. Nonadiabatic
couplings are then extracted from the propagated states in a Dynamical
Transition Orbital (DTO) gauge, and the nuclei are driven by the Mapping
Approach to Surface Hopping (MASH).

Everything is built on [PySCF](https://pyscf.org/); no other quantum-chemistry
backend is required.

## Installation

Requires Python >= 3.9. The only external dependencies are `pyscf`, `numpy`,
`scipy` and `matplotlib`, all of which are installed automatically:

```bash
git clone https://github.com/ruiyiQM/TDMASH_pack.git
cd TDMASH_pack
pip install -e .
```

The `-e` (editable) install is recommended while developing, so that changes to
the source take effect without reinstalling. Afterwards the package is
importable from anywhere:

```python
from tdmash import TDRKSSimulation, MASH, FixedPathDynamics
```

## Repository layout

```
src/tdmash/        the importable package
examples/          runnable input scripts for trans-diimide
scripts/           utilities for generating initial conditions
initial_conditions/  the sampled Wigner ensemble used in the paper
```

### Modules

| Module | Purpose |
| --- | --- |
| `tdmash.TDRHF`, `tdmash.TDRKS` | Spin-restricted real-time propagators (HF and KS) |
| `tdmash.TDUHF`, `tdmash.TDUKS` | Spin-unrestricted counterparts |
| `tdmash.Anderson` | Anderson mixing, used to accelerate the predictor-corrector iteration |
| `tdmash.MOM` | Maximum-overlap method, used to prepare the ΔSCF excited state at the first step |
| `tdmash.TDOverlap`, `tdmash.TDUOverlap` | Many-electron overlaps and the DTO gauge fixing |
| `tdmash.Spin` | MASH mapping variables, hopping criterion and momentum rescaling |
| `tdmash.FD` | Finite-difference nonadiabatic coupling vectors |
| `tdmash.PT` | Parallel-transport Crank-Nicolson electronic propagation |
| `tdmash.Verlet` | Velocity Verlet nuclear integrator |
| `tdmash.Ehrenfest` | Driver for fixed-path runs (`FixedPathDynamics`): nuclei on the excited-state surface, mapping variables propagated as a diagnostic, no hops |
| `tdmash.MASH` | Surface-hopping driver (`MASH`) |
| `tdmash.Wigner`, `tdmash.sampler` | Wigner sampling of nuclear initial conditions and of the initial mapping variables |

### Electronic propagation

Three propagation schemes are available, selected with the `propagation_method`
argument of the simulation classes:

- `'predictor_corrector'` — Crank-Nicolson with a plain predictor-corrector
  iteration for the midpoint Fock matrix.
- `'accelerated'` — the same scheme with Anderson mixing applied to the Fock
  matrix, which converges the iteration in far fewer cycles in the stiff,
  large-`α` regime. **This is the scheme used for the published results** and
  the one all the examples select.
- `'pt_cn'` — a self-consistent parallel-transport Crank-Nicolson solver for
  closed-shell RHF/RKS and UHF/UKS with `n_virtual=0`. It removes occupied-space
  rotations in the evolution equation and checks the original AO residual.
  Nonconvergence raises an error without advancing the electronic state.
  Each unrestricted spin channel has integer occupations of zero or one;
  alpha and beta Focks are rebuilt jointly from both trial spin densities.
  Restricted open shells, fractional occupations and imaginary time are
  rejected. The old `PT2` prototype
  and its `PTCN_TDRHFSimulation` class have been removed.

```python
sim = TDRKSSimulation(mf, dt=0.1, steps=100, propagation_method='pt_cn',
                     convergence_threshold=1e-10, max_iterations=80)
sim.run()
```

`convergence_threshold` for PT-CN is the Frobenius norm of the unpreconditioned
AO equation residual before final orthonormalization. It differs from the MO
change criterion of the other two methods. `sim.pt_cn_residual` and
`sim.pt_cn_iterations` describe the last accepted step. `mu` scales the
Hamiltonian; the endpoint field uses the physical time `t + dt`.

For unrestricted propagation, use `TDUHFSimulation` or `TDUKSSimulation` with
an integer-occupied UHF/UKS mean-field object. Unequal spin populations and an
empty spin channel are supported by PT-CN. Balanced spin blocks retain the
existing stacked-array representation; unequal blocks are stored separately,
without padding. The legacy CN paths currently require balanced spin blocks.
`sim.pt_cn_spin_residuals` reports the two residual norms; acceptance uses their
combined Frobenius norm. Single-state moving nuclei use the symmetric driver
below. Unrestricted two-state DTO/NAC and MASH remain separate integration work.
Study reports and result artifacts are kept locally under `docs/validation/`
and are excluded from version control.

### Symmetric Lowdin Ehrenfest dynamics

Current trajectory acceptance separates solver completion from accuracy in
nuclear positions, total dipoles and energy. Spin-density and force errors are
retained as diagnostics in the local study reports.

Unrestricted Fock matrices and energies retain the full complex density,
including exchange. The corresponding force restores the imaginary exchange
contribution omitted by PySCF's real integral-gradient contraction.

For moving nuclei, use the symmetric Velocity Verlet splitting with Lowdin
transport from the old geometry to the midpoint, electronic substeps covering
the nuclear interval at that fixed midpoint, then transport to the new endpoint
and a final half-kick.
The force includes the corresponding Lowdin Pulay term and RKS/UKS grid response.
The driver supports closed-shell RHF/RKS and integer-occupied UHF/UKS, positive
finite `mu`, no external field or density fitting, and positive timesteps with
`nuclei_dt` an integer multiple of electronic `dt`. Both spin channels use the same geometry transport, with
their Pulay contributions summed. `mu` is the electronic scaling factor; the
legacy driver argument `alpha` must remain 1 to avoid a second scaling convention.

```python
sim = TDRKSSimulation(mf, dt=0.1, nuclei_dt=0.1, steps=100,
                     ehrenfest=True, propagation_method='pt_cn',
                     convergence_threshold=1e-11, max_iterations=120)
sim.run_ehrenfest_midpoint_decoupled()
```

`sim.run()` also selects this driver for moving-nucleus PT-CN. For comparisons
with `'accelerated'` or `'predictor_corrector'`, call
`run_ehrenfest_midpoint_decoupled()` for both methods to match the nuclear
splitting, transport and force. This driver rejects nonconverged electronic
steps and restores the preceding nuclear/electronic state. Histories include
the initial state and synchronized endpoints.

For two electronic substeps per nuclear step, set `nuclei_dt=2*dt`; `steps`
counts nuclear steps. A failure in any electronic substep restores the whole
nuclear step. Large-alpha excited-state accuracy still requires separate
trajectory and state-identity validation.

The timestep comparison script checks positions, forces, dipoles, orthogonal
density matrices and energy drift against an independent density/nuclear ODE
reference at two tolerances. It freezes the error budget before running and
compares shared physical times. The output directory must be new:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/validate_pt_cn_timestep.py \
  --output build/pt_cn_timestep
```

A larger stable timestep is useful only when it passes the same observable
error budget. This short local comparison does not establish long-time or
large-system performance.

The longer Euler workflow is in `scripts/euler_pt_cn_bomd.py`. Site-specific
Slurm wrappers are kept locally. It compares approximately 100 fs trajectories against native PySCF
Velocity Verlet BOMD, with matched isotope masses and initial conditions.
It reports energy drift, bond and angle errors, phase-related observables and
cost rather than applying the short test's pointwise electronic gate. Raw
trajectories stay on Euler scratch; local records contain scripts and summaries.

`scripts/benchmark_pt_cn_bomd.py` adds paired timings to the frozen Euler study.
Each RHF/RKS timestep (5, 10, 20 au) runs four BOMD/PT-CN pairs in alternating
order on one allocated CPU, after warmup. Timings include fresh initialization
and common endpoint observables; archive writes and extra density diagnostics
are excluded. Every timed trajectory must reproduce its validated predecessor.
The report includes observable errors alongside paired runtime ratios.

`scripts/euler_c2h4_ehrenfest.py` tests the unrestricted MOM excited state of
ethylene, emphasizing `mu=10` with timesteps 4 and 5 au. It separates integration
convergence from target-state retention, compares fixed-reference and following
MOM-SCF dynamics, and records ground/target determinant fidelities. Raw results
remain on Euler scratch. Passing a short trajectory does not establish the
largest usable timestep or recovery from an MOM branch change.

### Regression tests

Run the tests with the Python environment in which TDMASH and its dependencies
are installed:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
cmake -S . -B build -DPython3_EXECUTABLE="$(command -v python)"
ctest --test-dir build --output-on-failure
```

CTest covers restricted/unrestricted PT-CN models, fixed- and moving-nucleus propagation regressions,
and the timestep, native BOMD and paired timing self-checks. See [tests/README.md](tests/README.md)
for scope.

### Treatment of frustrated hops

`MASH.py` calls `Spin.rescale_momentum_on_hop_v4`, in which a frustrated hop
reflects the entire spin vector without rescaling the momentum. This scheme was
adopted because the mapping variables can cross the equator of the Bloch sphere
repeatedly in this setting, and reflecting the spin alone prevents those
repeated crossings from accumulating spurious changes in the nuclear
trajectory.

`Spin.py` retains the earlier variants `v0`-`v3` for reference; they are not
used by the published results.

## Examples

All examples treat *trans*-diimide (N2H2) with the 6-31G* basis.

| Script | What it does |
| --- | --- |
| `examples/run_fixed_path_rks.py` | Single trajectory with the nuclei held on the excited-state surface, no hopping. This is the input used for Secs. 4.2-4.5 of the paper. |
| `examples/run_fixed_path_uks.py` | The same, in the unrestricted formalism |
| `examples/run_mash_rks.py` | Full MASH dynamics with state-specific forces and momentum rescaling |
| `examples/setup.py`, `examples/run_single.py` | Batch driver that generates and launches one directory per trajectory for an ensemble run |

```bash
python examples/run_fixed_path_rks.py
```

## Initial conditions

`initial_conditions/` holds the 100-member Wigner ensemble used for the
published results (coordinates and velocities in atomic units, plus the
sampled initial spin vectors). To regenerate it:

```bash
python scripts/generate_NHNH_Wigner.py
```

`tdmash.Wigner` implements the nuclear sampling and `tdmash.sampler` the
Bloch-sphere sampling of the initial mapping variables.

## Notes

- Times in the code are in atomic units (1 a.u. = 0.0242 fs). The published
  calculations use an electronic and nuclear time step of 0.4 a.u.
- A larger `α` requires a smaller time step, since the phase accumulated per
  step grows in proportion to `α`.
- Output `.npz` and trajectory files are ignored by git; see `.gitignore`.

## License

Released under the MIT License; see [`LICENSE`](LICENSE). The permissive terms
are deliberate: they allow the code to be incorporated into other packages,
including ones that are not themselves openly licensed.

## Citation

Please cite the paper above if you use this code.
