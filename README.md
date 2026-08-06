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
| `tdmash.PT2` | Predictor-corrector Crank-Nicolson propagation utilities |
| `tdmash.Verlet` | Velocity Verlet nuclear integrator |
| `tdmash.Ehrenfest` | Driver for fixed-path runs (`FixedPathDynamics`): nuclei on the excited-state surface, mapping variables propagated as a diagnostic, no hops |
| `tdmash.MASH` | Surface-hopping driver (`MASH`) |
| `tdmash.Wigner`, `tdmash.sampler` | Wigner sampling of nuclear initial conditions and of the initial mapping variables |

### Electronic propagation

Two propagation schemes are available, selected with the `propagation_method`
argument of the simulation classes:

- `'predictor_corrector'` — Crank-Nicolson with a plain predictor-corrector
  iteration for the midpoint Fock matrix.
- `'accelerated'` — the same scheme with Anderson mixing applied to the Fock
  matrix, which converges the iteration in far fewer cycles in the stiff,
  large-`α` regime. **This is the scheme used for the published results** and
  the one all the examples select.

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

<!-- TODO: choose a license before publishing (e.g. MIT or Apache-2.0) and add
     the corresponding LICENSE file. -->

## Citation

Please cite the paper above if you use this code.
