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

## Requirements

- Python >= 3.9
- `pyscf`, `numpy`, `scipy`, `matplotlib`

```bash
pip install pyscf numpy scipy matplotlib
```

The modules import each other by bare name, so run scripts from the repository
root or add it to `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/TDMASH_pack:$PYTHONPATH
```

## Module layout

| Module | Purpose |
| --- | --- |
| `TDRHF.py`, `TDRKS.py` | Spin-restricted real-time propagators (HF and KS) |
| `TDUHF.py`, `TDUKS.py` | Spin-unrestricted counterparts |
| `DIIS.py`, `Anderson.py` | Convergence acceleration for the SCF and for the predictor-corrector iteration |
| `MOM.py` | Maximum-overlap method, used to prepare the ΔSCF excited state at the first step |
| `TDOverlap.py`, `TDUOverlap.py` | Many-electron overlaps and the DTO gauge fixing |
| `Spin.py` | MASH mapping variables, hopping criterion and momentum rescaling |
| `FD.py` | Finite-difference nonadiabatic coupling vectors |
| `PT2.py` | Predictor-corrector Crank-Nicolson propagation utilities |
| `Verlet.py` | Velocity Verlet nuclear integrator |
| `Ehrenfest2.py` | Driver for fixed-path runs: nuclei on the excited-state surface, mapping variables propagated as a diagnostic, no hops |
| `MASH.py`, `MASH2.py` | Surface-hopping drivers |

### Which MASH driver to use

`MASH2.py` is the production driver and the one used for the published
results. It differs from `MASH.py` only in the treatment of a hop: `MASH2`
calls `Spin.rescale_momentum_on_hop_v4`, in which a frustrated hop reflects the
entire spin vector without rescaling the momentum, whereas `MASH.py` uses the
earlier `v1` prescription. The `v4` scheme was adopted because the mapping
variables can cross the equator of the Bloch sphere repeatedly in this setting,
and reflecting the spin alone prevents those repeated crossings from
accumulating spurious changes in the nuclear trajectory.

`Spin.py` retains the earlier variants `v0`-`v3` for reference; they are not
used by the published results.

## Examples

All examples treat *trans*-diimide (N2H2) with the 6-31G* basis.

| Script | What it does |
| --- | --- |
| `examples/run_fixed_path_rks.py` | Single trajectory with the nuclei held on the excited-state surface, no hopping. This is the input used for Secs. 4.2-4.5 of the paper. |
| `examples/run_fixed_path_uks.py` | The same, in the unrestricted formalism |
| `examples/run_mash_rks.py` | Full MASH dynamics with state-specific forces and momentum rescaling (`MASH2`, `v4`) |
| `examples/setup.py`, `examples/run_single.py` | Batch driver that generates and launches one directory per trajectory for an ensemble run |

```bash
cd TDMASH_pack
python examples/run_fixed_path_rks.py
```

## Initial conditions

`initial_conditions/` holds the 100-member Wigner ensemble used for the
published results (coordinates and velocities in atomic units, plus the
sampled initial spin vectors). To regenerate it:

```bash
python scripts/generate_NHNH_Wigner.py
```

`scripts/Wigner.py` implements the sampling itself and `scripts/sampler.py`
the Bloch-sphere sampling of the initial mapping variables.

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
