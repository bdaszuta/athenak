# GRA-style WeakRates policies

These controls are independent, default false, and read at construction.
They change the finite-step prescription; they do not change the raw rate
constants, FD approximation, EOS table, pgen, or source limiter.

```ini
<radiation_m1>
gra_source_dispatch = false

<weakrates>
gra_equilibrium_targets = false
gra_opacity_order = false
gra_target_mean = false
```

Set all four to true to select the combined comparison prescription. The
core dispatch details and controls are in `GRA_SOURCE_DISPATCH.md`.
Number/frame conversion is unconditional and has no enable switch.

## Target selection

`gra_equilibrium_targets = false` retains the existing `use_partial_equilibrium`
weighted predictor, its convergence controls, and its failure behavior.

When true, target selection uses coordinate mesh `dt` and the uncorrected
energy absorption/scattering rates of the two electron species:

```text
tau_s = [ka_s * (ka_s + ks_s)]^(-1/2)
tau = min(tau_nue, tau_anti)
```

The implementation evaluates this stably without overflowing the product.
Nonpositive absorption gives infinite relaxation time. The selected moments are:

- `tau >= dt`: local matter equilibrium.
- `tau <= dt/2`: full-equilibrium endpoint, or its best usable in-table approximation.
- `dt/2 < tau < dt`: `(1-lambda)*local + lambda*endpoint`, where
  `lambda = 2*(1-tau/dt)`.

Interpolation acts on number/energy moments, not on temperature or Ye.
The endpoint uses all four current radiation energy groups, the electron
lepton budget, and unit weights in the EOS trapped-neutrino solver. Each heavy
transport slot contains two states. A conserving interior or table-edge root
is accepted using the unchanged analytic residual tolerance, not the approximate
WeakRates FD target moments. The local-only regime does not need a full endpoint.

When an exact usable root is unavailable, table-bound exhaustion and Newton
failure are recoverable. Keep the original budgets and score in-table candidates:

```text
dE  = EOS_matter_energy + sum(three analytic neutrino-pair energies) - E_rhs
dYL = Ye + analytic_net_electron_neutrino_number_in_EOS_units/nb - YL_rhs
score = hypot(dE/S_E, dYL/S_L)
S_E = max(sum(current_J) + abs(e_local - e(nb,T_min,Ye_local)), eps*abs(E_rhs))
S_L = max(abs(Ye_local) + (current_N_nue + current_N_anti)/nb, eps*(Ye_max-Ye_min))
```

`eps` is machine epsilon, used only to keep scales positive. Scales are fixed
per cell. The energy scale measures warm-matter/radiation exchange rather than
rest-mass energy; the lepton scale avoids division by a cancelling signed budget.
Exactly zero `YL_rhs` bypasses the legacy EOS solve's singular norm. Its conserving
classification instead uses `S_L` for the lepton residual, with the same tolerance.

Retain the EOS-returned constrained, energy-only, or finite failed endpoint and
local safety target. Search all four table edges, including corners, using 16
coarse intervals and up to 48 golden refinements per sampled minimum. Refine the
returned, local, and best-found states inside the domain with up to 96 eight-direction
polls in normalized log-temperature/Ye coordinates. All accepted targets are scored
independently; EOS status is provenance, not acceptance. Retain the best usable
candidate even on stalled refinement. Equal scores are resolved by proximity to local
matter. This bounded search is not a global-minimum certificate for a nonconvex table.

Candidate acceptance checks final interpolated moments, spectral corrections,
Kirchhoff products/quotients, and enabled thermal reference requirements using
temporary coefficient copies. Thermal checks also require finite `q/B_E`, current
comoving-number/reference ratios, their pair product, and the resulting pair rate
and partner-dependent absorption. These checks cover target-dependent failures at
the current state; the consumer still validates future implicit trial states.
An unusable trial cannot modify the published state
or win merely because its analytic residual is small. Raw rates remain evaluated
at local matter; no target floors, EOS extrapolation, budget modification, or
weighted-predictor substitution is introduced. The evolved state, limiter, and
paired material feedback are unchanged.

Quality distinguishes local selection, conserving equilibrium, approximate
endpoint, local safety fallback, unsupported/invalid inputs, and no usable target.
One summary per pack per mesh timestep reports interior-cell fallback counts,
local-safety counts, evaluation totals, worst normalized score and separate
absolute normalized energy/lepton defects, and one representative cell with
signed defects and EOS status. Ghost cells are
still evaluated but excluded from these counts; no mesh-sized diagnostic arrays
or per-cell warning stream are added.

Abort only for unsupported/invalid inputs or no publishable candidate, including
the local safety target. A non-interior status, nonzero approximation defect, or
failed Newton solve alone does not abort or turn a usable cell transparent.
Existing original microscopic-invalidity zeroing and independent thermal/source
invariant failures retain their policies.

This path currently requires the native-nuclear, one-species CompOSE-like EOS
used by WeakRates. It does not inherit the `use_partial_equilibrium` enable
condition. The selector has no lapse or Lorentz-factor timestep conversion.

## Ordering and mean

| `gra_opacity_order` | `gra_target_mean` | Predictor input | Published correction reference |
|---|---|---|---|
| false | false | local-corrected rates | local moments |
| false | true | local-corrected rates | selected moments, after resetting spectral inputs |
| true | false | uncorrected rates | local moments, after selection |
| true | true | uncorrected rates | selected moments, after selection |

The target switch independently decides whether the selected moments come
from the standard predictor or the raw-timescale/full-endpoint policy.
Microscopic rates remain evaluated at local matter state, not endpoint matter.

For each species the factor is

```text
c = clamp[(J/N)*(N_reference/J_reference), 1/f, f]^2
f = opacity_corr_fac_max
```

Current `J` and `N` are fluid-frame quantities. The shared number conversion
uses unrepaired E/F/J before the separate moment-safety repair.
Scattering and electron CC absorption are multiplied by `c`. Electron emission
is reconstructed at the selected moments. Heavy spontaneous emission is
multiplied by `c` and retained; heavy absorption is inferred at the selected
moments. Resetting spectral inputs prevents multiplying a local factor and a
selected factor into the published coefficient.

Raw electron thermal `r,q` are not spectrally corrected or folded into transport
absorption. The separate thermal operator retains its enable control, clearing,
validation, and selected electron references. Existing density/temperature
masks and microscopic-invalidity zeroing are unchanged.

## Coupled electron thermal sources

When thermal emission is active, the coupled solver consumes the same two
stage-held CC regimes as the ordinary source path. The saved regimes are not
recomputed using thermal rates. Mixed electron regimes are supported:

- Nonstiff: CC/scattering E/F is forward Euler at the repaired pre-source baseline;
  thermal E/F remains backward Euler at the endpoint.
- Stiff: the existing combined CC/scattering/thermal backward-Euler equations.
- Equilibrium: the common GRA approximate reconstruction, with local solve inputs
  `eta_CC + alpha*V*q` and `ka_CC + alpha*lambda_s`. Thermal partner losses remain
  implicit; CC keeps GRA's literal stage-h denominators and thermal terms retain
  their lapse factor. These additions are not published transport opacities.

The coupled nine-variable solve remains responsible for the endpoint, including
implicit pair-number exchange and CC-number absorption. GRA dispatch overrides
legacy `src_update` selection for this path; when dispatch is false, its explicit
and implicit paths are unchanged apart from the unconditional frame conversion.
The shared limiter and paired material update still apply once to the complete
source increment. The ordinary electron source loop skips thermally active groups,
so no source is applied twice.

This is an extension of GRA's CC dispatch to the separate ATK thermal model,
not a claim that GRA contains this coupled thermal model. Moving-frame compiled
solver validation remains outstanding.

## Verification boundary

With all four options omitted or explicitly false, the standard source and
weighted-opacity paths remain unchanged apart from the unconditional number/frame
prescription. Focused source comparisons and interpreted formula checks cover
these paths, all 16 switch combinations, target-selection boundaries,
conserved SFHo roots, spectral clamping, and heavy Kirchhoff reconstruction.
They are not a C++ build, device test, or simulation validation. Compiled and
physical comparison tests remain required before production use.
