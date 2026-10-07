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
temporary coefficient copies. Deferred spectral previews use the same dedicated
lab N and W*(E-Fv) inputs as publication, not the converted J/n mean. Physical
comoving numbers remain separate for the thermal checks. Thermal checks also
require finite `q/B_E`, current comoving-number/reference ratios, their pair
product, and the resulting pair rate
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

For eligible positive inputs, each species uses the factor

```text
Fv = sum_i F_i*v^i
mean_incoming = W*(E-Fv)/N
c = clamp[mean_incoming*(N_reference/J_reference), 1/f, f]^2
f = opacity_corr_fac_max
```

Here E, F_i and N are the stored densitized lab moments, v^i is the ordinary
Eulerian velocity (not the primitive W*v^i), and W is the fluid Lorentz factor.
The shared density weight cancels in the incoming mean. This direct expression
matches GRA's default opacity-mean branch for finite positive eligible states;
it bypasses the number-conversion cap for spectral correction only. Both
opacity-order paths use it, including the default with all options false.
Nonpositive current/reference inputs or a nonfinite ratio retain the neutral
factor policy; no new floor or GRA zero/invalid-state policy is introduced.

The shared number conversion still uses unrepaired E/F/J and the capped factor
min(1-Fv/E, 1-1e-5) before separate moment safety. Its converted number and
energy moments still feed equilibrium selection and source accounting; they
are not replaced by the dedicated spectral inputs. Removing the cap from the
incoming mean can change predictor weights and selected targets indirectly,
without changing those algorithms. Selected thermal references may consequently
change, although raw thermal r,q remain uncorrected. This does not reproduce
GRA's alternative fiducial J/n estimate or its trapped-cell correction bypass.

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

The direct incoming opacity mean is unconditional, including with all four
options omitted or false. It intentionally changes the former capped J/n
spectral correction; number conversion and source accounting retain their
existing prescriptions. Historical pre-thermal bitwise opacity records are
not an unchanged-policy oracle for this revision and must not be silently
regenerated or exempted from comparison.

`tst/test_weakrates_opacity_mean.py BUILD_DIR` is the focused standalone runner
for the direct mean and real coefficient publication. It requires fresh
Makefiles objects from this checkout, double precision, MPI/OpenMP off, and
Serial or CUDA; WeakRates supports four groups with NuRATES disabled here.
It does not require a historical reference, configure/rebuild the code, or run
a simulation loop. The fixture covers counter-streaming/rest/co-streaming,
geometry and velocity conventions, ordering/target-mean combinations, species
closure, conversion-cap preservation, spectral-preview/thermal-input separation,
guards and stage hold. These are
regression sources only until compiled and executed; no passing native or
CUDA result is established for the direct-mean revision.

Earlier source comparisons and interpreted formula checks concerned the
prior paths, switch combinations, target-selection boundaries, conserved
SFHo roots, spectral clamping and heavy Kirchhoff reconstruction. They do not
verify this revision. Compiled and physical comparison tests remain required
before production use.
