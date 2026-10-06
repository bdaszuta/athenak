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
- `tau <= dt/2`: full conserved equilibrium.
- `dt/2 < tau < dt`: `(1-lambda)*local + lambda*full`, where
  `lambda = 2*(1-tau/dt)`.

Interpolation acts on number/energy moments, not on temperature or Ye.
The full endpoint uses all four current radiation energy groups, the electron
lepton budget, and unit weights in the EOS trapped-neutrino solver. Each heavy
transport slot contains two states. Analytic energy/lepton residuals and the
interior root status are checked before publishing moments. A constrained,
energy-only, failed, or nonconserving endpoint aborts explicitly; it is not
silently replaced by the weighted solution. The local-only regime does not
need a full endpoint.

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

## Verification boundary

With all four options omitted or explicitly false, the standard source and
weighted-opacity paths remain unchanged apart from the unconditional number/frame
prescription. Focused source comparisons and interpreted formula checks cover
these paths, all 16 switch combinations, target-selection boundaries,
conserved SFHo roots, spectral clamping, and heavy Kirchhoff reconstruction.
They are not a C++ build, device test, or simulation validation. Compiled and
physical comparison tests remain required before production use.
