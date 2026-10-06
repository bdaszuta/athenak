# GRA number conversion and optional energy dispatch

## Unconditional number/frame conversion

`radiation_number_gamma(E, J, F_d, v_u, W, params)` implements

```
Gamma = W * (E/J) * min(1 - sum_i(F_i v^i)/E, 1 - 1e-5)
```

when both `E` and `J` are strictly above the existing `rad_E_floor`, and
returns `W` otherwise. Inputs are the intended moments before comoving safety
repair; `v_u` is ordinary Eulerian velocity, not `W*v`. The fixed `1e-5` is
independent of `rad_eps`. No native E/N floors are changed. Source updates,
NuRates number-density reconstruction, and number transport share this scalar
conversion. The pre-existing safety projection of the spatial number current
and moment repairs remain in place.

## Parameters in `<radiation_m1>`

| Parameter | Default | Meaning |
|---|---:|---|
| `gra_source_dispatch` | `false` | Enable the frozen GRA-style **energy/momentum** dispatcher. |
| `gra_source_rho_min` | `0` | Minimum fluid rest-mass density, in **code units**, for equilibrium. |
| `gra_source_temp_min_mev` | `0` | Minimum temperature, in **MeV**, for equilibrium. |

Gates must be finite and nonnegative. A positive temperature gate requires a
fluid EOS. The gate uses cached C2P temperature for evolving matter, or the EOS
pressure-to-temperature inversion for fixed evolution, with explicit code-unit
to MeV conversion. The default zero temperature gate performs no EOS calls.

With the switch off, the existing `src_update`, source-limit thresholds,
analytic seed, and nonlinear solver arithmetic retain their original energy
behavior. No regime View is allocated. The scalar number conversion above is
unconditional, including when this switch is off.

## Enabled energy policy and precedence

This switch takes precedence over `src_update` **for E/F only**. At stage 1,
after opacity calculation, each active cell/species records a regime using the
full mesh timestep `dt` and the final available energy absorption/scattering
coefficients `ka`, `ks`:

1. **Equilibrium:** `dt*sqrt(ka*(ka+ks)) > 1`, `rho >= gra_source_rho_min`, and
   `T_MeV >= gra_source_temp_min_mev`.
2. Otherwise **nonstiff:** both `(dt/2)*ka < 1` and `(dt/2)*ks < 1`.
3. Otherwise **stiff**.

These strict optical-depth boundaries are intentional. The recorded regime
persists through stage 2. Classification contains no lapse or Lorentz factor.
The enabled policy does not consult `source_thin_limit`, `source_thick_limit`,
or `source_scat_limit`, and has no distinct scattering shortcut: scattering
uses FE when nonstiff and the nonlinear solve when stiff. This matches the
selected reference configuration with disabled thick/scattering source-limit
shortcuts, supplemented by the approved full-dt equilibrium criterion.

For each stage's `h = beta[stage-1]*dt`:

- **Nonstiff:** existing explicit source update evaluated on the raw
  transport/geometric predictor, not the prior stage's E/F.
- **Stiff:** existing nonlinear backward-Euler solver, with the recorded regime
  forced through both primary and Eddington fallback calls. No stage-local
  reclassification is allowed. The GRA approximate thick state is its seed.
- **Equilibrium:** directly publish the GRA approximate first-order thick state;
  do not run Newton. As in GRA's approximate integrator, first physicalize the
  transport predictor and compute its closure/comoving moments, then use

```
J'         = (W*Jstar + h*eta*volform)/(W + h*ka)
Htilde_i'  = W*Htilde_star_i/(W + h*(ka + ks))
Hn         = sum_i(v^i*Htilde_i')
E'         = (4*W^2 - 1)*J'/3 + 2*W*Hn
F_i'       = 4*W^2*J'*v_i/3 + W*(Hn*v_i + Htilde_i')
```

Here `Htilde` is the spatial Eulerian projection of comoving H. Its spatial
covariant components equal `Hstar_d(i)` because `n_i=0`. **No lapse appears in
these relaxation denominators.** This is not the legacy lapse-scaled,
anisotropic ATK analytic seed. The existing finite-state checks, floors,
closure refresh, and analytic fallback mechanism still apply. A failed stiff
solve falls back to its GRA approximate state; failed FE uses the physicalized
predictor. Nonfinite analytic states disable the complete cell source step as
before.

## Number, limiter, and matter coupling

The selected ATK number mode is retained independently of the energy regime:
`src_update=explicit` uses the old stage's E/F/closure for its number source;
`src_update=implicit` uses the existing number BE update or
`source_therm_limit` average-energy override. Equilibrium energy classification
**does not suppress number sources**. No GRA equilibrium-number policy is added.

Source increments remain measured against the **raw** predictor. The existing
raw-predictor limiter, repair behavior, and conservative matter-coupling
accounting are unchanged. The optional dispatcher changes neither opacity
microphysics nor EOS, FD reconstruction, pgen, or limiter algorithms.

## Reference and verification limits

Formula reference: GRA source tree at
`f534ca0f6fa0d55b90df1686fa9abd476730bf3b`,
`src/m1/m1_utils.hpp` (`ToFiducial`, lines 851–858) and
`src/m1/m1_integrators_implementation.cpp`
(`PrepareApproximateFirstOrder_E_F_d`, lines 112–212).

This implementation was checked using interpreted scalar/formula tests and
static source comparisons only. Those checks do not establish C++ compilation,
Kokkos execution, nonlinear convergence, or simulation-level agreement.
