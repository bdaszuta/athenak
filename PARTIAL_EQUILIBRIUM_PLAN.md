# Heavy-pair partial-equilibrium weight correction

Scope: correct heavy-pair weight aggregation in the existing frozen-coefficient
partial-equilibrium predictor. This is not an exact solve of the later M1 source
step. Do not change rate production, EOS residual mathematics, coefficient
publication, source update, configuration, solver tolerances, gates' thresholds,
softening count, or fallback policy. Do not apply the stashed rewrites.

The EOS has one heavy energy weight because its two heavy equilibrium energy
densities coincide. This does not imply equal effective opacities or equal
current radiation densities. BnsNurates can produce unequal heavy coefficients
with reconstructed distributions. WeakRates starts from one heavy emission
channel but applies the non-LTE correction separately to the two evolved heavy
groups before inferring their absorption coefficients.

For each existing corrected heavy energy absorption coefficient kappa_s and the
predictor's full-step dtau = dt*alpha/W, calculate

    w_s = (dtau*kappa_s)/(1 + dtau*kappa_s), s = 2,3.

Pass w_EOS_x = (w_2 + w_3)/2 to GetBetaEquilibriumPartial. Its energy RHS must
contain w_2*J_2 + w_3*J_3 instead of w_EOS_x*(J_2+J_3). This is valid because
the EOS heavy equilibrium contribution is w_EOS_x*(J_eq,2 + J_eq,3) and the two
equilibrium densities are equal. Retain individual electron energy and number
weights, with opposite lepton-number signs. For the supported three-group path,
use the sole heavy weight directly and omit absent group 3.

## 1. bugfix/radiation_m1

- In `src/radiation_m1/radiation_m1_helpers.hpp`, add one Kokkos-inline helper
  taking dtau, the two corrected heavy absorption coefficients, and the active
  group count; return the individual weights and the EOS mean. Keep all other
  per-cell work at the existing call site.
- In `CalcOpacityNurates_` in
  `src/radiation_m1/radiation_m1_calc_opacities_nurates.cpp`, replace the
  J-weighted opacity average with that helper. Include both heavy weights in
  the finite and maximum-weight gates. In the energy-excursion bound use the
  sum of the two separately weighted absolute equilibrium differences. In each
  existing softened retry, multiply the two individual weights by the same
  softening factor, pass their mean to the EOS, and add their separate current
  density terms to the energy RHS. Preserve the zero-interaction shortcut and
  all existing acceptance and fallback behavior.
- Correct the explanation beside the arithmetic and the heavy-weight
  documentation in `src/eos/primitive-solver/ps_types.hpp` and
  `src/eos/primitive-solver/eos_compose.hpp`. The latter must distinguish the
  one EOS equilibrium weight from the separate current-density RHS weights;
  update `src/eos/primitive-solver/eos.hpp` documentation if its description
  would otherwise imply the two must always be identical.

## 2. feature/weakrates

- After approval of Step 1's branch tip, rebase its four existing commits on
  that tip; inspect `git range-diff` and preserve unrelated worktree changes.
  Do not pop the targeted stashes.
- In `CalcOpacityWeakRates_` in
  `src/radiation_m1/radiation_m1_calc_opacities_weakrates.cpp`, reuse the helper
  and apply precisely the same heavy-pair treatment to the gates, softened EOS
  weight and energy RHS. Explain beside the heavy rate construction that its
  per-group non-LTE corrections can make the inferred absorption rates unequal.
  Keep the supplied charged-current absorption, heavy emission construction,
  and all final coefficient assignments unchanged.

## Verification ownership

The user does not want the agent to compile or run. After each change the agent
may inspect its source diff and check whitespace, but must not claim executable
validation. The user may run the existing host/device M1 math test and EOS unit
path after adding cases with an independent per-group weighted RHS. Distinct
checks: unequal kappa and unequal current J must reject opacity averaging;
unequal heavy departures of opposite sign must reject a combined-absolute-value
gate; equal rates, zero rates, and thin/fallback cases must retain their current
limits. Verify both backend call sites use the corrected post-processing
coefficients, with neither backend's published microphysics altered.

Boundary: this corrects a defined aggregation error inside the current
predictor only. BnsNurates' nonthermal emission and the source step's stage
weights, number shortcut, and frame factors preclude claiming that this is an
exact material-radiation source solve. A request for that broader behavior
requires a separate numerical-method decision, not an implicit extension of
this patch.
