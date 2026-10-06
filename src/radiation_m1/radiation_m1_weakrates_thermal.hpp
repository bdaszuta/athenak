#ifndef RADIATION_M1_RADIATION_M1_WEAKRATES_THERMAL_HPP_
#define RADIATION_M1_RADIATION_M1_WEAKRATES_THERMAL_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_weakrates_thermal.hpp
//! \brief Device-local coupled electron grey thermal and charged-current sources.

#include <limits>
#include <type_traits>

#include "radiation_m1/radiation_m1_calc_closure.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"
#include "radiation_m1/radiation_m1_params.hpp"
#include "radiation_m1/radiation_m1_weakrates.hpp"

namespace radiationm1 {
namespace weakrates {

// Failures belong to this operator, not to the shared four-variable solver.
// The caller owns the cell/stage diagnostic and abort on a terminal failure.
enum class ElectronThermalFailure {
  None,
  InvalidInput,
  InvalidReference,
  InvalidTrial,
  NonfiniteSource,
  FiniteDifference,
  SingularJacobian,
  Backtracking,
  MaxIterations,
  ClosureRetry,
  NumberResidual,
  GlobalInvalidation,
  AcceptanceFailure
};

KOKKOS_INLINE_FUNCTION
const char *ElectronThermalFailureName(const ElectronThermalFailure failure) {
  switch (failure) {
    case ElectronThermalFailure::None: return "None";
    case ElectronThermalFailure::InvalidInput: return "InvalidInput";
    case ElectronThermalFailure::InvalidReference: return "InvalidReference";
    case ElectronThermalFailure::InvalidTrial: return "InvalidTrial";
    case ElectronThermalFailure::NonfiniteSource: return "NonfiniteSource";
    case ElectronThermalFailure::FiniteDifference: return "FiniteDifference";
    case ElectronThermalFailure::SingularJacobian: return "SingularJacobian";
    case ElectronThermalFailure::Backtracking: return "Backtracking";
    case ElectronThermalFailure::MaxIterations: return "MaxIterations";
    case ElectronThermalFailure::ClosureRetry: return "ClosureRetry";
    case ElectronThermalFailure::NumberResidual: return "NumberResidual";
    case ElectronThermalFailure::GlobalInvalidation: return "GlobalInvalidation";
    case ElectronThermalFailure::AcceptanceFailure: return "AcceptanceFailure";
  }
  return "Unknown";
}

// All coefficients/references are undensitized. number_star and explicit_number
// are stored densitized N; explicit_moment is pre-source u0, not the repaired
// transport/geometric baseline. Only the explicit route reads the explicit data.
struct ElectronThermalSourceInput {
  ElectronThermalCoefficients thermal{};
  Real volume{};
  Real number_star[2]{};
  Real eta_number_cc[2]{};
  Real abs_number_cc[2]{};
  Real explicit_moment[2][4]{};
  Real explicit_number[2]{};
};

// Absolute unlimited endpoints, not deltas. CC and pair attribution are full-
// step densitized increments. Publication/floors use one caller-owned acceptance
// factor for both species; only CC contributes to matter electron lepton number.
struct ElectronThermalSourceResult {
  Real moment[2][4]{};
  Real number[2]{};
  Real cc_number_increment[2]{};
  Real pair_number_increment{};
  Real chi[2]{};
  SrcSignal status = SrcFail;
  ElectronThermalFailure failure = ElectronThermalFailure::InvalidInput;
};

// Residual scratch is a value record: J/H are physical comoving moments, while
// moment_increment and cc_number_increment are densitized stage increments.
// Gamma, n, lambda and chi all describe exactly the same evaluation state.
struct ElectronThermalResidualState {
  Real number[2]{};
  Real gamma[2]{};
  Real comoving_number[2]{};
  Real comoving_energy[2]{};
  Real comoving_flux[2][4]{};
  Real lambda[2]{};
  Real moment_increment[2][4]{};
  Real cc_number_increment[2]{};
  Real number_residual[2]{};
  Real chi[2]{};
  Real pair_number_increment{};
};
static_assert(std::is_trivially_copyable<ElectronThermalSourceInput>::value,
              "ElectronThermalSourceInput must be device-copyable");
static_assert(std::is_trivially_copyable<ElectronThermalSourceResult>::value,
              "ElectronThermalSourceResult must be device-copyable");
static_assert(std::is_trivially_copyable<ElectronThermalResidualState>::value,
              "ElectronThermalResidualState must be device-copyable");

KOKKOS_INLINE_FUNCTION
ElectronThermalFailure ValidateElectronThermalInput(
    const SrcParams electron[2], const ElectronThermalSourceInput &input,
    const RadiationM1Params &params) {
  if (!Kokkos::isfinite(input.volume) || input.volume <= 0.0 ||
      !Kokkos::isfinite(params.source_epsabs) || params.source_epsabs < 0.0 ||
      !Kokkos::isfinite(params.source_epsrel) || params.source_epsrel < 0.0 ||
      !Kokkos::isfinite(params.rad_N_floor) || params.rad_N_floor < 0.0 ||
      !Kokkos::isfinite(params.rad_eps) || params.rad_eps < 0.0 ||
      params.rad_eps >= 1.0 ||
      (params.src_update != Explicit && params.src_update != Implicit)) {
    return ElectronThermalFailure::InvalidInput;
  }
  const auto &thermal = input.thermal;
  if (!Kokkos::isfinite(thermal.r) || thermal.r < 0.0 ||
      !Kokkos::isfinite(thermal.q) || thermal.q < 0.0) {
    return ElectronThermalFailure::InvalidInput;
  }
  for (int s = 0; s < 2; ++s) {
    if (!Kokkos::isfinite(thermal.number[s]) || thermal.number[s] < 0.0 ||
        !Kokkos::isfinite(thermal.energy[s]) || thermal.energy[s] < 0.0 ||
        ((thermal.r > 0.0 || thermal.q > 0.0) && thermal.number[s] <= 0.0) ||
        (thermal.q > 0.0 && thermal.energy[s] <= 0.0)) {
      return ElectronThermalFailure::InvalidReference;
    }
    const auto &saved = electron[s];
    if (!Kokkos::isfinite(saved.cdt) || saved.cdt < 0.0 ||
        !Kokkos::isfinite(saved.alp) || saved.alp <= 0.0 ||
        saved.cdt != electron[0].cdt || saved.alp != electron[0].alp ||
        !Kokkos::isfinite(saved.Estar) || saved.Estar <= 0.0 ||
        !Kokkos::isfinite(saved.chi) || !Kokkos::isfinite(saved.W) || saved.W <= 0.0 ||
        !Kokkos::isfinite(saved.eta) || saved.eta < 0.0 ||
        !Kokkos::isfinite(saved.kabs) || saved.kabs < 0.0 ||
        !Kokkos::isfinite(saved.kscat) || saved.kscat < 0.0 ||
        !Kokkos::isfinite(input.number_star[s]) || input.number_star[s] < 0.0 ||
        !Kokkos::isfinite(input.eta_number_cc[s]) || input.eta_number_cc[s] < 0.0 ||
        !Kokkos::isfinite(input.abs_number_cc[s]) || input.abs_number_cc[s] < 0.0) {
      return ElectronThermalFailure::InvalidInput;
    }
    for (int a = 0; a < 4; ++a) {
      if (!Kokkos::isfinite(saved.Fstar_d(a)) ||
          !Kokkos::isfinite(saved.n_d(a)) || !Kokkos::isfinite(saved.n_u(a)) ||
          !Kokkos::isfinite(saved.u_d(a)) || !Kokkos::isfinite(saved.u_u(a)) ||
          !Kokkos::isfinite(saved.v_d(a)) || !Kokkos::isfinite(saved.v_u(a))) {
        return ElectronThermalFailure::InvalidInput;
      }
      for (int b = 0; b < 4; ++b) {
        if (!Kokkos::isfinite(saved.g_dd(a, b)) ||
            !Kokkos::isfinite(saved.g_uu(a, b)) ||
            !Kokkos::isfinite(saved.gamma_ud(a, b)) ||
            !Kokkos::isfinite(saved.proj_ud(a, b))) {
          return ElectronThermalFailure::InvalidInput;
        }
      }
    }
  }
  const Real D = electron[0].cdt * electron[0].alp;
  if (!Kokkos::isfinite(D) || !Kokkos::isfinite(D * input.volume)) {
    return ElectronThermalFailure::NonfiniteSource;
  }
  return ElectronThermalFailure::None;
}

// Reject invalid trials before the existing closure can clip E. Positive subfloor
// energies/numbers are legal; configured publication floors are not root domains.
KOKKOS_INLINE_FUNCTION
ElectronThermalFailure PrepareElectronThermalMoments(
    const BrentFunctor &brent, const Real moment[4], SrcParams &workspace,
    const RadiationM1Params &params, const RadiationM1Closure closure,
    const Real volume, Real &gamma, Real &energy, Real flux[4]) {
  for (int a = 0; a < 4; ++a) {
    if (!Kokkos::isfinite(moment[a])) return ElectronThermalFailure::InvalidTrial;
  }
  if (moment[0] <= 0.0) return ElectronThermalFailure::InvalidTrial;
  workspace.E = moment[0];
  pack_F_d(-workspace.alp * workspace.n_u(1),
           -workspace.alp * workspace.n_u(2),
           -workspace.alp * workspace.n_u(3), moment[1], moment[2], moment[3],
           workspace.F_d);
  const Real flux2 = tensor_dot(workspace.g_uu, workspace.F_d, workspace.F_d);
  if (!Kokkos::isfinite(flux2) || flux2 < 0.0 ||
      Kokkos::sqrt(flux2) / workspace.E > 1.0) {
    return ElectronThermalFailure::InvalidTrial;
  }
  tensor_contract(workspace.g_uu, workspace.F_d, workspace.F_u);
  if (calc_closure(brent, workspace.g_dd, workspace.g_uu, workspace.n_d,
                   workspace.W, workspace.u_u, workspace.v_d, workspace.proj_ud,
                   workspace.E, workspace.F_d, workspace.chi, workspace.P_dd,
                   params, closure) != LinalgSuccess ||
      !Kokkos::isfinite(workspace.chi)) {
    return ElectronThermalFailure::InvalidTrial;
  }
  assemble_rT(workspace.n_d, workspace.E, workspace.F_d, workspace.P_dd,
              workspace.T_dd);
  workspace.J = calc_J_from_rT(workspace.T_dd, workspace.u_u);
  gamma = radiation_number_gamma(workspace.E, workspace.J, workspace.F_d,
                                  workspace.v_u, workspace.W, params);
  calc_H_from_rT(workspace.T_dd, workspace.u_u, workspace.proj_ud, workspace.H_d);
  Real h_fac{};
  if (!enforce_comoving_moment_safety(workspace.g_dd, workspace.g_uu,
                                      workspace.u_u, workspace.J, workspace.H_d,
                                      params, &h_fac)) {
    return ElectronThermalFailure::InvalidTrial;
  }
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> current{};
  assemble_fnu(workspace.g_uu, workspace.T_dd, workspace.u_u, workspace.J,
               h_fac, current);
  energy = workspace.J / volume;
  if (!Kokkos::isfinite(gamma) || gamma <= 0.0 ||
      !Kokkos::isfinite(energy) || energy <= 0.0) {
    return ElectronThermalFailure::InvalidTrial;
  }
  for (int a = 0; a < 4; ++a) {
    flux[a] = workspace.H_d(a) / volume;
    if (!Kokkos::isfinite(current(a)) || !Kokkos::isfinite(flux[a])) {
      return ElectronThermalFailure::InvalidTrial;
    }
  }
  return ElectronThermalFailure::None;
}

// Approved phenomenological grey closure, not a microscopic inverse kernel:
// R = r*(1 - (n0/BN0)*(n1/BN1)); lambda_s = (q/BE_s)*(n_p/BN_p).
// One shared pair count and q per species, with no heavy multiplicity. Raw
// blocking is already present in the retained rates; neither another blocking
// factor nor the CC mean-energy correction is applied here. Selected references
// use the unchanged approximate predictor. This source-only extension does not
// add thermal damping to the transport/asymptotic-preserving flux operator.
KOKKOS_INLINE_FUNCTION
ElectronThermalFailure EvaluateElectronThermalRates(
    const ElectronThermalCoefficients &thermal, const Real number[2],
    Real lambda[2], Real &pair_rate) {
  pair_rate = 0.0;
  lambda[0] = lambda[1] = 0.0;
  Real ratio[2]{};
  if (thermal.r > 0.0 || thermal.q > 0.0) {
    for (int s = 0; s < 2; ++s) {
      if (!Kokkos::isfinite(thermal.number[s]) || thermal.number[s] <= 0.0) {
        return ElectronThermalFailure::InvalidReference;
      }
      ratio[s] = number[s] / thermal.number[s];
      if (!Kokkos::isfinite(ratio[s]) || ratio[s] < 0.0) {
        return ElectronThermalFailure::NonfiniteSource;
      }
    }
  }
  if (thermal.r > 0.0) {
    const Real product = ratio[0] * ratio[1];
    if (!Kokkos::isfinite(product)) return ElectronThermalFailure::NonfiniteSource;
    pair_rate = thermal.r * (1.0 - product);
    if (!Kokkos::isfinite(pair_rate)) return ElectronThermalFailure::NonfiniteSource;
  }
  if (thermal.q > 0.0) {
    for (int s = 0; s < 2; ++s) {
      if (!Kokkos::isfinite(thermal.energy[s]) || thermal.energy[s] <= 0.0) {
        return ElectronThermalFailure::InvalidReference;
      }
      const Real coefficient = thermal.q / thermal.energy[s];
      lambda[s] = coefficient * ratio[1 - s];
      if (!Kokkos::isfinite(coefficient) || !Kokkos::isfinite(lambda[s])) {
        return ElectronThermalFailure::NonfiniteSource;
      }
    }
  }
  return ElectronThermalFailure::None;
}

// Workspace is a copy of saved: its const CC coefficients never accumulate
// thermal terms. eta is already densitized, so only q receives the volume form.
KOKKOS_INLINE_FUNCTION
ElectronThermalFailure ProjectElectronThermalSources(
    SrcParams &workspace, const ElectronThermalSourceInput &input, const int species,
    ElectronThermalResidualState &state) {
  const Real emission = workspace.eta + input.volume * input.thermal.q;
  const Real absorption = workspace.kabs + state.lambda[species];
  if (!Kokkos::isfinite(emission) || !Kokkos::isfinite(absorption)) {
    return ElectronThermalFailure::NonfiniteSource;
  }
  calc_rad_sources(emission, absorption, workspace.kscat, workspace.u_d,
                   workspace.J, workspace.H_d, workspace.S_d);
  workspace.Edot = calc_rE_source(workspace.alp, workspace.n_u, workspace.S_d);
  calc_rF_source(workspace.alp, workspace.gamma_ud, workspace.S_d, workspace.tS_d);
  state.moment_increment[species][0] = workspace.cdt * workspace.Edot;
  for (int a = 1; a < 4; ++a) {
    state.moment_increment[species][a] = workspace.cdt * workspace.tS_d(a);
  }
  for (int a = 0; a < 4; ++a) {
    if (!Kokkos::isfinite(workspace.S_d(a)) ||
        !Kokkos::isfinite(workspace.tS_d(a)) ||
        !Kokkos::isfinite(state.moment_increment[species][a])) {
      return ElectronThermalFailure::NonfiniteSource;
    }
  }
  const Real D = workspace.cdt * workspace.alp;
  state.cc_number_increment[species] =
      D * (input.volume * input.eta_number_cc[species] -
           input.abs_number_cc[species] * state.number[species] / state.gamma[species]);
  if (!Kokkos::isfinite(workspace.Edot) ||
      !Kokkos::isfinite(state.cc_number_increment[species])) {
    return ElectronThermalFailure::NonfiniteSource;
  }
  return ElectronThermalFailure::None;
}

// Nine unknowns: E0,F0x,F0y,F0z,E1,F1x,F1y,F1z,xi. xi is a densitized
// pair-number increment, NOT the closure flux factor. Every call restores the
// original closure seeds and immutable CC/geometry/baseline records; perturbation
// order cannot affect the result. Reconstructed numbers are never floored.
KOKKOS_INLINE_FUNCTION
ElectronThermalFailure EvaluateElectronThermalResidual(
    const BrentFunctor &brent, const SrcParams electron[2],
    const ElectronThermalSourceInput &input, const RadiationM1Params &params,
    const RadiationM1Closure closure, const Real x[9], Real residual[9],
    ElectronThermalResidualState &state) {
  auto failure = ValidateElectronThermalInput(electron, input, params);
  if (failure != ElectronThermalFailure::None) return failure;
  if (!Kokkos::isfinite(x[8])) return ElectronThermalFailure::InvalidTrial;
  SrcParams workspace[2] = {electron[0], electron[1]};
  ElectronThermalResidualState trial{};
  const Real D = electron[0].cdt * electron[0].alp;
  const Real A = D * input.volume;
  for (int s = 0; s < 2; ++s) {
    failure = PrepareElectronThermalMoments(
        brent, &x[4 * s], workspace[s], params, closure, input.volume,
        trial.gamma[s], trial.comoving_energy[s], trial.comoving_flux[s]);
    if (failure != ElectronThermalFailure::None) return failure;
    trial.chi[s] = workspace[s].chi;
    const Real denominator = 1.0 + D * input.abs_number_cc[s] / trial.gamma[s];
    const Real numerator = input.number_star[s] + A * input.eta_number_cc[s] + x[8];
    if (!Kokkos::isfinite(denominator) || denominator <= 0.0 ||
        !Kokkos::isfinite(numerator)) return ElectronThermalFailure::InvalidTrial;
    trial.number[s] = numerator / denominator;
    const Real current_volume = input.volume * trial.gamma[s];
    if (!Kokkos::isfinite(trial.number[s]) || trial.number[s] < 0.0 ||
        !Kokkos::isfinite(current_volume) || current_volume <= 0.0) {
      return ElectronThermalFailure::InvalidTrial;
    }
    trial.comoving_number[s] = trial.number[s] / current_volume;
    if (!Kokkos::isfinite(trial.comoving_number[s])) {
      return ElectronThermalFailure::NonfiniteSource;
    }
  }
  Real pair_rate{};
  failure = EvaluateElectronThermalRates(input.thermal, trial.comoving_number,
                                         trial.lambda, pair_rate);
  if (failure != ElectronThermalFailure::None) return failure;
  trial.pair_number_increment = A * pair_rate;
  if (!Kokkos::isfinite(trial.pair_number_increment)) {
    return ElectronThermalFailure::NonfiniteSource;
  }
  for (int s = 0; s < 2; ++s) {
    failure = ProjectElectronThermalSources(workspace[s], input, s, trial);
    if (failure != ElectronThermalFailure::None) return failure;
    for (int a = 0; a < 4; ++a) {
      const Real baseline = (a == 0) ? electron[s].Estar : electron[s].Fstar_d(a);
      residual[4 * s + a] = x[4 * s + a] - baseline - trial.moment_increment[s][a];
    }
    // Independently check the original physical equation and CC attribution,
    // rather than merely testing the eliminated denominator or shared pair row.
    trial.number_residual[s] = trial.number[s] - input.number_star[s] -
                               trial.cc_number_increment[s] - x[8];
    if (!Kokkos::isfinite(trial.number_residual[s])) {
      return ElectronThermalFailure::NonfiniteSource;
    }
  }
  residual[8] = x[8] - trial.pair_number_increment;
  for (int a = 0; a < 9; ++a) {
    if (!Kokkos::isfinite(residual[a])) return ElectronThermalFailure::NonfiniteSource;
  }
  state = trial;
  return ElectronThermalFailure::None;
}

// Complete physical Jacobian, including closure, Gamma, eliminated N and the
// partner losses at EVERY perturbation. Central differences are preferred; one
// valid side is sufficient at a domain boundary. Shrinking never changes scales.
KOKKOS_INLINE_FUNCTION
ElectronThermalFailure BuildElectronThermalJacobian(
    const BrentFunctor &brent, const SrcParams electron[2],
    const ElectronThermalSourceInput &input, const RadiationM1Params &params,
    const RadiationM1Closure closure, const Real x[9], const Real column_scale[9],
    Real jacobian[9][9]) {
  Real base[9]{};
  ElectronThermalResidualState scratch{};
  const auto failure = EvaluateElectronThermalResidual(
      brent, electron, input, params, closure, x, base, scratch);
  if (failure != ElectronThermalFailure::None) return failure;
  const Real step_factor = Kokkos::sqrt(std::numeric_limits<Real>::epsilon());
  for (int j = 0; j < 9; ++j) {
    if (!Kokkos::isfinite(column_scale[j]) || column_scale[j] <= 0.0) {
      return ElectronThermalFailure::FiniteDifference;
    }
    Real h = step_factor * Kokkos::fmax(Kokkos::fabs(x[j]), column_scale[j]);
    bool complete = false;
    for (int shrink = 0; shrink <= 32; ++shrink) {
      Real plus[9], minus[9], fplus[9]{}, fminus[9]{};
      for (int a = 0; a < 9; ++a) plus[a] = minus[a] = x[a];
      plus[j] = x[j] + h;
      minus[j] = x[j] - h;
      const bool valid_plus = Kokkos::isfinite(plus[j]) && plus[j] != x[j] &&
          EvaluateElectronThermalResidual(brent, electron, input, params, closure,
                                         plus, fplus, scratch) ==
              ElectronThermalFailure::None;
      const bool valid_minus = Kokkos::isfinite(minus[j]) && minus[j] != x[j] &&
          EvaluateElectronThermalResidual(brent, electron, input, params, closure,
                                         minus, fminus, scratch) ==
              ElectronThermalFailure::None;
      if (valid_plus || valid_minus) {
        const Real denominator = valid_plus && valid_minus ? plus[j] - minus[j] :
                                 valid_plus ? plus[j] - x[j] : x[j] - minus[j];
        bool finite_column = Kokkos::isfinite(denominator) && denominator > 0.0;
        for (int i = 0; i < 9; ++i) {
          const Real difference = valid_plus && valid_minus ? fplus[i] - fminus[i] :
                                  valid_plus ? fplus[i] - base[i] : base[i] - fminus[i];
          jacobian[i][j] = difference / denominator;
          finite_column = finite_column && Kokkos::isfinite(jacobian[i][j]);
        }
        if (finite_column) {
          complete = true;
          break;
        }
      }
      if (!Kokkos::isfinite(h) || h <= 0.0 ||
          (plus[j] == x[j] && minus[j] == x[j])) break;
      h *= 0.5;
    }
    if (!complete) return ElectronThermalFailure::FiniteDifference;
  }
  return ElectronThermalFailure::None;
}

// In-place partial-pivot elimination for this one scaled 9x9 system. No arbitrary
// pivot threshold: reject zero/nonfinite pivots and nonfinite arithmetic/updates.
KOKKOS_INLINE_FUNCTION
ElectronThermalFailure SolveElectronThermalLinearSystem(
    Real matrix[9][9], Real rhs[9], Real update[9]) {
  for (int i = 0; i < 9; ++i) {
    if (!Kokkos::isfinite(rhs[i])) return ElectronThermalFailure::SingularJacobian;
    for (int j = 0; j < 9; ++j) {
      if (!Kokkos::isfinite(matrix[i][j])) {
        return ElectronThermalFailure::SingularJacobian;
      }
    }
  }
  for (int k = 0; k < 9; ++k) {
    int pivot = k;
    for (int i = k + 1; i < 9; ++i) {
      if (Kokkos::fabs(matrix[i][k]) > Kokkos::fabs(matrix[pivot][k])) pivot = i;
    }
    if (matrix[pivot][k] == 0.0 || !Kokkos::isfinite(matrix[pivot][k])) {
      return ElectronThermalFailure::SingularJacobian;
    }
    if (pivot != k) {
      for (int j = k; j < 9; ++j) {
        const Real value = matrix[k][j];
        matrix[k][j] = matrix[pivot][j];
        matrix[pivot][j] = value;
      }
      const Real value = rhs[k];
      rhs[k] = rhs[pivot];
      rhs[pivot] = value;
    }
    for (int i = k + 1; i < 9; ++i) {
      const Real factor = matrix[i][k] / matrix[k][k];
      if (!Kokkos::isfinite(factor)) return ElectronThermalFailure::SingularJacobian;
      matrix[i][k] = 0.0;
      for (int j = k + 1; j < 9; ++j) {
        matrix[i][j] -= factor * matrix[k][j];
        if (!Kokkos::isfinite(matrix[i][j])) {
          return ElectronThermalFailure::SingularJacobian;
        }
      }
      rhs[i] -= factor * rhs[k];
      if (!Kokkos::isfinite(rhs[i])) return ElectronThermalFailure::SingularJacobian;
    }
  }
  for (int i = 8; i >= 0; --i) {
    Real value = rhs[i];
    for (int j = i + 1; j < 9; ++j) value -= matrix[i][j] * update[j];
    if (matrix[i][i] == 0.0 || !Kokkos::isfinite(matrix[i][i])) {
      return ElectronThermalFailure::SingularJacobian;
    }
    update[i] = value / matrix[i][i];
    if (!Kokkos::isfinite(value) || !Kokkos::isfinite(update[i])) {
      return ElectronThermalFailure::SingularJacobian;
    }
  }
  return ElectronThermalFailure::None;
}

// Tolerances are physical, component-wise, and unchanged. Scaling is used only
// for the local matrix and line-search merit, never as a convergence criterion.
KOKKOS_INLINE_FUNCTION
bool ElectronThermalPhysicalConvergence(
    const SrcParams electron[2], const ElectronThermalSourceInput &input,
    const RadiationM1Params &params, const Real x[9], const Real residual[9],
    const ElectronThermalResidualState &state, bool &number_consistent) {
  number_consistent = true;
  bool converged = true;
  for (int s = 0; s < 2; ++s) {
    for (int a = 0; a < 4; ++a) {
      const Real baseline = a == 0 ? electron[s].Estar : electron[s].Fstar_d(a);
      const Real tolerance = params.source_epsabs + params.source_epsrel *
          Kokkos::fmax(Kokkos::fabs(x[4 * s + a]), Kokkos::fabs(baseline));
      converged = converged && Kokkos::isfinite(tolerance) &&
          Kokkos::isfinite(residual[4 * s + a]) &&
          Kokkos::fabs(residual[4 * s + a]) <= tolerance;
    }
    const Real tolerance = params.source_epsabs + params.source_epsrel *
        Kokkos::fmax(Kokkos::fabs(state.number[s]), Kokkos::fabs(input.number_star[s]));
    number_consistent = number_consistent && Kokkos::isfinite(tolerance) &&
        Kokkos::isfinite(state.number_residual[s]) &&
        Kokkos::fabs(state.number_residual[s]) <= tolerance;
  }
  const Real pair_scale = Kokkos::fmax(Kokkos::fabs(x[8]),
      Kokkos::fmax(input.number_star[0], input.number_star[1]));
  const Real tolerance = params.source_epsabs + params.source_epsrel * pair_scale;
  return converged && Kokkos::isfinite(tolerance) && Kokkos::isfinite(residual[8]) &&
         Kokkos::fabs(residual[8]) <= tolerance;
}

KOKKOS_INLINE_FUNCTION
Real ElectronThermalResidualMerit(const Real residual[9], const Real row_scale[9]) {
  Real merit = 0.0;
  for (int i = 0; i < 9; ++i) {
    const Real value = Kokkos::fabs(residual[i] / row_scale[i]);
    if (!Kokkos::isfinite(value)) return std::numeric_limits<Real>::infinity();
    merit = Kokkos::fmax(merit, value);
  }
  return merit;
}

// Fixed scales refer to the original baseline, including during closure retries.
KOKKOS_INLINE_FUNCTION
ElectronThermalFailure SolveElectronThermalNewton(
    const BrentFunctor &brent, const SrcParams electron[2],
    const ElectronThermalSourceInput &input, const RadiationM1Params &params,
    const RadiationM1Closure closure, const Real column_scale[9],
    const Real row_scale[9], Real x[9], ElectronThermalResidualState &endpoint) {
  Real residual[9]{};
  ElectronThermalResidualState state{};
  auto failure = EvaluateElectronThermalResidual(
      brent, electron, input, params, closure, x, residual, state);
  if (failure != ElectronThermalFailure::None) return failure;
  for (int iter = 0; ; ++iter) {
    bool number_consistent{};
    if (ElectronThermalPhysicalConvergence(electron, input, params, x, residual,
                                           state, number_consistent)) {
      if (!number_consistent) return ElectronThermalFailure::NumberResidual;
      endpoint = state;
      return ElectronThermalFailure::None;
    }
    if (iter >= params.source_maxiter) return ElectronThermalFailure::MaxIterations;
    Real matrix[9][9]{}, rhs[9]{}, scaled_update[9]{}, update[9]{};
    failure = BuildElectronThermalJacobian(brent, electron, input, params, closure,
                                           x, column_scale, matrix);
    if (failure != ElectronThermalFailure::None) return failure;
    for (int i = 0; i < 9; ++i) {
      rhs[i] = -residual[i] / row_scale[i];
      for (int j = 0; j < 9; ++j) {
        matrix[i][j] = matrix[i][j] * column_scale[j] / row_scale[i];
      }
    }
    failure = SolveElectronThermalLinearSystem(matrix, rhs, scaled_update);
    if (failure != ElectronThermalFailure::None) return failure;
    for (int j = 0; j < 9; ++j) {
      update[j] = column_scale[j] * scaled_update[j];
      if (!Kokkos::isfinite(update[j])) return ElectronThermalFailure::SingularJacobian;
    }
    const Real merit = ElectronThermalResidualMerit(residual, row_scale);
    if (!Kokkos::isfinite(merit)) return ElectronThermalFailure::NonfiniteSource;
    Real damping = 1.0;
    bool accepted = false;
    for (int attempt = 0; attempt < 32; ++attempt) {
      Real candidate[9]{}, trial_residual[9]{};
      ElectronThermalResidualState trial{};
      bool distinct = false;
      for (int j = 0; j < 9; ++j) {
        candidate[j] = x[j] + damping * update[j];
        distinct = distinct || candidate[j] != x[j];
      }
      failure = EvaluateElectronThermalResidual(brent, electron, input, params,
                                                closure, candidate, trial_residual, trial);
      if (distinct && failure == ElectronThermalFailure::None) {
        bool trial_number_consistent{};
        const bool physical = ElectronThermalPhysicalConvergence(
            electron, input, params, candidate, trial_residual, trial,
            trial_number_consistent);
        if ((physical && trial_number_consistent) ||
            ElectronThermalResidualMerit(trial_residual, row_scale) < merit) {
          for (int j = 0; j < 9; ++j) {
            x[j] = candidate[j];
            residual[j] = trial_residual[j];
          }
          state = trial;
          accepted = true;
          break;
        }
        if (physical && !trial_number_consistent) {
          return ElectronThermalFailure::NumberResidual;
        }
      }
      damping *= 0.5;
    }
    if (!accepted) return ElectronThermalFailure::Backtracking;
    // Next loop checks physical convergence BEFORE another Jacobian, including
    // after the final permitted accepted Newton iteration.
  }
}

// Evaluate both pre-source states before constructing either endpoint. Finite
// negative explicit endpoints are allowed: only the shared caller acceptance
// guard may limit them. No implicit number reconstruction or shortcut is used.
KOKKOS_INLINE_FUNCTION
ElectronThermalFailure EvaluateElectronThermalExplicit(
    const BrentFunctor &brent, const SrcParams electron[2],
    const ElectronThermalSourceInput &input, const RadiationM1Params &params,
    ElectronThermalSourceResult &result) {
  SrcParams workspace[2] = {electron[0], electron[1]};
  ElectronThermalResidualState state{};
  for (int s = 0; s < 2; ++s) {
    auto failure = PrepareElectronThermalMoments(
        brent, input.explicit_moment[s], workspace[s], params, params.closure_type,
        input.volume, state.gamma[s], state.comoving_energy[s], state.comoving_flux[s]);
    if (failure != ElectronThermalFailure::None) return failure;
    state.number[s] = input.explicit_number[s];
    const Real current_volume = input.volume * state.gamma[s];
    if (!Kokkos::isfinite(state.number[s]) || state.number[s] < 0.0 ||
        !Kokkos::isfinite(current_volume) || current_volume <= 0.0) {
      return ElectronThermalFailure::InvalidTrial;
    }
    state.comoving_number[s] = state.number[s] / current_volume;
    if (!Kokkos::isfinite(state.comoving_number[s])) {
      return ElectronThermalFailure::NonfiniteSource;
    }
  }
  Real pair_rate{};
  auto failure = EvaluateElectronThermalRates(input.thermal, state.comoving_number,
                                             state.lambda, pair_rate);
  if (failure != ElectronThermalFailure::None) return failure;
  const Real A = electron[0].cdt * electron[0].alp * input.volume;
  result.pair_number_increment = A * pair_rate;
  if (!Kokkos::isfinite(result.pair_number_increment)) {
    return ElectronThermalFailure::NonfiniteSource;
  }
  for (int s = 0; s < 2; ++s) {
    failure = ProjectElectronThermalSources(workspace[s], input, s, state);
    if (failure != ElectronThermalFailure::None) return failure;
    result.chi[s] = workspace[s].chi;
    result.cc_number_increment[s] = state.cc_number_increment[s];
    result.number[s] = input.number_star[s] + result.cc_number_increment[s] +
                       result.pair_number_increment;
    if (!Kokkos::isfinite(result.number[s])) return ElectronThermalFailure::NonfiniteSource;
    for (int a = 0; a < 4; ++a) {
      const Real baseline = a == 0 ? electron[s].Estar : electron[s].Fstar_d(a);
      result.moment[s][a] = baseline + state.moment_increment[s][a];
      if (!Kokkos::isfinite(result.moment[s][a])) {
        return ElectronThermalFailure::NonfiniteSource;
      }
    }
  }
  return ElectronThermalFailure::None;
}

// Caller supplies two SrcParams with long-lived referenced tensors and baseline
// Fstar_d. eta is V*eta_E_cc; kabs/kscat are unmodified CC/scattering inputs.
// Backend/option/source-enable/activity gates belong to the caller. Success is
// the unlimited coupled endpoint plus CC-only/shared-pair attribution; failure
// has no publishable endpoint. No floors, limiter, matter update or View here.
KOKKOS_INLINE_FUNCTION
bool UpdateElectronThermalSources(
    const BrentFunctor &brent, SrcParams electron[2],
    const ElectronThermalSourceInput &input, const RadiationM1Params &params,
    ElectronThermalSourceResult &out) {
  out = ElectronThermalSourceResult{};
  auto failure = ValidateElectronThermalInput(electron, input, params);
  if (failure != ElectronThermalFailure::None) {
    out.failure = failure;
    return false;
  }
  ElectronThermalSourceResult result{};
  const Real D = electron[0].cdt * electron[0].alp;
  if (D == 0.0) {
    // No solver and no explicit-state reads for a zero stage interval.
    for (int s = 0; s < 2; ++s) {
      result.number[s] = input.number_star[s];
      result.chi[s] = electron[s].chi;
      for (int a = 0; a < 4; ++a) {
        result.moment[s][a] = a == 0 ? electron[s].Estar : electron[s].Fstar_d(a);
      }
    }
  } else if (params.src_update == Explicit) {
    failure = EvaluateElectronThermalExplicit(brent, electron, input, params, result);
  } else {
    Real x[9]{}, column_scale[9]{}, row_scale[9]{};
    const Real A = D * input.volume;
    Real number_scale = Kokkos::fmax(input.number_star[0], input.number_star[1]);
    number_scale = Kokkos::fmax(number_scale, A * input.thermal.r);
    number_scale = Kokkos::fmax(number_scale, params.rad_N_floor);
    if (!Kokkos::isfinite(number_scale)) {
      out.failure = ElectronThermalFailure::NonfiniteSource;
      return false;
    }
    if (number_scale == 0.0) number_scale = std::numeric_limits<Real>::min();
    for (int s = 0; s < 2; ++s) {
      for (int a = 0; a < 4; ++a) {
        x[4 * s + a] = a == 0 ? electron[s].Estar : electron[s].Fstar_d(a);
        column_scale[4 * s + a] = row_scale[4 * s + a] = electron[s].Estar;
      }
    }
    column_scale[8] = row_scale[8] = number_scale;
    ElectronThermalResidualState endpoint{};
    failure = SolveElectronThermalNewton(brent, electron, input, params,
                                         params.closure_type, column_scale, row_scale,
                                         x, endpoint);
    if (failure != ElectronThermalFailure::None && params.closure_type != Eddington) {
      // Restart Eddington at the repaired baseline, not the failed iterate.
      // Its root is ONLY a seed; only a configured-closure root can be accepted.
      for (int s = 0; s < 2; ++s) {
        for (int a = 0; a < 4; ++a) {
          x[4 * s + a] = a == 0 ? electron[s].Estar : electron[s].Fstar_d(a);
        }
      }
      x[8] = 0.0;
      RadiationM1Params seed_params = params;
      seed_params.closure_type = Eddington;
      failure = SolveElectronThermalNewton(brent, electron, input, seed_params,
                                           Eddington, column_scale, row_scale, x, endpoint);
      if (failure == ElectronThermalFailure::None) {
        failure = SolveElectronThermalNewton(brent, electron, input, params,
                                             params.closure_type, column_scale, row_scale,
                                             x, endpoint);
      }
      if (failure != ElectronThermalFailure::None) {
        failure = ElectronThermalFailure::ClosureRetry;
      }
    }
    if (failure == ElectronThermalFailure::None) {
      result.pair_number_increment = x[8];
      for (int s = 0; s < 2; ++s) {
        result.number[s] = endpoint.number[s];
        result.cc_number_increment[s] = endpoint.cc_number_increment[s];
        result.chi[s] = endpoint.chi[s];
        for (int a = 0; a < 4; ++a) result.moment[s][a] = x[4 * s + a];
      }
    }
  }
  if (failure != ElectronThermalFailure::None) {
    out.failure = failure;
    return false;
  }
  result.failure = ElectronThermalFailure::None;
  result.status = params.src_update == Explicit ? SrcThin : SrcOk;
  out = result;
  return true;
}

}  // namespace weakrates
}  // namespace radiationm1
#endif  // RADIATION_M1_RADIATION_M1_WEAKRATES_THERMAL_HPP_
