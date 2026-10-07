#ifndef RADIATION_M1_RADIATION_M1_WEAKRATES_EQUILIBRIUM_HPP_
#define RADIATION_M1_RADIATION_M1_WEAKRATES_EQUILIBRIUM_HPP_
//========================================================================================
// AthenaK grey M1: GRA-style local/full/interpolated weak-equilibrium targets.
// Local microscopic rates and the existing FD approximation remain unchanged.
//========================================================================================

#include <limits>
#include <type_traits>

#include "eos/primitive-solver/ps_types.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "radiation_m1_weakrates.hpp"
#include "radiation_m1_weakrates_equilibrium_search.hpp"

namespace radiationm1 {
namespace weakrates {
namespace gra_equilibrium_detail {

// Refer to the EOS policy's status symbol only in a supported instantiation.
// Ideal-gas/piecewise-polytrope policies need not define NUEQ_INTERIOR. This
// header does not include EOSCompOSE or instantiate a particular policy.
template<class EOS, class = void>
struct HasInteriorStatus : std::false_type {};
template<class EOS>
struct HasInteriorStatus<EOS, std::void_t<decltype(EOS::NUEQ_INTERIOR)>>
    : std::true_type {};

// ka and ks have already been validated as finite/nonnegative. Avoid both
// ka*(ka+ks) and ka+ks overflow, including for unequal rates. An infinite tau
// for vanishingly small absorption is a valid local-regime selector result.
KOKKOS_INLINE_FUNCTION
Real RawEnergyTimescale(const Real ka, const Real ks) {
  if (ka == 0.0) return std::numeric_limits<Real>::infinity();
  if (ks == 0.0) return 1.0/ka;
  const Real scale = (ka > ks) ? ka : ks;
  const Real tau = ((1.0/Kokkos::sqrt(ka))/Kokkos::sqrt(scale))/
                   Kokkos::sqrt(ka/scale + ks/scale);
  return Kokkos::isfinite(tau) ? tau : std::numeric_limits<Real>::infinity();
}

}  // namespace gra_equilibrium_detail

//! Build the four transport-group targets selected by the GRA raw-opacity gate.
//!
//! Arrays have four entries: nu_e, anti-nu_e, and two heavy groups, each heavy
//! group representing TWO states. N is in EOS number-density units; J is in code
//! energy-density units. nb is in EOS number-density units. temp and ye are the
//! caller's locally evaluated matter temperature (code units) and electron
//! fraction. The supported full solve is the one-species CompOSE-like EOS with
//! native nuclear units (MeV, fm^-3). dt is the COORDINATE mesh timestep, not
//! dt*alpha/W. Raw rates are inverse code length (c_code=1), before mean-energy
//! corrections and before any non-CC-emission/absorption fold. Only entries 0/1
//! of raw_abs_1 and raw_scat_1 are read. The selector is ALWAYS raw, independent
//! of optional pipeline/mean-energy flags.
//!
//! local_number/local_energy must come from the unchanged WeakRates
//! ComputeEquilibriumDensities + ConvertToCodeUnits evaluation at local matter,
//! with 0.5 of its all-heavy entry in each heavy group. No extra density gate
//! or floor is imposed here beyond EOS-domain validation for a requested solve.
//!
//! tau = min_s[1/sqrt(ka_s*(ka_s+ks_s))], s=0,1.
//! tau < dt/2: full target; dt/2 <= tau < dt: lambda=2-2*tau/dt
//! mixture of full and local MOMENTS; otherwise: exact copies of local moments.
//!
//! Full RHS: e_rhs=GetEnergy(nb,temp,ye)+SUM_s J_s;
//! Yl=ye+(N_0-N_1)/nb. All five partial-equilibrium weights are exactly one.
//! Independently validate analytic residuals, including conserving edge roots.
//! Otherwise minimize scaled budget defects inside the closed table domain.
//! GetTrappedNeutrinos returns CODE number density; explicitly undo that
//! conversion before dividing by nb. Its three energy entries are PAIRS
//! (electron, muon, tau), already in code energy-density units.
//!
//! Selected FD moments use mu_l at the solver's (T_root,Ye_root). Never replace
//! Ye_root by the GRA postprocessed Ye before reevaluating mu_l. The residual
//! belongs to the analytic root, not to the FD approximation or to a mixture.
//!
//! Return typed quality after writing usable targets. Unusable outcomes leave
//! outputs unchanged. Table bounds, failed Newton solves and approximate budget
//! defects are not fatal. Yl==0 bypasses the legacy solver's singular norm.
//! Optional validation checks the complete coefficient/thermal pipeline on
//! copies before acceptance. Local selection needs no EOS solver.
//! Read-only input arrays must not overlap mutable output arrays. The function
//! does not modify current/local moments, microscopic coefficients, or raw r,q.
template<class EOS>
KOKKOS_INLINE_FUNCTION
GraEquilibriumResult BuildGraEquilibriumTargets(
    const EOS &eos, Real nb, Real temp, Real ye, Real dt,
    const Real *raw_abs_1, const Real *raw_scat_1,
    const Real *current_number, const Real *current_energy,
    const Real *local_number, const Real *local_energy,
    Real *selected_number, Real *selected_energy,
    const GraTargetValidation *validation = nullptr) {
  GraEquilibriumResult invalid{};
  for (int s = 0; s < 4; ++s) {
    if (!Kokkos::isfinite(local_number[s]) || local_number[s] < 0.0 ||
        !Kokkos::isfinite(local_energy[s]) || local_energy[s] < 0.0) return invalid;
  }
  if (!Kokkos::isfinite(dt) || dt < 0.0) return invalid;
  for (int s = 0; s < 2; ++s) {
    if (!Kokkos::isfinite(raw_abs_1[s]) || raw_abs_1[s] < 0.0 ||
        !Kokkos::isfinite(raw_scat_1[s]) || raw_scat_1[s] < 0.0) return invalid;
  }
  const Real tau_e = gra_equilibrium_detail::RawEnergyTimescale(
      raw_abs_1[0], raw_scat_1[0]);
  const Real tau_a = gra_equilibrium_detail::RawEnergyTimescale(
      raw_abs_1[1], raw_scat_1[1]);
  const Real tau = tau_e < tau_a ? tau_e : tau_a;
  if (dt == 0.0 || !(tau < dt)) {
    if (validation != nullptr && !(*validation)(local_number, local_energy)) {
      invalid.status = GraEquilibriumStatus::NoUsableTarget;
      return invalid;
    }
    for (int s = 0; s < 4; ++s) {
      selected_number[s] = local_number[s]; selected_energy[s] = local_energy[s];
    }
    GraEquilibriumResult result{};
    result.status = GraEquilibriumStatus::Local;
    result.temp = temp; result.ye = ye;
    return result;
  }
  if constexpr (!gra_equilibrium_detail::HasInteriorStatus<EOS>::value) {
    invalid.status = GraEquilibriumStatus::Unsupported;
    return invalid;
  } else {
    if (!Kokkos::isfinite(nb) || nb <= 0.0 || !Kokkos::isfinite(temp) ||
        temp <= 0.0 || !Kokkos::isfinite(ye) || eos.GetNSpecies() != 1) return invalid;
    // Metadata getters are non-const references; never modify the EOS here.
    const Primitive::UnitSystem code_units =
        const_cast<EOS &>(eos).GetCodeUnitSystem();
    const Primitive::UnitSystem eos_units =
        const_cast<EOS &>(eos).GetEOSUnitSystem();
    const Real t_to_eos = code_units.TemperatureConversion(eos_units);
    const Real mu_to_eos = code_units.ChemicalPotentialConversion(eos_units);
    const Real n_to_eos = code_units.NumberDensityConversion(eos_units);
    Primitive::UnitSystem cgs_units{};
    cgs_units.length = cgs_units.time = cgs_units.numberDensity = 1.0;
    cgs_units.volume = cgs_units.energy = 1.0;
    const UnitConversions conversions{
        code_units.LengthConversion(cgs_units),
        code_units.TimeConversion(cgs_units),
        eos_units.NumberDensityConversion(cgs_units),
        code_units.EnergyDensityConversion(cgs_units)};
    if (!Kokkos::isfinite(t_to_eos) || t_to_eos <= 0.0 ||
        !Kokkos::isfinite(mu_to_eos) || mu_to_eos <= 0.0 ||
        !Kokkos::isfinite(n_to_eos) || n_to_eos <= 0.0 ||
        !Kokkos::isfinite(conversions.number_cgs_per_eos) ||
        conversions.number_cgs_per_eos <= 0.0 ||
        !Kokkos::isfinite(conversions.energy_cgs_per_code) ||
        conversions.energy_cgs_per_code <= 0.0) return invalid;
    const Real t_min = eos.GetMinimumTemperature()/t_to_eos;
    const Real t_max = eos.GetMaximumTemperature()/t_to_eos;
    const Real y_min = eos.GetMinimumSpeciesFraction(0);
    const Real y_max = eos.GetMaximumSpeciesFraction(0);
    if (!Kokkos::isfinite(t_min) || !Kokkos::isfinite(t_max) ||
        !Kokkos::isfinite(y_min) || !Kokkos::isfinite(y_max) ||
        !(t_min > 0.0 && t_max > t_min && y_max > y_min) ||
        !(nb >= eos.GetMinimumDensity() && nb <= eos.GetMaximumDensity()) ||
        !(temp >= t_min && temp <= t_max && ye >= y_min && ye <= y_max)) return invalid;
    Real j_current = 0.0;
    for (int s = 0; s < 4; ++s) {
      if (!Kokkos::isfinite(current_number[s]) || current_number[s] < 0.0 ||
          !Kokkos::isfinite(current_energy[s]) || current_energy[s] < 0.0) return invalid;
      j_current += current_energy[s];
    }
    Real y_local[MAX_SPECIES] = {ye, 0.0, 0.0};
    const Real e_matter = eos.GetEnergy(nb, temp, y_local);
    // Preserve the original summation order and conservation norm.
    const Real e_rhs = e_matter + current_energy[0] + current_energy[1] +
                       current_energy[2] + current_energy[3];
    Real yl_rhs[MAX_SPECIES] = {
        ye + (current_number[0] - current_number[1])/nb, 0.0, 0.0};
    if (!Kokkos::isfinite(e_matter) || !Kokkos::isfinite(e_rhs) || e_rhs <= 0.0 ||
        !Kokkos::isfinite(yl_rhs[0]) || !Kokkos::isfinite(j_current)) return invalid;
    const Real e_cold = eos.GetEnergy(nb, t_min, y_local);
    const Real eps = std::numeric_limits<Real>::epsilon();
    const Real energy_scale = Kokkos::fmax(
        j_current + Kokkos::fabs(e_matter - e_cold), eps*Kokkos::fabs(e_rhs));
    const Real lepton_scale = Kokkos::fmax(
        Kokkos::fabs(ye) + current_number[0]/nb + current_number[1]/nb,
        eps*(y_max - y_min));
    if (!Kokkos::isfinite(e_cold) || !Kokkos::isfinite(energy_scale) ||
        energy_scale <= 0.0 || !Kokkos::isfinite(lepton_scale) ||
        lepton_scale <= 0.0) return invalid;
    const Real lambda = tau < 0.5*dt ? 1.0 : 2.0 - 2.0*(tau/dt);
    const gra_equilibrium_detail::GraProblem problem{
        nb, temp, ye, t_min, t_max, y_min, y_max,
        t_to_eos, mu_to_eos, n_to_eos, e_rhs, yl_rhs[0], lambda,
        energy_scale, lepton_scale, conversions, local_number, local_energy, validation};
    gra_equilibrium_detail::GraSearch<EOS> search{eos, problem};
    const Real weights[PEQ_NWEIGHTS] = {1.0, 1.0, 1.0, 1.0, 1.0};
    Real t_root = temp, y_root[MAX_SPECIES] = {ye, 0.0, 0.0};
    int status = -1;
    if (yl_rhs[0] != 0.0) {
      // Boolean/status are provenance, not acceptance. Even a failed frozen-Ye
      // solve can return a useful finite table endpoint.
      eos.GetBetaEquilibriumPartial(nb, e_rhs, yl_rhs, weights,
                                   t_root, y_root, temp, y_local, &status);
      search.Consider(t_root, y_root[0]);
      if (search.best.valid && search.best.conserving) {
        return search.Publish(selected_number, selected_energy, status);
      }
    }
    // Exact-solve failure is recoverable: retain budgets, usable seeds, and the
    // best candidate throughout bounded refinement of the closed table domain.
    search.Consider(temp, ye, true);
    search.Edges();
    search.Interior(t_root, y_root[0]);
    search.Interior(temp, ye);
    if (search.best.valid) {
      search.Interior(search.best.result.temp, search.best.result.ye);
    }
    return search.Publish(selected_number, selected_energy, status);
  }
}

}  // namespace weakrates
}  // namespace radiationm1
#endif  // RADIATION_M1_RADIATION_M1_WEAKRATES_EQUILIBRIUM_HPP_
