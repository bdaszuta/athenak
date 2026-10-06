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
//! Accept only NUEQ_INTERIOR and a finite, in-table root satisfying the ACTUAL
//! analytic EOS conservation residual. Do not validate it with approximate
//! FD moments. GetTrappedNeutrinos returns CODE number density; explicitly
//! undo that conversion before dividing by nb. Its three energy entries are
//! PAIRS (electron, muon, tau), already in code energy-density units.
//!
//! Selected FD moments use mu_l at the solver's (T_root,Ye_root). Never replace
//! Ye_root by the GRA postprocessed Ye before reevaluating mu_l. The residual
//! belongs to the analytic root, not to the FD approximation or to a mixture.
//!
//! Return true only after writing valid targets. On false outputs are unchanged;
//! the enabled parent path MUST diagnose failure, preferably Kokkos::abort.
//! Yl==0 is explicitly rejected for a requested full solve because the existing
//! EOS solver norm divides by Yl. This is a guarded failure, NOT an emulation of
//! GRA's complete Newton/fallback ladder. Local selection needs no solver.
//! Read-only input arrays must not overlap mutable output arrays. The function
//! does not modify current/local moments, microscopic coefficients, or raw r,q.
template<class EOS>
KOKKOS_INLINE_FUNCTION
bool BuildGraEquilibriumTargets(
    const EOS &eos, Real nb, Real temp, Real ye, Real dt,
    const Real *raw_abs_1, const Real *raw_scat_1,
    const Real *current_number, const Real *current_energy,
    const Real *local_number, const Real *local_energy,
    Real *selected_number, Real *selected_energy) {
  constexpr int ntransport = 4;
  constexpr Real residual_tolerance = 1.0e-7;  // existing EOS Newton tolerance
  for (int s = 0; s < ntransport; ++s) {
    if (!Kokkos::isfinite(local_number[s]) || local_number[s] < 0.0 ||
        !Kokkos::isfinite(local_energy[s]) || local_energy[s] < 0.0) {
      return false;
    }
  }
  if (!Kokkos::isfinite(dt) || dt < 0.0) return false;
  for (int s = 0; s < 2; ++s) {
    if (!Kokkos::isfinite(raw_abs_1[s]) || raw_abs_1[s] < 0.0 ||
        !Kokkos::isfinite(raw_scat_1[s]) || raw_scat_1[s] < 0.0) {
      return false;
    }
  }
  const Real tau_e = gra_equilibrium_detail::RawEnergyTimescale(
      raw_abs_1[0], raw_scat_1[0]);
  const Real tau_a = gra_equilibrium_detail::RawEnergyTimescale(
      raw_abs_1[1], raw_scat_1[1]);
  const Real tau = (tau_e < tau_a) ? tau_e : tau_a;
  if (dt == 0.0 || !(tau < dt)) {
    for (int s = 0; s < ntransport; ++s) {
      selected_number[s] = local_number[s];
      selected_energy[s] = local_energy[s];
    }
    return true;
  }

  if constexpr (!gra_equilibrium_detail::HasInteriorStatus<EOS>::value) {
    // The local regime remains usable without CompOSE-like weak equilibrium.
    return false;
  } else {
    if (!Kokkos::isfinite(nb) || nb <= 0.0 ||
        !Kokkos::isfinite(temp) || temp <= 0.0 || !Kokkos::isfinite(ye) ||
        eos.GetNSpecies() != 1) {
      return false;
    }
    // Live eos.hpp exposes metadata getters only as non-const. These getters
    // simply return references; copy metadata without modifying the EOS. The
    // const_cast can disappear when const getter overloads are introduced.
    const Primitive::UnitSystem code_units =
        const_cast<EOS &>(eos).GetCodeUnitSystem();
    const Primitive::UnitSystem eos_units =
        const_cast<EOS &>(eos).GetEOSUnitSystem();
    const Real t_to_eos = code_units.TemperatureConversion(eos_units);
    const Real mu_to_eos = code_units.ChemicalPotentialConversion(eos_units);
    const Real n_to_eos = code_units.NumberDensityConversion(eos_units);
    if (!Kokkos::isfinite(t_to_eos) || t_to_eos <= 0.0 ||
        !Kokkos::isfinite(mu_to_eos) || mu_to_eos <= 0.0 ||
        !Kokkos::isfinite(n_to_eos) || n_to_eos <= 0.0) {
      return false;
    }
    const Real t_min = eos.GetMinimumTemperature()/t_to_eos;
    const Real t_max = eos.GetMaximumTemperature()/t_to_eos;
    const Real y_min = eos.GetMinimumSpeciesFraction(0);
    const Real y_max = eos.GetMaximumSpeciesFraction(0);
    if (!(nb >= eos.GetMinimumDensity() && nb <= eos.GetMaximumDensity()) ||
        !(temp >= t_min && temp <= t_max && ye >= y_min && ye <= y_max)) {
      return false;
    }
    for (int s = 0; s < ntransport; ++s) {
      if (!Kokkos::isfinite(current_number[s]) || current_number[s] < 0.0 ||
          !Kokkos::isfinite(current_energy[s]) || current_energy[s] < 0.0) {
        return false;
      }
    }
    Real y_local[MAX_SPECIES] = {ye, 0.0, 0.0};
    const Real e_matter = eos.GetEnergy(nb, temp, y_local);
    const Real e_rhs = e_matter + current_energy[0] + current_energy[1] +
                       current_energy[2] + current_energy[3];
    Real yl_rhs[MAX_SPECIES] = {
        ye + (current_number[0] - current_number[1])/nb, 0.0, 0.0};
    if (!Kokkos::isfinite(e_matter) || !Kokkos::isfinite(e_rhs) || e_rhs <= 0.0 ||
        !Kokkos::isfinite(yl_rhs[0]) || yl_rhs[0] == 0.0) {
      return false;
    }
    const Real weights[PEQ_NWEIGHTS] = {1.0, 1.0, 1.0, 1.0, 1.0};
    Real t_root = temp;
    Real y_root[MAX_SPECIES] = {ye, 0.0, 0.0};
    int status = -1;
    if (!eos.GetBetaEquilibriumPartial(nb, e_rhs, yl_rhs, weights,
                                        t_root, y_root, temp, y_local, &status) ||
        status != EOS::NUEQ_INTERIOR || !Kokkos::isfinite(t_root) ||
        !Kokkos::isfinite(y_root[0]) || t_root <= 0.0 ||
        !(t_root >= t_min && t_root <= t_max &&
          y_root[0] >= y_min && y_root[0] <= y_max)) {
      return false;
    }
    Real net_number_code[3] = {};
    Real pair_energy_code[3] = {};
    eos.GetTrappedNeutrinos(nb, t_root, y_root,
                             net_number_code, pair_energy_code);
    for (int g = 0; g < 3; ++g) {
      if (!Kokkos::isfinite(net_number_code[g]) ||
          !Kokkos::isfinite(pair_energy_code[g]) || pair_energy_code[g] < 0.0) {
        return false;
      }
    }
    const Real net_number_eos = net_number_code[0]*n_to_eos;
    const Real e_root = eos.GetEnergy(nb, t_root, y_root);
    const Real e_total = e_root + pair_energy_code[0] + pair_energy_code[1] +
                         pair_energy_code[2];
    const Real r_lepton = (y_root[0] + net_number_eos/nb - yl_rhs[0])/yl_rhs[0];
    const Real r_energy = e_total/e_rhs - 1.0;
    if (!Kokkos::isfinite(e_root) || !Kokkos::isfinite(net_number_eos) ||
        !Kokkos::isfinite(r_lepton) || !Kokkos::isfinite(r_energy) ||
        Kokkos::fabs(r_lepton) + Kokkos::fabs(r_energy) > residual_tolerance) {
      return false;
    }

    const Real t_mev = t_root*t_to_eos;
    const Real mu_mev =
        eos.GetElectronLeptonChemicalPotential(nb, t_root, y_root)*mu_to_eos;
    const Real eta_nue = mu_mev/t_mev;
    if (!Kokkos::isfinite(t_mev) || t_mev <= 0.0 ||
        !Kokkos::isfinite(mu_mev) || !Kokkos::isfinite(eta_nue)) {
      return false;
    }
    // Construct just the CGS scale metadata used by conversions, on device.
    // Do not call host-only MakeCGS() or read a host static UnitSystem object.
    Primitive::UnitSystem cgs_units{};
    cgs_units.length = cgs_units.time = cgs_units.numberDensity = 1.0;
    cgs_units.volume = cgs_units.energy = 1.0;
    const UnitConversions conversions{
        code_units.LengthConversion(cgs_units),
        code_units.TimeConversion(cgs_units),
        eos_units.NumberDensityConversion(cgs_units),
        code_units.EnergyDensityConversion(cgs_units)};
    if (!Kokkos::isfinite(conversions.number_cgs_per_eos) ||
        conversions.number_cgs_per_eos <= 0.0 ||
        !Kokkos::isfinite(conversions.energy_cgs_per_code) ||
        conversions.energy_cgs_per_code <= 0.0) {
      return false;
    }
    EquilibriumDensities full = ComputeEquilibriumDensities(t_mev, eta_nue);
    ConvertToCodeUnits(conversions, full);
    if (!IsFinite(full)) return false;
    const Real lambda = (tau < 0.5*dt) ? 1.0 : 2.0 - 2.0*(tau/dt);
    Real n_candidate[ntransport] = {};
    Real j_candidate[ntransport] = {};
    for (int s = 0; s < ntransport; ++s) {
      const int rate_species = (s < 2) ? s : kNuX;
      const Real split = (s < 2) ? 1.0 : 0.5;
      const Real n_full = split*full.number[rate_species];
      const Real j_full = split*full.energy[rate_species];
      n_candidate[s] = lambda*n_full + (1.0 - lambda)*local_number[s];
      j_candidate[s] = lambda*j_full + (1.0 - lambda)*local_energy[s];
      if (!Kokkos::isfinite(n_candidate[s]) || n_candidate[s] < 0.0 ||
          !Kokkos::isfinite(j_candidate[s]) || j_candidate[s] < 0.0) {
        return false;
      }
    }
    for (int s = 0; s < ntransport; ++s) {
      selected_number[s] = n_candidate[s];
      selected_energy[s] = j_candidate[s];
    }
    return true;
  }
}

}  // namespace weakrates
}  // namespace radiationm1
#endif  // RADIATION_M1_RADIATION_M1_WEAKRATES_EQUILIBRIUM_HPP_
