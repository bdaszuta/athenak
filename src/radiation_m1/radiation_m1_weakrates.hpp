#ifndef RADIATION_M1_RADIATION_M1_WEAKRATES_HPP_
#define RADIATION_M1_RADIATION_M1_WEAKRATES_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_weakrates.hpp
//! \brief Device-callable GR-Athena++ weakrates point physics for grey M1.

#include <type_traits>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "eos/primitive-solver/fermi_dirac_integrals.hpp"
#include "radiation_m1_weakrates_constants.hpp"
#include "radiation_m1_weakrates_fermi.hpp"

namespace radiationm1 {

//! Runtime controls specific to the weakrates opacity backend.
struct WeakRatesParams {
  Real rho_min_cgs;
  Real temp_min_mev;
  Real opacity_corr_fac_max;
  Real peq_w_floor;
  Real peq_dlnT_tol;
  Real peq_dYe_tol;
  bool use_partial_equilibrium;
};
static_assert(std::is_trivially_copyable<WeakRatesParams>::value,
              "WeakRatesParams must be device-copyable");

namespace weakrates {

enum SpeciesIndex { kNuE = 0, kAntiNuE = 1, kNuX = 2, kNumRateSpecies = 3 };

//! Local thermodynamic state in the CGS/MeV convention of the rate formulas.
struct State {
  Real rho_cgs;
  Real temp_mev;
  Real ye;
  Real baryon_mass_cgs;
  Real eta_nue;
  Real eta_anue;
  Real eta_nux;
  Real eta_e;
  Real eta_np;
  Real eta_pn;
};
static_assert(std::is_trivially_copyable<State>::value, "State must be device-copyable");

//! Raw coefficients: emissivities are all-heavy in slot 2; opacities are intensive.
struct RawCoefficients {
  Real eta_0[kNumRateSpecies];         // cm^-3 s^-1
  Real eta_1[kNumRateSpecies];         // erg cm^-3 s^-1
  Real eta_0_non_cc[kNumRateSpecies];  // non-charged-current part, cm^-3 s^-1
  Real eta_1_non_cc[kNumRateSpecies];  // non-charged-current part, erg cm^-3 s^-1
  Real abs_0[kNumRateSpecies];         // cm^-1
  Real abs_1[kNumRateSpecies];         // cm^-1
  Real scat_1[kNumRateSpecies];        // cm^-1
};
static_assert(std::is_trivially_copyable<RawCoefficients>::value,
              "RawCoefficients must be device-copyable");

//! Equilibrium densities: slot 2 is the sum over all four heavy-lepton species.
struct EquilibriumDensities {
  Real number[kNumRateSpecies];  // cm^-3, then EOS number-density units after conversion
  Real energy[kNumRateSpecies];  // erg cm^-3, then code energy-density units after conversion
};
static_assert(std::is_trivially_copyable<EquilibriumDensities>::value,
              "EquilibriumDensities must be device-copyable");

//! Conversion factors from formula units to the M1 coefficient units.
struct UnitConversions {
  Real length_cgs_per_code;
  Real time_cgs_per_code;
  Real number_cgs_per_eos;
  Real energy_cgs_per_code;
};
static_assert(std::is_trivially_copyable<UnitConversions>::value,
              "UnitConversions must be device-copyable");

KOKKOS_INLINE_FUNCTION
bool IsFinite(const State &state) {
  return Kokkos::isfinite(state.rho_cgs) && Kokkos::isfinite(state.temp_mev) &&
         Kokkos::isfinite(state.ye) && Kokkos::isfinite(state.baryon_mass_cgs) &&
         Kokkos::isfinite(state.eta_nue) && Kokkos::isfinite(state.eta_anue) &&
         Kokkos::isfinite(state.eta_nux) && Kokkos::isfinite(state.eta_e) &&
         Kokkos::isfinite(state.eta_np) && Kokkos::isfinite(state.eta_pn);
}

KOKKOS_INLINE_FUNCTION
bool IsFinite(const RawCoefficients &coeffs) {
  bool finite = true;
  for (int s = 0; s < kNumRateSpecies; ++s) {
    finite = finite && Kokkos::isfinite(coeffs.eta_0[s]) &&
             Kokkos::isfinite(coeffs.eta_1[s]) &&
             Kokkos::isfinite(coeffs.eta_0_non_cc[s]) &&
             Kokkos::isfinite(coeffs.eta_1_non_cc[s]) &&
             Kokkos::isfinite(coeffs.abs_0[s]) &&
             Kokkos::isfinite(coeffs.abs_1[s]) && Kokkos::isfinite(coeffs.scat_1[s]);
  }
  return finite;
}

KOKKOS_INLINE_FUNCTION
bool IsFinite(const EquilibriumDensities &densities) {
  bool finite = true;
  for (int s = 0; s < kNumRateSpecies; ++s) {
    finite = finite && Kokkos::isfinite(densities.number[s]) &&
             Kokkos::isfinite(densities.energy[s]);
  }
  return finite;
}

//! Construct degeneracy parameters and nucleon blocking densities.
KOKKOS_INLINE_FUNCTION
State MakeState(const Real rho_cgs, const Real temp_mev, const Real ye,
                const Real baryon_mass_cgs, const Real mu_n, const Real mu_p,
                const Real mu_e) {
  State state{};
  state.rho_cgs = rho_cgs;
  state.temp_mev = temp_mev;
  state.ye = ye;
  state.baryon_mass_cgs = baryon_mass_cgs;
  state.eta_nue = (mu_p + mu_e - mu_n)/temp_mev;
  state.eta_anue = -state.eta_nue;
  state.eta_nux = 0.0;
  state.eta_e = mu_e/temp_mev;

  const Real nb_cgs = rho_cgs/baryon_mass_cgs;
  const Real xn = 1.0 - ye;
  const Real xp = ye;
  const Real eta_n = mu_n/temp_mev;
  const Real eta_p = mu_p/temp_mev;
  const Real eta_hat = eta_n - eta_p - q_np_mev/temp_mev;
  state.eta_np = nb_cgs*(xp - xn)/Kokkos::expm1(-eta_hat);
  state.eta_pn = nb_cgs*(xn - xp)/Kokkos::expm1(eta_hat);

  // Bruenn's non-degenerate limit avoids cancellation at low density.
  if (rho_cgs < 2.0e11) {
    state.eta_pn = nb_cgs*xp;
    state.eta_np = nb_cgs*xn;
  }
  if (state.eta_pn < 0.0) state.eta_pn = 0.0;
  if (state.eta_np < 0.0) state.eta_np = 0.0;
  return state;
}

//! Evaluate total and non-charged-current emission, absorption, and scattering in CGS.
KOKKOS_INLINE_FUNCTION
RawCoefficients ComputeRawCoefficients(const State &state) {
  RawCoefficients coeffs{};
  const Real rho = state.rho_cgs;
  const Real temp = state.temp_mev;
  const Real eta_nue = state.eta_nue;
  const Real eta_anue = state.eta_anue;
  const Real eta_nux = state.eta_nux;
  const Real eta_e = state.eta_e;
  const Real xn = 1.0 - state.ye;
  const Real xp = state.ye;

  const Real hc_3 = Cube(hc_mevcm);
  const Real temp_4 = Fourth(temp);
  const Real enr_p = 8.0*pi/hc_3*temp_4*fermi::F3(eta_e);
  const Real enr_m = 8.0*pi/hc_3*temp_4*fermi::F3(-eta_e);

  // Beta-process emission (Bruenn 1985).
  const Real me_mev = me_erg/mev_to_erg;
  const Real beta = pi*clight*(1.0 + 3.0*Square(alpha))*sigma_0/
                    (hc_3*Square(me_mev));
  Real block_e = 1.0 + Kokkos::exp(eta_nue - fermi::F5OverF4(eta_e));
  Real block_a = 1.0 + Kokkos::exp(eta_anue - fermi::F5OverF4(-eta_e));
  const Real temp_5 = Fourth(temp)*temp;
  const Real r_beta_nue = beta*state.eta_pn*temp_5*fermi::F4(eta_e)/block_e;
  const Real q_beta_nue = r_beta_nue*temp*fermi::F5OverF4(eta_e);
  const Real r_beta_anue = beta*state.eta_np*temp_5*fermi::F4(-eta_e)/block_a;
  const Real q_beta_anue = r_beta_anue*temp*fermi::F5OverF4(-eta_e);

  // Electron-positron pair annihilation (Ruffert et al. 1998).
  const Real pair_mean = 0.5*(fermi::F4OverF3(eta_e) + fermi::F4OverF3(-eta_e));
  block_e = 1.0 + Kokkos::exp(eta_nue - pair_mean);
  block_a = 1.0 + Kokkos::exp(eta_anue - pair_mean);
  Real block_x = 1.0 + Kokkos::exp(eta_nux - pair_mean);
  const Real pair_const = sigma_0*clight/Square(me_mev)*enr_m*enr_p;
  Real r_pair = pair_const/(36.0*block_e*block_a)*
                (Square(cv - ca) + Square(cv + ca));
  const Real r_pair_nue = r_pair;
  const Real r_pair_anue = r_pair;
  const Real q_pair_factor = temp*pair_mean;
  const Real q_pair_nue = r_pair*q_pair_factor;
  const Real q_pair_anue = r_pair*q_pair_factor;
  r_pair = pair_const/(9.0*Square(block_x))*
           (Square(cv - ca) + Square(cv + ca - 2.0));
  const Real r_pair_nux = r_pair;
  const Real q_pair_nux = r_pair*q_pair_factor;

  // Plasmon decay.
  const Real gamma = gamma_0*Kokkos::sqrt((Square(pi) + 3.0*Square(eta_e))/3.0);
  const Real plasmon_mean = 1.0 + 0.5*Square(gamma)/(1.0 + gamma);
  block_e = 1.0 + Kokkos::exp(eta_nue - plasmon_mean);
  block_a = 1.0 + Kokkos::exp(eta_anue - plasmon_mean);
  block_x = 1.0 + Kokkos::exp(eta_nux - plasmon_mean);
  const Real gamma_const = Cube(pi)*sigma_0*clight*Square(Fourth(temp))/
                           (Square(me_mev)*3.0*fine_structure*Square(hc_3))*
                           Square(Cube(gamma))*Kokkos::exp(-gamma)*(1.0 + gamma);
  Real r_gamma = Square(cv)*gamma_const/(block_e*block_a);
  const Real q_gamma_factor = 0.5*temp*(2.0 + Square(gamma)/(1.0 + gamma));
  const Real r_plasmon_nue = r_gamma;
  const Real q_plasmon_nue = r_gamma*q_gamma_factor;
  const Real r_plasmon_anue = r_gamma;
  const Real q_plasmon_anue = r_gamma*q_gamma_factor;
  r_gamma = Square(cv - 1.0)*4.0*gamma_const/Square(block_x);
  const Real r_plasmon_nux = r_gamma;
  const Real q_plasmon_nux = r_gamma*q_gamma_factor;

  // Nucleon-nucleon bremsstrahlung (Burrows et al. 2006).
  const Real q_brem = 0.5*2.0778e2*0.5/mev_to_erg*
                      (Square(xn) + Square(xp) + 28.0/3.0*xn*xp)*Square(rho)*
                      Kokkos::pow(temp, 5.5);
  const Real r_brem = 2.0*q_brem/(4.364*temp);

  coeffs.eta_0_non_cc[kNuE] = r_pair_nue + r_plasmon_nue + r_brem;
  coeffs.eta_0_non_cc[kAntiNuE] = r_pair_anue + r_plasmon_anue + r_brem;
  coeffs.eta_0_non_cc[kNuX] = r_pair_nux + r_plasmon_nux + 4.0*r_brem;
  coeffs.eta_1_non_cc[kNuE] =
      (q_pair_nue + q_plasmon_nue + q_brem)*mev_to_erg;
  coeffs.eta_1_non_cc[kAntiNuE] =
      (q_pair_anue + q_plasmon_anue + q_brem)*mev_to_erg;
  coeffs.eta_1_non_cc[kNuX] =
      (q_pair_nux + q_plasmon_nux + 4.0*q_brem)*mev_to_erg;
  coeffs.eta_0[kNuE] = r_beta_nue + r_pair_nue + r_plasmon_nue + r_brem;
  coeffs.eta_0[kAntiNuE] = r_beta_anue + r_pair_anue + r_plasmon_anue + r_brem;
  coeffs.eta_0[kNuX] = r_pair_nux + r_plasmon_nux + 4.0*r_brem;
  coeffs.eta_1[kNuE] =
      (q_beta_nue + q_pair_nue + q_plasmon_nue + q_brem)*mev_to_erg;
  coeffs.eta_1[kAntiNuE] =
      (q_beta_anue + q_pair_anue + q_plasmon_anue + q_brem)*mev_to_erg;
  coeffs.eta_1[kNuX] = (q_pair_nux + q_plasmon_nux + 4.0*q_brem)*mev_to_erg;

  // Charged-current absorption on free nucleons (Bruenn 1985).
  const Real abs_zeta = (1.0 + 3.0*Square(alpha))*0.25*sigma_0/Square(me_erg);
  Real block = 1.0 + Kokkos::exp(eta_e - fermi::F5OverF4(eta_nue));
  const Real zeta_nue = state.eta_np*abs_zeta/block;
  block = 1.0 + Kokkos::exp(-eta_e - fermi::F5OverF4(eta_anue));
  const Real zeta_anue = state.eta_pn*abs_zeta/block;
  const Real temp_erg_sq = Square(temp*mev_to_erg);
  coeffs.abs_0[kNuE] = zeta_nue*temp_erg_sq*fermi::F4OverF2(eta_nue);
  coeffs.abs_0[kAntiNuE] = zeta_anue*temp_erg_sq*fermi::F4OverF2(eta_anue);
  coeffs.abs_1[kNuE] = zeta_nue*temp_erg_sq*fermi::F5OverF3(eta_nue);
  coeffs.abs_1[kAntiNuE] = zeta_anue*temp_erg_sq*fermi::F5OverF3(eta_anue);

  // Isoenergetic scattering on free nucleons. The opacity is per neutrino.
  const Real nb_cgs = rho/state.baryon_mass_cgs;
  const Real scatter_n = nb_cgs*(1.0 + 5.0*Square(alpha))/24.0*
                         sigma_0/Square(me_erg);
  const Real scatter_p = nb_cgs*(4.0*Square(cv - 1.0) + 5.0*Square(alpha))/24.0*
                         sigma_0/Square(me_erg);
  const Real scatter_zeta = xn*scatter_n + xp*scatter_p;
  coeffs.scat_1[kNuE] = scatter_zeta*temp_erg_sq*fermi::F5OverF3(eta_nue);
  coeffs.scat_1[kAntiNuE] = scatter_zeta*temp_erg_sq*fermi::F5OverF3(eta_anue);
  coeffs.scat_1[kNuX] = scatter_zeta*temp_erg_sq*fermi::F5OverF3(eta_nux);

  return coeffs;
}

//! Compute local Fermi-Dirac equilibrium densities from temperature and degeneracy.
KOKKOS_INLINE_FUNCTION
EquilibriumDensities ComputeEquilibriumDensities(const Real temp_mev,
                                                 const Real eta_nue) {
  EquilibriumDensities densities{};
  const Real eta_anue = -eta_nue;
  const Real t2 = temp_mev*temp_mev;
  const Real t3 = t2*temp_mev;
  const Real t4 = t3*temp_mev;
  const Real prefactor = 4.0*pi/Cube(hc_mevcm);

  densities.number[kNuE] =
      prefactor*t3*Primitive::FermiDirac::F2(eta_nue);
  densities.number[kAntiNuE] =
      prefactor*t3*Primitive::FermiDirac::F2(eta_anue);
  densities.number[kNuX] =
      4.0*prefactor*t3*Primitive::FermiDirac::F2(0.0);
  densities.energy[kNuE] =
      prefactor*t4*Primitive::FermiDirac::F3(eta_nue)*mev_to_erg;
  densities.energy[kAntiNuE] =
      prefactor*t4*Primitive::FermiDirac::F3(eta_anue)*mev_to_erg;
  densities.energy[kNuX] =
      4.0*prefactor*t4*Primitive::FermiDirac::F3(0.0)*mev_to_erg;
  return densities;
}

//! Compute local Fermi-Dirac equilibrium densities for a weakrates state.
KOKKOS_INLINE_FUNCTION
EquilibriumDensities ComputeEquilibriumDensities(const State &state) {
  return ComputeEquilibriumDensities(state.temp_mev, state.eta_nue);
}

KOKKOS_INLINE_FUNCTION
void ConvertToCodeUnits(const UnitConversions &units, RawCoefficients &coeffs) {
  const Real number_rate = units.number_cgs_per_eos/units.time_cgs_per_code;
  const Real energy_rate = units.energy_cgs_per_code/units.time_cgs_per_code;
  for (int s = 0; s < kNumRateSpecies; ++s) {
    coeffs.eta_0[s] /= number_rate;
    coeffs.eta_1[s] /= energy_rate;
    coeffs.eta_0_non_cc[s] /= number_rate;
    coeffs.eta_1_non_cc[s] /= energy_rate;
    coeffs.abs_0[s] *= units.length_cgs_per_code;
    coeffs.abs_1[s] *= units.length_cgs_per_code;
    coeffs.scat_1[s] *= units.length_cgs_per_code;
  }
}

KOKKOS_INLINE_FUNCTION
void ConvertToCodeUnits(const UnitConversions &units, EquilibriumDensities &densities) {
  for (int s = 0; s < kNumRateSpecies; ++s) {
    densities.number[s] /= units.number_cgs_per_eos;
    densities.energy[s] /= units.energy_cgs_per_code;
  }
}

}  // namespace weakrates
}  // namespace radiationm1

#endif  // RADIATION_M1_RADIATION_M1_WEAKRATES_HPP_
