//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_calc_opacities_weakrates.cpp
//! \brief Calculate grey M1 coefficients with the GR-Athena++ weakrates formulas.

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "globals.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"
#include "radiation_m1/radiation_m1_weakrates.hpp"
#include "radiation_m1/radiation_m1_weakrates_equilibrium.hpp"

namespace radiationm1 {

TaskStatus RadiationM1::CalcOpacityWeakRates(Driver *pdrive, int stage) {
  // Coefficients are held fixed for the complete mesh timestep.
  if (stage > 1) return TaskStatus::complete;

  auto *ptest_nqt =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                     Primitive::ResetFloor> *>(pmy_pack->pdyngr);
  if (ptest_nqt != nullptr) {
    return CalcOpacityWeakRates_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                 Primitive::ResetFloor>(pdrive, stage);
  }

  auto *ptest_nlog = dynamic_cast<dyngr::DynGRMHDPS<
      Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor> *>(
      pmy_pack->pdyngr);
  if (ptest_nlog != nullptr) {
    return CalcOpacityWeakRates_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                                 Primitive::ResetFloor>(pdrive, stage);
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl;
  std::cout << "Weakrates requires EOSCompOSE with ResetFloor.\n";
  std::abort();
}

//----------------------------------------------------------------------------------------
//! \brief Publish five M1 coefficients and the option-enabled thermal fields.
template <class EOSPolicy, class ErrorPolicy>
TaskStatus RadiationM1::CalcOpacityWeakRates_(Driver *pdrive, int stage) {
  assert(nspecies == M1_TOTAL_NUM_SPECIES);

  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  const int is_op = is - 1, ie_op = ie + 1;
  const int js_op = pmy_pack->pmesh->multi_d ? js - 1 : js;
  const int je_op = pmy_pack->pmesh->multi_d ? je + 1 : je;
  const int ks_op = pmy_pack->pmesh->three_d ? ks - 1 : ks;
  const int ke_op = pmy_pack->pmesh->three_d ? ke + 1 : ke;

  const int nmb1 = pmy_pack->nmb_thispack - 1;
  const int nspecies_ = nspecies;
  const int nvars_ = nvars;

  auto &adm = pmy_pack->padm->adm;
  auto &radiation_mask_ = radiation_mask;
  auto &m1_params_ = params;
  const WeakRatesParams weakrates_params_ = weakrates_params;
  const bool thermal_enabled = params.opacity_type == WeakRates &&
                               weakrates_params_.use_electron_thermal_emission;

  auto &eta_0_ = eta_0;
  auto &abs_0_ = abs_0;
  auto &eta_1_ = eta_1;
  auto &abs_1_ = abs_1;
  auto &scat_1_ = scat_1;
  auto weakrates_thermal_ = weakrates_thermal;
  auto &u0_ = u0;
  auto &chi_ = chi;

  DvceArray5D<Real> w0_ = w0;
  if (ismhd) w0_ = pmy_pack->pmhd->w0;

  const Real dt_ = pmy_pack->pmesh->dt;
  auto *pdyngr = static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(
      pmy_pack->pdyngr);
  Primitive::EOS<EOSPolicy, ErrorPolicy> &eos = pdyngr->eos.ps.GetEOSMutable();
  auto temperature_ = pdyngr->temperature;
  // Fixed evolution skips full C2P, so only evolving MHD may use this array.
  const bool use_c2p_temperature_ = !pdyngr->IsFixedEvolution();
  const Real mb = eos.GetBaryonMass();

  const Primitive::UnitSystem code_units = eos.GetCodeUnitSystem();
  const Primitive::UnitSystem eos_units = eos.GetEOSUnitSystem();
  const Primitive::UnitSystem cgs_units = Primitive::MakeCGS();
  const Real rho_cgs_per_code = code_units.MassDensityConversion(cgs_units);
  weakrates::UnitConversions conversions{
      code_units.LengthConversion(cgs_units), code_units.TimeConversion(cgs_units),
      eos_units.NumberDensityConversion(cgs_units),
      code_units.EnergyDensityConversion(cgs_units)};
  const Real baryon_mass_cgs =
      mb*rho_cgs_per_code/conversions.number_cgs_per_eos;

  const Real nb_min = eos.GetMinimumDensity();
  const Real nb_max = eos.GetMaximumDensity();
  const Real temp_code_min = eos.GetMinimumTemperature()*
                             eos_units.TemperatureConversion(code_units);
  const Real temp_code_max = eos.GetMaximumTemperature()*
                             eos_units.TemperatureConversion(code_units);
  const Real ye_min = eos.GetMinimumSpeciesFraction(0);
  const Real ye_max = eos.GetMaximumSpeciesFraction(0);
  const Real eos_rho_min_cgs = nb_min*conversions.number_cgs_per_eos*baryon_mass_cgs;
  const Real rho_floor_cgs =
      std::max(weakrates_params_.rho_min_cgs, eos_rho_min_cgs);
  const Real temp_floor_mev =
      std::max(weakrates_params_.temp_min_mev, temp_code_min);

  constexpr Real peq_cv_eps = 1.0e-2;
  constexpr Real peq_trust_c = 10.0;
  constexpr int peq_max_halvings = 4;

  // Small per-pack diagnostics, allocated only for GRA selection. Count physical
  // interior cells once at stage 1; ghost evaluations do not inflate statistics.
  // These Views outlive the kernel and host readback; there is no mesh-sized log.
  DvceArray1D<unsigned long long> equilibrium_counts;
  DvceArray1D<Real> equilibrium_diagnostics;
  if (weakrates_params_.gra_equilibrium_targets) {
    equilibrium_counts = DvceArray1D<unsigned long long>("gra_equilibrium_counts", 3);
    equilibrium_diagnostics = DvceArray1D<Real>("gra_equilibrium_diagnostics", 14);
    Kokkos::deep_copy(equilibrium_counts, 0ULL);
    Kokkos::deep_copy(equilibrium_diagnostics, 0.0);
  }

  par_for(
      "radiation_m1_calc_opacity_weakrates", DevExeSpace(), 0, nmb1, ks_op, ke_op,
      js_op, je_op, is_op, ie_op,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        // Clear only evaluated cells before any early return, so mask/cutoff or
        // original microscopic invalidity cannot retain older thermal data.
        // The off-path View is empty and must never be indexed, even to zero it.
        if (thermal_enabled) {
          for (int f = 0; f < weakrates::kNumThermalFields; ++f) {
            weakrates_thermal_(m, f, k, j, i) = 0.0;
          }
        }
        if (radiation_mask_(m, k, j, i)) {
          for (int s = 0; s < nspecies_; ++s) {
            eta_0_(m, s, k, j, i) = 0.0;
            abs_0_(m, s, k, j, i) = 0.0;
            eta_1_(m, s, k, j, i) = 0.0;
            abs_1_(m, s, k, j, i) = 0.0;
            scat_1_(m, s, k, j, i) = 0.0;
          }
          return;
        }

        Real garr_dd[16];
        Real garr_uu[16];
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_dd{};
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_d{};
        pack_n_d(adm.alpha(m, k, j, i), n_d);
        adm::SpacetimeMetric(
            adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i),
            adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), garr_dd);
        adm::SpacetimeUpperMetric(
            adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i),
            adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), garr_uu);
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            g_dd(a, b) = garr_dd[a + 4*b];
            g_uu(a, b) = garr_uu[a + 4*b];
          }
        }

        const Real gam = adm::SpatialDet(
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
        const Real volform = Kokkos::sqrt(gam);

        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> proj_ud{};
        const Real w_lorentz =
            get_w_lorentz(w0_(m, IVX, k, j, i), w0_(m, IVY, k, j, i),
                          w0_(m, IVZ, k, j, i), g_dd);
        pack_u_u(
            w_lorentz/adm.alpha(m, k, j, i),
            w0_(m, IVX, k, j, i) -
                w_lorentz*adm.beta_u(m, 0, k, j, i)/adm.alpha(m, k, j, i),
            w0_(m, IVY, k, j, i) -
                w_lorentz*adm.beta_u(m, 1, k, j, i)/adm.alpha(m, k, j, i),
            w0_(m, IVZ, k, j, i) -
                w_lorentz*adm.beta_u(m, 2, k, j, i)/adm.alpha(m, k, j, i), u_u);
        pack_v_u(u_u(0), u_u(1), u_u(2), u_u(3), adm.alpha(m, k, j, i),
                 adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                 adm.beta_u(m, 2, k, j, i), v_u);
        tensor_contract(g_dd, u_u, u_d);
        tensor_contract(g_dd, v_u, v_d);
        calc_proj(u_d, u_u, proj_ud);

        Real J[M1_TOTAL_NUM_SPECIES]{};
        Real rnnu[M1_TOTAL_NUM_SPECIES]{};
        // Spectral inputs use the stored lab number and W*(E-F_i*v^i), not J/n
        // from the capped number conversion. The shared density weight cancels
        // in their ratio; spectral_energy is a numerator, not the comoving J.
        Real spectral_number[M1_TOTAL_NUM_SPECIES]{};
        Real spectral_energy[M1_TOTAL_NUM_SPECIES]{};
        bool point_finite = Kokkos::isfinite(volform) && volform > 0.0 &&
                            Kokkos::isfinite(w_lorentz) && w_lorentz > 0.0;
        for (int s = 0; s < nspecies_; ++s) {
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
          pack_F_d(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                   adm.beta_u(m, 2, k, j, i),
                   u0_(m, CombinedIdx(s, M1_FX_IDX, nvars_), k, j, i),
                   u0_(m, CombinedIdx(s, M1_FY_IDX, nvars_), k, j, i),
                   u0_(m, CombinedIdx(s, M1_FZ_IDX, nvars_), k, j, i), F_d);
          const Real E = u0_(m, CombinedIdx(s, M1_E_IDX, nvars_), k, j, i);
          const Real N = u0_(m, CombinedIdx(s, M1_N_IDX, nvars_), k, j, i);
          Real Fv = 0.0;
          for (int a = 1; a < 4; ++a) Fv += F_d(a)*v_u(a);
          spectral_number[s] = N;
          spectral_energy[s] = w_lorentz*(E - Fv);
          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
          apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d,
                        chi_(m, s, k, j, i), P_dd, m1_params_);
          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
          assemble_rT(n_d, E, F_d, P_dd, T_dd);

          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d{};
          J[s] = calc_J_from_rT(T_dd, u_u);
          calc_H_from_rT(T_dd, u_u, proj_ud, H_d);
          const Real gamma = radiation_number_gamma(
              E, J[s], F_d, v_u, w_lorentz, m1_params_);
          enforce_comoving_moment_safety(g_dd, g_uu, u_u, J[s], H_d, m1_params_);
          rnnu[s] = N/gamma;
          point_finite = point_finite && Kokkos::isfinite(J[s]) &&
                         Kokkos::isfinite(rnnu[s]);
        }

        Real nudens_0[M1_TOTAL_NUM_SPECIES]{};
        Real nudens_1[M1_TOTAL_NUM_SPECIES]{};
        for (int s = 0; s < nspecies_; ++s) {
          nudens_0[s] = rnnu[s]/volform;
          nudens_1[s] = J[s]/volform;
          point_finite = point_finite && Kokkos::isfinite(nudens_0[s]) &&
                         Kokkos::isfinite(nudens_1[s]);
        }

        const Real rho_code = w0_(m, IDN, k, j, i);
        const Real p = w0_(m, IPR, k, j, i);
        const Real rho_cgs = rho_code*rho_cgs_per_code;
        Real nb = rho_code/mb;
        nb = (nb > nb_min) ? nb : nb_min;
        nb = (nb < nb_max) ? nb : nb_max;
        const Real ye_input = w0_(m, IYF, k, j, i);
        Real ye = ye_input;
        ye = (ye > ye_min) ? ye : ye_min;
        ye = (ye < ye_max) ? ye : ye_max;
        Real ye_arr[3] = {ye, 0.0, 0.0};
        Real temp = use_c2p_temperature_
                        ? temperature_(m, 0, k, j, i)
                        : eos.GetTemperatureFromP(nb, p, ye_arr);
        const Real temp_unclamped = temp;
        temp = (temp > temp_code_min) ? temp : temp_code_min;
        temp = (temp < temp_code_max) ? temp : temp_code_max;

        point_finite = point_finite && Kokkos::isfinite(rho_cgs) &&
                       Kokkos::isfinite(p) && Kokkos::isfinite(nb) &&
                       Kokkos::isfinite(ye_input) &&
                       Kokkos::isfinite(ye) && Kokkos::isfinite(temp_unclamped) &&
                       Kokkos::isfinite(temp) && Kokkos::isfinite(baryon_mass_cgs) &&
                       baryon_mass_cgs > 0.0;
        if (!point_finite || rho_cgs < rho_floor_cgs ||
            temp_unclamped < temp_floor_mev) {
          for (int s = 0; s < nspecies_; ++s) {
            eta_0_(m, s, k, j, i) = 0.0;
            abs_0_(m, s, k, j, i) = 0.0;
            eta_1_(m, s, k, j, i) = 0.0;
            abs_1_(m, s, k, j, i) = 0.0;
            scat_1_(m, s, k, j, i) = 0.0;
          }
          return;
        }

        const Real rho_eval_cgs =
            nb*conversions.number_cgs_per_eos*baryon_mass_cgs;
        const Real xp = eos.GetProtonFraction(nb, temp, ye_arr);
        const Real xn = eos.GetNeutronFraction(nb, temp, ye_arr);
        const Real xh = eos.GetHeavyFraction(nb, temp, ye_arr);
        const Real abar = eos.GetMassNumber(nb, temp, ye_arr);
        const Real zbar = eos.GetChargeNumber(nb, temp, ye_arr);
        const Real mu_b = eos.GetBaryonChemicalPotential(nb, temp, ye_arr);
        const Real mu_q = eos.GetChargeChemicalPotential(nb, temp, ye_arr);
        const Real mu_le = eos.GetElectronLeptonChemicalPotential(nb, temp, ye_arr);
        const weakrates::State local_state = weakrates::MakeState(
            rho_eval_cgs, temp, baryon_mass_cgs, xn, xp, xh, abar, zbar, mu_b,
            mu_b + mu_q, mu_le - mu_q);
        weakrates::RawCoefficients raw =
            weakrates::ComputeRawCoefficients(local_state);
        weakrates::EquilibriumDensities local_eq =
            weakrates::ComputeEquilibriumDensities(local_state);
        weakrates::ConvertToCodeUnits(conversions, raw);
        weakrates::ConvertToCodeUnits(conversions, local_eq);

        point_finite = weakrates::IsFinite(local_state) && weakrates::IsFinite(raw) &&
                       weakrates::IsFinite(local_eq);
        Real eta_0_loc[M1_TOTAL_NUM_SPECIES]{};
        Real eta_1_loc[M1_TOTAL_NUM_SPECIES]{};
        Real abs_0_loc[M1_TOTAL_NUM_SPECIES]{};
        Real abs_1_loc[M1_TOTAL_NUM_SPECIES]{};
        Real scat_1_loc[M1_TOTAL_NUM_SPECIES]{};
        Real neq_0_loc[M1_TOTAL_NUM_SPECIES]{};
        Real neq_1_loc[M1_TOTAL_NUM_SPECIES]{};

        // Electron CC emissivity is reconstructed only in transport slots 0 and 1
        // from corrected CC absorption. Non-CC emission is published separately.
        // Kept commented out as reference.
        // eta_0_loc[0] = raw.eta_0[weakrates::kNuE];
        // eta_0_loc[1] = raw.eta_0[weakrates::kAntiNuE];
        // eta_1_loc[0] = raw.eta_1[weakrates::kNuE];
        // eta_1_loc[1] = raw.eta_1[weakrates::kAntiNuE];
        abs_0_loc[0] = raw.abs_0[weakrates::kNuE];
        abs_0_loc[1] = raw.abs_0[weakrates::kAntiNuE];
        abs_1_loc[0] = raw.abs_1[weakrates::kNuE];
        abs_1_loc[1] = raw.abs_1[weakrates::kAntiNuE];
        scat_1_loc[0] = raw.scat_1[weakrates::kNuE];
        scat_1_loc[1] = raw.scat_1[weakrates::kAntiNuE];
        neq_0_loc[0] = local_eq.number[weakrates::kNuE];
        neq_0_loc[1] = local_eq.number[weakrates::kAntiNuE];
        neq_1_loc[0] = local_eq.energy[weakrates::kNuE];
        neq_1_loc[1] = local_eq.energy[weakrates::kAntiNuE];

        constexpr Real heavy_split = 0.5;
        for (int s = 2; s < nspecies_; ++s) {
          eta_0_loc[s] = heavy_split*raw.eta_0[weakrates::kNuX];
          eta_1_loc[s] = heavy_split*raw.eta_1[weakrates::kNuX];
          abs_0_loc[s] = raw.abs_0[weakrates::kNuX];
          abs_1_loc[s] = raw.abs_1[weakrates::kNuX];
          scat_1_loc[s] = raw.scat_1[weakrates::kNuX];
          neq_0_loc[s] = heavy_split*local_eq.number[weakrates::kNuX];
          neq_1_loc[s] = heavy_split*local_eq.energy[weakrates::kNuX];
        }

        // Species-dependent Kirchhoff closure retains the microscopic input and
        // derives its partner coefficient. Local equilibrium supplies provisional
        // heavy absorption only for the unchanged predictor; final closure uses
        // the selected equilibrium target:
        //
        //   charged-current (nue, anua): the microphysics supplies the opacity
        //     (Bruenn-85 absorption); the non-LTE factor scales it and final
        //     emissivity follows from eta = kappa * n_eq at the selected target;
        //   emission-only (the heavy-lepton channel, which has no charged current at
        //     all): the microphysics supplies the emissivity; the factor scales that
        //     emissivity. Preserve that emission and infer final kappa = eta / n_eq
        //     at the selected target after prediction.
        //
        // Pair, plasmon and bremsstrahlung emission for nue/anua has no charged-current
        // counterpart, so CC closure cannot recover it from the opacity. The enabled
        // separate thermal operator carries it with partner-dependent inverse losses;
        // adding its emissivity over n_eq to transport opacity would install an
        // emission rate as an absorption length, and the transport operator would then
        // trap electron neutrinos and antineutrinos at that rate.
        //
        // The absorptive fold is kept commented out as the reference for what was
        // rejected: the separate thermal operator does not restore that fold.
        // abs_0_loc[0] += (neq_0_loc[0] > 0.0)
        //                   ? raw.eta_0_non_cc[weakrates::kNuE]/neq_0_loc[0] : 0.0;
        // abs_0_loc[1] += (neq_0_loc[1] > 0.0)
        //                   ? raw.eta_0_non_cc[weakrates::kAntiNuE]/neq_0_loc[1] : 0.0;
        // abs_1_loc[0] += (neq_1_loc[0] > 0.0)
        //                   ? raw.eta_1_non_cc[weakrates::kNuE]/neq_1_loc[0] : 0.0;
        // abs_1_loc[1] += (neq_1_loc[1] > 0.0)
        //                   ? raw.eta_1_non_cc[weakrates::kAntiNuE]/neq_1_loc[1] : 0.0;

        // The non-LTE factor uses each evolved group's mean energy. Although
        // the heavy emissivity starts from one channel, the corrected rates
        // and the absorption coefficients inferred below can differ by group.
        // Apply the non-LTE E_nu^2 correction from W*(E-F_i*v^i)/N. The factor
        // is referenced to the local matter equilibrium, which is the state every raw
        // coefficient above was evaluated at. Final selected-target closure is a
        // separate operation: it reconstructs electron CC emission and infers heavy
        // absorption while preserving corrected heavy emission. It does not
        // re-reference the correction or apply it to the separate thermal r,q.
        // False retains the standard corrected-coefficient predictor. True
        // builds targets with uncorrected inputs and corrects only afterward.
        if (!weakrates_params_.gra_opacity_order) {
          weakrates::ApplySpectralCorrection(
              nspecies_, weakrates_params_.opacity_corr_fac_max,
              spectral_number, spectral_energy, neq_0_loc, neq_1_loc,
              eta_0_loc, eta_1_loc, abs_0_loc, abs_1_loc, scat_1_loc);
        }

        // Weakrates has spontaneous heavy emission but no raw heavy absorption.
        // Infer provisional local-equilibrium absorption before forming predictor
        // weights. Final published heavy absorption instead uses the selected target.
        // The unchanged predictor remains approximate: it is not iterated with the
        // final absorption and does not include the separate electron pair cooling.
        for (int s = 2; s < nspecies_; ++s) {
          abs_0_loc[s] = (neq_0_loc[s] > 0.0) ? eta_0_loc[s]/neq_0_loc[s] : 0.0;
          abs_1_loc[s] = (neq_1_loc[s] > 0.0) ? eta_1_loc[s]/neq_1_loc[s] : 0.0;
        }

        for (int s = 0; s < nspecies_; ++s) {
          point_finite = point_finite && Kokkos::isfinite(eta_0_loc[s]) &&
                         Kokkos::isfinite(eta_1_loc[s]) &&
                         Kokkos::isfinite(abs_0_loc[s]) &&
                         Kokkos::isfinite(abs_1_loc[s]) &&
                         Kokkos::isfinite(scat_1_loc[s]) &&
                         Kokkos::isfinite(neq_0_loc[s]) &&
                         Kokkos::isfinite(neq_1_loc[s]);
          eta_0_loc[s] = (eta_0_loc[s] > 0.0) ? eta_0_loc[s] : 0.0;
          eta_1_loc[s] = (eta_1_loc[s] > 0.0) ? eta_1_loc[s] : 0.0;
          abs_0_loc[s] = (abs_0_loc[s] > 0.0) ? abs_0_loc[s] : 0.0;
          abs_1_loc[s] = (abs_1_loc[s] > 0.0) ? abs_1_loc[s] : 0.0;
          scat_1_loc[s] = (scat_1_loc[s] > 0.0) ? scat_1_loc[s] : 0.0;
        }

        Real temp_star = temp;
        Real ye_star = ye;
        if (point_finite && !weakrates_params_.gra_equilibrium_targets &&
            weakrates_params_.use_partial_equilibrium) {
          const Real dtau = dt_*adm.alpha(m, k, j, i)/w_lorentz;
          // Only equal heavy equilibrium densities share the EOS coefficient.
          // Current densities keep the weights of their own inferred opacities.
          Real w_12, w_13, w_1x;
          partial_equilibrium_heavy_weights(dtau, abs_1_loc[2], abs_1_loc[3],
                                            nspecies_, w_12, w_13, w_1x);
          const Real a_1p = dtau*abs_1_loc[0];
          const Real a_1m = dtau*abs_1_loc[1];
          const Real a_0p = dtau*abs_0_loc[0];
          const Real a_0m = dtau*abs_0_loc[1];
          const Real w_1p = a_1p/(1.0 + a_1p);
          const Real w_1m = a_1m/(1.0 + a_1m);
          const Real w_0p = a_0p/(1.0 + a_0p);
          const Real w_0m = a_0m/(1.0 + a_0m);
          const bool weights_finite = Kokkos::isfinite(w_1p) &&
                                      Kokkos::isfinite(w_1m) &&
                                      Kokkos::isfinite(w_12) &&
                                      Kokkos::isfinite(w_13) &&
                                      Kokkos::isfinite(w_0p) &&
                                      Kokkos::isfinite(w_0m);
          Real w_max = (w_1p > w_1m) ? w_1p : w_1m;
          w_max = (w_max > w_12) ? w_max : w_12;
          w_max = (w_max > w_13) ? w_max : w_13;
          w_max = (w_max > w_0p) ? w_max : w_0p;
          w_max = (w_max > w_0m) ? w_max : w_0m;

          if (weights_finite && w_max >= weakrates_params_.peq_w_floor) {
            Real y_part[3] = {ye, 0.0, 0.0};
            Real t_lo = temp*(1.0 - peq_cv_eps);
            Real t_hi = temp*(1.0 + peq_cv_eps);
            t_lo = (t_lo > temp_code_min) ? t_lo : temp_code_min;
            t_hi = (t_hi < temp_code_max) ? t_hi : temp_code_max;
            const Real cv_heat =
                (t_hi > t_lo)
                    ? (eos.GetEnergy(nb, t_hi, y_part) -
                       eos.GetEnergy(nb, t_lo, y_part))/(t_hi - t_lo)
                    : 0.0;
            const bool cv_ok = Kokkos::isfinite(cv_heat) && cv_heat > 0.0;
            // Opposing heavy departures can cancel as a sum, but still
            // contribute separately to the matter-energy excursion bound.
            const Real dJ_x_hat =
                w_12*Kokkos::fabs(neq_1_loc[2] - nudens_1[2]) +
                w_13*Kokkos::fabs(neq_1_loc[3] - nudens_1[3]);
            const Real dlnT_hat =
                cv_ok ? (w_1p*Kokkos::fabs(neq_1_loc[0] - nudens_1[0]) +
                         w_1m*Kokkos::fabs(neq_1_loc[1] - nudens_1[1]) +
                         dJ_x_hat)/(temp*cv_heat)
                      : 0.0;
            const Real dYe_hat =
                (w_0p*Kokkos::fabs(neq_0_loc[0] - nudens_0[0]) +
                 w_0m*Kokkos::fabs(neq_0_loc[1] - nudens_0[1]))/nb;

            if (cv_ok && !(dlnT_hat < weakrates_params_.peq_dlnT_tol &&
                           dYe_hat < weakrates_params_.peq_dYe_tol)) {
              const Real dlnT_trust = peq_trust_c*dlnT_hat;
              const Real dYe_trust = peq_trust_c*dYe_hat;
              const Real dlnT_max =
                  (dlnT_trust > weakrates_params_.peq_dlnT_tol)
                      ? dlnT_trust : weakrates_params_.peq_dlnT_tol;
              const Real dYe_max =
                  (dYe_trust > weakrates_params_.peq_dYe_tol)
                      ? dYe_trust : weakrates_params_.peq_dYe_tol;
              const Real e_mat = eos.GetEnergy(nb, temp, y_part);

              Real soften = 1.0;
              for (int n_soft = 0; n_soft <= peq_max_halvings;
                   ++n_soft, soften *= 0.5) {
                const Real weights[PEQ_NWEIGHTS] = {
                    soften*w_1p, soften*w_1m, soften*w_1x,
                    soften*w_0p, soften*w_0m};
                // The EOS mean multiplies the equal equilibrium densities;
                // each current heavy density still has its own softened weight.
                const Real e_rhs = e_mat + weights[PEQ_W1_NUE]*nudens_1[0] +
                                   weights[PEQ_W1_ANUE]*nudens_1[1] +
                                   soften*(w_12*nudens_1[2] +
                                           w_13*nudens_1[3]);
                Real yl_rhs[3] = {
                    ye + (weights[PEQ_W0_NUE]*nudens_0[0] -
                          weights[PEQ_W0_ANUE]*nudens_0[1])/nb, 0.0, 0.0};
                Real t_try = temp;
                Real ye_try[3] = {ye, 0.0, 0.0};
                const bool ok = eos.GetBetaEquilibriumPartial(
                    nb, e_rhs, yl_rhs, weights, t_try, ye_try, temp, y_part);
                if (ok && Kokkos::isfinite(t_try) && Kokkos::isfinite(ye_try[0]) &&
                    Kokkos::fabs(Kokkos::log(t_try/temp)) <= dlnT_max &&
                    Kokkos::fabs(ye_try[0] - ye) <= dYe_max) {
                  temp_star = t_try;
                  ye_star = ye_try[0];
                  break;
                }
              }
            }
          }
        }

        Real neq_0_star[M1_TOTAL_NUM_SPECIES]{};
        Real neq_1_star[M1_TOTAL_NUM_SPECIES]{};
        if (point_finite && weakrates_params_.gra_equilibrium_targets) {
          // Raw selector uses mesh dt, not dt*alpha/W. Validate prospective
          // coefficients on copies so a subnormal or overflowing trial cannot
          // win selection and merely move the failure to the thermal consumer.
          const weakrates::GraTargetValidation validation{
              &raw, &weakrates_params_, nudens_0, nudens_1,
              neq_0_loc, neq_1_loc, eta_0_loc, eta_1_loc,
              abs_0_loc, abs_1_loc, scat_1_loc, thermal_enabled,
              spectral_number, spectral_energy};
          const auto equilibrium = weakrates::BuildGraEquilibriumTargets(
              eos, nb, temp, ye, dt_, raw.abs_1, raw.scat_1,
              nudens_0, nudens_1, neq_0_loc, neq_1_loc,
              neq_0_star, neq_1_star, &validation);
          if (!equilibrium.usable()) {
            Kokkos::printf("WEAKRATES GRA equilibrium invalid/unusable target: "
                           "cell=(%d,%d,%d,%d), quality=%d, eos_status=%d\n",
                           m, k, j, i, static_cast<int>(equilibrium.status),
                           equilibrium.eos_status);
            Kokkos::abort("WEAKRATES GRA equilibrium: invalid inputs/configuration "
                          "or no publishable target, including local fallback");
          }
          if (equilibrium.approximate() && i >= is && i <= ie &&
              j >= js && j <= je && k >= ks && k <= ke) {
            const auto previous = Kokkos::atomic_fetch_add(&equilibrium_counts(0), 1ULL);
            if (equilibrium.status == weakrates::GraEquilibriumStatus::LocalFallback) {
              Kokkos::atomic_add(&equilibrium_counts(1), 1ULL);
            }
            Kokkos::atomic_add(&equilibrium_counts(2),
                              static_cast<unsigned long long>(equilibrium.evaluations));
            Kokkos::atomic_max(&equilibrium_diagnostics(0), equilibrium.score);
            Kokkos::atomic_max(&equilibrium_diagnostics(1),
                              equilibrium.normalized_energy_defect);
            Kokkos::atomic_max(&equilibrium_diagnostics(2),
                              equilibrium.normalized_lepton_defect);
            if (previous == 0ULL) {
              // One representative (not necessarily worst) fallback per pack.
              equilibrium_diagnostics(3) = equilibrium.temp;
              equilibrium_diagnostics(4) = equilibrium.ye;
              equilibrium_diagnostics(5) = equilibrium.energy_defect;
              equilibrium_diagnostics(6) = equilibrium.lepton_defect;
              equilibrium_diagnostics(7) = equilibrium.score;
              equilibrium_diagnostics(8) = equilibrium.eos_status;
              equilibrium_diagnostics(9) = equilibrium.on_boundary ? 1.0 : 0.0;
              equilibrium_diagnostics(10) = m;
              equilibrium_diagnostics(11) = k;
              equilibrium_diagnostics(12) = j;
              equilibrium_diagnostics(13) = i;
            }
          }
        } else if (point_finite) {
          Real ye_star_arr[3] = {ye_star, 0.0, 0.0};
          const Real mu_le_star =
              eos.GetElectronLeptonChemicalPotential(nb, temp_star, ye_star_arr);
          const Real eta_nue_star = mu_le_star/temp_star;
          weakrates::EquilibriumDensities star_eq =
              weakrates::ComputeEquilibriumDensities(temp_star, eta_nue_star);
          weakrates::ConvertToCodeUnits(conversions, star_eq);
          bool star_finite = Kokkos::isfinite(eta_nue_star) &&
                             weakrates::IsFinite(star_eq);
          neq_0_star[0] = star_eq.number[weakrates::kNuE];
          neq_0_star[1] = star_eq.number[weakrates::kAntiNuE];
          neq_1_star[0] = star_eq.energy[weakrates::kNuE];
          neq_1_star[1] = star_eq.energy[weakrates::kAntiNuE];
          for (int s = 2; s < nspecies_; ++s) {
            neq_0_star[s] = heavy_split*star_eq.number[weakrates::kNuX];
            neq_1_star[s] = heavy_split*star_eq.energy[weakrates::kNuX];
          }
          for (int s = 0; s < nspecies_; ++s) {
            star_finite = star_finite && Kokkos::isfinite(neq_0_star[s]) &&
                          Kokkos::isfinite(neq_1_star[s]) &&
                          neq_0_star[s] >= 0.0 && neq_1_star[s] >= 0.0;
          }
          if (!star_finite) {
            for (int s = 0; s < nspecies_; ++s) {
              neq_0_star[s] = neq_0_loc[s];
              neq_1_star[s] = neq_1_loc[s];
            }
          }
        }

        if (weakrates_params_.gra_opacity_order ||
            weakrates_params_.gra_target_mean) {
          if (weakrates_params_.gra_target_mean) {
            // Reset only spectral inputs, not thermal r/q or target selection.
            // With standard ordering the predictor used local-corrected rates;
            // the published coefficients must not receive both correction factors.
            for (int s = 0; s < nspecies_; ++s) {
              const int rs = (s < 2) ? s : weakrates::kNuX;
              scat_1_loc[s] = raw.scat_1[rs];
              if (s < 2) {
                abs_0_loc[s] = raw.abs_0[rs];
                abs_1_loc[s] = raw.abs_1[rs];
              } else {
                eta_0_loc[s] = heavy_split*raw.eta_0[weakrates::kNuX];
                eta_1_loc[s] = heavy_split*raw.eta_1[weakrates::kNuX];
              }
            }
          }
          const Real *reference_number =
              weakrates_params_.gra_target_mean ? neq_0_star : neq_0_loc;
          const Real *reference_energy =
              weakrates_params_.gra_target_mean ? neq_1_star : neq_1_loc;
          weakrates::ApplySpectralCorrection(
              nspecies_, weakrates_params_.opacity_corr_fac_max,
              spectral_number, spectral_energy, reference_number, reference_energy,
              eta_0_loc, eta_1_loc, abs_0_loc, abs_1_loc, scat_1_loc);
        }

        // Always finalize, even after prior invalidity or with thermal sources off.
        // Retain electron CC absorption and corrected spontaneous heavy emission;
        // the helper accumulates the original five-field finite checks.
        point_finite = weakrates::FinalizeKirchhoffCoefficients(
            nspecies_, point_finite, neq_0_star, neq_1_star, eta_0_loc, eta_1_loc,
            abs_0_loc, abs_1_loc, scat_1_loc);

        weakrates::ElectronThermalCoefficients thermal{};
        if (thermal_enabled && point_finite) {
          // Use converted raw r,q without correction or species splitting, and
          // the already-selected electron references. New-model failure is fatal,
          // not the original microscopic-invalidity transparent-cell policy.
          if (!weakrates::BuildElectronThermalCoefficients(
                  raw, neq_0_star, neq_1_star, thermal)) {
            Kokkos::abort("WEAKRATES electron thermal coefficients: invalid rates, "
                          "electron-rate symmetry, or selected references");
          }
        }

        if (!point_finite) {
          for (int s = 0; s < nspecies_; ++s) {
            eta_0_(m, s, k, j, i) = 0.0;
            abs_0_(m, s, k, j, i) = 0.0;
            eta_1_(m, s, k, j, i) = 0.0;
            abs_1_(m, s, k, j, i) = 0.0;
            scat_1_(m, s, k, j, i) = 0.0;
          }
          return;
        }

        for (int s = 0; s < nspecies_; ++s) {
          eta_0_(m, s, k, j, i) = (eta_0_loc[s] > 0.0) ? eta_0_loc[s] : 0.0;
          abs_0_(m, s, k, j, i) = (abs_0_loc[s] > 0.0) ? abs_0_loc[s] : 0.0;
          eta_1_(m, s, k, j, i) = (eta_1_loc[s] > 0.0) ? eta_1_loc[s] : 0.0;
          abs_1_(m, s, k, j, i) = (abs_1_loc[s] > 0.0) ? abs_1_loc[s] : 0.0;
          scat_1_(m, s, k, j, i) = (scat_1_loc[s] > 0.0) ? scat_1_loc[s] : 0.0;
        }
        // All six fields are undensitized and stage-held, including zero-rate
        // enabled cells. Consumers distinguish enablement from local activity.
        if (thermal_enabled) {
          weakrates_thermal_(m, weakrates::kThermalNumberRate, k, j, i) = thermal.r;
          weakrates_thermal_(m, weakrates::kThermalEnergyRate, k, j, i) = thermal.q;
          weakrates_thermal_(m, weakrates::kThermalNumberNuE, k, j, i) =
              thermal.number[0];
          weakrates_thermal_(m, weakrates::kThermalNumberAntiNuE, k, j, i) =
              thermal.number[1];
          weakrates_thermal_(m, weakrates::kThermalEnergyNuE, k, j, i) =
              thermal.energy[0];
          weakrates_thermal_(m, weakrates::kThermalEnergyAntiNuE, k, j, i) =
              thermal.energy[1];
        }
      });
  if (weakrates_params_.gra_equilibrium_targets) {
    auto counts = Kokkos::create_mirror(equilibrium_counts);
    Kokkos::deep_copy(counts, equilibrium_counts);
    if (counts(0) > 0ULL) {
      auto diagnostic = Kokkos::create_mirror(equilibrium_diagnostics);
      Kokkos::deep_copy(diagnostic, equilibrium_diagnostics);
      std::cout << "WEAKRATES GRA equilibrium fallback: rank=" << global_variable::my_rank
                << ", cycle=" << pmy_pack->pmesh->ncycle
                << ", cells=" << counts(0) << ", local_safety=" << counts(1)
                << ", evaluations=" << counts(2) << ", max_score=" << diagnostic(0)
                << ", max_abs_dE_over_SE=" << diagnostic(1)
                << ", max_abs_dYL_over_SL=" << diagnostic(2)
                << ", representative_cell=(" << static_cast<int>(diagnostic(10))
                << ',' << static_cast<int>(diagnostic(11))
                << ',' << static_cast<int>(diagnostic(12))
                << ',' << static_cast<int>(diagnostic(13))
                << "), T=" << diagnostic(3) << ", Ye=" << diagnostic(4)
                << ", dE=" << diagnostic(5) << ", dYL=" << diagnostic(6)
                << ", score=" << diagnostic(7)
                << ", eos_status=" << static_cast<int>(diagnostic(8))
                << ", boundary=" << static_cast<int>(diagnostic(9)) << '\n';
    }
  }
  return TaskStatus::complete;
}

}  // namespace radiationm1
