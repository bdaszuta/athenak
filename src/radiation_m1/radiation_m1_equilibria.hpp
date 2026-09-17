#ifndef RADIATION_M1_EQUILIBRIA_HPP
#define RADIATION_M1_EQUILIBRIA_HPP
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_equilibria.hpp
//  \brief backend-neutral fluid-frame equilibrium densities for the M1 module
//
//  Consumers such as the conservative M1 projection ask for the fluid-frame equilibrium
//  number and energy densities of the neutrino transport groups without naming an opacity
//  backend. Each backend keeps its own physical constants and its own unit convention: the
//  neutrino backends evaluate the equilibrium integrals in the NGS system and convert to
//  EOS number-density and code energy-density units internally. The caller supplies only
//  the matter state (temperature in MeV, electron-neutrino degeneracy parameter) and the
//  unit systems.
//
//  The group layout is [nue, anue, nux, anux]; the two heavy-lepton groups share the summed
//  nux density equally. The compiled capacity M1_TOTAL_NUM_SPECIES is four for every build
//  and the active transport-group count follows the selected opacity backend, so this
//  interface always fills equilibrium_nspecies groups.

#include <math.h>

#include "athena.hpp"
#include "eos/primitive-solver/fermi_dirac_integrals.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "radiation_m1/radiation_m1_params.hpp"
#include "radiation_m1/radiation_m1_weakrates.hpp"

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \brief number of transport groups filled by radiationm1::equilibrium_densities

constexpr int equilibrium_nspecies = 4;

static_assert(M1_TOTAL_NUM_SPECIES == equilibrium_nspecies,
              "the equilibrium interface requires a four-group M1 state layout");

//----------------------------------------------------------------------------------------
//! \fn bool radiationm1::equilibrium_available
//  \brief does this opacity backend supply fluid-frame equilibrium densities?
//
//  A backend added on another branch extends this predicate together with the dispatch in
//  equilibrium_densities. A consumer that requires the equilibrium state must treat a
//  false answer as unsupported rather than assuming a particular backend.

KOKKOS_INLINE_FUNCTION
bool equilibrium_available(const RadiationM1OpacityType opacity_type) {
  return opacity_type == BnsNurates || opacity_type == WeakRates;
}

//----------------------------------------------------------------------------------------
//! \fn bool radiationm1::equilibrium_densities
//  \brief fluid-frame equilibrium number and energy densities in code units
//
//  \param opacity_type selected opacity backend; its own constants and unit convention are
//                      applied internally
//  \param code_units   code unit system
//  \param eos_units    EOS unit system
//  \param ngs_units    NGS unit system used by the neutrino equilibrium integrals
//  \param temp         matter temperature in MeV
//  \param eta_nue      electron-neutrino degeneracy parameter
//  \param n            returned number densities, [nue, anue, nux, anux], code units
//  \param J            returned energy densities, [nue, anue, nux, anux], code units
//  Returns false for a nonfinite state or for a backend without equilibrium densities.

KOKKOS_INLINE_FUNCTION
bool equilibrium_densities(const RadiationM1OpacityType opacity_type,
                           const Primitive::UnitSystem &code_units,
                           const Primitive::UnitSystem &eos_units,
                           const Primitive::UnitSystem &ngs_units,
                           const Real temp, const Real eta_nue,
                           Real n[equilibrium_nspecies], Real J[equilibrium_nspecies]) {
  if (!Kokkos::isfinite(eta_nue) || !Kokkos::isfinite(temp)) {
    return false;
  }

  Real n_nue, n_anue, n_nux_total;
  Real J_nue, J_anue, J_nux_total;
  if (opacity_type == BnsNurates) {
    const Real eta_anue = -eta_nue;
    const Real hc_mevnm = 1.23984172e-10 * 1.0e7;
    const Real hc_mevnm3 = hc_mevnm * hc_mevnm * hc_mevnm;
    const Real temp3 = temp * temp * temp;
    const Real temp4 = temp3 * temp;

    n_nue = 4.0 * M_PI / hc_mevnm3 * temp3 *
            Primitive::FermiDirac::F2(eta_nue);
    n_anue = 4.0 * M_PI / hc_mevnm3 * temp3 *
             Primitive::FermiDirac::F2(eta_anue);
    n_nux_total = 16.0 * M_PI / hc_mevnm3 * temp3 *
                  Primitive::FermiDirac::F2(0.0);
    J_nue = 4.0 * M_PI / hc_mevnm3 * temp4 *
            Primitive::FermiDirac::F3(eta_nue);
    J_anue = 4.0 * M_PI / hc_mevnm3 * temp4 *
             Primitive::FermiDirac::F3(eta_anue);
    J_nux_total = 16.0 * M_PI / hc_mevnm3 * temp4 *
                  Primitive::FermiDirac::F3(0.0);

    const Real unit_num_dens = eos_units.NumberDensityConversion(ngs_units);
    const Real unit_ene_dens = code_units.EnergyDensityConversion(ngs_units);
    n_nue /= unit_num_dens;
    n_anue /= unit_num_dens;
    n_nux_total /= unit_num_dens;
    J_nue /= unit_ene_dens;
    J_anue /= unit_ene_dens;
    J_nux_total /= unit_ene_dens;
  } else if (opacity_type == WeakRates) {
    // The weakrates backend evaluates its equilibrium densities in CGS and converts them to EOS
    // number-density and code energy-density units with its own conversion struct. That struct is
    // defined against CGS, the unit system whose derived units are all unity (see
    // Primitive::MakeCGS), so each factor is the corresponding UnitSystem conversion with
    // b = CGS evaluated here:
    //   LengthConversion(b)        = b.length/length                    -> 1/code_units.length
    //   TimeConversion(b)          = b.time/time                        -> 1/code_units.time
    //   NumberDensityConversion(b) = b.numberDensity/numberDensity      -> 1/eos_units.numberDensity
    //   EnergyDensityConversion(b) = (volume/b.volume)*(b.energy/energy) -> code_units.volume/code_units.energy
    // The object Primitive::CGS cannot be named: this function is device code, while that object is
    // host-only. tst/test_m1_equilibria_weakrates.py checks these factors against the unit-system
    // definitions and the unity CGS fields they assume.
    const radiationm1::weakrates::UnitConversions conversions{
        1.0 / code_units.length,
        1.0 / code_units.time,
        1.0 / eos_units.numberDensity,
        code_units.volume / code_units.energy};
    radiationm1::weakrates::EquilibriumDensities densities =
        radiationm1::weakrates::ComputeEquilibriumDensities(temp, eta_nue);
    radiationm1::weakrates::ConvertToCodeUnits(conversions, densities);
    n_nue = densities.number[radiationm1::weakrates::kNuE];
    n_anue = densities.number[radiationm1::weakrates::kAntiNuE];
    n_nux_total = densities.number[radiationm1::weakrates::kNuX];
    J_nue = densities.energy[radiationm1::weakrates::kNuE];
    J_anue = densities.energy[radiationm1::weakrates::kAntiNuE];
    J_nux_total = densities.energy[radiationm1::weakrates::kNuX];
  } else {
    return false;
  }

  n[0] = n_nue;
  n[1] = n_anue;
  J[0] = J_nue;
  J[1] = J_anue;
  for (int s = 2; s < equilibrium_nspecies; ++s) {
    n[s] = 0.5 * n_nux_total;
    J[s] = 0.5 * J_nux_total;
  }
  return true;
}

}  // namespace radiationm1

#endif  // RADIATION_M1_EQUILIBRIA_HPP
