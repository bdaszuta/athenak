// Real CCSN task, M1 source and stress-energy regressions. One case per process:
// CCSN owns static bounce/empirical state. No production implementation is included here.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "driver/driver.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"
#include "radiation_m1/radiation_m1_macro.hpp"
#include "z4c/tmunu.hpp"

void Deleptonization(Mesh *, Real);  // Existing CCSN callback; defined in ccsn.cpp.

namespace {
void Require(bool ok, const std::string &why) {
  if (!ok) throw std::runtime_error(why);
}

template <typename View>
auto HostCopy(const View &view) {
  auto saved = Kokkos::create_mirror(view);  // Allocates even for Serial Views.
  Kokkos::deep_copy(saved, view);
  return saved;
}

// An error bound for each independently rounded material/radiation identity.
Real Bound(Real a, Real b, Real c, Real d = 0) {
  return 128 * std::numeric_limits<Real>::epsilon() *
         (std::abs(a) + std::abs(b) + std::abs(c) + std::abs(d));
}

void Near(Real actual, Real expected, Real scale, const std::string &why) {
  Require(std::isfinite(actual) && std::isfinite(expected) &&
          std::abs(actual - expected) <=
              128 * std::numeric_limits<Real>::epsilon() * scale, why);
}

std::vector<Real> Moments(Mesh &mesh) {
  auto &rad = *mesh.pmb_pack->pradm1;
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rad.u0);
  std::vector<Real> values;
  const auto &b = mesh.mb_indcs;
  for (int i = b.is; i <= b.ie; ++i) {
    for (int q = 0; q < rad.nvarstot; ++q) values.push_back(h(0, q, b.ks, b.js, i));
  }
  return values;  // Independent host-owned storage (Serial mirrors can alias Views).
}

Real Trap(ParameterInput &pin, Mesh &mesh) {
  return pin.GetReal("problem", "rho_trap") /
      mesh.pmb_pack->pdyngr->GetCodeUnitSystem().MassDensityConversion(
          Primitive::MakeCGS());
}

// Set a stationary, EOS-interior, no-field state across active and ghost cells.
// P2C/C2P exercise the real material layout; restore the equality probe after
// C2P because recovered floating-point rho need not be bit-identical to rho_trap.
template <typename EOS>
void ControlledState(Mesh &mesh, Driver &driver, EOS &eos, Real trap) {
  auto *pack = mesh.pmb_pack;
  const auto &b = mesh.mb_indcs;
  const Real mb = eos.GetBaryonMass();
  const Real Y = 0.42, T = 8.0;
  Require(std::isfinite(trap) && trap > 0 &&
          0.5 * trap / mb > eos.GetMinimumDensity() &&
          1.5 * trap / mb < eos.GetMaximumDensity() &&
          T > eos.GetMinimumTemperature() && T < eos.GetMaximumTemperature() &&
          Y > eos.GetMinimumSpeciesFraction(0) &&
          Y < eos.GetMaximumSpeciesFraction(0), "controlled state outside CompOSE table");
  Kokkos::deep_copy(pack->pmhd->b0.x1f, 0.0);
  Kokkos::deep_copy(pack->pmhd->b0.x2f, 0.0);
  Kokkos::deep_copy(pack->pmhd->b0.x3f, 0.0);
  Kokkos::deep_copy(pack->pmhd->bcc0, 0.0);
  auto metric = Kokkos::create_mirror_view(pack->padm->u_adm);
  Kokkos::deep_copy(metric, 0.0);
  for (int i = 0; i < static_cast<int>(metric.extent(4)); ++i) {
    const Real gxx = (i == b.is + 3) ? 4.0 : 1.0;
    metric(0, adm::ADM::I_ADM_GXX, b.ks, b.js, i) = gxx;
    metric(0, adm::ADM::I_ADM_GYY, b.ks, b.js, i) = 1.0;
    metric(0, adm::ADM::I_ADM_GZZ, b.ks, b.js, i) = 1.0;
    metric(0, adm::ADM::I_ADM_ALPHA, b.ks, b.js, i) = 1.0;
    metric(0, adm::ADM::I_ADM_PSI4, b.ks, b.js, i) = 1.0;
  }
  Kokkos::deep_copy(pack->padm->u_adm, metric);
  auto w = Kokkos::create_mirror(pack->pmhd->w0);
  Kokkos::deep_copy(w, 0.0);
  for (int i = 0; i < static_cast<int>(w.extent(4)); ++i) {
    Real rho = trap;
    if (i == b.is || i == b.is + 4) rho *= 0.5;
    if (i == b.is + 2) rho *= 1.5;
    if (i == b.is + 3) rho *= 0.75;
    w(0, IDN, b.ks, b.js, i) = rho;
    w(0, IYF, b.ks, b.js, i) = Y;
  }
  Kokkos::deep_copy(pack->pmhd->w0, w);
  auto eos_device = eos;
  auto primitive = pack->pmhd->w0;
  par_for("ccsn_test_pressure", DevExeSpace(), 0, pack->nmb_thispack - 1,
          b.ks, b.ke, b.js, b.je, 0, static_cast<int>(w.extent(4)) - 1,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
            Real fractions[MAX_SPECIES] = {0.0};
            fractions[0] = primitive(m, IYF, k, j, i);
            primitive(m, IPR, k, j, i) = eos_device.GetPressure(
                primitive(m, IDN, k, j, i) / mb, T, fractions);
          });
  Kokkos::fence();
  auto exact = HostCopy(pack->pmhd->w0);
  pack->pdyngr->PrimToConInit(0, static_cast<int>(w.extent(4)) - 1,
                              b.js, b.je, b.ks, b.ke);
  pack->pdyngr->ConToPrim(&driver, 0);
  auto published = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                        pack->pdyngr->temperature);
  for (int i = b.is; i <= b.ie; ++i) {
    Require(std::isfinite(published(0, 0, b.ks, b.js, i)) &&
            published(0, 0, b.ks, b.js, i) > eos.GetMinimumTemperature(),
            "C2P failed to publish an interior temperature");
  }
  // Restore exact threshold density with consistent pressure and conserved fields.
  Kokkos::deep_copy(pack->pmhd->w0, exact);
  pack->pdyngr->PrimToConInit(0, static_cast<int>(w.extent(4)) - 1,
                              b.js, b.je, b.ks, b.ke);
  Kokkos::fence();
}

void Select(Mesh &mesh, Driver &driver) {
  driver.ExecuteTaskList(&mesh, "before_timeintegrator", 0);
  Kokkos::fence();
}

void CheckMembership(Mesh &mesh, int offset, bool expected) {
  auto &rad = *mesh.pmb_pack->pradm1;
  Require(rad.CouplingRegionEnabled(), "regional restriction unexpectedly disabled");
  auto region = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                    rad.coupling_region);
  const auto &b = mesh.mb_indcs;
  Require(region(0, b.ks, b.js, b.is + offset) == expected,
          "frozen selector disagrees with primitive rest density");
}

// This probes the real additive implementation, not an independently copied gate.
void CheckTmunu(Mesh &mesh, Driver &driver, bool regional) {
  auto *pack = mesh.pmb_pack;
  auto &rad = *pack->pradm1;
  const auto &b = mesh.mb_indcs;
  pack->pdyngr->SetTmunu(&driver, 1);  // Reset before EACH additive call.
  Kokkos::fence();
  auto before = HostCopy(pack->ptmunu->u_tmunu);
  rad.SetTmunu(&driver, 1);
  Kokkos::fence();
  auto after = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                   pack->ptmunu->u_tmunu);
  auto u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rad.u0);
  auto region = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                    rad.coupling_region);
  const int S[] = {Tmunu::I_Tmunu_Sx, Tmunu::I_Tmunu_Sy, Tmunu::I_Tmunu_Sz};
  const int P[] = {Tmunu::I_Tmunu_Sxx, Tmunu::I_Tmunu_Syy, Tmunu::I_Tmunu_Szz};
  for (int i = b.is; i <= b.ie; ++i) {
    const bool selected = !regional || region(0, b.ks, b.js, i);
    const Real gxx = (i == b.is + 3) ? 4.0 : 1.0;
    const Real vol = (i == b.is + 3) ? 2.0 : 1.0;
    Real energy = 0, flux[3] = {0, 0, 0};
    for (int s = 0; s < rad.nspecies; ++s) {
      energy += u(0, radiationm1::CombinedIdx(s, M1_E_IDX, rad.nvars), b.ks, b.js, i);
      for (int a = 0; a < 3; ++a) {
        flux[a] += u(0, radiationm1::CombinedIdx(s, M1_FX_IDX + a, rad.nvars),
                     b.ks, b.js, i);
      }
    }
    auto check = [&](int component, Real addition) {
      const Real x = before(0, component, b.ks, b.js, i);
      const Real y = after(0, component, b.ks, b.js, i);
      if (selected) {
        Near(y - x, addition, std::abs(x) + std::abs(y) + std::abs(addition),
             "radiation Tmunu addition/undensitization mismatch");
      } else {
        Require(y == x, "unselected cell received radiation Tmunu");
      }
    };
    check(Tmunu::I_Tmunu_E, energy / vol);
    for (int a = 0; a < 3; ++a) {
      check(S[a], flux[a] / vol);
      check(P[a], energy * (a == 0 ? gxx : 1.0) / (3 * vol));
    }
  }
}

void SeedRadiation(Mesh &mesh, Real trap, Real mb) {
  auto &rad = *mesh.pmb_pack->pradm1;
  auto h = Kokkos::create_mirror_view(rad.u0);
  Kokkos::deep_copy(h, 0.0);
  for (int i = 0; i < static_cast<int>(h.extent(4)); ++i) {
    for (int s = 0; s < rad.nspecies; ++s) {
      Real E = 0.03 * trap * (1.0 + 0.1 * s);
      h(0, radiationm1::CombinedIdx(s, M1_E_IDX, rad.nvars), 0, 0, i) = E;
      h(0, radiationm1::CombinedIdx(s, M1_FX_IDX, rad.nvars), 0, 0, i) = 0.1 * E;
      h(0, radiationm1::CombinedIdx(s, M1_FY_IDX, rad.nvars), 0, 0, i) = 0.08 * E;
      h(0, radiationm1::CombinedIdx(s, M1_FZ_IDX, rad.nvars), 0, 0, i) = 0.06 * E;
      h(0, radiationm1::CombinedIdx(s, M1_N_IDX, rad.nvars), 0, 0, i) =
          0.01 * trap / mb * (1.0 + 0.2 * s);
    }
  }
  Kokkos::deep_copy(rad.u0, h);
  Kokkos::deep_copy(rad.uflx.x1f, 0.0);
  Kokkos::deep_copy(rad.uflx.x2f, 0.0);
  Kokkos::deep_copy(rad.uflx.x3f, 0.0);
  Kokkos::deep_copy(rad.eta_0, 0.0);
  Kokkos::deep_copy(rad.eta_1, 0.0);
  Kokkos::deep_copy(rad.abs_0, 20.0);
  Kokkos::deep_copy(rad.abs_1, 20.0);
  Kokkos::deep_copy(rad.scat_1, 10.0);
  rad.SetMask(nullptr, 1);
  rad.FloorAndCalcClosure(nullptr, 1);
  Kokkos::fence();
}

void Sources(Mesh &mesh, Driver &driver, Real mb, bool exterior_coupled) {
  auto *pack = mesh.pmb_pack;
  auto &rad = *pack->pradm1;
  const auto &b = mesh.mb_indcs;
  auto initial_matter = HostCopy(pack->pmhd->u0);
  auto initial_rad = HostCopy(rad.u0);
  rad.CopyCons(&driver, 1);
  rad.TimeUpdate(&driver, 1);
  Kokkos::fence();
  auto stage1_matter = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                           pack->pmhd->u0);
  for (int i = b.is; i <= b.ie; ++i) {
    for (int q : {IDN, IM1, IM2, IM3, IEN, IYF}) {
      Require(stage1_matter(0, q, b.ks, b.js, i) ==
                  initial_matter(0, q, b.ks, b.js, i),
              "stage 1 deposited material source");
    }
  }
  rad.FloorAndCalcClosure(&driver, 2);
  rad.TimeUpdate(&driver, 2);  // No second CopyCons: u1 is the full-step base.
  Kokkos::fence();
  auto final_matter = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                          pack->pmhd->u0);
  auto final_rad = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rad.u0);
  auto region = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                    rad.coupling_region);
  int resolved = 0;
  for (int i = b.is; i <= b.ie; ++i) {
    bool selected = !rad.CouplingRegionEnabled() || region(0, b.ks, b.js, i);
    if (i == b.is) Require(selected == exterior_coupled, "wrong source region at exterior");
    auto delta_m = [&](int q) {
      return final_matter(0, q, b.ks, b.js, i) -
             initial_matter(0, q, b.ks, b.js, i);
    };
    Require(delta_m(IDN) == 0.0, "collision source changed conserved D");
    Real radiation[5] = {}, rad_operands[5] = {};
    for (int s = 0; s < rad.nspecies; ++s) {
      for (int q = 0; q < rad.nvars; ++q) {
        int v = radiationm1::CombinedIdx(s, q, rad.nvars);
        const Real old = initial_rad(0, v, b.ks, b.js, i);
        const Real now = final_rad(0, v, b.ks, b.js, i);
        Require(std::isfinite(now), "nonfinite accepted M1 source");
        if (q <= M1_FZ_IDX) {
          radiation[q] += now - old;
          rad_operands[q] += std::abs(old) + std::abs(now);
        }
        if (q == M1_N_IDX && s < 2) {
          radiation[M1_N_IDX] += (s == 0 ? 1 : -1) * (now - old);
          rad_operands[M1_N_IDX] += std::abs(old) + std::abs(now);
        }
      }
    }
    if (selected) {
      for (int q = 0; q <= M1_FZ_IDX; ++q) {
        int material_index = q == 0 ? IEN : IM1 + q - 1;
        Real old = initial_matter(0, material_index, b.ks, b.js, i);
        Real now = final_matter(0, material_index, b.ks, b.js, i);
        Real defect = (now - old) + radiation[q];
        Real limit = Bound(old, now, radiation[q], rad_operands[q]);
        Require(std::isfinite(defect) && std::abs(defect) <= limit,
                "unpaired accepted energy/momentum collision source");
        if (i == b.is + 1) {
          Require(std::abs(now - old) > limit && std::abs(radiation[q]) > limit,
                  "energy or three-momentum source unresolved");
          ++resolved;
        }
      }
      Real old = initial_matter(0, IYF, b.ks, b.js, i);
      Real now = final_matter(0, IYF, b.ks, b.js, i);
      Real defect = (now - old) + mb * radiation[M1_N_IDX];
      Real limit = Bound(old, now, mb * radiation[M1_N_IDX],
                         mb * rad_operands[M1_N_IDX]);
      Require(std::isfinite(defect) && std::abs(defect) <= limit,
              "unpaired accepted electron-lepton collision source");
      if (i == b.is + 1) {
        Require(std::abs(now - old) > limit, "electron-lepton source unresolved");
        ++resolved;
      }
    } else {
      for (int q : {IEN, IM1, IM2, IM3, IYF})
        Require(delta_m(q) == 0.0, "one-way exterior deposited material source");
      if (i == b.is) {
        Real E = radiation[M1_E_IDX];
        Require(std::abs(E) > Bound(initial_rad(0, 0, b.ks, b.js, i),
                                    final_rad(0, 0, b.ks, b.js, i), E),
                "exterior radiation failed to evolve");
      }
    }
  }
  Require(resolved == 5, "not all local exchange channels resolved");
}

// Controlled source coefficients above are prescribed. This separate case
// checks that the real NuRATES path computes rates on both sides of the region.
void PhysicalOpacities(Mesh &mesh, Driver &driver, Real trap, Real mb) {
  auto &rad = *mesh.pmb_pack->pradm1;
  const auto &b = mesh.mb_indcs;
  SeedRadiation(mesh, trap, mb);
  const auto saved = Moments(mesh);
  rad.CalcOpacityNurates(&driver, 1);
  Kokkos::fence();
  Require(saved == Moments(mesh), "opacity evaluation changed radiation moments");
  auto eta0 = HostCopy(rad.eta_0);
  auto eta1 = HostCopy(rad.eta_1);
  auto abs0 = HostCopy(rad.abs_0);
  auto abs1 = HostCopy(rad.abs_1);
  auto scat1 = HostCopy(rad.scat_1);
  for (int i : {b.is, b.is + 1}) {  // exterior and selected core
    Real electron_emission = 0.0;
    Real electron_absorption = 0.0;
    for (int s = 0; s < rad.nspecies; ++s) {
      const Real rates[] = {eta0(0, s, b.ks, b.js, i),
                            eta1(0, s, b.ks, b.js, i),
                            abs0(0, s, b.ks, b.js, i),
                            abs1(0, s, b.ks, b.js, i),
                            scat1(0, s, b.ks, b.js, i)};
      for (Real rate : rates)
        Require(std::isfinite(rate) && rate >= 0.0,
                "nonfinite or negative physical NuRATES coefficient");
      if (s < 2) {
        electron_emission += rates[0];
        electron_absorption += rates[2];
      }
    }
    Require(electron_emission > 0.0 && electron_absorption > 0.0,
            "no resolved physical electron-neutrino rates");
  }
}

// Coordinate excision changes after selection; it must not change density bits.
void CurrentExcision(Mesh &mesh, Driver &driver, Real trap, Real mb) {
  auto *pack = mesh.pmb_pack;
  auto &rad = *pack->pradm1;
  auto &coord = *pack->pcoord;
  const auto &b = mesh.mb_indcs;
  Kokkos::realloc(coord.excision_floor, rad.u0.extent(0), rad.u0.extent(2),
                  rad.u0.extent(3), rad.u0.extent(4));
  Kokkos::realloc(coord.excision_flux, rad.u0.extent(0), rad.u0.extent(2),
                  rad.u0.extent(3), rad.u0.extent(4));
  Kokkos::deep_copy(coord.excision_floor, false);
  Kokkos::deep_copy(coord.excision_flux, false);
  auto floor = Kokkos::create_mirror_view(coord.excision_floor);
  floor(0, b.ks, b.js, b.is + 2) = true;  // selected
  floor(0, b.ks, b.js, b.is + 4) = true;  // exterior
  Kokkos::deep_copy(coord.excision_floor, floor);
  coord.coord_data.bh_excise = true;
  CheckMembership(mesh, 2, true);
  CheckMembership(mesh, 4, false);

  auto before_w = HostCopy(pack->pmhd->w0);
  auto before_T = HostCopy(pack->pdyngr->temperature);
  Deleptonization(&mesh, mesh.dt);
  Kokkos::fence();
  auto after_w = HostCopy(pack->pmhd->w0);
  auto after_T = HostCopy(pack->pdyngr->temperature);
  for (int i : {b.is + 2, b.is + 4}) {
    Require(after_w(0, IYF, b.ks, b.js, i) == before_w(0, IYF, b.ks, b.js, i) &&
                after_w(0, IPR, b.ks, b.js, i) == before_w(0, IPR, b.ks, b.js, i) &&
                after_T(0, 0, b.ks, b.js, i) == before_T(0, 0, b.ks, b.js, i),
            "current excision or selection failed to exclude empirical map");
  }
  Require(after_w(0, IYF, b.ks, b.js, b.is) <
              before_w(0, IYF, b.ks, b.js, b.is),
          "unexcised exterior did not apply empirical map");

  SeedRadiation(mesh, trap, mb);  // Calls actual SetMask after the coordinate change.
  auto mask = HostCopy(rad.radiation_mask);
  auto radiation = HostCopy(rad.u0);
  for (int i : {b.is + 2, b.is + 4}) {
    Require(mask(0, b.ks, b.js, i), "coordinate excision not in radiation mask");
    for (int v = 0; v < rad.nvarstot; ++v)
      Require(radiation(0, v, b.ks, b.js, i) == 0.0,
              "excised radiation moment not zeroed");
  }
  Require(!mask(0, b.ks, b.js, b.is) &&
              !mask(0, b.ks, b.js, b.is + 1) &&
              radiation(0, 0, b.ks, b.js, b.is) > 0.0,
          "nonexcised radiation was disabled by regional selection");
  pack->pdyngr->SetTmunu(&driver, 1);
  Kokkos::fence();
  auto before_t = HostCopy(pack->ptmunu->u_tmunu);
  rad.SetTmunu(&driver, 1);
  Kokkos::fence();
  auto after_t = HostCopy(pack->ptmunu->u_tmunu);
  for (int i : {b.is + 2, b.is + 4}) {
    Require(after_t(0, Tmunu::I_Tmunu_E, b.ks, b.js, i) ==
                before_t(0, Tmunu::I_Tmunu_E, b.ks, b.js, i),
            "excised cell received radiation gravity");
  }
  Require(after_t(0, Tmunu::I_Tmunu_E, b.ks, b.js, b.is + 1) >
              before_t(0, Tmunu::I_Tmunu_E, b.ks, b.js, b.is + 1),
          "unexcised selected cell missed radiation gravity");

  auto before_m = HostCopy(pack->pmhd->u0);
  Sources(mesh, driver, mb, false);
  auto after_m = HostCopy(pack->pmhd->u0);
  auto after_r = HostCopy(rad.u0);
  for (int i : {b.is + 2, b.is + 4}) {
    for (int q : {IDN, IM1, IM2, IM3, IEN, IYF})
      Require(after_m(0, q, b.ks, b.js, i) ==
                  before_m(0, q, b.ks, b.js, i),
              "excised cell received material source");
    for (int v = 0; v < rad.nvarstot; ++v)
      Require(after_r(0, v, b.ks, b.js, i) == 0.0,
              "excised moment revived during source update");
  }
}

template <typename EOS>
void Run(const std::string &which, ParameterInput &pin, Mesh &mesh, Driver &driver,
         EOS &eos, Real mb) {
  auto *pack = mesh.pmb_pack;
  auto &rad = *pack->pradm1;
  const auto &b = mesh.mb_indcs;
  const Real trap = Trap(pin, mesh);
  Require(rad.nspecies == 4 && rad.nvars == M1_N_IDX + 1,
          "expected four-group number-evolving M1");
  const auto empty = Moments(mesh);
  Select(mesh, driver);
  Require(empty == Moments(mesh), "selector changed radiation moments");
  for (int n = 0; n < 5; ++n) CheckMembership(mesh, n, n == 1 || n == 2);
  auto mat = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pack->pmhd->u0);
  Require(mat(0, IDN, b.ks, b.js, b.is + 3) > trap,
          "D-vs-rest-density probe failed to densitize with sqrt(gamma)=2");
  if (which == "selection") {
    auto w = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pack->pmhd->w0);
    w(0, IDN, b.ks, b.js, b.is + 1) = trap * 0.5;
    w(0, IDN, b.ks, b.js, b.is) = trap * 1.5;
    Kokkos::deep_copy(pack->pmhd->w0, w);
    auto eos_device = eos;
    auto primitive = pack->pmhd->w0;
    par_for("ccsn_test_cross_pressure", DevExeSpace(), 0, pack->nmb_thispack - 1,
            b.ks, b.ke, b.js, b.je, b.is, b.is + 1,
            KOKKOS_LAMBDA(int m, int k, int j, int i) {
              Real fractions[MAX_SPECIES] = {0.0};
              fractions[0] = primitive(m, IYF, k, j, i);
              primitive(m, IPR, k, j, i) = eos_device.GetPressure(
                  primitive(m, IDN, k, j, i) / mb, 8.0, fractions);
            });
    pack->pdyngr->PrimToConInit(b.is, b.ie, b.js, b.je, b.ks, b.ke);
    // Neither empirical exclusion nor the feedback gates resample changed rho.
    CheckMembership(mesh, 0, false);
    CheckMembership(mesh, 1, true);
    // The post-hydro map uses the frozen selector, not the changed density.
    auto before_map = HostCopy(pack->pmhd->w0);
    auto before_temp = HostCopy(pack->pdyngr->temperature);
    Deleptonization(&mesh, mesh.dt);
    Kokkos::fence();
    auto after_map = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                         pack->pmhd->w0);
    auto after_temp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                          pack->pdyngr->temperature);
    Require(after_map(0, IYF, b.ks, b.js, b.is + 1) ==
                before_map(0, IYF, b.ks, b.js, b.is + 1) &&
            after_map(0, IPR, b.ks, b.js, b.is + 1) ==
                before_map(0, IPR, b.ks, b.js, b.is + 1) &&
            after_temp(0, 0, b.ks, b.js, b.is + 1) ==
                before_temp(0, 0, b.ks, b.js, b.is + 1) &&
            after_map(0, IYF, b.ks, b.js, b.is) <
                before_map(0, IYF, b.ks, b.js, b.is),
            "frozen empirical complement changed within the step");
    SeedRadiation(mesh, trap, mb);
    CheckTmunu(mesh, driver, true);
    Sources(mesh, driver, mb, false);
    const auto saved = Moments(mesh);
    Select(mesh, driver);
    Require(saved == Moments(mesh), "next producer changed radiation state");
    CheckMembership(mesh, 0, true);
    CheckMembership(mesh, 1, false);
    rad.SetCouplingRegionEnabled(true);
    Require(saved == Moments(mesh), "same-value regional setter cleared moments");
  } else if (which == "source_explicit" || which == "source_implicit") {
    SeedRadiation(mesh, trap, mb);
    Sources(mesh, driver, mb, false);
  } else if (which == "opacity") {
    PhysicalOpacities(mesh, driver, trap, mb);
  } else if (which == "excision") {
    CurrentExcision(mesh, driver, trap, mb);
  } else if (which == "tmunu") {
    SeedRadiation(mesh, trap, mb);
    CheckTmunu(mesh, driver, true);
    auto saved = Moments(mesh);
    rad.SetCouplingRegionEnabled(false);
    Require(saved == Moments(mesh), "regional disable cleared radiation moments");
    CheckTmunu(mesh, driver, false);
    rad.SetCouplingRegionEnabled(true);
    Require(saved == Moments(mesh), "regional reenable cleared radiation moments");
  } else if (which == "bounce") {
    SeedRadiation(mesh, trap, mb);
    const auto saved = Moments(mesh);
    auto w0 = HostCopy(pack->pmhd->w0);
    auto T0 = HostCopy(pack->pdyngr->temperature);
    Deleptonization(&mesh, mesh.dt);  // Real empirical map followed by bounce check.
    Kokkos::fence();
    Require(pin.GetBoolean("problem", "post_bounce") &&
            std::isfinite(pin.GetReal("problem", "t_bounce")) &&
            pin.GetReal("problem", "D_max_bounce") > 0,
            "forced bounce did not persist diagnostics");
    Require(rad.CouplingRegionEnabled() && saved == Moments(mesh),
            "bounce prematurely changed regional feedback or radiation");
    CheckMembership(mesh, 0, false);
    auto w1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pack->pmhd->w0);
    auto T1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                  pack->pdyngr->temperature);
    Require(w0(0, IYF, b.ks, b.js, b.is + 1) ==
                w1(0, IYF, b.ks, b.js, b.is + 1) &&
            w0(0, IPR, b.ks, b.js, b.is + 1) ==
                w1(0, IPR, b.ks, b.js, b.is + 1) &&
            T0(0, 0, b.ks, b.js, b.is + 1) ==
                T1(0, 0, b.ks, b.js, b.is + 1),
            "empirical map touched selected cell");
    Require(w1(0, IYF, b.ks, b.js, b.is) < w0(0, IYF, b.ks, b.js, b.is),
            "empirical complement did not evolve Ye");
    CheckTmunu(mesh, driver, true);  // Same detecting step still excludes exterior.
    Sources(mesh, driver, mb, false);  // The detecting split is still regional.
    const auto after_split = Moments(mesh);
    Select(mesh, driver);       // First following full-step boundary.
    Require(!rad.CouplingRegionEnabled() && after_split == Moments(mesh),
            "next-step global coupling failed or reset radiation");
    CheckTmunu(mesh, driver, false);
    SeedRadiation(mesh, trap, mb);
    Sources(mesh, driver, mb, true);  // Previously exterior now deposits material.
    auto y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pack->pmhd->w0);
    Deleptonization(&mesh, mesh.dt);
    Kokkos::fence();
    auto ypost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pack->pmhd->w0);
    Require(y(0, IYF, b.ks, b.js, b.is) == ypost(0, IYF, b.ks, b.js, b.is),
            "empirical map reactivated after bounce");
  } else {
    throw std::runtime_error("unknown native case");
  }
  std::cout << "PASS " << which << '\n';
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 5) return 2;  // input, EOS, progenitor, named case
  global_variable::my_rank = 0;
  global_variable::nranks = 1;
  Kokkos::initialize();
  int result = 0;
  try {
    {
      ParameterInput pin;
      std::ifstream input(argv[1]);
      Require(input.good(), "missing CCSN input skeleton");
      pin.LoadFromStream(input);
      pin.SetInteger("mesh", "nx1", 8);
      pin.SetInteger("mesh", "nx2", 1);
      pin.SetInteger("mesh", "nx3", 1);
      pin.SetInteger("mesh", "nghost", 2);
      pin.SetInteger("meshblock", "nx1", 8);
      pin.SetInteger("meshblock", "nx2", 1);
      pin.SetInteger("meshblock", "nx3", 1);
      pin.SetString("mhd", "table", argv[2]);
      pin.SetString("mhd", "units", "geometric_solar");
      pin.SetBoolean("mhd", "fixed", false);
      pin.SetInteger("mhd", "nscalars", 1);
      pin.SetString("mhd", "dyn_eos", "compose");
      pin.SetString("mhd", "dyn_error", "reset_floor");
      pin.SetString("problem", "pgen_name", "ccsn");
      pin.SetString("problem", "progenitor", argv[3]);
      pin.SetBoolean("problem", "interpolate_temp", true);
      pin.SetReal("problem", "Omega_0", 0.0);
      pin.SetReal("problem", "Omega_A", 1.0);
      pin.SetReal("problem", "B0_amp", 0.0);
      pin.SetReal("problem", "rho_trap", 1.0e12);
      pin.SetBoolean("problem", "prebounce_core_coupling", true);
      pin.SetBoolean("problem", "recoupling_approach", true);
      pin.SetBoolean("problem", "detect_bounce", true);
      pin.SetString("problem", "bounce_detection_method", "entropy");
      pin.SetString("problem", "deleptonization_method", "Liebendoerfer");
      pin.SetBoolean("problem", "equilibriate_post_bounce", false);
      pin.SetBoolean("problem", "conservative_bounce_projection", false);
      pin.SetBoolean("problem", "inject_equilibrium", false);
      pin.SetBoolean("problem", "update_conserved", false);
      pin.SetReal("problem", "bdm_rho_min", std::string(argv[4]) == "bounce" ? 0.0 : 1e99);
      pin.SetReal("problem", "bdm_spb_max", std::string(argv[4]) == "bounce" ? -1e99 : 3.0);
      pin.SetReal("problem", "bdm_r_max", 1e99);
      pin.SetBoolean("bns_nurates", "use_partial_equilibrium", false);
      pin.SetBoolean("bns_nurates", "use_abs_em", true);
      pin.SetInteger("bns_nurates", "eq_warmup_cycles", 0);
      pin.SetString("radiation_m1", "opacity_type", "bns-nurates");
      pin.SetBoolean("radiation_m1", "matter_sources", true);
      pin.SetBoolean("radiation_m1", "backreact", true);
      pin.SetBoolean("radiation_m1", "backreact_tmunu", true);
      pin.SetBoolean("radiation_m1", "gr_sources", false);
      pin.SetBoolean("radiation_m1", "theta_limiter", false);
      pin.SetString("radiation_m1", "closure_fun", "eddington");
      pin.SetString("radiation_m1", "src_update",
                    std::string(argv[4]) == "source_implicit" ? "implicit" : "explicit");
      Mesh mesh(&pin);
      mesh.BuildTreeFromScratch(&pin);
      mesh.AddCoordinatesAndPhysics(&pin);
      mesh.pgen = std::make_unique<ProblemGenerator>(&pin, &mesh);
      Kokkos::Timer timer;
      Driver driver(&pin, &mesh, -1.0, &timer);
      driver.InitBoundaryValuesAndPrimitives(&mesh);
      mesh.dt = 1e-4;
      mesh.ncycle = 1;
      auto *base = mesh.pmb_pack->pdyngr;
      auto *normal = dynamic_cast<dyngr::DynGRMHDPS<
          Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor> *>(base);
      auto *nqt = dynamic_cast<dyngr::DynGRMHDPS<
          Primitive::EOSCompOSE<Primitive::NQTLogs>, Primitive::ResetFloor> *>(base);
      Require(normal != nullptr || nqt != nullptr, "unsupported active CompOSE specialization");
      auto execute = [&](auto *hydro) {
        auto &eos = hydro->eos.ps.GetEOSMutable();
        ControlledState(mesh, driver, eos, Trap(pin, mesh));
        std::cout << "backend=" << DevExeSpace::name() << " sizeof(Real)=" << sizeof(Real)
                  << " case=" << argv[4] << '\n';
        Require(sizeof(Real) == 8, "native exchange requires full precision");
        Run(argv[4], pin, mesh, driver, eos, eos.GetBaryonMass());
      };
      if (normal) execute(normal);
      else execute(nqt);
    }
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << argv[4] << ": " << error.what() << '\n';
    result = 1;
  }
  Kokkos::finalize();
  return result;
}
