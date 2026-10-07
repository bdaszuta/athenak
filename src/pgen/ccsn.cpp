//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ccsn.cpp
//  \brief Problem generator for CCSN simulations. Only works when ADM is
//  enabled.
//         Stellar profiles from Woosley:2002zz and similar.
//         Compile with '-D PROBLEM=ccsn'.

#include <math.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/eos.hpp"
#include "eos/primitive-solver/eos_compose.hpp"
#include "eos/primitive-solver/eos_hybrid.hpp"
#include "eos/primitive-solver/piecewise_polytrope.hpp"
#include "eos/primitive-solver/reset_floor.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "eos/primitive_solver_hyd.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mhd/mhd.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_macro.hpp"
#include "z4c/z4c.hpp"

#include "pgen/ccsn.hpp"
#include "pgen/ccsn_m1_projection.hpp"

// The sibling QOL port owns the paired history/reducer capacity increase.
// Refuse an unsupported combination instead of overrunning the history arrays.
static_assert(NHISTORY_VARIABLES >= 32, "CCSN requires the QOL history capacity");

//----------------------------------------------------------------------------------------
// Global state (anonymous namespace)

namespace {

// Species index for Ye
static const int IYE = 0;

// Refinement parameters
Real delta_min_m;
Real delta_max_m;

// Refinement strategy
enum class RefinementMethod {none, MaxMassInCell};
RefinementMethod refm;

} // namespace

//----------------------------------------------------------------------------------------
//! \fn StellarProfile::StellarProfile
//  \brief Read profile file, compute Phi(r) from rho(r) forward integration,
//         store metric components, copy to Kokkos::View for GPU access.

StellarProfile::StellarProfile(ParameterInput *pin, bool interpolate_Phi)
    : siz(0) {
  const std::string fname = pin->GetString("problem", "progenitor");
  std::ifstream in(fname);
  auto fail = [&](int line_number, const std::string &reason) {
    std::cerr << "### FATAL: stellar profile " << fname << ":" << line_number
              << ": " << reason << std::endl;
    std::exit(EXIT_FAILURE);
  };
  if (!in.is_open()) fail(0, "cannot open file");

  // Geometric-solar r, mass, velocity, density and potential; T in MeV,
  // Ye and entropy per baryon dimensionless. File columns are
  // r mass vel rho T Ye P e s om ab phi [phiZ]. All stored channels,
  // including unused metric/optional channels, have defined host values.
  std::vector<Real> h_pr;
  std::vector<std::array<Real, num_vars>> h_pvars;
  std::string line;
  int line_number = 0;
  while (std::getline(in, line)) {
    ++line_number;
    line = line.substr(0, line.find('#'));
    std::istringstream row(line);
    row >> std::ws;
    if (row.eof()) continue;
    Real rad;
    std::array<Real, num_vars> values{};
    values[gxx] = 1.0;  // unused when the ICs interpolate phi directly
    if (!(row >> rad)) fail(line_number, "invalid radius");
    for (int v = mass; v <= phi; ++v) {
      if (!(row >> values[v]) || !std::isfinite(values[v])) {
        fail(line_number, "expected finite columns r through phi");
      }
    }
    row >> std::ws;
    if (!row.eof()) {
      if (!(row >> values[phiZ]) || !std::isfinite(values[phiZ])) {
        fail(line_number, "invalid optional phiZ column");
      }
      row >> std::ws;
      if (!row.eof()) fail(line_number, "unexpected trailing column or text");
    }
    if (!std::isfinite(rad) || rad < 0.0 ||
        (!h_pr.empty() && rad <= h_pr.back())) {
      fail(line_number, "radii must be finite, nonnegative and strictly increasing");
    }
    h_pr.push_back(rad);
    h_pvars.push_back(values);
  }
  if (!in.eof()) fail(line_number, "read error");
  if (h_pr.size() < 2 || h_pr.size() > std::numeric_limits<int>::max()) {
    fail(line_number, "profile requires at least two representable radial points");
  }
  siz = static_cast<int>(h_pr.size());

  // Retain the donor's forward Newtonian potential integration and outer shift.
  // In the other mode, the tabulated phi column supplies the metric in SetCCSNIC.
  if (!interpolate_Phi) {
    std::vector<Real> h_Phi(siz, 0.0);
    Real massi = 4.0 / 3.0 * M_PI * pow(h_pr[1], 3) * h_pvars[0][rho];
    h_Phi[1] = massi / h_pr[1];
    for (int i = 2; i < siz; ++i) {
      Real dr = h_pr[i] - h_pr[i - 1];
      massi += 4.0 * M_PI * pow(h_pr[i - 1], 2) * h_pvars[i - 1][rho] * dr;
      h_Phi[i] = h_Phi[i - 1] + massi / pow(h_pr[i], 2) * dr;
    }
    const Real dPhi = h_Phi[siz - 1] - massi / h_pr[siz - 1];
    for (int i = 0; i < siz; ++i) {
      h_Phi[i] -= dPhi;
      if (!(1.0 + 2.0 * h_Phi[i] > 0.0) ||
          !(1.0 - 2.0 * h_Phi[i] > 0.0) || !std::isfinite(h_Phi[i])) {
        fail(0, "integrated potential does not define a finite positive metric");
      }
      h_pvars[i][alp] = sqrt(1.0 + 2.0 * h_Phi[i]);
      h_pvars[i][gxx] = 1.0 - 2.0 * h_Phi[i];
    }
  }

  d_pr = Kokkos::View<Real *>("StellarProfile::pr", siz);
  d_pvars = Kokkos::View<Real **>("StellarProfile::pvars", num_vars, siz);
  auto h_pr_mirror = Kokkos::create_mirror_view(d_pr);
  auto h_pvars_mirror = Kokkos::create_mirror_view(d_pvars);
  for (int i = 0; i < siz; ++i) {
    h_pr_mirror(i) = h_pr[i];
    for (int v = 0; v < num_vars; ++v) h_pvars_mirror(v, i) = h_pvars[i][v];
  }
  Kokkos::deep_copy(d_pr, h_pr_mirror);
  Kokkos::deep_copy(d_pvars, h_pvars_mirror);
}

// Bounce diagnostics for history output.
static Real t_bounce_val = 0.0;
static Real D_max_bounce_val = 0.0;

//----------------------------------------------------------------------------------------
//! \fn HistoryMaxT
//  \brief EOS-templated kernel for max temperature. Separate from the main
//         reduce so the original 4-variable path is never touched by EOS
//         dispatch failures.

template <class EOSPolicy, class ErrorPolicy>
static void HistoryMaxT(Mesh *pm, Real &temp_max) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto *pdg =
      dynamic_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmbp->pdyngr);
  if (pdg == nullptr)
    return;
  auto &temperature_ = pdg->temperature;
  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  int nmb = pmbp->nmb_thispack;
  const int nmkji = nmb * nx3 * nx2 * nx1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;

  Real tmax = 0.0;
  Kokkos::parallel_reduce(
      "T_hist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, Real &mb_temp) {
        int m = idx / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / nx1;
        int i = (idx - m * nkji - k * nji - j * nx1) + is;
        k += ks;
        j += js;
        Real rho = w0_(m, IDN, k, j, i);
        if (rho > 0.0) {
          mb_temp = fmax(mb_temp, temperature_(m, 0, k, j, i));
        }
      },
      Kokkos::Max<Real>(tmax));

  temp_max = tmax;
}

//----------------------------------------------------------------------------------------
//! \fn History
//! \brief 32 outputs: CCSN, GRA-standard M1 extrema, and field diagnostics.
//
void History(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;

  pdata->nhist = 32;
  pdata->label[0] = "max_rho";
  pdata->label[1] = "min_alpha";
  pdata->label[2] = "max_press";
  pdata->label[3] = "max_MassInCell";
  pdata->label[4] = "max_level";
  pdata->label[5] = "max_T";
  pdata->label[6] = "max_MassPerMB";
  pdata->label[7] = "t_bounce";
  pdata->label[8] = "D_max_bounce";
  for (int nuidx = 0; nuidx < 4; ++nuidx) {
    const int ihist = 9 + 4 * nuidx;
    const std::string gs = "0" + std::to_string(nuidx);
    pdata->label[ihist] = "max_sc_nG_" + gs;
    pdata->label[ihist + 1] = "max_sc_E_" + gs;
    pdata->label[ihist + 2] = "min_sc_nG_" + gs;
    pdata->label[ihist + 3] = "min_sc_E_" + gs;
  }
  pdata->label[25] = "max_B2";
  pdata->label[26] = "max_b2";
  pdata->label[27] = "div_B";
  pdata->label[28] = "div_B_norm2";
  pdata->label[29] = "E_B";
  pdata->label[30] = "Etau_B";
  pdata->label[31] = "bphisq";

  auto &w0_ = pm->pmb_pack->pmhd->w0;
  auto &adm = pm->pmb_pack->padm->adm;
  auto &u0_ = pm->pmb_pack->pmhd->u0;
  auto &size = pm->pmb_pack->pmb->mb_size;

  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  int nmb = pmbp->nmb_thispack;
  const int nmkji = nmb * nx3 * nx2 * nx1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;

  // --- 4-variable reduce (original working code, no EOS dependency) ---
  Real rho_max = 0.0;
  Real alpha_min = std::numeric_limits<Real>::max();
  Real press_max = 0.0;
  Real mass_max = 0.0;

  Kokkos::parallel_reduce(
      "CCSNHist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, Real &mb_rho, Real &mb_alp, Real &mb_press,
                    Real &mb_mass) {
        int m = idx / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / nx1;
        int i = (idx - m * nkji - k * nji - j * nx1) + is;
        k += ks;
        j += js;

        mb_rho = fmax(mb_rho, w0_(m, IDN, k, j, i));
        mb_alp = fmin(mb_alp, adm.alpha(m, k, j, i));
        mb_press = fmax(mb_press, w0_(m, IPR, k, j, i));

        Real vol = size.d_view(m).dx1 * size.d_view(m).dx2 * size.d_view(m).dx3;
        Real mcell = vol * u0_(m, IDN, k, j, i);
        mb_mass = fmax(mb_mass, mcell);
      },
      Kokkos::Max<Real>(rho_max), Kokkos::Min<Real>(alpha_min),
      Kokkos::Max<Real>(press_max), Kokkos::Max<Real>(mass_max));

  // --- max_T via EOS template dispatch ---
  Real temp_max = 0.0;
  using namespace Primitive;
  switch (pmbp->pdyngr->eos_policy) {
    case DynGRMHD_EOS::eos_ideal:
      HistoryMaxT<IdealGas, ResetFloor>(pm, temp_max);
      break;
    case DynGRMHD_EOS::eos_piecewise_poly:
      HistoryMaxT<PiecewisePolytrope, ResetFloor>(pm, temp_max);
      break;
    case DynGRMHD_EOS::eos_compose:
      HistoryMaxT<EOSCompOSE<NQTLogs>, ResetFloor>(pm, temp_max);
      HistoryMaxT<EOSCompOSE<NormalLogs>, ResetFloor>(pm, temp_max);
      break;
    case DynGRMHD_EOS::eos_hybrid:
      HistoryMaxT<EOSHybrid<NQTLogs>, ResetFloor>(pm, temp_max);
      HistoryMaxT<EOSHybrid<NormalLogs>, ResetFloor>(pm, temp_max);
      break;
  }

  // --- max_MassPerMB via team reduce ---
  Kokkos::View<Real *, DevExeSpace> mb_mass_dev("mb_mass_dev", nmb);
  par_for_outer(
      "hist_mb_sum", DevExeSpace(), 0, 0, 0, nmb - 1,
      KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
        Real team_sum = 0.0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(tmember, nkji),
            [=](const int idx, Real &lsum) {
              int k = idx / nji;
              int j = (idx - k * nji) / nx1;
              int i = (idx - k * nji - j * nx1) + is;
              j += js;
              k += ks;
              Real vol = size.d_view(m).dx1 * size.d_view(m).dx2 *
                         size.d_view(m).dx3;
              lsum += vol * u0_(m, IDN, k, j, i);
            },
            Kokkos::Sum<Real>(team_sum));
        mb_mass_dev(m) = team_sum;
      });
  auto h_mb_mass = Kokkos::create_mirror_view(mb_mass_dev);
  Kokkos::deep_copy(h_mb_mass, mb_mass_dev);
  Real max_mb_mass = 0.0;
  for (int m = 0; m < nmb; ++m) {
    max_mb_mass = fmax(max_mb_mass, h_mb_mass(m));
  }

  Real max_sc_nG[4]{};
  Real max_sc_E[4]{};
  Real min_sc_nG[4]{};
  Real min_sc_E[4]{};
  if (pmbp->pradm1 != nullptr) {
    auto &m1_u0 = pmbp->pradm1->u0;
    const int m1_nvars = pmbp->pradm1->nvars;
    const int m1_nspecies = pmbp->pradm1->nspecies;
    for (int nuidx = 0;
         nuidx < 4 && nuidx < m1_nspecies && m1_nvars > M1_N_IDX;
         ++nuidx) {
      max_sc_nG[nuidx] = -std::numeric_limits<Real>::infinity();
      max_sc_E[nuidx] = -std::numeric_limits<Real>::infinity();
      min_sc_nG[nuidx] = std::numeric_limits<Real>::infinity();
      min_sc_E[nuidx] = std::numeric_limits<Real>::infinity();
      Kokkos::parallel_reduce(
          "CCSNHistM1", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
          KOKKOS_LAMBDA(const int idx, Real &mb_max_nG, Real &mb_max_E,
                        Real &mb_min_nG, Real &mb_min_E) {
            int m = idx / nkji;
            int k = (idx - m * nkji) / nji;
            int j = (idx - m * nkji - k * nji) / nx1;
            int i = (idx - m * nkji - k * nji - j * nx1) + is;
            k += ks;
            j += js;
            const Real sc_nG =
                m1_u0(m, nuidx * m1_nvars + M1_N_IDX, k, j, i);
            const Real sc_E =
                m1_u0(m, nuidx * m1_nvars + M1_E_IDX, k, j, i);
            mb_max_nG = fmax(mb_max_nG, sc_nG);
            mb_max_E = fmax(mb_max_E, sc_E);
            mb_min_nG = fmin(mb_min_nG, sc_nG);
            mb_min_E = fmin(mb_min_E, sc_E);
          },
          Kokkos::Max<Real>(max_sc_nG[nuidx]), Kokkos::Max<Real>(max_sc_E[nuidx]),
          Kokkos::Min<Real>(min_sc_nG[nuidx]), Kokkos::Min<Real>(min_sc_E[nuidx]));
    }
  }

  auto &b0_ = pmbp->pmhd->b0;
  auto &bcc0_ = pmbp->pmhd->bcc0;
  Real B2_max = 0.0;
  Real b2_max = 0.0;
  Real div_B = 0.0;
  Real div_B_norm2 = 0.0;
  Real E_B = 0.0;
  Real Etau_B = 0.0;
  Real bphisq = 0.0;
  Kokkos::parallel_reduce(
      "CCSNHistB", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, Real &mb_B2, Real &mb_b2, Real &mb_div_B,
                    Real &mb_div_B_norm2, Real &mb_E_B, Real &mb_Etau_B,
                    Real &mb_bphisq) {
        int m = idx / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / nx1;
        int i = (idx - m * nkji - k * nji - j * nx1) + is;
        k += ks;
        j += js;

        const Real dx1 = size.d_view(m).dx1;
        const Real dx2 = size.d_view(m).dx2;
        const Real dx3 = size.d_view(m).dx3;
        const Real vol = dx1 * dx2 * dx3;
        const Real x = CellCenterX(i - is, nx1, size.d_view(m).x1min,
                                   size.d_view(m).x1max);
        const Real y = CellCenterX(j - js, nx2, size.d_view(m).x2min,
                                   size.d_view(m).x2max);

        const Real detg = adm::SpatialDet(
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
        const Real sqrt_det_gamma = sqrt(detg);
        const Real oo_sqrt_det_gamma = 1.0 / sqrt_det_gamma;

        Real Wsq = 1.0;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            Wsq += w0_(m, IVX + a, k, j, i) * w0_(m, IVX + b, k, j, i) *
                   adm.g_dd(m, a, b, k, j, i);
          }
        }
        const Real W = sqrt(Wsq);

        Real B2 = 0.0;
        Real Bv = 0.0;
        Real beta_d[3] = {0.0};
        Real b_u[3] = {0.0};
        Real b_d[3] = {0.0};
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            B2 += bcc0_(m, a, k, j, i) * adm.g_dd(m, a, b, k, j, i) *
                  bcc0_(m, b, k, j, i);
            Bv += bcc0_(m, a, k, j, i) * adm.g_dd(m, a, b, k, j, i) *
                  w0_(m, IVX + b, k, j, i) / W;
            beta_d[a] += adm.g_dd(m, a, b, k, j, i) * adm.beta_u(m, b, k, j, i);
          }
        }
        const Real alpha_b0 = W * Bv * oo_sqrt_det_gamma;
        const Real b0 = alpha_b0 / adm.alpha(m, k, j, i);
        const Real b2 = (SQR(alpha_b0) + SQR(oo_sqrt_det_gamma) * B2) / Wsq;
        for (int a = 0; a < 3; ++a) {
          const Real vtil_u = w0_(m, IVX + a, k, j, i) / W -
                              adm.beta_u(m, a, k, j, i) / adm.alpha(m, k, j, i);
          b_u[a] = (bcc0_(m, a, k, j, i) * oo_sqrt_det_gamma +
                    adm.alpha(m, k, j, i) * b0 * W * vtil_u) /
                   W;
        }
        for (int a = 0; a < 3; ++a) {
          b_d[a] = beta_d[a] * b0;
          for (int b = 0; b < 3; ++b) {
            b_d[a] += adm.g_dd(m, a, b, k, j, i) * b_u[b];
          }
        }

        const Real divB =
            (b0_.x1f(m, k, j, i + 1) - b0_.x1f(m, k, j, i)) / dx1 +
            (b0_.x2f(m, k, j + 1, i) - b0_.x2f(m, k, j, i)) / dx2 +
            (b0_.x3f(m, k + 1, j, i) - b0_.x3f(m, k, j, i)) / dx3;
        const Real rho_cyl = sqrt(SQR(x) + SQR(y));
        const Real bphi_u = (y * b_u[0] - x * b_u[1]) / rho_cyl;
        const Real bphi_d = (y * b_d[0] - x * b_d[1]) / rho_cyl;

        mb_B2 = fmax(mb_B2, fabs(B2));
        mb_b2 = fmax(mb_b2, fabs(b2));
        mb_div_B += divB * vol;
        mb_div_B_norm2 += SQR(divB) * vol;
        mb_E_B += sqrt_det_gamma * W * b2 * vol;
        mb_Etau_B += sqrt_det_gamma *
                     (b2 * (Wsq - 0.5) - SQR(alpha_b0)) * vol;
        mb_bphisq += sqrt_det_gamma * bphi_d * bphi_u * vol;
      },
      Kokkos::Max<Real>(B2_max), Kokkos::Max<Real>(b2_max),
      Kokkos::Sum<Real>(div_B), Kokkos::Sum<Real>(div_B_norm2),
      Kokkos::Sum<Real>(E_B), Kokkos::Sum<Real>(Etau_B),
      Kokkos::Sum<Real>(bphisq));
  E_B *= 0.5;

  // --- bundled MPI reduction ---
#if MPI_PARALLEL_ENABLED
  // Pack all MAX reductions: rho, press, mass_cell, temp, mass_mb
  Real max_arr[5] = {rho_max, press_max, mass_max, temp_max, max_mb_mass};
  Real m1_max_arr[8];
  Real m1_min_arr[8];
  for (int nuidx = 0; nuidx < 4; ++nuidx) {
    m1_max_arr[2 * nuidx] = max_sc_nG[nuidx];
    m1_max_arr[2 * nuidx + 1] = max_sc_E[nuidx];
    m1_min_arr[2 * nuidx] = min_sc_nG[nuidx];
    m1_min_arr[2 * nuidx + 1] = min_sc_E[nuidx];
  }
  Real field_max_arr[2] = {B2_max, b2_max};
  Real field_sum_arr[5] = {div_B, div_B_norm2, E_B, Etau_B, bphisq};
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, max_arr, 5, MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, m1_max_arr, 8, MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, m1_min_arr, 8, MPI_ATHENA_REAL, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, field_max_arr, 2, MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, field_sum_arr, 5, MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(max_arr, max_arr, 5, MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&alpha_min, &alpha_min, 1, MPI_ATHENA_REAL, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(m1_max_arr, m1_max_arr, 8, MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(m1_min_arr, m1_min_arr, 8, MPI_ATHENA_REAL, MPI_MIN, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(field_max_arr, field_max_arr, 2, MPI_ATHENA_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(field_sum_arr, field_sum_arr, 5, MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
  }
  rho_max     = max_arr[0];
  press_max   = max_arr[1];
  mass_max    = max_arr[2];
  temp_max    = max_arr[3];
  max_mb_mass = max_arr[4];
  for (int nuidx = 0; nuidx < 4; ++nuidx) {
    max_sc_nG[nuidx] = m1_max_arr[2 * nuidx];
    max_sc_E[nuidx] = m1_max_arr[2 * nuidx + 1];
    min_sc_nG[nuidx] = m1_min_arr[2 * nuidx];
    min_sc_E[nuidx] = m1_min_arr[2 * nuidx + 1];
  }
  B2_max = field_max_arr[0];
  b2_max = field_max_arr[1];
  div_B = field_sum_arr[0];
  div_B_norm2 = field_sum_arr[1];
  E_B = field_sum_arr[2];
  Etau_B = field_sum_arr[3];
  bphisq = field_sum_arr[4];
#endif

  // --- write hdata ---
  pdata->hdata[0] = rho_max;
  pdata->hdata[1] = alpha_min;
  pdata->hdata[2] = press_max;
  pdata->hdata[3] = mass_max;
  // --- max_level: dynamic active max from leaf-block list ---
  int local_max_lev = 0;
  int my_rank = global_variable::my_rank;
  int gid_start = pm->gids_eachrank[my_rank];
  int nmb_local = pm->nmb_eachrank[my_rank];
  for (int i = 0; i < nmb_local; ++i) {
    local_max_lev = std::max(local_max_lev,
                             pm->lloc_eachmb[gid_start + i].level);
  }
  int global_max_lev = local_max_lev;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local_max_lev, &global_max_lev, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  pdata->hdata[4] = static_cast<Real>(global_max_lev);
  pdata->hdata[5] = temp_max;
  pdata->hdata[6] = max_mb_mass;

  // --- bounce diagnostics ---
  pdata->hdata[7] = t_bounce_val;
  pdata->hdata[8] = D_max_bounce_val;
  for (int nuidx = 0; nuidx < 4; ++nuidx) {
    const int ihist = 9 + 4 * nuidx;
    pdata->hdata[ihist] = max_sc_nG[nuidx];
    pdata->hdata[ihist + 1] = max_sc_E[nuidx];
    pdata->hdata[ihist + 2] = min_sc_nG[nuidx];
    pdata->hdata[ihist + 3] = min_sc_E[nuidx];
  }
  pdata->hdata[25] = B2_max;
  pdata->hdata[26] = b2_max;
  pdata->hdata[27] = div_B;
  pdata->hdata[28] = div_B_norm2;
  pdata->hdata[29] = E_B;
  pdata->hdata[30] = Etau_B;
  pdata->hdata[31] = bphisq;

#if MPI_PARALLEL_ENABLED
  // The history output framework (outputs/history.cpp) performs an
  // unconditional MPI_SUM over hdata across all ranks. This function has
  // already reduced the correct global max/min values onto rank 0, so zero
  // out non-root ranks' contributions to prevent the framework's sum from
  // double-counting each rank's local values.
  if (global_variable::my_rank != 0) {
    for (int n = 0; n < pdata->nhist; ++n) {
      pdata->hdata[n] = 0.0;
    }
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn RefinementCondition
//  \brief Refine/derefine by the largest coordinate-cell baryon mass,
//  dx1*dx2*dx3*u0(IDN), with the configured hysteresis thresholds.

void RefinementCondition(MeshBlockPack *pmbp) {
  if (refm == RefinementMethod::none)
    return;

  Mesh *pmesh = pmbp->pmesh;
  int nmb = pmbp->nmb_thispack;
  int mbs = pmesh->gids_eachrank[global_variable::my_rank];
  auto &refine_flag = pmesh->pmr->refine_flag;
  auto &indcs = pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  auto &u0 = pmbp->pmhd->u0;
  auto &size = pmbp->pmb->mb_size;
  const Real dmax = delta_max_m;
  const Real dmin = delta_min_m;

  if (refm == RefinementMethod::MaxMassInCell) {
    par_for_outer(
        "amr_mass_cell", DevExeSpace(), 0, 0, 0, nmb - 1,
        KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
          Real team_mmax = 0.0;
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(tmember, nkji),
              [=](const int idx, Real &mmax) {
                int k = idx / nji;
                int j = (idx - k * nji) / nx1;
                int i = (idx - k * nji - j * nx1) + is;
                j += js;
                k += ks;
                Real vol = size.d_view(m).dx1 * size.d_view(m).dx2 *
                           size.d_view(m).dx3;
                Real mcell = vol * u0(m, IDN, k, j, i);
                mmax = fmax(mcell, mmax);
              },
              Kokkos::Max<Real>(team_mmax));
          if (team_mmax > dmax) {
            refine_flag.d_view(m + mbs) = 1;
          } else if (team_mmax < dmin) {
            refine_flag.d_view(m + mbs) = -1;
          }
        });
  }

  refine_flag.modify_device();
  refine_flag.sync_host();
}

//----------------------------------------------------------------------------------------
// Full-step CompOSE deleptonization
//----------------------------------------------------------------------------------------

// Global state for deleptonization
static bool delept_active = false;
static bool prebounce_core_coupling = false;
static bool delept_update_entropy = true;
static Real delept_E_nu_avg;
static Real delept_rho_trap;
static Real delept_log10_rho1, delept_log10_rho2, delept_Ye_2, delept_Ye_c,
    delept_Ye_H;
static ParameterInput *delept_pin = nullptr;

// Called by the pgen-local post-hydro task, once per full timestep.
void Deleptonization(Mesh *pm, const Real dt);
static void BounceCheck(Mesh *pm, ParameterInput *pin);

//----------------------------------------------------------------------------------------
//! \fn SetCCSNIC
//! \brief Templated IC kernel: interpolate profile onto grid with EOS access.
//!        Interpolate density linearly and optionally obtain pressure from
//!        interpolated temperature. The native floor/P2C pass follows.

template <class EOSPolicy, class ErrorPolicy>
static void SetCCSNIC(
    MeshBlockPack *pmbp,
    const StellarProfile &star,
    Real rho_cut,
    Real Omega_0, Real Omega_A,
    bool interpolate_Phi, bool interpolate_temp, bool interpolate_offset_r) {

  auto *pdg = dynamic_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(
      pmbp->pdyngr);
  if (pdg == nullptr) return;

  auto eos = pdg->eos.ps.GetEOS();
  Real mb = eos.GetBaryonMass();
  Real ye_max = eos.GetMaximumSpeciesFraction(0); // IYE = 0

  auto d_pr = star.d_pr;
  auto d_pvars = star.d_pvars;
  int star_siz = star.siz;

  auto &w0_ = pmbp->pmhd->w0;
  int nvars_ = pmbp->pmhd->nmhd;
  int nscal_ = pmbp->pmhd->nscalars;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int ng = indcs.ng;
  int n1 = indcs.nx1 + 2 * ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &size = pmbp->pmb->mb_size;
  auto &adm = pmbp->padm->adm;

  par_for("pgen_ccsn", DevExeSpace(), 0, nmb1, 0, (n3 - 1), 0, (n2 - 1),
          0, (n1 - 1), KOKKOS_LAMBDA(int m, int k, int j, int i) {

    // --- cell-center computation ---
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i - is, indcs.nx1, x1min, x1max);
    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j - js, indcs.nx2, x2min, x2max);
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k - ks, indcs.nx3, x3min, x3max);
    Real r = sqrt(SQR(x1v) + SQR(x2v) + SQR(x3v));
    Real s_cyl = sqrt(SQR(x1v) + SQR(x2v));

    // --- metric ---
    Real alp, A;
    if (interpolate_Phi) {
      Real const Phi = StellarProfile::Eval(d_pr, d_pvars,
          StellarProfile::phi, r, star_siz, false, interpolate_offset_r);
      A = 1.0 - 2.0 * Phi;
      alp = sqrt(1.0 + 2.0 * Phi);
    } else {
      alp = StellarProfile::Eval(d_pr, d_pvars,
          StellarProfile::alp, r, star_siz, false, interpolate_offset_r);
      A = StellarProfile::Eval(d_pr, d_pvars,
          StellarProfile::gxx, r, star_siz, false, interpolate_offset_r);
    }

    // --- hydro: rho (linear interpolation, matching GRA) ---
    Real rho_1d = StellarProfile::Eval(d_pr, d_pvars,
        StellarProfile::rho, r, star_siz, false, interpolate_offset_r);

    // --- Ye with EOS-query clamp ---
    Real ye = StellarProfile::Eval(d_pr, d_pvars,
        StellarProfile::ye, r, star_siz, false, interpolate_offset_r);
    ye = fmin(ye, ye_max);

    // --- pressure: GRA-equivalent per-cell EOS computation ---
    Real press;
    if (interpolate_temp) {
      Real T = StellarProfile::Eval(d_pr, d_pvars,
          StellarProfile::temp, r, star_siz, false, interpolate_offset_r);
      Real n = rho_1d / mb;
      Real Y[MAX_SPECIES] = {0.};
      Y[0] = ye;
      press = eos.GetPressure(n, T, Y);
    } else {
      press = StellarProfile::Eval(d_pr, d_pvars,
          StellarProfile::press, r, star_siz, false, interpolate_offset_r);
    }

    // --- velocity ---
    Real vr = StellarProfile::Eval(d_pr, d_pvars,
        StellarProfile::vel, r, star_siz, false, interpolate_offset_r);
    Real vx = vr * x1v / r;
    Real vy = vr * x2v / r;
    Real vz = vr * x3v / r;

    // --- rotation ---
    if (Omega_0 > 0.0 && Omega_A > 0.0) {
      Real const Omega = OmegaLaw(s_cyl, Omega_0, Omega_A);
      vx -= Omega * x2v;
      vy += Omega * x1v;
    }

    // --- atmosphere cutoff (set rho=0, floor pass handles the rest) ---
    if (rho_cut > 0.0 && rho_1d < rho_cut) {
      rho_1d = 0.0;
    }

    // --- set primitives ---
    Real W = 1.0 / sqrt(1.0 - A * (SQR(vx) + SQR(vy) + SQR(vz)));
    w0_(m, IDN, k, j, i) = rho_1d;
    w0_(m, IPR, k, j, i) = press;
    w0_(m, IVX, k, j, i) = W * vx;
    w0_(m, IVY, k, j, i) = W * vy;
    w0_(m, IVZ, k, j, i) = W * vz;
    if (nscal_ >= 1) w0_(m, nvars_, k, j, i) = ye;

    // --- set ADM variables ---
    adm.alpha(m, k, j, i) = alp;
    adm.g_dd(m, 0, 0, k, j, i) = A;
    adm.g_dd(m, 0, 1, k, j, i) = 0.0;
    adm.g_dd(m, 0, 2, k, j, i) = 0.0;
    adm.g_dd(m, 1, 1, k, j, i) = A;
    adm.g_dd(m, 1, 2, k, j, i) = 0.0;
    adm.g_dd(m, 2, 2, k, j, i) = A;
    Real det = adm::SpatialDet(A, 0.0, 0.0, A, 0.0, A);
    adm.psi4(m, k, j, i) = pow(det, 1.0 / 3.0);
    adm.beta_u(m, 0, k, j, i) = 0.0;
    adm.beta_u(m, 1, k, j, i) = 0.0;
    adm.beta_u(m, 2, k, j, i) = 0.0;
    adm.vK_dd(m, 0, 0, k, j, i) = 0.0;
    adm.vK_dd(m, 0, 1, k, j, i) = 0.0;
    adm.vK_dd(m, 0, 2, k, j, i) = 0.0;
    adm.vK_dd(m, 1, 1, k, j, i) = 0.0;
    adm.vK_dd(m, 1, 2, k, j, i) = 0.0;
    adm.vK_dd(m, 2, 2, k, j, i) = 0.0;
  });
}

//----------------------------------------------------------------------------------------
//! \fn ApplyPrimitiveFloorsIC
//! \brief Apply primitive floors to IC state (GRA: ApplyPrimitiveFloors
//!        before PrimitiveToConserved).  Handles atmosphere cells where
//!        the profile cutoff set rho = 0.

template <class EOSPolicy, class ErrorPolicy>
static void ApplyPrimitiveFloorsIC(MeshBlockPack *pmbp) {
  auto *pdg = dynamic_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(
      pmbp->pdyngr);
  if (pdg == nullptr) return;

  auto eos = pdg->eos.ps.GetEOS();
  Real mb = eos.GetBaryonMass();
  auto &w0_ = pmbp->pmhd->w0;
  int nmhd = pmbp->pmhd->nmhd;
  int nscal = pmbp->pmhd->nscalars;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;

  par_for("floor_ic", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real rho = w0_(m, IDN, k, j, i);
    Real n = rho / mb;
    Real Wvu[3] = {w0_(m, IVX, k, j, i),
                   w0_(m, IVY, k, j, i),
                   w0_(m, IVZ, k, j, i)};
    Real P = w0_(m, IPR, k, j, i);
    Real Y[MAX_SPECIES] = {0.};
    if (nscal >= 1) Y[0] = w0_(m, nmhd, k, j, i);

    Real T;
    if (n <= 0.0) {
      T = 0.0;
    } else {
      T = eos.GetTemperatureFromP(n, P, Y);
    }

    if (eos.ApplyPrimitiveFloor(n, Wvu, P, T, Y)) {
      w0_(m, IDN, k, j, i) = n * mb;
      w0_(m, IVX, k, j, i) = Wvu[0];
      w0_(m, IVY, k, j, i) = Wvu[1];
      w0_(m, IVZ, k, j, i) = Wvu[2];
      w0_(m, IPR, k, j, i) = P;
      if (nscal >= 1) w0_(m, nmhd, k, j, i) = Y[0];
    }
  });
  Kokkos::fence();
}

//----------------------------------------------------------------------------------------
//! \fn BounceEntropyImpl
//! \brief Entropy-based bounce check using tabulated EOS.

template <class EOSPolicy, class ErrorPolicy>
static void BounceEntropyImpl(MeshBlockPack *pmbp, bool &at_bounce,
                              Real rho_gate, Real r2_gate, Real spb_gate) {
  auto *pdyngr =
      dynamic_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmbp->pdyngr);
  if (pdyngr == nullptr)
    return;

  auto &eos_ = pdyngr->eos.ps.GetEOS();
  auto &temperature_ = pdyngr->temperature;
  auto &w0_ = pmbp->pmhd->w0;
  int nmhd = pmbp->pmhd->nmhd;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, nx1 = indcs.nx1;
  int js = indcs.js, nx2 = indcs.nx2;
  int ks = indcs.ks, nx3 = indcs.nx3;
  int nmb = pmbp->nmb_thispack;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  const int nmkji = nmb * nkji;

  const Real mb = eos_.GetBaryonMass();

  int bounce_flag = 0;
  Kokkos::parallel_reduce(
      "bounce_entropy", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, int &lflag) {
        int m = idx / nkji;
        int rem = idx % nkji;
        int k = rem / nji + ks;
        rem = rem % nji;
        int j = rem / nx1 + js;
        int i = rem % nx1 + is;

        Real rho = w0_(m, IDN, k, j, i);
        if (rho < rho_gate || rho <= 0.0)
          return;

        Real x1v = CellCenterX(i - is, nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
        Real x2v = CellCenterX(j - js, nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max);
        Real x3v = CellCenterX(k - ks, nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max);
        if (x1v * x1v + x2v * x2v + x3v * x3v > r2_gate)
          return;

        Real n = rho / mb;
        Real Ye = w0_(m, nmhd, k, j, i);
        Real Y[MAX_SPECIES] = {0.};
        Y[0] = Ye;
        Real T = temperature_(m, 0, k, j, i);
        Real spb = eos_.GetEntropyPerBaryon(n, T, Y);
        if (spb > spb_gate)
          lflag = 1;
      },
      Kokkos::Max<int>(bounce_flag));
  at_bounce = at_bounce || (bounce_flag == 1);
}

//----------------------------------------------------------------------------------------
//! \fn DeleptonizationImpl
//  \brief Liebendoerfer deleptonization with entropy correction (tabulated
//  EOS).
//         Invert with the existing CompOSE temperature interpolation policy.

struct EntropyInversionFailure {
  int rank, gid, k, j, i;
  Real rho, rho_cgs, n, p;
  Real Ye_old, Ye_new, delta_Ye;
  Real T_old, T_new, T_min, T_max;
  Real mu_nu, E_nu_avg;
  Real s_per_baryon, s_at_T_min, s_at_T_max;
  int update_entropy, correction_active;
};

template <class EOSPolicy, class ErrorPolicy>
void DeleptonizationImpl(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto *pdyngr =
      dynamic_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmbp->pdyngr);
  if (pdyngr == nullptr)
    return;

  auto &eos_ = pdyngr->eos.ps.GetEOS();
  auto &temperature_ = pdyngr->temperature;
  auto &w0_ = pmbp->pmhd->w0;
  int nmhd = pmbp->pmhd->nmhd;

  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;

  const Real mb = eos_.GetBaryonMass();
  const Real rho_min = eos_.GetMinimumDensity() * mb;
  const Real T_min = eos_.GetMinimumTemperature();
  const Real T_max = eos_.GetMaximumTemperature();
  const int rank = global_variable::my_rank;
  const int gid_start = pmbp->gids;

  const Real E_nu_avg = delept_E_nu_avg;
  const Real rho_trap = delept_rho_trap;
  const Real log10_r1 = delept_log10_rho1;
  const Real log10_r2 = delept_log10_rho2;
  const Real Ye2 = delept_Ye_2;
  const Real Yec = delept_Ye_c;
  const Real YeH = delept_Ye_H;

  const bool do_entropy = delept_update_entropy;
  const bool regional = prebounce_core_coupling;
  const bool use_region = regional && pmbp->pradm1 != nullptr &&
                          pmbp->pradm1->CouplingRegionEnabled();
  DvceArray4D<bool> region;
  if (use_region) region = pmbp->pradm1->coupling_region;
  const bool bh_excise = regional && pmbp->pcoord->coord_data.bh_excise;
  auto excision = pmbp->pcoord->excision_floor;
  // Read unit metadata on the host; eos_ remains const in the kernel.
  const Real rho_code_to_cgs = pdyngr->GetCodeUnitSystem().MassDensityConversion(
      Primitive::MakeCGS());

  Kokkos::View<int, DevExeSpace> entropy_failed("ccsn_entropy_failed");
  Kokkos::View<EntropyInversionFailure, DevExeSpace> entropy_failure(
      "ccsn_entropy_failure");
  Kokkos::deep_copy(entropy_failed, 0);

  par_for(
      "deleptonization", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        if (regional && ((bh_excise && excision(m, k, j, i)) ||
                         (use_region && region(m, k, j, i))))
          return;
        Real rho = w0_(m, IDN, k, j, i);
        if (rho <= rho_min)
          return;

        Real n = rho / mb;
        Real p = w0_(m, IPR, k, j, i);
        Real Ye_old = w0_(m, nmhd, k, j, i);

        // Use the temperature published with this cell-centered C2P state.
        Real Y_old[MAX_SPECIES] = {0.};
        Y_old[0] = Ye_old;
        Real T_old = temperature_(m, 0, k, j, i);

        // Compute target Ye
        Real rho_cgs = rho * rho_code_to_cgs;
        Real Ye_bar = Ye_of_rho(rho_cgs, log10_r1, log10_r2, Ye2, Yec, YeH);
        Real delta_Ye = fmin(0.0, Ye_bar - Ye_old);
        if (delta_Ye >= 0.0)
          return;

        Real Ye_new = Ye_old + delta_Ye;
        w0_(m, nmhd, k, j, i) = Ye_new;

        Real Y_new[MAX_SPECIES] = {0.};
        Y_new[0] = Ye_new;

        // mu_nu from pre-update composition (Y_old).
        // CompOSE identity: mu_nu = mu_e + mu_p - mu_n = mu_l.
        // GetElectronLeptonChemicalPotential returns mu_l (ECMUL).
        Real mu_nu = eos_.GetElectronLeptonChemicalPotential(n, T_old, Y_old);

        // --- Entropy update (port of GRA Liebendoerfer) -----------------------

        Real s_per_baryon;
        if (do_entropy) {
          // entropy from pre-update composition (Y_old)
          s_per_baryon = eos_.GetEntropyPerBaryon(n, T_old, Y_old);

          // entropy correction for neutrino losses
          if ((mu_nu > E_nu_avg) && (rho < rho_trap)) {
            s_per_baryon -= delta_Ye * (mu_nu - E_nu_avg) / T_old;
          }
        } else {
          // entropy from updated composition (Y_new)
          s_per_baryon = eos_.GetEntropyPerBaryon(n, T_old, Y_new);
        }

        // recover temperature from entropy
        Real T_new = eos_.GetTemperatureFromEntropy(n, s_per_baryon, Y_new);

        // In geometric-solar units both wrapper and EOS temperatures are MeV.
        // Reject invalid results before clamping can conceal them.
        if (!Kokkos::isfinite(T_new) || !(T_new > 0.0)) {
          const Real s_at_T_min = eos_.GetEntropyPerBaryon(n, T_min, Y_new);
          const Real s_at_T_max = eos_.GetEntropyPerBaryon(n, T_max, Y_new);
          // The first failing cell owns the report. The host reads it only
          // after this kernel completes, so the flag cannot expose a partial record.
          if (Kokkos::atomic_compare_exchange(&entropy_failed(), 0, 1) == 0) {
            entropy_failure() = {
                rank, gid_start + m, k, j, i,
                rho, rho_cgs, n, p, Ye_old, Ye_new, delta_Ye,
                T_old, T_new, T_min, T_max, mu_nu, E_nu_avg,
                s_per_baryon, s_at_T_min, s_at_T_max,
                do_entropy ? 1 : 0,
                (do_entropy && mu_nu > E_nu_avg && rho < rho_trap) ? 1 : 0};
          }
          return;
        }
        // Bound finite positive results before another EOS lookup. This local
        // policy clips endpoint roundoff and larger excursions without changing
        // the shared inverse or guaranteeing entropy preservation after clipping.
        T_new = fmax(T_min, fmin(T_max, T_new));
        w0_(m, IPR, k, j, i) = eos_.GetPressure(n, T_new, Y_new);
        temperature_(m, 0, k, j, i) = T_new;
      });

  // Finish all failing writes before inspecting the device-side record.
  Kokkos::fence();
  auto h_entropy_failed = Kokkos::create_mirror_view(entropy_failed);
  Kokkos::deep_copy(h_entropy_failed, entropy_failed);
  if (h_entropy_failed()) {
    auto h_entropy_failure = Kokkos::create_mirror_view(entropy_failure);
    Kokkos::deep_copy(h_entropy_failure, entropy_failure);
    const auto &f = h_entropy_failure();
    char message[1024];
    std::snprintf(
        message, sizeof(message),
        "CCSN entropy inversion FAIL rank=%d gid=%d cell=(%d,%d,%d)"
        " rho=% .17e rho_cgs=% .17e n=% .17e P=% .17e"
        " Ye_old=% .17e Ye_new=% .17e dYe=% .17e"
        " T_old=% .17e T_new=% .17e Tmin=% .17e Tmax=% .17e"
        " mu_nu=% .17e E_nu_avg=% .17e"
        " s_target=% .17e s_Tmin=% .17e s_Tmax=% .17e"
        " update_entropy=%d correction_active=%d",
        f.rank, f.gid, f.k, f.j, f.i, f.rho, f.rho_cgs, f.n, f.p,
        f.Ye_old, f.Ye_new, f.delta_Ye, f.T_old, f.T_new, f.T_min,
        f.T_max, f.mu_nu, f.E_nu_avg, f.s_per_baryon, f.s_at_T_min,
        f.s_at_T_max, f.update_entropy, f.correction_active);
    Kokkos::abort(message);
  }

  // The native moving, magnetized GRMHD P2C includes momentum and magnetic terms.
  pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
}

//----------------------------------------------------------------------------------------
//! \fn Deleptonization
//  \brief Full-step CompOSE map and bounce check, enrolled by a local callable.

void Deleptonization(Mesh *pm, const Real dt) {
  // The empirical update is a full-step map, not a rate scaled by an RK weight.
  if (delept_active) {
    using namespace Primitive;
    DeleptonizationImpl<EOSCompOSE<NQTLogs>, ResetFloor>(pm);
    DeleptonizationImpl<EOSCompOSE<NormalLogs>, ResetFloor>(pm);
  }
  BounceCheck(pm, delept_pin);
}

//----------------------------------------------------------------------------------------
// Entropy bounce detection and persistent transition state
//----------------------------------------------------------------------------------------

// Global state
static bool detect_bounce = false;
static bool bounce_detected = false;
static bool recoupling_approach = false;
static bool equilibriate_post_bounce = true;
static bool conservative_bounce_projection = false;
// Controls of the conservative bounce projection, filled from the deck in
// UserProblem when the projection is selected. Every threshold the solver
// applies is a member, so a failure line can be read against exactly the values
// the solve used.
static ccsn::ProjectionControls projection_controls;
static bool stop_at_bounce = false;
static Real eql_rho_min_cgs;
static Real eql_t_min_mev;
static std::string bounce_method_str = "None";
static Real bdm_rho_min = 1.0e10;
static Real bdm_spb_max = 3.0;
static Real bdm_r_max = 30.0;

//----------------------------------------------------------------------------------------
//! Print, once, the run-level constants a failed projection line is read against.
//!
//! Called on the failure path only, so a projection that succeeds still prints
//! nothing. The per-cell lines are in the projection's own code units; the
//! conversions and floors here are shared by every cell and are what turn those
//! numbers into physical ones and separate a floor hit from a genuine solve
//! failure. rho_cgs_per_code converts a code mass density to g/cc, so
//! rho_cgs = rho_code * mb * rho_cgs_per_code, and n_cgs_per_code converts the
//! reported baryon number density to 1/cc. E_floor and N_floor are the
//! configured radiation floors; the projection scales each by 1/sqrt(detg) of
//! its cell, and detg follows from that line's metric. Every projection control
//! is printed too, because those are the thresholds the failing test was applied
//! against and a reader of the line needs them to reproduce the decision.
template <class EOSPolicy>
static void ProjectionFailureContext(
    const EOSPolicy &eos, const Primitive::UnitSystem &code_units,
    const Primitive::UnitSystem &cgs_units, const Real mb,
    const ccsn::ProjectionControls &ctl, Real energy_floor, Real number_floor) {
  Kokkos::printf(
      "CCSN M1 projection context: mb=% .9e rho_cgs_per_code=% .9e"
      " n_cgs_per_code=% .9e len_cgs_per_code=% .9e time_cgs_per_code=% .9e"
      " E_cgs_per_code=% .9e eql_rho_min_cgs=% .9e eql_t_min_mev=% .9e"
      " tol=% .9e maxiter=%d chain_tol=% .9e slack=% .9e mom_floor=% .9e"
      " ls_maxiter=%d jac_step=% .9e armijo=% .9e backtrack=% .9e"
      " pivot=% .9e E_floor=% .9e N_floor=% .9e"
      " n_atm=% .9e T_atm=% .9e n_threshold=% .9e n_min=% .9e n_max=% .9e"
      " T_min=% .9e T_max=% .9e h_min=% .9e v_max=% .9e\n",
      mb, code_units.MassDensityConversion(cgs_units),
      code_units.NumberDensityConversion(cgs_units),
      code_units.LengthConversion(cgs_units),
      code_units.TimeConversion(cgs_units),
      code_units.EnergyDensityConversion(cgs_units), eql_rho_min_cgs,
      eql_t_min_mev, ctl.tolerance, ctl.max_iterations, ctl.chain_tolerance,
      ctl.acceptance_slack, ctl.momentum_floor, ctl.line_search_maxiter,
      ctl.jacobian_step, ctl.armijo, ctl.backtrack_factor, ctl.pivot_threshold,
      energy_floor, number_floor, eos.GetDensityFloor(),
      eos.GetTemperatureFloor(), eos.GetThreshold(), eos.GetMinimumDensity(),
      eos.GetMaximumDensity(), eos.GetMinimumTemperature(),
      eos.GetMaximumTemperature(), eos.GetMinimumEnthalpy(),
      eos.GetMaxVelocity());
}

//! \fn ConservativeM1ProjectionImpl
//! \brief Replace the matter-only bounce state with a local conservative
//!        matter+radiation equilibrium in cells above the EQL cutoff.
//!
//! A cell the solver refuses is printed once, counted, and skipped: nothing of
//! its state is written, so it keeps the pre-bounce moments and the conserved
//! state it entered with, and it is unmasked with the rest at the end of the
//! transition. That is the same state a cell below the equilibrium cutoff is
//! left in, and it is local: one cell cannot end the run. Run-level conditions
//! stay fatal - a missing M1 or dynamics object by the caller's return value,
//! and an unmatched EOS policy in the dispatcher.
template <class EOSPolicy, class ErrorPolicy>
static bool ConservativeM1ProjectionImpl(MeshBlockPack *pmbp) {
  if (pmbp->pradm1 == nullptr)
    return false;
  auto *pdyngr =
      dynamic_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(pmbp->pdyngr);
  if (pdyngr == nullptr)
    return false;

  auto &ps_ = pdyngr->eos.ps;
  auto &eos_mut = ps_.GetEOSMutable();
  const Real mb = eos_mut.GetBaryonMass();
  const Primitive::UnitSystem code_units = eos_mut.GetCodeUnitSystem();
  const Primitive::UnitSystem eos_units = eos_mut.GetEOSUnitSystem();
  const Primitive::UnitSystem ngs_units = Primitive::MakeNGS();
  const Primitive::UnitSystem cgs_units = Primitive::MakeCGS();
  const Real rho_code_to_cgs = code_units.MassDensityConversion(cgs_units);
  const Real eql_rho_min_code = eql_rho_min_cgs / rho_code_to_cgs;

  auto &mhd_u0 = pmbp->pmhd->u0;
  auto &mhd_w0 = pmbp->pmhd->w0;
  auto &mhd_b0 = pmbp->pmhd->b0;
  auto &temperature = pdyngr->temperature;
  auto &m1_u0 = pmbp->pradm1->u0;
  auto &adm = pmbp->padm->adm;
  auto &excision_floor = pmbp->pcoord->excision_floor;
  const bool excise = pmbp->pcoord->coord_data.bh_excise;
  const int nmhd = pmbp->pmhd->nmhd;
  const int nscalars = pmbp->pmhd->nscalars;
  const int nvars = pmbp->pradm1->nvars;
  const radiationm1::RadiationM1OpacityType opacity_type =
      pmbp->pradm1->params.opacity_type;
  const Real energy_floor = pmbp->pradm1->params.rad_E_floor;
  const Real number_floor = pmbp->pradm1->params.rad_N_floor;
  // Every threshold the solve applies comes from projection_controls, which
  // UserProblem fills from the deck and validates by name. Nothing here is a
  // relaxation of the radiation toward a target: every point the solver evaluates
  // lies on the equilibrium manifold and the line search only shortens a step by
  // the configured backtracking factor.
  const ccsn::ProjectionControls ctl = projection_controls;

  auto &indcs = pmbp->pmesh->mb_indcs;
  const int is = indcs.is, nx1 = indcs.nx1;
  const int js = indcs.js, nx2 = indcs.nx2;
  const int ks = indcs.ks, nx3 = indcs.nx3;
  const int nmb = pmbp->nmb_thispack;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  const int nmkji = nmb * nkji;
  const Real eql_t_min = eql_t_min_mev;

  // Identity captured for the failure lines: the pack-local block index m is
  // only unique within one rank and refinement level, so each line reports the
  // rank and the global meshblock id gids + m as well.
  const int rank = global_variable::my_rank;
  const int gid_start = pmbp->gids;
  int failures = 0;
  Kokkos::parallel_reduce(
      "ccsn_conservative_m1_projection",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, int &local_failures) {
        int m = idx / nkji;
        int rem = idx % nkji;
        int k = rem / nji + ks;
        rem %= nji;
        int j = rem / nx1 + js;
        int i = rem % nx1 + is;

        if (excise && excision_floor(m, k, j, i))
          return;

        const Real rho = mhd_w0(m, IDN, k, j, i);
        if (rho < eql_rho_min_code)
          return;

        Real initial_prim[NPRIM] = {};
        initial_prim[PRH] = rho / mb;
        initial_prim[PVX] = mhd_w0(m, IVX, k, j, i);
        initial_prim[PVY] = mhd_w0(m, IVY, k, j, i);
        initial_prim[PVZ] = mhd_w0(m, IVZ, k, j, i);
        initial_prim[PPR] = mhd_w0(m, IPR, k, j, i);
        for (int n = 0; n < nscalars; ++n) {
          initial_prim[PYF + n] = mhd_w0(m, nmhd + n, k, j, i);
        }
        initial_prim[PTM] = temperature(m, 0, k, j, i);
        if (initial_prim[PTM] < eql_t_min)
          return;

        Real g3d[NSPMETRIC], g3u[NSPMETRIC];
        g3d[S11] = adm.g_dd(m, 0, 0, k, j, i);
        g3d[S12] = adm.g_dd(m, 0, 1, k, j, i);
        g3d[S13] = adm.g_dd(m, 0, 2, k, j, i);
        g3d[S22] = adm.g_dd(m, 1, 1, k, j, i);
        g3d[S23] = adm.g_dd(m, 1, 2, k, j, i);
        g3d[S33] = adm.g_dd(m, 2, 2, k, j, i);
        const Real detg = Primitive::GetDeterminant(g3d);
        if (!(detg > 0.0) || !Kokkos::isfinite(detg)) {
          // The metric decides whether the projection's undensitized variables
          // exist at all, so this site reports the metric instead of a state
          // that cannot be derived from it. The z group holds the three velocity
          // components the primitive solver stores, W v^i (the projection's z^i
          // in PVX..PVZ), and not v^i.
          Kokkos::printf(
              "CCSN M1 projection FAIL rank=%d gid=%d cell=(%d,%d,%d) site=107"
              " detail=0 iter=-1 diff=% .9e"
              " | prim rho=% .17e zx=% .9e zy=% .9e zz=% .9e P=% .9e T=% .17e"
              " Ye=% .17e"
              " | met=% .9e % .9e % .9e % .9e % .9e % .9e detg=% .17e\n",
              rank, gid_start + m, k, j, i, -1.0, initial_prim[PRH],
              initial_prim[PVX], initial_prim[PVY], initial_prim[PVZ],
              initial_prim[PPR], initial_prim[PTM], initial_prim[PYF], g3d[S11],
              g3d[S12], g3d[S13], g3d[S22], g3d[S23], g3d[S33], detg);
          local_failures++;
          return;
        }
        const Real sdetg = Kokkos::sqrt(detg);
        const Real isdetg = 1.0 / sdetg;
        Primitive::InvertMatrix(g3u, g3d, detg);

        Real b[NMAG] = {
            0.5 * (mhd_b0.x1f(m, k, j, i) + mhd_b0.x1f(m, k, j, i + 1)) *
                isdetg,
            0.5 * (mhd_b0.x2f(m, k, j, i) + mhd_b0.x2f(m, k, j + 1, i)) *
                isdetg,
            0.5 * (mhd_b0.x3f(m, k, j, i) + mhd_b0.x3f(m, k + 1, j, i)) *
                isdetg};

        Real old_cons[NCONS] = {};
        old_cons[CDN] = mhd_u0(m, IDN, k, j, i) * isdetg;
        old_cons[CSX] = mhd_u0(m, IM1, k, j, i) * isdetg;
        old_cons[CSY] = mhd_u0(m, IM2, k, j, i) * isdetg;
        old_cons[CSZ] = mhd_u0(m, IM3, k, j, i) * isdetg;
        old_cons[CTA] = mhd_u0(m, IEN, k, j, i) * isdetg;
        for (int n = 0; n < nscalars; ++n) {
          old_cons[CYD + n] = mhd_u0(m, nmhd + n, k, j, i) * isdetg;
        }

        Real old_rad[ccsn::projection_nspecies][ccsn::projection_nvars] = {};
        for (int s = 0; s < ccsn::projection_nspecies; ++s) {
          for (int v = 0; v < ccsn::projection_nvars; ++v) {
            old_rad[s][v] =
                m1_u0(m, radiationm1::CombinedIdx(s, v, nvars), k, j, i) * isdetg;
          }
        }

        Real final_prim[NPRIM], final_cons[NCONS];
        Real final_rad[ccsn::projection_nspecies][ccsn::projection_nvars];
        // The report is overwritten by every path that returns a failure, so this
        // initialisation only states the convention: code 0 is "no failure
        // recorded", and iteration, residual and worst_term start at -1 for
        // "nothing has been evaluated yet", as the solver itself sets them.
        ccsn::ProjectionFailureReport fail = {0, 0, -1, -1.0, 0, -1};
        // The solve works in the cell's conserved variables, so initial_prim is
        // not an input to it; the primitives enter through the failure line.
        const bool success = ccsn::ProjectM1Cell(
            ps_, opacity_type, code_units, eos_units, ngs_units,
            old_cons, b, g3d, g3u, old_rad, ctl,
            energy_floor * isdetg, number_floor * isdetg, final_prim,
            final_cons, final_rad, &fail);
        if (!success) {
          ccsn::PrintProjectionFailureLine(rank, gid_start + m, k, j, i, fail,
                                           initial_prim, old_cons, b, g3d,
                                           old_rad);
          local_failures++;
          return;
        }

        mhd_u0(m, IM1, k, j, i) = final_cons[CSX] * sdetg;
        mhd_u0(m, IM2, k, j, i) = final_cons[CSY] * sdetg;
        mhd_u0(m, IM3, k, j, i) = final_cons[CSZ] * sdetg;
        mhd_u0(m, IEN, k, j, i) = final_cons[CTA] * sdetg;
        for (int n = 0; n < nscalars; ++n) {
          mhd_u0(m, nmhd + n, k, j, i) = final_cons[CYD + n] * sdetg;
        }

        mhd_w0(m, IDN, k, j, i) = final_prim[PRH] * mb;
        mhd_w0(m, IVX, k, j, i) = final_prim[PVX];
        mhd_w0(m, IVY, k, j, i) = final_prim[PVY];
        mhd_w0(m, IVZ, k, j, i) = final_prim[PVZ];
        mhd_w0(m, IPR, k, j, i) = final_prim[PPR];
        temperature(m, 0, k, j, i) = final_prim[PTM];
        for (int n = 0; n < nscalars; ++n) {
          mhd_w0(m, nmhd + n, k, j, i) = final_prim[PYF + n];
        }

        for (int s = 0; s < ccsn::projection_nspecies; ++s) {
          for (int v = 0; v < ccsn::projection_nvars; ++v) {
            m1_u0(m, radiationm1::CombinedIdx(s, v, nvars), k, j, i) =
                final_rad[s][v] * sdetg;
          }
        }
      },
      Kokkos::Sum<int>(failures));

  if (failures != 0) {
    // The per-cell lines above are the actionable record; this line is their
    // count and the run-level constants that turn those numbers into physical
    // ones. The run continues: a cell that keeps its pre-bounce state is worth
    // less than the post-bounce evolution it would otherwise stop.
    ProjectionFailureContext(eos_mut, code_units, cgs_units, mb, ctl,
                             energy_floor, number_floor);
    std::cerr << "### WARNING: conservative M1 bounce projection left "
              << failures << " cell(s) unprojected on rank "
              << global_variable::my_rank << std::endl;
  }
  return true;
}

static void ConservativeM1Projection(MeshBlockPack *pmbp) {
  using namespace Primitive;
  bool handled = false;
  handled = ConservativeM1ProjectionImpl<EOSCompOSE<NQTLogs>, ResetFloor>(pmbp) ||
            handled;
  handled = ConservativeM1ProjectionImpl<EOSCompOSE<NormalLogs>, ResetFloor>(pmbp) ||
            handled;
  if (!handled) {
    std::cerr << "### FATAL: conservative M1 projection requires CompOSE EOS"
              << std::endl;
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#else
    std::abort();
#endif
  }
}

// Per-cycle bounce check, called from Deleptonization after delept work
static void BounceCheck(Mesh *pm, ParameterInput *pin) {
  if (!detect_bounce || bounce_detected ||
      pin->GetOrAddBoolean("problem", "post_bounce", false))
    return;

  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &w0_ = pmbp->pmhd->w0;
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;
  int nmb = pmbp->nmb_thispack;

  bool at_bounce = false;

  {
    // Full EOS entropy check via template dispatch
    const Real rho_gate = bdm_rho_min /
        pmbp->pdyngr->GetCodeUnitSystem().MassDensityConversion(Primitive::MakeCGS());
    const Real r2_gate = bdm_r_max * bdm_r_max;
    const Real spb_gate = bdm_spb_max;
    switch (pmbp->pdyngr->eos_policy) {
    case DynGRMHD_EOS::eos_compose: {
      using namespace Primitive;
      BounceEntropyImpl<EOSCompOSE<NQTLogs>, ResetFloor>(
          pmbp, at_bounce, rho_gate, r2_gate, spb_gate);
      BounceEntropyImpl<EOSCompOSE<NormalLogs>, ResetFloor>(
          pmbp, at_bounce, rho_gate, r2_gate, spb_gate);
      break;
    }
    default:
      break;
    }

  }

  // MPI-synchronize bounce flag
#if MPI_PARALLEL_ENABLED
  {
    int local_bounce = at_bounce ? 1 : 0;
    int global_bounce = 0;
    MPI_Allreduce(&local_bounce, &global_bounce, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    at_bounce = (global_bounce != 0);
  }
#endif

  if (at_bounce) {
    // Save bounce diagnostics for history output BEFORE modifying state.
    // Need global max of u0(IDN) across all ranks (each rank only sees its
    // local cells). Compute here since we are already synchronized.
    const Real bounce_time = pm->time + pm->dt;
    t_bounce_val = bounce_time;
    {
      auto &u0_bounce = pmbp->pmhd->u0;
      const int nkji_b = nx3 * nx2 * nx1;
      const int nji_b = nx2 * nx1;
      const int nmkji_b = nmb * nkji_b;
      Real D_local = 0.0;
      Kokkos::parallel_reduce(
          "bounce_Dmax", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji_b),
          KOKKOS_LAMBDA(const int idx, Real &lmax) {
            int m = idx / nkji_b;
            int rem = idx % nkji_b;
            int k = rem / nji_b + ks;
            rem = rem % nji_b;
            int j = rem / nx1 + js;
            int i = rem % nx1 + is;
            lmax = fmax(lmax, u0_bounce(m, IDN, k, j, i));
          }, Kokkos::Max<Real>(D_local));
      Real D_global = D_local;
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(&D_local, &D_global, 1, MPI_ATHENA_REAL, MPI_MAX,
                    MPI_COMM_WORLD);
#endif
      D_max_bounce_val = D_global;
    }

    const char *m1_state = " unavailable.";
    if (pmbp->pradm1 != nullptr) {
      if (prebounce_core_coupling) {
        // The detecting step (including its M1 split) keeps the frozen region.
        m1_state = " regional feedback retained; full-domain feedback begins next full step.";
      } else if (recoupling_approach) {
        pmbp->pradm1->SetCouplingEnabled(true);
        m1_state = " evolved state preserved; configured fluid/geometry feedback restored.";
      } else {
        if (equilibriate_post_bounce) {
          ConservativeM1Projection(pmbp);
          m1_state = " conservative equilibrium projected; U refresh follows.";
        } else {
          m1_state = " unmasked without equilibrium injection.";
        }
        pmbp->pradm1->SetEvolutionEnabled(true);
      }
    }
    bounce_detected = true;
    if (!prebounce_core_coupling) delept_active = false;
    if (global_variable::my_rank == 0) {
      std::cout << (prebounce_core_coupling
          ? "### CCSN: Bounce detected! Global deleptonization shutdown at next full step."
          : "### CCSN: Bounce detected! Deleptonization disabled.")
                << std::endl;
    }
    if (global_variable::my_rank == 0) {
      std::cout << "### CCSN: M1" << m1_state << std::endl;
    }
    // Record bounce time (stored in ParameterInput for restart/post-processing)
    pin->SetReal("problem", "t_bounce", bounce_time);
    pin->SetReal("problem", "D_max_bounce", D_max_bounce_val);
    pin->SetBoolean("problem", "post_bounce", true);
    pin->SetString("problem", "deleptonization_method", "None");
    // Stop after post-integrator communication/C2P, before the first M1 split.
    if (stop_at_bounce) {
      pin->SetBoolean("problem", "stop_before_m1", true);
      if (global_variable::my_rank == 0) {
        std::cout << "### CCSN: stop_at_bounce requested at t = " << bounce_time
                  << std::endl;
      }
    }
  }
}

// Freeze the primitive-density selector once before each full RK step. The
// active pack and bounds may have changed during AMR since the last call.
static void PrepareCoreCoupling(Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto *prad = pmbp->pradm1;
  if (bounce_detected) {
    prad->SetCouplingRegionEnabled(false);
    delept_active = false;
    return;
  }
  prad->SetCouplingRegionEnabled(true);
  auto region = prad->coupling_region;
  auto w0 = pmbp->pmhd->w0;
  auto &indcs = pm->mb_indcs;
  const Real rho_trap = delept_rho_trap;
  par_for("ccsn_core_coupling", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
          indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
          KOKKOS_LAMBDA(int m, int k, int j, int i) {
            region(m, k, j, i) = w0(m, IDN, k, j, i) >= rho_trap;
          });
}

// Append only this pgen's post-hydro work. The scheduler owns receive progress;
// no pointer into replaceable block, coordinate or boundary metadata is captured.
static void EnrollCCSNTasks(Mesh *pm) {
  if (prebounce_core_coupling) {
    auto &before = *pm->pmb_pack->tl_map.at("before_timeintegrator");
    TaskID before_dep(0);
    for (int id = 1; id <= before.Size(); ++id)
      before_dep = before_dep | TaskID(id);
    before.AddTask([pm](Driver *d, int stage) {
      PrepareCoreCoupling(pm);
      return TaskStatus::complete;
    }, before_dep);
  }
  auto &tl = *pm->pmb_pack->tl_map.at("after_timeintegrator");
  TaskID dep(0);
  for (int id = 1; id <= tl.Size(); ++id) dep = dep | TaskID(id);
  dep = tl.AddTask([pm](Driver *d, int stage) {
    Deleptonization(pm, pm->dt);
    return TaskStatus::complete;
  }, dep);
  auto *pmhd = pm->pmb_pack->pmhd;
  auto *pdg = pm->pmb_pack->pdyngr;
  dep = tl.AddTask(&mhd::MHD::InitRecvU, pmhd, dep);
  dep = tl.AddTask(&mhd::MHD::RestrictU, pmhd, dep);
  dep = tl.AddTask(&mhd::MHD::SendU, pmhd, dep);
  dep = tl.AddTask(&mhd::MHD::RecvU, pmhd, dep);
  dep = tl.AddTask(&mhd::MHD::Prolongate, pmhd, dep);
  dep = tl.AddTask(&mhd::MHD::ApplyPhysicalBCs, pmhd, dep);
  // Virtual dispatch reaches the existing concrete EOS-specific DynGRMHD C2P,
  // including ghost zones, exactly as in the accepted post-M1 refresh.
  dep = tl.AddTask([pdg](Driver *d, int stage) {
    return pdg->ConToPrim(d, stage);
  }, dep);
  dep = tl.AddTask(&mhd::MHD::ClearSendU, pmhd, dep);
  dep = tl.AddTask(&mhd::MHD::ClearRecvU, pmhd, dep);
  auto *prad = pm->pmb_pack->pradm1;
  if (prad != nullptr) {
    using radiationm1::RadiationM1;
    // A negative stage selects the existing U-only API: no AMR flux receives.
    dep = tl.AddTask([prad](Driver *d, int stage) {
      return prad->InitRecv(d, -1);
    }, dep);
    dep = tl.AddTask(&RadiationM1::RestrictU, prad, dep);
    dep = tl.AddTask(&RadiationM1::SendU, prad, dep);
    dep = tl.AddTask(&RadiationM1::RecvU, prad, dep);
    dep = tl.AddTask(&RadiationM1::Prolongate, prad, dep);
    dep = tl.AddTask(&RadiationM1::ApplyPhysicalBCs, prad, dep);
    dep = tl.AddTask([prad](Driver *d, int stage) {
      return prad->ClearSend(d, -1);
    }, dep);
    dep = tl.AddTask([prad](Driver *d, int stage) {
      return prad->ClearRecv(d, -1);
    }, dep);
    // Boundary/prolongation writes are also mask/closure inputs, even when the
    // operator-split evolution is disabled or this is the bounce-stop step.
    dep = tl.AddTask(&RadiationM1::SetMask, prad, dep);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem
//! \brief Register CCSN full-step work and set massive-star initial data.
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  const Real INF = std::numeric_limits<Real>::infinity();
  if (!pmbp->pcoord->is_dynamical_relativistic || pmbp->pdyngr == nullptr ||
      pmbp->pmhd == nullptr || pmbp->padm == nullptr) {
    std::cerr << "### FATAL: CCSN requires dynamical GRMHD and ADM" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->pmhd->nscalars != 1) {
    // The profile supplies one composition variable, matching CompOSE's
    // SetNSpecies requirement even when its assertions are compiled out.
    std::cerr << "### FATAL: CCSN requires exactly one scalar, Ye" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->pdyngr->eos_policy == DynGRMHD_EOS::eos_compose &&
      pin->GetString("mhd", "units") != "geometric_solar") {
    // This profile and its MeV thresholds use the donor's geometric-solar
    // normalization. Supporting other units needs conversion of all IC fields.
    std::cerr << "### FATAL: CCSN profiles require mhd/units=geometric_solar"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  user_ref_func = &RefinementCondition;
  user_hist_func = &History;

  // Construction happens once, including on restart. Startup AMR calls this
  // method again only to reevaluate initial data on the new hierarchy.
  if (!user_tasks_enrolled_) {
    pin->SetBoolean("problem", "stop_before_m1", false);
    delept_pin = pin;
    const Real rho_code_to_cgs = pmbp->pdyngr->GetCodeUnitSystem().MassDensityConversion(
        Primitive::MakeCGS());
    const std::string method = pin->GetOrAddString("problem", "refinement_method", "none");
    if (method == "MaxMassInCell") {
      refm = RefinementMethod::MaxMassInCell;
    } else if (method == "none") {
      refm = RefinementMethod::none;
    } else {
      std::cerr << "### FATAL: CCSN supports refinement_method=MaxMassInCell or none"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    delta_min_m = pin->GetOrAddReal("problem", "delta_min_m", -INF);
    delta_max_m = pin->GetOrAddReal("problem", "delta_max_m", INF);

    const std::string dlp_method =
        pin->GetOrAddString("problem", "deleptonization_method", "Liebendoerfer");
    // Simple was an alias for this same entropy update with CompOSE in the
    // donor. Retain that input spelling, not its non-tabulated EOS algorithm.
    delept_active = dlp_method == "Liebendoerfer" || dlp_method == "Simple";
    if (!delept_active && dlp_method != "None" && dlp_method != "none") {
      std::cerr << "### FATAL: CCSN supports Liebendoerfer/Simple or None deleptonization"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    delept_update_entropy = pin->GetOrAddBoolean("problem", "update_entropy", true);
    if (pin->GetOrAddBoolean("problem", "update_conserved", false)) {
      // The donor's rest-only tau overwrite is not a general moving GRMHD
      // thermodynamic update. Keep the full native P2C route in this port.
      std::cerr << "### FATAL: CCSN requires update_conserved=false (native GRMHD P2C)"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    delept_E_nu_avg = pin->GetOrAddReal("problem", "E_nu_avg", 10.0);
    const Real rho_trap_cgs = pin->GetOrAddReal("problem", "rho_trap", 1.0e12);
    delept_rho_trap = rho_trap_cgs / rho_code_to_cgs;
    delept_log10_rho1 = pin->GetOrAddReal("deleptonization", "log10_rho1", 7.795);
    delept_log10_rho2 = pin->GetOrAddReal("deleptonization", "log10_rho2", 12.816);
    delept_Ye_2 = pin->GetOrAddReal("deleptonization", "Ye_2", 0.308);
    delept_Ye_c = pin->GetOrAddReal("deleptonization", "Ye_c", 0.0412);
    delept_Ye_H = pin->GetOrAddReal("deleptonization", "Ye_H", 0.257);

    detect_bounce = pin->GetOrAddBoolean("problem", "detect_bounce", false);
    recoupling_approach = pin->GetOrAddBoolean("problem", "recoupling_approach", false);
    prebounce_core_coupling =
        pin->GetOrAddBoolean("problem", "prebounce_core_coupling", false);
    const bool inject_equilibrium = pin->GetOrAddBoolean("problem", "inject_equilibrium", false);
    equilibriate_post_bounce =
        pin->GetOrAddBoolean("problem", "equilibriate_post_bounce", true);
    conservative_bounce_projection =
        pin->GetOrAddBoolean("problem", "conservative_bounce_projection", false);
    // Controls of the conservative bounce projection. Every threshold, tolerance
    // and iteration cap the solve applies is read here and nowhere else, so the
    // header holds no numeric policy and a failure line can be read against the
    // values the solve used. Defaults reproduce the values the solver was
    // developed and verified with; a run that changes one is changing the
    // decision a cell is judged by, not merely a performance knob.
    projection_controls.tolerance =
        pin->GetOrAddReal("problem", "projection_tolerance", 1.0e-8);
    projection_controls.max_iterations =
        pin->GetOrAddInteger("problem", "projection_maxiter", 50);
    projection_controls.chain_tolerance =
        pin->GetOrAddReal("problem", "projection_chain_tolerance", 1.0e-10);
    projection_controls.acceptance_slack =
        pin->GetOrAddReal("problem", "projection_acceptance_slack", 10.0);
    projection_controls.momentum_floor =
        pin->GetOrAddReal("problem", "projection_momentum_floor", 1.0e-3);
    projection_controls.line_search_maxiter =
        pin->GetOrAddInteger("problem", "projection_line_search_maxiter", 25);
    projection_controls.jacobian_step =
        pin->GetOrAddReal("problem", "projection_jacobian_step", 1.0e-6);
    projection_controls.armijo =
        pin->GetOrAddReal("problem", "projection_armijo", 1.0e-4);
    projection_controls.backtrack_factor =
        pin->GetOrAddReal("problem", "projection_backtrack_factor", 0.5);
    projection_controls.pivot_threshold =
        pin->GetOrAddReal("problem", "projection_pivot_threshold", 1.0e-14);
    stop_at_bounce = pin->GetOrAddBoolean("problem", "stop_at_bounce", false);
    if (inject_equilibrium ||
        (detect_bounce && equilibriate_post_bounce && !conservative_bounce_projection)) {
      std::cerr << "### FATAL: nonconservative M1 injection is not supported; select "
                   "conservative_bounce_projection or disable equilibriate_post_bounce"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (recoupling_approach && (equilibriate_post_bounce ||
        conservative_bounce_projection || pmbp->pradm1 == nullptr ||
        !pmbp->pradm1->params.matter_sources)) {
      std::cerr << "### FATAL: recoupling requires M1 matter_sources=true and "
                   "equilibriate_post_bounce=false, conservative_bounce_projection=false"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if ((delept_active || detect_bounce) &&
        pmbp->pdyngr->eos_policy != DynGRMHD_EOS::eos_compose) {
      std::cerr << "### FATAL: CCSN deleptonization/entropy bounce requires CompOSE"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (conservative_bounce_projection) {
      const bool projection_opacity = pmbp->pradm1 != nullptr &&
          radiationm1::equilibrium_available(pmbp->pradm1->params.opacity_type);
      if (!detect_bounce || recoupling_approach || !equilibriate_post_bounce ||
          !projection_opacity ||
          pmbp->pradm1->nspecies != radiationm1::equilibrium_nspecies ||
          pmbp->pradm1->nvars != 5 ||
          !pmbp->pradm1->params.matter_sources || !pmbp->pradm1->params.backreact) {
        std::cerr << "### FATAL: conservative projection requires entropy bounce, "
                     "an opacity backend with four-group equilibrium densities, "
                     "matter sources/backreaction and equilibrium mode"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // Every control is validated by name, so a rejected run says which key is
      // wrong. The message keeps the phrase the prebuilt test matches on.
      auto require = [](const char *key, bool ok, const char *what) {
        if (!ok) {
          std::cerr << "### FATAL: invalid conservative projection controls: "
                    << key << " " << what << std::endl;
          std::exit(EXIT_FAILURE);
        }
      };
      const ccsn::ProjectionControls &pc = projection_controls;
      require("problem/projection_tolerance",
              pc.tolerance > 0.0 && std::isfinite(pc.tolerance),
              "must be positive and finite");
      require("problem/projection_maxiter", pc.max_iterations >= 1,
              "must be at least 1");
      require("problem/projection_chain_tolerance",
              pc.chain_tolerance > 0.0 && std::isfinite(pc.chain_tolerance),
              "must be positive and finite");
      require("problem/projection_acceptance_slack",
              pc.acceptance_slack >= 1.0 && std::isfinite(pc.acceptance_slack),
              "must be at least 1: it factors the convergence threshold");
      require("problem/projection_momentum_floor",
              pc.momentum_floor > 0.0 && std::isfinite(pc.momentum_floor),
              "must be positive and finite");
      require("problem/projection_line_search_maxiter",
              pc.line_search_maxiter >= 1, "must be at least 1");
      require("problem/projection_jacobian_step",
              pc.jacobian_step > 0.0 && std::isfinite(pc.jacobian_step),
              "must be positive and finite");
      require("problem/projection_armijo",
              pc.armijo > 0.0 && pc.armijo < 1.0,
              "must lie strictly between 0 and 1");
      require("problem/projection_backtrack_factor",
              pc.backtrack_factor > 0.0 && pc.backtrack_factor < 1.0,
              "must lie strictly between 0 and 1");
      require("problem/projection_pivot_threshold",
              pc.pivot_threshold >= 0.0 && std::isfinite(pc.pivot_threshold),
              "must be non-negative and finite");
      eql_rho_min_cgs = pin->GetReal("problem", "eql_rho_min_cgs");
      eql_t_min_mev = pin->GetReal("problem", "eql_t_min_mev");
    }
    if (detect_bounce) {
      bounce_method_str = pin->GetOrAddString("problem", "bounce_detection_method", "entropy");
      if (bounce_method_str != "entropy") {
        std::cerr << "### FATAL: CCSN supports bounce_detection_method=entropy" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      bdm_rho_min = pin->GetOrAddReal("problem", "bdm_rho_min", 1.0e10);
      bdm_spb_max = pin->GetOrAddReal("problem", "bdm_spb_max", 3.0);
      bdm_r_max = pin->GetOrAddReal("problem", "bdm_r_max", 30.0);
    }

    bounce_detected = restart && pin->GetOrAddBoolean("problem", "post_bounce", false);
    pin->SetBoolean("problem", "post_bounce", bounce_detected);
    t_bounce_val = bounce_detected ? pin->GetOrAddReal("problem", "t_bounce", 0.0) : 0.0;
    D_max_bounce_val = bounce_detected ? pin->GetOrAddReal("problem", "D_max_bounce", 0.0) : 0.0;
    if (bounce_detected) delept_active = false;
    if (prebounce_core_coupling) {
      auto reject = [](bool invalid, const char *reason) {
        if (invalid) {
          std::cerr << "### FATAL: prebounce_core_coupling requires " << reason
                    << std::endl;
          std::exit(EXIT_FAILURE);
        }
      };
      reject(!recoupling_approach, "problem/recoupling_approach=true");
      reject(!detect_bounce, "problem/detect_bounce=true");
      reject(equilibriate_post_bounce || conservative_bounce_projection ||
                 inject_equilibrium,
             "problem/equilibriate_post_bounce=false, "
             "problem/conservative_bounce_projection=false, "
             "problem/inject_equilibrium=false");
      reject(pmbp->pdyngr->IsFixedEvolution(), "mhd/fixed=false");
      reject(pmbp->pdyngr->eos_policy != DynGRMHD_EOS::eos_compose,
             "mhd/dyn_eos=compose");
      reject(pmbp->pradm1 == nullptr, "radiation_m1 module");
      auto *prad = pmbp->pradm1;
      reject(!radiationm1::equilibrium_available(prad->params.opacity_type),
             "radiation_m1/opacity_type with four-group equilibrium");
      reject(prad->nspecies != radiationm1::equilibrium_nspecies ||
                 prad->nvars != M1_N_IDX + 1,
             "four-group number-evolving M1 layout");
      reject(!prad->params.matter_sources || !prad->params.backreact,
             "radiation_m1/matter_sources=true and radiation_m1/backreact=true");
      reject(pmbp->pz4c != nullptr && !prad->params.backreact_tmunu,
             "radiation_m1/backreact_tmunu=true with Z4c");
      reject(!std::isfinite(rho_trap_cgs) || rho_trap_cgs <= 0.0 ||
                 !std::isfinite(rho_code_to_cgs) || rho_code_to_cgs <= 0.0 ||
                 !std::isfinite(delept_rho_trap) || delept_rho_trap <= 0.0,
             "finite positive problem/rho_trap and density conversion");
      reject(!bounce_detected && !delept_active,
             "pre-bounce problem/deleptonization_method=Liebendoerfer or Simple");
    }
    if (delept_active || detect_bounce) EnrollCCSNTasks(pmy_mesh_);
    user_tasks_enrolled_ = true;
  }

  // Initial data are never replayed on restart.
  if (!restart) {
  // Read parameters -----------------------------------------------------------

  bool interpolate_Phi =
      pin->GetOrAddBoolean("problem", "interpolate_Phi", false);
  bool interpolate_temp =
      pin->GetOrAddBoolean("problem", "interpolate_temp", false);
  bool interpolate_offset_r =
      pin->GetOrAddBoolean("problem", "interpolate_offset_r", false);
  bool id_floor_primitives =
      pin->GetOrAddBoolean("problem", "id_floor_primitives", true);

  const Real Omega_0 = pin->GetOrAddReal("problem", "Omega_0", 0.0);
  const Real Omega_A = pin->GetOrAddReal("problem", "Omega_A", 0.0);
  if (pin->GetOrAddReal("problem", "OmegaGRB_rfe", 0.0) != 0.0 ||
      pin->GetOrAddReal("problem", "OmegaGRB_drtrans", 0.0) != 0.0 ||
      pin->GetOrAddReal("problem", "OmegaGRB_dropfac", 1.0) != 1.0) {
    std::cerr << "### FATAL: OmegaGRB is not implemented; use Omega_0/Omega_A" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real B0_amp = pin->GetOrAddReal("problem", "B0_amp", 0.0);
  const Real B0_rad = pin->GetOrAddReal("problem", "B0_rad", 0.0);
  const Real rho_cut = pin->GetOrAddReal("problem", "rho_cut", -INF);

  // Read stellar profile
  StellarProfile star(pin, interpolate_Phi);

  // Mesh variables for B-field init, PrimToConInit, and Z4c init
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2 * ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  int &ie = indcs.ie, &je = indcs.je, &ke = indcs.ke;
  int nmb1 = pmbp->nmb_thispack - 1;
  auto &size = pmbp->pmb->mb_size;
  auto &adm = pmbp->padm->adm;

  // IC kernel: interpolate profile onto grid (templated for EOS access)
  using namespace Primitive;
  switch (pmbp->pdyngr->eos_policy) {
    case DynGRMHD_EOS::eos_ideal:
      SetCCSNIC<IdealGas, ResetFloor>(pmbp, star, rho_cut,
          Omega_0, Omega_A,
          interpolate_Phi, interpolate_temp, interpolate_offset_r);
      break;
    case DynGRMHD_EOS::eos_piecewise_poly:
      SetCCSNIC<PiecewisePolytrope, ResetFloor>(pmbp, star, rho_cut,
          Omega_0, Omega_A,
          interpolate_Phi, interpolate_temp, interpolate_offset_r);
      break;
    case DynGRMHD_EOS::eos_compose:
      SetCCSNIC<EOSCompOSE<NQTLogs>, ResetFloor>(pmbp, star, rho_cut,
          Omega_0, Omega_A,
          interpolate_Phi, interpolate_temp, interpolate_offset_r);
      SetCCSNIC<EOSCompOSE<NormalLogs>, ResetFloor>(pmbp, star, rho_cut,
          Omega_0, Omega_A,
          interpolate_Phi, interpolate_temp, interpolate_offset_r);
      break;
    case DynGRMHD_EOS::eos_hybrid:
      SetCCSNIC<EOSHybrid<NQTLogs>, ResetFloor>(pmbp, star, rho_cut,
          Omega_0, Omega_A,
          interpolate_Phi, interpolate_temp, interpolate_offset_r);
      SetCCSNIC<EOSHybrid<NormalLogs>, ResetFloor>(pmbp, star, rho_cut,
          Omega_0, Omega_A,
          interpolate_Phi, interpolate_temp, interpolate_offset_r);
      break;
  }

  //----------------------------------------------------------------------------
  // B-field initialization (gated on B0_amp > 0 && B0_rad > 0)

  if (B0_amp > 0.0 && B0_rad > 0.0) {
    int ncells1 = indcs.nx1 + 2 * ng;
    int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
    int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
    int nmb = pmbp->nmb_thispack;

    DvceArray4D<Real> a1, a2, a3;
    Kokkos::realloc(a1, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(a2, nmb, ncells3, ncells2, ncells1);
    Kokkos::realloc(a3, nmb, ncells3, ncells2, ncells1);

    auto &nghbr = pmbp->pmb->nghbr;
    auto &mblev = pmbp->pmb->mb_lev;

    // Pass 1: vector potential at faces
    par_for(
        "pgen_potential", DevExeSpace(), 0, nmb - 1, ks, ke + 1, js, je + 1, is,
        ie + 1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
          Real &x1min = size.d_view(m).x1min;
          Real &x1max = size.d_view(m).x1max;
          int nx1 = indcs.nx1;
          Real x1v = CellCenterX(i - is, nx1, x1min, x1max);
          Real x1f = LeftEdgeX(i - is, nx1, x1min, x1max);

          Real &x2min = size.d_view(m).x2min;
          Real &x2max = size.d_view(m).x2max;
          int nx2 = indcs.nx2;
          Real x2v = CellCenterX(j - js, nx2, x2min, x2max);
          Real x2f = LeftEdgeX(j - js, nx2, x2min, x2max);

          Real &x3min = size.d_view(m).x3min;
          Real &x3max = size.d_view(m).x3max;
          int nx3 = indcs.nx3;
          Real x3v = CellCenterX(k - ks, nx3, x3min, x3max);
          Real x3f = LeftEdgeX(k - ks, nx3, x3min, x3max);

          Real dx1 = size.d_view(m).dx1;
          Real dx2 = size.d_view(m).dx2;
          Real dx3 = size.d_view(m).dx3;

          a1(m, k, j, i) = VectorPotential(x1v, x2f, x3f, B0_amp, B0_rad, 1);
          a2(m, k, j, i) = VectorPotential(x1f, x2v, x3f, B0_amp, B0_rad, 2);
          a3(m, k, j, i) = VectorPotential(x1f, x2f, x3v, B0_amp, B0_rad, 3);

          // AMR: sub-sample at fine/coarse boundaries for A1
          if ((nghbr.d_view(m, 8).lev > mblev.d_view(m) && j == js) ||
              (nghbr.d_view(m, 9).lev > mblev.d_view(m) && j == js) ||
              (nghbr.d_view(m, 10).lev > mblev.d_view(m) && j == js) ||
              (nghbr.d_view(m, 11).lev > mblev.d_view(m) && j == js) ||
              (nghbr.d_view(m, 12).lev > mblev.d_view(m) && j == je + 1) ||
              (nghbr.d_view(m, 13).lev > mblev.d_view(m) && j == je + 1) ||
              (nghbr.d_view(m, 14).lev > mblev.d_view(m) && j == je + 1) ||
              (nghbr.d_view(m, 15).lev > mblev.d_view(m) && j == je + 1) ||
              (nghbr.d_view(m, 24).lev > mblev.d_view(m) && k == ks) ||
              (nghbr.d_view(m, 25).lev > mblev.d_view(m) && k == ks) ||
              (nghbr.d_view(m, 26).lev > mblev.d_view(m) && k == ks) ||
              (nghbr.d_view(m, 27).lev > mblev.d_view(m) && k == ks) ||
              (nghbr.d_view(m, 28).lev > mblev.d_view(m) && k == ke + 1) ||
              (nghbr.d_view(m, 29).lev > mblev.d_view(m) && k == ke + 1) ||
              (nghbr.d_view(m, 30).lev > mblev.d_view(m) && k == ke + 1) ||
              (nghbr.d_view(m, 31).lev > mblev.d_view(m) && k == ke + 1) ||
              (nghbr.d_view(m, 40).lev > mblev.d_view(m) && j == js &&
               k == ks) ||
              (nghbr.d_view(m, 41).lev > mblev.d_view(m) && j == js &&
               k == ks) ||
              (nghbr.d_view(m, 42).lev > mblev.d_view(m) && j == je + 1 &&
               k == ks) ||
              (nghbr.d_view(m, 43).lev > mblev.d_view(m) && j == je + 1 &&
               k == ks) ||
              (nghbr.d_view(m, 44).lev > mblev.d_view(m) && j == js &&
               k == ke + 1) ||
              (nghbr.d_view(m, 45).lev > mblev.d_view(m) && j == js &&
               k == ke + 1) ||
              (nghbr.d_view(m, 46).lev > mblev.d_view(m) && j == je + 1 &&
               k == ke + 1) ||
              (nghbr.d_view(m, 47).lev > mblev.d_view(m) && j == je + 1 &&
               k == ke + 1)) {
            Real xl = x1v + 0.25 * dx1;
            Real xr = x1v - 0.25 * dx1;
            a1(m, k, j, i) =
                0.5 * (VectorPotential(xl, x2f, x3f, B0_amp, B0_rad, 1) +
                       VectorPotential(xr, x2f, x3f, B0_amp, B0_rad, 1));
          }

          // AMR: sub-sample for A2
          if ((nghbr.d_view(m, 0).lev > mblev.d_view(m) && i == is) ||
              (nghbr.d_view(m, 1).lev > mblev.d_view(m) && i == is) ||
              (nghbr.d_view(m, 2).lev > mblev.d_view(m) && i == is) ||
              (nghbr.d_view(m, 3).lev > mblev.d_view(m) && i == is) ||
              (nghbr.d_view(m, 4).lev > mblev.d_view(m) && i == ie + 1) ||
              (nghbr.d_view(m, 5).lev > mblev.d_view(m) && i == ie + 1) ||
              (nghbr.d_view(m, 6).lev > mblev.d_view(m) && i == ie + 1) ||
              (nghbr.d_view(m, 7).lev > mblev.d_view(m) && i == ie + 1) ||
              (nghbr.d_view(m, 24).lev > mblev.d_view(m) && k == ks) ||
              (nghbr.d_view(m, 25).lev > mblev.d_view(m) && k == ks) ||
              (nghbr.d_view(m, 26).lev > mblev.d_view(m) && k == ks) ||
              (nghbr.d_view(m, 27).lev > mblev.d_view(m) && k == ks) ||
              (nghbr.d_view(m, 28).lev > mblev.d_view(m) && k == ke + 1) ||
              (nghbr.d_view(m, 29).lev > mblev.d_view(m) && k == ke + 1) ||
              (nghbr.d_view(m, 30).lev > mblev.d_view(m) && k == ke + 1) ||
              (nghbr.d_view(m, 31).lev > mblev.d_view(m) && k == ke + 1) ||
              (nghbr.d_view(m, 32).lev > mblev.d_view(m) && i == is &&
               k == ks) ||
              (nghbr.d_view(m, 33).lev > mblev.d_view(m) && i == is &&
               k == ks) ||
              (nghbr.d_view(m, 34).lev > mblev.d_view(m) && i == ie + 1 &&
               k == ks) ||
              (nghbr.d_view(m, 35).lev > mblev.d_view(m) && i == ie + 1 &&
               k == ks) ||
              (nghbr.d_view(m, 36).lev > mblev.d_view(m) && i == is &&
               k == ke + 1) ||
              (nghbr.d_view(m, 37).lev > mblev.d_view(m) && i == is &&
               k == ke + 1) ||
              (nghbr.d_view(m, 38).lev > mblev.d_view(m) && i == ie + 1 &&
               k == ke + 1) ||
              (nghbr.d_view(m, 39).lev > mblev.d_view(m) && i == ie + 1 &&
               k == ke + 1)) {
            Real xl = x2v + 0.25 * dx2;
            Real xr = x2v - 0.25 * dx2;
            a2(m, k, j, i) =
                0.5 * (VectorPotential(x1f, xl, x3f, B0_amp, B0_rad, 2) +
                       VectorPotential(x1f, xr, x3f, B0_amp, B0_rad, 2));
          }

          // AMR: sub-sample for A3
          if ((nghbr.d_view(m, 0).lev > mblev.d_view(m) && i == is) ||
              (nghbr.d_view(m, 1).lev > mblev.d_view(m) && i == is) ||
              (nghbr.d_view(m, 2).lev > mblev.d_view(m) && i == is) ||
              (nghbr.d_view(m, 3).lev > mblev.d_view(m) && i == is) ||
              (nghbr.d_view(m, 4).lev > mblev.d_view(m) && i == ie + 1) ||
              (nghbr.d_view(m, 5).lev > mblev.d_view(m) && i == ie + 1) ||
              (nghbr.d_view(m, 6).lev > mblev.d_view(m) && i == ie + 1) ||
              (nghbr.d_view(m, 7).lev > mblev.d_view(m) && i == ie + 1) ||
              (nghbr.d_view(m, 8).lev > mblev.d_view(m) && j == js) ||
              (nghbr.d_view(m, 9).lev > mblev.d_view(m) && j == js) ||
              (nghbr.d_view(m, 10).lev > mblev.d_view(m) && j == js) ||
              (nghbr.d_view(m, 11).lev > mblev.d_view(m) && j == js) ||
              (nghbr.d_view(m, 12).lev > mblev.d_view(m) && j == je + 1) ||
              (nghbr.d_view(m, 13).lev > mblev.d_view(m) && j == je + 1) ||
              (nghbr.d_view(m, 14).lev > mblev.d_view(m) && j == je + 1) ||
              (nghbr.d_view(m, 15).lev > mblev.d_view(m) && j == je + 1) ||
              (nghbr.d_view(m, 16).lev > mblev.d_view(m) && i == is &&
               j == js) ||
              (nghbr.d_view(m, 17).lev > mblev.d_view(m) && i == is &&
               j == js) ||
              (nghbr.d_view(m, 18).lev > mblev.d_view(m) && i == ie + 1 &&
               j == js) ||
              (nghbr.d_view(m, 19).lev > mblev.d_view(m) && i == ie + 1 &&
               j == js) ||
              (nghbr.d_view(m, 20).lev > mblev.d_view(m) && i == is &&
               j == je + 1) ||
              (nghbr.d_view(m, 21).lev > mblev.d_view(m) && i == is &&
               j == je + 1) ||
              (nghbr.d_view(m, 22).lev > mblev.d_view(m) && i == ie + 1 &&
               j == je + 1) ||
              (nghbr.d_view(m, 23).lev > mblev.d_view(m) && i == ie + 1 &&
               j == je + 1)) {
            Real xl = x3v + 0.25 * dx3;
            Real xr = x3v - 0.25 * dx3;
            a3(m, k, j, i) =
                0.5 * (VectorPotential(x1f, x2f, xl, B0_amp, B0_rad, 3) +
                       VectorPotential(x1f, x2f, xr, B0_amp, B0_rad, 3));
          }
        });

    // Pass 2: face-centered B = curl(A)
    auto &b0 = pmbp->pmhd->b0;
    par_for(
        "pgen_Bfc", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          Real dx1 = size.d_view(m).dx1;
          Real dx2 = size.d_view(m).dx2;
          Real dx3 = size.d_view(m).dx3;

          b0.x1f(m, k, j, i) = ((a3(m, k, j + 1, i) - a3(m, k, j, i)) / dx2 -
                                (a2(m, k + 1, j, i) - a2(m, k, j, i)) / dx3);
          b0.x2f(m, k, j, i) = ((a1(m, k + 1, j, i) - a1(m, k, j, i)) / dx3 -
                                (a3(m, k, j, i + 1) - a3(m, k, j, i)) / dx1);
          b0.x3f(m, k, j, i) = ((a2(m, k, j, i + 1) - a2(m, k, j, i)) / dx1 -
                                (a1(m, k, j + 1, i) - a1(m, k, j, i)) / dx2);

          // Extra face components at block edges
          if (i == ie) {
            b0.x1f(m, k, j, i + 1) =
                ((a3(m, k, j + 1, i + 1) - a3(m, k, j, i + 1)) / dx2 -
                 (a2(m, k + 1, j, i + 1) - a2(m, k, j, i + 1)) / dx3);
          }
          if (j == je) {
            b0.x2f(m, k, j + 1, i) =
                ((a1(m, k + 1, j + 1, i) - a1(m, k, j + 1, i)) / dx3 -
                 (a3(m, k, j + 1, i + 1) - a3(m, k, j + 1, i)) / dx1);
          }
          if (k == ke) {
            b0.x3f(m, k + 1, j, i) =
                ((a2(m, k + 1, j, i + 1) - a2(m, k + 1, j, i)) / dx1 -
                 (a1(m, k + 1, j + 1, i) - a1(m, k + 1, j, i)) / dx2);
          }
        });

    // Pass 3: cell-centered B from face averages
    auto &bcc_ = pmbp->pmhd->bcc0;
    par_for(
        "pgen_Bcc", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          bcc_(m, IBX, k, j, i) =
              0.5 * (b0.x1f(m, k, j, i) + b0.x1f(m, k, j, i + 1));
          bcc_(m, IBY, k, j, i) =
              0.5 * (b0.x2f(m, k, j, i) + b0.x2f(m, k, j + 1, i));
          bcc_(m, IBZ, k, j, i) =
              0.5 * (b0.x3f(m, k, j, i) + b0.x3f(m, k + 1, j, i));
        });
  }

  //----------------------------------------------------------------------------
  // Convert primitives to conserved

  // Primitive floor pass (GRA: ApplyPrimitiveFloors before PrimitiveToConserved)
  if (id_floor_primitives) {
    using namespace Primitive;
    switch (pmbp->pdyngr->eos_policy) {
      case DynGRMHD_EOS::eos_ideal:
        ApplyPrimitiveFloorsIC<IdealGas, ResetFloor>(pmbp);
        break;
      case DynGRMHD_EOS::eos_piecewise_poly:
        ApplyPrimitiveFloorsIC<PiecewisePolytrope, ResetFloor>(pmbp);
        break;
      case DynGRMHD_EOS::eos_compose:
        ApplyPrimitiveFloorsIC<EOSCompOSE<NQTLogs>, ResetFloor>(pmbp);
        ApplyPrimitiveFloorsIC<EOSCompOSE<NormalLogs>, ResetFloor>(pmbp);
        break;
      case DynGRMHD_EOS::eos_hybrid:
        ApplyPrimitiveFloorsIC<EOSHybrid<NQTLogs>, ResetFloor>(pmbp);
        ApplyPrimitiveFloorsIC<EOSHybrid<NormalLogs>, ResetFloor>(pmbp);
        break;
    }
  }

  pmbp->pdyngr->PrimToConInit(0, (n1 - 1), 0, (n2 - 1), 0, (n3 - 1));

  // Z4c initialization
  if (pmbp->pz4c != nullptr) {
    switch (indcs.ng) {
    case 2:
      pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
      pmbp->pz4c->ADMConstraints<2>(pmbp);
      break;
    case 3:
      pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
      pmbp->pz4c->ADMConstraints<3>(pmbp);
      break;
    case 4:
      pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
      pmbp->pz4c->ADMConstraints<4>(pmbp);
      break;
    }
    // Init Z4c gauge from ADM.
    {
      auto &z4c_ = pmbp->pz4c->z4c;
      int n1g = indcs.nx1 + 2 * ng;
      int n2g = (indcs.nx2 > 1) ? indcs.nx2 + 2 * ng : 1;
      int n3g = (indcs.nx3 > 1) ? indcs.nx3 + 2 * ng : 1;
      par_for("ccsn_z4c_gauge", DevExeSpace(), 0, nmb1, 0, n3g - 1, 0, n2g - 1,
              0, n1g - 1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
                z4c_.alpha(m, k, j, i)   = adm.alpha(m, k, j, i);
                z4c_.beta_u(m, 0, k, j, i) = adm.beta_u(m, 0, k, j, i);
                z4c_.beta_u(m, 1, k, j, i) = adm.beta_u(m, 1, k, j, i);
                z4c_.beta_u(m, 2, k, j, i) = adm.beta_u(m, 2, k, j, i);
                z4c_.vTheta(m, k, j, i)  = 0.0;
              });
    }
  }
  } // !restart

  // M1 lifecycle: either masked until conservative projection at bounce, or
  // evolved one-way before bounce and recoupled without changing M1 state.
  if (pmbp->pradm1 != nullptr) {
    if (prebounce_core_coupling) {
      // Reuse the ordinary fresh-state zeroing; a restart retains its moments.
      if (!restart) pmbp->pradm1->SetEvolutionEnabled(false);
      pmbp->pradm1->SetEvolutionEnabled(true);
      pmbp->pradm1->SetCouplingEnabled(true);
      pmbp->pradm1->SetCouplingRegionEnabled(!bounce_detected);
    } else if (recoupling_approach) {
      // Fresh refined ICs start with no radiation. Restart must retain the
      // transported state, including a pre-bounce one-way state.
      if (!restart) pmbp->pradm1->SetEvolutionEnabled(false);
      pmbp->pradm1->SetEvolutionEnabled(true);
      pmbp->pradm1->SetCouplingEnabled(bounce_detected);
    } else {
      pmbp->pradm1->SetCouplingEnabled(true);
      pmbp->pradm1->SetEvolutionEnabled(bounce_detected);
    }
  }

  return;
}
