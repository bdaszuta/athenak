//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_update.cpp
//! \brief beam time update for grey M1

#include <iostream>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "globals.hpp"
#include "radiation_m1.hpp"
#include "radiation_m1_calc_closure.hpp"
#include "radiation_m1_helpers.hpp"
#include "radiation_m1_nurates.hpp"
#include "radiation_m1_sources.hpp"
#include "radiation_m1_closure.hpp"
#include "z4c/z4c.hpp"

namespace radiationm1 {

TaskStatus RadiationM1::TimeUpdate(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;

  // Here we are using dynamic_cast to infer which derived type pdyngr is
  auto *ptest_nqt =
      dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                                     Primitive::ResetFloor> *>(
          pmy_pack->pdyngr);
  if (ptest_nqt != nullptr) {
    switch (indcs.ng) {
      case 2:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor, 2>(pdrive, stage);
        break;
      case 3:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor, 3>(pdrive, stage);
        break;
      case 4:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor, 4>(pdrive, stage);
        break;
    }
  }

  auto *ptest_nlog = dynamic_cast<dyngr::DynGRMHDPS<
      Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor> *>(
      pmy_pack->pdyngr);
  if (ptest_nlog != nullptr) {
    switch (indcs.ng) {
      case 2:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                           Primitive::ResetFloor, 2>(pdrive, stage);
        break;
      case 3:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                           Primitive::ResetFloor, 3>(pdrive, stage);
        break;
      case 4:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                           Primitive::ResetFloor, 4>(pdrive, stage);
        break;
    }
  }

  if (!ismhd && !ishydro) {
    switch (indcs.ng) {
      case 2:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor, 2>(pdrive, stage);
        break;
      case 3:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor, 3>(pdrive, stage);
        break;
      case 4:
        return TimeUpdate_<Primitive::EOSCompOSE<Primitive::NQTLogs>,
                           Primitive::ResetFloor, 4>(pdrive, stage);
        break;
    }
  }

  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl;
  std::cout << "Unsupported EOS type!\n";
  abort();
}

//! \file radiation_m1_update.cpp
//! \brief perform update for M1 Steps
//!  1. F^m   = F^k + dt/2 [ A[F^k] + S[F^m]   ]
//!  2. F^k+1 = F^k + dt   [ A[F^m] + S[F^k+1] ]
//!  At each step we solve an implicit problem in the form
//!     F = F^* + cdt S[F]
//!  Where F^* = F^k + cdt A
template <class EOSPolicy, class ErrorPolicy, int NGHOST>
TaskStatus RadiationM1::TimeUpdate_(Driver *d, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;

  auto &u0_ = u0;
  auto &chi_ = chi;
  auto &u1_ = u1;
  auto flx1 = uflx.x1f;
  auto flx2 = uflx.x2f;
  auto flx3 = uflx.x3f;

  auto &eta_0_ = eta_0;
  auto &abs_0_ = abs_0;
  auto &eta_1_ = eta_1;
  auto &abs_1_ = abs_1;
  auto &scat_1_ = scat_1;

  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &nvars_ = nvars;
  auto &nspecies_ = nspecies;
  auto &radiation_mask_ = radiation_mask;

  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &params_ = pmy_pack->pradm1->params;

  DvceArray5D<Real> w0_ = w0;
  DvceArray5D<Real> umhd0_;
  if (ismhd) {
    w0_ = pmy_pack->pmhd->w0;
    umhd0_ = pmy_pack->pmhd->u0;
  }

  auto &BrentFunc_ = pmy_pack->pradm1->BrentFunc;
  auto &HybridsjFunc_ = pmy_pack->pradm1->HybridsjFunc;

  Real mb{};
  if (ismhd || ishydro) {
    Primitive::EOS<EOSPolicy, ErrorPolicy> &eos =
        static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(
            pmy_pack->pdyngr)
            ->eos.ps.GetEOSMutable();
    mb = eos.GetBaryonMass();
  }

  Real beta[2] = {0.5, 1.};
  Real beta_dt = (beta[stage - 1]) * (pmy_pack->pmesh->dt);

  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  bool ismhd_ = ismhd;
  bool ishydro_ = ishydro;

  auto source_regime_ = source_regime;
  const Real mesh_dt = pmy_pack->pmesh->dt;
  // TimeUpdate depends on CalcOpacity: stage-1 coefficients are available here.
  // Store once per cycle; the second stage must not reclassify with its own h.
  if (params_.gra_source_dispatch && stage == 1 && params_.matter_sources) {
    if (params_.gra_source_temp_min_mev > 0.0) {
      auto *pdyngr = static_cast<dyngr::DynGRMHDPS<EOSPolicy, ErrorPolicy> *>(
          pmy_pack->pdyngr);
      Primitive::EOS<EOSPolicy, ErrorPolicy> &eos = pdyngr->eos.ps.GetEOSMutable();
      auto temperature_ = pdyngr->temperature;
      const bool use_c2p_temperature = !pdyngr->IsFixedEvolution();
      const Real code_to_mev =
          eos.GetCodeUnitSystem().TemperatureConversion(Primitive::MakeNuclear());
      par_for(
          "radiation_m1_gra_source_regime", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            const Real rho = w0_(m, IDN, k, j, i);
            Real Y = w0_(m, IYF, k, j, i);
            const Real T = radiation_mask_(m, k, j, i) ? 0.0 :
                (use_c2p_temperature ? temperature_(m, 0, k, j, i) :
                 eos.GetTemperatureFromP(rho / mb, w0_(m, IPR, k, j, i), &Y));
            for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
              source_regime_(m, nuidx, k, j, i) = gra_source_regime(
                  mesh_dt, abs_1_(m, nuidx, k, j, i),
                  scat_1_(m, nuidx, k, j, i), rho, T * code_to_mev, params_);
            }
          });
    } else {
      // Default zero temperature gate needs neither EOS calls nor a fluid EOS.
      par_for(
          "radiation_m1_gra_source_regime", DevExeSpace(), 0, nmb1,
          ks, ke, js, je, is, ie,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
            for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
              source_regime_(m, nuidx, k, j, i) = gra_source_regime(
                  mesh_dt, abs_1_(m, nuidx, k, j, i),
                  scat_1_(m, nuidx, k, j, i), w0_(m, IDN, k, j, i), 0.0, params_);
            }
          });
    }
  }

  par_for(
      "radiation_m1_update", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        // Excised cells: keep radiation fields pinned at zero and skip the flux
        // divergence, geometric, and matter-source updates (mirrors THC's
        // thc_M1_calc_update mask handling).
        if (radiation_mask_(m, k, j, i)) {
          for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
            u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) = 0;
            u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i) = 0;
            u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i) = 0;
            u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i) = 0;
            if (nspecies_ > 1) {
              u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) = 0;
            }
          }
          return;
        }

        // [A] Compute gr quantities: metric, shift, extrinsic curvature, etc.
        Real garr_dd[16];
        Real garr_uu[16];
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_dd{};
        AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gamma_uu{};
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> K_dd{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> gamma_ud{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> beta_u{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> beta_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_u{};
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
            g_dd(a, b) = garr_dd[a + b * 4];
            g_uu(a, b) = garr_uu[a + b * 4];
          }
        }
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            gamma_uu(a, b) =
                g_uu(a + 1, b + 1) +
                adm.beta_u(m, a, k, j, i) * adm.beta_u(m, b, k, j, i) /
                    (adm.alpha(m, k, j, i) * adm.alpha(m, k, j, i));
            K_dd(a, b) = adm.vK_dd(m, a, b, k, j, i);
          }
        }
        pack_n_d(adm.alpha(m, k, j, i), n_d);
        tensor_contract(g_uu, n_d, n_u);
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            gamma_ud(a, b) = (a == b) + n_u(a) * n_d(b);
          }
        }

        pack_beta_u(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                    adm.beta_u(m, 2, k, j, i), beta_u);
        tensor_contract(g_dd, beta_u, beta_d);

        Real gam = adm::SpatialDet(
            adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
            adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
            adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i));
        Real volform = Kokkos::sqrt(gam);

        // [B] Compute derivatives of gr quantities (note: 3-arrays, no \p_t!)
        // [B.1] Derivatives of lapse (\p_i alpha)
        Real ideltax[3] = {1 / mbsize.d_view(m).dx1, 1 / mbsize.d_view(m).dx2,
                           1 / mbsize.d_view(m).dx3};
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d{};
        dalpha_d(0) = Dx<NGHOST>(0, ideltax, adm.alpha, m, k, j, i);
        dalpha_d(1) =
            (multi_d) ? Dx<NGHOST>(1, ideltax, adm.alpha, m, k, j, i) : 0.;
        dalpha_d(2) =
            (three_d) ? Dx<NGHOST>(2, ideltax, adm.alpha, m, k, j, i) : 0.;

        // [B.2] Derivatives of shift (\p_i beta_u(j))
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dbeta_du{};
        for (int a = 0; a < 3; ++a) {
          dbeta_du(0, a) = Dx<NGHOST>(0, ideltax, adm.beta_u, m, a, k, j, i);
          dbeta_du(1, a) =
              (multi_d) ? Dx<NGHOST>(1, ideltax, adm.beta_u, m, a, k, j, i)
                        : 0.;
          dbeta_du(2, a) =
              (three_d) ? Dx<NGHOST>(2, ideltax, adm.beta_u, m, a, k, j, i)
                        : 0.;
        }

        // [B.3] Derivatives of spatial metric (\p_k gamma_ij)
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd{};
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            dg_ddd(0, a, b) =
                Dx<NGHOST>(0, ideltax, adm.g_dd, m, a, b, k, j, i);
            dg_ddd(1, a, b) =
                (multi_d) ? Dx<NGHOST>(1, ideltax, adm.g_dd, m, a, b, k, j, i)
                          : 0.;
            dg_ddd(2, a, b) =
                (three_d) ? Dx<NGHOST>(2, ideltax, adm.g_dd, m, a, b, k, j, i)
                          : 0.;
          }
        }

        // [C] Compute fluid quantities
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d{};
        AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> proj_ud{};

        Real w_lorentz{};
        w_lorentz = get_w_lorentz(w0_(m, IVX, k, j, i), w0_(m, IVY, k, j, i),
                                  w0_(m, IVZ, k, j, i), g_dd);
        pack_u_u(w_lorentz / adm.alpha(m, k, j, i),
                 w0_(m, IVX, k, j, i) -
                     w_lorentz * beta_u(1) / adm.alpha(m, k, j, i),
                 w0_(m, IVY, k, j, i) -
                     w_lorentz * beta_u(2) / adm.alpha(m, k, j, i),
                 w0_(m, IVZ, k, j, i) -
                     w_lorentz * beta_u(3) / adm.alpha(m, k, j, i),
                 u_u);
        pack_v_u(u_u(0), u_u(1), u_u(2), u_u(3), adm.alpha(m, k, j, i),
                 adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
                 adm.beta_u(m, 2, k, j, i), v_u);
        tensor_contract(g_dd, u_u, u_d);
        tensor_contract(g_dd, v_u, v_d);
        calc_proj(u_d, u_u, proj_ud);

        // [E] Compute contribution from flux and geometric sources
        Real rEFN[M1_TOTAL_NUM_SPECIES][5];
        for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
          // [E.1: Contribution from fluxes]
          for (int var = 0; var < nvars_; ++var) {
            rEFN[nuidx][var] =
                -(flx1(m, CombinedIdx(nuidx, var, nvars_), k, j, i + 1) -
                  flx1(m, CombinedIdx(nuidx, var, nvars_), k, j, i)) *
                ideltax[0];
            if (multi_d) {
              rEFN[nuidx][var] +=
                  -(flx2(m, CombinedIdx(nuidx, var, nvars_), k, j + 1, i) -
                    flx2(m, CombinedIdx(nuidx, var, nvars_), k, j, i)) *
                  ideltax[1];
            }
            if (three_d) {
              rEFN[nuidx][var] +=
                  -(flx3(m, CombinedIdx(nuidx, var, nvars_), k + 1, j, i) -
                    flx3(m, CombinedIdx(nuidx, var, nvars_), k, j, i)) *
                  ideltax[2];
            }
          }
          // [E.2 Contribution from geometric sources]
          // Load lab radiation quantities
          if (params_.gr_sources) {
            AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
            const Real E =
                u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i);
            pack_F_d(beta_u(1), beta_u(2), beta_u(3),
                     u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
                     u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
                     u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i),
                     F_d);
            AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> F3_d{};
            F3_d(0) = F_d(1);
            F3_d(1) = F_d(2);
            F3_d(2) = F_d(3);

            AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
            apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d,
                          chi_(m, nuidx, k, j, i), P_dd, params_);
            AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> P3_dd{};
            for (int a = 0; a < 3; ++a) {
              for (int b = 0; b < 3; ++b) {
                P3_dd(a, b) = P_dd(a + 1, b + 1);
              }
            }
            AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> P3_uu{};
            tensor_contract2(gamma_uu, P3_dd, P3_uu);

            // geometric sources
            rEFN[nuidx][M1_E_IDX] +=
                adm.alpha(m, k, j, i) * tensor_dot(P3_uu, K_dd) -
                tensor_dot(gamma_uu, F3_d, dalpha_d);

            for (int a = 0; a < 3; ++a) {
              rEFN[nuidx][a + 1] -= E * dalpha_d(a);
              for (int b = 0; b < 3; ++b) {
                rEFN[nuidx][a + 1] += F3_d(b) * dbeta_du(a, b);
              }
              for (int b = 0; b < 3; ++b) {
                for (int c = 0; c < 3; ++c) {
                  rEFN[nuidx][a + 1] += adm.alpha(m, k, j, i) / 2. *
                                        P3_uu(b, c) * dg_ddd(a, b, c);
                }
              }
            }
          }
        }

        // [F] Compute contribution from matter sources
        Real DrEFN[M1_TOTAL_NUM_SPECIES][5]{};
        Real source_pred[M1_TOTAL_NUM_SPECIES][5]{};
        Real source_base[M1_TOTAL_NUM_SPECIES][5]{};
        bool source_step_is_finite = true;
        for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
          const Real Epred =
              u1_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) +
              beta_dt * rEFN[nuidx][M1_E_IDX];
          source_pred[nuidx][M1_E_IDX] = Epred;

          const Real Fxpred =
              u1_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i) +
              beta_dt * rEFN[nuidx][M1_FX_IDX];
          const Real Fypred =
              u1_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i) +
              beta_dt * rEFN[nuidx][M1_FY_IDX];
          const Real Fzpred =
              u1_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i) +
              beta_dt * rEFN[nuidx][M1_FZ_IDX];
          source_pred[nuidx][M1_FX_IDX] = Fxpred;
          source_pred[nuidx][M1_FY_IDX] = Fypred;
          source_pred[nuidx][M1_FZ_IDX] = Fzpred;

          Real Ebase = Epred;
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fbase_d{};
          pack_F_d(beta_u(1), beta_u(2), beta_u(3), Fxpred, Fypred, Fzpred,
                   Fbase_d);
          const Real Fpred2 = tensor_dot(g_uu, Fbase_d, Fbase_d);
          source_step_is_finite =
              source_step_is_finite && Kokkos::isfinite(Fpred2);
          apply_floor(g_uu, Ebase, Fbase_d, params_);
          source_base[nuidx][M1_E_IDX] = Ebase;
          source_base[nuidx][M1_FX_IDX] = Fbase_d(1);
          source_base[nuidx][M1_FY_IDX] = Fbase_d(2);
          source_base[nuidx][M1_FZ_IDX] = Fbase_d(3);
          if (nspecies_ > 1) {
            const Real Npred =
                u1_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) +
                beta_dt * rEFN[nuidx][M1_N_IDX];
            source_pred[nuidx][M1_N_IDX] = Npred;
            source_base[nuidx][M1_N_IDX] = floor_number(Npred, params_);
          }

          const int nsrc = (nspecies_ > 1) ? 5 : 4;
          for (int var = 0; var < nsrc; ++var) {
            source_step_is_finite =
                source_step_is_finite &&
                Kokkos::isfinite(source_pred[nuidx][var]);
          }
        }
        Real theta{};
        if (params_.matter_sources) {
          for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
            bool source_coeffs_are_finite =
                Kokkos::isfinite(volform * eta_1_(m, nuidx, k, j, i)) &&
                Kokkos::isfinite(abs_1_(m, nuidx, k, j, i)) &&
                Kokkos::isfinite(scat_1_(m, nuidx, k, j, i));
            if (nspecies_ > 1) {
              source_coeffs_are_finite =
                  source_coeffs_are_finite &&
                  Kokkos::isfinite(volform * eta_0_(m, nuidx, k, j, i)) &&
                  Kokkos::isfinite(abs_0_(m, nuidx, k, j, i));
            }
            source_step_is_finite =
                source_step_is_finite && source_coeffs_are_finite;
            if (!source_coeffs_are_finite) {
              continue;
            }

            if (!params_.gra_source_dispatch && params_.src_update == Explicit) {
              // radiation fields
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> S_d{};
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> tS_d{};
              pack_F_d(beta_u(1), beta_u(2), beta_u(3),
                       u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
                       u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
                       u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i),
                       F_d);
              const Real E =
                  u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i);
              AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
              apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E,
                            F_d, chi_(m, nuidx, k, j, i), P_dd, params_);

              // compute radiation quantities in fluid frame
              AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
              assemble_rT(n_d, E, F_d, P_dd, T_dd);
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d{};
              Real J = calc_J_from_rT(T_dd, u_u);
              calc_H_from_rT(T_dd, u_u, proj_ud, H_d);
              const Real Gamma =
                  radiation_number_gamma(E, J, F_d, v_u, w_lorentz, params_);
              enforce_comoving_moment_safety(g_dd, g_uu, u_u, J, H_d, params_);

              // Compute radiation sources
              calc_rad_sources(eta_1_(m, nuidx, k, j, i) * volform,
                               abs_1_(m, nuidx, k, j, i),
                               scat_1_(m, nuidx, k, j, i), u_d, J, H_d, S_d);
              DrEFN[nuidx][M1_E_IDX] =
                  beta_dt * calc_rE_source(adm.alpha(m, k, j, i), n_u, S_d);

              calc_rF_source(adm.alpha(m, k, j, i), gamma_ud, S_d, tS_d);
              DrEFN[nuidx][M1_FX_IDX] = beta_dt * tS_d(1);
              DrEFN[nuidx][M1_FY_IDX] = beta_dt * tS_d(2);
              DrEFN[nuidx][M1_FZ_IDX] = beta_dt * tS_d(3);

              if (nspecies_ > 1) {
                const Real N =
                    u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i);
                DrEFN[nuidx][M1_N_IDX] =
                    beta_dt * adm.alpha(m, k, j, i) *
                    (volform * eta_0_(m, nuidx, k, j, i) -
                     abs_0_(m, nuidx, k, j, i) * N / Gamma);
              }
            }

            if (params_.gra_source_dispatch || params_.src_update == Implicit) {
              const Real chi_old = chi_(m, nuidx, k, j, i);
              const RadiationM1SourceRegime regime = params_.gra_source_dispatch
                  ? static_cast<RadiationM1SourceRegime>(
                        source_regime_(m, nuidx, k, j, i))
                  : SourceLegacy;
              // Boost to the fluid frame, compute fluid matter interaction and
              // boost back. Use these values for implicit solve

              // Use the physicalized transport/geometric predictor to build a
              // closure-safe analytic source seed.
              const Real Estar = source_base[nuidx][M1_E_IDX];
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fstar_d{};
              pack_F_d(beta_u(1), beta_u(2), beta_u(3),
                       source_base[nuidx][M1_FX_IDX],
                       source_base[nuidx][M1_FY_IDX],
                       source_base[nuidx][M1_FZ_IDX], Fstar_d);
              const Real Nstar = source_base[nuidx][M1_N_IDX];

              const Real Epred = source_pred[nuidx][M1_E_IDX];
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fpred_d{};
              pack_F_d(beta_u(1), beta_u(2), beta_u(3),
                       source_pred[nuidx][M1_FX_IDX],
                       source_pred[nuidx][M1_FY_IDX],
                       source_pred[nuidx][M1_FZ_IDX], Fpred_d);
              const Real Npred = source_pred[nuidx][M1_N_IDX];

              // Compute quantities in the fluid frame
              AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
              calc_closure(BrentFunc_, g_dd, g_uu, n_d, w_lorentz, u_u, v_d,
                           proj_ud, Estar, Fstar_d, chi_(m, nuidx, k, j, i),
                           P_dd, params_, params_.closure_type);
              AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> rT_dd{};
              assemble_rT(n_d, Estar, Fstar_d, P_dd, rT_dd);
              Real Jstar = calc_J_from_rT(rT_dd, u_u);
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Hstar_d{};
              calc_H_from_rT(rT_dd, u_u, proj_ud, Hstar_d);
              const Real Gamma_star = radiation_number_gamma(
                  Estar, Jstar, Fstar_d, v_u, w_lorentz, params_);
              Real Jstar_number = Jstar;
              if (nspecies_ > 1) {
                AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Hstar_number_d{};
                for (int a = 0; a < 4; ++a) {
                  Hstar_number_d(a) = Hstar_d(a);
                }
                enforce_comoving_moment_safety(
                    g_dd, g_uu, u_u, Jstar_number, Hstar_number_d, params_);
              }

              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fnew_d{};
              Real Enew{};
              Real Jnew{};
              if (params_.gra_source_dispatch) {
                if (regime == SourceNonstiff) {
                  // FE evaluates sources on Epred/Fpred in source_update_ll;
                  // this physicalized state is only its failure fallback.
                  Enew = Estar;
                  for (int a = 0; a < 4; ++a) Fnew_d(a) = Fstar_d(a);
                } else {
                  // Equilibrium publishes this literal GRA approximation;
                  // ordinary stiff uses it only as the existing BE solver seed.
                  gra_approximate_thick_update(
                      beta_dt, adm.alpha(m, k, j, i), n_u, w_lorentz, Jstar,
                      Hstar_d, v_d, v_u, volform * eta_1_(m, nuidx, k, j, i),
                      abs_1_(m, nuidx, k, j, i), scat_1_(m, nuidx, k, j, i),
                      Enew, Fnew_d);
                  chi_(m, nuidx, k, j, i) = 1.0 / 3.0;
                }
              } else {
                // Estimate interaction with matter (unchanged legacy seed)
                const Real dtau = beta_dt * (adm.alpha(m, k, j, i) / w_lorentz);
                Jnew = (Jstar + dtau * eta_1_(m, nuidx, k, j, i) * volform) /
                            (1. + dtau * abs_1_(m, nuidx, k, j, i));

                // Only three components of H^a are independent H^0 is found by
                // requiring H^a u_a = 0
                const Real khat =
                    (abs_1_(m, nuidx, k, j, i) + scat_1_(m, nuidx, k, j, i));
                AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Hnew_d{};
                for (int a = 1; a < 4; ++a) {
                  Hnew_d(a) = Hstar_d(a) / (1. + dtau * khat);
                }
                Hnew_d(0) = 0.0;
                for (int a = 1; a < 4; ++a) {
                  Hnew_d(0) -= Hnew_d(a) * (u_u(a) / u_u(0));
                }

                // Update Tmunu.  Repair J/H before using them to build both xi
                // and Tmunu; a finite positive comoving J remains valid even
                // when J <= rad_E_floor.
                const bool comoving_state_is_valid =
                    enforce_comoving_moment_safety(g_dd, g_uu, u_u, Jnew, Hnew_d, params_);
                const Real H2 = comoving_moment_squared(g_dd, g_uu, u_u, Hnew_d);
                const Real normH = Kokkos::sqrt(H2);
                const Real inormH = (normH > 0. ? 1. / normH : 0.);
                const Real xi = comoving_state_is_valid
                                    ? normH / Jnew
                                    : 0.;
                chi_(m, nuidx, k, j, i) = closure_fun(xi, params_.closure_type);

                const Real dthick = 3. * (1. - chi_(m, nuidx, k, j, i)) / 2.;
                const Real dthin = 1. - dthick;

                for (int a = 0; a < 4; ++a) {
                  for (int b = a; b < 4; ++b) {
                    rT_dd(a, b) =
                        Jnew * u_d(a) * u_d(b) + Hnew_d(a) * u_d(b) +
                        Hnew_d(b) * u_d(a) +
                        dthin * Jnew *
                            ((Hnew_d(a) * inormH) * (Hnew_d(b) * inormH)) +
                        dthick * Jnew * (g_dd(a, b) + u_d(a) * u_d(b)) / 3.;
                  }
                }

                // Boost back to the lab frame.  A nonfinite analytic state is not
                // repaired into a charged matter interaction: retain the physicalized
                // predictor and disable the complete source step for this cell.
                Enew = calc_J_from_rT(rT_dd, n_u);
                calc_H_from_rT(rT_dd, n_u, gamma_ud, Fnew_d);
              }
              bool analytic_state_is_finite =
                  radiation_state_is_finite(g_uu, Enew, Fnew_d);
              if (!analytic_state_is_finite) {
                source_step_is_finite = false;
                Enew = Estar;
                for (int a = 0; a < 4; ++a) {
                  Fnew_d(a) = Fstar_d(a);
                }
              }
              apply_floor(g_uu, Enew, Fnew_d, params_);

              const Real E_analytic = Enew;
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_analytic_d{};
              for (int a = 0; a < 4; ++a) {
                F_analytic_d(a) = Fnew_d(a);
              }
              const Real chi_analytic = chi_(m, nuidx, k, j, i);

              SrcSignal src_signal = SrcFail;
              if (analytic_state_is_finite && regime != SourceEquilibrium) {
                src_signal = source_update(
                    BrentFunc_, HybridsjFunc_, beta_dt, adm.alpha(m, k, j, i),
                    g_dd, g_uu, n_d, n_u, gamma_ud, u_d, u_u, v_d, v_u,
                    proj_ud, w_lorentz, Estar, Fstar_d, Epred, Fpred_d,
                    volform * eta_1_(m, nuidx, k, j, i),
                    abs_1_(m, nuidx, k, j, i), scat_1_(m, nuidx, k, j, i),
                    chi_(m, nuidx, k, j, i), Enew, Fnew_d, params_,
                    params_.closure_type, regime);
              }
              if (src_signal == SrcFail) {
                Enew = E_analytic;
                for (int a = 0; a < 4; ++a) {
                  Fnew_d(a) = F_analytic_d(a);
                }
                chi_(m, nuidx, k, j, i) = chi_analytic;
              }
              apply_floor(g_uu, Enew, Fnew_d, params_);

              // Update closure for the accepted nonlinear or analytic state.
              calc_closure(BrentFunc_, g_dd, g_uu, n_d, w_lorentz, u_u, v_d,
                           proj_ud, Enew, Fnew_d, chi_(m, nuidx, k, j, i), P_dd,
                           params_, params_.closure_type);

              // Compute new radiation energy density in the fluid frame
              AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
              assemble_rT(n_d, Enew, Fnew_d, P_dd, T_dd);
              Jnew = calc_J_from_rT(T_dd, u_u);

              // Compute changes in radiation energy and momentum
              DrEFN[nuidx][M1_E_IDX] = Enew - Epred;
              DrEFN[nuidx][M1_FX_IDX] = Fnew_d(1) - Fpred_d(1);
              DrEFN[nuidx][M1_FY_IDX] = Fnew_d(2) - Fpred_d(2);
              DrEFN[nuidx][M1_FZ_IDX] = Fnew_d(3) - Fpred_d(3);

              if (nspecies_ > 1 && params_.src_update == Implicit) {
                // Match the late safety repair formerly applied to Jstar before
                // deriving the equilibrium average energy.
                Jstar = Jstar_number;
                AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Hnumber_d{};
                calc_H_from_rT(T_dd, u_u, proj_ud, Hnumber_d);
                const Real Gamma = radiation_number_gamma(
                    Enew, Jnew, Fnew_d, v_u, w_lorentz, params_);
                enforce_comoving_moment_safety(g_dd, g_uu, u_u, Jnew, Hnumber_d,
                                               params_);

                if (params_.source_therm_limit < 0 ||
                    beta_dt * abs_0_(m, nuidx, k, j, i) <
                        params_.source_therm_limit) {
                  //
                  // N^k+1 = N^* + dt ( eta - abs N^k+1 )
                  DrEFN[nuidx][M1_N_IDX] =
                      (Npred + beta_dt * adm.alpha(m, k, j, i) * volform *
                                   eta_0_(m, nuidx, k, j, i)) /
                          (1 + beta_dt * adm.alpha(m, k, j, i) *
                                   abs_0_(m, nuidx, k, j, i) / Gamma) -
                      Npred;
                } else {
                  // The neutrino number density is updated assuming the
                  // neutrino average energies are those of the equilibrium
                  const Real nueave = Gamma_star * Jstar / Nstar;
                  DrEFN[nuidx][M1_N_IDX] =
                      (nueave > 0 ? Gamma * Jnew / nueave - Npred : 0.0);
                }
              }
              if (nspecies_ > 1 && params_.src_update == Explicit) {
                // GRA overrides energy dispatch, never the selected ATK number
                // mode. Evaluate explicit number on the old stage state/closure.
                const Real Eold =
                    u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i);
                AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fold_d{};
                pack_F_d(beta_u(1), beta_u(2), beta_u(3),
                    u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
                    u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
                    u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i), Fold_d);
                AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> Pnumber_dd{};
                apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud,
                              Eold, Fold_d, chi_old, Pnumber_dd, params_);
                AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> Tnumber_dd{};
                assemble_rT(n_d, Eold, Fold_d, Pnumber_dd, Tnumber_dd);
                const Real Jold = calc_J_from_rT(Tnumber_dd, u_u);
                const Real Gamma_old = radiation_number_gamma(
                    Eold, Jold, Fold_d, v_u, w_lorentz, params_);
                const Real Nold =
                    u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i);
                DrEFN[nuidx][M1_N_IDX] = beta_dt * adm.alpha(m, k, j, i) *
                    (volform * eta_0_(m, nuidx, k, j, i) -
                     abs_0_(m, nuidx, k, j, i) * Nold / Gamma_old);
              }
            }
          }

          for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
            const int nsrc = (nspecies_ > 1) ? 5 : 4;
            for (int var = 0; var < nsrc; ++var) {
              source_step_is_finite =
                  source_step_is_finite &&
                  Kokkos::isfinite(DrEFN[nuidx][var]) &&
                  Kokkos::isfinite(source_pred[nuidx][var] +
                                   DrEFN[nuidx][var]);
            }
          }

          // [G] Limit signed material exchange while retaining each species'
          // depletion bound. Budgets use the E/N increments stored after floors.
          theta = (source_step_is_finite && params_.source_limiter != 0.0)
                      ? 1.0
                      : 0.0;
          if (source_step_is_finite && params_.theta_limiter &&
              params_.source_limiter > 0.0) {
            const Real limiter = params_.source_limiter;
            const Real tau =
                (ismhd_ || ishydro_) ? umhd0_(m, IEN, k, j, i) : 0.;
            const Real cons_dens =
                (ismhd_ || ishydro_)
                    ? volform * w_lorentz * w0_(m, IDN, k, j, i)
                    : 0.;
            const Real Y_e =
                (nspecies_ > 1 && (ismhd_ || ishydro_))
                    ? w0_(m, IYF, k, j, i) : 0.;

            Real dE_net_full = 0.0;
            Real Npred_e[2]{}, Nbase_e[2]{}, dN_e[2]{};
            for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
              const Real Epred = source_pred[nuidx][M1_E_IDX];
              const Real Ebase = source_base[nuidx][M1_E_IDX];
              const Real dE = DrEFN[nuidx][M1_E_IDX];
              const Real dE_full =
                  Kokkos::fmax(params_.rad_E_floor, Epred + dE) - Ebase;
              const Real dE_loss_full = Kokkos::fmax(-dE_full, 0.0);
              const Real dE_loss_max = limiter * Kokkos::fmax(Ebase, 0.0);
              if (dE_loss_full > dE_loss_max) {
                theta = Kokkos::fmin(theta, dE_loss_max / (-dE));
              }
              dE_net_full += dE_full;

              if (nspecies_ > 1) {
                const Real Npred = source_pred[nuidx][M1_N_IDX];
                const Real Nbase = source_base[nuidx][M1_N_IDX];
                const Real dN = DrEFN[nuidx][M1_N_IDX];
                const Real dN_full =
                    Kokkos::fmax(params_.rad_N_floor, Npred + dN) - Nbase;
                const Real dN_loss_max = limiter * Kokkos::fmax(Nbase, 0.0);
                if (-dN_full > dN_loss_max) {
                  theta = Kokkos::fmin(theta, dN_loss_max / (-dN));
                }
                if (nuidx < 2) {
                  Npred_e[nuidx] = Npred;
                  Nbase_e[nuidx] = Nbase;
                  dN_e[nuidx] = dN;
                }
              }
            }

            const Real dE_net_max = limiter * Kokkos::fmax(tau, 0.0);
            if (!(Kokkos::isfinite(tau) && Kokkos::isfinite(dE_net_full) &&
                  Kokkos::isfinite(dE_net_max))) {
              theta = 0.0;
            } else if (dE_net_full > dE_net_max) {
              // Each floored energy is convex in theta and its baseline is its
              // theta=0 value. Thus this ratio also bounds net cooling between
              // floor crossings; net heating does not spend the cooling budget.
              theta = Kokkos::fmin(theta, dE_net_max / dE_net_full);
            }
            theta = Kokkos::fmax(0.0, Kokkos::fmin(1.0, theta));

            Real dQ_up_max = 0.0, dQ_down_max = 0.0;
            if (nspecies_ > 1) {
              if (!(Kokkos::isfinite(cons_dens) && cons_dens > 0.0 &&
                    Kokkos::isfinite(Y_e))) {
                theta = 0.0;
              } else {
                dQ_up_max = limiter * cons_dens *
                    Kokkos::fmax(params_.source_Ye_max - Y_e, 0.0);
                dQ_down_max = limiter * cons_dens *
                    Kokkos::fmax(Y_e - params_.source_Ye_min, 0.0);
                theta = limit_source_ye(Npred_e, Nbase_e, dN_e,
                    params_.rad_N_floor, mb, dQ_down_max, dQ_up_max, theta);
              }
            }

            // Verify actual rounded scalar exchanges before [H] publishes any
            // species. If rounding invalidates a bound, try one smaller ceiling
            // and recompute the floor-aware composition bound; smaller theta is
            // not assumed safe. No budget slack or iterative source solve is added.
            for (int trial = 0; trial < 2 && theta > 0.0; ++trial) {
              Real dE_net = 0.0;
              bool within_bounds = true;
              for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
                const Real dE = Kokkos::fmax(params_.rad_E_floor,
                    source_pred[nuidx][M1_E_IDX] +
                    theta * DrEFN[nuidx][M1_E_IDX]) - source_base[nuidx][M1_E_IDX];
                dE_net += dE;
                within_bounds = within_bounds && Kokkos::isfinite(dE) &&
                    dE >= -limiter * Kokkos::fmax(source_base[nuidx][M1_E_IDX], 0.0);
                if (nspecies_ > 1) {
                  const Real dN = Kokkos::fmax(params_.rad_N_floor,
                      source_pred[nuidx][M1_N_IDX] +
                      theta * DrEFN[nuidx][M1_N_IDX]) - source_base[nuidx][M1_N_IDX];
                  within_bounds = within_bounds && Kokkos::isfinite(dN) &&
                      dN >= -limiter * Kokkos::fmax(source_base[nuidx][M1_N_IDX], 0.0);
                }
              }
              within_bounds = within_bounds && Kokkos::isfinite(dE_net) &&
                              dE_net <= dE_net_max;
              if (nspecies_ > 1) {
                const Real dQ = source_electron_change(theta, Npred_e, Nbase_e, dN_e,
                                                      params_.rad_N_floor, mb);
                within_bounds = within_bounds && Kokkos::isfinite(dQ) &&
                                dQ >= -dQ_down_max && dQ <= dQ_up_max;
              }
              if (within_bounds) break;
              if (trial == 1 || !Kokkos::isfinite(dE_net)) {
                theta = 0.0;
              } else {
                theta /= 2.0;
                if (nspecies_ > 1) {
                  theta = limit_source_ye(Npred_e, Nbase_e, dN_e,
                      params_.rad_N_floor, mb, dQ_down_max, dQ_up_max, theta);
                }
              }
            }
          }  // if (params_.theta_limiter && params_.source_limiter > 0.0)

          source_step_is_finite =
              source_step_is_finite && Kokkos::isfinite(theta);
          if (source_step_is_finite) {
            for (int nuidx = 0; nuidx < nspecies_; ++nuidx) {
              const Real Ecandidate = source_pred[nuidx][M1_E_IDX] +
                                      theta * DrEFN[nuidx][M1_E_IDX];
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fcandidate_d{};
              pack_F_d(
                  beta_u(1), beta_u(2), beta_u(3),
                  source_pred[nuidx][M1_FX_IDX] +
                      theta * DrEFN[nuidx][M1_FX_IDX],
                  source_pred[nuidx][M1_FY_IDX] +
                      theta * DrEFN[nuidx][M1_FY_IDX],
                  source_pred[nuidx][M1_FZ_IDX] +
                      theta * DrEFN[nuidx][M1_FZ_IDX],
                  Fcandidate_d);
              source_step_is_finite =
                  source_step_is_finite && Kokkos::isfinite(Ecandidate) &&
                  Kokkos::isfinite(tensor_dot(g_uu, Fcandidate_d, Fcandidate_d));
              if (nspecies_ > 1) {
                source_step_is_finite =
                    source_step_is_finite &&
                    Kokkos::isfinite(source_pred[nuidx][M1_N_IDX] +
                                     theta * DrEFN[nuidx][M1_N_IDX]);
              }
            }
          }
          if (!source_step_is_finite) {
            theta = 0.0;
          }
        } else {
          theta = 0.0;
        }  // if (params_.matter_sources)

        // [H] Update fields
        const bool apply_source_step = source_step_is_finite && theta > 0.0;
        // Sum only the two material budgets that were limited in [G]. Momentum
        // feedback retains the existing per-species repaired-flux accounting.
        Real dE_source = 0.0;
        Real dN_electron[2]{};
        for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
          const Real Ebase = source_base[nuidx][M1_E_IDX];
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fbase_d{};
          pack_F_d(beta_u(1), beta_u(2), beta_u(3),
                   source_base[nuidx][M1_FX_IDX],
                   source_base[nuidx][M1_FY_IDX],
                   source_base[nuidx][M1_FZ_IDX], Fbase_d);

          Real Ecandidate = Ebase;
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fcandidate_d{};
          for (int a = 0; a < 4; ++a) {
            Fcandidate_d(a) = Fbase_d(a);
          }
          if (apply_source_step) {
            Ecandidate = source_pred[nuidx][M1_E_IDX] +
                         theta * DrEFN[nuidx][M1_E_IDX];
            const Real Fxf = source_pred[nuidx][M1_FX_IDX] +
                             theta * DrEFN[nuidx][M1_FX_IDX];
            const Real Fyf = source_pred[nuidx][M1_FY_IDX] +
                             theta * DrEFN[nuidx][M1_FY_IDX];
            const Real Fzf = source_pred[nuidx][M1_FZ_IDX] +
                             theta * DrEFN[nuidx][M1_FZ_IDX];
            pack_F_d(adm.beta_u(m, 0, k, j, i),
                     adm.beta_u(m, 1, k, j, i),
                     adm.beta_u(m, 2, k, j, i), Fxf, Fyf, Fzf, Fcandidate_d);
            apply_floor(g_uu, Ecandidate, Fcandidate_d, params_);
          }

          Real Ef{}, dE_actual{};
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Ff_d{}, dF_actual_d{};
          select_source_candidate(apply_source_step, Ebase, Fbase_d, Ecandidate,
                                  Fcandidate_d, Ef, Ff_d, dE_actual, dF_actual_d);

          AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> Pfinal_dd{};
          calc_closure(BrentFunc_, g_dd, g_uu, n_d, w_lorentz, u_u, v_d,
                       proj_ud, Ef, Ff_d, chi_(m, nuidx, k, j, i), Pfinal_dd,
                       params_, params_.closure_type);

          u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i) = Ef;
          u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i) = Ff_d(1);
          u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i) = Ff_d(2);
          u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i) = Ff_d(3);

          Real dN_actual = 0.0;
          if (nspecies_ > 1) {
            const Real Nbase = source_base[nuidx][M1_N_IDX];
            Real Nf = Nbase;
            if (apply_source_step) {
              Nf = floor_number(source_pred[nuidx][M1_N_IDX] +
                                    theta * DrEFN[nuidx][M1_N_IDX],
                                params_);
            }
            dN_actual = Nf - Nbase;
            u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) = Nf;
          }

          dE_source += dE_actual;
          if (nspecies_ > 1 && nuidx < 2) dN_electron[nuidx] = dN_actual;

          const Real dFx_actual = dF_actual_d(1);
          const Real dFy_actual = dF_actual_d(2);
          const Real dFz_actual = dF_actual_d(3);
          if (params_.backreact && stage == 2 && (ismhd_)) {
            umhd0_(m, IM1, k, j, i) -= dFx_actual;
            umhd0_(m, IM2, k, j, i) -= dFy_actual;
            umhd0_(m, IM3, k, j, i) -= dFz_actual;
          }
        }
        if (params_.backreact && stage == 2 && ismhd_) {
          // Apply net E/Q once, using the same order and electron signs as [G].
          // Canceling species exchange must not round through intermediate
          // material states before it is combined.
          umhd0_(m, IEN, k, j, i) -= dE_source;
          if (nspecies_ > 1) {
            umhd0_(m, IYF, k, j, i) += mb * (dN_electron[1] - dN_electron[0]);
          }
        }
      });
  return TaskStatus::complete;
}
}  // namespace radiationm1
