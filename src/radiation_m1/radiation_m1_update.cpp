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

            if (params_.src_update == Explicit) {
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
              Real h_fac{};
              const bool comoving_state_is_valid =
                  enforce_comoving_moment_safety(g_dd, g_uu, u_u, J, H_d, params_, &h_fac);

              // Gamma is the Eulerian 0-component of the number current
              // fnu = u + H/J, evaluated from the same (possibly
              // safety-rescaled) H used for the number source.
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> fnu_u{};
              assemble_fnu(g_uu, T_dd, u_u, J, h_fac, fnu_u);
              const Real Gamma_candidate = adm.alpha(m, k, j, i) * fnu_u(0);
              const Real Gamma = comoving_state_is_valid &&
                                         Kokkos::isfinite(Gamma_candidate) &&
                                         Gamma_candidate > 0.0
                                     ? Gamma_candidate
                                     : w_lorentz;

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

            if (params_.src_update == Implicit) {
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
              Real Gamma_star{};
              Real Jstar_number = Jstar;
              if (nspecies_ > 1) {
                AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Hstar_number_d{};
                for (int a = 0; a < 4; ++a) {
                  Hstar_number_d(a) = Hstar_d(a);
                }
                Real hstar_fac{};
                const bool star_number_state_is_valid = enforce_comoving_moment_safety(
                    g_dd, g_uu, u_u, Jstar_number, Hstar_number_d, params_, &hstar_fac);
                AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> fstar_u{};
                assemble_fnu(g_uu, rT_dd, u_u, Jstar_number, hstar_fac, fstar_u);
                const Real Gamma_star_candidate = adm.alpha(m, k, j, i) * fstar_u(0);
                Gamma_star = star_number_state_is_valid &&
                                     Kokkos::isfinite(Gamma_star_candidate) &&
                                     Gamma_star_candidate > 0.0
                                 ? Gamma_star_candidate
                                 : w_lorentz;
              }

              // Estimate interaction with matter
              const Real dtau = beta_dt * (adm.alpha(m, k, j, i) / w_lorentz);
              Real Jnew = (Jstar + dtau * eta_1_(m, nuidx, k, j, i) * volform) /
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

              // Boost back to the lab frame
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fnew_d{};
              Real Enew = calc_J_from_rT(rT_dd, n_u);
              calc_H_from_rT(rT_dd, n_u, gamma_ud, Fnew_d);
              apply_floor(g_uu, Enew, Fnew_d, params_);

              const Real E_analytic = Enew;
              AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_analytic_d{};
              for (int a = 0; a < 4; ++a) {
                F_analytic_d(a) = Fnew_d(a);
              }
              const Real chi_analytic = chi_(m, nuidx, k, j, i);

              const SrcSignal src_signal = source_update(
                  BrentFunc_, HybridsjFunc_, beta_dt, adm.alpha(m, k, j, i),
                  g_dd, g_uu, n_d, n_u, gamma_ud, u_d, u_u, v_d, v_u, proj_ud,
                  w_lorentz, Estar, Fstar_d, Epred, Fpred_d,
                  volform * eta_1_(m, nuidx, k, j, i),
                  abs_1_(m, nuidx, k, j, i), scat_1_(m, nuidx, k, j, i),
                  chi_(m, nuidx, k, j, i), Enew, Fnew_d, params_,
                  params_.closure_type);
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

              if (nspecies_ > 1) {
                // Match the late safety repair formerly applied to Jstar before
                // deriving the equilibrium average energy.
                Jstar = Jstar_number;
                AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Hnumber_d{};
                calc_H_from_rT(T_dd, u_u, proj_ud, Hnumber_d);
                Real hnumber_fac{};
                const bool number_state_is_valid =
                    enforce_comoving_moment_safety(g_dd, g_uu, u_u, Jnew, Hnumber_d,
                                                   params_, &hnumber_fac);

                AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> fnumber_u{};
                assemble_fnu(g_uu, T_dd, u_u, Jnew, hnumber_fac, fnumber_u);
                const Real Gamma_candidate = adm.alpha(m, k, j, i) * fnumber_u(0);
                const Real Gamma = number_state_is_valid &&
                                           Kokkos::isfinite(Gamma_candidate) &&
                                           Gamma_candidate > 0.0
                                       ? Gamma_candidate
                                       : w_lorentz;

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

          // [G] Limit the source step using conservative one-sided bounds on
          // the increments that [H] stores after applying the E/N floors.
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
                (ismhd_ || ishydro_) ? w0_(m, IYF, k, j, i) : 0.;

            Real dE_gain_full = 0.0;
            Real dYe_up_full = 0.0;
            Real dYe_down_full = 0.0;
            Real dYe_up_rate = 0.0;
            Real dYe_down_rate = 0.0;

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
              dE_gain_full += Kokkos::fmax(dE_full, 0.0);

              if (nspecies_ > 1) {
                const Real Npred = source_pred[nuidx][M1_N_IDX];
                const Real Nbase = source_base[nuidx][M1_N_IDX];
                const Real dN = DrEFN[nuidx][M1_N_IDX];
                const Real dN_full =
                    Kokkos::fmax(params_.rad_N_floor, Npred + dN) - Nbase;
                const Real dN_loss_full = Kokkos::fmax(-dN_full, 0.0);
                const Real dN_gain_full = Kokkos::fmax(dN_full, 0.0);
                const Real dN_loss_max =
                    limiter * Kokkos::fmax(Nbase, 0.0);

                if (dN_loss_full > dN_loss_max) {
                  theta = Kokkos::fmin(theta, dN_loss_max / (-dN));
                }

                const Real dN_loss_rate =
                    (Npred > params_.rad_N_floor && dN < 0.0) ? -dN : 0.0;
                if (nuidx == 0) {
                  dYe_up_full += mb * dN_loss_full;
                  dYe_up_rate += mb * dN_loss_rate;
                  dYe_down_full += mb * dN_gain_full;
                  dYe_down_rate += mb * dN_gain_full;
                } else if (nuidx == 1) {
                  dYe_up_full += mb * dN_gain_full;
                  dYe_up_rate += mb * dN_gain_full;
                  dYe_down_full += mb * dN_loss_full;
                  dYe_down_rate += mb * dN_loss_rate;
                }
              }
            }

            const Real dE_gain_max = limiter * Kokkos::fmax(tau, 0.0);
            if (dE_gain_full > dE_gain_max) {
              theta = Kokkos::fmin(theta, dE_gain_max / dE_gain_full);
            }

            if (nspecies_ > 1) {
              if (!(Kokkos::isfinite(cons_dens) && cons_dens > 0.0)) {
                theta = 0.0;
              } else {
                dYe_up_full /= cons_dens;
                dYe_down_full /= cons_dens;
                dYe_up_rate /= cons_dens;
                dYe_down_rate /= cons_dens;

                const Real dYe_up_max =
                    limiter *
                    Kokkos::fmax(params_.source_Ye_max - Y_e, 0.0);
                const Real dYe_down_max =
                    limiter *
                    Kokkos::fmax(Y_e - params_.source_Ye_min, 0.0);

                if (dYe_up_full > dYe_up_max) {
                  theta = Kokkos::fmin(theta, dYe_up_max / dYe_up_rate);
                }
                if (dYe_down_full > dYe_down_max) {
                  theta =
                      Kokkos::fmin(theta, dYe_down_max / dYe_down_rate);
                }
              }
            }

            theta = Kokkos::fmax(0.0, Kokkos::fmin(1.0, theta));
          }  //  if (params_.theta_limiter && params_.source_limiter >= 0)

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
        for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
          const Real Ebase = source_base[nuidx][M1_E_IDX];
          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Fbase_d{};
          pack_F_d(beta_u(1), beta_u(2), beta_u(3),
                   source_base[nuidx][M1_FX_IDX],
                   source_base[nuidx][M1_FY_IDX],
                   source_base[nuidx][M1_FZ_IDX], Fbase_d);

          AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Ff_d{};
          Real Ef = Ebase;
          for (int a = 0; a < 4; ++a) {
            Ff_d(a) = Fbase_d(a);
          }
          if (apply_source_step) {
            Ef = source_pred[nuidx][M1_E_IDX] +
                 theta * DrEFN[nuidx][M1_E_IDX];
            const Real Fxf = source_pred[nuidx][M1_FX_IDX] +
                             theta * DrEFN[nuidx][M1_FX_IDX];
            const Real Fyf = source_pred[nuidx][M1_FY_IDX] +
                             theta * DrEFN[nuidx][M1_FY_IDX];
            const Real Fzf = source_pred[nuidx][M1_FZ_IDX] +
                             theta * DrEFN[nuidx][M1_FZ_IDX];
            pack_F_d(adm.beta_u(m, 0, k, j, i),
                     adm.beta_u(m, 1, k, j, i),
                     adm.beta_u(m, 2, k, j, i), Fxf, Fyf, Fzf, Ff_d);
            apply_floor(g_uu, Ef, Ff_d, params_);
          }

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

          const Real dE_actual = Ef - Ebase;
          const Real dFx_actual = Ff_d(1) - Fbase_d(1);
          const Real dFy_actual = Ff_d(2) - Fbase_d(2);
          const Real dFz_actual = Ff_d(3) - Fbase_d(3);
          if (params_.backreact && stage == 2 && (ismhd_)) {
            umhd0_(m, IEN, k, j, i) -= dE_actual;
            umhd0_(m, IM1, k, j, i) -= dFx_actual;
            umhd0_(m, IM2, k, j, i) -= dFy_actual;
            umhd0_(m, IM3, k, j, i) -= dFz_actual;
            if (nspecies_ > 1) {
              if (nuidx == 0) {
                umhd0_(m, IYF, k, j, i) -= mb * dN_actual;
              } else if (nuidx == 1) {
                umhd0_(m, IYF, k, j, i) += mb * dN_actual;
              }
            }
          }
        }
      });
  return TaskStatus::complete;
}
}  // namespace radiationm1
