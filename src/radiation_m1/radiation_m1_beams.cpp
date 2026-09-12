//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_beams.cpp
//! \brief beam initial data for grey M1

#include <coordinates/cell_locations.hpp>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "radiation_m1.hpp"

namespace radiationm1 {

// Beams from left wall of domain (1d, single species)
void ApplyBeamSources1D(Mesh *pmesh) {
  auto &indcs = pmesh->mb_indcs;
  int &is = indcs.is;
  int nmb1 = pmesh->pmb_pack->nmb_thispack - 1;
  auto nvars_ = pmesh->pmb_pack->pradm1->nvars;
  auto &nspecies_ = pmesh->pmb_pack->pradm1->nspecies;
  auto &mb_bcs = pmesh->pmb_pack->pmb->mb_bcs;

  int &ng = indcs.ng;
  auto &u0_ = pmesh->pmb_pack->pradm1->u0;
  auto &beam_source_1_vals_ = pmesh->pmb_pack->pradm1->rad_m1_beam.beam_source_vals;


  par_for(
      "radiation_m1_beams_populate_1d", DevExeSpace(), 0, nmb1, 0, nvars_-2,
      KOKKOS_LAMBDA(int m, int n) {
        switch (mb_bcs.d_view(m, BoundaryFace::inner_x1)) {
          case BoundaryFlag::outflow:
            for (int i = 0; i < ng; ++i) {
                for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
                  u0_(m, CombinedIdx(nuidx, n, nvars_), 0, 0, is - i - 1) = beam_source_1_vals_(n); 
                }
            }
            break;
          default:
            break;
        }
      });
}

// Beams from left wall of domain (2d, single species)
void ApplyBeamSources2D(Mesh *pmesh) {
  auto &indcs = pmesh->mb_indcs;
  int &is = indcs.is;
  int &js = indcs.js;

  int nmb1 = pmesh->pmb_pack->nmb_thispack - 1;
  auto nvars_ = pmesh->pmb_pack->pradm1->nvars;
  auto &nspecies_ = pmesh->pmb_pack->pradm1->nspecies;
  auto &mb_bcs = pmesh->pmb_pack->pmb->mb_bcs;
  auto &size = pmesh->pmb_pack->pmb->mb_size;

  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;

  auto &u0_ = pmesh->pmb_pack->pradm1->u0;
  auto &beam_source_1_vals_ = pmesh->pmb_pack->pradm1->rad_m1_beam.beam_source_vals;
  auto &beam_source_1_y1_ = pmesh->pmb_pack->pradm1->rad_m1_beam.beam_ymin;
  auto &beam_source_1_y2_ = pmesh->pmb_pack->pradm1->rad_m1_beam.beam_ymax;


  par_for(
      "radiation_m1_beams_populate_2d", DevExeSpace(), 0, nmb1, 0, nvars_-2, 0,
      (n3 - 1), 0, (n2 - 1), KOKKOS_LAMBDA(int m, int n, int k, int j) {
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real x2 = CellCenterX(j - js, nx2, x2min, x2max);

        switch (mb_bcs.d_view(m, BoundaryFace::inner_x1)) {
          case BoundaryFlag::outflow:
            if (beam_source_1_y1_ <= x2 && x2 <= beam_source_1_y2_) {
              for (int i = 0; i < ng; ++i) {
                for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
                  u0_(m, CombinedIdx(nuidx, n, nvars_), k, j, is - i - 1) = beam_source_1_vals_(n);
                }
              }
            }
            break;
          default:
            break;
        }
      });
}

// Beam for the M1 beam test around black hole (2d, single species)
void ApplyBeamSourcesBlackHole(Mesh *pmesh) {
  auto &indcs = pmesh->mb_indcs;
  int &is = indcs.is;
  int &js = indcs.js;

  int nmb1 = pmesh->pmb_pack->nmb_thispack - 1;
  auto &mb_bcs = pmesh->pmb_pack->pmb->mb_bcs;
  auto &size = pmesh->pmb_pack->pmb->mb_size;
  int &ng = indcs.ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2 * ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2 * ng) : 1;
  auto &params_ = pmesh->pmb_pack->pradm1->params;
  auto nvars_ = pmesh->pmb_pack->pradm1->nvars;
  auto &nspecies_ = pmesh->pmb_pack->pradm1->nspecies;

  auto &u0_ = pmesh->pmb_pack->pradm1->u0;
  adm::ADM::ADM_vars &adm = pmesh->pmb_pack->padm->adm;

  auto &beam_source_1_y1_ = pmesh->pmb_pack->pradm1->rad_m1_beam.beam_ymin;
  auto &beam_source_1_y2_ = pmesh->pmb_pack->pradm1->rad_m1_beam.beam_ymax;


  par_for(
      "radiation_m1_beam_populate_black_hole", DevExeSpace(), 0, nmb1, 0, (n3 - 1), 0,
      (n2 - 1), KOKKOS_LAMBDA(int m, int k, int j) {
        Real &x2min = size.d_view(m).x2min;
        Real &x2max = size.d_view(m).x2max;
        int nx2 = indcs.nx2;
        Real x2 = CellCenterX(j - js, nx2, x2min, x2max);

        switch (mb_bcs.d_view(m, BoundaryFace::inner_x1)) {
          case BoundaryFlag::vacuum:
            if (beam_source_1_y1_ <= x2 && x2 <= beam_source_1_y2_) {
              for (int i = 0; i < ng; ++i) {
                // Use the destination ghost's geometry, not the loop counter.
                const int ig = is - i - 1;
                // Calculate the 4-metric and sqrt(det(3-metric)).
                Real garr_dd[16];
                AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_dd{};
                // Real garr_uu[16];
                // adm::SpacetimeUpperMetric(
                //     adm.alpha(m, k, j, ig), adm.beta_u(m, 0, k, j, ig),
                //     adm.beta_u(m, 1, k, j, ig), adm.beta_u(m, 2, k, j, ig),
                //     adm.g_dd(m, 0, 0, k, j, ig), adm.g_dd(m, 0, 1, k, j, ig),
                //     adm.g_dd(m, 0, 2, k, j, ig), adm.g_dd(m, 1, 1, k, j, ig),
                //     adm.g_dd(m, 1, 2, k, j, ig), adm.g_dd(m, 2, 2, k, j, ig), garr_uu);
                adm::SpacetimeMetric(
                    adm.alpha(m, k, j, ig), adm.beta_u(m, 0, k, j, ig),
                    adm.beta_u(m, 1, k, j, ig), adm.beta_u(m, 2, k, j, ig),
                    adm.g_dd(m, 0, 0, k, j, ig), adm.g_dd(m, 0, 1, k, j, ig),
                    adm.g_dd(m, 0, 2, k, j, ig), adm.g_dd(m, 1, 1, k, j, ig),
                    adm.g_dd(m, 1, 2, k, j, ig), adm.g_dd(m, 2, 2, k, j, ig), garr_dd);
                for (int a = 0; a < 4; ++a) {
                  for (int b = 0; b < 4; ++b) {
                    g_dd(a, b) = garr_dd[a + b * 4];
                  }
                }
                const Real gam = adm::SpatialDet(
                    adm.g_dd(m, 0, 0, k, j, ig), adm.g_dd(m, 0, 1, k, j, ig),
                    adm.g_dd(m, 0, 2, k, j, ig), adm.g_dd(m, 1, 1, k, j, ig),
                    adm.g_dd(m, 1, 2, k, j, ig), adm.g_dd(m, 2, 2, k, j, ig));
                const Real volform = Kokkos::sqrt(gam);

                // Solve g_xx*a^2 + 2*beta_x*a + beta2 = alpha^2*(1-rad_eps).
                const Real g_xx = adm.g_dd(m, 0, 0, k, j, ig);
                Real beta_x = 0;
                Real beta2 = 0;
                for (int idx = 0; idx < 3; idx++) {
                  beta_x += adm.g_dd(m, 0, idx, k, j, ig) * adm.beta_u(m, idx, k, j, ig);
                  for (int idx2 = 0; idx2 < 3; idx2++) {
                    beta2 += adm.g_dd(m, idx, idx2, k, j, ig) *
                             adm.beta_u(m, idx, k, j, ig) * adm.beta_u(m, idx2, k, j, ig);
                  }
                }
                const Real a =
                    (-beta_x + sqrt(beta_x * beta_x - g_xx *
                                    (beta2 - adm.alpha(m, k, j, ig) *
                                     adm.alpha(m, k, j, ig) *
                                        (1 - params_.rad_eps)))) / g_xx;

                Real E = volform * 1.;
                Real Fx = a * E / adm.alpha(m, k, j, ig) +
                          adm.beta_u(m, 0, k, j, ig) * E / adm.alpha(m, k, j, ig);
                Real Fy = adm.beta_u(m, 1, k, j, ig) * E / adm.alpha(m, k, j, ig);
                Real Fz = adm.beta_u(m, 2, k, j, ig) * E / adm.alpha(m, k, j, ig);
                AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_u{};
                AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
                F_u(0) = 0;
                F_u(1) = Fx;
                F_u(2) = Fy;
                F_u(3) = Fz;
                tensor_contract(g_dd, F_u, F_d);
                // apply_floor(g_uu, E, F_d, params_);
                for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
                  u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, ig) = E;
                  u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, ig) = F_d(1);
                  u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, ig) = F_d(2);
                  u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, ig) = F_d(3);
                  if (nspecies_ > 1) {
                    u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, ig) = 1.0;
                  }
                }
              }
            }
            else {
              for (int i = 0; i < ng; ++i) {
                const int ig = is - i - 1;
                for (int nuidx = 0; nuidx < nspecies_; nuidx++) {
                  u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, ig) =
                      0.0;
                  u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, ig) =
                      0.0;
                  u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, ig) =
                      0.0;
                  u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, ig) =
                      0.0;
                  if (nspecies_ > 1) {
                    u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, ig) =
                        0.0;
                  }
                }
              }
            }
            break;
          default:
            break;
        }
      });
}

}  // namespace radiationm1