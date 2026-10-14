//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_fluxes.cpp
//! \brief calculate 3D fluxes for grey M1

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_helpers.hpp"
#include "eos/eos.hpp"
#include "mhd/mhd.hpp"
#include "reconstruct/recon.hpp"


namespace radiationm1 {

// TODO: this code is computing some quantities, such as the metric many
//       times for each grid point... it could be made much more efficient
KOKKOS_INLINE_FUNCTION
void CalcFlux(const int m, const int k, const int j, const int i, const int nuidx,
              const int dir, const DvceArray5D<Real> &u0_, const DvceArray5D<Real> &w0_,
              const DvceArray5D<Real> &chi_,
              const adm::ADM::ADM_vars &adm, const RadiationM1Params params_,
              const int nvars_, const int nspecies_, Real flux[5],
              Real &cmax) {
  // load 4-metric, 3-metric inverse, shift, normal
  Real garr_dd[16];
  Real garr_uu[16];
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_dd{};
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gamma_uu{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> beta_u{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> beta_d{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_d{};
  pack_n_d(adm.alpha(m, k, j, i), n_d);
  adm::SpacetimeMetric(adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i),
                       adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                       adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                       adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                       adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), garr_dd);
  adm::SpacetimeUpperMetric(
      adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
      adm.beta_u(m, 2, k, j, i), adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
      adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i), adm.g_dd(m, 1, 2, k, j, i),
      adm.g_dd(m, 2, 2, k, j, i), garr_uu);
  pack_beta_u(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
              adm.beta_u(m, 2, k, j, i), beta_u);
  tensor_contract(g_dd, beta_u, beta_d);
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      g_dd(a, b) = garr_dd[a + b * 4];
      g_uu(a, b) = garr_uu[a + b * 4];
    }
  }
  for (int a = 1; a < 4; ++a) {
    for (int b = 1; b < 4; ++b) {
      gamma_uu(a - 1, b - 1) =
          garr_uu[a + b * 4] +
          beta_u(a) * beta_u(b) / (adm.alpha(m, k, j, i) * adm.alpha(m, k, j, i));
    }
  }

  // load Lorentz factor, four velocity, three velocity, projection
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
               w_lorentz * adm.beta_u(m, 0, k, j, i) / adm.alpha(m, k, j, i),
           w0_(m, IVY, k, j, i) -
               w_lorentz * adm.beta_u(m, 1, k, j, i) / adm.alpha(m, k, j, i),
           w0_(m, IVZ, k, j, i) -
               w_lorentz * adm.beta_u(m, 2, k, j, i) / adm.alpha(m, k, j, i),
           u_u);
  pack_v_u(u_u(0), u_u(1), u_u(2), u_u(3), adm.alpha(m, k, j, i),
           adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
           adm.beta_u(m, 2, k, j, i), v_u);
  tensor_contract(g_dd, u_u, u_d);
  tensor_contract(g_dd, v_u, v_d);
  calc_proj(u_d, u_u, proj_ud);

  const Real E = u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i);
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_u{};
  pack_F_d(beta_u(1), beta_u(2), beta_u(3),
           u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
           u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
           u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i), F_d);
  tensor_contract(g_uu, F_d, F_u);

  // lab frame pressure
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> P_ud{};
  apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d,
                chi_(m, nuidx, k, j, i), P_dd, params_);
  tensor_contract(g_uu, P_dd, P_ud);

  // compute fluid frame quantities
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
  assemble_rT(n_d, E, F_d, P_dd, T_dd);
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> H_d{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> fnu_u{};

  Real J = calc_J_from_rT(T_dd, u_u);
  calc_H_from_rT(T_dd, u_u, proj_ud, H_d);
  const Real Gamma =
      radiation_number_gamma(E, J, F_d, v_u, w_lorentz, params_);
  Real h_fac{};
  enforce_comoving_moment_safety(g_dd, g_uu, u_u, J, H_d, params_, &h_fac);

  Real N{};
  if (nspecies_ > 1) {
    N = u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i);
  }
  assemble_fnu(g_uu, T_dd, u_u, J, h_fac, fnu_u);

  // Density uses the GRA scalar conversion above; retain the existing safety
  // projection of the spatial number current used for transport.

  // Note that nnu is densitized here
  Real nnu{};
  if (nspecies_ > 1) {
    nnu = N / Gamma;
  }

  flux[0] = calc_E_flux(adm.alpha(m, k, j, i), beta_u, E, F_u, dir);
  flux[1] = calc_F_flux(adm.alpha(m, k, j, i), beta_u, F_d, P_ud, dir, 1);
  flux[2] = calc_F_flux(adm.alpha(m, k, j, i), beta_u, F_d, P_ud, dir, 2);
  flux[3] = calc_F_flux(adm.alpha(m, k, j, i), beta_u, F_d, P_ud, dir, 3);
  if (nspecies_ > 1) {
    flux[4] = adm.alpha(m, k, j, i) * nnu * fnu_u(dir);
    // number_advection = hydro: only the part relative to the fluid, alpha n (fnu - u)^i
    // = alpha n h_fac H^i/J, is built here; the part moving with the fluid,
    // alpha n u^i = (n/rho) F_D, is added on the faces from the hydro's rest-mass flux
    // (CalculateFluxes)
    if (params_.number_advection_hydro) {
      flux[4] = adm.alpha(m, k, j, i) * nnu * (fnu_u(dir) - u_u(dir));
    }
  } else {
    flux[4] = 0;
  }

  // Speed of light -- note that gamma_uu has NDIM=3
  const Real clam[2] = {
      adm.alpha(m, k, j, i) * Kokkos::sqrt(gamma_uu(dir - 1, dir - 1)) - beta_u(dir),
      adm.alpha(m, k, j, i) * Kokkos::sqrt(gamma_uu(dir - 1, dir - 1)) + beta_u(dir)};
  const Real clight = Kokkos::fmax(Kokkos::abs(clam[0]), Kokkos::abs(clam[1]));
  cmax = clight;
}

//! \fn Real CalcNnuW
//  \brief (N/Gamma) W: the densitized fluid-frame number density sqrt(gamma) n times the
//  Lorentz factor, so that (N/Gamma) W / D = n/rho (neutrinos per unit rest mass), with
//  D = sqrt(gamma) W rho the hydro's conserved density. Same closure and the same
//  number/frame conversion (radiation_number_gamma, from the unrepaired J) as CalcFlux,
//  so that this n is the n of the flux.
KOKKOS_INLINE_FUNCTION
Real CalcNnuW(const int m, const int k, const int j, const int i, const int nuidx,
              const DvceArray5D<Real> &u0_, const DvceArray5D<Real> &w0_,
              const DvceArray5D<Real> &chi_, const adm::ADM::ADM_vars &adm,
              const RadiationM1Params &params_, const int nvars_) {
  Real garr_dd[16];
  Real garr_uu[16];
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_uu{};
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g_dd{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> n_d{};
  pack_n_d(adm.alpha(m, k, j, i), n_d);
  adm::SpacetimeMetric(adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i),
                       adm.beta_u(m, 1, k, j, i), adm.beta_u(m, 2, k, j, i),
                       adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
                       adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i),
                       adm.g_dd(m, 1, 2, k, j, i), adm.g_dd(m, 2, 2, k, j, i), garr_dd);
  adm::SpacetimeUpperMetric(
      adm.alpha(m, k, j, i), adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
      adm.beta_u(m, 2, k, j, i), adm.g_dd(m, 0, 0, k, j, i), adm.g_dd(m, 0, 1, k, j, i),
      adm.g_dd(m, 0, 2, k, j, i), adm.g_dd(m, 1, 1, k, j, i), adm.g_dd(m, 1, 2, k, j, i),
      adm.g_dd(m, 2, 2, k, j, i), garr_uu);
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      g_dd(a, b) = garr_dd[a + b * 4];
      g_uu(a, b) = garr_uu[a + b * 4];
    }
  }
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_u{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> u_d{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> proj_ud{};
  const Real w_lorentz = get_w_lorentz(w0_(m, IVX, k, j, i), w0_(m, IVY, k, j, i),
                                       w0_(m, IVZ, k, j, i), g_dd);
  pack_u_u(w_lorentz / adm.alpha(m, k, j, i),
           w0_(m, IVX, k, j, i) -
               w_lorentz * adm.beta_u(m, 0, k, j, i) / adm.alpha(m, k, j, i),
           w0_(m, IVY, k, j, i) -
               w_lorentz * adm.beta_u(m, 1, k, j, i) / adm.alpha(m, k, j, i),
           w0_(m, IVZ, k, j, i) -
               w_lorentz * adm.beta_u(m, 2, k, j, i) / adm.alpha(m, k, j, i),
           u_u);
  pack_v_u(u_u(0), u_u(1), u_u(2), u_u(3), adm.alpha(m, k, j, i),
           adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
           adm.beta_u(m, 2, k, j, i), v_u);
  tensor_contract(g_dd, u_u, u_d);
  tensor_contract(g_dd, v_u, v_d);
  calc_proj(u_d, u_u, proj_ud);

  const Real E = u0_(m, CombinedIdx(nuidx, M1_E_IDX, nvars_), k, j, i);
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> F_d{};
  pack_F_d(adm.beta_u(m, 0, k, j, i), adm.beta_u(m, 1, k, j, i),
           adm.beta_u(m, 2, k, j, i),
           u0_(m, CombinedIdx(nuidx, M1_FX_IDX, nvars_), k, j, i),
           u0_(m, CombinedIdx(nuidx, M1_FY_IDX, nvars_), k, j, i),
           u0_(m, CombinedIdx(nuidx, M1_FZ_IDX, nvars_), k, j, i), F_d);
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> P_dd{};
  apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d,
                chi_(m, nuidx, k, j, i), P_dd, params_);
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> T_dd{};
  assemble_rT(n_d, E, F_d, P_dd, T_dd);
  const Real J = calc_J_from_rT(T_dd, u_u);
  const Real Gamma = radiation_number_gamma(E, J, F_d, v_u, w_lorentz, params_);
  return u0_(m, CombinedIdx(nuidx, M1_N_IDX, nvars_), k, j, i) / Gamma * w_lorentz;
}

TaskStatus RadiationM1::CalculateFluxes(Driver *pdrive, int stage) {
  RegionIndcs &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &ie = indcs.ie;
  int &js = indcs.js, &je = indcs.je;
  int &ks = indcs.ks, &ke = indcs.ke;

  auto nmb1 = pmy_pack->nmb_thispack - 1;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &nspecies_ = pmy_pack->pradm1->nspecies;
  auto nvars_ = pmy_pack->pradm1->nvars;
  auto &params_ = pmy_pack->pradm1->params;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;

  auto &u0_ = pmy_pack->pradm1->u0;
  auto &chi_ = pmy_pack->pradm1->chi;
  auto &xi_ = pmy_pack->pradm1->xi;
  adm::ADM::ADM_vars &adm = pmy_pack->padm->adm;

  DvceArray5D<Real> w0_ = w0;
  if (ismhd) {
    w0_ = pmy_pack->pmhd->w0;
  }

  auto &abs_1_ = pmy_pack->pradm1->abs_1;
  auto &scat_1_ = pmy_pack->pradm1->scat_1;

  // number_advection = hydro: neutrinos per unit rest mass y = n/rho = (N/Gamma) W / D in
  // all cells the reconstruction reads. D is the hydro's conserved density at the time
  // level of this M1 stage (u^n = pmhd->u1 in stage 1, the midpoint in stage 2), so that
  // a uniform y stays exactly uniform, as a uniform Ye does in the hydro. The face states
  // of y are then built with the hydro's own reconstruction, and the part of the number
  // flux moving with the fluid is F_acc * y_upwind, F_acc = the hydro's step-averaged
  // rest-mass flux.
  const bool num_hydro = params_.number_advection_hydro && (nspecies_ > 1);
  auto ynu_ = ynu;
  auto ynu_l_ = ynu_l;
  auto ynu_r_ = ynu_r;
  DvceArray4D<Real> macc1_, macc2_, macc3_;
  if (num_hydro) {
    if (!ismhd || !pmy_pack->pmhd->accumulate_mass_flux) {
      std::cerr << "Error: radiation_m1/number_advection = hydro needs the MHD "
                << "step-averaged mass flux (<mhd> block)" << std::endl;
      exit(EXIT_FAILURE);
    }
    macc1_ = pmy_pack->pmhd->mflx_acc.x1f;
    macc2_ = pmy_pack->pmhd->mflx_acc.x2f;
    macc3_ = pmy_pack->pmhd->mflx_acc.x3f;
    auto &hu0 = pmy_pack->pmhd->u0;
    auto &hu1 = pmy_pack->pmhd->u1;
    const bool first = (stage == 1);
    const int ng1 = indcs.ng - 1;  // the widest (WENOZ/PPM) stencil reaches 3 cells out
    const int yil = is - ng1, yiu = ie + ng1;
    const int yjl = multi_d ? js - ng1 : js, yju = multi_d ? je + ng1 : je;
    const int ykl = three_d ? ks - ng1 : ks, yku = three_d ? ke + ng1 : ke;
    par_for("radiation_m1_ynu", DevExeSpace(), 0, nmb1, ykl, yku, yjl, yju, yil, yiu, 0,
            nspecies_ - 1,
            KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                          const int nuidx) {
      const Real dref = first ? hu1(m, IDN, k, j, i)
                              : 0.5 * (hu1(m, IDN, k, j, i) + hu0(m, IDN, k, j, i));
      ynu_(m, nuidx, k, j, i) =
          CalcNnuW(m, k, j, i, nuidx, u0_, w0_, chi_, adm, params_, nvars_) / dref;
    });
  }
  auto recon_ = ismhd ? pmy_pack->pmhd->recon_method : ReconstructionMethod::dc;

  //--------------------------------------------------------------------------------------
  // i-direction

  auto &flx1_ = uflx.x1f;
  int il = is, iu = ie + 1, jl = js, ju = je, kl = ks, ku = ke;
  if (num_hydro) {  // face states of y: cells [il-1, iu] give faces [il, iu]
    ReconDispatch<IVX>(recon_, "radiation_m1_ynu_x1", nmb1, kl, ku, jl, ju, il - 1, iu,
                       pmy_pack->pmhd->peos->eos_data, false, nspecies_, ynu_, ynu_l_,
                       ynu_r_);
  }

  par_for(
      "radiation_m1_flux_x1", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu, 0,
      nspecies_ - 1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i, const int nuidx) {
        int dir = 1;
        Real flux_j[5]{};
        Real flux_jp1[5]{};
        Real cmax_j, cmax_jp1;
        CalcFlux(m, k, j, i - 1, nuidx, dir, u0_, w0_, chi_, adm, params_, nvars_,
                 nspecies_, flux_j, cmax_j);
        CalcFlux(m, k, j, i, nuidx, dir, u0_, w0_, chi_, adm, params_, nvars_,
                 nspecies_, flux_jp1, cmax_jp1);

        Real flux_jp12_lo[5]{};
        Real flux_jp12_ho[5]{};
        Real cmax_jp12 = Kokkos::fmax(cmax_j, cmax_jp1);

        Real kappa_ave = 0;
        if (params_.matter_sources) {
          kappa_ave = 0.5 * (abs_1_(m, nuidx, k, j, i - 1) + abs_1_(m, nuidx, k, j, i) +
                             scat_1_(m, nuidx, k, j, i - 1) + scat_1_(m, nuidx, k, j, i));
        }
        Real A_jp12;
        if (params_.flux_hybrid == FluxHybridMinModC) {
          // GRA HybridizeMinModC: hybridise on the closure flux factor averaged
          // over the two cells flanking the face.
          A_jp12 = Kokkos::fmin(1.0, 0.5 * (xi_(m, nuidx, k, j, i - 1) +
                                            xi_(m, nuidx, k, j, i)));
        } else {
          A_jp12 = Kokkos::fmin(1., 1. / (kappa_ave * mbsize.d_view(m).dx1));
        }

        for (int momidx = 0; momidx < nvars_; ++momidx) {
          const Real ujm = u0_(m, CombinedIdx(nuidx, momidx, nvars_), k, j, i - 2);
          const Real uj = u0_(m, CombinedIdx(nuidx, momidx, nvars_), k, j, i - 1);
          const Real ujp = u0_(m, CombinedIdx(nuidx, momidx, nvars_), k, j, i);
          const Real ujpp = u0_(m, CombinedIdx(nuidx, momidx, nvars_), k, j, i + 1);

          const Real dup = ujpp - ujp;
          const Real duc = ujp - uj;
          const Real dum = uj - ujm;

          bool sawtooth = false;
          Real phi_jp12 = 0;
          if (dup * duc > 0 && dum * duc > 0) {
            phi_jp12 = minmod2(dum / duc, dup / duc, params_.minmod_theta);
          } else if (dup * duc < 0 && dum * duc < 0) {
            sawtooth = true;
          }

          flux_jp12_ho[momidx] = (flux_j[momidx] + flux_jp1[momidx]) / 2.;
          flux_jp12_lo[momidx] =
              (flux_j[momidx] + flux_jp1[momidx]) / 2. -
              cmax_jp12 *
                  (u0_(m, CombinedIdx(nuidx, momidx, nvars_), k, j, i) -
                   u0_(m, CombinedIdx(nuidx, momidx, nvars_), k, j, i - 1)) /
                  2.;

          flx1_(m, CombinedIdx(nuidx, momidx, nvars_), k, j, i) =
              flux_jp12_ho[momidx] - (sawtooth ? 1 : A_jp12) * (1 - phi_jp12) *
                                         (flux_jp12_ho[momidx] - flux_jp12_lo[momidx]);
          if (num_hydro && momidx == M1_N_IDX) {  // moving with the fluid
            const Real fd = macc1_(m, k, j, i);
            flx1_(m, CombinedIdx(nuidx, momidx, nvars_), k, j, i) +=
                fd * Kokkos::fmax(fd >= 0. ? ynu_l_(m, nuidx, k, j, i)
                                            : ynu_r_(m, nuidx, k, j, i), 0.);
          }
        }
      });

  //--------------------------------------------------------------------------------------
  // j-direction

  if (multi_d) {
    auto &flx2_ = uflx.x2f;
    il = is, iu = ie, jl = js, ju = je + 1, kl = ks, ku = ke;
    if (num_hydro) {
      ReconDispatch<IVY>(recon_, "radiation_m1_ynu_x2", nmb1, kl, ku, jl - 1, ju, il, iu,
                         pmy_pack->pmhd->peos->eos_data, false, nspecies_, ynu_, ynu_l_,
                         ynu_r_);
    }

    par_for(
        "radiation_m1_flux_x2", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu, 0,
        nspecies_ - 1,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                      const int nuidx) {
          int dir = 2;
          Real flux_j[5]{};
          Real flux_jp1[5]{};
          Real cmax_j, cmax_jp1;
          CalcFlux(m, k, j - 1, i, nuidx, dir, u0_, w0_, chi_, adm, params_,
                   nvars_, nspecies_, flux_j, cmax_j);
          CalcFlux(m, k, j, i, nuidx, dir, u0_, w0_, chi_, adm, params_, nvars_,
                   nspecies_, flux_jp1, cmax_jp1);

          Real flux_jp12_lo[5]{};
          Real flux_jp12_ho[5]{};
          Real cmax_jp12 = Kokkos::fmax(cmax_j, cmax_jp1);

          Real kappa_ave = 0;
          if (params_.matter_sources) {
            kappa_ave =
                0.5 * (abs_1_(m, nuidx, k, j - 1, i) + abs_1_(m, nuidx, k, j, i) +
                       scat_1_(m, nuidx, k, j - 1, i) + scat_1_(m, nuidx, k, j, i));
          }
          Real A_jp12;
          if (params_.flux_hybrid == FluxHybridMinModC) {
            A_jp12 = Kokkos::fmin(1.0, 0.5 * (xi_(m, nuidx, k, j - 1, i) +
                                              xi_(m, nuidx, k, j, i)));
          } else {
            A_jp12 = Kokkos::fmin(1., 1. / (kappa_ave * mbsize.d_view(m).dx2));
          }

          for (int momidx = 0; momidx < nvars_; ++momidx) {
            const Real ujm = u0_(m, CombinedIdx(nuidx, momidx, nvars_), k, j - 2, i);
            const Real uj = u0_(m, CombinedIdx(nuidx, momidx, nvars_), k, j - 1, i);
            const Real ujp = u0_(m, CombinedIdx(nuidx, momidx, nvars_), k, j, i);
            const Real ujpp = u0_(m, CombinedIdx(nuidx, momidx, nvars_), k, j + 1, i);

            const Real dup = ujpp - ujp;
            const Real duc = ujp - uj;
            const Real dum = uj - ujm;

            bool sawtooth = false;
            Real phi_jp12 = 0;
            if (dup * duc > 0 && dum * duc > 0) {
              phi_jp12 = minmod2(dum / duc, dup / duc, params_.minmod_theta);
            } else if (dup * duc < 0 && dum * duc < 0) {
              sawtooth = true;
            }

            flux_jp12_ho[momidx] = (flux_j[momidx] + flux_jp1[momidx]) / 2.;
            flux_jp12_lo[momidx] =
                (flux_j[momidx] + flux_jp1[momidx]) / 2. -
                cmax_jp12 *
                    (u0_(m, CombinedIdx(nuidx, momidx, nvars_), k, j, i) -
                     u0_(m, CombinedIdx(nuidx, momidx, nvars_), k, j - 1, i)) /
                    2.;

            flx2_(m, CombinedIdx(nuidx, momidx, nvars_), k, j, i) =
                flux_jp12_ho[momidx] - (sawtooth ? 1 : A_jp12) * (1 - phi_jp12) *
                                           (flux_jp12_ho[momidx] - flux_jp12_lo[momidx]);
            if (num_hydro && momidx == M1_N_IDX) {  // moving with the fluid
              const Real fd = macc2_(m, k, j, i);
              flx2_(m, CombinedIdx(nuidx, momidx, nvars_), k, j, i) +=
                  fd * Kokkos::fmax(fd >= 0. ? ynu_l_(m, nuidx, k, j, i)
                                              : ynu_r_(m, nuidx, k, j, i), 0.);
            }
          }
        });
  }

  //--------------------------------------------------------------------------------------
  // k-direction

  if (three_d) {
    auto &flx3_ = uflx.x3f;
    il = is, iu = ie, jl = js, ju = je, kl = ks, ku = ke + 1;
    if (num_hydro) {
      ReconDispatch<IVZ>(recon_, "radiation_m1_ynu_x3", nmb1, kl - 1, ku, jl, ju, il, iu,
                         pmy_pack->pmhd->peos->eos_data, false, nspecies_, ynu_, ynu_l_,
                         ynu_r_);
    }

    par_for(
        "radiation_m1_flux_x3", DevExeSpace(), 0, nmb1, kl, ku, jl, ju, il, iu, 0,
        nspecies_ - 1,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                      const int nuidx) {
          int dir = 3;
          Real flux_j[5]{};
          Real flux_jp1[5]{};
          Real cmax_j, cmax_jp1;
          CalcFlux(m, k - 1, j, i, nuidx, dir, u0_, w0_, chi_, adm, params_,
                   nvars_, nspecies_, flux_j, cmax_j);
          CalcFlux(m, k, j, i, nuidx, dir, u0_, w0_, chi_, adm, params_, nvars_,
                   nspecies_, flux_jp1, cmax_jp1);

          Real flux_jp12_lo[5]{};
          Real flux_jp12_ho[5]{};
          Real cmax_jp12 = Kokkos::fmax(cmax_j, cmax_jp1);

          Real kappa_ave = 0;
          if (params_.matter_sources) {
            kappa_ave =
                0.5 * (abs_1_(m, nuidx, k - 1, j, i) + abs_1_(m, nuidx, k, j, i) +
                       scat_1_(m, nuidx, k - 1, j, i) + scat_1_(m, nuidx, k, j, i));
          }
          Real A_jp12;
          if (params_.flux_hybrid == FluxHybridMinModC) {
            A_jp12 = Kokkos::fmin(1.0, 0.5 * (xi_(m, nuidx, k - 1, j, i) +
                                              xi_(m, nuidx, k, j, i)));
          } else {
            A_jp12 = Kokkos::fmin(1., 1. / (kappa_ave * mbsize.d_view(m).dx3));
          }

          for (int momidx = 0; momidx < nvars_; ++momidx) {
            const Real ujm = u0_(m, CombinedIdx(nuidx, momidx, nvars_), k - 2, j, i);
            const Real uj = u0_(m, CombinedIdx(nuidx, momidx, nvars_), k - 1, j, i);
            const Real ujp = u0_(m, CombinedIdx(nuidx, momidx, nvars_), k, j, i);
            const Real ujpp = u0_(m, CombinedIdx(nuidx, momidx, nvars_), k + 1, j, i);

            const Real dup = ujpp - ujp;
            const Real duc = ujp - uj;
            const Real dum = uj - ujm;

            bool sawtooth = false;
            Real phi_jp12 = 0;
            if (dup * duc > 0 && dum * duc > 0) {
              phi_jp12 = minmod2(dum / duc, dup / duc, params_.minmod_theta);
            } else if (dup * duc < 0 && dum * duc < 0) {
              sawtooth = true;
            }

            flux_jp12_ho[momidx] = (flux_j[momidx] + flux_jp1[momidx]) / 2.;
            flux_jp12_lo[momidx] =
                (flux_j[momidx] + flux_jp1[momidx]) / 2. -
                cmax_jp12 *
                    (u0_(m, CombinedIdx(nuidx, momidx, nvars_), k, j, i) -
                     u0_(m, CombinedIdx(nuidx, momidx, nvars_), k - 1, j, i)) /
                    2.;

            flx3_(m, CombinedIdx(nuidx, momidx, nvars_), k, j, i) =
                flux_jp12_ho[momidx] - (sawtooth ? 1 : A_jp12) * (1 - phi_jp12) *
                                           (flux_jp12_ho[momidx] - flux_jp12_lo[momidx]);
            if (num_hydro && momidx == M1_N_IDX) {  // moving with the fluid
              const Real fd = macc3_(m, k, j, i);
              flx3_(m, CombinedIdx(nuidx, momidx, nvars_), k, j, i) +=
                  fd * Kokkos::fmax(fd >= 0. ? ynu_l_(m, nuidx, k, j, i)
                                              : ynu_r_(m, nuidx, k, j, i), 0.);
            }
          }
        });
  }
  return TaskStatus::complete;
}
}  // namespace radiationm1
