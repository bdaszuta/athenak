#ifndef RADIATION_M1_SOURCES_HPP
#define RADIATION_M1_SOURCES_HPP

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_roots_hybridsj.hpp"

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_macro.hpp
//  \brief macros for Grey M1 radiation class

namespace radiationm1 {

// A candidate state is usable only when both the energy and flux norm are finite.
KOKKOS_INLINE_FUNCTION
bool radiation_state_is_finite(
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu, const Real E,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d) {
  if (!Kokkos::isfinite(E)) return false;
  for (int a = 0; a < 4; ++a) {
    if (!Kokkos::isfinite(F_d(a))) return false;
  }
  return Kokkos::isfinite(tensor_dot(g_uu, F_d, F_d));
}

// Publish either the accepted source candidate or the physicalized predictor and
// return exactly the radiation increment that must be charged to matter.
KOKKOS_INLINE_FUNCTION
void select_source_candidate(
    const bool accept_candidate, const Real Ebase,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fbase_d,
    const Real Ecandidate,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fcandidate_d, Real &Efinal,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Ffinal_d, Real &dE,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &dF_d) {
  Efinal = accept_candidate ? Ecandidate : Ebase;
  dE = Efinal - Ebase;
  for (int a = 0; a < 4; ++a) {
    Ffinal_d(a) = accept_candidate ? Fcandidate_d(a) : Fbase_d(a);
    dF_d(a) = Ffinal_d(a) - Fbase_d(a);
  }
}

//----------------------------------------------------------------------------------------
//! \fn MathSignal radiationm1::RadiationM1::prepare_closure
//  \brief Sets F_d, F_u, E and computes chi, P_dd in src_params
KOKKOS_INLINE_FUNCTION
MathSignal prepare_closure(const BrentFunctor &BrentFunc, const Real q[4],
                           SrcParams &src_params, const RadiationM1Params &m1_params,
                           const RadiationM1Closure &closure_type) {
  for (int a = 0; a < M1_MULTIROOTS_DIM; ++a) {
    if (!Kokkos::isfinite(q[a])) {
      return LinalgEbadfunc;
    }
  }

  src_params.E = Kokkos::fmax(q[0], 0.0);
  pack_F_d(-src_params.alp * src_params.n_u(1), -src_params.alp * src_params.n_u(2),
           -src_params.alp * src_params.n_u(3), q[1], q[2], q[3], src_params.F_d);

  tensor_contract(src_params.g_uu, src_params.F_d, src_params.F_u);

  return calc_closure(BrentFunc, src_params.g_dd, src_params.g_uu, src_params.n_d,
                      src_params.W, src_params.u_u, src_params.v_d, src_params.proj_ud,
                      src_params.E, src_params.F_d, src_params.chi, src_params.P_dd,
                      m1_params, closure_type);
}

//----------------------------------------------------------------------------------------
//! \fn MathSignal radiationm1::RadiationM1::prepare_sources
//  \brief Sets T_dd, J, H_d, S_d, Edot and tS_d in src_params
KOKKOS_INLINE_FUNCTION
MathSignal prepare_sources(const Real q[4], SrcParams &src_params,
                           const RadiationM1Params &params_) {
  assemble_rT(src_params.n_d, src_params.E, src_params.F_d, src_params.P_dd,
              src_params.T_dd);

  src_params.J = calc_J_from_rT(src_params.T_dd, src_params.u_u);
  calc_H_from_rT(src_params.T_dd, src_params.u_u, src_params.proj_ud, src_params.H_d);
  enforce_comoving_moment_safety(src_params.g_dd, src_params.g_uu, src_params.u_u,
                                 src_params.J, src_params.H_d, params_);

  calc_rad_sources(src_params.eta, src_params.kabs, src_params.kscat, src_params.u_d,
                   src_params.J, src_params.H_d, src_params.S_d);

  src_params.Edot = calc_rE_source(src_params.alp, src_params.n_u, src_params.S_d);
  calc_rF_source(src_params.alp, src_params.gamma_ud, src_params.S_d, src_params.tS_d);

  if (!Kokkos::isfinite(src_params.J) || !Kokkos::isfinite(src_params.Edot)) {
    return LinalgEbadfunc;
  }
  for (int a = 0; a < 4; ++a) {
    if (!Kokkos::isfinite(src_params.H_d(a)) ||
        !Kokkos::isfinite(src_params.S_d(a)) ||
        !Kokkos::isfinite(src_params.tS_d(a))) {
      return LinalgEbadfunc;
    }
  }
  return LinalgSuccess;
}

//----------------------------------------------------------------------------------------
//! \fn MathSignal radiationm1::RadiationM1::prepare
//  \brief Calls prepare_closure and prepare_sources
KOKKOS_INLINE_FUNCTION
MathSignal prepare(const BrentFunctor &BrentFunc, const Real q[4], SrcParams &src_params,
                   const RadiationM1Params &m1_params,
                   const RadiationM1Closure &closure_type) {
  auto ierr = prepare_closure(BrentFunc, q, src_params, m1_params, closure_type);
  if (ierr != LinalgSuccess) {
    return ierr;
  }
  ierr = prepare_sources(q, src_params, m1_params);
  if (ierr != LinalgSuccess) {
    return ierr;
  }
  return LinalgSuccess;
}

//----------------------------------------------------------------------------------------
//! \fn HybridsjSignal radiationm1::explicit_update
//  \brief Computes (E/F)_new = (E/F)* + cdt * tS_d[E/F]
KOKKOS_INLINE_FUNCTION
void explicit_update(const SrcParams &src_params, Real &Enew,
                     AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fnew_d) {
  Enew = src_params.Estar + src_params.cdt * src_params.Edot;
  Fnew_d(1) = src_params.Fstar_d(1) + src_params.cdt * src_params.tS_d(1);
  Fnew_d(2) = src_params.Fstar_d(2) + src_params.cdt * src_params.tS_d(2);
  Fnew_d(3) = src_params.Fstar_d(3) + src_params.cdt * src_params.tS_d(3);
  // F_0 = g_0i F^i = beta_i F^i = beta^i F_i
  Fnew_d(0) = - src_params.alp * src_params.n_u(1) * Fnew_d(1)
    - src_params.alp * src_params.n_u(2) * Fnew_d(2) 
    - src_params.alp * src_params.n_u(3) * Fnew_d(3);
}


// Classify once using the full mesh dt for thermalisation, then half dt for
// nonstiff/stiff. No lapse/W factors, source-limit options, or number override.
KOKKOS_INLINE_FUNCTION
RadiationM1SourceRegime gra_source_regime(
    const Real dt, const Real kabs, const Real kscat, const Real rho,
    const Real temp_mev, const RadiationM1Params &params) {
  if (dt * Kokkos::sqrt(kabs * (kabs + kscat)) > 1.0 &&
      rho >= params.gra_source_rho_min &&
      temp_mev >= params.gra_source_temp_min_mev) {
    return SourceEquilibrium;
  }
  const Real half_dt = 0.5 * dt;
  if (half_dt * kabs < 1.0 && half_dt * kscat < 1.0) {
    return SourceNonstiff;
  }
  return SourceStiff;
}

// Literal GRA approximate first-order thick reconstruction. Hstar_d(i) is
// already the spatial Eulerian projection (n_i=0); Hn = v^i Htilde_i.
// In particular, the relaxation denominators contain no lapse.
KOKKOS_INLINE_FUNCTION
void gra_approximate_thick_update(
    const Real h, const Real alp,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_u,
    const Real W, const Real Jstar,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Hstar_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_u,
    const Real eta_vol, const Real kabs, const Real kscat, Real &Enew,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fnew_d) {
  const Real Jnew = (W * Jstar + h * eta_vol) / (W + h * kabs);
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Htilde_d{};
  Real Hn = 0.0;
  for (int a = 1; a < 4; ++a) {
    Htilde_d(a) = W * Hstar_d(a) / (W + h * (kabs + kscat));
    Hn += v_u(a) * Htilde_d(a);
  }
  const Real W2 = W * W;
  Enew = (4.0 * W2 - 1.0) / 3.0 * Jnew + 2.0 * W * Hn;
  for (int a = 1; a < 4; ++a) {
    Fnew_d(a) = 4.0 / 3.0 * W2 * Jnew * v_d(a) +
                W * (Hn * v_d(a) + Htilde_d(a));
  }
  Fnew_d(0) = -alp * (n_u(1) * Fnew_d(1) + n_u(2) * Fnew_d(2) +
                      n_u(3) * Fnew_d(3));
}

// forward declaration
KOKKOS_INLINE_FUNCTION
SrcSignal source_update_eddington(
    const BrentFunctor &BrentFunc, const HybridsjFunctor &HybridsjFunc, const Real &cdt,
    const Real &alp, const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &gamma_ud,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud, const Real &W,
    const Real &Eold, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fold_d,
    const Real &Estar, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fstar_d,
    const Real &eta, const Real &kabs, const Real &kscat, Real &chi, Real &Enew,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fnew_d,
    const RadiationM1Params &m1_params);


// Solves the implicit problem
// .  q^new = q^star + dt S[q^new]
// The source term is S^a = (eta - ka J) u^a - (ka + ks) H^a and includes
// also emission.
KOKKOS_INLINE_FUNCTION
SrcSignal source_update_ll(
    const BrentFunctor &BrentFunc, const HybridsjFunctor &HybridsjFunc, const Real &cdt,
    const Real &alp, const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &gamma_ud,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud, const Real &W,
    const Real &, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &,
    const Real &Estar, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fstar_d,
    const Real &eta, const Real &kabs, const Real &kscat, Real &chi, Real &Enew,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fnew_d,
    const RadiationM1Params &m1_params, const RadiationM1Closure &closure_type,
    const RadiationM1SourceRegime forced_regime = SourceLegacy) {
  SrcParams src_params(cdt, alp, g_dd, g_uu, n_d, n_u, gamma_ud, u_d, u_u, v_d, v_u,
                       proj_ud, W, Estar, Fstar_d, chi, eta, kabs, kscat);

  // non stiff limit, explicit update
  if (forced_regime == SourceNonstiff ||
      (forced_regime == SourceLegacy &&
       cdt * kabs < m1_params.source_thin_limit &&
       cdt * kscat < m1_params.source_thin_limit)) {
    const Real qstar[] = {src_params.Estar, src_params.Fstar_d(1),
                          src_params.Fstar_d(2), src_params.Fstar_d(3)};
    MathSignal ierr = prepare(BrentFunc, qstar, src_params, m1_params, closure_type);
    if (ierr != LinalgSuccess) {
      return SrcFail;
    }
    explicit_update(src_params, Enew, Fnew_d);

    const Real q[4] = {Enew, Fnew_d(1), Fnew_d(2), Fnew_d(3)};
    ierr = prepare_closure(BrentFunc, q, src_params, m1_params, closure_type);
    if (ierr != LinalgSuccess) {
      return SrcFail;
    }
    chi = src_params.chi;
    return SrcThin;
  }

  // cannot capture case tau << dt, go to equilibrium
  if (forced_regime == SourceLegacy && m1_params.source_thick_limit > 0 &&
      cdt * cdt * (kabs * (kabs + kscat)) >
          m1_params.source_thick_limit * m1_params.source_thick_limit) {
    return SrcEquil;
  }

  // scattering dominated limit
  if (forced_regime == SourceLegacy && m1_params.source_scat_limit > 0 &&
      cdt * kscat > m1_params.source_scat_limit) {
    return SrcScat;
  }

  // begin multiroots solve
  HybridsjState hybridsj_state{};
  HybridsjParams hybridsj_params{};

  // initial guess
  Real q[4] = {Enew, Fnew_d(1), Fnew_d(2), Fnew_d(3)};
  Real x[4];
  for (int i = 0; i < 4; i++) {
    x[i] = q[i];
    hybridsj_params.x[i] = x[i];
  }
  auto ierr = HybridsjInitialize(HybridsjFunc, hybridsj_state, hybridsj_params, BrentFunc,
                                 src_params, m1_params);
  if (ierr != LinalgSuccess) {
    return SrcFail;
  }

  int iter = 0;
  do {
    if (iter >= m1_params.source_maxiter) {
#ifdef DEBUG_BUILD
      Kokkos::printf(
          "source_update: Exceeded maximum number of iterations of Hybridsj.\n");
#endif
      return SrcFail;
    }

    ierr = HybridsjIterate(HybridsjFunc, hybridsj_state, hybridsj_params, BrentFunc,
                           src_params, m1_params);
    iter++;

    // the solver is stuck!
    if (ierr == LinalgEnoprogj || ierr == LinalgEbadfunc) {
#ifdef DEBUG_BUILD
      if (ierr == LinalgEbadfunc) {
        Kokkos::printf("source_update: NaNs/Infs found in implicit solve.\n");
      } else {
        Kokkos::printf("source_update: Non-linear solver is stuck.\n");
      }
#endif
      return SrcFail;
    } else if (ierr != LinalgSuccess) {
#ifdef DEBUG_BUILD
      Kokkos::printf("source_update: Unexpected error in source solver!\n");
#endif
      return SrcFail;
    }
    ierr = HybridsjTestDelta(hybridsj_params.dx, hybridsj_params.x,
                             m1_params.source_epsabs, m1_params.source_epsrel);
  } while (ierr == LinalgContinue);

  // A small dogleg step is not sufficient for convergence.  The source
  // Jacobian treats chi as fixed while the residual can recompute chi(q), so a
  // stalled iterate must not be returned as SrcOk.  hybridsj_params.f is kept
  // consistent with hybridsj_params.x: initialization evaluates f(x), accepted
  // steps copy f_trial together with x_trial, and rejected steps leave both
  // unchanged.
  for (int a = 0; a < M1_MULTIROOTS_DIM; ++a) {
    const Real xstar =
        (a == 0) ? src_params.Estar : src_params.Fstar_d(a);
    const Real scale =
        Kokkos::fmax(Kokkos::fabs(hybridsj_params.x[a]), Kokkos::fabs(xstar));
    const Real tolerance =
        m1_params.source_epsabs + m1_params.source_epsrel * scale;

    if (!Kokkos::isfinite(hybridsj_params.f[a]) ||
        Kokkos::fabs(hybridsj_params.f[a]) > tolerance) {
#ifdef DEBUG_BUILD
      Kokkos::printf(
          "source_update: small step but large residual in component %d "
          "(f=%.3e, tol=%.3e).\n",
          a, hybridsj_params.f[a], tolerance);
#endif
      return SrcFail;
    }
  }

  Enew = hybridsj_params.x[0];
  Fnew_d(1) = hybridsj_params.x[1];
  Fnew_d(2) = hybridsj_params.x[2];
  Fnew_d(3) = hybridsj_params.x[3];
  // F_0 = g_0i F^i = beta_i F^i = beta^i F_i
  Fnew_d(0) =
      -alp * n_u(1) * Fnew_d(1) - alp * n_u(2) * Fnew_d(2) - alp * n_u(3) * Fnew_d(3);

  const MathSignal closure_status =
      prepare_closure(BrentFunc, hybridsj_params.x, src_params, m1_params, closure_type);
  if (closure_status != LinalgSuccess) {
    return SrcFail;
  }
  chi = src_params.chi;

  return SrcOk;
}

KOKKOS_INLINE_FUNCTION
SrcSignal source_update(
    const BrentFunctor &BrentFunc, const HybridsjFunctor &HybridsjFunc, const Real &cdt,
    const Real &alp, const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &gamma_ud,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud, const Real &W,
    const Real &Eold, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fold_d,
    const Real &Estar, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fstar_d,
    const Real &eta, const Real &kabs, const Real &kscat, Real &chi, Real &Enew,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &Fnew_d,
    const RadiationM1Params &m1_params, const RadiationM1Closure &closure_type,
    const RadiationM1SourceRegime forced_regime = SourceLegacy) {
  SrcSignal ierr = source_update_ll(
      BrentFunc, HybridsjFunc, cdt, alp, g_dd, g_uu, n_d, n_u, gamma_ud, u_d,
      u_u, v_d, v_u, proj_ud, W, Eold, Fold_d, Estar, Fstar_d, eta, kabs, kscat,
      chi, Enew, Fnew_d, m1_params, closure_type, forced_regime);
  if (ierr == SrcFail && closure_type != Eddington) {
    RadiationM1Params fallback_params = m1_params;
    fallback_params.closure_type = Eddington;
    const SrcSignal fallback_signal = source_update_ll(
        BrentFunc, HybridsjFunc, cdt, alp, g_dd, g_uu, n_d, n_u, gamma_ud, u_d,
        u_u, v_d, v_u, proj_ud, W, Eold, Fold_d, Estar, Fstar_d, eta, kabs,
        kscat, chi, Enew, Fnew_d, fallback_params, Eddington, forced_regime);
    ierr = (fallback_signal == SrcFail) ? SrcFail : SrcEddington;
#ifdef DEBUG_BUILD
    if (ierr == SrcFail) {
      Kokkos::printf("source_update: The source solver has failed!\n");
    }
    else {
      Kokkos::printf("source_update: Backup solver worked!\n");
    }
#endif
  }
  return ierr;
}

}  // namespace radiationm1

#endif  // RADIATION_M1_SOURCES_HPP
