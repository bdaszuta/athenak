#ifndef RADIATION_M1_CALC_CLOSURE_HPP
#define RADIATION_M1_CALC_CLOSURE_HPP
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_calc_closure.hpp
//! \brief compute closure and inv. closure using Brent-Dekker routines

#include "radiation_m1/radiation_m1_roots_brent.hpp"
#include "radiation_m1/radiation_m1_roots_fns.hpp"

namespace radiationm1 {

// Computes the closure in the lab frame with a rootfinding procedure.
// On return `chi` is the Eddington factor and `xi` is the closure flux factor
// sqrt(H_a H^a)/J that produced it.  Every exit path sets a usable closure;
// the return value distinguishes convergence from a bounded fallback.  Callers
// that do not need xi use the forwarding overload below.
KOKKOS_INLINE_FUNCTION MathSignal calc_closure(
    BrentFunctor BrentFunc, const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d, const Real &w_lorentz,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud, const Real &E,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d, Real &chi, Real &xi,
    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &P_dd,
    const RadiationM1Params &m1_params, const RadiationM1Closure &closure_type) {
  // Special cases for which no rootfinding is needed.
  if (closure_type == Eddington) {
    chi = 1. / 3.;
    xi = 0.0;
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi, P_dd,
                  m1_params);
    return LinalgSuccess;
  }
  if (closure_type == Thin) {
    chi = 1.0;
    xi = 1.0;
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi, P_dd,
                  m1_params);
    return LinalgSuccess;
  }
  if (closure_type == Minerbo || closure_type == Kershaw) {
    // Newton-Raphson rootfinder (opt-in via closure_solver = newton).
    // Seeded with the relativistic-aberration guess xi0 = |F - v E| / E.
    if (closure_type == Minerbo && m1_params.closure_solver == ClosureNewton) {
      AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Hg_d{};
      for (int a = 0; a < 4; ++a) {
        Hg_d(a) = F_d(a) - v_d(a) * E;
      }
      const Real xi0 =
          (E > m1_params.rad_E_floor)
              ? Kokkos::fmin(1.0, Kokkos::sqrt(Kokkos::fmax(0.0,
                                     tensor_dot(g_uu, Hg_d, Hg_d))) / E)
              : 0.5;
      Real xi_root{};
      const MathSignal ierr = NewtonClosure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d,
                                            proj_ud, E, F_d, m1_params, xi0, xi_root);
      if (ierr == LinalgSuccess) {
        chi = closure_fun(xi_root, closure_type);
        xi = xi_root;
        apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi,
                      P_dd, m1_params);
        return LinalgSuccess;
      }
#ifdef DEBUG_BUILD
      Kokkos::printf("Newton M1 closure failed; falling back to Brent-Dekker.\n");
#endif
    }

    Real x_lo = 0.0;
    Real x_md = 0.5;
    Real x_hi = 1.0;
    Real root{};
    BrentState state{};

    MathSignal ierr =
        BrentInitialize(BrentFunc, x_lo, x_hi, root, state, g_dd, g_uu, n_d, w_lorentz,
                        u_u, v_d, proj_ud, E, F_d, m1_params, closure_type);

    if (ierr == LinalgEinval) {
      // No bracket: retain the finite endpoint with the smaller residual.
      if (Kokkos::fabs(state.f_b) < Kokkos::fabs(state.f_a)) {
        xi = 1.0;
        chi = 1.0;
      } else {
        xi = 0.0;
        chi = 1. / 3.;
      }
      apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi, P_dd,
                    m1_params);
      return ierr;
    }
    if (ierr != LinalgSuccess) {
      // A nonfinite closure function cannot define a bounded residual choice.
      xi = 0.0;
      chi = 1. / 3.;
      apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi, P_dd,
                    m1_params);
      return ierr;
    }

    // Initialization can already contain an exact endpoint root.  Report it as
    // converged even when the configured iteration budget is zero.
    if (state.f_a == 0.0 || state.f_b == 0.0) {
      xi = (state.f_a == 0.0) ? state.a : state.b;
      xi = Kokkos::fmax(0.0, Kokkos::fmin(xi, 1.0));
      chi = closure_fun(xi, closure_type);
      apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi, P_dd,
                    m1_params);
      return LinalgSuccess;
    }

    ierr = LinalgContinue;
    int iter = 0;
    while (ierr == LinalgContinue && iter < m1_params.closure_maxiter) {
      ++iter;
      const MathSignal iterate_status =
          BrentIterate(BrentFunc, x_lo, x_hi, root, state, g_dd, g_uu, n_d,
                       w_lorentz, u_u, v_d, proj_ud, E, F_d, m1_params,
                       closure_type);
      if (iterate_status != LinalgSuccess) {
        ierr = iterate_status;
        break;
      }
      x_md = root;
      ierr = BrentTestInterval(x_lo, x_hi, m1_params.closure_epsilon, 0);
    }

    if (ierr == LinalgContinue) {
      // Exhaustion is not convergence.  Retain the finite bounded iterate or
      // bracket endpoint with the smallest closure residual.
      x_md = state.b;
      Real best_residual = Kokkos::fabs(state.f_b);
      if (Kokkos::isfinite(state.f_a) &&
          Kokkos::fabs(state.f_a) < best_residual) {
        x_md = state.a;
        best_residual = Kokkos::fabs(state.f_a);
      }
      if (Kokkos::isfinite(state.f_c) &&
          Kokkos::fabs(state.f_c) < best_residual) {
        x_md = state.c;
      }
    }

    if (ierr == LinalgEbadfunc || !Kokkos::isfinite(x_md)) {
      xi = 0.0;
      chi = 1. / 3.;
    } else {
      xi = Kokkos::fmax(0.0, Kokkos::fmin(x_md, 1.0));
      chi = closure_fun(xi, closure_type);
    }
    apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi, P_dd,
                  m1_params);
#ifdef DEBUG_BUILD
    if (ierr == LinalgContinue) {
      Kokkos::printf(
          "Maximum number of iterations exceeded when computing the M1 "
          "closure\n");
    }
#endif
    return ierr;
  }

  // Unknown closure types retain a finite Eddington state for propagation.
  xi = 0.0;
  chi = 1. / 3.;
  apply_closure(g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E, F_d, chi, P_dd,
                m1_params);
  return LinalgInvalid;
}

// Overload for callers that do not need the closure flux factor.  It forwards
// to the primary and discards xi, so those call sites stay unchanged.
KOKKOS_INLINE_FUNCTION MathSignal calc_closure(
    BrentFunctor BrentFunc, const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d, const Real &w_lorentz,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud, const Real &E,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d, Real &chi,
    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &P_dd,
    const RadiationM1Params &m1_params, const RadiationM1Closure &closure_type) {
  Real xi_unused{};
  return calc_closure(BrentFunc, g_dd, g_uu, n_d, w_lorentz, u_u, v_d, proj_ud, E,
                      F_d, chi, xi_unused, P_dd, m1_params, closure_type);
}

}  // namespace radiationm1
#endif  // RADIATION_M1_CALC_CLOSURE_HPP
