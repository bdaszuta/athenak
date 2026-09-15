// Focused host and execution-space regressions for M1 norms and closure tensors.
// Build/run with tst/test_m1_math_findings.py and an existing CMake build.
#include <cmath>
#include <iostream>
#include <limits>

#include "radiation_m1/radiation_m1_calc_closure.hpp"

using Metric = AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2>;
using Vector = AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>;
constexpr Real kClosureTolerance = 64 * std::numeric_limits<Real>::epsilon();

// Each test calls the production helper, without reproducing its algorithm.
KOKKOS_INLINE_FUNCTION
Real Evaluate(const int test) {
  Metric gd{}, gu{};
  gd(0, 0) = gu(0, 0) = -1;
  for (int a = 1; a < 4; ++a) gd(a, a) = gu(a, a) = 1;
  Vector u{}, h{};
  if (test == 0) {
    // Positive-definite non-diagonal spatial metric, W=2, H.u=0.
    gd(1, 2) = 1.0 / 2.0;
    gu(1, 1) = gu(2, 2) = 4.0 / 3.0;
    gu(1, 2) = -2.0 / 3.0;
    u(0) = 2; u(1) = u(2) = 1;
    h(0) = 1; h(1) = h(2) = -1;
    return radiationm1::comoving_moment_squared(gd, gu, u, h);
  }
  if (test == 1 || test == 2) {
    // Diagonal high-W and direct low-W branches, with exact normalization.
    u(1) = (test == 1) ? 2 : 1.0 / 2.0;
    u(0) = Kokkos::sqrt(1 + u(1) * u(1));
    h(1) = 1; h(2) = 2; h(0) = -u(1) / u(0);
    return radiationm1::comoving_moment_squared(gd, gu, u, h);
  }
  Vector nd{}, vd{}, fd{}, ud{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> proj{};
  u(0) = 1; ud(0) = -1; nd(0) = -1; fd(1) = 1.0 / 2.0;
  radiationm1::calc_proj(ud, u, proj);
  radiationm1::RadiationM1Params params{};
  params.closure_newton_bracket = (test != 4);
  params.closure_epsilon = 1e-5;
  params.closure_maxiter = (test == 5) ? 1 : 10;
  params.rad_E_floor = 1e-30;
  params.rad_eps = 1e-14;
  Real xi{};
  const radiationm1::MathSignal ierr = radiationm1::NewtonClosure(
      gd, gu, nd, 1, u, vd, proj, 1, fd, params,
      (test == 5) ? 1.0 / 4.0 : 1.0 / 2.0, xi);
  if (test == 5) return ierr == radiationm1::LinalgContinue ? 1.0 : 0.0;
  return ierr == radiationm1::LinalgSuccess ? xi : -1.0;
}

// Dimensionless trace, spatiality, closure, and source-derivative errors.
// All states use signature (-,+,+,+), with Eulerian E and covariant F_a.
KOKKOS_INLINE_FUNCTION
Real ClosureError(const int test) {
  const bool curved = test == 1 || test >= 6;
  const bool moving = test == 1 || test >= 4;
  const bool solve = test >= 2 && test <= 9;
  const bool nonzero_flux = test == 10 || test == 11 || test == 13;
  const Real alpha = curved ? 3.0 / 2.0 : 1.0;
  const Real beta[3] = {curved ? 1.0 / 4.0 : 0.0,
                        curved ? -1.0 / 8.0 : 0.0,
                        curved ? 1.0 / 16.0 : 0.0};
  Real gamma[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  Real invgamma[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  if (curved) {
    gamma[0][1] = gamma[1][0] = 1.0 / 2.0;
    invgamma[0][0] = invgamma[1][1] = 4.0 / 3.0;
    invgamma[0][1] = invgamma[1][0] = -2.0 / 3.0;
  }
  Metric gd{}, gu{};
  gd(0, 0) = -alpha * alpha;
  gu(0, 0) = -1.0 / (alpha * alpha);
  for (int a = 0; a < 3; ++a) {
    gu(0, a + 1) = beta[a] / (alpha * alpha);
    for (int b = 0; b < 3; ++b) {
      gd(0, 0) += gamma[a][b] * beta[a] * beta[b];
      gd(0, a + 1) += gamma[a][b] * beta[b];
      gd(a + 1, b + 1) = gamma[a][b];
      gu(a + 1, b + 1) = invgamma[a][b] - beta[a] * beta[b] / (alpha * alpha);
    }
  }
  Vector nd{}, nu{}, vu{}, vd{}, u{}, ud{}, fd{};
  nd(0) = -alpha;
  nu(0) = 1.0 / alpha;
  vu(1) = moving ? 1.0 / 2.0 : 0.0;
  const Real W = 1.0 / Kokkos::sqrt(1.0 - vu(1) * vu(1));
  u(0) = W / alpha;
  for (int a = 0; a < 3; ++a) {
    nu(a + 1) = -beta[a] / alpha;
    u(a + 1) = W * (vu(a + 1) - beta[a] / alpha);
  }
  tensor_contract(gd, u, ud);
  tensor_contract(gd, vu, vd);
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> proj{};
  radiationm1::calc_proj(ud, u, proj);
  if (nonzero_flux) {
    pack_F_d(beta[0], beta[1], beta[2], 1.0 / 4.0, -1.0 / 8.0, 1.0 / 16.0, fd);
  }
  radiationm1::RadiationM1Params params{};
  params.closure_solver = test % 2 ? radiationm1::ClosureNewton
                                  : radiationm1::ClosureBrent;
  params.closure_newton_bracket = true;
  params.closure_epsilon = kClosureTolerance;
  params.closure_maxiter = 64;
  params.rad_E_floor = 1e-30;
  params.rad_eps = 1e-14;
  Real E = 1.0;
  if (test == 8 || test == 9) {
    E = params.rad_E_floor / 2;
    pack_F_d(beta[0], beta[1], beta[2], params.rad_E_floor / 4, 0, 0, fd);
    radiationm1::apply_floor(gu, E, fd, params);
    if (E != params.rad_E_floor) return 1.0;
    for (int a = 0; a < 4; ++a) if (fd(a) != 0.0) return 1.0;
  }
  Real chi = test < 2 || test == 10 ? 1.0 : 2.0 / 3.0;
  Metric P{};
  if (solve) {
    radiationm1::BrentFunctor fn{};
    radiationm1::calc_closure(fn, gd, gu, nd, W, u, vd, proj, E, fd, chi,
                             P, params, radiationm1::Minerbo);
  } else {
    radiationm1::apply_closure(gd, gu, nd, W, u, vd, proj, E, fd, chi, P, params);
  }
  Real trace = 0.0, error = 0.0;
  for (int a = 0; a < 4; ++a) {
    Real spatial = 0.0;
    for (int b = 0; b < 4; ++b) {
      trace += gu(a, b) * (P(a, b) / E);
      spatial += nu(b) * (P(a, b) / E);
    }
    error += Kokkos::abs(spatial);
  }
  error += Kokkos::abs(trace - 1.0);
  if (test < 2 || test == 10 || test == 11) {
    // Independent thin tensor: isotropic for zero F, rank one otherwise.
    const Real F2 = tensor_dot(gu, fd, fd);
    Metric thick{};
    radiationm1::calc_Pthick(gd, gu, nd, W, vd, E, fd, thick);
    const Real dthick = 3.0 * (1.0 - chi) / 2.0;
    for (int a = 0; a < 4; ++a) {
      for (int b = a; b < 4; ++b) {
        const Real thin = nonzero_flux ? E / F2 * fd(a) * fd(b)
                                       : E / 3.0 * (gd(a, b) + nd(a) * nd(b));
        const Real expected = dthick * thick(a, b) + (1.0 - dthick) * thin;
        error += Kokkos::abs((P(a, b) - expected) / E);
      }
    }
  }
  if (solve) {
    // Check the published tensor against the defining comoving flux relation.
    Metric T{};
    Vector H{};
    radiationm1::assemble_rT(nd, E, fd, P, T);
    const Real J = radiationm1::calc_J_from_rT(T, u);
    radiationm1::calc_H_from_rT(T, u, proj, H);
    const Real xi = Kokkos::sqrt(tensor_dot(gu, H, H)) / J;
    error += Kokkos::abs(chi - radiationm1::closure_fun(xi, radiationm1::Minerbo));
    if (moving && !(chi > 1.0 / 3.0)) return 1.0;
  }
  if (test == 12 || test == 13) {
    // At fixed chi and F, the source is affine in E. A unit energy increment
    // gives its energy derivative without a small finite-difference step.
    Real sources[2][4]{};
    for (int sample = 0; sample < 2; ++sample) {
      Metric Ps{}, T{};
      Vector H{}, S{};
      radiationm1::apply_closure(gd, gu, nd, W, u, vd, proj,
                                 E + sample, fd, chi, Ps, params);
      radiationm1::assemble_rT(nd, E + sample, fd, Ps, T);
      const Real J = radiationm1::calc_J_from_rT(T, u);
      radiationm1::calc_H_from_rT(T, u, proj, H);
      radiationm1::calc_rad_sources(0, 2, 3, ud, J, H, S);
      sources[sample][0] = radiationm1::calc_rE_source(alpha, nu, S);
      for (int a = 1; a < 4; ++a) sources[sample][a] = alpha * S(a);
    }
    Vector fu{};
    tensor_contract(gu, fd, fu);
    Real F2 = tensor_dot(fu, fd);
    const Real v2 = tensor_dot(vu, vd);
    const Real q[4] = {E, fd(1), fd(2), fd(3)};
    const Real cdt = 1.0 / 8.0;
    Real jac[4][4]{};
    radiationm1::source_jacobian(q, fu, F2, chi, 2, 3, vu, vd, v2, W,
                                alpha, cdt, q, jac);
    for (int a = 0; a < 4; ++a) {
      const Real expected = (a == 0) - cdt * (sources[1][a] - sources[0][a]);
      error += Kokkos::abs(jac[a][0] - expected);
    }
  }
  return error;
}

// Compare a deliberately exhausted Newton closure with the direct Brent result.
KOKKOS_INLINE_FUNCTION
Real NewtonFallbackError() {
  Metric gd{}, gu{};
  gd(0, 0) = gu(0, 0) = -1;
  for (int a = 1; a < 4; ++a) gd(a, a) = gu(a, a) = 1;

  Vector nd{}, vu{}, vd{}, u{}, ud{}, fd{};
  nd(0) = -1;
  vu(1) = 1.0 / 2.0;
  const Real W = 1.0 / Kokkos::sqrt(1.0 - vu(1) * vu(1));
  u(0) = W;
  u(1) = W * vu(1);
  tensor_contract(gd, u, ud);
  tensor_contract(gd, vu, vd);
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> proj{};
  radiationm1::calc_proj(ud, u, proj);

  radiationm1::RadiationM1Params params{};
  params.closure_newton_bracket = true;
  params.closure_epsilon = kClosureTolerance;
  params.closure_maxiter = 1;
  params.rad_E_floor = 1e-30;
  params.rad_eps = 1e-14;
  radiationm1::BrentFunctor fn{};
  Real chi_newton{}, chi_brent{};
  Metric P_newton{}, P_brent{};
  params.closure_solver = radiationm1::ClosureNewton;
  radiationm1::calc_closure(fn, gd, gu, nd, W, u, vd, proj, 1, fd,
                           chi_newton, P_newton, params, radiationm1::Minerbo);
  params.closure_solver = radiationm1::ClosureBrent;
  radiationm1::calc_closure(fn, gd, gu, nd, W, u, vd, proj, 1, fd,
                           chi_brent, P_brent, params, radiationm1::Minerbo);

  Real error = Kokkos::abs(chi_newton - chi_brent);
  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      error += Kokkos::abs(P_newton(a, b) - P_brent(a, b));
    }
  }
  return error;
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failed = 0;
  {
    constexpr int ntest = 21;
    const Real expected[ntest] = {1.0 / 3.0, 21.0 / 5.0, 24.0 / 5.0,
                                  1.0 / 2.0, 1.0 / 2.0, 1.0};
    const char *names[ntest] = {
        "norm-nondiagonal", "norm-high-W", "norm-low-W", "newton-exact-bracket",
        "newton-exact-pure", "newton-exhaustion-signal", "zero-thin-static",
        "zero-thin-moving-shifted", "zero-static-brent", "zero-static-newton",
        "zero-moving-brent", "zero-moving-newton", "zero-shifted-brent",
        "zero-shifted-newton", "floor-shifted-brent", "floor-shifted-newton",
        "nonzero-thin", "nonzero-blend", "zero-source-energy-column",
        "nonzero-source-energy-column", "newton-fallback-brent"};
    // Explicit host evaluation remains distinct from the device-kernel result.
    Real host[ntest];
    for (int n = 0; n < ntest; ++n) {
      if (n < 6) {
        host[n] = Evaluate(n);
      } else if (n < 20) {
        host[n] = ClosureError(n - 6);
      } else {
        host[n] = NewtonFallbackError();
      }
    }
    Kokkos::View<Real *> device("m1_findings", ntest);
    Kokkos::parallel_for("m1_findings", Kokkos::RangePolicy<>(0, ntest),
                        KOKKOS_LAMBDA(const int n) {
                          if (n < 6) {
                            device(n) = Evaluate(n);
                          } else if (n < 20) {
                            device(n) = ClosureError(n - 6);
                          } else {
                            device(n) = NewtonFallbackError();
                          }
                        });
    const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device);
    for (int n = 0; n < ntest; ++n) {
      // Tensor checks include frame contractions and a closure solve to 64 eps.
      // Keep the original six scalar tolerances unchanged.
      const Real tol = (n < 6 ? 32 : 512) * std::numeric_limits<Real>::epsilon();
      const bool pass_host = std::abs(host[n] - expected[n]) <= tol;
      const bool pass_device = std::abs(mirror(n) - expected[n]) <= tol;
      const bool exact_root = (n != 3 && n != 4) ||
                              (host[n] == expected[n] && mirror(n) == expected[n]);
      std::cout << ((pass_host && pass_device && exact_root) ? "PASS " : "FAIL ")
                << "case=" << n << " " << names[n] << " host=" << host[n]
                << " execution_space=" << Kokkos::DefaultExecutionSpace::name()
                << " value=" << mirror(n) << " expected=" << expected[n] << '\n';
      failed += !(pass_host && pass_device && exact_root);
    }
  }
  Kokkos::finalize();
  return failed ? 1 : 0;
}
