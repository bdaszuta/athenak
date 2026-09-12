// Focused host and execution-space regressions for the M1 norm and Newton root.
// Build/run with tst/test_m1_math_findings.py and an existing CMake build.
#include <cmath>
#include <iostream>
#include <limits>

#include "radiation_m1/radiation_m1_calc_closure.hpp"

using Metric = AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2>;
using Vector = AthenaPointTensor<Real, TensorSymm::NONE, 4, 1>;

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
  // test=5 deliberately exhausts one iteration and must print a warning.
  return radiationm1::NewtonClosure(gd, gu, nd, 1, u, vd, proj, 1, fd,
                                    params, (test == 5) ? 1.0 / 4.0 : 1.0 / 2.0);
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int failed = 0;
  {
    constexpr int ntest = 6;
    const Real expected[ntest] = {1.0 / 3.0, 21.0 / 5.0, 24.0 / 5.0,
                                  1.0 / 2.0, 1.0 / 2.0, 5.0 / 8.0};
    // Explicit host evaluation remains distinct from the device-kernel result.
    Real host[ntest];
    for (int n = 0; n < ntest; ++n) host[n] = Evaluate(n);
    Kokkos::View<Real *> device("m1_findings", ntest);
    Kokkos::parallel_for("m1_findings", Kokkos::RangePolicy<>(0, ntest),
                        KOKKOS_LAMBDA(const int n) { device(n) = Evaluate(n); });
    const auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), device);
    for (int n = 0; n < ntest; ++n) {
      const Real tol = 32 * std::numeric_limits<Real>::epsilon();
      const bool pass_host = std::abs(host[n] - expected[n]) <= tol;
      const bool pass_device = std::abs(mirror(n) - expected[n]) <= tol;
      const bool exact_root = (n != 3 && n != 4) ||
                              (host[n] == expected[n] && mirror(n) == expected[n]);
      std::cout << ((pass_host && pass_device && exact_root) ? "PASS " : "FAIL ")
                << "case=" << n << " host=" << host[n]
                << " execution_space=" << Kokkos::DefaultExecutionSpace::name()
                << " value=" << mirror(n) << " expected=" << expected[n] << '\n';
      failed += !(pass_host && pass_device && exact_root);
    }
  }
  Kokkos::finalize();
  return failed ? 1 : 0;
}
