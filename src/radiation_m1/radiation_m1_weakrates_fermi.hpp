#ifndef RADIATION_M1_RADIATION_M1_WEAKRATES_FERMI_HPP_
#define RADIATION_M1_RADIATION_M1_WEAKRATES_FERMI_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_weakrates_fermi.hpp
//! \brief Takahashi approximations used by the GR-Athena++ weakrates formulas.

#include <Kokkos_Core.hpp>

#include "athena.hpp"

namespace radiationm1 {
namespace weakrates {
namespace fermi {

inline constexpr Real eta0 = 1.0e-3;

KOKKOS_INLINE_FUNCTION
Real F2P(const Real eta) {
  constexpr Real k21 = 3.2899;
  constexpr Real k22 = 1.8246;
  const Real x = 1.0/eta;
  return (1.0/3.0 + k21*x*x)/(1.0 - Kokkos::exp(-k22*eta));
}

KOKKOS_INLINE_FUNCTION
Real F2M(const Real eta) {
  constexpr Real a21 = 0.1092;
  constexpr Real a22 = 0.8908;
  return 2.0/(1.0 + a21*Kokkos::exp(a22*eta));
}

KOKKOS_INLINE_FUNCTION
Real F2(const Real eta) {
  if (eta > eta0) {
    return eta*eta*eta*F2P(eta);
  }
  return Kokkos::exp(eta)*F2M(eta);
}

KOKKOS_INLINE_FUNCTION
Real F3P(const Real eta) {
  constexpr Real k31 = 4.9348;
  constexpr Real k32 = 11.3644;
  constexpr Real k33 = 1.9039;
  const Real x = 1.0/eta;
  const Real x2 = x*x;
  const Real x4 = x2*x2;
  return (0.25 + k31*x2 + k32*x4)/(1.0 + Kokkos::exp(-k33*eta));
}

KOKKOS_INLINE_FUNCTION
Real F3M(const Real eta) {
  constexpr Real a31 = 0.0559;
  constexpr Real a32 = 0.9069;
  return 6.0/(1.0 + a31*Kokkos::exp(a32*eta));
}

KOKKOS_INLINE_FUNCTION
Real F3(const Real eta) {
  if (eta > eta0) {
    return eta*eta*eta*eta*F3P(eta);
  }
  return Kokkos::exp(eta)*F3M(eta);
}

KOKKOS_INLINE_FUNCTION
Real F4P(const Real eta) {
  constexpr Real k41 = 6.5797;
  constexpr Real k42 = 45.4576;
  constexpr Real k43 = 1.9484;
  const Real x2 = 1.0/(eta*eta);
  const Real x4 = x2*x2;
  return (0.2 + k41*x2 + k42*x4)/(1.0 - Kokkos::exp(-k43*eta));
}

KOKKOS_INLINE_FUNCTION
Real F4M(const Real eta) {
  constexpr Real a41 = 0.0287;
  constexpr Real a42 = 0.9257;
  return 24.0/(1.0 + a41*Kokkos::exp(a42*eta));
}

KOKKOS_INLINE_FUNCTION
Real F4(const Real eta) {
  if (eta > eta0) {
    return eta*eta*eta*eta*eta*F4P(eta);
  }
  return Kokkos::exp(eta)*F4M(eta);
}

KOKKOS_INLINE_FUNCTION
Real F5P(const Real eta) {
  constexpr Real k51 = 8.2247;
  constexpr Real k52 = 113.6439;
  constexpr Real k53 = 236.5323;
  constexpr Real k54 = 1.9727;
  const Real x2 = 1.0/(eta*eta);
  const Real x4 = x2*x2;
  const Real x6 = x4*x2;
  return (1.0/6.0 + k51*x2 + k52*x4 + k53*x6)/
         (1.0 + Kokkos::exp(-k54*eta));
}

KOKKOS_INLINE_FUNCTION
Real F5M(const Real eta) {
  constexpr Real a51 = 0.0147;
  constexpr Real a52 = 0.9431;
  return 120.0/(1.0 + a51*Kokkos::exp(a52*eta));
}

KOKKOS_INLINE_FUNCTION
Real F5(const Real eta) {
  if (eta > eta0) {
    return eta*eta*eta*eta*eta*eta*F5P(eta);
  }
  return Kokkos::exp(eta)*F5M(eta);
}

KOKKOS_INLINE_FUNCTION
Real F4OverF2(const Real eta) {
  if (eta > eta0) {
    return eta*eta*F4P(eta)/F2P(eta);
  }
  return F4M(eta)/F2M(eta);
}

KOKKOS_INLINE_FUNCTION
Real F4OverF3(const Real eta) {
  if (eta > eta0) {
    return eta*F4P(eta)/F3P(eta);
  }
  return F4M(eta)/F3M(eta);
}

KOKKOS_INLINE_FUNCTION
Real F5OverF3(const Real eta) {
  if (eta > eta0) {
    return eta*eta*F5P(eta)/F3P(eta);
  }
  return F5M(eta)/F3M(eta);
}

KOKKOS_INLINE_FUNCTION
Real F5OverF4(const Real eta) {
  if (eta > eta0) {
    return eta*F5P(eta)/F4P(eta);
  }
  return F5M(eta)/F4M(eta);
}

}  // namespace fermi
}  // namespace weakrates
}  // namespace radiationm1

#endif  // RADIATION_M1_RADIATION_M1_WEAKRATES_FERMI_HPP_
