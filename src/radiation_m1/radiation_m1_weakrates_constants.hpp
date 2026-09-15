#ifndef RADIATION_M1_RADIATION_M1_WEAKRATES_CONSTANTS_HPP_
#define RADIATION_M1_RADIATION_M1_WEAKRATES_CONSTANTS_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_weakrates_constants.hpp
//! \brief Physical constants used by the GR-Athena++ weakrates formulas.

#include <Kokkos_Core.hpp>

#include "athena.hpp"

namespace radiationm1 {
namespace weakrates {

// These values are pinned to the GR-Athena++ weakrates implementation.
inline constexpr Real clight = 2.99792458e10;       // cm s^-1
inline constexpr Real mev_to_erg = 1.60217733e-6;   // erg MeV^-1
inline constexpr Real hc_mevcm = 1.23984172e-10;    // MeV cm
inline constexpr Real pi = 3.14159265358979323846;
inline constexpr Real me_erg = 8.187108692567103e-7;
inline constexpr Real sigma_0 = 1.76e-44;            // cm^2
inline constexpr Real alpha = 1.23;
inline constexpr Real cv = 0.5 + 2.0*0.23;
inline constexpr Real ca = 0.5;
inline constexpr Real gamma_0 = 5.565e-2;
inline constexpr Real fine_structure = 1.0/137.036;
inline constexpr Real q_np_mev = 1.293333;

KOKKOS_INLINE_FUNCTION
constexpr Real Square(const Real x) { return x*x; }

KOKKOS_INLINE_FUNCTION
constexpr Real Cube(const Real x) { return x*x*x; }

KOKKOS_INLINE_FUNCTION
constexpr Real Fourth(const Real x) { return x*x*x*x; }

}  // namespace weakrates
}  // namespace radiationm1

#endif  // RADIATION_M1_RADIATION_M1_WEAKRATES_CONSTANTS_HPP_
