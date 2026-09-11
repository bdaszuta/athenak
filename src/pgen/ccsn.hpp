#ifndef PGEN_CCSN_HPP_
#define PGEN_CCSN_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ccsn.hpp
//  \brief CCSN pgen utilities: StellarProfile, rotation laws, vector potential,
//         deleptonization fit. Device helpers are inline for Kokkos.

#include <math.h>

#include "athena.hpp"
#include "parameter_input.hpp"

//----------------------------------------------------------------------------------------
//! \class StellarProfile
//  \brief Read and interpolate 1D stellar profiles.
//         Storage in Kokkos::View for GPU access.
class StellarProfile {
 public:
  enum vars {
    alp  = 0,
    gxx  = 1,
    mass = 2,
    vel  = 3,
    rho  = 4,
    temp = 5,
    ye   = 6,
    press = 7,
    e    = 8,
    s    = 9,
    om   = 10,
    ab   = 11,
    phi  = 12,
    phiZ = 13,
    num_vars = 14,
  };

  StellarProfile(ParameterInput *pin, bool interpolate_Phi);

  // Interpolation function.  use_log: interpolate in log space (for rho, P).
  // interpolate_offset_r is handled internally by shifting rad before search.
  KOKKOS_INLINE_FUNCTION
  static Real Eval(const Kokkos::View<Real*> &d_pr,
                   const Kokkos::View<Real**> &d_pvars,
                   int vi, Real rad, int siz,
                   bool use_log = false,
                   bool interpolate_offset_r = false) {
    Real offset = interpolate_offset_r ? d_pr(0) / 2.0 : 0.0;
    Real r_eff = rad - offset;
    if (r_eff <= d_pr(0)) {
      return d_pvars(vi, 0);
    }
    if (r_eff >= d_pr(siz - 1)) {
      return d_pvars(vi, siz - 1);
    }
    // Binary search for interpolation interval
    int lo = 0, hi = siz;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (d_pr(mid) < r_eff) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    int idx = lo - 1;
    Real lam = (r_eff - d_pr(idx)) / (d_pr(idx + 1) - d_pr(idx));
    Real v0 = d_pvars(vi, idx);
    Real v1 = d_pvars(vi, idx + 1);
    if (use_log && v0 > 0.0 && v1 > 0.0) {
      return exp(log(v0) * (1.0 - lam) + log(v1) * lam);
    }
    return v0 * (1.0 - lam) + v1 * lam;
  }

  Kokkos::View<Real*> d_pr;
  Kokkos::View<Real**> d_pvars;
  int siz;
};

//----------------------------------------------------------------------------------------
// Rotation laws

KOKKOS_INLINE_FUNCTION
Real OmegaLaw(Real rad, Real Omega_0, Real Omega_A) {
  return Omega_0 / (1.0 + SQR(rad / Omega_A));
}

//----------------------------------------------------------------------------------------
// Vector potential for magnetic field initialization
// A_phi dipole model: https://arxiv.org/abs/1004.2896
//                      https://arxiv.org/abs/1403.1230

KOKKOS_INLINE_FUNCTION
Real VectorPotential(Real xp, Real yp, Real zp,
                     Real B0_amp, Real B0_rad,
                     int component) {
  Real const rad_cyl    = sqrt(SQR(xp) + SQR(yp));
  Real const oo_rad_cyl = 1.0 / rad_cyl;
  Real const rad        = sqrt(SQR(xp) + SQR(yp) + SQR(zp));

  // Spherical-to-Cartesian Jacobian: only azimuthal component contributes
  Real const dphdx = -yp * SQR(oo_rad_cyl);
  Real const dphdy =  xp * SQR(oo_rad_cyl);
  Real const dphdz = 0.0;

  // A_phi dipole
  Real Aph = 0.0;
  if (B0_amp > 0.0 && B0_rad > 0.0) {
    Aph = B0_amp * rad_cyl / (pow(rad, 3) + pow(B0_rad, 3));
  }

  Real const Ax = dphdx * Aph;
  Real const Ay = dphdy * Aph;
  Real const Az = dphdz * Aph;

  if (component == 1) return Ax;
  if (component == 2) return Ay;
  if (component == 3) return Az;
  return 0.0;  // unreachable
}

//----------------------------------------------------------------------------------------
// Deleptonization: Ye(rho) parametrization (Liebendoerfer 2005, Sherwood 2017 SFHo)

KOKKOS_INLINE_FUNCTION
Real Ye_of_rho(Real rho_cgs, Real log10_rho1, Real log10_rho2,
               Real Ye_2, Real Ye_c, Real Ye_H) {
  Real const Ye_1       = 0.5;
  Real const log10_rhoH = 15.0;
  Real const log10_rho  = log10(rho_cgs);

  Real const x = fmax(-1.0, fmin(1.0,
      (2.0 * log10_rho - log10_rho2 - log10_rho1) /
      (log10_rho2 - log10_rho1)));

  Real const m = (Ye_H - Ye_2) / (log10_rhoH - log10_rho2);

  if (log10_rho > log10_rho2) {
    return Ye_2 + m * (log10_rho - log10_rho2);
  } else {
    return 0.5 * (Ye_2 + Ye_1) + 0.5 * x * (Ye_2 - Ye_1) +
           Ye_c * (1.0 - fabs(x) +
                   4.0 * fabs(x) * (fabs(x) - 0.5) * (fabs(x) - 1.0));
  }
}

#endif  // PGEN_CCSN_HPP_
