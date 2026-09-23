#ifndef RADIATION_M1_CLOSURE_HPP
#define RADIATION_M1_CLOSURE_HPP
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file radiation_m1_helpers.hpp
//  \brief helper functions for grey M1

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "radiation_m1/radiation_m1_macro.hpp"
#include "radiation_m1/radiation_m1_params.hpp"
#include "radiation_m1/radiation_m1_tensors.hpp"

namespace radiationm1 {

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::minmod2
//  \brief double minmod
KOKKOS_INLINE_FUNCTION
Real minmod2(Real rl, Real rp, Real th) {
  return Kokkos::fmin(1.0, Kokkos::fmin(th * rl, th * rp));
}

//----------------------------------------------------------------------------------------
//! \fn int radiationm1::CombinedIdx
//  \brief given flavor index, variable index and total no. of vars compute
//  combined idx
KOKKOS_INLINE_FUNCTION
int CombinedIdx(int nuidx, int varidx, int nvars) { return varidx + nuidx * nvars; }

// The EOS has one heavy energy weight because its two equilibrium densities
// coincide. The current densities and their absorption weights need not coincide.
KOKKOS_INLINE_FUNCTION
void partial_equilibrium_heavy_weights(const Real dtau, const Real kappa_2,
                                       const Real kappa_3, const int nspecies,
                                       Real &w_2, Real &w_3, Real &w_eq) {
  const Real a_2 = dtau * kappa_2;
  w_2 = a_2 / (1.0 + a_2);
  w_3 = 0.0;
  if (nspecies > 3) {
    const Real a_3 = dtau * kappa_3;
    w_3 = a_3 / (1.0 + a_3);
  }
  w_eq = (nspecies > 3) ? 0.5 * (w_2 + w_3) : w_2;
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::calc_proj
//  \brief Fluid projector: delta^a_b + u^a u_b
KOKKOS_INLINE_FUNCTION
void calc_proj(const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d,
               const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
               AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud) {
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      proj_ud(a, b) = (a == b) + u_u(a) * u_d(b);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::calc_K_from_rT
//  \brief Project out the radiation pressure tensor (in any frame)
KOKKOS_INLINE_FUNCTION
void calc_K_from_rT(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &rT_dd,
                    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud,
                    AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &K_dd) {
  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      K_dd(a, b) = 0.;
      for (int c = 0; c < 4; ++c) {
        for (int d = 0; d < 4; ++d) {
          K_dd(a, b) += proj_ud(c, a) * proj_ud(d, b) * rT_dd(c, d);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::calc_Pthin
//  \brief Compute lab thin pressure with an isotropic zero-norm completion.
//  E, F_d, and P_dd use the same densitization. At zero flux norm there is no
//  beam direction: use E (g_ab + n_a n_b)/3 to retain spatial trace E. This is
//  a lab-frame convention at the singular state, not a fluid-frame thick limit.
KOKKOS_INLINE_FUNCTION
void calc_Pthin(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
                const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
                const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
                const Real &E, const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
                AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &P_dd) {
  const Real F2 = tensor_dot(g_uu, F_d, F_d);
  if (F2 == 0.0) {
    const Real Eo3 = E / 3.0;
    for (int a = 0; a < 4; ++a) {
      for (int b = a; b < 4; ++b) {
        P_dd(a, b) = Eo3 * (g_dd(a, b) + n_d(a) * n_d(b));
      }
    }
    return;
  }
  Real fac = (F2 > 0 ? E / F2 : 0);
  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      P_dd(a, b) = fac * F_d(a) * F_d(b);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::calc_Pthick
//  \brief Compute radiation pressure in the thick limit
KOKKOS_INLINE_FUNCTION
void calc_Pthick(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
                 const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
                 const Real &W,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
                 const Real &E,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
                 AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &P_dd) {
  const Real v_dot_F = tensor_dot(g_uu, v_d, F_d);
  const Real W2 = W * W;
  const Real coef = 1. / (2. * W2 + 1.);

  // J/3
  const Real Jo3 = coef * ((2. * W2 - 1.) * E - 2. * W2 * v_dot_F);

  // tH = gamma_ud H_d
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> tH_d{};
  for (int a = 0; a < 4; ++a) {
    tH_d(a) = F_d(a) / W + coef * W * v_d(a) * ((4. * W2 + 1.) * v_dot_F - 4. * W2 * E);
  }

  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      P_dd(a, b) = Jo3 * (4. * W2 * v_d(a) * v_d(b) + g_dd(a, b) + n_d(a) * n_d(b));
      P_dd(a, b) += W * (tH_d(a) * v_d(b) + tH_d(b) * v_d(a));
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::flux_factor
//  \brief Compute the flux factor xi = H_a H^a / J^2
KOKKOS_INLINE_FUNCTION
Real flux_factor(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
                 const Real &J, AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &H_d) {
  const bool use_flux = Kokkos::isfinite(J) && (J > 0.0);
  const Real xi = (use_flux ? tensor_dot(g_uu, H_d, H_d) / (J*J) : 0);
  return Kokkos::fmax(0.0, Kokkos::fmin(xi, 1.0));
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::assemble_fnu
//  \brief Assemble the unit-norm radiation number current without cancellation.
KOKKOS_INLINE_FUNCTION
void assemble_fnu(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
                  const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &T_dd,
                  const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
                  const Real &J, const Real h_fac,
                  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &fnu_u) {
  if (!(h_fac > 0.0)) {
    for (int a = 0; a < 4; ++a) {
      fnu_u(a) = u_u(a);
    }
    return;
  }

  // Before H -> h_fac H, fnu^a = -T^a_b u^b / J.  Apply the accepted
  // projection as h_fac fnu_raw + (1 - h_fac) u, avoiding u + H/J.
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> T_ud{};
  tensor_contract(g_uu, T_dd, T_ud);
  for (int a = 0; a < 4; ++a) {
    Real fnu_raw = 0.0;
    for (int b = 0; b < 4; ++b) {
      fnu_raw -= T_ud(a, b) * u_u(b) / J;
    }
    fnu_u(a) = h_fac * fnu_raw + (1.0 - h_fac) * u_u(a);
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::floor_number
//  \brief Enforce a finite radiation-number floor
KOKKOS_INLINE_FUNCTION
Real floor_number(const Real &N, const RadiationM1Params &params) {
  return Kokkos::isfinite(N) ? Kokkos::fmax(params.rad_N_floor, N)
                             : params.rad_N_floor;
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::assemble_rT
//  \brief Assemble the radiation stress tensor in any frame
KOKKOS_INLINE_FUNCTION
void assemble_rT(const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d,
                 const Real &J,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &H_d,
                 const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &K_dd,
                 AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &rT_dd) {
  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      rT_dd(a, b) = J * u_d(a) * u_d(b) + H_d(a) * u_d(b) + H_d(b) * u_d(a) + K_dd(a, b);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::calc_J_from_rT
//  \brief Project out the radiation energy (in any frame)
KOKKOS_INLINE_FUNCTION
Real calc_J_from_rT(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &rT_dd,
                    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u) {
  return tensor_dot(rT_dd, u_u, u_u);
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::calc_H_from_rT
//  \brief Project out the radiation fluxes (in any frame)
KOKKOS_INLINE_FUNCTION
void calc_H_from_rT(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &rT_dd,
                    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
                    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud,
                    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &H_d) {
  for (int a = 0; a < 4; ++a) {
    H_d(a) = 0.;
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        H_d(a) -= proj_ud(b, a) * u_u(c) * rT_dd(b, c);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::apply_closure
//  \brief Computes the closure in the lab frame given the Eddington factor chi
KOKKOS_INLINE_FUNCTION
void apply_closure(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
                   const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
                   const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_d,
                   const Real &w_lorentz,
                   const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
                   const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &v_d,
                   const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &proj_ud,
                   const Real &E,
                   const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
                   const Real &chi, AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &P_dd,
                   const RadiationM1Params &params) {
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> Pthin_dd{};
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> Pthick_dd{};

  calc_Pthin(g_dd, g_uu, n_d, E, F_d, Pthin_dd);
  calc_Pthick(g_dd, g_uu, n_d, w_lorentz, v_d, E, F_d, Pthick_dd);

  const Real dthick = 3. * (1 - chi) / 2.;
  const Real dthin = 1. - dthick;

  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      P_dd(a, b) = dthick * Pthick_dd(a, b) + dthin * Pthin_dd(a, b);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::calc_E_flux
//  \brief Compute the radiation energy flux
KOKKOS_INLINE_FUNCTION
Real calc_E_flux(const Real &alp,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &beta_u,
                 const Real &E,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_u,
                 const int &direction) {  // 1:x 2:y 3:z
  return alp * F_u(direction) - beta_u(direction) * E;
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::calc_F_flux
//  \brief Compute the flux of neutrino energy flux
KOKKOS_INLINE_FUNCTION
Real calc_F_flux(const Real &alp,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &beta_u,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
                 const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &P_ud,
                 const int &direction,    // 1:x 2:y 3:z
                 const int &component) {  // 1:x 2:y 3:z
  return alp * P_ud(direction, component) - beta_u(direction) * F_d(component);
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::calc_rad_sources
//  \brief Computes the sources S_a = [eta - k_abs J] u_a - [k_abs + k_scat] H_a
//  WARNING: be consistent with the densitization of eta, J, and H_d
KOKKOS_INLINE_FUNCTION
void calc_rad_sources(const Real &eta, const Real &kabs, const Real &kscat,
                      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_d,
                      const Real &J,
                      const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &H_d,
                      AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &S_d) {
  for (int a = 0; a < 4; ++a) {
    S_d(a) = (eta - kabs * J) * u_d(a) - (kabs + kscat) * H_d(a);
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::calc_rE_source
//  \brief Computes the source term for E: -alp n^a S_a
KOKKOS_INLINE_FUNCTION
Real calc_rE_source(const Real &alp,
                    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &n_u,
                    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &S_d) {
  return -alp * tensor_dot(n_u, S_d);
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::calc_rF_source
//  \brief Computes the source term for F_a: alp gamma^b_a S_b
KOKKOS_INLINE_FUNCTION
void calc_rF_source(const Real &alp,
                    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 2> &gamma_ud,
                    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &S_d,
                    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &tS_d) {
  for (int a = 0; a < 4; ++a) {
    tS_d(a) = 0.0;
    for (int b = 0; b < 4; ++b) {
      tS_d(a) += alp * gamma_ud(b, a) * S_d(b);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn Real radiationm1::comoving_moment_squared
//  \brief Compute H_a H^a from non-cancelling spatial components.
KOKKOS_INLINE_FUNCTION
Real comoving_moment_squared(
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &H_d) {
  const Real alpha = 1.0 / Kokkos::sqrt(-g_uu(0, 0));
  const Real W = alpha * u_u(0);
  if (W * W <= 2.0) {
    return tensor_dot(g_uu, H_d, H_d);
  }

  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> beta_u{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_u{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> v_d{};
  AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> Hperp_d{};
  for (int a = 1; a < 4; ++a) {
    beta_u(a) = -g_uu(0, a) / g_uu(0, 0);
    v_u(a) = u_u(a) / W + beta_u(a) / alpha;
  }
  // Lower only after every component is available for off-diagonal metrics.
  for (int a = 1; a < 4; ++a) {
    for (int b = 1; b < 4; ++b) {
      v_d(a) += g_dd(a, b) * v_u(b);
    }
  }

  const Real v2 = 1.0 - 1.0 / (W * W);
  Real Hdotv = 0.0;
  for (int a = 1; a < 4; ++a) {
    Hdotv += H_d(a) * v_u(a);
  }
  for (int a = 1; a < 4; ++a) {
    Hperp_d(a) = H_d(a) - Hdotv * v_d(a) / v2;
  }

  Real Hperp2 = 0.0;
  for (int a = 1; a < 4; ++a) {
    for (int b = 1; b < 4; ++b) {
      const Real gamma_uu =
          g_uu(a, b) - beta_u(a) * beta_u(b) * g_uu(0, 0);
      Hperp2 += gamma_uu * Hperp_d(a) * Hperp_d(b);
    }
  }
  return Hperp2 + (Hdotv * Hdotv / v2) / (W * W);
}

//----------------------------------------------------------------------------------------
//! \fn bool radiationm1::enforce_comoving_moment_safety
//  \brief Enforce finite, positive, and causal fluid-frame radiation moments without
//  applying the Eulerian energy floor.
KOKKOS_INLINE_FUNCTION
bool enforce_comoving_moment_safety(
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_dd,
    const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu,
    const AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &u_u, Real &J,
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &H_d,
    const RadiationM1Params &params, Real *h_fac = nullptr) {
  if (h_fac != nullptr) {
    *h_fac = 0.0;
  }
  bool state_is_valid = Kokkos::isfinite(J) && (J > 0.0);
  for (int a = 0; a < 4; ++a) {
    state_is_valid = state_is_valid && Kokkos::isfinite(H_d(a));
  }

  // J or H itself is invalid: use the documented invalid-state fallback.
  if (!state_is_valid) {
    J = params.rad_E_floor;
    for (int a = 0; a < 4; ++a) {
      H_d(a) = 0.0;
    }
    return false;
  }

  // Keep H_a u^a = 0 exactly after the projector contraction.
  H_d(0) = 0.0;
  for (int a = 1; a < 4; ++a) {
    H_d(0) -= u_u(a) * H_d(a) / u_u(0);
  }
  const Real H2 = comoving_moment_squared(g_dd, g_uu, u_u, H_d);
  if (!Kokkos::isfinite(H2)) {
    J = params.rad_E_floor;
    for (int a = 0; a < 4; ++a) {
      H_d(a) = 0.0;
    }
    return false;
  }

  // A finite positive J remains physical.  A finite H2 < 0 is a
  // cancellation-corrupted, unresolved flux, so discard H without replacing J.
  // This preserves absorption/emission through J.
  if (H2 < 0.0) {
    for (int a = 0; a < 4; ++a) {
      H_d(a) = 0.0;
    }
    return false;
  }

  const Real lim = J * J * (1 - params.rad_eps);
  Real fac = 1.0;
  if (H2 > lim) {
    fac = Kokkos::sqrt(lim / H2);
    for (int a = 0; a < 4; ++a) {
      H_d(a) *= fac;
    }
  }
  if (h_fac != nullptr) {
    *h_fac = fac;
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn void radiationm1::apply_floor
//  \brief Enforce the absolute floor and causal limit on Eulerian moments E and F_a.
//  If E drops below rad_E_floor, F is zeroed (isotropic floor state), mirroring GRA.
KOKKOS_INLINE_FUNCTION
void apply_floor(const AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> &g_uu, Real &E,
                 AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> &F_d,
                 const RadiationM1Params &params) {
  bool state_is_finite = Kokkos::isfinite(E);
  for (int a = 0; a < 4; ++a) {
    state_is_finite = state_is_finite && Kokkos::isfinite(F_d(a));
  }

  const Real F2 = state_is_finite ? tensor_dot(g_uu, F_d, F_d) : 0.0;
  if (!state_is_finite || !Kokkos::isfinite(F2)) {
    E = params.rad_E_floor;
    for (int a = 0; a < 4; ++a) {
      F_d(a) = 0.0;
    }
    return;
  }

  const bool E_below_floor = E < params.rad_E_floor;
  E = Kokkos::fmax(params.rad_E_floor, E);
  if (E_below_floor) {
    for (int a = 0; a < 4; ++a) {
      F_d(a) = 0.0;
    }
    return;
  }

  const Real lim = E * E * (1 - params.rad_eps);
  if (F2 > lim) {
    const Real fac = Kokkos::sqrt(lim / F2);
    for (int a = 0; a < 4; ++a) {
      F_d(a) *= fac;
    }
  }
}

}  // namespace radiationm1
#endif  // RADIATION_M1_CLOSURE_HPP
