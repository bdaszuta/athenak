//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_gauge_rhs.hpp
//! \brief Shared current-state lapse and shift RHS, excluding numerical dissipation.

#ifndef Z4C_Z4C_GAUGE_RHS_HPP_
#define Z4C_Z4C_GAUGE_RHS_HPP_

#include <cmath>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "z4c/z4c.hpp"
#include "z4c/driftcontrol/driftcontrol.hpp"

namespace z4c {

// Copy only consumed coefficients, not unused/uninitialized members of Options.
struct GaugeRhsCoefficients {
  Real chi_psi_power, chi_div_floor, chi_min_floor;
  Real lapse_oplog, lapse_harmonicf, lapse_harmonic, lapse_advect;
  bool slow_start_lapse;
  Real ssl_damping_amp, ssl_damping_time, ssl_damping_index;
  Real shift_ggamma, shift_alpha2ggamma, shift_hh, shift_advect, shift_eta;
};

// Immutable by-value host snapshot for one sample; no borrowed device pointers.
// Rebuild when sample time, options, or the live controller state changes.
struct GaugeRhsContext {
  GaugeRhsCoefficients opt;
  bool dc_on;
  int dc_variety;
  Real dc_inv_s2;
  Real dc_inv_tau[3], dc_c[3], dc_corr[3], dc_target[3], dc_gsupp[3];
};

// Cartesian coordinate components in geometrized code units, from current u0.
// chi is raw; chi_guarded is its division floor. khat is Khat, not Khat+2*Theta.
// gam is the evolved conformal Gamma, not a connection reconstructed from g.
struct GaugeRhsPoint {
  Real chi, chi_guarded, alpha, khat;
  Real beta[3], gam[3];
};

// Centered coordinate derivatives of raw fields and already-upwinded advection
// sums, accumulated in Cartesian direction order 0,1,2 by the caller.
struct GaugeRhsDerivatives {
  Real dalpha[3], dchi[3];
  Real Lalpha, Lbeta[3];
};

using GaugeInverseMetric =
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2>;

// Coordinate-time gauge derivatives; callers retain ownership of numerical KO.
struct GaugeRhsNoKO {
  Real alpha;
  Real beta[3];
};

// Host only: read current controller state without advancing it. The caller must
// supply pdc when drift control is enabled; a disabled controller is never read.
// time is the actual mesh/sample time, also passed unchanged to EvaluateGaugeRhs.
inline GaugeRhsContext MakeGaugeRhsContext(
    const Z4c::Options &opt, const DriftControl *pdc, Real time) {
  GaugeRhsContext c{};
  c.opt.chi_psi_power = opt.chi_psi_power;
  c.opt.chi_div_floor = opt.chi_div_floor;
  c.opt.chi_min_floor = opt.chi_min_floor;
  c.opt.lapse_oplog = opt.lapse_oplog;
  c.opt.lapse_harmonicf = opt.lapse_harmonicf;
  c.opt.lapse_harmonic = opt.lapse_harmonic;
  c.opt.lapse_advect = opt.lapse_advect;
  c.opt.slow_start_lapse = opt.slow_start_lapse;
  c.opt.ssl_damping_amp = opt.ssl_damping_amp;
  c.opt.ssl_damping_time = opt.ssl_damping_time;
  c.opt.ssl_damping_index = opt.ssl_damping_index;
  c.opt.shift_ggamma = opt.shift_ggamma;
  c.opt.shift_alpha2ggamma = opt.shift_alpha2ggamma;
  c.opt.shift_hh = opt.shift_hh;
  c.opt.shift_advect = opt.shift_advect;
  c.opt.shift_eta = opt.shift_eta;
  c.dc_on = opt.enable_driftcontrol;
  c.dc_variety = opt.dc_variety;
  if (!c.dc_on) return c;

  Real const fixed[3] = {opt.dc_fixed_x, opt.dc_fixed_y, opt.dc_fixed_z};
  c.dc_inv_s2 = 1.0 / SQR(opt.dc_damping_scale);
  for (int a = 0; a < 3; ++a) {
    c.dc_c[a] = (opt.dc_gaussian_center == DriftControl::CenterTracker)
                   ? pdc->GetPos(a) : fixed[a];
  }

  if (c.dc_variety == DriftControl::Relaxation) {
    for (int a = 0; a < 3; ++a) {
      c.dc_inv_tau[a] = 1.0 / opt.dc_relaxation_time;
      Real const e = pdc->GetPos(a) - fixed[a];
      c.dc_target[a] = opt.dc_kappa * e;
      if (opt.dc_gamma_suppress > 0.0) {
        Real const sc = 1.0 / (1.0 + opt.dc_gamma_suppress
                                  * Kokkos::fabs(e) / opt.dc_damping_scale);
        c.dc_gsupp[a] = opt.shift_ggamma * (1.0 - sc);
      }
    }
  } else if (c.dc_variety == DriftControl::PID) {
    for (int a = 0; a < 3; ++a) {
      Real const e = pdc->GetPos(a) - fixed[a];
      c.dc_corr[a] = -(opt.dc_Kp * e + opt.dc_Ki * pdc->GetIntegral(a)
                     + opt.dc_Kd * pdc->GetVel(a));
    }
  } else if (c.dc_variety == DriftControl::BDOB) {
    // GetU already owns BDOB gain, ramp, authority/slew limits, and scheduling.
    for (int a = 0; a < 3; ++a) {
      c.dc_corr[a] = -pdc->GetU(a);
    }
  } else if (c.dc_variety == DriftControl::DOB) {
    Real const wc2 = SQR(opt.dc_omega_c);
    Real const twzc = 2.0 * opt.dc_zeta * opt.dc_omega_c;
    for (int a = 0; a < 3; ++a) {
      Real const e = pdc->GetPos(a) - fixed[a];
      Real const v = pdc->GetVel(a);
      Real const fhat = pdc->GetP(a) + opt.dc_omega_o * v;
      c.dc_corr[a] = -(wc2 * e + twzc * v + fhat);
    }
  } else {
    // Existing fallback is the oscillator branch; retain it.
    Real const tau = opt.dc_damping_time;
    Real const zeta = opt.dc_damping_coeff;
    Real const inv_tau2 = 1.0 / SQR(tau);
    for (int a = 0; a < 3; ++a) {
      c.dc_corr[a] = -(2.0 * tau * zeta * pdc->GetVel(a)
                     + (pdc->GetPos(a) - fixed[a])) * inv_tau2;
    }
  }

  if (c.dc_variety != DriftControl::BDOB) {
    Real const gain[3] = {opt.dc_gain_x, opt.dc_gain_y, opt.dc_gain_z};
    for (int a = 0; a < 3; ++a) {
      c.dc_corr[a] *= gain[a];
      c.dc_gsupp[a] *= gain[a];
      c.dc_inv_tau[a] *= gain[a];
    }
    Real const ramp = pdc->RampFactor(time);
    if (ramp < 1.0) {
      for (int a = 0; a < 3; ++a) {
        c.dc_corr[a] *= ramp;
        c.dc_gsupp[a] *= ramp;
        c.dc_inv_tau[a] *= ramp;
      }
    }
  }
  return c;
}

// Device-callable, side-effect-free gauge law. g_uu is inverse CONFORMAL metric;
// x is the absolute global Cartesian cell center, not an extraction-frame offset.
// Preserve lapse, slow-start, base shift, harmonic, then drift addition order.
KOKKOS_INLINE_FUNCTION
GaugeRhsNoKO EvaluateGaugeRhs(
    const GaugeRhsPoint &u, const GaugeRhsDerivatives &d,
    const GaugeInverseMetric &g_uu, const Real x[3], Real time,
    const GaugeRhsContext &ctx) {
  auto const &opt = ctx.opt;
  GaugeRhsNoKO out{};

  Real const f = opt.lapse_oplog * opt.lapse_harmonicf
               + opt.lapse_harmonic * u.alpha;
  out.alpha = opt.lapse_advect * d.Lalpha - f * u.alpha * u.khat;
  if (opt.slow_start_lapse) {
    Real W2 = (u.chi > opt.chi_min_floor) ? u.chi : opt.chi_min_floor;
    Real W = pow(W2, 0.5);
    out.alpha += opt.ssl_damping_amp * (W - u.alpha)
                 * pow(W, opt.ssl_damping_index)
                 * exp(-0.5 * pow(time / opt.ssl_damping_time, 2));
  }

  for (int a = 0; a < 3; ++a) {
    out.beta[a] = opt.shift_ggamma * u.gam[a]
                + opt.shift_advect * d.Lbeta[a];
    out.beta[a] -= opt.shift_eta * u.beta[a];
  }
  for (int a = 0; a < 3; ++a) {
    out.beta[a] += opt.shift_alpha2ggamma * SQR(u.alpha) * u.gam[a];
    for (int b = 0; b < 3; ++b) {
      out.beta[a] += opt.shift_hh * u.alpha * u.chi_guarded
                     * (0.5 * u.alpha * d.dchi[b] - d.dalpha[b]) * g_uu(a,b);
    }
  }

  if (ctx.dc_on) {
    Real const r2 = SQR(x[0] - ctx.dc_c[0])
                  + SQR(x[1] - ctx.dc_c[1])
                  + SQR(x[2] - ctx.dc_c[2]);
    Real const g = Kokkos::exp(-r2 * ctx.dc_inv_s2);
    if (ctx.dc_variety == DriftControl::Relaxation) {
      for (int a = 0; a < 3; ++a) {
        out.beta[a] -= ctx.dc_inv_tau[a] * (u.beta[a] - ctx.dc_target[a]) * g;
        out.beta[a] -= ctx.dc_gsupp[a] * u.gam[a] * g;
      }
    } else {
      for (int a = 0; a < 3; ++a) {
        out.beta[a] -= ctx.dc_corr[a] * g;
      }
    }
  }
  return out;
}

}  // namespace z4c

#endif  // Z4C_Z4C_GAUGE_RHS_HPP_
