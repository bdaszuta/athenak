#ifndef EOS_PRIMITIVE_SOLVER_FERMI_DIRAC_INTEGRALS_HPP_
#define EOS_PRIMITIVE_SOLVER_FERMI_DIRAC_INTEGRALS_HPP_
//========================================================================================
// Complete Fermi-Dirac integrals needed by the CompOSE neutrino-equilibrium solver.
//
// The rational minimax approximations below are adapted from bns_nurates.
//
// Copyright 2024 Albino Perego, David Radice, Federico Maria Guercilena,
// Leonardo Chiesa, and Maitraya Bhattacharyya
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list
// of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or other
// materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors may be
// used to endorse or promote products derived from this software without specific prior
// written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
// SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//========================================================================================

#include <Kokkos_Core.hpp>

#include "../../athena.hpp"

namespace Primitive {
namespace FermiDirac {

inline constexpr Real fdi_exp_two = 7.38905609893065023;

//! Complete Fermi-Dirac integral F_1(eta).
KOKKOS_INLINE_FUNCTION
Real F1(const Real eta) {
  const Real x = -Kokkos::fabs(eta);
  Real fd = 0.0;

  constexpr Real a[10] = {1.0,
                          22189.1070807945062,
                          4915.92700908746777,
                          322.901386168881348,
                          5.9897442965804548,
                          0.00397641173774375092,
                          88756.428323178025,
                          25002.3197546553836,
                          2389.06277237306633,
                          88.376214553692756};
  constexpr Real b[17] = {
      145.488167182330098,  251.392824471576922,   56.6537141912783024,
      17.9918985363509694,  20.1369115558099802,   7.09659390228556164,
      0.199701180197912643, -0.403173132925886253, 0.0792966701498222697,
      606.0757707716040,    374.1806357435014,     252.1367051536344,
      27.2746245830016,     -61.57766112137513,    -53.72117554363975,
      -25.678454878692950,  -7.1995819520154718};
  constexpr Real c[2] = {1.64493406684822644, 0.5};

  if (x < -2.0) {
    const Real ex = Kokkos::exp(x);
    const Real t = ex*fdi_exp_two;
    fd = ex*(a[0] -
             ex*(a[1] + t*(a[2] + t*(a[3] + t*(a[4] + t*a[5]))))/
                (a[6] + t*(a[7] + t*(a[8] + t*(a[9] + t)))));
  } else if (x <= 0.0) {
    const Real s = -0.5*x;
    const Real t = 1.0 - s;
    fd = (b[0] +
          t*(b[1] +
             t*(b[2] +
                t*(b[3] +
                   t*(b[4] + t*(b[5] + t*(b[6] + t*(b[7] - t*b[8]))))))))/
         (b[9] +
          s*(b[10] +
             s*(b[11] +
                s*(b[12] +
                   s*(b[13] + s*(b[14] + s*(b[15] + s*(b[16] - s))))))));
  }
  if (eta > 0.0) {
    fd = -fd + c[0] + c[1]*eta*eta;
  }
  return fd;
}

//! Complete Fermi-Dirac integral F_2(eta).
KOKKOS_INLINE_FUNCTION
Real F2(const Real eta) {
  const Real x = -Kokkos::fabs(eta);
  Real fd = 0.0;

  constexpr Real a[8] = {2.0,
                         1914.06748184935743,
                         273.085756700981399,
                         8.5861610217850095,
                         0.0161890243763741414,
                         7656.2699273974454,
                         1399.35442210906621,
                         72.929152915475392};
  constexpr Real b[17] = {
      2711.49678259128843,  1299.85460914884154,  222.606134197895041,
      172.881855215582924,  112.951038040682055,  24.0376482128898634,
      -2.68393549333878715, -2.14077421411719935, 0.326188299771397236,
      2517.1726659917047,   3038.7689794575778,   2541.7823512372631,
      1428.0589853413436,   531.62378035996132,   122.54595216479181,
      8.395768655115050,    -3.9142702096919080};
  constexpr Real c[2] = {3.28986813369645287, 1.0/3.0};

  if (x < -2.0) {
    const Real ex = Kokkos::exp(x);
    const Real t = ex*fdi_exp_two;
    fd = ex*(a[0] -
             ex*(a[1] + t*(a[2] + t*(a[3] + t*a[4])))/
                (a[5] + t*(a[6] + t*(a[7] + t))));
  } else if (x <= 0.0) {
    const Real s = -0.5*x;
    const Real t = 1.0 - s;
    fd = (b[0] +
          t*(b[1] +
             t*(b[2] +
                t*(b[3] +
                   t*(b[4] + t*(b[5] + t*(b[6] + t*(b[7] - t*b[8]))))))))/
         (b[9] +
          s*(b[10] +
             s*(b[11] +
                s*(b[12] +
                   s*(b[13] + s*(b[14] + s*(b[15] + s*(b[16] - s))))))));
  }
  if (eta > 0.0) {
    fd += eta*(c[0] + c[1]*eta*eta);
  }
  return fd;
}

//! Complete Fermi-Dirac integral F_3(eta).
KOKKOS_INLINE_FUNCTION
Real F3(const Real eta) {
  const Real x = -Kokkos::fabs(eta);
  Real fd = 0.0;

  constexpr Real a[8] = {6.0,
                         5121.6401850302408,
                         664.28706260743472,
                         19.0856927562699544,
                         0.0410982603688952131,
                         13657.7071600806539,
                         2136.54222460571183,
                         92.376788603062645};
  constexpr Real b[15] = {
      7881.24597452900838, 4323.07526636309661, 1260.13125873282465,
      653.359212389160499, 354.630774329461644, 113.373708671587772,
      19.9559488532742796, 1.59407954898394322, 2570.7250703533430,
      2972.7443644211129,  2393.9995533270879,  1259.0724833462608,
      459.86413596901097,  112.60906419590854,  16.468882811659000};
  constexpr Real c[3] = {11.3643939539669510, 4.93480220054467931, 0.25};

  if (x < -2.0) {
    const Real ex = Kokkos::exp(x);
    const Real t = ex*fdi_exp_two;
    fd = ex*(a[0] -
             ex*(a[1] + t*(a[2] + t*(a[3] + t*a[4])))/
                (a[5] + t*(a[6] + t*(a[7] + t))));
  } else if (x <= 0.0) {
    const Real s = -0.5*x;
    const Real t = 1.0 - s;
    fd = (b[0] +
          t*(b[1] +
             t*(b[2] +
                t*(b[3] +
                   t*(b[4] + t*(b[5] + t*(b[6] + t*b[7])))))))/
         (b[8] +
          s*(b[9] +
             s*(b[10] +
                s*(b[11] + s*(b[12] + s*(b[13] + s*(b[14] + s)))))));
  }
  if (eta > 0.0) {
    const Real eta2 = eta*eta;
    fd = -fd + c[0] + eta2*(c[1] + eta2*c[2]);
  }
  return fd;
}

}  // namespace FermiDirac
}  // namespace Primitive

#endif  // EOS_PRIMITIVE_SOLVER_FERMI_DIRAC_INTEGRALS_HPP_
