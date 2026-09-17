#ifndef PGEN_CCSN_M1_PROJECTION_HPP_
#define PGEN_CCSN_M1_PROJECTION_HPP_

#include <math.h>

#include "athena.hpp"
#include "eos/primitive-solver/primitive_solver.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "radiation_m1/radiation_m1_equilibria.hpp"
#include "radiation_m1/radiation_m1_macro.hpp"
#include "radiation_m1/radiation_m1_params.hpp"

namespace ccsn {

constexpr int projection_nspecies = 4;
constexpr int projection_nvars = 5;
constexpr int projection_nue = 0;
constexpr int projection_anue = 1;

static_assert(projection_nspecies == radiationm1::equilibrium_nspecies,
              "the CCSN projection requires the equilibrium interface's group count");

//! Compute the four transport-group equilibrium densities for the active backend.
//!
//! The backend, its physical constants and its unit convention are hidden behind
//! the module's equilibrium interface. The matter state enters as temperature and
//! electron-neutrino degeneracy; the groups return as [nue, anue, nux, anux].
KOKKOS_INLINE_FUNCTION bool EquilibriumDensities(
    const radiationm1::RadiationM1OpacityType opacity_type, const Real mu_n,
    const Real mu_p, const Real mu_e, const Real temp,
    const Primitive::UnitSystem &code_units,
    const Primitive::UnitSystem &eos_units,
    const Primitive::UnitSystem &ngs_units,
    Real n[projection_nspecies], Real J[projection_nspecies]) {
  const Real eta_nue = (mu_p + mu_e - mu_n) / temp;
  return radiationm1::equilibrium_densities(opacity_type, code_units, eos_units,
                                            ngs_units, temp, eta_nue, n, J);
}

//! Build the equilibrium Eulerian M1 moments for the supplied matter state.
//!
//! The selected backend supplies fluid-frame equilibrium densities in its own
//! physical-constant convention. Isotropic fluid-frame radiation is then boosted
//! with the matter velocity into [E, F_x, F_y, F_z, N]. All moments are
//! undensitized. Returns false for a nonphysical or nonfinite state.
template <class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION bool EquilibriumM1Target(
    const Primitive::PrimitiveSolver<EOSPolicy, ErrorPolicy> &ps,
    const radiationm1::RadiationM1OpacityType opacity_type,
    const Primitive::UnitSystem &code_units,
    const Primitive::UnitSystem &eos_units,
    const Primitive::UnitSystem &ngs_units,
    const Real prim[NPRIM],
    const Real g3d[NSPMETRIC], Real target[projection_nspecies][projection_nvars]) {
  const auto &eos = ps.GetEOS();
  const Real nb = prim[PRH];
  const Real temp = prim[PTM];
  Real Y[MAX_SPECIES] = {0.0};
  for (int s = 0; s < eos.GetNSpecies(); ++s) {
    Y[s] = prim[PYF + s];
  }
  if (!(nb > 0.0) || !(temp > 0.0) || !Kokkos::isfinite(nb) ||
      !Kokkos::isfinite(temp)) {
    return false;
  }

  const Real mu_b = eos.GetBaryonChemicalPotential(nb, temp, Y);
  const Real mu_q = eos.GetChargeChemicalPotential(nb, temp, Y);
  const Real mu_l = eos.GetElectronLeptonChemicalPotential(nb, temp, Y);
  const Real mu_n = mu_b;
  const Real mu_p = mu_b + mu_q;
  const Real mu_e = mu_l - mu_q;
  if (!Kokkos::isfinite(mu_n) || !Kokkos::isfinite(mu_p) ||
      !Kokkos::isfinite(mu_e)) {
    return false;
  }

  Real n[projection_nspecies], J[projection_nspecies];
  if (!EquilibriumDensities(opacity_type, mu_n, mu_p, mu_e, temp, code_units,
                            eos_units, ngs_units, n, J)) {
    return false;
  }

  const Real zx = prim[PVX];
  const Real zy = prim[PVY];
  const Real zz = prim[PVZ];
  const Real zdx = g3d[S11] * zx + g3d[S12] * zy + g3d[S13] * zz;
  const Real zdy = g3d[S12] * zx + g3d[S22] * zy + g3d[S23] * zz;
  const Real zdz = g3d[S13] * zx + g3d[S23] * zy + g3d[S33] * zz;
  const Real W2 = 1.0 + zx * zdx + zy * zdy + zz * zdz;
  if (!(W2 >= 1.0) || !Kokkos::isfinite(W2)) {
    return false;
  }
  const Real W = Kokkos::sqrt(W2);

  for (int s = 0; s < projection_nspecies; ++s) {
    if (!(n[s] > 0.0) || !(J[s] > 0.0) || !Kokkos::isfinite(n[s]) ||
        !Kokkos::isfinite(J[s])) {
      return false;
    }
    target[s][M1_E_IDX] = ONE_3RD * J[s] * (4.0 * W2 - 1.0);
    const Real flux_fac = 4.0 * ONE_3RD * J[s] * W;
    target[s][M1_FX_IDX] = flux_fac * zdx;
    target[s][M1_FY_IDX] = flux_fac * zdy;
    target[s][M1_FZ_IDX] = flux_fac * zdz;
    target[s][M1_N_IDX] = W * n[s];
    for (int v = 0; v < projection_nvars; ++v) {
      if (!Kokkos::isfinite(target[s][v])) {
        return false;
      }
    }
  }
  return true;
}

//! Contraction of two spatial covectors with the inverse metric, g^ij a_i b_j.
//!
//! Used to measure differences of lowered components, so that the measure does
//! not depend on how the spatial coordinates are laid out. A maximum over
//! coordinate components is not invariant: for the same physical difference it
//! changes by up to sqrt(3) when the axes are rotated, so a criterion built on it
//! would accept in one basis what it rejects in another.
KOKKOS_INLINE_FUNCTION Real MetricContractU(const Real g3u[NSPMETRIC],
                                            const Real a[3], const Real b[3]) {
  const Real gb0 = g3u[S11] * b[0] + g3u[S12] * b[1] + g3u[S13] * b[2];
  const Real gb1 = g3u[S12] * b[0] + g3u[S22] * b[1] + g3u[S23] * b[2];
  const Real gb2 = g3u[S13] * b[0] + g3u[S23] * b[1] + g3u[S33] * b[2];
  return a[0] * gb0 + a[1] * gb1 + a[2] * gb2;
}

//! Metric norm sqrt(g^ij a_i a_j) of a lowered spatial vector.
//!
//! The self-contraction is non-negative for a positive-definite inverse metric;
//! the clamp only guards a round-off negative under the square root.
KOKKOS_INLINE_FUNCTION Real MetricNormU(const Real g3u[NSPMETRIC],
                                        const Real a[3]) {
  return Kokkos::sqrt(Kokkos::fmax(0.0, MetricContractU(g3u, a, a)));
}

//! Return the largest normalized difference between two four-species M1 states.
//!
//! Energy and number density are frame-dependent scalars but not
//! coordinate-dependent, so each is compared as a scalar against the target
//! energy and the target number density. The three flux components are the
//! components of one spatial covector and are measured as a single metric norm,
//! sqrt(g^ij dF_i dF_j), against the target energy, never componentwise: the
//! energy scale is the same for all three because a flux and an energy density
//! share units, and the norm is what makes the test independent of the
//! coordinates. The M1 floors provide nonzero scales when a target is small.
KOKKOS_INLINE_FUNCTION Real ProjectionDifference(
    const Real state[projection_nspecies][projection_nvars],
    const Real target[projection_nspecies][projection_nvars],
    const Real g3u[NSPMETRIC], Real energy_floor, Real number_floor) {
  Real error = 0.0;
  for (int s = 0; s < projection_nspecies; ++s) {
    const Real energy_scale =
        Kokkos::fmax(Kokkos::fabs(target[s][M1_E_IDX]), energy_floor);
    error = Kokkos::fmax(
        error,
        Kokkos::fabs(state[s][M1_E_IDX] - target[s][M1_E_IDX]) / energy_scale);
    const Real flux_diff[3] = {state[s][M1_FX_IDX] - target[s][M1_FX_IDX],
                               state[s][M1_FY_IDX] - target[s][M1_FY_IDX],
                               state[s][M1_FZ_IDX] - target[s][M1_FZ_IDX]};
    error = Kokkos::fmax(error, MetricNormU(g3u, flux_diff) / energy_scale);
    const Real number_scale =
        Kokkos::fmax(Kokkos::fabs(target[s][M1_N_IDX]), number_floor);
    error = Kokkos::fmax(
        error,
        Kokkos::fabs(state[s][M1_N_IDX] - target[s][M1_N_IDX]) / number_scale);
  }
  return error;
}

//! Fixed right-hand side of the equilibrium solve.
//!
//! The entering matter and radiation states define the cell's total energy,
//! momentum and electron lepton number, in the order of projection_nvars.
//! These totals are what the previous complement subtraction held fixed; the
//! solver preserves them exactly, so they are evaluated once per cell and
//! never modified. Energy and momentum are the sums of already-lowered
//! components, so no metric factor enters here. Inputs are undensitized; mb
//! converts the lepton-number difference to a mass density, as `cons[CYD]`
//! stores D * Ye.
KOKKOS_INLINE_FUNCTION void ConservedTotals(
    const Real old_cons[NCONS],
    const Real old_rad[projection_nspecies][projection_nvars], Real mb,
    Real tot[projection_nvars]) {
  tot[0] = old_cons[CTA];
  tot[1] = old_cons[CSX];
  tot[2] = old_cons[CSY];
  tot[3] = old_cons[CSZ];
  for (int s = 0; s < projection_nspecies; ++s) {
    tot[0] += old_rad[s][M1_E_IDX];
    tot[1] += old_rad[s][M1_FX_IDX];
    tot[2] += old_rad[s][M1_FY_IDX];
    tot[3] += old_rad[s][M1_FZ_IDX];
  }
  tot[4] = old_cons[CYD] +
           mb * (old_rad[projection_nue][M1_N_IDX] -
                 old_rad[projection_anue][M1_N_IDX]);
}

//! Conservative contribution of an M1 state to those totals.
//!
//! Componentwise addition of lowered components: a flux contributes to the
//! momentum it shares components with, so no metric factor appears here. mb
//! converts the lepton-number difference to a mass density.
KOKKOS_INLINE_FUNCTION void FieldConserved(
    const Real rad[projection_nspecies][projection_nvars], Real mb,
    Real field[projection_nvars]) {
  field[0] = 0.0;
  field[1] = 0.0;
  field[2] = 0.0;
  field[3] = 0.0;
  for (int s = 0; s < projection_nspecies; ++s) {
    field[0] += rad[s][M1_E_IDX];
    field[1] += rad[s][M1_FX_IDX];
    field[2] += rad[s][M1_FY_IDX];
    field[3] += rad[s][M1_FZ_IDX];
  }
  field[4] = mb * (rad[projection_nue][M1_N_IDX] -
                   rad[projection_anue][M1_N_IDX]);
}

//! Recover primitives from a trial matter complement without allowing repairs.
//!
//! ConToPrim operates on local copies because projection may not alter live
//! state until the cell is accepted. Recovery fails if C2P reports an error,
//! applies a floor or adjustment, changes any conserved value or magnetic-field
//! component, or produces a nonfinite active primitive. On success, prim and
//! cons_out contain the recovered active primitives and unchanged conserved
//! state.
//!
//! fail_code names the rejection that fired, so a failed projection reports its
//! cause instead of only that it failed: 0 accepted, 2 conserved floor,
//! 3 primitive floor, 4 conserved adjustment, 5 conserved value changed or
//! nonfinite, 6 active primitive nonfinite, 7 magnetic field changed, and
//! 20 + the Primitive::Error enumerator when C2P returned an error (21
//! RHO_TOO_BIG, 22 RHO_TOO_SMALL, 23 NANS_IN_CONS, 24 MAG_TOO_BIG, 25
//! BRACKETING_FAILED, 26 NO_SOLUTION, 27 CONS_FLOOR, 28 PRIM_FLOOR, 29
//! CONS_ADJUSTED). The tests are applied in the same order as the
//! pre-instrumentation combined condition, so the reported code is the first
//! condition that was true.
template <class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION bool RecoverMatter(
    const Primitive::PrimitiveSolver<EOSPolicy, ErrorPolicy> &ps,
    const Real cons_in[NCONS], const Real b_in[NMAG],
    const Real g3d[NSPMETRIC], const Real g3u[NSPMETRIC], Real prim[NPRIM],
    Real cons_out[NCONS], int *fail_code) {
  Real b[NMAG] = {b_in[IBX], b_in[IBY], b_in[IBZ]};
  Real metric[NSPMETRIC], inverse_metric[NSPMETRIC];
  Real original_cons[NCONS];
  for (int n = 0; n < NSPMETRIC; ++n) {
    metric[n] = g3d[n];
    inverse_metric[n] = g3u[n];
  }
  for (int n = 0; n < NCONS; ++n) {
    original_cons[n] = cons_in[n];
    cons_out[n] = cons_in[n];
  }

  const Primitive::SolverResult result =
      ps.ConToPrim(prim, cons_out, b, metric, inverse_metric);
  if (result.error != Primitive::Error::SUCCESS) {
    // Name the C2P failure itself: "could not solve" reads differently from
    // "solved but repaired", and the two need different responses.
    *fail_code = 20 + static_cast<int>(result.error);
    return false;
  }
  if (result.cons_floor) {
    *fail_code = 2;
    return false;
  }
  if (result.prim_floor) {
    *fail_code = 3;
    return false;
  }
  if (result.cons_adjusted) {
    *fail_code = 4;
    return false;
  }
  for (int n = 0; n < NCONS; ++n) {
    if (!Kokkos::isfinite(cons_out[n]) || cons_out[n] != original_cons[n]) {
      *fail_code = 5;
      return false;
    }
  }
  const int nprim_active = PYF + ps.GetEOS().GetNSpecies();
  for (int n = 0; n < nprim_active; ++n) {
    if (!Kokkos::isfinite(prim[n])) {
      *fail_code = 6;
      return false;
    }
  }
  if (b[IBX] != b_in[IBX] || b[IBY] != b_in[IBY] || b[IBZ] != b_in[IBZ]) {
    *fail_code = 7;
    return false;
  }
  *fail_code = 0;
  return true;
}

//! Outcome of a failed projection, handed back to the caller.
//!
//! code is the rejection site: 100 the equilibrium field is not computable, 101
//! the matter recovery was rejected, 103 the exact complement of the converged
//! field is not recoverable, 104 its field is not computable, 105 the acceptance
//! test of the converged pair refused at least once and the step budget ran out
//! before the pair moved inside it, 106 the iteration budget ran out, 110
//! the cell's own state is not admissible at the start point, 111 a Jacobian
//! column could not be formed, 112 the Jacobian is numerically singular, 113 the
//! line search found no reducing step.
//!
//! detail is the RecoverMatter code at the sites that relay a chain rejection of
//! a point, which are 101, 103, 104 and 110 (2-7, or 20 + the Primitive::Error
//! enumerator when C2P returned an error), 8 when the recovered primitives do
//! not reproduce the requested conserved state, the failed column index at 111,
//! and 0 elsewhere - including at 110 when the failure is an invalid scale
//! rather than a refused point. iteration is the iteration the failure was
//! detected in and -1 where no iteration applies.
//!
//! residual is the scaled residual of the last point the chain evaluated
//! successfully at every site but 105, where it is the acceptance-test difference
//! of the last refused pair, and -1 when none has been evaluated yet; evals is the
//! number of chain evaluations attempted, a refused probe included, and
//! worst_term the term that carried the measure (0 energy, 1 momentum, 4
//! electron lepton number, -1 before the first evaluation); the pair records the
//! measure together with the term that produced it.
//!
//! matter_rho, matter_T and matter_Ye describe the last point whose chain
//! evaluation completed, which is the state the failing test saw at every site
//! but 100 and 104; there the test applies to C2P's state, which is recorded as
//! c2p instead, so the three hold the previous point (0 before the first). trial
//! is the undensitized matter state handed to
//! C2P and is filled at 100, 101, 103 and 104, the sites that relay a chain
//! refusal of a point, so the line prints there the state the chain refused; at
//! 110 the chain fills it with the entering state, which the caller prints
//! itself, so the line does not repeat it and does not print trial at 110.
//! c2p_rho, c2p_T and c2p_Ye are what C2P left behind when it returned a state
//! rather than an error, and c2p_valid is 1 exactly where they were recorded: on
//! an error path C2P leaves its output array undefined, so they are filled only
//! for detail below 20.
//!
//! The report exists so that the caller can print one self-contained line per
//! failed cell. Device output from different threads and ranks is ordered
//! arbitrarily, so a failure reported across several prints cannot be matched up
//! with its own state afterwards; every site records here, and nothing in this
//! file prints.
struct ProjectionFailureReport {
  int code;
  int detail;
  int iteration;
  Real residual;
  int evals;
  int worst_term;
  Real matter_rho;
  Real matter_T;
  Real matter_Ye;
  Real trial[NCONS];
  Real c2p_rho;
  Real c2p_T;
  Real c2p_Ye;
  int c2p_valid;
};

//! Record one rejected projection trial, without printing.
//!
//! Codes: 100 equilibrium field not computable, 101 matter recovery rejected,
//! 103 exact complement of the converged field not recoverable, 104 its field
//! not computable, 105 the acceptance test of the converged pair refused at
//! least once and the step budget ran out, 106 iteration budget exhausted, 110
//! cell state not admissible at the start point, 111 Jacobian column not
//! formable, 112 Jacobian numerically singular, 113 line search found no
//! reducing step. detail carries the RecoverMatter code
//! at 101, 103, 104 and 110, 8 for a failed consistency check, the column index
//! for 111, and 0 otherwise - including at 110 when the start point is refused
//! for an invalid scale rather than by the chain. residual is the scaled residual
//! of the last point the chain evaluated successfully, the acceptance-test
//! difference for 105, and -1 when no residual has been evaluated yet.
//!
//! trial_cons is the complemented matter state handed to C2P and c2p_prim the
//! array C2P writes its state into; each is passed only where it is defined and
//! is null elsewhere, so a trial that C2P rejected with an error never reads the
//! output array C2P leaves undefined on that path. matter_rho, matter_temperature
//! and matter_electron_fraction describe the last point whose chain evaluation
//! completed, which at 100 and 104 is the previous point rather than the C2P
//! state the failing test saw. fail_out may be null.
KOKKOS_INLINE_FUNCTION
void RecordProjectionFailure(ProjectionFailureReport *fail_out, int code,
                             int detail, int iteration, Real matter_rho,
                             Real matter_temperature,
                             Real matter_electron_fraction, Real residual,
                             const Real trial_cons[NCONS] = nullptr,
                             const Real c2p_prim[NPRIM] = nullptr) {
  if (fail_out == nullptr) {
    return;
  }
  fail_out->code = code;
  fail_out->detail = detail;
  fail_out->iteration = iteration;
  fail_out->residual = residual;
  fail_out->matter_rho = matter_rho;
  fail_out->matter_T = matter_temperature;
  fail_out->matter_Ye = matter_electron_fraction;
  if (trial_cons != nullptr) {
    for (int n = 0; n < NCONS; ++n) {
      fail_out->trial[n] = trial_cons[n];
    }
  }
  fail_out->c2p_valid = 0;
  if (c2p_prim != nullptr && detail < 20) {
    // C2P returned a state rather than an error. Its rho, T and Ye identify
    // which floor or adjustment branch fired: a density at the atmosphere value
    // means W = D/rho collapsed, and a density near the input with T at the
    // temperature floor means the solve returned a sub-floor temperature. They
    // are also the state the closure test saw at 100 and the state the
    // consistency check rejected at detail 8.
    fail_out->c2p_valid = 1;
    fail_out->c2p_rho = c2p_prim[PRH];
    fail_out->c2p_T = c2p_prim[PTM];
    fail_out->c2p_Ye = c2p_prim[PYF];
  }
}

//! Print one failed projection as a single self-contained line, and nothing else.
//!
//! One line per failed cell, because device output from different threads and
//! ranks is ordered arbitrarily: a failure reported across several prints cannot
//! be matched up with its own state afterwards, and a parallel log has no other
//! way to associate the two. The line names the cell (MPI rank, global meshblock
//! id, meshblock-local (k,j,i)) and the failure site, then the complete state the
//! projection consumed and produced there:
//!
//!   site    the failing test, with detail its sub-code and iter the iteration it
//!           fired in
//!   diff    the measure recorded with the failure, evals the chain evaluations
//!           attempted to reach it, and worst_term the term that carried that
//!           measure (0 energy, 1 momentum, 4 electron lepton number, -1 when
//!           nothing has been measured)
//!   prim    the matter primitives the caller entered with: rho, the three
//!           velocity components W v^i, P, T, Ye
//!   cons    the undensitized matter conserved state it entered with
//!   met     the spatial metric g_ij, so g3u, detg and every contraction follow,
//!   B_u     and the contravariant field B^i, the C2P magnetization input
//!   m1      the undensitized four-species M1 state, (E,Fx,Fy,Fz,N) per species
//!   fail    rho, T and Ye of the last point whose chain evaluation completed:
//!           at 100 and 104 that is the previous point, the state the failing
//!           test saw being c2p instead
//!   trial   D, tau, the lowered momentum components S_i and D Ye of the state
//!           the chain refused, valid only when trial=1, that is at the sites
//!           100, 101, 103 and 104, where the refused point differs from the
//!           entering state printed as cons; at 110 the two are the same state,
//!           so cons already carries it
//!   c2p     rho, T and Ye C2P returned, valid only when c2p=1, that is when C2P
//!           returned a state rather than an error, which is any detail below 20
//!
//! Everything is in the projection's own variables and units, so the failing call
//! can be replayed offline without further conversion: rho is the baryon number
//! density in 1/code-length^3, T is in MeV, and the matter and radiation
//! conserved arrays are undensitized (divided by sqrt(detg)). The velocity
//! components are the ones the primitive solver stores, W v^i (the projection's
//! z^i in PVX..PVZ), and not v^i: the replay takes v^i = z^i / W with
//! W = sqrt(1 + g_ij z^i z^j) from the metric the line prints. The run-level
//! conversions and floor parameters shared by every cell are printed once by the
//! caller instead of on every line.
KOKKOS_INLINE_FUNCTION
void PrintProjectionFailureLine(int rank, int gid, int k, int j, int i,
                                const ProjectionFailureReport &fail,
                                const Real prim[NPRIM], const Real cons[NCONS],
                                const Real b[NMAG], const Real g3d[NSPMETRIC],
                                const Real rad[projection_nspecies]
                                            [projection_nvars]) {
  // The sites that relay a chain refusal carry the refused point itself, which
  // the line needs if the call is to be replayable offline. At 110 the chain
  // fills trial with the entering state, which the cons group above already
  // prints, so trial is left out there.
  const int has_trial = (fail.code == 100 || fail.code == 101 ||
                         fail.code == 103 || fail.code == 104)
                            ? 1
                            : 0;
  Kokkos::printf(
      "CCSN M1 projection FAIL rank=%d gid=%d cell=(%d,%d,%d) site=%d"
      " detail=%d iter=%d diff=% .9e evals=%d worst_term=%d"
      " | prim rho=% .17e zx=% .9e zy=% .9e zz=% .9e P=% .9e T=% .17e"
      " Ye=% .17e"
      " | cons D=% .17e tau=% .17e Sx=% .9e Sy=% .9e Sz=% .9e DYe=% .17e"
      " | met=% .9e % .9e % .9e % .9e % .9e % .9e"
      " | B_u=% .9e % .9e % .9e"
      " | m1 nue E=% .9e Fx=% .9e Fy=% .9e Fz=% .9e N=% .9e"
      " anue E=% .9e Fx=% .9e Fy=% .9e Fz=% .9e N=% .9e"
      " nux E=% .9e Fx=% .9e Fy=% .9e Fz=% .9e N=% .9e"
      " anux E=% .9e Fx=% .9e Fy=% .9e Fz=% .9e N=% .9e"
      " | fail rho=% .9e T=% .9e Ye=% .9e"
      " | trial=%d D=% .9e tau=% .9e Sx=% .9e Sy=% .9e Sz=% .9e DYe=% .9e"
      " | c2p=%d rho=% .9e T=% .9e Ye=% .9e\n",
      rank, gid, k, j, i, fail.code, fail.detail, fail.iteration,
      fail.residual, fail.evals, fail.worst_term, prim[PRH], prim[PVX],
      prim[PVY], prim[PVZ], prim[PPR],
      prim[PTM], prim[PYF], cons[CDN], cons[CTA], cons[CSX], cons[CSY],
      cons[CSZ], cons[CYD], g3d[S11], g3d[S12], g3d[S13], g3d[S22], g3d[S23],
      g3d[S33], b[IBX], b[IBY], b[IBZ], rad[0][M1_E_IDX], rad[0][M1_FX_IDX],
      rad[0][M1_FY_IDX], rad[0][M1_FZ_IDX], rad[0][M1_N_IDX], rad[1][M1_E_IDX],
      rad[1][M1_FX_IDX], rad[1][M1_FY_IDX], rad[1][M1_FZ_IDX],
      rad[1][M1_N_IDX], rad[2][M1_E_IDX], rad[2][M1_FX_IDX],
      rad[2][M1_FY_IDX], rad[2][M1_FZ_IDX], rad[2][M1_N_IDX],
      rad[3][M1_E_IDX], rad[3][M1_FX_IDX], rad[3][M1_FY_IDX],
      rad[3][M1_FZ_IDX], rad[3][M1_N_IDX], fail.matter_rho, fail.matter_T,
      fail.matter_Ye, has_trial, fail.trial[CDN], fail.trial[CTA],
      fail.trial[CSX], fail.trial[CSY], fail.trial[CSZ], fail.trial[CYD],
      fail.c2p_valid, fail.c2p_rho, fail.c2p_T, fail.c2p_Ye);
}

//! Controls of the conservative bounce projection.
//!
//! Every threshold, tolerance and iteration cap the solver applies is a member,
//! so the solver holds no numeric policy of its own: the caller fills this from
//! the deck and the failure context line prints the same values the solve used.
//! The members are the solver's contract with the caller, not tuning constants
//! hidden in the header.
//!
//! chain_tolerance is the relative tolerance of the consistency check each chain
//! evaluation performs against PrimToCon. It grades a state the code
//! reconstructed itself, so it can be no tighter than the primitive solver that
//! produced it: tightening it below that solver's own accuracy rejects every
//! point. The energy term of that check is anchored to the larger of the cell's
//! total energy density and its rest-mass energy density, because the conserved
//! energy is rebuilt by subtracting the rest-mass term and is bounded by it, not
//! by the leftover itself. momentum_floor is the smallest physical speed, in units
//! of D, accepted as a momentum scale, which gives the residual a nonzero scale in
//! a cell at rest. jacobian_step is the relative finite-difference step of the
//! scaled unknowns. armijo and backtrack_factor define the line search, with
//! alpha = backtrack_factor^k for k = 0 to line_search_maxiter - 1. armijo,
//! backtrack_factor, jacobian_step and pivot_threshold change the path the solve
//! takes, not the criterion it is judged by: pivot_threshold is the pivot floor
//! of Solve5x5 relative to the largest entry of the Jacobian.
//!
//! acceptance_slack is the factor between the convergence threshold of the
//! residual and the acceptance threshold of the written pair, and it exists
//! because the two measure different quantities. The residual is scaled by the
//! cell totals, while the acceptance difference is scaled by the field alone, so
//! it exceeds the residual by the dimensionless response of the equilibrium field
//! to the matter state (about four for the energy channel, of order the
//! logarithmic Ye response of the trapped number density for the lepton channel).
//! Setting acceptance_slack to one makes the acceptance test the stricter of the
//! two by that factor.
struct ProjectionControls {
  Real tolerance;            //! scaled residual that ends the Newton solve
  int max_iterations;        //! Newton steps the solve may take
  Real chain_tolerance;      //! PrimToCon round trip every chain point must pass
  Real acceptance_slack;     //! acceptance bound, as a factor on tolerance
  Real momentum_floor;       //! smallest momentum scale, in units of D
  int line_search_maxiter;   //! backtracking trials per Newton step
  Real jacobian_step;        //! relative FD step of the scaled unknowns
  Real armijo;               //! sufficient-decrease constant of the line search
  Real backtrack_factor;     //! step-shortening factor per backtracking trial
  Real pivot_threshold;      //! relative pivot floor of the 5x5 solve
};

//! Outcome of one equilibrium-chain evaluation.
//!
//! ok is false when the point is not admissible, and site and detail then name
//! the rejection in the vocabulary of ProjectionFailureReport: site 100 the
//! field is not computable, site 101 the matter recovery was rejected or the
//! consistency check failed, with detail the RecoverMatter code (2-7, or 20 plus
//! the Primitive::Error enumerator) or 8 for a failed consistency check.
//!
//! The chain reports 100 when its third step refused and 101 otherwise, so the
//! driver relabels the 101 case by which of its own calls refused: 110 for the
//! entering state at iteration 0, and 103 or 104 for the acceptance point. The
//! relabelled code is what reaches the report, so a failure line is read against
//! the driver's table, not against this one.
//!
//! cons is the matter conserved array the chain built and is defined on every
//! path, including a rejected one. prim holds the last state C2P computed and is
//! meaningful only where C2P wrote it; rad holds the last field computed and is
//! meaningful only on success.
struct EquilibriumChainResult {
  bool ok;
  int site;
  int detail;
  Real cons[NCONS];
  Real prim[NPRIM];
  Real rad[projection_nspecies][projection_nvars];
};

//! One evaluation of the equilibrium chain: matter state -> primitives -> field.
//!
//! x holds the five matter unknowns in the order of projection_nvars, that is
//! (tau_m, S_m^x, S_m^y, S_m^z, D Ye_m), and is placed into the conserved array
//! in the slots those quantities occupy. The density and every unrelated
//! conserved component are copied from old_cons and are never solved for, so the
//! electron fraction and the baryon number of the cell are structural rather
//! than outcomes of the solve.
//!
//! The recovered primitives are then checked against x with the code's own
//! algebraic inverse, ps.PrimToCon, because a C2P can return a state pinned at an
//! EOS or C2P floor while reporting success: such a state is not a solution of
//! the projection however well the residual algebra closes. The check is
//! relative, in the same scales and with the same invariant measure the residual
//! uses, and its tolerance is the caller's chain_tolerance. Every evaluation is a
//! point on the equilibrium manifold: the matter comes from the strict C2P and the
//! field is the exact equilibrium field of that same matter, so nothing here
//! blends two matter states. All work is stack-local and nothing is carried
//! between calls.
template <class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION bool M1EquilibriumChain(
    const Primitive::PrimitiveSolver<EOSPolicy, ErrorPolicy> &ps,
    const radiationm1::RadiationM1OpacityType opacity_type,
    const Primitive::UnitSystem &code_units,
    const Primitive::UnitSystem &eos_units,
    const Primitive::UnitSystem &ngs_units,
    const Real old_cons[NCONS], const Real x[projection_nvars],
    const Real b[NMAG], const Real g3d[NSPMETRIC], const Real g3u[NSPMETRIC],
    const Real scale[projection_nvars], Real chain_tolerance,
    EquilibriumChainResult *out) {
  for (int n = 0; n < NCONS; ++n) {
    out->cons[n] = old_cons[n];
  }
  out->cons[CTA] = x[0];
  out->cons[CSX] = x[1];
  out->cons[CSY] = x[2];
  out->cons[CSZ] = x[3];
  out->cons[CYD] = x[4];

  int recover_code = 0;
  Real recovered_cons[NCONS];
  if (!RecoverMatter(ps, out->cons, b, g3d, g3u, out->prim, recovered_cons,
                     &recover_code)) {
    out->ok = false;
    out->site = 101;
    out->detail = recover_code;
    return false;
  }

  // Consistency check. PrimToCon is algebraic, so a state that C2P returned
  // unchanged but pinned at the temperature floor fails here even though no floor
  // or adjustment flag was set and every conserved value round-tripped exactly.
  Real prim_copy[NPRIM];
  const int nprim_active = PYF + ps.GetEOS().GetNSpecies();
  // Only the active primitives are read: C2P writes the first nprim_active of
  // them and leaves the rest of out->prim holding whatever the stack had, and
  // PrimToCon reads no further, so copying the whole array would read
  // indeterminate values that nothing consumes.
  for (int n = 0; n < nprim_active; ++n) {
    prim_copy[n] = out->prim[n];
  }
  Real b_copy[NMAG] = {b[IBX], b[IBY], b[IBZ]};
  Real metric[NSPMETRIC];
  for (int n = 0; n < NSPMETRIC; ++n) {
    metric[n] = g3d[n];
  }
  Real cons_from_prim[NCONS];
  const Primitive::Error inverse_error =
      ps.PrimToCon(prim_copy, cons_from_prim, b_copy, metric);
  if (inverse_error != Primitive::Error::SUCCESS ||
      !Kokkos::isfinite(cons_from_prim[CDN]) ||
      Kokkos::fabs(cons_from_prim[CDN] - old_cons[CDN]) >
          chain_tolerance * Kokkos::fabs(old_cons[CDN])) {
    out->ok = false;
    out->site = 101;
    out->detail = 8;
    return false;
  }
  const Real check[projection_nvars] = {
      cons_from_prim[CTA], cons_from_prim[CSX], cons_from_prim[CSY],
      cons_from_prim[CSZ], cons_from_prim[CYD]};
  for (int j = 0; j < projection_nvars; ++j) {
    if (!Kokkos::isfinite(check[j])) {
      out->ok = false;
      out->site = 101;
      out->detail = 8;
      return false;
    }
  }
  // The difference is measured exactly as ResidualNorm measures the residual:
  // energy and electron lepton number against their own scales, and the three
  // momentum components as one metric norm against the shared momentum scale,
  // so the criterion is the same in every spatial coordinate basis. A
  // componentwise maximum is not invariant and would reject here what it accepts
  // in another basis, and this check decides whether the run continues.
  //
  // The energy term is the one exception, and it is anchored rather than scaled
  // by its own total: PrimToCon rebuilds tau = (H W^2 + B^2 - p - ...) - D, so the
  // rest-mass term D is what that subtraction cancels, and D bounds the attainable
  // absolute accuracy of the energy term. Anchored to the leftover tau alone, the
  // check asks a relative tolerance of a cancelling difference: at tau/D = 0.01 it
  // needs the enthalpy to about 1e-12, which refuses ordinary cells at their first
  // evaluation while the density, momentum and lepton terms keep three to four
  // decades of margin. Anchoring to the larger of the two grades the state the
  // solver actually reconstructed, and still refuses a state pinned at a clamp:
  // such a state violates the energy term by O(tau / D) against the new anchor.
  const Real energy_check_scale =
      Kokkos::fmax(scale[0], Kokkos::fabs(old_cons[CDN]));
  const Real check_momentum[3] = {check[1] - x[1], check[2] - x[2],
                                  check[3] - x[3]};
  if (Kokkos::fabs(check[0] - x[0]) > chain_tolerance * energy_check_scale ||
      MetricNormU(g3u, check_momentum) > chain_tolerance * scale[1] ||
      Kokkos::fabs(check[4] - x[4]) > chain_tolerance * scale[4]) {
    out->ok = false;
    out->site = 101;
    out->detail = 8;
    return false;
  }

  if (!EquilibriumM1Target(ps, opacity_type, code_units, eos_units, ngs_units,
                           out->prim, g3d, out->rad)) {
    out->ok = false;
    out->site = 100;
    out->detail = 0;
    return false;
  }
  out->ok = true;
  out->site = 0;
  out->detail = 0;
  return true;
}

//! Conservative field contribution and residual at one chain point.
//!
//! R_j = x_j + field_j - tot_j, so R = 0 states that the matter state x is the
//! exact complement of its own equilibrium field, at unchanged cell totals. The
//! field is returned as well because the acceptance test needs it to rebuild the
//! complement it will write.
KOKKOS_INLINE_FUNCTION void ChainResidual(
    const Real x[projection_nvars],
    const Real rad[projection_nspecies][projection_nvars], Real mb,
    const Real tot[projection_nvars], Real field[projection_nvars],
    Real residual[projection_nvars]) {
  FieldConserved(rad, mb, field);
  for (int j = 0; j < projection_nvars; ++j) {
    residual[j] = x[j] + field[j] - tot[j];
  }
}

//! The invariant measure of a residual at one point.
//!
//! Energy and electron lepton number are frame-dependent scalars, not
//! coordinate-dependent, so each is measured against its own scale. The three
//! momentum components are the components of one covector and are measured as a
//! single metric norm against the shared momentum scale, so the measure is
//! unchanged by any change of spatial coordinates: a componentwise maximum is
//! not, and would accept in one basis what it rejects in another. worst_term,
//! when not null, receives the term that carried the maximum: 0 energy,
//! 1 momentum, 4 electron lepton number.
KOKKOS_INLINE_FUNCTION Real ResidualNorm(const Real residual[projection_nvars],
                                         const Real scale[projection_nvars],
                                         const Real g3u[NSPMETRIC],
                                         int *worst_term = nullptr) {
  const Real momentum[3] = {residual[1], residual[2], residual[3]};
  Real value[3];
  value[0] = Kokkos::fabs(residual[0]) / scale[0];
  value[1] = MetricNormU(g3u, momentum) / scale[1];
  value[2] = Kokkos::fabs(residual[4]) / scale[4];
  const int term[3] = {0, 1, 4};
  Real norm = value[0];
  int worst = 0;
  for (int m = 1; m < 3; ++m) {
    if (value[m] > norm) {
      norm = value[m];
      worst = term[m];
    }
  }
  if (worst_term != nullptr) {
    *worst_term = worst;
  }
  return norm;
}

//! Solve a 5x5 linear system by Gaussian elimination with partial pivoting.
//!
//! Stack-local and without branching on the values: the caller needs a
//! deterministic answer, and a pivot below pivot_threshold relative to the
//! largest entry of the matrix is reported as a singular system rather than
//! turned into a division by a near-zero number. du is undefined on failure.
KOKKOS_INLINE_FUNCTION bool Solve5x5(
    const Real J[projection_nvars][projection_nvars],
    const Real rhs[projection_nvars], Real pivot_threshold,
    Real du[projection_nvars]) {
  Real a[projection_nvars][projection_nvars];
  Real b[projection_nvars];
  Real largest = 0.0;
  for (int i = 0; i < projection_nvars; ++i) {
    b[i] = rhs[i];
    for (int j = 0; j < projection_nvars; ++j) {
      a[i][j] = J[i][j];
      largest = Kokkos::fmax(largest, Kokkos::fabs(J[i][j]));
    }
  }
  const Real threshold = pivot_threshold * largest;
  for (int col = 0; col < projection_nvars; ++col) {
    int pivot = col;
    Real best = Kokkos::fabs(a[col][col]);
    for (int row = col + 1; row < projection_nvars; ++row) {
      if (Kokkos::fabs(a[row][col]) > best) {
        best = Kokkos::fabs(a[row][col]);
        pivot = row;
      }
    }
    if (!(best > threshold)) {
      return false;
    }
    if (pivot != col) {
      for (int j = col; j < projection_nvars; ++j) {
        const Real swap = a[col][j];
        a[col][j] = a[pivot][j];
        a[pivot][j] = swap;
      }
      const Real swap = b[col];
      b[col] = b[pivot];
      b[pivot] = swap;
    }
    const Real inverse = 1.0 / a[col][col];
    for (int row = col + 1; row < projection_nvars; ++row) {
      const Real factor = a[row][col] * inverse;
      if (factor == 0.0) {
        continue;
      }
      for (int j = col; j < projection_nvars; ++j) {
        a[row][j] -= factor * a[col][j];
      }
      b[row] -= factor * b[col];
    }
  }
  for (int row = projection_nvars - 1; row >= 0; --row) {
    Real sum = b[row];
    for (int j = row + 1; j < projection_nvars; ++j) {
      sum -= a[row][j] * du[j];
    }
    du[row] = sum / a[row][row];
  }
  return true;
}

//! Solve one cell's conservative matter-radiation equilibrium projection.
//!
//! The unknowns are the matter's own undensitized conserved variables,
//! (tau_m, S_m^x, S_m^y, S_m^z, D Ye_m). The radiation is eliminated by its
//! closure and enters only through the equilibrium field of the current matter,
//! so a damped Newton iteration with a backtracking line search drives the five
//! conservation statements to zero along the equilibrium manifold: every iterate
//! is a state where matter and field are in equilibrium, and the residual says
//! only whether their sum still equals the totals the cell entered with. The
//! density is not an unknown and is never written back.
//!
//! The finite-difference Jacobian is rebuilt every iteration from fresh chain
//! evaluations, and nothing metric-dependent or state-dependent is carried in or
//! out of a call: a cell whose metric or state changed since the last step is
//! treated exactly like any other.
//!
//! On convergence the state about to be written is re-derived through its own
//! exact complement, and the pair is written only if the field of that
//! complement reproduces the field written beside it within the caller's
//! acceptance bound, ctl.acceptance_slack * ctl.tolerance. That test bounds the
//! mutual consistency of the pair, not its conservation: the written matter is
//! the exact complement of the written field, so the entering totals survive by
//! construction whatever the test says. A pair that fails the test is not
//! written: the solve continues from the acceptance point instead, so a refusal
//! costs iterations rather than the cell, and the step budget bounds the retry.
//! All work is stack-local; output arrays are populated only on success and the
//! caller commits them.
//!
//! Every threshold the solve applies is a member of ctl, so a failure line can be
//! read against exactly the values the solve used; the header holds no numeric
//! policy of its own.
//!
//! fail_out receives code 0 on success and otherwise names the failing site as
//! recorded by RecordProjectionFailure, together with the solver-level
//! diagnostics of that failure; it may be null.
template <class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION bool ProjectM1Cell(
    const Primitive::PrimitiveSolver<EOSPolicy, ErrorPolicy> &ps,
    const radiationm1::RadiationM1OpacityType opacity_type,
    const Primitive::UnitSystem &code_units,
    const Primitive::UnitSystem &eos_units,
    const Primitive::UnitSystem &ngs_units,
    const Real old_cons[NCONS], const Real b[NMAG], const Real g3d[NSPMETRIC],
    const Real g3u[NSPMETRIC],
    const Real old_rad[projection_nspecies][projection_nvars],
    const ProjectionControls &ctl, Real energy_floor, Real number_floor,
    Real final_prim[NPRIM], Real final_cons[NCONS],
    Real final_rad[projection_nspecies][projection_nvars],
    ProjectionFailureReport *fail_out) {
  const int nprim_active = PYF + ps.GetEOS().GetNSpecies();
  const Real mb = ps.GetEOS().GetBaryonMass();

  // Totals held fixed by the solve, and the scales of the residual. The three
  // momentum entries share one scale so that the scaled unknowns transform
  // covariantly: both the metric norm of the total momentum and the momentum
  // floor are invariant under a change of spatial coordinates, so the scaled
  // point rotates with them.
  Real tot[projection_nvars];
  ConservedTotals(old_cons, old_rad, mb, tot);
  const Real total_momentum[3] = {tot[1], tot[2], tot[3]};
  Real scale[projection_nvars];
  scale[0] = Kokkos::fmax(Kokkos::fabs(tot[0]), energy_floor);
  const Real momentum_scale =
      Kokkos::fmax(MetricNormU(g3u, total_momentum),
                   ctl.momentum_floor * Kokkos::fabs(old_cons[CDN]));
  scale[1] = momentum_scale;
  scale[2] = momentum_scale;
  scale[3] = momentum_scale;
  scale[4] = Kokkos::fmax(Kokkos::fabs(tot[4]), mb * number_floor);

  int evals = 0;
  // -1 until something has been evaluated, so a failure line can tell "the energy
  // term carried the measure" from "nothing has been measured yet".
  int worst_term = -1;
  Real norm = -1.0;
  // Set when the acceptance test refuses a converged pair, so a budget failure
  // can say that the acceptance test was the binding constraint rather than a
  // residual that never converged, and hold the difference it refused.
  bool acceptance_refused = false;
  Real acceptance_difference = -1.0;
  Real matter_rho = 0.0;
  Real matter_T = 0.0;
  Real matter_Ye = 0.0;
  if (fail_out != nullptr) {
    fail_out->code = 0;
    fail_out->detail = 0;
    fail_out->iteration = -1;
    fail_out->residual = -1.0;
    fail_out->evals = 0;
    fail_out->worst_term = -1;
  }

  // Record a rejection together with the solver-level diagnostics of the point
  // it fired at, so one failure line carries both the site and the measure.
  auto Record = [&](int code, int detail, int iteration, Real residual,
                    const Real *trial_cons, const Real *c2p_prim) {
    if (fail_out != nullptr) {
      fail_out->evals = evals;
      fail_out->worst_term = worst_term;
      RecordProjectionFailure(fail_out, code, detail, iteration, matter_rho,
                              matter_T, matter_Ye, residual, trial_cons,
                              c2p_prim);
    }
  };

  for (int j = 0; j < projection_nvars; ++j) {
    if (!(scale[j] > 0.0) || !Kokkos::isfinite(scale[j])) {
      // Without a positive scale the cell's own totals define no solve at all,
      // which is the statement of site 110 one step earlier: nothing has been
      // evaluated yet, so no residual is reported either.
      Record(110, 0, 0, -1.0, nullptr, nullptr);
      return false;
    }
  }

  Real x[projection_nvars] = {old_cons[CTA], old_cons[CSX], old_cons[CSY],
                              old_cons[CSZ], old_cons[CYD]};
  // Scaled unknowns and scaled residual of the linearization about x: scaled_x is
  // x_j / scale_j and scaled_residual is R_j / scale_j, the variables the
  // finite-difference step and the Jacobian are expressed in. rad holds the field
  // of the last point the chain accepted, which the acceptance test compares
  // against.
  Real scaled_x[projection_nvars];
  Real scaled_residual[projection_nvars];
  Real rad[projection_nspecies][projection_nvars];

  // Newton iteration. ctl.max_iterations is a budget of steps, not of tested
  // points: every point a permitted step produces is measured and tested before
  // the budget is declared exhausted, so a step whose result was never evaluated
  // cannot hide a root the budget allows.
  for (int iteration = 0;; ++iteration) {
    // 1. The chain and the measure at the current point. The first iteration's
    //    point is the cell's own matter state, so a refusal there is that state
    //    being inadmissible rather than an iteration going wrong; every later
    //    point comes from a step the line search already accepted.
    EquilibriumChainResult chain;
    ++evals;
    if (!M1EquilibriumChain(ps, opacity_type, code_units, eos_units, ngs_units,
                            old_cons, x, b, g3d, g3u, scale,
                            ctl.chain_tolerance, &chain)) {
      Record(chain.site == 100 ? 100 : (iteration == 0 ? 110 : 101),
             chain.detail, iteration, norm, chain.cons, chain.prim);
      return false;
    }
    Real field[projection_nvars];
    Real residual[projection_nvars];
    ChainResidual(x, chain.rad, mb, tot, field, residual);
    norm = ResidualNorm(residual, scale, g3u, &worst_term);
    for (int s = 0; s < projection_nspecies; ++s) {
      for (int v = 0; v < projection_nvars; ++v) {
        rad[s][v] = chain.rad[s][v];
      }
    }
    matter_rho = chain.prim[PRH];
    matter_T = chain.prim[PTM];
    matter_Ye = chain.prim[PYF];

    if (norm <= ctl.tolerance) {
      // Acceptance. The state to be written is the exact complement of the
      // field just computed, so it leaves the entering totals behind to
      // roundoff. Re-derive it independently and require the field of its own
      // complement to reproduce the field it came from within the acceptance
      // slack: only a pair that passes this is written.
      Real x_exact[projection_nvars];
      for (int j = 0; j < projection_nvars; ++j) {
        x_exact[j] = tot[j] - field[j];
      }
      EquilibriumChainResult acceptance;
      ++evals;
      if (!M1EquilibriumChain(ps, opacity_type, code_units, eos_units, ngs_units,
                              old_cons, x_exact, b, g3d, g3u,
                              scale, ctl.chain_tolerance, &acceptance)) {
        Record(acceptance.site == 100 ? 104 : 103, acceptance.detail,
               iteration, norm, acceptance.cons, acceptance.prim);
        return false;
      }
      const Real difference = ProjectionDifference(rad, acceptance.rad, g3u,
                                                  energy_floor, number_floor);
      if (difference > ctl.acceptance_slack * ctl.tolerance) {
        // The written field and the field of its own complement disagree beyond
        // the acceptance slack. Do not write the pair and do not abandon the
        // cell: the acceptance point is a point of the same solve, so take it
        // as the new iterate and keep solving. The step budget bounds the retry,
        // and a cell that runs out of budget is reported as 105, which says the
        // acceptance test was binding rather than that the residual stalled.
        acceptance_refused = true;
        acceptance_difference = difference;
        // The retry re-enters the convergence branch above, and the budget test
        // below this block sits behind the continue, so it would never see a
        // pair that keeps being refused: consult the budget here, on the path
        // that spends the steps, so a converged pair that never moves inside the
        // slack is bounded like any other. Site 105 says that bound was the
        // binding one rather than a residual that stalled, and the reported
        // iteration is the budget itself, as at the check below.
        if (iteration >= ctl.max_iterations) {
          Record(105, 0, ctl.max_iterations, acceptance_difference, nullptr,
                 nullptr);
          return false;
        }
        for (int j = 0; j < projection_nvars; ++j) {
          x[j] = x_exact[j];
        }
        continue;
      }
      for (int n = 0; n < NCONS; ++n) {
        final_cons[n] = acceptance.cons[n];
      }
      for (int n = 0; n < nprim_active; ++n) {
        final_prim[n] = acceptance.prim[n];
      }
      for (int s = 0; s < projection_nspecies; ++s) {
        for (int v = 0; v < projection_nvars; ++v) {
          final_rad[s][v] = rad[s][v];
        }
      }
      return true;
    }
    // The budget is a step budget, and the point the last permitted step
    // produced has just been measured and tested above, so the budget is
    // declared exhausted only after every point it allows has had its chance.
    // The reported iteration is the budget itself, as it was when the check sat
    // in the loop header. The site separates a residual that never converged
    // (106, reporting the residual) from a residual that converged and whose
    // pair the acceptance test refused at least once (105, reporting that
    // difference, so the line says how far outside the slack the pair stayed).
    if (iteration >= ctl.max_iterations) {
      Record(acceptance_refused ? 105 : 106, 0, ctl.max_iterations,
             acceptance_refused ? acceptance_difference : norm, nullptr,
             nullptr);
      return false;
    }
    // 2. Finite-difference Jacobian, rebuilt from fresh evaluations on every
    //    iteration. The step is a fixed fraction of the scaled unknown it probes.
    //    The measure and the scaled system are what is invariant, not the step:
    //    it is per component and is absolute rather than relative where a scaled
    //    unknown sits below one, so it fixes the path the iteration takes, not
    //    the criterion it is judged by. Each column probes forward first and
    //    backward second; a column
    //    that neither probe can form is left zero and recorded, because no step
    //    in that unknown can be taken from this point.
    for (int j = 0; j < projection_nvars; ++j) {
      scaled_x[j] = x[j] / scale[j];
      scaled_residual[j] = residual[j] / scale[j];
    }
    Real jacobian[projection_nvars][projection_nvars];
    int failed_column = -1;
    for (int j = 0; j < projection_nvars; ++j) {
      const Real step =
          ctl.jacobian_step * Kokkos::fmax(Kokkos::fabs(scaled_x[j]), 1.0);
      bool formed = false;
      for (int attempt = 0; attempt < 2 && !formed; ++attempt) {
        const Real signed_step = (attempt == 0) ? step : -step;
        Real probe[projection_nvars];
        Real probe_field[projection_nvars];
        Real probe_residual[projection_nvars];
        EquilibriumChainResult probe_chain;
        for (int m = 0; m < projection_nvars; ++m) {
          probe[m] = x[m];
        }
        probe[j] += signed_step * scale[j];
        ++evals;
        if (!M1EquilibriumChain(ps, opacity_type, code_units, eos_units,
                                ngs_units, old_cons, probe, b,
                                g3d, g3u, scale, ctl.chain_tolerance,
                                &probe_chain)) {
          continue;
        }
        ChainResidual(probe, probe_chain.rad, mb, tot, probe_field,
                      probe_residual);
        for (int m = 0; m < projection_nvars; ++m) {
          const Real probe_scaled_residual = probe_residual[m] / scale[m];
          const Real delta = probe_scaled_residual - scaled_residual[m];
          jacobian[m][j] = (attempt == 0) ? delta / step : -delta / step;
        }
        formed = true;
      }
      if (!formed) {
        for (int m = 0; m < projection_nvars; ++m) {
          jacobian[m][j] = 0.0;
        }
        if (failed_column < 0) {
          failed_column = j;
        }
      }
    }

    // 3. Newton step, from the linearization about the current point.
    Real rhs[projection_nvars];
    Real du[projection_nvars];
    for (int m = 0; m < projection_nvars; ++m) {
      rhs[m] = -scaled_residual[m];
    }
    if (!Solve5x5(jacobian, rhs, ctl.pivot_threshold, du)) {
      Record((failed_column >= 0) ? 111 : 112,
             (failed_column >= 0) ? failed_column : 0, iteration, norm, nullptr,
             nullptr);
      return false;
    }

    // 4. Backtracking line search along the Newton direction: alpha is the
    //    backtracking factor to the k, and the step taken is the first one that
    //    lowers the measure by the Armijo fraction of it. A trial the chain
    //    refuses is skipped rather than shortened into, so every point the search
    //    evaluates is on the equilibrium manifold and no factor here relaxes the
    //    field toward anything.
    bool stepped = false;
    Real alpha = 1.0;
    for (int k = 0; k < ctl.line_search_maxiter && !stepped; ++k) {
      Real trial[projection_nvars];
      Real trial_field[projection_nvars];
      Real trial_residual[projection_nvars];
      EquilibriumChainResult trial_chain;
      for (int m = 0; m < projection_nvars; ++m) {
        trial[m] = x[m] + alpha * du[m] * scale[m];
      }
      ++evals;
      if (M1EquilibriumChain(ps, opacity_type, code_units, eos_units, ngs_units,
                             old_cons, trial, b, g3d, g3u,
                             scale, ctl.chain_tolerance, &trial_chain)) {
        ChainResidual(trial, trial_chain.rad, mb, tot, trial_field,
                      trial_residual);
        int trial_worst_term = -1;
        const Real trial_norm =
            ResidualNorm(trial_residual, scale, g3u, &trial_worst_term);
        if (trial_norm < (1.0 - ctl.armijo * alpha) * norm) {
          for (int m = 0; m < projection_nvars; ++m) {
            x[m] = trial[m];
          }
          norm = trial_norm;
          worst_term = trial_worst_term;
          stepped = true;
        }
      }
      alpha *= ctl.backtrack_factor;
    }
    if (!stepped) {
      Record(113, 0, iteration, norm, nullptr, nullptr);
      return false;
    }
  }
  // Unreachable: the loop returns through its convergence test, its budget check
  // or one of the failure records above. Kept so the function still ends in a
  // return statement.
  return false;
}

} // namespace ccsn

#endif // PGEN_CCSN_M1_PROJECTION_HPP_
