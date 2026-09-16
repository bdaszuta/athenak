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

//! Return the largest normalized difference between two four-species M1 states.
//!
//! Energy and each flux component use the target energy as their scale, while
//! number density uses the target number density. The corresponding M1 floors
//! provide nonzero scales when a target is small.
KOKKOS_INLINE_FUNCTION Real ProjectionDifference(
    const Real state[projection_nspecies][projection_nvars],
    const Real target[projection_nspecies][projection_nvars], Real energy_floor,
    Real number_floor) {
  Real error = 0.0;
  for (int s = 0; s < projection_nspecies; ++s) {
    const Real energy_scale =
        Kokkos::fmax(Kokkos::fabs(target[s][M1_E_IDX]), energy_floor);
    for (int v = M1_E_IDX; v <= M1_FZ_IDX; ++v) {
      error = Kokkos::fmax(
          error, Kokkos::fabs(state[s][v] - target[s][v]) / energy_scale);
    }
    const Real number_scale =
        Kokkos::fmax(Kokkos::fabs(target[s][M1_N_IDX]), number_floor);
    error = Kokkos::fmax(
        error,
        Kokkos::fabs(state[s][M1_N_IDX] - target[s][M1_N_IDX]) / number_scale);
  }
  return error;
}

//! Construct the matter state complementary to a proposed radiation state.
//!
//! The old matter and radiation states define fixed total energy, momentum,
//! and electron lepton number. This routine subtracts the proposed radiation
//! contribution from those totals, preserving them exactly while leaving the
//! matter density and any unrelated conserved components unchanged. Inputs and
//! output are undensitized.
KOKKOS_INLINE_FUNCTION void MatterComplement(
    const Real old_cons[NCONS],
    const Real old_rad[projection_nspecies][projection_nvars],
    const Real rad[projection_nspecies][projection_nvars], Real mb,
    Real cons[NCONS]) {
  for (int n = 0; n < NCONS; ++n) {
    cons[n] = old_cons[n];
  }

  Real total_energy = old_cons[CTA];
  Real total_momentum[3] = {old_cons[CSX], old_cons[CSY], old_cons[CSZ]};
  for (int s = 0; s < projection_nspecies; ++s) {
    total_energy += old_rad[s][M1_E_IDX];
    total_momentum[0] += old_rad[s][M1_FX_IDX];
    total_momentum[1] += old_rad[s][M1_FY_IDX];
    total_momentum[2] += old_rad[s][M1_FZ_IDX];
  }
  const Real total_lepton =
      old_cons[CYD] + mb * (old_rad[projection_nue][M1_N_IDX] -
                            old_rad[projection_anue][M1_N_IDX]);

  cons[CTA] = total_energy;
  cons[CSX] = total_momentum[0];
  cons[CSY] = total_momentum[1];
  cons[CSZ] = total_momentum[2];
  for (int s = 0; s < projection_nspecies; ++s) {
    cons[CTA] -= rad[s][M1_E_IDX];
    cons[CSX] -= rad[s][M1_FX_IDX];
    cons[CSY] -= rad[s][M1_FY_IDX];
    cons[CSZ] -= rad[s][M1_FZ_IDX];
  }
  cons[CYD] = total_lepton -
              mb * (rad[projection_nue][M1_N_IDX] -
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

//! Outcome of a failed projection trial, handed back to the caller.
//!
//! code is the rejection site (100-106); detail is the RecoverMatter code at the
//! recovery sites (2-7, or 20 + the Primitive::Error enumerator when C2P
//! returned an error) and 0 elsewhere; iteration is the index of the failing
//! trial; residual is the most recent relaxed residual, the exact-pass residual
//! at 105, and -1 when no residual has been evaluated yet.
//!
//! matter_rho, matter_T and matter_Ye describe the matter state the failing test
//! was applied to. That is not always the state the caller entered with: every
//! accepted recovery replaces the iterate. trial is the undensitized complemented
//! matter state handed to C2P and is defined only at sites 101 and 103. c2p_rho,
//! c2p_T and c2p_Ye are what C2P left behind when it returned a floored or
//! adjusted state rather than an error and are defined only for detail 2-5.
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
  Real matter_rho;
  Real matter_T;
  Real matter_Ye;
  Real trial[NCONS];
  Real c2p_rho;
  Real c2p_T;
  Real c2p_Ye;
};

//! Record one rejected projection trial, without printing.
//!
//! Codes: 100 iteration target not computable, 101 relaxed-trial recovery
//! rejected, 102 recovered-state target not computable, 103 exact-trial
//! recovery rejected, 104 exact-pass target not computable, 105 exact-pass
//! residual above tolerance, 106 iteration budget exhausted. detail carries the
//! RecoverMatter code for 101 and 103 and is 0 otherwise. residual is the most
//! recent relaxed residual, the exact-pass residual for 105, and -1 when no
//! residual has been evaluated yet.
//!
//! trial_cons is the complemented matter state handed to C2P and c2p_prim the
//! array C2P writes its state into; each is passed only where it is defined and
//! is null elsewhere, so a trial that C2P rejected with an error never reads the
//! output array C2P leaves undefined on that path. matter_rho, matter_temperature
//! and matter_electron_fraction describe the matter state the failing test saw.
//! fail_out may be null.
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
  if (c2p_prim != nullptr && detail >= 2 && detail <= 5) {
    // C2P returned a state rather than an error. Its rho, T and Ye identify
    // which floor or adjustment branch fired: a density at the atmosphere value
    // means W = D/rho collapsed, and a density near the input with T at the
    // temperature floor means the solve returned a sub-floor temperature.
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
//!   prim    the matter primitives the caller entered with: rho, v^i, P, T, Ye
//!   cons    the undensitized matter conserved state it entered with
//!   met     the spatial metric g_ij, so g3u, detg and every contraction follow,
//!   B_u     and the contravariant field B^i, the C2P magnetization input
//!   m1      the undensitized four-species M1 state, (E,Fx,Fy,Fz,N) per species
//!   fail    rho, T and Ye of the matter state the failing test was applied to
//!   trial   D, tau, S^i and D Ye of the complemented state handed to C2P, valid
//!           only when trial=1, i.e. at sites 101 and 103
//!   c2p     rho, T and Ye C2P returned, valid only when c2p=1, that is when C2P
//!           returned a floored or adjusted state rather than an error
//!
//! Everything is in the projection's own variables and units, so the failing call
//! can be replayed offline without further conversion: rho is the baryon number
//! density in 1/code-length^3, T is in MeV, and the matter and radiation
//! conserved arrays are undensitized (divided by sqrt(detg)). The run-level
//! conversions and floor parameters shared by every cell are printed once by the
//! caller instead of on every line.
KOKKOS_INLINE_FUNCTION
void PrintProjectionFailureLine(int rank, int gid, int k, int j, int i,
                                const ProjectionFailureReport &fail,
                                const Real prim[NPRIM], const Real cons[NCONS],
                                const Real b[NMAG], const Real g3d[NSPMETRIC],
                                const Real rad[projection_nspecies]
                                            [projection_nvars]) {
  const int has_trial = (fail.code == 101 || fail.code == 103) ? 1 : 0;
  const int has_c2p = (fail.detail >= 2 && fail.detail <= 5) ? 1 : 0;
  Kokkos::printf(
      "CCSN M1 projection FAIL rank=%d gid=%d cell=(%d,%d,%d) site=%d"
      " detail=%d iter=%d diff=% .9e"
      " | prim rho=% .17e vx=% .9e vy=% .9e vz=% .9e P=% .9e T=% .17e"
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
      fail.residual, prim[PRH], prim[PVX], prim[PVY], prim[PVZ], prim[PPR],
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
      has_c2p, fail.c2p_rho, fail.c2p_T, fail.c2p_Ye);
}

//! Solve one cell's conservative matter-radiation equilibrium projection.
//!
//! Each iteration computes the equilibrium M1 target from the current matter,
//! under-relaxes the radiation state toward it, constructs the exact matter
//! complement, and recovers new matter primitives. Once the relaxed state is
//! within tolerance, an unrelaxed target/complement/recovery pass verifies that
//! the final radiation still matches equilibrium for the final matter. All
//! work is stack-local; output arrays are populated only on success. The caller
//! is responsible for committing them to the live MHD and M1 arrays.
//!
//! fail_out receives code 0 on success and otherwise names the last rejected
//! trial as recorded by RecordProjectionFailure, whose remaining fields stay
//! undefined unless a site recorded a failure; it may be null.
template <class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION bool ProjectM1Cell(
    const Primitive::PrimitiveSolver<EOSPolicy, ErrorPolicy> &ps,
    const radiationm1::RadiationM1OpacityType opacity_type,
    const Primitive::UnitSystem &code_units,
    const Primitive::UnitSystem &eos_units,
    const Primitive::UnitSystem &ngs_units,
    const Real initial_prim[NPRIM],
    const Real old_cons[NCONS], const Real b[NMAG], const Real g3d[NSPMETRIC],
    const Real g3u[NSPMETRIC],
    const Real old_rad[projection_nspecies][projection_nvars], Real tolerance,
    int max_iterations, Real relaxation, Real energy_floor, Real number_floor,
    Real final_prim[NPRIM], Real final_cons[NCONS],
    Real final_rad[projection_nspecies][projection_nvars],
    ProjectionFailureReport *fail_out) {
  const int nprim_active = PYF + ps.GetEOS().GetNSpecies();
  Real prim[NPRIM];
  Real rad[projection_nspecies][projection_nvars];
  int recover_code = 0;
  Real last_diff = -1.0;
  if (fail_out != nullptr) {
    fail_out->code = 0;
    fail_out->detail = 0;
    fail_out->iteration = -1;
  }
  for (int n = 0; n < nprim_active; ++n) {
    prim[n] = initial_prim[n];
  }
  for (int s = 0; s < projection_nspecies; ++s) {
    for (int v = 0; v < projection_nvars; ++v) {
      rad[s][v] = old_rad[s][v];
    }
  }

  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    Real target[projection_nspecies][projection_nvars];
    if (!EquilibriumM1Target(ps, opacity_type, code_units, eos_units,
                             ngs_units, prim, g3d, target)) {
      RecordProjectionFailure(fail_out, 100, 0, iteration, prim[PRH], prim[PTM],
                              prim[PYF], last_diff);
      return false;
    }

    Real trial_rad[projection_nspecies][projection_nvars];
    for (int s = 0; s < projection_nspecies; ++s) {
      for (int v = 0; v < projection_nvars; ++v) {
        trial_rad[s][v] = rad[s][v] + relaxation * (target[s][v] - rad[s][v]);
      }
    }

    Real trial_cons[NCONS];
    MatterComplement(old_cons, old_rad, trial_rad, ps.GetEOS().GetBaryonMass(),
                     trial_cons);
    Real recovered_cons[NCONS], recovered_prim[NPRIM];
    if (!RecoverMatter(ps, trial_cons, b, g3d, g3u, recovered_prim,
                       recovered_cons, &recover_code)) {
      // The refused trial is the C2P input, and the primitives C2P left behind
      // are read from recovered_prim only for a returned state, never an error.
      RecordProjectionFailure(fail_out, 101, recover_code, iteration, prim[PRH],
                              prim[PTM], prim[PYF], last_diff, trial_cons,
                              recovered_prim);
      return false;
    }

    Real recovered_target[projection_nspecies][projection_nvars];
    if (!EquilibriumM1Target(ps, opacity_type, code_units, eos_units,
                             ngs_units, recovered_prim, g3d,
                             recovered_target)) {
      RecordProjectionFailure(fail_out, 102, 0, iteration, prim[PRH], prim[PTM],
                              prim[PYF], last_diff);
      return false;
    }

    last_diff = ProjectionDifference(trial_rad, recovered_target, energy_floor,
                                     number_floor);
    if (last_diff <= tolerance) {
      Real exact_cons[NCONS], exact_prim[NPRIM];
      MatterComplement(old_cons, old_rad, recovered_target,
                       ps.GetEOS().GetBaryonMass(), exact_cons);
      if (!RecoverMatter(ps, exact_cons, b, g3d, g3u, exact_prim, final_cons,
                         &recover_code)) {
        RecordProjectionFailure(fail_out, 103, recover_code, iteration,
                                prim[PRH], prim[PTM], prim[PYF], last_diff,
                                exact_cons, exact_prim);
        return false;
      }
      Real exact_target[projection_nspecies][projection_nvars];
      const bool exact_target_ok = EquilibriumM1Target(
          ps, opacity_type, code_units, eos_units, ngs_units,
          exact_prim, g3d, exact_target);
      const Real exact_diff =
          exact_target_ok
              ? ProjectionDifference(recovered_target, exact_target, energy_floor,
                                     number_floor)
              : 0.0;
      if (!exact_target_ok || exact_diff > tolerance) {
        RecordProjectionFailure(fail_out, exact_target_ok ? 105 : 104, 0,
                                iteration, prim[PRH], prim[PTM], prim[PYF],
                                exact_diff);
        for (int n = 0; n < nprim_active; ++n) {
          prim[n] = exact_prim[n];
        }
        for (int s = 0; s < projection_nspecies; ++s) {
          for (int v = 0; v < projection_nvars; ++v) {
            rad[s][v] = recovered_target[s][v];
          }
        }
        continue;
      }

      for (int n = 0; n < nprim_active; ++n) {
        final_prim[n] = exact_prim[n];
      }
      for (int s = 0; s < projection_nspecies; ++s) {
        for (int v = 0; v < projection_nvars; ++v) {
          final_rad[s][v] = recovered_target[s][v];
        }
      }
      return true;
    }

    for (int n = 0; n < nprim_active; ++n) {
      prim[n] = recovered_prim[n];
    }
    for (int s = 0; s < projection_nspecies; ++s) {
      for (int v = 0; v < projection_nvars; ++v) {
        rad[s][v] = trial_rad[s][v];
      }
    }
  }
  RecordProjectionFailure(fail_out, 106, 0, max_iterations, prim[PRH], prim[PTM],
                          prim[PYF], last_diff);
  return false;
}

} // namespace ccsn

#endif // PGEN_CCSN_M1_PROJECTION_HPP_
