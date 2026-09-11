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
template <class EOSPolicy, class ErrorPolicy>
KOKKOS_INLINE_FUNCTION bool RecoverMatter(
    const Primitive::PrimitiveSolver<EOSPolicy, ErrorPolicy> &ps,
    const Real cons_in[NCONS], const Real b_in[NMAG],
    const Real g3d[NSPMETRIC], const Real g3u[NSPMETRIC], Real prim[NPRIM],
    Real cons_out[NCONS]) {
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
  if (result.error != Primitive::Error::SUCCESS || result.cons_floor ||
      result.prim_floor || result.cons_adjusted) {
    return false;
  }
  for (int n = 0; n < NCONS; ++n) {
    if (!Kokkos::isfinite(cons_out[n]) || cons_out[n] != original_cons[n]) {
      return false;
    }
  }
  const int nprim_active = PYF + ps.GetEOS().GetNSpecies();
  for (int n = 0; n < nprim_active; ++n) {
    if (!Kokkos::isfinite(prim[n])) {
      return false;
    }
  }
  return b[IBX] == b_in[IBX] && b[IBY] == b_in[IBY] && b[IBZ] == b_in[IBZ];
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
    Real final_rad[projection_nspecies][projection_nvars]) {
  const int nprim_active = PYF + ps.GetEOS().GetNSpecies();
  Real prim[NPRIM];
  Real rad[projection_nspecies][projection_nvars];
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
                       recovered_cons)) {
      return false;
    }

    Real recovered_target[projection_nspecies][projection_nvars];
    if (!EquilibriumM1Target(ps, opacity_type, code_units, eos_units,
                             ngs_units, recovered_prim, g3d,
                             recovered_target)) {
      return false;
    }

    if (ProjectionDifference(trial_rad, recovered_target, energy_floor,
                             number_floor) <= tolerance) {
      Real exact_cons[NCONS], exact_prim[NPRIM];
      MatterComplement(old_cons, old_rad, recovered_target,
                       ps.GetEOS().GetBaryonMass(), exact_cons);
      if (!RecoverMatter(ps, exact_cons, b, g3d, g3u, exact_prim, final_cons)) {
        return false;
      }
      Real exact_target[projection_nspecies][projection_nvars];
      if (!EquilibriumM1Target(ps, opacity_type, code_units, eos_units,
                               ngs_units, exact_prim, g3d,
                               exact_target) ||
          ProjectionDifference(recovered_target, exact_target, energy_floor,
                               number_floor) > tolerance) {
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
  return false;
}

} // namespace ccsn

#endif // PGEN_CCSN_M1_PROJECTION_HPP_
