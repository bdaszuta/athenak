#ifndef RADIATION_M1_RADIATION_M1_WEAKRATES_EQUILIBRIUM_SEARCH_HPP_
#define RADIATION_M1_RADIATION_M1_WEAKRATES_EQUILIBRIUM_SEARCH_HPP_
//========================================================================================
// Bounded best-found WeakRates targets. All state is cell-local device storage;
// no EOS extrapolation, changes to conserved budgets, or evolved-state writes.
//========================================================================================

#include <limits>

#include "eos/primitive-solver/ps_types.hpp"
#include "eos/primitive-solver/unit_system.hpp"
#include "radiation_m1_weakrates.hpp"

namespace radiationm1 {
namespace weakrates {

//! Target quality, distinct from the generic EOS solver's status.
enum class GraEquilibriumStatus {
  Local, Conserving, Approximate, LocalFallback,
  InvalidInput, Unsupported, NoUsableTarget
};

//! Defects refer to the analytic endpoint, not its approximate FD moments or mix.
//! Temperature/energy use code units; Ye/lepton defect are dimensionless.
struct GraEquilibriumResult {
  GraEquilibriumStatus status = GraEquilibriumStatus::InvalidInput;
  Real temp = 0.0, ye = 0.0;
  Real energy_defect = 0.0, lepton_defect = 0.0, score = 0.0;
  // Absolute component defects divided by the fixed cell exchange scales.
  Real normalized_energy_defect = 0.0, normalized_lepton_defect = 0.0;
  int eos_status = -1, evaluations = 0;
  bool on_boundary = false;

  KOKKOS_INLINE_FUNCTION bool usable() const {
    return status == GraEquilibriumStatus::Local ||
           status == GraEquilibriumStatus::Conserving || approximate();
  }
  KOKKOS_INLINE_FUNCTION bool approximate() const {
    return status == GraEquilibriumStatus::Approximate ||
           status == GraEquilibriumStatus::LocalFallback;
  }
};

//! Read-only snapshots of the producer's coefficient pipeline at selection time.
//! All arrays have four entries. Pointers refer only to the invoking cell's stack.
//! Candidate trials use copies; rejected targets cannot change caller coefficients.
struct GraTargetValidation {
  const RawCoefficients *raw = nullptr;
  const WeakRatesParams *params = nullptr;
  const Real *current_number = nullptr, *current_energy = nullptr;
  const Real *local_number = nullptr, *local_energy = nullptr;
  const Real *eta_0 = nullptr, *eta_1 = nullptr;
  const Real *abs_0 = nullptr, *abs_1 = nullptr, *scat_1 = nullptr;
  bool thermal_enabled = false;

  KOKKOS_INLINE_FUNCTION bool operator()(const Real *number,
                                          const Real *energy) const {
    if (raw == nullptr || params == nullptr || current_number == nullptr ||
        current_energy == nullptr || local_number == nullptr ||
        local_energy == nullptr || eta_0 == nullptr || eta_1 == nullptr ||
        abs_0 == nullptr || abs_1 == nullptr || scat_1 == nullptr) return false;
    Real en[4], ej[4], an[4], aj[4], sc[4];
    for (int s = 0; s < 4; ++s) {
      en[s] = eta_0[s]; ej[s] = eta_1[s];
      an[s] = abs_0[s]; aj[s] = abs_1[s]; sc[s] = scat_1[s];
    }
    if (params->gra_opacity_order || params->gra_target_mean) {
      if (params->gra_target_mean) {
        for (int s = 0; s < 4; ++s) {
          const int rs = s < 2 ? s : kNuX;
          sc[s] = raw->scat_1[rs];
          if (s < 2) {
            an[s] = raw->abs_0[rs]; aj[s] = raw->abs_1[rs];
          } else {
            en[s] = 0.5*raw->eta_0[kNuX]; ej[s] = 0.5*raw->eta_1[kNuX];
          }
        }
      }
      ApplySpectralCorrection(4, params->opacity_corr_fac_max,
          current_number, current_energy,
          params->gra_target_mean ? number : local_number,
          params->gra_target_mean ? energy : local_energy, en, ej, an, aj, sc);
    }
    if (!FinalizeKirchhoffCoefficients(4, true, number, energy,
                                      en, ej, an, aj, sc)) return false;
    if (!thermal_enabled) return true;
    ElectronThermalCoefficients thermal{};
    if (!BuildElectronThermalCoefficients(*raw, number, energy, thermal)) return false;
    // References and current numbers are physical comoving code-unit densities,
    // matching EvaluateElectronThermalRates. Check its target-dependent divisions
    // and products before publication. Future implicit iterates still require
    // the independent source consumer's validity and acceptance checks.
    Real ratio[2]{};
    if (thermal.r > 0.0 || thermal.q > 0.0) {
      for (int s = 0; s < 2; ++s) {
        ratio[s] = current_number[s]/thermal.number[s];
        if (!Kokkos::isfinite(ratio[s]) || ratio[s] < 0.0) return false;
      }
    }
    if (thermal.r > 0.0) {
      const Real product = ratio[0]*ratio[1];
      if (!Kokkos::isfinite(product) ||
          !Kokkos::isfinite(thermal.r*(1.0 - product))) return false;
    }
    if (thermal.q > 0.0) {
      for (int s = 0; s < 2; ++s) {
        const Real coefficient = thermal.q/thermal.energy[s];
        if (!Kokkos::isfinite(coefficient) ||
            !Kokkos::isfinite(coefficient*ratio[1 - s])) return false;
      }
    }
    return true;
  }
};

namespace gra_equilibrium_detail {

//! Stable norm; unrepresentably large normalized defects saturate, never abort.
KOKKOS_INLINE_FUNCTION
Real DefectNorm(Real a, Real b) {
  const Real limit = std::numeric_limits<Real>::max();
  a = Kokkos::fmin(Kokkos::fabs(a), limit);
  b = Kokkos::fmin(Kokkos::fabs(b), limit);
  const Real scale = Kokkos::fmax(a, b);
  if (scale == 0.0) return 0.0;
  const Real factor = Kokkos::sqrt((a/scale)*(a/scale) + (b/scale)*(b/scale));
  return scale > limit/factor ? limit : scale*factor;
}

//! Fixed problem metadata and read-only references. Number conversion explicitly
//! undoes GetTrappedNeutrinos' code-density output before division by EOS nb.
struct GraProblem {
  Real nb, temp, ye, t_min, t_max, y_min, y_max;
  Real t_to_eos, mu_to_eos, n_to_eos, e_rhs, yl_rhs, lambda;
  Real energy_scale, lepton_scale;
  UnitConversions conversions;
  const Real *local_number, *local_energy;
  const GraTargetValidation *validation;
};

//! A trial is scored independently of EOS status; only publishable trials compete.
struct GraCandidate {
  GraEquilibriumResult result{};
  Real number[4]{}, energy[4]{};
  bool valid = false, conserving = false, local = false;
};

template<class EOS>
struct GraSearch {
  const EOS &eos;
  const GraProblem &p;
  GraCandidate best{};
  int evaluations = 0;

  KOKKOS_INLINE_FUNCTION
  GraCandidate Evaluate(const Real temp, const Real ye, const bool local = false) {
    ++evaluations;
    GraCandidate c{};
    if (!Kokkos::isfinite(temp) || !Kokkos::isfinite(ye) ||
        !(temp >= p.t_min && temp <= p.t_max &&
          ye >= p.y_min && ye <= p.y_max)) return c;
    Real y[MAX_SPECIES] = {ye, 0.0, 0.0};
    Real net[3]{}, pair[3]{};
    eos.GetTrappedNeutrinos(p.nb, temp, y, net, pair);
    for (int g = 0; g < 3; ++g) {
      if (!Kokkos::isfinite(net[g]) || !Kokkos::isfinite(pair[g]) ||
          pair[g] < 0.0) return c;
    }
    const Real n_eos = net[0]*p.n_to_eos;
    const Real e_mat = eos.GetEnergy(p.nb, temp, y);
    const Real e_total = e_mat + pair[0] + pair[1] + pair[2];
    c.result.temp = temp;
    c.result.ye = ye;
    c.result.energy_defect = e_total - p.e_rhs;
    c.result.lepton_defect = ye + n_eos/p.nb - p.yl_rhs;
    if (!Kokkos::isfinite(n_eos) || !Kokkos::isfinite(e_mat) ||
        !Kokkos::isfinite(c.result.energy_defect) ||
        !Kokkos::isfinite(c.result.lepton_defect)) return c;
    const Real defect_limit = std::numeric_limits<Real>::max();
    c.result.normalized_energy_defect = Kokkos::fmin(
        Kokkos::fabs(c.result.energy_defect/p.energy_scale), defect_limit);
    c.result.normalized_lepton_defect = Kokkos::fmin(
        Kokkos::fabs(c.result.lepton_defect/p.lepton_scale), defect_limit);
    c.result.score = DefectNorm(c.result.normalized_energy_defect,
                               c.result.normalized_lepton_defect);
    // Preserve the original norm and tolerance wherever the signed budget is
    // nonzero. At zero budget use the fixed, uncancelled population scale.
    const Real rl = p.yl_rhs != 0.0 ? c.result.lepton_defect/p.yl_rhs :
                                     c.result.lepton_defect/p.lepton_scale;
    const Real re = e_total/p.e_rhs - 1.0;
    c.conserving = Kokkos::isfinite(rl) && Kokkos::isfinite(re) &&
                  Kokkos::fabs(rl) + Kokkos::fabs(re) <= 1.0e-7;
    c.result.on_boundary = temp == p.t_min || temp == p.t_max ||
                           ye == p.y_min || ye == p.y_max;
    c.local = local;
    if (local) {
      // Preserve the caller's microscopic local evaluation, including its
      // rounding, when it is the only usable safety target.
      for (int s = 0; s < 4; ++s) {
        c.number[s] = p.local_number[s]; c.energy[s] = p.local_energy[s];
      }
    } else {
      const Real t_mev = temp*p.t_to_eos;
      const Real mu_mev =
          eos.GetElectronLeptonChemicalPotential(p.nb, temp, y)*p.mu_to_eos;
      const Real eta = mu_mev/t_mev;
      if (!Kokkos::isfinite(t_mev) || t_mev <= 0.0 ||
          !Kokkos::isfinite(mu_mev) || !Kokkos::isfinite(eta)) return c;
      EquilibriumDensities full = ComputeEquilibriumDensities(t_mev, eta);
      ConvertToCodeUnits(p.conversions, full);
      if (!IsFinite(full)) return c;
      for (int s = 0; s < 4; ++s) {
        const int rs = s < 2 ? s : kNuX;
        const Real split = s < 2 ? 1.0 : 0.5;
        const Real n_full = split*full.number[rs];
        const Real j_full = split*full.energy[rs];
        c.number[s] = p.lambda*n_full + (1.0 - p.lambda)*p.local_number[s];
        c.energy[s] = p.lambda*j_full + (1.0 - p.lambda)*p.local_energy[s];
      }
    }
    for (int s = 0; s < 4; ++s) {
      if (!Kokkos::isfinite(c.number[s]) || c.number[s] < 0.0 ||
          !Kokkos::isfinite(c.energy[s]) || c.energy[s] < 0.0) return c;
    }
    if (p.validation != nullptr && !(*p.validation)(c.number, c.energy)) return c;
    c.valid = true;
    return c;
  }

  KOKKOS_INLINE_FUNCTION
  Real Distance(const GraCandidate &c) const {
    const Real dlogt = (Kokkos::log(c.result.temp) - Kokkos::log(p.temp))/
                      (Kokkos::log(p.t_max) - Kokkos::log(p.t_min));
    const Real dy = (c.result.ye - p.ye)/(p.y_max - p.y_min);
    return DefectNorm(dlogt, dy);
  }

  KOKKOS_INLINE_FUNCTION
  Real Consider(const Real temp, const Real ye, const bool local = false) {
    const GraCandidate c = Evaluate(temp, ye, local);
    if (!c.valid) return std::numeric_limits<Real>::infinity();
    if (!best.valid || c.result.score < best.result.score ||
        (c.result.score == best.result.score && Distance(c) < Distance(best))) {
      best = c;
    }
    return c.result.score;
  }

  // Coordinates are normalized log(T) and Ye. Return exact table endpoints;
  // exp(log(T_max)) can otherwise round one ULP outside the closed domain.
  KOKKOS_INLINE_FUNCTION
  Real Temperature(const Real x) const {
    if (x <= 0.0) return p.t_min;
    if (x >= 1.0) return p.t_max;
    return Kokkos::fmax(p.t_min, Kokkos::fmin(p.t_max,
        Kokkos::exp(Kokkos::log(p.t_min) +
                    x*(Kokkos::log(p.t_max) - Kokkos::log(p.t_min)))));
  }
  KOKKOS_INLINE_FUNCTION
  Real Fraction(const Real x) const {
    if (x <= 0.0) return p.y_min;
    if (x >= 1.0) return p.y_max;
    return p.y_min + x*(p.y_max - p.y_min);
  }
  KOKKOS_INLINE_FUNCTION
  Real Point(const Real x, const Real y) {
    return Consider(Temperature(x), Fraction(y));
  }
  KOKKOS_INLINE_FUNCTION
  Real EdgePoint(const int edge, const Real x) {
    return edge < 2 ? Point(static_cast<Real>(edge), x) :
                      Point(x, static_cast<Real>(edge - 2));
  }

  //! Refine every sampled local minimum, not just the first bracket. The bounded
  //! golden search is best-found only; tabulated residuals need not be convex.
  KOKKOS_INLINE_FUNCTION
  void Edges() {
    constexpr int segments = 16, refinements = 48;
    constexpr Real golden = 0.6180339887498948482;
    for (int edge = 0; edge < 4; ++edge) {
      Real scores[segments + 1];
      for (int j = 0; j <= segments; ++j) {
        scores[j] = EdgePoint(edge, static_cast<Real>(j)/segments);
      }
      for (int j = 0; j <= segments; ++j) {
        if (!Kokkos::isfinite(scores[j])) continue;
        const Real left = j > 0 ? scores[j - 1] :
                                 std::numeric_limits<Real>::infinity();
        const Real right = j < segments ? scores[j + 1] :
                                         std::numeric_limits<Real>::infinity();
        if (!(scores[j] <= left && scores[j] <= right &&
              (scores[j] < left || scores[j] < right))) continue;
        Real a = static_cast<Real>(j > 0 ? j - 1 : 0)/segments;
        Real b = static_cast<Real>(j < segments ? j + 1 : segments)/segments;
        Real u = b - golden*(b - a), v = a + golden*(b - a);
        Real fu = EdgePoint(edge, u), fv = EdgePoint(edge, v);
        for (int it = 0; it < refinements; ++it) {
          if (fu <= fv) {
            b = v; v = u; fv = fu; u = b - golden*(b - a);
            fu = EdgePoint(edge, u);
          } else {
            a = u; u = v; fu = fv; v = a + golden*(b - a);
            fv = EdgePoint(edge, v);
          }
        }
      }
    }
  }

  //! Safeguarded derivative-free interior refinement. Retain failed-solve and
  //! local seeds without extrapolating or trusting their EOS status. Poll all
  //! eight directions before moving; shrink only when none improves the score.
  KOKKOS_INLINE_FUNCTION
  void Interior(const Real temp, const Real ye) {
    if (!Kokkos::isfinite(temp) || !Kokkos::isfinite(ye) ||
        !(temp >= p.t_min && temp <= p.t_max &&
          ye >= p.y_min && ye <= p.y_max)) return;
    Real x = (Kokkos::log(temp) - Kokkos::log(p.t_min))/
             (Kokkos::log(p.t_max) - Kokkos::log(p.t_min));
    Real y = (ye - p.y_min)/(p.y_max - p.y_min);
    Real score = Point(x, y), step = 0.25;
    for (int it = 0; it < 96; ++it) {
      Real next_x = x, next_y = y, next_score = score;
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          if (dx == 0 && dy == 0) continue;
          const Real tx = Kokkos::fmax(0.0, Kokkos::fmin(1.0, x + step*dx));
          const Real ty = Kokkos::fmax(0.0, Kokkos::fmin(1.0, y + step*dy));
          const Real trial = Point(tx, ty);
          if (trial < next_score) {
            next_x = tx; next_y = ty; next_score = trial;
          }
        }
      }
      if (next_score < score) {
        x = next_x; y = next_y; score = next_score;
      } else {
        step *= 0.5;
        if (step <= std::numeric_limits<Real>::epsilon()) break;
      }
    }
  }

  //! Publish once after acceptance. On failure, all caller output arrays remain
  //! untouched. Local safety fallback is distinguished from a refined endpoint.
  KOKKOS_INLINE_FUNCTION
  GraEquilibriumResult Publish(Real *number, Real *energy, const int eos_status) {
    GraEquilibriumResult result{};
    result.status = GraEquilibriumStatus::NoUsableTarget;
    if (best.valid) {
      result = best.result;
      result.status = best.conserving ? GraEquilibriumStatus::Conserving :
          (best.local ? GraEquilibriumStatus::LocalFallback :
                        GraEquilibriumStatus::Approximate);
      for (int s = 0; s < 4; ++s) {
        number[s] = best.number[s]; energy[s] = best.energy[s];
      }
    }
    result.eos_status = eos_status;
    result.evaluations = evaluations;
    return result;
  }
};

}  // namespace gra_equilibrium_detail
}  // namespace weakrates
}  // namespace radiationm1
#endif  // RADIATION_M1_RADIATION_M1_WEAKRATES_EQUILIBRIUM_SEARCH_HPP_
