// Native regression of the adapter AND final opacity-kernel publication.
// Uses the stock single-zone pgen and a real CompOSE table; no mocked rates.
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>

#include "athena.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "radiation_m1/radiation_m1.hpp"

int main(int argc, char **argv) {
  if (argc != 3) return 2;  // single-zone input, CompOSE table
  global_variable::my_rank = 0;
  global_variable::nranks = 1;
  Kokkos::initialize();
  int failures = 0;
  {
    ParameterInput pin;
    std::ifstream input(argv[1]);
    pin.LoadFromStream(input);
    pin.SetString("mhd", "table", argv[2]);
    pin.SetBoolean("bns_nurates", "use_partial_equilibrium", false);
    pin.SetInteger("bns_nurates", "eq_warmup_cycles", 0);
    pin.SetReal("problem", "temp", 10);
    Mesh mesh(&pin);
    mesh.BuildTreeFromScratch(&pin);
    mesh.AddCoordinatesAndPhysics(&pin);
    mesh.pgen = std::make_unique<ProblemGenerator>(&pin, &mesh);
    mesh.dt = 1e-10;
    mesh.ncycle = 1;
    auto *pack = mesh.pmb_pack;
    auto &rad = *pack->pradm1;
    auto &np = rad.nurates_params;
    // The default secondary sentinel must copy the primary rule unchanged.
    bool alias_ok = np.quad_nx_2 == -1 && np.quadrature.nx == np.quadrature_2.nx;
    for (int q = 0; q < np.quadrature.nx; ++q) {
      alias_ok = alias_ok && np.quadrature.points[q] == np.quadrature_2.points[q] &&
                 np.quadrature.w[q] == np.quadrature_2.w[q];
    }
    std::cout << (alias_ok ? "PASS " : "FAIL ") << "secondary quadrature alias\n";
    failures += !alias_ok;
    auto *hyd = dynamic_cast<dyngr::DynGRMHDPS<
        Primitive::EOSCompOSE<Primitive::NormalLogs>, Primitive::ResetFloor> *>(
        pack->pdyngr);
    if (hyd == nullptr) return 2;
    auto &eos = hyd->eos.ps.GetEOSMutable();
    const auto cu = eos.GetCodeUnitSystem();
    const auto eu = eos.GetEOSUnitSystem();
    const auto nu = Primitive::MakeNGS();
    const int i = mesh.mb_indcs.is;
    const auto wh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pack->pmhd->w0);
    Real nb = wh(0, IDN, 0, 0, i) / eos.GetBaryonMass();
    Real y = wh(0, IYF, 0, 0, i);
    Real temp = eos.GetTemperatureFromP(nb, wh(0, IPR, 0, 0, i), &y);
    Real yp = eos.GetProtonFraction(nb, temp, &y);
    Real yn = eos.GetNeutronFraction(nb, temp, &y);
    Real mun = eos.GetBaryonChemicalPotential(nb, temp, &y);
    Real muq = eos.GetChargeChemicalPotential(nb, temp, &y);
    Real mup = mun + muq;
    Real mue = eos.GetElectronLeptonChemicalPotential(nb, temp, &y) - muq;
    Real neq[4]{}, jeq[4]{};
    radiationm1::NeutrinoDens(mun, mup, mue, temp, neq[0], neq[1], neq[2],
                             jeq[0], jeq[1], jeq[2], np, cu, eu, nu);
    neq[2] /= 2; neq[3] = neq[2];
    jeq[2] /= 2; jeq[3] = jeq[2];
    np.use_pair = np.use_brem = np.use_iso = false;
    // NEPS-only, beta-only, unsplit NEPS and mixed beta+NEPS cover all K/E arms.
    // Dilute at two fixed mean energies to test non-unit thermal correction
    // independently of occupation. Unsplit rates retain their total coefficients.
    for (int scenario = 0; scenario < 4; ++scenario) {
      np.use_nonthermal_separated = scenario != 2;
      np.use_abs_em = scenario == 1 || scenario == 3;
      np.use_inelastic_scatt = scenario != 1;
      for (int eq = 0; eq < 2; ++eq) {
        np.use_equilibrium_distribution = eq;
        for (int kirchhoff = 0; kirchhoff < 2; ++kirchhoff) {
          np.use_kirchhoff_law = kirchhoff;
          for (int sample = 0; sample < 4; ++sample) {
            const Real f = sample % 2 ? 1.0 / 4.0 : 1.0;
            const Real mean_factor = sample < 2 ? 1.0 : 2.0;
            Real n[4], j[4], chi[4], e0[4], e1[4], a0[4], a1[4];
            Real s0[4], s1[4], en[4], an[4], an0[4];
            auto uh = Kokkos::create_mirror_view(rad.u0);
            Kokkos::deep_copy(uh, 0.0);
            for (int s = 0; s < 4; ++s) {
              n[s] = f * neq[s]; j[s] = f * jeq[s] * mean_factor;
              chi[s] = 1.0 / 3.0;
              for (int x = 0; x < static_cast<int>(uh.extent(4)); ++x) {
                uh(0, radiationm1::CombinedIdx(s, M1_E_IDX, rad.nvars), 0, 0, x) = j[s];
                uh(0, radiationm1::CombinedIdx(s, M1_N_IDX, rad.nvars), 0, 0, x) = n[s];
              }
            }
            Kokkos::deep_copy(rad.u0, uh);
            radiationm1::ComputeNuratesOpacities(
                nb, temp, yp, yn, mun, mup, mue, n, j, chi,
                e0, e1, a0, a1, s0, s1, en, an, an0, np, cu, eu, nu);
            rad.FloorAndCalcClosure(nullptr, 1);
            rad.CalcOpacityNurates(nullptr, 1);
            const auto eh0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rad.eta_0);
            const auto ah0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rad.abs_0);
            const auto eh1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rad.eta_1);
            bool pass = true;
            for (int s = 0; s < 4; ++s) {
              // Only electron-flavour thermal coefficients receive the capped
              // mean-energy correction. NEPS gets occupation, not this factor.
              const Real corr = eq && s < 2
                  ? std::fmin(mean_factor * mean_factor, np.opacity_corr_fac_max) : 1;
              Real expect = (kirchhoff && a1[s] - an[s] > 0)
                  ? (a1[s] - an[s]) * jeq[s] : std::fmax(e1[s] - en[s], 0.0);
              expect = corr * expect + en[s] * (eq ? f : 1);
              pass = pass && std::isfinite(eh1(0, s, 0, 0, i));
              pass = pass && std::abs(eh1(0, s, 0, 0, i) - expect) <=
                  1e-12 * std::abs(expect);
              if (scenario == 0) {
                pass = pass && e0[s] == 0 && a0[s] == 0 && an0[s] == 0 && en[s] > 0;
                pass = pass && eh0(0, s, 0, 0, i) == 0 && ah0(0, s, 0, 0, i) == 0;
              } else {
                const Real expect0 = kirchhoff
                    ? (a0[s] > 0 ? corr * a0[s] * neq[s] : e0[s]) : corr * e0[s];
                pass = pass && std::abs(eh0(0, s, 0, 0, i) - expect0) <=
                    1e-12 * std::abs(expect0);
                pass = pass && (scenario == 3 ? en[s] > 0 && an0[s] == 0 : en[s] == 0);
              }
            }
            std::cout << (pass ? "PASS " : "FAIL ") << "native opacity scenario="
                      << scenario << " E=" << eq << " K=" << kirchhoff << " f=" << f
                      << " mean_factor=" << mean_factor << '\n';
            failures += !pass;
          }
        }
      }
    }
  }
  Kokkos::finalize();
  return failures ? 1 : 0;
}
