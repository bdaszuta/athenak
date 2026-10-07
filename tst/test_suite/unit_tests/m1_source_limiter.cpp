// Source-only native TimeUpdate regression; run only after execution is approved.
// Reuses the stock single-zone deck and reduced CompOSE table, but does not run
// a pgen, an opacity producer, a Driver, or C2P. Synthetic nonnegative coefficients
// isolate the limiter while real mesh/EOS resources and the production kernel run.
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "radiation_m1/radiation_m1.hpp"

namespace {
using radiationm1::CombinedIdx;
using Hydro = dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Primitive::NormalLogs>,
                               Primitive::ResetFloor>;

// Independent storage is required on Serial as well as CUDA: mirror_view
// may alias the evolving source, invalidating the pre-update oracle baseline.
template <class View>
auto Snapshot(const View &view) {
  auto host = Kokkos::create_mirror(view);
  Kokkos::deep_copy(host, view);
  return host;
}

// The rest-frame geometry has alpha=1/2, sqrt(gamma)=2, W=1 and J=E.
// Above the energy floor, the fixed number-frame cap gives Gamma=1-1e-5.
// Stage dt is chosen so h=beta_stage*dt=1 in either stage. Independently:
// dE = eta_1 - abs_1*E/2, dFx = -(abs_1+scat_1)*Fx/2,
// dN = eta_0 - abs_0*N/(2*Gamma). Absorption coefficients include Gamma
// below so the intended number changes are binary-exact. The decimal energy
// case instead checks strict bounds and nonzero acceptance, not an exact theta.
constexpr Real kNumberGamma = 1.0 - 1.0e-5;
struct Case {
  Real rates[4][5]{};  // eta_0, abs_0, eta_1, abs_1, scat_1
  Real theta = 1.0;
  Real limiter = 1.0 / 2.0;
  Real tau = 4.0;
  bool zero_ye_budget = false;
  bool matter = true;
  bool limiting = true;
  bool feedback = true;
  bool masked = false;
  bool floor_only = false;
  bool check_energy_bound = false;
};

const char *case_names[] = {
    "energy-signed-cancellation", "Ye-emission-cancellation-at-endpoint",
    "Ye-absorption-cancellation-at-endpoint", "individual-energy-depletion",
    "individual-number-depletion", "E-F-lepton-feedback",
    "zero-source-limiter-gate", "matter-sources-off-gate", "theta-limiter-off",
    "negative-source-limiter", "backreaction-off", "excision-mask",
    "nonfinite-cell-rejection", "floor-only-no-charge", "energy-net-positive-control",
    "energy-cancellation-zero-tau", "energy-cancellation-negative-tau",
    "rounded-energy-bound-nonzero-acceptance"};

Case MakeCase(int test) {
  Case c;
  if (test == 0 || test == 15 || test == 16) {
    c.tau = test == 15 ? 0.0 : (test == 16 ? -1.0 / 8.0 : 1.0 / 8.0);
    c.rates[0][2] = 1.0; c.rates[1][3] = 1.0 / 2.0;
    // dE=(+1,-1): net matter debit vanishes, despite a small tau budget.
  } else if (test == 1 || test == 2) {
    c.zero_ye_budget = true;
    if (test == 1) c.rates[0][0] = c.rates[1][0] = 1.0;
    else c.rates[0][1] = c.rates[1][1] = kNumberGamma / 4.0;
  } else if (test == 3 || test == 8 || test == 9) {
    c.rates[0][3] = 1.0;
    c.limiter = 1.0 / 4.0;
    c.theta = 1.0 / 2.0;  // dE=-2, loss allowance=(1/4)*4=1.
    if (test == 8) { c.limiting = false; c.theta = 1.0; }
    if (test == 9) { c.limiter = -1.0; c.theta = 1.0; }
  } else if (test == 4) {
    c.rates[0][1] = kNumberGamma / 2.0;
    c.limiter = 1.0 / 8.0;
    c.theta = 1.0 / 2.0;  // dN=-2, loss allowance=(1/8)*8=1.
  } else {
    c.rates[0][2] = 1.0 / 2.0; c.rates[1][2] = 1.0 / 4.0;
    c.rates[0][4] = 1.0 / 2.0; c.rates[1][4] = 1.0 / 4.0;
    c.rates[0][0] = 1.0 / 4.0; c.rates[1][0] = 1.0 / 2.0;
    if (test == 6) { c.limiter = 0.0; c.theta = 0.0; }
    if (test == 7) { c.matter = false; c.theta = 0.0; }
    if (test == 10) c.feedback = false;
    if (test == 11) c.masked = true;
    if (test == 12) {
      c.rates[0][0] = std::numeric_limits<Real>::quiet_NaN();
      c.theta = 0.0;
    }
    if (test == 13) {
      c = Case{};
      c.floor_only = true; c.limiter = 0.0; c.theta = 0.0;
    }
    if (test == 14) {
      c.rates[1][2] = 1.0 / 2.0;
      c.tau = 1.0 / 2.0;
      c.theta = 1.0 / 4.0;  // Net dE=1, matter budget=(1/2)*(1/2)=1/4.
    }
    if (test == 17) {
      // A decimal-valued binding energy budget probes the strict endpoint
      // check. Require a nonzero safe step, not a particular rounded theta.
      c = Case{}; c.tau = 0.3; c.rates[0][2] = 0.2;
      c.check_energy_bound = true;
    }
  }
  return c;
}

class Fixture {
 public:
  ParameterInput pin;
  std::unique_ptr<Mesh> mesh;
  Real mb;

  Fixture(const char *deck, const char *table, int ng) {
    std::ifstream stream(deck);
    if (!stream) throw std::runtime_error("missing single-zone deck");
    pin.LoadFromStream(stream);
    pin.SetString("mhd", "table", table);
    pin.SetString("mhd", "units", "geometric_solar");
    pin.SetBoolean("mhd", "use_NQT", false);
    pin.SetBoolean("coord", "minkowski", true);
    pin.SetInteger("mesh", "nx1", 8);
    pin.SetInteger("meshblock", "nx1", 8);
    pin.SetInteger("mesh", "nghost", ng);
    pin.SetString("radiation_m1", "opacity_type", "bns-nurates");
    pin.SetBoolean("bns_nurates", "use_partial_equilibrium", false);
    mesh = std::make_unique<Mesh>(&pin);
    mesh->BuildTreeFromScratch(&pin);
    mesh->AddCoordinatesAndPhysics(&pin);
    auto *hyd = dynamic_cast<Hydro *>(mesh->pmb_pack->pdyngr);
    if (!hyd) throw std::runtime_error("NormalLogs CompOSE fixture mismatch");
    mb = hyd->eos.ps.GetEOSMutable().GetBaryonMass();
    if (!(std::isfinite(mb) && mb > 0.0)) throw std::runtime_error("invalid mb");
    if (Rad().nspecies != 4 || Rad().nvars != 5 || mesh->pmb_pack->nmb_thispack != 1)
      throw std::runtime_error("fixture requires one block and four real transport groups");
    auto adm = mesh->pmb_pack->padm->adm;
    const int cells = Rad().u0.extent(4);
    par_for("m1_limiter_uniform_geometry", DevExeSpace(), 0, 0, 0, 0, 0, 0,
            0, cells - 1, KOKKOS_LAMBDA(int m, int k, int j, int i) {
      adm.alpha(m,k,j,i) = 1.0 / 2.0;
      adm.psi4(m,k,j,i) = Kokkos::cbrt(4.0);
      for (int a = 0; a < 3; ++a) {
        adm.beta_u(m,a,k,j,i) = 0.0;
        for (int b = a; b < 3; ++b) {
          adm.g_dd(m,a,b,k,j,i) = a == b ? (a == 0 ? 4.0 : 1.0) : 0.0;
          adm.vK_dd(m,a,b,k,j,i) = 0.0;
        }
      }
    });
  }

  radiationm1::RadiationM1 &Rad() { return *mesh->pmb_pack->pradm1; }

  bool Run(int test, int stage) {
    const Case c = MakeCase(test);
    auto &rad = Rad();
    auto *pack = mesh->pmb_pack;
    auto &p = rad.params;
    p.src_update = radiationm1::Explicit;
    p.closure_type = radiationm1::Eddington;
    p.gr_sources = false; p.gra_source_dispatch = false;
    p.matter_sources = c.matter; p.backreact = c.feedback;
    p.theta_limiter = c.limiting; p.source_limiter = c.limiter;
    p.rad_E_floor = p.rad_N_floor = 1.0 / 16.0;
    const Real ye_width = 16.0 * std::fmax(mb, 1.0);
    p.source_Ye_min = c.zero_ye_budget ? 1.0 / 4.0 : 1.0 / 4.0 - ye_width;
    p.source_Ye_max = c.zero_ye_budget ? 1.0 / 4.0 : 1.0 / 4.0 + ye_width;
    mesh->dt = stage == 1 ? 2.0 : 1.0;
    Kokkos::deep_copy(rad.u0, 0.0);
    Kokkos::deep_copy(rad.chi, 1.0 / 3.0);
    Kokkos::deep_copy(rad.radiation_mask, c.masked);
    for (auto view : {rad.eta_0, rad.abs_0, rad.eta_1, rad.abs_1, rad.scat_1})
      Kokkos::deep_copy(view, 0.0);
    Kokkos::deep_copy(rad.uflx.x1f, 0.0);
    Kokkos::deep_copy(rad.uflx.x2f, 0.0);
    Kokkos::deep_copy(rad.uflx.x3f, 0.0);
    Kokkos::deep_copy(pack->pmhd->w0, 0.0);
    Kokkos::deep_copy(pack->pmhd->u0, 0.0);
    auto r = Snapshot(rad.u0);
    auto w = Snapshot(pack->pmhd->w0);
    auto u = Snapshot(pack->pmhd->u0);
    // Deliberately seed conservative budgets independently of the EOS pressure:
    // no primitive recovery or atmosphere repair may conceal a wrong debit.
    for (int i = 0; i < static_cast<int>(r.extent(4)); ++i) {
      w(0,IDN,0,0,i) = 1.0; w(0,IPR,0,0,i) = 1.0;
      w(0,IYF,0,0,i) = 1.0 / 4.0;
      u(0,IDN,0,0,i) = 2.0; u(0,IEN,0,0,i) = c.tau;
      u(0,IM1,0,0,i) = 1.0 / 4.0;
      u(0,IYF,0,0,i) = 1.0 / 2.0;
      for (int s = 0; s < 4; ++s) {
        r(0,CombinedIdx(s,M1_E_IDX,rad.nvars),0,0,i) =
            c.floor_only && s == 0 ? 1.0 / 32.0 : 4.0;
        r(0,CombinedIdx(s,M1_N_IDX,rad.nvars),0,0,i) =
            c.floor_only && s == 0 ? 1.0 / 32.0 : 8.0;
        r(0,CombinedIdx(s,M1_FX_IDX,rad.nvars),0,0,i) =
            c.floor_only && s == 0 ? 0.0 : 1.0 / 8.0;
      }
    }
    Kokkos::deep_copy(rad.u0, r); Kokkos::deep_copy(rad.u1, r);
    Kokkos::deep_copy(pack->pmhd->w0, w); Kokkos::deep_copy(pack->pmhd->u0, u);
    int channel = 0;
    for (auto view : {rad.eta_0, rad.abs_0, rad.eta_1, rad.abs_1, rad.scat_1}) {
      auto host = Snapshot(view);
      for (int i = 0; i < static_cast<int>(host.extent(4)); ++i)
        for (int s = 0; s < 4; ++s) host(0,s,0,0,i) = c.rates[s][channel];
      Kokkos::deep_copy(view, host);
      ++channel;
    }
    bool pass = rad.TimeUpdate(nullptr, stage) == TaskStatus::complete;
    const auto final_r = Snapshot(rad.u0);
    const auto final_u = Snapshot(pack->pmhd->u0);
    const auto final_w = Snapshot(pack->pmhd->w0);
    for (int i = mesh->mb_indcs.is; i <= mesh->mb_indcs.ie; ++i) {
      Real de = 0.0, df = 0.0, dn[2]{};
      for (int s = 0; s < 4; ++s) {
        const Real e0 = r(0,CombinedIdx(s,M1_E_IDX,rad.nvars),0,0,i);
        const Real f0 = r(0,CombinedIdx(s,M1_FX_IDX,rad.nvars),0,0,i);
        const Real n0 = r(0,CombinedIdx(s,M1_N_IDX,rad.nvars),0,0,i);
        const Real eb = std::fmax(p.rad_E_floor, e0);
        const Real nb = std::fmax(p.rad_N_floor, n0);
        Real ef = eb, ff = f0, nf = nb;
        // Avoid any nonfinite arithmetic in the independent rejection oracle.
        if (c.theta > 0.0) {
          ef = std::fmax(p.rad_E_floor, e0 + c.theta * (c.rates[s][2] - c.rates[s][3]*e0/2.0));
          ff = f0 - c.theta * (c.rates[s][3] + c.rates[s][4])*f0/2.0;
          nf = std::fmax(p.rad_N_floor, n0 + c.theta * (c.rates[s][0] - c.rates[s][1]*n0/(2.0*kNumberGamma)));
        }
        if (c.check_energy_bound && s == 0) {
          ef = final_r(0,CombinedIdx(s,M1_E_IDX,rad.nvars),0,0,i);
          pass = pass && std::isfinite(ef) && ef > eb &&
                 ef - eb <= c.limiter * c.tau;
        }
        if (c.masked) ef = ff = nf = 0.0;
        pass = pass && final_r(0,CombinedIdx(s,M1_E_IDX,rad.nvars),0,0,i) == ef &&
            final_r(0,CombinedIdx(s,M1_FX_IDX,rad.nvars),0,0,i) == ff &&
            final_r(0,CombinedIdx(s,M1_N_IDX,rad.nvars),0,0,i) == nf &&
            final_r(0,CombinedIdx(s,M1_FY_IDX,rad.nvars),0,0,i) == 0.0 &&
            final_r(0,CombinedIdx(s,M1_FZ_IDX,rad.nvars),0,0,i) == 0.0;
        if (c.feedback && stage == 2 && !c.masked) {
          de += ef - eb; df += ff - f0;
          if (s < 2) dn[s] = nf - nb;
        }
      }
      pass = pass && final_u(0,IEN,0,0,i) == c.tau - de &&
          final_u(0,IM1,0,0,i) == 1.0 / 4.0 - df;
      const Real q = 1.0 / 2.0 + mb * (dn[1] - dn[0]);
      // Only EOS mb and host/device floating-point contraction require rounding
      // allowance. Radiation states, theta and canceling charge are exact.
      const Real qtol = 16.0 * std::numeric_limits<Real>::epsilon() *
          (1.0 / 2.0 + std::abs(mb) * (std::abs(dn[0]) + std::abs(dn[1])));
      pass = pass && std::isfinite(final_u(0,IYF,0,0,i)) &&
          (dn[0] == dn[1] ? final_u(0,IYF,0,0,i) == q :
           std::abs(final_u(0,IYF,0,0,i) - q) <= qtol);
      for (int v = 0; v < static_cast<int>(u.extent(1)); ++v)
        if (v != IEN && v != IM1 && v != IYF)
          pass = pass && final_u(0,v,0,0,i) == u(0,v,0,0,i);
      for (int v = 0; v < static_cast<int>(w.extent(1)); ++v)
        pass = pass && final_w(0,v,0,0,i) == w(0,v,0,0,i);
    }
    std::cout << (pass ? "PASS " : "FAIL ") << case_names[test]
              << " stage=" << stage << " NGHOST=" << mesh->mb_indcs.ng
              << " execution_space=" << DevExeSpace::name() << '\n';
    return pass;
  }
};
}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) return 2;  // stock deck and reduced NormalLogs CompOSE table
  global_variable::my_rank = 0; global_variable::nranks = 1;
  Kokkos::initialize();
  int failed = 0;
  try {
    for (int ng : {2, 3, 4}) {
      Fixture fixture(argv[1], argv[2], ng);
      for (int stage : {1, 2})
        for (int test = 0; test < static_cast<int>(sizeof(case_names)/sizeof(case_names[0])); ++test)
          failed += !fixture.Run(test, stage);
    }
  } catch (const std::exception &error) {
    std::cerr << "FAIL native M1 limiter fixture: " << error.what() << '\n';
    ++failed;
  }
  Kokkos::finalize();
  return failed ? 1 : 0;
}
