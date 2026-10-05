// Focused WEAKRATES device and real-wrapper regression. No pgen/Driver loop.
// WEAKRATES_LEGACY_CAPTURE compiles against the heavy-fixed PRE-THERMAL epoch:
// every fixture used for off-path records uses only pre-thermal interfaces.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "radiation_m1/radiation_m1.hpp"
#include "radiation_m1/radiation_m1_calc_closure.hpp"
#include "radiation_m1/radiation_m1_weakrates.hpp"
#ifndef WEAKRATES_LEGACY_CAPTURE
#include "radiation_m1/radiation_m1_weakrates_thermal.hpp"
#endif

namespace {
using namespace radiationm1;  // NOLINT(build/namespaces)
namespace wr = radiationm1::weakrates;
using Records = std::map<std::string, std::vector<Real>>;
int failures = 0;

void Check(bool pass, const std::string &name) {
  std::cout << (pass ? "PASS " : "FAIL ") << name << '\n';
  failures += !pass;
}

KOKKOS_INLINE_FUNCTION bool CoefficientClose(Real actual, Real expected) {
  return Kokkos::isfinite(actual) && Kokkos::isfinite(expected) &&
      (expected == 0.0 ? actual == 0.0 :
       Kokkos::fabs(actual - expected) <= 1.e-12*Kokkos::fabs(expected));
}

KOKKOS_INLINE_FUNCTION bool ResidualClose(Real residual, Real endpoint,
                                         Real baseline,
                                         const RadiationM1Params &p) {
  return Kokkos::isfinite(residual) && Kokkos::fabs(residual) <=
      p.source_epsabs + p.source_epsrel*
      Kokkos::fmax(Kokkos::fabs(endpoint), Kokkos::fabs(baseline));
}

// Explicit allocations are mandatory: create_mirror_view may alias on Serial.
template<class View> auto Snapshot(const View &view) {
  auto host = Kokkos::create_mirror(view);
  Kokkos::deep_copy(host, view);
  return host;
}

template<class View> std::vector<Real> Values(const View &view) {
  auto host = Snapshot(view);
  return std::vector<Real>(host.data(), host.data() + host.size());
}

bool SameBits(const std::vector<Real> &a, const std::vector<Real> &b) {
  return a.size() == b.size() &&
      std::memcmp(a.data(), b.data(), a.size()*sizeof(Real)) == 0;
}

void Save(Records &records, const std::string &key, const std::vector<Real> &v) {
  if (v.empty() || !records.emplace(key, v).second)
    throw std::runtime_error("empty/duplicate fixture record " + key);
}

void WriteRecords(const Records &records, const std::string &path) {
  static_assert(sizeof(Real) == sizeof(std::uint64_t), "double precision required");
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot open record output");
  for (const auto &entry : records) {
    out << entry.first;
    for (Real value : entry.second) {
      std::uint64_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      out << ' ' << std::hex << std::setfill('0') << std::setw(16) << bits;
    }
    out << '\n';
  }
  if (!out) throw std::runtime_error("cannot write record output");
}

void HeavyBoundary() {
  // All test arithmetic including independent quotients executes on DevExeSpace.
  constexpr int count = 22;
  DvceArray1D<int> result("heavy_boundary_result", count);
  Kokkos::parallel_for("weakrates_heavy_boundary",
      Kokkos::RangePolicy<DevExeSpace>(0, count), KOKKOS_LAMBDA(int c) {
    Real bn[4] = {3, 5, 2, 2}, be[4] = {7, 11, 8, 8};
    Real en[4]{}, ee[4]{};
    Real an[4] = {2, 3, 0, 0}, ae[4] = {4, 5, 0, 0};
    const Real total_raw_number = 8., total_raw_energy = 16.;
    const Real correction[2] = {.5,2.};
    for (int s = 2; s < 4; ++s) {
      en[s] = .5*total_raw_number*correction[s-2];
      ee[s] = .5*total_raw_energy*correction[s-2];
      an[s] = en[s]/2.; ae[s] = ee[s]/8.;
    }
    Real scat[4] = {1, 2, 3, 4};
    bool incoming = true;
    if (c < 3) {
      const Real ratio = c == 0 ? .5 : (c == 1 ? 1. : 2.);
      for (int s = 2; s < 4; ++s) {
        bn[s] *= ratio*ratio*ratio;
        be[s] *= ratio*ratio*ratio*ratio;
      }
    } else if (c < 11) {
      const int s = 2 + (c - 3)/4;
      const int arm = (c - 3)%4;
      if (arm == 0) bn[s] = s == 2 ? 0. : -0.;
      if (arm == 1) be[s] = s == 2 ? -0. : 0.;
      if (arm == 2) en[s] = ee[s] = 0.;
      if (arm == 3) { en[s] = ee[s] = 0.; bn[s] = be[s] = 0.; }
    } else if (c < 15) {
      const int s = 2 + (c - 11)/2;
      if ((c - 11)%2 == 0) {
        en[s] = std::numeric_limits<Real>::max();
        bn[s] = std::numeric_limits<Real>::min();
      } else {
        ee[s] = std::numeric_limits<Real>::max();
        be[s] = std::numeric_limits<Real>::min();
      }
    } else if (c < 19) {
      const int s = 2 + (c - 15)/2;
      if ((c - 15)%2 == 0) {
        en[s] = std::numeric_limits<Real>::min();
        bn[s] = std::numeric_limits<Real>::max();
      } else {
        ee[s] = std::numeric_limits<Real>::min();
        be[s] = std::numeric_limits<Real>::max();
      }
    } else if (c == 19) {
      incoming = false;
    } else if (c == 20) {
      scat[3] = std::numeric_limits<Real>::infinity();
    } else {
      // Prior overflow must not be rescued by finite final target absorption.
      incoming = false;
      an[2] = std::numeric_limits<Real>::infinity();
    }
    Real saved_n[4], saved_e[4], saved_sc[4], expect_n[4], expect_e[4];
    for (int s = 0; s < 4; ++s) {
      saved_n[s] = en[s]; saved_e[s] = ee[s]; saved_sc[s] = scat[s];
      expect_n[s] = s < 2 ? an[s]*bn[s] : (bn[s] > 0. ? en[s]/bn[s] : 0.);
      expect_e[s] = s < 2 ? ae[s]*be[s] : (be[s] > 0. ? ee[s]/be[s] : 0.);
    }
    const bool finite = wr::FinalizeKirchhoffCoefficients(
        4, incoming, bn, be, en, ee, an, ae, scat);
    bool pass = finite == (c < 11 || (c >= 15 && c < 19));
    for (int s = 0; s < 4; ++s) {
      pass = pass && scat[s] == saved_sc[s];
      if (s < 2) {
        pass = pass && en[s] == expect_n[s] && ee[s] == expect_e[s] &&
               an[s] == (s == 0 ? 2. : 3.) && ae[s] == (s == 0 ? 4. : 5.);
      } else {
        pass = pass && en[s] == saved_n[s] && ee[s] == saved_e[s];
        pass = pass && (Kokkos::isfinite(expect_n[s]) ? an[s] == expect_n[s] :
                       !Kokkos::isfinite(an[s]));
        pass = pass && (Kokkos::isfinite(expect_e[s]) ? ae[s] == expect_e[s] :
                       !Kokkos::isfinite(ae[s]));
        if (c < 3) pass = pass && ee[s]/en[s] == saved_e[s]/saved_n[s];
      }
    }
    result(c) = pass;
  });
  const auto host = Snapshot(result);
  for (int c = 0; c < count; ++c)
    Check(host(c), "heavy finalizer boundary " + std::to_string(c));
}

// All-cell device initialization/reference calculation is independent of the
// new finalizer/builder. EOS tables are never evaluated on the host.
template<class Logs> class Fixture {
 public:
  ParameterInput pin;
  std::unique_ptr<Mesh> mesh;
  dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Logs>, Primitive::ResetFloor> *hyd;
  DvceArray5D<Real> reference;
  Real mb;
  int option;

  Fixture(const std::string &deck, const std::string &table, int ng,
          int thermal_option, bool predictor, const std::string &backend = "weakrates")
      : option(thermal_option) {
    std::ifstream stream(deck);
    if (!stream) throw std::runtime_error("missing deck");
    pin.LoadFromStream(stream);
    pin.SetString("radiation_m1", "opacity_type", backend);
    pin.SetBoolean("coord", "minkowski", true);
    pin.SetString("mhd", "units", "geometric_solar");
    pin.SetString("mhd", "table", table);
    pin.SetBoolean("mhd", "use_NQT", std::is_same<Logs, Primitive::NQTLogs>::value);
    pin.SetInteger("mesh", "nx1", 8);
    pin.SetInteger("meshblock", "nx1", 8);
    pin.SetInteger("mesh", "nghost", ng);
    pin.SetReal("weakrates", "rho_min_cgs", 0.);
    pin.SetReal("weakrates", "temp_min_mev", 0.);
    pin.SetReal("weakrates", "opacity_corr_fac_max", 1.75);
    pin.SetReal("weakrates", "peq_w_floor", 1.e-3);
    pin.SetReal("weakrates", "peq_dlnT_tol", 1.e-4);
    pin.SetReal("weakrates", "peq_dYe_tol", 1.e-4);
    pin.SetBoolean("weakrates", "use_partial_equilibrium", predictor);
#ifndef WEAKRATES_LEGACY_CAPTURE
    if (thermal_option >= 0)
      pin.SetBoolean("weakrates", "use_electron_thermal_emission", thermal_option != 0);
#endif
    mesh = std::make_unique<Mesh>(&pin);
    mesh->BuildTreeFromScratch(&pin);
    mesh->AddCoordinatesAndPhysics(&pin);
    mesh->dt = 0.;
    hyd = dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Logs>,
                                      Primitive::ResetFloor> *>(mesh->pmb_pack->pdyngr);
    if (!hyd) throw std::runtime_error("typed EOS fixture mismatch");
    auto &eos = hyd->eos.ps.GetEOSMutable();
    mb = eos.GetBaryonMass();
    auto &rad = Rad();
    reference = DvceArray5D<Real>("independent_rate_reference", rad.eta_0.extent(0),
                                 26, rad.eta_0.extent(2), rad.eta_0.extent(3),
                                 rad.eta_0.extent(4));
    Initialize();
  }

  RadiationM1 &Rad() { return *mesh->pmb_pack->pradm1; }

  void Initialize(Real metric_x = 1., Real lapse = 1., Real zx = 0.) {
    auto &rad = Rad();
    auto *pack = mesh->pmb_pack;
    auto adm = pack->padm->adm;
    auto w = pack->pmhd->w0;
    auto temp_cache = hyd->temperature;
    auto u = rad.u0;
    auto chi = rad.chi;
    auto ref = reference;
    auto &eos = hyd->eos.ps.GetEOSMutable();
    const auto cu = eos.GetCodeUnitSystem();
    const auto eu = eos.GetEOSUnitSystem();
    const auto cg = Primitive::MakeCGS();
    const Real density_conversion = cu.MassDensityConversion(cg);
    const wr::UnitConversions units{cu.LengthConversion(cg), cu.TimeConversion(cg),
        eu.NumberDensityConversion(cg), cu.EnergyDensityConversion(cg)};
    const Real baryon_cgs = mb*density_conversion/units.number_cgs_per_eos;
    const Real rho = pin.GetReal("problem", "rho");
    const Real ye_input = pin.GetReal("problem", "Y_e");
    const Real nb_lo = eos.GetMinimumDensity(), nb_hi = eos.GetMaximumDensity();
    const Real t_lo = eos.GetMinimumTemperature()*eu.TemperatureConversion(cu);
    const Real t_hi = eos.GetMaximumTemperature()*eu.TemperatureConversion(cu);
    const Real y_lo = eos.GetMinimumSpeciesFraction(0), y_hi = eos.GetMaximumSpeciesFraction(0);
    const Real baryon = mb;
    const int cells = u.extent(4), blocks = pack->nmb_thispack, nv = rad.nvars;
    const int species = rad.nspecies;
    Kokkos::deep_copy(w, 0.);
    Kokkos::deep_copy(u, 0.);
    Kokkos::deep_copy(chi, 1./3.);
    Kokkos::deep_copy(temp_cache, 0.);
    Kokkos::deep_copy(pack->pmhd->u0, 0.);
    Kokkos::deep_copy(rad.radiation_mask, false);
    Kokkos::deep_copy(reference, 0.);
    par_for("weakrates_fixture_reference", DevExeSpace(), 0, blocks - 1,
            0, 0, 0, 0, 0, cells - 1,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
      adm.alpha(m,k,j,i) = lapse;
      adm.psi4(m,k,j,i) = Kokkos::cbrt(metric_x);
      for (int a = 0; a < 3; ++a) {
        adm.beta_u(m,a,k,j,i) = 0.;
        for (int b = a; b < 3; ++b) {
          adm.g_dd(m,a,b,k,j,i) = a == b ? (a == 0 ? metric_x : 1.) : 0.;
          adm.vK_dd(m,a,b,k,j,i) = 0.;
        }
      }
      const Real nb = Kokkos::fmax(nb_lo, Kokkos::fmin(rho/baryon, nb_hi));
      const Real t = Kokkos::fmax(t_lo, Kokkos::fmin(10., t_hi));
      const Real y = Kokkos::fmax(y_lo, Kokkos::fmin(ye_input, y_hi));
      Real ya[3] = {y, 0., 0.};
      w(m,IDN,k,j,i) = rho;
      w(m,IYF,k,j,i) = y;
      w(m,IPR,k,j,i) = eos.GetPressure(nb, t, ya);
      w(m,IVX,k,j,i) = zx;
      temp_cache(m,0,k,j,i) = t;
      const Real mu_b = eos.GetBaryonChemicalPotential(nb,t,ya);
      const Real mu_q = eos.GetChargeChemicalPotential(nb,t,ya);
      const Real mu_le = eos.GetElectronLeptonChemicalPotential(nb,t,ya);
      const wr::State state = wr::MakeState(nb*units.number_cgs_per_eos*baryon_cgs,
          t,baryon_cgs,eos.GetNeutronFraction(nb,t,ya),eos.GetProtonFraction(nb,t,ya),
          eos.GetHeavyFraction(nb,t,ya),eos.GetMassNumber(nb,t,ya),
          eos.GetChargeNumber(nb,t,ya),mu_b,mu_b+mu_q,mu_le-mu_q);
      auto raw = wr::ComputeRawCoefficients(state);
      auto local = wr::ComputeEquilibriumDensities(state);
      auto target = wr::ComputeEquilibriumDensities(t,mu_le/t);
      wr::ConvertToCodeUnits(units,raw);
      wr::ConvertToCodeUnits(units,local);
      wr::ConvertToCodeUnits(units,target);
      const Real mean[4] = {1.,1.25,.75,1.5};
      for (int s = 0; s < 4; ++s) {
        const int slot = s < 2 ? s : wr::kNuX;
        const Real split = s < 2 ? 1. : .5;
        const Real bn = split*local.number[slot], be = split*local.energy[slot];
        const Real n = .25*bn, e = .25*be*mean[s];
        if (s < species) {
          u(m,CombinedIdx(s,M1_E_IDX,nv),k,j,i) = e;
          if (species > 1) u(m,CombinedIdx(s,M1_N_IDX,nv),k,j,i) = n;
          chi(m,s,k,j,i) = 1./3.;
        }
        Real ratio = (e/n)*(bn/be);
        ratio = Kokkos::fmax(1./1.75,Kokkos::fmin(ratio,1.75));
        const Real correction = ratio*ratio;
        const Real tn = split*target.number[slot], te = split*target.energy[slot];
        const Real en = s < 2 ? correction*raw.abs_0[slot]*tn : correction*split*raw.eta_0[slot];
        const Real ee = s < 2 ? correction*raw.abs_1[slot]*te : correction*split*raw.eta_1[slot];
        ref(m,s,k,j,i) = en;
        ref(m,4+s,k,j,i) = ee;
        ref(m,8+s,k,j,i) = s < 2 ? correction*raw.abs_0[slot] : en/tn;
        ref(m,12+s,k,j,i) = s < 2 ? correction*raw.abs_1[slot] : ee/te;
        ref(m,16+s,k,j,i) = correction*raw.scat_1[slot];
      }
      ref(m,20,k,j,i) = raw.eta_0_non_cc[wr::kNuE];
      ref(m,21,k,j,i) = raw.eta_1_non_cc[wr::kNuE];
      ref(m,22,k,j,i) = target.number[wr::kNuE];
      ref(m,23,k,j,i) = target.number[wr::kAntiNuE];
      ref(m,24,k,j,i) = target.energy[wr::kNuE];
      ref(m,25,k,j,i) = target.energy[wr::kAntiNuE];
    });
    Kokkos::deep_copy(rad.u1, rad.u0);
    Kokkos::deep_copy(rad.uflx.x1f, 0.);
    Kokkos::deep_copy(rad.uflx.x2f, 0.);
    Kokkos::deep_copy(rad.uflx.x3f, 0.);
    Kokkos::deep_copy(pack->pmhd->bcc0, 0.);
    Kokkos::deep_copy(pack->pmhd->b0.x1f, 0.);
    Kokkos::deep_copy(pack->pmhd->b0.x2f, 0.);
    Kokkos::deep_copy(pack->pmhd->b0.x3f, 0.);
    hyd->eos.PrimToCons(pack->pmhd->w0,pack->pmhd->bcc0,pack->pmhd->u0,
                        0,cells-1,0,0,0,0);
    Kokkos::fence();
  }

  void SeedCoefficients(Real value) {
    auto &r = Rad();
    Kokkos::deep_copy(r.eta_0,value); Kokkos::deep_copy(r.eta_1,value);
    Kokkos::deep_copy(r.abs_0,value); Kokkos::deep_copy(r.abs_1,value);
    Kokkos::deep_copy(r.scat_1,value);
#ifndef WEAKRATES_LEGACY_CAPTURE
    if (r.weakrates_thermal.size()) Kokkos::deep_copy(r.weakrates_thermal,value);
#endif
  }

  std::vector<Real> Coefficients() {
    auto &r = Rad();
    std::vector<Real> result;
    for (auto view : {r.eta_0,r.eta_1,r.abs_0,r.abs_1,r.scat_1}) {
      const auto values = Values(view);
      result.insert(result.end(),values.begin(),values.end());
    }
    return result;
  }

  void ZeroLegacy() {
    auto &r = Rad();
    for (auto view : {r.eta_0,r.eta_1,r.abs_0,r.abs_1,r.scat_1}) Kokkos::deep_copy(view,0.);
  }

  Real SourceDt() {
    DvceArray1D<Real> scale("real_rate_stage_scale",1);
    auto ref = reference;
    const int i = mesh->mb_indcs.is;
    Kokkos::parallel_for("weakrates_source_dt",Kokkos::RangePolicy<DevExeSpace>(0,1),
        KOKKOS_LAMBDA(int) {
      Real dt = std::numeric_limits<Real>::max();
      const Real nr = ref(0,20,0,0,i), er = ref(0,21,0,0,i);
      if (nr > 0.) for (int s = 0; s < 2; ++s) dt = Kokkos::fmin(dt,ref(0,22+s,0,0,i)/nr);
      if (er > 0.) for (int s = 0; s < 2; ++s) dt = Kokkos::fmin(dt,ref(0,24+s,0,0,i)/er);
      scale(0) = dt/64.;
    });
    const auto host = Snapshot(scale);
    if (!(std::isfinite(host(0)) && host(0) > 0. && host(0) < std::numeric_limits<Real>::max()/64.))
      throw std::runtime_error("real-rate fixture needs finite positive timestep");
    return host(0);
  }
};

#ifndef WEAKRATES_LEGACY_CAPTURE
// Caller-owned geometry tensors outlive both SrcParams records in each kernel.
struct Geometry {
  AthenaPointTensor<Real,TensorSymm::SYM2,4,2> gd{}, gu{};
  AthenaPointTensor<Real,TensorSymm::NONE,4,1> nd{}, nu{}, ud{}, uu{}, vd{}, vu{};
  AthenaPointTensor<Real,TensorSymm::NONE,4,2> spatial{}, projector{};
  Real alpha, volume, W;
  KOKKOS_INLINE_FUNCTION Geometry(Real mx = 1., Real alp = 1., Real z = 0.)
      : alpha(alp), volume(Kokkos::sqrt(mx)), W(Kokkos::sqrt(1.+mx*z*z)) {
    gd(0,0) = -alp*alp; gu(0,0) = -1./(alp*alp);
    for (int a = 1; a < 4; ++a) {
      gd(a,a) = a == 1 ? mx : 1.; gu(a,a) = 1./gd(a,a);
    }
    nd(0) = -alp; nu(0) = 1./alp;
    uu(0) = W/alp; uu(1) = z;
    vu(1) = z/W;
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        ud(a) += gd(a,b)*uu(b); vd(a) += gd(a,b)*vu(b);
      }
    }
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) {
      spatial(a,b) = (a == b) + nu(a)*nd(b);
      projector(a,b) = (a == b) + uu(a)*ud(b);
    }
  }
};

struct OracleState {
  Real gamma[2]{}, n[2]{}, J[2]{}, H[2][4]{}, lambda[2]{};
  Real moment_increment[2][4]{}, cc[2]{}, pair{};
};

// Independent reference: existing selected-closure calculation, followed by
// direct tensor contractions and the plan equations. Never calls the new
// thermal residual, rate evaluator, projector, builder or update helper.
KOKKOS_INLINE_FUNCTION bool Oracle(const Geometry &g, const Real moment[2][4],
    const Real number[2], const wr::ElectronThermalSourceInput &in,
    const Real eta_energy[2], const Real absorption[2], const Real scattering[2],
    Real dt, const RadiationM1Params &p, OracleState &out) {
  for (int s = 0; s < 2; ++s) {
    AthenaPointTensor<Real,TensorSymm::NONE,4,1> fd{};
    AthenaPointTensor<Real,TensorSymm::SYM2,4,2> pd{}, td{};
    for (int a = 1; a < 4; ++a) fd(a) = moment[s][a];
    Real chi = 1./3., xi = 0.;
    if (calc_closure(BrentFunctor{},g.gd,g.gu,g.nd,g.W,g.uu,g.vd,g.projector,
        moment[s][0],fd,chi,xi,pd,p,p.closure_type) != LinalgSuccess) return false;
    for (int a = 0; a < 4; ++a) for (int b = a; b < 4; ++b)
      td(a,b) = moment[s][0]*g.nd(a)*g.nd(b) + fd(a)*g.nd(b) +
                 fd(b)*g.nd(a) + pd(a,b);
    Real densitized_J = 0.;
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b)
      densitized_J += td(a,b)*g.uu(a)*g.uu(b);
    if (!(Kokkos::isfinite(densitized_J) && densitized_J > 0.)) return false;
    out.J[s] = densitized_J/g.volume;
    for (int a = 0; a < 4; ++a) {
      Real h = 0.;
      for (int b = 0; b < 4; ++b) for (int c = 0; c < 4; ++c)
        h -= g.projector(b,a)*g.uu(c)*td(b,c);
      out.H[s][a] = h/g.volume;
    }
    // Orthogonality/current safety convention, with a noncancelling norm for
    // this diagonal metric and x-directed fluid boost.
    out.H[s][0] = -g.uu(1)*out.H[s][1]/g.uu(0);
    const Real h2 = g.gu(1,1)*out.H[s][1]*out.H[s][1]/(g.W*g.W) +
        out.H[s][2]*out.H[s][2] + out.H[s][3]*out.H[s][3];
    if (!Kokkos::isfinite(h2) || h2 < 0. ||
        h2 > (1.-p.rad_eps)*out.J[s]*out.J[s]) return false;
    // fnu^0 = -g^{0a} T_ab u^b/J. These deliberately causal
    // oracle states need no safety projection; reject nonfinite setup instead.
    Real f0 = 0.;
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b)
      f0 -= g.gu(0,a)*td(a,b)*g.uu(b)/densitized_J;
    out.gamma[s] = g.alpha*f0;
    if (!(Kokkos::isfinite(out.gamma[s]) && out.gamma[s] > 0.)) return false;
    out.n[s] = number[s]/(g.volume*out.gamma[s]);
  }
  if (in.thermal.r > 0.) out.pair = dt*g.alpha*g.volume*in.thermal.r*
      (1.-(out.n[0]/in.thermal.number[0])*(out.n[1]/in.thermal.number[1]));
  for (int s = 0; s < 2; ++s) {
    out.lambda[s] = in.thermal.q > 0. ? (in.thermal.q/in.thermal.energy[s])*
        (out.n[1-s]/in.thermal.number[1-s]) : 0.;
    Real source[4]{};
    for (int a = 0; a < 4; ++a) source[a] =
        (g.volume*(eta_energy[s]+in.thermal.q) -
         (absorption[s]+out.lambda[s])*g.volume*out.J[s])*g.ud(a) -
        (absorption[s]+out.lambda[s]+scattering[s])*g.volume*out.H[s][a];
    for (int b = 0; b < 4; ++b) out.moment_increment[s][0] -= dt*g.alpha*g.nu(b)*source[b];
    for (int a = 1; a < 4; ++a) for (int b = 0; b < 4; ++b)
      out.moment_increment[s][a] += dt*g.alpha*g.spatial(b,a)*source[b];
    out.cc[s] = dt*g.alpha*(g.volume*in.eta_number_cc[s] -
        in.abs_number_cc[s]*number[s]/out.gamma[s]);
    if (!Kokkos::isfinite(out.cc[s]) || !Kokkos::isfinite(out.pair)) return false;
    for (int a = 0; a < 4; ++a)
      if (!Kokkos::isfinite(out.moment_increment[s][a])) return false;
  }
  return true;
}

void ThermalBuilderBoundary() {
  constexpr int count = 14;
  DvceArray1D<int> result("thermal_builder_result",count);
  Kokkos::parallel_for("thermal_builder_boundary",Kokkos::RangePolicy<DevExeSpace>(0,count),
      KOKKOS_LAMBDA(int c) {
    wr::RawCoefficients raw{};
    raw.eta_0_non_cc[0] = raw.eta_0_non_cc[1] = .5;
    raw.eta_1_non_cc[0] = raw.eta_1_non_cc[1] = 2.;
    Real bn[4] = {2.,4.,3.,5.}, be[4] = {7.,11.,13.,17.};
    if (c == 1) raw.eta_0_non_cc[1] = .75;
    if (c == 2) raw.eta_1_non_cc[1] = 3.;
    if (c == 3) raw.eta_0_non_cc[0] = raw.eta_0_non_cc[1] = -.5;
    if (c == 4) raw.eta_1_non_cc[0] = raw.eta_1_non_cc[1] = std::numeric_limits<Real>::infinity();
    if (c == 5) bn[0] = 0.;
    if (c == 6) bn[1] = -1.;
    if (c == 7) be[0] = 0.;
    if (c == 8) be[1] = std::numeric_limits<Real>::quiet_NaN();
    if (c >= 9) {
      raw.eta_0_non_cc[0] = raw.eta_0_non_cc[1] = 0.;
      raw.eta_1_non_cc[0] = raw.eta_1_non_cc[1] = 0.;
      bn[0] = bn[1] = be[0] = be[1] = 0.;
      if (c == 10) raw.eta_0_non_cc[0] = raw.eta_0_non_cc[1] = 1.;
      if (c == 11) { bn[0] = 2.; bn[1] = 4.; raw.eta_0_non_cc[0] = raw.eta_0_non_cc[1] = 1.; }
      if (c == 12) { bn[0] = 2.; bn[1] = 4.; raw.eta_1_non_cc[0] = raw.eta_1_non_cc[1] = 1.; }
      if (c == 13) bn[1] = -1.;
    }
    wr::ElectronThermalCoefficients out{91.,92.,{93.,94.},{95.,96.}};
    const bool ok = wr::BuildElectronThermalCoefficients(raw,bn,be,out);
    bool pass = ok == (c == 0 || c == 9 || c == 11);
    if (ok) pass = pass && out.r == raw.eta_0_non_cc[0] && out.q == raw.eta_1_non_cc[0] &&
        out.number[0] == bn[0] && out.number[1] == bn[1] && out.energy[0] == be[0] && out.energy[1] == be[1];
    else pass = pass && out.r == 91. && out.q == 92. && out.number[0] == 93. &&
        out.number[1] == 94. && out.energy[0] == 95. && out.energy[1] == 96.;
    result(c) = pass;
  });
  const auto host = Snapshot(result);
  for (int c = 0; c < count; ++c) Check(host(c),"thermal coefficient builder "+std::to_string(c));
}

void ThermalDeviceSources(RadiationM1Params params) {
  // Both routes, three closures, rest/boosted nonunit geometry. Positive setup
  // states are mandatory; no difficult manufactured case is skipped.
  constexpr int count = 44;
  DvceArray1D<int> result("coupled_source_result",count);
  Kokkos::parallel_for("thermal_device_sources",Kokkos::RangePolicy<DevExeSpace>(0,count),
      KOKKOS_LAMBDA(int c) {
    auto p = params;
    p.src_update = c%2 ? Explicit : Implicit;
    p.closure_type = Eddington;
    p.theta_limiter = false;
    wr::ElectronThermalSourceInput in{};
    in.volume = 1.;
    in.thermal = {.5,0.,{2.,2.},{4.,8.}};
    Real dt = 2.;
    Real baseline[2][4] = {{2.,0.,0.,0.},{4.,0.,0.,0.}};
    Real energy_cc[2]{}, energy_abs[2]{}, scattering[2]{};
    Real expected_moment[2][4]{};
    Real expected_number[2] = {.5,.5};
    in.number_star[0] = in.number_star[1] = .5;
    bool boost = false, manufactured = false, equilibrium = false;
    if (c < 4) {
      const Real old = c < 2 ? .5 : 8.;
      in.number_star[0] = in.number_star[1] = old;
      const Real root = 2.*(old+dt*in.thermal.r)/
          (1.+Kokkos::sqrt(1.+4.*(dt*in.thermal.r/4.)*(old+dt*in.thermal.r)));
      expected_number[0] = expected_number[1] = p.src_update == Implicit ? root :
          old + dt*in.thermal.r*(1.-(old/2.)*(old/2.));
    } else if (c < 8) {
      in.thermal.r = 0.; in.thermal.q = .5;
      in.number_star[0] = in.number_star[1] = 2.;
      expected_number[0] = expected_number[1] = 2.;
    } else if (c < 12) {
      in.thermal.r = in.thermal.q = 0.;
      dt = c >= 10 ? 0. : dt;
      if (c >= 10) in.thermal.r = in.thermal.q = .5;
      expected_number[0] = expected_number[1] = .5;
    } else if (c < 20) {
      // Empty partner (each index), both empty, and unequal dilute populations.
      in.thermal.q = .5; dt = .125;
      if (c < 14) { in.number_star[0] = 0.; in.number_star[1] = 1.; }
      else if (c < 16) { in.number_star[0] = 1.; in.number_star[1] = 0.; }
      else if (c < 18) { in.number_star[0] = in.number_star[1] = 0.; }
      else { in.number_star[0] = .25; in.number_star[1] = 1.5; }
    } else if (c < 32) {
      manufactured = true;
      boost = (c-20)/6 != 0;
      const int closure = ((c-20)%6)/2;
      p.closure_type = closure == 0 ? Eddington : (closure == 1 ? Minerbo : Kershaw);
      dt = .125;
      in.thermal = {.25,.5,{2.,4.},{5.,7.}};
      baseline[0][0] = 2.; baseline[1][0] = 3.;
      baseline[0][1] = .3; baseline[1][1] = -.2;
      baseline[0][2] = .1; baseline[1][2] = .15;
      in.number_star[0] = 2.; in.number_star[1] = 3.;
      for (int s = 0; s < 2; ++s) {
        in.eta_number_cc[s] = s == 0 ? .125 : .25;
        in.abs_number_cc[s] = s == 0 ? .125 : .25;
        energy_cc[s] = s == 0 ? .125 : .25;
        energy_abs[s] = s == 0 ? .25 : .125;
        scattering[s] = s == 0 ? .25 : .5;
      }
    } else {
      equilibrium = true;
      p.source_maxiter = 0;  // Initial physical convergence precedes the budget.
      boost = c >= 38;
      p.closure_type = ((c-32)%6)/2 == 0 ? Eddington :
          (((c-32)%6)/2 == 1 ? Minerbo : Kershaw);
      in.thermal = {.5,.5,{2.,4.},{2.,4.}};
      dt = .5;
    }
    Geometry g(boost ? 4. : 1.,boost ? .75 : 1.,boost ? .25 : 0.);
    in.volume = g.volume;
    if (equilibrium) {
      for (int s = 0; s < 2; ++s) {
        const Real j = in.thermal.energy[s]*g.volume;
        AthenaPointTensor<Real,TensorSymm::SYM2,4,2> tensor{};
        for (int a = 0; a < 4; ++a) for (int b = a; b < 4; ++b)
          tensor(a,b) = j*g.ud(a)*g.ud(b)+(j/3.)*(g.gd(a,b)+g.ud(a)*g.ud(b));
        baseline[s][0] = 0.;
        for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b)
          baseline[s][0] += tensor(a,b)*g.nu(a)*g.nu(b);
        for (int a = 1; a < 4; ++a) {
          baseline[s][a] = 0.;
          for (int b = 0; b < 4; ++b) for (int d = 0; d < 4; ++d)
            baseline[s][a] -= g.spatial(b,a)*g.nu(d)*tensor(b,d);
        }
        in.number_star[s] = g.volume*g.W*in.thermal.number[s];
        in.abs_number_cc[s] = s == 0 ? .125 : .25;
        in.eta_number_cc[s] = in.abs_number_cc[s]*in.thermal.number[s];
        energy_abs[s] = s == 0 ? .25 : .125;
        energy_cc[s] = energy_abs[s]*in.thermal.energy[s];
        expected_number[s] = in.number_star[s];
      }
    }
    for (int s = 0; s < 2; ++s) for (int a = 0; a < 4; ++a)
      expected_moment[s][a] = baseline[s][a];
    if (c >= 4 && c < 8) for (int s = 0; s < 2; ++s) {
      const Real old = baseline[s][0], be = in.thermal.energy[s];
      expected_moment[s][0] = p.src_update == Implicit ?
          (old+dt*in.thermal.q)/(1.+dt*in.thermal.q/be) : old+dt*in.thermal.q*(1.-old/be);
    }
    bool pass = true;
    if (manufactured) {
      OracleState prescribed{};
      pass = Oracle(g,baseline,in.number_star,in,energy_cc,energy_abs,scattering,dt,p,prescribed);
      if (boost) pass = pass && prescribed.gamma[0] != prescribed.gamma[1];
      for (int s = 0; s < 2; ++s) {
        expected_number[s] = in.number_star[s];
        in.number_star[s] -= prescribed.cc[s]+prescribed.pair;
        pass = pass && in.number_star[s] > 0.;
        for (int a = 0; a < 4; ++a) baseline[s][a] -= prescribed.moment_increment[s][a];
        pass = pass && baseline[s][0] > 0.;
      }
      // Explicit evaluation remains at the prescribed endpoint, while repaired
      // baselines are source-subtracted; distinguishes u0 from source_base.
      for (int s = 0; s < 2; ++s) {
        in.explicit_number[s] = expected_number[s];
        for (int a = 0; a < 4; ++a) in.explicit_moment[s][a] = expected_moment[s][a];
      }
    } else for (int s = 0; s < 2; ++s) {
      in.explicit_number[s] = in.number_star[s];
      for (int a = 0; a < 4; ++a) in.explicit_moment[s][a] = baseline[s][a];
    }
    if (dt == 0.) for (int s = 0; s < 2; ++s) {
      in.explicit_number[s] = std::numeric_limits<Real>::quiet_NaN();
      for (int a = 0; a < 4; ++a) in.explicit_moment[s][a] = std::numeric_limits<Real>::quiet_NaN();
    }
    AthenaPointTensor<Real,TensorSymm::NONE,4,1> fs[2]{};
    for (int s = 0; s < 2; ++s) for (int a = 1; a < 4; ++a) fs[s](a) = baseline[s][a];
    SrcParams electron[2] = {
      SrcParams(dt,g.alpha,g.gd,g.gu,g.nd,g.nu,g.spatial,g.ud,g.uu,g.vd,g.vu,g.projector,
          g.W,baseline[0][0],fs[0],1./3.,g.volume*energy_cc[0],energy_abs[0],scattering[0]),
      SrcParams(dt,g.alpha,g.gd,g.gu,g.nd,g.nu,g.spatial,g.ud,g.uu,g.vd,g.vu,g.projector,
          g.W,baseline[1][0],fs[1],1./3.,g.volume*energy_cc[1],energy_abs[1],scattering[1])};
    wr::ElectronThermalSourceResult out{};
    const bool ok = wr::UpdateElectronThermalSources(BrentFunctor{},electron,in,p,out);
    pass = pass && ok && out.failure == wr::ElectronThermalFailure::None &&
        out.status == (p.src_update == Implicit ? SrcOk : SrcThin);
    if (ok) {
      Real state_moment[2][4], state_number[2];
      for (int s = 0; s < 2; ++s) {
        state_number[s] = p.src_update == Implicit || dt == 0. ? out.number[s] : in.explicit_number[s];
        for (int a = 0; a < 4; ++a) state_moment[s][a] = p.src_update == Implicit || dt == 0. ? out.moment[s][a] : in.explicit_moment[s][a];
      }
      OracleState oracle{};
      pass = pass && Oracle(g,state_moment,state_number,in,energy_cc,energy_abs,scattering,dt,p,oracle);
      for (int s = 0; s < 2; ++s) {
        pass = pass && ResidualClose(out.number[s]-in.number_star[s]-oracle.cc[s]-oracle.pair,
                                    out.number[s],in.number_star[s],p);
        pass = pass && ResidualClose(out.cc_number_increment[s]-oracle.cc[s],out.number[s],in.number_star[s],p);
        for (int a = 0; a < 4; ++a) pass = pass && ResidualClose(
            out.moment[s][a]-baseline[s][a]-oracle.moment_increment[s][a],out.moment[s][a],baseline[s][a],p);
        if (c < 12 || manufactured || equilibrium) {
          pass = pass && ResidualClose(out.number[s]-expected_number[s],out.number[s],in.number_star[s],p);
          for (int a = 0; a < 4; ++a) pass = pass && ResidualClose(out.moment[s][a]-expected_moment[s][a],out.moment[s][a],baseline[s][a],p);
        }
        if (c >= 8 && c < 12) {
          pass = pass && out.number[s] == in.number_star[s] && out.cc_number_increment[s] == 0.;
          for (int a = 0; a < 4; ++a) pass = pass && out.moment[s][a] == baseline[s][a];
        }
        if (equilibrium && !boost && p.src_update == Explicit) {
          pass = pass && out.number[s] == in.number_star[s] && out.pair_number_increment == 0.;
          for (int a = 0; a < 4; ++a) pass = pass && out.moment[s][a] == baseline[s][a];
        }
      }
      pass = pass && ResidualClose(out.pair_number_increment-oracle.pair,
          out.pair_number_increment,Kokkos::fmax(in.number_star[0],in.number_star[1]),p);
    }
    result(c) = pass;
  });
  const auto host = Snapshot(result);
  for (int c = 0; c < count; ++c) Check(host(c),"coupled device oracle "+std::to_string(c));
}
#endif  // !WEAKRATES_LEGACY_CAPTURE

#ifndef WEAKRATES_LEGACY_CAPTURE
void ThermalNumericalBoundaries(RadiationM1Params params) {
  DvceArray1D<int> result("thermal_numerical_invariants",4);
  Kokkos::parallel_for("thermal_numerical_boundaries",Kokkos::RangePolicy<DevExeSpace>(0,4),
      KOKKOS_LAMBDA(int c) {
    bool pass = true;
    if (c == 3) {
      // A cyclic row permutation forces pivoting. The independently prescribed
      // binary solution detects both row-swap and back-substitution errors.
      Real matrix[9][9]{}, rhs[9]{}, update[9]{};
      for (int i = 0; i < 9; ++i) {
        const int j = (i+1)%9;
        matrix[i][j] = i%2 ? 2. : 4.;
        rhs[i] = matrix[i][j]*(j+1.);
      }
      pass = wr::SolveElectronThermalLinearSystem(matrix,rhs,update) == wr::ElectronThermalFailure::None;
      for (int j = 0; j < 9; ++j) pass = pass && update[j] == j+1.;
    } else {
      auto p = params; p.src_update = Implicit; p.closure_type = Eddington;
      Geometry g;
      AthenaPointTensor<Real,TensorSymm::NONE,4,1> fs[2]{};
      wr::ElectronThermalSourceInput in{};
      in.volume = 1.; in.thermal = {.5,.5,{2.,4.},{4.,8.}};
      in.number_star[0] = 1.; in.number_star[1] = 3.;
      Real x[9] = {2.,0.,0.,0.,4.,0.,0.,0.,.25};
      if (c == 2) {
        in.thermal.r = in.thermal.q = 0.;
        x[1] = x[0]; x[5] = x[4]; x[8] = 0.;
        fs[0](1) = x[1]; fs[1](1) = x[5];
      }
      SrcParams electron[2] = {
        SrcParams(.5,1.,g.gd,g.gu,g.nd,g.nu,g.spatial,g.ud,g.uu,g.vd,g.vu,g.projector,g.W,2.,fs[0],1./3.,0.,0.,0.),
        SrcParams(.5,1.,g.gd,g.gu,g.nd,g.nu,g.spatial,g.ud,g.uu,g.vd,g.vu,g.projector,g.W,4.,fs[1],1./3.,0.,0.,0.)};
      Real residual[9]{}, repeated[9]{}, jacobian[9][9]{};
      wr::ElectronThermalResidualState state{}, other{};
      pass = wr::EvaluateElectronThermalResidual(BrentFunctor{},electron,in,p,Eddington,x,residual,state) == wr::ElectronThermalFailure::None;
      Real columns[9] = {2.,2.,2.,2.,4.,4.,4.,4.,3.};
      pass = pass && wr::BuildElectronThermalJacobian(BrentFunctor{},electron,in,p,Eddington,x,columns,jacobian) == wr::ElectronThermalFailure::None;
      // Every perturbation restores the immutable CC and closure seeds. Calling
      // the base again must be bitwise repeatable, not cumulatively augmented.
      pass = pass && wr::EvaluateElectronThermalResidual(BrentFunctor{},electron,in,p,Eddington,x,repeated,other) == wr::ElectronThermalFailure::None;
      for (int i = 0; i < 9; ++i) pass = pass && repeated[i] == residual[i];
      for (int s = 0; s < 2; ++s) pass = pass && electron[s].eta == 0. && electron[s].kabs == 0. && electron[s].kscat == 0. && electron[s].chi == 1./3.;
      if (c == 0) {
        // Direct rest-frame residual and eliminated-number oracle, no production
        // residual/helper used to construct these expected values.
        const Real n[2] = {1.25,3.25};
        const Real pair = .5*.5*(1.-(n[0]/2.)*(n[1]/4.));
        for (int s = 0; s < 2; ++s) {
          const Real energy = x[4*s];
          const Real be = s == 0 ? 4. : 8.;
          const Real bp = s == 0 ? 4. : 2.;
          const Real lambda = (.5/be)*(n[1-s]/bp);
          pass = pass && state.number[s] == n[s] && state.gamma[s] == 1.;
          pass = pass && CoefficientClose(residual[4*s],-.5*(.5-lambda*energy));
          for (int a = 1; a < 4; ++a) pass = pass && residual[4*s+a] == 0.;
          pass = pass && state.number_residual[s] == 0.;
        }
        pass = pass && CoefficientClose(residual[8],x[8]-pair);
      } else {
        // The full rest-frame coupled Jacobian has partner dependence in its
        // shared xi column; a frozen independent-species Jacobian fails here.
        for (int i = 0; i < 9; ++i) for (int j = 0; j < 9; ++j) {
          Real expected = 0.;
          if (i == j) {
            if (c == 2 || i == 8) expected = 1.;
            else {
              const int s = i/4;
              const Real be = s == 0 ? 4. : 8.;
              const Real bp = s == 0 ? 4. : 2.;
              const Real partner = s == 0 ? 3.25 : 1.25;
              expected = 1.+.5*(.5/be)*(partner/bp);
            }
          }
          if (c == 1 && j == 8) {
            if (i == 0) expected = .5*(.5/4.)*2./4.;
            if (i == 4) expected = .5*(.5/8.)*4./2.;
            if (i == 8) expected = 1.+(.5*.5/8.)*(1.25+3.25);
          }
          pass = pass && CoefficientClose(jacobian[i][j],expected);
        }
      }
    }
    result(c) = pass;
  });
  const auto host = Snapshot(result);
  for (int c = 0; c < 4; ++c) Check(host(c),"production residual/Jacobian/pivot invariants "+std::to_string(c));
}

void ThermalFailureBoundary(RadiationM1Params params) {
  constexpr int count = 21;
  DvceArray1D<int> result("thermal_failures",count);
  Kokkos::parallel_for("thermal_failure_boundary",Kokkos::RangePolicy<DevExeSpace>(0,count),
      KOKKOS_LAMBDA(int c) {
    auto p = params;
    p.src_update = Implicit; p.closure_type = Eddington;
    Geometry g(1.,c == 17 ? 0. : 1.,0.);
    Real dt = c == 2 ? -1. : 1.;
    AthenaPointTensor<Real,TensorSymm::NONE,4,1> fs[2]{};
    if (c == 5) fs[0](1) = 3.;
    SrcParams electron[2] = {
      SrcParams(dt,g.alpha,g.gd,g.gu,g.nd,g.nu,g.spatial,g.ud,g.uu,g.vd,g.vu,g.projector,g.W,2.,fs[0],1./3.,0.,0.,0.),
      SrcParams(dt,g.alpha,g.gd,g.gu,g.nd,g.nu,g.spatial,g.ud,g.uu,g.vd,g.vu,g.projector,g.W,4.,fs[1],1./3.,0.,0.,0.)};
    wr::ElectronThermalSourceInput in{};
    in.volume = 1.; in.thermal = {.5,0.,{2.,2.},{4.,8.}};
    in.number_star[0] = in.number_star[1] = 1.;
    for (int s = 0; s < 2; ++s) { in.explicit_number[s] = 1.; in.explicit_moment[s][0] = s == 0 ? 2. : 4.; }
    auto expected = wr::ElectronThermalFailure::InvalidInput;
    if (c == 0) in.volume = 0.;
    if (c == 1) in.thermal.r = std::numeric_limits<Real>::quiet_NaN();
    if (c == 3) { in.thermal.number[1] = 0.; expected = wr::ElectronThermalFailure::InvalidReference; }
    if (c == 4) { in.thermal.q = 1.; in.thermal.energy[0] = 0.; expected = wr::ElectronThermalFailure::InvalidReference; }
    if (c == 5) expected = wr::ElectronThermalFailure::InvalidTrial;
    if (c == 6) { in.thermal.q = std::numeric_limits<Real>::max(); in.thermal.energy[0] = std::numeric_limits<Real>::min(); expected = wr::ElectronThermalFailure::NonfiniteSource; }
    if (c == 7) { p.source_maxiter = 0; expected = wr::ElectronThermalFailure::MaxIterations; }
    if (c == 8) { p.source_maxiter = 0; p.closure_type = Minerbo; expected = wr::ElectronThermalFailure::ClosureRetry; }
    if (c == 9) { p.src_update = Explicit; in.explicit_number[0] = -1.; expected = wr::ElectronThermalFailure::InvalidTrial; }
    if (c == 18) in.thermal.r = -.5;
    if (c == 19) { in.thermal.energy[1] = std::numeric_limits<Real>::quiet_NaN(); expected = wr::ElectronThermalFailure::InvalidReference; }
    if (c == 20) { p.src_update = Explicit; in.explicit_moment[0][1] = std::numeric_limits<Real>::infinity(); expected = wr::ElectronThermalFailure::InvalidTrial; }
    bool pass = true;
    if (c < 10 || c >= 17) {
      wr::ElectronThermalSourceResult out{};
      const bool ok = wr::UpdateElectronThermalSources(BrentFunctor{},electron,in,p,out);
      pass = !ok && out.status == SrcFail && out.failure == expected;
    } else if (c < 13) {
      Real matrix[9][9]{}, rhs[9]{}, update[9]{};
      if (c == 11) matrix[0][0] = std::numeric_limits<Real>::infinity();
      if (c == 12) rhs[3] = std::numeric_limits<Real>::quiet_NaN();
      pass = wr::SolveElectronThermalLinearSystem(matrix,rhs,update) == wr::ElectronThermalFailure::SingularJacobian;
    } else if (c < 15) {
      Real x[9] = {2.,0.,0.,0.,4.,0.,0.,0.,0.};
      Real scale[9] = {2.,2.,2.,2.,4.,4.,4.,4.,1.}, jac[9][9]{};
      if (c == 13) scale[0] = 0.;
      else {
        // Valid zero-number/zero-rate base, but no representable pair-coordinate
        // perturbation: sqrt(epsilon)*minimum subnormal rounds to zero.
        scale[8] = std::numeric_limits<Real>::denorm_min();
        in.thermal.r = in.thermal.q = 0.;
        in.number_star[0] = in.number_star[1] = 0.;
      }
      pass = wr::BuildElectronThermalJacobian(BrentFunctor{},electron,in,p,Eddington,x,scale,jac) == wr::ElectronThermalFailure::FiniteDifference;
    } else if (c == 15) {
      // Strict physical convergence forbids calling a representationally zero
      // Newton update convergence. Row scaling underflows rhs, not physics.
      in.thermal.r = 1.e-300;
      p.source_epsabs = p.source_epsrel = 0.;
      Real x[9] = {2.,0.,0.,0.,4.,0.,0.,0.,0.};
      Real columns[9] = {2.,2.,2.,2.,4.,4.,4.,4.,1.};
      Real rows[9] = {2.,2.,2.,2.,4.,4.,4.,4.,std::numeric_limits<Real>::max()};
      wr::ElectronThermalResidualState state{};
      pass = wr::SolveElectronThermalNewton(BrentFunctor{},electron,in,p,Eddington,columns,rows,x,state) == wr::ElectronThermalFailure::Backtracking;
    } else {
      // 0.8 - 1 - (-0.2) is not a zero residual in double precision: an
      // eliminated number denominator alone must not certify convergence.
      in.thermal.r = in.thermal.q = 0.;
      in.abs_number_cc[0] = in.abs_number_cc[1] = .25;
      p.source_epsabs = p.source_epsrel = 0.;
      wr::ElectronThermalSourceResult out{};
      pass = !wr::UpdateElectronThermalSources(BrentFunctor{},electron,in,p,out) &&
          out.status == SrcFail && out.failure == wr::ElectronThermalFailure::NumberResidual;
    }
    result(c) = pass;
  });
  const auto host = Snapshot(result);
  for (int c = 0; c < count; ++c) Check(host(c),"named thermal failure "+std::to_string(c));
}
#endif

template<class Logs> void PublicationCases(Fixture<Logs> &f, const std::string &prefix,
                                           Records &records) {
  auto &rad = f.Rad();
#ifndef WEAKRATES_LEGACY_CAPTURE
  const bool enabled = f.option != 0;
  Check(enabled ? rad.weakrates_thermal.extent(1) == wr::kNumThermalFields &&
      rad.weakrates_thermal.extent(0) == rad.eta_0.extent(0) &&
      rad.weakrates_thermal.extent(4) == rad.eta_0.extent(4) : rad.weakrates_thermal.size() == 0,
      prefix+" construction-time allocation");
  Check(rad.weakrates_params.use_electron_thermal_emission == enabled,
        prefix+" default/explicit option value");
#endif
  const auto ind = f.mesh->mb_indcs;
  const int point = ind.is + 1;
  for (int scenario = 0; scenario < 5; ++scenario) {
    f.Initialize();
    f.SeedCoefficients(-77.);
    rad.weakrates_params.rho_min_cgs = 0.;
    rad.weakrates_params.temp_min_mev = 0.;
    if (scenario == 1) {
      auto mask = rad.radiation_mask;
      Kokkos::parallel_for("set_test_mask",Kokkos::RangePolicy<DevExeSpace>(0,1),
          KOKKOS_LAMBDA(int) { mask(0,0,0,point) = true; });
    }
    if (scenario == 2) rad.weakrates_params.rho_min_cgs = std::numeric_limits<Real>::max();
    if (scenario == 3) rad.weakrates_params.temp_min_mev = std::numeric_limits<Real>::max();
    if (scenario == 4) {
      auto temperature = f.hyd->temperature;
      Kokkos::parallel_for("set_invalid_temperature",Kokkos::RangePolicy<DevExeSpace>(0,1),
          KOKKOS_LAMBDA(int) { temperature(0,0,0,0,point) = std::numeric_limits<Real>::quiet_NaN(); });
    }
    Check(rad.CalcOpacityWeakRates(nullptr,1) == TaskStatus::complete,
          prefix+" producer wrapper status "+std::to_string(scenario));
    auto ref = Snapshot(f.reference);
    bool pass = true, setup = true;
    int field = 0;
    for (auto view : {rad.eta_0,rad.eta_1,rad.abs_0,rad.abs_1,rad.scat_1}) {
      auto host = Snapshot(view);
      for (int m = 0; m < f.mesh->pmb_pack->nmb_thispack; ++m)
        for (int i = 0; i < static_cast<int>(view.extent(4)); ++i)
          for (int s = 0; s < 4; ++s) {
            const bool evaluated = i >= ind.is-1 && i <= ind.ie+1;
            const bool zero = scenario == 2 || scenario == 3 ||
                ((scenario == 1 || scenario == 4) && i == point);
            const Real expected = !evaluated ? -77. : (zero ? 0. : ref(m,4*field+s,0,0,i));
            if (evaluated && !zero) setup = setup && std::isfinite(expected) && expected >= 0.;
            pass = pass && (!evaluated || zero ? host(m,s,0,0,i) == expected :
                            CoefficientClose(host(m,s,0,0,i),expected));
          }
      ++field;
    }
    for (int s = 0; s < 2; ++s) setup = setup && ref(0,22+s,0,0,point) > 0. && ref(0,24+s,0,0,point) > 0.;
    Check(setup,prefix+" positive finite rate/reference setup "+std::to_string(scenario));
    Check(pass,prefix+" five-field/domain/outer-ghost publication "+std::to_string(scenario));
#ifndef WEAKRATES_LEGACY_CAPTURE
    if (enabled) {
      auto host = Snapshot(rad.weakrates_thermal);
      bool thermal_pass = true;
      for (int m = 0; m < f.mesh->pmb_pack->nmb_thispack; ++m)
        for (int i = 0; i < static_cast<int>(host.extent(4)); ++i)
          for (int a = 0; a < wr::kNumThermalFields; ++a) {
            const bool evaluated = i >= ind.is-1 && i <= ind.ie+1;
            const bool zero = scenario == 2 || scenario == 3 ||
                ((scenario == 1 || scenario == 4) && i == point);
            const Real expected = !evaluated ? -77. : (zero ? 0. : ref(m,20+a,0,0,i));
            thermal_pass = thermal_pass && (!evaluated || zero ? host(m,a,0,0,i) == expected :
                                           CoefficientClose(host(m,a,0,0,i),expected));
          }
      Check(thermal_pass,prefix+" six uncorrected thermal fields/domain/return zeroing "+std::to_string(scenario));
      Save(records,prefix+"/thermal-"+std::to_string(scenario),Values(rad.weakrates_thermal));
    }
#endif
    Save(records,prefix+"/coeff-"+std::to_string(scenario),f.Coefficients());
  }
  // Later stages are a held-coefficient contract, including allocated outer ghosts.
  int sentinel = -101;
  for (auto view : {rad.eta_0,rad.eta_1,rad.abs_0,rad.abs_1,rad.scat_1}) Kokkos::deep_copy(view,static_cast<Real>(sentinel--));
#ifndef WEAKRATES_LEGACY_CAPTURE
  if (enabled) Kokkos::deep_copy(rad.weakrates_thermal,-109.);
  const auto thermal_before = enabled ? Values(rad.weakrates_thermal) : std::vector<Real>{};
#endif
  const auto before = f.Coefficients();
  Check(rad.CalcOpacityWeakRates(nullptr,2) == TaskStatus::complete && SameBits(before,f.Coefficients()),prefix+" stage-2 five-field hold");
#ifndef WEAKRATES_LEGACY_CAPTURE
  if (enabled) Check(SameBits(thermal_before,Values(rad.weakrates_thermal)),prefix+" stage-2 thermal hold");
#endif
  Save(records,prefix+"/stage2-coeff",f.Coefficients());
  rad.weakrates_params.rho_min_cgs = rad.weakrates_params.temp_min_mev = 0.;
}

template<class Logs> void ConsumerCases(Fixture<Logs> &f, const std::string &prefix,
                                        Records &records) {
  auto &rad = f.Rad();
  const auto original_params = rad.params;
  const int point = f.mesh->mb_indcs.is;
  for (int mode = 0; mode < 2; ++mode) for (int stage = 1; stage <= 2; ++stage)
      for (int scenario = 0; scenario < 12; ++scenario) {
    // Every stage starts from freshly initialized primitive/conserved/radiation
    // state, including the complete NGHOST stencil. No Driver/C2P is invoked.
    f.mesh->dt = 0.;
    f.Initialize(4.,.75,.25);
    rad.params = original_params;
    rad.params.src_update = mode == 0 ? Implicit : Explicit;
    rad.params.theta_limiter = scenario == 9 || scenario == 10;
    if (scenario == 9 || scenario == 10) rad.params.source_limiter = 1.e-8;
    if (scenario == 11) {
      rad.params.source_thin_limit = 1.e6;
      rad.params.source_thick_limit = 0.;
      rad.params.source_scat_limit = 0.;
      rad.params.source_therm_limit = 0.;
    }
    Check(rad.CalcOpacityWeakRates(nullptr,1) == TaskStatus::complete,
          prefix+" real producer before source");
    f.mesh->dt = f.SourceDt();
    if (scenario == 1 || scenario == 7 || scenario == 9 || scenario == 11) f.ZeroLegacy();
    if (scenario == 2) rad.params.matter_sources = false;
    if (scenario == 3) rad.params.source_limiter = 0.;
    if (scenario == 8) rad.params.backreact = false;
#ifndef WEAKRATES_LEGACY_CAPTURE
    const bool enabled = f.option != 0;
    if (enabled && scenario == 4) Kokkos::deep_copy(rad.weakrates_thermal,0.);
    // Poison storage behind deliberate no-read gates: NaN must not invoke the
    // thermal model when matter sources or all source acceptance are disabled.
    if (enabled && (scenario == 2 || scenario == 3))
      Kokkos::deep_copy(rad.weakrates_thermal,std::numeric_limits<Real>::quiet_NaN());
#endif
    if (scenario == 5) {
      auto mask = rad.radiation_mask;
      Kokkos::parallel_for("consumer_mask",Kokkos::RangePolicy<DevExeSpace>(0,1),
          KOKKOS_LAMBDA(int) { mask(0,0,0,point) = true; });
    }
    if (scenario == 6) {
#ifndef WEAKRATES_LEGACY_CAPTURE
      // Active-enabled counterpart is tested in a fatal native child. Keep this
      // successful fixture inactive so original global invalidity is observable.
      if (enabled) Kokkos::deep_copy(rad.weakrates_thermal,0.);
#endif
      auto eta = rad.eta_1;
      Kokkos::parallel_for("legacy_heavy_invalid",Kokkos::RangePolicy<DevExeSpace>(0,1),
          KOKKOS_LAMBDA(int) { eta(0,2,0,0,point) = std::numeric_limits<Real>::quiet_NaN(); });
    }
    if (scenario == 7) {
      auto u1 = rad.u1;
      const int nv = rad.nvars;
      const Real ef = rad.params.rad_E_floor, nf = rad.params.rad_N_floor;
      Kokkos::parallel_for("repair_incoming_baseline",Kokkos::RangePolicy<DevExeSpace>(0,1),
          KOKKOS_LAMBDA(int) {
        u1(0,CombinedIdx(0,M1_E_IDX,nv),0,0,point) = .5*ef;
        u1(0,CombinedIdx(0,M1_N_IDX,nv),0,0,point) = .5*nf;
      });
    }
    DvceArray5D<Real> before("independent_source_baseline",rad.u1.extent(0),
        rad.u1.extent(1),rad.u1.extent(2),rad.u1.extent(3),rad.u1.extent(4));
    DvceArray5D<Real> explicit_before("independent_explicit_state",rad.u0.extent(0),
        rad.u0.extent(1),rad.u0.extent(2),rad.u0.extent(3),rad.u0.extent(4));
    auto fluid = f.mesh->pmb_pack->pmhd->u0;
    DvceArray5D<Real> matter_before("independent_matter_state",fluid.extent(0),
        fluid.extent(1),fluid.extent(2),fluid.extent(3),fluid.extent(4));
    Kokkos::deep_copy(before,rad.u1);
    Kokkos::deep_copy(explicit_before,rad.u0);
    Kokkos::deep_copy(matter_before,fluid);
    const auto before_host = Values(fluid);
    Check(rad.TimeUpdate(nullptr,stage) == TaskStatus::complete,
          prefix+" direct TimeUpdate mode="+std::to_string(mode)+" stage="+std::to_string(stage)+" scenario="+std::to_string(scenario));
    const std::string key = prefix+"/source-"+std::to_string(mode)+"-"+std::to_string(stage)+"-"+std::to_string(scenario);
    Save(records,key+"-rad",Values(rad.u0));
    Save(records,key+"-matter",Values(fluid));
    Save(records,key+"-chi",Values(rad.chi));
    if (stage == 1 || scenario == 2 || scenario == 3 || scenario == 8)
      Check(SameBits(before_host,Values(fluid)),key+" stage/source/backreaction matter gate");
#ifndef WEAKRATES_LEGACY_CAPTURE
    const bool active = enabled && scenario != 2 && scenario != 3 && scenario != 4 && scenario != 6;
    if (active && scenario != 5) {
      DvceArray1D<int> outcome("actual_consumer_oracle",1);
      auto after = rad.u0;
      auto thermal = rad.weakrates_thermal;
      auto eta_n = rad.eta_0, abs_n = rad.abs_0;
      auto eta_e = rad.eta_1, abs_e = rad.abs_1, scat = rad.scat_1;
      const auto p = rad.params;
      const Real dt = f.mesh->dt*(stage == 1 ? .5 : 1.);
      const Real mb = f.mb;
      const int nv = rad.nvars;
      const bool pure = scenario == 1 || scenario == 7 || scenario == 9 || scenario == 11;
      Kokkos::parallel_for("actual_timeupdate_oracle",Kokkos::RangePolicy<DevExeSpace>(0,1),
          KOKKOS_LAMBDA(int) {
        Geometry g(4.,.75,.25);
        Real baseline[2][4]{}, current[2][4]{}, numbers[2]{}, old_number[2]{};
        wr::ElectronThermalSourceInput in{};
        in.volume = g.volume;
        in.thermal = {thermal(0,wr::kThermalNumberRate,0,0,point),
            thermal(0,wr::kThermalEnergyRate,0,0,point),
            {thermal(0,wr::kThermalNumberNuE,0,0,point),thermal(0,wr::kThermalNumberAntiNuE,0,0,point)},
            {thermal(0,wr::kThermalEnergyNuE,0,0,point),thermal(0,wr::kThermalEnergyAntiNuE,0,0,point)}};
        Real eeta[2], eabs[2], escat[2];
        bool pass = true;
        for (int s = 0; s < 2; ++s) {
          AthenaPointTensor<Real,TensorSymm::NONE,4,1> fd{};
          for (int a = 0; a < 4; ++a) baseline[s][a] = before(0,CombinedIdx(s,a,nv),0,0,point);
          for (int a = 1; a < 4; ++a) fd(a) = baseline[s][a];
          apply_floor(g.gu,baseline[s][0],fd,p);
          for (int a = 1; a < 4; ++a) baseline[s][a] = fd(a);
          old_number[s] = floor_number(before(0,CombinedIdx(s,M1_N_IDX,nv),0,0,point),p);
          in.number_star[s] = old_number[s];
          in.eta_number_cc[s] = eta_n(0,s,0,0,point);
          in.abs_number_cc[s] = abs_n(0,s,0,0,point);
          eeta[s] = eta_e(0,s,0,0,point); eabs[s] = abs_e(0,s,0,0,point); escat[s] = scat(0,s,0,0,point);
          in.explicit_number[s] = explicit_before(0,CombinedIdx(s,M1_N_IDX,nv),0,0,point);
          numbers[s] = mode == 0 ? after(0,CombinedIdx(s,M1_N_IDX,nv),0,0,point) : in.explicit_number[s];
          for (int a = 0; a < 4; ++a) {
            in.explicit_moment[s][a] = explicit_before(0,CombinedIdx(s,a,nv),0,0,point);
            current[s][a] = mode == 0 ? after(0,CombinedIdx(s,a,nv),0,0,point) : in.explicit_moment[s][a];
          }
        }
        Real theta = 1.;
        OracleState expected{};
        if (scenario == 9 || scenario == 10) {
          // Independently solved unlimited endpoint supplies a caller-wiring
          // witness. Its physics is separately checked by direct contractions.
          AthenaPointTensor<Real,TensorSymm::NONE,4,1> fs[2]{};
          for (int s = 0; s < 2; ++s) for (int a = 1; a < 4; ++a) fs[s](a) = baseline[s][a];
          SrcParams electron[2] = {
            SrcParams(dt,g.alpha,g.gd,g.gu,g.nd,g.nu,g.spatial,g.ud,g.uu,g.vd,g.vu,g.projector,g.W,baseline[0][0],fs[0],1./3.,g.volume*eeta[0],eabs[0],escat[0]),
            SrcParams(dt,g.alpha,g.gd,g.gu,g.nd,g.nu,g.spatial,g.ud,g.uu,g.vd,g.vu,g.projector,g.W,baseline[1][0],fs[1],1./3.,g.volume*eeta[1],eabs[1],escat[1])};
          wr::ElectronThermalSourceResult unlimited{};
          pass = wr::UpdateElectronThermalSources(BrentFunctor{},electron,in,p,unlimited);
          // Use the better-resolved number channel, not an assumed equality of
          // rounded stored deltas across populations with unequal references.
          const Real dn0 = unlimited.number[0]-old_number[0];
          const Real dn1 = unlimited.number[1]-old_number[1];
          const int witness = Kokkos::fabs(dn0)/old_number[0] >=
              Kokkos::fabs(dn1)/old_number[1] ? 0 : 1;
          const Real dn = unlimited.number[witness]-old_number[witness];
          theta = dn != 0. ? (after(0,CombinedIdx(witness,M1_N_IDX,nv),0,0,point)-old_number[witness])/dn : -1.;
          pass = pass && theta > 0. && theta < 1.;
          for (int s = 0; s < 2; ++s) {
            if (mode == 0) {
              numbers[s] = unlimited.number[s];
              for (int a = 0; a < 4; ++a) current[s][a] = unlimited.moment[s][a];
            }
            pass = pass && ResidualClose(after(0,CombinedIdx(s,M1_N_IDX,nv),0,0,point)-old_number[s]-theta*(unlimited.number[s]-old_number[s]),
                after(0,CombinedIdx(s,M1_N_IDX,nv),0,0,point),old_number[s],p);
          }
        }
        pass = pass && Oracle(g,current,numbers,in,eeta,eabs,escat,dt,p,expected);
        Real sum[4]{};
        for (int s = 0; s < 4; ++s) for (int a = 0; a < 4; ++a) {
          Real base = before(0,CombinedIdx(s,a,nv),0,0,point);
          if (s < 2) base = baseline[s][a];
          sum[a] += after(0,CombinedIdx(s,a,nv),0,0,point)-base;
          if (s < 2) pass = pass && ResidualClose(
              after(0,CombinedIdx(s,a,nv),0,0,point)-base-theta*expected.moment_increment[s][a],
              after(0,CombinedIdx(s,a,nv),0,0,point),base,p);
        }
        for (int s = 0; s < 2; ++s) pass = pass && ResidualClose(
            after(0,CombinedIdx(s,M1_N_IDX,nv),0,0,point)-old_number[s]-theta*(expected.cc[s]+expected.pair),
            after(0,CombinedIdx(s,M1_N_IDX,nv),0,0,point),old_number[s],p);
        if (pure) pass = pass && expected.cc[0] == 0. && expected.cc[1] == 0. && expected.pair != 0. &&
            (after(0,CombinedIdx(0,M1_FX_IDX,nv),0,0,point) != 0. ||
             after(0,CombinedIdx(1,M1_FX_IDX,nv),0,0,point) != 0.);
        if (stage == 2 && p.backreact) {
          for (int a = 0; a < 4; ++a) {
            const int var = a == 0 ? IEN : IVX+a-1;
            pass = pass && ResidualClose(fluid(0,var,0,0,point)-matter_before(0,var,0,0,point)+sum[a],
                sum[a],0.,p);
          }
          const Real expected_ye = mb*theta*(expected.cc[1]-expected.cc[0]);
          if (!pure) pass = pass && expected_ye != 0.;
          pass = pass && ResidualClose(
              (fluid(0,IYF,0,0,point)-matter_before(0,IYF,0,0,point))/mb - theta*(expected.cc[1]-expected.cc[0]),
              Kokkos::fmax(after(0,CombinedIdx(0,M1_N_IDX,nv),0,0,point),after(0,CombinedIdx(1,M1_N_IDX,nv),0,0,point)),
              Kokkos::fmax(old_number[0],old_number[1]),p);
          if (pure) pass = pass && fluid(0,IYF,0,0,point) == matter_before(0,IYF,0,0,point);
        }
        if (scenario == 7) pass = pass && before(0,CombinedIdx(0,M1_E_IDX,nv),0,0,point) < p.rad_E_floor;
        outcome(0) = pass;
      });
      Check(Snapshot(outcome)(0),key+" independent source/accepted exchange/CC-only Ye");
    }
#endif
  }
  rad.params = original_params;
}

#ifndef WEAKRATES_LEGACY_CAPTURE
template<class Logs> void AcceptanceCases(Fixture<Logs> &f, const std::string &prefix) {
  if (f.option == 0) return;  // Off path has no pair-specific guard; records cover it.
  auto &rad = f.Rad();
  const auto original = rad.params;
  const int cells = rad.u0.extent(4), point = f.mesh->mb_indcs.is, nv = rad.nvars;
  for (int mode = 0; mode < 2; ++mode) for (int scenario = 0; scenario < 8; ++scenario) {
    // Scenario 6 checks explicit overshoot and the corresponding safe implicit root.
    f.Initialize(); f.ZeroLegacy();
    rad.params = original;
    rad.params.src_update = mode == 0 ? Implicit : Explicit;
    rad.params.theta_limiter = scenario == 7;
    if (scenario == 7) {
      rad.params.source_limiter = .125;
      rad.params.source_Ye_min = 0.;
      rad.params.source_Ye_max = 1.;
    }
    rad.params.rad_N_floor = 1.;
    rad.params.rad_E_floor = scenario == 3 || scenario == 4 ? 1. : .125;
    f.mesh->dt = 1.;
    auto u = rad.u0, u1 = rad.u1, thermal = rad.weakrates_thermal;
    auto eta_n = rad.eta_0, abs_n = rad.abs_0;
    auto input_matter = f.mesh->pmb_pack->pmhd->u0;
    auto input_primitive = f.mesh->pmb_pack->pmhd->w0;
    const Real baryon = f.mb;
    const auto p = rad.params;
    Kokkos::parallel_for("acceptance_inputs",Kokkos::RangePolicy<DevExeSpace>(0,cells),
        KOKKOS_LAMBDA(int i) {
      thermal(0,wr::kThermalNumberRate,0,0,i) = scenario == 3 || scenario == 4 || scenario == 6 ? 0. : 1.;
      thermal(0,wr::kThermalEnergyRate,0,0,i) = scenario == 6 ? 6. : (scenario == 3 || scenario == 4 ? 2. : 0.);
      thermal(0,wr::kThermalNumberNuE,0,0,i) = scenario == 6 ? 1.25 : (scenario == 7 ? 4. : .5);
      thermal(0,wr::kThermalNumberAntiNuE,0,0,i) = scenario == 6 ? 3. : (scenario == 7 ? 4. : .5);
      thermal(0,wr::kThermalEnergyNuE,0,0,i) = scenario == 6 ? 2. : .5;
      thermal(0,wr::kThermalEnergyAntiNuE,0,0,i) = scenario == 6 ? 4. : .5;
      if (scenario == 7) {
        // Source-only synthetic matter input makes the gross Ye bound
        // independently resolvable: rho/mb=4, Ye=1/2, limiter=1/8.
        // No producer/EOS/C2P call follows this input change.
        input_primitive(0,IDN,0,0,i) = input_matter(0,IDN,0,0,i) = 4.*baryon;
        input_primitive(0,IYF,0,0,i) = .5;
        input_matter(0,IYF,0,0,i) = 2.*baryon;
      }
      for (int s = 0; s < 4; ++s) {
        if (scenario == 5 && s < 2) {
          eta_n(0,s,0,0,i) = s == 0 ? .25 : .125;
          abs_n(0,s,0,0,i) = s == 0 ? .25 : .5;
        }
        const Real n = s == 0 ? (scenario == 0 ? 1. :
            (scenario == 2 ? 1.+1./1048576. : 1.25)) : 3.;
        const Real e = s == 0 ? (scenario == 3 ? 1.25 : 2.) : 4.;
        for (int a = 0; a < 5; ++a) {
          const Real value = a == M1_E_IDX ? e : (a == M1_N_IDX ? n :
              (scenario == 6 && a == M1_FX_IDX && s == 0 ? 1.5 :
               (scenario == 4 && a == M1_FX_IDX && s < 2 ? 1.75 : 0.)));
          u(0,CombinedIdx(s,a,nv),0,0,i) = u1(0,CombinedIdx(s,a,nv),0,0,i) = value;
        }
      }
    });
    // An independent before View avoids aliasing even on Serial.
    DvceArray5D<Real> before("acceptance_snapshot",u.extent(0),u.extent(1),u.extent(2),u.extent(3),u.extent(4));
    Kokkos::deep_copy(before,u);
    auto matter = f.mesh->pmb_pack->pmhd->u0;
    const auto matter_before = Values(matter);
    DvceArray5D<Real> matter_snapshot("acceptance_matter_snapshot",matter.extent(0),matter.extent(1),matter.extent(2),matter.extent(3),matter.extent(4));
    Kokkos::deep_copy(matter_snapshot,matter);
    DvceArray1D<wr::ElectronThermalSourceResult> unlimited("unlimited_endpoint",1);
    DvceArray1D<int> setup("acceptance_setup",1);
    Kokkos::parallel_for("unlimited_subfloor_root",Kokkos::RangePolicy<DevExeSpace>(0,1),
        KOKKOS_LAMBDA(int) {
      Geometry g;
      AthenaPointTensor<Real,TensorSymm::NONE,4,1> fs[2]{};
      wr::ElectronThermalSourceInput in{};
      in.volume = 1.;
      in.thermal = {thermal(0,wr::kThermalNumberRate,0,0,point),
          thermal(0,wr::kThermalEnergyRate,0,0,point),
          {thermal(0,wr::kThermalNumberNuE,0,0,point),thermal(0,wr::kThermalNumberAntiNuE,0,0,point)},
          {thermal(0,wr::kThermalEnergyNuE,0,0,point),thermal(0,wr::kThermalEnergyAntiNuE,0,0,point)}};
      for (int s = 0; s < 2; ++s) {
        in.eta_number_cc[s] = eta_n(0,s,0,0,point);
        in.abs_number_cc[s] = abs_n(0,s,0,0,point);
        in.number_star[s] = before(0,CombinedIdx(s,M1_N_IDX,nv),0,0,point);
        in.explicit_number[s] = in.number_star[s];
        for (int a = 0; a < 4; ++a) {
          in.explicit_moment[s][a] = before(0,CombinedIdx(s,a,nv),0,0,point);
          if (a > 0) fs[s](a) = in.explicit_moment[s][a];
        }
      }
      SrcParams electron[2] = {
        SrcParams(1.,1.,g.gd,g.gu,g.nd,g.nu,g.spatial,g.ud,g.uu,g.vd,g.vu,g.projector,g.W,in.explicit_moment[0][0],fs[0],1./3.,0.,0.,0.),
        SrcParams(1.,1.,g.gd,g.gu,g.nd,g.nu,g.spatial,g.ud,g.uu,g.vd,g.vu,g.projector,g.W,in.explicit_moment[1][0],fs[1],1./3.,0.,0.,0.)};
      wr::ElectronThermalSourceResult out{};
      bool ok = wr::UpdateElectronThermalSources(BrentFunctor{},electron,in,p,out);
      if (scenario < 3 || scenario == 5) ok = ok && out.number[0] < p.rad_N_floor;
      else if (scenario < 5) ok = ok && out.moment[0][0] < p.rad_E_floor;
      else if (scenario == 6) {
        // Rest-frame relaxation at J=B_E leaves E unchanged. Explicit
        // stepping overshoots F while the implicit root is causal. Floors
        // alone permit theta=1 in both modes. Independently check the full
        // and half candidates so deleting causal backoff cannot pass.
        Real preliminary = 1.;
        bool half_safe = true;
        for (int s = 0; s < 2; ++s) {
          const Real e = before(0,CombinedIdx(s,M1_E_IDX,nv),0,0,point);
          const Real n = before(0,CombinedIdx(s,M1_N_IDX,nv),0,0,point);
          if (out.moment[s][0] < e)
            preliminary = Kokkos::fmin(preliminary,(e-p.rad_E_floor)/(e-out.moment[s][0]));
          if (out.number[s] < n)
            preliminary = Kokkos::fmin(preliminary,(n-p.rad_N_floor)/(n-out.number[s]));
          const Real eh = e+.5*(out.moment[s][0]-e);
          const Real nh = n+.5*(out.number[s]-n);
          Real f2 = 0.;
          for (int a = 1; a < 4; ++a) {
            const Real fb = before(0,CombinedIdx(s,a,nv),0,0,point);
            const Real fh = fb+.5*(out.moment[s][a]-fb);
            f2 += fh*fh;
          }
          half_safe = half_safe && eh >= p.rad_E_floor && nh >= p.rad_N_floor &&
                      f2 <= (1.-p.rad_eps)*eh*eh;
        }
        Real full_flux2 = 0.;
        for (int a = 1; a < 4; ++a) full_flux2 += out.moment[0][a]*out.moment[0][a];
        const Real causal_bound = (1.-p.rad_eps)*out.moment[0][0]*out.moment[0][0];
        ok = ok && preliminary == 1. && half_safe &&
             (mode == 0 ? full_flux2 <= causal_bound : full_flux2 > causal_bound);
      } else {
        // Pure pair number gain has no species-loss, energy or floor bound.
        // Gross Ye up/down limits alone each allow a shared count of 1/4,
        // even though accepted matter Ye is unchanged by the pair operator.
        ok = ok && out.pair_number_increment > .5 &&
             out.cc_number_increment[0] == 0. && out.cc_number_increment[1] == 0.;
        for (int s = 0; s < 2; ++s)
          ok = ok && out.moment[s][0] == before(0,CombinedIdx(s,M1_E_IDX,nv),0,0,point);
      }
      setup(0) = ok; unlimited(0) = out;
    });
    Check(Snapshot(setup)(0),prefix+" independent acceptance boundary setup mode="+std::to_string(mode)+" scenario="+std::to_string(scenario));
    Check(rad.TimeUpdate(nullptr,2) == TaskStatus::complete,prefix+" paired guard actual TimeUpdate");
    DvceArray1D<int> accepted("pair_acceptance_check",1);
    const Real mb = f.mb;
    Kokkos::parallel_for("pair_acceptance_oracle",Kokkos::RangePolicy<DevExeSpace>(0,1),
        KOKKOS_LAMBDA(int) {
      const auto out = unlimited(0);
      const int axis = scenario == 6 ? M1_FX_IDX :
                       (scenario < 3 || scenario == 5 || scenario == 7 ? M1_N_IDX : M1_E_IDX);
      const Real base0 = before(0,CombinedIdx(0,axis,nv),0,0,point);
      const Real end0 = axis == M1_N_IDX ? out.number[0] : out.moment[0][axis];
      const Real theta = (u(0,CombinedIdx(0,axis,nv),0,0,point)-base0)/(end0-base0);
      bool pass = Kokkos::isfinite(theta) && theta >= 0. && theta <= 1.;
      if (scenario == 0) pass = pass && theta == 0.;
      else if (scenario == 6) pass = pass && theta == (mode == 0 ? 1. : .5);
      else pass = pass && theta > 0. && theta < 1.;
      if (scenario == 7) {
        // Independent gross-bound oracle, not an expected factor inferred
        // from publication: limiter*(1/2)*rho/mb = 1/4 for each pair count.
        for (int s = 0; s < 2; ++s) {
          const Real nb = before(0,CombinedIdx(s,M1_N_IDX,nv),0,0,point);
          const Real n = u(0,CombinedIdx(s,M1_N_IDX,nv),0,0,point);
          pass = pass && ResidualClose(n-nb-.25,n,nb,p);
        }
      }
      for (int s = 0; s < 2; ++s) {
        const Real n = u(0,CombinedIdx(s,M1_N_IDX,nv),0,0,point);
        const Real nb = before(0,CombinedIdx(s,M1_N_IDX,nv),0,0,point);
        const Real e = u(0,CombinedIdx(s,M1_E_IDX,nv),0,0,point);
        Real f2 = 0.;
        pass = pass && n >= p.rad_N_floor && e >= p.rad_E_floor;
        pass = pass && ResidualClose(n-nb-theta*(out.number[s]-nb),n,nb,p);
        for (int a = 0; a < 4; ++a) {
          const Real base = before(0,CombinedIdx(s,a,nv),0,0,point);
          const Real stored = u(0,CombinedIdx(s,a,nv),0,0,point);
          pass = pass && ResidualClose(stored-base-theta*(out.moment[s][a]-base),stored,base,p);
          if (a > 0) f2 += stored*stored;
          if (theta == 0.) pass = pass && stored == base;
        }
        pass = pass && f2 <= (1.-p.rad_eps)*e*e;
      }
      if (scenario == 5) {
        Real cc[2]{};
        for (int s = 0; s < 2; ++s) {
          const Real nb = before(0,CombinedIdx(s,M1_N_IDX,nv),0,0,point);
          const Real evaluated_number = mode == 0 ? out.number[s] : nb;
          cc[s] = eta_n(0,s,0,0,point)-abs_n(0,s,0,0,point)*evaluated_number;
          pass = pass && ResidualClose(out.cc_number_increment[s]-cc[s],out.number[s],nb,p);
        }
        const Real ye_number_residual =
            (matter(0,IYF,0,0,point)-matter_snapshot(0,IYF,0,0,point))/mb - theta*(cc[1]-cc[0]);
        pass = pass && cc[1] != cc[0] && ResidualClose(ye_number_residual,
            Kokkos::fmax(out.number[0],out.number[1]),
            Kokkos::fmax(before(0,CombinedIdx(0,M1_N_IDX,nv),0,0,point),before(0,CombinedIdx(1,M1_N_IDX,nv),0,0,point)),p);
      }
      accepted(0) = pass;
    });
    Check(Snapshot(accepted)(0),prefix+" shared acceptance floors/causal backoff/gross bound/counts scenario="+std::to_string(scenario));
    auto host = Snapshot(matter);
    // All input matter cells are unchanged except stage-2 accepted exchange;
    // pure pair attribution must never alter the conserved Ye slot.
    auto original_host = Kokkos::create_mirror(matter);
    std::copy(matter_before.begin(),matter_before.end(),original_host.data());
    if (scenario != 5) Check(host(0,IYF,0,0,point) == original_host(0,IYF,0,0,point),prefix+" floor-limited pair is lepton neutral");
    if (scenario == 0) Check(SameBits(matter_before,Values(matter)),prefix+" deliberate zero acceptance has zero matter exchange");
  }
  rad.params = original;
}
#endif

#ifndef WEAKRATES_LEGACY_CAPTURE
template<class Logs> void OtherBackendRouting(const std::string &deck,
    const std::string &table, int ng, const std::string &backend) {
  Fixture<Logs> f(deck,table,ng,1,false,backend);
  auto &rad = f.Rad();
  Check(rad.weakrates_thermal.size() == 0,backend+" ignores enabled weakrates key allocation");
  for (int mode = 0; mode < 2; ++mode) for (int stage = 1; stage <= 2; ++stage) {
    f.Initialize(); f.ZeroLegacy();
    f.mesh->dt = 0.;
    rad.params.src_update = mode == 0 ? Implicit : Explicit;
    Check(rad.TimeUpdate(nullptr,stage) == TaskStatus::complete,
          backend+" empty thermal View routing NGHOST="+std::to_string(ng));
    Check(rad.weakrates_thermal.size() == 0,backend+" no thermal reallocation");
  }
}

template<class Logs> void FatalCase(const std::string &deck, const std::string &table,
    int ng, int mode, const std::string &reason) {
  const bool off = reason == "producer-model-off" || reason == "acceptance-off" ||
                   reason == "producer-raw-invalid-off";
  Fixture<Logs> f(deck,table,ng,off ? 0 : 1,false);
  auto &rad = f.Rad();
  if (reason == "producer-raw-invalid" || reason == "producer-raw-invalid-off") {
    // The scratch TU injects NaN before original raw-rate validation, outside
    // the thermal option gate. Prior microscopic invalidity must retain its
    // all-cell zeroing policy in both states, never enter the new-model abort.
    f.SeedCoefficients(-77.);
    Check(rad.CalcOpacityWeakRates(nullptr,1) == TaskStatus::complete,
          "raw-invalid real producer returns original zeroed coefficients");
    bool pass = true;
    for (auto view : {rad.eta_0,rad.eta_1,rad.abs_0,rad.abs_1,rad.scat_1}) {
      const auto host = Snapshot(view);
      for (int i = 0; i < view.extent(4); ++i) {
        const bool evaluated = i >= f.mesh->mb_indcs.is-1 && i <= f.mesh->mb_indcs.ie+1;
        for (int s = 0; s < 4; ++s)
          pass = pass && host(0,s,0,0,i) == (evaluated ? 0. : -77.);
      }
    }
    Check(pass,"raw-invalid on/off preserves all-five-field zeroing and outer ghosts");
    if (off) {
      Check(rad.weakrates_thermal.size() == 0,"raw-invalid off path retains empty thermal View");
    } else {
      const auto host = Snapshot(rad.weakrates_thermal);
      pass = true;
      for (int i = 0; i < rad.weakrates_thermal.extent(4); ++i) {
        const bool evaluated = i >= f.mesh->mb_indcs.is-1 && i <= f.mesh->mb_indcs.ie+1;
        for (int a = 0; a < wr::kNumThermalFields; ++a)
          pass = pass && host(0,a,0,0,i) == (evaluated ? 0. : -77.);
      }
      Check(pass,"raw-invalid on path retains zero thermal fields and outer ghosts");
    }
    return;
  }
  if (reason == "producer-zero") {
    // This native binary links a scratch producer with both raw electron
    // thermal channels set to zero AFTER original coefficient construction.
    // All six fields must nevertheless be published, including target references.
    Fixture<Logs> disabled(deck,table,ng,0,false);
    auto &old = disabled.Rad();
    for (int stage = 1; stage <= 2; ++stage) {
      f.mesh->dt = disabled.mesh->dt = 0.;
      f.Initialize(); disabled.Initialize();
      f.SeedCoefficients(-77.); disabled.SeedCoefficients(-77.);
      rad.CalcOpacityWeakRates(nullptr,1); old.CalcOpacityWeakRates(nullptr,1);
      auto thermal = Snapshot(rad.weakrates_thermal);
      auto reference = Snapshot(f.reference);
      bool pass = rad.weakrates_thermal.extent(1) == wr::kNumThermalFields;
      for (int i = f.mesh->mb_indcs.is-1; i <= f.mesh->mb_indcs.ie+1; ++i) {
        pass = pass && thermal(0,wr::kThermalNumberRate,0,0,i) == 0. &&
               thermal(0,wr::kThermalEnergyRate,0,0,i) == 0.;
        for (int a = 2; a < wr::kNumThermalFields; ++a)
          pass = pass && CoefficientClose(thermal(0,a,0,0,i),reference(0,20+a,0,0,i));
      }
      Check(pass,"zero-rate real producer publishes all six fields");
      Check(SameBits(f.Coefficients(),disabled.Coefficients()),"zero-rate producer leaves original five fields unchanged");
      f.mesh->dt = disabled.mesh->dt = f.SourceDt();
      rad.params.src_update = old.params.src_update = mode == 0 ? Implicit : Explicit;
      rad.TimeUpdate(nullptr,stage); old.TimeUpdate(nullptr,stage);
      Check(SameBits(Values(rad.u0),Values(old.u0)) && SameBits(Values(rad.chi),Values(old.chi)) &&
            SameBits(Values(f.mesh->pmb_pack->pmhd->u0),Values(disabled.mesh->pmb_pack->pmhd->u0)),
            "zero-rate enabled actual stage follows legacy source/acceptance/attribution route");
    }
    return;
  }
  if (reason == "producer-model" || reason == "producer-model-off") {
    // Python replaces ONLY this producer's object with a scratch fault-injected
    // copy. The actual wrapper/kernel/builder/abort path is then exercised.
    f.SeedCoefficients(-77.);
    rad.CalcOpacityWeakRates(nullptr,1);
    if (off) {
      Check(rad.weakrates_thermal.size() == 0,"fault-injected producer remains gated off");
      auto reference = Snapshot(f.reference);
      int field = 0;
      bool pass = true;
      for (auto view : {rad.eta_0,rad.eta_1,rad.abs_0,rad.abs_1,rad.scat_1}) {
        auto host = Snapshot(view);
        for (int i = f.mesh->mb_indcs.is-1; i <= f.mesh->mb_indcs.ie+1; ++i)
          for (int s = 0; s < 4; ++s)
            pass = pass && CoefficientClose(host(0,s,0,0,i),reference(0,4*field+s,0,0,i));
        ++field;
      }
      Check(pass,"fault-injected off producer retains original five fields and heavy fix");
      return;
    }
    Check(false,"fault-injected active producer must abort"); return;
  }
  rad.CalcOpacityWeakRates(nullptr,1);
  f.mesh->dt = f.SourceDt();
  f.ZeroLegacy();
  rad.params.src_update = mode == 0 ? Implicit : Explicit;
  const int point = f.mesh->mb_indcs.is;
  auto thermal = rad.weakrates_thermal;
  auto eta = rad.eta_1;
  if (reason == "stored-rate" || reason == "stored-reference" || reason == "global-heavy") {
    const int kind = reason == "stored-rate" ? 0 : (reason == "stored-reference" ? 1 : 2);
    Kokkos::parallel_for("fatal_stored_input",Kokkos::RangePolicy<DevExeSpace>(0,1),
        KOKKOS_LAMBDA(int) {
      if (kind == 0) thermal(0,wr::kThermalNumberRate,0,0,point) = std::numeric_limits<Real>::quiet_NaN();
      if (kind == 1) thermal(0,wr::kThermalNumberAntiNuE,0,0,point) = 0.;
      if (kind == 2) eta(0,2,0,0,point) = std::numeric_limits<Real>::quiet_NaN();
    });
  } else if (reason == "acceptance" || reason == "acceptance-off") {
    // D=0 succeeds without a source solve. Configured final transport closure
    // still requires iterations for this causal nonisotropic state; zero budget
    // must diagnose AcceptanceFailure rather than publish a closure fallback.
    f.mesh->dt = 0.;
    rad.params.closure_type = Minerbo;
    rad.params.closure_maxiter = 0;
    auto u = rad.u0, u1 = rad.u1;
    const int nv = rad.nvars;
    Kokkos::parallel_for("fatal_final_closure",Kokkos::RangePolicy<DevExeSpace>(0,1),
        KOKKOS_LAMBDA(int) {
      for (int s = 0; s < 2; ++s) {
        const Real e = u(0,CombinedIdx(s,M1_E_IDX,nv),0,0,point);
        u(0,CombinedIdx(s,M1_FX_IDX,nv),0,0,point) = .25*e;
        u1(0,CombinedIdx(s,M1_FX_IDX,nv),0,0,point) = .25*e;
      }
    });
  } else throw std::runtime_error("unknown fatal fixture");
  rad.TimeUpdate(nullptr,2);
  Kokkos::fence();
  if (off) { Check(rad.weakrates_thermal.size() == 0,"disabled feature retains legacy closure fallback"); return; }
  Check(false,"active thermal "+reason+" must abort");
}
#endif

template<class Logs> void Matrix(const std::string &deck, const std::string &table,
    const std::string &logs, Records &legacy) {
  for (int ng : {2,3,4}) for (bool predictor : {false,true}) {
    const std::string prefix = logs+"/ng"+std::to_string(ng)+"/peq"+std::to_string(predictor);
    {
      Fixture<Logs> off(deck,table,ng,0,predictor);
      PublicationCases(off,prefix,legacy);
      ConsumerCases(off,prefix,legacy);
    }
#ifndef WEAKRATES_LEGACY_CAPTURE
    Records omitted, explicit_on;
    {
      Fixture<Logs> defaults(deck,table,ng,-1,predictor);
      PublicationCases(defaults,prefix,omitted);
      ConsumerCases(defaults,prefix,omitted);
      AcceptanceCases(defaults,prefix+"/omitted");
    }
    {
      Fixture<Logs> enabled(deck,table,ng,1,predictor);
      PublicationCases(enabled,prefix,explicit_on);
      ConsumerCases(enabled,prefix,explicit_on);
      AcceptanceCases(enabled,prefix+"/explicit-on");
      if (ng == 2 && !predictor && logs == "NormalLogs") {
        ThermalDeviceSources(enabled.Rad().params);
        ThermalNumericalBoundaries(enabled.Rad().params);
        ThermalFailureBoundary(enabled.Rad().params);
      }
    }
    bool same = omitted.size() == explicit_on.size();
    for (const auto &entry : omitted) {
      const auto found = explicit_on.find(entry.first);
      same = same && found != explicit_on.end() && SameBits(entry.second,found->second);
    }
    Check(same,prefix+" omitted-key/explicit-on bitwise coefficients and both source routes");
    // Both original coefficient paths independently have the heavy fixed oracle.
    // Enabled zero-rate publication and gates are compared to off records where
    // appropriate; independent positive-rate off preservation uses saved JSON.
    for (int mode = 0; mode < 2; ++mode) for (int stage = 1; stage <= 2; ++stage)
        for (int scenario : {2,3,4,6}) {
      const std::string key = prefix+"/source-"+std::to_string(mode)+"-"+std::to_string(stage)+"-"+std::to_string(scenario);
      for (const std::string suffix : {"-rad","-matter","-chi"})
        Check(SameBits(omitted.at(key+suffix),legacy.at(key+suffix)),key+suffix+" inactive enabled equals legacy route");
    }
#endif
  }
#ifndef WEAKRATES_LEGACY_CAPTURE
  for (int ng : {2,3,4}) {
    OtherBackendRouting<Logs>(deck,table,ng,"none");
#if ENABLE_NURATES
    OtherBackendRouting<Logs>(deck,table,ng,"bns-nurates");
#endif
  }
#endif
}
}  // namespace

int main(int argc, char **argv) {
  static_assert(sizeof(Real) == 8,"weakrates emission suite requires double precision");
  if (argc < 6) {
    std::cerr << "FAIL usage: native DECK NORMAL_TABLE NQT_TABLE --records PATH | --fatal CASE NG LOGS MODE\n";
    return 2;
  }
  global_variable::my_rank = 0;
  global_variable::nranks = 1;
  Kokkos::initialize();
  try {
    // All mesh/EOS/reference Views are destroyed before Kokkos::finalize.
    const std::string route = argv[4];
#ifndef WEAKRATES_LEGACY_CAPTURE
    if (route == "--fatal") {
      if (argc != 9) throw std::runtime_error("fatal fixture requires NG LOGS MODE");
      const int ng = std::stoi(argv[6]), mode = std::stoi(argv[8]);
      if (std::string(argv[7]) == "NQTLogs")
        FatalCase<Primitive::NQTLogs>(argv[1],argv[3],ng,mode,argv[5]);
      else if (std::string(argv[7]) == "NormalLogs")
        FatalCase<Primitive::NormalLogs>(argv[1],argv[2],ng,mode,argv[5]);
      else throw std::runtime_error("unknown log policy");
    } else
#endif
    if (route == "--records") {
      Records legacy;
      HeavyBoundary();
#ifndef WEAKRATES_LEGACY_CAPTURE
      ThermalBuilderBoundary();
#endif
      Matrix<Primitive::NormalLogs>(argv[1],argv[2],"NormalLogs",legacy);
      Matrix<Primitive::NQTLogs>(argv[1],argv[3],"NQTLogs",legacy);
      WriteRecords(legacy,argv[5]);
    } else throw std::runtime_error("unknown native route");
  } catch (const std::exception &e) {
    Check(false,std::string("native fixture setup: ")+e.what());
  }
  Kokkos::finalize();
  std::cout << (failures ? "FAIL " : "PASS ") << "weakrates emission native suite\n";
  return failures ? 1 : 0;
}
