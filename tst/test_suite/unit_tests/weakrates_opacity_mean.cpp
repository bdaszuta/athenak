// Focused direct-opacity-mean regression against the real WeakRates producer.
// No Driver, pgen invocation, source update, or simulation timestep is needed.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
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
#include "radiation_m1/radiation_m1_helpers.hpp"
#include "radiation_m1/radiation_m1_weakrates.hpp"
#include "radiation_m1/radiation_m1_weakrates_equilibrium_search.hpp"

namespace {
using namespace radiationm1;  // NOLINT(build/namespaces)
namespace wr = radiationm1::weakrates;
int failures = 0;

void Check(bool pass, const std::string &name) {
  std::cout << (pass ? "PASS " : "FAIL ") << name << '\n';
  failures += !pass;
}

bool Close(Real actual, Real expected) {
  // Keep the existing emission fixture's double-precision coefficient tolerance.
  return std::isfinite(actual) && std::isfinite(expected) &&
      (expected == 0. ? actual == 0. :
       std::abs(actual - expected) <= 1.e-12*std::abs(expected));
}

// create_mirror_view can alias a device View on Serial. Always allocate snapshots.
template<class View> auto Snapshot(const View &view) {
  auto host = Kokkos::create_mirror(view);
  Kokkos::deep_copy(host, view);
  return host;
}

template<class View> std::vector<Real> Values(const View &view) {
  const auto host = Snapshot(view);
  return std::vector<Real>(host.data(), host.data() + host.size());
}

bool SameBits(const std::vector<Real> &a, const std::vector<Real> &b) {
  return a.size() == b.size() &&
      std::memcmp(a.data(), b.data(), a.size()*sizeof(Real)) == 0;
}

void HelperCases() {
  constexpr int count = 15;
  DvceArray2D<Real> result("spectral_helper_channels", count, 20);
  Kokkos::parallel_for("spectral_helper_cases",
      Kokkos::RangePolicy<DevExeSpace>(0, count), KOKKOS_LAMBDA(int c) {
    Real n[4], e[4], bn[4], be[4], en[4], ee[4], an[4], ae[4], sc[4];
    for (int s = 0; s < 4; ++s) {
      n[s] = 1.; bn[s] = 1.; be[s] = 10.;
      // Algebraic counter-stream witness: Fv/E=-0.1 and direct mean 12 MeV.
      // The helper sees only the numerator/number pair, not these frame inputs.
      const Real W = 1.25, E = 12./(W*(1. - (-.1)));
      e[s] = W*(E - (-.1*E));
      en[s] = 2.; ee[s] = 3.; an[s] = 5.; ae[s] = 7.; sc[s] = 11.;
      if (c == 1) e[s] = 1.;
      if (c == 2) e[s] = 40.;
      if (c == 4) n[s] = 0.;
      if (c == 5) n[s] = -1.;
      if (c == 6) e[s] = 0.;
      if (c == 7) e[s] = -1.;
      if (c == 8) bn[s] = 0.;
      if (c == 9) bn[s] = -1.;
      if (c == 10) be[s] = 0.;
      if (c == 11) be[s] = -1.;
      if (c == 12) {
        n[s] = std::numeric_limits<Real>::min();
        e[s] = std::numeric_limits<Real>::max();
      }
      if (c == 13) e[s] = std::numeric_limits<Real>::infinity();
      if (c == 14) e[s] = std::numeric_limits<Real>::quiet_NaN();
    }
    wr::ApplySpectralCorrection(4, c == 3 ? 1. : 1.75, n, e, bn, be,
                                en, ee, an, ae, sc);
    for (int s = 0; s < 4; ++s) {
      result(c, s) = en[s]; result(c, 4+s) = ee[s];
      result(c, 8+s) = an[s]; result(c, 12+s) = ae[s];
      result(c, 16+s) = sc[s];
    }
  });
  const auto host = Snapshot(result);
  const char *names[count] = {"12/10 counter-stream factor 1.44", "lower clamp",
      "upper clamp", "unity clamp", "zero number", "negative number",
      "zero numerator", "negative numerator", "zero reference number",
      "negative reference number", "zero reference energy", "negative reference energy",
      "overflow ratio neutral fallback", "infinite ratio neutral fallback",
      "NaN numerator neutral fallback"};
  for (int c = 0; c < count; ++c) {
    const Real factor = c == 0 ? 1.44 :
        (c == 1 ? (1./1.75)*(1./1.75) : (c == 2 ? 1.75*1.75 : 1.));
    bool pass = true;
    for (int s = 0; s < 4; ++s) {
      pass = pass && Close(host(c,s), 2.*(s < 2 ? 1. : factor)) &&
          Close(host(c,4+s), 3.*(s < 2 ? 1. : factor)) &&
          Close(host(c,8+s), 5.*(s < 2 ? factor : 1.)) &&
          Close(host(c,12+s), 7.*(s < 2 ? factor : 1.)) &&
          Close(host(c,16+s), 11.*factor);
    }
    Check(pass, std::string("device helper ") + names[c]);
  }
}

void NumberConversionCases() {
  DvceArray1D<Real> result("unchanged_number_gamma", 3);
  Kokkos::parallel_for("unchanged_number_gamma_cases",
      Kokkos::RangePolicy<DevExeSpace>(0, 3), KOKKOS_LAMBDA(int c) {
    RadiationM1Params p{};
    p.rad_E_floor = 1.e-30;
    AthenaPointTensor<Real, TensorSymm::NONE, 4, 1> fd{}, vu{};
    const Real s = c == 0 ? -.1 : (c == 1 ? 0. : .1);
    const Real W = c == 1 ? 1. : 1.25;
    const Real E = 8., J = c == 1 ? 8. : 6.;
    vu(1) = c == 1 ? 0. : .4;
    fd(1) = c == 1 ? 0. : s*E/.4;
    // A deliberate temporal distraction must not enter the spatial contraction.
    fd(0) = 100.; vu(0) = 100.;
    result(c) = radiation_number_gamma(E, J, fd, vu, W, p);
  });
  const auto host = Snapshot(result);
  const char *names[3] = {"counter-stream cap retained", "rest epsilon cap retained",
                          "co-stream conversion retained"};
  for (int c = 0; c < 3; ++c) {
    const Real s = c == 0 ? -.1 : (c == 1 ? 0. : .1);
    const Real W = c == 1 ? 1. : 1.25;
    const Real J = c == 1 ? 8. : 6.;
    const Real expected = W*(8./J)*std::min(1.-s, 1.-1.e-5);
    Check(Close(host(c), expected), std::string("shared number conversion ") + names[c]);
  }
}

// Preview and publication must use the same spectral pair, while thermal
// feasibility must use physical comoving number density. Synthetic coefficients
// near Real's representable limit distinguish those otherwise benign means.
void ValidationCases() {
  constexpr int count = 24;
  DvceArray2D<Real> result("target_validation_spectral_inputs", count, 23);
  Kokkos::parallel_for("target_validation_spectral_inputs",
      Kokkos::RangePolicy<DevExeSpace>(0, count), KOKKOS_LAMBDA(int c) {
    const int route = c/8, arm = c%8;
    const Real limit = std::numeric_limits<Real>::max();
    WeakRatesParams p{};
    p.opacity_corr_fac_max = 1.75;
    p.gra_opacity_order = route != 1;
    p.gra_target_mean = route != 0;
    wr::RawCoefficients raw{};
    Real n[4], e[4], sn[4], se[4], ln[4], le[4], tn[4], te[4];
    Real en[4], ej[4], an[4], aj[4], sc[4];
    const bool near_limit = arm < 2 || arm >= 6;
    const bool thermal = arm == 2 || arm == 3;
    for (int s = 0; s < 3; ++s) {
      raw.abs_0[s] = s == 0 && near_limit ? limit*.6 : 1.;
      raw.abs_1[s] = 2.; raw.scat_1[s] = 1.;
      raw.eta_0[s] = 2.; raw.eta_1[s] = 4.;
      if (thermal && s < 2) {
        raw.eta_0_non_cc[s] = 1.; raw.eta_1_non_cc[s] = 1.;
      }
    }
    for (int s = 0; s < 4; ++s) {
      ln[s] = 1.; le[s] = 10.; tn[s] = 1.5; te[s] = 15.;
      n[s] = 1.; e[s] = arm == 1 || arm == 6 ? 12. : 8.;
      sn[s] = 1.; se[s] = arm == 0 ? 12. : 8.;
      if (arm == 2) { sn[s] = limit/4.; se[s] = limit/2.; }
      if (arm == 3) { n[s] = limit/4.; e[s] = limit/2.; }
      const int rs = s < 2 ? s : wr::kNuX;
      en[s] = s < 2 ? 0. : .5*raw.eta_0[rs];
      ej[s] = s < 2 ? 0. : .5*raw.eta_1[rs];
      an[s] = raw.abs_0[rs]; aj[s] = raw.abs_1[rs]; sc[s] = raw.scat_1[rs];
    }
    // This is the producer's uncorrected selection-time snapshot in the deferred
    // route. Target-mean routes reset it from raw before applying their factor.
    Real saved[4][13];
    for (int s = 0; s < 4; ++s) {
      saved[s][0] = en[s]; saved[s][1] = ej[s]; saved[s][2] = an[s];
      saved[s][3] = aj[s]; saved[s][4] = sc[s];
      saved[s][5] = n[s]; saved[s][6] = e[s];
      saved[s][7] = sn[s]; saved[s][8] = se[s];
      saved[s][9] = ln[s]; saved[s][10] = le[s];
      saved[s][11] = tn[s]; saved[s][12] = te[s];
    }
    // Older aggregate initialization deliberately omits the optional pair in
    // arms 6/7. Each one-sided null mismatch must be rejected before any use.
    wr::GraTargetValidation validation{&raw,&p,n,e,ln,le,en,ej,an,aj,sc,thermal};
    if (arm < 6) {
      validation.spectral_number = arm == 5 ? nullptr : sn;
      validation.spectral_energy = arm == 4 ? nullptr : se;
    }
    result(c,0) = validation(tn,te);
    bool unchanged = true;
    for (int s = 0; s < 4; ++s) {
      unchanged = unchanged && en[s] == saved[s][0] && ej[s] == saved[s][1] &&
          an[s] == saved[s][2] && aj[s] == saved[s][3] && sc[s] == saved[s][4] &&
          n[s] == saved[s][5] && e[s] == saved[s][6] && sn[s] == saved[s][7] &&
          se[s] == saved[s][8] && ln[s] == saved[s][9] && le[s] == saved[s][10] &&
          tn[s] == saved[s][11] && te[s] == saved[s][12];
    }
    result(c,2) = unchanged;
    // Exercise the actual correction/finalizer publication operations on an
    // independent copy. Expected validity below is NOT obtained from them.
    wr::ApplySpectralCorrection(4,p.opacity_corr_fac_max,
        arm >= 6 ? n : sn,arm >= 6 ? e : se,
        p.gra_target_mean ? tn : ln,p.gra_target_mean ? te : le,en,ej,an,aj,sc);
    result(c,1) = wr::FinalizeKirchhoffCoefficients(4,true,tn,te,en,ej,an,aj,sc);
    for (int s = 0; s < 4; ++s) {
      result(c,3+5*s) = en[s]; result(c,4+5*s) = ej[s];
      result(c,5+5*s) = an[s]; result(c,6+5*s) = aj[s];
      result(c,7+5*s) = sc[s];
    }
  });
  const auto host = Snapshot(result);
  const Real limit = std::numeric_limits<Real>::max();
  const char *arms[8] = {"spectral-reject/current-accept", "spectral-accept/current-reject",
      "huge-lab-number/finite-physical-number", "finite-lab-number/huge-physical-number",
      "number-only-null-mismatch", "energy-only-null-mismatch",
      "omitted-pair/current-reject", "omitted-pair/current-accept"};
  for (int c = 0; c < count; ++c) {
    const int route = c/8, arm = c%8;
    const Real mean = arm == 0 || arm == 6 ? 12. : (arm == 2 ? 2. : 8.);
    const Real ratio = mean/10.;
    const Real clamped = std::max(1./1.75,std::min(ratio,1.75));
    const Real factor = clamped*clamped;
    const bool near_limit = arm < 2 || arm >= 6;
    const Real base_abs = near_limit ? limit*.6 : 1.;
    // Division bounds avoid computing an overflowing product as our oracle.
    const bool finite_abs = base_abs <= limit/factor;
    const bool finite_emission = finite_abs && base_abs*factor <= limit/1.5;
    const bool coefficients_valid = finite_abs && finite_emission;
    const Real physical_n = arm == 3 ? limit/4. : 1.;
    const Real physical_ratio = physical_n/1.5;
    const bool thermal_valid = (arm != 2 && arm != 3) ||
        (physical_ratio <= limit/physical_ratio &&
         (1./15.) <= limit/physical_ratio);
    const bool expected = coefficients_valid && thermal_valid && arm != 4 && arm != 5;
    const std::string label = "target preview/route="+std::to_string(route)+
        "/"+arms[arm];
    Check((host(c,0) != 0.) == expected,label+" independent coefficient/thermal bounds");
    Check((host(c,1) != 0.) == coefficients_valid,label+" publication finite boundary");
    Check(host(c,2) != 0.,label+" read-only trial inputs");
    bool fields = true;
    for (int s = 0; s < 4; ++s) {
      const Real absorption_n = factor*(s == 0 ? base_abs : 1.);
      const Real absorption_e = 2.*factor;
      const Real emission_n = s < 2 ? absorption_n*1.5 : factor;
      const Real emission_e = s < 2 ? absorption_e*15. : 2.*factor;
      const Real want_abs_n = s < 2 ? absorption_n : emission_n/1.5;
      const Real want_abs_e = s < 2 ? absorption_e : emission_e/15.;
      fields = fields &&
          (s == 0 && !finite_emission ? std::isinf(host(c,3+5*s)) :
           Close(host(c,3+5*s),emission_n)) &&
          Close(host(c,4+5*s),emission_e) && Close(host(c,5+5*s),want_abs_n) &&
          Close(host(c,6+5*s),want_abs_e) && Close(host(c,7+5*s),factor);
    }
    Check(fields,label+" independent five-field spectral-factor oracle");
  }
}

// Diagonal spatial geometry; z is the stored W*v primitive, not coordinate u^i.
// Shift and lapse affect the spacetime metric, never the direct spectral ratio.
struct Case {
  std::string name;
  std::array<Real, 3> metric, z, shift;
  Real lapse, contraction, factor_max;
  std::array<Real, 4> mean_ratio;
};

// Reference columns use EOS number-density units, code energy-density units,
// inverse code length for opacities, and density/code time for emissivities.
enum ReferenceColumn {
  kLocalN, kLocalE, kTargetN, kTargetE, kRawEtaN, kRawEtaE,
  kRawAbsN, kRawAbsE, kRawScat, kColumns
};

template<class Logs> class Fixture {
 public:
  ParameterInput pin;
  std::unique_ptr<Mesh> mesh;
  dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Logs>, Primitive::ResetFloor> *hyd;
  DvceArray2D<Real> reference;
  DvceArray1D<Real> raw_thermal;

  Fixture(const std::string &deck, const std::string &table) {
    std::ifstream stream(deck);
    if (!stream) throw std::runtime_error("missing input deck " + deck);
    pin.LoadFromStream(stream);
    pin.SetString("radiation_m1", "opacity_type", "weakrates");
    pin.SetString("radiation_m1", "closure_fun", "eddington");
    pin.SetBoolean("coord", "minkowski", true);
    pin.SetString("mhd", "units", "geometric_solar");
    pin.SetString("mhd", "table", table);
    pin.SetBoolean("mhd", "fixed", false);  // Producer reads the temperature cache.
    pin.SetBoolean("mhd", "use_NQT", std::is_same<Logs, Primitive::NQTLogs>::value);
    pin.SetInteger("mesh", "nx1", 8);
    pin.SetInteger("meshblock", "nx1", 8);
    pin.SetInteger("mesh", "nghost", 2);
    pin.SetReal("weakrates", "rho_min_cgs", 0.);
    pin.SetReal("weakrates", "temp_min_mev", 0.);
    pin.SetReal("weakrates", "opacity_corr_fac_max", 1.75);
    pin.SetBoolean("weakrates", "use_partial_equilibrium", false);
    pin.SetBoolean("weakrates", "gra_equilibrium_targets", false);
    pin.SetBoolean("weakrates", "use_electron_thermal_emission", true);
    // Leave order/target-mean absent to cover their construction-time defaults.
    mesh = std::make_unique<Mesh>(&pin);
    mesh->BuildTreeFromScratch(&pin);
    mesh->AddCoordinatesAndPhysics(&pin);
    mesh->dt = 0.;
    hyd = dynamic_cast<dyngr::DynGRMHDPS<Primitive::EOSCompOSE<Logs>,
                                      Primitive::ResetFloor> *>(mesh->pmb_pack->pdyngr);
    if (!hyd) throw std::runtime_error("typed CompOSE fixture mismatch");
    if (mesh->pmb_pack->nmb_thispack != 1 || Rad().nspecies != 4 || Rad().nvars != 5)
      throw std::runtime_error("fixture requires one block and four WeakRates groups");
    Check(!Rad().weakrates_params.gra_opacity_order &&
          !Rad().weakrates_params.gra_target_mean, "constructor local/early defaults");
    reference = DvceArray2D<Real>("independent_raw_local_selected", 4, kColumns);
    raw_thermal = DvceArray1D<Real>("independent_raw_thermal", 2);
    InitializeMatter();
  }

  RadiationM1 &Rad() { return *mesh->pmb_pack->pradm1; }

  void InitializeMatter() {
    auto w = mesh->pmb_pack->pmhd->w0;
    auto temperature = hyd->temperature;
    auto ref = reference;
    auto thermal = raw_thermal;
    auto &eos = hyd->eos.ps.GetEOSMutable();
    const auto cu = eos.GetCodeUnitSystem(), eu = eos.GetEOSUnitSystem();
    const auto cg = Primitive::MakeCGS();
    const Real density_conversion = cu.MassDensityConversion(cg);
    const wr::UnitConversions units{cu.LengthConversion(cg), cu.TimeConversion(cg),
        eu.NumberDensityConversion(cg), cu.EnergyDensityConversion(cg)};
    const Real mb = eos.GetBaryonMass();
    const Real baryon_cgs = mb*density_conversion/units.number_cgs_per_eos;
    const Real rho = pin.GetReal("problem", "rho");
    const Real ye_input = pin.GetReal("problem", "Y_e");
    const Real nb_lo = eos.GetMinimumDensity(), nb_hi = eos.GetMaximumDensity();
    const Real t_lo = eos.GetMinimumTemperature()*eu.TemperatureConversion(cu);
    const Real t_hi = eos.GetMaximumTemperature()*eu.TemperatureConversion(cu);
    const Real y_lo = eos.GetMinimumSpeciesFraction(0);
    const Real y_hi = eos.GetMaximumSpeciesFraction(0);
    const int cells = w.extent(4);
    Kokkos::deep_copy(w, 0.);
    Kokkos::deep_copy(temperature, 0.);
    // EOS tables and microscopic formulas are device-only. This reference never
    // calls the spectral correction, finalizer, or target-selection algorithms.
    Kokkos::parallel_for("opacity_mean_matter_reference",
        Kokkos::RangePolicy<DevExeSpace>(0, 1), KOKKOS_LAMBDA(int) {
      const Real nb = Kokkos::fmax(nb_lo, Kokkos::fmin(rho/mb, nb_hi));
      const Real t = Kokkos::fmax(t_lo, Kokkos::fmin(10., t_hi));
      const Real y = Kokkos::fmax(y_lo, Kokkos::fmin(ye_input, y_hi));
      Real ya[3] = {y, 0., 0.};
      const Real pressure = eos.GetPressure(nb, t, ya);
      for (int i = 0; i < cells; ++i) {
        w(0,IDN,0,0,i) = rho; w(0,IYF,0,0,i) = y;
        w(0,IPR,0,0,i) = pressure; temperature(0,0,0,0,i) = t;
      }
      const Real mu_b = eos.GetBaryonChemicalPotential(nb,t,ya);
      const Real mu_q = eos.GetChargeChemicalPotential(nb,t,ya);
      const Real mu_le = eos.GetElectronLeptonChemicalPotential(nb,t,ya);
      const wr::State state = wr::MakeState(nb*units.number_cgs_per_eos*baryon_cgs,
          t,baryon_cgs,eos.GetNeutronFraction(nb,t,ya),eos.GetProtonFraction(nb,t,ya),
          eos.GetHeavyFraction(nb,t,ya),eos.GetMassNumber(nb,t,ya),
          eos.GetChargeNumber(nb,t,ya),mu_b,mu_b+mu_q,mu_le-mu_q);
      auto raw = wr::ComputeRawCoefficients(state);
      auto local = wr::ComputeEquilibriumDensities(state);
      // With both predictors off, the selected target is fixed local matter
      // T,Ye evaluated with mu_le/T. Keep it separate from the rate-state mean.
      auto target = wr::ComputeEquilibriumDensities(t,mu_le/t);
      wr::ConvertToCodeUnits(units,raw);
      wr::ConvertToCodeUnits(units,local);
      wr::ConvertToCodeUnits(units,target);
      for (int s = 0; s < 4; ++s) {
        const int slot = s < 2 ? s : wr::kNuX;
        const Real split = s < 2 ? 1. : .5;
        ref(s,kLocalN) = split*local.number[slot];
        ref(s,kLocalE) = split*local.energy[slot];
        ref(s,kTargetN) = split*target.number[slot];
        ref(s,kTargetE) = split*target.energy[slot];
        ref(s,kRawEtaN) = split*raw.eta_0[slot];
        ref(s,kRawEtaE) = split*raw.eta_1[slot];
        ref(s,kRawAbsN) = raw.abs_0[slot];
        ref(s,kRawAbsE) = raw.abs_1[slot];
        ref(s,kRawScat) = raw.scat_1[slot];
      }
      thermal(0) = raw.eta_0_non_cc[wr::kNuE];
      thermal(1) = raw.eta_1_non_cc[wr::kNuE];
    });
    Kokkos::fence();
  }

  void Initialize(const Case &c) {
    auto &r = Rad();
    auto *pack = mesh->pmb_pack;
    auto adm = pack->padm->adm;
    auto w = pack->pmhd->w0;
    const int cells = w.extent(4);
    // Only POD values cross the device boundary, not Case's std::string.
    const Real mx = c.metric[0], my = c.metric[1], mz = c.metric[2];
    const Real zx = c.z[0], zy = c.z[1], zz = c.z[2];
    const Real bx = c.shift[0], by = c.shift[1], bz = c.shift[2], lapse = c.lapse;
    Kokkos::parallel_for("opacity_mean_geometry",
        Kokkos::RangePolicy<DevExeSpace>(0, cells), KOKKOS_LAMBDA(int i) {
      adm.alpha(0,0,0,i) = lapse;
      adm.psi4(0,0,0,i) = Kokkos::cbrt(mx*my*mz);
      for (int a = 0; a < 3; ++a) {
        adm.beta_u(0,a,0,0,i) = a == 0 ? bx : (a == 1 ? by : bz);
        for (int b = a; b < 3; ++b) {
          adm.g_dd(0,a,b,0,0,i) = a == b ? (a == 0 ? mx : (a == 1 ? my : mz)) : 0.;
          adm.vK_dd(0,a,b,0,0,i) = 0.;
        }
      }
      w(0,IVX,0,0,i) = zx; w(0,IVY,0,0,i) = zy; w(0,IVZ,0,0,i) = zz;
    });
    const auto ref = Snapshot(reference);
    auto u = Snapshot(r.u0);
    const Real W = std::sqrt(1.+mx*zx*zx+my*zy*zy+mz*zz*zz);
    const Real volume = std::sqrt(mx*my*mz);
    const std::array<Real, 3> v = {zx/W, zy/W, zz/W};
    Real v2 = 0.;
    for (int a = 0; a < 3; ++a) v2 += c.metric[a]*v[a]*v[a];
    if (v2 == 0. && c.contraction != 0.)
      throw std::runtime_error("rest fixture cannot have Fv/E != 0");
    for (int i = 0; i < cells; ++i) for (int s = 0; s < 4; ++s) {
      // All three stored moments share sqrt(det(gamma)); no lapse densitization.
      const Real N = volume*.25*ref(s,kLocalN);
      const Real E = N*c.mean_ratio[s]*(ref(s,kLocalE)/ref(s,kLocalN))/
                     (W*(1.-c.contraction));
      u(0,CombinedIdx(s,M1_N_IDX,r.nvars),0,0,i) = N;
      u(0,CombinedIdx(s,M1_E_IDX,r.nvars),0,0,i) = E;
      for (int a = 0; a < 3; ++a)
        u(0,CombinedIdx(s,M1_FX_IDX+a,r.nvars),0,0,i) =
            v2 == 0. ? 0. : E*c.contraction*c.metric[a]*v[a]/v2;
    }
    Kokkos::deep_copy(r.u0,u);
    Kokkos::deep_copy(r.u1,r.u0);
    Kokkos::deep_copy(r.chi,1./3.);
    Kokkos::deep_copy(r.radiation_mask,false);
    for (auto view : {r.eta_0,r.eta_1,r.abs_0,r.abs_1,r.scat_1,r.weakrates_thermal})
      Kokkos::deep_copy(view,-77.);
    Kokkos::deep_copy(pack->pmhd->bcc0,0.);
    Kokkos::deep_copy(pack->pmhd->b0.x1f,0.);
    Kokkos::deep_copy(pack->pmhd->b0.x2f,0.);
    Kokkos::deep_copy(pack->pmhd->b0.x3f,0.);
    hyd->eos.PrimToCons(pack->pmhd->w0,pack->pmhd->bcc0,pack->pmhd->u0,
                        0,cells-1,0,0,0,0);
    Kokkos::fence();
  }

  std::vector<Real> Coefficients() {
    auto &r = Rad();
    std::vector<Real> result;
    for (auto view : {r.eta_0,r.eta_1,r.abs_0,r.abs_1,r.scat_1,r.weakrates_thermal}) {
      const auto values = Values(view);
      result.insert(result.end(),values.begin(),values.end());
    }
    return result;
  }

  std::vector<Real> Evolved() {
    auto &r = Rad();
    auto &m = *mesh->pmb_pack->pmhd;
    std::vector<Real> result;
    for (auto view : {r.u0,r.u1,m.u0,m.w0,m.bcc0,hyd->temperature}) {
      const auto values = Values(view);
      result.insert(result.end(),values.begin(),values.end());
    }
    for (auto view : {m.b0.x1f,m.b0.x2f,m.b0.x3f}) {
      const auto values = Values(view);
      result.insert(result.end(),values.begin(),values.end());
    }
    return result;
  }

  void Run(const Case &c, bool order, bool selected, bool masked,
           const std::string &prefix) {
    Initialize(c);
    auto &r = Rad();
    r.weakrates_params.gra_opacity_order = order;
    r.weakrates_params.gra_target_mean = selected;
    r.weakrates_params.opacity_corr_fac_max = c.factor_max;
    const auto ind = mesh->mb_indcs;
    const int mask_i = ind.is+1;
    if (masked) {
      auto mask = r.radiation_mask;
      Kokkos::parallel_for("opacity_mean_mask",
          Kokkos::RangePolicy<DevExeSpace>(0,1), KOKKOS_LAMBDA(int) {
        mask(0,0,0,mask_i) = true;
      });
    }
    const auto evolved_before = Evolved();
    const auto incoming = Snapshot(r.u0);
    const auto ref = Snapshot(reference);
    const auto thermal_ref = Snapshot(raw_thermal);
    const std::string label = prefix+"/"+c.name+"/order="+std::to_string(order)+
        "/selected="+std::to_string(selected)+(masked ? "/masked" : "");
    Check(r.CalcOpacityWeakRates(nullptr,1) == TaskStatus::complete, label+" status");
    Check(SameBits(evolved_before,Evolved()), label+" evolved radiation/matter unchanged");
    const Real W = std::sqrt(1.+c.metric[0]*c.z[0]*c.z[0]+
        c.metric[1]*c.z[1]*c.z[1]+c.metric[2]*c.z[2]*c.z[2]);
    const std::array<Real, 3> v = {c.z[0]/W,c.z[1]/W,c.z[2]/W};
    std::array<std::array<Real, 5>, 4> expected{};
    bool setup = true;
    for (int s = 0; s < 4; ++s) {
      const Real E = incoming(0,CombinedIdx(s,M1_E_IDX,r.nvars),0,0,ind.is);
      const Real N = incoming(0,CombinedIdx(s,M1_N_IDX,r.nvars),0,0,ind.is);
      Real Fv = 0.;
      for (int a = 0; a < 3; ++a)
        Fv += incoming(0,CombinedIdx(s,M1_FX_IDX+a,r.nvars),0,0,ind.is)*v[a];
      // Independent host oracle from prescribed raw E,F,N,gamma_ij and W*v.
      // No closure, number conversion, or production spectral helper enters it.
      const Real mean = W*(E-Fv)/N;
      const Real bn = ref(s,selected ? kTargetN : kLocalN);
      const Real be = ref(s,selected ? kTargetE : kLocalE);
      const Real ratio = mean*(bn/be);
      const Real clamped = std::max(1./c.factor_max,std::min(ratio,c.factor_max));
      const Real factor = clamped*clamped;
      setup = setup && N > 0. && E > 0. && bn > 0. && be > 0. &&
          Close(Fv/E,c.contraction) &&
          Close(mean,c.mean_ratio[s]*ref(s,kLocalE)/ref(s,kLocalN)) &&
          ref(s,kRawScat) > 0.;
      const Real tn = ref(s,kTargetN), te = ref(s,kTargetE);
      if (s < 2) {
        expected[s][2] = factor*ref(s,kRawAbsN);
        expected[s][3] = factor*ref(s,kRawAbsE);
        expected[s][0] = expected[s][2]*tn;
        expected[s][1] = expected[s][3]*te;
        setup = setup && ref(s,kRawAbsN) > 0. && ref(s,kRawAbsE) > 0.;
      } else {
        expected[s][0] = factor*ref(s,kRawEtaN);
        expected[s][1] = factor*ref(s,kRawEtaE);
        expected[s][2] = expected[s][0]/tn;
        expected[s][3] = expected[s][1]/te;
        setup = setup && ref(s,kRawEtaN) > 0. && ref(s,kRawEtaE) > 0.;
      }
      expected[s][4] = factor*ref(s,kRawScat);
    }
    Check(setup, label+" positive independent oracle and prescribed stream");
    int field = 0;
    const char *fields[5] = {"eta_0", "eta_1", "abs_0", "abs_1", "scat_1"};
    for (auto view : {r.eta_0,r.eta_1,r.abs_0,r.abs_1,r.scat_1}) {
      const auto host = Snapshot(view);
      for (int s = 0; s < 4; ++s) {
        bool pass = true;
        for (int i = 0; i < static_cast<int>(host.extent(4)); ++i) {
          const bool evaluated = i >= ind.is-1 && i <= ind.ie+1;
          const Real value = !evaluated ? -77. :
              (masked && i == mask_i ? 0. : expected[s][field]);
          pass = pass && Close(host(0,s,0,0,i),value);
        }
        Check(pass,label+"/"+fields[field]+"/species="+std::to_string(s));
      }
      ++field;
    }
    const auto thermal = Snapshot(r.weakrates_thermal);
    bool thermal_pass = thermal_ref(0) > 0. && thermal_ref(1) > 0.;
    for (int i = 0; i < static_cast<int>(thermal.extent(4)); ++i)
      for (int f = 0; f < wr::kNumThermalFields; ++f) {
        const bool evaluated = i >= ind.is-1 && i <= ind.ie+1;
        const Real raw = f < 2 ? thermal_ref(f) :
            (f < 4 ? ref(f-2,kTargetN) : ref(f-4,kTargetE));
        const Real value = !evaluated ? -77. : (masked && i == mask_i ? 0. : raw);
        thermal_pass = thermal_pass && Close(thermal(0,f,0,0,i),value);
      }
    Check(thermal_pass,label+" uncorrected raw thermal r/q and selected references");
    // Stage 2 must hold coefficients even when the incoming state has changed.
    const auto held = Coefficients();
    Kokkos::deep_copy(r.u0,-3.);
    Kokkos::deep_copy(hyd->temperature,20.);
    const auto stage2_evolved = Evolved();
    Check(r.CalcOpacityWeakRates(nullptr,2) == TaskStatus::complete,label+" stage2 status");
    Check(SameBits(held,Coefficients()),label+" stage2 five-field/thermal bitwise hold");
    Check(SameBits(stage2_evolved,Evolved()),label+" stage2 evolved state unchanged");
    // Restore the independent matter/cache setup before the next case.
    InitializeMatter();
  }
};

template<class Logs> void Matrix(const std::string &deck, const std::string &table,
                                 const std::string &logs) {
  Fixture<Logs> fixture(deck,table);
  const std::array<Real,4> means = {1.2,.85,1.4,.7};
  const std::array<Real,4> clamps = {.2,3.,.4,2.5};
  const std::array<Real,3> flat = {1.,1.,1.}, curved = {1.7,.8,1.3};
  const std::array<Real,3> zero = {0.,0.,0.}, moving = {.4,.12,-.07};
  const std::array<Real,3> shift = {.08,-.03,.02};
  const std::vector<Case> cases = {
      {"rest",flat,zero,zero,1.,0.,1.75,means},
      {"counter",flat,moving,zero,1.,-.1,1.75,means},
      {"co",flat,moving,zero,1.,.1,1.75,means},
      {"metric-counter",curved,moving,shift,.73,-.1,1.75,means},
      {"metric-co",curved,moving,shift,.73,.1,1.75,means},
      {"metric-rest",curved,zero,shift,.73,0.,1.75,means},
      {"lower-upper-clamps",curved,moving,shift,.73,-.1,1.75,clamps},
      {"unity-clamp",curved,moving,shift,.73,-.1,1.,means}};
  for (bool order : {false,true}) for (bool selected : {false,true}) {
    for (const auto &c : cases) fixture.Run(c,order,selected,false,logs);
    fixture.Run(cases[3],order,selected,true,logs);
  }
}
}  // namespace

int main(int argc, char **argv) {
  static_assert(sizeof(Real) == 8,"direct opacity mean fixture requires double precision");
  if (argc != 4) {
    std::cerr << "FAIL usage: native DECK NORMAL_TABLE NQT_TABLE\n";
    return 2;
  }
  global_variable::my_rank = 0;
  global_variable::nranks = 1;
  Kokkos::initialize();
  try {
    HelperCases();
    NumberConversionCases();
    ValidationCases();
    // Each small mesh and its EOS Views are released before constructing the next.
    Matrix<Primitive::NormalLogs>(argv[1],argv[2],"NormalLogs");
    Matrix<Primitive::NQTLogs>(argv[1],argv[3],"NQTLogs");
  } catch (const std::exception &e) {
    Check(false,std::string("native fixture setup: ")+e.what());
  }
  Kokkos::finalize();
  std::cout << (failures ? "FAIL " : "PASS ") << "weakrates opacity mean native suite\n";
  return failures ? 1 : 0;
}
