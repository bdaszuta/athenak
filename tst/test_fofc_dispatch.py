"""Exercise the literal host FOFC dispatch with tagged correction calls.

Usage: python tst/test_fofc_dispatch.py [--source-only]
The source-only mode checks scalar option/flux/PP wiring without compiling.
The default mode also compiles and exercises the literal host FOFC dispatch.
Neither mode checks numerical kernels, admissibility or mesh writeback.
The production enum and dispatch block are extracted without rewriting.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile


def check_scalar_pp_baseline(root):
    """Check selectable scalar baseline wiring, not numerical positivity."""
    source = (root / "src/dyn_grmhd/dyn_grmhd_fofc.cpp").read_text()
    guard = "  if (use_fofc_ && scalar_pplimiter && nscal_ > 0) {"
    assert source.count(guard) == 1
    start = source.index(guard)
    end = source.index("  // reset FOFC flag", start)
    pp = source[start:end]
    assert "#if 0" not in source[:end], "scalar PP must not be disabled"
    assert pp.count("if (scalar_upwind_) {") == 3
    assert "InsertFluxes(" not in pp, "scalar PP must not replace fluid fluxes"
    assert pp.count("flx_lo[n] = flux[CYD + n];") == 3
    for direction, axis, left, normal in (
        (1, "IVX", "k, j, i-1", "IBX"),
        (2, "IVY", "k, j-1, i", "IBY"),
        (3, "IVZ", "k-1, j, i", "IBZ"),
    ):
        assert f"ExtractPrimitives(wl, w0_, temperature_, eos_, nmhd_, nscal_, m, {left});" in pp
        assert f"bl[{normal}] = br[{normal}] = b0_.x{direction}f(m, k, j, i);" in pp
        assert f"adm::Face{direction}Metric(" in pp
        for solver in ("LLF", "HLLE"):
            assert f"SingleState{solver}_DYNGR<{axis}>(" in pp
        assert f"flx{direction}(m,nmhd_+n,k,j,i) = flx_lo[n]" in pp
        assert not re.search(rf"flx{direction}\(m,IDN,k,j,i\)\s*=(?!=)", pp)
    config = (root / "src/dyn_grmhd/dyn_grmhd.cpp").read_text()
    assert 'scalar_upwind = pin->GetOrAddBoolean("mhd", "scalar_upwind", true);' in config
    assert pp.count("rsolver_method_ == DynGRMHD_RSolver::llf_dyngr") == 3
    assert pp.count("rsolver_method_ == DynGRMHD_RSolver::hlle_dyngr") == 3
    print("PASS selectable scalar PP FOFC baseline source wiring")


def check_scalar_transport(root):
    """Check all selectable producers and correction gates, not numerical fluxes."""
    directory = root / "src/dyn_grmhd"
    fluxes = (directory / "dyn_grmhd_fluxes.cpp").read_text()
    correction = (directory / "dyn_grmhd_fofc.cpp").read_text()
    assert fluxes.count("if (scalar_upwind_ && nvars > nhyd) {") == 3
    assert "#if 0" not in fluxes
    assert fluxes.count("nh, ns, scalar_upwind_, adm,") == 3
    assert correction.count("if (fofc_flag || fofc_excision || !scalar_upwind_) {") == 6
    assert correction.count("if (scalar_upwind_) {") == 9
    assert correction.count("if (fofc_flag || (use_fofc_ && fofc_scal_(m,n,k,j,i))) {") == 6
    # Scalar-only correction must not publish the candidate fluid flux.
    guarded = re.findall(
        r"if \(fofc_flag \|\| fofc_excision\) \{\s*InsertFluxes\(", correction)
    assert len(guarded) == correction.count("InsertFluxes(") == 6
    for direction, axis, left_index in (
        (1, "IVX", "k,j,i-1"),
        (2, "IVY", "k,j-1,i"),
        (3, "IVZ", "k-1,j,i"),
    ):
        assert (f"flx_lo[n] = w0_(m,nmhd_+n,{left_index}) "
                f"* flx{direction}(m,IDN,k,j,i);") in correction
        assert (f"flx_lo[n] = w0_(m,nmhd_+n,k,j,i) "
                f"* flx{direction}(m,IDN,k,j,i);") in correction
        for solver in ("LLF", "HLLE"):
            calls = re.findall(
                rf"SingleState{solver}_DYNGR<{axis}>\((.*?)\);", correction, re.S)
            assert len(calls) == 3
            assert all("nmhd_, nscal_, scalar_upwind_," in call for call in calls)
    shared = (directory / "rsolvers/flux_dyn_grmhd.hpp").read_text()
    assert shared.count("if (!scalar_upwind) {") == 1
    for side in ("l", "r"):
        assert f"cons_{side}[CYD + n] = cons_{side}[CDN]*prim_{side}[PYF + n];" in shared
        assert f"flux_{side}[CYD + n] = flux_{side}[CDN]*prim_{side}[PYF + n];" in shared
    for solver, count in (("llf", 2), ("hlle", 4)):
        code = (directory / f"rsolvers/{solver}_dyn_grmhd.hpp").read_text()
        assert code.count("if (!scalar_upwind) {") == count
        calls = re.findall(r"SingleStateFlux<ivx>\((.*?)\);", code, re.S)
        assert len(calls) == 2
        assert all("scalar_upwind" in call for call in calls)
    print("PASS scalar option, normal flux and FOFC source wiring")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true",
                        help="check scalar source wiring without compiling or running C++")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    check_scalar_pp_baseline(root)
    check_scalar_transport(root)
    if args.source_only:
        return
    header = (root / "src/dyn_grmhd/dyn_grmhd.hpp").read_text()
    source = (root / "src/dyn_grmhd/dyn_grmhd_fluxes.cpp").read_text()
    enum_match = re.search(r"enum class DynGRMHD_RSolver \{[^}]+\};", header)
    assert enum_match is not None
    enum = enum_match.group()
    guard = "  if (pmy_pack->pmhd->use_fofc || pmy_pack->pcoord->coord_data.bh_excise) {"
    assert source.count(guard) == 1
    start = source.index(guard)
    end = source.index("\n\n  return TaskStatus::complete;", start)
    dispatch = source[start:end]
    program = r'''
#include <iostream>
#include <initializer_list>
ENUM
struct MHD { bool use_fofc; };
struct Coordinates { struct { bool bh_excise; } coord_data; };
struct Pack { MHD *pmhd; Coordinates *pcoord; };
struct Probe {
  Pack *pmy_pack;
  DynGRMHD_RSolver fofc_method;
  int calls = 0, selected = -1;
  template<DynGRMHD_RSolver method>
  void FOFC(void *, int) { ++calls; selected = static_cast<int>(method); }
  template<DynGRMHD_RSolver rsolver_method_>
  void Run(void *pdriver, int stage) {
DISPATCH
  }
};
template<DynGRMHD_RSolver main_solver>
bool Check(DynGRMHD_RSolver correction, bool enabled, bool excision) {
  MHD mhd{enabled};
  Coordinates coord{{excision}};
  Pack pack{&mhd, &coord};
  Probe probe{&pack, correction};
  probe.Run<main_solver>(nullptr, 1);
  const bool active = enabled || excision;
  const int expected = active ? static_cast<int>(correction) : -1;
  const bool pass = probe.calls == static_cast<int>(active) && probe.selected == expected;
  std::cout << (pass ? "PASS " : "FAIL ")
            << "main=" << static_cast<int>(main_solver)
            << " correction=" << static_cast<int>(correction)
            << " fofc=" << enabled << " excision=" << excision
            << " calls=" << probe.calls << " selected=" << probe.selected
            << " expected=" << expected << '\n';
  return pass;
}
int main() {
  int failed = 0;
  for (auto correction : {DynGRMHD_RSolver::llf_dyngr, DynGRMHD_RSolver::hlle_dyngr}) {
    for (bool enabled : {false, true}) {
      for (bool excision : {false, true}) {
        failed += !Check<DynGRMHD_RSolver::llf_dyngr>(correction, enabled, excision);
        failed += !Check<DynGRMHD_RSolver::hlle_dyngr>(correction, enabled, excision);
      }
    }
  }
  return failed ? 1 : 0;
}
'''.replace("ENUM", enum).replace("DISPATCH", dispatch)
    with tempfile.TemporaryDirectory(prefix="hermes-verify-fofc-dispatch-") as tmp:
        cpp = Path(tmp) / "dispatch.cpp"
        exe = Path(tmp) / "dispatch"
        cpp.write_text(program)
        subprocess.run(["c++", "-std=c++17", str(cpp), "-o", str(exe)], check=True)
        subprocess.run([str(exe)], check=True)


if __name__ == "__main__":
    main()
