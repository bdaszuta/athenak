"""Exercise the literal host FOFC dispatch with tagged correction calls.

Usage: python tst/test_fofc_dispatch.py
This checks solver selection and activation, not numerical FOFC kernels or mesh
writeback. The production enum and dispatch block are extracted without rewriting.
"""
from pathlib import Path
import re
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parent.parent
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
