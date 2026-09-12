#!/usr/bin/env python3
"""Focused host checks for refinement, excision and output; no AthenaK build.

Run: python3 -B tst/test_geometry_output_findings.py
Requires only the Python standard library and a C++17 compiler (CXX or c++).
Production methods/snippets are extracted unchanged and compiled serially with
UBSan. Small stand-ins supply mesh storage and streams, not the tested logic.
These checks do not exercise Kokkos/MPI, native binary I/O, or restart loading;
restart timing is tested by resuming from an explicitly restored clock/position.
All generated files live in a temporary /tmp/hermes-verify-* directory.
"""

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def source(path):
    return (ROOT / "src" / path).read_text(encoding="utf-8")


def between(text, start, end):
    return text.split(start, 1)[1].split(end, 1)[0]


def definition(text, signature):
    """Extract a definition, counting braces in these known simple functions."""
    start = text.index(signature)
    opening = text.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


SUPPORT = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <new>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using Real = double;
constexpr int NDIM = 3;
#define SQ(X) ((X)*(X))
namespace global_variable { int my_rank = 0; }
struct HostMemSpace {};
struct DevExeSpace {};
template <class T> struct HostStorage {
  std::vector<T> values;
  T &h_view(int i) { return values.at(i); }
  int modifications = 0, syncs = 0;
  template <class Space> void modify() { ++modifications; }
  template <class Space> void sync() { ++syncs; }
  // The HIP branch uses named ownership operations for aliased memory spaces.
  void modify_host() { ++modifications; }
  void sync_device() { ++syncs; }
};
struct RegionSize { Real x1min, x1max, x2min, x2max, x3min, x3max; };
struct MeshBlock { HostStorage<RegionSize> mb_size; };
struct MeshRefinement { HostStorage<int> refine_flag; };
struct LogicalLocation { int level; };
struct MeshBlockPack;
struct Mesh {
  MeshRefinement *pmr;
  std::vector<int> gids_eachrank{1};
  std::vector<LogicalLocation> lloc_eachmb{{2}, {2}};
  int root_level = 2;
  Real time = 0.0, dt = 0.125;
  int ncycle = 0;
  MeshBlockPack *pmb_pack;
};
struct CompactObjectTracker {
  Mesh const *pmesh;
  Real pos[NDIM]{}, vel[NDIM]{1.0, -2.0, 0.5};
  int out_every = 1, reflevel = 1;
  Real radius = 0.5;
  std::ostringstream ofile;
  Real GetPos(int a) const { return pos[a]; }
  int GetReflevel() const { return reflevel; }
  Real GetRadius() const { return radius; }
  void WriteTracker();
  void AdvanceODE();
};
struct Z4c { std::vector<CompactObjectTracker *> ptracker; };
struct RadiationM1 {
  int nspecies = 4, nvars = 5;
  int u0 = 0, chi = 0, eta_0 = 0, abs_0 = 0, eta_1 = 0, abs_1 = 0, scat_1 = 0;
};
struct MeshBlockPack {
  Mesh *pmesh;
  MeshBlock *pmb;
  Z4c *pz4c;
  int nmb_thispack = 1;
  RadiationM1 *pradm1 = nullptr;
  void *phydro = nullptr, *pmhd = nullptr, *pdyngr = nullptr, *pturb = nullptr;
  void *prad = nullptr, *padm = nullptr, *ptmunu = nullptr;
  void *ppart = nullptr, *pgrav = nullptr;
};
namespace z4c {
struct Z4c_AMR {
  enum Method { Trivial, Tracker, Chi, dChi } method = Tracker;
  std::vector<Real> radius{0.5};
  std::vector<int> reflevel{1};
  void Refine(MeshBlockPack *pmy_pack);
  void RefineTracker(MeshBlockPack *pmbp);
  void RefineRadii(MeshBlockPack *pmbp);
  void RefineChiMin(MeshBlockPack *) { std::abort(); }
  void RefineDchiMax(MeshBlockPack *) { std::abort(); }
};
}
struct OutputVariableInfo {
  std::string label;
  int data_index;
  int *data_ptr;
  OutputVariableInfo(std::string name, int index, int *data)
      : label(name), data_index(index), data_ptr(data) {}
};
struct ParameterInput {
  std::set<std::string> blocks;
  std::map<std::string, bool> booleans;
  bool DoesBlockExist(std::string key) { return blocks.count(key); }
  bool GetOrAddBoolean(std::string, std::string key, bool fallback) {
    return booleans.count(key) ? booleans.at(key) : fallback;
  }
  Real GetReal(std::string, std::string) { return 0.0; }
};
'''


CHECKS = r'''
void CheckGeometry(std::string mode) {
  MeshRefinement mr;
  mr.refine_flag.values = {7, -1};  // Sentinel before this rank's first block.
  Mesh mesh;
  mesh.pmr = &mr;
  MeshBlock block;
  block.mb_size.values = {{-1, 1, -1, 1, -1, 1}};
  Z4c z4c;
  MeshBlockPack pack{&mesh, &block, &z4c};
  z4c::Z4c_AMR amr;
  auto &flags = mr.refine_flag;
  if (mode == "empty") {
    for (int flag : {-1, 0, 1}) {
      flags.values[1] = flag;
      amr.RefineTracker(&pack);
      assert(flags.values[0] == 7 && flags.values[1] == flag);
      assert(flags.modifications == 0 && flags.syncs == 0);
    }
    // Returning from the tracker criterion must not skip the radial criterion.
    flags.values[1] = 0;
    amr.Refine(&pack);
    assert(flags.values[1] == 1);
  } else if (mode == "single") {
    CompactObjectTracker tracker;
    z4c.ptracker = {&tracker};
    for (int level : {0, 1, 2}) {
      mesh.lloc_eachmb[1].level = mesh.root_level + level;
      amr.RefineTracker(&pack);
      assert(flags.values[1] == 1 - level);
    }
    mesh.lloc_eachmb[1].level = mesh.root_level;
    tracker.pos[0] = 2.0;
    amr.RefineTracker(&pack);
    assert(flags.values[1] == -1);
    tracker.pos[0] = 1.25;  // Face intersection, no corner within the radius.
    amr.RefineTracker(&pack);
    assert(flags.values[1] == 1);
    tracker.pos[0] = 0.0;
    tracker.radius = 0.0;  // Containment still refines at zero tracker radius.
    amr.RefineTracker(&pack);
    assert(flags.values[1] == 1);
    assert(flags.modifications == 6 && flags.syncs == 6);
  } else {
    const RegionSize boxes[] = {
      {-1, 1, -1, 1, -1, 1},       // Sphere enclosed by box.
      {0.25, 1, -1, 1, -1, 1},    // Face intersection.
      {0.25, 1, 0.25, 1, -1, 1},  // Edge intersection.
      {-1, -0.25, -1, 1, -1, 1},  // Negative-side face.
      {0.1, 0.2, 0.1, 0.2, 0.1, 0.2},  // Corner already inside.
      {1, 2, -1, 1, -1, 1},       // Disjoint.
      {0.5, 1, -1, 1, -1, 1}      // Tangency: strict < must stay strict.
    };
    for (int i = 0; i < 7; ++i) {
      block.mb_size.values[0] = boxes[i];
      for (int level : {0, 1, 2}) {
        mesh.lloc_eachmb[1].level = mesh.root_level + level;
        for (int old_flag : {-1, 0, 1}) {
          flags.values[1] = old_flag;
          amr.RefineRadii(&pack);
          int expected = old_flag;
          if (i < 5 && level == 0) expected = 1;
          if (i < 5 && level == 1 && old_flag == -1) expected = 0;
          assert(flags.values[0] == 7 && flags.values[1] == expected);
        }
      }
    }
    assert(flags.modifications == 63 && flags.syncs == 63);
  }
}
void CheckCoordinates() {
  // Default initialization, not value initialization, on nonzero storage.
  alignas(CoordData) unsigned char storage[sizeof(CoordData)];
  std::memset(storage, 0xa5, sizeof(storage));
  auto *data = new (storage) CoordData;
  assert(!data->bh_excise);
  ParameterInput pin;
  pin.blocks = {"z4c"};
  InitCoord(*data, &pin);  // Vacuum Z4c, default general_rel=false.
  assert(!data->bh_excise);
  pin.blocks = {"hydro"};
  InitCoord(*data, &pin);  // Newtonian AMR recreates the same coordinate defaults.
  assert(!data->bh_excise);
  pin.booleans = {{"general_rel", true}, {"excise", true}};
  InitCoord(*data, &pin);
  assert(data->bh_excise);
  pin.booleans["excise"] = false;
  InitCoord(*data, &pin);
  assert(!data->bh_excise);
  pin.blocks = {"z4c", "mhd"};
  pin.booleans = {{"excise", true}};
  InitCoord(*data, &pin);
  assert(data->bh_excise);
  pin.booleans["minkowski"] = true;
  InitCoord(*data, &pin);
  assert(!data->bh_excise);
  data->~CoordData();
}
void CheckTracker() {
  for (int cadence : {1, 3}) {
    for (int start_cycle : {0, 5}) {
      Mesh mesh;
      mesh.ncycle = start_cycle;
      mesh.time = start_cycle * mesh.dt;
      CompactObjectTracker tracker;
      tracker.pmesh = &mesh;
      tracker.out_every = cadence;
      tracker.ofile << std::setprecision(19);
      const Real origin[3] = {0.25, -0.5, 1.0};
      for (int a = 0; a < NDIM; ++a) {
        tracker.pos[a] = origin[a] + tracker.vel[a] * mesh.time;
      }
      for (int step = 0; step < 7; ++step) {
        tracker.AdvanceODE();
        tracker.WriteTracker();
        mesh.time += mesh.dt;
        ++mesh.ncycle;
      }
      std::istringstream rows(tracker.ofile.str());
      int cycle;
      Real time, pos[3], vel[3];
      for (int expected = start_cycle + 1; expected <= start_cycle + 7; ++expected) {
        if (expected % cadence != 0) continue;
        assert(rows >> cycle >> time >> pos[0] >> pos[1] >> pos[2]
                    >> vel[0] >> vel[1] >> vel[2]);
        assert(cycle == expected && time == cycle * mesh.dt);
        for (int a = 0; a < NDIM; ++a) {
          assert(pos[a] == origin[a] + tracker.vel[a] * time);
          assert(vel[a] == tracker.vel[a]);
        }
      }
      assert(!(rows >> cycle));
      auto previous = tracker.ofile.str();
      global_variable::my_rank = 1;
      tracker.WriteTracker();
      assert(previous == tracker.ofile.str());
      global_variable::my_rank = 0;
    }
  }
}
int main(int argc, char **argv) {
  assert(argc > 1);
  std::string mode = argv[1];
  if (mode == "output") {
    Mesh mesh;
    Z4c z4c;
    MeshBlockPack pack{&mesh, nullptr, &z4c};
    RadiationM1 radiation;
    pack.pradm1 = &radiation;
    mesh.pmb_pack = &pack;
    OutputParameters params{};
    params.variable = argv[2];
    params.block_name = "output73";
    params.file_type = argc > 3 ? "pdf" : "bin";
    if (argc > 3) { params.variable_2 = argv[3]; params.nbin2 = 8; }
    auto fields = ValidateAndRegister(&mesh, params);
    // A marker stands in for the writer; invalid input must never reach it.
    std::ofstream file("writer_reached.txt");
    for (const auto &field : fields) {
      file << field.label << " " << field.data_index << "\n";
    }
    std::cout << fields.size() << "\n";
  } else if (mode == "coordinates") {
    CheckCoordinates();
  } else if (mode == "tracker") {
    CheckTracker();
  } else {
    CheckGeometry(mode);
  }
}
'''


class GeometryOutputFindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="hermes-verify-geometry-", dir="/tmp")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.work = Path(cls.temp.name)
        amr = source("z4c/z4c_amr.cpp")
        tracker = source("z4c/compact_object_tracker.cpp")
        coord = source("coordinates/coordinates.hpp")
        output = source("outputs/basetype_output.cpp")
        header = source("outputs/outputs.hpp")
        registry = definition(header, "static const char *var_choice") + ";"
        cls.choices = re.findall(r'"([^"\n]+)"', re.sub(r"//[^\n]*", "", registry))
        count_macro = re.search(r"#define NOUTPUT_CHOICES \d+", header)
        assert count_macro is not None
        chunks = [SUPPORT, definition(coord, "enum class ExcisionScheme") + ";",
                  definition(coord, "struct CoordData {") + ";",
                  definition(header, "struct OutputParameters {") + ";",
                  count_macro.group(), registry]
        chunks.append("namespace radiationm1 {" + definition(
            source("radiation_m1/radiation_m1_helpers.hpp"), "int CombinedIdx(") + "}")
        for name in ("Refine", "RefineTracker", "RefineRadii"):
            chunks.append("namespace z4c {" + definition(amr, f"void Z4c_AMR::{name}(") + "}")
        chunks.append(definition(tracker, "void CompactObjectTracker::WriteTracker()"))
        chunks.append("void CompactObjectTracker::AdvanceODE() {" + definition(
            tracker, "for (int a = 0; a < NDIM; ++a)") + "}")
        initialization = between(source("coordinates/coordinates.cpp"),
                                 "// Check for relativistic dynamics", "    if (coord_data.bh_excise)")
        chunks.append("void InitCoord(CoordData &coord_data, ParameterInput *pin) {\n"
                      "bool is_dynamical_relativistic = false;\n"
                      "bool is_special_relativistic = false, is_general_relativistic = false;\n"
                      + initialization + "}}")
        validation = between(output, "// check for valid choice of variables",
                             "// Now load STL vector of output variables")
        registration = between(output, "// radiation m1 lab energy density",
                               "// lower time-component of the fluid four-velocity")
        chunks.append("std::vector<OutputVariableInfo> ValidateAndRegister("
                      "Mesh *pm, OutputParameters out_params) {\n" + validation
                      + "std::vector<OutputVariableInfo> outvars; int derived_var = 0;\n"
                      + registration + "return outvars; }")
        chunks.append(CHECKS)
        cpp = cls.work / "geometry_output_findings.cpp"
        cpp.write_text("\n".join(chunks), encoding="ascii")
        cls.exe = cls.work / "geometry_output_findings"
        command = shlex.split(os.environ.get("CXX", "c++")) + [
            "-std=c++17", "-O0", "-g", "-D_GLIBCXX_ASSERTIONS",
            "-fsanitize=undefined", "-fno-sanitize-recover=all",
            str(cpp), "-o", str(cls.exe)]
        subprocess.run(command, check=True, cwd=cls.work, timeout=90)

    def run_check(self, *args):
        return subprocess.run([str(self.exe), *args], cwd=self.work,
                              capture_output=True, text=True, timeout=10)

    def assert_passes(self, mode):
        result = self.run_check(mode)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_empty_tracker_preserves_flags(self):
        self.assert_passes("empty")

    def test_single_tracker_unchanged(self):
        self.assert_passes("single")

    def test_sphere_box_and_level_decisions(self):
        self.assert_passes("radial")

    def test_default_and_gr_overrides(self):
        self.assert_passes("coordinates")

    def test_completed_step_cadence_and_resumed_clock(self):
        self.assert_passes("tracker")

    def test_reserved_choice_fails_before_writer(self):
        for choices in [("rad_m1_vel",), ("r_sph", "rad_m1_vel")]:
            with self.subTest(choices=choices):
                marker = self.work / "writer_reached.txt"
                marker.unlink(missing_ok=True)
                result = self.run_check("output", *choices)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("rad_m1_vel", result.stdout)
                self.assertIn("output73", result.stdout)
                self.assertIn("not implemented", result.stdout)
                self.assertFalse(marker.exists())

    def test_supported_m1_registrations_and_indices(self):
        self.assertEqual(len(self.choices), 175)
        self.assertEqual(self.choices[163], "rad_m1_vel")
        self.assertEqual(self.choices[164:166], ["rad_m1_moments", "rad_m1_opacities"])
        self.assertEqual(self.choices[174], "r_sph")
        for choice in self.choices[154:172]:
            if choice == "rad_m1_vel":
                continue
            with self.subTest(choice=choice):
                components = {"rad_m1_F": 3, "rad_m1_H": 3, "rad_m1_fnu": 4,
                              "rad_m1_moments": 5, "rad_m1_opacities": 5}.get(choice, 1)
                result = self.run_check("output", choice)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(int(result.stdout), 4 * components)
                rows = (self.work / "writer_reached.txt").read_text().splitlines()
                self.assertEqual(len(rows), 4 * components)
                self.assertEqual(len(set(rows)), len(rows))
                if choice == "rad_m1_E":
                    self.assertEqual(rows, [f"E:{i} {5 * i}" for i in range(4)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
