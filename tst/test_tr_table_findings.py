"""Compile and exercise the real Table reader with C++17 and sanitizers.

Run: python tst/test_tr_table_findings.py -v
Optional full-size table: SFHO_TABLE=/path/to/SFHo.athtab python ... -v
No AthenaK/Kokkos build is needed. Compilation and cases run sequentially.
"""

import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHECK = r"""
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include "utils/tr_table.hpp"

int main(int argc, char* argv[]) {
  if (argc != 4) return 2;
  const std::string mode = argv[1];
  TableReader::Table table;
  const auto result = table.ReadTable(argv[2]);
  if (mode == "short") {
    assert(result.error != TableReader::ReadResult::SUCCESS);
    assert(!table.IsInitialize());
    assert(table.GetRawData() == nullptr);
    assert(!table.HasField("nb") && !table.HasField("Q1") && !table.HasField("Q2"));
    assert(result.message.find(argv[2]) != std::string::npos);
    assert(result.message.find("expected 48 bytes") != std::string::npos);
    assert(result.message.find("read " + std::string(argv[3])) != std::string::npos);
    assert(result.message.find("Swapped") == std::string::npos);
    std::cout << result.message;
    return 0;
  }
  if (result.error != TableReader::ReadResult::SUCCESS) {
    std::cerr << result.message;
    return 1;
  }
  assert(table.IsInitialize());
  assert(table.GetRawData() != nullptr);
  assert(table.HasField("nb") && table.HasField("Q1"));
  if (mode == "sfho") {
    assert(table.GetNDimensions() == 3);
    assert(table.HasField("yq") && table.HasField("t"));
    const auto& points = table.GetPointInfo();
    std::size_t npoints = 1;
    std::size_t coordinates = 0;
    for (const auto& point : points) {
      npoints *= point.second;
      coordinates += point.second;
    }
    assert(npoints == table.GetNPoints());
    assert(coordinates + npoints * table.GetFieldNames().size() == table.GetMemSize());
    assert(std::isfinite(table["Q1"][0]));
    assert(std::isfinite(table["Q1"][npoints - 1]));
    std::cout << "PASS SFHo: " << table.GetMemSize() << " doubles\n";
    return 0;
  }
  assert(table.GetNDimensions() == 1 && table.GetNPoints() == 2);
  assert(table.GetMemSize() == 6);
  const double expected[] = {0.25, 0.5, 1.25, -2.5, 3.75, 4.125};
  for (std::size_t i = 0; i < 6; ++i) assert(table.GetRawData()[i] == expected[i]);
  assert(table["Q1"] == table.GetRawData() + 2);
  assert(table["Q2"] == table.GetRawData() + 4);
  assert(table.GetScalars().at("mn") == 939.56535);
  if (mode == "empty") {
    assert(table.GetMetadata().at("empty").empty());
    assert(table.GetMetadata().at("spaces").empty());
    assert(table.GetMetadata().at("tabs").empty());
  }
  std::cout << "PASS " << mode << '\n';
}
"""


def table_bytes(endianness, whitespace=False, empty=False):
    metadata = f"version=1.0\nendianness={endianness}\nprecision=double\n"
    if empty:
        metadata += "empty= \nspaces=   \ntabs=\t\t\n"
    points = "\t nb\t = \t2\t \n" if whitespace else "nb=2\n"
    fields = "\t Q1 \t \nQ2\t\t\n" if whitespace else "Q1\nQ2\n"
    header = (
        "<metadatabegin>\n" + metadata + "<metadataend>\n"
        "<scalarsbegin>\nmn=939.56535\n<scalarsend>\n"
        "<pointsbegin>\n" + points + "<pointsend>\n"
        "<fieldsbegin>\n" + fields + "<fieldsend>\n"
    ).encode("ascii")
    order = "<" if endianness == "little" else ">"
    payload = struct.pack(order + "6d", 0.25, 0.5, 1.25, -2.5, 3.75, 4.125)
    return header, payload


class TableReaderFindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="hermes-verify-tr-table-", dir="/tmp")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.scratch = Path(cls.temporary.name)
        check = cls.scratch / "check.cpp"
        check.write_text(CHECK, encoding="ascii")
        cls.binary = cls.scratch / "check"
        command = shlex.split(os.environ.get("CXX", "c++")) + [
            "-std=c++17", "-O0", "-g", "-D_GLIBCXX_ASSERTIONS",
            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
            "-fno-omit-frame-pointer", "-I", str(ROOT / "src"), str(check),
            str(ROOT / "src/utils/tr_table.cpp"), "-o", str(cls.binary),
        ]
        subprocess.run(command, check=True, timeout=120)

    def run_reader(self, mode, path, byte_count=0):
        environment = os.environ.copy()
        environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
        result = subprocess.run(
            [str(self.binary), mode, str(path), str(byte_count)],
            capture_output=True, text=True, env=environment, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout.strip()

    def test_complete_payload_both_byte_orders(self):
        for order in ("little", "big"):
            with self.subTest(order=order):
                header, payload = table_bytes(order)
                path = self.scratch / "complete.athtab"
                path.write_bytes(header + payload)
                self.run_reader("complete", path)

    def test_trailing_tabs_and_mixed_whitespace(self):
        for order in ("little", "big"):
            with self.subTest(order=order):
                header, payload = table_bytes(order, whitespace=True)
                path = self.scratch / "whitespace.athtab"
                path.write_bytes(header + payload)
                self.run_reader("whitespace", path)

    def test_empty_and_whitespace_only_metadata_values(self):
        header, payload = table_bytes("little", empty=True)
        path = self.scratch / "empty.athtab"
        path.write_bytes(header + payload)
        self.run_reader("empty", path)

    def test_truncated_payload_both_byte_orders(self):
        for order in ("little", "big"):
            header, payload = table_bytes(order)
            for byte_count in (0, 1, 7, 8, 47):
                with self.subTest(order=order, bytes=byte_count):
                    path = self.scratch / "short.athtab"
                    path.write_bytes(header + payload[:byte_count])
                    self.run_reader("short", path, byte_count)

    def test_shipped_reduced_sfho(self):
        path = ROOT / "tst/inputs/tables/SFHo_reduced.athtab"
        print(self.run_reader("sfho", path))

    @unittest.skipUnless(os.environ.get("SFHO_TABLE"), "SFHO_TABLE not supplied")
    def test_supplied_full_sfho(self):
        print(self.run_reader("sfho", Path(os.environ["SFHO_TABLE"])))


if __name__ == "__main__":
    unittest.main()
