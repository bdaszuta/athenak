"""Run the real ParameterInput/serial I/O code without an AthenaK build.

Run: python tst/test_parameter_roundtrip_findings.py -v
Only unrelated athena.hpp types and Kokkos::abort are replaced by a preinclude;
serialization, parsing, dumping, and file loading use the production sources.
Both Real aliases are compiled separately, sequentially, with C++17.
"""

import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TYPES = r"""
#define ATHENA_HPP_
#define MPI_PARALLEL_ENABLED 0
#define OPENMP_PARALLEL_ENABLED 0
#include <cstdint>
#include <cstdlib>
#if SINGLE_PRECISION_ENABLED
using Real = float;
#else
using Real = double;
#endif
namespace Kokkos {
[[noreturn]] inline void abort(const char*) { std::abort(); }
}
"""
CHECK = r"""
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include "parameter_input.hpp"

namespace global_variable {
int my_rank = 0;
int nranks = 1;
}

bool Exact(Real actual, Real expected) {
  return actual == expected && std::signbit(actual) == std::signbit(expected);
}

int main(int argc, char* argv[]) {
  if (argc != 2) return 2;
  const Real values[] = {
      Real(0.0012345678901234567), Real(3.14159265358979),
      Real(2181.2345678900001), Real(0), -Real(0), Real(1), Real(-1),
      std::nextafter(Real(1), Real(2)), std::nextafter(Real(1), Real(0)),
      std::numeric_limits<Real>::epsilon(), std::numeric_limits<Real>::min(),
      std::numeric_limits<Real>::max(), std::numeric_limits<Real>::lowest(),
      std::numeric_limits<Real>::denorm_min(), Real(1e-20), Real(-1e20)};
  constexpr std::size_t count = sizeof(values) / sizeof(values[0]);
  ParameterInput source;
  for (int pass = 0; pass < 2; ++pass) {
    for (std::size_t i = 0; i < count; ++i) {
      const Real value = pass == 0 ? values[i] : -values[i];
      if (!Exact(source.SetReal("problem", "value" + std::to_string(i), value),
                 value)) return 3;
    }
    std::stringstream dump;
    source.ParameterDump(dump);
    ParameterInput stream_reload;
    stream_reload.LoadFromStream(dump);
    {
      std::ofstream output(argv[1], std::ios::binary);
      output << dump.str();
      if (!output) return 4;
    }
    IOWrapper input;
    input.Open(argv[1], IOWrapper::FileMode::read);
    ParameterInput file_reload;
    file_reload.LoadFromFile(input);
    input.Close();
    for (std::size_t i = 0; i < count; ++i) {
      const std::string name = "value" + std::to_string(i);
      const Real expected = pass == 0 ? values[i] : -values[i];
      for (auto* parameters : {&source, &stream_reload, &file_reload}) {
        const Real actual = parameters->GetReal("problem", name);
        if (!Exact(actual, expected)) {
          std::cerr << std::setprecision(std::numeric_limits<Real>::max_digits10)
                    << "round-trip mismatch: " << name << ": " << expected
                    << " -> " << actual << '\n';
          return 1;
        }
      }
    }
  }
  std::cout << "PASS " << (sizeof(Real) == sizeof(float) ? "float" : "double")
            << ": " << count << " values, insert/update, stream/file reload\n";
}
"""


class ParameterRoundTripFindings(unittest.TestCase):
    def test_setreal_dump_reload_exact(self):
        with tempfile.TemporaryDirectory(
                prefix="hermes-verify-parameter-", dir="/tmp") as directory:
            scratch = Path(directory)
            types = scratch / "types.hpp"
            check = scratch / "check.cpp"
            types.write_text(TYPES, encoding="ascii")
            check.write_text(CHECK, encoding="ascii")
            for single_precision in (0, 1):
                with self.subTest(single_precision=single_precision):
                    binary = scratch / f"check-{single_precision}"
                    command = shlex.split(os.environ.get("CXX", "c++")) + [
                        "-std=c++17", "-O0", "-g", "-D_GLIBCXX_ASSERTIONS",
                        f"-DSINGLE_PRECISION_ENABLED={single_precision}",
                        "-include", str(types), "-I", str(ROOT / "src"),
                        str(check), str(ROOT / "src/parameter_input.cpp"),
                        str(ROOT / "src/outputs/io_wrapper.cpp"),
                        "-o", str(binary),
                    ]
                    subprocess.run(command, check=True, timeout=120)
                    result = subprocess.run(
                        [str(binary), str(scratch / "parameters.txt")],
                        capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0,
                                     result.stdout + result.stderr)
                    print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
