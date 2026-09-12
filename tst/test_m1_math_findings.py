"""Compile native M1 regressions using a configured Unix Makefiles build.

Usage: python tst/test_m1_math_findings.py /absolute/path/to/build
Requires CMAKE_EXPORT_COMPILE_COMMANDS=ON and built Kokkos libraries. The
selected compiler/backend is reused; a Serial run is not a GPU verification.
"""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def main():
    build = Path(sys.argv[1]).resolve()
    source = Path(__file__).parent / "test_suite/unit_tests/m1_math_findings.cpp"
    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next(e for e in entries if e["file"].endswith("/src/main.cpp"))
    flags = shlex.split(entry["command"])
    # Drop only the original input/output selection, retaining backend flags.
    compile_flags = flags[:flags.index("-o")]
    link = shlex.split((build / "src/CMakeFiles/athena.dir/link.txt").read_text())
    libraries = link[link.index("athena") + 1:]
    with tempfile.TemporaryDirectory(prefix="hermes-verify-m1-math-") as tmp:
        exe = Path(tmp) / "m1_math_findings"
        subprocess.run(compile_flags + [str(source), "-o", str(exe)] + libraries,
                       cwd=entry["directory"], check=True)
        run = subprocess.run([str(exe)], check=True, text=True, capture_output=True)
        print(run.stdout, end="")
        warning = "Maximum number of iterations exceeded when computing the M1 closure (Newton)"
        if run.stdout.count(warning) != 2:
            raise AssertionError("Expected one exhausted host and one kernel warning")
        print("PASS exact roots, bounded exhaustion, and both nonconvergence warnings")


if __name__ == "__main__":
    main()
