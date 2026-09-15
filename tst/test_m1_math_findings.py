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
        run = subprocess.run([str(exe)], text=True, capture_output=True)
        print(run.stdout, end="")
        if run.returncode != 0:
            raise AssertionError("M1 math regression executable failed")
        warning = "Maximum number of iterations exceeded when computing the M1 closure"
        fallback = "Newton M1 closure failed; falling back to Brent-Dekker."
        debug_build = any(flag == "-DDEBUG_BUILD" for flag in compile_flags)
        if not debug_build and (warning in run.stdout or fallback in run.stdout):
            raise AssertionError("Release build printed a closure-solver diagnostic")
        print("PASS exact roots, exhaustion signal, release silence, and Brent fallback")


if __name__ == "__main__":
    main()
