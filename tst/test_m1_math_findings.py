"""Compile and run native M1 math regressions with an AthenaK build.

Usage: python tst/test_m1_math_findings.py /absolute/path/to/build
The selected compiler and Kokkos backend are reused; a CPU execution-space run
is not GPU verification.
"""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def _build_commands(build):
    """Return the Athena compiler command, link libraries, and build cwd."""
    link_file = build / "src/CMakeFiles/athena.dir/link.txt"
    link = shlex.split(link_file.read_text())
    libraries = link[link.index("athena") + 1:]

    compile_commands = build / "compile_commands.json"
    if compile_commands.exists():
        entries = json.loads(compile_commands.read_text())
        entry = next(e for e in entries if e["file"].endswith("/src/main.cpp"))
        flags = shlex.split(entry["command"])
        # Drop only the original input/output selection, retaining backend flags.
        compile_flags = flags[:flags.index("-o")]
        return compile_flags, libraries, Path(entry["directory"])

    # The standard test suite does not request compile_commands.json. Reuse the
    # compiler and target flags emitted by its Unix Makefiles build instead.
    make_flags = {}
    flags_file = build / "src/CMakeFiles/athena.dir/flags.make"
    for line in flags_file.read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            make_flags[key] = value
    compile_flags = [link[0]]
    for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        compile_flags.extend(shlex.split(make_flags.get(key, "")))
    return compile_flags, libraries, build / "src"


def run(build):
    build = Path(build).resolve()
    source = Path(__file__).parent / "test_suite/unit_tests/m1_math_findings.cpp"
    compile_flags, libraries, build_cwd = _build_commands(build)
    with tempfile.TemporaryDirectory(prefix="hermes-verify-m1-math-") as tmp:
        exe = Path(tmp) / "m1_math_findings"
        subprocess.run(compile_flags + [str(source), "-o", str(exe)] + libraries,
                       cwd=build_cwd, check=True)
        completed = subprocess.run([str(exe)], text=True, capture_output=True)
        print(completed.stdout, end="")
        if completed.returncode != 0:
            raise AssertionError("M1 math regression executable failed")
        warning = "Maximum number of iterations exceeded when computing the M1 closure"
        fallback = "Newton M1 closure failed; falling back to Brent-Dekker."
        debug_build = any(flag == "-DDEBUG_BUILD" for flag in compile_flags)
        if not debug_build and (warning in completed.stdout or fallback in completed.stdout):
            raise AssertionError("Release build printed a closure-solver diagnostic")
        print("PASS exact roots, closure/source failure status, release silence, "
              "and Brent fallback")


def main():
    run(sys.argv[1])


if __name__ == "__main__":
    main()
