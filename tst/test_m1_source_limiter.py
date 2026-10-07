#!/usr/bin/env python3
"""Source-only addition; execution is paused until explicitly approved.

Future usage: python3 tst/test_m1_source_limiter.py /absolute/path/to/build
Requires already rebuilt Unix Makefiles AthenaK objects, compile_commands.json,
Athena_ENABLE_NURATES=ON, double precision, and MPI/OpenMP off. NuRATES is needed
only for the real four-group constructor; the fixture seeds coefficients itself.
Reuses the configured compiler/Kokkos backend (Serial or CUDA), compiles just the
fixture, and substitutes its main object in the existing native link command.
Never configures/rebuilds AthenaK, installs anything, or runs a simulation/C2P.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def cache_values(path):
    values = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            key, value = line.split("=", 1)
            values[key.split(":", 1)[0]] = value
    return values


def enabled(value):
    return value.upper() in {"1", "ON", "TRUE", "YES"}


def arguments(entry):
    return entry.get("arguments") or shlex.split(entry["command"])


def fixture_flags(entry):
    args = arguments(entry)
    flags = []
    source = Path(entry["file"]).resolve()
    paired = {"-o", "-MF", "-MT", "-MQ"}
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in paired:
            if index + 1 >= len(args):
                raise ValueError("truncated compiler option " + arg)
            index += 2
            continue
        if arg in {"-c", "-MD", "-MMD", "-MP"} or any(
                arg.startswith(p) and arg != p for p in paired):
            index += 1
            continue
        if not arg.startswith("-") and Path(entry["directory"], arg).resolve() == source:
            index += 1
            continue
        flags.append(arg)
        index += 1
    return flags


def metadata(build):
    cache = cache_values(build / "CMakeCache.txt")
    if cache.get("CMAKE_GENERATOR") != "Unix Makefiles":
        raise ValueError("Unix Makefiles build metadata required; no Ninja")
    if Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve() != ROOT:
        raise ValueError("build does not belong to this source checkout")
    for key in ("Athena_SINGLE_PRECISION", "Athena_ENABLE_MPI", "Athena_ENABLE_OPENMP"):
        if key not in cache or enabled(cache[key]):
            raise ValueError(key + " must explicitly be OFF")
    if not enabled(cache.get("Athena_ENABLE_NURATES", "OFF")):
        raise ValueError("real four-group constructor requires Athena_ENABLE_NURATES=ON")
    cuda = enabled(cache.get("Kokkos_ENABLE_CUDA", "OFF"))
    for backend in ("OPENMP", "THREADS", "HIP", "SYCL", "OPENMPTARGET", "HPX"):
        if enabled(cache.get("Kokkos_ENABLE_" + backend, "OFF")):
            raise ValueError("fixture supports Serial or CUDA, not " + backend)
    if not cuda and not enabled(cache.get("Kokkos_ENABLE_SERIAL", "OFF")):
        raise ValueError("Serial or CUDA execution space required")
    entries = json.loads((build / "compile_commands.json").read_text())
    mains = [e for e in entries if Path(e["file"]).resolve() == ROOT / "src/main.cpp"]
    if len(mains) != 1:
        raise ValueError("expected exactly one main.cpp compile command")
    # Refuse old production objects: this runner must exercise the edited helper
    # and TimeUpdate, not an accidentally stale binary from before the bug fix.
    for entry in entries:
        args = arguments(entry)
        if "-o" not in args:
            raise ValueError("missing object output in build metadata")
        obj = Path(entry["directory"], args[args.index("-o") + 1]).resolve()
        dep = Path(str(obj) + ".d")
        if not obj.is_file() or not dep.is_file():
            raise ValueError("missing built object/dependency: " + str(obj))
        text = dep.read_text().replace("\\\n", " ")
        names = shlex.split(text.split(":", 1)[1].split("\n", 1)[0])
        for name in [entry["file"], *names]:
            source = Path(entry["directory"], name).resolve()
            if source.is_file() and source.stat().st_mtime_ns > obj.stat().st_mtime_ns:
                raise ValueError("stale build; rebuild after approval: " + str(source))
    entry = mains[0]
    link = shlex.split((build / "src/CMakeFiles/athena.dir/link.txt").read_text())
    if sum(x.endswith("/main.cpp.o") for x in link) != 1 or "-o" not in link:
        raise ValueError("link requires exactly one replaceable main and output")
    return entry, fixture_flags(entry), link


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    args = parser.parse_args()
    entry, flags, link = metadata(args.build_dir.resolve())
    deck = ROOT / "inputs/tests/rad_m1_singlezone.athinput"
    table = ROOT / "tst/inputs/tables/SFHo_reduced.athtab"
    if not deck.is_file() or not table.is_file():
        raise ValueError("missing stock deck or reduced CompOSE table")
    scratch = Path(os.environ.get("TMPDIR", Path.home() / ".hermes/cache/scratch"))
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="6")
    with tempfile.TemporaryDirectory(prefix="m1-source-limiter-", dir=scratch) as name:
        tmp = Path(name)
        obj, exe = tmp / "limiter.o", tmp / "limiter"
        fixture = ROOT / "tst/test_suite/unit_tests/m1_source_limiter.cpp"
        subprocess.run(flags + ["-c", str(fixture), "-o", str(obj)],
                       cwd=entry["directory"], env=env, check=True)
        command = [str(obj) if x.endswith("/main.cpp.o") else x for x in link]
        command[command.index("-o") + 1] = str(exe)
        command = [x for x in command if not x.startswith("-Wl,--dependency-file=")]
        subprocess.run(command, cwd=entry["directory"], env=env, check=True)
        subprocess.run([str(exe), str(deck), str(table)], cwd=tmp, env=env, check=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, IndexError, subprocess.CalledProcessError) as error:
        print("FAIL native M1 source limiter: " + str(error), file=sys.stderr)
        sys.exit(1)
