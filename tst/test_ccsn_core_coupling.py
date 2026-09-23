"""Link a native CCSN regression main against the exact already-built AthenaK objects.

Usage: python tst/test_ccsn_core_coupling.py BUILD EOS PROGENITOR
The caller must build the matching, QOL-integrated athena target first. This
script compiles only the replacement test main; it never configures/rebuilds athena.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


CASES = ("selection", "source_explicit", "source_implicit", "opacity", "excision", "tmunu", "bounce")


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def main():
    require(len(sys.argv) == 4, "usage: test_ccsn_core_coupling.py BUILD EOS PROGENITOR")
    root = Path(__file__).resolve().parents[1]
    build, eos, progenitor = (Path(arg).resolve() for arg in sys.argv[1:])
    for path in (eos, progenitor):
        require(path.is_file(), f"missing input: {path}")
    cache = (build / "CMakeCache.txt").read_text()
    values = dict((key.split(":", 1)[0], value)
                  for line in cache.splitlines()
                  if line and not line.startswith(("#", "//")) and "=" in line
                  for key, value in [line.split("=", 1)])
    source = values.get("athena_SOURCE_DIR") or values.get("CMAKE_HOME_DIRECTORY")
    require(source is not None and Path(source).resolve() == root,
            "CMakeCache source root does not match this checkout")
    for key, expected in (("PROBLEM", "ccsn"), ("Athena_SINGLE_PRECISION", "OFF"),
                          ("Athena_ENABLE_NURATES", "ON"), ("Athena_ENABLE_MPI", "OFF")):
        require(values.get(key) == expected, f"required CMakeCache setting: {key}={expected}")
    entries = json.loads((build / "compile_commands.json").read_text())
    matches = [e for e in entries if Path(e["file"]).resolve() == root / "src/main.cpp"]
    require(len(matches) == 1, "expected exactly one compile command for this main.cpp")
    entry = matches[0]
    flags = shlex.split(entry["command"]) if "command" in entry else list(entry["arguments"])
    require(flags.count("-o") == 1 and flags.count("-c") == 1,
            "cannot identify unique main object/source in compiler command")
    original_source = Path(entry["file"]).resolve()
    source_positions = [i for i, flag in enumerate(flags)
                        if flag.endswith("main.cpp") and
                        (Path(entry["directory"]) / flag).resolve() == original_source]
    require(len(source_positions) == 1, "compiler command does not name exactly one main.cpp")
    link = shlex.split((build / "src/CMakeFiles/athena.dir/link.txt").read_text())
    objects = [i for i, item in enumerate(link) if item.endswith("/main.cpp.o")]
    require(len(objects) == 1 and link.count("-o") == 1,
            "link command must contain one main.cpp.o and one output")
    require((build / "src/athena").is_file(), "build matching athena target before this test")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="ccsn-core-coupling-") as directory:
        scratch = Path(directory)
        obj, exe = scratch / "ccsn_core_coupling.o", scratch / "ccsn_core_coupling"
        flags[flags.index("-o") + 1] = str(obj)
        flags[source_positions[0]] = str(root / "tst/test_suite/unit_tests/ccsn_core_coupling.cpp")
        subprocess.run(flags, cwd=entry["directory"], env=env, check=True)
        link[objects[0]] = str(obj)
        link[link.index("-o") + 1] = str(exe)
        link = [x for x in link if not x.startswith("-Wl,--dependency-file=")]
        subprocess.run(link, cwd=entry["directory"], env=env, check=True)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        print(f"source={root} revision={revision} build={build} compiler={flags[0]}", flush=True)
        for case in CASES:
            result = subprocess.run([str(exe), str(root / "inputs/tests/rad_m1_singlezone.athinput"),
                                     str(eos), str(progenitor), case], cwd=scratch,
                                    env=env, text=True, capture_output=True, timeout=120)
            require(result.returncode == 0 and f"PASS {case}\n" in result.stdout,
                    f"{case} failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
            print(result.stdout, end="", flush=True)
        print(f"PASS all {len(CASES)} native CCSN cases", flush=True)


if __name__ == "__main__":
    main()
