#!/usr/bin/env python3
"""Direct WeakRates opacity mean regression against a fresh, existing build.

Future execution, after separate authorization:
    python3 tst/test_weakrates_opacity_mean.py BUILD_DIR

Requires the retained emission runner's Unix Makefiles, C++17, double precision,
MPI/OpenMP-off, built_in_pgens, Serial/CUDA and object-freshness contracts. This
runner compiles only its standalone fixture and replaces main.cpp.o in the
existing link command. It never configures or rebuilds AthenaK, installs packages,
changes submodules, or captures/loads legacy references. NuRATES may be OFF.
All generated objects, executables and native output live under TMPDIR scratch.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tst/test_suite/unit_tests/weakrates_opacity_mean.cpp"
SUCCESS = "PASS weakrates opacity mean native suite"


def build_metadata(build):
    """Reuse read-only freshness/compile/link discovery, not legacy execution."""
    sys.dont_write_bytecode = True
    path = ROOT / "tst/test_weakrates_emission.py"
    spec = importlib.util.spec_from_file_location("weakrates_emission_metadata", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import existing WeakRates build metadata")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry, flags, link, inputs, record = module.metadata(build)
    # The shared metadata routine fingerprints its own fixture. Replace that
    # informational fingerprint; no old reference file is consulted or generated.
    record = dict(record)
    record["fixture_sha256"] = module.digest(SOURCE)
    record["fixture"] = str(SOURCE.relative_to(ROOT))
    return entry, flags, link, inputs, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    args = parser.parse_args()
    entry, flags, link, inputs, record = build_metadata(args.build_dir.resolve())
    scratch = Path(os.environ.get("TMPDIR", Path.home() / ".hermes/cache/scratch")).resolve()
    if scratch == ROOT or ROOT in scratch.parents:
        raise ValueError("TMPDIR scratch must be outside the source checkout")
    scratch.mkdir(parents=True, exist_ok=True)
    print("WeakRates opacity mean metadata: " + json.dumps(record, sort_keys=True), flush=True)
    with tempfile.TemporaryDirectory(prefix="weakrates-opacity-mean-", dir=scratch) as name:
        tmp = Path(name)
        obj, exe = tmp / "opacity_mean.o", tmp / "opacity_mean"
        subprocess.run(flags + ["-c", str(SOURCE), "-o", str(obj)],
                       cwd=entry["directory"], check=True)
        command = [str(obj) if x.endswith("/main.cpp.o") else x for x in link]
        command[command.index("-o") + 1] = str(exe)
        # Newer host linkers can publish a dependency file beside the original
        # target. Suppress that output rather than mutate the user's build tree.
        command = [x for x in command if not x.startswith("-Wl,--dependency-file=")]
        subprocess.run(command, cwd=entry["directory"], check=True)
        run = subprocess.run([str(exe), *map(str, inputs)], cwd=tmp,
                             text=True, capture_output=True)
        (tmp / "native_stdout.txt").write_text(run.stdout)
        (tmp / "native_stderr.txt").write_text(run.stderr)
        print(run.stdout, end="", flush=True)
        print(run.stderr, end="", file=sys.stderr, flush=True)
        if run.returncode != 0 or SUCCESS not in run.stdout.splitlines():
            raise AssertionError("native direct-opacity-mean suite failed or did not complete")
        if any(line.startswith("FAIL ") for line in run.stdout.splitlines()):
            raise AssertionError("native fixture reported a failure despite its completion marker")
        print("PASS weakrates opacity mean runner (" + record["backend"] + ")")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, IndexError, TypeError, AssertionError,
            subprocess.CalledProcessError) as exc:
        print("FAIL weakrates opacity mean: " + str(exc), file=sys.stderr)
        sys.exit(1)
