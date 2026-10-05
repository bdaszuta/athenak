#!/usr/bin/env python3
"""Focused device regression against an already rebuilt AthenaK (stdlib only).

Approved future execution:
 python3 tst/test_weakrates_emission.py BUILD_DIR --legacy-reference REFERENCE_JSON
 WEAKRATES_LEGACY_REFERENCE=REFERENCE_JSON python3 tst/test_weakrates_emission.py BUILD_DIR
 python3 tst/test_weakrates_emission.py BUILD_DIR --capture-legacy-reference REFERENCE_JSON

Capture requires a genuinely heavy-fixed, pre-thermal production source epoch.
Preserve that source separately: start at b1c9f93b797a716825169c0fa4e55cc90d453de7
and apply ONLY the plan's heavy finalizer and calculator call/comment changes.
Use this same fixture there and an existing rebuilt Makefiles build. The capture
macro omits all thermal interfaces. Never use the completed off path as its own
independent oracle. No reference is supplied or synthesized here. Keep one file
per Serial/CUDA x NuRATES-off/on row outside scratch/checkouts. This runner never
configures/rebuilds AthenaK, installs dependencies, or changes production files.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ("radiation_m1_calc_opacities_weakrates.cpp", "radiation_m1_weakrates.hpp",
              "radiation_m1.hpp", "radiation_m1.cpp", "radiation_m1_update.cpp")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_values(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.split(":", 1)[0]] = value
    return result


def truth(value):
    return value.upper() in {"1", "ON", "TRUE", "YES"}


def compile_flags(entry):
    args = entry.get("arguments") or shlex.split(entry["command"])
    flags = []
    source = Path(entry["file"]).resolve()
    paired = {"-o", "-MF", "-MT", "-MQ"}
    single = {"-c", "-MD", "-MMD", "-MP"}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in paired:
            if i + 1 >= len(args):
                raise ValueError("truncated compiler option " + arg)
            i += 2
            continue
        if arg in single or any(arg.startswith(x) and arg != x for x in paired):
            i += 1
            continue
        if not arg.startswith("-") and Path(entry["directory"], arg).resolve() == source:
            i += 1
            continue
        flags.append(arg)
        i += 1
    if not flags:
        raise ValueError("empty compile command")
    return flags


def metadata(build):
    cache = cache_values(build / "CMakeCache.txt")
    if cache.get("CMAKE_GENERATOR") != "Unix Makefiles":
        raise ValueError("Unix Makefiles link metadata is required (no Ninja)")
    if Path(cache.get("CMAKE_HOME_DIRECTORY", "")).resolve() != ROOT:
        raise ValueError("build/source association does not match test checkout")
    if not truth(cache.get("CMAKE_EXPORT_COMPILE_COMMANDS", "OFF")):
        raise ValueError("CMAKE_EXPORT_COMPILE_COMMANDS=ON is required")
    for key in ("Athena_SINGLE_PRECISION", "Athena_ENABLE_MPI", "Athena_ENABLE_OPENMP"):
        if key not in cache or truth(cache[key]):
            raise ValueError(key + " must explicitly be OFF")
    if cache.get("PROBLEM") != "built_in_pgens":
        raise ValueError("PROBLEM=built_in_pgens is required")
    entries = json.loads((build / "compile_commands.json").read_text())
    mains = [e for e in entries if Path(e["file"]).resolve() == ROOT / "src/main.cpp"]
    if len(mains) != 1:
        raise ValueError("expected exactly one main.cpp compile command")
    entry = mains[0]
    cwd = Path(entry["directory"]).resolve()
    if cwd != build and build not in cwd.parents:
        raise ValueError("compile directory is outside build")
    flags = compile_flags(entry)
    if "-std=c++17" not in flags:
        raise ValueError("the approved C++17 compile standard is required")
    if any(x in {"-DSINGLE_PRECISION_ENABLED=1", "-DSINGLE_PRECISION_ENABLED"} for x in flags):
        raise ValueError("single precision compile definition is forbidden")
    # Validate existing object/dependency freshness without rebuilding anything.
    for e in entries:
        args = e.get("arguments") or shlex.split(e["command"])
        if "-o" not in args:
            raise ValueError("missing object output metadata")
        obj = Path(e["directory"], args[args.index("-o") + 1]).resolve()
        dep = Path(str(obj) + ".d")
        if not obj.is_file() or not dep.is_file():
            raise ValueError("missing built object/dependency file: " + str(obj))
        text = dep.read_text().replace("\\\n", " ")
        deps = shlex.split(text.split(":", 1)[1].split("\n", 1)[0])
        for name in [e["file"]] + deps:
            p = Path(e["directory"], name).resolve()
            if p.is_file() and p.stat().st_mtime_ns > obj.stat().st_mtime_ns:
                raise ValueError("stale build: " + str(p) + " newer than " + str(obj))
    link = shlex.split((build / "src/CMakeFiles/athena.dir/link.txt").read_text())
    if sum(x.endswith("/main.cpp.o") for x in link) != 1 or "-o" not in link:
        raise ValueError("link must contain one replaceable main and output")
    selected = {k: v for k, v in cache.items() if k.startswith(("Athena_", "Kokkos_"))
                or k in {"CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS", "CMAKE_BUILD_TYPE", "PROBLEM"}}
    cuda = truth(cache.get("Kokkos_ENABLE_CUDA", "OFF"))
    for backend in ("OPENMP", "THREADS", "HIP", "SYCL", "OPENMPTARGET", "HPX"):
        if truth(cache.get("Kokkos_ENABLE_" + backend, "OFF")):
            raise ValueError("unsupported active Kokkos backend " + backend)
    if not cuda and not truth(cache.get("Kokkos_ENABLE_SERIAL", "OFF")):
        raise ValueError("suite requires Serial or CUDA")
    normalize = lambda xs: [x.replace(str(build), "<BUILD>").replace(str(ROOT), "<SOURCE>") for x in xs]
    selected = {k: normalize([v])[0] for k, v in selected.items()}
    inputs = [ROOT / "inputs/tests/rad_m1_singlezone.athinput",
              ROOT / "tst/inputs/tables/SFHo_reduced.athtab",
              ROOT / "tst/inputs/tables/SFHo_reduced_NQT.athtab"]
    # Resolve compiler/launcher binaries without running them. CUDA wrappers also
    # need their underlying nvcc and host compiler fingerprint, not just a script.
    def executable(token):
        local = Path(entry["directory"], token)
        path = local.resolve() if local.is_file() else shutil.which(token)
        if not path:
            raise ValueError("cannot resolve compiler executable " + token)
        return Path(path).resolve()
    compiler = executable(cache["CMAKE_CXX_COMPILER"])
    binaries = [executable(flags[0]), compiler]
    if cuda:
        wrapper = compiler
        default_host = "g++"
        launchers = [i for i, arg in enumerate(flags)
                     if Path(arg).name == "kokkos_launch_compiler"]
        if launchers:
            # Kokkos's RULE_LAUNCH_COMPILE prefixes the command with
            # launcher, nvcc_wrapper, expected CXX, actual CXX. Identify the
            # executed compiler chain, not just the cached host compiler.
            if len(launchers) != 1 or launchers[0] + 3 >= len(flags):
                raise ValueError("invalid Kokkos CUDA launcher metadata")
            i = launchers[0]
            launcher = executable(flags[i])
            wrapper = executable(flags[i + 1])
            expected = executable(flags[i + 2])
            actual = executable(flags[i + 3])
            if expected != compiler or actual != expected or "-DKOKKOS_DEPENDENCE" not in flags:
                raise ValueError("Kokkos launcher does not select the CUDA compiler")
            binaries.extend([launcher, wrapper, expected, actual])
            if wrapper.parent == launcher.parent:
                # The unconfigured launcher uses expected CXX; an installed
                # launcher may instead embed its configured default host.
                installed = re.search(r':\s*\$\{NVCC_WRAPPER_DEFAULT_COMPILER:="([^"]+)"\}',
                                      launcher.read_text())
                if installed is None:
                    raise ValueError("unrecognized Kokkos launcher host-compiler default")
                default_host = (str(expected) if installed[1] == "@NVCC_WRAPPER_DEFAULT_COMPILER@"
                                else installed[1])
        if wrapper.name == "nvcc_wrapper":
            text = wrapper.read_text()
            if 'host_compiler=${NVCC_WRAPPER_DEFAULT_COMPILER:-"g++"}' not in text:
                raise ValueError("unrecognized CUDA wrapper: compiler identity requires explicit metadata")
            nvcc = str(Path(os.environ["CUDA_ROOT"]) / "bin/nvcc") if os.environ.get("CUDA_ROOT") else "nvcc"
            host = os.environ.get("NVCC_WRAPPER_DEFAULT_COMPILER") or default_host
            for i, arg in enumerate(flags):
                if arg in {"-ccbin", "--compiler-bindir"}:
                    if i + 1 >= len(flags):
                        raise ValueError("missing CUDA host-compiler argument")
                    host = flags[i + 1]
                elif arg.startswith(("-ccbin=", "--compiler-bindir=")):
                    host = arg.split("=", 1)[1]
            binaries.extend([executable(nvcc), executable(host)])
        elif launchers:
            raise ValueError("unrecognized Kokkos CUDA compiler redirect")
    compiler_files = {normalize([str(p)])[0]: digest(p) for p in binaries}
    record = {"schema": 1, "settings": selected, "backend": "CUDA" if cuda else "Serial",
              "precision": "double", "compiler_files": compiler_files,
              "compile": normalize(flags), "link": normalize(link),
              "fixture_sha256": digest(ROOT / "tst/test_suite/unit_tests/weakrates_emission.cpp"),
              "inputs": {str(p.relative_to(ROOT)): digest(p) for p in inputs}}
    return entry, flags, link, inputs, record


def source_checks():
    sys.dont_write_bytecode = True  # No generated __pycache__ in the checkout.
    for name in ("test_weakrates_configuration", "test_m1_equilibria_weakrates"):
        spec = importlib.util.spec_from_file_location(name, ROOT / "tst" / (name + ".py"))
        if spec is None or spec.loader is None:
            raise ValueError("cannot load retained source test " + name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for key, value in sorted(vars(module).items()):
            if key.startswith("test_") and callable(value):
                value()
                print("PASS retained source check " + key)


def records(path):
    parsed = {}
    for line in path.read_text().splitlines():
        key, *values = line.split()
        if key in parsed or not values or any(not re.fullmatch(r"[0-9a-f]{16}", v) for v in values):
            raise ValueError("invalid/duplicate full-precision fixture record: " + key)
        parsed[key] = values
    if not parsed:
        raise ValueError("native fixture produced no records")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--capture-legacy-reference", type=Path)
    group.add_argument("--legacy-reference", type=Path)
    args = parser.parse_args()
    entry, flags, link, inputs, meta = metadata(args.build_dir.resolve())
    capture = args.capture_legacy_reference
    saved = None
    reference = args.legacy_reference or (Path(os.environ["WEAKRATES_LEGACY_REFERENCE"])
                                         if os.environ.get("WEAKRATES_LEGACY_REFERENCE") else None)
    if capture:
        capture = capture.resolve()
        if capture == ROOT or ROOT in capture.parents:
            raise ValueError("legacy reference must be outside the checkout")
        if capture.exists():
            raise ValueError("refusing to overwrite legacy reference")
        code = "\n".join((ROOT / "src/radiation_m1" / n).read_text() for n in PRODUCTION)
        if "FinalizeKirchhoffCoefficients" not in code or "weakrates_thermal" in code or "use_electron_thermal_emission" in code:
            raise ValueError("capture requires heavy-fixed PRE-THERMAL sources; completed off path is not independent")
    else:
        if reference is None:
            raise ValueError("missing heavy-only reference: use --legacy-reference or WEAKRATES_LEGACY_REFERENCE")
        saved = json.loads(reference.read_text())
        if not isinstance(saved, dict):
            raise ValueError("legacy reference must be a JSON object")
        if saved.get("metadata") != meta or saved.get("provenance") != "heavy-fixed-pre-thermal":
            raise ValueError("legacy reference compiler/backend/precision/fixture/input metadata mismatch")
        if not isinstance(saved.get("records"), dict) or not saved["records"]:
            raise ValueError("missing legacy fixture records")
        source_checks()
    scratch = Path(os.environ.get("TMPDIR", Path.home() / ".hermes/cache/scratch"))
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="weakrates-emission-", dir=scratch) as name:
        tmp = Path(name)
        obj, exe, data = tmp / "weakrates.o", tmp / "weakrates", tmp / "records.txt"
        extra = ["-DWEAKRATES_LEGACY_CAPTURE=1"] if capture else []
        source = ROOT / "tst/test_suite/unit_tests/weakrates_emission.cpp"
        subprocess.run(flags + extra + ["-c", str(source), "-o", str(obj)], cwd=entry["directory"], check=True)
        command = [str(obj) if x.endswith("/main.cpp.o") else x for x in link]
        command[command.index("-o") + 1] = str(exe)
        command = [x for x in command if not x.startswith("-Wl,--dependency-file=")]
        subprocess.run(command, cwd=entry["directory"], check=True)
        subprocess.run([str(exe), *map(str, inputs), "--records", str(data)], cwd=tmp, check=True)
        actual = records(data)
        if capture:
            payload = {"metadata": meta, "provenance": "heavy-fixed-pre-thermal", "records": actual,
                       "legacy_source_sha256": {n: digest(ROOT / "src/radiation_m1" / n) for n in PRODUCTION}}
            capture = capture.resolve()
            capture.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=capture.parent, prefix=".weakrates-reference-", delete=False) as f:
                pending = Path(f.name)
                json.dump(payload, f, sort_keys=True, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(pending, capture)  # atomic publication, refuses overwrite
            finally:
                pending.unlink()
            print("PASS captured independent heavy-only reference " + str(capture))
        else:
            assert saved is not None
            if actual != saved["records"]:
                missing = set(saved["records"]) - set(actual)
                extra_keys = set(actual) - set(saved["records"])
                changed = [k for k in actual.keys() & saved["records"].keys() if actual[k] != saved["records"][k]]
                raise AssertionError(f"legacy bitwise mismatch: missing={sorted(missing)}, extra={sorted(extra_keys)}, changed={changed}")
            print("PASS explicit-off bitwise heavy-only coefficients/radiation/matter")
            def child(binary, case, ng, logs, mode, fatal=True):
                command = [str(binary), *map(str, inputs), "--fatal", case, str(ng), logs, str(mode)]
                run = subprocess.run(command, cwd=tmp, text=True, capture_output=True)
                output = run.stdout + run.stderr
                if fatal:
                    expected = {"global-heavy": "GlobalInvalidation", "acceptance": "AcceptanceFailure",
                                "stored-rate": "InvalidInput", "stored-reference": "InvalidReference"}.get(case)
                    if run.returncode == 0 or "WEAKRATES electron thermal" not in output or (expected and expected not in output):
                        raise AssertionError("missing native thermal fatal diagnostic: " + case + "\n" + output)
                elif run.returncode or "PASS weakrates emission native suite" not in output:
                    raise AssertionError("disabled fatal fixture changed legacy behavior: " + case + "\n" + output)
                print(f"PASS native {'abort' if fatal else 'off gate'} {case} NGHOST={ng} {logs} mode={mode}")
            for ng in (2, 3, 4):
                for logs in ("NormalLogs", "NQTLogs"):
                    for mode in (0, 1):
                        for case in ("stored-rate", "stored-reference", "global-heavy", "acceptance"):
                            child(exe, case, ng, logs, mode)
                        child(exe, "acceptance-off", ng, logs, mode, fatal=False)
            # Test-only fault injection after the original five-field validation:
            # raw formulas always return symmetric thermal rates, so the actual
            # producer's new-model abort needs an injected asymmetry. Compile a
            # scratch COPY of its existing TU, replace that one object, and never
            # alter checkout source or the ordinary numerical executable.
            producer = ROOT / "src/radiation_m1/radiation_m1_calc_opacities_weakrates.cpp"
            source_text = producer.read_text()
            anchor = "weakrates::ElectronThermalCoefficients thermal{};"
            if source_text.count(anchor) != 1:
                raise ValueError("producer fault-injection boundary changed; review required")
            injected = "raw.eta_0_non_cc[weakrates::kAntiNuE] = -1.0;\n        " + anchor
            copied = tmp / "fault_calc_opacities_weakrates.cpp"
            copied.write_text(source_text.replace(anchor, injected))
            entries = json.loads((args.build_dir.resolve() / "compile_commands.json").read_text())
            pe = [e for e in entries if Path(e["file"]).resolve() == producer]
            if len(pe) != 1:
                raise ValueError("missing producer compile metadata for fatal test")
            fault_obj, fault_exe = tmp / "fault_producer.o", tmp / "fault_weakrates"
            subprocess.run(compile_flags(pe[0]) + ["-c", str(copied), "-o", str(fault_obj)], cwd=pe[0]["directory"], check=True)
            name = "/radiation_m1_calc_opacities_weakrates.cpp.o"
            if sum(x.endswith(name) for x in command) != 1:
                raise ValueError("missing producer object in native link metadata")
            fault_link = [str(fault_obj) if x.endswith(name) else x for x in command]
            fault_link[fault_link.index("-o") + 1] = str(fault_exe)
            subprocess.run(fault_link, cwd=entry["directory"], check=True)
            for ng in (2, 3, 4):
                for logs in ("NormalLogs", "NQTLogs"):
                    for mode in (0, 1):
                        child(fault_exe, "producer-model", ng, logs, mode)
                        child(fault_exe, "producer-model-off", ng, logs, mode, fatal=False)
            # An allocated zero-rate cell still publishes its four references.
            # This is a separate actual-kernel witness, not merely zeroing the
            # thermal View after publication in the ordinary routing fixture.
            zero_lines = "\n".join(
                f"raw.{channel}[weakrates::{species}] = 0.0;"
                for channel in ("eta_0_non_cc", "eta_1_non_cc")
                for species in ("kNuE", "kAntiNuE"))
            zero_source, zero_obj, zero_exe = tmp / "zero_producer.cpp", tmp / "zero_producer.o", tmp / "zero_weakrates"
            zero_source.write_text(source_text.replace(anchor, zero_lines + "\n        " + anchor))
            subprocess.run(compile_flags(pe[0]) + ["-c", str(zero_source), "-o", str(zero_obj)], cwd=pe[0]["directory"], check=True)
            zero_link = [str(zero_obj) if x.endswith(name) else x for x in command]
            zero_link[zero_link.index("-o") + 1] = str(zero_exe)
            subprocess.run(zero_link, cwd=entry["directory"], check=True)
            for ng in (2, 3, 4):
                for logs in ("NormalLogs", "NQTLogs"):
                    for mode in (0, 1):
                        child(zero_exe, "producer-zero", ng, logs, mode, fatal=False)
            # Unlike the new-model fault above, this invalidates a raw rate
            # BEFORE the original microscopic finite check, in both option
            # states. Its required outcome is legacy all-cell zeroing, not abort.
            raw_anchor = "point_finite = weakrates::IsFinite(local_state) && weakrates::IsFinite(raw) &&"
            if source_text.count(raw_anchor) != 1:
                raise ValueError("original raw-rate validation boundary changed; review required")
            invalid_source, invalid_obj, invalid_exe = tmp / "invalid_raw_producer.cpp", tmp / "invalid_raw_producer.o", tmp / "invalid_raw_weakrates"
            invalid_source.write_text("#include <limits>\n" + source_text.replace(
                raw_anchor, "raw.eta_0_non_cc[weakrates::kNuE] = std::numeric_limits<Real>::quiet_NaN();\n        " + raw_anchor))
            subprocess.run(compile_flags(pe[0]) + ["-c", str(invalid_source), "-o", str(invalid_obj)], cwd=pe[0]["directory"], check=True)
            invalid_link = [str(invalid_obj) if x.endswith(name) else x for x in command]
            invalid_link[invalid_link.index("-o") + 1] = str(invalid_exe)
            subprocess.run(invalid_link, cwd=entry["directory"], check=True)
            for ng in (2, 3, 4):
                for logs in ("NormalLogs", "NQTLogs"):
                    for mode in (0, 1):
                        for case in ("producer-raw-invalid", "producer-raw-invalid-off"):
                            child(invalid_exe, case, ng, logs, mode, fatal=False)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, IndexError, TypeError, AssertionError, subprocess.CalledProcessError) as exc:
        print("FAIL weakrates emission: " + str(exc), file=sys.stderr)
        sys.exit(1)
