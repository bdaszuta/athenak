"""Native opacity and quadrature regressions with a NuRATES-enabled Serial build.

Usage: python tst/test_m1_opacity_findings.py /absolute/path/to/build
Requires built AthenaK objects and CMAKE_EXPORT_COMPILE_COMMANDS=ON. Uses the
shipped reduced SFHo table and real rate evaluations, without evolving matter.
"""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def main():
    build = Path(sys.argv[1]).resolve()
    root = Path(__file__).resolve().parents[1]
    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next(e for e in entries if e["file"].endswith("/src/main.cpp"))
    flags = shlex.split(entry["command"])
    flags = flags[:flags.index("-o")]
    link = shlex.split((build / "src/CMakeFiles/athena.dir/link.txt").read_text())
    with tempfile.TemporaryDirectory(prefix="hermes-verify-m1-opacity-") as tmp:
        tmp = Path(tmp)
        obj, exe = tmp / "opacity.o", tmp / "opacity"
        source = root / "tst/test_suite/unit_tests/m1_opacity_findings.cpp"
        subprocess.run(flags + ["-c", str(source), "-o", str(obj)],
                       cwd=entry["directory"], check=True)
        link = [str(obj) if x.endswith("/main.cpp.o") else x for x in link]
        link[link.index("-o") + 1] = str(exe)
        link = [x for x in link if not x.startswith("-Wl,--dependency-file=")]
        subprocess.run(link, cwd=entry["directory"], check=True)
        subprocess.run([str(exe), str(root / "inputs/tests/rad_m1_singlezone.athinput"),
                        str(root / "tst/inputs/tables/SFHo_reduced.athtab")],
                       cwd=tmp, check=True)
        # Fixed MHD skips C2P, so NuRATES must retain pressure-based temperature recovery.
        fixed = tmp / "fixed.athinput"
        text = (root / "inputs/tests/rad_m1_singlezone.athinput").read_text()
        text = text.replace("<mhd>", "<mhd>\nfixed = true")
        fixed.write_text(text, encoding="ascii")
        subprocess.run([str(build / "src/athena"), "-i", str(fixed),
                        f"mhd/table={root / 'tst/inputs/tables/SFHo_reduced.athtab'}",
                        "time/nlim=1"], cwd=tmp, check=True)
        print("PASS fixed-MHD opacity temperature fallback")
        # CLI overrides require keys to exist in the deck before constructors.
        # Add startup-only keys to a temporary copy, leaving shipped inputs alone.
        beam = tmp / "quadrature.athinput"
        text = (root / "inputs/tests/radiation_beam_1d.athinput").read_text()
        text = text.replace("<time>", "<time>\nnlim = 0")
        text = text.replace("<radiation_m1>", "<radiation_m1>\nopacity_type = bns-nurates")
        text = text.replace("metric = kerrschild", "metric = minkowski")
        text += "\n<bns_nurates>\nnurates_quad_nx = 6\nnurates_quad_nx_2 = -1\n"
        beam.write_text(text, encoding="ascii")
        common = [str(build / "src/athena"), "-i", str(beam),
                  "mesh/nx1=8", "mesh/nx2=8", "meshblock/nx1=8", "meshblock/nx2=8",
                  "time/nlim=0"]
        for n, n2, valid in [(6, -1, True), (1, 1, True), (10, 20, True),
                             (0, -1, False), (11, -1, False), (21, -1, False),
                             (6, 0, False), (6, 21, False), (6, -2, False)]:
            run = subprocess.run(common + [f"bns_nurates/nurates_quad_nx={n}",
                                           f"bns_nurates/nurates_quad_nx_2={n2}"],
                                 cwd=tmp, text=True, capture_output=True)
            if valid:
                if run.returncode:
                    raise AssertionError(run.stdout + run.stderr)
            elif run.returncode == 0 or "must be" not in run.stderr:
                raise AssertionError(run.stdout + run.stderr)
            print(f"PASS quadrature orders ({n}, {n2}) accepted={valid}")


if __name__ == "__main__":
    main()
