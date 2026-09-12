"""Beam geometry, inflow and restart checks; no build or third-party modules.

Run source/algebra checks: python3 -B tst/test_m1_beam_findings.py
Also run actual inflow/restart checks with a double-precision built-in-pgen binary:
ATHENA_M1_BEAM_EXECUTABLE=/absolute/path/athena python3 -B tst/test_m1_beam_findings.py
The optional runs use one process/thread and temporary /tmp/hermes-verify-* data.
Source checks are not a substitute for the optional executable checks.
"""

import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def source(relative):
    """Strip line comments so disabled metric calls cannot satisfy assertions."""
    return re.sub(r"//[^\n]*", "", (ROOT / relative).read_text())


class BeamSourceChecks(unittest.TestCase):
    def test_destination_geometry(self):
        text = source("src/radiation_m1/radiation_m1_beams.cpp")
        text = text.split("void ApplyBeamSourcesBlackHole", 1)[1]
        active = text.split("for (int i = 0; i < ng; ++i)", 1)[1]
        active = active.split("else", 1)[0]
        self.assertRegex(active, r"const int ig\s*=\s*is\s*-\s*i\s*-\s*1;")
        reads = re.findall(r"adm\.(?:alpha|beta_u|g_dd)\(([^)]*)\)", active)
        self.assertGreater(len(reads), 20)
        for arguments in reads:
            self.assertEqual(arguments.split(",")[-1].strip(), "ig", arguments)
        writes = re.findall(r"u0_\(.*?,\s*k,\s*j,\s*([^)]*)\)", active)
        self.assertEqual(len(writes), 5)
        self.assertEqual(set(writes), {"ig"})

    def check_beam_norm(self, metric, beta, alpha):
        text = source("src/radiation_m1/radiation_m1_beams.cpp")
        expression = re.search(r"const Real a\s*=\s*(.*?);", text, re.S).group(1)
        expression = re.sub(r"adm\.alpha\([^)]*\)", "alpha", expression)
        expression = expression.replace("params_.rad_eps", "eps")
        beta_x = sum(metric[0][j] * beta[j] for j in range(3))
        beta2 = sum(metric[i][j] * beta[i] * beta[j]
                    for i in range(3) for j in range(3))
        # Evaluate the production scalar expression, not a copy of its formula.
        a = eval(" ".join(expression.split()), {"__builtins__": {}},
                 dict(sqrt=math.sqrt, alpha=alpha, eps=0.01, beta_x=beta_x,
                      beta2=beta2, g_xx=metric[0][0]))
        flux = [(a * (i == 0) + beta[i]) / alpha for i in range(3)]
        norm = sum(metric[i][j] * flux[i] * flux[j]
                   for i in range(3) for j in range(3))
        self.assertAlmostEqual(norm, 0.99, delta=2e-14)
        return a

    def test_isotropic_and_shifted_metrics(self):
        self.check_beam_norm([[1.4641, 0, 0], [0, 1.4641, 0],
                              [0, 0, 1.4641]], [0, 0, 0], 9 / 11)
        self.check_beam_norm([[1.4, 0.1, 0.02], [0.1, 1.2, 0.03],
                              [0.02, 0.03, 1.1]], [0.1, 0.06, -0.02], 0.8)

    def test_minkowski_unchanged(self):
        a = self.check_beam_norm([[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                [0, 0, 0], 1)
        self.assertEqual(a, math.sqrt(0.99))

    def test_population_gate_and_copy(self):
        text = source("src/pgen/tests/rad_m1_beams.cpp")
        block = text.split("Kokkos::realloc(beam_vals_host, 4);", 1)[1]
        condition = re.search(r"if\s*\((.*?)\)\s*\{\s*Real E", block, re.S).group(1)
        condition = condition.replace("pmbp->pmesh->", "")
        condition = condition.replace("||", " or ").replace("&&", " and ")
        for one_d, two_d, metric, expected in [
                (True, False, "minkowski", True),
                (True, False, "isotropic", True),
                (False, True, "minkowski", True),
                (False, True, "isotropic", False)]:
            with self.subTest(one_d=one_d, metric=metric):
                self.assertEqual(eval(" ".join(condition.split()),
                                      {"__builtins__": {}},
                                      dict(one_d=one_d, two_d=two_d, metric=metric)),
                                 expected)
        self.assertIn("Real E = 1;", block)
        self.assertIn("Real Fx = E;", block)
        self.assertIn("apply_floor(g_uu, E, F_d, pmbp->pradm1->params);", block)
        self.assertIn("Kokkos::deep_copy(beam_vals, beam_vals_host);", block)

    def test_restart_setup_before_state_guard(self):
        text = source("src/pgen/tests/rad_m1_beams.cpp")
        guard = re.search(r"if\s*\(restart\)\s*\{?\s*return;", text)
        self.assertIsNotNone(guard)
        setup, initial = text[:guard.start()], text[guard.end():]
        for token in ["user_bcs = true;", "params.beam_sources = true;",
                      "user_bcs_func = radiationm1::ApplyBeamSources1D;",
                      "user_bcs_func = radiationm1::ApplyBeamSources2D;",
                      "user_bcs_func = radiationm1::ApplyBeamSourcesBlackHole;",
                      'GetOrAddReal("problem", "beam_ymin"',
                      'GetOrAddReal("problem", "beam_ymax"',
                      "Kokkos::realloc(beam_vals, 4);",
                      "Kokkos::deep_copy(beam_vals, beam_vals_host);"]:
            self.assertIn(token, setup)
        self.assertNotIn("par_for(", setup)
        self.assertNotRegex(setup, r"(?:adm\.\w+|w0_|u0_?)\(")
        self.assertNotRegex(text, r"\bu0_?\(")
        self.assertIn("adm.g_dd(m, a, b, k, j, i) =", initial)
        self.assertIn("w0_(m, IVX, k, j, i) = 0.;", initial)


def beam_input(two_d, metric):
    """Small single-block case; the sampled y cell lies inside the beam interval."""
    bc = "outflow" if metric == "minkowski" else "vacuum"
    deck = f"""<job>
basename = beam
<mesh>
nghost = 4
nx1 = 16
x1min = 4
x1max = 6
ix1_bc = {bc}
ox1_bc = {bc}
nx2 = {16 if two_d else 1}
x2min = 3
x2max = 5
ix2_bc = {bc if two_d else 'periodic'}
ox2_bc = {bc if two_d else 'periodic'}
nx3 = 1
x3min = -0.5
x3max = 0.5
ix3_bc = periodic
ox3_bc = periodic
<meshblock>
nx1 = 16
nx2 = {16 if two_d else 1}
nx3 = 1
<time>
evolution = dynamic
integrator = rk2
cfl_number = 0.2
tlim = 1
nlim = 2
<coord>
general_rel = true
m = 1
a = 0
excise = false
<adm>
metric = {metric}
bh_mass = 1
<radiation_m1>
closure_fun = minerbo
rad_E_floor = 1e-12
rad_eps = 0.01
backreact = false
backreact_tmunu = false
matter_sources = false
<problem>
pgen_name = rad_m1_beamtest
beam_ymin = 3.25
beam_ymax = 3.75
<output3>
file_type = rst
dt = 1
"""
    for number, variable, label in [(1, "rad_m1_E", "e"), (2, "rad_m1_F", "f")]:
        deck += f"""<output{number}>
file_type = tab
variable = {variable}
id = {label}
dcycle = 1
ghost_zones = true
data_format = %.17e
slice_x3 = 0
"""
        if two_d:
            deck += "slice_x2 = 3.5\n"
    return deck


def table_at_cycle(directory, label, cycle):
    """Read fields from the right: table headers include degenerate coordinates."""
    for path in sorted((directory / "tab").glob(f"beam.{label}.*.tab")):
        lines = path.read_text().splitlines()
        if int(re.search(r"cycle=(\d+)", lines[0]).group(1)) != cycle:
            continue
        names = [name for name in lines[1].split() if ":" in name]
        rows = [line.split() for line in lines[2:] if line.strip()]
        return {int(row[1]): (float(row[2]), dict(zip(names, map(float, row[-len(names):]))))
                for row in rows}
    raise AssertionError(f"Missing cycle={cycle}, {label} table in {directory}")


class BeamExecutableChecks(unittest.TestCase):
    def test_inflow_and_restart(self):
        executable = os.environ.get("ATHENA_M1_BEAM_EXECUTABLE")
        if not executable:
            self.skipTest("set ATHENA_M1_BEAM_EXECUTABLE for actual inflow/restart runs")
        executable = str(Path(executable).resolve(strict=True))
        environment = dict(os.environ, OMP_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix="hermes-verify-beams-", dir="/tmp") as tmp:
            for two_d, metric in [(False, "minkowski"), (True, "minkowski"),
                                  (True, "isotropic"), (True, "kerrschild")]:
                with self.subTest(two_d=two_d, metric=metric):
                    case = Path(tmp) / f"{two_d}-{metric}"
                    case.mkdir()
                    pin = case / "beam.athinput"
                    pin.write_text(beam_input(two_d, metric))
                    for name, arguments in [("full", ["-i", str(pin)]),
                                             ("split", ["-i", str(pin), "time/nlim=1"]),
                                             ("resume", None)]:
                        run = case / name
                        run.mkdir()
                        if arguments is None:
                            checkpoint = sorted((case / "split/rst").glob("*.rst"))[-1]
                            arguments = ["-r", str(checkpoint), "time/nlim=2"]
                        result = subprocess.run([executable, *arguments], cwd=run,
                                                env=environment, text=True,
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT, timeout=120)
                        self.assertEqual(result.returncode, 0, result.stdout)
                    energy = table_at_cycle(case / "full", "e", 0)
                    flux = table_at_cycle(case / "full", "f", 0)
                    ghosts = [i for i, (x, _) in energy.items() if x < 4]
                    self.assertEqual(len(ghosts), 4)
                    for i in ghosts:
                        x, fields = energy[i]
                        # Slice 3.5 selects j=4 in the active mesh, centered at 3.5625.
                        y = 3.5625
                        r = math.hypot(x, y)
                        psi4 = (1 + 0.5 / r) ** 4 if metric == "isotropic" else 1
                        volume = (math.sqrt(1 + 2 / r) if metric == "kerrschild"
                                  else psi4 ** 1.5)
                        for label, value in fields.items():
                            species = label.split(":")[1]
                            f = [flux[i][1][f"F{axis}:{species}"] for axis in "xyz"]
                            self.assertGreater(f[0], 0)
                            norm = sum(v * v for v in f) / psi4
                            if metric == "kerrschild":
                                norm -= (2 / (r + 2)) * ((x * f[0] + y * f[1]) / r) ** 2
                            self.assertAlmostEqual(value / volume, 1, delta=2e-13)
                            self.assertAlmostEqual(norm / value ** 2, 0.99, delta=2e-13)
                    for label in ["e", "f"]:
                        full = table_at_cycle(case / "full", label, 2)
                        resumed = table_at_cycle(case / "resume", label, 2)
                        self.assertEqual(full.keys(), resumed.keys())
                        for i in full:
                            self.assertEqual(full[i][0], resumed[i][0])
                            self.assertEqual(full[i][1].keys(), resumed[i][1].keys())
                            for field, value in full[i][1].items():
                                self.assertTrue(math.isclose(value, resumed[i][1][field],
                                                            rel_tol=2e-13, abs_tol=2e-14),
                                                (label, i, field, value, resumed[i]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
