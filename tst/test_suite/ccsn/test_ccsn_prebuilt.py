"""Opt-in CCSN regressions using a supplied executable, EOS and real profile.

No configure/build helpers are imported or called. Run this module directly with
pytest only after runtime verification is authorized; see tst/CCSN_VERIFICATION.md.
"""

import copy
import importlib
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = {"matter": "mhd_u", "radiation": "rad_m1_moments", "geometry": "z4c"}
# This is a forced lifecycle event, NOT a physical bounce-time prediction.
TRIGGER = {"bdm_rho_min": "0", "bdm_spb_max": "-1e99", "bdm_r_max": "1e99"}


def _parameters(lines):
    """Read native input/ParameterDump text, not C++ source patterns."""
    blocks = {}
    block = None
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "<par_end>":
            break
        if line.startswith("<") and line.endswith(">"):
            block = blocks.setdefault(line[1:-1], {})
        else:
            assert block is not None and "=" in line, f"Invalid input line: {raw}"
            key, value = line.split("=", 1)
            block[key.strip()] = value.strip()
    return blocks


def _write_input(path, blocks):
    path.write_text("\n".join(
        f"<{name}>\n" + "\n".join(f"{key} = {value}" for key, value in values.items())
        for name, values in blocks.items()
    ) + "\n", encoding="ascii")


@pytest.fixture
def ccsn(monkeypatch):
    """Missing external assets skip; supplied but unsuitable inputs fail."""
    assets = {}
    for name in ("EXE", "INPUT", "EOS", "PROGENITOR"):
        variable = f"CCSN_TEST_{name}"
        value = os.environ.get(variable)
        if not value:
            pytest.skip(f"{variable} is required; no CCSN binaries/data are built or fetched")
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            pytest.skip(f"{variable} is not an available file: {path}")
        assets[name] = path
    if not os.access(assets["EXE"], os.X_OK):
        pytest.skip(f"CCSN_TEST_EXE is not executable: {assets['EXE']}")
    np = pytest.importorskip("numpy", reason="native binary reader requires numpy")
    pytest.importorskip("h5py", reason="native bin_convert imports h5py")
    monkeypatch.syspath_prepend(str(ROOT / "vis" / "python"))
    reader = importlib.import_module("bin_convert")
    blocks = _parameters(assets["INPUT"].read_text(encoding="ascii").splitlines())
    for name in ("mesh", "meshblock", "time", "mhd", "adm", "z4c", "radiation_m1", "problem"):
        assert name in blocks, f"CCSN_TEST_INPUT requires <{name}>"
    assert blocks["mhd"].get("dyn_eos") == "compose", "Supply a CompOSE CCSN deck"
    assert blocks["mhd"].get("units", "geometric_solar") == "geometric_solar"
    assert blocks["radiation_m1"].get("opacity_type") in (
        "bns-nurates", "weakrates"), "Supply a BNS-NuRates or weakrates CCSN deck"
    assert blocks["mhd"].get("nscalars") == "1", "This scope uses Ye as its only scalar"
    assets.update(blocks=blocks, np=np, reader=reader)
    return assets


def _deck(ccsn, mode="transport", feedback=(True, True)):
    blocks = copy.deepcopy(ccsn["blocks"])
    # Keep physical domain, EOS/floor/reaction controls and real ICs from the
    # supplied deck. Bound topology and work; do not inherit production outputs.
    for name in list(blocks):
        if name.startswith(("output", "refined_region", "amr_criterion")):
            del blocks[name]
    blocks.setdefault("job", {})["basename"] = "ccsn_test"
    for axis in ("1", "2", "3"):
        blocks["mesh"][f"nx{axis}"] = "16"
        blocks["meshblock"][f"nx{axis}"] = "8"
    blocks["mesh"]["nghost"] = "4"
    blocks["mesh_refinement"] = {"refinement": "none"}
    blocks["time"].update(evolution="dynamic", nlim="3", tlim="1e30",
                          start_time="0", cfl_number="0.05", ndiag="1")
    blocks["mhd"]["table"] = str(ccsn["EOS"])
    blocks["mhd"]["fixed"] = "false"
    blocks.setdefault("coord", {})["excise"] = "false"
    blocks["radiation_m1"].update(
        matter_sources="true", gr_sources="true",
        backreact=str(feedback[0]).lower(), backreact_tmunu=str(feedback[1]).lower())
    blocks["problem"].update(
        pgen_name="ccsn", progenitor=str(ccsn["PROGENITOR"]), user_hist="true",
        refinement_method="none", update_conserved="false", detect_bounce="true",
        bounce_detection_method="entropy", bdm_rho_min="1e99", bdm_spb_max="3.0",
        inject_equilibrium="false", equilibrium_injected="false",
        stop_at_bounce="false", post_bounce="false",
        t_bounce="0", D_max_bounce="0", stop_before_m1="false",
        prebounce_core_coupling=str(mode == "core").lower(),
        recoupling_approach=str(mode in ("transport", "core")).lower(),
        equilibriate_post_bounce=str(mode in ("projection", "direct")).lower(),
        conservative_bounce_projection=str(mode == "projection").lower())
    if mode == "core":
        blocks["problem"]["deleptonization_method"] = "Liebendoerfer"
        blocks["problem"]["rho_trap"] = "1.0e12"
        blocks["radiation_m1"].update(backreact="true", backreact_tmunu="true")
    if mode in ("projection", "direct"):
        for name in ("eql_rho_min_cgs", "eql_t_min_mev"):
            assert name in blocks["problem"], f"Supply physical equilibrium cutoff {name}"
    for number, (name, variable) in enumerate(OUTPUTS.items(), 1):
        blocks[f"output{number}"] = dict(file_type="bin", variable=variable,
                                         id=name, dcycle="1", single_file_per_rank="false")
    blocks["output4"] = dict(file_type="rst", dcycle="1", single_file_per_rank="false")
    blocks["output5"] = dict(file_type="hst", dcycle="1", user_hist_only="true",
                             data_format="%24.17e")
    return blocks


def _run(ccsn, directory, blocks=None, restart=None, overrides=(), ranks=1,
         failure=None):
    """Use isolated paths, fixed rank/thread caps and a process-group timeout."""
    directory.mkdir(parents=True)
    command = [str(ccsn["EXE"])]
    if ranks == 2:
        launcher = os.environ.get("CCSN_TEST_MPIEXEC")
        if not launcher:
            pytest.skip("CCSN_TEST_MPIEXEC is required for the two-rank MPI cases")
        launcher_path = Path(launcher).expanduser()
        if not (launcher_path.is_absolute() and launcher_path.is_file()
                and os.access(launcher_path, os.X_OK)):
            pytest.skip(f"CCSN_TEST_MPIEXEC is not an available absolute executable: {launcher}")
        # An executable path only, not arbitrary shell flags or a rank override.
        command = [str(launcher_path), "-n", "2"] + command
    else:
        assert ranks == 1
    if restart is None:
        input_path = directory / "test.athinput"
        _write_input(input_path, blocks)
        command += ["-i", str(input_path)]
    else:
        command += ["-r", str(restart)]
    command += list(overrides)
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    (directory / "command.txt").write_text(shlex.join(command) + "\n", encoding="ascii")
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=directory, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            returncode = process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            pytest.fail(f"CCSN exceeded 120 seconds; process group killed; see {directory / 'run.log'}")
    text = (directory / "run.log").read_text(encoding="utf-8", errors="replace")
    if failure is None:
        assert returncode == 0, text
        assert "FATAL" not in text, text
    else:
        assert returncode != 0, "Invalid CCSN input was accepted"
        assert failure in text, text
    return text


def _snapshot(ccsn, directory, first=False):
    np = ccsn["np"]
    result: dict[str, Any] = {"fields": {}}
    for name in OUTPUTS:
        files = sorted((directory / "bin").glob(f"ccsn_test.{name}.*.bin"))
        assert files, f"No native {name} output under {directory}"
        data = ccsn["reader"].read_binary(str(files[0] if first else files[-1]))
        if "cycle" not in result:
            result.update(cycle=data["cycle"], time=data["time"],
                          topology=data["mb_logical"],
                          parameters=_parameters(data["header"]))
        assert data["cycle"] == result["cycle"]
        assert data["time"] == result["time"]
        np.testing.assert_array_equal(data["mb_logical"], result["topology"])
        assert data["n_mbs"] == 8, "The small non-AMR deck must have eight blocks"
        result["fields"][name] = {}
        for variable, values in data["mb_data"].items():
            values = np.asarray(values, dtype=np.float64)
            assert np.isfinite(values).all(), f"Nonfinite {name}/{variable}"
            result["fields"][name][variable] = values
    assert set(result["fields"]["radiation"]) == {
        f"{var}:{species}" for var in ("E", "N", "Fx", "Fy", "Fz") for species in range(4)
    }
    assert {"dens", "ener", "mom1", "mom2", "mom3", "r_00"} <= set(result["fields"]["matter"])
    return result


def _same(ccsn, left, right, groups=OUTPUTS):
    """Require identical stored fields; do not hide differences behind a ULP band."""
    np = ccsn["np"]
    assert left["cycle"] == right["cycle"]
    assert left["time"] == pytest.approx(right["time"], rel=1e-13, abs=0)
    np.testing.assert_array_equal(left["topology"], right["topology"])
    for group in groups:
        assert left["fields"][group].keys() == right["fields"][group].keys()
        for name, expected in left["fields"][group].items():
            actual = right["fields"][group][name]
            np.testing.assert_array_equal(actual, expected, err_msg=f"{group}/{name}")


def _flag(snapshot, block, key):
    value = snapshot["parameters"][block][key].lower()
    assert value in ("true", "false", "1", "0"), f"Invalid boolean {block}/{key}={value}"
    return value in ("true", "1")


def _mode(snapshot, mode, feedback, bounced):
    assert _flag(snapshot, "problem", "prebounce_core_coupling") == (mode == "core")
    assert _flag(snapshot, "problem", "recoupling_approach") == (mode in ("transport", "core"))
    assert _flag(snapshot, "problem", "equilibriate_post_bounce") == (mode in ("projection", "direct"))
    assert _flag(snapshot, "problem", "conservative_bounce_projection") == (mode == "projection")
    assert not _flag(snapshot, "problem", "inject_equilibrium")
    assert not _flag(snapshot, "problem", "update_conserved")
    assert _flag(snapshot, "problem", "detect_bounce")
    assert _flag(snapshot, "problem", "post_bounce") == bounced
    for key, configured in zip(("backreact", "backreact_tmunu"), feedback):
        assert _flag(snapshot, "radiation_m1", key) == configured


@pytest.mark.parametrize("feedback", ((True, True), (True, False), (False, True), (False, False)))
def test_legacy_omitted_and_explicit_false_core_option(ccsn, tmp_path, feedback):
    """The extra input key must not alter stored legacy fields or feedback settings."""
    deck = _deck(ccsn, feedback=feedback)
    deck["time"]["nlim"] = "1"
    explicit = tmp_path / "explicit-false"
    _run(ccsn, explicit, deck)
    recorded = _snapshot(ccsn, explicit)
    _mode(recorded, "transport", feedback, False)
    del deck["problem"]["prebounce_core_coupling"]
    omitted = tmp_path / "omitted"
    _run(ccsn, omitted, deck)
    other = _snapshot(ccsn, omitted)
    _mode(other, "transport", feedback, False)
    _same(ccsn, recorded, other)


@pytest.mark.parametrize("block, key, value, conflict", (
    ("problem", "recoupling_approach", "false", "recoupling_approach"),
    ("problem", "detect_bounce", "false", "detect_bounce"),
    ("problem", "equilibriate_post_bounce", "true", "equilibriate_post_bounce"),
    ("problem", "conservative_bounce_projection", "true", "conservative_bounce_projection"),
    ("problem", "deleptonization_method", "None", "deleptonization_method"),
    ("problem", "rho_trap", "0", "rho_trap"),
    ("problem", "rho_trap", "-1", "rho_trap"),
    ("problem", "rho_trap", "1e999", "rho_trap"),
    ("mhd", "fixed", "true", "mhd/fixed=false"),
    ("mhd", "dyn_eos", "ideal", "CCSN deleptonization/entropy bounce requires CompOSE"),
    ("mhd", "nscalars", "0", "one scalar"),
    ("radiation_m1", "opacity_type", "none", "radiation_m1/opacity_type"),
    ("radiation_m1", "opacity_type", "toy", "radiation_m1/opacity_type with four-group equilibrium"),
    ("radiation_m1", "matter_sources", "false", "matter_sources"),
    ("radiation_m1", "backreact", "false", "backreact"),
    ("radiation_m1", "backreact_tmunu", "false", "backreact_tmunu"),
))
def test_core_incompatible_inputs_rejected(ccsn, tmp_path, block, key, value, conflict):
    deck = _deck(ccsn, "core")
    deck[block][key] = value
    deck["time"]["nlim"] = "0"
    log = _run(ccsn, tmp_path / "rejected", deck, failure=conflict)
    assert "### FATAL" in log


def test_core_requires_m1_module(ccsn, tmp_path):
    deck = _deck(ccsn, "core")
    del deck["radiation_m1"]
    deck["time"]["nlim"] = "0"
    log = _run(ccsn, tmp_path / "rejected", deck, failure="recoupling requires M1")
    assert "### FATAL" in log


@pytest.mark.parametrize("ranks", (1, 2))
@pytest.mark.parametrize("method", ("Liebendoerfer", "Simple"))
def test_core_fresh_restart(ccsn, tmp_path, ranks, method):
    """Real-profile M1 and Z4c continuity; no threshold-membership inference."""
    deck = _deck(ccsn, "core")
    deck["problem"]["deleptonization_method"] = method
    whole = tmp_path / "whole"
    _run(ccsn, whole, deck, ranks=ranks)
    expected = _snapshot(ccsn, whole)
    assert expected["cycle"] == 3
    _mode(expected, "core", (True, True), False)
    assert float(expected["parameters"]["problem"]["rho_trap"]) == pytest.approx(1.0e12)
    first = _snapshot(ccsn, whole, first=True)
    assert first["cycle"] == 0
    assert any((expected["fields"]["radiation"][f"E:{s}"] >
                first["fields"]["radiation"][f"E:{s}"]).any() for s in range(4)), (
        "No resolved M1 evolution; supply real emitting data")
    _history(ccsn, whole)
    short = tmp_path / "short"
    deck["time"]["nlim"] = "1"
    _run(ccsn, short, deck, ranks=ranks)
    checkpoint = _restart(short)
    absent_profile = tmp_path / "must_not_read_progenitor"
    reload = tmp_path / "reload"
    _run(ccsn, reload, restart=checkpoint, ranks=ranks,
         overrides=("time/nlim=1", f"problem/progenitor={absent_profile}"))
    _same(ccsn, _snapshot(ccsn, short), _snapshot(ccsn, reload))
    resume = tmp_path / "resume"
    _run(ccsn, resume, restart=checkpoint, ranks=ranks,
         overrides=("time/nlim=3", f"problem/progenitor={absent_profile}"))
    actual = _snapshot(ccsn, resume)
    _mode(actual, "core", (True, True), False)
    _same(ccsn, expected, actual)
    _history(ccsn, resume)


@pytest.mark.parametrize("ranks", (1, 2))
def test_core_bounce_next_full_step(ccsn, tmp_path, ranks):
    """An uninterrupted forced bounce persists and expands on the next step."""
    deck = _deck(ccsn, "core")
    deck["problem"].update(TRIGGER)
    deck["time"]["nlim"] = "2"
    directory = tmp_path / "uninterrupted"
    log = _run(ccsn, directory, deck, ranks=ranks)
    assert "next full step" in log, log
    assert "full-domain" in log, log
    state = _snapshot(ccsn, directory)
    assert state["cycle"] == 2
    _mode(state, "core", (True, True), True)
    hist = _history(ccsn, directory)
    assert hist["t_bounce"] > 0 and hist["D_max_bounce"] > 0


@pytest.mark.parametrize("ranks", (1, 2))
def test_core_stop_at_bounce_postbounce_restart(ccsn, tmp_path, ranks):
    """A stopped split is retained, not compared to uninterrupted evolution."""
    pre = tmp_path / "prebounce"
    deck = _deck(ccsn, "core")
    deck["time"]["nlim"] = "1"
    _run(ccsn, pre, deck, ranks=ranks)
    before = _snapshot(ccsn, pre)
    _mode(before, "core", (True, True), False)
    assert any((before["fields"]["radiation"][f"E:{s}"] > 0).any()
               for s in range(4)), "No radiation to test retention"
    forced = tmp_path / "stop"
    overrides = [f"problem/{key}={value}" for key, value in TRIGGER.items()]
    overrides += ["problem/stop_at_bounce=true", "time/nlim=3"]
    log = _run(ccsn, forced, restart=_restart(pre), overrides=overrides, ranks=ranks)
    assert "Terminating at bounce before the M1 split" in log
    assert "next full step" in log, log
    stopped = _snapshot(ccsn, forced)
    assert stopped["cycle"] == 2
    _mode(stopped, "core", (True, True), True)
    assert stopped["parameters"]["problem"]["deleptonization_method"] == "None"
    for name, values in before["fields"]["radiation"].items():
        ccsn["np"].testing.assert_array_equal(stopped["fields"]["radiation"][name], values)
    hist = _history(ccsn, forced)
    assert hist["t_bounce"] == pytest.approx(stopped["time"], rel=1e-13)
    assert hist["D_max_bounce"] > 0
    checkpoint = _restart(forced)
    reload = tmp_path / "reload"
    _run(ccsn, reload, restart=checkpoint, overrides=("time/nlim=2",), ranks=ranks)
    _same(ccsn, stopped, _snapshot(ccsn, reload))
    continuation = tmp_path / "continuation"
    _run(ccsn, continuation, restart=checkpoint, ranks=ranks,
         overrides=("time/nlim=4", "problem/stop_at_bounce=false"))
    after = _snapshot(ccsn, continuation)
    assert after["cycle"] == 4 and after["time"] > stopped["time"]
    _mode(after, "core", (True, True), True)
    assert not _flag(after, "problem", "stop_before_m1")
    after_hist = _history(ccsn, continuation)
    for name in ("t_bounce", "D_max_bounce"):
        assert after_hist[name] == pytest.approx(hist[name], rel=1e-13)


def _restart(directory):
    files = sorted((directory / "rst").glob("ccsn_test.*.rst"))
    assert files, f"No native restart under {directory}"
    return files[-1]


def _history(ccsn, directory):
    lines = (directory / "ccsn_test.user.hst").read_text(encoding="ascii").splitlines()
    labels = re.findall(r"\[\d+\]=([^\s]+)", lines[1])
    expected = ["time", "dt", "max_rho", "min_alpha", "max_press", "max_MassInCell",
                "max_level", "max_T", "max_MassPerMB", "t_bounce", "D_max_bounce"]
    for species in range(4):
        expected += [f"{name}_{species:02d}" for name in
                     ("max_sc_nG", "max_sc_E", "min_sc_nG", "min_sc_E")]
    expected += ["max_B2", "max_b2", "div_B", "div_B_norm2", "E_B", "Etau_B", "bphisq"]
    assert labels == expected, "CCSN requires all 32 complete, untruncated history labels"
    rows = [[float(v) for v in line.split()] for line in lines if not line.startswith("#")]
    assert rows and all(len(row) == len(expected) for row in rows)
    assert ccsn["np"].isfinite(rows).all()
    assert (directory / "ccsn_test.z4c.hst").is_file(), "Requires QOL history filename fix"
    assert not (directory / "ccsn_test.z4c.user.hst").exists()
    return dict(zip(labels, rows[-1]))


@pytest.mark.parametrize("mode", ("transport", "projection"))
@pytest.mark.parametrize("ranks", (1, 2))
def test_prebounce_fresh_and_restart(ccsn, tmp_path, mode, ranks):
    """Three-cycle fresh vs one-plus-two restart, with no profile on restart."""
    deck = _deck(ccsn, mode)
    whole = tmp_path / "whole"
    _run(ccsn, whole, deck, ranks=ranks)
    expected = _snapshot(ccsn, whole)
    assert expected["cycle"] == 3
    _mode(expected, mode, (True, True), False)
    first = _snapshot(ccsn, whole, first=True)
    assert first["cycle"] == 0
    radiation = expected["fields"]["radiation"]
    if mode == "projection":
        assert all((values == 0).all() for values in radiation.values()), "Pre-bounce mask leaked"
    else:
        assert any((radiation[f"E:{s}"] > first["fields"]["radiation"][f"E:{s}"]).any()
                   for s in range(4)), "No resolved pre-bounce M1 evolution; supply emitting real data"
    hist = _history(ccsn, whole)
    assert hist["t_bounce"] == 0 and hist["D_max_bounce"] == 0
    short = tmp_path / "short"
    deck["time"]["nlim"] = "1"
    _run(ccsn, short, deck, ranks=ranks)
    checkpoint = _restart(short)
    missing = tmp_path / "must_not_read_progenitor"
    reload = tmp_path / "reload"
    _run(ccsn, reload, restart=checkpoint, ranks=ranks,
         overrides=("time/nlim=1", f"problem/progenitor={missing}"))
    _same(ccsn, _snapshot(ccsn, short), _snapshot(ccsn, reload))
    resume = tmp_path / "resume"
    _run(ccsn, resume, restart=checkpoint, ranks=ranks,
         overrides=("time/nlim=3", f"problem/progenitor={missing}"))
    actual = _snapshot(ccsn, resume)
    _mode(actual, mode, (True, True), False)
    _same(ccsn, expected, actual)
    _history(ccsn, resume)


@pytest.mark.parametrize("feedback", ((True, True), (True, False), (False, True), (False, False)))
def test_recoupling_feedback_and_postbounce_restart(ccsn, tmp_path, feedback):
    """Configured flags survive masking/transition; restart retains evolved M1."""
    off = tmp_path / "off"
    deck = _deck(ccsn, feedback=(False, False))
    deck["time"]["nlim"] = "1"
    _run(ccsn, off, deck)
    baseline = _snapshot(ccsn, off)
    active = tmp_path / "configured"
    deck = _deck(ccsn, feedback=feedback)
    deck["time"]["nlim"] = "1"
    _run(ccsn, active, deck)
    before = _snapshot(ccsn, active)
    _mode(before, "transport", feedback, False)
    # Both material and geometry must match the feedback-off case BEFORE bounce.
    # This tests the lifecycle gate, not just the serialized input booleans.
    _same(ccsn, baseline, before)
    initial = _snapshot(ccsn, active, first=True)
    assert any((before["fields"]["radiation"][f"E:{s}"] >
                initial["fields"]["radiation"][f"E:{s}"]).any() for s in range(4)), (
        "Need resolved emission above the initialized radiation to test recoupling")
    bounce = tmp_path / "bounce"
    overrides = [f"problem/{key}={value}" for key, value in TRIGGER.items()]
    overrides += ["problem/stop_at_bounce=true", "time/nlim=3"]
    text = _run(ccsn, bounce, restart=_restart(active), overrides=overrides)
    stopped = _snapshot(ccsn, bounce)
    assert stopped["cycle"] == 2, "Bounce stop must follow exactly one full hydro step"
    assert "Terminating at bounce before the M1 split" in text
    _mode(stopped, "transport", feedback, True)
    # No radiation evolution occurs between this restart and the bounce stop:
    # the pre-existing moments must not be replaced by equilibrium or cleared.
    for key, expected in before["fields"]["radiation"].items():
        ccsn["np"].testing.assert_array_equal(stopped["fields"]["radiation"][key], expected)
    hist = _history(ccsn, bounce)
    assert hist["t_bounce"] == pytest.approx(stopped["time"], rel=1e-13)
    assert hist["D_max_bounce"] > 0
    checkpoint = _restart(bounce)
    reload = tmp_path / "reload"
    _run(ccsn, reload, restart=checkpoint, overrides=("time/nlim=2",))
    _same(ccsn, stopped, _snapshot(ccsn, reload))
    continued = tmp_path / "continued"
    # A stopped run alone never exercises restored feedback. Continue two steps.
    _run(ccsn, continued, restart=checkpoint,
         overrides=("time/nlim=4", "problem/stop_at_bounce=false"))
    after = _snapshot(ccsn, continued)
    assert after["cycle"] == 4 and after["time"] > stopped["time"]
    _mode(after, "transport", feedback, True)
    assert not _flag(after, "problem", "stop_before_m1")
    after_hist = _history(ccsn, continued)
    for key in ("t_bounce", "D_max_bounce"):
        assert after_hist[key] == pytest.approx(hist[key], rel=1e-13)


def test_projection_transition_and_postbounce_restart(ccsn, tmp_path):
    """Check a resolved projection/stop/restart transition, not full-precision balance."""
    deck = _deck(ccsn, "projection")
    # This progenitor peaks below the inherited 1e11 g/cm^3 cutoff at t=0;
    # select genuinely hot cells without changing the projection solver.
    deck["problem"]["eql_rho_min_cgs"] = "1e7"
    # Disable the empirical Ye map on both sides to isolate the local projection.
    deck["problem"]["deleptonization_method"] = "None"
    deck["time"]["nlim"] = "1"
    control_dir = tmp_path / "control"
    _run(ccsn, control_dir, deck)
    control = _snapshot(ccsn, control_dir)
    deck["problem"].update(TRIGGER)
    deck["problem"]["stop_at_bounce"] = "true"
    deck["time"]["nlim"] = "3"
    projected_dir = tmp_path / "projected"
    _run(ccsn, projected_dir, deck)
    projected = _snapshot(ccsn, projected_dir)
    assert projected["cycle"] == control["cycle"] == 1
    assert projected["time"] == control["time"]
    _mode(projected, "projection", (True, True), True)
    _same(ccsn, control, projected, groups=("geometry",))
    np = ccsn["np"]
    old = control["fields"]["matter"]
    new = projected["fields"]["matter"]
    rad = projected["fields"]["radiation"]
    assert any((rad[f"E:{s}"] > 0).any() for s in range(4)), (
        "No cell projected: supply real data/domain and physical EQL cutoffs selecting hot matter")
    assert (new["ener"] != old["ener"]).any(), (
        "No resolved material projection: radiation floors alone are not projection coverage")
    np.testing.assert_array_equal(new["dens"], old["dens"])
    # Binary fields are float32 even in a double build. These dumps cannot
    # certify the unchanged 1e-8 projection tolerance. Full-precision energy,
    # momentum and electron-lepton balance remain separate required checks;
    # no storage-precision tolerance is substituted for the solver criterion.
    checkpoint = _restart(projected_dir)
    reload = tmp_path / "reload"
    _run(ccsn, reload, restart=checkpoint, overrides=("time/nlim=1",))
    _same(ccsn, projected, _snapshot(ccsn, reload))
    continuation = tmp_path / "continuation"
    _run(ccsn, continuation, restart=checkpoint,
         overrides=("time/nlim=3", "problem/stop_at_bounce=false"))
    after = _snapshot(ccsn, continuation)
    assert after["cycle"] == 3
    _mode(after, "projection", (True, True), True)
    old_hist = _history(ccsn, projected_dir)
    new_hist = _history(ccsn, continuation)
    for key in ("t_bounce", "D_max_bounce"):
        assert new_hist[key] == pytest.approx(old_hist[key], rel=1e-13)


def _equilibrium_deck(ccsn, mode="masked"):
    """Use real thermodynamics and a resolved hot-cell gate, never synthetic ICs."""
    deck = _deck(ccsn, mode)
    for key in ("eql_rho_min_cgs", "eql_t_min_mev"):
        assert key in deck["problem"], f"Supply physical equilibrium cutoff {key}"
    # Match the existing conservative transition case's test-only density gate.
    # Keep the supplied temperature cutoff, EOS and profile unchanged. No selected
    # hot cells is a failure, not fabricated coverage or a reason to lower floors.
    deck["problem"]["eql_rho_min_cgs"] = "1e7"
    if mode != "core":
        deck["problem"]["deleptonization_method"] = "None"
    return deck


def _equilibrium_signal(ccsn, snapshot, baseline=None):
    """Require resolved energy AND number moments in every equilibrium group."""
    np = ccsn["np"]
    radiation = snapshot["fields"]["radiation"]
    parameters = snapshot["parameters"]["radiation_m1"]
    for species in range(4):
        for variable, floor_key in (("E", "rad_E_floor"), ("N", "rad_N_floor")):
            key = f"{variable}:{species}"
            values = radiation[key]
            assert (values > 0).any(), f"No positive {key}: need selected real hot cells"
            # A factor of two separates a signal from the float32 rounding of an
            # initialized floor; it is not a solver/conservation tolerance.
            floor = float(parameters[floor_key])
            assert (values > 2 * floor).any(), f"Only floor-scale {key}, not equilibrium coverage"
            if baseline is not None:
                old = baseline["fields"]["radiation"][key]
                assert not np.array_equal(values, old), f"Initialization did not change {key}"


def _manual_state(snapshot, mode="masked", bounce_init=False, conservative=False):
    """A consumed request persists as a marker, never as a fictitious bounce."""
    assert not _flag(snapshot, "problem", "inject_equilibrium")
    assert _flag(snapshot, "problem", "equilibrium_injected")
    assert not _flag(snapshot, "problem", "post_bounce")
    assert _flag(snapshot, "problem", "conservative_bounce_projection") == conservative
    assert _flag(snapshot, "problem", "equilibriate_post_bounce") == bounce_init
    assert _flag(snapshot, "problem", "recoupling_approach") == (mode in ("transport", "core"))
    assert _flag(snapshot, "problem", "prebounce_core_coupling") == (mode == "core")
    assert not _flag(snapshot, "problem", "update_conserved")
    assert float(snapshot["parameters"]["problem"]["t_bounce"]) == 0
    assert float(snapshot["parameters"]["problem"]["D_max_bounce"]) == 0
    for key in ("backreact", "backreact_tmunu"):
        assert _flag(snapshot, "radiation_m1", key), "Do not rewrite configured feedback"


@pytest.mark.parametrize("conservative", (False, True))
def test_manual_equilibrium_requires_native_matter_recovery(ccsn, tmp_path, conservative):
    """Fixed evolution cannot supply recovered startup temperature or primitives."""
    deck = _equilibrium_deck(ccsn)
    deck["time"]["nlim"] = "0"
    deck["mhd"]["fixed"] = "true"
    deck["problem"].update(inject_equilibrium="true", detect_bounce="false",
                           conservative_bounce_projection=str(conservative).lower())
    _run(ccsn, tmp_path / "fixed-rejected", deck, failure="mhd/fixed=false")


def test_manual_conservative_startup_without_bounce_init(ccsn, tmp_path):
    """An explicit request also reaches the retained conservative selector."""
    deck = _equilibrium_deck(ccsn)
    deck["time"]["nlim"] = "0"
    deck["problem"].update(detect_bounce="false", inject_equilibrium="true",
                           conservative_bounce_projection="true")
    directory = tmp_path / "manual-conservative"
    log = _run(ccsn, directory, deck)
    assert "manual M1 equilibrium projected conservatively." in log
    state = _snapshot(ccsn, directory)
    assert state["cycle"] == 0 and state["time"] == 0
    _manual_state(state, conservative=True)
    _equilibrium_signal(ccsn, state)
    # Resolved moments and consumed state must already be present at startup.
    # Full-precision conservation remains the projection fixture's contract.
    _same(ccsn, state, _snapshot(ccsn, directory, first=True))


def test_direct_bounce_preserves_same_hydro_step_matter(ccsn, tmp_path):
    """Force the same entropy stop with and without automatic direct injection."""
    deck = _equilibrium_deck(ccsn, "direct")
    deck["problem"].update(TRIGGER, stop_at_bounce="true")
    deck["time"]["nlim"] = "3"
    direct_dir = tmp_path / "direct"
    log = _run(ccsn, direct_dir, deck)
    direct = _snapshot(ccsn, direct_dir)
    assert "Terminating at bounce before the M1 split" in log
    _mode(direct, "direct", (True, True), True)
    assert not _flag(direct, "problem", "equilibrium_injected"), "Bounce is not a manual request"
    deck["problem"]["equilibriate_post_bounce"] = "false"
    control_dir = tmp_path / "uninitialized"
    log = _run(ccsn, control_dir, deck)
    control = _snapshot(ccsn, control_dir)
    assert "Terminating at bounce before the M1 split" in log
    assert direct["cycle"] == control["cycle"] == 1
    assert direct["time"] == control["time"], "Controls must stop after the same hydro step"
    _mode(control, "masked", (True, True), True)
    _same(ccsn, control, direct, groups=("matter", "geometry"))
    assert all((values == 0).all() for values in control["fields"]["radiation"].values())
    _equilibrium_signal(ccsn, direct, baseline=control)


@pytest.mark.parametrize("mode, bounce_init", (
    ("masked", False), ("masked", True), ("transport", False), ("core", False)))
def test_manual_direct_startup_no_steps(ccsn, tmp_path, mode, bounce_init):
    """Explicit direct startup is independent of the automatic-bounce switch."""
    deck = _equilibrium_deck(ccsn, mode)
    deck["time"]["nlim"] = "0"
    deck["problem"]["equilibriate_post_bounce"] = str(bounce_init).lower()
    control_dir = tmp_path / "startup-control"
    _run(ccsn, control_dir, deck)
    control = _snapshot(ccsn, control_dir)
    assert control["cycle"] == 0 and control["time"] == 0
    assert not _flag(control, "problem", "equilibrium_injected")
    deck["problem"]["inject_equilibrium"] = "true"
    injected_dir = tmp_path / "startup-manual"
    _run(ccsn, injected_dir, deck)
    injected = _snapshot(ccsn, injected_dir)
    _same(ccsn, control, injected, groups=("matter", "geometry"))
    _manual_state(injected, mode, bounce_init)
    _equilibrium_signal(ccsn, injected, baseline=control)
    # The first native output must already see the consumed request and moments:
    # end-of-run injection alone is too late to satisfy startup initialization.
    _same(ccsn, injected, _snapshot(ccsn, injected_dir, first=True))


@pytest.mark.parametrize("mode", ("masked", "transport", "core"))
def test_manual_direct_restart_no_steps(ccsn, tmp_path, mode):
    """Inject into checkpointed current matter without replaying real-profile ICs."""
    deck = _equilibrium_deck(ccsn, mode)
    deck["time"]["nlim"] = "1"
    seed_dir = tmp_path / "seed"
    _run(ccsn, seed_dir, deck)
    seed = _snapshot(ccsn, seed_dir)
    checkpoint = _restart(seed_dir)
    missing = tmp_path / "must-not-read-progenitor"
    overrides = ("time/nlim=1", f"problem/progenitor={missing}")
    control_dir = tmp_path / "restart-control"
    _run(ccsn, control_dir, restart=checkpoint, overrides=overrides)
    control = _snapshot(ccsn, control_dir)
    _same(ccsn, seed, control)
    injected_dir = tmp_path / "restart-manual"
    _run(ccsn, injected_dir, restart=checkpoint,
         overrides=overrides + ("problem/inject_equilibrium=true",))
    injected = _snapshot(ccsn, injected_dir)
    assert injected["cycle"] == 1
    _same(ccsn, control, injected, groups=("matter", "geometry"))
    _manual_state(injected, mode)
    _equilibrium_signal(ccsn, injected, baseline=control)
    reload_dir = tmp_path / "consumed-no-step-reload"
    _run(ccsn, reload_dir, restart=_restart(injected_dir), overrides=overrides)
    reloaded = _snapshot(ccsn, reload_dir)
    _manual_state(reloaded, mode)
    _same(ccsn, injected, reloaded)


@pytest.mark.parametrize("mode", ("masked", "transport", "core"))
def test_consumed_manual_request_evolved_restart_not_replayed_or_erased(ccsn, tmp_path, mode):
    """A persisted manual marker keeps evolved pre-bounce M1 through a roundtrip."""
    deck = _equilibrium_deck(ccsn, mode)
    deck["problem"]["inject_equilibrium"] = "true"
    deck["time"]["nlim"] = "1"
    evolved_dir = tmp_path / "manual-then-evolved"
    _run(ccsn, evolved_dir, deck)
    initialized = _snapshot(ccsn, evolved_dir, first=True)
    evolved = _snapshot(ccsn, evolved_dir)
    assert initialized["cycle"] == 0 and evolved["cycle"] == 1
    _manual_state(initialized, mode)
    _manual_state(evolved, mode)
    _equilibrium_signal(ccsn, initialized)
    _equilibrium_signal(ccsn, evolved)
    assert any(not ccsn["np"].array_equal(values, initialized["fields"]["radiation"][key])
               for key, values in evolved["fields"]["radiation"].items()), (
        "No resolved post-injection evolution: cannot distinguish a replay from retention")
    reload_dir = tmp_path / "evolved-reload"
    _run(ccsn, reload_dir, restart=_restart(evolved_dir),
         overrides=("time/nlim=1", f"problem/progenitor={tmp_path / 'absent-progenitor'}"))
    reloaded = _snapshot(ccsn, reload_dir)
    _manual_state(reloaded, mode)
    _same(ccsn, evolved, reloaded)


@pytest.mark.parametrize("change, message", (
    ({"conservative_bounce_projection": "true"}, "recoupling requires"),
    ({"bounce_detection_method": "density"}, "bounce_detection_method=entropy"),
    ({"bounce_detection_method": "None"}, "bounce_detection_method=entropy"),
    ({"refinement_method": "MaxDensity"}, "refinement_method=MaxMassInCell or none"),
    ({"refinement_method": "unsupported"}, "refinement_method=MaxMassInCell or none"),
    ({"update_conserved": "true"}, "update_conserved=false"),
))
def test_deferred_modes_rejected(ccsn, tmp_path, change, message):
    deck = _deck(ccsn)
    deck["problem"].update(change)
    deck["time"]["nlim"] = "0"
    _run(ccsn, tmp_path / "rejected", deck, failure=message)


@pytest.mark.parametrize("key, value", (("projection_tolerance", "0"),
                                        ("projection_maxiter", "0")))
def test_projection_controls_rejected(ccsn, tmp_path, key, value):
    deck = _deck(ccsn, "projection")
    deck["problem"][key] = value
    deck["time"]["nlim"] = "0"
    _run(ccsn, tmp_path / "rejected", deck, failure="invalid conservative projection controls")


def _profile_rows(ccsn):
    rows = [line.split("#", 1)[0].split() for line in
            ccsn["PROGENITOR"].read_text(encoding="ascii").splitlines()]
    rows = [row for row in rows if row]
    # mass..phi occupies eleven channels; the radius makes twelve columns.
    assert len(rows) >= 2 and all(len(row) in (12, 13) for row in rows), (
        "CCSN_TEST_PROGENITOR must supply real rows r mass vel rho T Ye P e s om ab phi [phiZ]")
    return rows


@pytest.mark.parametrize("case, message", (
    ("missing", "cannot open file"),
    ("empty", "at least two"),
    ("one-row", "at least two"),
    ("short-row", "expected finite columns r through phi"),
    ("bad-radius", "invalid radius"),
    ("nonfinite-column", "expected finite columns r through phi"),
    ("negative-radius", "radii must be finite, nonnegative and strictly increasing"),
    ("duplicate-radius", "radii must be finite, nonnegative and strictly increasing"),
    ("decreasing-radius", "radii must be finite, nonnegative and strictly increasing"),
    ("bad-optional", "invalid optional phiZ column"),
    ("trailing-text", "unexpected trailing column or text"),
))
def test_profile_row_rejection(ccsn, tmp_path, case, message):
    # These are deliberately malformed copies of REAL supplied rows. They are
    # parser inputs only; no synthetic physical profile or expected result.
    rows = _profile_rows(ccsn)[:2]
    if case == "empty":
        rows = []
    elif case == "one-row":
        rows = rows[:1]
    elif case == "short-row":
        rows[0] = rows[0][:11]
    elif case == "bad-radius":
        rows[0][0] = "not-a-radius"
    elif case == "nonfinite-column":
        rows[0][4] = "1e999"
    elif case == "negative-radius":
        rows[0][0] = "-1"
    elif case == "duplicate-radius":
        rows[1][0] = rows[0][0]
    elif case == "decreasing-radius":
        rows.reverse()
    elif case == "bad-optional":
        rows[0] = rows[0][:12] + ["not-phiZ"]
    elif case == "trailing-text":
        rows[0] = rows[0][:12] + ["0", "unexpected"]
    profile = tmp_path / "invalid-profile.txt"
    if case != "missing":
        profile.write_text("# Deliberately invalid parser input\n" +
                           "\n".join(" ".join(row) for row in rows) + "\n", encoding="ascii")
    deck = _deck(ccsn)
    deck["time"]["nlim"] = "0"
    deck["problem"].update(progenitor=str(profile), detect_bounce="false",
                           deleptonization_method="None")
    log = _run(ccsn, tmp_path / "rejected", deck, failure=message)
    assert "stellar profile" in log and str(profile) + ":" in log


def test_profile_optional_column_and_comments(ccsn, tmp_path):
    """Exercise both accepted widths without changing any active physical data."""
    rows = _profile_rows(ccsn)
    results = []
    for width in (12, 13):
        profile = tmp_path / f"profile-{width}.txt"
        # phiZ is optional/unused by these ICs; zero is its documented default.
        text = "# Same real profile; whitespace and inline comments\n\n"
        text += "\n".join("  " + " ".join(row[:12] + (["0"] if width == 13 else [])) +
                          "  # retained real row" for row in rows) + "\n"
        profile.write_text(text, encoding="ascii")
        deck = _deck(ccsn)
        deck["time"]["nlim"] = "0"
        deck["problem"].update(progenitor=str(profile), detect_bounce="false",
                               deleptonization_method="None")
        directory = tmp_path / f"accepted-{width}"
        _run(ccsn, directory, deck)
        results.append(_snapshot(ccsn, directory))
    _same(ccsn, *results)
