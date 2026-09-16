#!/usr/bin/env python3
"""Source checks for the backend-neutral conservative CCSN projection."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def printf_arguments(call):
    """Return a printf call's format string and its top-level arguments."""
    body = call[call.index("(") + 1 : call.rindex(")")]
    parts, depth, current, in_string, escaped = [], 0, "", False, False
    for char in body:
        if in_string:
            current += char
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(current)
            current = ""
            continue
        current += char
    parts.append(current)
    format_string = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', parts[0]))
    return format_string, parts[1:]


def test_projection_header_is_backend_neutral():
    source = (ROOT / "src/pgen/ccsn_m1_projection.hpp").read_text(encoding="utf-8")

    assert "#if ENABLE_NURATES" not in source
    assert "NuratesParams" not in source
    assert "radiation_m1_nurates.hpp" not in source
    assert "radiation_m1_weakrates.hpp" not in source
    assert "weakrates" not in source
    assert "radiationm1::WeakRates" not in source


def test_projection_header_uses_the_shared_equilibrium_interface():
    source = (ROOT / "src/pgen/ccsn_m1_projection.hpp").read_text(encoding="utf-8")

    assert "radiation_m1/radiation_m1_equilibria.hpp" in source
    assert "radiationm1::equilibrium_densities(" in source
    assert "static_assert(projection_nspecies == radiationm1::equilibrium_nspecies" in source
    # the backend physics itself must not reappear here
    assert "Primitive::FermiDirac" not in source
    assert "1.23984172e-10 * 1.0e7" not in source
    assert "16.0 * M_PI / hc_mevnm3" not in source
    assert "NumberDensityConversion(ngs_units)" not in source
    assert "EnergyDensityConversion(ngs_units)" not in source


def test_projection_driver_asks_the_interface_which_backends_work():
    source = (ROOT / "src/pgen/ccsn.cpp").read_text(encoding="utf-8")
    projection = source.split("static bool ConservativeM1ProjectionImpl", 1)[1].split(
        "// Per-cycle bounce check", 1
    )[0]
    validation = source.split("if (conservative_bounce_projection)", 1)[1].split(
        "if (detect_bounce)", 1
    )[0]

    assert "#if ENABLE_NURATES" not in projection
    assert "nurates_params" not in projection
    assert "params.opacity_type" in projection
    assert "radiationm1::equilibrium_available(" in validation
    assert "radiationm1::equilibrium_nspecies" in validation
    assert "radiationm1::BnsNurates" not in validation
    assert "radiationm1::WeakRates" not in validation
    assert "#if ENABLE_NURATES" not in validation


def test_documentation_states_the_backend_neutral_contract():
    documentation = (ROOT / "doc/ccsn.md").read_text(encoding="utf-8")
    verification = (ROOT / "tst/CCSN_VERIFICATION.md").read_text(encoding="utf-8")

    statement = "The projection source names no backend."
    assert statement in documentation
    assert statement in verification


def test_projection_failure_instrumentation_reports_every_site():
    header = (ROOT / "src/pgen/ccsn_m1_projection.hpp").read_text(encoding="utf-8")
    source = (ROOT / "src/pgen/ccsn.cpp").read_text(encoding="utf-8")

    # RecoverMatter names the rejection that fired, one code per test, in the
    # same order as the combined condition it replaced.
    assert "int *fail_code" in header
    for code in (0, 2, 3, 4, 5, 6, 7):
        assert f"*fail_code = {code};" in header
    # A C2P error is named by its enumerator, not lumped into one code.
    assert "*fail_code = 20 + static_cast<int>(result.error);" in header
    assert (header.index("result.error") < header.index("result.cons_floor") <
            header.index("result.prim_floor") < header.index("result.cons_adjusted"))

    # Every rejection site records into the caller's report, and the recording
    # path prints nothing: a failure reported across several prints could not be
    # matched up with its own state in a parallel log.
    assert "ProjectionFailureReport *fail_out" in header
    assert "RecordProjectionFailure(fail_out, 100, 0, iteration," in header
    assert "RecordProjectionFailure(fail_out, 101, recover_code, iteration," in header
    assert "RecordProjectionFailure(fail_out, 102, 0, iteration," in header
    assert "RecordProjectionFailure(fail_out, 103, recover_code, iteration," in header
    assert "RecordProjectionFailure(fail_out, exact_target_ok ? 105 : 104, 0," in header
    assert "RecordProjectionFailure(fail_out, 106, 0, max_iterations," in header
    assert "ReportProjectionTrial" not in header
    assert header.count("Kokkos::printf") == 1

    # The refused trial and the primitives C2P returned are recorded only where
    # they are defined, so a trial C2P rejected with an error never reports the
    # output array C2P leaves undefined on that path.
    assert "last_diff, trial_cons," in header
    assert "exact_cons, exact_prim);" in header
    assert "if (c2p_prim != nullptr && detail >= 2 && detail <= 5) {" in header

    # One self-contained line per failed cell: identity, site, the full matter
    # state, the full M1 state, the refused trial and the C2P status reach one
    # printf, and that printf ends the line exactly once.
    report = header.split("void PrintProjectionFailureLine", 1)[1].split("\n}\n", 1)[0]
    assert "rank=%d gid=%d cell=(%d,%d,%d) site=%d" in report
    for group in ("| prim rho=", "| cons D=", "| met=", "| B_u=",
                  "| fail rho=", "| trial=%d D=", "| c2p=%d rho="):
        assert group in report, group
    assert "| m1 nue E=" in report
    assert report.count("E=% .9e Fx=% .9e Fy=% .9e Fz=% .9e N=% .9e") == 4
    for species in (" nue E=", " anue E=", " nux E=", " anux E="):
        assert report.count(species) == 1
    assert report.count("\\n") == 1
    assert report.count("Kokkos::printf") == 1
    assert "const int has_trial = (fail.code == 101 || fail.code == 103)" in report
    assert "const int has_c2p = (fail.detail >= 2 && fail.detail <= 5)" in report

    # The caller prints nothing while the projection succeeds, prints one line
    # per failed cell, and reports the metric instead of a state that cannot be
    # derived from it when the metric is not invertible.
    projection = source.split("static bool ConservativeM1ProjectionImpl", 1)[1]
    assert "ccsn::ProjectionFailureReport fail = {0, 0, -1};" in projection
    assert ("ccsn::PrintProjectionFailureLine(rank, gid_start + m, k, j, i, fail,"
            in projection)
    assert "site=107" in projection
    # The only direct print in the projection is that metric report: the accepted
    # path prints nothing and the failure line comes from the header helper.
    assert projection.count("Kokkos::printf") == 1
    assert "Kokkos::printf" not in projection.split("site=107", 1)[1]
    # The run-level context line is printed once, after a failure is counted and
    # before the abort, and never on the accepted path.
    assert "ProjectionAbortContext(" in projection.split("if (failures != 0) {", 1)[1]
    assert "### FATAL: conservative M1 bounce projection failed in " in projection

    # Acceptance semantics are untouched: the relaxed tolerance test and the
    # exact-pass residual test still gate success.
    assert "if (last_diff <= tolerance) {" in header
    assert "exact_diff > tolerance" in header


def test_projection_failure_line_format_matches_its_arguments():
    header = (ROOT / "src/pgen/ccsn_m1_projection.hpp").read_text(encoding="utf-8")
    body = header.split("void PrintProjectionFailureLine", 1)[1].split("\n}\n", 1)[0]
    call = body[body.index("Kokkos::printf") : body.index(");") + 2]
    format_string, arguments = printf_arguments(call)

    # Kokkos::printf forwards to printf through a variadic template, so the
    # compiler cannot check this format against its argument list. A mismatch
    # would corrupt the only diagnostic of a failed bounce projection, so the
    # counts are checked here instead.
    conversions = re.findall(
        r"%[-+ #0]*[\d.*]*(?:hh|h|ll|l|j|z|t|L)?[diouxXeEfFgGaAcspn]",
        format_string)
    assert len(conversions) == len(arguments)
    assert "" not in [argument.strip() for argument in arguments]
    assert format_string.count("\\n") == 1
