#!/usr/bin/env python3
"""Source checks for backend-portable conservative CCSN projection."""

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
    assert "ReportProjectionTrial" not in header
    assert header.count("Kokkos::printf") == 1

    # One code per rejection site of the coupled equilibrium solve: 100 the field
    # is not computable, 101 the chain refused the point, 103/104 the acceptance
    # step of the pair about to be written, 105 the acceptance test refused that pair
    # and the budget ran out, 106 the iteration budget, 110
    # the cell's own state is inadmissible at the start, 111/112/113 the Newton
    # step, and 8 the PrimToCon consistency check.
    for site in ("chain.site == 100 ? 100 : (iteration == 0 ? 110 : 101)",
                 "acceptance.site == 100 ? 104 : 103",
                 "Record(acceptance_refused ? 105 : 106, 0, ctl.max_iterations,",
                 "Record(113, 0, iteration, norm, nullptr, nullptr);",
                 "Record((failed_column >= 0) ? 111 : 112,",
                 "out->detail = 8;"):
        assert site in header, site
    # A refused pair costs the iterations it needs rather than the cell: the
    # acceptance test is bounded by the convergence threshold times a control,
    # because the two tests measure different quantities and the same bound would
    # make the acceptance test the stricter of the two.
    assert "ctl.acceptance_slack * ctl.tolerance" in header
    assert "acceptance_refused = true;" in header
    assert "acceptance_difference = difference;" in header
    assert "acceptance_refused ? acceptance_difference : norm" in header

    # The refused trial and the primitives C2P returned are recorded only where
    # they are defined, so a trial C2P rejected with an error never reports the
    # output array C2P leaves undefined on that path: the fill stops at the error
    # codes (20 and above) and the report carries whether it happened.
    assert "chain.cons, chain.prim);" in header
    assert "acceptance.cons, acceptance.prim);" in header
    assert "if (c2p_prim != nullptr && detail < 20) {" in header
    assert "int c2p_valid;" in header
    assert "fail_out->c2p_valid = 1;" in header

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
    # The line prints the refused point itself at the four sites that relay a
    # chain refusal, so a failure can be replayed offline from the line alone; at
    # 110 the state it would repeat is already printed as cons.
    assert "const int has_trial = (fail.code == 100 || fail.code == 101 ||" in report
    assert "fail.code == 103 || fail.code == 104)" in report
    assert "fail.c2p_valid, fail.c2p_rho, fail.c2p_T, fail.c2p_Ye);" in report

    # The caller prints nothing while the projection succeeds, prints one line
    # per failed cell, and reports the metric instead of a state that cannot be
    # derived from it when the metric is not invertible.
    projection = source.split("static bool ConservativeM1ProjectionImpl", 1)[1]
    assert "ccsn::ProjectionFailureReport fail = {0, 0, -1, -1.0, 0, -1};" in projection
    assert ("ccsn::PrintProjectionFailureLine(rank, gid_start + m, k, j, i, fail,"
            in projection)
    assert "site=107" in projection
    # The only direct print in the projection is that metric report: the accepted
    # path prints nothing and the failure line comes from the header helper.
    assert projection.count("Kokkos::printf") == 1
    assert "Kokkos::printf" not in projection.split("site=107", 1)[1]
    # The run-level context line is printed once, after the failures are counted,
    # and never on the accepted path.
    assert "ProjectionFailureContext(" in projection.split("if (failures != 0) {", 1)[1]
    assert ("### WARNING: conservative M1 bounce projection left " in projection)
    # A cell the solver cannot project is skipped rather than fatal: one cell
    # keeps its pre-bounce state and the run continues, so the per-cell
    # implementation contains no aborting call. The dispatcher may still abort on
    # an unmatched EOS policy, which is a run-level condition.
    impl = projection.split("static void ConservativeM1Projection(", 1)[0]
    assert "MPI_Abort" not in impl
    assert "std::abort" not in impl

    # Acceptance semantics: the acceptance test that gated success before still
    # gates it, now applied to the state the coupled solve writes. What changed is
    # that the unknowns are the matter's conserved variables rather than a radiation
    # blend, that the test and the convergence measure are metric-invariant, and
    # that a refused pair is iterated rather than abandoned.
    assert "if (norm <= ctl.tolerance) {" in header
    assert "ctl.acceptance_slack * ctl.tolerance" in header
    assert "EquilibriumChainResult" in header
    assert "ConservedTotals(old_cons, old_rad, mb, tot)" in header
    assert "Solve5x5(" in header
    assert "MetricContractU" in header
    assert "MetricNormU" in header
    assert "ResidualNorm(residual, scale, g3u, &worst_term)" in header
    # The term that carried the measure starts at -1, "nothing measured yet", and
    # has one name in the report. The field that repeated it as phi_max is gone
    # rather than renamed: it held the same number as residual in every record,
    # including the acceptance-test difference at 105, so no honest second name
    # existed for it.
    assert "int worst_term = -1;" in header
    assert "int *worst_term = nullptr" in header
    assert "fail_out->worst_term = worst_term;" in header
    assert "phi_max" not in header
    assert "phi_index" not in header
    assert "lane" not in header
    # The report and the failure line name the same set of sites for the refused
    # state and for the chain's rejection sub-code, so the two cannot drift apart.
    assert "101, 103, 104 and 110" in header
    assert "filled at 100, 101, 103 and 104" in header
    # The scaled unknowns and the scaled residual say what they are; the
    # two-letter shorthand u/phi is gone.
    assert "Real scaled_x[projection_nvars];" in header
    assert "Real scaled_residual[projection_nvars];" in header
    assert "probe_scaled_residual" in header
    assert "Real phi[" not in header
    # The chain copies only the primitive slots C2P writes, so it never reads the
    # indeterminate tail of the chain result's array.
    assert "prim_copy[n] = out->prim[n];" in header
    assert "for (int n = 0; n < NPRIM; ++n) {" not in header
    assert header.count(
        "const int nprim_active = PYF + ps.GetEOS().GetNSpecies();") == 3
    # The driver carries only the field it needs across iterations: the unused
    # primitive copy is gone.
    assert "prim[n] = chain.prim[n];" not in header
    # The failure line prints the chain evaluations attempted and the term that
    # carried the measure, so one line separates "the entering state was refused"
    # from "the last permitted Newton step was refused" and names what was
    # unsatisfied.
    assert "evals=%d worst_term=%d" in header
    # evals counts attempts rather than successes: the counter is raised before
    # each of the four chain calls, so a refused probe is included and the number
    # on a failure line is what the cell actually cost.
    assert header.count("++evals;") == 4
    for call in ("&chain)) {", "&acceptance)) {",
                 "&probe_chain)) {", "&trial_chain)) {"):
        assert "++evals;" in header.split(call, 1)[0][-400:], call
    assert "ProjectionDifference(rad, acceptance.rad, g3u," in header
    assert "ctl.jacobian_step" in header
    assert "Solve5x5(jacobian, rhs, ctl.pivot_threshold, du)" in header
    # The PrimToCon consistency check measures its difference the way the residual
    # is measured, so no acceptance criterion depends on the spatial coordinate
    # basis: a componentwise maximum would reject in one basis what it accepts in
    # another, and a rejection here stops the run.
    assert "MetricNormU(g3u, check_momentum)" in header
    # The under-relaxation of the field is gone from every layer, and every
    # threshold the line search applies is a control rather than a constant.
    assert "alpha *= ctl.backtrack_factor;" in header
    assert "ctl.line_search_maxiter" in header
    assert "1.0 - ctl.armijo * alpha" in header
    assert "relaxation" not in header
    assert "projection_relaxation" not in source
    # The budget is counted in steps, and every point a permitted step produces is
    # measured and tested before the budget is declared exhausted, so no evaluated
    # point is discarded and no root the budget allows is missed.
    assert "for (int iteration = 0;; ++iteration) {" in header
    assert "if (iteration >= ctl.max_iterations) {" in header
    # The relaxation-toward-a-target loop and its complement helper are gone, and
    # the primitives are no longer an input to the solver.
    assert "MatterComplement" not in header
    assert "trial_rad" not in header
    assert "const Real initial_prim[NPRIM]," not in header
    # The solver holds no numeric policy: every threshold, tolerance and iteration
    # cap is a member of ProjectionControls, and the caller reads each one from the
    # deck, so a failure line can be read against the values the solve used.
    assert "struct ProjectionControls" in header
    assert "const ProjectionControls &ctl," in header
    assert "constexpr Real k" not in header
    assert "constexpr int k" not in header
    assert source.count('GetOrAddReal("problem", "projection_') == 8
    assert source.count('GetOrAddInteger("problem", "projection_') == 2
    for key in ("projection_tolerance", "projection_maxiter",
                "projection_chain_tolerance", "projection_acceptance_slack",
                "projection_momentum_floor", "projection_line_search_maxiter",
                "projection_jacobian_step", "projection_armijo",
                "projection_backtrack_factor", "projection_pivot_threshold"):
        assert f'"{key}"' in source, key
    # Each control is validated by name, and the phrase the behavioural test
    # matches on survives the per-key message.
    assert "invalid conservative projection controls: " in source
    assert source.count("require(\"problem/projection_") == 10


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


def test_ccsn_cell_centered_temperature_state_and_entropy_failure_line():
    source = (ROOT / "src/pgen/ccsn.cpp").read_text(encoding="utf-8")
    history = source.split("static void HistoryMaxT", 1)[1].split(
        "//! \\fn History", 1)[0]
    bounce = source.split("static void BounceEntropyImpl", 1)[1].split(
        "//! \\fn DeleptonizationImpl", 1)[0]
    deleptonization = source.split("void DeleptonizationImpl", 1)[1].split(
        "//! \\fn Deleptonization", 1)[0]
    projection = source.split("static bool ConservativeM1ProjectionImpl", 1)[1].split(
        "// Per-cycle bounce check", 1)[0]

    for cell_centered_path in (history, bounce, deleptonization, projection):
        assert "GetTemperatureFromP" not in cell_centered_path
    assert "temperature_(m, 0, k, j, i)" in history
    assert "temperature_(m, 0, k, j, i)" in bounce
    assert "Real T_old = temperature_(m, 0, k, j, i);" in deleptonization
    assert "temperature_(m, 0, k, j, i) = T_new;" in deleptonization
    assert "initial_prim[PTM] = temperature(m, 0, k, j, i);" in projection

    call_start = deleptonization.index("Kokkos::printf")
    call = deleptonization[call_start : deleptonization.index(");", call_start) + 2]
    format_string, arguments = printf_arguments(call)
    conversions = re.findall(
        r"%[-+ #0]*[\d.*]*(?:hh|h|ll|l|j|z|t|L)?[diouxXeEfFgGaAcspn]",
        format_string)
    assert len(conversions) == len(arguments)
    for field in ("rank=", "gid=", "cell=", "rho=", "rho_cgs=", "n=", "P=",
                  "Ye_old=", "Ye_new=", "dYe=", "T_old=", "T_new=", "Tmin=",
                  "Tmax=", "mu_nu=", "E_nu_avg=", "s_target=", "s_Tmin=",
                  "s_Tmax=", "update_entropy=", "correction_active="):
        assert field in format_string, field
    assert format_string.count("\\n") == 1
