#!/usr/bin/env python3
"""Static CCSN direct/manual initialization contracts; no build or simulation.

These checks constrain source routing and publication order. They do not certify
numerical equilibrium, backend accuracy, conservation, AMR or device execution.
The opt-in real-profile lifecycle cases live in test_suite/ccsn/test_ccsn_prebuilt.py.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# Preserve quoted diagnostic strings while removing comments. Function/block
# extraction then masks literals too, so braces in diagnostics cannot end a body.
TOKENS = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*', re.S)
LITERALS = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')


def _source(relative):
    source = (ROOT / relative).read_text(encoding="utf-8")
    return TOKENS.sub(lambda match: " " if match.group().startswith(("//", "/*"))
                      else match.group(), source)


def _compact(source):
    return re.sub(r"\s+", "", source)


def _block(source, anchor):
    """Extract one balanced C++ brace body after a required regex anchor."""
    masked = LITERALS.sub(lambda match: " " * len(match.group()), source)
    match = re.search(anchor, masked)
    assert match is not None, f"Missing source region: {anchor}"
    start = masked.index("{", match.end())
    depth = 1
    for index in range(start + 1, len(masked)):
        if masked[index] == "{":
            depth += 1
        elif masked[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1:index]
    raise AssertionError(f"Unclosed source region: {anchor}")


def _function(source, name):
    return _block(source, rf"\b{re.escape(name)}\s*\([^;{{}}]*\)\s*(?={{)")


def test_bounce_and_manual_requests_share_one_selector():
    source = _source("src/pgen/ccsn.cpp")
    dispatch = _compact(_function(source, "InitializeM1Equilibrium"))
    assert "if(conservative_bounce_projection){ConservativeM1Projection(pmbp);}" in dispatch
    assert "else{DirectM1Equilibrium(pmbp);}" in dispatch
    assert "equilibriate_post_bounce" not in dispatch
    assert "inject_equilibrium" not in dispatch
    bounce = _function(source, "BounceCheck")
    automatic = _block(bounce, r"if\s*\(equilibriate_post_bounce\)")
    assert "InitializeM1Equilibrium(pmbp);" in automatic
    assert bounce.count("InitializeM1Equilibrium(") == 1
    assert "ConservativeM1Projection(pmbp)" not in bounce
    assert "DirectM1Equilibrium(pmbp)" not in bounce
    assert "equilibrium_injected" not in bounce
    manual = _function(source, "ManualM1Equilibrium")
    assert "InitializeM1Equilibrium(pm->pmb_pack);" in manual
    assert "equilibriate_post_bounce" not in manual


def test_direct_setter_uses_current_matter_and_never_projects_or_converts_it():
    direct = _function(_source("src/pgen/ccsn.cpp"), "DirectM1EquilibriumImpl")
    code = _compact(direct)
    assert "auto&w0=pmbp->pmhd->w0;" in code
    assert "auto&temperature=pdyngr->temperature;" in code
    assert "auto&u0=pmbp->pradm1->u0;" in code
    assert "constRealtemp=temperature(m,0,k,j,i);" in code
    assert "prim[PRH]=rho/mb;" in code
    assert "prim[PTM]=temp;" in code
    assert "prim[PYF]=w0(m,nmhd,k,j,i);" in code
    for component in ("PVX", "PVY", "PVZ", "PPR"):
        assert f"prim[{component}]=w0(" in code
    assert "ccsn::EquilibriumM1Target(" in direct
    for forbidden in ("ProjectM1Cell", "ConservativeM1Projection", "PrimToCon",
                      "ConToPrim", "GetTemperatureFromP", "pmhd->u0", "temperature("):
        if forbidden == "temperature(":
            assert not re.search(r"\btemperature\([^;]*?\)\s*=(?!=)", direct)
        else:
            assert forbidden not in direct, forbidden
    assert not re.search(r"\bw0\([^;]*?\)\s*=(?!=)", direct)


def test_direct_publication_is_after_all_group_validation():
    direct = _function(_source("src/pgen/ccsn.cpp"), "DirectM1EquilibriumImpl")
    code = _compact(direct)
    target = code.index("ccsn::EquilibriumM1Target(")
    densitize = code.index("target[s][v]*=sdetg;")
    finite = code.index("Kokkos::isfinite(target[s][v])", densitize)
    reject = code.index("if(!valid){", finite)
    publish = code.index("u0(m,radiationm1::CombinedIdx(s,v,nvars),k,j,i)=target[s][v];")
    assert target < densitize < finite < reject < publish
    refusal = _block(direct, r"if\s*\(!valid\)")
    assert "return;" in refusal and "++local_failures;" in refusal
    assert "u0(" not in refusal
    assert code.count("u0(") == 1, "No group may publish before full-cell acceptance"
    assert "s<ccsn::projection_nspecies" in code
    assert "v<ccsn::projection_nvars" in code
    for guard in ("if(excise&&excision(m,k,j,i))return;",
                  "if(rho<rho_min||temp<t_gate)return;",
                  "prim[PRH]>=nb_min", "prim[PRH]<=nb_max",
                  "temp>=t_min", "temp<=t_max", "prim[PYF]>=y_min",
                  "prim[PYF]<=y_max", "Kokkos::isfinite(detg)", "detg>0.0",
                  "g3d[S11]>0.0", "Kokkos::isfinite(minor2)", "minor2>0.0"):
        assert guard in code, guard
        assert code.index(guard) < target


def test_direct_target_reuses_eos_and_backend_neutral_equilibrium_helper():
    source = _source("src/pgen/ccsn.cpp")
    direct = _function(source, "DirectM1EquilibriumImpl")
    header = _source("src/pgen/ccsn_m1_projection.hpp")
    target = _function(header, "EquilibriumM1Target")
    densities = _function(header, "EquilibriumDensities")
    for backend_detail in ("radiationm1::BnsNurates", "radiationm1::WeakRates",
                           "nurates_params", "#if ENABLE_NURATES", "FermiDirac"):
        assert backend_detail not in direct + target + densities
    assert "params.opacity_type" in direct
    assert "GetCodeUnitSystem()" in direct and "GetEOSUnitSystem()" in direct
    assert "TemperatureConversion(code_units)" in direct
    for chemical in ("GetBaryonChemicalPotential", "GetChargeChemicalPotential",
                     "GetElectronLeptonChemicalPotential"):
        assert chemical in target
    assert "EquilibriumDensities(opacity_type," in target
    assert "radiationm1::equilibrium_densities(" in densities
    assert "ProjectM1Cell" not in target and "PrimToCon" not in target
    for component in ("M1_E_IDX", "M1_FX_IDX", "M1_FY_IDX", "M1_FZ_IDX", "M1_N_IDX"):
        assert f"target[s][{component}]" in target
    dispatch = _function(source, "DirectM1Equilibrium")
    for logs in ("NQTLogs", "NormalLogs"):
        assert f"DirectM1EquilibriumImpl<EOSCompOSE<{logs}>, ResetFloor>" in dispatch


def test_manual_callback_consumes_request_without_manufacturing_a_bounce():
    manual = _compact(_function(_source("src/pgen/ccsn.cpp"), "ManualM1Equilibrium"))
    initialize = manual.index("InitializeM1Equilibrium(pm->pmb_pack);")
    assert initialize < manual.index("manual_equilibrium_pending=false;")
    assert initialize < manual.index("manual_equilibrium_active=true;")
    assert initialize < manual.index('pin->SetBoolean("problem","inject_equilibrium",false);')
    assert initialize < manual.index('pin->SetBoolean("problem","equilibrium_injected",true);')
    for forbidden in ("post_bounce", "bounce_detected", "t_bounce", "D_max_bounce",
                      "SetCouplingEnabled", "backreact", "SetEvolutionEnabled"):
        assert forbidden not in manual, forbidden


def test_request_registration_and_marker_restore_are_not_repeated_by_amr():
    user = _function(_source("src/pgen/ccsn.cpp"), "ProblemGenerator::UserProblem")
    once = _block(user, r"if\s*\(!user_tasks_enrolled_\)")
    code = _compact(once)
    assert "manual_equilibrium_pending=inject_equilibrium;" in code
    assert ('manual_equilibrium_active=restart&&pin->GetOrAddBoolean('
            '"problem","equilibrium_injected",false);') in code
    assert 'pin->SetBoolean("problem","equilibrium_injected",manual_equilibrium_active);' in code
    assert "if(inject_equilibrium)pgen_init_func=&ManualM1Equilibrium;" in code
    assert user.count("pgen_init_func") == once.count("pgen_init_func") == 1
    assert "InitializeM1Equilibrium(" not in user
    assert "DirectM1Equilibrium(" not in user
    assert "user_tasks_enrolled_=true;" in code
    header = _compact(_source("src/pgen/pgen.hpp"))
    assert "ProblemInitializeFnPtrpgen_init_func=nullptr;" in header
    for path in (ROOT / "src/mesh").glob("*.cpp"):
        assert "pgen_init_func" not in _source(path.relative_to(ROOT)), path.name


def test_manual_marker_and_pending_request_preserve_prebounce_radiation():
    user = _compact(_function(_source("src/pgen/ccsn.cpp"), "ProblemGenerator::UserProblem"))
    assert ("SetEvolutionEnabled(bounce_detected||manual_equilibrium_pending||"
            "manual_equilibrium_active);") in user
    # Fresh ICs zero normally; restart pending injection must not erase the old
    # radiation that the conservative path includes in its input totals.
    assert ("if(!restart&&manual_equilibrium_pending){"
            "pmbp->pradm1->SetEvolutionEnabled(false);}") in user
    assert "SetCouplingEnabled(bounce_detected);" in user
    assert "SetCouplingRegionEnabled(!bounce_detected);" in user


def test_common_cutoff_and_layout_checks_include_direct_manual_and_consumed_marker():
    user = _function(_source("src/pgen/ccsn.cpp"), "ProblemGenerator::UserProblem")
    code = _compact(user)
    assert "constboolmanual_equilibrium=inject_equilibrium||manual_equilibrium_active;" in code
    assert "constboolbounce_equilibrium=detect_bounce&&equilibriate_post_bounce;" in code
    common = _block(user, r"if\s*\(bounce_equilibrium\s*\|\|\s*manual_equilibrium\)")
    checks = _compact(common)
    fixed = _block(common, r"if\s*\(pmbp->pdyngr->IsFixedEvolution\(\)\)")
    assert "mhd/fixed=false" in fixed and "std::exit(EXIT_FAILURE);" in fixed
    assert "radiationm1::equilibrium_available(prad->params.opacity_type)" in checks
    assert "prad->nspecies!=radiationm1::equilibrium_nspecies" in checks
    assert "prad->nvars!=M1_N_IDX+1" in checks
    for key in ("eql_rho_min_cgs", "eql_t_min_mev"):
        read = f'pin->GetReal("problem","{key}")'
        assert code.count(read) == 1 and read in checks
        assert f"!std::isfinite({key})" in checks
        assert f"{key}<0.0" in checks
    assert "!std::isfinite(rho_code_to_cgs)" in checks
    assert "rho_code_to_cgs<=0.0" in checks
    assert "radiationm1::BnsNurates" not in common and "radiationm1::WeakRates" not in common
    assert "nonconservative M1 injection is not supported" not in user


def test_manual_projection_exception_does_not_enable_automatic_core_or_recoupling_init():
    user = _function(_source("src/pgen/ccsn.cpp"), "ProblemGenerator::UserProblem")
    code = _compact(user)
    assert "if(recoupling_approach&&(equilibriate_post_bounce||" in code
    core = _compact(_block(user, r"if\s*\(prebounce_core_coupling\)"))
    assert "reject(equilibriate_post_bounce||" in core
    assert "(conservative_bounce_projection&&!manual_equilibrium)" in code
    assert "(conservative_bounce_projection&&!manual_equilibrium)" in core
    assert "reject(inject_equilibrium" not in core
    projection = _compact(_block(user, r"if\s*\(conservative_bounce_projection\)"))
    assert "!(bounce_equilibrium||manual_equilibrium)" in projection
    assert "params.matter_sources" in projection and "params.backreact" in projection


def test_startup_callback_order_is_after_native_c2p_masks_before_dt_and_output():
    source = _source("src/driver/driver.cpp")
    initialize = _function(source, "Driver::Initialize")
    code = _compact(initialize)
    callback = _compact(_block(initialize, r"if\s*\(pmesh->pgen->pgen_init_func\s*!=\s*nullptr\)"))
    call = code.index("initialize(pin,pmesh);")
    assert code.index("InitBoundaryValuesAndPrimitives(pmesh,res_flag);") < call
    assert code.index("UpdateExcisionMasks();") < call
    assert code.index("SetMask(this,0);") < call
    assert callback.index("pmesh->pgen->pgen_init_func=nullptr;") < callback.index("initialize(pin,pmesh);")
    refresh = code.index("InitBoundaryValuesAndPrimitives(pmesh,res_flag);", call)
    closure = code.index("FloorAndCalcClosure(this,0);", call)
    assert call < refresh < closure < code.index("NewTimeStep(") < code.index("LoadOutputData(")
    assert callback.count("initialize(pin,pmesh);") == 1
    # Native startup recovery must remain the producer of matter and ADM, not a
    # replacement rest-state conversion buried in the equilibrium callback.
    native = _function(source, "Driver::InitBoundaryValuesAndPrimitives")
    assert "ConToPrim" in native and "Z4cToADM" in native
