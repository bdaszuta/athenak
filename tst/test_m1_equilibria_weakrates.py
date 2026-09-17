#!/usr/bin/env python3
"""Source checks for the weakrates extension of the M1 equilibrium interface."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def strip_comments(source):
    """Drop // comments so an explanation naming a host object is not mistaken for a use."""
    return "\n".join(line.split("//")[0] for line in source.splitlines())


def test_weakrates_extends_the_equilibrium_interface():
    source = (
        ROOT / "src/radiation_m1/radiation_m1_equilibria.hpp"
    ).read_text(encoding="utf-8")
    code = strip_comments(source)

    assert "opacity_type == BnsNurates || opacity_type == WeakRates" in code
    assert "weakrates::ComputeEquilibriumDensities(temp, eta_nue)" in code
    assert "weakrates::ConvertToCodeUnits(conversions, densities)" in code

    # The conversion factors are the UnitSystem conversions against CGS evaluated in device code.
    assert "1.0 / code_units.length" in code
    assert "1.0 / code_units.time" in code
    assert "1.0 / eos_units.numberDensity" in code
    assert "code_units.volume / code_units.energy" in code

    # Primitive::CGS is a host object: naming it in this device function fails a GPU build.
    assert "Primitive::CGS" not in code
    assert "Conversion(Primitive::CGS)" not in code


def test_conversion_factors_agree_with_the_unit_system_definitions():
    """The hand-evaluated factors above assume the UnitSystem conversion formulas and a unity CGS.

    Recheck both premises against their definitions, so an edit to either is caught here rather
    than as a silently wrong equilibrium density.
    """
    header = (
        ROOT / "src/eos/primitive-solver/unit_system.hpp"
    ).read_text(encoding="utf-8")
    compact = re.sub(r"\s+", "", header)

    assert "returnb.length/length;" in compact
    assert "returnb.time/time;" in compact
    assert "returnb.numberDensity/numberDensity;" in compact
    assert "return(volume/b.volume)*(b.energy/energy);" in compact

    host = (
        ROOT / "src/eos/primitive-solver/unit_system.cpp"
    ).read_text(encoding="utf-8")
    body = host.split("Primitive::UnitSystem Primitive::MakeCGS()", 1)[1]
    body = body[: body.index("\n}")]
    values = re.findall(
        r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", strip_comments(body)
    )

    assert len(values) == 14
    # Five physical constants, then length, time, numberDensity, volume, mass, energy, pressure,
    # temperature and chemical potential, all unity for CGS.
    assert values[5:] == ["1.0"] * 9


def test_weakrates_equilibrium_state_uses_the_shared_group_count():
    source = (
        ROOT / "src/radiation_m1/radiation_m1.cpp"
    ).read_text(encoding="utf-8")

    assert "radiationm1::equilibrium_available(params.opacity_type)" in source
    assert "nspecies = M1_TOTAL_NUM_SPECIES;" not in source
