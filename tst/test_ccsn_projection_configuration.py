#!/usr/bin/env python3
"""Source checks for the backend-neutral conservative CCSN projection."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
