#!/usr/bin/env python3
"""Source checks for the backend-neutral M1 equilibrium interface."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_equilibrium_interface_is_backend_neutral():
    source = (
        ROOT / "src/radiation_m1/radiation_m1_equilibria.hpp"
    ).read_text(encoding="utf-8")

    assert "bool equilibrium_available(const RadiationM1OpacityType opacity_type)" in source
    assert "bool equilibrium_densities(const RadiationM1OpacityType opacity_type," in source
    assert "constexpr int equilibrium_nspecies = 4;" in source
    assert "static_assert(M1_TOTAL_NUM_SPECIES == equilibrium_nspecies," in source
    # Backends added on other branches extend the availability predicate and the dispatch in
    # this same header, so this test must not require the absence of their names. What it can
    # require is that the interface, not a consumer, owns the selection.
    assert "opacity_type == BnsNurates" in source


def test_bns_nurates_equilibrium_physics_lives_in_the_interface():
    source = (
        ROOT / "src/radiation_m1/radiation_m1_equilibria.hpp"
    ).read_text(encoding="utf-8")

    assert "1.23984172e-10 * 1.0e7" in source
    assert "4.0 * M_PI / hc_mevnm3" in source
    assert "16.0 * M_PI / hc_mevnm3" in source
    assert "Primitive::FermiDirac::F2" in source
    assert "Primitive::FermiDirac::F3" in source
    assert "NumberDensityConversion(ngs_units)" in source
    assert "EnergyDensityConversion(ngs_units)" in source
    assert "0.5 * n_nux_total" in source


def test_group_count_follows_the_backend_and_capacity_is_fixed():
    cpp = (ROOT / "src/radiation_m1/radiation_m1.cpp").read_text(encoding="utf-8")

    assert (
        "radiationm1::equilibrium_available(params.opacity_type) ? M1_TOTAL_NUM_SPECIES : 1"
        in cpp
    )
    assert "nspecies = M1_TOTAL_NUM_SPECIES;" not in cpp

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    species_setup = cmake.split("#set user problem generator flag", 1)[0]

    assert species_setup.count("set(M1_TOTAL_NUM_SPECIES 4)") == 1
    assert species_setup.index("set(M1_TOTAL_NUM_SPECIES 4)") < species_setup.index(
        "if(Athena_ENABLE_NURATES)"
    )
