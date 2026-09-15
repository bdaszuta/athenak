#!/usr/bin/env python3
"""Source checks for weakrates species capacity and equilibrium-density entry points."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_weakrates_has_four_species_with_or_without_nurates():
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    species_setup = cmake.split("#set user problem generator flag", 1)[0]

    assert species_setup.count("set(M1_TOTAL_NUM_SPECIES 4)") == 1
    assert "set(M1_TOTAL_NUM_SPECIES 3)" not in species_setup
    assert species_setup.index("set(M1_TOTAL_NUM_SPECIES 4)") < species_setup.index(
        "if(Athena_ENABLE_NURATES)"
    )


def test_equilibrium_densities_accept_thermodynamic_state_directly():
    source = (
        ROOT / "src/radiation_m1/radiation_m1_weakrates.hpp"
    ).read_text(encoding="utf-8")

    assert "ComputeEquilibriumDensities(const Real temp_mev," in source
    assert "const Real eta_nue)" in source
    state_overload = source.split(
        "EquilibriumDensities ComputeEquilibriumDensities(const State &state)", 1
    )[1]
    assert "ComputeEquilibriumDensities(state.temp_mev, state.eta_nue)" in state_overload


def test_weakrates_uses_c2p_temperature_when_available():
    source = (
        ROOT / "src/radiation_m1/radiation_m1_calc_opacities_weakrates.cpp"
    ).read_text(encoding="utf-8")
    compact = " ".join(source.split())

    assert "auto temperature_ = pdyngr->temperature;" in compact
    assert "!pdyngr->IsFixedEvolution()" in compact
    assert "? temperature_(m, 0, k, j, i)" in compact
    assert ": eos.GetTemperatureFromP(nb, p, ye_arr)" in compact


def test_weakrates_kernel_uses_the_four_slot_mapping_only():
    source = (
        ROOT / "src/radiation_m1/radiation_m1_calc_opacities_weakrates.cpp"
    ).read_text(encoding="utf-8")

    assert "assert(nspecies == M1_TOTAL_NUM_SPECIES);" in source
    assert "constexpr Real heavy_split = 0.5;" in source
    assert "nspecies_ == 4" not in source
