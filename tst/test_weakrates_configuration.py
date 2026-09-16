#!/usr/bin/env python3
"""Source checks for weakrates species capacity, equilibrium, and EOS fractions."""

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


def test_weakrates_uses_tabulated_free_nucleon_fractions():
    header = (
        ROOT / "src/radiation_m1/radiation_m1_weakrates.hpp"
    ).read_text(encoding="utf-8")
    kernel = (
        ROOT / "src/radiation_m1/radiation_m1_calc_opacities_weakrates.cpp"
    ).read_text(encoding="utf-8")

    assert "Real xn;" in header
    assert "Real xp;" in header
    assert "const Real xn = state.xn;" in header
    assert "const Real xp = state.xp;" in header
    assert "1.0 - ye" not in header
    assert "state.ye" not in header

    get_xp = "const Real xp = eos.GetProtonFraction(nb, temp, ye_arr);"
    get_xn = "const Real xn = eos.GetNeutronFraction(nb, temp, ye_arr);"
    make_state = "const weakrates::State local_state = weakrates::MakeState("
    assert get_xp in kernel
    assert get_xn in kernel
    assert kernel.index(get_xp) < kernel.index(make_state)
    assert kernel.index(get_xn) < kernel.index(make_state)
    assert "rho_eval_cgs, temp, baryon_mass_cgs, xn, xp" in kernel
    assert "ComputeEquilibriumDensities(temp_star, eta_nue_star)" in kernel


def test_weakrates_coherent_nucleus_scattering_wiring():
    eos_header = (
        ROOT / "src/eos/primitive-solver/eos.hpp"
    ).read_text(encoding="utf-8")
    compose_header = (
        ROOT / "src/eos/primitive-solver/eos_compose.hpp"
    ).read_text(encoding="utf-8")
    compose_cpp = (
        ROOT / "src/eos/primitive-solver/eos_compose.cpp"
    ).read_text(encoding="utf-8")
    wr_header = (
        ROOT / "src/radiation_m1/radiation_m1_weakrates.hpp"
    ).read_text(encoding="utf-8")
    kernel = (
        ROOT / "src/radiation_m1/radiation_m1_calc_opacities_weakrates.cpp"
    ).read_text(encoding="utf-8")

    # EOS loader stores the heavy-nucleus fields and the mass fraction X_h.
    assert "ECXH" in compose_header
    assert "ECAN" in compose_header
    assert "ECZN" in compose_header
    assert "ECNVARS = 12" in compose_header
    assert 'table.HasField("Y[N]")' in compose_cpp
    assert 'table.HasField("A[N]")' in compose_cpp
    assert 'table.HasField("Z[N]")' in compose_cpp
    assert "const Real xh = an*table_yn[iflat];" not in compose_cpp
    assert "Real xh = an_cl*yn;" in compose_cpp
    assert "xh = (xh < 1.0) ? xh : 1.0;" in compose_cpp
    assert "zn = (zn > an_cl) ? an_cl : zn;" in compose_cpp

    # EOS accessors expose the three heavy-nucleus quantities.
    assert "HeavyFraction(Real n, Real T, Real *Y)" in compose_header
    assert "MassNumber(Real n, Real T, Real *Y)" in compose_header
    assert "ChargeNumber(Real n, Real T, Real *Y)" in compose_header
    assert "EOSPolicy::HeavyFraction(" in eos_header
    assert "EOSPolicy::MassNumber(" in eos_header
    assert "EOSPolicy::ChargeNumber(" in eos_header

    # Weakrates state carries the fractions and the coherent term.
    assert "Real xh;" in wr_header
    assert "Real abar;" in wr_header
    assert "Real zbar;" in wr_header
    assert "0.0625" in wr_header
    assert "Square(1.0 - state.zbar/state.abar)" in wr_header
    assert "state.xh*scatter_h" in wr_header

    # The kernel queries all three and passes them to MakeState.
    assert "const Real xh = eos.GetHeavyFraction(nb, temp, ye_arr);" in kernel
    assert "const Real abar = eos.GetMassNumber(nb, temp, ye_arr);" in kernel
    assert "const Real zbar = eos.GetChargeNumber(nb, temp, ye_arr);" in kernel
    assert (
        "rho_eval_cgs, temp, baryon_mass_cgs, xn, xp, xh, abar, zbar" in kernel
    )


def test_weakrates_kernel_uses_the_four_slot_mapping_only():
    source = (
        ROOT / "src/radiation_m1/radiation_m1_calc_opacities_weakrates.cpp"
    ).read_text(encoding="utf-8")

    assert "assert(nspecies == M1_TOTAL_NUM_SPECIES);" in source
    assert "constexpr Real heavy_split = 0.5;" in source
    assert "nspecies_ == 4" not in source
