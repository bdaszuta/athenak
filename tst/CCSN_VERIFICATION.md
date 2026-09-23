# CCSN prebuilt-executable regressions

The pytest suite consumes a separately built CCSN executable and supplied data;
it does not configure or build AthenaK. The native wrapper compiles a focused
regression main against the matching build's objects.

## Inputs and invocation

`test_suite/ccsn/test_ccsn_prebuilt.py` uses native pytest discovery and the
existing `vis/python/bin_convert.py` reader. It never calls build helpers. Do
not use `run_test_suite.py` for this scope: that entry point automatically builds.

Provide absolute paths through these test-only environment variables:

- `CCSN_TEST_EXE`: an already built CCSN executable, with CompOSE, four-species
  M1 with a four-group equilibrium opacity backend (BNS-NuRates, or the standalone
  weakrates backend), and dynamical GRMHD/Z4c support. Use the
  reviewed integrated source, including the QOL 32-entry history
  capacity/labels/filename changes. The standalone feature branch's CCSN
  `static_assert(NHISTORY_VARIABLES >= 32)` does not compile with its 20-entry
  history capacity; do not remove the assertion as a test workaround.
- `CCSN_TEST_INPUT`: a real CCSN deck, with `<mhd> dyn_eos=compose`, one Ye scalar,
  geometric-solar units, `<adm>`, `<z4c>`, `<radiation_m1>` and `<problem>`.
  Keep suitable domain bounds, floors, magnetic/rotation controls, reaction
  settings and physical EQL cutoffs in this deck.
- `CCSN_TEST_EOS`: the actual compatible CompOSE table; not a guessed local path.
- `CCSN_TEST_PROGENITOR`: the actual geometric-solar stellar profile, with rows
  `r mass vel rho T Ye P e s om ab phi [phiZ]`, T in MeV.
- `CCSN_TEST_MPIEXEC`: optional absolute path to an MPI launcher accepting
  `-n 2`; the supplied CCSN executable must itself support MPI. No launcher flags
  or alternate rank count are accepted. Without this, only MPI cases skip.

Missing assets/dependencies produce explicit skips. Supplied but incompatible
inputs, missing output, nonfinite fields or no resolved radiation/projection
signal fail; they are not reported as successful physics coverage. The supplied
reference production deck still has `FILLMEATCMD`; it is not a resolved progenitor.
Python requires pytest, numpy and h5py (the native reader imports h5py).

The standalone weakrates backend works with Athena_ENABLE_NURATES either OFF or ON;
BNS-NuRates requires Athena_ENABLE_NURATES=ON. Complete backend/build coverage
therefore runs this suite with weakrates executables from both builds and with a
BNS-NuRates executable from the enabled build. The projection source names no backend.
The selected backend reaches it only through the module equilibrium interface.

From this checkout's `tst/` directory, with the environment variables above:

```sh
python -m pytest -q -rs test_suite/ccsn/test_ccsn_prebuilt.py
```

Do not use pytest-xdist or other concurrent test workers. Each process uses one
OpenMP thread; MPI cases use exactly two ranks. Inputs are copied into pytest's
`tmp_path`, reduced to a non-AMR 16x16x16 root grid with eight 8x8x8 blocks and four
ghost cells, and capped at four total cycles (most launches use zero to three).
Each launch has a 120-second process-group timeout. Commands, native binary /
restart / history outputs and logs stay in that case's temporary directory.
Production decks, profiles, tables and executables are never edited or generated.
No CMake options, builds or executable downloads are part of these tests.

## Paths exercised by the prebuilt suite

- Real-profile startup and malformed-row rejection: missing/empty/single-row
  files, short rows, bad/nonfinite fields, negative/duplicate/decreasing radii,
  invalid optional phiZ and trailing text. Malformed cases derive from actual
  supplied rows, not invented stellar models. Both accepted column counts,
  blank lines and inline comments must give the same initialized fields.
- Bounded pre-bounce fresh vs interrupted/restarted evolution for both M1 modes,
  serial and two-rank MPI. Restart points to a deliberately absent progenitor:
  reopening ICs fails. Native matter, radiation and Z4c fields are compared.
  Masked radiation must remain zero; transport must produce resolved evolution.
- Opt-in core mode: startup rejects incompatible recoupling/bounce/projection,
  empirical-method, EOS, scalar, M1 opacity/module/source/backreaction and
  density settings (including zero, negative and nonfinite `rho_trap`). A fresh
  `Simple` alias and post-bounce `None` restart are admitted. Three-cycle fresh
  and one-plus-two restart runs compare matter, radiation and Z4c fields with
  no progenitor on restart; omitted vs explicit false compares legacy fields
  for each configured feedback combination.
- Core-mode forced bounce, uninterrupted through the following full step, checks
  the deferred-expansion diagnostic and persisted bounce history. The separate
  stop-before-M1 case retains radiation moments through the event and zero-step
  post-bounce reload, then evolves from that checkpoint with empirical `None`.
  Both fresh/restart and forced-bounce cases include optional two-rank MPI;
  without `CCSN_TEST_MPIEXEC` those MPI cases **skip**, not pass. Neither an
  end-of-step binary dump nor the default low-density smoke profile proves that
  cells above `rho_trap` were selected.
- Recoupling with all four configured `(backreact, backreact_tmunu)` combinations.
  Pre-bounce fields must agree with the feedback-off control; configured flags
  must remain intact. A forced entropy event stops before M1, retains the evolved
  moments, survives a zero-step restart, and continues two post-bounce steps with
  `stop_at_bounce=false`, preserving `t_bounce` and `D_max_bounce`.
- Conservative projection at a forced entropy event: compare equal-time
  post-hydro states against a masked non-bouncing control with deleptonization
  disabled on both sides. Require an actual projected cell and unchanged D.
  Full-precision conservation is not certified by these dumps. Then reload and
  continue the projected checkpoint. This test alone sets
  `problem/eql_rho_min_cgs=1e7` because the supplied progenitor's initial peak
  density is below the inherited `1e11` cutoff; production inputs are unchanged.
- Explicit rejection of nonconservative injection, alternate bounce/refinement
  criteria, incompatible recoupling/projection and invalid projection controls.
  `update_conserved=true` is rejected; all valid cases retain native GRMHD P2C.
- All 32 untruncated CCSN history labels, finite fields, distinct user/Z4c history
  filenames and persisted bounce diagnostics.

`bdm_spb_max` is entropy in s/k_B per baryon. Normal smoke runs use a deliberately
unreachable density gate; transition cases lower the entropy threshold only to
force the real entropy predicate immediately. Those events are lifecycle tests,
not physical bounce predictions or a validation of the entropy-unit accessor.

## Direct native core-coupling checks (separate execution)

Build the reviewed integrated `PROBLEM=ccsn` target, then invoke
`python tst/test_ccsn_core_coupling.py BUILD EOS PROGENITOR`. This is **not**
the prebuilt-only pytest command above: it compiles/links a focused C++ test
against the actual CCSN/M1/GRMHD/EOS objects from that build. The entry point
must verify the CMake source root, CCSN problem choice, double precision,
NuRATES availability and serial MPI-off configuration; reuse the recorded
`main.cpp` compile command and `athena` link command, replacing only its object
and output. No old objects from another checkout may be linked. Run cases in
separate processes to avoid CCSN's file-scope state crossing cases.

The native entry point contains seven cases: selection/frozen crossings,
controlled explicit and implicit sources, physical NuRATES coefficients on both
sides of the selector, current-time coordinate excision, Tmunu addition, and
bounce handoff. Interface transport, limited sources, AMR regeneration, and a
direct legacy-disabled comparison from section 8.3 are not implemented in this
entry point. A missing, skipped or nonexecuted native case cannot be reported as
a pass; the physical-opacity case checks finite, nonnegative and nonzero rates,
not their calibrated accuracy.

Report Serial, two-rank MPI and device (MPI-off CUDA/HIP) outcomes separately.
A Serial result is not a device result: repeat the native source against an
approved device build without discarding its compiler flags. A two-rank
prebuilt run checks global bounce history/continuation but does not measure
per-cell deposition or substitute for native AMR and excision checks. The
prebuilt suite disables AMR and writes float32 fields even for a double build;
its field equality checks establish lifecycle smoke behavior, not full-precision
collision balance or a calibrated core. Compare false-mode outputs against the
pre-change integrated baseline using the same compiler/backend for a regression
claim; comparing omitted and explicit false on one new build is not enough.

## Remaining manual or higher-precision verification

The supplied real progenitor and CompOSE table exercise the bounded serial
smoke cases, but native binary fields are float32 even with a double executable.
State-preserving comparisons require exact stored-field equality, without ULP
bands or excluded cells; restart-roundoff differences must be diagnosed rather
than silently hidden.
These dumps do NOT certify the projection solver tolerance (normally 1e-8) or its
convergence. The projection solves the coupled matter/radiation equilibrium in the
matter's conserved variables, so check separately: the invariant scaled residual
and the Newton iteration and evaluation counts at the failing cells, the same with
a nonzero field, the acceptance difference of the state that is written together
with its ratio to the acceptance slack (a margin near one is a pair the containment
path is about to iterate on or catch), and that
the written matter state is the exact complement of the written field, so combined
matter+radiation totals and the electron lepton number (using the active EOS baryon
mass) are preserved to roundoff. Also check that a state pinned at a C2P floor is
rejected rather than written (site 101, detail 8, the PrimToCon consistency check),
that a refused cell is skipped without stopping the run and is counted in the
projection WARNING line,
moving/magnetized P2C consistency, and cell-solver failure/no-live-write behavior.

Post-bounce continuation exercises restored feedback but does not independently
measure each feedback channel's source term or detect every possible accidental
forcing-on of a configured-off channel. That needs direct material/Tmunu source
instrumentation. Also remaining: natural bounce/entropy-table units and inversion,
exact callback count/stage ordering, repeated startup AMR/direct refined ICs,
coarse-fine U-only communication without flux receives, excision composition,
AMR/load-balancing restarts, HIP ownership/reallocation, device accuracy and
magnetic normalization. Two-rank non-AMR completion alone does not prove those.
