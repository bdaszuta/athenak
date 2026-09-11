# CCSN prebuilt-executable regressions

Source only: these tests have not been executed. No CCSN build, simulation,
pytest invocation, test-module import or numerical experiment was performed
while authoring them. Runtime verification requires separate authorization.

## Inputs and eventual invocation

`test_suite/ccsn/test_ccsn_prebuilt.py` uses native pytest discovery and the
existing `vis/python/bin_convert.py` reader. It never calls build helpers. Do
not use `run_test_suite.py` for this scope: that entry point automatically builds.

Provide absolute paths through these test-only environment variables:

- `CCSN_TEST_EXE`: an already built CCSN executable, with CompOSE, four-species
  M1 with a four-group equilibrium opacity backend (BNS-NuRates, or the standalone
  weakrates backend), and dynamical GRMHD/Z4c support. Use the
  reviewed integrated source, including the QOL 32-entry history
  capacity/labels/filename changes.
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

After runtime authorization, from this checkout's `tst/` directory:

```sh
python -m pytest -q -rs test_suite/ccsn/test_ccsn_prebuilt.py
```

Do not use pytest-xdist or other concurrent test workers. Each process uses one
OpenMP thread; MPI cases use exactly two ranks. Inputs are copied into pytest's
`tmp_path`, reduced to a non-AMR 16x16x16 root grid with eight 8x8x8 blocks and four
ghost cells, and capped at six total cycles (most launches use zero to three).
Each launch has a 120-second process-group timeout. Commands, native binary /
restart / history outputs and logs stay in that case's temporary directory.
Production decks, profiles, tables and executables are never edited or generated.
No CMake options, builds or executable downloads are part of these tests.

## Covered paths when later executed

- Real-profile startup and malformed-row rejection: missing/empty/single-row
  files, short rows, bad/nonfinite fields, negative/duplicate/decreasing radii,
  invalid optional phiZ and trailing text. Malformed cases derive from actual
  supplied rows, not invented stellar models. Both accepted column counts,
  blank lines and inline comments must give the same initialized fields.
- Bounded pre-bounce fresh vs interrupted/restarted evolution for both M1 modes,
  serial and two-rank MPI. Restart points to a deliberately absent progenitor:
  reopening ICs fails. Native matter, radiation and Z4c fields are compared.
  Masked radiation must remain zero; transport must produce resolved evolution.
- Recoupling with all four configured `(backreact, backreact_tmunu)` combinations.
  Pre-bounce fields must agree with the feedback-off control; configured flags
  must remain intact. A forced entropy event stops before M1, retains the evolved
  moments, survives a zero-step restart, and continues two post-bounce steps with
  `stop_at_bounce=false`, preserving `t_bounce` and `D_max_bounce`.
- Conservative projection at a forced entropy event: compare equal-time
  post-hydro states against a masked non-bouncing control with deleptonization
  disabled on both sides. Require an actual projected cell and unchanged D.
  Full-precision conservation is not certified by these dumps. Then reload and
  continue the projected checkpoint. Physical EQL cutoffs are not weakened.
- Explicit rejection of nonconservative injection, alternate bounce/refinement
  criteria, incompatible recoupling/projection and invalid projection controls.
  `update_conserved=true` is rejected; all valid cases retain native GRMHD P2C.
- All 32 untruncated CCSN history labels, finite fields, distinct user/Z4c history
  filenames and persisted bounce diagnostics.

`bdm_spb_max` is entropy in s/k_B per baryon. Normal smoke runs use a deliberately
unreachable density gate; transition cases lower the entropy threshold only to
force the real entropy predicate immediately. Those events are lifecycle tests,
not physical bounce predictions or a validation of the entropy-unit accessor.

## Remaining manual or higher-precision verification

All runtime results remain unverified. These tests require real, emitting/hot
input data on the small retained domain; no such dataset was resolved here.
Native binary fields are float32 even with a double executable. State-preserving
comparisons require exact stored-field equality, without ULP bands or excluded
cells; restart-roundoff differences must be diagnosed rather than silently hidden.
These dumps do NOT certify the unchanged projection solver tolerance (normally
1e-8). Check full-precision equilibrium residuals and matter+radiation totals,
electron-lepton conservation using the active EOS baryon mass, moving/magnetized
P2C consistency and cell-solver failure/no-live-write behavior separately.

Post-bounce continuation exercises restored feedback but does not independently
measure each feedback channel's source term or detect every possible accidental
forcing-on of a configured-off channel. That needs direct material/Tmunu source
instrumentation. Also remaining: natural bounce/entropy-table units and inversion,
exact callback count/stage ordering, repeated startup AMR/direct refined ICs,
coarse-fine U-only communication without flux receives, excision composition,
AMR/load-balancing restarts, HIP ownership/reallocation, device accuracy and
magnetic normalization. Two-rank non-AMR completion alone does not prove those.
