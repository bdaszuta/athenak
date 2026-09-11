# CCSN problem generator

This source port uses `PROBLEM=ccsn` with dynamical GRMHD and ADM. It needs the
QOL branch's paired 32-entry history/reduction capacities. No global task-list
assembly changes are required. The accepted radiation solver, explicit-stage
mask/closure/stress-energy ordering, and post-M1 material refresh are retained.

No build, test, simulation or accelerator validation was performed during the
port. See `tst/` for the separately invocable regression sources. The actual
progenitor and table are external inputs; this repository does not supply a
replacement for the reference deck's `FILLMEATCMD`.

## Initial data and startup AMR

The profile columns are:

    r mass vel rho T Ye P e s om ab phi [phiZ]

The mapping requires the donor's geometric-solar profile normalization and MeV
temperature; CompOSE runs reject other `mhd/units` settings. Density conversion
for CGS deleptonization/bounce thresholds comes
from the active EOS code unit system, not a copied density constant. `bdm_r_max`
is a coordinate radius in code units. The profile reader accepts whitespace and
comments, requires at least two strictly increasing nonnegative radii, validates
finite numeric fields and column counts, and initializes every channel copied to
the device. The optional phiZ channel is zero when absent. `interpolate_Phi=true`
uses the tabulated phi; otherwise the donor's forward Newtonian potential
integration supplies the initial metric. This is not a relativistic constraint
solve or a new stellar-equilibrium construction.

`refinement_method=MaxMassInCell` uses the largest cell baryon mass
`dx1*dx2*dx3*u0(IDN)` in a block. `delta_min_m` and `delta_max_m` supply the
coarsening/refinement thresholds. `none` disables the pgen criterion. The QOL
startup loop repeatedly evaluates initial data on the resulting hierarchy,
including boundary/C2P initialization, before initial outputs. It bypasses only
runtime refinement cadence and notifies mesh-update caches after redistribution.
It does not run on restart. The initial one-block-per-rank restriction remains:
startup AMR cannot rescue an over-ranked root mesh.

Magnetic initial data retain the donor's vector-potential coefficient and
edge-sampling/discrete-curl procedure. `B0_amp` has not been calibrated here as a
specified central field in gauss; neither magnetic normalization nor an
alternative regularized field model has been validated. Ordinary rotation uses
`Omega_0` and `Omega_A`; nondefault disconnected OmegaGRB controls are rejected.

## Once-per-step material update

The pgen appends its local chain once per ProblemGenerator instance, including
restart construction. Repeated startup IC evaluations do not append more tasks.
The chain runs after hydro RK and before the M1 split:

    Liebendoerfer update / entropy bounce / selected transition
    -> material U receive setup, restriction, exchange, prolongation, BCs, C2P
    -> material U completion
    -> radiation U receive setup, restriction, exchange, prolongation, BCs
    -> radiation U completion and mask refresh

MHD uses its existing U communication methods and native coupled prolongation/BC
path. Radiation receive setup and completion use the existing methods with
`stage=-1`, so no restricted-flux receives are posted. The scheduler retains
ownership of incomplete receives. The separate accepted post-M1 refresh is not
replaced by this pre-M1 chain.

Deleptonization and entropy bounce require CompOSE. `Liebendoerfer` and the
legacy `Simple` spelling select the same CompOSE entropy update; `None` disables
it. `update_conserved=false` is required: the native P2C maps the thermodynamic
change into moving, magnetized GRMHD conserved variables. The donor's rest-only
tau overwrite is deliberately not ported. `update_entropy` retains its selected
behavior. The entropy accessor is per baryon, not the existing per-mass accessor;
`bdm_spb_max` is entropy per baryon in k_B for geometric-solar units.

Entropy inversion delegates to the existing CompOSE interpolation helper. It
does not change the shared energy/pressure inversion or add a table-boundary
clamping policy. Out-of-range behavior inherits that helper's assertion or
extrapolation behavior; the pgen rejects nonfinite, nonpositive or out-of-table
temperatures before another lookup. The table's bracketing and round-trip behavior need
verification before production use.

## M1 activation choices

Masked pre-bounce evolution with conservative equilibrium projection:

    <problem>
    detect_bounce = true
    bounce_detection_method = entropy
    recoupling_approach = false
    inject_equilibrium = false
    equilibriate_post_bounce = true
    conservative_bounce_projection = true

Supply `eql_rho_min_cgs` and `eql_t_min_mev`. This requires four-species M1 with an
opacity backend that supplies four-group equilibrium densities, Ye, material sources and
configured material backreaction. The backend is selected through the deck's
`radiation_m1/opacity_type` value: BNS-NuRates requires Athena_ENABLE_NURATES=ON, and the
standalone weakrates backend works with it either OFF or ON. Each backend supplies the
equilibrium target using the same constants and units as its opacity implementation.
The projection source names no backend. The projection uses unchanged donor
tolerances and rejects failed or repaired C2P
trial states; accepted cells preserve combined matter/radiation energy, momentum
and electron lepton number by constructing the complementary material conserved state.

Evolving one-way M1 before bounce, followed by recoupling without replacing its
moments:

    <problem>
    detect_bounce = true
    bounce_detection_method = entropy
    recoupling_approach = true
    inject_equilibrium = false
    equilibriate_post_bounce = false
    conservative_bounce_projection = false

This mode keeps radiation transport and radiation matter-source updates active.
Only feedback to fluid and geometry is suppressed before bounce. At bounce the
configured `radiation_m1/backreact` and `backreact_tmunu` settings become effective;
settings configured false remain false. They are not overwritten in parameter
storage. This one-way interval is deliberately not conservative coupled evolution.

The coupling switch is independent of the global evolution/excision mask. Every
mask refresh composes lifecycle disabling with coordinate excision, including
ghosts and newly active AMR slots, and invalidates closure state. Masked cells
have E/F/N and chi zeroed. Unmasking and recoupling preserve unexcised moments.

## Bounce stop and restart

`stop_at_bounce=true` terminates after the selected transition and completed
material/radiation U refresh, before the first activated M1 split. The driver
counts the completed hydro step with its physical timestamp and lets finalization
write the stop state before another AMR redistribution. It reports bounce
termination rather than a wall-clock limit.

`post_bounce`, `t_bounce`, `D_max_bounce` and the disabled deleptonization setting
are recorded in restart parameters. Reconstruction restores the selected M1 mode
before initialization consumers and does not replay ICs on restart. A pre-bounce
recoupling-mode restart retains its evolved radiation rather than zeroing it.
For a post-bounce continuation test, explicitly set `stop_at_bounce=false`; a
bounce-stop-only run does not test continuing recoupled evolution.

Nonconservative equilibrium injection and alternative bounce/refinement criteria
are deferred and rejected rather than silently substituted. Projection conservation,
entropy-table behavior, incomplete MPI receive progress, AMR/restart transitions,
all 32 histories, and actual MI300A ownership transitions remain runtime checks.
