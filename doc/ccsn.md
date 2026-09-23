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
The projection source names no backend. The projection solves the coupled matter/radiation equilibrium directly: the
unknowns are the matter's own undensitized conserved variables (energy, momentum
and electron lepton number), the radiation is eliminated by its closure, and a
damped Newton iteration with a line search drives the five conservation statements
to zero. Donor C2P tolerances are unchanged, and a failed or repaired C2P state is
rejected at every evaluation, including the state that is finally written. The
written matter state is the exact complement of the field it is paired with, so
accepted cells preserve combined matter/radiation energy, momentum and electron
lepton number to roundoff: that is a property of the pair that is constructed, not
of any tolerance.
Ten controls reach the projection, all read from `<problem>` with the defaults
below; the solver holds no threshold, tolerance or iteration cap of its own, and
the failure context line prints every one of them so a failure can be read against
the values the solve used. A run that changes one is changing the decision a cell
is judged by:

| key | default | what it bounds |
|---|---|---|
| `projection_tolerance` | `1e-8` | relative bound on the scaled residual that ends the Newton solve |
| `projection_maxiter` | `50` | budget of Newton steps; counted so the point the last permitted step produces is measured and tested before the budget is declared exhausted |
| `projection_chain_tolerance` | `1e-10` | PrimToCon round trip every chain point must pass; this grades a state the code reconstructed itself, so it can be no tighter than the primitive solver that produced it; its energy term is measured against the larger of the cell's total and rest-mass energy density, since the conserved energy is rebuilt by subtracting the rest-mass term |
| `projection_acceptance_slack` | `10.0` | factor on `projection_tolerance` giving the acceptance bound of the written pair |
| `projection_momentum_floor` | `1e-3` | smallest momentum scale, as a speed in units of D, so a cell at rest still has a nonzero scale |
| `projection_line_search_maxiter` | `25` | backtracking trials per Newton step |
| `projection_jacobian_step` | `1e-6` | relative finite-difference step of the scaled unknowns |
| `projection_armijo` | `1e-4` | sufficient-decrease constant of the line search |
| `projection_backtrack_factor` | `0.5` | step-shortening factor per backtracking trial |
| `projection_pivot_threshold` | `1e-14` | relative pivot floor of the 5x5 solve |

The last five change the path the solve takes rather than the criterion it is
judged by: loosening `projection_armijo` or the backtracking factor can stop a
cell converging, so change them with a reason and re-run the convergence checks.
The deck's `projection_relaxation` key is retired: a Newton step with a line
search has no relaxation to tune.
Before a pair is written it is re-derived through its own exact complement and
required to reproduce the field it came from within `projection_acceptance_slack`
times `projection_tolerance`. That test bounds the mutual consistency of the
pair, and it is deliberately looser than the convergence test: the two measure
different quantities - the residual is scaled by the cell totals and the acceptance
difference by the field alone, so the same bound would make the acceptance test the
stricter of the two by the field's dimensionless response to the matter. A pair
that fails it is not written; the solve continues from the acceptance point, so a
refusal costs iterations, bounded by the step budget.
Nothing is printed while the projection succeeds. A cell that cannot be projected
reports one line per failed cell: device output from different threads and ranks is
not ordered, so a failure split across several prints cannot be matched up with its
own state afterwards. The line names the cell (MPI rank, global meshblock id and
meshblock-local k, j, i), the failing site as `site`, `detail` and `iter`, and the
state the projection consumed there: the matter primitives (`prim`: rho, the three
velocity components `W v^i`, P, T, Ye), the undensitized matter conserved state
(`cons`), the spatial metric (`met`),
the contravariant field (`B_u`), the undensitized four-species M1 state (`m1`:
E, Fx, Fy, Fz and N per species, in nue, anue, nux, anux order) and the last
point whose chain evaluation completed (`fail`; at 100 and 104 that is the
previous point, the state the test saw being the `c2p` group). A site that relays
a chain refusal also carries the state the chain refused (`trial`, valid when
`trial=1`, that is sites 100, 101, 103 and 104; at 110 the chain holds the
entering state, which the `cons` group above already prints) and the primitives
C2P returned (`c2p`, valid when `c2p=1`, that is whenever C2P returned a state
rather than an error, which is any `detail` below 20). Every quantity is in the
projection's own variables and units, so the
failing call can be replayed offline without further conversion: rho is a baryon
number density in 1/code-length^3, T is in MeV, and the matter and radiation
conserved arrays are undensitized. The velocity components are the ones the
primitive solver stores, `W v^i` (`zx`, `zy`, `zz` in the line, the projection's
`z^i` in `PVX..PVZ`), and not `v^i`: the replay takes `v^i = z^i / W` with
`W = sqrt(1 + g_ij z^i z^j)` from the metric the line prints. The metric site (107)
has no undensitized state to report, so its line carries the cell, the metric and
the matter primitives instead.

`site` is the failing test: 100 the equilibrium field is not computable, 101 the
matter recovery or its consistency check refused the point, 103 the exact
complement of the converged field is not recoverable, 104 its equilibrium field is
not computable, 105 the acceptance test of the converged pair refused at least once
and the step budget ran out before the pair moved inside the slack, 106
the iteration budget is exhausted, 107 the spatial metric is not invertible, 110
the cell's own state is not admissible at the start point, 111 a Jacobian column
could not be formed, 112 the Jacobian is numerically singular, 113 the line search
found no step that lowered the residual.
`diff` is the scaled residual of the last point the chain evaluated successfully,
the acceptance difference at 105, and -1 before any residual has been evaluated;
`evals` is the number of chained evaluations attempted by then (a refused probe
counts), and `worst_term` is the
term that carried that residual, as the term index 0 energy, 1 momentum, 4 electron
lepton number (-1 before the first evaluation). Sites 101, 103, 104 and 110 carry
the rejection sub-code as `detail` (2 conserved floor, 3 primitive floor, 4
conserved adjustment, 5 conserved value changed or nonfinite, 6 active primitive
nonfinite, 7 magnetic field changed, or 20 plus the Primitive::Error enumerator
when C2P returned an error: 21 RHO_TOO_BIG, 22 RHO_TOO_SMALL, 23 NANS_IN_CONS, 24
MAG_TOO_BIG, 25 BRACKETING_FAILED, 26 NO_SOLUTION, 27 CONS_FLOOR, 28 PRIM_FLOOR, 29
CONS_ADJUSTED), 8 when the primitives C2P returned do not reproduce the requested
conserved state through the code's own PrimToCon, the failed column index at 111,
and 0 otherwise - including at 110 when the start point is refused for an invalid
scale rather than by the chain. `trial` carries the state the chain refused and is
printed at 100, 101, 103 and 104; at 110 that state is the entering state the `cons`
group already prints. The code-to-CGS conversions, the equilibrium gate, the
projection controls, the M1 floors and the EOS error-policy floors that are
identical for every cell are printed once as a context line, and the rank-local
failure count is reported in a
`### WARNING: conservative M1 bounce projection left N cell(s) unprojected on rank
R` line.
A failed cell is counted and skipped, not fatal: nothing of its state is written,
so it keeps the pre-bounce moments and the conserved state it entered with and is
unmasked with the rest of the grid. That is the same condition a cell below the
equilibrium cutoff is in, and it is why a post-bounce state should be read against
that WARNING line: a skipped cell carries no equilibrium field. Only run-level
conditions stay fatal - the startup validation of the projection's controls and of
the backends it needs, and an EOS policy the projection has no instantiation for.

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

### Opt-in pre-bounce core coupling

`<problem>/prebounce_core_coupling=false` by default. When true, it extends
recoupling rather than selecting the conservative projection. Merge these keys
into a suitable CCSN deck with its actual EOS, progenitor, units, opacity backend,
reaction controls, floors, mesh and output settings:

    <problem>
    prebounce_core_coupling = true
    recoupling_approach = true
    detect_bounce = true
    bounce_detection_method = entropy
    deleptonization_method = Liebendoerfer
    rho_trap = 1.0e12
    equilibriate_post_bounce = false
    conservative_bounce_projection = false
    inject_equilibrium = false
    update_conserved = false

    <radiation_m1>
    matter_sources = true
    backreact = true
    backreact_tmunu = true

`Simple` is also accepted as the pre-bounce empirical-method spelling. The new
mode requires dynamical relativistic GRMHD/ADM with CompOSE, exactly one `Ye`
scalar, four-group number-evolving M1 with an equilibrium-capable opacity backend,
material collision sources and both material/gravity feedback switches. If Z4c
evolves spacetime, `backreact_tmunu=true` is required. Projection and equilibrium
injection are incompatible. `gr_sources` independently controls geometric terms
in the radiation equation; it is not the radiation-gravity feedback switch.
Invalid combinations and nonfinite or nonpositive `rho_trap` fail at startup.
Post-bounce restarts accept the persisted `deleptonization_method=None`.

`rho_trap` is a rest-mass density in g/cm^3 (default `1.0e12`), converted with
the active code unit system. Immediately before each full explicit step, the
selection task compares primitive rest density `w0(IDN)` with the converted
threshold. Equality belongs to the coupled core. This is not a comparison of
the metric-densitized conserved `D`, an average, a radial cutoff or an optical
depth. Membership is frozen through hydro, the post-hydro empirical update and
both M1 split stages. A cell crossing the threshold changes treatment at the
next full-step selection, not partway through the current step. Excision remains
a separate *current-time* override and is never frozen into this selection.

| Phase and region | M1 transport/collisions | M1 material feedback | Radiation gravity, when spacetime evolves | Empirical map |
|---|---|---|---|---|
| Before bounce, selected (`rho >= rho_trap`) | On | On | On | Off |
| Before bounce, unselected | On | Off | Off | On |
| Bounce-detection step | On, unless stopping before split | Frozen regional treatment | Frozen regional treatment | Complement already updated |
| First following full step and later | On | On in nonexcised cells | On in nonexcised cells | Off |
| Excised cells | Existing excision handling | Off | Off | Off in this mode |

Fresh runs retain the existing recoupling initialization with zero radiation
moments; an initially dense core is **not** seeded with an equilibrium reservoir.
Pre-bounce restarts retain stored moments and select cells from recovered
primitives at the first pre-integrator task, not from unrecovered restart data.
After runtime AMR or redistribution, that task reads current blocks and recovered
primitives again; selection is neither transferred with AMR nor stored in restart.
The radiation mask and evolved moments are not reset on density crossings.

In this mode the detecting step keeps its regional selection for the remaining
M1 split. The persisted bounce state disables the empirical map and expands
feedback throughout the nonexcised domain at the **next** full-step boundary;
the expansion does not project or clear the moments. By contrast, ordinary
recoupling still changes its global feedback switch immediately at bounce.
With `stop_at_bounce=true`, the detecting step finishes its material refresh but
skips that M1 split. A subsequent restart begins globally coupled from the
stored radiation and does not replay the skipped split. It is not bitwise
equivalent to an uninterrupted trajectory.

This regional phase is not globally conservative: exterior radiation collisions
have no opposite material deposit, and empirical exterior changes do not fill a
matching neutrino reservoir. Radiation may carry that mismatch into the core;
global recoupling also changes gravitational sources without a projection.
Only the accepted local collision increments in selected, nonexcised cells have
paired material energy, momentum and electron-lepton deposits. A float32 binary
dump cannot certify those full-precision identities; see the direct native checks
described in `tst/CCSN_VERIFICATION.md`. A physical improvement or an exactly
conservative interface is not established by this mode.

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
