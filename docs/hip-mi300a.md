# HIP / MI300A build recipe

This recipe was checked against source only. No configure, compilation, test,
simulation, hardware validation, or performance measurement was performed.
Commands below are for a later, separately authorized build.

## Prerequisites

- Use an MI300A APU with a site-supported ROCm compiler/runtime, HMM/XNACK support,
  and an MPI installation compatible with that toolchain. Set `ROCM_PATH` to its
  installation prefix. Device-buffer MPI communication requires suitable HIP-aware
  MPI support; selecting `Athena_ENABLE_MPI` does not establish that support.
- The checkout must already contain its pinned `kokkos/` sources at
  `6739bc623081648af9e752b616d9671527922cbf`. The HIP cleanup worktree has an
  uninitialized submodule; source verification used the read-only, clean copy at
  `../../project_collapsar/kokkos`, which is at the same revision. No dependency
  was initialized, copied, or updated for this change.
- Use a fresh out-of-source build directory, Unix Makefiles (not Ninja), and at
  most six build jobs. Do not reuse a CUDA or discrete-MI300X cache.

## Backend and architecture selection

From the root of a prepared checkout:

```sh
: "${ROCM_PATH:?Set ROCM_PATH to the site ROCm installation}"
cmake -S . -B build-hip-mi300a -G "Unix Makefiles" \
  -DCMAKE_CXX_COMPILER="$ROCM_PATH/bin/hipcc" \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_HIP=ON \
  -DKokkos_ARCH_AMD_GFX942_APU=ON \
  -DKokkos_ARCH_AMD_GFX942=OFF \
  -DKokkos_ENABLE_CUDA=OFF \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_OPENMP=OFF \
  -DAthena_ENABLE_OPENMP=OFF \
  -DAthena_ENABLE_MPI=ON \
  -DAthena_ENABLE_NURATES=OFF \
  -DAthena_SINGLE_PRECISION=OFF \
  -DPROBLEM=built_in_pgens
cmake --build build-hip-mi300a --parallel 6
```

`built_in_pgens` and NuRates OFF make this recipe independent of the separate
CCSN feature. Only after integration supplies `src/pgen/ccsn.cpp` and the approved
`bns_nurates/include` dependency path should a CCSN build use `-DPROBLEM=ccsn`
and `-DAthena_ENABLE_NURATES=ON`. The requested NuRates revision is
`d48bfc5ac422f3608d84b29229660cc74807c396`; this change does not vendor or update it.
Athena MPI/OpenMP/NuRates options alone do not select HIP or the APU architecture.

### Source evidence for the flags

All Kokkos paths below refer to the pinned revision above:

- `cmake/kokkos_arch.cmake:114-134,245-251,988-1029` declares
  `Kokkos_ARCH_AMD_GFX942_APU`, maps it to `gfx942`, and supplies
  `--offload-arch=gfx942` for HIP. The separate `AMD_GFX942` option is not the APU
  option; selecting both architectures is rejected.
- `cmake/KokkosCore_config.h.in:141` exports `KOKKOS_ARCH_AMD_GFX942_APU`.
  `core/src/setup/Kokkos_Setup_HIP.hpp:20-25,57-61` defines
  `KOKKOS_IMPL_HIP_UNIFIED_MEMORY` for HIP plus that APU macro. Do not define this
  internal macro manually or substitute a generic unified-memory option.
- `core/src/HIP/Kokkos_HIP_Space.hpp:306-314` makes `HIPSpace` host-accessible
  under that macro. `core/src/Kokkos_HostSpace.hpp:144-169` then retains the
  allocation space for a host mirror, while selecting host execution.
- `core/src/HIP/Kokkos_HIP.cpp:77-95` requires XNACK for APU host allocations and
  warns about the wrong hardware. For a later launch, set `HSA_XNACK=1` in the
  job environment and inspect the actual runtime configuration (for example,
  `KOKKOS_PRINT_CONFIGURATION=1`). An environment variable cannot enable missing
  system support or establish correct rank-to-device placement.

## DualView operations and pending verification

`HostMemSpace` is a storage alias, not a logical-side selector. Under the APU
macro it is `HIPSpace`; otherwise it stays `HostSpace`. Existing layouts,
allocations, and owning handles are retained. Host writes use `modify_host()`
then `sync_device()`; device writes use `modify_device()` then `sync_host()`.
This includes the multigrid execution-space calls and compact-object lapse
mirror. No template-form DualView `modify`/`sync` calls remain in retained `src/`.
The host tracker loop in `pgen/dyn_grmhd/dynbbh.cpp` also writes `h_view`, matching
its host-to-device handoff; actual kernel writes remain on `d_view`.

Shared storage still needs execution ordering. At this Kokkos revision,
`containers/src/Kokkos_DualView.hpp:645-728` performs a no-execution-instance
`deep_copy` for a dirty opposite-side handoff; the equal-span path in
`core/src/Kokkos_CopyViews.hpp:1163-1172` fences even when pointers alias. No
blanket fences were added. A clean-side sync is not a general fence, and a future
custom execution instance or dependency revision requires a fresh ordering audit.

`tst/unit/dualview_roles.cpp` is an uncompiled, unrun standalone regression
candidate. It checks mirror assignment, dirty-side reset, repeated device/host
handoffs, and growth/shrink reallocation with `LayoutRight`. It is not registered
in the existing Python simulation suite. A future standalone target must use
C++17, link the same `Kokkos::kokkos`, and include this checkout's `src/` and the
Athena build directory containing generated `config.hpp`.

Pending authorized validation: build and run that candidate on CPU, discrete HIP,
and MI300A; exercise actual repeated AMR, particle exchange, interpolation/output,
and restart paths with the intended MPI setup. A Debug build enables the existing
Kokkos bounds and DualView checks. None of these results is claimed here.
