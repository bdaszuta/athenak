//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Standalone source-only regression candidate. See docs/hip-mi300a.md.
// Checks explicit DualView handoffs and host-mirror storage after reallocation.

#include <initializer_list>
#include <iostream>
#include <type_traits>

#include "athena.hpp"

namespace {
using Values = DualArray2D<int>;
constexpr bool separate_sides =
    !std::is_same_v<Values::t_dev::device_type, Values::t_host::device_type>;

static_assert(std::is_same_v<Values::t_dev::array_layout, Kokkos::LayoutRight>);
static_assert(std::is_same_v<Values::t_host::array_layout, Kokkos::LayoutRight>);
#if defined(KOKKOS_IMPL_HIP_UNIFIED_MEMORY)
static_assert(std::is_same_v<HostMemSpace, Kokkos::HIPSpace>);
static_assert(std::is_same_v<Values::t_dev::memory_space,
                             Values::t_host::memory_space>);
static_assert(separate_sides, "HIP APU still needs distinct logical sides");
#else
static_assert(std::is_same_v<HostMemSpace, Kokkos::HostSpace>);
#endif

void Require(bool condition, const char *message) {
  if (!condition) Kokkos::abort(message);
}

void RoundTrip(Values &values, int seed) {
  const int rows = values.extent_int(0);
  const int cols = values.extent_int(1);
  auto device = values.d_view;
  Kokkos::parallel_for("dualview_device_write", Kokkos::RangePolicy<DevExeSpace>(0, rows),
      KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < cols; ++j) device(i,j) = seed + i*cols + j;
      });
  values.modify_device();
  Require(!separate_sides || values.need_sync_host(), "Device write not recorded");
  values.sync_host();
  Require(!values.need_sync_host() && !values.need_sync_device(),
          "Device-to-host handoff left a dirty side");

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      Require(values.h_view(i,j) == seed + i*cols + j, "Device-to-host value mismatch");
      values.h_view(i,j) += 7;
    }
  }
  values.modify_host();
  Require(!separate_sides || values.need_sync_device(), "Host write not recorded");
  values.sync_device();
  Require(!values.need_sync_host() && !values.need_sync_device(),
          "Host-to-device handoff left a dirty side");

  Kokkos::parallel_for("dualview_device_update", Kokkos::RangePolicy<DevExeSpace>(0, rows),
      KOKKOS_LAMBDA(int i) {
        for (int j = 0; j < cols; ++j) device(i,j) *= 2;
      });
  values.modify_device();
  values.sync_host();
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      Require(values.h_view(i,j) == 2*(seed + i*cols + j + 7),
              "Host-to-device-to-host value mismatch");
    }
  }
  // The handoff above completes device work before the borrowed handle expires.
}
}  // namespace

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    // Match the HostArray::HostMirror assignment used by EOS table readers.
    DvceArray2D<int> device("mirror_storage", 2, 3);
    HostArray2D<int>::HostMirror host = Kokkos::create_mirror_view(device);
    Kokkos::deep_copy(device, 11);
    Kokkos::deep_copy(host, device);
    Require(host(1,2) == 11, "HostArray mirror value mismatch");

    Values values("dualview_roles", 4, 3);
    for (int rows : {4, 7, 2}) {
      // Reallocation deliberately discards a host-owned value; no device work is pending.
      values.h_view(0,0) = -1;
      values.modify_host();
      Kokkos::realloc(values, rows, 3);
      Require(!values.need_sync_host() && !values.need_sync_device(),
              "Reallocation did not reset the dirty sides");
      RoundTrip(values, rows);
      RoundTrip(values, rows + 1);
    }
  }
  Kokkos::finalize();
  std::cout << "DualView role and mirror checks passed\n";
  return 0;
}
