//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wave_extract_rwz.cpp
//! \brief Read-only, completed-state Regge-Wheeler-Zerilli extraction.
//!
//! Physical derivatives are produced on cells, communicated with independent Z4c
//! point-field buffers, and then interpolated only by the owning block. Geometry
//! and harmonics live on device; only compact integrals and final modes use host
//! storage. The inherited radial-map approximations are retained deliberately.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/finite_diff.hpp"
#include "utils/legendre_roots.hpp"
#include "utils/spherical_harm.hpp"
#include "z4c/wave_extract_rwz.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_gauge_rhs.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
// Unexpected failures must not strand peers in the next diagnostic collective.
[[noreturn]] void RWZAbort(const char *reason, int root = 0) {
  if (global_variable::my_rank == root) {
    std::cerr << "### FATAL ERROR: RWZ extraction: " << reason << std::endl;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

bool ProductFits(std::initializer_list<std::size_t> dims, std::size_t limit) {
  std::size_t product = 1;
  for (const auto n : dims) {
    if (n != 0 && product > limit/n) return false;
    product *= n;
  }
  return true;
}

// Value and selected Cartesian derivatives of one interpolated field. Mask bits
// 0,1,2 select x,y,z derivatives in code-coordinate units. The caller has already
// admitted the owner and stencil bounds. Keep the native i,j,k sum order and
// weight association while loading each field value only once per stencil cell.
struct RWZInterpolatedField {
  Real value;
  Real derivative[3];
};

template <int NGHOST, int DerivativeMask>
KOKKOS_INLINE_FUNCTION
RWZInterpolatedField InterpolateRWZField(
    const DvceArray5D<Real> &fields, int var, const RegionIndcs &ind,
    const IndAndWghts<NGHOST> &iw) {
  RWZInterpolatedField sample{};
  for (int i = 0; i < 2*NGHOST; ++i) {
    for (int j = 0; j < 2*NGHOST; ++j) {
      const Real w12 = iw.wght1[i]*iw.wght2[j];
      for (int k = 0; k < 2*NGHOST; ++k) {
        const Real value = fields(iw.ii0,var,
            iw.ii3-(NGHOST-k-ind.ks)+1,iw.ii2-(NGHOST-j-ind.js)+1,
            iw.ii1-(NGHOST-i-ind.is)+1);
        const Real weight = w12*iw.wght3[k];
        sample.value += weight*value;
        if constexpr ((DerivativeMask & 1) != 0) {
          const Real weight_x = iw.dwght1[i]*iw.wght2[j]*iw.wght3[k];
          sample.derivative[0] += weight_x*value;
        }
        if constexpr ((DerivativeMask & 2) != 0) {
          const Real weight_y = iw.wght1[i]*iw.dwght2[j]*iw.wght3[k];
          sample.derivative[1] += weight_y*value;
        }
        if constexpr ((DerivativeMask & 4) != 0) {
          const Real weight_z = w12*iw.dwght3[k];
          sample.derivative[2] += weight_z*value;
        }
      }
    }
  }
  return sample;
}

// Calling a completed boundary operation twice can repost receives or reuse
// payloads. Repeat only the operation currently reporting incomplete.
template <typename Operation>
void CompleteBoundary(Operation operation) {
  TaskStatus status;
  do { status = operation(); } while (status == TaskStatus::incomplete);
  if (status != TaskStatus::complete) RWZAbort("auxiliary communication failed");
}
}  // namespace

namespace z4c {
WaveExtractRWZ::WaveExtractRWZ(MeshBlockPack *pack, ParameterInput *pin) try
    : pack_(pack), device_error_("rwz_error",1),
      host_error_("rwz_host_error",1) {
  Kokkos::deep_copy(device_error_,0);
  auto *pm = pack_->pmesh;
  if (pack_->pz4c == nullptr || pack_->padm == nullptr || !pm->three_d) {
    local_error_ = invalid_configuration;
  }
  CheckPhase("Cartesian Z4c setup");
  const auto ind = pm->mb_indcs;
  const auto &opt = pack_->pz4c->opt;
  const std::string block = "rwz_extraction";
  const int nr = pin->GetOrAddInteger(block,"num_radii",0);
  lmax_ = pin->GetOrAddInteger(block,"lmax",2);
  ntheta_ = pin->GetOrAddInteger(block,"ntheta",32);
  nphi_ = pin->GetOrAddInteger(block,"nphi",64);
  mpi_root_ = pin->GetOrAddInteger(block,"mpi_root",0);
  output_digits_ = pin->GetOrAddInteger(block,"output_digits",16);
  bitant_ = pin->GetOrAddBoolean("mesh","bitant",false);
  subtract_background_ = pin->GetOrAddBoolean(block,"subtract_background",false);
  extra_output_ = pin->GetOrAddBoolean(block,"extra_output",false);
  verbose_ = pin->GetOrAddBoolean(block,"verbose",false);
  const std::string method = pin->GetOrAddString(block,"method_integrals","riemann");
  gauss_legendre_ = method == "gausslegendre";
  const std::string radius = pin->GetOrAddString(block,"method_areal_radius","areal");
  if (radius == "areal") radius_method_ = RadiusMethod::areal;
  else if (radius == "areal_simple") radius_method_ = RadiusMethod::areal_simple;
  else if (radius == "average_schw") radius_method_ = RadiusMethod::average_schw;
  else if (radius == "schw_gthth") radius_method_ = RadiusMethod::schw_gthth;
  else if (radius == "schw_gphph") radius_method_ = RadiusMethod::schw_gphph;
  else local_error_ = invalid_configuration;
  if (nr <= 0 || lmax_ < 2 || lmax_ > 8 || ntheta_ <= 0 || nphi_ <= 0 ||
      nphi_%2 != 0 || mpi_root_ < 0 || mpi_root_ >= global_variable::nranks ||
      output_digits_ <= 0 || (!gauss_legendre_ && method != "riemann" &&
      method != "midpoint") || ind.ng < 2 || ind.ng > 4 ||
      (pm->multilevel && ind.ng%2 != 0) || !std::isfinite(opt.chi_psi_power) ||
      opt.chi_psi_power == 0 || !std::isfinite(4./opt.chi_psi_power)) {
    local_error_ = invalid_configuration;
  }
  // Use a valid diagnostic root even while reporting a bad root option.
  if (mpi_root_ < 0 || mpi_root_ >= global_variable::nranks) mpi_root_ = 0;
  CheckPhase("configuration");
  if (ntheta_ < lmax_+1 || nphi_ < 2*lmax_+1 ||
      (gauss_legendre_ && static_cast<long long>(nphi_) != 2LL*ntheta_)) {
    local_error_ = invalid_configuration;
  }
  if (bitant_ && (pm->mesh_size.x3min != 0 ||
                 pm->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::reflect)) {
    local_error_ = invalid_configuration;
  }
  const int nx[3] = {ind.nx1,ind.nx2,ind.nx3};
  const int cnx[3] = {ind.cnx1,ind.cnx2,ind.cnx3};
  for (int d = 0; d < 3; ++d) {
    if (nx[d] <= 1 || nx[d] > std::numeric_limits<int>::max()-2*ind.ng ||
        (pm->multilevel && (nx[d] < 2*ind.ng || nx[d]%2 != 0))) {
      local_error_ = invalid_configuration;
    }
    for (int side = 0; side < 2; ++side) {
      const auto bc = pm->mesh_bcs[2*d+side];
      const bool extrapolate = bc == BoundaryFlag::outflow ||
                               bc == BoundaryFlag::diode || bc == BoundaryFlag::vacuum;
      if (bc != BoundaryFlag::periodic && bc != BoundaryFlag::reflect && !extrapolate) {
        local_error_ = invalid_configuration;
      }
      if (extrapolate && (pm->multilevel ? cnx[d] : nx[d]) < opt.extrap_order) {
        local_error_ = invalid_configuration;
      }
    }
  }
  if (opt.extrap_order < 2 || opt.extrap_order > 4) local_error_ = invalid_configuration;
  nmode_ = (lmax_+1)*(lmax_+1)-4;
  const std::size_t imax = std::numeric_limits<int>::max();
  const std::size_t real_max = std::numeric_limits<std::size_t>::max()/sizeof(Real);
  const std::size_t nt = ntheta_, np = nphi_, ns = nr, nm = nmode_;
  // The two-dimensional harmonic launch is flattened into an int by par_for.
  if (!ProductFits({nt,np},imax) || !ProductFits({nt,np,nm},imax) ||
      !ProductFits({ns,nt,np},imax) ||
      !ProductFits({ns,NBackground},imax) || !ProductFits({ns,NMultipoles,nm,2},imax) ||
      !ProductFits({nt,np,nm,2},real_max)) local_error_ = invalid_configuration;
  CheckPhase("mesh, boundary and storage eligibility");
  npoint_ = ntheta_*nphi_;
  shells_.resize(nr);
  for (int n = 0; n < nr; ++n) {
    auto &s = shells_[n];
    const auto suffix = std::to_string(n);
    s.R = pin->GetOrAddReal(block,"radius_"+suffix,10.0);
    s.center[0] = pin->GetOrAddReal(block,"center_x_"+suffix,0.0);
    s.center[1] = pin->GetOrAddReal(block,"center_y_"+suffix,0.0);
    s.center[2] = pin->GetOrAddReal(block,"center_z_"+suffix,0.0);
    if (!std::isfinite(s.R) || s.R <= 0) local_error_ = invalid_configuration;
    for (int d = 0; d < 3; ++d) {
      if (!std::isfinite(s.center[d])) local_error_ = invalid_configuration;
    }
  }
  CheckPhase("sphere configuration");
  InitializeOutput(pin);
  CheckPhase("output paths");
  const int capacity = std::max(pack_->nmb_thispack,pm->nmb_maxperrank);
  const std::size_t n1 = nx[0]+2*ind.ng, n2 = nx[1]+2*ind.ng, n3 = nx[2]+2*ind.ng;
  if (capacity < 0 || !ProductFits({static_cast<std::size_t>(capacity),n3,n2,n1},imax) ||
      !ProductFits({static_cast<std::size_t>(capacity),34,n3,n2,n1},real_max)) {
    local_error_ = invalid_configuration;
  }
  // RestrictCC flattens active blocks, 34 fields, and coarse cells into int.
  if (pm->multilevel && !ProductFits({static_cast<std::size_t>(pack_->nmb_thispack),34,
      static_cast<std::size_t>(cnx[2]),static_cast<std::size_t>(cnx[1]),
      static_cast<std::size_t>(cnx[0])},imax)) local_error_ = invalid_configuration;
  CheckPhase("auxiliary storage");
  auxiliary_ = DvceArray5D<Real>("rwz_aux",capacity,34,n3,n2,n1);
  if (pm->multilevel) {
    coarse_auxiliary_ = DvceArray5D<Real>("rwz_coarse_aux",capacity,34,
        cnx[2]+2*ind.ng,cnx[1]+2*ind.ng,cnx[0]+2*ind.ng);
  }
  background_ = DvceArray2D<Real>("rwz_background",nr,NBackground);
  multipoles_ = DvceArray4D<Real>("rwz_multipoles",nr,NMultipoles,nmode_,2);
  host_background_ = HostArray2D<Real>("rwz_host_background",nr,NBackground);
  host_multipoles_ = HostArray4D<Real>("rwz_host_multipoles",nr,NMultipoles,nmode_,2);
  boundary_ = std::make_unique<MeshBoundaryValuesCC>(pack_,pin,true);
  boundary_->InitializeBuffers(34, /*allocate_flux=*/false);
  InitializeTables();
  CheckPhase("angular geometry and harmonics");
} catch (const std::exception &e) {
  RWZAbort(e.what());
} catch (...) {
  RWZAbort("unexpected setup failure");
}

WaveExtractRWZ::~WaveExtractRWZ() {
  // Every Load is synchronous. Destruction does not add a diagnostic collective.
  for (auto &s : shells_) {
    for (auto *file : s.files) if (file != nullptr) std::fclose(file);
  }
}

void WaveExtractRWZ::CheckPhase(const char *phase) {
  Kokkos::fence("RWZ phase completion");
  Kokkos::deep_copy(host_error_,device_error_);
  int error = std::max(local_error_,host_error_(0));
#if MPI_PARALLEL_ENABLED
  if (MPI_Allreduce(MPI_IN_PLACE,&error,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD) != MPI_SUCCESS) {
    RWZAbort("MPI error during diagnostic failure synchronization",mpi_root_);
  }
#endif
  if (error != 0) {
    const char *names[] = {"success", "invalid_configuration", "invalid_ownership",
        "invalid_conformal_state", "invalid_auxiliary", "invalid_sample", "invalid_area",
        "invalid_background", "singular_radial_map", "invalid_two_metric",
        "invalid_projection", "singular_master_coefficient", "output_io_error"};
    std::ostringstream message;
    message << phase << ": " << (error > 0 && error <= output_io_error
                               ? names[error] : "unknown diagnostic error");
    RWZAbort(message.str().c_str(),mpi_root_);
  }
  local_error_ = 0;
  Kokkos::deep_copy(device_error_,0);
}

void WaveExtractRWZ::Load(Real sample_time, int sample_cycle) try {
  auto *pm = pack_->pmesh;
  const auto ind = pm->mb_indcs;
  const int ng = ind.ng, nmb = pack_->nmb_thispack;
  if (!std::isfinite(sample_time) || sample_cycle < 0 ||
      nmb > auxiliary_.extent_int(0)) local_error_ = invalid_configuration;
  // Recheck the active restriction launch after AMR or load balancing, using the
  // existing admission gate rather than adding a collective before restriction.
  if (pm->multilevel && !ProductFits({static_cast<std::size_t>(nmb),34,
      static_cast<std::size_t>(ind.cnx3),static_cast<std::size_t>(ind.cnx2),
      static_cast<std::size_t>(ind.cnx1)},std::numeric_limits<int>::max())) {
    local_error_ = invalid_configuration;
  }
  CheckPhase("sample identity and current pack");
  if (ownership_epoch_ != pm->GetAMRLoadBalanceUpdateSeq()) RefreshOwnership();
  auto u0 = pack_->pz4c->u0;
  auto error = device_error_;
  const Real exponent = 4./pack_->pz4c->opt.chi_psi_power;
  // Validate the exact full-ghost conversion/stencil range before taking powers.
  // All evolved Z4c lanes are consumed by ADM conversion or gauge stencils.
  par_for("RWZ raw state admission",DevExeSpace(),0,nmb-1,
      ind.ks-ng,ind.ke+ng,ind.js-ng,ind.je+ng,ind.is-ng,ind.ie+ng,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
    for (int n = 0; n < Z4c::nz4c; ++n) {
      if (!Kokkos::isfinite(u0(m,n,k,j,i))) {
        Kokkos::atomic_max(&error(0),int(invalid_auxiliary));
        return;
      }
    }
    const Real chi = u0(m,Z4c::I_Z4C_CHI,k,j,i);
    if (!(chi > 0)) {
      Kokkos::atomic_max(&error(0),int(invalid_conformal_state));
      return;
    }
    const Real psi4 = Kokkos::pow(chi,exponent);
    if (!Kokkos::isfinite(psi4) || !(psi4 > 0)) {
      Kokkos::atomic_max(&error(0),int(invalid_conformal_state));
    }
  });
  CheckPhase("raw ghosted state before ADM conversion");
  pack_->pz4c->Z4cToADM(pack_);
  Kokkos::fence("RWZ physical ADM refresh");
  PrepareAuxiliary(sample_time);
  CheckPhase("auxiliary producer");
  ExchangeAuxiliary();
  CheckPhase("auxiliary halo");
  SampleAndAccumulateBackground();
  CheckPhase("owned sphere and local background");
  Kokkos::deep_copy(host_background_,background_);
#if MPI_PARALLEL_ENABLED
  if (MPI_Allreduce(MPI_IN_PLACE,host_background_.data(),int(host_background_.size()),
                    MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD) != MPI_SUCCESS) {
    RWZAbort("MPI background integral reduction failed",mpi_root_);
  }
#endif
  for (std::size_t n = 0; n < host_background_.size(); ++n) {
    if (!std::isfinite(host_background_.data()[n])) local_error_ = invalid_background;
  }
  CheckPhase("reduced background");
  FinalizeBackgroundAndTransform();
  CheckPhase("radial map and transformed samples");
  AccumulateMultipoles();
  CheckPhase("local multipoles");
  Kokkos::deep_copy(host_multipoles_,multipoles_);
#if MPI_PARALLEL_ENABLED
  if (MPI_Allreduce(MPI_IN_PLACE,host_multipoles_.data(),int(host_multipoles_.size()),
                    MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD) != MPI_SUCCESS) {
    RWZAbort("MPI multipole integral reduction failed",mpi_root_);
  }
#endif
  for (std::size_t n = 0; n < host_multipoles_.size(); ++n) {
    if (!std::isfinite(host_multipoles_.data()[n])) local_error_ = invalid_projection;
  }
  CheckPhase("reduced multipoles");
  FinalizeMultipoles();
  CheckPhase("gauge invariants and master functions");
  if (verbose_ && global_variable::my_rank == mpi_root_) {
    std::cout << "RWZ extracted cycle=" << sample_cycle << " time=" << sample_time
              << " on " << shells_.size() << " spheres" << std::endl;
  }
} catch (const std::exception &e) {
  RWZAbort(e.what());
} catch (...) {
  RWZAbort("unexpected extraction failure");
}

void WaveExtractRWZ::InitializeTables() {
  HostArray1D<Real> htheta("rwz_theta_nodes",ntheta_);
  HostArray1D<Real> hweight("rwz_theta_weights",ntheta_);
  if (gauss_legendre_) {
    const auto roots = RootsAndWeights(ntheta_);
    for (int i = 0; i < ntheta_; ++i) {
      if (!std::isfinite(roots[0][i]) || roots[0][i] <= -1 || roots[0][i] >= 1 ||
          !std::isfinite(roots[1][i]) || roots[1][i] <= 0) {
        local_error_ = invalid_configuration;
        return;
      }
      htheta(i) = std::acos(roots[0][i]);
      hweight(i) = roots[1][i];
    }
  } else {
    for (int i = 0; i < ntheta_; ++i) {
      htheta(i) = (i+0.5)*M_PI/ntheta_;
      hweight(i) = std::sin(htheta(i))*M_PI/ntheta_;
    }
  }
  DvceArray1D<Real> theta_nodes("rwz_theta_nodes_device",ntheta_);
  DvceArray1D<Real> theta_weights("rwz_theta_weights_device",ntheta_);
  Kokkos::deep_copy(theta_nodes,htheta);
  Kokkos::deep_copy(theta_weights,hweight);
  const int npoint = npoint_, nmode = nmode_, nphi = nphi_;
  const int ng = pack_->pmesh->mb_indcs.ng;
  const bool bitant = bitant_;
  auto error = device_error_;
  // Angles, quadrature and basis depend only on angular resolution and lmax,
  // not on radius, center or mesh topology. Initialize one manager-owned copy.
  angular_.theta = DvceArray1D<Real>("rwz_theta",npoint);
  angular_.phi = DvceArray1D<Real>("rwz_phi",npoint);
  angular_.weights = DvceArray1D<Real>("rwz_weights",npoint);
  for (int h = 0; h < NHarmonic; ++h) {
    angular_.harmonics[h] = DvceArray3D<Real>(
        "rwz_harmonic_"+std::to_string(h),npoint,nmode,2);
  }
  {
    auto theta = angular_.theta, phi = angular_.phi, weight = angular_.weights;
    par_for("RWZ fixed angles",DevExeSpace(),0,npoint-1,KOKKOS_LAMBDA(int q) {
      const Real th = theta_nodes(q/nphi), ph = (q%nphi+0.5)*2*M_PI/nphi;
      theta(q) = th; phi(q) = ph;
      weight(q) = theta_weights(q/nphi)*2*M_PI/nphi;
      if (!Kokkos::isfinite(th) || !Kokkos::isfinite(ph) ||
          !(Kokkos::sin(th) > 0) || !Kokkos::isfinite(weight(q)) || !(weight(q) > 0)) {
        Kokkos::atomic_max(&error(0),int(invalid_configuration));
      }
    });
    auto hy = angular_.harmonics[Y], ht = angular_.harmonics[Ytheta];
    auto hp = angular_.harmonics[Yphi];
    auto hw = angular_.harmonics[W], hx = angular_.harmonics[X];
    par_for("RWZ tensor harmonic table",DevExeSpace(),0,npoint-1,0,nmode-1,
        KOKKOS_LAMBDA(int q, int lm) {
      const int l = static_cast<int>(Kokkos::sqrt(Real(lm+4)));
      const int m = lm+4-l*l-l;
      const Real th = theta(q), ph = phi(q), sn = Kokkos::sin(th);
      if (!(sn > 0)) return;  // angular failure is propagated before first Load
      const Real cot = Kokkos::cos(th)/sn;
      Real yr, yi, tr, ti, pr, pi;
      SphericalHarmFirstDerivs(&yr,&yi,&tr,&ti,&pr,&pi,l,m,th,ph);
      hy(q,lm,0) = yr; hy(q,lm,1) = yi;
      ht(q,lm,0) = tr; ht(q,lm,1) = ti;
      hp(q,lm,0) = pr; hp(q,lm,1) = pi;
      const Real coef = 2*m*m/(sn*sn)-l*(l+1);
      hw(q,lm,0) = -2*cot*tr+coef*yr;
      hw(q,lm,1) = -2*cot*ti+coef*yi;
      hx(q,lm,0) = -2*m*(ti-cot*yi);
      hx(q,lm,1) =  2*m*(tr-cot*yr);
      for (int c = 0; c < 2; ++c) {
        if (!Kokkos::isfinite(hy(q,lm,c)) || !Kokkos::isfinite(ht(q,lm,c)) ||
            !Kokkos::isfinite(hp(q,lm,c)) || !Kokkos::isfinite(hw(q,lm,c)) ||
            !Kokkos::isfinite(hx(q,lm,c))) {
          Kokkos::atomic_max(&error(0),int(invalid_configuration));
        }
      }
    });
  }
  for (std::size_t ir = 0; ir < shells_.size(); ++ir) {
    auto &s = shells_[ir];
    const std::string tag = "rwz_"+std::to_string(ir)+"_";
    s.theta = angular_.theta;
    s.phi = angular_.phi;
    s.weights = angular_.weights;
    s.x_lookup = DvceArray2D<Real>(tag+"lookup",npoint,3);
    s.normal = DvceArray2D<Real>(tag+"normal",npoint,3);
    s.reflected = DvceArray1D<int>(tag+"reflected",npoint);
    s.owner = DvceArray1D<int>(tag+"owner",npoint);
    Kokkos::deep_copy(s.owner,-1);
    s.jacobian = DvceArray3D<Real>(tag+"jacobian",npoint,3,3);
    s.inverse_jacobian = DvceArray3D<Real>(tag+"inverse_jacobian",npoint,3,3);
    for (int h = 0; h < NHarmonic; ++h) s.harmonics[h] = angular_.harmonics[h];
    if (ng == 2) s.interp2 = DvceArray1D<IndAndWghts<2>>(tag+"interpolation",npoint);
    if (ng == 3) s.interp3 = DvceArray1D<IndAndWghts<3>>(tag+"interpolation",npoint);
    if (ng == 4) s.interp4 = DvceArray1D<IndAndWghts<4>>(tag+"interpolation",npoint);
    s.gamma = DvceArray3D<Real>(tag+"gamma",npoint,NDerivative,6);
    s.beta_d = DvceArray3D<Real>(tag+"beta_d",npoint,NDerivative,3);
    s.alpha = DvceArray3D<Real>(tag+"alpha",npoint,NDerivative,1);
    s.beta2 = DvceArray3D<Real>(tag+"beta2",npoint,NDerivative,1);
    // Final host modes are used only by the selected writer rank.
    if (global_variable::my_rank == mpi_root_) {
      auto mode = [&](ModeField &f, const char *name) {
        f.data = HostArray3D<Real>(tag+name,NDerivative,nmode,2);
      };
      mode(s.h00,"h00"); mode(s.h01,"h01"); mode(s.h11,"h11");
      mode(s.h0,"h0"); mode(s.h1,"h1"); mode(s.G,"G"); mode(s.K,"K");
      mode(s.H0,"H0"); mode(s.H1,"H1"); mode(s.H,"H");
      s.kappa_dd.data = HostArray3D<Real>(tag+"kappa_dd",3,nmode,2);
      s.kappa_d = HostArray3D<Real>(tag+"kappa_d",2,nmode,2);
      auto final = [&](HostArray2D<Real> &f, const char *name) {
        f = HostArray2D<Real>(tag+name,nmode,2);
      };
      final(s.kappa,"kappa"); final(s.Tr_kappa_dd,"Tr_kappa_dd");
      final(s.Psie,"Psie"); final(s.Psio,"Psio");
      final(s.Psie_dyn,"Psie_dyn"); final(s.Psio_dyn,"Psio_dyn");
      final(s.Psie_sch,"Psie_sch"); final(s.Psio_sch,"Psio_sch");
      final(s.Qplus,"Qplus"); final(s.Qstar,"Qstar");
      final(s.Psie_dr,"Psie_dr"); final(s.Psio_dr,"Psio_dr");
      final(s.Qplus_dr,"Qplus_dr"); final(s.Qstar_dr,"Qstar_dr");
    }
    const Real R = s.R, cx = s.center[0], cy = s.center[1], cz = s.center[2];
    auto theta = s.theta, phi = s.phi, weight = s.weights;
    auto lookup = s.x_lookup, normal = s.normal;
    auto reflected = s.reflected;
    auto jac = s.jacobian, invjac = s.inverse_jacobian;
    par_for("RWZ fixed geometry",DevExeSpace(),0,npoint-1,KOKKOS_LAMBDA(int q) {
      const Real th = theta(q), ph = phi(q);
      const Real sn = Kokkos::sin(th), cs = Kokkos::cos(th);
      const Real cp = Kokkos::cos(ph), sp = Kokkos::sin(ph);
      const Real Rsin = R*sn;
      if (!(sn > 0) || !(Rsin > 0) || !Kokkos::isfinite(Rsin)) {
        Kokkos::atomic_max(&error(0),int(invalid_configuration));
        return;
      }
      const Real n[3] = {sn*cp,sn*sp,cs};
      const Real et[3] = {cs*cp,cs*sp,-sn}, ep[3] = {-sp,cp,0};
      const Real center[3] = {cx,cy,cz};
      // Keep the unreflected point only through setup admission and bitant parity.
      Real logical[3];
      for (int a = 0; a < 3; ++a) {
        logical[a] = center[a]+R*n[a];
        lookup(q,a) = (bitant && a == 2) ? Kokkos::fabs(logical[a]) : logical[a];
        normal(q,a) = n[a];
        jac(q,a,0) = n[a]; jac(q,a,1) = R*et[a]; jac(q,a,2) = Rsin*ep[a];
        invjac(q,0,a) = n[a]; invjac(q,1,a) = et[a]/R;
        invjac(q,2,a) = ep[a]/Rsin;
        if (!Kokkos::isfinite(logical[a]) || !Kokkos::isfinite(invjac(q,1,a)) ||
            !Kokkos::isfinite(invjac(q,2,a))) {
          Kokkos::atomic_max(&error(0),int(invalid_configuration));
        }
      }
      reflected(q) = bitant && logical[2] < 0;
      if (!Kokkos::isfinite(weight(q)) || !(weight(q) > 0)) {
        Kokkos::atomic_max(&error(0),int(invalid_configuration));
      }
    });
  }
  // The temporary angular-node Views must outlive both geometry and basis kernels.
  Kokkos::fence("RWZ angular table ownership boundary");
}

void WaveExtractRWZ::RefreshOwnership() {
  switch (pack_->pmesh->mb_indcs.ng) {
    case 2: RefreshOwnershipImpl<2>(); break;
    case 3: RefreshOwnershipImpl<3>(); break;
    case 4: RefreshOwnershipImpl<4>(); break;
    default: local_error_ = invalid_configuration;
  }
  CheckPhase("sphere ownership and interpolation support");
  ownership_epoch_ = pack_->pmesh->GetAMRLoadBalanceUpdateSeq();
}

template <int NGHOST>
void WaveExtractRWZ::RefreshOwnershipImpl() {
  const auto ind = pack_->pmesh->mb_indcs;
  auto size = pack_->pmb->mb_size;
  const int nmb = pack_->nmb_thispack, np = npoint_;
  DvceArray2D<int> counts("rwz_local_owner_counts",shells_.size(),np);
  auto error = device_error_;
  for (std::size_t ir = 0; ir < shells_.size(); ++ir) {
    auto &s = shells_[ir];
    auto lookup = s.x_lookup;
    auto owner = s.owner;
    auto interpolation = [&]() {
      if constexpr (NGHOST == 2) return s.interp2;
      if constexpr (NGHOST == 3) return s.interp3;
      if constexpr (NGHOST == 4) return s.interp4;
    }();
    par_for("RWZ refresh ownership",DevExeSpace(),0,np-1,KOKKOS_LAMBDA(int q) {
      Real x[3] = {lookup(q,0),lookup(q,1),lookup(q,2)};
      int count = 0;
      for (int m = 0; m < nmb; ++m) {
        const auto b = size.d_view(m);
        if (x[0] >= b.x1min && x[0] < b.x1max &&
            x[1] >= b.x2min && x[1] < b.x2max &&
            x[2] >= b.x3min && x[2] < b.x3max) {
          // IndicesAndWeights divides by spacing and casts before returning.
          // Reject a bad candidate before entering that shared helper.
          if (!Kokkos::isfinite(b.dx1) || !(b.dx1 > 0) ||
              !Kokkos::isfinite(b.dx2) || !(b.dx2 > 0) ||
              !Kokkos::isfinite(b.dx3) || !(b.dx3 > 0)) {
            Kokkos::atomic_max(&error(0),int(invalid_ownership));
            counts(ir,q) = 0;
            owner(q) = -1;
            return;
          }
          ++count;
        }
      }
      counts(ir,q) = count;
      const auto iw = IndicesAndWeights<NGHOST>(ind,size,x,nmb);
      interpolation(q) = iw;
      owner(q) = iw.ii0;
      if (!iw.point_exist) return;
      const int start[3] = {iw.ii1-NGHOST+ind.is+1,iw.ii2-NGHOST+ind.js+1,
                            iw.ii3-NGHOST+ind.ks+1};
      const int n[3] = {ind.nx1+2*ind.ng,ind.nx2+2*ind.ng,ind.nx3+2*ind.ng};
      bool valid = count == 1 && iw.ii0 >= 0 && iw.ii0 < nmb;
      for (int d = 0; d < 3; ++d) {
        valid = valid && start[d] >= 0 && start[d]+2*NGHOST <= n[d];
      }
      for (int t = 0; t < 2*NGHOST; ++t) {
        valid = valid && Kokkos::isfinite(iw.wght1[t]) && Kokkos::isfinite(iw.wght2[t]) &&
            Kokkos::isfinite(iw.wght3[t]) && Kokkos::isfinite(iw.dwght1[t]) &&
            Kokkos::isfinite(iw.dwght2[t]) && Kokkos::isfinite(iw.dwght3[t]);
      }
      if (!valid) Kokkos::atomic_max(&error(0),int(invalid_ownership));
    });
  }
  Kokkos::fence("RWZ ownership before MPI");
  HostArray2D<int> global_counts("rwz_global_owner_counts",shells_.size(),np);
  Kokkos::deep_copy(global_counts,counts);
#if MPI_PARALLEL_ENABLED
  if (MPI_Allreduce(MPI_IN_PLACE,global_counts.data(),int(global_counts.size()),
                    MPI_INT,MPI_SUM,MPI_COMM_WORLD) != MPI_SUCCESS) {
    RWZAbort("MPI owner-count reduction failed",mpi_root_);
  }
#endif
  for (std::size_t n = 0; n < global_counts.size(); ++n) {
    if (global_counts.data()[n] != 1) local_error_ = invalid_ownership;
  }
}

void WaveExtractRWZ::PrepareAuxiliary(Real sample_time) {
  const auto &o = pack_->pz4c->opt;
  const auto *pdc = pack_->pz4c->pdrift_control.get();
  auto finite = [](std::initializer_list<Real> values) {
    for (Real v : values) if (!std::isfinite(v)) return false;
    return true;
  };
  bool valid = finite({sample_time, o.chi_psi_power, o.chi_div_floor,
      o.lapse_oplog, o.lapse_harmonicf, o.lapse_harmonic, o.lapse_advect,
      o.shift_ggamma, o.shift_alpha2ggamma, o.shift_hh, o.shift_advect,
      o.shift_eta, pack_->pz4c->diss});
  if (o.slow_start_lapse) {
    valid = valid && finite({o.chi_min_floor,o.ssl_damping_amp,
                            o.ssl_damping_time,o.ssl_damping_index}) &&
            o.ssl_damping_time != 0;
    if (valid) {
      const Real ratio = sample_time/o.ssl_damping_time;
      valid = std::isfinite(ratio) && std::isfinite(std::pow(ratio,2));
    }
  }
  // Validate only the controller branches actually consumed by the snapshot.
  // These admission checks are diagnostic-only, not new evolution restrictions.
  if (o.enable_driftcontrol) {
    valid = valid && pdc != nullptr && finite({o.dc_damping_scale,
        o.dc_fixed_x,o.dc_fixed_y,o.dc_fixed_z});
    const Real scale2 = SQR(o.dc_damping_scale);
    valid = valid && std::isfinite(scale2) && scale2 != 0;
    if (pdc != nullptr) {
      const Real fixed[3] = {o.dc_fixed_x,o.dc_fixed_y,o.dc_fixed_z};
      for (int a = 0; a < 3; ++a) {
        if (o.dc_gaussian_center == DriftControl::CenterTracker ||
            o.dc_variety != DriftControl::BDOB) valid = valid && finite({pdc->GetPos(a)});
        if (o.dc_variety == DriftControl::Relaxation) {
          valid = valid && finite({o.dc_relaxation_time,o.dc_kappa,o.dc_gamma_suppress}) &&
                  o.dc_relaxation_time != 0;
          if (valid && o.dc_gamma_suppress > 0) {
            const Real denominator = 1.0+o.dc_gamma_suppress*
                std::fabs(pdc->GetPos(a)-fixed[a])/o.dc_damping_scale;
            valid = std::isfinite(denominator) && denominator != 0;
          }
        } else if (o.dc_variety == DriftControl::PID) {
          valid = valid && finite({o.dc_Kp,o.dc_Ki,o.dc_Kd,
                                   pdc->GetIntegral(a),pdc->GetVel(a)});
        } else if (o.dc_variety == DriftControl::BDOB) {
          valid = valid && finite({pdc->GetU(a)});
        } else if (o.dc_variety == DriftControl::DOB) {
          valid = valid && finite({o.dc_omega_c,o.dc_omega_o,o.dc_zeta,
                                   pdc->GetP(a),pdc->GetVel(a)});
        } else {
          const Real tau2 = SQR(o.dc_damping_time);
          valid = valid && finite({o.dc_damping_time,o.dc_damping_coeff,
                                   pdc->GetVel(a),tau2}) && tau2 != 0;
        }
      }
      if (o.dc_variety != DriftControl::BDOB) {
        const Real ramp = pdc->RampFactor(sample_time);
        valid = valid && finite({o.dc_gain_x,o.dc_gain_y,o.dc_gain_z,ramp}) &&
                ramp >= 0 && ramp <= 1;
      }
    }
  }
  if (!valid) { local_error_ = invalid_auxiliary; return; }
  switch (pack_->pmesh->mb_indcs.ng) {
    case 2: PrepareAuxiliaryImpl<2>(sample_time); break;
    case 3: PrepareAuxiliaryImpl<3>(sample_time); break;
    case 4: PrepareAuxiliaryImpl<4>(sample_time); break;
    default: local_error_ = invalid_configuration;
  }
}

template <int NGHOST>
void WaveExtractRWZ::PrepareAuxiliaryImpl(Real sample_time) {
  const auto ind = pack_->pmesh->mb_indcs;
  auto size = pack_->pmb->mb_size;
  auto z = pack_->pz4c->z4c;
  auto u0 = pack_->pz4c->u0;
  auto physical = pack_->padm->adm;
  auto aux = auxiliary_;
  auto error = device_error_;
  const int nmb = pack_->nmb_thispack;
  const GaugeRhsContext ctx = MakeGaugeRhsContext(
      pack_->pz4c->opt,pack_->pz4c->pdrift_control.get(),sample_time);
  for (int a = 0; a < 3; ++a) {
    if (!std::isfinite(ctx.dc_inv_tau[a]) || !std::isfinite(ctx.dc_c[a]) ||
        !std::isfinite(ctx.dc_corr[a]) || !std::isfinite(ctx.dc_target[a]) ||
        !std::isfinite(ctx.dc_gsupp[a])) local_error_ = invalid_auxiliary;
  }
  if (!std::isfinite(ctx.dc_inv_s2)) local_error_ = invalid_auxiliary;
  if (local_error_ != 0) return;
  const Real ko_diss = pack_->pz4c->diss;
  par_for("RWZ auxiliary physical",DevExeSpace(),0,nmb-1,
      ind.ks,ind.ke,ind.js,ind.je,ind.is,ind.ie,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const auto b = size.d_view(m);
    if (!Kokkos::isfinite(b.dx1) || !(b.dx1 > 0) ||
        !Kokkos::isfinite(b.dx2) || !(b.dx2 > 0) ||
        !Kokkos::isfinite(b.dx3) || !(b.dx3 > 0)) {
      Kokkos::atomic_max(&error(0),int(invalid_auxiliary));
      return;
    }
    const Real idx[3] = {1/b.dx1,1/b.dx2,1/b.dx3};
    for (int c = 0; c < 3; ++c) {
      if (!Kokkos::isfinite(idx[c])) {
        Kokkos::atomic_max(&error(0),int(invalid_auxiliary));
        return;
      }
    }
    GaugeRhsPoint u{};
    GaugeRhsDerivatives d{};
    u.chi = z.chi(m,k,j,i);
    u.chi_guarded = u.chi > ctx.opt.chi_div_floor ? u.chi : ctx.opt.chi_div_floor;
    u.alpha = z.alpha(m,k,j,i);
    u.khat = z.vKhat(m,k,j,i);
    for (int a = 0; a < 3; ++a) {
      u.beta[a] = z.beta_u(m,a,k,j,i);
      u.gam[a] = z.vGam_u(m,a,k,j,i);
    }
    for (int c = 0; c < 3; ++c) {
      d.dalpha[c] = Dx<NGHOST>(c,idx,z.alpha,m,k,j,i);
      d.dchi[c] = Dx<NGHOST>(c,idx,z.chi,m,k,j,i);
      aux(m,27+c,k,j,i) = d.dalpha[c];
      d.Lalpha += Lx<NGHOST>(c,idx,z.beta_u,z.alpha,m,c,k,j,i);
      for (int a = 0; a < 3; ++a) {
        aux(m,18+3*c+a,k,j,i) = Dx<NGHOST>(c,idx,z.beta_u,m,a,k,j,i);
        d.Lbeta[a] += Lx<NGHOST>(c,idx,z.beta_u,z.beta_u,m,c,a,k,j,i);
      }
      if (!Kokkos::isfinite(d.dalpha[c]) || !Kokkos::isfinite(d.dchi[c])) {
        Kokkos::atomic_max(&error(0),int(invalid_auxiliary));
        return;
      }
    }
    const int sa[6] = {0,0,0,1,1,2}, sb[6] = {0,1,2,1,2,2};
    const Real psi4 = physical.psi4(m,k,j,i);
    for (int c = 0; c < 3; ++c) {
      for (int s = 0; s < 6; ++s) {
        const int a = sa[s], b = sb[s];
        aux(m,6*c+s,k,j,i) = psi4*(Dx<NGHOST>(c,idx,z.g_dd,m,a,b,k,j,i)
            +(4./ctx.opt.chi_psi_power)*z.g_dd(m,a,b,k,j,i)*d.dchi[c]/u.chi);
      }
    }
    const Real gxx = z.g_dd(m,0,0,k,j,i), gxy = z.g_dd(m,0,1,k,j,i);
    const Real gxz = z.g_dd(m,0,2,k,j,i), gyy = z.g_dd(m,1,1,k,j,i);
    const Real gyz = z.g_dd(m,1,2,k,j,i), gzz = z.g_dd(m,2,2,k,j,i);
    const Real minor2 = gxx*gyy-gxy*gxy;
    const Real det = adm::SpatialDet(gxx,gxy,gxz,gyy,gyz,gzz);
    if (!(gxx > 0) || !Kokkos::isfinite(minor2) || !(minor2 > 0) ||
        !Kokkos::isfinite(det) || !(det > 0)) {
      Kokkos::atomic_max(&error(0),int(invalid_auxiliary));
      return;
    }
    const Real invdet = 1./det;
    if (!Kokkos::isfinite(invdet)) {
      Kokkos::atomic_max(&error(0),int(invalid_auxiliary));
      return;
    }
    GaugeInverseMetric g_uu;
    adm::SpatialInv(invdet,gxx,gxy,gxz,gyy,gyz,gzz,
        &g_uu(0,0),&g_uu(0,1),&g_uu(0,2),&g_uu(1,1),&g_uu(1,2),&g_uu(2,2));
    bool valid = Kokkos::isfinite(d.Lalpha);
    for (int a = 0; a < 3; ++a) {
      valid = valid && Kokkos::isfinite(d.Lbeta[a]);
      for (int b = a; b < 3; ++b) valid = valid && Kokkos::isfinite(g_uu(a,b));
    }
    for (int n = 0; n < 30; ++n) valid = valid && Kokkos::isfinite(aux(m,n,k,j,i));
    // Slow-start uses raw chi and its separate floor, not the physical psi4 law.
    if (ctx.opt.slow_start_lapse) {
      const Real W2 = u.chi > ctx.opt.chi_min_floor ? u.chi : ctx.opt.chi_min_floor;
      if (!Kokkos::isfinite(W2) || !(W2 > 0)) valid = false;
      if (valid) {
        const Real ww = Kokkos::pow(W2,0.5);
        if (!Kokkos::isfinite(ww) || !(ww > 0)) valid = false;
        if (valid) {
          const Real power = Kokkos::pow(ww,ctx.opt.ssl_damping_index);
          valid = Kokkos::isfinite(power) && Kokkos::isfinite(
              ctx.opt.ssl_damping_amp*(ww-u.alpha)*power*
              Kokkos::exp(-0.5*Kokkos::pow(sample_time/ctx.opt.ssl_damping_time,2)));
        }
      }
    }
    Real x[3] = {};
    if (ctx.dc_on) {
      x[0] = CellCenterX(i-ind.is,ind.nx1,b.x1min,b.x1max);
      x[1] = CellCenterX(j-ind.js,ind.nx2,b.x2min,b.x2max);
      x[2] = CellCenterX(k-ind.ks,ind.nx3,b.x3min,b.x3max);
      const Real r2 = SQR(x[0]-ctx.dc_c[0])+SQR(x[1]-ctx.dc_c[1])+SQR(x[2]-ctx.dc_c[2]);
      valid = valid && Kokkos::isfinite(r2) && Kokkos::isfinite(-r2*ctx.dc_inv_s2);
    }
    if (!valid) {
      Kokkos::atomic_max(&error(0),int(invalid_auxiliary));
      return;
    }
    GaugeRhsNoKO q = EvaluateGaugeRhs(u,d,g_uu,x,sample_time,ctx);
    // Stored diss already includes its parity sign and 2^(-2*ng) factor.
    // Evolution retains its separate, unchanged all-variable KO pass.
    for (int c = 0; c < 3; ++c) {
      q.alpha += Diss<NGHOST>(c,idx,u0,m,Z4c::I_Z4C_ALPHA,k,j,i)*ko_diss;
      for (int a = 0; a < 3; ++a) {
        q.beta[a] += Diss<NGHOST>(c,idx,u0,m,Z4c::I_Z4C_BETAX+a,k,j,i)*ko_diss;
      }
    }
    aux(m,30,k,j,i) = q.alpha;
    for (int a = 0; a < 3; ++a) aux(m,31+a,k,j,i) = q.beta[a];
    for (int n = 30; n < 34; ++n) {
      if (!Kokkos::isfinite(aux(m,n,k,j,i))) {
        Kokkos::atomic_max(&error(0),int(invalid_auxiliary));
      }
    }
  });
}

void WaveExtractRWZ::ExchangeAuxiliary() {
  auto *pm = pack_->pmesh;
  const int nmb = pack_->nmb_thispack;
  DvceArray5D<Real> fine = Kokkos::subview(auxiliary_,std::make_pair(0,nmb),
      Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
  DvceArray5D<Real> coarse;
  Kokkos::fence("RWZ physical auxiliaries before exchange");
  if (pm->multilevel) {
    coarse = Kokkos::subview(coarse_auxiliary_,std::make_pair(0,nmb),
        Kokkos::ALL,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    pm->pmr->RestrictCC(fine,coarse,true);
    Kokkos::fence("RWZ auxiliary restriction");
  }
  CompleteBoundary([&]() { return boundary_->InitRecv(34); });
  CompleteBoundary([&]() { return boundary_->PackAndSendCC(fine,coarse); });
  CompleteBoundary([&]() { return boundary_->RecvAndUnpackCC(fine,coarse); });
  CompleteBoundary([&]() { return boundary_->ClearSend(); });
  CompleteBoundary([&]() { return boundary_->ClearRecv(); });
  Kokkos::fence("RWZ auxiliary exchange completion");
  if (pm->multilevel) {
    AuxiliaryBCs(coarse,true);
    boundary_->ProlongateCC(fine,coarse,true);
    Kokkos::fence("RWZ auxiliary prolongation");
  }
  AuxiliaryBCs(fine,false);
}

void WaveExtractRWZ::AuxiliaryBCs(DvceArray5D<Real> fields, bool coarse) {
  if (pack_->pmesh->strictly_periodic) return;
  const auto ind = pack_->pmesh->mb_indcs;
  const int lo[3] = {coarse ? ind.cis : ind.is, coarse ? ind.cjs : ind.js,
                     coarse ? ind.cks : ind.ks};
  const int hi[3] = {coarse ? ind.cie : ind.ie, coarse ? ind.cje : ind.je,
                     coarse ? ind.cke : ind.ke};
  const int lengths[3] = {fields.extent_int(4),fields.extent_int(3),fields.extent_int(2)};
  const int nmb = pack_->nmb_thispack, ng = ind.ng, order = pack_->pz4c->opt.extrap_order;
  auto bcs = pack_->pmb->mb_bcs;
  // Axis ordering matches Z4c: later passes repair temporary corner values from
  // earlier axes. Use this bank's tensor indices, never evolved-variable indices.
  for (int axis = 0; axis < 3; ++axis) {
    const int t1 = axis == 0 ? 1 : 0, t2 = axis == 2 ? 1 : 2;
    par_for("RWZ auxiliary physical boundary",DevExeSpace(),0,nmb-1,
        0,lengths[t2]-1,0,lengths[t1]-1,
        KOKKOS_LAMBDA(int m, int v2, int v1) {
      for (int side = 0; side < 2; ++side) {
        const auto bc = bcs.d_view(m,2*axis+side);
        const bool reflect = bc == BoundaryFlag::reflect;
        if (!reflect && bc != BoundaryFlag::outflow && bc != BoundaryFlag::diode &&
            bc != BoundaryFlag::vacuum) continue;
        const int b = side == 0 ? lo[axis] : hi[axis], inward = side == 0 ? 1 : -1;
        for (int n = 0; n < 34; ++n) {
          int occurrences = 0;
          if (n < 18) {
            const int a[6] = {0,0,0,1,1,2}, bb[6] = {0,1,2,1,2,2};
            occurrences = (n/6 == axis)+(a[n%6] == axis)+(bb[n%6] == axis);
          } else if (n < 27) {
            occurrences = ((n-18)/3 == axis)+((n-18)%3 == axis);
          } else if (n < 30) occurrences = n-27 == axis;
          else if (n > 30) occurrences = n-31 == axis;
          const Real parity = occurrences%2 == 0 ? 1. : -1.;
          for (int delta = 1; delta <= ng; ++delta) {
            int dest[3] = {}, src[3] = {};
            dest[t1] = src[t1] = v1; dest[t2] = src[t2] = v2;
            dest[axis] = b-inward*delta;
            Real value;
            if (reflect) {
              src[axis] = b+inward*(delta-1);
              value = parity*fields(m,n,src[2],src[1],src[0]);
            } else {
              Real f[4] = {};
              for (int t = 0; t < order; ++t) {
                src[axis] = b+inward*t;
                f[t] = fields(m,n,src[2],src[1],src[0]);
              }
              if (order == 2) value = f[0]+delta*(f[0]-f[1]);
              else if (order == 3) value = 0.5*(f[0]*(1+delta)*(2+delta)
                  +delta*(f[2]+delta*f[2]-2*f[1]*(2+delta)));
              else value = (-3.0*f[1]*delta*(2+delta)*(3+delta)
                  +f[0]*(1+delta)*(2+delta)*(3+delta)
                  +delta*(1+delta)*(-f[3]*(2+delta)+3*f[2]*(3+delta)))/6.0;
            }
            fields(m,n,dest[2],dest[1],dest[0]) = value;
          }
        }
      }
    });
    Kokkos::fence("RWZ ordered auxiliary boundary axis");
  }
}

void WaveExtractRWZ::SampleAndAccumulateBackground() {
  // Integral atomics require a clear on the sampling kernels' default instance.
  // Successful sampling overwrites every consumed owned-point sphere slot;
  // consumers skip nonowners and unproduced scalar/vector D11 slots.
  const DevExeSpace exec;
  Kokkos::deep_copy(exec, background_, Real(0));
  switch (pack_->pmesh->mb_indcs.ng) {
    case 2: SampleAndAccumulateBackgroundImpl<2>(); break;
    case 3: SampleAndAccumulateBackgroundImpl<3>(); break;
    case 4: SampleAndAccumulateBackgroundImpl<4>(); break;
    default: local_error_ = invalid_configuration; break;
  }
}

template <int NGHOST>
void WaveExtractRWZ::SampleAndAccumulateBackgroundImpl() {
  // Rebind current physical ADM and evolved gauge state on every sample.
  const auto adm_state = pack_->padm->u_adm;
  const auto z4c_state = pack_->pz4c->u0;
  const auto aux = auxiliary_;
  const auto indcs = pack_->pmesh->mb_indcs;
  const int nmb = pack_->nmb_thispack;
  const auto errors = device_error_;
  const auto background = background_;
  const auto radius_method = radius_method_;
  const int npoint = npoint_;
  for (int ir = 0; ir < static_cast<int>(shells_.size()); ++ir) {
    const auto &s = shells_[ir];
    const Real R = s.R;
    const auto interp = [&]() {
      if constexpr (NGHOST == 2) return s.interp2;
      else if constexpr (NGHOST == 3) return s.interp3;
      else return s.interp4;
    }();
    const auto owner = s.owner;
    const auto reflected = s.reflected;
    const auto normal = s.normal;
    const auto x_lookup = s.x_lookup;
    const auto jac = s.jacobian;
    const auto invjac = s.inverse_jacobian;
    const auto theta = s.theta;
    const auto phi = s.phi;
    const auto weights = s.weights;
    const auto gamma = s.gamma;
    const auto beta_d = s.beta_d;
    const auto alpha = s.alpha;
    const auto beta2 = s.beta2;
    Kokkos::parallel_for("rwz_sample_background",
        Kokkos::RangePolicy<DevExeSpace>(0, npoint), KOKKOS_LAMBDA(const int q) {
      // Ownership precedes even coordinate divisions and determinant construction.
      if (owner(q) < 0) return;
      const auto iw = interp(q);
      if (!iw.point_exist || iw.ii0 != owner(q) || iw.ii0 >= nmb ||
          !Kokkos::isfinite(R) || !(R > 0) ||
          !Kokkos::isfinite(theta(q)) || !Kokkos::isfinite(phi(q)) ||
          !Kokkos::isfinite(weights(q)) || !(weights(q) > 0)) {
        Kokkos::atomic_max(&errors(0), int(invalid_sample));
        return;
      }
      const Real R2 = R*R;
      const Real st = Kokkos::sin(theta(q));
      const Real st2 = st*st;
      if (!Kokkos::isfinite(R2) || !(R2 > 0) ||
          !Kokkos::isfinite(st) || !(st > 0) || !(st2 > 0)) {
        Kokkos::atomic_max(&errors(0), int(invalid_sample));
        return;
      }
      const Real invR = Real(1)/R;
      const Real invR2 = invR*invR;
      const Real invst = Real(1)/st;
      const Real mu = invst*invst;
      if (!Kokkos::isfinite(invR) || !Kokkos::isfinite(invR2) ||
          !Kokkos::isfinite(invst) || !Kokkos::isfinite(mu)) {
        Kokkos::atomic_max(&errors(0), int(invalid_sample));
        return;
      }
      bool bad = false;
      // Fixed geometry was admitted during initialization and is immutable.
      Real n[3];
      for (int a = 0; a < 3; ++a) n[a] = normal(q,a);
      const int ilo = iw.ii1 - NGHOST + indcs.is + 1;
      const int jlo = iw.ii2 - NGHOST + indcs.js + 1;
      const int klo = iw.ii3 - NGHOST + indcs.ks + 1;
      bad = bad || ilo < 0 || jlo < 0 || klo < 0 ||
            iw.ii0 >= static_cast<int>(adm_state.extent(0)) ||
            iw.ii0 >= static_cast<int>(z4c_state.extent(0)) ||
            iw.ii0 >= static_cast<int>(aux.extent(0));
      for (int a = 2; a < 5; ++a) {
        const int last = (a == 2 ? klo : (a == 3 ? jlo : ilo)) + 2*NGHOST - 1;
        bad = bad || last >= static_cast<int>(adm_state.extent(a)) ||
              last >= static_cast<int>(z4c_state.extent(a)) ||
              last >= static_cast<int>(aux.extent(a));
      }
      if (bad) {
        Kokkos::atomic_max(&errors(0), int(invalid_sample));
        return;
      }
      // Interpolation reads every stencil operand, including zero-weight terms.
      // Its result checks below precede all nonlinear metric operations.
      bool bad_aux = false;
      // Compact Cartesian bundles retain symmetric field and derivative-pair aliasing.
      Real cg[NDerivative][6] = {}, ck[6] = {}, cb[NDerivative][3] = {};
      Real ca[NDerivative] = {}, dg[3][6] = {}, dk[3][6] = {};
      Real ddg[3][3][6] = {}, db[3][3] = {}, ddb[3][3][3] = {};
      Real da[3] = {}, dda[3][3] = {};
      Real av[34];
      const bool refl = reflected(q) != 0;
      // Metric gradients retain I_d[d_c gamma_ab], c <= d. Select each derivative
      // mask before entering its stencil, and share that pass with the value.
      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          const int ab = Sym(a,b);
          for (int c = 0; c < 3; ++c) {
            const auto sample = c == 0 ?
                InterpolateRWZField<NGHOST,7>(aux,6*c+ab,indcs,iw) :
                (c == 1 ? InterpolateRWZField<NGHOST,6>(aux,6*c+ab,indcs,iw) :
                          InterpolateRWZField<NGHOST,4>(aux,6*c+ab,indcs,iw));
            av[6*c+ab] = sample.value;
            bad_aux = bad_aux || !Kokkos::isfinite(sample.value);
            for (int d = c; d < 3; ++d) {
              const Real sign_abcd =
                  (refl && ((a == 2) + (b == 2) + (c == 2) + (d == 2))%2)
                  ? Real(-1) : Real(1);
              const Real v = sign_abcd*sample.derivative[d];
              ddg[c][d][ab] = ddg[d][c][ab] = v;
              bad_aux = bad_aux || !Kokkos::isfinite(v);
            }
          }
        }
      }
      // Lapse/shift retain the OTHER order, I_c[d_d f], c <= d.
      for (int d = 0; d < 3; ++d) {
        const auto lapse = d == 0 ?
            InterpolateRWZField<NGHOST,1>(aux,27+d,indcs,iw) :
            (d == 1 ? InterpolateRWZField<NGHOST,3>(aux,27+d,indcs,iw) :
                      InterpolateRWZField<NGHOST,7>(aux,27+d,indcs,iw));
        av[27+d] = lapse.value;
        bad_aux = bad_aux || !Kokkos::isfinite(lapse.value);
        for (int c = 0; c <= d; ++c) {
          const Real sign_cd = (refl && ((c == 2) + (d == 2))%2) ? Real(-1) : Real(1);
          const Real v = sign_cd*lapse.derivative[c];
          dda[c][d] = dda[d][c] = v;
          bad_aux = bad_aux || !Kokkos::isfinite(v);
        }
        for (int a = 0; a < 3; ++a) {
          const auto shift = d == 0 ?
              InterpolateRWZField<NGHOST,1>(aux,18+3*d+a,indcs,iw) :
              (d == 1 ? InterpolateRWZField<NGHOST,3>(aux,18+3*d+a,indcs,iw) :
                        InterpolateRWZField<NGHOST,7>(aux,18+3*d+a,indcs,iw));
          av[18+3*d+a] = shift.value;
          bad_aux = bad_aux || !Kokkos::isfinite(shift.value);
          for (int c = 0; c <= d; ++c) {
            const Real sign_cda = (refl && ((c == 2) + (d == 2) + (a == 2))%2)
                                 ? Real(-1) : Real(1);
            const Real v = sign_cda*shift.derivative[c];
            ddb[c][d][a] = ddb[d][c][a] = v;
            bad_aux = bad_aux || !Kokkos::isfinite(v);
          }
        }
      }
      // Gauge time derivatives need values only, not interpolant derivatives.
      for (int v = 30; v < 34; ++v) {
        av[v] = InterpolateLagrange<NGHOST>(aux,v,indcs,iw);
        bad_aux = bad_aux || !Kokkos::isfinite(av[v]);
      }
      ca[D00] = InterpolateLagrange<NGHOST>(z4c_state,Z4c::I_Z4C_ALPHA,indcs,iw);
      ca[D01] = av[30];
      bad = !Kokkos::isfinite(ca[D00]);
      for (int a = 0; a < 3; ++a) {
        const Real sign_a = (refl && a == 2) ? Real(-1) : Real(1);
        cb[D00][a] = sign_a*InterpolateLagrange<NGHOST>(z4c_state,
            Z4c::I_Z4C_BETAX + a,indcs,iw);
        cb[D01][a] = sign_a*av[31+a];
        da[a] = sign_a*av[27+a];
        bad = bad || !Kokkos::isfinite(cb[D00][a]);
        for (int b = 0; b < 3; ++b) {
          const Real sign_ab = (refl && ((a == 2) + (b == 2))%2) ? Real(-1) : Real(1);
          db[b][a] = sign_ab*av[18+3*b+a];
        }
        for (int b = a; b < 3; ++b) {
          const int ab = Sym(a,b);
          const Real sign_ab = (refl && ((a == 2) + (b == 2))%2) ? Real(-1) : Real(1);
          cg[D00][ab] = sign_ab*InterpolateLagrange<NGHOST>(adm_state,
              adm::ADM::I_ADM_GXX + ab,indcs,iw);
          const auto curvature = InterpolateRWZField<NGHOST,7>(adm_state,
              adm::ADM::I_ADM_KXX + ab,indcs,iw);
          ck[ab] = sign_ab*curvature.value;
          bad = bad || !Kokkos::isfinite(cg[D00][ab]) || !Kokkos::isfinite(ck[ab]);
          for (int c = 0; c < 3; ++c) {
            const Real sign_abc = (refl && ((a == 2) + (b == 2) + (c == 2))%2)
                                 ? Real(-1) : Real(1);
            dg[c][ab] = sign_abc*av[6*c+ab];
            dk[c][ab] = sign_abc*curvature.derivative[c];
            bad = bad || !Kokkos::isfinite(dk[c][ab]);
          }
        }
      }
      if (bad || bad_aux) {
        Kokkos::atomic_max(&errors(0), int(bad_aux ? invalid_auxiliary : invalid_sample));
        return;
      }
      // Positive-definiteness is tested on the physical, parity-restored metric.
      const Real gxx = cg[D00][0], gxy = cg[D00][1], gxz = cg[D00][2];
      const Real gyy = cg[D00][3], gyz = cg[D00][4], gzz = cg[D00][5];
      const Real minor2 = gxx*gyy-gxy*gxy;
      const Real det = -gxz*gxz*gyy+2*gxy*gxz*gyz-gxx*gyz*gyz
                       -gxy*gxy*gzz+gxx*gyy*gzz;
      if (!(gxx > 0) || !Kokkos::isfinite(minor2) || !(minor2 > 0) ||
          !Kokkos::isfinite(det) || !(det > 0)) {
        Kokkos::atomic_max(&errors(0), int(invalid_sample));
        return;
      }
      const Real sqrtdet = Kokkos::sqrt(det), invdet = Real(1)/det;
      if (!Kokkos::isfinite(sqrtdet) || !Kokkos::isfinite(invdet)) {
        Kokkos::atomic_max(&errors(0), int(invalid_sample));
        return;
      }
      Real gu[6] = {(gyy*gzz-gyz*gyz)*invdet, (gxz*gyz-gxy*gzz)*invdet,
                    (-gxz*gyy+gxy*gyz)*invdet, (gxx*gzz-gxz*gxz)*invdet,
                    (gxy*gxz-gxx*gyz)*invdet, minor2*invdet};
      for (int ab = 0; ab < 6; ++ab) bad = bad || !Kokkos::isfinite(gu[ab]);
      if (bad) {
        Kokkos::atomic_max(&errors(0), int(invalid_sample));
        return;
      }
      // ADM metric evolution and its spatial derivative use physical first
      // derivatives; interpolant derivatives supply only dK and second derivatives.
      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          const int ab = Sym(a,b);
          cg[D01][ab] = -2*ca[D00]*ck[ab];
          for (int c = 0; c < 3; ++c) {
            cg[D01][ab] += dg[c][ab]*cb[D00][c]
                          +cg[D00][Sym(a,c)]*db[b][c]
                          +cg[D00][Sym(b,c)]*db[a][c];
            Real dgt = -2*(ca[D00]*dk[c][ab]+da[c]*ck[ab]);
            for (int d = 0; d < 3; ++d) {
              dgt += dg[c][Sym(a,d)]*db[b][d]+cg[D00][Sym(a,d)]*ddb[c][b][d]
                    +dg[c][Sym(b,d)]*db[a][d]+cg[D00][Sym(b,d)]*ddb[c][a][d]
                    +db[c][d]*dg[d][ab]+cb[D00][d]*ddg[c][d][ab];
              cg[D20][ab] += n[c]*n[d]*ddg[c][d][ab];
            }
            cg[D10][ab] += n[c]*dg[c][ab];
            cg[D11][ab] += n[c]*dgt;
          }
        }
        for (int c = 0; c < 3; ++c) {
          cb[D10][a] += n[c]*db[c][a];
          for (int d = 0; d < 3; ++d) cb[D20][a] += n[c]*n[d]*ddb[c][d][a];
        }
      }
      for (int c = 0; c < 3; ++c) {
        ca[D10] += n[c]*da[c];
        for (int d = 0; d < 3; ++d) ca[D20] += n[c]*n[d]*dda[c][d];
      }
      for (int D = 0; D < NDerivative; ++D) {
        bad = bad || !Kokkos::isfinite(ca[D]);
        for (int a = 0; a < 3; ++a) bad = bad || !Kokkos::isfinite(cb[D][a]);
        for (int ab = 0; ab < 6; ++ab) bad = bad || !Kokkos::isfinite(cg[D][ab]);
      }
      if (bad) {
        Kokkos::atomic_max(&errors(0), int(invalid_sample));
        return;
      }
      Real L[NBackground] = {};
      Real trk = 0, imass = 0;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          trk += gu[Sym(a,b)]*ck[Sym(a,b)];
          for (int c = 0; c < 3; ++c) {
            imass += gu[Sym(a,b)]*(dg[b][Sym(a,c)]-dg[c][Sym(a,b)])*n[c];
          }
        }
      }
      const Real vol_adm = R2*weights(q);
      if (!Kokkos::isfinite(trk) || !Kokkos::isfinite(imass) ||
          !Kokkos::isfinite(vol_adm)) {
        Kokkos::atomic_max(&errors(0), int(invalid_sample));
        return;
      }
      L[I_adm_M] = Real(0.25)*imass*sqrtdet*vol_adm;
      // K_c^d depends only on c,d. Retain its e-sum and the outer charge order.
      Real mixed_k[3][3] = {};
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          for (int e = 0; e < 3; ++e) mixed_k[c][d] += gu[Sym(e,d)]*ck[Sym(c,e)];
        }
      }
      for (int a = 0; a < 3; ++a) {
        Real ip = 0, ij = 0;
        for (int b = 0; b < 3; ++b) {
          for (int c = 0; c < 3; ++c) {
            const Real eps = (a == b || b == c || c == a) ? Real(0)
                : (((a == 0 && b == 1) || (a == 1 && b == 2) ||
                    (a == 2 && b == 0)) ? Real(1) : Real(-1));
            for (int d = 0; d < 3; ++d) {
              // Preserve the source's explicit b sum, including its trace term.
              ip += (gu[Sym(b,c)]*ck[Sym(d,b)]-(d == c ? trk : Real(0)))
                    *(a == d ? Real(1) : Real(0))*n[c];
              // Reflected GLOBAL lookup coordinates are intentional GRA behavior.
              ij += (mixed_k[c][d]-(c == d ? trk : Real(0)))*eps*x_lookup(q,b)*n[d];
            }
          }
        }
        L[I_adm_Px+a] = Real(0.5)*ip*sqrtdet*vol_adm;
        L[I_adm_Jx+a] = Real(0.5)*ij*sqrtdet*vol_adm;
      }
      for (int v = I_adm_M; v <= I_adm_Jz; ++v) bad = bad || !Kokkos::isfinite(L[v]);
      if (bad) {
        Kokkos::atomic_max(&errors(0), int(invalid_sample));
        return;
      }
      // Spherical coordinate components, with explicit radial Jacobian products.
      // A time-independent fixed sphere has no Jacobian time derivative, but its
      // angular columns DO contribute p_A S_dot to the mixed radial-time metric.
      Real sg[NDerivative][6] = {}, bu[NDerivative][3] = {};
      Real bd[NDerivative][3] = {}, b2[NDerivative] = {};
      const Real p[3] = {0, invR, invR};
      const Real pv[3] = {0, -invR, -invR};
      const Real pv2[3] = {0, 2*invR2, 2*invR2};
      for (int A = 0; A < 3; ++A) {
        for (int B = A; B < 3; ++B) {
          Real S[NDerivative] = {};
          for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
              const Real JJ = jac(q,a,A)*jac(q,b,B);
              for (int D = 0; D < NDerivative; ++D) S[D] += JJ*cg[D][Sym(a,b)];
            }
          }
          const int AB = Sym(A,B);
          const Real pAB = p[A]+p[B];
          sg[D00][AB] = S[D00];
          sg[D01][AB] = S[D01];
          sg[D10][AB] = pAB*S[D00]+S[D10];
          sg[D20][AB] = 2*p[A]*p[B]*S[D00]+2*pAB*S[D10]+S[D20];
          sg[D11][AB] = pAB*S[D01]+S[D11];
        }
        Real U[NDerivative] = {};
        for (int a = 0; a < 3; ++a) {
          for (int D = 0; D < NDerivative; ++D) U[D] += invjac(q,A,a)*cb[D][a];
        }
        bu[D00][A] = U[D00];
        bu[D01][A] = U[D01];
        bu[D10][A] = pv[A]*U[D00]+U[D10];
        bu[D20][A] = pv2[A]*U[D00]+2*pv[A]*U[D10]+U[D20];
      }
      for (int D = 0; D < NDerivative; ++D) {
        for (int ab = 0; ab < 6; ++ab) bad = bad || !Kokkos::isfinite(sg[D][ab]);
        for (int a = 0; a < 3; ++a) bad = bad || !Kokkos::isfinite(bu[D][a]);
      }
      if (bad) {
        Kokkos::atomic_max(&errors(0), int(invalid_sample));
        return;
      }
      // Lower the shift in the ORIGINAL metric. This precedes subtraction/map.
      for (int A = 0; A < 3; ++A) {
        for (int B = 0; B < 3; ++B) {
          const int AB = Sym(A,B);
          bd[D00][A] += sg[D00][AB]*bu[D00][B];
          bd[D10][A] += sg[D10][AB]*bu[D00][B]+sg[D00][AB]*bu[D10][B];
          bd[D20][A] += sg[D20][AB]*bu[D00][B]+2*sg[D10][AB]*bu[D10][B]
                       +sg[D00][AB]*bu[D20][B];
          bd[D01][A] += sg[D01][AB]*bu[D00][B]+sg[D00][AB]*bu[D01][B];
        }
        for (int D = 0; D < NDerivative; ++D) bad = bad || !Kokkos::isfinite(bd[D][A]);
      }
      if (bad) {
        Kokkos::atomic_max(&errors(0), int(invalid_sample));
        return;
      }
      for (int A = 0; A < 3; ++A) {
        b2[D00] += bu[D00][A]*bd[D00][A];
        b2[D10] += bu[D10][A]*bd[D00][A]+bu[D00][A]*bd[D10][A];
        b2[D20] += bu[D20][A]*bd[D00][A]+2*bu[D10][A]*bd[D10][A]
                   +bu[D00][A]*bd[D20][A];
        b2[D01] += bu[D01][A]*bd[D00][A]+bu[D00][A]*bd[D01][A];
      }
      for (int D = 0; D < NDerivative; ++D) bad = bad || !Kokkos::isfinite(b2[D]);
      if (bad) {
        Kokkos::atomic_max(&errors(0), int(invalid_sample));
        return;
      }
      // All five radius definitions use the physical, preconversion angular metric.
      const Real induced_det = sg[D00][3]*sg[D00][5]-sg[D00][4]*sg[D00][4];
      if (!Kokkos::isfinite(induced_det) || !(induced_det > 0)) {
        Kokkos::atomic_max(&errors(0), int(invalid_area));
        return;
      }
      const Real w = weights(q);
      Real fr[NDerivative] = {};
      if (radius_method == RadiusMethod::areal ||
          radius_method == RadiusMethod::areal_simple) {
        Real D = sg[D00][3]*sg[D00][5];
        Real DR = sg[D10][3]*sg[D00][5]+sg[D00][3]*sg[D10][5];
        Real DRR = sg[D20][3]*sg[D00][5]+2*sg[D10][3]*sg[D10][5]
                   +sg[D00][3]*sg[D20][5];
        Real Dt = sg[D01][3]*sg[D00][5]+sg[D00][3]*sg[D01][5];
        Real DRt = sg[D11][3]*sg[D00][5]+sg[D10][3]*sg[D01][5]
                   +sg[D01][3]*sg[D10][5]+sg[D00][3]*sg[D11][5];
        if (radius_method == RadiusMethod::areal) {
          D = induced_det;
          DR -= 2*sg[D00][4]*sg[D10][4];
          DRR -= 2*sg[D10][4]*sg[D10][4]+2*sg[D00][4]*sg[D20][4];
          Dt -= 2*sg[D00][4]*sg[D01][4];
          DRt -= 2*sg[D10][4]*sg[D01][4]+2*sg[D00][4]*sg[D11][4];
        }
        if (!Kokkos::isfinite(D) || !(D > 0) || !Kokkos::isfinite(DR) ||
            !Kokkos::isfinite(DRR) || !Kokkos::isfinite(Dt) || !Kokkos::isfinite(DRt)) {
          Kokkos::atomic_max(&errors(0), int(invalid_area));
          return;
        }
        const Real area = Kokkos::sqrt(D);
        if (!Kokkos::isfinite(area) || !(area > 0)) {
          Kokkos::atomic_max(&errors(0), int(invalid_area));
          return;
        }
        const Real invarea = Real(1)/area;
        // Normalize determinant derivatives before quadratic terms: cubing the
        // inverse area can overflow Real even when the integrand is finite.
        const Real DR_invarea = DR*invarea, Dt_invarea = Dt*invarea;
        const Real u = w*invst;
        if (!Kokkos::isfinite(invarea) || !Kokkos::isfinite(u)) {
          Kokkos::atomic_max(&errors(0), int(invalid_area));
          return;
        }
        fr[D00] = area*u;
        fr[D10] = Real(0.25)*DR*invarea*u;
        fr[D20] = (Real(0.25)*DRR*invarea-Real(0.125)*DR_invarea*DR_invarea*invarea)*u;
        // F_t is ALREADY one half of d_t(radius squared); no extra 1/2 later.
        fr[D01] = Real(0.25)*Dt*invarea*u;
        fr[D11] = (Real(0.25)*DRt*invarea-Real(0.125)*DR_invarea*Dt_invarea*invarea)*u;
      } else {
        Real base[NDerivative] = {};
        for (int D = 0; D < NDerivative; ++D) {
          switch (radius_method) {
            case RadiusMethod::average_schw:
              base[D] = Real(0.5)*(sg[D][3]+sg[D][5]*mu);
              break;
            case RadiusMethod::schw_gthth:
              base[D] = sg[D][3];
              break;
            case RadiusMethod::schw_gphph:
              base[D] = sg[D][5]*mu;
              break;
            default:
              Kokkos::atomic_max(&errors(0), int(invalid_area));
              return;
          }
          bad = bad || !Kokkos::isfinite(base[D]);
        }
        if (bad || !(base[D00] > 0)) {
          Kokkos::atomic_max(&errors(0), int(invalid_area));
          return;
        }
        fr[D00] = base[D00]*w;
        for (int D = D10; D < NDerivative; ++D) fr[D] = Real(0.5)*base[D]*w;
      }
      L[Irsch2] = fr[D00];
      L[Idrsch_dri] = fr[D10];
      L[Id2rsch_dri2] = fr[D20];
      L[Idot_rsch2] = fr[D01];
      L[Idrsch_dri_dot] = fr[D11];
      L[Ig00] = w*(-ca[D00]*ca[D00]+b2[D00]);
      L[Ig0R] = w*bd[D00][0];
      L[IgRR] = w*sg[D00][0];
      L[IgRt] = w*sg[D00][1];
      L[Igtt] = w*sg[D00][3];
      L[Igpp] = w*sg[D00][5];
      L[IdR_g00] = w*(-2*ca[D00]*ca[D10]+b2[D10]);
      L[IdR_g0R] = w*bd[D10][0];
      L[IdR_gRR] = w*sg[D10][0];
      L[IdR2_g00] = w*(-2*ca[D00]*ca[D20]-2*ca[D10]*ca[D10]+b2[D20]);
      L[IdR2_g0R] = w*bd[D20][0];
      L[IdR2_gRR] = w*sg[D20][0];
      L[IdR_gtt] = w*sg[D10][3];
      L[Idot_g00] = w*(-2*ca[D00]*ca[D01]+b2[D01]);
      L[Idot_g0R] = w*bd[D01][0];
      L[Idot_gRR] = w*sg[D01][0];
      for (int v = Irsch2; v < NBackground; ++v) bad = bad || !Kokkos::isfinite(L[v]);
      if (bad) {
        Kokkos::atomic_max(&errors(0), int(invalid_area));
        return;
      }
      // Commit only fully admitted owned points. Consumers never read the
      // unproduced D11 scalar/vector slots or any nonowned point.
      for (int D = 0; D < NDerivative; ++D) {
        for (int ab = 0; ab < 6; ++ab) gamma(q,D,ab) = sg[D][ab];
        if (D == D11) continue;
        alpha(q,D,0) = ca[D];
        beta2(q,D,0) = b2[D];
        for (int A = 0; A < 3; ++A) {
          beta_d(q,D,A) = bd[D][A];
        }
      }
      for (int v = 0; v < NBackground; ++v) Kokkos::atomic_add(&background(ir,v), L[v]);
    });
    // Finite point integrands can still overflow a compact local sum. Admit the
    // completed buffer before the manager's host/collective phase boundary.
    Kokkos::parallel_for("rwz_check_local_background",
        Kokkos::RangePolicy<DevExeSpace>(0, NBackground), KOKKOS_LAMBDA(const int v) {
      if (!Kokkos::isfinite(background(ir,v))) {
        Kokkos::atomic_max(&errors(0), int(v <= I_adm_Jz ? invalid_sample : invalid_area));
      }
    });
  }
}

void WaveExtractRWZ::AccumulateMultipoles() {
  // Background finalization has already performed the ordered subtraction/map.
  // No sphere fields are copied to host or globally replicated for projections.
  // The default projection instance orders this clear before its atomics.
  const DevExeSpace exec;
  Kokkos::deep_copy(exec, multipoles_, Real(0));
  const auto multipoles = multipoles_;
  const auto errors = device_error_;
  const int lmax = lmax_;
  const int nmode = nmode_;
  const int npoint = npoint_;
  const bool subtract_background = subtract_background_;
  for (int ir = 0; ir < static_cast<int>(shells_.size()); ++ir) {
    const auto &s = shells_[ir];
    const Real r = s.Schwarzschild_radius;
    // The t-t background component is unchanged by the inherited radial map:
    // this is exactly the normalized, presubtraction B[Ig00] snapshot.
    const Real background_g00 = s.g_dd(0,0);
    const auto owner = s.owner;
    const auto theta = s.theta;
    const auto weights = s.weights;
    const auto gamma = s.gamma;
    const auto beta_d = s.beta_d;
    const auto alpha = s.alpha;
    const auto beta2 = s.beta2;
    const auto y = s.harmonics[Y];
    const auto yt = s.harmonics[Ytheta];
    const auto yp = s.harmonics[Yphi];
    const auto hw = s.harmonics[W];
    const auto hx = s.harmonics[X];
    Kokkos::parallel_for("rwz_multipole_projection",
        Kokkos::RangePolicy<DevExeSpace>(0, npoint), KOKKOS_LAMBDA(const int q) {
      if (owner(q) < 0) return;
      if (!Kokkos::isfinite(r) || !(r > 0) ||
          !Kokkos::isfinite(theta(q)) || !Kokkos::isfinite(weights(q)) ||
          !(weights(q) > 0) || (subtract_background && !Kokkos::isfinite(background_g00))) {
        Kokkos::atomic_max(&errors(0), int(invalid_projection));
        return;
      }
      const Real r2 = r*r;
      const Real st = Kokkos::sin(theta(q));
      const Real st2 = st*st;
      if (!Kokkos::isfinite(r2) || !(r2 > Real(1e-20)) ||
          !Kokkos::isfinite(st) || !(st > 0) || !(st2 > 0)) {
        Kokkos::atomic_max(&errors(0), int(invalid_projection));
        return;
      }
      const Real invr = Real(1)/r, invst = Real(1)/st;
      const Real invr2 = invr*invr, invr3 = invr2*invr, invr4 = invr3*invr;
      const Real mu = invst*invst;
      if (!Kokkos::isfinite(invr) || !Kokkos::isfinite(invr2) ||
          !Kokkos::isfinite(invr3) || !Kokkos::isfinite(invr4) ||
          !Kokkos::isfinite(invst) || !Kokkos::isfinite(mu)) {
        Kokkos::atomic_max(&errors(0), int(invalid_projection));
        return;
      }
      bool bad = false;
      Real g[NDerivative][6] = {}, bd[NDerivative][3] = {};
      Real a[NDerivative] = {}, b2[NDerivative] = {}, f00[NDerivative] = {};
      // Read only the post-map slots that have an actual downstream consumer.
      // In particular beta_d[D11] and unchanged radial D20/D11 are not derivatives
      // to be invented or read by this projection contract.
      for (int D = 0; D < NDerivative; ++D) {
        for (int ab = 0; ab < 6; ++ab) {
          if (ab >= 3 || D == D00 || D == D10 || D == D01 ||
              (D == D11 && ab > 0)) {
            g[D][ab] = gamma(q,D,ab);
            bad = bad || !Kokkos::isfinite(g[D][ab]);
          }
        }
        if (D == D00 || D == D10 || D == D01) {
          a[D] = alpha(q,D,0);
          b2[D] = beta2(q,D,0);
          bad = bad || !Kokkos::isfinite(a[D]) || !Kokkos::isfinite(b2[D]);
          for (int A = 0; A < 3; ++A) {
            bd[D][A] = beta_d(q,D,A);
            bad = bad || !Kokkos::isfinite(bd[D][A]);
          }
        } else if (D == D20) {
          for (int A = 1; A < 3; ++A) {
            bd[D][A] = beta_d(q,D,A);
            bad = bad || !Kokkos::isfinite(bd[D][A]);
          }
        }
      }
      if (bad) {
        Kokkos::atomic_max(&errors(0), int(invalid_projection));
        return;
      }
      f00[D00] = -a[D00]*a[D00]+b2[D00]
                  -(subtract_background ? background_g00 : Real(0));
      f00[D10] = -2*a[D00]*a[D10]+b2[D10];
      f00[D01] = -2*a[D00]*a[D01]+b2[D01];
      // Columns are T=theta-theta minus phi-phi/sin^2, C=theta-phi/sin^2,
      // and U=the angular trace. E applies ONLY the prescribed 1/r^2 rules;
      // no derivative of a time-dependent radius prefactor is introduced.
      Real angular[NDerivative][3] = {}, E[NDerivative][3] = {};
      for (int D = 0; D < NDerivative; ++D) {
        angular[D][0] = g[D][3]-g[D][5]*mu;
        angular[D][1] = g[D][4]*mu;
        angular[D][2] = g[D][3]+g[D][5]*mu;
        for (int A = 0; A < 3; ++A) bad = bad || !Kokkos::isfinite(angular[D][A]);
      }
      if (bad) {
        Kokkos::atomic_max(&errors(0), int(invalid_projection));
        return;
      }
      for (int A = 0; A < 3; ++A) {
        E[D00][A] = angular[D00][A]*invr2;
        E[D10][A] = angular[D10][A]*invr2-2*angular[D00][A]*invr3;
        E[D20][A] = angular[D20][A]*invr2-4*angular[D10][A]*invr3
                    +6*angular[D00][A]*invr4;
        E[D01][A] = angular[D01][A]*invr2;
        E[D11][A] = angular[D11][A]*invr2-2*angular[D01][A]*invr3;
        for (int D = 0; D < NDerivative; ++D) bad = bad || !Kokkos::isfinite(E[D][A]);
      }
      bad = bad || !Kokkos::isfinite(f00[D00]) || !Kokkos::isfinite(f00[D10]) ||
            !Kokkos::isfinite(f00[D01]);
      if (bad) {
        Kokkos::atomic_max(&errors(0), int(invalid_projection));
        return;
      }
      for (int l = 2; l <= lmax; ++l) {
        const Real lambda = Real(l)*Real(l+1);
        const Real tensor_norm = lambda*(lambda-2);
        if (!Kokkos::isfinite(lambda) || !(lambda > 2) ||
            !Kokkos::isfinite(tensor_norm) || !(tensor_norm > 0)) {
          Kokkos::atomic_max(&errors(0), int(invalid_projection));
          return;
        }
        const Real invlambda = Real(1)/lambda;
        const Real invtensor = Real(1)/tensor_norm;
        const Real invoddvector = invst*invlambda;
        const Real invoddtensor = invst*invtensor;
        if (!Kokkos::isfinite(invlambda) || !Kokkos::isfinite(invtensor) ||
            !Kokkos::isfinite(invoddvector) || !Kokkos::isfinite(invoddtensor)) {
          Kokkos::atomic_max(&errors(0), int(invalid_projection));
          return;
        }
        for (int m = -l; m <= l; ++m) {
          const int lm = l*l+l+m-4;
          for (int c = 0; c < 2; ++c) {
            // All sampled fields are real; conjugate every complex basis once.
            // X already contains its explicit imaginary factor in the fixed table.
            const Real sign = c == 0 ? Real(1) : Real(-1);
            const Real Yc = sign*y(q,lm,c), Ytc = sign*yt(q,lm,c);
            const Real Ypc = sign*yp(q,lm,c), Wc = sign*hw(q,lm,c);
            const Real Xc = sign*hx(q,lm,c);
            Real P[NMultipoles] = {};
            P[Ih00] = f00[D00]*Yc;
            P[Ih01] = bd[D00][0]*Yc;
            P[Ih11] = g[D00][0]*Yc;
            P[Ih0] = (bd[D00][1]*Ytc+bd[D00][2]*Ypc*mu)*invlambda;
            P[Ih1] = (g[D00][1]*Ytc+g[D00][2]*Ypc*mu)*invlambda;
            P[IG] = (E[D00][0]*Wc+2*E[D00][1]*Xc)*invtensor;
            P[IK] = Real(0.5)*E[D00][2]*Yc;
            P[Ih00_dr] = f00[D10]*Yc;
            P[Ih01_dr] = bd[D10][0]*Yc;
            P[Ih11_dr] = g[D10][0]*Yc;
            P[Ih0_dr] = (bd[D10][1]*Ytc+bd[D10][2]*Ypc*mu)*invlambda;
            P[Ih1_dr] = (g[D10][1]*Ytc+g[D10][2]*Ypc*mu)*invlambda;
            P[IG_dr] = (E[D10][0]*Wc+2*E[D10][1]*Xc)*invtensor;
            P[IK_dr] = Real(0.5)*E[D10][2]*Yc;
            P[IG_dr2] = (E[D20][0]*Wc+2*E[D20][1]*Xc)*invtensor;
            P[IK_dr2] = Real(0.5)*E[D20][2]*Yc;
            P[IG_dr_dot] = (E[D11][0]*Wc+2*E[D11][1]*Xc)*invtensor;
            P[IK_dr_dot] = Real(0.5)*E[D11][2]*Yc;
            P[Ih00_dot] = f00[D01]*Yc;
            P[Ih01_dot] = bd[D01][0]*Yc;
            P[Ih11_dot] = g[D01][0]*Yc;
            P[Ih0_dot] = (bd[D01][1]*Ytc+bd[D01][2]*Ypc*mu)*invlambda;
            P[Ih1_dot] = (g[D01][1]*Ytc+g[D01][2]*Ypc*mu)*invlambda;
            P[IG_dot] = (E[D01][0]*Wc+2*E[D01][1]*Xc)*invtensor;
            P[IK_dot] = Real(0.5)*E[D01][2]*Yc;
            P[IH0] = (-bd[D00][1]*Ypc+bd[D00][2]*Ytc)*invoddvector;
            P[IH1] = (-g[D00][1]*Ypc+g[D00][2]*Ytc)*invoddvector;
            P[IH] = (-angular[D00][0]*Xc+2*g[D00][4]*Wc)*invoddtensor;
            P[IH0_dr] = (-bd[D10][1]*Ypc+bd[D10][2]*Ytc)*invoddvector;
            P[IH1_dr] = (-g[D10][1]*Ypc+g[D10][2]*Ytc)*invoddvector;
            P[IH_dr] = (-angular[D10][0]*Xc+2*g[D10][4]*Wc)*invoddtensor;
            P[IH0_dot] = (-bd[D01][1]*Ypc+bd[D01][2]*Ytc)*invoddvector;
            P[IH1_dot] = (-g[D01][1]*Ypc+g[D01][2]*Ytc)*invoddvector;
            P[IH_dot] = (-angular[D01][0]*Xc+2*g[D01][4]*Wc)*invoddtensor;
            P[IH0_dr2] = (-bd[D20][1]*Ypc+bd[D20][2]*Ytc)*invoddvector;
            P[IH_dr2] = (-angular[D20][0]*Xc+2*g[D20][4]*Wc)*invoddtensor;
            P[IH1_dr_dot] = (-g[D11][1]*Ypc+g[D11][2]*Ytc)*invoddvector;
            // Trace correction is intentionally NOT included here. The completed
            // local G integral is added once by the compact kernel below.
            for (int v = 0; v < NMultipoles; ++v) {
              P[v] *= weights(q);
              bad = bad || !Kokkos::isfinite(P[v]);
            }
            if (bad) {
              Kokkos::atomic_max(&errors(0), int(invalid_projection));
              return;
            }
            for (int v = 0; v < NMultipoles; ++v) {
              Kokkos::atomic_add(&multipoles(ir,v,lm,c), P[v]);
            }
          }
        }
      }
    });
    // Default execution-space ordering completes the local G projection before
    // this point-free compact correction. Each (mode,complex,derivative) is unique.
    Kokkos::parallel_for("rwz_local_trace_correction",
        Kokkos::RangePolicy<DevExeSpace>(0, nmode), KOKKOS_LAMBDA(const int lm) {
      int l = 2;
      while (lm >= (l+1)*(l+1)-4) ++l;
      const Real lambda = Real(l)*Real(l+1);
      const int gslot[NDerivative] = {IG, IG_dr, IG_dr2, IG_dot, IG_dr_dot};
      const int kslot[NDerivative] = {IK, IK_dr, IK_dr2, IK_dot, IK_dr_dot};
      for (int D = 0; D < NDerivative; ++D) {
        for (int c = 0; c < 2; ++c) {
          const Real ge = multipoles(ir,gslot[D],lm,c);
          const Real ke = multipoles(ir,kslot[D],lm,c);
          if (!Kokkos::isfinite(ge) || !Kokkos::isfinite(ke)) {
            Kokkos::atomic_max(&errors(0), int(invalid_projection));
            return;
          }
          const Real corrected = ke+Real(0.5)*lambda*ge;
          if (!Kokkos::isfinite(corrected)) {
            Kokkos::atomic_max(&errors(0), int(invalid_projection));
            return;
          }
          multipoles(ir,kslot[D],lm,c) = corrected;
        }
      }
      // Check every completed local sum, including the scalar/vector slots that
      // do not enter the trace correction, before the caller's collective gate.
      for (int v = 0; v < NMultipoles; ++v) {
        for (int c = 0; c < 2; ++c) {
          if (!Kokkos::isfinite(multipoles(ir,v,lm,c))) {
            Kokkos::atomic_max(&errors(0), int(invalid_projection));
            return;
          }
        }
      }
    });
  }
}

void WaveExtractRWZ::FinalizeBackgroundAndTransform() {
  if (host_background_.extent(0) != shells_.size() ||
      host_background_.extent(1) != NBackground || npoint_ <= 0) {
    local_error_ = invalid_configuration;
    return;
  }
  if (device_error_.extent(0) != 1) {
    local_error_ = invalid_configuration;
    return;
  }
  for (const auto& shell : shells_) {
    if (shell.owner.extent(0) != static_cast<std::size_t>(npoint_) ||
        shell.theta.extent(0) != static_cast<std::size_t>(npoint_) ||
        shell.gamma.extent(0) != static_cast<std::size_t>(npoint_) ||
        shell.gamma.extent(1) != NDerivative || shell.gamma.extent(2) != 6 ||
        shell.beta_d.extent(0) != static_cast<std::size_t>(npoint_) ||
        shell.beta_d.extent(1) != NDerivative || shell.beta_d.extent(2) != 3 ||
        shell.alpha.extent(0) != static_cast<std::size_t>(npoint_) ||
        shell.alpha.extent(1) != NDerivative || shell.alpha.extent(2) != 1 ||
        shell.beta2.extent(0) != static_cast<std::size_t>(npoint_) ||
        shell.beta2.extent(1) != NDerivative || shell.beta2.extent(2) != 1) {
      local_error_ = invalid_configuration;
      return;
    }
  }
  const Real div_4PI = 1.0 / (4.0 * std::acos(-1.0));
  constexpr Real eps_det = 1e-12;
  std::vector<Real> jacobian_dot(shells_.size());
  for (std::size_t is = 0; is < shells_.size(); ++is) {
    auto& shell = shells_[is];
    // Normalize every slot exactly once, including the seven ADM charges.
    // Keep this compact normalized buffer for the later constant-h00
    // projection.
    for (int slot = 0; slot < NBackground; ++slot) {
      host_background_(is, slot) *= div_4PI;
      if (!std::isfinite(host_background_(is, slot))) {
        local_error_ = invalid_background;
        return;
      }
    }
    shell.ADM_M = host_background_(is, I_adm_M);
    for (int a = 0; a < 3; ++a) {
      shell.ADM_P[a] = host_background_(is, I_adm_Px + a);
      shell.ADM_J[a] = host_background_(is, I_adm_Jx + a);
    }

    const Real rsch2 = host_background_(is, Irsch2);
    if (!std::isfinite(rsch2) || rsch2 <= 1e-20) {
      local_error_ = invalid_background;
      return;
    }
    const Real rsch = std::sqrt(rsch2);
    if (!std::isfinite(rsch) || rsch <= 0) {
      local_error_ = invalid_background;
      return;
    }
    const Real div_rsch = 1.0 / rsch;
    if (!std::isfinite(div_rsch) || div_rsch == 0) {
      local_error_ = invalid_background;
      return;
    }
    const Real dot_rsch = div_rsch * host_background_(is, Idot_rsch2);
    // TODO We do not have second time derivatives; dot2_rsch remains zero.
    // This approximation is not reconstructed from previous output samples.
    const Real dot2_rsch = 0.0;
    (void)dot2_rsch;
    Real drsch_dri = host_background_(is, Idrsch_dri);
    Real d2rsch_dri2 = host_background_(is, Id2rsch_dri2);
    Real drsch_dri_dot = host_background_(is, Idrsch_dri_dot);
    const Real g00 = host_background_(is, Ig00);
    const Real g0R = host_background_(is, Ig0R);
    const Real gRR = host_background_(is, IgRR);
    const Real gRt = host_background_(is, IgRt);
    const Real gtt = host_background_(is, Igtt);
    const Real gpp = host_background_(is, Igpp);
    (void)gRt;
    (void)gpp;
    // IgRt and Igpp were normalized too, but are not subtraction operands:
    // phi-phi subtraction deliberately uses Igtt*sin(theta)^2, never Igpp.
    const Real dR_g00 = host_background_(is, IdR_g00);
    const Real dR_g0R = host_background_(is, IdR_g0R);
    const Real dR_gRR = host_background_(is, IdR_gRR);
    const Real dR_gtt = host_background_(is, IdR_gtt);
    const Real dR2_g00 = host_background_(is, IdR2_g00);
    const Real dR2_g0R = host_background_(is, IdR2_g0R);
    const Real dR2_gRR = host_background_(is, IdR2_gRR);
    const Real dot_g00 = host_background_(is, Idot_g00);
    const Real dot_g0R = host_background_(is, Idot_g0R);
    const Real dot_gRR = host_background_(is, Idot_gRR);
    if (!std::isfinite(dot_rsch)) {
      local_error_ = invalid_background;
      return;
    }

    // Radial map: source overwrites and missing third-derivative term retained.
    drsch_dri *= div_rsch;
    if (!std::isfinite(drsch_dri) || drsch_dri == 0) {
      local_error_ = singular_radial_map;
      return;
    }
    const Real dri_drsch = 1.0 / drsch_dri;
    if (!std::isfinite(dri_drsch) || dri_drsch == 0) {
      local_error_ = singular_radial_map;
      return;
    }
    d2rsch_dri2 = -div_rsch * (SQR(drsch_dri) - d2rsch_dri2);
    if (!std::isfinite(d2rsch_dri2)) {
      local_error_ = singular_radial_map;
      return;
    }
    const Real d2ri_drsch2 = -std::pow(dri_drsch, 3) * d2rsch_dri2;
    if (!std::isfinite(d2ri_drsch2)) {
      local_error_ = singular_radial_map;
      return;
    }
    const Real d3rsch_dri3 = -3.0 * div_rsch * drsch_dri *
                             d2rsch_dri2;  // + omitted d3r_g_ab, next 1/R order
    if (!std::isfinite(d3rsch_dri3)) {
      local_error_ = singular_radial_map;
      return;
    }
    const Real d3ri_drsch3 = -std::pow(dri_drsch, 4) * d3rsch_dri3 +
                             3.0 * SQR(d2rsch_dri2) * std::pow(dri_drsch, 5);
    if (!std::isfinite(d3ri_drsch3)) {
      local_error_ = singular_radial_map;
      return;
    }
    drsch_dri_dot = div_rsch * (-dot_rsch * drsch_dri + drsch_dri_dot);
    if (!std::isfinite(drsch_dri_dot)) {
      local_error_ = singular_radial_map;
      return;
    }
    const Real dri_drsch_dot = -SQR(dri_drsch) * drsch_dri_dot;
    if (!std::isfinite(dri_drsch_dot)) {
      local_error_ = singular_radial_map;
      return;
    }
    auto& g_dd = shell.g_dd;
    auto& g_dr_dd = shell.g_dr_dd;
    auto& g_dr2_dd = shell.g_dr2_dd;
    auto& g_dot_dd = shell.g_dot_dd;
    auto& g_uu = shell.g_uu;
    auto& g_dr_uu = shell.g_dr_uu;
    auto& g_dr2_uu = shell.g_dr2_uu;
    auto& g_dot_uu = shell.g_dot_uu;
    auto& Gamma_udd = shell.Gamma_udd;
    auto& Gamma_dyn_udd = shell.Gamma_dyn_udd;
    auto& norm_Delta_Gamma = shell.norm_Delta_Gamma;
    auto& Schwarzschild_radius = shell.Schwarzschild_radius;
    auto& Schwarzschild_mass = shell.Schwarzschild_mass;
    // time derivatives of g0r and grr
    const Real dot_g0r = dri_drsch_dot * g0R + dri_drsch * dot_g0R;
    if (!std::isfinite(dot_g0r)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real dot_grr =
        2.0 * dri_drsch * dri_drsch_dot * gRR + SQR(dri_drsch) * dot_gRR;
    if (!std::isfinite(dot_grr)) {
      local_error_ = invalid_two_metric;
      return;
    }

    const Real dr2_g00 = d2ri_drsch2 * dR_g00 + SQR(dri_drsch) * dR2_g00;
    if (!std::isfinite(dr2_g00)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real dr2_g0r = d3ri_drsch3 * g0R +
                         3.0 * d2ri_drsch2 * dri_drsch * dR_g0R +
                         std::pow(dri_drsch, 3) * dR2_g0R;
    if (!std::isfinite(dr2_g0r)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real dr2_grr =
        2.0 * gRR * (SQR(d2ri_drsch2) + dri_drsch * d3ri_drsch3) +
        5.0 * SQR(dri_drsch) * d2ri_drsch2 * dR_gRR +
        std::pow(dri_drsch, 4) * dR2_gRR;
    if (!std::isfinite(dr2_grr)) {
      local_error_ = invalid_two_metric;
      return;
    }

    const Real dr_g00 = dri_drsch * dR_g00;
    if (!std::isfinite(dr_g00)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real dr_g0r = d2ri_drsch2 * g0R + SQR(dri_drsch) * dR_g0R;
    if (!std::isfinite(dr_g0r)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real dr_grr =
        2.0 * dri_drsch * d2ri_drsch2 * gRR + std::pow(dri_drsch, 3) * dR_gRR;
    if (!std::isfinite(dr_grr)) {
      local_error_ = invalid_two_metric;
      return;
    }

    const Real g0r = dri_drsch * g0R;
    if (!std::isfinite(g0r)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real grr = SQR(dri_drsch) * gRR;
    if (!std::isfinite(grr)) {
      local_error_ = invalid_two_metric;
      return;
    }

    // Inverse metric & Christoffel's symbols of the background metric

    // Determinant
    const Real detg = g00 * grr - SQR(g0r);
    if (!std::isfinite(detg) || detg >= 0 || std::fabs(detg) < eps_det) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real div_detg = 1.0 / detg;
    if (!std::isfinite(div_detg) || div_detg == 0) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real div_detg2 = SQR(div_detg);
    if (!std::isfinite(div_detg2)) {
      local_error_ = invalid_two_metric;
      return;
    }

    // Inverse matrix
    const Real g00_uu = grr * div_detg;
    if (!std::isfinite(g00_uu)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real g0r_uu = -g0r * div_detg;
    if (!std::isfinite(g0r_uu)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real grr_uu = g00 * div_detg;
    if (!std::isfinite(grr_uu)) {
      local_error_ = invalid_two_metric;
      return;
    }

    // Derivatives of the inverse metric
    const Real dr_div_detg =
        -div_detg2 * (dr_g00 * grr + g00 * dr_grr - 2.0 * g0r * dr_g0r);
    if (!std::isfinite(dr_div_detg)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real dr_g00_uu = dr_grr * div_detg + grr * dr_div_detg;
    if (!std::isfinite(dr_g00_uu)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real dr_g0r_uu = -dr_g0r * div_detg - g0r * dr_div_detg;
    if (!std::isfinite(dr_g0r_uu)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real dr_grr_uu = dr_g00 * div_detg + g00 * dr_div_detg;
    if (!std::isfinite(dr_grr_uu)) {
      local_error_ = invalid_two_metric;
      return;
    }

    const Real dot_div_detg =
        -div_detg2 * (dot_g00 * grr + g00 * dot_grr - 2.0 * g0r * dot_g0r);
    if (!std::isfinite(dot_div_detg)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real dot_g00_uu = dot_grr * div_detg + grr * dot_div_detg;
    if (!std::isfinite(dot_g00_uu)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real dot_g0r_uu = -dot_g0r * div_detg - g0r * dot_div_detg;
    if (!std::isfinite(dot_g0r_uu)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real dot_grr_uu = dot_g00 * div_detg + g00 * dot_div_detg;
    if (!std::isfinite(dot_grr_uu)) {
      local_error_ = invalid_two_metric;
      return;
    }

    const Real dr2_div_detg =
        -div_detg2 * (dr2_g00 * grr + 2.0 * dr_g00 * dr_grr + g00 * dr2_grr -
                      2.0 * SQR(dr_g0r) - 2.0 * g0r * dr2_g0r) +
        2.0 * SQR(dr_div_detg) / div_detg;
    if (!std::isfinite(dr2_div_detg)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real dr2_g00_uu =
        dr2_grr * div_detg + 2.0 * dr_grr * dr_div_detg + grr * dr2_div_detg;
    if (!std::isfinite(dr2_g00_uu)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real dr2_g0r_uu =
        -dr2_g0r * div_detg - 2.0 * dr_g0r * dr_div_detg - g0r * dr2_div_detg;
    if (!std::isfinite(dr2_g0r_uu)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real dr2_grr_uu =
        dr2_g00 * div_detg + 2.0 * dr_g00 * dr_div_detg + g00 * dr2_div_detg;
    if (!std::isfinite(dr2_grr_uu)) {
      local_error_ = invalid_two_metric;
      return;
    }

    // G stands for Gamma, these are the Christoffel symbols.
    // Assuming time independence:
    const Real G000 = -0.5 * g0r_uu * dr_g00;
    if (!std::isfinite(G000)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real Gr00 = -0.5 * grr_uu * dr_g00;
    if (!std::isfinite(Gr00)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real G00r = 0.5 * g00_uu * dr_g00;
    if (!std::isfinite(G00r)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real Gr0r = 0.5 * g0r_uu * dr_g00;
    if (!std::isfinite(Gr0r)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real G0rr = g00_uu * dr_g0r + 0.5 * g0r_uu * dr_grr;
    if (!std::isfinite(G0rr)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real Grrr = g0r_uu * dr_g0r + 0.5 * grr_uu * dr_grr;
    if (!std::isfinite(Grrr)) {
      local_error_ = invalid_two_metric;
      return;
    }

    // Generic (time-dependent)
    // *_dyn stands for "dynamic": it uses time derivatives of the bakgrund
    // metric
    const Real G000_dyn =
        0.5 * g00_uu * dot_g00 + g0r_uu * dot_g0r - 0.5 * g0r_uu * dr_g00;
    if (!std::isfinite(G000_dyn)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real Gr00_dyn =
        grr_uu * dot_g0r - 0.5 * grr_uu * dr_g00 + 0.5 * g0r_uu * dot_g00;
    if (!std::isfinite(Gr00_dyn)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real G00r_dyn = 0.5 * g00_uu * dr_g00 + 0.5 * g0r_uu * dot_grr;
    if (!std::isfinite(G00r_dyn)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real Gr0r_dyn = 0.5 * g0r_uu * dr_g00 + 0.5 * grr_uu * dot_grr;
    if (!std::isfinite(Gr0r_dyn)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real G0rr_dyn =
        g00_uu * dr_g0r - 0.5 * g00_uu * dot_grr + 0.5 * g0r_uu * dr_grr;
    if (!std::isfinite(G0rr_dyn)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real Grrr_dyn =
        g0r_uu * dr_g0r - 0.5 * g0r_uu * dot_grr + 0.5 * grr_uu * dr_grr;
    if (!std::isfinite(Grrr_dyn)) {
      local_error_ = invalid_two_metric;
      return;
    }

    // Compute differences
    const Real Delta_G000 = G000 - G000_dyn;
    if (!std::isfinite(Delta_G000)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real Delta_Gr00 = Gr00 - Gr00_dyn;
    if (!std::isfinite(Delta_Gr00)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real Delta_G00r = G00r - G00r_dyn;
    if (!std::isfinite(Delta_G00r)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real Delta_Gr0r = Gr0r - Gr0r_dyn;
    if (!std::isfinite(Delta_Gr0r)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real Delta_G0rr = G0rr - G0rr_dyn;
    if (!std::isfinite(Delta_G0rr)) {
      local_error_ = invalid_two_metric;
      return;
    }
    const Real Delta_Grrr = Grrr - Grrr_dyn;
    if (!std::isfinite(Delta_Grrr)) {
      local_error_ = invalid_two_metric;
      return;
    }

    norm_Delta_Gamma = SQR(Delta_G000) + SQR(Delta_G00r) + SQR(Delta_G0rr);
    if (!std::isfinite(norm_Delta_Gamma)) {
      local_error_ = invalid_two_metric;
      return;
    }
    norm_Delta_Gamma += SQR(Delta_Gr00) + SQR(Delta_Gr0r) + SQR(Delta_Grrr);
    if (!std::isfinite(norm_Delta_Gamma) || norm_Delta_Gamma < 0) {
      local_error_ = invalid_two_metric;
      return;
    }
    norm_Delta_Gamma = std::sqrt(norm_Delta_Gamma);
    if (!std::isfinite(norm_Delta_Gamma)) {
      local_error_ = invalid_two_metric;
      return;
    }

    // The background is now in the correct coordinates.
    // Store everything.

    Schwarzschild_radius = rsch;
    if (!std::isfinite(Schwarzschild_radius)) {
      local_error_ = invalid_two_metric;
      return;
    }
    Schwarzschild_mass = 0.5 * rsch * (1 - grr_uu);
    if (!std::isfinite(Schwarzschild_mass)) {
      local_error_ = invalid_two_metric;
      return;
    }

    g_dd(0, 0) = g00;
    if (!std::isfinite(g_dd(0, 0))) {
      local_error_ = invalid_two_metric;
      return;
    }
    g_dd(0, 1) = g0r;
    if (!std::isfinite(g_dd(0, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }
    g_dd(1, 1) = grr;
    if (!std::isfinite(g_dd(1, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }

    g_dr_dd(0, 0) = dr_g00;
    if (!std::isfinite(g_dr_dd(0, 0))) {
      local_error_ = invalid_two_metric;
      return;
    }
    g_dr_dd(0, 1) = dr_g0r;
    if (!std::isfinite(g_dr_dd(0, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }
    g_dr_dd(1, 1) = dr_grr;
    if (!std::isfinite(g_dr_dd(1, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }

    g_dr2_dd(0, 0) = dr2_g00;
    if (!std::isfinite(g_dr2_dd(0, 0))) {
      local_error_ = invalid_two_metric;
      return;
    }
    g_dr2_dd(0, 1) = dr2_g0r;
    if (!std::isfinite(g_dr2_dd(0, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }
    g_dr2_dd(1, 1) = dr2_grr;
    if (!std::isfinite(g_dr2_dd(1, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }

    g_dot_dd(0, 0) = dot_g00;
    if (!std::isfinite(g_dot_dd(0, 0))) {
      local_error_ = invalid_two_metric;
      return;
    }
    g_dot_dd(0, 1) = dot_g0r;
    if (!std::isfinite(g_dot_dd(0, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }
    g_dot_dd(1, 1) = dot_grr;
    if (!std::isfinite(g_dot_dd(1, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }

    g_uu(0, 0) = g00_uu;
    if (!std::isfinite(g_uu(0, 0))) {
      local_error_ = invalid_two_metric;
      return;
    }
    g_uu(0, 1) = g0r_uu;
    if (!std::isfinite(g_uu(0, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }
    g_uu(1, 1) = grr_uu;
    if (!std::isfinite(g_uu(1, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }

    g_dr_uu(0, 0) = dr_g00_uu;
    if (!std::isfinite(g_dr_uu(0, 0))) {
      local_error_ = invalid_two_metric;
      return;
    }
    g_dr_uu(0, 1) = dr_g0r_uu;
    if (!std::isfinite(g_dr_uu(0, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }
    g_dr_uu(1, 1) = dr_grr_uu;
    if (!std::isfinite(g_dr_uu(1, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }

    g_dr2_uu(0, 0) = dr2_g00_uu;
    if (!std::isfinite(g_dr2_uu(0, 0))) {
      local_error_ = invalid_two_metric;
      return;
    }
    g_dr2_uu(0, 1) = dr2_g0r_uu;
    if (!std::isfinite(g_dr2_uu(0, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }
    g_dr2_uu(1, 1) = dr2_grr_uu;
    if (!std::isfinite(g_dr2_uu(1, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }

    g_dot_uu(0, 0) = dot_g00_uu;  // TODO
    if (!std::isfinite(g_dot_uu(0, 0))) {
      local_error_ = invalid_two_metric;
      return;
    }
    g_dot_uu(0, 1) = dot_g0r_uu;
    if (!std::isfinite(g_dot_uu(0, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }
    g_dot_uu(1, 1) = dot_grr_uu;  // TODO
    if (!std::isfinite(g_dot_uu(1, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }

    Gamma_udd(0, 0, 0) = G000;
    if (!std::isfinite(Gamma_udd(0, 0, 0))) {
      local_error_ = invalid_two_metric;
      return;
    }
    Gamma_udd(0, 0, 1) = G00r;
    if (!std::isfinite(Gamma_udd(0, 0, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }
    Gamma_udd(0, 1, 1) = G0rr;
    if (!std::isfinite(Gamma_udd(0, 1, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }
    Gamma_udd(1, 0, 0) = Gr00;
    if (!std::isfinite(Gamma_udd(1, 0, 0))) {
      local_error_ = invalid_two_metric;
      return;
    }
    Gamma_udd(1, 0, 1) = Gr0r;
    if (!std::isfinite(Gamma_udd(1, 0, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }
    Gamma_udd(1, 1, 1) = Grrr;
    if (!std::isfinite(Gamma_udd(1, 1, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }

    Gamma_dyn_udd(0, 0, 0) = G000_dyn;
    if (!std::isfinite(Gamma_dyn_udd(0, 0, 0))) {
      local_error_ = invalid_two_metric;
      return;
    }
    Gamma_dyn_udd(0, 0, 1) = G00r_dyn;  // N.B. SYM2
    if (!std::isfinite(Gamma_dyn_udd(0, 0, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }
    Gamma_dyn_udd(0, 1, 1) = G0rr_dyn;
    if (!std::isfinite(Gamma_dyn_udd(0, 1, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }
    Gamma_dyn_udd(1, 0, 0) = Gr00_dyn;
    if (!std::isfinite(Gamma_dyn_udd(1, 0, 0))) {
      local_error_ = invalid_two_metric;
      return;
    }
    Gamma_dyn_udd(1, 0, 1) = Gr0r_dyn;  // N.B. SYM2
    if (!std::isfinite(Gamma_dyn_udd(1, 0, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }
    Gamma_dyn_udd(1, 1, 1) = Grrr_dyn;
    if (!std::isfinite(Gamma_dyn_udd(1, 1, 1))) {
      local_error_ = invalid_two_metric;
      return;
    }

    for (int a = 0; a < 6; ++a) {
      shell.Delta_Gamma.v[a] = shell.Gamma_udd.v[a] - shell.Gamma_dyn_udd.v[a];
      if (!std::isfinite(shell.Delta_Gamma.v[a])) {
        local_error_ = invalid_two_metric;
        return;
      }
    }
    jacobian_dot[is] = dri_drsch_dot;
    shell.dot_rsch = dot_rsch;
    shell.drsch_dri = drsch_dri;
    shell.d2rsch_dri2 = d2rsch_dri2;
    shell.dri_drsch = dri_drsch;
    shell.d2ri_drsch2 = d2ri_drsch2;
    shell.d3ri_drsch3 = d3ri_drsch3;
  }
  // Launch only after every shell's compact host background has been admitted.
  // One serial work item per logical sample retains symmetric off-diagonal
  // aliasing.
  for (std::size_t is = 0; is < shells_.size(); ++is) {
    const auto& shell = shells_[is];
    const auto gamma = shell.gamma;
    const auto beta_d = shell.beta_d;
    const auto alpha = shell.alpha;
    const auto beta2 = shell.beta2;
    const auto owner = shell.owner;
    const auto theta = shell.theta;
    const auto error = device_error_;
    const Real dri_drsch = shell.dri_drsch;
    const Real d2ri_drsch2 = shell.d2ri_drsch2;
    // Only this admitted scalar, never the host vector, enters the device
    // capture.
    const Real dri_drsch_dot = jacobian_dot[is];
    const Real gRR = host_background_(is, IgRR);
    const Real gtt = host_background_(is, Igtt);
    const Real dR_gRR = host_background_(is, IdR_gRR);
    const Real dR_gtt = host_background_(is, IdR_gtt);
    const bool subtract_background = subtract_background_;
    Kokkos::parallel_for(
        "rwz_radial_sample_transform",
        Kokkos::RangePolicy<DevExeSpace>(0, npoint_), KOKKOS_LAMBDA(int q) {
          if (owner(q) < 0) return;
          const Real sinth = Kokkos::sin(theta(q));
          if (!Kokkos::isfinite(sinth) || sinth <= 0) {
            Kokkos::atomic_max(&error(0), int(invalid_sample));
            return;
          }
          // Validate only the initialized slots consumed by
          // projection/transformation.
          bool finite_slots = true;
          for (int d = 0; d < NDerivative; ++d) {
            if (d == D00 || d == D10 || d == D20 || d == D01) {
              finite_slots = finite_slots && Kokkos::isfinite(alpha(q, d, 0)) &&
                             Kokkos::isfinite(beta2(q, d, 0));
            }
            for (int a = 0; a < 3; ++a) {
              if (d == D00 || d == D10 || d == D01 || (d == D20 && a > 0)) {
                finite_slots =
                    finite_slots && Kokkos::isfinite(beta_d(q, d, a));
              }
            }
            for (int a = 0; a < 6; ++a) {
              if (d == D00 || d == D10 || d == D01 || (d == D20 && a >= 3) ||
                  (d == D11 && a >= 1)) {
                finite_slots = finite_slots && Kokkos::isfinite(gamma(q, d, a));
              }
            }
          }
          if (!finite_slots) {
            Kokkos::atomic_max(&error(0), int(invalid_sample));
            return;
          }
          if (subtract_background) {
            // Exactly six changes, before radial conversion. Never re-lower the
            // shift.
            const Real sinth2 = SQR(sinth);
            gamma(q, D00, Sym(0, 0)) -= gRR;
            gamma(q, D00, Sym(1, 1)) -= gtt;
            gamma(q, D00, Sym(2, 2)) -= gtt * sinth2;
            gamma(q, D10, Sym(0, 0)) -= dR_gRR;
            gamma(q, D10, Sym(1, 1)) -= dR_gtt;
            gamma(q, D10, Sym(2, 2)) -= dR_gtt * sinth2;
          }
          // The three full angular loops deliberately update (theta,phi) TWICE.
          // alpha radial derivatives
          alpha(q, D20, 0) = d2ri_drsch2 * alpha(q, D10, 0) +
                             SQR(dri_drsch) * alpha(q, D20, 0);
          alpha(q, D10, 0) *= dri_drsch;

          // dr2_beta_th and dr2_beta_ph
          for (int a = 1; a < 3; a++)
            beta_d(q, D20, a) = d2ri_drsch2 * beta_d(q, D10, a) +
                                SQR(dri_drsch) * beta_d(q, D20, a);

          // dr_beta_i
          beta_d(q, D10, 0) = d2ri_drsch2 * beta_d(q, D00, 0) +
                              SQR(dri_drsch) * beta_d(q, D10, 0);
          for (int a = 1; a < 3; a++) beta_d(q, D10, a) *= dri_drsch;

          // dot_beta_r
          beta_d(q, D01, 0) =
              dri_drsch_dot * beta_d(q, D00, 0) + dri_drsch * beta_d(q, D01, 0);

          // beta_r
          beta_d(q, D00, 0) *= dri_drsch;

          // dr2_gamma_thth, dr2_gamma_thph, dr2_gamma_phph
          for (int a = 1; a < 3; a++)
            for (int b = 1; b < 3; b++) {
              gamma(q, D20, Sym(a, b)) =
                  d2ri_drsch2 * gamma(q, D10, Sym(a, b)) +
                  SQR(dri_drsch) * gamma(q, D20, Sym(a, b));
            }

          // dr_dot_gamma_thth, dr_dot_gamma_thph, dr_dot_gamma_phph
          for (int a = 1; a < 3; a++)
            for (int b = 1; b < 3; b++) {
              gamma(q, D11, Sym(a, b)) *= dri_drsch;
            }

          // dr_dot_gamma_rth, dr_dot_gamma_rph
          for (int a = 1; a < 3; a++)
            gamma(q, D11, Sym(0, a)) =
                dri_drsch_dot * dri_drsch * gamma(q, D10, Sym(0, a)) +
                SQR(dri_drsch) * gamma(q, D11, Sym(0, a)) +
                d2ri_drsch2 *
                    gamma(q, D01, Sym(0, a));  // Lacks term dr_dri_drsch_dot

          // dr_gamma_ij
          gamma(q, D10, Sym(0, 0)) =
              2.0 * dri_drsch * d2ri_drsch2 * gamma(q, D00, Sym(0, 0)) +
              Kokkos::pow(dri_drsch, 3) * gamma(q, D10, Sym(0, 0));
          for (int a = 1; a < 3; a++)
            gamma(q, D10, Sym(0, a)) =
                d2ri_drsch2 * gamma(q, D00, Sym(0, a)) +
                SQR(dri_drsch) * gamma(q, D10, Sym(0, a));
          for (int a = 1; a < 3; a++)
            for (int b = 1; b < 3; b++) {
              gamma(q, D10, Sym(a, b)) *= dri_drsch;
            }

          // dot_gamma_ri
          gamma(q, D01, Sym(0, 0)) =
              2.0 * dri_drsch * dri_drsch_dot * gamma(q, D00, Sym(0, 0)) +
              SQR(dri_drsch) * gamma(q, D01, Sym(0, 0));

          for (int a = 1; a < 3; a++)
            gamma(q, D01, Sym(0, a)) =
                dri_drsch_dot * gamma(q, D00, Sym(0, a)) +
                dri_drsch * gamma(q, D01, Sym(0, a));

          // gamma_ri
          gamma(q, D00, Sym(0, 0)) *= SQR(dri_drsch);
          for (int a = 1; a < 3; a++) gamma(q, D00, Sym(0, a)) *= dri_drsch;

          // beta2 is a scalar, so only derivatives transform
          beta2(q, D20, 0) = d2ri_drsch2 * beta2(q, D10, 0) +
                             SQR(dri_drsch) * beta2(q, D20, 0);
          beta2(q, D10, 0) *= dri_drsch;
          // The subtracted metric is a perturbation: do not impose positive
          // area. All downstream-consumed transformed slots must nevertheless
          // be finite.
          finite_slots = true;
          for (int d = 0; d < NDerivative; ++d) {
            if (d == D00 || d == D10 || d == D20 || d == D01) {
              finite_slots = finite_slots && Kokkos::isfinite(alpha(q, d, 0)) &&
                             Kokkos::isfinite(beta2(q, d, 0));
            }
            for (int a = 0; a < 3; ++a) {
              if (d == D00 || d == D10 || d == D01 || (d == D20 && a > 0)) {
                finite_slots =
                    finite_slots && Kokkos::isfinite(beta_d(q, d, a));
              }
            }
            for (int a = 0; a < 6; ++a) {
              if (d == D00 || d == D10 || d == D01 || (d == D20 && a >= 3) ||
                  (d == D11 && a >= 1)) {
                finite_slots = finite_slots && Kokkos::isfinite(gamma(q, d, a));
              }
            }
          }
          if (!finite_slots) {
            Kokkos::atomic_max(&error(0), int(invalid_sample));
            return;
          }
        });
  }
  Kokkos::fence();
}

void WaveExtractRWZ::FinalizeMultipoles() {
  // Background/projection remain distributed; only the writer uses final modes.
  if (global_variable::my_rank != mpi_root_) return;
  if (lmax_ < 2 || lmax_ > 8 || nmode_ != (lmax_ + 1) * (lmax_ + 1) - 4 ||
      host_multipoles_.extent(0) != shells_.size() ||
      host_multipoles_.extent(1) != NMultipoles ||
      host_multipoles_.extent(2) != static_cast<std::size_t>(nmode_) ||
      host_multipoles_.extent(3) != 2) {
    local_error_ = invalid_configuration;
    return;
  }
  for (std::size_t is = 0; is < shells_.size(); ++is) {
    auto& shell = shells_[is];
    // These independently allocated HostSpace Views are contiguous. Clear the
    // complete allocations, including unused slots, without deep_copy's global
    // device fences. The reduced-multipole phase has already synchronized data.
    std::array<ModeField*, 10> fields = {
        &shell.h00, &shell.h01, &shell.h11, &shell.h0, &shell.h1,
        &shell.G,   &shell.K,   &shell.H0,  &shell.H1, &shell.H};
    for (auto* field : fields) {
      if (field->data.extent(0) != NDerivative ||
          field->data.extent(1) < static_cast<std::size_t>(nmode_) ||
          field->data.extent(2) != 2) {
        local_error_ = invalid_configuration;
        return;
      }
      std::fill_n(field->data.data(), field->data.size(), Real(0));
    }
    std::array<HostArray2D<Real>, 14> scalar_outputs = {
        shell.Psie,     shell.Psio,       shell.Psie_dyn, shell.Psio_dyn,
        shell.Psie_sch, shell.Psio_sch,   shell.Qplus,    shell.Qstar,
        shell.Psie_dr,  shell.Psio_dr,    shell.Qplus_dr, shell.Qstar_dr,
        shell.kappa,    shell.Tr_kappa_dd};
    for (auto field : scalar_outputs) {
      if (field.extent(0) < static_cast<std::size_t>(nmode_) ||
          field.extent(1) != 2) {
        local_error_ = invalid_configuration;
        return;
      }
      std::fill_n(field.data(), field.size(), Real(0));
    }
    if (shell.kappa_dd.data.extent(0) != 3 ||
        shell.kappa_dd.data.extent(1) < static_cast<std::size_t>(nmode_) ||
        shell.kappa_dd.data.extent(2) != 2 || shell.kappa_d.extent(0) != 2 ||
        shell.kappa_d.extent(1) < static_cast<std::size_t>(nmode_) ||
        shell.kappa_d.extent(2) != 2) {
      local_error_ = invalid_configuration;
      return;
    }
    std::fill_n(shell.kappa_dd.data.data(), shell.kappa_dd.data.size(), Real(0));
    std::fill_n(shell.kappa_d.data(), shell.kappa_d.size(), Real(0));
    shell.norm_Tr_kappa_dd = 0.0;
    auto& h00 = shell.h00;
    auto& h01 = shell.h01;
    auto& h11 = shell.h11;
    auto& h0 = shell.h0;
    auto& h1 = shell.h1;
    auto& G = shell.G;
    auto& K = shell.K;
    auto& H0 = shell.H0;
    auto& H1 = shell.H1;
    auto& H = shell.H;
    auto& Psie = shell.Psie;
    auto& Psio = shell.Psio;
    auto& Psie_dyn = shell.Psie_dyn;
    auto& Psio_dyn = shell.Psio_dyn;
    auto& Psie_sch = shell.Psie_sch;
    auto& Psio_sch = shell.Psio_sch;
    auto& Qplus = shell.Qplus;
    auto& Qstar = shell.Qstar;
    auto& Psie_dr = shell.Psie_dr;
    auto& Psio_dr = shell.Psio_dr;
    auto& Qplus_dr = shell.Qplus_dr;
    auto& Qstar_dr = shell.Qstar_dr;
    auto& kappa_dd = shell.kappa_dd;
    auto& kappa_d = shell.kappa_d;
    auto& kappa = shell.kappa;
    auto& Tr_kappa_dd = shell.Tr_kappa_dd;
    auto& norm_Tr_kappa_dd = shell.norm_Tr_kappa_dd;
    auto& g_dd = shell.g_dd;
    auto& g_uu = shell.g_uu;
    auto& g_dr_dd = shell.g_dr_dd;
    auto& g_dr_uu = shell.g_dr_uu;
    auto& g_dr2_dd = shell.g_dr2_dd;
    auto& g_dr2_uu = shell.g_dr2_uu;
    auto& g_dot_dd = shell.g_dot_dd;
    auto& g_dot_uu = shell.g_dot_uu;
    auto& Gamma_dyn_udd = shell.Gamma_dyn_udd;
    auto unpack = [&](ModeField& dest, int derivative, int slot) {
      for (int lm = 0; lm < nmode_; ++lm) {
        for (int c = 0; c < 2; ++c) {
          const Real value = host_multipoles_(is, slot, lm, c);
          if (!std::isfinite(value)) {
            local_error_ = invalid_projection;
            return false;
          }
          dest(derivative, lm, c) = value;
        }
      }
      return true;
    };
    // Plain copies: projection supplied all r/lambda/product-rule prefactors.
    // There is deliberately no 1/(4*pi) multipole normalization here.
    // even scalar
    if (!unpack(h00, D00, Ih00)) return;
    if (!unpack(h01, D00, Ih01)) return;
    if (!unpack(h11, D00, Ih11)) return;
    if (!unpack(h00, D10, Ih00_dr)) return;
    if (!unpack(h01, D10, Ih01_dr)) return;
    if (!unpack(h11, D10, Ih11_dr)) return;
    if (!unpack(h00, D01, Ih00_dot)) return;
    if (!unpack(h01, D01, Ih01_dot)) return;
    if (!unpack(h11, D01, Ih11_dot)) return;

    // even vector
    if (!unpack(h0, D00, Ih0)) return;
    if (!unpack(h1, D00, Ih1)) return;
    if (!unpack(h0, D10, Ih0_dr)) return;
    if (!unpack(h1, D10, Ih1_dr)) return;
    if (!unpack(h0, D01, Ih0_dot)) return;
    if (!unpack(h1, D01, Ih1_dot)) return;

    // even tensor + trace
    if (!unpack(G, D00, IG)) return;
    if (!unpack(K, D00, IK)) return;
    if (!unpack(G, D10, IG_dr)) return;
    if (!unpack(K, D10, IK_dr)) return;
    if (!unpack(G, D01, IG_dot)) return;
    if (!unpack(K, D01, IK_dot)) return;
    if (!unpack(G, D20, IG_dr2)) return;
    if (!unpack(K, D20, IK_dr2)) return;
    if (!unpack(G, D11, IG_dr_dot)) return;
    if (!unpack(K, D11, IK_dr_dot)) return;

    // odd vector
    if (!unpack(H0, D00, IH0)) return;
    if (!unpack(H0, D10, IH0_dr)) return;
    if (!unpack(H0, D20, IH0_dr2)) return;
    if (!unpack(H0, D01, IH0_dot)) return;
    if (!unpack(H1, D00, IH1)) return;
    if (!unpack(H1, D10, IH1_dr)) return;
    if (!unpack(H1, D01, IH1_dot)) return;
    if (!unpack(H1, D11, IH1_dr_dot)) return;

    // odd tensor
    if (!unpack(H, D00, IH)) return;
    if (!unpack(H, D10, IH_dr)) return;
    if (!unpack(H, D20, IH_dr2)) return;
    if (!unpack(H, D01, IH_dot)) return;

    // Background producer already enforced its original r^2 > 1e-20 gate.
    // Recheck every scalar and independent tensor entry used by these stages.
    const Real rsch = shell.Schwarzschild_radius;
    const Real Schwarzschild_radius = shell.Schwarzschild_radius;
    const Real Schwarzschild_mass = shell.Schwarzschild_mass;
    const Real dot_rsch = shell.dot_rsch;
    if (!std::isfinite(rsch) || rsch <= 0 ||
        !std::isfinite(Schwarzschild_mass) || !std::isfinite(dot_rsch)) {
      local_error_ = invalid_background;
      return;
    }
    const Real checked_r2 = SQR(rsch);
    if (!std::isfinite(checked_r2) || checked_r2 <= 0) {
      local_error_ = invalid_background;
      return;
    }
    if (!std::isfinite(1.0 / rsch) || !std::isfinite(1.0 / checked_r2) ||
        !std::isfinite(SQR(1.0 / rsch))) {
      local_error_ = invalid_background;
      return;
    }
    const std::array<Sym2, 8> background_tensors = {
        g_dd, g_uu, g_dr_dd, g_dr_uu, g_dr2_dd, g_dr2_uu, g_dot_dd, g_dot_uu};
    for (const auto& tensor : background_tensors) {
      for (int a = 0; a < 3; ++a) {
        if (!std::isfinite(tensor.v[a])) {
          local_error_ = invalid_two_metric;
          return;
        }
      }
    }
    for (int a = 0; a < 6; ++a) {
      if (!std::isfinite(Gamma_dyn_udd.v[a])) {
        local_error_ = invalid_two_metric;
        return;
      }
    }
    const Real detg = g_dd(0, 0) * g_dd(1, 1) - SQR(g_dd(0, 1));
    constexpr Real eps_det = 1e-12;
    if (!std::isfinite(detg) || detg >= 0 || std::fabs(detg) < eps_det) {
      local_error_ = invalid_two_metric;
      return;
    }
    if (g_uu(1, 1) == 0) {
      // Retained dr_coef_G_dr division is not removed by cancellation at a
      // horizon.
      local_error_ = singular_master_coefficient;
      return;
    }
    if (!std::isfinite(g_dr_uu(1, 1) / g_uu(1, 1))) {
      local_error_ = singular_master_coefficient;
      return;
    }

    {  // Gauge invariants first, before the master-function stage.
      const Real r = rsch;
      if (!std::isfinite(r)) {
        local_error_ = invalid_projection;
        return;
      }
      const Real r2 = SQR(rsch);
      if (!std::isfinite(r2)) {
        local_error_ = invalid_projection;
        return;
      }
      const Real rdot = dot_rsch;
      if (!std::isfinite(rdot)) {
        local_error_ = invalid_projection;
        return;
      }
      const Real div_r = 1.0 / rsch;
      if (!std::isfinite(div_r)) {
        local_error_ = invalid_projection;
        return;
      }

      // TODO This is a placeholder, these second drvt of the G multipole are
      // not computed ATM!
      const Real G_dt2 = 0.0;
      if (!std::isfinite(G_dt2)) {
        local_error_ = invalid_projection;
        return;
      }

      norm_Tr_kappa_dd = 0.0;
      if (!std::isfinite(norm_Tr_kappa_dd)) {
        local_error_ = invalid_projection;
        return;
      }

      for (int l = 2; l <= lmax_; ++l) {
        for (int m = -l; m <= l; ++m) {
          const int lm = l * l + l + m - 4;
          for (int c = 0; c < 2; ++c) {
            // even-parity
            // -----------------------------------------

            // gauge-invariant tensor kappa_{AB}
            // NOTE: G_dt2 is a placeholder (set to 0), so kappa_dd(0,0,...)
            // is approximate. See TODO above.

            kappa_dd(0, 0, lm, c) = h00(lm, c);
            if (!std::isfinite(kappa_dd(0, 0, lm, c))) {
              local_error_ = invalid_projection;
              return;
            }
            kappa_dd(0, 0, lm, c) += -2.0 * h0(D01, lm, c) +
                                     2.0 * (Gamma_dyn_udd(0, 0, 0) * h0(lm, c) +
                                            Gamma_dyn_udd(1, 0, 0) * h1(lm, c));
            if (!std::isfinite(kappa_dd(0, 0, lm, c))) {
              local_error_ = invalid_projection;
              return;
            }
            kappa_dd(0, 0, lm, c) +=
                2.0 * r * rdot * G(D01, lm, c) +
                r2 * (G_dt2 - Gamma_dyn_udd(0, 0, 0) * G(D01, lm, c) -
                      Gamma_dyn_udd(1, 0, 0) * G(D10, lm, c));
            if (!std::isfinite(kappa_dd(0, 0, lm, c))) {
              local_error_ = invalid_projection;
              return;
            }

            kappa_dd(0, 1, lm, c) = h01(lm, c);
            if (!std::isfinite(kappa_dd(0, 1, lm, c))) {
              local_error_ = invalid_projection;
              return;
            }
            kappa_dd(0, 1, lm, c) += -h1(D01, lm, c) - h0(D10, lm, c) +
                                     2.0 * (Gamma_dyn_udd(0, 0, 1) * h0(lm, c) +
                                            Gamma_dyn_udd(1, 0, 1) * h1(lm, c));
            if (!std::isfinite(kappa_dd(0, 1, lm, c))) {
              local_error_ = invalid_projection;
              return;
            }
            kappa_dd(0, 1, lm, c) +=
                r * rdot * G(D10, lm, c) + r * G(D01, lm, c) +
                r2 * (G(D11, lm, c) - Gamma_dyn_udd(0, 0, 1) * G(D01, lm, c) -
                      Gamma_dyn_udd(1, 0, 1) * G(D10, lm, c));
            if (!std::isfinite(kappa_dd(0, 1, lm, c))) {
              local_error_ = invalid_projection;
              return;
            }

            kappa_dd(1, 1, lm, c) = h11(lm, c);
            if (!std::isfinite(kappa_dd(1, 1, lm, c))) {
              local_error_ = invalid_projection;
              return;
            }
            kappa_dd(1, 1, lm, c) += -2.0 * h1(D10, lm, c) +
                                     2.0 * (Gamma_dyn_udd(0, 1, 1) * h0(lm, c) +
                                            Gamma_dyn_udd(1, 1, 1) * h1(lm, c));
            if (!std::isfinite(kappa_dd(1, 1, lm, c))) {
              local_error_ = invalid_projection;
              return;
            }
            kappa_dd(1, 1, lm, c) +=
                2.0 * r * G(D10, lm, c) +
                r2 * (G(D20, lm, c) - Gamma_dyn_udd(0, 1, 1) * G(D01, lm, c) -
                      Gamma_dyn_udd(1, 1, 1) * G(D10, lm, c));
            if (!std::isfinite(kappa_dd(1, 1, lm, c))) {
              local_error_ = invalid_projection;
              return;
            }

            // gauge-invariant scalar kappa
            kappa(lm, c) = K(lm, c);
            if (!std::isfinite(kappa(lm, c))) {
              local_error_ = invalid_projection;
              return;
            }
            kappa(lm, c) -=
                (g_uu(0, 0) * rdot * (2.0 * h0(lm, c) - r2 * G(D01, lm, c)) +
                 g_uu(0, 1) * (rdot * (2.0 * h1(lm, c) - r2 * G(D10, lm, c)) +
                               2.0 * h0(lm, c) - r2 * G(D01, lm, c)) +
                 g_uu(1, 1) * (2.0 * h1(lm, c) - r2 * G(D10, lm, c))) *
                div_r;
            if (!std::isfinite(kappa(lm, c))) {
              local_error_ = invalid_projection;
              return;
            }

            // odd-parity
            // -----------------------------------------

            // BD: verify
            //     kappa_A = H_A - 1/2 \nabla_A H + H (\nabla_A r)/r
            // ??: kappa_d(0, lm, c) = H0 - H(D01) + H * 2.0 * rdot * div_r
            // ->: kappa_d(0, lm, c) = H0 - 0.5 * H(D01) + H * rdot * div_r
            // ??: kappa_d(1, lm, c) = H1 - H(D10) + H * 2.0 * div_r
            // ->: kappa_d(1, lm, c) = H1 - 0.5 * H(D10) + H * div_r
            kappa_d(0, lm, c) =
                H0(lm, c) - 0.5 * H(D01, lm, c) + H(lm, c) * rdot * div_r;
            if (!std::isfinite(kappa_d(0, lm, c))) {
              local_error_ = invalid_projection;
              return;
            }
            kappa_d(1, lm, c) =
                H1(lm, c) - 0.5 * H(D10, lm, c) + H(lm, c) * div_r;
            if (!std::isfinite(kappa_d(1, lm, c))) {
              local_error_ = invalid_projection;
              return;
            }

            // Trace constraint: Tr(kappa_AB) = g^{AB} kappa_{AB}
            Tr_kappa_dd(lm, c) = g_uu(0, 0) * kappa_dd(0, 0, lm, c) +
                                 2.0 * g_uu(0, 1) * kappa_dd(0, 1, lm, c) +
                                 g_uu(1, 1) * kappa_dd(1, 1, lm, c);
            if (!std::isfinite(Tr_kappa_dd(lm, c))) {
              local_error_ = invalid_projection;
              return;
            }
            // G_dt2=0 also makes this trace/norm an approximate constraint
            // diagnostic.
            const Real trace_square = SQR(Tr_kappa_dd(lm, c));
            if (!std::isfinite(trace_square) || trace_square < 0) {
              local_error_ = invalid_projection;
              return;
            }
            norm_Tr_kappa_dd += trace_square;
            if (!std::isfinite(norm_Tr_kappa_dd)) {
              local_error_ = invalid_projection;
              return;
            }

          }  // real&imag

        }  // for m
      }  // for l

      if (!std::isfinite(norm_Tr_kappa_dd) || norm_Tr_kappa_dd < 0) {
        local_error_ = invalid_projection;
        return;
      }
      norm_Tr_kappa_dd = std::sqrt(norm_Tr_kappa_dd);
      if (!std::isfinite(norm_Tr_kappa_dd)) {
        local_error_ = invalid_projection;
        return;
      }
    }
    {  // Source static/dynamic master expressions and four radial families.
      using std::pow;
      const Real M = Schwarzschild_mass;
      if (!std::isfinite(M)) {
        local_error_ = singular_master_coefficient;
        return;
      }
      const Real r = Schwarzschild_radius;
      if (!std::isfinite(r)) {
        local_error_ = singular_master_coefficient;
        return;
      }
      const Real r2 = SQR(r);
      if (!std::isfinite(r2)) {
        local_error_ = singular_master_coefficient;
        return;
      }
      const Real rdot = dot_rsch;
      if (!std::isfinite(rdot)) {
        local_error_ = singular_master_coefficient;
        return;
      }
      const Real div_r = 1.0 / r;
      if (!std::isfinite(div_r)) {
        local_error_ = singular_master_coefficient;
        return;
      }

      const Real S = (1.0 - 2.0 * M * div_r);
      if (!std::isfinite(S)) {
        local_error_ = singular_master_coefficient;
        return;
      }
      const Real abs_detg =
          std::fabs(g_dd(0, 0) * g_dd(1, 1) - SQR(g_dd(0, 1)));
      if (!std::isfinite(abs_detg) || abs_detg <= 0) {
        local_error_ = invalid_two_metric;
        return;
      }
      const Real sqrt_abs_detg = std::sqrt(abs_detg);
      if (!std::isfinite(sqrt_abs_detg) || sqrt_abs_detg <= 0) {
        local_error_ = invalid_two_metric;
        return;
      }
      const Real div_sqrtdetg = 1.0 / (std::sqrt(abs_detg));
      if (!std::isfinite(div_sqrtdetg)) {
        local_error_ = singular_master_coefficient;
        return;
      }
      if (!std::isfinite(std::pow(div_sqrtdetg, 3))) {
        local_error_ = singular_master_coefficient;
        return;
      }

      for (int l = 2; l <= lmax_; ++l) {
        const Real lambda = l * (l + 1);
        if (!std::isfinite(lambda) || lambda == 0 || lambda - 2.0 == 0) {
          local_error_ = singular_master_coefficient;
          return;
        }
        const Real even_denominator = (lambda - 2) * r + 6.0 * M;
        if (!std::isfinite(even_denominator) || even_denominator == 0) {
          local_error_ = singular_master_coefficient;
          return;
        }
        const Real div_lambda_2 = 1.0 / (lambda - 2.0);
        if (!std::isfinite(div_lambda_2)) {
          local_error_ = singular_master_coefficient;
          return;
        }
        const Real fac_Psie = (2.0 * r) / ((lambda - 2) * r + 6.0 * M);
        if (!std::isfinite(fac_Psie)) {
          local_error_ = singular_master_coefficient;
          return;
        }
        const Real r_div_lambda = r / lambda;
        if (!std::isfinite(r_div_lambda)) {
          local_error_ = singular_master_coefficient;
          return;
        }
        const Real Lam = (l - 1) * (l + 2) + 6 * M * div_r;
        if (!std::isfinite(Lam) || Lam == 0) {
          local_error_ = singular_master_coefficient;
          return;
        }
        const Real even_derivative_denominator = r2 * Lam;
        if (!std::isfinite(even_derivative_denominator) ||
            even_derivative_denominator == 0) {
          local_error_ = singular_master_coefficient;
          return;
        }
        const Real qplus_radicand = 2 * (l - 1) * (l + 2) / lambda;
        if (!std::isfinite(qplus_radicand) || qplus_radicand < 0) {
          local_error_ = singular_master_coefficient;
          return;
        }
        const Real sch_even_radicand = 2.0 * lambda * (lambda - 2.0);
        if (!std::isfinite(sch_even_radicand) || sch_even_radicand <= 0) {
          local_error_ = singular_master_coefficient;
          return;
        }
        const Real sch_even_root = std::sqrt(sch_even_radicand);
        if (!std::isfinite(sch_even_root) || sch_even_root <= 0) {
          local_error_ = singular_master_coefficient;
          return;
        }
        const Real Qpnorm = std::sqrt(2 * (l - 1) * (l + 2) / lambda) / Lam;
        if (!std::isfinite(Qpnorm)) {
          local_error_ = singular_master_coefficient;
          return;
        }
        const Real dr_fac_Psie = fac_Psie * 6.0 * M / (r2 * Lam);
        if (!std::isfinite(dr_fac_Psie)) {
          local_error_ = singular_master_coefficient;
          return;
        }

        for (int m = -l; m <= l; ++m) {
          const int lm = l * l + l + m - 4;
          for (int c = 0; c < 2; ++c) {
            // Even parity in Schwarschild coordinates
            // -----------------------------------------

            const Real Qplus_ =
                Qpnorm * (lambda * S * (r2 * G(D10, lm, c) - 2 * h1(lm, c)) +
                          2.0 * r * S * (S * h11(lm, c) - r * K(D10, lm, c)) +
                          r * Lam * K(lm, c));
            if (!std::isfinite(Qplus_)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Psie_sch(lm, c) =
                Qplus_ / (std::sqrt(2.0 * lambda * (lambda - 2.0)));
            if (!std::isfinite(Psie_sch(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }
            Qplus(lm, c) = Qplus_;
            if (!std::isfinite(Qplus(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Qplus_dr(lm, c) =
                6.0 * M * SQR(div_r) * Qplus(lm, c) / Lam +
                Qpnorm * (lambda * 2.0 * M * SQR(div_r) *
                              (r2 * G(D10, lm, c) - 2.0 * h1(lm, c)) +
                          lambda * S *
                              (2.0 * r * G(D10, lm, c) + r2 * G(D20, lm, c) -
                               2.0 * h1(D10, lm, c)) +
                          2.0 * (S * h11(lm, c) - r * K(D10, lm, c)) +
                          2.0 * r * S *
                              (2.0 * M * SQR(div_r) * h11(lm, c) +
                               S * h11(D10, lm, c) - K(D10, lm, c) -
                               r * K(D20, lm, c)) +
                          (lambda - 2.0) * K(lm, c) + r * Lam * K(D10, lm, c));
            if (!std::isfinite(Qplus_dr(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }

            // Odd parity in Schwarzschild coordinates
            // -----------------------------------------

            const Real Psio_sch_ =
                r *
                (H1(D01, lm, c) - H0(D10, lm, c) + 2.0 * div_r * H0(lm, c)) *
                div_lambda_2;
            if (!std::isfinite(Psio_sch_)) {
              local_error_ = singular_master_coefficient;
              return;
            }
            Psio_sch(lm, c) = Psio_sch_;
            if (!std::isfinite(Psio_sch(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }

            const Real Qstar_ =
                div_r * S *
                (H1(lm, c) - 0.5 * H(D10, lm, c) + H(lm, c) * div_r);
            if (!std::isfinite(Qstar_)) {
              local_error_ = singular_master_coefficient;
              return;
            }
            Qstar(lm, c) = Qstar_;
            if (!std::isfinite(Qstar(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Qstar_dr(lm, c) =
                SQR(div_r) * (1.0 - 2.0 * S) *
                    (H1(lm, c) - 0.5 * H(D10, lm, c) + H(lm, c) * div_r) +
                div_r * S *
                    (H1(D10, lm, c) - 0.5 * H(D20, lm, c) +
                     div_r * H(D10, lm, c) - SQR(div_r) * H(lm, c));
            if (!std::isfinite(Qstar_dr(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }

            // Even parity in general coordinates (static)
            // -----------------------------------------

            const Real term1_K = K(lm, c);
            if (!std::isfinite(term1_K)) {
              local_error_ = singular_master_coefficient;
              return;
            }
            const Real term1_hG =
                (-2.0 * div_r) *
                (g_uu(1, 1) * (h1(lm, c) - 0.5 * r2 * G(D10, lm, c)) +
                 g_uu(0, 1) * (h0(lm, c) - 0.5 * r2 * G(D01, lm, c)));
            if (!std::isfinite(term1_hG)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            const Real term_hAB = SQR(g_uu(0, 1)) * h00(lm, c) +
                                  2.0 * g_uu(0, 1) * g_uu(1, 1) * h01(lm, c) +
                                  SQR(g_uu(1, 1)) * h11(lm, c);
            if (!std::isfinite(term_hAB)) {
              local_error_ = singular_master_coefficient;
              return;
            }
            const Real term2_K =
                -r * (g_uu(1, 1) * K(D10, lm, c) + g_uu(0, 1) * K(D01, lm, c));
            if (!std::isfinite(term2_K)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Real coef_h0 =
                -r * std::pow(g_uu(0, 1), 3) * g_dr_dd(0, 0) +
                2.0 * r * g_uu(1, 1) *
                    (g_uu(0, 0) * g_uu(1, 1) * g_dr_dd(0, 1) + g_dr_uu(0, 1)) +
                g_uu(0, 1) * g_uu(1, 1) *
                    (-2.0 + 2.0 * r * g_uu(0, 0) * g_dr_dd(0, 0) +
                     r * g_uu(1, 1) * g_dr_dd(1, 1));
            if (!std::isfinite(coef_h0)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            coef_h0 *= div_r;
            if (!std::isfinite(coef_h0)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Real coef_h1 = 2.0 * SQR(g_uu(1, 1)) *
                               (-1.0 + r * g_uu(0, 1) * g_dr_dd(0, 1)) +
                           r * std::pow(g_uu(1, 1), 3) * g_dr_dd(1, 1) +
                           r * g_uu(1, 1) * SQR(g_uu(0, 1)) * g_dr_dd(0, 0) +
                           2.0 * r * g_uu(1, 1) * g_dr_uu(1, 1);
            if (!std::isfinite(coef_h1)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            coef_h1 *= div_r;
            if (!std::isfinite(coef_h1)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Real coef_G_dr =
                2.0 * g_uu(1, 1) * (-1.0 + r * g_uu(0, 1) * g_dr_dd(0, 1)) +
                r * SQR(g_uu(1, 1)) * g_dr_dd(1, 1) +
                r * SQR(g_uu(0, 1)) * g_dr_dd(0, 0) + 2.0 * r * g_dr_uu(1, 1);
            if (!std::isfinite(coef_G_dr)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            coef_G_dr *= -0.5 * r * g_uu(1, 1);
            if (!std::isfinite(coef_G_dr)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Real coef_G_dot =
                r * std::pow(g_uu(0, 1), 3) * g_dr_dd(0, 0) -
                2 * r * g_uu(1, 1) *
                    (g_uu(0, 0) * g_uu(1, 1) * g_dr_dd(0, 1) + g_dr_uu(0, 1)) -
                g_uu(0, 1) * g_uu(1, 1) *
                    (-2.0 + 2.0 * r * g_uu(0, 0) * g_dr_dd(0, 0) +
                     r * g_uu(1, 1) * g_dr_dd(1, 1));
            if (!std::isfinite(coef_G_dot)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            coef_G_dot *= 0.5 * r;
            if (!std::isfinite(coef_G_dot)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Psie(lm, c) = term1_K;
            if (!std::isfinite(Psie(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }
            Psie(lm, c) += term1_hG;
            if (!std::isfinite(Psie(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }
            Psie(lm, c) +=
                fac_Psie * (term_hAB + coef_h0 * h0(lm, c) +
                            coef_h1 * h1(lm, c) + coef_G_dr * G(D10, lm, c) +
                            coef_G_dot * G(D01, lm, c) + term2_K);
            if (!std::isfinite(Psie(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }
            Psie(lm, c) *= r_div_lambda;
            if (!std::isfinite(Psie(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }

            const Real dr_term1_hG =
                -div_r * term1_hG -
                2.0 * div_r *
                    (g_dr_uu(1, 1) * (h1(lm, c) - 0.5 * r2 * G(D10, lm, c)) +
                     g_dr_uu(0, 1) * (h0(lm, c) - 0.5 * r2 * G(D01, lm, c)) +
                     g_uu(1, 1) * (h1(D10, lm, c) - r * G(D10, lm, c) -
                                   0.5 * r2 * G(D20, lm, c)) +
                     g_uu(0, 1) * (h0(D10, lm, c) - r * G(D01, lm, c) -
                                   0.5 * r2 * G(D11, lm, c)));
            if (!std::isfinite(dr_term1_hG)) {
              local_error_ = singular_master_coefficient;
              return;
            }
            const Real dr_term_hAB =
                SQR(g_uu(0, 1)) * h00(D10, lm, c) +
                2.0 * g_uu(0, 1) * g_dr_uu(0, 1) * h00(lm, c) +
                2.0 * g_uu(0, 1) * g_uu(1, 1) * h01(D10, lm, c) +
                2.0 *
                    (g_dr_uu(0, 1) * g_uu(1, 1) + g_uu(0, 1) * g_dr_uu(1, 1)) *
                    h01(lm, c) +
                SQR(g_uu(1, 1)) * h11(D10, lm, c) +
                2.0 * g_uu(1, 1) * g_dr_uu(1, 1) * h11(lm, c);
            if (!std::isfinite(dr_term_hAB)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            const Real dr_coef_h0 =
                -std::pow(g_uu(0, 1), 3) * g_dr2_dd(0, 0) -
                3.0 * SQR(g_uu(0, 1)) * g_dr_uu(0, 1) * g_dr_dd(0, 0) +
                2.0 * g_dr_uu(1, 1) *
                    (g_uu(0, 0) * g_uu(1, 1) * g_dr_dd(0, 1) + g_dr_uu(0, 1)) +
                2.0 * g_uu(1, 1) *
                    ((g_uu(0, 0) * g_dr_uu(1, 1) + g_dr_uu(0, 0) * g_uu(1, 1)) *
                         g_dr_dd(0, 1) +
                     g_uu(0, 0) * g_uu(1, 1) * g_dr2_dd(0, 1) +
                     g_dr2_uu(0, 1)) +
                (g_dr_uu(0, 1) * g_uu(1, 1) + g_uu(0, 1) * g_dr_uu(1, 1)) *
                    (-2.0 / r + 2.0 * g_uu(0, 0) * g_dr_dd(0, 0) +
                     g_uu(1, 1) * g_dr_dd(1, 1)) +
                g_uu(0, 1) * g_uu(1, 1) *
                    (2.0 / r2 + 2.0 * g_dr_uu(0, 0) * g_dr_dd(0, 0) +
                     2.0 * g_uu(0, 0) * g_dr2_dd(0, 0) +
                     g_uu(1, 1) * g_dr2_dd(1, 1) +
                     g_dr_uu(1, 1) * g_dr_dd(1, 1));
            if (!std::isfinite(dr_coef_h0)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            const Real dr_coef_h1 =
                4.0 * g_uu(1, 1) * g_dr_uu(1, 1) *
                    (-div_r + g_uu(0, 1) * g_dr_dd(0, 1)) +
                2.0 * SQR(g_uu(1, 1)) *
                    (1.0 / r2 + g_dr_uu(0, 1) * g_dr_dd(0, 1) +
                     g_uu(0, 1) * g_dr2_dd(0, 1)) +
                3.0 * SQR(g_uu(1, 1)) * g_dr_uu(1, 1) * g_dr_dd(1, 1) +
                std::pow(g_uu(1, 1), 3) * g_dr2_dd(1, 1) +
                g_dr_uu(1, 1) * SQR(g_uu(0, 1)) * g_dr_dd(0, 0) +
                g_uu(1, 1) * (2.0 * g_uu(0, 1) * g_dr_uu(0, 1) * g_dr_dd(0, 0) +
                              // BD: verify Psi^(e)_dr, h_1 coefficient
                              //     SQR bracket placement
                              // ??: SQR(g_uu(0, 1) * g_dr2_dd(0, 0))
                              // ->: SQR(g_uu(0, 1)) * g_dr2_dd(0, 0)
                              SQR(g_uu(0, 1)) * g_dr2_dd(0, 0)) +
                // BD: verify Psi^(e)_dr, h_1 coefficient
                //     d/dr(2r g^{11} \partial_r g^{11})
                // ??: 2.0 * g_dr_uu(1, 1) * g_dr_dd(1, 1) + 2.0 * g_uu(1, 1) *
                // g_dr2_dd(1, 1)
                // ->: 2.0 * SQR(g_dr_uu(1, 1)) + 2.0 * g_uu(1, 1) * g_dr2_uu(1,
                // 1)
                2.0 * SQR(g_dr_uu(1, 1)) + 2.0 * g_uu(1, 1) * g_dr2_uu(1, 1);
            if (!std::isfinite(dr_coef_h1)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            const Real dr_coef_G_dr =
                coef_G_dr * (div_r + g_dr_uu(1, 1) / g_uu(1, 1)) -
                0.5 * r * g_uu(1, 1) *
                    (2.0 * g_dr_uu(1, 1) *
                         (-1.0 + r * g_uu(0, 1) * g_dr_dd(0, 1)) +
                     2.0 * g_uu(1, 1) *
                         (g_uu(0, 1) * g_dr_dd(0, 1) +
                          r * g_dr_uu(0, 1) * g_dr_dd(0, 1) +
                          r * g_uu(0, 1) * g_dr2_dd(0, 1)) +
                     SQR(g_uu(1, 1)) * g_dr_dd(1, 1) +
                     SQR(g_uu(0, 1)) * g_dr_dd(0, 0) +
                     r * 2.0 * g_uu(1, 1) * g_dr_uu(1, 1) * g_dr_dd(1, 1) +
                     r * SQR(g_uu(1, 1)) * g_dr2_dd(1, 1) +
                     r * 2.0 * g_uu(0, 1) * g_dr_uu(0, 1) * g_dr_dd(0, 0) +
                     r * SQR(g_uu(0, 1)) * g_dr2_dd(0, 0) +
                     2.0 * g_dr_uu(1, 1) + 2.0 * r * g_dr2_uu(1, 1));
            if (!std::isfinite(dr_coef_G_dr)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            const Real dr_coef_G_dot =
                coef_G_dot * div_r +
                0.5 * r *
                    (std::pow(g_uu(0, 1), 3) *
                         (g_dr_dd(0, 0) + r * g_dr2_dd(0, 0)) +
                     3.0 * r * SQR(g_uu(0, 1)) * g_dr_uu(0, 1) * g_dr_dd(0, 0) -
                     2.0 * (g_uu(1, 1) + r * g_dr_uu(1, 1)) *
                         (g_uu(0, 0) * g_uu(1, 1) * g_dr_dd(0, 1) +
                          g_dr_uu(0, 1)) -
                     2.0 * r * g_uu(1, 1) *
                         (g_dr_uu(0, 0) * g_uu(1, 1) * g_dr_dd(0, 1) +
                          g_uu(0, 0) * g_dr_uu(1, 1) * g_dr_dd(0, 1) +
                          g_uu(0, 0) * g_uu(1, 1) * g_dr2_dd(0, 1) +
                          // BD: verify: Psi^(e)_dr, Gdot coefficient
                          //   missing \partial^2_r g^{01} from
                          //   d/dr(-2r g^{11} \partial_r g^{01})
                          // ??: (term missing)
                          // ->: g_dr2_uu(0, 1)
                          g_dr2_uu(0, 1)) -
                     (g_dr_uu(0, 1) * g_uu(1, 1) + g_uu(0, 1) * g_dr_uu(1, 1)) *
                         (-2.0 + 2.0 * r * g_uu(0, 0) * g_dr_dd(0, 0) +
                          r * g_uu(1, 1) * g_dr_dd(1, 1)) -
                     g_uu(0, 1) * g_uu(1, 1) *
                         (2.0 * g_uu(0, 0) * g_dr_dd(0, 0) +
                          2.0 * r * g_dr_uu(0, 0) * g_dr_dd(0, 0) +
                          2.0 * r * g_uu(0, 0) * g_dr2_dd(0, 0) +
                          g_uu(1, 1) * g_dr_dd(1, 1) +
                          r * g_dr_uu(1, 1) * g_dr_dd(1, 1) +
                          r * g_uu(1, 1) * g_dr2_dd(1, 1)));
            if (!std::isfinite(dr_coef_G_dot)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            const Real dr_term2_K =
                term2_K * div_r - r * (g_dr_uu(1, 1) * K(D10, lm, c) +
                                       g_uu(1, 1) * K(D20, lm, c) +
                                       g_dr_uu(0, 1) * K(D01, lm, c) +
                                       g_uu(0, 1) * K(D11, lm, c));
            if (!std::isfinite(dr_term2_K)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Psie_dr(lm, c) =
                div_r * Psie(lm, c) +
                r_div_lambda *
                    (K(D10, lm, c) + dr_term1_hG +
                     dr_fac_Psie *
                         (term_hAB + coef_h0 * h0(lm, c) + coef_h1 * h1(lm, c) +
                          coef_G_dr * G(D10, lm, c) +
                          coef_G_dot * G(D01, lm, c) + term2_K) +
                     fac_Psie *
                         (dr_term_hAB + coef_h0 * h0(D10, lm, c) +
                          dr_coef_h0 * h0(lm, c) + coef_h1 * h1(D10, lm, c) +
                          dr_coef_h1 * h1(lm, c) + coef_G_dr * G(D20, lm, c) +
                          dr_coef_G_dr * G(D10, lm, c) +
                          coef_G_dot * G(D11, lm, c) +
                          dr_coef_G_dot * G(D01, lm, c) + dr_term2_K));
            if (!std::isfinite(Psie_dr(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }

            // Odd parity in general coordinates (static)
            // -----------------------------------------

            Psio(lm, c) = Psio_sch(lm, c) * div_sqrtdetg;
            if (!std::isfinite(Psio(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }

            const Real dr_absdetg =
                // BD: verify Psi^(o)_dr, \partial_r|det g|:
                //     det g < 0 so \partial_r|det g| = -\partial_r(det g)
                // ??: std::fabs(\partial_r det g)
                // ->: -(\partial_r det g)
                -(g_dr_dd(0, 0) * g_dd(1, 1) + g_dd(0, 0) * g_dr_dd(1, 1) -
                  2.0 * g_dd(0, 1) * g_dr_dd(0, 1));
            if (!std::isfinite(dr_absdetg)) {
              local_error_ = singular_master_coefficient;
              return;
            }
            Psio_dr(lm, c) =
                (div_sqrtdetg -
                 0.5 * r * std::pow(div_sqrtdetg, 3) * dr_absdetg) *
                    div_lambda_2 *
                    (H1(D01, lm, c) - H0(D10, lm, c) +
                     2.0 * div_r * H0(lm, c)) +
                r * div_sqrtdetg * div_lambda_2 *
                    (H1(D11, lm, c) - H0(D20, lm, c) +
                     2.0 * div_r * H0(D10, lm, c) -
                     2.0 * SQR(div_r) * H0(lm, c));
            if (!std::isfinite(Psio_dr(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }

            // Even parity in general coordinates (dynamic)
            // -----------------------------------------

            const Real term2_hG =
                (-2.0 * rdot * div_r) *
                (g_uu(0, 0) * (h0(lm, c) - 0.5 * r2 * G(D01, lm, c)) +
                 g_uu(0, 1) * (h1(lm, c) - 0.5 * r2 * G(D10, lm, c)));
            if (!std::isfinite(term2_hG)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            const Real term_hAB_rdot =
                rdot * h00(lm, c) *
                    (rdot * SQR(g_uu(0, 0)) + 2.0 * g_uu(0, 0) * g_uu(0, 1)) +
                2.0 * rdot * h01(lm, c) *
                    (rdot * g_uu(0, 0) * g_uu(0, 1) + SQR(g_uu(0, 1)) +
                     g_uu(0, 0) * g_uu(1, 1)) +
                rdot * h11(lm, c) *
                    (rdot * SQR(g_uu(0, 1)) + 2.0 * g_uu(0, 1) * g_uu(1, 1));
            if (!std::isfinite(term_hAB_rdot)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Real coef_h0_t =
                2.0 * r * std::pow(g_uu(0, 1), 3) * g_dot_dd(0, 1) +
                2.0 * r * g_uu(0, 1) *
                    g_dot_uu(0, 1)  // CHECK g_dot_uu(0,1) in the notes, but eq
                                    // full of typos
                + (-r * g_uu(0, 0) * SQR(g_uu(1, 1)) +
                   2.0 * r * g_uu(1, 1) * SQR(g_uu(0, 1))) *
                      g_dot_dd(1, 1) +
                r * SQR(g_uu(0, 1)) * g_uu(0, 0) * g_dot_dd(0, 0);
            if (!std::isfinite(coef_h0_t)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            coef_h0_t *= div_r;
            if (!std::isfinite(coef_h0_t)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Real coef_h0_rdot =
                r * rdot * pow(g_uu(0, 0), 3) * g_dot_dd(0, 0) +
                2.0 * r * g_uu(1, 1) * g_dr_uu(0, 0) +
                2.0 * r * g_uu(0, 1) *
                    (g_dr_uu(0, 1) + g_dot_uu(0, 0) + rdot * g_dr_uu(0, 0)) +
                r * pow(g_uu(0, 1), 3) *
                    (2.0 * g_dot_dd(1, 1) + rdot * g_dr_dd(1, 1)) +
                2.0 * SQR(g_uu(0, 1)) *
                    (-2.0 + r * g_uu(1, 1) * g_dr_dd(1, 1)) +
                SQR(g_uu(0, 0)) * (-2.0 * SQR(rdot) +
                                   r * rdot * g_uu(0, 1) *
                                       (g_dr_dd(0, 0) + 2.0 * g_dot_dd(0, 1)) +
                                   2.0 * r *
                                       (g_uu(0, 1) * g_dot_dd(0, 0) +
                                        g_uu(1, 1) * g_dr_dd(0, 0))) +
                2.0 * r * g_uu(0, 0) * g_dot_uu(0, 1) -
                2.0 * g_uu(0, 0) * g_uu(1, 1) +
                rdot * g_uu(0, 0) *
                    (2.0 * r * g_dot_uu(0, 0) - 6.0 * g_uu(0, 1) +
                     r * SQR(g_uu(0, 1)) *
                         (2.0 * g_dr_dd(0, 1) + g_dot_dd(1, 1))) +
                4.0 * r * g_uu(0, 0) * g_uu(0, 1) *
                    (g_uu(0, 1) * g_dot_dd(0, 1) + g_uu(1, 1) * g_dr_dd(0, 1));
            if (!std::isfinite(coef_h0_rdot)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            coef_h0_rdot *= rdot * div_r;
            if (!std::isfinite(coef_h0_rdot)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Real coef_h1_t = SQR(g_uu(0, 1)) * g_dot_dd(0, 0) +
                             2.0 * g_uu(0, 1) * g_uu(1, 1) * g_dot_dd(0, 1) +
                             SQR(g_uu(1, 1)) * g_dot_dd(1, 1);
            if (!std::isfinite(coef_h1_t)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            coef_h1_t *= g_uu(0, 1);  // TODO CHECK notes for factor *div_r
            if (!std::isfinite(coef_h1_t)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Real coef_h1_rdot =
                -2.0 * SQR(rdot) * g_uu(0, 0) * g_uu(0, 1) +
                2.0 * r * pow(g_uu(0, 1), 3) * g_dr_dd(0, 0) +
                2.0 * r * SQR(g_uu(0, 1)) *
                    (g_uu(0, 0) * g_dot_dd(0, 0) +
                     2.0 * g_uu(1, 1) * g_dr_dd(0, 1)) +
                2.0 * r * g_uu(1, 1) *
                    (g_dr_uu(0, 1) + g_uu(0, 0) * g_uu(1, 1) * g_dot_dd(1, 1)) +
                2.0 * g_uu(0, 1) *
                    (r * (g_dr_uu(1, 1) + g_dot_uu(0, 1)) +
                     g_uu(1, 1) *
                         (-3.0 + 2.0 * r * g_uu(0, 0) * g_dot_dd(0, 1)) +
                     r * SQR(g_uu(1, 1)) * g_dr_dd(1, 1)) +
                r * rdot * SQR(g_uu(0, 0)) *
                    (g_uu(0, 1) * g_dot_dd(0, 0) -
                     g_uu(1, 1) * (g_dr_dd(0, 0) - 2.0 * g_dot_dd(0, 1))) +
                2.0 * rdot * g_uu(0, 0) *
                    (r * g_dot_uu(0, 1) + r * SQR(g_uu(0, 1)) * g_dr_dd(0, 0) -
                     g_uu(1, 1) +
                     r * g_uu(0, 1) * g_uu(1, 1) * g_dot_dd(1, 1)) +
                2.0 * r * rdot * g_uu(0, 1) * g_dr_uu(0, 1) +
                r * rdot * pow(g_uu(0, 1), 3) *
                    (2.0 * g_dr_dd(0, 1) - g_dot_dd(1, 1)) +
                rdot * SQR(g_uu(0, 1)) *
                    (-4.0 + r * g_uu(1, 1) * g_dr_dd(1, 1));
            if (!std::isfinite(coef_h1_rdot)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            coef_h1_rdot *= rdot * div_r;
            if (!std::isfinite(coef_h1_rdot)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Real coef_G_dr_t = 2.0 * g_uu(0, 1) * g_uu(1, 1) * g_dot_dd(0, 1) +
                               SQR(g_uu(0, 1)) * g_dot_dd(0, 0) +
                               SQR(g_uu(1, 1)) * g_dot_dd(1, 1);
            if (!std::isfinite(coef_G_dr_t)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            coef_G_dr_t *= -0.5 * r2 * g_uu(0, 1);
            if (!std::isfinite(coef_G_dr_t)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Real coef_G_dr_rdot =
                -2.0 * SQR(rdot) * g_uu(0, 0) * g_uu(0, 1) +
                2.0 * r * pow(g_uu(0, 1), 3) * g_dr_dd(0, 0) +
                2.0 * r * SQR(g_uu(0, 1)) *
                    (g_uu(0, 0) * g_dot_dd(0, 0) +
                     2.0 * g_uu(1, 1) * g_dr_dd(0, 1)) +
                2.0 * r * g_uu(1, 1) *
                    (g_dr_uu(0, 1) + g_uu(0, 0) * g_uu(1, 1) * g_dot_dd(1, 1)) +
                2.0 * r * g_uu(0, 1) * (g_dr_uu(1, 1) + g_dot_uu(0, 1)) +
                2.0 * g_uu(0, 1) * g_uu(1, 1) *
                    (-3.0 + 2.0 * r * g_uu(0, 0) * g_dot_dd(0, 1)) +
                2.0 * r * g_uu(0, 1) * SQR(g_uu(1, 1)) * g_dr_dd(1, 1) +
                r * rdot * SQR(g_uu(0, 0)) *
                    (g_uu(0, 1) * g_dot_dd(0, 0) -
                     g_uu(1, 1) * (g_dr_dd(0, 0) - 2.0 * g_dot_dd(0, 1))) +
                2.0 * rdot * g_uu(0, 0) *
                    (r * g_dot_uu(0, 1) + r * SQR(g_uu(0, 1)) * g_dr_dd(0, 0) -
                     g_uu(1, 1) +
                     r * g_uu(0, 1) * g_uu(1, 1) * g_dot_dd(1, 1)) +
                2.0 * r * rdot * g_uu(0, 1) * g_dr_uu(0, 1) +
                r * rdot * pow(g_uu(0, 1), 3) *
                    (2.0 * g_dr_dd(0, 1) - g_dot_dd(1, 1)) +
                rdot * SQR(g_uu(0, 1)) *
                    (-4.0 + r * g_uu(1, 1) * g_dr_dd(1, 1));
            if (!std::isfinite(coef_G_dr_rdot)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            coef_G_dr_rdot *= -0.5 * r * rdot;
            if (!std::isfinite(coef_G_dr_rdot)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Real coef_G_dot_t =
                -2.0 * std::pow(g_uu(0, 1), 3) * g_dot_dd(0, 1) -
                2.0 * g_uu(0, 1) * g_dot_uu(0, 1)  // CHECK in the notes
                - SQR(g_uu(0, 1)) * g_uu(0, 0) * g_dot_dd(0, 0) +
                g_uu(1, 1) * (g_uu(1, 1) * g_uu(0, 0) - 2.0 * SQR(g_uu(0, 1))) *
                    g_dot_dd(1, 1);
            if (!std::isfinite(coef_G_dot_t)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            coef_G_dot_t *= 0.5 * r2;  // TODO CHECK again, notes
            if (!std::isfinite(coef_G_dot_t)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Real coef_G_dot_rdot =
                r * rdot * pow(g_uu(0, 0), 3) * g_dot_dd(0, 0) +
                2.0 * r * g_uu(1, 1) * g_dr_uu(0, 0) +
                2.0 * r * g_uu(0, 1) *
                    (g_dr_uu(0, 1) + g_dot_uu(0, 0) + rdot * g_dr_uu(0, 0)) +
                r * pow(g_uu(0, 1), 3) *
                    (2.0 * g_dot_dd(1, 1) + rdot * g_dr_dd(1, 1)) +
                2.0 * SQR(g_uu(0, 1)) *
                    (-2.0 + r * g_uu(1, 1) * g_dr_dd(1, 1)) +
                SQR(g_uu(0, 0)) * (-2.0 * SQR(rdot) +
                                   r * rdot * g_uu(0, 1) *
                                       (g_dr_dd(0, 0) + 2.0 * g_dot_dd(0, 1)) +
                                   2.0 * r *
                                       (g_uu(0, 1) * g_dot_dd(0, 0) +
                                        g_uu(1, 1) * g_dr_dd(0, 0))) +
                2.0 * r * g_uu(0, 0) * g_dot_uu(0, 1) -
                2.0 * g_uu(0, 0) * g_uu(1, 1) +
                rdot * g_uu(0, 0) *
                    (2.0 * r * g_dot_uu(0, 0) - 6.0 * g_uu(0, 1) +
                     r * SQR(g_uu(0, 1)) *
                         (2.0 * g_dr_dd(0, 1) + g_dot_dd(1, 1))) +
                4.0 * r * g_uu(0, 0) * g_uu(0, 1) *
                    (g_uu(0, 1) * g_dot_dd(0, 1) + g_uu(1, 1) * g_dr_dd(0, 1));
            if (!std::isfinite(coef_G_dot_rdot)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            coef_G_dot_rdot *= -0.5 * r * rdot;
            if (!std::isfinite(coef_G_dot_rdot)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            const Real term2_K_rdot =
                -r * rdot *
                (g_uu(0, 0) * K(D01, lm, c) + g_uu(0, 1) * K(D10, lm, c));
            if (!std::isfinite(term2_K_rdot)) {
              local_error_ = singular_master_coefficient;
              return;
            }

            Psie_dyn(lm, c) = term1_K;
            if (!std::isfinite(Psie_dyn(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }
            Psie_dyn(lm, c) += term1_hG + term2_hG;
            if (!std::isfinite(Psie_dyn(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }
            Psie_dyn(lm, c) +=
                fac_Psie *
                (term_hAB + term_hAB_rdot +
                 (coef_h0 + coef_h0_t + coef_h0_rdot) * h0(lm, c) +
                 (coef_h1 + coef_h1_t + coef_h1_rdot) * h1(lm, c) +
                 (coef_G_dr + coef_G_dr_t + coef_G_dr_rdot) * G(D10, lm, c) +
                 (coef_G_dot + coef_G_dot_t + coef_G_dot_rdot) * G(D01, lm, c) +
                 term2_K + term2_K_rdot);
            if (!std::isfinite(Psie_dyn(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }
            Psie_dyn(lm, c) *= r_div_lambda;
            if (!std::isfinite(Psie_dyn(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }

            // Odd parity in general coordinates (dynamic)
            // -----------------------------------------

            Psio_dyn(lm, c) = (r * (H1(D01, lm, c) - H0(D10, lm, c)) +
                               2.0 * (H0(lm, c) - H1(lm, c) * rdot));
            if (!std::isfinite(Psio_dyn(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }
            Psio_dyn(lm, c) *= div_lambda_2 * div_sqrtdetg;
            if (!std::isfinite(Psio_dyn(lm, c))) {
              local_error_ = singular_master_coefficient;
              return;
            }

          }  // real& imag
        }  // for m
      }  // for l
    }
  }
}

//----------------------------------------------------------------------------------------
// Read filename defaults on every rank, then render active targets on the writer
// before any file is opened. Canonicalize existing prefixes to detect relative,
// dot-component, and symlink aliases. Relative basenames retain run-dir meaning.
void WaveExtractRWZ::InitializeOutput(ParameterInput *pin) {
  if (local_error_ != 0) return;
  try {
    const std::array<const char *, NFiles> keys = {{
      "filename_diagnostic", "filename_adm", "filename_hlm", "filename_psie",
      "filename_psio", "filename_psie_dyn", "filename_psio_dyn", "filename_Qplus",
      "filename_Qstar", "filename_H1_dot", "filename_H0_dr", "filename_H0",
      "filename_H1", "filename_H", "filename_H_dr", "filename_psie_dr",
      "filename_psio_dr", "filename_Qplus_dr", "filename_Qstar_dr"
    }};
    const std::array<const char *, NFiles> defaults = {{
      "diagnostic", "adm", "wave_rwz", "wave_psie", "wave_psio", "wave_psie_dyn",
      "wave_psio_dyn", "wave_Qplus", "wave_Qstar", "wave_H1dot", "wave_H0dr",
      "wave_H0", "wave_H1", "wave_H", "wave_Hdr", "wave_psie_dr", "wave_psio_dr",
      "wave_Qplus_dr", "wave_Qstar_dr"
    }};
    std::array<std::string, NFiles> basenames;
    for (int f = 0; f < NFiles; ++f) {
      basenames[f] = pin->GetOrAddString("rwz_extraction", keys[f], defaults[f]);
    }
    // Parameter defaults stay identical on all ranks; only the writer needs
    // filesystem paths. The caller retains its all-rank output-path phase gate.
    if (global_variable::my_rank != mpi_root_) return;
    const int nfiles = extra_output_ ? NFiles : H1DotFile;
    std::set<std::string> targets;
    for (auto &shell : shells_) {
      std::ostringstream radius;
      radius.imbue(std::locale::classic());
      radius << std::setfill('0') << std::setw(5) << std::fixed
             << std::setprecision(2) << shell.R;
      if (!radius || !std::isfinite(shell.R) || shell.R <= 0) {
        local_error_ = invalid_configuration;
        return;
      }
      for (int f = 0; f < nfiles; ++f) {
        if (basenames[f].find('\0') != std::string::npos) {
          local_error_ = invalid_configuration;
          return;
        }
        const std::string rendered = basenames[f] + "_r" + radius.str() + ".txt";
        std::error_code error;
        const auto absolute = std::filesystem::absolute(rendered, error);
        if (error) {
          local_error_ = invalid_configuration;
          return;
        }
        const auto normalized = std::filesystem::weakly_canonical(absolute, error);
        if (error || normalized.empty()) {
          // Permission/stat failures are not evidence that a target is absent.
          local_error_ = invalid_configuration;
          return;
        }
        shell.paths[f] = normalized.string();
        if (!targets.insert(shell.paths[f]).second) {
          local_error_ = invalid_configuration;
          return;
        }
      }
    }
  } catch (...) {
    // Returned error codes above follow the normal setup collective. Unexpected
    // allocation/path exceptions instead stop the job without another collective.
    RWZAbort("unexpected output-path construction failure",mpi_root_);
  }
}

//----------------------------------------------------------------------------------------
// Root-only, lazy append. A returned failure is broadcast by RWZOutput before any
// identity/cadence mutation. Earlier flushed files may contain a partial sample;
// no truncation or multi-file rollback is attempted. All initial targets pass
// descriptor preflight before the first header or row. Handles belong to the manager.
int WaveExtractRWZ::Write(Real sample_time, int sample_cycle) {
  if (global_variable::my_rank != mpi_root_) return 0;
  if (local_error_ != 0) return local_error_;
  const auto fail = [&](const std::string &path, const char *operation) {
    std::cerr << "### FATAL ERROR in RWZ output: " << operation << " '"
              << path << "'" << std::endl;
    local_error_ = output_io_error;
    return local_error_;
  };
  try {
    if (!std::isfinite(sample_time) || sample_cycle < 0 || output_digits_ <= 0) {
      return fail("", "invalid sample identity or precision for");
    }
    const int nfiles = extra_output_ ? NFiles : H1DotFile;
    // A successful first preflight opens every active target together. Failed
    // preflights set local_error_ and cannot be retried with partial handles.
    const bool first_write = shells_.front().files[Diagnostic] == nullptr;
    std::vector<std::array<bool, NFiles>> needs_header;
    if (first_write) {
      needs_header.resize(shells_.size());
      std::set<std::pair<dev_t, ino_t>> targets;
      for (std::size_t ir = 0; ir < shells_.size(); ++ir) {
        auto &shell = shells_[ir];
        for (int f = 0; f < nfiles; ++f) {
          const std::string &path = shell.paths[f];
          if (path.empty()) return fail(path, "uninitialized target");
          // Nonblocking open prevents a FIFO from hanging before its type check.
          // O_APPEND/O_CREAT retain fopen("a") history and creation semantics.
          const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND |
                                O_NONBLOCK | O_CLOEXEC, 0666);
          if (fd < 0) return fail(path, "could not open append descriptor for");
          struct stat info;
          if (::fstat(fd, &info) != 0 || info.st_size < 0 || !S_ISREG(info.st_mode)) {
            const int close_status = ::close(fd);
            return fail(path, close_status == 0 ? "could not stat regular file" :
                        "could not close rejected descriptor for");
          }
          // Canonical names do not identify hard links. Compare the open files
          // before any target receives a header or row; later samples reuse them.
          if (!targets.insert({info.st_dev, info.st_ino}).second) {
            const int close_status = ::close(fd);
            return fail(path, close_status == 0 ? "target aliases another output" :
                        "could not close rejected descriptor for");
          }
          std::FILE *file = ::fdopen(fd, "a");
          if (file == nullptr) {
            const int close_status = ::close(fd);
            return fail(path, close_status == 0 ? "could not create append stream for" :
                        "could not close rejected descriptor for");
          }
          shell.files[f] = file;
          if (std::fseek(file, 0, SEEK_END) != 0 || std::ferror(file) != 0) {
            return fail(path, "could not seek to append position in");
          }
          needs_header[ir][f] = info.st_size == 0;
        }
      }
    }
    for (std::size_t ir = 0; ir < shells_.size(); ++ir) {
      auto &shell = shells_[ir];
      for (int f = 0; f < nfiles; ++f) {
        const std::string &path = shell.paths[f];
        std::FILE *file = shell.files[f];
        if (std::ferror(file) != 0) return fail(path, "stream already failed for");

        if (first_write && needs_header[ir][f]) {
          bool header_ok = true;
          if (f == Diagnostic) {
            header_ok = std::fprintf(file,
                "# 1:iter 2:time 3:SchwarzschildRradius 4:SchwarzschildMass "
                "5:SchwarzschildRadius_dot 6:NormDeltaGamma "
                "7:NormTracekappaAB\n") >= 0;
          } else if (f == ADMCharges) {
            header_ok = std::fprintf(file,
                "# 1:iter 2:time 3:ADM_M 4:ADM_Px 5:ADM_Py 6:ADM_Pz "
                "7:ADM_Jx 8:ADM_Jy 9:ADM_Jz\n") >= 0;
          } else {
            header_ok = std::fprintf(file, "# 1:iter 2:time") >= 0;
            int column = 3;
            for (int l = 2; l <= lmax_ && header_ok; ++l) {
              for (int m = -l; m <= l && header_ok; ++m) {
                header_ok = std::fprintf(file,
                    " %d:l=%d-m=%d-Re %d:l=%d-m=%d-Im",
                    column, l, m, column + 1, l, m) >= 0;
                column += 2;
              }
            }
            header_ok = header_ok && std::fputc('\n', file) != EOF;
          }
          if (!header_ok || std::ferror(file) != 0 || std::fflush(file) != 0 ||
              std::ferror(file) != 0) return fail(path, "could not write/flush header to");
        }

        if (std::fprintf(file, "%d %.*e", sample_cycle, output_digits_,
                         static_cast<double>(sample_time)) < 0 ||
            std::ferror(file) != 0) return fail(path, "could not write row identity to");
        const auto write_real = [&](Real value) {
          return std::isfinite(value) &&
                 std::fprintf(file, " %.*e", output_digits_,
                              static_cast<double>(value)) >= 0 &&
                 std::ferror(file) == 0;
        };
        if (f == Diagnostic) {
          const Real values[] = {shell.Schwarzschild_radius, shell.Schwarzschild_mass,
                                 shell.dot_rsch, shell.norm_Delta_Gamma,
                                 shell.norm_Tr_kappa_dd};
          for (Real value : values) {
            if (!write_real(value)) return fail(path, "could not write diagnostic to");
          }
        } else if (f == ADMCharges) {
          // Background finalization already applied the source ADM normalization.
          const Real values[] = {shell.ADM_M, shell.ADM_P[0], shell.ADM_P[1],
                                 shell.ADM_P[2], shell.ADM_J[0], shell.ADM_J[1],
                                 shell.ADM_J[2]};
          for (Real value : values) {
            if (!write_real(value)) return fail(path, "could not write ADM charges to");
          }
        } else {
          // Target enums put Hlm third, unlike GRA's last entry. Dispatch explicitly
          // rather than inheriting the source enum-offset array indexing.
          const HostArray2D<Real> *data = nullptr;
          const ModeField *field = nullptr;
          int derivative = D00;
          switch (f) {
            case Hlm: break;
            case PsieFile: data = &shell.Psie; break;
            case PsioFile: data = &shell.Psio; break;
            case PsieDynFile: data = &shell.Psie_dyn; break;
            case PsioDynFile: data = &shell.Psio_dyn; break;
            case QplusFile: data = &shell.Qplus; break;
            case QstarFile: data = &shell.Qstar; break;
            case H1DotFile: field = &shell.H1; derivative = D01; break;
            case H0DrFile: field = &shell.H0; derivative = D10; break;
            case H0File: field = &shell.H0; break;
            case H1File: field = &shell.H1; break;
            case HFile: field = &shell.H; break;
            case HDrFile: field = &shell.H; derivative = D10; break;
            case PsieDrFile: data = &shell.Psie_dr; break;
            case PsioDrFile: data = &shell.Psio_dr; break;
            case QplusDrFile: data = &shell.Qplus_dr; break;
            case QstarDrFile: data = &shell.Qstar_dr; break;
            default: return fail(path, "invalid file-to-mode mapping for");
          }
          for (int l = 2; l <= lmax_; ++l) {
            // This is the configured Cartesian coordinate radius, not areal r.
            const Real normalization = f == Hlm ?
                std::sqrt(static_cast<Real>((l-1)*l*(l+1)*(l+2))) / shell.R : 0;
            for (int m = -l; m <= l; ++m) {
              const int lm = l*l + l + m - 4;
              if (lm < 0 || lm >= nmode_) return fail(path, "invalid mode index for");
              Real real_part, imag_part;
              if (f == Hlm) {
                // sqrt((l-1)l(l+1)(l+2)) * (Psie + i Psio) / R.
                real_part = normalization * (shell.Psie(lm,0) - shell.Psio(lm,1));
                imag_part = normalization * (shell.Psie(lm,1) + shell.Psio(lm,0));
              } else if (data != nullptr) {
                real_part = (*data)(lm,0);
                imag_part = (*data)(lm,1);
              } else {
                real_part = (*field)(derivative,lm,0);
                imag_part = (*field)(derivative,lm,1);
              }
              if (!write_real(real_part) || !write_real(imag_part)) {
                return fail(path, "could not write finite complex mode to");
              }
            }
          }
        }
        if (std::fputc('\n', file) == EOF || std::ferror(file) != 0 ||
            std::fflush(file) != 0 || std::ferror(file) != 0) {
          return fail(path, "could not finish/flush row to");
        }
      }
    }
  } catch (const std::exception &error) {
    // Never let a root-only formatting/filesystem exception strand the other ranks
    // in their status broadcast. The adapter owns the collective fatal decision.
    std::cerr << "### FATAL ERROR in RWZ output: " << error.what() << std::endl;
    local_error_ = output_io_error;
    return local_error_;
  }
  return 0;
}

}  // namespace z4c
