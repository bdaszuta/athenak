#ifndef Z4C_WAVE_EXTRACT_RWZ_HPP_
#define Z4C_WAVE_EXTRACT_RWZ_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file wave_extract_rwz.hpp
//! \brief Completed-time RWZ extraction on fixed Cartesian coordinate spheres.
//!
//! The diagnostic borrows evolved state and owns its communicated derivative bank.
//! Cartesian tensors use xx,xy,xz,yy,yz,zz; sphere tensors use coordinate components,
//! not an orthonormal frame. Length and time are in geometrized simulation units.
//! Only compact integral buffers cross to host. All kernels finish before output.

#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"
#include "utils/inline_interpolator.hpp"

class MeshBlockPack;
class MeshBoundaryValuesCC;
class ParameterInput;

namespace z4c {
class WaveExtractRWZ {
 public:
  WaveExtractRWZ(MeshBlockPack *pack, ParameterInput *pin);
  ~WaveExtractRWZ();
  void Load(Real sample_time, int sample_cycle);
  int Write(Real sample_time, int sample_cycle);
  int root_rank() const { return mpi_root_; }

  // NVCC extended lambdas require public enclosing methods and captured types.
  // These implementation entry points do not expose the manager-owned state.
  enum class RadiusMethod { areal, areal_simple, average_schw, schw_gthth, schw_gphph };
  void InitializeTables();
  template <int NGHOST> void RefreshOwnershipImpl();
  template <int NGHOST> void PrepareAuxiliaryImpl(Real sample_time);
  void AuxiliaryBCs(DvceArray5D<Real> fields, bool coarse);
  template <int NGHOST> void SampleAndAccumulateBackgroundImpl();
  void FinalizeBackgroundAndTransform();
  void AccumulateMultipoles();

 private:
  enum { D00, D10, D20, D01, D11, NDerivative };
  enum { invalid_configuration = 1, invalid_ownership, invalid_conformal_state,
         invalid_auxiliary, invalid_sample, invalid_area, invalid_background,
         singular_radial_map, invalid_two_metric, invalid_projection,
         singular_master_coefficient, output_io_error };
  enum { I_adm_M, I_adm_Px, I_adm_Py, I_adm_Pz,
         I_adm_Jx, I_adm_Jy, I_adm_Jz,
         Irsch2, Idrsch_dri, Id2rsch_dri2, Idot_rsch2, Idrsch_dri_dot,
         Ig00, Ig0R, IgRR, IgRt, Igtt, Igpp,
         IdR_g00, IdR_g0R, IdR_gRR, IdR2_g00, IdR2_g0R, IdR2_gRR,
         IdR_gtt, Idot_g00, Idot_g0R, Idot_gRR, NBackground };
  enum { Ih00, Ih01, Ih11, Ih0, Ih1, IG, IK,
         Ih00_dr, Ih01_dr, Ih11_dr, Ih0_dr, Ih1_dr, IG_dr, IK_dr,
         IG_dr2, IK_dr2, IG_dr_dot, IK_dr_dot,
         Ih00_dot, Ih01_dot, Ih11_dot, Ih0_dot, Ih1_dot, IG_dot, IK_dot,
         IH0, IH1, IH, IH0_dr, IH1_dr, IH_dr, IH0_dot, IH1_dot, IH_dot,
         IH0_dr2, IH_dr2, IH1_dr_dot, NMultipoles };
  enum { Y, Ytheta, Yphi, W, X, NHarmonic };
  enum { Diagnostic, ADMCharges, Hlm, PsieFile, PsioFile, PsieDynFile, PsioDynFile,
         QplusFile, QstarFile, H1DotFile, H0DrFile, H0File, H1File, HFile, HDrFile,
         PsieDrFile, PsioDrFile, QplusDrFile, QstarDrFile, NFiles };

  // Small value types retain symmetric aliasing. Repeated angular off-diagonal
  // updates in the inherited radial map deliberately hit the same storage slot.
  struct Sym2 {
    Real v[3] = {};
    KOKKOS_INLINE_FUNCTION
    Real &operator()(int i, int j) { return v[i == j ? 2*i : 1]; }
    KOKKOS_INLINE_FUNCTION
    Real operator()(int i, int j) const { return v[i == j ? 2*i : 1]; }
  };
  struct Connection2 {
    Real v[6] = {};
    KOKKOS_INLINE_FUNCTION
    Real &operator()(int i, int j, int k) { return v[3*i + (j == k ? 2*j : 1)]; }
    KOKKOS_INLINE_FUNCTION
    Real operator()(int i, int j, int k) const {
      return v[3*i + (j == k ? 2*j : 1)];
    }
  };
  struct ModeField {
    HostArray3D<Real> data;  // (derivative slot,mode,ReIm), independently allocated
    Real &operator()(int d, int lm, int c) { return data(d,lm,c); }
    Real operator()(int d, int lm, int c) const { return data(d,lm,c); }
    Real &operator()(int lm, int c) { return data(D00,lm,c); }
    Real operator()(int lm, int c) const { return data(D00,lm,c); }
  };
  struct ModeTensor {
    HostArray3D<Real> data;  // symmetric (tt,tr,rr),mode,ReIm
    Real &operator()(int i, int j, int lm, int c) {
      return data(i == j ? 2*i : 1,lm,c);
    }
    Real operator()(int i, int j, int lm, int c) const {
      return data(i == j ? 2*i : 1,lm,c);
    }
  };
  // One allocation of each radius-independent table serves every shell. The
  // manager initializes these on device; shells retain read-only View handles.
  struct AngularTables {
    DvceArray1D<Real> theta, phi, weights;
    std::array<DvceArray3D<Real>, NHarmonic> harmonics;  // (point,mode,ReIm)
  };
  struct Shell {
    Real R = 0, center[3] = {};
    // Shared angular aliases and shell-specific geometry survive mesh changes.
    DvceArray1D<const Real> theta, phi, weights;
    // Unreflected positions are setup-local; only lookup positions persist.
    DvceArray2D<Real> x_lookup, normal;
    DvceArray1D<int> reflected, owner;
    DvceArray3D<Real> jacobian, inverse_jacobian;
    std::array<DvceArray3D<const Real>, NHarmonic> harmonics;
    // Exactly one interpolation View is allocated for the active ghost width.
    DvceArray1D<IndAndWghts<2>> interp2;
    DvceArray1D<IndAndWghts<3>> interp3;
    DvceArray1D<IndAndWghts<4>> interp4;
    // All bundles use (point,derivative slot,component). Only consumed slots of
    // owned points are defined for the current sample; consumers skip the rest.
    DvceArray3D<Real> gamma, beta_d, alpha, beta2;
    Real Schwarzschild_radius = 0, Schwarzschild_mass = 0, dot_rsch = 0;
    Real drsch_dri = 0, d2rsch_dri2 = 0, dri_drsch = 0;
    Real d2ri_drsch2 = 0, d3ri_drsch3 = 0;
    Real ADM_M = 0, ADM_P[3] = {}, ADM_J[3] = {};
    Sym2 g_dd, g_uu, g_dr_dd, g_dr_uu, g_dr2_dd, g_dr2_uu, g_dot_dd, g_dot_uu;
    Connection2 Gamma_udd, Gamma_dyn_udd, Delta_Gamma;
    Real norm_Delta_Gamma = 0, norm_Tr_kappa_dd = 0;
    ModeField h00, h01, h11, h0, h1, G, K, H0, H1, H;
    ModeTensor kappa_dd;
    HostArray3D<Real> kappa_d;
    HostArray2D<Real> kappa, Tr_kappa_dd;
    HostArray2D<Real> Psie, Psio, Psie_dyn, Psio_dyn, Psie_sch, Psio_sch;
    HostArray2D<Real> Qplus, Qstar, Psie_dr, Psio_dr, Qplus_dr, Qstar_dr;
    std::array<std::string, NFiles> paths;
    std::array<std::FILE *, NFiles> files = {};
  };

  KOKKOS_INLINE_FUNCTION
  static int Sym(int a, int b) {
    if (a > b) { int t = a; a = b; b = t; }
    return a == 0 ? b : (a == 1 ? b + 2 : 5);
  }

  void InitializeOutput(ParameterInput *pin);
  void RefreshOwnership();
  void PrepareAuxiliary(Real sample_time);
  void ExchangeAuxiliary();
  void SampleAndAccumulateBackground();
  void FinalizeMultipoles();
  void CheckPhase(const char *phase);

  MeshBlockPack *pack_;  // borrowed stable pack; mutable block/state handles are rebound
  int lmax_ = 2, ntheta_ = 32, nphi_ = 64, npoint_ = 0, nmode_ = 0;
  int mpi_root_ = 0, output_digits_ = 16, ownership_epoch_ = -1;
  int local_error_ = 0;
  bool bitant_ = false, subtract_background_ = false, extra_output_ = false;
  bool verbose_ = false, gauss_legendre_ = false;
  RadiusMethod radius_method_ = RadiusMethod::areal;
  AngularTables angular_;  // immutable after InitializeTables; outlives shell aliases
  std::vector<Shell> shells_;
  std::unique_ptr<MeshBoundaryValuesCC> boundary_;
  DvceArray5D<Real> auxiliary_, coarse_auxiliary_;
  DvceArray2D<Real> background_;
  DvceArray4D<Real> multipoles_;
  HostArray2D<Real> host_background_;
  HostArray4D<Real> host_multipoles_;
  DvceArray1D<int> device_error_;
  HostArray1D<int> host_error_;  // independent staging, reused after each phase fence
};
}  // namespace z4c
#endif  // Z4C_WAVE_EXTRACT_RWZ_HPP_
