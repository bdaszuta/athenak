//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rwz.cpp
//! \brief Completed-time RWZ output and exact restart/sample bookkeeping.

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "outputs.hpp"
#include "parameter_input.hpp"
#include "z4c/wave_extract_rwz.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {
// All ranks must reject a bad identity/cadence before any extraction or file write.
// In particular, root must not exit while other ranks wait for writer status.
[[noreturn]] void AbortRWZ(const std::string &block, const char *reason) {
  if (global_variable::my_rank == 0) {
    std::cout << "### FATAL ERROR in RWZ output block '" << block << "': "
              << reason << std::endl;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

void CheckRWZState(bool invalid, const std::string &block, const char *reason) {
  int error = invalid ? 1 : 0;
#if MPI_PARALLEL_ENABLED
  int global_error = 0;
  if (MPI_Allreduce(&error, &global_error, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS) {
    AbortRWZ(block, "could not reduce validation status");
  }
  error = global_error;
#endif
  if (error != 0) AbortRWZ(block, reason);
}

// Sample identity and scheduling time must survive restart exactly, independently
// of the lower, user-selected precision of waveform rows. Do not use SetReal.
std::string EncodeRWZReal(Real value) {
  std::ostringstream text;
  text.imbue(std::locale::classic());
  text << std::scientific
       << std::setprecision(std::numeric_limits<Real>::max_digits10 - 1) << value;
  return text ? text.str() : std::string();
}

bool DecodeRWZReal(const std::string &text, Real &value) {
  std::istringstream input(text);
  input.imbue(std::locale::classic());
  if (!(input >> value) || !std::isfinite(value)) return false;
  input >> std::ws;
  return input.eof();
}
}  // namespace

//----------------------------------------------------------------------------------------
// Validate the persisted pair before allocating the extraction manager. Integer
// parameters use the normal readers; only Real identity text needs strict parsing.
RWZOutput::RWZOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
    BaseTypeOutput(pin, pm, op) {
  last_written_cycle_ = pin->GetOrAddInteger(op.block_name,
                                            "rwz_last_written_cycle", -1);
  const std::string time_text = pin->GetOrAddString(op.block_name,
                                                   "rwz_last_written_time", "none");
  const bool unset_cycle = last_written_cycle_ == -1;
  const bool unset_time = time_text == "none";
  bool invalid = unset_cycle != unset_time || last_written_cycle_ < -1;
  if (last_written_cycle_ >= 0 && !unset_time) {
    invalid = !DecodeRWZReal(time_text, last_written_time_) || invalid;
  }
  CheckRWZState(invalid, op.block_name,
                "invalid persisted rwz_last_written_cycle/time pair");
  CheckRWZState(!std::isfinite(op.last_time) || !std::isfinite(op.dt) ||
                op.file_number < 0 || op.dcycle < 0 ||
                (op.dcycle == 0 && op.dt <= 0), op.block_name,
                "invalid scheduling clock, cadence, or file counter");
  rwz_ = std::make_unique<z4c::WaveExtractRWZ>(pm->pmb_pack, pin);
}

// The manager is complete here, not at the forward declaration in outputs.hpp.
RWZOutput::~RWZOutput() = default;

//----------------------------------------------------------------------------------------
// Freeze the actual committed sample identity, with no dt offset. Reset pending
// first so an exact duplicate cannot leave an earlier load eligible for writing.
void RWZOutput::LoadOutputData(Mesh *pm) {
  pending_ = false;
  CheckRWZState(!std::isfinite(pm->time) || pm->ncycle < 0, out_params.block_name,
                "invalid actual sample time/cycle");
  if (last_written_cycle_ >= 0 && pm->ncycle == last_written_cycle_ &&
      pm->time == last_written_time_) return;
  pending_time_ = pm->time;
  pending_cycle_ = pm->ncycle;
  rwz_->Load(pending_time_, pending_cycle_);
  pending_ = true;
}

//----------------------------------------------------------------------------------------
// Root confirms every active file row and flush before any rank advances output
// state. Cross-file rollback is not attempted if a later file fails.
void RWZOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) try {
  if (!pending_) return;
  // Larger codes give the original earlier validation categories priority across
  // ranks. Stage only locally valid prospective state, then reject globally once
  // before any file write or bookkeeping mutation.
  enum { valid_write, serialization_error, next_clock_error, clock_error, identity_error };
  int error = valid_write;
  int next_file_number = 0;
  Real next_last_time = 0;
  std::string sample_text, cadence_text;
  if (!std::isfinite(pending_time_) || pending_cycle_ < 0 ||
      pm->time != pending_time_ || pm->ncycle != pending_cycle_) {
    error = identity_error;
  } else if (out_params.file_number < 0 ||
             out_params.file_number == std::numeric_limits<int>::max() ||
             !std::isfinite(out_params.last_time) || !std::isfinite(out_params.dt)) {
    error = clock_error;
  } else {
    // The counter increment is reached only after its range check.
    next_file_number = out_params.file_number + 1;
    next_last_time = out_params.last_time < 0 ? pending_time_ :
                     out_params.last_time + out_params.dt;
    if (!std::isfinite(next_last_time)) {
      error = next_clock_error;
    } else {
      sample_text = EncodeRWZReal(pending_time_);
      cadence_text = EncodeRWZReal(next_last_time);
      if (sample_text.empty() || cadence_text.empty()) error = serialization_error;
    }
  }
#if MPI_PARALLEL_ENABLED
  if (MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS) {
    AbortRWZ(out_params.block_name, "could not reduce validation status");
  }
#endif
  if (error != valid_write) {
    const char *reasons[] = {"success",
        "could not serialize the sample identity or scheduling clock",
        "scheduling clock would become nonfinite",
        "file counter overflow or nonfinite cadence",
        "mesh identity changed after RWZ load"};
    AbortRWZ(out_params.block_name, reasons[error]);
  }

  int status = 0;
  if (global_variable::my_rank == rwz_->root_rank()) {
    status = rwz_->Write(pending_time_, pending_cycle_);
  }
#if MPI_PARALLEL_ENABLED
  if (MPI_Bcast(&status, 1, MPI_INT, rwz_->root_rank(), MPI_COMM_WORLD) != MPI_SUCCESS) {
    AbortRWZ(out_params.block_name, "could not broadcast writer status");
  }
#endif
  if (status != 0) AbortRWZ(out_params.block_name, "RWZ file output failed");

  last_written_time_ = pending_time_;
  last_written_cycle_ = pending_cycle_;
  out_params.last_time = next_last_time;
  out_params.file_number = next_file_number;
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetInteger(out_params.block_name, "rwz_last_written_cycle", last_written_cycle_);
  pin->SetString(out_params.block_name, "rwz_last_written_time", sample_text);
  pin->SetString(out_params.block_name, "last_time", cadence_text);
  pending_ = false;
} catch (...) {
  // A rank-local allocation/serialization exception must not strand peers in
  // the next validation or writer-status collective. No new collective here.
  AbortRWZ(out_params.block_name, "unexpected output bookkeeping failure");
}
