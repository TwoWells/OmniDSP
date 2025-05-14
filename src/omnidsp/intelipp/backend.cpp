/**
 * @file backend.cpp (IntelIPP)
 * @brief Implements the IntelIPP::Backend class methods.
 */

#include "backend.hpp"  // Corresponding header for IntelIPP::Backend

#include <ipps.h>  // Main IPP header, e.g., for ippsGetLibVersion

#include <OmniDSP/core_types.hpp>  // For Status, OmniExpected, OmniException, BackendType etc.
// Public Plan headers are not strictly needed here for defining Impl factories,
// but can be useful for context or if any public plan utilities were used.
// #include <OmniDSP/fft.hpp>
// #include <OmniDSP/filter.hpp>
// #include <OmniDSP/resample.hpp>
#include <OmniDSP/coefs/iir_filter.hpp>  // For Coefs::SOS (used in method signatures)
#include <OmniDSP/design/resample.hpp>  // For Design::Resample (used in method signatures)
#include <OmniDSP/window.hpp>  // For WindowSetup, Design::Resample (used in method signatures)
#include <exception>  // For std::exception, std::bad_alloc
#include <iostream>  // For error logging (std::cerr), though spdlog is preferred
#include <memory>    // For std::unique_ptr, std::make_unique
#include <new>       // For std::bad_alloc
#include <vector>    // For std::vector<> types like F32Vec etc.

// spdlog for logging
#include "spdlog/spdlog.h"

// Include necessary IPP backend implementation headers for PlanImpl classes
#include "fft.hpp"  // Defines IntelIPP::FFTPlanImpl, IntelIPP::RFFTPlanImpl
#include "fir_filter.hpp"  // Defines IntelIPP::FIRFilterPlanImpl
#include "iir_filter.hpp"  // Defines IntelIPP::IIRFilterPlanImpl
#include "resample.hpp"    // Defines IntelIPP::ResamplePlanImpl
#include "window.hpp"      // Defines the generate_*_window_intelipp helpers

// Include the interface backend header for Abstract::Backend (needed for
// factory func return type) This is already included via default/backend.hpp ->
// omnidsp/interface/backend.hpp #include "../interface/backend.hpp"

namespace OmniDSP::IntelIPP {

  //--------------------------------------------------------------------------
  // Backend Method Definitions
  //--------------------------------------------------------------------------

  Backend::Backend()
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    // Initialize IPP library (optional but recommended)
    // ippInit(); // Might be needed depending on IPP version/linking
    const IppLibraryVersion* libVersion = ippsGetLibVersion();
    if (libVersion) {
      logger->info(
          "Intel IPP backend initialized: {} {} (Build: {})",
          libVersion->Name,
          libVersion->Version,
          libVersion->BuildDate);
    }
    else {
      logger->warn("Could not retrieve Intel IPP library version info.");
    }
  }

  Backend::~Backend()
  {
    // Cleanup IPP if needed
    // auto logger = spdlog::get("OmniDSP");
    // if (logger) logger->trace("IntelIPP::Backend destructed.");
  }

  BackendType Backend::get_backend() const { return BackendType::IntelIPP; }

  // --- Overridden Window Generation ---
  // These call IPP-specific helper functions defined in intelipp/window.hpp/cpp
  [[nodiscard]] Status Backend::bartlett_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) return Status::SizeMismatch;
    return IntelIPP::generate_bartlett_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::bartlett_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) return Status::SizeMismatch;
    return IntelIPP::generate_bartlett_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::blackman_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) return Status::SizeMismatch;
    return IntelIPP::generate_blackman_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::blackman_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) return Status::SizeMismatch;
    return IntelIPP::generate_blackman_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::hamming_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) return Status::SizeMismatch;
    return IntelIPP::generate_hamming_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::hamming_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) return Status::SizeMismatch;
    return IntelIPP::generate_hamming_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::hann_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) return Status::SizeMismatch;
    return IntelIPP::generate_hann_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::hann_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) return Status::SizeMismatch;
    return IntelIPP::generate_hann_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::kaiser_window_f32(
      size_t length, double beta, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) return Status::SizeMismatch;
    if (beta < 0.0) return Status::InvalidArgument;  // Basic validation
    return IntelIPP::generate_kaiser_window_intelipp(
        beta, output.first(length));
  }
  [[nodiscard]] Status Backend::kaiser_window_f64(
      size_t length, double beta, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) return Status::SizeMismatch;
    if (beta < 0.0) return Status::InvalidArgument;
    return IntelIPP::generate_kaiser_window_intelipp(
        beta, output.first(length));
  }
  [[nodiscard]] Status Backend::rectangular_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) return Status::SizeMismatch;
    return IntelIPP::generate_rectangular_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::rectangular_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) return Status::SizeMismatch;
    return IntelIPP::generate_rectangular_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::triangular_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) return Status::SizeMismatch;
    return IntelIPP::generate_triangular_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::triangular_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) return Status::SizeMismatch;
    return IntelIPP::generate_triangular_window_intelipp(output.first(length));
  }

  // --- Overridden Plan Impl Factories ---
  // These create the IPP-specific *Impl objects.

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>
  Backend::create_fft_plan_impl_c32(size_t length) const
  {
    try {
      // Construct the IPP-specific FFTPlanImpl
      return std::make_unique<IntelIPP::FFTPlanImpl<C32>>(length);
    }
    catch (const OmniException& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fft_plan_impl_c32 failed: {} (Status: {})",
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::bad_alloc& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fft_plan_impl_c32 allocation failed: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>(
          std::unexpect, Status::AllocationError);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fft_plan_impl_c32 exception: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>(
          std::unexpect, Status::BackendError);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>
  Backend::create_fft_plan_impl_c64(size_t length) const
  {
    try {
      return std::make_unique<IntelIPP::FFTPlanImpl<C64>>(length);
    }
    catch (const OmniException& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fft_plan_impl_c64 failed: {} (Status: {})",
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::bad_alloc& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fft_plan_impl_c64 allocation failed: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>(
          std::unexpect, Status::AllocationError);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fft_plan_impl_c64 exception: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>(
          std::unexpect, Status::BackendError);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>
  Backend::create_rfft_plan_impl_f32(size_t length) const
  {
    try {
      return std::make_unique<IntelIPP::RFFTPlanImpl<F32>>(length);
    }
    catch (const OmniException& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_rfft_plan_impl_f32 failed: {} (Status: {})",
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::bad_alloc& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_rfft_plan_impl_f32 allocation failed: {}",
          e.what());
      return OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>(
          std::unexpect, Status::AllocationError);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_rfft_plan_impl_f32 exception: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>(
          std::unexpect, Status::BackendError);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>
  Backend::create_rfft_plan_impl_f64(size_t length) const
  {
    try {
      return std::make_unique<IntelIPP::RFFTPlanImpl<F64>>(length);
    }
    catch (const OmniException& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_rfft_plan_impl_f64 failed: {} (Status: {})",
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::bad_alloc& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_rfft_plan_impl_f64 allocation failed: {}",
          e.what());
      return OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>(
          std::unexpect, Status::AllocationError);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_rfft_plan_impl_f64 exception: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>(
          std::unexpect, Status::BackendError);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ResampleProcessorImpl<F32>>>
  Backend::create_resample_processor_impl_f32(
      const Design::Resample& spec) const
  {
    try {
      // Pass 'this' if the IPP ResampleProcessorImpl needs to access other
      // backend features (e.g. filter design) For now, assuming it's
      // self-contained or takes what it needs from Design::Resample.
      return std::make_unique<IntelIPP::ResampleProcessorImpl<F32>>(this, spec);
    }
    catch (const OmniException& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_resample_processor_impl_f32 failed: {} (Status: "
          "{})",
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<
          std::unique_ptr<Abstract::ResampleProcessorImpl<F32>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::bad_alloc& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_resample_processor_impl_f32 allocation failed: {}",
          e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::ResampleProcessorImpl<F32>>>(
          std::unexpect, Status::AllocationError);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_resample_processor_impl_f32 exception: {}",
          e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::ResampleProcessorImpl<F32>>>(
          std::unexpect, Status::BackendError);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ResampleProcessorImpl<F64>>>
  Backend::create_resample_processor_impl_f64(
      const Design::Resample& spec) const
  {
    try {
      return std::make_unique<IntelIPP::ResampleProcessorImpl<F64>>(this, spec);
    }
    catch (const OmniException& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_resample_processor_impl_f64 failed: {} (Status: "
          "{})",
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<
          std::unique_ptr<Abstract::ResampleProcessorImpl<F64>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::bad_alloc& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_resample_processor_impl_f64 allocation failed: {}",
          e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::ResampleProcessorImpl<F64>>>(
          std::unexpect, Status::AllocationError);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_resample_processor_impl_f64 exception: {}",
          e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::ResampleProcessorImpl<F64>>>(
          std::unexpect, Status::BackendError);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::FIRFilterProcessorImpl<F32>>>
  Backend::create_fir_filter_processor_impl_f32(
      const F32Vec& coefficients) const
  {
    try {
      return std::make_unique<IntelIPP::FIRFilterProcessorImpl<F32>>(
          coefficients);
    }
    catch (const OmniException& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fir_filter_processor_impl_f32 failed: {} (Status: "
          "{})",
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<
          std::unique_ptr<Abstract::FIRFilterProcessorImpl<F32>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::bad_alloc& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fir_filter_processor_impl_f32 allocation failed: "
          "{}",
          e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::FIRFilterProcessorImpl<F32>>>(
          std::unexpect, Status::AllocationError);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fir_filter_processor_impl_f32 exception: {}",
          e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::FIRFilterProcessorImpl<F32>>>(
          std::unexpect, Status::BackendError);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::FIRFilterProcessorImpl<F64>>>
  Backend::create_fir_filter_processor_impl_f64(
      const F64Vec& coefficients) const
  {
    try {
      return std::make_unique<IntelIPP::FIRFilterProcessorImpl<F64>>(
          coefficients);
    }
    catch (const OmniException& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fir_filter_processor_impl_f64 failed: {} (Status: "
          "{})",
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<
          std::unique_ptr<Abstract::FIRFilterProcessorImpl<F64>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::bad_alloc& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fir_filter_processor_impl_f64 allocation failed: "
          "{}",
          e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::FIRFilterProcessorImpl<F64>>>(
          std::unexpect, Status::AllocationError);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fir_filter_processor_impl_f64 exception: {}",
          e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::FIRFilterProcessorImpl<F64>>>(
          std::unexpect, Status::BackendError);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::FIRFilterProcessorImpl<C32>>>
  Backend::create_fir_filter_processor_impl_c32(
      const C32Vec& coefficients) const
  {
    try {
      return std::make_unique<IntelIPP::FIRFilterProcessorImpl<C32>>(
          coefficients);
    }
    catch (const OmniException& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fir_filter_processor_impl_c32 failed: {} (Status: "
          "{})",
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<
          std::unique_ptr<Abstract::FIRFilterProcessorImpl<C32>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::bad_alloc& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fir_filter_processor_impl_c32 allocation failed: "
          "{}",
          e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::FIRFilterProcessorImpl<C32>>>(
          std::unexpect, Status::AllocationError);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fir_filter_processor_impl_c32 exception: {}",
          e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::FIRFilterProcessorImpl<C32>>>(
          std::unexpect, Status::BackendError);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::FIRFilterProcessorImpl<C64>>>
  Backend::create_fir_filter_processor_impl_c64(
      const C64Vec& coefficients) const
  {
    try {
      return std::make_unique<IntelIPP::FIRFilterProcessorImpl<C64>>(
          coefficients);
    }
    catch (const OmniException& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fir_filter_processor_impl_c64 failed: {} (Status: "
          "{})",
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<
          std::unique_ptr<Abstract::FIRFilterProcessorImpl<C64>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::bad_alloc& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fir_filter_processor_impl_c64 allocation failed: "
          "{}",
          e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::FIRFilterProcessorImpl<C64>>>(
          std::unexpect, Status::AllocationError);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_fir_filter_processor_impl_c64 exception: {}",
          e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::FIRFilterProcessorImpl<C64>>>(
          std::unexpect, Status::BackendError);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::IIRFilterProcessorImpl<F32>>>
  Backend::create_iir_filter_processor_impl_f32(
      const Coefs::IIRFilterSOS& sos_coefficients) const
  {
    try {
      return std::make_unique<IntelIPP::IIRFilterProcessorImpl<F32>>(
          sos_coefficients);
    }
    catch (const OmniException& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_iir_filter_processor_impl_f32 failed: {} (Status: "
          "{})",
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<
          std::unique_ptr<Abstract::IIRFilterProcessorImpl<F32>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::bad_alloc& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_iir_filter_processor_impl_f32 allocation failed: "
          "{}",
          e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::IIRFilterProcessorImpl<F32>>>(
          std::unexpect, Status::AllocationError);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_iir_filter_processor_impl_f32 exception: {}",
          e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::IIRFilterProcessorImpl<F32>>>(
          std::unexpect, Status::BackendError);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::IIRFilterProcessorImpl<F64>>>
  Backend::create_iir_filter_processor_impl_f64(
      const Coefs::IIRFilterSOS& sos_coefficients) const
  {
    try {
      return std::make_unique<IntelIPP::IIRFilterProcessorImpl<F64>>(
          sos_coefficients);
    }
    catch (const OmniException& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_iir_filter_processor_impl_f64 failed: {} (Status: "
          "{})",
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<
          std::unique_ptr<Abstract::IIRFilterProcessorImpl<F64>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::bad_alloc& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_iir_filter_processor_impl_f64 allocation failed: "
          "{}",
          e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::IIRFilterProcessorImpl<F64>>>(
          std::unexpect, Status::AllocationError);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "IntelIPP::create_iir_filter_processor_impl_f64 exception: {}",
          e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::IIRFilterProcessorImpl<F64>>>(
          std::unexpect, Status::BackendError);
    }
  }

  // Convolution/Correlation plan impl and CQT processor impl factories and
  // Filter design methods are inherited from Default::Backend.

}  // namespace OmniDSP::IntelIPP

// --- Factory function for IntelIPP Backend ---
// This function is declared in "omnidsp/interface/backend.hpp"
// and defined here, in the IntelIPP backend's .cpp file.
namespace OmniDSP::Abstract {
  std::unique_ptr<Backend> create_intelipp_backend()
  {
    // Use the concrete class from the IntelIPP namespace
    return std::make_unique<IntelIPP::Backend>();
  }
}  // namespace OmniDSP::Abstract
