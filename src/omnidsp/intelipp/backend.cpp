/**
 * @file backend.cpp (IntelIPP)
 * @brief Implements the Backend class methods using Intel IPP.
 */

#include "backend.hpp"  // Includes default/backend.hpp now

#include <ipps.h>  // Include IPP header for ippsGetLibVersion

#include <OmniDSP/core_types.hpp>  // For Status, OmniExpected, OmniException, Backend enum etc.
#include <OmniDSP/fft.hpp>       // For FFTPlan, RFFTPlan public classes
#include <OmniDSP/filter.hpp>    // For FIR/IIR public classes, IIRFilterCoef
#include <OmniDSP/resample.hpp>  // For ResamplePlan public class, ResampleSpec
#include <OmniDSP/window.hpp>    // For WindowSpec
#include <exception>             // For std::exception, std::bad_alloc
#include <iostream>              // For error logging (std::cerr)
#include <memory>                // For std::unique_ptr, std::make_unique
#include <new>                   // For std::bad_alloc
#include <vector>                // For std::vector<> types like F32Vec etc.

// Include necessary IPP backend implementation headers
#include "fft.hpp"         // Defines IntelIPP FFT plan impls
#include "fir_filter.hpp"  // Defines IntelIPP Filter plan impls
#include "iir_filter.hpp"  // Defines IntelIPP Filter plan impls
#include "resample.hpp"    // Defines IntelIPP Resample plan impls
#include "window.hpp"      // Defines the generate_*_window_intelipp helpers

// Include the interface backend header for AbstractBackend (needed for factory
// func return type)
#include "../interface/backend.hpp"

namespace OmniDSP::IntelIPP {

  // Error Handling Macros remain useful
#define OMNI_RETURN_IPP_ERROR(T, status_code, message)                         \
  do {                                                                         \
    std::cerr << "Intel IPP Backend Error: " << message                        \
              << " (Status: " << static_cast<int>(status_code) << ")"          \
              << std::endl;                                                    \
    return OmniExpected<T>(std::unexpect, status_code);                        \
  }                                                                            \
  while (0)

#define OMNI_PROPAGATE_IPP_ERROR(T, expr)                                      \
  if (!(expr)) {                                                               \
    return OmniExpected<T>(std::unexpect, (expr).error());                     \
  }

  //--------------------------------------------------------------------------
  // Backend Method Definitions
  //--------------------------------------------------------------------------

  Backend::Backend()
  {
    // Initialize IPP library (optional but recommended)
    // ippInit(); // Might be needed depending on IPP version/linking
    const IppLibraryVersion* libVersion;
    libVersion = ippsGetLibVersion();  // Example: Get library info
    if (libVersion) {
      std::cout << "Intel IPP backend initialized: " << libVersion->Name << " "
                << libVersion->Version << " (Build: " << libVersion->BuildDate
                << ")" << std::endl;
    }
    else {
      std::cerr << "Warning: Could not retrieve Intel IPP library version info."
                << std::endl;
    }
  }

  Backend::~Backend()
  {
    // Cleanup IPP if needed (e.g., if ippInit was called)
  }

  BackendType Backend::get_backend() const
  {
    return BackendType::IntelIPP;  // Override to identify correctly
  }

  // --- DSP Operations (One-Off Implementations) ---
  // *** REMOVED *** Definitions for fft_c32, ifft_c32, rfft_f32, irfft_c32,
  // *** REMOVED *** convolve_*, correlate_* are now INHERITED from
  // DefaultBackend.
  // *** REMOVED *** They will automatically use the overridden IPP plan
  // factories below.

  // --- Overridden Window Generation ---
  // Provide definitions ONLY for functions marked 'override' in the header.

  [[nodiscard]] Status Backend::bartlett_window_f32(
      size_t length, std::span<F32> output) const
  {
    // Basic size check (optional, could rely on helper)
    if (output.size() < length) return Status::SizeMismatch;
    // Call the specific IPP helper function
    return IntelIPP::generate_bartlett_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::bartlett_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return IntelIPP::generate_bartlett_window_intelipp(output.first(length));
  }

  [[nodiscard]] Status Backend::blackman_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return IntelIPP::generate_blackman_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::blackman_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return IntelIPP::generate_blackman_window_intelipp(output.first(length));
  }

  // Flattop and Gaussian are NOT overridden - definitions inherited.

  [[nodiscard]] Status Backend::hamming_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return IntelIPP::generate_hamming_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::hamming_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return IntelIPP::generate_hamming_window_intelipp(output.first(length));
  }

  [[nodiscard]] Status Backend::hann_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return IntelIPP::generate_hann_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::hann_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return IntelIPP::generate_hann_window_intelipp(output.first(length));
  }

  [[nodiscard]] Status Backend::kaiser_window_f32(
      size_t length, double beta, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    // Add validation for beta if needed, although the helper might do it
    if (beta < 0.0) return Status::InvalidArgument;
    return IntelIPP::generate_kaiser_window_intelipp(
        beta, output.first(length));
  }
  [[nodiscard]] Status Backend::kaiser_window_f64(
      size_t length, double beta, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    if (beta < 0.0) return Status::InvalidArgument;
    return IntelIPP::generate_kaiser_window_intelipp(
        beta, output.first(length));
  }

  [[nodiscard]] Status Backend::rectangular_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return IntelIPP::generate_rectangular_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::rectangular_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return IntelIPP::generate_rectangular_window_intelipp(output.first(length));
  }

  [[nodiscard]] Status Backend::triangular_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    // IPP Triangular uses the Bartlett implementation helper
    return IntelIPP::generate_triangular_window_intelipp(output.first(length));
  }
  [[nodiscard]] Status Backend::triangular_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    // IPP Triangular uses the Bartlett implementation helper
    return IntelIPP::generate_triangular_window_intelipp(output.first(length));
  }

  // --- Overridden Plan Factories (Public API) ---
  // These create the public Plan<> handles using the IPP implementations.

  [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C32>>>
  Backend::create_fft_plan_c32(size_t length) const
  {
    try {
      // Create the IPP-specific implementation
      auto pimpl = std::make_unique<FFTPlanImpl<C32>>(length);
      // Wrap it in the public handle using the static helper
      return FFTPlan<C32>::create_from_impl(std::move(pimpl));
    }
    catch (const OmniException& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FFTPlan<C32>>, e.get_status(), e.what());
    }
    catch (const std::bad_alloc& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FFTPlan<C32>>, Status::AllocationError, e.what());
    }
    catch (const std::exception& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FFTPlan<C32>>, Status::Failure, e.what());
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C64>>>
  Backend::create_fft_plan_c64(size_t length) const
  {
    try {
      auto pimpl = std::make_unique<FFTPlanImpl<C64>>(length);
      return FFTPlan<C64>::create_from_impl(std::move(pimpl));
    }
    catch (const OmniException& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FFTPlan<C64>>, e.get_status(), e.what());
    }
    catch (const std::bad_alloc& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FFTPlan<C64>>, Status::AllocationError, e.what());
    }
    catch (const std::exception& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FFTPlan<C64>>, Status::Failure, e.what());
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F32>>>
  Backend::create_rfft_plan_f32(size_t length) const
  {
    try {
      auto pimpl
          = std::make_unique<IntelIPP::IntelIPPRFFTPlanImpl<F32>>(length);
      return RFFTPlan<F32>::create_from_impl(std::move(pimpl));
    }
    catch (const OmniException& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<RFFTPlan<F32>>, e.get_status(), e.what());
    }
    catch (const std::bad_alloc& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<RFFTPlan<F32>>, Status::AllocationError, e.what());
    }
    catch (const std::exception& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<RFFTPlan<F32>>, Status::Failure, e.what());
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F64>>>
  Backend::create_rfft_plan_f64(size_t length) const
  {
    try {
      auto pimpl
          = std::make_unique<IntelIPP::IntelIPPRFFTPlanImpl<F64>>(length);
      return RFFTPlan<F64>::create_from_impl(std::move(pimpl));
    }
    catch (const OmniException& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<RFFTPlan<F64>>, e.get_status(), e.what());
    }
    catch (const std::bad_alloc& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<RFFTPlan<F64>>, Status::AllocationError, e.what());
    }
    catch (const std::exception& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<RFFTPlan<F64>>, Status::Failure, e.what());
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<F32>>>
  Backend::create_resample_plan_f32(const ResampleSpec& spec) const
  {
    try {
      auto pimpl
          = std::make_unique<IntelIPP::ResamplePlanImpl<F32>>(this, spec);
      return ResamplePlan<F32>::create_from_impl(std::move(pimpl));
    }
    catch (const OmniException& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<ResamplePlan<F32>>, e.get_status(), e.what());
    }
    catch (const std::bad_alloc& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<ResamplePlan<F32>>,
          Status::AllocationError,
          e.what());
    }
    catch (const std::exception& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<ResamplePlan<F32>>, Status::Failure, e.what());
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<F64>>>
  Backend::create_resample_plan_f64(const ResampleSpec& spec) const
  {
    try {
      auto pimpl
          = std::make_unique<IntelIPP::ResamplePlanImpl<F64>>(this, spec);
      return ResamplePlan<F64>::create_from_impl(std::move(pimpl));
    }
    catch (const OmniException& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<ResamplePlan<F64>>, e.get_status(), e.what());
    }
    catch (const std::bad_alloc& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<ResamplePlan<F64>>,
          Status::AllocationError,
          e.what());
    }
    catch (const std::exception& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<ResamplePlan<F64>>, Status::Failure, e.what());
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<F32>>>
  Backend::create_fir_filter_plan_f32(const F32Vec& coefficients) const
  {
    try {
      auto pimpl
          = std::make_unique<IntelIPP::FIRFilterPlanImpl<F32>>(coefficients);
      return FIRFilterPlan<F32>::create_from_impl(std::move(pimpl));
    }
    catch (const OmniException& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FIRFilterPlan<F32>>, e.get_status(), e.what());
    }
    catch (const std::bad_alloc& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FIRFilterPlan<F32>>,
          Status::AllocationError,
          e.what());
    }
    catch (const std::exception& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FIRFilterPlan<F32>>, Status::Failure, e.what());
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<F64>>>
  Backend::create_fir_filter_plan_f64(const F64Vec& coefficients) const
  {
    try {
      auto pimpl
          = std::make_unique<IntelIPP::FIRFilterPlanImpl<F64>>(coefficients);
      return FIRFilterPlan<F64>::create_from_impl(std::move(pimpl));
    }
    catch (const OmniException& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FIRFilterPlan<F64>>, e.get_status(), e.what());
    }
    catch (const std::bad_alloc& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FIRFilterPlan<F64>>,
          Status::AllocationError,
          e.what());
    }
    catch (const std::exception& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FIRFilterPlan<F64>>, Status::Failure, e.what());
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<C32>>>
  Backend::create_fir_filter_plan_c32(const C32Vec& coefficients) const
  {
    try {
      auto pimpl
          = std::make_unique<IntelIPP::FIRFilterPlanImpl<C32>>(coefficients);
      return FIRFilterPlan<C32>::create_from_impl(std::move(pimpl));
    }
    catch (const OmniException& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FIRFilterPlan<C32>>, e.get_status(), e.what());
    }
    catch (const std::bad_alloc& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FIRFilterPlan<C32>>,
          Status::AllocationError,
          e.what());
    }
    catch (const std::exception& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FIRFilterPlan<C32>>, Status::Failure, e.what());
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<C64>>>
  Backend::create_fir_filter_plan_c64(const C64Vec& coefficients) const
  {
    try {
      auto pimpl
          = std::make_unique<IntelIPP::FIRFilterPlanImpl<C64>>(coefficients);
      return FIRFilterPlan<C64>::create_from_impl(std::move(pimpl));
    }
    catch (const OmniException& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FIRFilterPlan<C64>>, e.get_status(), e.what());
    }
    catch (const std::bad_alloc& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FIRFilterPlan<C64>>,
          Status::AllocationError,
          e.what());
    }
    catch (const std::exception& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<FIRFilterPlan<C64>>, Status::Failure, e.what());
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F32>>>
  Backend::create_iir_filter_plan_f32(
      const std::vector<IIRFilterCoef>& sos_coefficients) const
  {
    try {
      auto pimpl = std::make_unique<IntelIPP::IIRFilterPlanImpl<F32>>(
          sos_coefficients);
      return IIRFilterPlan<F32>::create_from_impl(std::move(pimpl));
    }
    catch (const OmniException& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<IIRFilterPlan<F32>>, e.get_status(), e.what());
    }
    catch (const std::bad_alloc& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<IIRFilterPlan<F32>>,
          Status::AllocationError,
          e.what());
    }
    catch (const std::exception& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<IIRFilterPlan<F32>>, Status::Failure, e.what());
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F64>>>
  Backend::create_iir_filter_plan_f64(
      const std::vector<IIRFilterCoef>& sos_coefficients) const
  {
    try {
      auto pimpl = std::make_unique<IntelIPP::IIRFilterPlanImpl<F64>>(
          sos_coefficients);
      return IIRFilterPlan<F64>::create_from_impl(std::move(pimpl));
    }
    catch (const OmniException& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<IIRFilterPlan<F64>>, e.get_status(), e.what());
    }
    catch (const std::bad_alloc& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<IIRFilterPlan<F64>>,
          Status::AllocationError,
          e.what());
    }
    catch (const std::exception& e) {
      OMNI_RETURN_IPP_ERROR(
          std::unique_ptr<IIRFilterPlan<F64>>, Status::Failure, e.what());
    }
  }

  // Definitions for Convolution/Correlation/CQT plan factories are inherited.
  // Definitions for Filter design methods are inherited.

}  // namespace OmniDSP::IntelIPP

// This function needs to be defined in the global OmniDSP::abstract namespace
// as declared in interface/backend.hpp
namespace OmniDSP::Abstract {
  std::unique_ptr<Backend> create_intelipp_backend()
  {
    // Use the concrete class from the IntelIPP namespace
    return std::make_unique<::OmniDSP::IntelIPP::Backend>();
  }
}  // namespace OmniDSP::Abstract
