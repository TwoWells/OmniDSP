/**
 * @file backend.cpp (default)
 * @brief Implements the DefaultBackend class methods using standard C++.
 */

#include "backend.hpp"  // Corresponding header for Default backend declarations

#include "convolution.hpp"  // DefaultConvolutionPlanImpl, DefaultCorrelationPlanImpl
#include "cqt.hpp"          // DefaultCQTPlanImpl
#include "fft.hpp"          // DefaultFFTPlanImpl, DefaultRFFTPlanImpl
#include "filter.hpp"    // DefaultFIRFilterPlanImpl, DefaultIIRFilterPlanImpl
#include "resample.hpp"  // DefaultResamplePlanImpl
#include "window.hpp"  // Default window implementations (span-based) - needed for generate_window

// Include headers for the public Plan classes (needed for factory return types
// and Pimpl access/construction)
#include <OmniDSP/convolution.hpp>  // Includes ConvolutionPlan and CorrelationPlan
#include <OmniDSP/core_types.hpp>
#include <OmniDSP/cqt.hpp>
#include <OmniDSP/fft.hpp>     // Includes FFTPlan, RFFTPlan
#include <OmniDSP/filter.hpp>  // Needed for non-templated FIRFilterSpec, IIRFilterSpec, IIRFilterCoef, FIRFilterPlan, IIRFilterPlan
#include <OmniDSP/resample.hpp>  // Includes ResamplePlan, ResampleSpec
#include <OmniDSP/window.hpp>    // Needed for WindowSpec

// Include standard library headers needed for default implementations
#include <algorithm>
#include <cmath>
#include <complex>
#include <expected>
#include <iostream>
#include <limits>  // For numeric_limits
#include <memory>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>  // Needed for variant in factories
#include <vector>

// Include Boost Bessel function for Kaiser window (needed by default impl in
// window.hpp/cpp) This might not be strictly needed here if generate_window is
// fully defined elsewhere #include <boost/math/special_functions/bessel.hpp>

namespace OmniDSP {
  namespace backend {

    // --- Helper Function from convolution.cpp ---
    // (Ensure this helper is available, perhaps move to a common utility
    // header)
    namespace convolution_detail {
      inline size_t next_power_of_two(size_t n)
      {
        if (n == 0) return 1;  // Smallest valid FFT size often 1 or 2
        if ((n > 0) && ((n & (n - 1)) == 0)) {
          return n;
        }
        if (n > (std::numeric_limits<size_t>::max() / 2U)) {
          if (n == (~(std::numeric_limits<size_t>::max() >> 1))) return n;
          return std::numeric_limits<size_t>::max();
        }
        size_t power = 1;
        while (power < n) {
          if (power > (std::numeric_limits<size_t>::max() / 2U))
            return std::numeric_limits<size_t>::max();
          power <<= 1;
          if (power == 0)
            return std::numeric_limits<size_t>::max();  // Check overflow
        }
        return power;
      }
    }  // namespace convolution_detail

    /** @brief Helper function to check a condition and return a status. */
    inline Status check_status(
        bool condition,
        Status error_status,
        const std::string& error_message = "")
    { /* ... implementation ... */
      if (!condition) {
        if (!error_message.empty()) {
          std::cerr << "OmniDSP Default Backend Error: " << error_message
                    << std::endl;
        }
        return error_status;
      }
      return Status::Success;
    }

//--------------------------------------------------------------------------
// Error Handling Macros
//--------------------------------------------------------------------------
#define OMNI_PROPAGATE_ERROR(T, expr)                                          \
  if (!(expr)) {                                                               \
    return OmniExpected<T>(std::unexpect, (expr).error());                     \
  }
#define OMNI_RETURN_ERROR(T, status_code)                                      \
  return OmniExpected<T>(std::unexpect, status_code);

    //--------------------------------------------------------------------------
    // DefaultBackend Method Definitions
    //--------------------------------------------------------------------------
    DefaultBackend::DefaultBackend() {}
    DefaultBackend::~DefaultBackend() {}
    Backend DefaultBackend::get_backend() const { return Backend::Default; }

    // --- DSP Operations (One-Off Implementations) ---
    // Calls to create_*_plan_impl_* are correct here (3 args)
    [[nodiscard]] OmniExpected<F32Vec> DefaultBackend::convolve_f32(
        const F32Vec& input,
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      auto plan_expected
          = create_convolution_plan_impl_f32(kernel, type, method);
      OMNI_PROPAGATE_ERROR(F32Vec, plan_expected);
      auto& plan = plan_expected.value();
      size_t output_len = plan->get_output_length(input.size());
      F32Vec output(output_len);
      Status status = plan->execute(std::span(input), std::span(output));
      if (status != Status::Success) {
        OMNI_RETURN_ERROR(F32Vec, status);
      }
      return output;
    }
    [[nodiscard]] OmniExpected<F64Vec> DefaultBackend::convolve_f64(
        const F64Vec& input,
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      auto plan_expected
          = create_convolution_plan_impl_f64(kernel, type, method);
      OMNI_PROPAGATE_ERROR(F64Vec, plan_expected);
      auto& plan = plan_expected.value();
      size_t output_len = plan->get_output_length(input.size());
      F64Vec output(output_len);
      Status status = plan->execute(std::span(input), std::span(output));
      if (status != Status::Success) {
        OMNI_RETURN_ERROR(F64Vec, status);
      }
      return output;
    }
    [[nodiscard]] OmniExpected<C32Vec> DefaultBackend::convolve_c32(
        const C32Vec& input,
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      auto plan_expected
          = create_convolution_plan_impl_c32(kernel, type, method);
      OMNI_PROPAGATE_ERROR(C32Vec, plan_expected);
      auto& plan = plan_expected.value();
      size_t output_len = plan->get_output_length(input.size());
      C32Vec output(output_len);
      Status status = plan->execute(std::span(input), std::span(output));
      if (status != Status::Success) {
        OMNI_RETURN_ERROR(C32Vec, status);
      }
      return output;
    }
    [[nodiscard]] OmniExpected<C64Vec> DefaultBackend::convolve_c64(
        const C64Vec& input,
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      auto plan_expected
          = create_convolution_plan_impl_c64(kernel, type, method);
      OMNI_PROPAGATE_ERROR(C64Vec, plan_expected);
      auto& plan = plan_expected.value();
      size_t output_len = plan->get_output_length(input.size());
      C64Vec output(output_len);
      Status status = plan->execute(std::span(input), std::span(output));
      if (status != Status::Success) {
        OMNI_RETURN_ERROR(C64Vec, status);
      }
      return output;
    }
    [[nodiscard]] OmniExpected<F32Vec> DefaultBackend::correlate_f32(
        const F32Vec& input,
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      auto plan_expected
          = create_correlation_plan_impl_f32(kernel, type, method);
      OMNI_PROPAGATE_ERROR(F32Vec, plan_expected);
      auto& plan = plan_expected.value();
      size_t output_len = plan->get_output_length(input.size());
      F32Vec output(output_len);
      Status status = plan->execute(std::span(input), std::span(output));
      if (status != Status::Success) {
        OMNI_RETURN_ERROR(F32Vec, status);
      }
      return output;
    }
    [[nodiscard]] OmniExpected<F64Vec> DefaultBackend::correlate_f64(
        const F64Vec& input,
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      auto plan_expected
          = create_correlation_plan_impl_f64(kernel, type, method);
      OMNI_PROPAGATE_ERROR(F64Vec, plan_expected);
      auto& plan = plan_expected.value();
      size_t output_len = plan->get_output_length(input.size());
      F64Vec output(output_len);
      Status status = plan->execute(std::span(input), std::span(output));
      if (status != Status::Success) {
        OMNI_RETURN_ERROR(F64Vec, status);
      }
      return output;
    }
    [[nodiscard]] OmniExpected<C32Vec> DefaultBackend::correlate_c32(
        const C32Vec& input,
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      auto plan_expected
          = create_correlation_plan_impl_c32(kernel, type, method);
      OMNI_PROPAGATE_ERROR(C32Vec, plan_expected);
      auto& plan = plan_expected.value();
      size_t output_len = plan->get_output_length(input.size());
      C32Vec output(output_len);
      Status status = plan->execute(std::span(input), std::span(output));
      if (status != Status::Success) {
        OMNI_RETURN_ERROR(C32Vec, status);
      }
      return output;
    }
    [[nodiscard]] OmniExpected<C64Vec> DefaultBackend::correlate_c64(
        const C64Vec& input,
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      auto plan_expected
          = create_correlation_plan_impl_c64(kernel, type, method);
      OMNI_PROPAGATE_ERROR(C64Vec, plan_expected);
      auto& plan = plan_expected.value();
      size_t output_len = plan->get_output_length(input.size());
      C64Vec output(output_len);
      Status status = plan->execute(std::span(input), std::span(output));
      if (status != Status::Success) {
        OMNI_RETURN_ERROR(C64Vec, status);
      }
      return output;
    }
    [[nodiscard]] OmniExpected<C32Vec> DefaultBackend::fft_c32(
        const C32Vec& input) const
    {
      auto plan_expected = create_fft_plan_impl_c32(input.size());
      OMNI_PROPAGATE_ERROR(C32Vec, plan_expected);
      auto& plan = plan_expected.value();
      C32Vec output(input.size());
      Status status = plan->fft(std::span(input), std::span(output));
      if (status != Status::Success) {
        OMNI_RETURN_ERROR(C32Vec, status);
      }
      return output;
    }
    [[nodiscard]] OmniExpected<C64Vec> DefaultBackend::fft_c64(
        const C64Vec& input) const
    {
      auto plan_expected = create_fft_plan_impl_c64(input.size());
      OMNI_PROPAGATE_ERROR(C64Vec, plan_expected);
      auto& plan = plan_expected.value();
      C64Vec output(input.size());
      Status status = plan->fft(std::span(input), std::span(output));
      if (status != Status::Success) {
        OMNI_RETURN_ERROR(C64Vec, status);
      }
      return output;
    }
    [[nodiscard]] OmniExpected<C32Vec> DefaultBackend::ifft_c32(
        const C32Vec& input) const
    {
      auto plan_expected = create_fft_plan_impl_c32(input.size());
      OMNI_PROPAGATE_ERROR(C32Vec, plan_expected);
      auto& plan = plan_expected.value();
      C32Vec output(input.size());
      Status status = plan->ifft(std::span(input), std::span(output));
      if (status != Status::Success) {
        OMNI_RETURN_ERROR(C32Vec, status);
      }
      return output;
    }
    [[nodiscard]] OmniExpected<C64Vec> DefaultBackend::ifft_c64(
        const C64Vec& input) const
    {
      auto plan_expected = create_fft_plan_impl_c64(input.size());
      OMNI_PROPAGATE_ERROR(C64Vec, plan_expected);
      auto& plan = plan_expected.value();
      C64Vec output(input.size());
      Status status = plan->ifft(std::span(input), std::span(output));
      if (status != Status::Success) {
        OMNI_RETURN_ERROR(C64Vec, status);
      }
      return output;
    }
    [[nodiscard]] OmniExpected<C32Vec> DefaultBackend::rfft_f32(
        const F32Vec& input) const
    {
      auto plan_expected = create_rfft_plan_impl_f32(input.size());
      OMNI_PROPAGATE_ERROR(C32Vec, plan_expected);
      auto& plan = plan_expected.value();
      size_t output_len = (input.empty()) ? 0 : (input.size() / 2) + 1;
      C32Vec output(output_len);
      Status status = plan->rfft(std::span(input), std::span(output));
      if (status != Status::Success) {
        OMNI_RETURN_ERROR(C32Vec, status);
      }
      return output;
    }
    [[nodiscard]] OmniExpected<C64Vec> DefaultBackend::rfft_f64(
        const F64Vec& input) const
    {
      auto plan_expected = create_rfft_plan_impl_f64(input.size());
      OMNI_PROPAGATE_ERROR(C64Vec, plan_expected);
      auto& plan = plan_expected.value();
      size_t output_len = (input.empty()) ? 0 : (input.size() / 2) + 1;
      C64Vec output(output_len);
      Status status = plan->rfft(std::span(input), std::span(output));
      if (status != Status::Success) {
        OMNI_RETURN_ERROR(C64Vec, status);
      }
      return output;
    }
    [[nodiscard]] OmniExpected<F32Vec> DefaultBackend::irfft_c32(
        const C32Vec& input, size_t output_length) const
    {
      if (output_length == 0) {
        return input.empty() ? OmniExpected<F32Vec>(F32Vec{})
                             : OmniExpected<F32Vec>(
                                   std::unexpect, Status::InvalidArgument);
      }
      size_t expected_input_len = (output_length / 2) + 1;
      if (input.size() != expected_input_len) {
        std::cerr << "Default irfft_c32 error: Input size (" << input.size()
                  << ") does not match expected size (" << expected_input_len
                  << ") for output length " << output_length << "."
                  << std::endl;
        OMNI_RETURN_ERROR(F32Vec, Status::SizeMismatch);
      }
      auto plan_expected = create_rfft_plan_impl_f32(output_length);
      OMNI_PROPAGATE_ERROR(F32Vec, plan_expected);
      auto& plan = plan_expected.value();
      F32Vec output(output_length);
      Status status = plan->irfft(std::span(input), std::span(output));
      if (status != Status::Success) {
        OMNI_RETURN_ERROR(F32Vec, status);
      }
      return output;
    }
    [[nodiscard]] OmniExpected<F64Vec> DefaultBackend::irfft_c64(
        const C64Vec& input, size_t output_length) const
    {
      if (output_length == 0) {
        return input.empty() ? OmniExpected<F64Vec>(F64Vec{})
                             : OmniExpected<F64Vec>(
                                   std::unexpect, Status::InvalidArgument);
      }
      size_t expected_input_len = (output_length / 2) + 1;
      if (input.size() != expected_input_len) {
        std::cerr << "Default irfft_c64 error: Input size (" << input.size()
                  << ") does not match expected size (" << expected_input_len
                  << ") for output length " << output_length << "."
                  << std::endl;
        OMNI_RETURN_ERROR(F64Vec, Status::SizeMismatch);
      }
      auto plan_expected = create_rfft_plan_impl_f64(output_length);
      OMNI_PROPAGATE_ERROR(F64Vec, plan_expected);
      auto& plan = plan_expected.value();
      F64Vec output(output_length);
      Status status = plan->irfft(std::span(input), std::span(output));
      if (status != Status::Success) {
        OMNI_RETURN_ERROR(F64Vec, status);
      }
      return output;
    }

    // --- Window Generation (Public API Implementations) ---
    // These call the internal generate_window_* which then calls the span-based
    // helpers
    [[nodiscard]] Status DefaultBackend::bartlett_window_f32(
        size_t length, std::span<F32> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      return generate_window_f32(WindowSpec(WindowType::Bartlett), target_span);
    }
    [[nodiscard]] Status DefaultBackend::bartlett_window_f64(
        size_t length, std::span<F64> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      return generate_window_f64(WindowSpec(WindowType::Bartlett), target_span);
    }
    [[nodiscard]] Status DefaultBackend::blackman_window_f32(
        size_t length, std::span<F32> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      return generate_window_f32(WindowSpec(WindowType::Blackman), target_span);
    }
    [[nodiscard]] Status DefaultBackend::blackman_window_f64(
        size_t length, std::span<F64> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      return generate_window_f64(WindowSpec(WindowType::Blackman), target_span);
    }
    [[nodiscard]] Status DefaultBackend::flattop_window_f32(
        size_t length, std::span<F32> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      return generate_window_f32(WindowSpec(WindowType::Flattop), target_span);
    }
    [[nodiscard]] Status DefaultBackend::flattop_window_f64(
        size_t length, std::span<F64> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      return generate_window_f64(WindowSpec(WindowType::Flattop), target_span);
    }
    [[nodiscard]] Status DefaultBackend::gaussian_window_f32(
        size_t length, double stddev, std::span<F32> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      try {
        return generate_window_f32(WindowSpec::Gaussian(stddev), target_span);
      }
      catch (const std::invalid_argument&) {
        return Status::InvalidArgument;
      }
    }
    [[nodiscard]] Status DefaultBackend::gaussian_window_f64(
        size_t length, double stddev, std::span<F64> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      try {
        return generate_window_f64(WindowSpec::Gaussian(stddev), target_span);
      }
      catch (const std::invalid_argument&) {
        return Status::InvalidArgument;
      }
    }
    [[nodiscard]] Status DefaultBackend::hamming_window_f32(
        size_t length, std::span<F32> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      return generate_window_f32(WindowSpec(WindowType::Hamming), target_span);
    }
    [[nodiscard]] Status DefaultBackend::hamming_window_f64(
        size_t length, std::span<F64> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      return generate_window_f64(WindowSpec(WindowType::Hamming), target_span);
    }
    [[nodiscard]] Status DefaultBackend::hann_window_f32(
        size_t length, std::span<F32> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      return generate_window_f32(WindowSpec(WindowType::Hann), target_span);
    }
    [[nodiscard]] Status DefaultBackend::hann_window_f64(
        size_t length, std::span<F64> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      return generate_window_f64(WindowSpec(WindowType::Hann), target_span);
    }
    [[nodiscard]] Status DefaultBackend::kaiser_window_f32(
        size_t length, double beta, std::span<F32> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      try {
        return generate_window_f32(WindowSpec::Kaiser(beta), target_span);
      }
      catch (const std::invalid_argument&) {
        return Status::InvalidArgument;
      }
    }
    [[nodiscard]] Status DefaultBackend::kaiser_window_f64(
        size_t length, double beta, std::span<F64> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      try {
        return generate_window_f64(WindowSpec::Kaiser(beta), target_span);
      }
      catch (const std::invalid_argument&) {
        return Status::InvalidArgument;
      }
    }
    [[nodiscard]] Status DefaultBackend::rectangular_window_f32(
        size_t length, std::span<F32> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      return generate_window_f32(
          WindowSpec(WindowType::Rectangular), target_span);
    }
    [[nodiscard]] Status DefaultBackend::rectangular_window_f64(
        size_t length, std::span<F64> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      return generate_window_f64(
          WindowSpec(WindowType::Rectangular), target_span);
    }
    [[nodiscard]] Status DefaultBackend::triangular_window_f32(
        size_t length, std::span<F32> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      return generate_window_f32(
          WindowSpec(WindowType::Triangular), target_span);
    }
    [[nodiscard]] Status DefaultBackend::triangular_window_f64(
        size_t length, std::span<F64> output) const
    {
      if (output.size() < length) return Status::SizeMismatch;
      auto target_span = output.first(length);
      return generate_window_f64(
          WindowSpec(WindowType::Triangular), target_span);
    }

    // --- Plan Factories (Implementations) ---
    // Use the static create_from_impl helper on the public Plan class

    // *** UPDATED: Return types and create_from_impl calls corrected ***
    [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C32>>>
    DefaultBackend::create_fft_plan_c32(size_t length) const
    {
      auto pimpl_expected = create_fft_plan_impl_c32(length);
      OMNI_PROPAGATE_ERROR(std::unique_ptr<FFTPlan<C32>>, pimpl_expected);
      return FFTPlan<C32>::create_from_impl(std::move(pimpl_expected.value()));
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C64>>>
    DefaultBackend::create_fft_plan_c64(size_t length) const
    {
      auto pimpl_expected = create_fft_plan_impl_c64(length);
      OMNI_PROPAGATE_ERROR(std::unique_ptr<FFTPlan<C64>>, pimpl_expected);
      return FFTPlan<C64>::create_from_impl(std::move(pimpl_expected.value()));
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F32>>>
    DefaultBackend::create_rfft_plan_f32(size_t length) const
    {
      auto pimpl_expected = create_rfft_plan_impl_f32(length);
      OMNI_PROPAGATE_ERROR(std::unique_ptr<RFFTPlan<F32>>, pimpl_expected);
      return RFFTPlan<F32>::create_from_impl(std::move(pimpl_expected.value()));
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F64>>>
    DefaultBackend::create_rfft_plan_f64(size_t length) const
    {
      auto pimpl_expected = create_rfft_plan_impl_f64(length);
      OMNI_PROPAGATE_ERROR(std::unique_ptr<RFFTPlan<F64>>, pimpl_expected);
      return RFFTPlan<F64>::create_from_impl(std::move(pimpl_expected.value()));
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<F32>>>
    DefaultBackend::create_cqt_plan_f32(
        F32 sample_rate,
        F32 min_freq,
        F32 max_freq,
        int bins_per_octave,
        const WindowSpec& window_spec) const
    {
      auto pimpl_expected = create_cqt_plan_impl_f32(
          sample_rate, min_freq, max_freq, bins_per_octave, window_spec);
      OMNI_PROPAGATE_ERROR(std::unique_ptr<CQTPlan<F32>>, pimpl_expected);
      return CQTPlan<F32>::create_from_impl(std::move(pimpl_expected.value()));
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<F64>>>
    DefaultBackend::create_cqt_plan_f64(
        F64 sample_rate,
        F64 min_freq,
        F64 max_freq,
        int bins_per_octave,
        const WindowSpec& window_spec) const
    {
      auto pimpl_expected = create_cqt_plan_impl_f64(
          sample_rate, min_freq, max_freq, bins_per_octave, window_spec);
      OMNI_PROPAGATE_ERROR(std::unique_ptr<CQTPlan<F64>>, pimpl_expected);
      return CQTPlan<F64>::create_from_impl(std::move(pimpl_expected.value()));
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<F32>>>
    DefaultBackend::create_resample_plan_f32(const ResampleSpec& spec) const
    {
      try {
        auto pimpl = std::make_unique<DefaultResamplePlanImpl<F32>>(this, spec);
        return ResamplePlan<F32>::create_from_impl(std::move(pimpl));
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Default ResamplePlan<F32>: " << e.what()
                  << std::endl;
        OMNI_RETURN_ERROR(std::unique_ptr<ResamplePlan<F32>>, Status::Failure);
      }
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<F64>>>
    DefaultBackend::create_resample_plan_f64(const ResampleSpec& spec) const
    {
      try {
        auto pimpl = std::make_unique<DefaultResamplePlanImpl<F64>>(this, spec);
        return ResamplePlan<F64>::create_from_impl(std::move(pimpl));
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Default ResamplePlan<F64>: " << e.what()
                  << std::endl;
        OMNI_RETURN_ERROR(std::unique_ptr<ResamplePlan<F64>>, Status::Failure);
      }
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<F32>>>
    DefaultBackend::create_convolution_plan_f32(
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      auto pimpl_expected
          = create_convolution_plan_impl_f32(kernel, type, method);
      OMNI_PROPAGATE_ERROR(
          std::unique_ptr<ConvolutionPlan<F32>>, pimpl_expected);
      return ConvolutionPlan<F32>::create_from_impl(
          std::move(pimpl_expected.value()));
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<F64>>>
    DefaultBackend::create_convolution_plan_f64(
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      auto pimpl_expected
          = create_convolution_plan_impl_f64(kernel, type, method);
      OMNI_PROPAGATE_ERROR(
          std::unique_ptr<ConvolutionPlan<F64>>, pimpl_expected);
      return ConvolutionPlan<F64>::create_from_impl(
          std::move(pimpl_expected.value()));
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<C32>>>
    DefaultBackend::create_convolution_plan_c32(
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      auto pimpl_expected
          = create_convolution_plan_impl_c32(kernel, type, method);
      OMNI_PROPAGATE_ERROR(
          std::unique_ptr<ConvolutionPlan<C32>>, pimpl_expected);
      return ConvolutionPlan<C32>::create_from_impl(
          std::move(pimpl_expected.value()));
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<C64>>>
    DefaultBackend::create_convolution_plan_c64(
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      auto pimpl_expected
          = create_convolution_plan_impl_c64(kernel, type, method);
      OMNI_PROPAGATE_ERROR(
          std::unique_ptr<ConvolutionPlan<C64>>, pimpl_expected);
      return ConvolutionPlan<C64>::create_from_impl(
          std::move(pimpl_expected.value()));
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<F32>>>
    DefaultBackend::create_correlation_plan_f32(
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      auto pimpl_expected
          = create_correlation_plan_impl_f32(kernel, type, method);
      OMNI_PROPAGATE_ERROR(
          std::unique_ptr<CorrelationPlan<F32>>, pimpl_expected);
      return CorrelationPlan<F32>::create_from_impl(
          std::move(pimpl_expected.value()));
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<F64>>>
    DefaultBackend::create_correlation_plan_f64(
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      auto pimpl_expected
          = create_correlation_plan_impl_f64(kernel, type, method);
      OMNI_PROPAGATE_ERROR(
          std::unique_ptr<CorrelationPlan<F64>>, pimpl_expected);
      return CorrelationPlan<F64>::create_from_impl(
          std::move(pimpl_expected.value()));
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<C32>>>
    DefaultBackend::create_correlation_plan_c32(
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      auto pimpl_expected
          = create_correlation_plan_impl_c32(kernel, type, method);
      OMNI_PROPAGATE_ERROR(
          std::unique_ptr<CorrelationPlan<C32>>, pimpl_expected);
      return CorrelationPlan<C32>::create_from_impl(
          std::move(pimpl_expected.value()));
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<C64>>>
    DefaultBackend::create_correlation_plan_c64(
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      auto pimpl_expected
          = create_correlation_plan_impl_c64(kernel, type, method);
      OMNI_PROPAGATE_ERROR(
          std::unique_ptr<CorrelationPlan<C64>>, pimpl_expected);
      return CorrelationPlan<C64>::create_from_impl(
          std::move(pimpl_expected.value()));
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<F32>>>
    DefaultBackend::create_fir_filter_plan_f32(const F32Vec& coefficients) const
    {
      try {
        auto pimpl
            = std::make_unique<DefaultFIRFilterPlanImpl<F32>>(coefficients);
        return FIRFilterPlan<F32>::create_from_impl(std::move(pimpl));
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Default FIRFilterPlan<F32>: " << e.what()
                  << std::endl;
        OMNI_RETURN_ERROR(std::unique_ptr<FIRFilterPlan<F32>>, Status::Failure);
      }
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<F64>>>
    DefaultBackend::create_fir_filter_plan_f64(const F64Vec& coefficients) const
    {
      try {
        auto pimpl
            = std::make_unique<DefaultFIRFilterPlanImpl<F64>>(coefficients);
        return FIRFilterPlan<F64>::create_from_impl(std::move(pimpl));
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Default FIRFilterPlan<F64>: " << e.what()
                  << std::endl;
        OMNI_RETURN_ERROR(std::unique_ptr<FIRFilterPlan<F64>>, Status::Failure);
      }
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<C32>>>
    DefaultBackend::create_fir_filter_plan_c32(const C32Vec& coefficients) const
    {
      try {
        auto pimpl
            = std::make_unique<DefaultFIRFilterPlanImpl<C32>>(coefficients);
        return FIRFilterPlan<C32>::create_from_impl(std::move(pimpl));
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Default FIRFilterPlan<C32>: " << e.what()
                  << std::endl;
        OMNI_RETURN_ERROR(std::unique_ptr<FIRFilterPlan<C32>>, Status::Failure);
      }
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<C64>>>
    DefaultBackend::create_fir_filter_plan_c64(const C64Vec& coefficients) const
    {
      try {
        auto pimpl
            = std::make_unique<DefaultFIRFilterPlanImpl<C64>>(coefficients);
        return FIRFilterPlan<C64>::create_from_impl(std::move(pimpl));
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Default FIRFilterPlan<C64>: " << e.what()
                  << std::endl;
        OMNI_RETURN_ERROR(std::unique_ptr<FIRFilterPlan<C64>>, Status::Failure);
      }
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F32>>>
    DefaultBackend::create_iir_filter_plan_f32(
        const std::vector<IIRFilterCoef>& sos_coefficients) const
    {
      try {
        auto pimpl
            = std::make_unique<DefaultIIRFilterPlanImpl<F32>>(sos_coefficients);
        return IIRFilterPlan<F32>::create_from_impl(std::move(pimpl));
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Default IIRFilterPlan<F32>: " << e.what()
                  << std::endl;
        OMNI_RETURN_ERROR(std::unique_ptr<IIRFilterPlan<F32>>, Status::Failure);
      }
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F64>>>
    DefaultBackend::create_iir_filter_plan_f64(
        const std::vector<IIRFilterCoef>& sos_coefficients) const
    {
      try {
        auto pimpl
            = std::make_unique<DefaultIIRFilterPlanImpl<F64>>(sos_coefficients);
        return IIRFilterPlan<F64>::create_from_impl(std::move(pimpl));
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Default IIRFilterPlan<F64>: " << e.what()
                  << std::endl;
        OMNI_RETURN_ERROR(std::unique_ptr<IIRFilterPlan<F64>>, Status::Failure);
      }
    }

    // --- Filter Design ---
    // Forward to standalone implementation functions (defined in filter.cpp)
    // *** Assumes these extern functions exist and are correctly defined
    // elsewhere ***
    template <typename T>  // Assuming templated implementation
    extern OmniExpected<std::vector<T>> design_fir_filter_impl(
        const FIRFilterSpec& spec);
    extern OmniExpected<std::vector<IIRFilterCoef>> design_iir_filter_impl(
        const IIRFilterSpec& spec);  // IIR design returns non-templated coefs

    [[nodiscard]] OmniExpected<F32Vec> DefaultBackend::design_fir_filter_f32(
        const FIRFilterSpec& spec) const
    {
      return design_fir_filter_impl<F32>(spec);
    }
    [[nodiscard]] OmniExpected<F64Vec> DefaultBackend::design_fir_filter_f64(
        const FIRFilterSpec& spec) const
    {
      return design_fir_filter_impl<F64>(spec);
    }
    [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>>
    DefaultBackend::design_iir_filter_f32(const IIRFilterSpec& spec) const
    {
      return design_iir_filter_impl(spec);
    }
    [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>>
    DefaultBackend::design_iir_filter_f64(const IIRFilterSpec& spec) const
    {
      return design_iir_filter_impl(spec);
    }

    // --- Internal Implementation Factories (Private) ---
    [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlanImpl<C32>>>
    DefaultBackend::create_fft_plan_impl_c32(size_t length) const
    {
      try {
        return std::make_unique<DefaultFFTPlanImpl<C32>>(length);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating DefaultFFTPlanImpl<C32>: " << e.what()
                  << std::endl;
        OMNI_RETURN_ERROR(std::unique_ptr<FFTPlanImpl<C32>>, Status::Failure);
      }
    }
    [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlanImpl<C64>>>
    DefaultBackend::create_fft_plan_impl_c64(size_t length) const
    {
      try {
        return std::make_unique<DefaultFFTPlanImpl<C64>>(length);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating DefaultFFTPlanImpl<C64>: " << e.what()
                  << std::endl;
        OMNI_RETURN_ERROR(std::unique_ptr<FFTPlanImpl<C64>>, Status::Failure);
      }
    }
    [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlanImpl<F32>>>
    DefaultBackend::create_rfft_plan_impl_f32(size_t length) const
    {
      try {
        return std::make_unique<DefaultRFFTPlanImpl<F32>>(length);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating DefaultRFFTPlanImpl<F32>: " << e.what()
                  << std::endl;
        OMNI_RETURN_ERROR(std::unique_ptr<RFFTPlanImpl<F32>>, Status::Failure);
      }
    }
    [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlanImpl<F64>>>
    DefaultBackend::create_rfft_plan_impl_f64(size_t length) const
    {
      try {
        return std::make_unique<DefaultRFFTPlanImpl<F64>>(length);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating DefaultRFFTPlanImpl<F64>: " << e.what()
                  << std::endl;
        OMNI_RETURN_ERROR(std::unique_ptr<RFFTPlanImpl<F64>>, Status::Failure);
      }
    }
    // Definition for the factory function used by the public
    // create_convolution_plan_*
    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlanImpl<F32>>>
    DefaultBackend::create_convolution_plan_impl_f32(
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      try {
        // Create the FFT plan variant first
        size_t kernel_length = kernel.size();
        if (kernel_length == 0) return std::unexpected(Status::InvalidArgument);
        // Determine required FFT length based on kernel and potential signal
        // size (use kernel * 2 as proxy)
        size_t min_fft_len = kernel_length + kernel_length - 1;
        size_t fft_length = convolution_detail::next_power_of_two(min_fft_len);
        if (fft_length == 0)
          return std::unexpected(
              Status::InvalidArgument);      // Overflow or too large
        if (fft_length < 2) fft_length = 2;  // RFFT needs length >= 2

        auto rfft_plan_expected
            = create_rfft_plan_impl_f32(fft_length);  // Use 'this->' implicitly
        if (!rfft_plan_expected)
          return std::unexpected(rfft_plan_expected.error());

        // Use the alias defined within DefaultConvolutionPlanImpl
        DefaultConvolutionPlanImpl<F32>::FFTPlanImplVariant fft_variant;
        fft_variant.emplace<std::unique_ptr<RFFTPlanImpl<F32>>>(
            std::move(rfft_plan_expected.value()));

        // Pass the variant to the constructor
        return std::make_unique<DefaultConvolutionPlanImpl<F32>>(
            std::move(fft_variant), kernel, type, method);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating DefaultConvolutionPlanImpl<F32>: "
                  << e.what() << std::endl;
        OMNI_RETURN_ERROR(
            std::unique_ptr<ConvolutionPlanImpl<F32>>, Status::Failure);
      }
    }
    // Definition for the factory function used by the public
    // create_convolution_plan_*
    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlanImpl<F64>>>
    DefaultBackend::create_convolution_plan_impl_f64(
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      try {
        size_t kernel_length = kernel.size();
        if (kernel_length == 0) return std::unexpected(Status::InvalidArgument);
        size_t min_fft_len = kernel_length + kernel_length - 1;
        size_t fft_length = convolution_detail::next_power_of_two(min_fft_len);
        if (fft_length == 0) return std::unexpected(Status::InvalidArgument);
        if (fft_length < 2) fft_length = 2;

        auto rfft_plan_expected = create_rfft_plan_impl_f64(fft_length);
        if (!rfft_plan_expected)
          return std::unexpected(rfft_plan_expected.error());

        DefaultConvolutionPlanImpl<F64>::FFTPlanImplVariant fft_variant;
        fft_variant.emplace<std::unique_ptr<RFFTPlanImpl<F64>>>(
            std::move(rfft_plan_expected.value()));

        return std::make_unique<DefaultConvolutionPlanImpl<F64>>(
            std::move(fft_variant), kernel, type, method);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating DefaultConvolutionPlanImpl<F64>: "
                  << e.what() << std::endl;
        OMNI_RETURN_ERROR(
            std::unique_ptr<ConvolutionPlanImpl<F64>>, Status::Failure);
      }
    }
    // Definition for the factory function used by the public
    // create_convolution_plan_*
    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlanImpl<C32>>>
    DefaultBackend::create_convolution_plan_impl_c32(
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      try {
        size_t kernel_length = kernel.size();
        if (kernel_length == 0) return std::unexpected(Status::InvalidArgument);
        size_t min_fft_len = kernel_length + kernel_length - 1;
        size_t fft_length = convolution_detail::next_power_of_two(min_fft_len);
        if (fft_length == 0) return std::unexpected(Status::InvalidArgument);

        auto cfft_plan_expected = create_fft_plan_impl_c32(fft_length);
        if (!cfft_plan_expected)
          return std::unexpected(cfft_plan_expected.error());

        DefaultConvolutionPlanImpl<C32>::FFTPlanImplVariant fft_variant;
        fft_variant.emplace<std::unique_ptr<FFTPlanImpl<C32>>>(
            std::move(cfft_plan_expected.value()));

        return std::make_unique<DefaultConvolutionPlanImpl<C32>>(
            std::move(fft_variant), kernel, type, method);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating DefaultConvolutionPlanImpl<C32>: "
                  << e.what() << std::endl;
        OMNI_RETURN_ERROR(
            std::unique_ptr<ConvolutionPlanImpl<C32>>, Status::Failure);
      }
    }
    // Definition for the factory function used by the public
    // create_convolution_plan_*
    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlanImpl<C64>>>
    DefaultBackend::create_convolution_plan_impl_c64(
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      try {
        size_t kernel_length = kernel.size();
        if (kernel_length == 0) return std::unexpected(Status::InvalidArgument);
        size_t min_fft_len = kernel_length + kernel_length - 1;
        size_t fft_length = convolution_detail::next_power_of_two(min_fft_len);
        if (fft_length == 0) return std::unexpected(Status::InvalidArgument);

        auto cfft_plan_expected = create_fft_plan_impl_c64(fft_length);
        if (!cfft_plan_expected)
          return std::unexpected(cfft_plan_expected.error());

        DefaultConvolutionPlanImpl<C64>::FFTPlanImplVariant fft_variant;
        fft_variant.emplace<std::unique_ptr<FFTPlanImpl<C64>>>(
            std::move(cfft_plan_expected.value()));

        return std::make_unique<DefaultConvolutionPlanImpl<C64>>(
            std::move(fft_variant), kernel, type, method);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating DefaultConvolutionPlanImpl<C64>: "
                  << e.what() << std::endl;
        OMNI_RETURN_ERROR(
            std::unique_ptr<ConvolutionPlanImpl<C64>>, Status::Failure);
      }
    }
    // Definition for the factory function used by the public
    // create_correlation_plan_*
    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlanImpl<F32>>>
    DefaultBackend::create_correlation_plan_impl_f32(
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      try {
        size_t kernel_length = kernel.size();
        if (kernel_length == 0) return std::unexpected(Status::InvalidArgument);
        size_t min_fft_len = kernel_length + kernel_length - 1;
        size_t fft_length = convolution_detail::next_power_of_two(min_fft_len);
        if (fft_length == 0) return std::unexpected(Status::InvalidArgument);
        if (fft_length < 2) fft_length = 2;

        auto rfft_plan_expected = create_rfft_plan_impl_f32(fft_length);
        if (!rfft_plan_expected)
          return std::unexpected(rfft_plan_expected.error());

        DefaultCorrelationPlanImpl<F32>::FFTPlanImplVariant fft_variant;
        fft_variant.emplace<std::unique_ptr<RFFTPlanImpl<F32>>>(
            std::move(rfft_plan_expected.value()));

        return std::make_unique<DefaultCorrelationPlanImpl<F32>>(
            std::move(fft_variant), kernel, type, method);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating DefaultCorrelationPlanImpl<F32>: "
                  << e.what() << std::endl;
        OMNI_RETURN_ERROR(
            std::unique_ptr<CorrelationPlanImpl<F32>>, Status::Failure);
      }
    }
    // Definition for the factory function used by the public
    // create_correlation_plan_*
    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlanImpl<F64>>>
    DefaultBackend::create_correlation_plan_impl_f64(
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      try {
        size_t kernel_length = kernel.size();
        if (kernel_length == 0) return std::unexpected(Status::InvalidArgument);
        size_t min_fft_len = kernel_length + kernel_length - 1;
        size_t fft_length = convolution_detail::next_power_of_two(min_fft_len);
        if (fft_length == 0) return std::unexpected(Status::InvalidArgument);
        if (fft_length < 2) fft_length = 2;

        auto rfft_plan_expected = create_rfft_plan_impl_f64(fft_length);
        if (!rfft_plan_expected)
          return std::unexpected(rfft_plan_expected.error());

        DefaultCorrelationPlanImpl<F64>::FFTPlanImplVariant fft_variant;
        fft_variant.emplace<std::unique_ptr<RFFTPlanImpl<F64>>>(
            std::move(rfft_plan_expected.value()));

        return std::make_unique<DefaultCorrelationPlanImpl<F64>>(
            std::move(fft_variant), kernel, type, method);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating DefaultCorrelationPlanImpl<F64>: "
                  << e.what() << std::endl;
        OMNI_RETURN_ERROR(
            std::unique_ptr<CorrelationPlanImpl<F64>>, Status::Failure);
      }
    }
    // Definition for the factory function used by the public
    // create_correlation_plan_*
    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlanImpl<C32>>>
    DefaultBackend::create_correlation_plan_impl_c32(
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      try {
        size_t kernel_length = kernel.size();
        if (kernel_length == 0) return std::unexpected(Status::InvalidArgument);
        size_t min_fft_len = kernel_length + kernel_length - 1;
        size_t fft_length = convolution_detail::next_power_of_two(min_fft_len);
        if (fft_length == 0) return std::unexpected(Status::InvalidArgument);

        auto cfft_plan_expected = create_fft_plan_impl_c32(fft_length);
        if (!cfft_plan_expected)
          return std::unexpected(cfft_plan_expected.error());

        DefaultCorrelationPlanImpl<C32>::FFTPlanImplVariant fft_variant;
        fft_variant.emplace<std::unique_ptr<FFTPlanImpl<C32>>>(
            std::move(cfft_plan_expected.value()));

        return std::make_unique<DefaultCorrelationPlanImpl<C32>>(
            std::move(fft_variant), kernel, type, method);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating DefaultCorrelationPlanImpl<C32>: "
                  << e.what() << std::endl;
        OMNI_RETURN_ERROR(
            std::unique_ptr<CorrelationPlanImpl<C32>>, Status::Failure);
      }
    }
    // Definition for the factory function used by the public
    // create_correlation_plan_*
    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlanImpl<C64>>>
    DefaultBackend::create_correlation_plan_impl_c64(
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const
    {
      try {
        size_t kernel_length = kernel.size();
        if (kernel_length == 0) return std::unexpected(Status::InvalidArgument);
        size_t min_fft_len = kernel_length + kernel_length - 1;
        size_t fft_length = convolution_detail::next_power_of_two(min_fft_len);
        if (fft_length == 0) return std::unexpected(Status::InvalidArgument);

        auto cfft_plan_expected = create_fft_plan_impl_c64(fft_length);
        if (!cfft_plan_expected)
          return std::unexpected(cfft_plan_expected.error());

        DefaultCorrelationPlanImpl<C64>::FFTPlanImplVariant fft_variant;
        fft_variant.emplace<std::unique_ptr<FFTPlanImpl<C64>>>(
            std::move(cfft_plan_expected.value()));

        return std::make_unique<DefaultCorrelationPlanImpl<C64>>(
            std::move(fft_variant), kernel, type, method);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating DefaultCorrelationPlanImpl<C64>: "
                  << e.what() << std::endl;
        OMNI_RETURN_ERROR(
            std::unique_ptr<CorrelationPlanImpl<C64>>, Status::Failure);
      }
    }

    [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlanImpl<F32>>>
    DefaultBackend::create_cqt_plan_impl_f32(
        F32 sample_rate,
        F32 min_freq,
        F32 max_freq,
        int bins_per_octave,
        const WindowSpec& window_spec) const
    {
      try {
        // Pass 'this' as the owner
        return std::make_unique<DefaultCQTPlanImpl<F32>>(
            this,
            sample_rate,
            min_freq,
            max_freq,
            bins_per_octave,
            window_spec);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating DefaultCQTPlanImpl<F32>: " << e.what()
                  << std::endl;
        OMNI_RETURN_ERROR(std::unique_ptr<CQTPlanImpl<F32>>, Status::Failure);
      }
    }
    [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlanImpl<F64>>>
    DefaultBackend::create_cqt_plan_impl_f64(
        F64 sample_rate,
        F64 min_freq,
        F64 max_freq,
        int bins_per_octave,
        const WindowSpec& window_spec) const
    {
      try {
        // Pass 'this' as the owner
        return std::make_unique<DefaultCQTPlanImpl<F64>>(
            this,
            sample_rate,
            min_freq,
            max_freq,
            bins_per_octave,
            window_spec);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating DefaultCQTPlanImpl<F64>: " << e.what()
                  << std::endl;
        OMNI_RETURN_ERROR(std::unique_ptr<CQTPlanImpl<F64>>, Status::Failure);
      }
    }

    // Internal window generation helper implementation (UPDATED SIGNATURE)
    // Calls the span-based helpers from window.hpp/window.cpp
    [[nodiscard]] Status DefaultBackend::generate_window_f32(
        const WindowSpec& spec, std::span<F32> output) const
    {
      // Check for empty span early
      if (output.empty()) {
        return Status::Success;
      }
      // Call the templated generate_window function (defined in window.hpp)
      return generate_window<F32>(spec, output.size(), output);
    }

    [[nodiscard]] Status DefaultBackend::generate_window_f64(
        const WindowSpec& spec, std::span<F64> output) const
    {
      if (output.empty()) {
        return Status::Success;
      }
      // Call the templated generate_window function (defined in window.hpp)
      return generate_window<F64>(spec, output.size(), output);
    }

  }  // namespace backend
}  // namespace OmniDSP
