/**
 * @file backend.cpp (Default)
 * @brief Implements the Backend class methods using standard C++.
 */

#include "backend.hpp"  // Corresponding header for Default backend declarations

#include "convolution.hpp"  // DefaultConvolutionPlanImpl, DefaultCorrelationPlanImpl
#include "cqt.hpp"          // DefaultCQTPlanImpl
#include "fft.hpp"          // DefaultFFTPlanImpl, DefaultRFFTPlanImpl
#include "filter.hpp"    // DefaultFIRFilterPlanImpl, DefaultIIRFilterPlanImpl
#include "resample.hpp"  // DefaultResamplePlanImpl
#include "window.hpp"  // Default window implementations (span-based) - needed for generate_window

// Include headers for the public Plan classes
#include <OmniDSP/convolution.hpp>
#include <OmniDSP/core_types.hpp>
#include <OmniDSP/cqt.hpp>
#include <OmniDSP/fft.hpp>
#include <OmniDSP/filter.hpp>  // Includes FIRCoefs alias now
#include <OmniDSP/resample.hpp>
#include <OmniDSP/window.hpp>

// Include standard library headers needed for default implementations
#include <algorithm>
#include <cmath>
#include <complex>
#include <expected>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace OmniDSP::Default {

  // Forward declare the internal design helper functions (defined in
  // filter.cpp) These declarations allow this file to call them.
  template <typename T>
  [[nodiscard]] OmniExpected<FIRCoefs<T>> generate_fir_filter_coeffs(
      const FIRFilterSpec& spec);
  [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>>
  generate_iir_filter_coeffs(const IIRFilterSpec& spec);

  // --- Helper Function from convolution.cpp ---
  namespace convolution_detail {
    inline size_t next_power_of_two(size_t n)
    {
      if (n == 0) return 1;
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
        if (power == 0) return std::numeric_limits<size_t>::max();
      }
      return power;
    }
  }  // namespace convolution_detail

  /** @brief Helper function to check a condition and return a status. */
  inline Status check_status(
      bool condition,
      Status error_status,
      const std::string& error_message = "")
  {
    if (!condition) {
      if (!error_message.empty()) {
        std::cerr << "OmniDSP Default BackendType Error: " << error_message
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
  // Backend Method Definitions
  //--------------------------------------------------------------------------
  Backend::Backend() {}
  Backend::~Backend() {}
  BackendType Backend::get_backend() const { return BackendType::Default; }

  // --- DSP Operations (One-Off Implementations) ---
  // ... (convolve_*, correlate_*, fft_*, etc. implementations remain the
  // same) ...
  [[nodiscard]] OmniExpected<F32Vec> Backend::convolve_f32(
      const F32Vec& input,
      const F32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    // auto plan_expected = create_convolution_plan_impl_f32(kernel, type,
    // method);
    auto plan_expected
        = this->create_convolution_plan_f32(kernel, type, method);
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
  [[nodiscard]] OmniExpected<F64Vec> Backend::convolve_f64(
      const F64Vec& input,
      const F64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    // auto plan_expected = create_convolution_plan_impl_f64(kernel, type,
    // method);
    auto plan_expected
        = this->create_convolution_plan_f64(kernel, type, method);
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
  [[nodiscard]] OmniExpected<C32Vec> Backend::convolve_c32(
      const C32Vec& input,
      const C32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    // auto plan_expected = create_convolution_plan_impl_c32(kernel, type,
    // method);
    auto plan_expected
        = this->create_convolution_plan_c32(kernel, type, method);
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
  [[nodiscard]] OmniExpected<C64Vec> Backend::convolve_c64(
      const C64Vec& input,
      const C64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    // auto plan_expected = create_convolution_plan_impl_c64(kernel, type,
    // method);
    auto plan_expected
        = this->create_convolution_plan_c64(kernel, type, method);
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
  [[nodiscard]] OmniExpected<F32Vec> Backend::correlate_f32(
      const F32Vec& input,
      const F32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    // auto plan_expected = create_correlation_plan_impl_f32(kernel, type,
    // method);
    auto plan_expected
        = this->create_correlation_plan_f32(kernel, type, method);
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
  [[nodiscard]] OmniExpected<F64Vec> Backend::correlate_f64(
      const F64Vec& input,
      const F64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    // auto plan_expected = create_correlation_plan_impl_f64(kernel, type,
    // method);
    auto plan_expected
        = this->create_correlation_plan_f64(kernel, type, method);
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
  [[nodiscard]] OmniExpected<C32Vec> Backend::correlate_c32(
      const C32Vec& input,
      const C32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    // auto plan_expected = create_correlation_plan_impl_c32(kernel, type,
    // method);
    auto plan_expected
        = this->create_correlation_plan_c32(kernel, type, method);
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
  [[nodiscard]] OmniExpected<C64Vec> Backend::correlate_c64(
      const C64Vec& input,
      const C64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    // auto plan_expected = create_correlation_plan_impl_c64(kernel, type,
    // method);
    auto plan_expected
        = this->create_correlation_plan_c64(kernel, type, method);
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
  [[nodiscard]] OmniExpected<C32Vec> Backend::fft_c32(const C32Vec& input) const
  {
    // auto plan_expected = create_fft_plan_impl_c32(input.size());
    auto plan_expected = this->create_fft_plan_c32(input.size());
    OMNI_PROPAGATE_ERROR(C32Vec, plan_expected);
    auto& plan = plan_expected.value();
    C32Vec output(input.size());
    Status status = plan->fft(std::span(input), std::span(output));
    if (status != Status::Success) {
      OMNI_RETURN_ERROR(C32Vec, status);
    }
    return output;
  }
  [[nodiscard]] OmniExpected<C64Vec> Backend::fft_c64(const C64Vec& input) const
  {
    // auto plan_expected = create_fft_plan_impl_c64(input.size());
    auto plan_expected = this->create_fft_plan_c64(input.size());
    OMNI_PROPAGATE_ERROR(C64Vec, plan_expected);
    auto& plan = plan_expected.value();
    C64Vec output(input.size());
    Status status = plan->fft(std::span(input), std::span(output));
    if (status != Status::Success) {
      OMNI_RETURN_ERROR(C64Vec, status);
    }
    return output;
  }
  [[nodiscard]] OmniExpected<C32Vec> Backend::ifft_c32(
      const C32Vec& input) const
  {
    // auto plan_expected = create_fft_plan_impl_c32(input.size());
    auto plan_expected = create_fft_plan_c32(input.size());
    OMNI_PROPAGATE_ERROR(C32Vec, plan_expected);
    auto& plan = plan_expected.value();
    C32Vec output(input.size());
    Status status = plan->ifft(std::span(input), std::span(output));
    if (status != Status::Success) {
      OMNI_RETURN_ERROR(C32Vec, status);
    }
    return output;
  }
  [[nodiscard]] OmniExpected<C64Vec> Backend::ifft_c64(
      const C64Vec& input) const
  {
    // auto plan_expected = create_fft_plan_impl_c64(input.size());
    auto plan_expected = create_fft_plan_c64(input.size());
    OMNI_PROPAGATE_ERROR(C64Vec, plan_expected);
    auto& plan = plan_expected.value();
    C64Vec output(input.size());
    Status status = plan->ifft(std::span(input), std::span(output));
    if (status != Status::Success) {
      OMNI_RETURN_ERROR(C64Vec, status);
    }
    return output;
  }
  [[nodiscard]] OmniExpected<C32Vec> Backend::rfft_f32(
      const F32Vec& input) const
  {
    // auto plan_expected = create_rfft_plan_impl_f32(input.size());
    auto plan_expected = create_rfft_plan_f32(input.size());
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
  [[nodiscard]] OmniExpected<C64Vec> Backend::rfft_f64(
      const F64Vec& input) const
  {
    // auto plan_expected = create_rfft_plan_impl_f64(input.size());
    auto plan_expected = create_rfft_plan_f64(input.size());
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
  [[nodiscard]] OmniExpected<F32Vec> Backend::irfft_c32(
      const C32Vec& input, size_t output_length) const
  {
    if (output_length == 0) {
      return input.empty()
                 ? OmniExpected<F32Vec>(F32Vec{})
                 : OmniExpected<F32Vec>(std::unexpect, Status::InvalidArgument);
    }
    size_t expected_input_len = (output_length / 2) + 1;
    if (input.size() != expected_input_len) {
      std::cerr << "Default irfft_c32 error: Input size (" << input.size()
                << ") does not match expected size (" << expected_input_len
                << ") for output length " << output_length << "." << std::endl;
      OMNI_RETURN_ERROR(F32Vec, Status::SizeMismatch);
    }
    // auto plan_expected = create_rfft_plan_impl_f32(output_length);
    auto plan_expected = this->create_rfft_plan_f32(output_length);
    OMNI_PROPAGATE_ERROR(F32Vec, plan_expected);
    auto& plan = plan_expected.value();
    F32Vec output(output_length);
    Status status = plan->irfft(std::span(input), std::span(output));
    if (status != Status::Success) {
      OMNI_RETURN_ERROR(F32Vec, status);
    }
    return output;
  }
  [[nodiscard]] OmniExpected<F64Vec> Backend::irfft_c64(
      const C64Vec& input, size_t output_length) const
  {
    if (output_length == 0) {
      return input.empty()
                 ? OmniExpected<F64Vec>(F64Vec{})
                 : OmniExpected<F64Vec>(std::unexpect, Status::InvalidArgument);
    }
    size_t expected_input_len = (output_length / 2) + 1;
    if (input.size() != expected_input_len) {
      std::cerr << "Default irfft_c64 error: Input size (" << input.size()
                << ") does not match expected size (" << expected_input_len
                << ") for output length " << output_length << "." << std::endl;
      OMNI_RETURN_ERROR(F64Vec, Status::SizeMismatch);
    }
    // auto plan_expected = create_rfft_plan_impl_f64(output_length);
    auto plan_expected = this->create_rfft_plan_f64(output_length);
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
  // ... (Window implementations remain the same, calling generate_window<T>)
  // ...
  [[nodiscard]] Status Backend::bartlett_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    auto target_span = output.first(length);
    return generate_window_f32(WindowSpec(WindowType::Bartlett), target_span);
  }
  [[nodiscard]] Status Backend::bartlett_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    auto target_span = output.first(length);
    return generate_window_f64(WindowSpec(WindowType::Bartlett), target_span);
  }
  [[nodiscard]] Status Backend::blackman_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    auto target_span = output.first(length);
    return generate_window_f32(WindowSpec(WindowType::Blackman), target_span);
  }
  [[nodiscard]] Status Backend::blackman_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    auto target_span = output.first(length);
    return generate_window_f64(WindowSpec(WindowType::Blackman), target_span);
  }
  [[nodiscard]] Status Backend::flattop_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    auto target_span = output.first(length);
    return generate_window_f32(WindowSpec(WindowType::Flattop), target_span);
  }
  [[nodiscard]] Status Backend::flattop_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    auto target_span = output.first(length);
    return generate_window_f64(WindowSpec(WindowType::Flattop), target_span);
  }
  [[nodiscard]] Status Backend::gaussian_window_f32(
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
  [[nodiscard]] Status Backend::gaussian_window_f64(
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
  [[nodiscard]] Status Backend::hamming_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    auto target_span = output.first(length);
    return generate_window_f32(WindowSpec(WindowType::Hamming), target_span);
  }
  [[nodiscard]] Status Backend::hamming_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    auto target_span = output.first(length);
    return generate_window_f64(WindowSpec(WindowType::Hamming), target_span);
  }
  [[nodiscard]] Status Backend::hann_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    auto target_span = output.first(length);
    return generate_window_f32(WindowSpec(WindowType::Hann), target_span);
  }
  [[nodiscard]] Status Backend::hann_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    auto target_span = output.first(length);
    return generate_window_f64(WindowSpec(WindowType::Hann), target_span);
  }
  [[nodiscard]] Status Backend::kaiser_window_f32(
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
  [[nodiscard]] Status Backend::kaiser_window_f64(
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
  [[nodiscard]] Status Backend::rectangular_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    auto target_span = output.first(length);
    return generate_window_f32(
        WindowSpec(WindowType::Rectangular), target_span);
  }
  [[nodiscard]] Status Backend::rectangular_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    auto target_span = output.first(length);
    return generate_window_f64(
        WindowSpec(WindowType::Rectangular), target_span);
  }
  [[nodiscard]] Status Backend::triangular_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    auto target_span = output.first(length);
    return generate_window_f32(WindowSpec(WindowType::Triangular), target_span);
  }
  [[nodiscard]] Status Backend::triangular_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    auto target_span = output.first(length);
    return generate_window_f64(WindowSpec(WindowType::Triangular), target_span);
  }

  // --- Plan Factories (Implementations) ---
  // ... (Plan factory implementations remain the same) ...
  [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C32>>>
  Backend::create_fft_plan_c32(size_t length) const
  {
    // auto pimpl_expected = create_fft_plan_impl_c32(length);
    auto pimpl_expected = this->create_fft_plan_impl_c32(length);
    OMNI_PROPAGATE_ERROR(std::unique_ptr<FFTPlan<C32>>, pimpl_expected);
    return FFTPlan<C32>::create_from_impl(std::move(pimpl_expected.value()));
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C64>>>
  Backend::create_fft_plan_c64(size_t length) const
  {
    // auto pimpl_expected = create_fft_plan_impl_c64(length);
    auto pimpl_expected = this->create_fft_plan_impl_c64(length);
    OMNI_PROPAGATE_ERROR(std::unique_ptr<FFTPlan<C64>>, pimpl_expected);
    return FFTPlan<C64>::create_from_impl(std::move(pimpl_expected.value()));
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F32>>>
  Backend::create_rfft_plan_f32(size_t length) const
  {
    // auto pimpl_expected = create_rfft_plan_impl_f32(length);
    auto pimpl_expected = this->create_rfft_plan_impl_f32(length);
    OMNI_PROPAGATE_ERROR(std::unique_ptr<RFFTPlan<F32>>, pimpl_expected);
    return RFFTPlan<F32>::create_from_impl(std::move(pimpl_expected.value()));
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F64>>>
  Backend::create_rfft_plan_f64(size_t length) const
  {
    // auto pimpl_expected = create_rfft_plan_impl_f64(length);
    auto pimpl_expected = this->create_rfft_plan_impl_f64(length);
    OMNI_PROPAGATE_ERROR(std::unique_ptr<RFFTPlan<F64>>, pimpl_expected);
    return RFFTPlan<F64>::create_from_impl(std::move(pimpl_expected.value()));
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<F32>>>
  Backend::create_cqt_plan_f32(
      F32 sample_rate,
      F32 min_freq,
      F32 max_freq,
      int bins_per_octave,
      const WindowSpec& window_spec) const
  {
    // auto pimpl_expected = create_cqt_plan_impl_f32(
    auto pimpl_expected = this->create_cqt_plan_impl_f32(
        sample_rate, min_freq, max_freq, bins_per_octave, window_spec);
    OMNI_PROPAGATE_ERROR(std::unique_ptr<CQTPlan<F32>>, pimpl_expected);
    return CQTPlan<F32>::create_from_impl(std::move(pimpl_expected.value()));
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<F64>>>
  Backend::create_cqt_plan_f64(
      F64 sample_rate,
      F64 min_freq,
      F64 max_freq,
      int bins_per_octave,
      const WindowSpec& window_spec) const
  {
    // auto pimpl_expected = create_cqt_plan_impl_f64(
    auto pimpl_expected = this->create_cqt_plan_impl_f64(
        sample_rate, min_freq, max_freq, bins_per_octave, window_spec);
    OMNI_PROPAGATE_ERROR(std::unique_ptr<CQTPlan<F64>>, pimpl_expected);
    return CQTPlan<F64>::create_from_impl(std::move(pimpl_expected.value()));
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<F32>>>
  Backend::create_resample_plan_f32(const ResampleSpec& spec) const
  {
    try {
      auto pimpl = std::make_unique<ResamplePlanImpl<F32>>(this, spec);
      return ResamplePlan<F32>::create_from_impl(std::move(pimpl));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating Default ResamplePlan<F32>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(std::unique_ptr<ResamplePlan<F32>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<F64>>>
  Backend::create_resample_plan_f64(const ResampleSpec& spec) const
  {
    try {
      auto pimpl = std::make_unique<ResamplePlanImpl<F64>>(this, spec);
      return ResamplePlan<F64>::create_from_impl(std::move(pimpl));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating Default ResamplePlan<F64>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(std::unique_ptr<ResamplePlan<F64>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<F32>>>
  Backend::create_convolution_plan_f32(
      const F32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    auto pimpl_expected
        = create_convolution_plan_impl_f32(kernel, type, method);
    OMNI_PROPAGATE_ERROR(std::unique_ptr<ConvolutionPlan<F32>>, pimpl_expected);
    return ConvolutionPlan<F32>::create_from_impl(
        std::move(pimpl_expected.value()));
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<F64>>>
  Backend::create_convolution_plan_f64(
      const F64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    auto pimpl_expected
        = create_convolution_plan_impl_f64(kernel, type, method);
    OMNI_PROPAGATE_ERROR(std::unique_ptr<ConvolutionPlan<F64>>, pimpl_expected);
    return ConvolutionPlan<F64>::create_from_impl(
        std::move(pimpl_expected.value()));
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<C32>>>
  Backend::create_convolution_plan_c32(
      const C32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    auto pimpl_expected
        = create_convolution_plan_impl_c32(kernel, type, method);
    OMNI_PROPAGATE_ERROR(std::unique_ptr<ConvolutionPlan<C32>>, pimpl_expected);
    return ConvolutionPlan<C32>::create_from_impl(
        std::move(pimpl_expected.value()));
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<C64>>>
  Backend::create_convolution_plan_c64(
      const C64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    auto pimpl_expected
        = create_convolution_plan_impl_c64(kernel, type, method);
    OMNI_PROPAGATE_ERROR(std::unique_ptr<ConvolutionPlan<C64>>, pimpl_expected);
    return ConvolutionPlan<C64>::create_from_impl(
        std::move(pimpl_expected.value()));
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<F32>>>
  Backend::create_correlation_plan_f32(
      const F32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    auto pimpl_expected
        = create_correlation_plan_impl_f32(kernel, type, method);
    OMNI_PROPAGATE_ERROR(std::unique_ptr<CorrelationPlan<F32>>, pimpl_expected);
    return CorrelationPlan<F32>::create_from_impl(
        std::move(pimpl_expected.value()));
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<F64>>>
  Backend::create_correlation_plan_f64(
      const F64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    auto pimpl_expected
        = create_correlation_plan_impl_f64(kernel, type, method);
    OMNI_PROPAGATE_ERROR(std::unique_ptr<CorrelationPlan<F64>>, pimpl_expected);
    return CorrelationPlan<F64>::create_from_impl(
        std::move(pimpl_expected.value()));
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<C32>>>
  Backend::create_correlation_plan_c32(
      const C32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    auto pimpl_expected
        = create_correlation_plan_impl_c32(kernel, type, method);
    OMNI_PROPAGATE_ERROR(std::unique_ptr<CorrelationPlan<C32>>, pimpl_expected);
    return CorrelationPlan<C32>::create_from_impl(
        std::move(pimpl_expected.value()));
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<C64>>>
  Backend::create_correlation_plan_c64(
      const C64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    auto pimpl_expected
        = create_correlation_plan_impl_c64(kernel, type, method);
    OMNI_PROPAGATE_ERROR(std::unique_ptr<CorrelationPlan<C64>>, pimpl_expected);
    return CorrelationPlan<C64>::create_from_impl(
        std::move(pimpl_expected.value()));
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<F32>>>
  Backend::create_fir_filter_plan_f32(const F32Vec& coefficients) const
  {
    try {
      auto pimpl = std::make_unique<FIRFilterPlanImpl<F32>>(coefficients);
      return FIRFilterPlan<F32>::create_from_impl(std::move(pimpl));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating Default FIRFilterPlan<F32>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(std::unique_ptr<FIRFilterPlan<F32>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<F64>>>
  Backend::create_fir_filter_plan_f64(const F64Vec& coefficients) const
  {
    try {
      auto pimpl = std::make_unique<FIRFilterPlanImpl<F64>>(coefficients);
      return FIRFilterPlan<F64>::create_from_impl(std::move(pimpl));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating Default FIRFilterPlan<F64>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(std::unique_ptr<FIRFilterPlan<F64>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<C32>>>
  Backend::create_fir_filter_plan_c32(const C32Vec& coefficients) const
  {
    try {
      auto pimpl = std::make_unique<FIRFilterPlanImpl<C32>>(coefficients);
      return FIRFilterPlan<C32>::create_from_impl(std::move(pimpl));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating Default FIRFilterPlan<C32>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(std::unique_ptr<FIRFilterPlan<C32>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<C64>>>
  Backend::create_fir_filter_plan_c64(const C64Vec& coefficients) const
  {
    try {
      auto pimpl = std::make_unique<FIRFilterPlanImpl<C64>>(coefficients);
      return FIRFilterPlan<C64>::create_from_impl(std::move(pimpl));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating Default FIRFilterPlan<C64>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(std::unique_ptr<FIRFilterPlan<C64>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F32>>>
  Backend::create_iir_filter_plan_f32(
      const std::vector<IIRFilterCoef>& sos_coefficients) const
  {
    try {
      auto pimpl = std::make_unique<IIRFilterPlanImpl<F32>>(sos_coefficients);
      return IIRFilterPlan<F32>::create_from_impl(std::move(pimpl));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating Default IIRFilterPlan<F32>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(std::unique_ptr<IIRFilterPlan<F32>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F64>>>
  Backend::create_iir_filter_plan_f64(
      const std::vector<IIRFilterCoef>& sos_coefficients) const
  {
    try {
      auto pimpl = std::make_unique<IIRFilterPlanImpl<F64>>(sos_coefficients);
      return IIRFilterPlan<F64>::create_from_impl(std::move(pimpl));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating Default IIRFilterPlan<F64>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(std::unique_ptr<IIRFilterPlan<F64>>, Status::Failure);
    }
  }

  // --- Filter Design ---
  // *** Calls the renamed helper functions defined in filter.cpp ***
  [[nodiscard]] OmniExpected<FIRCoefs<F32>> Backend::design_fir_filter_f32(
      const FIRFilterSpec& spec) const
  {
    return generate_fir_filter_coeffs<F32>(spec);  // Call helper
  }
  [[nodiscard]] OmniExpected<FIRCoefs<F64>> Backend::design_fir_filter_f64(
      const FIRFilterSpec& spec) const
  {
    return generate_fir_filter_coeffs<F64>(spec);  // Call helper
  }
  [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>>
  Backend::design_iir_filter_f32(const IIRFilterSpec& spec) const
  {
    // IIR design returns non-templated coefs, pass F32 spec context if needed
    return generate_iir_filter_coeffs(spec);  // Call helper
  }
  [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>>
  Backend::design_iir_filter_f64(const IIRFilterSpec& spec) const
  {
    // IIR design returns non-templated coefs, pass F64 spec context if needed
    return generate_iir_filter_coeffs(spec);  // Call helper
  }

  // --- Internal Implementation Factories (Private) ---
  // ... (Internal factory implementations remain the same) ...
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>
  Backend::create_fft_plan_impl_c32(size_t length) const
  {
    try {
      return std::make_unique<FFTPlanImpl<C32>>(length);
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating FFTPlanImpl<C32>: " << e.what() << std::endl;
      OMNI_RETURN_ERROR(
          std::unique_ptr<Abstract::FFTPlanImpl<C32>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>
  Backend::create_fft_plan_impl_c64(size_t length) const
  {
    try {
      return std::make_unique<FFTPlanImpl<C64>>(length);
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating FFTPlanImpl<C64>: " << e.what() << std::endl;
      OMNI_RETURN_ERROR(
          std::unique_ptr<Abstract::FFTPlanImpl<C64>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>
  Backend::create_rfft_plan_impl_f32(size_t length) const
  {
    try {
      return std::make_unique<RFFTPlanImpl<F32>>(length);
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating RFFTPlanImpl<F32>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(
          std::unique_ptr<Abstract::RFFTPlanImpl<F32>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>
  Backend::create_rfft_plan_impl_f64(size_t length) const
  {
    try {
      return std::make_unique<RFFTPlanImpl<F64>>(length);
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating RFFTPlanImpl<F64>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(
          std::unique_ptr<Abstract::RFFTPlanImpl<F64>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ConvolutionPlanImpl<F32>>>
  Backend::create_convolution_plan_impl_f32(
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
      ConvolutionPlanImpl<F32>::FFTPlanImplVariant fft_variant;
      fft_variant.emplace<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>(
          std::move(rfft_plan_expected.value()));
      return std::make_unique<ConvolutionPlanImpl<F32>>(
          std::move(fft_variant), kernel, type, method);
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating ConvolutionPlanImpl<F32>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(
          std::unique_ptr<Abstract::ConvolutionPlanImpl<F32>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ConvolutionPlanImpl<F64>>>
  Backend::create_convolution_plan_impl_f64(
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
      ConvolutionPlanImpl<F64>::FFTPlanImplVariant fft_variant;
      fft_variant.emplace<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>(
          std::move(rfft_plan_expected.value()));
      return std::make_unique<ConvolutionPlanImpl<F64>>(
          std::move(fft_variant), kernel, type, method);
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating ConvolutionPlanImpl<F64>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(
          std::unique_ptr<Abstract::ConvolutionPlanImpl<F64>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ConvolutionPlanImpl<C32>>>
  Backend::create_convolution_plan_impl_c32(
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
      ConvolutionPlanImpl<C32>::FFTPlanImplVariant fft_variant;
      fft_variant.emplace<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>(
          std::move(cfft_plan_expected.value()));
      return std::make_unique<ConvolutionPlanImpl<C32>>(
          std::move(fft_variant), kernel, type, method);
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating ConvolutionPlanImpl<C32>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(
          std::unique_ptr<Abstract::ConvolutionPlanImpl<C32>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ConvolutionPlanImpl<C64>>>
  Backend::create_convolution_plan_impl_c64(
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
      ConvolutionPlanImpl<C64>::FFTPlanImplVariant fft_variant;
      fft_variant.emplace<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>(
          std::move(cfft_plan_expected.value()));
      return std::make_unique<ConvolutionPlanImpl<C64>>(
          std::move(fft_variant), kernel, type, method);
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating ConvolutionPlanImpl<C64>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(
          std::unique_ptr<Abstract::ConvolutionPlanImpl<C64>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::CorrelationPlanImpl<F32>>>
  Backend::create_correlation_plan_impl_f32(
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
      CorrelationPlanImpl<F32>::FFTPlanImplVariant fft_variant;
      fft_variant.emplace<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>(
          std::move(rfft_plan_expected.value()));
      return std::make_unique<CorrelationPlanImpl<F32>>(
          std::move(fft_variant), kernel, type, method);
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating CorrelationPlanImpl<F32>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(
          std::unique_ptr<Abstract::CorrelationPlanImpl<F32>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::CorrelationPlanImpl<F64>>>
  Backend::create_correlation_plan_impl_f64(
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
      CorrelationPlanImpl<F64>::FFTPlanImplVariant fft_variant;
      fft_variant.emplace<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>(
          std::move(rfft_plan_expected.value()));
      return std::make_unique<CorrelationPlanImpl<F64>>(
          std::move(fft_variant), kernel, type, method);
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating CorrelationPlanImpl<F64>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(
          std::unique_ptr<Abstract::CorrelationPlanImpl<F64>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::CorrelationPlanImpl<C32>>>
  Backend::create_correlation_plan_impl_c32(
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
      CorrelationPlanImpl<C32>::FFTPlanImplVariant fft_variant;
      fft_variant.emplace<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>(
          std::move(cfft_plan_expected.value()));
      return std::make_unique<CorrelationPlanImpl<C32>>(
          std::move(fft_variant), kernel, type, method);
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating CorrelationPlanImpl<C32>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(
          std::unique_ptr<Abstract::CorrelationPlanImpl<C32>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::CorrelationPlanImpl<C64>>>
  Backend::create_correlation_plan_impl_c64(
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
      CorrelationPlanImpl<C64>::FFTPlanImplVariant fft_variant;
      fft_variant.emplace<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>(
          std::move(cfft_plan_expected.value()));
      return std::make_unique<CorrelationPlanImpl<C64>>(
          std::move(fft_variant), kernel, type, method);
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating CorrelationPlanImpl<C64>: " << e.what()
                << std::endl;
      OMNI_RETURN_ERROR(
          std::unique_ptr<Abstract::CorrelationPlanImpl<C64>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::CQTPlanImpl<F32>>>
  Backend::create_cqt_plan_impl_f32(
      F32 sample_rate,
      F32 min_freq,
      F32 max_freq,
      int bins_per_octave,
      const WindowSpec& window_spec) const
  {
    try {
      return std::make_unique<CQTPlanImpl<F32>>(
          this, sample_rate, min_freq, max_freq, bins_per_octave, window_spec);
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating CQTPlanImpl<F32>: " << e.what() << std::endl;
      OMNI_RETURN_ERROR(
          std::unique_ptr<Abstract::CQTPlanImpl<F32>>, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::CQTPlanImpl<F64>>>
  Backend::create_cqt_plan_impl_f64(
      F64 sample_rate,
      F64 min_freq,
      F64 max_freq,
      int bins_per_octave,
      const WindowSpec& window_spec) const
  {
    try {
      return std::make_unique<CQTPlanImpl<F64>>(
          this, sample_rate, min_freq, max_freq, bins_per_octave, window_spec);
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating CQTPlanImpl<F64>: " << e.what() << std::endl;
      OMNI_RETURN_ERROR(
          std::unique_ptr<Abstract::CQTPlanImpl<F64>>, Status::Failure);
    }
  }

  // --- Internal window generation helper implementation ---
  // Calls the span-based helpers from window.cpp
  [[nodiscard]] Status Backend::generate_window_f32(
      const WindowSpec& spec, std::span<F32> output) const
  {
    if (output.empty()) {
      return Status::Success;
    }
    // Call the appropriate free function from default/window.cpp
    switch (spec.get_type()) {
      case WindowType::Bartlett:
        return bartlett_window(output);
      case WindowType::Blackman:
        return blackman_window(output);
      case WindowType::Flattop:
        return flattop_window(output);
      case WindowType::Hamming:
        return hamming_window(output);
      case WindowType::Hann:
        return hann_window(output);
      case WindowType::Rectangular:
        return rectangular_window(output);
      case WindowType::Triangular:
        return triangular_window(output);
      case WindowType::Gaussian:
        if (!spec.get_param().has_value()) return Status::InvalidArgument;
        return gaussian_window(
            static_cast<F32>(spec.get_param().value()), output);
      case WindowType::Kaiser:
        if (!spec.get_param().has_value()) return Status::InvalidArgument;
        return kaiser_window(
            static_cast<F32>(spec.get_param().value()), output);
      default:
        return Status::InvalidArgument;
    }
  }

  [[nodiscard]] Status Backend::generate_window_f64(
      const WindowSpec& spec, std::span<F64> output) const
  {
    if (output.empty()) {
      return Status::Success;
    }
    switch (spec.get_type()) {
      case WindowType::Bartlett:
        return Default ::bartlett_window(output);
      case WindowType::Blackman:
        return Default ::blackman_window(output);
      case WindowType::Flattop:
        return Default ::flattop_window(output);
      case WindowType::Hamming:
        return Default ::hamming_window(output);
      case WindowType::Hann:
        return Default ::hann_window(output);
      case WindowType::Rectangular:
        return Default ::rectangular_window(output);
      case WindowType::Triangular:
        return Default ::triangular_window(output);
      case WindowType::Gaussian:
        if (!spec.get_param().has_value()) return Status::InvalidArgument;
        return Default ::gaussian_window(
            spec.get_param().value(), output);  // Pass double directly
      case WindowType::Kaiser:
        if (!spec.get_param().has_value()) return Status::InvalidArgument;
        return Default ::kaiser_window(
            spec.get_param().value(), output);  // Pass double directly
      default:
        return Status::InvalidArgument;
    }
  }
}  // namespace OmniDSP::Default

// This function needs to be defined in the global OmniDSP::Abstract namespace
// as declared in interface/backend.hpp
namespace OmniDSP::Abstract {
  std::unique_ptr<Backend> create_default_backend()
  {
    // Use the concrete class from the Default namespace
    return std::make_unique<::OmniDSP::Default ::Backend>();
  }
}  // namespace OmniDSP::Abstract
