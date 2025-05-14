/**
 * @file backend.cpp (Default)
 * @brief Implements the Default::Backend class methods.
 */

#include "backend.hpp"  // Corresponding header for Default backend declarations

// Include headers for Default plan implementations
#include "convolution.hpp"  // Defines Default::ConvolutionPlanImpl etc.
#include "cqt.hpp"          // Defines Default::CQTPlanImpl
#include "fft.hpp"  // Defines Default::FFTPlanImpl, Default::RFFTPlanImpl
#include "fir_filter.hpp"  // Defines Default::FIRFilterPlanImpl
#include "iir_filter.hpp"  // Defines Default::IIRFilterPlanImpl
#include "resample.hpp"    // Defines Default::ResamplePlanImpl
// The local "default/window.hpp" is used by the specific window overrides.

// Public API headers (needed for types used in method signatures and for public
// Plan classes)
#include <OmniDSP/convolution.hpp>  // For ConvolutionPlan
#include <OmniDSP/core_types.hpp>   // For F32, F64, Status, OmniExpected etc.
#include <OmniDSP/cqt.hpp>          // For CQTPlan
#include <OmniDSP/fft.hpp>          // For FFTPlan, RFFTPlan
#include <OmniDSP/fir_filter.hpp>  // For FIRFilterPlan, Design::FIRFilter, Coefs::FIRFilter
#include <OmniDSP/iir_filter.hpp>  // For IIRFilterPlan, Design::IIRFilter, Coefs::IIRFilterSOS
#include <OmniDSP/resample.hpp>  // For ResamplePlan, Design::Resample
#include <OmniDSP/window.hpp>  // For WindowSetup, WindowParams, and the free OmniDSP::generate_window

// Standard library headers
#include <algorithm>
#include <cmath>
#include <complex>  // For std::complex
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
#include <vector>

// spdlog for logging
#include "spdlog/spdlog.h"

namespace OmniDSP::Default {

  // Forward declare the internal design helper functions (defined in
  // filter.cpp or similar, ensure they are accessible and correctly templated)
  // Assuming Coefs::FIRFilter<T> is std::vector<T>
  template <typename T>
  [[nodiscard]] OmniExpected<std::vector<T>>
  generate_fir_filter_coeffs(  // Changed Coefs::FIRFilter<T> to std::vector<T>
                               // for clarity
      const Design::FIRFilter& spec);

  [[nodiscard]] OmniExpected<Coefs::IIRFilterSOS> generate_iir_filter_coeffs(
      const Design::IIRFilter& spec);

  namespace detail {
    inline size_t next_power_of_two(size_t n)
    {
      if (n == 0) return 1;  // Should ideally not happen for FFT lengths
      // Check if n is already a power of two
      if ((n > 0) && ((n & (n - 1)) == 0)) {
        return n;
      }
      size_t power = 1;
      while (power < n
             && power < (std::numeric_limits<size_t>::max() / 2U)
                            + 1) {  // Prevent overflow
        power <<= 1;
      }
      return power;
    }
  }  // namespace detail

  //--------------------------------------------------------------------------
  // Backend Method Definitions
  //--------------------------------------------------------------------------
  Backend::Backend()
  {
    // auto logger = spdlog::get("OmniDSP");
    // if (logger) logger->trace("Default::Backend constructed.");
  }
  Backend::~Backend()
  {
    // auto logger = spdlog::get("OmniDSP");
    // if (logger) logger->trace("Default::Backend destructed.");
  }

  BackendType Backend::get_backend() const { return BackendType::Default; }

  // --- DSP Operations (One-Off Implementations) ---

  [[nodiscard]] OmniExpected<F32Vec> Backend::convolve_f32(
      const F32Vec& input,
      const F32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (kernel.empty())
      return OmniExpected<F32Vec>(std::unexpect, OmniStatus::InvalidArgument);
    Params::Convolution params(
        input.size(),
        kernel.size(),
        type,
        method);  // Max input length is current input size
    auto pimpl_expected
        = this->create_convolution_plan_impl_f32(params, kernel);
    if (!pimpl_expected) {
      return OmniExpected<F32Vec>(std::unexpect, pimpl_expected.error());
    }
    // The public ConvolutionPlan::create now also takes Params and span.
    // For one-off, we are directly using the Impl.
    auto plan_impl = std::move(pimpl_expected.value());
    size_t output_len = plan_impl->get_output_length(input.size());
    F32Vec output(output_len);
    OmniStatus status = plan_impl->execute(input, output);
    if (status != OmniStatus::Success) {
      return OmniExpected<F32Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<F64Vec> Backend::convolve_f64(
      const F64Vec& input,
      const F64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (kernel.empty())
      return OmniExpected<F64Vec>(std::unexpect, OmniStatus::InvalidArgument);
    Params::Convolution params(input.size(), kernel.size(), type, method);
    auto pimpl_expected
        = this->create_convolution_plan_impl_f64(params, kernel);
    if (!pimpl_expected) {
      return OmniExpected<F64Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan_impl = std::move(pimpl_expected.value());
    size_t output_len = plan_impl->get_output_length(input.size());
    F64Vec output(output_len);
    OmniStatus status = plan_impl->execute(input, output);
    if (status != OmniStatus::Success) {
      return OmniExpected<F64Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<C32Vec> Backend::convolve_c32(
      const C32Vec& input,
      const C32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (kernel.empty())
      return OmniExpected<C32Vec>(std::unexpect, OmniStatus::InvalidArgument);
    Params::Convolution params(input.size(), kernel.size(), type, method);
    auto pimpl_expected
        = this->create_convolution_plan_impl_c32(params, kernel);
    if (!pimpl_expected) {
      return OmniExpected<C32Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan_impl = std::move(pimpl_expected.value());
    size_t output_len = plan_impl->get_output_length(input.size());
    C32Vec output(output_len);
    OmniStatus status = plan_impl->execute(input, output);
    if (status != OmniStatus::Success) {
      return OmniExpected<C32Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<C64Vec> Backend::convolve_c64(
      const C64Vec& input,
      const C64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (kernel.empty())
      return OmniExpected<C64Vec>(std::unexpect, OmniStatus::InvalidArgument);
    Params::Convolution params(input.size(), kernel.size(), type, method);
    auto pimpl_expected
        = this->create_convolution_plan_impl_c64(params, kernel);
    if (!pimpl_expected) {
      return OmniExpected<C64Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan_impl = std::move(pimpl_expected.value());
    size_t output_len = plan_impl->get_output_length(input.size());
    C64Vec output(output_len);
    OmniStatus status = plan_impl->execute(input, output);
    if (status != OmniStatus::Success) {
      return OmniExpected<C64Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<F32Vec> Backend::correlate_f32(
      const F32Vec& input,
      const F32Vec& template_coeffs,  // Renamed from kernel for clarity
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (template_coeffs.empty())
      return OmniExpected<F32Vec>(std::unexpect, OmniStatus::InvalidArgument);
    Params::Correlation params(
        input.size(), template_coeffs.size(), type, method);
    auto pimpl_expected
        = this->create_correlation_plan_impl_f32(params, template_coeffs);
    if (!pimpl_expected) {
      return OmniExpected<F32Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan_impl = std::move(pimpl_expected.value());
    size_t output_len = plan_impl->get_output_length(input.size());
    F32Vec output(output_len);
    OmniStatus status = plan_impl->execute(input, output);
    if (status != OmniStatus::Success) {
      return OmniExpected<F32Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<F64Vec> Backend::correlate_f64(
      const F64Vec& input,
      const F64Vec& template_coeffs,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (template_coeffs.empty())
      return OmniExpected<F64Vec>(std::unexpect, OmniStatus::InvalidArgument);
    Params::Correlation params(
        input.size(), template_coeffs.size(), type, method);
    auto pimpl_expected
        = this->create_correlation_plan_impl_f64(params, template_coeffs);
    if (!pimpl_expected) {
      return OmniExpected<F64Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan_impl = std::move(pimpl_expected.value());
    size_t output_len = plan_impl->get_output_length(input.size());
    F64Vec output(output_len);
    OmniStatus status = plan_impl->execute(input, output);
    if (status != OmniStatus::Success) {
      return OmniExpected<F64Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<C32Vec> Backend::correlate_c32(
      const C32Vec& input,
      const C32Vec& template_coeffs,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (template_coeffs.empty())
      return OmniExpected<C32Vec>(std::unexpect, OmniStatus::InvalidArgument);
    Params::Correlation params(
        input.size(), template_coeffs.size(), type, method);
    auto pimpl_expected
        = this->create_correlation_plan_impl_c32(params, template_coeffs);
    if (!pimpl_expected) {
      return OmniExpected<C32Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan_impl = std::move(pimpl_expected.value());
    size_t output_len = plan_impl->get_output_length(input.size());
    C32Vec output(output_len);
    OmniStatus status = plan_impl->execute(input, output);
    if (status != OmniStatus::Success) {
      return OmniExpected<C32Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<C64Vec> Backend::correlate_c64(
      const C64Vec& input,
      const C64Vec& template_coeffs,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    if (template_coeffs.empty())
      return OmniExpected<C64Vec>(std::unexpect, OmniStatus::InvalidArgument);
    Params::Correlation params(
        input.size(), template_coeffs.size(), type, method);
    auto pimpl_expected
        = this->create_correlation_plan_impl_c64(params, template_coeffs);
    if (!pimpl_expected) {
      return OmniExpected<C64Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan_impl = std::move(pimpl_expected.value());
    size_t output_len = plan_impl->get_output_length(input.size());
    C64Vec output(output_len);
    OmniStatus status = plan_impl->execute(input, output);
    if (status != OmniStatus::Success) {
      return OmniExpected<C64Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<C32Vec> Backend::fft_c32(const C32Vec& input) const
  {
    auto pimpl_expected = this->create_fft_plan_impl_c32(input.size());
    if (!pimpl_expected) {
      return OmniExpected<C32Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan
        = FFTPlan<C32>::create_from_impl(std::move(pimpl_expected.value()));
    if (!plan) {
      spdlog::get("OmniDSP")->error(
          "Default::fft_c32: Failed to create public FFTPlan from impl.");
      return OmniExpected<C32Vec>(std::unexpect, OmniStatus::Failure);
    }

    C32Vec output(input.size());
    OmniStatus status = plan->fft(input, output);
    if (status != OmniStatus::Success) {
      return OmniExpected<C32Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<C64Vec> Backend::fft_c64(const C64Vec& input) const
  {
    auto pimpl_expected = this->create_fft_plan_impl_c64(input.size());
    if (!pimpl_expected) {
      return OmniExpected<C64Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan
        = FFTPlan<C64>::create_from_impl(std::move(pimpl_expected.value()));
    if (!plan) {
      spdlog::get("OmniDSP")->error(
          "Default::fft_c64: Failed to create public FFTPlan from impl.");
      return OmniExpected<C64Vec>(std::unexpect, OmniStatus::Failure);
    }

    C64Vec output(input.size());
    OmniStatus status = plan->fft(input, output);
    if (status != OmniStatus::Success) {
      return OmniExpected<C64Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<C32Vec> Backend::ifft_c32(
      const C32Vec& input) const
  {
    auto pimpl_expected = this->create_fft_plan_impl_c32(input.size());
    if (!pimpl_expected) {
      return OmniExpected<C32Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan
        = FFTPlan<C32>::create_from_impl(std::move(pimpl_expected.value()));
    if (!plan) {
      spdlog::get("OmniDSP")->error(
          "Default::ifft_c32: Failed to create public FFTPlan from impl.");
      return OmniExpected<C32Vec>(std::unexpect, OmniStatus::Failure);
    }

    C32Vec output(input.size());
    OmniStatus status = plan->ifft(input, output);
    if (status != OmniStatus::Success) {
      return OmniExpected<C32Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<C64Vec> Backend::ifft_c64(
      const C64Vec& input) const
  {
    auto pimpl_expected = this->create_fft_plan_impl_c64(input.size());
    if (!pimpl_expected) {
      return OmniExpected<C64Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan
        = FFTPlan<C64>::create_from_impl(std::move(pimpl_expected.value()));
    if (!plan) {
      spdlog::get("OmniDSP")->error(
          "Default::ifft_c64: Failed to create public FFTPlan from impl.");
      return OmniExpected<C64Vec>(std::unexpect, OmniStatus::Failure);
    }

    C64Vec output(input.size());
    OmniStatus status = plan->ifft(input, output);
    if (status != OmniStatus::Success) {
      return OmniExpected<C64Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<C32Vec> Backend::rfft_f32(
      const F32Vec& input) const
  {
    auto pimpl_expected = this->create_rfft_plan_impl_f32(input.size());
    if (!pimpl_expected) {
      return OmniExpected<C32Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan
        = RFFTPlan<F32>::create_from_impl(std::move(pimpl_expected.value()));
    if (!plan) {
      spdlog::get("OmniDSP")->error(
          "Default::rfft_f32: Failed to create public RFFTPlan from impl.");
      return OmniExpected<C32Vec>(std::unexpect, OmniStatus::Failure);
    }

    size_t output_len = (input.empty()) ? 0 : (input.size() / 2) + 1;
    C32Vec output(output_len);
    OmniStatus status = plan->rfft(input, output);
    if (status != OmniStatus::Success) {
      return OmniExpected<C32Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<C64Vec> Backend::rfft_f64(
      const F64Vec& input) const
  {
    auto pimpl_expected = this->create_rfft_plan_impl_f64(input.size());
    if (!pimpl_expected) {
      return OmniExpected<C64Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan
        = RFFTPlan<F64>::create_from_impl(std::move(pimpl_expected.value()));
    if (!plan) {
      spdlog::get("OmniDSP")->error(
          "Default::rfft_f64: Failed to create public RFFTPlan from impl.");
      return OmniExpected<C64Vec>(std::unexpect, OmniStatus::Failure);
    }

    size_t output_len = (input.empty()) ? 0 : (input.size() / 2) + 1;
    C64Vec output(output_len);
    OmniStatus status = plan->rfft(input, output);
    if (status != OmniStatus::Success) {
      return OmniExpected<C64Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<F32Vec> Backend::irfft_c32(
      const C32Vec& input, size_t output_length) const
  {
    if (output_length == 0) {
      return input.empty() ? F32Vec{}
                           : OmniExpected<F32Vec>(
                                 std::unexpect, OmniStatus::InvalidArgument);
    }
    size_t expected_input_len = (output_length / 2) + 1;
    if (input.size() != expected_input_len) {
      spdlog::get("OmniDSP")->error(
          "Default::irfft_c32: Input size ({}) does not match expected size "
          "({}) for output length {}.",
          input.size(),
          expected_input_len,
          output_length);
      return OmniExpected<F32Vec>(std::unexpect, OmniStatus::SizeMismatch);
    }

    auto pimpl_expected = this->create_rfft_plan_impl_f32(output_length);
    if (!pimpl_expected) {
      return OmniExpected<F32Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan
        = RFFTPlan<F32>::create_from_impl(std::move(pimpl_expected.value()));
    if (!plan) {
      spdlog::get("OmniDSP")->error(
          "Default::irfft_c32: Failed to create public RFFTPlan from impl.");
      return OmniExpected<F32Vec>(std::unexpect, OmniStatus::Failure);
    }

    F32Vec output(output_length);
    OmniStatus status = plan->irfft(input, output);
    if (status != OmniStatus::Success) {
      return OmniExpected<F32Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<F64Vec> Backend::irfft_c64(
      const C64Vec& input, size_t output_length) const
  {
    if (output_length == 0) {
      return input.empty() ? F64Vec{}
                           : OmniExpected<F64Vec>(
                                 std::unexpect, OmniStatus::InvalidArgument);
    }
    size_t expected_input_len = (output_length / 2) + 1;
    if (input.size() != expected_input_len) {
      spdlog::get("OmniDSP")->error(
          "Default::irfft_c64: Input size ({}) does not match expected size "
          "({}) for output length {}.",
          input.size(),
          expected_input_len,
          output_length);
      return OmniExpected<F64Vec>(std::unexpect, OmniStatus::SizeMismatch);
    }

    auto pimpl_expected = this->create_rfft_plan_impl_f64(output_length);
    if (!pimpl_expected) {
      return OmniExpected<F64Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan
        = RFFTPlan<F64>::create_from_impl(std::move(pimpl_expected.value()));
    if (!plan) {
      spdlog::get("OmniDSP")->error(
          "Default::irfft_c64: Failed to create public RFFTPlan from impl.");
      return OmniExpected<F64Vec>(std::unexpect, OmniStatus::Failure);
    }

    F64Vec output(output_length);
    OmniStatus status = plan->irfft(input, output);
    if (status != OmniStatus::Success) {
      return OmniExpected<F64Vec>(std::unexpect, status);
    }
    return output;
  }
  // --- Window Generation (Overrides for specific virtuals from
  // Abstract::Backend) ---
  // ... (Existing implementations for bartlett_window_f32, etc. - keeping them
  // concise here)
  OmniStatus Backend::bartlett_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::bartlett_window_f32: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Bartlett, static_cast<int>(length));
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::bartlett_window_f32: Error creating WindowSetup: {}",
          e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::bartlett_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::bartlett_window_f64: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Bartlett, static_cast<int>(length));
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::bartlett_window_f64: Error creating WindowSetup: {}",
          e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::blackman_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::blackman_window_f32: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Blackman, static_cast<int>(length));
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::blackman_window_f32: {}", e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::blackman_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::blackman_window_f64: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Blackman, static_cast<int>(length));
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::blackman_window_f64: {}", e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::flattop_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::flattop_window_f32: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Flattop, static_cast<int>(length));
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::flattop_window_f32: {}", e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::flattop_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::flattop_window_f64: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Flattop, static_cast<int>(length));
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::flattop_window_f64: {}", e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::gaussian_window_f32(
      size_t length, double stddev, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::gaussian_window_f32: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(
          WindowType::Gaussian,
          static_cast<int>(length),
          WindowParams{{"sigma", stddev}});
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::gaussian_window_f32: {}", e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::gaussian_window_f64(
      size_t length, double stddev, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::gaussian_window_f64: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(
          WindowType::Gaussian,
          static_cast<int>(length),
          WindowParams{{"sigma", stddev}});
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::gaussian_window_f64: {}", e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::hamming_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::hamming_window_f32: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Hamming, static_cast<int>(length));
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::hamming_window_f32: {}", e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::hamming_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::hamming_window_f64: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Hamming, static_cast<int>(length));
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::hamming_window_f64: {}", e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::hann_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::hann_window_f32: Output span too small. Expected {}, got "
          "{}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Hann, static_cast<int>(length));
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error("Default::hann_window_f32: {}", e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::hann_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::hann_window_f64: Output span too small. Expected {}, got "
          "{}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Hann, static_cast<int>(length));
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error("Default::hann_window_f64: {}", e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::kaiser_window_f32(
      size_t length, double beta_param, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::kaiser_window_f32: Output span too small. Expected {}, got "
          "{}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(
          WindowType::Kaiser,
          static_cast<int>(length),
          WindowParams{{"beta", beta_param}});
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error("Default::kaiser_window_f32: {}", e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::kaiser_window_f64(
      size_t length, double beta_param, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::kaiser_window_f64: Output span too small. Expected {}, got "
          "{}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(
          WindowType::Kaiser,
          static_cast<int>(length),
          WindowParams{{"beta", beta_param}});
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error("Default::kaiser_window_f64: {}", e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::rectangular_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::rectangular_window_f32: Output span too small. Expected "
          "{}, got {}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Rectangular, static_cast<int>(length));
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::rectangular_window_f32: {}", e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::rectangular_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::rectangular_window_f64: Output span too small. Expected "
          "{}, got {}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Rectangular, static_cast<int>(length));
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::rectangular_window_f64: {}", e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::triangular_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::triangular_window_f32: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Triangular, static_cast<int>(length));
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::triangular_window_f32: {}", e.what());
      return OmniStatus::InvalidArgument;
    }
  }
  OmniStatus Backend::triangular_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::triangular_window_f64: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return OmniStatus::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Triangular, static_cast<int>(length));
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? OmniStatus::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::triangular_window_f64: {}", e.what());
      return OmniStatus::InvalidArgument;
    }
  }

  // --- Plan Impl Factories ---
  // ... (Existing implementations for create_fft_plan_impl_c32, etc. - keeping
  // them concise here)
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>
  Backend::create_fft_plan_impl_c32(size_t length) const
  {
    try {
      return std::make_unique<FFTPlanImpl<C32>>(length);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::FFTPlanImpl<C32>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>
  Backend::create_fft_plan_impl_c64(size_t length) const
  {
    try {
      return std::make_unique<FFTPlanImpl<C64>>(length);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::FFTPlanImpl<C64>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>
  Backend::create_rfft_plan_impl_f32(size_t length) const
  {
    try {
      return std::make_unique<RFFTPlanImpl<F32>>(length);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::RFFTPlanImpl<F32>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>
  Backend::create_rfft_plan_impl_f64(size_t length) const
  {
    try {
      return std::make_unique<RFFTPlanImpl<F64>>(length);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::RFFTPlanImpl<F64>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<F32>>>
  Backend::create_cqt_processor_impl_f32(const Design::CQT& spec) const
  {
    try {
      return std::make_unique<CQTProcessorImpl<F32>>(this, spec);
    }
    catch (const OmniException& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::CQTProcessorImpl<F32> from Design::CQT: {} "
          "(Status: "
          "{})",
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<F32>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::CQTProcessorImpl<F32> from Design::CQT: {}",
          e.what());
      return OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<F32>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<F64>>>
  Backend::create_cqt_processor_impl_f64(const Design::CQT& spec) const
  {
    try {
      return std::make_unique<Default::CQTProcessorImpl<F64>>(this, spec);
    }
    catch (const OmniException& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::CQTProcessorImpl<F64> from Design::CQT: {} "
          "(Status: "
          "{})",
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<F64>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::CQTProcessorImpl<F64> from Design::CQT: {}",
          e.what());
      return OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<F64>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ResampleProcessorImpl<F32>>>
  Backend::create_resample_processor_impl_f32(
      const Design::Resample& spec) const
  {
    try {
      return std::make_unique<ResampleProcessorImpl<F32>>(this, spec);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::ResampleProcessorImpl<F32>: {}", e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::ResampleProcessorImpl<F32>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ResampleProcessorImpl<F64>>>
  Backend::create_resample_processor_impl_f64(
      const Design::Resample& spec) const
  {
    try {
      return std::make_unique<ResampleProcessorImpl<F64>>(this, spec);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::ResampleProcessorImpl<F64>: {}", e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::ResampleProcessorImpl<F64>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  // ConvolutionPlanImpl Factories (Reverted to explicit, verbose form)
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ConvolutionPlanImpl<F32>>>
  Backend::create_convolution_plan_impl_f32(
      const Params::Convolution& params,
      std::span<const F32> kernel_coeffs) const
  {
    try {
      Default::ConvolutionPlanImpl<F32>::FFTPlanImplVariant fft_plan_variant;
      if (params.method_hint_ == ConvolutionMethod::FFT
          || (params.method_hint_ == ConvolutionMethod::Auto
              && !kernel_coeffs.empty())) {
        size_t kernel_len = kernel_coeffs.size();
        size_t fft_len = params.max_input_length_ + kernel_len - 1;
        if (params.max_input_length_ == 0 && kernel_len == 0) {
          fft_len = 0;
        }
        else if (params.max_input_length_ == 0 || kernel_len == 0) {
          fft_len = std::max(params.max_input_length_, kernel_len);
        }

        if (fft_len > 0) {
          fft_len = detail::next_power_of_two(fft_len);
          if (fft_len == 0) {
            spdlog::get("OmniDSP")->error(
                "FFT length calculation overflowed for F32 convolution.");
            return OmniExpected<
                std::unique_ptr<Abstract::ConvolutionPlanImpl<F32>>>(
                std::unexpect, OmniStatus::InvalidArgument);
          }
          if (fft_len < 2) fft_len = 2;  // RFFT often needs min length 2
          auto rfft_plan_expected = this->create_rfft_plan_impl_f32(fft_len);
          if (!rfft_plan_expected) {
            return OmniExpected<
                std::unique_ptr<Abstract::ConvolutionPlanImpl<F32>>>(
                std::unexpect, rfft_plan_expected.error());
          }
          fft_plan_variant
              .emplace<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>(
                  std::move(rfft_plan_expected.value()));
        }
        else if (
            params.method_hint_
            == ConvolutionMethod::FFT) {  // FFT forced, but lengths are zero
          // Create a minimal FFT plan if FFT is explicitly requested for
          // zero-length operation
          auto rfft_plan_expected
              = this->create_rfft_plan_impl_f32(2);  // Minimal sensible default
          if (!rfft_plan_expected) {
            return OmniExpected<
                std::unique_ptr<Abstract::ConvolutionPlanImpl<F32>>>(
                std::unexpect, rfft_plan_expected.error());
          }
          fft_plan_variant
              .emplace<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>(
                  std::move(rfft_plan_expected.value()));
        }
      }
      return std::make_unique<Default::ConvolutionPlanImpl<F32>>(
          std::move(fft_plan_variant), params, kernel_coeffs);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::ConvolutionPlanImpl<F32>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::ConvolutionPlanImpl<F32>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ConvolutionPlanImpl<F64>>>
  Backend::create_convolution_plan_impl_f64(
      const Params::Convolution& params,
      std::span<const F64> kernel_coeffs) const
  {
    try {
      Default::ConvolutionPlanImpl<F64>::FFTPlanImplVariant fft_plan_variant;
      if (params.method_hint_ == ConvolutionMethod::FFT
          || (params.method_hint_ == ConvolutionMethod::Auto
              && !kernel_coeffs.empty())) {
        size_t kernel_len = kernel_coeffs.size();
        size_t fft_len = params.max_input_length_ + kernel_len - 1;
        if (params.max_input_length_ == 0 && kernel_len == 0) {
          fft_len = 0;
        }
        else if (params.max_input_length_ == 0 || kernel_len == 0) {
          fft_len = std::max(params.max_input_length_, kernel_len);
        }

        if (fft_len > 0) {
          fft_len = detail::next_power_of_two(fft_len);
          if (fft_len == 0) {
            spdlog::get("OmniDSP")->error(
                "FFT length calculation overflowed for F64 convolution.");
            return OmniExpected<
                std::unique_ptr<Abstract::ConvolutionPlanImpl<F64>>>(
                std::unexpect, OmniStatus::InvalidArgument);
          }
          if (fft_len < 2) fft_len = 2;
          auto rfft_plan_expected = this->create_rfft_plan_impl_f64(fft_len);
          if (!rfft_plan_expected) {
            return OmniExpected<
                std::unique_ptr<Abstract::ConvolutionPlanImpl<F64>>>(
                std::unexpect, rfft_plan_expected.error());
          }
          fft_plan_variant
              .emplace<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>(
                  std::move(rfft_plan_expected.value()));
        }
        else if (params.method_hint_ == ConvolutionMethod::FFT) {
          auto rfft_plan_expected = this->create_rfft_plan_impl_f64(2);
          if (!rfft_plan_expected) {
            return OmniExpected<
                std::unique_ptr<Abstract::ConvolutionPlanImpl<F64>>>(
                std::unexpect, rfft_plan_expected.error());
          }
          fft_plan_variant
              .emplace<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>(
                  std::move(rfft_plan_expected.value()));
        }
      }
      return std::make_unique<Default::ConvolutionPlanImpl<F64>>(
          std::move(fft_plan_variant), params, kernel_coeffs);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::ConvolutionPlanImpl<F64>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::ConvolutionPlanImpl<F64>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ConvolutionPlanImpl<C32>>>
  Backend::create_convolution_plan_impl_c32(
      const Params::Convolution& params,
      std::span<const C32> kernel_coeffs) const
  {
    try {
      Default::ConvolutionPlanImpl<C32>::FFTPlanImplVariant fft_plan_variant;
      if (params.method_hint_ == ConvolutionMethod::FFT
          || (params.method_hint_ == ConvolutionMethod::Auto
              && !kernel_coeffs.empty())) {
        size_t kernel_len = kernel_coeffs.size();
        size_t fft_len = params.max_input_length_ + kernel_len - 1;
        if (params.max_input_length_ == 0 && kernel_len == 0) {
          fft_len = 0;
        }
        else if (params.max_input_length_ == 0 || kernel_len == 0) {
          fft_len = std::max(params.max_input_length_, kernel_len);
        }

        if (fft_len > 0) {
          fft_len = detail::next_power_of_two(fft_len);
          if (fft_len == 0) {
            spdlog::get("OmniDSP")->error(
                "FFT length calculation overflowed for C32 convolution.");
            return OmniExpected<
                std::unique_ptr<Abstract::ConvolutionPlanImpl<C32>>>(
                std::unexpect, OmniStatus::InvalidArgument);
          }
          auto cfft_plan_expected = this->create_fft_plan_impl_c32(fft_len);
          if (!cfft_plan_expected) {
            return OmniExpected<
                std::unique_ptr<Abstract::ConvolutionPlanImpl<C32>>>(
                std::unexpect, cfft_plan_expected.error());
          }
          fft_plan_variant.emplace<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>(
              std::move(cfft_plan_expected.value()));
        }
        else if (params.method_hint_ == ConvolutionMethod::FFT) {
          auto cfft_plan_expected = this->create_fft_plan_impl_c32(
              1);  // CFFT can often handle length 1
          if (!cfft_plan_expected) {
            return OmniExpected<
                std::unique_ptr<Abstract::ConvolutionPlanImpl<C32>>>(
                std::unexpect, cfft_plan_expected.error());
          }
          fft_plan_variant.emplace<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>(
              std::move(cfft_plan_expected.value()));
        }
      }
      return std::make_unique<Default::ConvolutionPlanImpl<C32>>(
          std::move(fft_plan_variant), params, kernel_coeffs);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::ConvolutionPlanImpl<C32>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::ConvolutionPlanImpl<C32>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::ConvolutionPlanImpl<C64>>>
  Backend::create_convolution_plan_impl_c64(
      const Params::Convolution& params,
      std::span<const C64> kernel_coeffs) const
  {
    try {
      Default::ConvolutionPlanImpl<C64>::FFTPlanImplVariant fft_plan_variant;
      if (params.method_hint_ == ConvolutionMethod::FFT
          || (params.method_hint_ == ConvolutionMethod::Auto
              && !kernel_coeffs.empty())) {
        size_t kernel_len = kernel_coeffs.size();
        size_t fft_len = params.max_input_length_ + kernel_len - 1;
        if (params.max_input_length_ == 0 && kernel_len == 0) {
          fft_len = 0;
        }
        else if (params.max_input_length_ == 0 || kernel_len == 0) {
          fft_len = std::max(params.max_input_length_, kernel_len);
        }

        if (fft_len > 0) {
          fft_len = detail::next_power_of_two(fft_len);
          if (fft_len == 0) {
            spdlog::get("OmniDSP")->error(
                "FFT length calculation overflowed for C64 convolution.");
            return OmniExpected<
                std::unique_ptr<Abstract::ConvolutionPlanImpl<C64>>>(
                std::unexpect, OmniStatus::InvalidArgument);
          }
          auto cfft_plan_expected = this->create_fft_plan_impl_c64(fft_len);
          if (!cfft_plan_expected) {
            return OmniExpected<
                std::unique_ptr<Abstract::ConvolutionPlanImpl<C64>>>(
                std::unexpect, cfft_plan_expected.error());
          }
          fft_plan_variant.emplace<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>(
              std::move(cfft_plan_expected.value()));
        }
        else if (params.method_hint_ == ConvolutionMethod::FFT) {
          auto cfft_plan_expected = this->create_fft_plan_impl_c64(1);
          if (!cfft_plan_expected) {
            return OmniExpected<
                std::unique_ptr<Abstract::ConvolutionPlanImpl<C64>>>(
                std::unexpect, cfft_plan_expected.error());
          }
          fft_plan_variant.emplace<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>(
              std::move(cfft_plan_expected.value()));
        }
      }
      return std::make_unique<Default::ConvolutionPlanImpl<C64>>(
          std::move(fft_plan_variant), params, kernel_coeffs);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::ConvolutionPlanImpl<C64>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::ConvolutionPlanImpl<C64>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  // CorrelationPlanImpl Factories (Reverted to explicit, verbose form)
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::CorrelationPlanImpl<F32>>>
  Backend::create_correlation_plan_impl_f32(
      const Params::Correlation& params,
      std::span<const F32> template_coeffs) const
  {
    try {
      Default::CorrelationPlanImpl<F32>::FFTPlanImplVariant fft_plan_variant;
      if (params.method_hint_ == ConvolutionMethod::FFT
          || (params.method_hint_ == ConvolutionMethod::Auto
              && !template_coeffs.empty())) {
        size_t template_len = template_coeffs.size();
        size_t fft_len = params.max_input_length_ + template_len - 1;
        if (params.max_input_length_ == 0 && template_len == 0) {
          fft_len = 0;
        }
        else if (params.max_input_length_ == 0 || template_len == 0) {
          fft_len = std::max(params.max_input_length_, template_len);
        }

        if (fft_len > 0) {
          fft_len = detail::next_power_of_two(fft_len);
          if (fft_len == 0) {
            spdlog::get("OmniDSP")->error(
                "FFT length calculation overflowed for F32 correlation.");
            return OmniExpected<
                std::unique_ptr<Abstract::CorrelationPlanImpl<F32>>>(
                std::unexpect, OmniStatus::InvalidArgument);
          }
          if (fft_len < 2) fft_len = 2;
          auto rfft_plan_expected = this->create_rfft_plan_impl_f32(fft_len);
          if (!rfft_plan_expected) {
            return OmniExpected<
                std::unique_ptr<Abstract::CorrelationPlanImpl<F32>>>(
                std::unexpect, rfft_plan_expected.error());
          }
          fft_plan_variant
              .emplace<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>(
                  std::move(rfft_plan_expected.value()));
        }
        else if (params.method_hint_ == ConvolutionMethod::FFT) {
          auto rfft_plan_expected = this->create_rfft_plan_impl_f32(2);
          if (!rfft_plan_expected) {
            return OmniExpected<
                std::unique_ptr<Abstract::CorrelationPlanImpl<F32>>>(
                std::unexpect, rfft_plan_expected.error());
          }
          fft_plan_variant
              .emplace<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>(
                  std::move(rfft_plan_expected.value()));
        }
      }
      return std::make_unique<Default::CorrelationPlanImpl<F32>>(
          std::move(fft_plan_variant), params, template_coeffs);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::CorrelationPlanImpl<F32>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::CorrelationPlanImpl<F32>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::CorrelationPlanImpl<F64>>>
  Backend::create_correlation_plan_impl_f64(
      const Params::Correlation& params,
      std::span<const F64> template_coeffs) const
  {
    try {
      Default::CorrelationPlanImpl<F64>::FFTPlanImplVariant fft_plan_variant;
      if (params.method_hint_ == ConvolutionMethod::FFT
          || (params.method_hint_ == ConvolutionMethod::Auto
              && !template_coeffs.empty())) {
        size_t template_len = template_coeffs.size();
        size_t fft_len = params.max_input_length_ + template_len - 1;
        if (params.max_input_length_ == 0 && template_len == 0) {
          fft_len = 0;
        }
        else if (params.max_input_length_ == 0 || template_len == 0) {
          fft_len = std::max(params.max_input_length_, template_len);
        }

        if (fft_len > 0) {
          fft_len = detail::next_power_of_two(fft_len);
          if (fft_len == 0) {
            spdlog::get("OmniDSP")->error(
                "FFT length calculation overflowed for F64 correlation.");
            return OmniExpected<
                std::unique_ptr<Abstract::CorrelationPlanImpl<F64>>>(
                std::unexpect, OmniStatus::InvalidArgument);
          }
          if (fft_len < 2) fft_len = 2;
          auto rfft_plan_expected = this->create_rfft_plan_impl_f64(fft_len);
          if (!rfft_plan_expected) {
            return OmniExpected<
                std::unique_ptr<Abstract::CorrelationPlanImpl<F64>>>(
                std::unexpect, rfft_plan_expected.error());
          }
          fft_plan_variant
              .emplace<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>(
                  std::move(rfft_plan_expected.value()));
        }
        else if (params.method_hint_ == ConvolutionMethod::FFT) {
          auto rfft_plan_expected = this->create_rfft_plan_impl_f64(2);
          if (!rfft_plan_expected) {
            return OmniExpected<
                std::unique_ptr<Abstract::CorrelationPlanImpl<F64>>>(
                std::unexpect, rfft_plan_expected.error());
          }
          fft_plan_variant
              .emplace<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>(
                  std::move(rfft_plan_expected.value()));
        }
      }
      return std::make_unique<Default::CorrelationPlanImpl<F64>>(
          std::move(fft_plan_variant), params, template_coeffs);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::CorrelationPlanImpl<F64>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::CorrelationPlanImpl<F64>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::CorrelationPlanImpl<C32>>>
  Backend::create_correlation_plan_impl_c32(
      const Params::Correlation& params,
      std::span<const C32> template_coeffs) const
  {
    try {
      Default::CorrelationPlanImpl<C32>::FFTPlanImplVariant fft_plan_variant;
      if (params.method_hint_ == ConvolutionMethod::FFT
          || (params.method_hint_ == ConvolutionMethod::Auto
              && !template_coeffs.empty())) {
        size_t template_len = template_coeffs.size();
        size_t fft_len = params.max_input_length_ + template_len - 1;
        if (params.max_input_length_ == 0 && template_len == 0) {
          fft_len = 0;
        }
        else if (params.max_input_length_ == 0 || template_len == 0) {
          fft_len = std::max(params.max_input_length_, template_len);
        }

        if (fft_len > 0) {
          fft_len = detail::next_power_of_two(fft_len);
          if (fft_len == 0) {
            spdlog::get("OmniDSP")->error(
                "FFT length calculation overflowed for C32 correlation.");
            return OmniExpected<
                std::unique_ptr<Abstract::CorrelationPlanImpl<C32>>>(
                std::unexpect, OmniStatus::InvalidArgument);
          }
          auto cfft_plan_expected = this->create_fft_plan_impl_c32(fft_len);
          if (!cfft_plan_expected) {
            return OmniExpected<
                std::unique_ptr<Abstract::CorrelationPlanImpl<C32>>>(
                std::unexpect, cfft_plan_expected.error());
          }
          fft_plan_variant.emplace<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>(
              std::move(cfft_plan_expected.value()));
        }
        else if (params.method_hint_ == ConvolutionMethod::FFT) {
          auto cfft_plan_expected = this->create_fft_plan_impl_c32(1);
          if (!cfft_plan_expected) {
            return OmniExpected<
                std::unique_ptr<Abstract::CorrelationPlanImpl<C32>>>(
                std::unexpect, cfft_plan_expected.error());
          }
          fft_plan_variant.emplace<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>(
              std::move(cfft_plan_expected.value()));
        }
      }
      return std::make_unique<Default::CorrelationPlanImpl<C32>>(
          std::move(fft_plan_variant), params, template_coeffs);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::CorrelationPlanImpl<C32>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::CorrelationPlanImpl<C32>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::CorrelationPlanImpl<C64>>>
  Backend::create_correlation_plan_impl_c64(
      const Params::Correlation& params,
      std::span<const C64> template_coeffs) const
  {
    try {
      Default::CorrelationPlanImpl<C64>::FFTPlanImplVariant fft_plan_variant;
      if (params.method_hint_ == ConvolutionMethod::FFT
          || (params.method_hint_ == ConvolutionMethod::Auto
              && !template_coeffs.empty())) {
        size_t template_len = template_coeffs.size();
        size_t fft_len = params.max_input_length_ + template_len - 1;
        if (params.max_input_length_ == 0 && template_len == 0) {
          fft_len = 0;
        }
        else if (params.max_input_length_ == 0 || template_len == 0) {
          fft_len = std::max(params.max_input_length_, template_len);
        }

        if (fft_len > 0) {
          fft_len = detail::next_power_of_two(fft_len);
          if (fft_len == 0) {
            spdlog::get("OmniDSP")->error(
                "FFT length calculation overflowed for C64 correlation.");
            return OmniExpected<
                std::unique_ptr<Abstract::CorrelationPlanImpl<C64>>>(
                std::unexpect, OmniStatus::InvalidArgument);
          }
          auto cfft_plan_expected = this->create_fft_plan_impl_c64(fft_len);
          if (!cfft_plan_expected) {
            return OmniExpected<
                std::unique_ptr<Abstract::CorrelationPlanImpl<C64>>>(
                std::unexpect, cfft_plan_expected.error());
          }
          fft_plan_variant.emplace<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>(
              std::move(cfft_plan_expected.value()));
        }
        else if (params.method_hint_ == ConvolutionMethod::FFT) {
          auto cfft_plan_expected = this->create_fft_plan_impl_c64(1);
          if (!cfft_plan_expected) {
            return OmniExpected<
                std::unique_ptr<Abstract::CorrelationPlanImpl<C64>>>(
                std::unexpect, cfft_plan_expected.error());
          }
          fft_plan_variant.emplace<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>(
              std::move(cfft_plan_expected.value()));
        }
      }
      return std::make_unique<Default::CorrelationPlanImpl<C64>>(
          std::move(fft_plan_variant), params, template_coeffs);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::CorrelationPlanImpl<C64>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::CorrelationPlanImpl<C64>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::FIRFilterProcessorImpl<F32>>>
  Backend::create_fir_filter_processor_impl_f32(
      const F32Vec& coefficients) const
  {
    try {
      return std::make_unique<FIRFilterProcessorImpl<F32>>(coefficients);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::FIRPlanImpl<F32>: {}", e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::FIRFilterProcessorImpl<F32>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::FIRFilterProcessorImpl<F64>>>
  Backend::create_fir_filter_processor_impl_f64(
      const F64Vec& coefficients) const
  {
    try {
      return std::make_unique<FIRFilterProcessorImpl<F64>>(coefficients);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::FIRPlanImpl<F64>: {}", e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::FIRFilterProcessorImpl<F64>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::FIRFilterProcessorImpl<C32>>>
  Backend::create_fir_filter_processor_impl_c32(
      const C32Vec& coefficients) const
  {
    try {
      return std::make_unique<FIRFilterProcessorImpl<C32>>(coefficients);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::FIRPlanImpl<C32>: {}", e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::FIRFilterProcessorImpl<C32>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::FIRFilterProcessorImpl<C64>>>
  Backend::create_fir_filter_processor_impl_c64(
      const C64Vec& coefficients) const
  {
    try {
      return std::make_unique<FIRFilterProcessorImpl<C64>>(coefficients);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::FIRPlanImpl<C64>: {}", e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::FIRFilterProcessorImpl<C64>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::IIRFilterProcessorImpl<F32>>>
  Backend::create_iir_filter_processor_impl_f32(
      const Coefs::IIRFilterSOS& sos_coefficients) const
  {
    try {
      return std::make_unique<IIRFilterProcessorImpl<F32>>(sos_coefficients);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::IIRPlanImpl<F32>: {}", e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::IIRFilterProcessorImpl<F32>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }
  [[nodiscard]] OmniExpected<
      std::unique_ptr<Abstract::IIRFilterProcessorImpl<F64>>>
  Backend::create_iir_filter_processor_impl_f64(
      const Coefs::IIRFilterSOS& sos_coefficients) const
  {
    try {
      return std::make_unique<IIRFilterProcessorImpl<F64>>(sos_coefficients);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::IIRPlanImpl<F64>: {}", e.what());
      return OmniExpected<
          std::unique_ptr<Abstract::IIRFilterProcessorImpl<F64>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  // --- Filter Design ---
  [[nodiscard]] OmniExpected<Coefs::FIRFilter<F32>>
  Backend::design_fir_filter_f32(const Design::FIRFilter& spec) const
  {
    // Assuming Coefs::FIRFilter<F32> is std::vector<F32>
    return generate_fir_filter_coeffs<F32>(spec);
  }

  [[nodiscard]] OmniExpected<Coefs::FIRFilter<F64>>
  Backend::design_fir_filter_f64(const Design::FIRFilter& spec) const
  {
    // Assuming Coefs::FIRFilter<F64> is std::vector<F64>
    return generate_fir_filter_coeffs<F64>(spec);
  }

  // Implementations for complex FIR filter design
  [[nodiscard]] OmniExpected<Coefs::FIRFilter<C32>>
  Backend::design_fir_filter_c32(const Design::FIRFilter& spec) const
  {
    // generate_fir_filter_coeffs<std::complex<float>> should return
    // OmniExpected<std::vector<std::complex<float>>> which is compatible with
    // OmniExpected<C32Vec> if C32Vec is std::vector<std::complex<float>> and
    // Coefs::FIRFilter<T> is std::vector<T>.
    return generate_fir_filter_coeffs<C32>(spec);
  }

  [[nodiscard]] OmniExpected<Coefs::FIRFilter<C64>>
  Backend::design_fir_filter_c64(const Design::FIRFilter& spec) const
  {
    // generate_fir_filter_coeffs<std::complex<double>> should return
    // OmniExpected<std::vector<std::complex<double>>> which is compatible with
    // OmniExpected<C64Vec> if C64Vec is std::vector<std::complex<double>> and
    // Coefs::FIRFilter<T> is std::vector<T>.
    return generate_fir_filter_coeffs<C64>(spec);
  }

  [[nodiscard]] OmniExpected<Coefs::IIRFilterSOS>
  Backend::design_iir_filter_f32(const Design::IIRFilter& spec) const
  {
    return generate_iir_filter_coeffs(spec);
  }

  [[nodiscard]] OmniExpected<Coefs::IIRFilterSOS>
  Backend::design_iir_filter_f64(const Design::IIRFilter& spec) const
  {
    return generate_iir_filter_coeffs(spec);
  }

}  // namespace OmniDSP::Default

// --- Factory function for Default Backend ---
namespace OmniDSP::Abstract {
  std::unique_ptr<Backend> create_default_backend()
  {
    return std::make_unique<Default::Backend>();
  }
}  // namespace OmniDSP::Abstract
