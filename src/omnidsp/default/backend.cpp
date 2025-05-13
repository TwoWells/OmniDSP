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
#include <OmniDSP/filter.hpp>  // For FIRFilterPlan, IIRFilterPlan, FIRCoefs, IIRFilterCoef, Design::FIRFilter, Design::IIRFilter
#include <OmniDSP/resample.hpp>  // For ResamplePlan, Design::Resample
#include <OmniDSP/window.hpp>  // For WindowSetup, WindowParams, and the free OmniDSP::generate_window

// Standard library headers
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
#include <vector>

// spdlog for logging
#include "spdlog/spdlog.h"

namespace OmniDSP::Default {

  // Forward declare the internal design helper functions (defined in
  // filter.cpp)
  template <typename T>
  [[nodiscard]] OmniExpected<FIRCoefs<T>> generate_fir_filter_coeffs(
      const Design::FIRFilter& spec);
  [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>>
  generate_iir_filter_coeffs(const Design::IIRFilter& spec);

  namespace convolution_detail {
    inline size_t next_power_of_two(size_t n)
    {
      if (n == 0) return 1;
      if ((n > 0) && ((n & (n - 1)) == 0)) {
        return n;
      }
      size_t power = 1;
      while (power < n
             && power < (std::numeric_limits<size_t>::max() / 2U) + 1) {
        power <<= 1;
      }
      return power;
    }
  }  // namespace convolution_detail

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
  // These create a temporary plan using the backend's *Impl factory,
  // wrap it in a public Plan object, execute, and return the result.

  [[nodiscard]] OmniExpected<F32Vec> Backend::convolve_f32(
      const F32Vec& input,
      const F32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    auto pimpl_expected
        = this->create_convolution_plan_impl_f32(kernel, type, method);
    if (!pimpl_expected) {
      return OmniExpected<F32Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan = ConvolutionPlan<F32>::create_from_impl(
        std::move(pimpl_expected.value()));
    if (!plan) {
      spdlog::get("OmniDSP")->error(
          "Default::convolve_f32: Failed to create public ConvolutionPlan from "
          "impl.");
      return OmniExpected<F32Vec>(std::unexpect, Status::Failure);
    }

    size_t output_len = plan->get_output_length(input.size());
    F32Vec output(output_len);
    Status status = plan->execute(input, output);
    if (status != Status::Success) {
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
    auto pimpl_expected
        = this->create_convolution_plan_impl_f64(kernel, type, method);
    if (!pimpl_expected) {
      return OmniExpected<F64Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan = ConvolutionPlan<F64>::create_from_impl(
        std::move(pimpl_expected.value()));
    if (!plan) {
      spdlog::get("OmniDSP")->error(
          "Default::convolve_f64: Failed to create public ConvolutionPlan from "
          "impl.");
      return OmniExpected<F64Vec>(std::unexpect, Status::Failure);
    }

    size_t output_len = plan->get_output_length(input.size());
    F64Vec output(output_len);
    Status status = plan->execute(input, output);
    if (status != Status::Success) {
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
    auto pimpl_expected
        = this->create_convolution_plan_impl_c32(kernel, type, method);
    if (!pimpl_expected) {
      return OmniExpected<C32Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan = ConvolutionPlan<C32>::create_from_impl(
        std::move(pimpl_expected.value()));
    if (!plan) {
      spdlog::get("OmniDSP")->error(
          "Default::convolve_c32: Failed to create public ConvolutionPlan from "
          "impl.");
      return OmniExpected<C32Vec>(std::unexpect, Status::Failure);
    }

    size_t output_len = plan->get_output_length(input.size());
    C32Vec output(output_len);
    Status status = plan->execute(input, output);
    if (status != Status::Success) {
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
    auto pimpl_expected
        = this->create_convolution_plan_impl_c64(kernel, type, method);
    if (!pimpl_expected) {
      return OmniExpected<C64Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan = ConvolutionPlan<C64>::create_from_impl(
        std::move(pimpl_expected.value()));
    if (!plan) {
      spdlog::get("OmniDSP")->error(
          "Default::convolve_c64: Failed to create public ConvolutionPlan from "
          "impl.");
      return OmniExpected<C64Vec>(std::unexpect, Status::Failure);
    }

    size_t output_len = plan->get_output_length(input.size());
    C64Vec output(output_len);
    Status status = plan->execute(input, output);
    if (status != Status::Success) {
      return OmniExpected<C64Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<F32Vec> Backend::correlate_f32(
      const F32Vec& input,
      const F32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    auto pimpl_expected
        = this->create_correlation_plan_impl_f32(kernel, type, method);
    if (!pimpl_expected) {
      return OmniExpected<F32Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan = CorrelationPlan<F32>::create_from_impl(
        std::move(pimpl_expected.value()));
    if (!plan) {
      spdlog::get("OmniDSP")->error(
          "Default::correlate_f32: Failed to create public CorrelationPlan "
          "from impl.");
      return OmniExpected<F32Vec>(std::unexpect, Status::Failure);
    }

    size_t output_len = plan->get_output_length(input.size());
    F32Vec output(output_len);
    Status status = plan->execute(input, output);
    if (status != Status::Success) {
      return OmniExpected<F32Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<F64Vec> Backend::correlate_f64(
      const F64Vec& input,
      const F64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    auto pimpl_expected
        = this->create_correlation_plan_impl_f64(kernel, type, method);
    if (!pimpl_expected) {
      return OmniExpected<F64Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan = CorrelationPlan<F64>::create_from_impl(
        std::move(pimpl_expected.value()));
    if (!plan) {
      spdlog::get("OmniDSP")->error(
          "Default::correlate_f64: Failed to create public CorrelationPlan "
          "from impl.");
      return OmniExpected<F64Vec>(std::unexpect, Status::Failure);
    }

    size_t output_len = plan->get_output_length(input.size());
    F64Vec output(output_len);
    Status status = plan->execute(input, output);
    if (status != Status::Success) {
      return OmniExpected<F64Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<C32Vec> Backend::correlate_c32(
      const C32Vec& input,
      const C32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    auto pimpl_expected
        = this->create_correlation_plan_impl_c32(kernel, type, method);
    if (!pimpl_expected) {
      return OmniExpected<C32Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan = CorrelationPlan<C32>::create_from_impl(
        std::move(pimpl_expected.value()));
    if (!plan) {
      spdlog::get("OmniDSP")->error(
          "Default::correlate_c32: Failed to create public CorrelationPlan "
          "from impl.");
      return OmniExpected<C32Vec>(std::unexpect, Status::Failure);
    }

    size_t output_len = plan->get_output_length(input.size());
    C32Vec output(output_len);
    Status status = plan->execute(input, output);
    if (status != Status::Success) {
      return OmniExpected<C32Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<C64Vec> Backend::correlate_c64(
      const C64Vec& input,
      const C64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    auto pimpl_expected
        = this->create_correlation_plan_impl_c64(kernel, type, method);
    if (!pimpl_expected) {
      return OmniExpected<C64Vec>(std::unexpect, pimpl_expected.error());
    }
    auto plan = CorrelationPlan<C64>::create_from_impl(
        std::move(pimpl_expected.value()));
    if (!plan) {
      spdlog::get("OmniDSP")->error(
          "Default::correlate_c64: Failed to create public CorrelationPlan "
          "from impl.");
      return OmniExpected<C64Vec>(std::unexpect, Status::Failure);
    }

    size_t output_len = plan->get_output_length(input.size());
    C64Vec output(output_len);
    Status status = plan->execute(input, output);
    if (status != Status::Success) {
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
      return OmniExpected<C32Vec>(std::unexpect, Status::Failure);
    }

    C32Vec output(input.size());
    Status status = plan->fft(input, output);
    if (status != Status::Success) {
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
      return OmniExpected<C64Vec>(std::unexpect, Status::Failure);
    }

    C64Vec output(input.size());
    Status status = plan->fft(input, output);
    if (status != Status::Success) {
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
      return OmniExpected<C32Vec>(std::unexpect, Status::Failure);
    }

    C32Vec output(input.size());
    Status status = plan->ifft(input, output);
    if (status != Status::Success) {
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
      return OmniExpected<C64Vec>(std::unexpect, Status::Failure);
    }

    C64Vec output(input.size());
    Status status = plan->ifft(input, output);
    if (status != Status::Success) {
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
      return OmniExpected<C32Vec>(std::unexpect, Status::Failure);
    }

    size_t output_len = (input.empty()) ? 0 : (input.size() / 2) + 1;
    C32Vec output(output_len);
    Status status = plan->rfft(input, output);
    if (status != Status::Success) {
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
      return OmniExpected<C64Vec>(std::unexpect, Status::Failure);
    }

    size_t output_len = (input.empty()) ? 0 : (input.size() / 2) + 1;
    C64Vec output(output_len);
    Status status = plan->rfft(input, output);
    if (status != Status::Success) {
      return OmniExpected<C64Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<F32Vec> Backend::irfft_c32(
      const C32Vec& input, size_t output_length) const
  {
    if (output_length == 0) {
      return input.empty()
                 ? F32Vec{}
                 : OmniExpected<F32Vec>(std::unexpect, Status::InvalidArgument);
    }
    size_t expected_input_len = (output_length / 2) + 1;
    if (input.size() != expected_input_len) {
      spdlog::get("OmniDSP")->error(
          "Default::irfft_c32: Input size ({}) does not match expected size "
          "({}) for output length {}.",
          input.size(),
          expected_input_len,
          output_length);
      return OmniExpected<F32Vec>(std::unexpect, Status::SizeMismatch);
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
      return OmniExpected<F32Vec>(std::unexpect, Status::Failure);
    }

    F32Vec output(output_length);
    Status status = plan->irfft(input, output);
    if (status != Status::Success) {
      return OmniExpected<F32Vec>(std::unexpect, status);
    }
    return output;
  }

  [[nodiscard]] OmniExpected<F64Vec> Backend::irfft_c64(
      const C64Vec& input, size_t output_length) const
  {
    if (output_length == 0) {
      return input.empty()
                 ? F64Vec{}
                 : OmniExpected<F64Vec>(std::unexpect, Status::InvalidArgument);
    }
    size_t expected_input_len = (output_length / 2) + 1;
    if (input.size() != expected_input_len) {
      spdlog::get("OmniDSP")->error(
          "Default::irfft_c64: Input size ({}) does not match expected size "
          "({}) for output length {}.",
          input.size(),
          expected_input_len,
          output_length);
      return OmniExpected<F64Vec>(std::unexpect, Status::SizeMismatch);
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
      return OmniExpected<F64Vec>(std::unexpect, Status::Failure);
    }

    F64Vec output(output_length);
    Status status = plan->irfft(input, output);
    if (status != Status::Success) {
      return OmniExpected<F64Vec>(std::unexpect, status);
    }
    return output;
  }

  // --- Window Generation (Overrides for specific virtuals from
  // Abstract::Backend) ---
  Status Backend::bartlett_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::bartlett_window_f32: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Bartlett, static_cast<int>(length));
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::bartlett_window_f32: Error creating WindowSetup: {}",
          e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::bartlett_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::bartlett_window_f64: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Bartlett, static_cast<int>(length));
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::bartlett_window_f64: Error creating WindowSetup: {}",
          e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::blackman_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::blackman_window_f32: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Blackman, static_cast<int>(length));
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::blackman_window_f32: {}", e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::blackman_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::blackman_window_f64: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Blackman, static_cast<int>(length));
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::blackman_window_f64: {}", e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::flattop_window_f32(size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::flattop_window_f32: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Flattop, static_cast<int>(length));
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::flattop_window_f32: {}", e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::flattop_window_f64(size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::flattop_window_f64: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Flattop, static_cast<int>(length));
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::flattop_window_f64: {}", e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::gaussian_window_f32(
      size_t length, double stddev, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::gaussian_window_f32: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(
          WindowType::Gaussian,
          static_cast<int>(length),
          WindowParams{{"sigma", stddev}});
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::gaussian_window_f32: {}", e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::gaussian_window_f64(
      size_t length, double stddev, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::gaussian_window_f64: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(
          WindowType::Gaussian,
          static_cast<int>(length),
          WindowParams{{"sigma", stddev}});
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::gaussian_window_f64: {}", e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::hamming_window_f32(size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::hamming_window_f32: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Hamming, static_cast<int>(length));
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::hamming_window_f32: {}", e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::hamming_window_f64(size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::hamming_window_f64: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Hamming, static_cast<int>(length));
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::hamming_window_f64: {}", e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::hann_window_f32(size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::hann_window_f32: Output span too small. Expected {}, got "
          "{}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Hann, static_cast<int>(length));
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error("Default::hann_window_f32: {}", e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::hann_window_f64(size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::hann_window_f64: Output span too small. Expected {}, got "
          "{}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Hann, static_cast<int>(length));
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error("Default::hann_window_f64: {}", e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::kaiser_window_f32(
      size_t length, double beta_param, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::kaiser_window_f32: Output span too small. Expected {}, got "
          "{}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(
          WindowType::Kaiser,
          static_cast<int>(length),
          WindowParams{{"beta", beta_param}});
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error("Default::kaiser_window_f32: {}", e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::kaiser_window_f64(
      size_t length, double beta_param, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::kaiser_window_f64: Output span too small. Expected {}, got "
          "{}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(
          WindowType::Kaiser,
          static_cast<int>(length),
          WindowParams{{"beta", beta_param}});
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error("Default::kaiser_window_f64: {}", e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::rectangular_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::rectangular_window_f32: Output span too small. Expected "
          "{}, got {}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Rectangular, static_cast<int>(length));
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::rectangular_window_f32: {}", e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::rectangular_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::rectangular_window_f64: Output span too small. Expected "
          "{}, got {}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Rectangular, static_cast<int>(length));
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::rectangular_window_f64: {}", e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::triangular_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::triangular_window_f32: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Triangular, static_cast<int>(length));
      auto result = generate_window<F32>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::triangular_window_f32: {}", e.what());
      return Status::InvalidArgument;
    }
  }
  Status Backend::triangular_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length && length > 0) {
      spdlog::get("OmniDSP")->warn(
          "Default::triangular_window_f64: Output span too small. Expected {}, "
          "got {}.",
          length,
          output.size());
      return Status::SizeMismatch;
    }
    try {
      WindowSetup setup(WindowType::Triangular, static_cast<int>(length));
      auto result = generate_window<F64>(setup, output.first(length));
      return result ? Status::Success : result.error();
    }
    catch (const std::invalid_argument& e) {
      spdlog::get("OmniDSP")->error(
          "Default::triangular_window_f64: {}", e.what());
      return Status::InvalidArgument;
    }
  }

  // --- Plan Impl Factories ---
  // These methods create the Default backend's specific plan implementations.
  // They are called by the public create_*_plan methods in OmniDSP (via
  // Abstract::Backend's overridden methods).

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
          std::unexpect, Status::Failure);
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
          std::unexpect, Status::Failure);
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
          std::unexpect, Status::Failure);
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
          std::unexpect, Status::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::CQTPlanImpl<F32>>>
  Backend::create_cqt_plan_impl_f32(
      const Design::CQT& spec) const  // Updated signature
  {
    try {
      // Call the CQTPlanImpl constructor with 'this' (owner_backend) and the
      // 'spec'
      return std::make_unique<CQTPlanImpl<F32>>(this, spec);
    }
    catch (const OmniException& e) {  // Catch OmniException specifically if
                                      // thrown by CQTPlanImpl
      spdlog::get("OmniDSP")->error(
          "Error creating Default::CQTPlanImpl<F32> from Design::CQT: {} "
          "(Status: "
          "{})",
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<std::unique_ptr<Abstract::CQTPlanImpl<F32>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::CQTPlanImpl<F32> from Design::CQT: {}",
          e.what());
      return OmniExpected<std::unique_ptr<Abstract::CQTPlanImpl<F32>>>(
          std::unexpect,
          Status::Failure);  // Or a more specific error if identifiable
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::CQTPlanImpl<F64>>>
  Backend::create_cqt_plan_impl_f64(
      const Design::CQT& spec) const  // Updated signature
  {
    try {
      // Call the CQTPlanImpl constructor with 'this' (owner_backend) and the
      // 'spec'
      return std::make_unique<Default::CQTPlanImpl<F64>>(this, spec);
    }
    catch (const OmniException& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::CQTPlanImpl<F64> from Design::CQT: {} "
          "(Status: "
          "{})",
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<std::unique_ptr<Abstract::CQTPlanImpl<F64>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::CQTPlanImpl<F64> from Design::CQT: {}",
          e.what());
      return OmniExpected<std::unique_ptr<Abstract::CQTPlanImpl<F64>>>(
          std::unexpect, Status::Failure);  // Or a more specific error
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::ResamplePlanImpl<F32>>>
  Backend::create_resample_plan_impl_f32(const Design::Resample& spec) const
  {
    try {
      return std::make_unique<ResamplePlanImpl<F32>>(this, spec);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::ResamplePlanImpl<F32>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::ResamplePlanImpl<F32>>>(
          std::unexpect, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::ResamplePlanImpl<F64>>>
  Backend::create_resample_plan_impl_f64(const Design::Resample& spec) const
  {
    try {
      return std::make_unique<ResamplePlanImpl<F64>>(this, spec);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::ResamplePlanImpl<F64>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::ResamplePlanImpl<F64>>>(
          std::unexpect, Status::Failure);
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
      ConvolutionPlanImpl<F32>::FFTPlanImplVariant fft_plan_variant;
      if (method == ConvolutionMethod::FFT || (method == ConvolutionMethod::Auto /* && decide_if_fft_is_better() based on kernel/input sizes */)) {
        size_t kernel_length = kernel.size();
        if (kernel_length == 0)
          return OmniExpected<
              std::unique_ptr<Abstract::ConvolutionPlanImpl<F32>>>(
              std::unexpect, Status::InvalidArgument);

        size_t temp_input_len_for_fft_calc = kernel_length;
        size_t fft_len = convolution_detail::next_power_of_two(
            kernel_length + temp_input_len_for_fft_calc - 1);
        if (fft_len == 0)
          return OmniExpected<
              std::unique_ptr<Abstract::ConvolutionPlanImpl<F32>>>(
              std::unexpect, Status::InvalidArgument);
        if (fft_len < 2) fft_len = 2;

        auto rfft_plan_expected = this->create_rfft_plan_impl_f32(fft_len);
        if (!rfft_plan_expected)
          return OmniExpected<
              std::unique_ptr<Abstract::ConvolutionPlanImpl<F32>>>(
              std::unexpect, rfft_plan_expected.error());
        fft_plan_variant.emplace<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>(
            std::move(rfft_plan_expected.value()));
      }
      return std::make_unique<ConvolutionPlanImpl<F32>>(
          std::move(fft_plan_variant), kernel, type, method);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error(
          "Error creating Default::ConvolutionPlanImpl<F32>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::ConvolutionPlanImpl<F32>>>(
          std::unexpect, Status::Failure);
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
      ConvolutionPlanImpl<F64>::FFTPlanImplVariant fft_plan_variant;
      if (method == ConvolutionMethod::FFT
          || (method == ConvolutionMethod::Auto)) {
        size_t kernel_length = kernel.size();
        if (kernel_length == 0)
          return OmniExpected<
              std::unique_ptr<Abstract::ConvolutionPlanImpl<F64>>>(
              std::unexpect, Status::InvalidArgument);
        size_t temp_input_len_for_fft_calc = kernel_length;
        size_t fft_len = convolution_detail::next_power_of_two(
            kernel_length + temp_input_len_for_fft_calc - 1);
        if (fft_len == 0)
          return OmniExpected<
              std::unique_ptr<Abstract::ConvolutionPlanImpl<F64>>>(
              std::unexpect, Status::InvalidArgument);
        if (fft_len < 2) fft_len = 2;
        auto rfft_plan_expected = this->create_rfft_plan_impl_f64(fft_len);
        if (!rfft_plan_expected)
          return OmniExpected<
              std::unique_ptr<Abstract::ConvolutionPlanImpl<F64>>>(
              std::unexpect, rfft_plan_expected.error());
        fft_plan_variant.emplace<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>(
            std::move(rfft_plan_expected.value()));
      }
      return std::make_unique<ConvolutionPlanImpl<F64>>(
          std::move(fft_plan_variant), kernel, type, method);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::ConvPlanImpl<F64>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::ConvolutionPlanImpl<F64>>>(
          std::unexpect, Status::Failure);
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
      ConvolutionPlanImpl<C32>::FFTPlanImplVariant fft_plan_variant;
      if (method == ConvolutionMethod::FFT
          || (method == ConvolutionMethod::Auto)) {
        size_t kernel_length = kernel.size();
        if (kernel_length == 0)
          return OmniExpected<
              std::unique_ptr<Abstract::ConvolutionPlanImpl<C32>>>(
              std::unexpect, Status::InvalidArgument);
        size_t temp_input_len_for_fft_calc = kernel_length;
        size_t fft_len = convolution_detail::next_power_of_two(
            kernel_length + temp_input_len_for_fft_calc - 1);
        if (fft_len == 0)
          return OmniExpected<
              std::unique_ptr<Abstract::ConvolutionPlanImpl<C32>>>(
              std::unexpect, Status::InvalidArgument);
        auto cfft_plan_expected = this->create_fft_plan_impl_c32(fft_len);
        if (!cfft_plan_expected)
          return OmniExpected<
              std::unique_ptr<Abstract::ConvolutionPlanImpl<C32>>>(
              std::unexpect, cfft_plan_expected.error());
        fft_plan_variant.emplace<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>(
            std::move(cfft_plan_expected.value()));
      }
      return std::make_unique<ConvolutionPlanImpl<C32>>(
          std::move(fft_plan_variant), kernel, type, method);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::ConvPlanImpl<C32>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::ConvolutionPlanImpl<C32>>>(
          std::unexpect, Status::Failure);
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
      ConvolutionPlanImpl<C64>::FFTPlanImplVariant fft_plan_variant;
      if (method == ConvolutionMethod::FFT
          || (method == ConvolutionMethod::Auto)) {
        size_t kernel_length = kernel.size();
        if (kernel_length == 0)
          return OmniExpected<
              std::unique_ptr<Abstract::ConvolutionPlanImpl<C64>>>(
              std::unexpect, Status::InvalidArgument);
        size_t temp_input_len_for_fft_calc = kernel_length;
        size_t fft_len = convolution_detail::next_power_of_two(
            kernel_length + temp_input_len_for_fft_calc - 1);
        if (fft_len == 0)
          return OmniExpected<
              std::unique_ptr<Abstract::ConvolutionPlanImpl<C64>>>(
              std::unexpect, Status::InvalidArgument);
        auto cfft_plan_expected = this->create_fft_plan_impl_c64(fft_len);
        if (!cfft_plan_expected)
          return OmniExpected<
              std::unique_ptr<Abstract::ConvolutionPlanImpl<C64>>>(
              std::unexpect, cfft_plan_expected.error());
        fft_plan_variant.emplace<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>(
            std::move(cfft_plan_expected.value()));
      }
      return std::make_unique<ConvolutionPlanImpl<C64>>(
          std::move(fft_plan_variant), kernel, type, method);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::ConvPlanImpl<C64>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::ConvolutionPlanImpl<C64>>>(
          std::unexpect, Status::Failure);
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
      CorrelationPlanImpl<F32>::FFTPlanImplVariant fft_plan_variant;
      if (method == ConvolutionMethod::FFT
          || (method == ConvolutionMethod::Auto)) {
        size_t kernel_length = kernel.size();
        if (kernel_length == 0)
          return OmniExpected<
              std::unique_ptr<Abstract::CorrelationPlanImpl<F32>>>(
              std::unexpect, Status::InvalidArgument);
        size_t temp_input_len_for_fft_calc = kernel_length;
        size_t fft_len = convolution_detail::next_power_of_two(
            kernel_length + temp_input_len_for_fft_calc - 1);
        if (fft_len == 0)
          return OmniExpected<
              std::unique_ptr<Abstract::CorrelationPlanImpl<F32>>>(
              std::unexpect, Status::InvalidArgument);
        if (fft_len < 2) fft_len = 2;
        auto rfft_plan_expected = this->create_rfft_plan_impl_f32(fft_len);
        if (!rfft_plan_expected)
          return OmniExpected<
              std::unique_ptr<Abstract::CorrelationPlanImpl<F32>>>(
              std::unexpect, rfft_plan_expected.error());
        fft_plan_variant.emplace<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>(
            std::move(rfft_plan_expected.value()));
      }
      return std::make_unique<CorrelationPlanImpl<F32>>(
          std::move(fft_plan_variant), kernel, type, method);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::CorrPlanImpl<F32>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::CorrelationPlanImpl<F32>>>(
          std::unexpect, Status::Failure);
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
      CorrelationPlanImpl<F64>::FFTPlanImplVariant fft_plan_variant;
      if (method == ConvolutionMethod::FFT
          || (method == ConvolutionMethod::Auto)) {
        size_t kernel_length = kernel.size();
        if (kernel_length == 0)
          return OmniExpected<
              std::unique_ptr<Abstract::CorrelationPlanImpl<F64>>>(
              std::unexpect, Status::InvalidArgument);
        size_t temp_input_len_for_fft_calc = kernel_length;
        size_t fft_len = convolution_detail::next_power_of_two(
            kernel_length + temp_input_len_for_fft_calc - 1);
        if (fft_len == 0)
          return OmniExpected<
              std::unique_ptr<Abstract::CorrelationPlanImpl<F64>>>(
              std::unexpect, Status::InvalidArgument);
        if (fft_len < 2) fft_len = 2;
        auto rfft_plan_expected = this->create_rfft_plan_impl_f64(fft_len);
        if (!rfft_plan_expected)
          return OmniExpected<
              std::unique_ptr<Abstract::CorrelationPlanImpl<F64>>>(
              std::unexpect, rfft_plan_expected.error());
        fft_plan_variant.emplace<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>(
            std::move(rfft_plan_expected.value()));
      }
      return std::make_unique<CorrelationPlanImpl<F64>>(
          std::move(fft_plan_variant), kernel, type, method);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::CorrPlanImpl<F64>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::CorrelationPlanImpl<F64>>>(
          std::unexpect, Status::Failure);
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
      CorrelationPlanImpl<C32>::FFTPlanImplVariant fft_plan_variant;
      if (method == ConvolutionMethod::FFT
          || (method == ConvolutionMethod::Auto)) {
        size_t kernel_length = kernel.size();
        if (kernel_length == 0)
          return OmniExpected<
              std::unique_ptr<Abstract::CorrelationPlanImpl<C32>>>(
              std::unexpect, Status::InvalidArgument);
        size_t temp_input_len_for_fft_calc = kernel_length;
        size_t fft_len = convolution_detail::next_power_of_two(
            kernel_length + temp_input_len_for_fft_calc - 1);
        if (fft_len == 0)
          return OmniExpected<
              std::unique_ptr<Abstract::CorrelationPlanImpl<C32>>>(
              std::unexpect, Status::InvalidArgument);
        auto cfft_plan_expected = this->create_fft_plan_impl_c32(fft_len);
        if (!cfft_plan_expected)
          return OmniExpected<
              std::unique_ptr<Abstract::CorrelationPlanImpl<C32>>>(
              std::unexpect, cfft_plan_expected.error());
        fft_plan_variant.emplace<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>(
            std::move(cfft_plan_expected.value()));
      }
      return std::make_unique<CorrelationPlanImpl<C32>>(
          std::move(fft_plan_variant), kernel, type, method);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::CorrPlanImpl<C32>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::CorrelationPlanImpl<C32>>>(
          std::unexpect, Status::Failure);
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
      CorrelationPlanImpl<C64>::FFTPlanImplVariant fft_plan_variant;
      if (method == ConvolutionMethod::FFT
          || (method == ConvolutionMethod::Auto)) {
        size_t kernel_length = kernel.size();
        if (kernel_length == 0)
          return OmniExpected<
              std::unique_ptr<Abstract::CorrelationPlanImpl<C64>>>(
              std::unexpect, Status::InvalidArgument);
        size_t temp_input_len_for_fft_calc = kernel_length;
        size_t fft_len = convolution_detail::next_power_of_two(
            kernel_length + temp_input_len_for_fft_calc - 1);
        if (fft_len == 0)
          return OmniExpected<
              std::unique_ptr<Abstract::CorrelationPlanImpl<C64>>>(
              std::unexpect, Status::InvalidArgument);
        auto cfft_plan_expected = this->create_fft_plan_impl_c64(fft_len);
        if (!cfft_plan_expected)
          return OmniExpected<
              std::unique_ptr<Abstract::CorrelationPlanImpl<C64>>>(
              std::unexpect, cfft_plan_expected.error());
        fft_plan_variant.emplace<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>(
            std::move(cfft_plan_expected.value()));
      }
      return std::make_unique<CorrelationPlanImpl<C64>>(
          std::move(fft_plan_variant), kernel, type, method);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::CorrPlanImpl<C64>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::CorrelationPlanImpl<C64>>>(
          std::unexpect, Status::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FIRFilterPlanImpl<F32>>>
  Backend::create_fir_filter_plan_impl_f32(const F32Vec& coefficients) const
  {
    try {
      return std::make_unique<FIRFilterPlanImpl<F32>>(coefficients);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::FIRPlanImpl<F32>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::FIRFilterPlanImpl<F32>>>(
          std::unexpect, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FIRFilterPlanImpl<F64>>>
  Backend::create_fir_filter_plan_impl_f64(const F64Vec& coefficients) const
  {
    try {
      return std::make_unique<FIRFilterPlanImpl<F64>>(coefficients);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::FIRPlanImpl<F64>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::FIRFilterPlanImpl<F64>>>(
          std::unexpect, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FIRFilterPlanImpl<C32>>>
  Backend::create_fir_filter_plan_impl_c32(const C32Vec& coefficients) const
  {
    try {
      return std::make_unique<FIRFilterPlanImpl<C32>>(coefficients);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::FIRPlanImpl<C32>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::FIRFilterPlanImpl<C32>>>(
          std::unexpect, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FIRFilterPlanImpl<C64>>>
  Backend::create_fir_filter_plan_impl_c64(const C64Vec& coefficients) const
  {
    try {
      return std::make_unique<FIRFilterPlanImpl<C64>>(coefficients);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::FIRPlanImpl<C64>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::FIRFilterPlanImpl<C64>>>(
          std::unexpect, Status::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::IIRFilterPlanImpl<F32>>>
  Backend::create_iir_filter_plan_impl_f32(
      const std::vector<IIRFilterCoef>& sos_coefficients) const
  {
    try {
      return std::make_unique<IIRFilterPlanImpl<F32>>(sos_coefficients);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::IIRPlanImpl<F32>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::IIRFilterPlanImpl<F32>>>(
          std::unexpect, Status::Failure);
    }
  }
  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::IIRFilterPlanImpl<F64>>>
  Backend::create_iir_filter_plan_impl_f64(
      const std::vector<IIRFilterCoef>& sos_coefficients) const
  {
    try {
      return std::make_unique<IIRFilterPlanImpl<F64>>(sos_coefficients);
    }
    catch (const std::exception& e) {
      spdlog::get("OmniDSP")->error("Default::IIRPlanImpl<F64>: {}", e.what());
      return OmniExpected<std::unique_ptr<Abstract::IIRFilterPlanImpl<F64>>>(
          std::unexpect, Status::Failure);
    }
  }

  // --- Filter Design ---
  [[nodiscard]] OmniExpected<FIRCoefs<F32>> Backend::design_fir_filter_f32(
      const Design::FIRFilter& spec) const
  {
    return generate_fir_filter_coeffs<F32>(spec);
  }
  [[nodiscard]] OmniExpected<FIRCoefs<F64>> Backend::design_fir_filter_f64(
      const Design::FIRFilter& spec) const
  {
    return generate_fir_filter_coeffs<F64>(spec);
  }
  [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>>
  Backend::design_iir_filter_f32(const Design::IIRFilter& spec) const
  {
    return generate_iir_filter_coeffs(spec);
  }
  [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>>
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
