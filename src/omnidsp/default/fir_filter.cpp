/**
 * @file fir_filter.cpp (Default)
 * @brief Implements Default backend FIRFilterProcessorImpl and FIR design
 * helper.
 */

#include "fir_filter.hpp"  // Corresponding header

#include <OmniDSP/core_types.hpp>  // Core types, Status, OmniExpected, C32Vec, C64Vec
#include <OmniDSP/fir_filter.hpp>  // Public Design::FIRFilter, FIRCoefs
#include <OmniDSP/iir_filter.hpp>  // Public Design::FIRFilter (Likely a typo, should be fir_filter.hpp or not needed here)
#include <OmniDSP/types/filter.hpp>  // For FilterType enum
#include <OmniDSP/window.hpp>        // For OmniDSP::generate_window
#include <algorithm>  // For std::copy, std::fill, std::min, std::max
#include <cmath>  // For std::sin, std::cos, std::abs, std::log10, std::pow, std::sqrt
#include <complex>  // For std::complex, std::real, std::imag
#include <expected>
#include <iostream>
#include <limits>   // For std::numeric_limits
#include <numbers>  // For std::numbers::pi_v
#include <numeric>  // For std::accumulate, std::iota
#include <span>
#include <stdexcept>
#include <vector>

#include "spdlog/spdlog.h"

namespace OmniDSP::Default {

  //--------------------------------------------------------------------------
  // FIR Filter Design Implementation (Windowed Sinc)
  //--------------------------------------------------------------------------
  // Definition of the template function
  template <typename T>
  [[nodiscard]] OmniExpected<FIRCoefs<T>>
  generate_fir_filter_coeffs(  // Assuming FIRCoefs<T> is std::vector<T>
      const Design::FIRFilter& spec)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      // Fallback or handle missing logger
      // For simplicity, we'll proceed, but in a real app, this might be an
      // issue. std::cerr << "Warning: OmniDSP logger not found in
      // generate_fir_filter_coeffs." << std::endl;
    }
    else {
      // Optional: Log entry into the function
      // logger->trace("generate_fir_filter_coeffs called for type {}",
      // typeid(T).name());
    }

    if (!spec.validate_consistency()) {
      if (logger)
        logger->error(
            "Invalid Design::FIRFilter provided to "
            "Default::generate_fir_filter_coeffs. Order={}, SR={}, C1={}, "
            "WinLen={}",
            spec.order,
            spec.sample_rate,
            spec.cutoff1,
            spec.window_setup.length);
      return std::unexpected(Status::InvalidArgument);
    }

    size_t num_taps = spec.num_taps();
    if (num_taps == 0) {
      if (logger)
        logger->error(
            "FIR filter order must result in at least 1 tap (num_taps was 0).");
      return std::unexpected(Status::InvalidArgument);
    }

    // Normalized frequencies
    double fn1 = spec.cutoff1 / spec.sample_rate;
    std::optional<double> fn2_opt = std::nullopt;
    if (spec.cutoff2.has_value()) {
      fn2_opt = spec.cutoff2.value() / spec.sample_rate;
      if (fn2_opt.value() <= fn1
          && (spec.type == FilterType::Bandpass
              || spec.type == FilterType::Bandstop)) {
        if (logger)
          logger->error(
              "FIR design: cutoff2 ({}) must be greater than cutoff1 ({}) for "
              "bandpass/bandstop filters.",
              spec.cutoff2.value(),
              spec.cutoff1);
        return std::unexpected(Status::InvalidArgument);
      }
    }

    FIRCoefs<T> ideal_coeffs_vec(num_taps);  // This is std::vector<T>
    std::span<T> ideal_coeffs = ideal_coeffs_vec;

    const double center = static_cast<double>(spec.order) / 2.0;
    constexpr double pi = std::numbers::pi_v<double>;

    for (size_t n = 0; n < num_taps; ++n) {
      double m_double = static_cast<double>(n) - center;
      T sinc_val1;

      // Sinc for cutoff1
      if (std::abs(m_double) < 1e-9) {  // m_double is close to zero
        sinc_val1 = static_cast<T>(2.0 * fn1);
      }
      else {
        sinc_val1 = static_cast<T>(
            std::sin(2.0 * pi * fn1 * m_double) / (pi * m_double));
      }

      if constexpr (
          std::is_same_v<T, std::complex<float>>
          || std::is_same_v<T, std::complex<double>>) {
        // For complex types, the ideal sinc response is real.
        // If a complex filter design (e.g. Hilbert transformer) was intended,
        // this basic windowed-sinc for real filters would need significant
        // changes. Assuming for now that even for complex T, we are designing a
        // filter with real ideal response.
      }

      switch (spec.type) {
        case FilterType::Lowpass:
          ideal_coeffs[n] = sinc_val1;
          break;
        case FilterType::Highpass: {
          T delta_sinc;  // Sinc of 0.5 (Nyquist) for the dirac delta component
          if (std::abs(m_double) < 1e-9) {
            delta_sinc = static_cast<T>(1.0);  // 2.0 * 0.5
          }
          else {
            delta_sinc
                = static_cast<T>(std::sin(pi * m_double) / (pi * m_double));
          }
          ideal_coeffs[n] = delta_sinc - sinc_val1;
          break;
        }
        case FilterType::Bandpass: {
          if (!fn2_opt.has_value()) {
            if (logger)
              logger->error(
                  "Default FIR design (Bandpass): Missing fn2 (cutoff2) in "
                  "spec.");
            return std::unexpected(Status::InvalidArgument);
          }
          double fn2 = fn2_opt.value();
          T sinc_val2;
          if (std::abs(m_double) < 1e-9) {
            sinc_val2 = static_cast<T>(2.0 * fn2);
          }
          else {
            sinc_val2 = static_cast<T>(
                std::sin(2.0 * pi * fn2 * m_double) / (pi * m_double));
          }
          ideal_coeffs[n]
              = sinc_val2 - sinc_val1;  // h_bp(n) = h_lp_c2(n) - h_lp_c1(n)
          break;
        }
        case FilterType::Bandstop: {
          if (!fn2_opt.has_value()) {
            if (logger)
              logger->error(
                  "Default FIR design (Bandstop): Missing fn2 (cutoff2) in "
                  "spec.");
            return std::unexpected(Status::InvalidArgument);
          }
          double fn2 = fn2_opt.value();
          T sinc_val2;
          if (std::abs(m_double) < 1e-9) {
            sinc_val2 = static_cast<T>(2.0 * fn2);
          }
          else {
            sinc_val2 = static_cast<T>(
                std::sin(2.0 * pi * fn2 * m_double) / (pi * m_double));
          }
          T delta_sinc;
          if (std::abs(m_double) < 1e-9) {
            delta_sinc = static_cast<T>(1.0);
          }
          else {
            delta_sinc
                = static_cast<T>(std::sin(pi * m_double) / (pi * m_double));
          }
          // h_bs(n) = delta(n) - (h_lp_c2(n) - h_lp_c1(n))
          ideal_coeffs[n] = delta_sinc - (sinc_val2 - sinc_val1);
          break;
        }
        default:
          if (logger)
            logger->error(
                "Default FIR design: Unknown filter type specified in spec.");
          return std::unexpected(Status::InvalidArgument);
      }
    }

    // Apply window
    std::vector<T> window_coeffs_vec_storage(
        num_taps);  // Ensure this is vector<T>
    std::span<T> window_coeffs_span = window_coeffs_vec_storage;

    // The generate_window function needs to be able to handle complex T if T is
    // complex. Typically, windows are real. If T is complex, and
    // generate_window expects real output, this needs careful handling.
    // Assuming generate_window<T> works correctly for complex T by applying a
    // real window to both real and imaginary parts, or that it's specialized.
    // For now, let's assume generate_window<T> is appropriate.
    OmniExpected<void> window_gen_status
        = OmniDSP::generate_window<T>(spec.window_setup, window_coeffs_span);

    if (!window_gen_status) {
      if (logger)
        logger->error(
            "Failed to generate window coefficients during Default FIR design. "
            "Status: {}",
            static_cast<int>(window_gen_status.error()));
      return std::unexpected(window_gen_status.error());
    }

    FIRCoefs<T> final_coeffs_vec(num_taps);  // This is std::vector<T>
    std::span<T> final_coeffs = final_coeffs_vec;

    for (size_t n = 0; n < num_taps; ++n) {
      final_coeffs[n] = ideal_coeffs[n]
                        * window_coeffs_span[n];  // window_coeffs_vec_storage
    }

    // Normalize gain for lowpass and bandpass filters
    // For complex filters, normalization might need to consider magnitude or
    // specific component. This normalization is typical for real filters.
    if (spec.type == FilterType::Lowpass || spec.type == FilterType::Bandpass) {
      if constexpr (
          std::is_arithmetic_v<T> || std::is_same_v<T, std::complex<float>>
          || std::is_same_v<T, std::complex<double>>) {
        T sum_val = T{0};  // Use T{0} for complex types as well
        for (size_t n = 0; n < num_taps; ++n) {
          sum_val += final_coeffs[n];
        }

        // Use magnitude for complex sum to check if it's effectively zero
        // For real T, std::abs is fine. For complex T, std::abs gives
        // magnitude.
        if (std::abs(sum_val)
            > static_cast<typename Utils::GetRealType<T>>(1e-9)) {
          T inv_sum
              = static_cast<T>(1.0) / sum_val;  // This works for complex T too
          for (size_t n = 0; n < num_taps; ++n) {
            final_coeffs[n] *= inv_sum;
          }
        }
      }
    }
    return final_coeffs_vec;  // Return the std::vector<T>
  }

  // --- FIRFilterProcessorImpl ---
  template <typename T>
  FIRFilterProcessorImpl<T>::FIRFilterProcessorImpl(
      const std::vector<T>& coefficients)
      : coefficients_(coefficients),
        state_(coefficients.empty() ? 0 : coefficients.size() - 1, T{0})
  {
    if (coefficients_.empty()) {
      auto logger = spdlog::get("OmniDSP");
      if (!logger) logger = spdlog::default_logger();  // Basic fallback
      if (logger)
        logger->error(
            "Default::FIRFilterProcessorImpl: Coefficients cannot be empty.");
      throw std::invalid_argument(
          "FIR filter coefficients cannot be empty for "
          "FIRFilterProcessorImpl.");
    }
  }

  template <typename T>
  FIRFilterProcessorImpl<T>::~FIRFilterProcessorImpl() = default;

  template <typename T>
  Status FIRFilterProcessorImpl<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (coefficients_.empty()) {  // Should have been caught by constructor
      std::fill(output.begin(), output.end(), T{0});
      return Status::Success;  // Or an error status
    }
    if (output.size() < input.size()) {
      return Status::SizeMismatch;
    }
    if (input.empty()) {
      // If input is empty, output should also be empty up to input.size().
      // The rest of output (if larger) is untouched or could be zeroed.
      // For consistency, let's ensure output matching input size is zeroed.
      std::fill(
          output.begin(),
          output.begin() + std::min(input.size(), output.size()),
          T{0});
      return Status::Success;
    }

    size_t num_taps = coefficients_.size();
    size_t state_len = state_.size();  // This is num_taps - 1

    for (size_t i = 0; i < input.size(); ++i) {
      T current_input_sample = input[i];
      T acc = T{0};  // Accumulator

      // Convolve: y[i] = b[0]*x[i] + b[1]*x[i-1] + ... + b[N-1]*x[i-N+1]
      // x[i-k] for k > 0 are in state_
      // state_[0] is x[i-1], state_[1] is x[i-2], ..., state_[state_len-1] is
      // x[i-state_len]

      acc += coefficients_[0] * current_input_sample;  // b[0]*x[i]

      for (size_t k = 1; k < num_taps; ++k) {  // k from 1 to num_taps-1
        // coefficients_[k] multiplies state_[k-1] (which is x[i-k])
        if ((k - 1) < state_len) {  // Ensure we don't read past state_ (should
                                    // always be true if state_len = num_taps-1)
          acc += coefficients_[k] * state_[k - 1];
        }
      }
      output[i] = acc;

      // Update state (shift delay line)
      if (state_len > 0) {
        for (size_t k = state_len - 1; k > 0; --k) {
          state_[k] = state_[k - 1];
        }
        state_[0]
            = current_input_sample;  // Newest sample enters the delay line
      }
    }

    // If output buffer is larger than input, zero out the rest
    if (output.size() > input.size()) {
      std::fill(output.begin() + input.size(), output.end(), T{0});
    }
    return Status::Success;
  }

  template <typename T>
  Status FIRFilterProcessorImpl<T>::reset()
  {
    std::fill(state_.begin(), state_.end(), T{0});
    return Status::Success;
  }

  template <typename T>
  size_t FIRFilterProcessorImpl<T>::get_order() const
  {
    return coefficients_.empty() ? 0 : coefficients_.size() - 1;
  }

  template <typename T>
  size_t FIRFilterProcessorImpl<T>::get_num_taps() const
  {
    return coefficients_.size();
  }

  template <typename T>
  std::span<const T> FIRFilterProcessorImpl<T>::get_coefficients() const
  {
    return std::span<const T>(coefficients_);
  }

  // Explicit template instantiations for FIRFilterProcessorImpl
  template class FIRFilterProcessorImpl<F32>;
  template class FIRFilterProcessorImpl<F64>;
  template class FIRFilterProcessorImpl<C32>;  // C32 is std::complex<F32>
  template class FIRFilterProcessorImpl<C64>;  // C64 is std::complex<F64>

  // Explicit template instantiations for get_coefficients (member function)
  // These are often not strictly necessary if the class itself is explicitly
  // instantiated, but can be good practice or required by some compilers/build
  // systems.
  template std::span<const F32> FIRFilterProcessorImpl<F32>::get_coefficients()
      const;
  template std::span<const F64> FIRFilterProcessorImpl<F64>::get_coefficients()
      const;
  template std::span<const C32> FIRFilterProcessorImpl<C32>::get_coefficients()
      const;
  template std::span<const C64> FIRFilterProcessorImpl<C64>::get_coefficients()
      const;

  // Explicit template instantiations for generate_fir_filter_coeffs
  // Assuming FIRCoefs<T> is std::vector<T>
  template OmniExpected<FIRCoefs<F32>> generate_fir_filter_coeffs<F32>(
      const Design::FIRFilter& spec);
  template OmniExpected<FIRCoefs<F64>> generate_fir_filter_coeffs<F64>(
      const Design::FIRFilter& spec);

  // NEW: Explicit instantiations for complex types
  // The linker error showed it was looking for a function returning
  // std::expected<std::vector<std::complex<...>>> This means
  // FIRCoefs<std::complex<float>> should be std::vector<std::complex<float>>.
  template OmniExpected<FIRCoefs<C32>> generate_fir_filter_coeffs<C32>(
      const Design::FIRFilter& spec);
  template OmniExpected<FIRCoefs<C64>> generate_fir_filter_coeffs<C64>(
      const Design::FIRFilter& spec);

}  // namespace OmniDSP::Default
