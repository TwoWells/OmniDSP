/**
 * @file fir_filter.cpp (Default)
 * @brief Implements Default backend FIRFilterPlanImpl and FIR design helper.
 */

#include "fir_filter.hpp"  // Corresponding header

#include <OmniDSP/core_types.hpp>    // Core types, Status, OmniExpected
#include <OmniDSP/filter.hpp>        // Public Design::FIRFilter
#include <OmniDSP/types/filter.hpp>  // For FilterType enum
#include <OmniDSP/window.hpp>        // For OmniDSP::generate_window
#include <algorithm>  // For std::copy, std::fill, std::min, std::max
#include <cmath>      // For std::sin, std::cos, std::abs
#include <complex>
#include <expected>
#include <iostream>
#include <numbers>  // For std::numbers::pi_v
#include <numeric>  // For std::accumulate
#include <span>
#include <stdexcept>
#include <vector>

#include "spdlog/spdlog.h"

namespace OmniDSP::Default {

  //--------------------------------------------------------------------------
  // FIR Filter Design Implementation (Windowed Sinc)
  //--------------------------------------------------------------------------
  template <typename T>
  [[nodiscard]] OmniExpected<FIRCoefs<T>> generate_fir_filter_coeffs(
      const Design::FIRFilter& spec)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (!spec.validate_consistency()) {
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
      logger->error(
          "FIR filter order must result in at least 1 tap (num_taps was 0).");
      return std::unexpected(Status::InvalidArgument);
    }

    double fn1 = spec.cutoff1 / spec.sample_rate;
    std::optional<double> fn2 = std::nullopt;
    if (spec.cutoff2.has_value()) {
      fn2 = spec.cutoff2.value() / spec.sample_rate;
    }

    FIRCoefs<T> ideal_coeffs(num_taps);
    const double center = static_cast<double>(spec.order) / 2.0;
    constexpr double pi = std::numbers::pi_v<double>;

    for (size_t n = 0; n < num_taps; ++n) {
      double m_double = static_cast<double>(n) - center;
      T sinc_val1;
      if (std::abs(m_double) < 1e-9) {
        sinc_val1 = static_cast<T>(2.0 * fn1);
      }
      else {
        sinc_val1 = static_cast<T>(
            std::sin(2.0 * pi * fn1 * m_double) / (pi * m_double));
      }

      switch (spec.type) {
        case FilterType::Lowpass:
          ideal_coeffs[n] = sinc_val1;
          break;
        case FilterType::Highpass: {
          T delta_sinc
              = (std::abs(m_double) < 1e-9)
                    ? static_cast<T>(1.0)
                    : static_cast<T>(std::sin(pi * m_double) / (pi * m_double));
          ideal_coeffs[n] = delta_sinc - sinc_val1;
          break;
        }
        case FilterType::Bandpass: {
          if (!fn2.has_value()) {
            logger->error(
                "Default FIR design (Bandpass): Missing fn2 (cutoff2) in "
                "spec.");
            return std::unexpected(Status::InvalidArgument);
          }
          double fn2_val = fn2.value();
          T sinc_val2;
          if (std::abs(m_double) < 1e-9) {
            sinc_val2 = static_cast<T>(2.0 * fn2_val);
          }
          else {
            sinc_val2 = static_cast<T>(
                std::sin(2.0 * pi * fn2_val * m_double) / (pi * m_double));
          }
          ideal_coeffs[n] = sinc_val2 - sinc_val1;
          break;
        }
        case FilterType::Bandstop: {
          if (!fn2.has_value()) {
            logger->error(
                "Default FIR design (Bandstop): Missing fn2 (cutoff2) in "
                "spec.");
            return std::unexpected(Status::InvalidArgument);
          }
          double fn2_val = fn2.value();
          T sinc_val2;
          if (std::abs(m_double) < 1e-9) {
            sinc_val2 = static_cast<T>(2.0 * fn2_val);
          }
          else {
            sinc_val2 = static_cast<T>(
                std::sin(2.0 * pi * fn2_val * m_double) / (pi * m_double));
          }
          T delta_sinc
              = (std::abs(m_double) < 1e-9)
                    ? static_cast<T>(1.0)
                    : static_cast<T>(std::sin(pi * m_double) / (pi * m_double));
          ideal_coeffs[n] = delta_sinc - (sinc_val2 - sinc_val1);
          break;
        }
        default:
          logger->error(
              "Default FIR design: Unknown filter type specified in spec.");
          return std::unexpected(Status::InvalidArgument);
      }
    }

    std::vector<T> window_coeffs_vec(num_taps);
    std::span<T> window_coeffs_span = window_coeffs_vec;
    OmniExpected<void> window_gen_status
        = OmniDSP::generate_window<T>(spec.window_setup, window_coeffs_span);

    if (!window_gen_status) {
      logger->error(
          "Failed to generate window coefficients during Default FIR design. "
          "Status: {}",
          static_cast<int>(window_gen_status.error()));
      return std::unexpected(window_gen_status.error());
    }

    FIRCoefs<T> final_coeffs(num_taps);
    for (size_t n = 0; n < num_taps; ++n) {
      final_coeffs[n] = ideal_coeffs[n] * window_coeffs_vec[n];
    }

    if (spec.type == FilterType::Lowpass || spec.type == FilterType::Bandpass) {
      T sum = std::accumulate(final_coeffs.begin(), final_coeffs.end(), T{0.0});
      if (std::abs(sum) > static_cast<T>(1e-9)) {
        T inv_sum = static_cast<T>(1.0) / sum;
        for (auto& coeff : final_coeffs) {
          coeff *= inv_sum;
        }
      }
    }
    return final_coeffs;
  }

  // --- FIRFilterPlanImpl ---
  template <typename T>
  FIRFilterPlanImpl<T>::FIRFilterPlanImpl(const std::vector<T>& coefficients)
      : coefficients_(coefficients),
        state_(coefficients.empty() ? 0 : coefficients.size() - 1, T{0})
  {
    if (coefficients_.empty()) {
      auto logger = spdlog::get("OmniDSP");
      if (!logger) logger = spdlog::default_logger();
      logger->error(
          "Default::FIRFilterPlanImpl: Coefficients cannot be empty.");
      throw std::invalid_argument(
          "FIR filter coefficients cannot be empty for FIRFilterPlanImpl.");
    }
  }

  template <typename T>
  FIRFilterPlanImpl<T>::~FIRFilterPlanImpl() = default;

  template <typename T>
  Status FIRFilterPlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (coefficients_.empty()) {
      std::fill(output.begin(), output.end(), T{0});
      return Status::Success;
    }
    if (output.size() < input.size()) {
      return Status::SizeMismatch;
    }
    if (input.empty()) {
      if (!output.empty()) {
        std::fill(
            output.begin(),
            output.begin() + std::min(input.size(), output.size()),
            T{0});
      }
      return Status::Success;
    }

    size_t num_taps = coefficients_.size();
    size_t state_len = state_.size();

    for (size_t i = 0; i < input.size(); ++i) {
      T current_input_sample = input[i];
      T acc = T{0};
      acc += coefficients_[0] * current_input_sample;
      for (size_t k = 1; k < num_taps; ++k) {
        if ((k - 1) < state_len) {
          acc += coefficients_[k] * state_[k - 1];
        }
      }
      output[i] = acc;
      if (state_len > 0) {
        for (size_t k = state_len - 1; k > 0; --k) {
          state_[k] = state_[k - 1];
        }
        state_[0] = current_input_sample;
      }
    }
    if (output.size() > input.size()) {
      std::fill(output.begin() + input.size(), output.end(), T{0});
    }
    return Status::Success;
  }

  template <typename T>
  Status FIRFilterPlanImpl<T>::reset()
  {
    std::fill(state_.begin(), state_.end(), T{0});
    return Status::Success;
  }

  template <typename T>
  size_t FIRFilterPlanImpl<T>::get_order() const
  {
    return coefficients_.empty() ? 0 : coefficients_.size() - 1;
  }

  template <typename T>
  size_t FIRFilterPlanImpl<T>::get_num_taps() const
  {
    return coefficients_.size();
  }

  template <typename T>
  std::span<const T> FIRFilterPlanImpl<T>::get_coefficients() const
  {
    return std::span<const T>(coefficients_);
  }

  // Explicit template instantiations for FIRFilterPlanImpl
  template class FIRFilterPlanImpl<F32>;
  template class FIRFilterPlanImpl<F64>;
  template class FIRFilterPlanImpl<C32>;
  template class FIRFilterPlanImpl<C64>;

  // Explicit template instantiations for get_coefficients
  template std::span<const F32> FIRFilterPlanImpl<F32>::get_coefficients()
      const;
  template std::span<const F64> FIRFilterPlanImpl<F64>::get_coefficients()
      const;
  template std::span<const C32> FIRFilterPlanImpl<C32>::get_coefficients()
      const;
  template std::span<const C64> FIRFilterPlanImpl<C64>::get_coefficients()
      const;

  // Explicit template instantiations for generate_fir_filter_coeffs
  template OmniExpected<FIRCoefs<F32>> generate_fir_filter_coeffs<F32>(
      const Design::FIRFilter& spec);
  template OmniExpected<FIRCoefs<F64>> generate_fir_filter_coeffs<F64>(
      const Design::FIRFilter& spec);

}  // namespace OmniDSP::Default
