/**
 * @file filter.cpp (default)
 * @brief Implements Default backend FIRFilterPlanImpl, IIRFilterPlanImpl,
 * and filter design helper functions.
 */

#include "filter.hpp"  // Includes DefaultFIRFilterPlanImpl/DefaultIIRFilterPlanImpl declarations

#include <OmniDSP/filter.hpp>  // Public filter interfaces and specs (FIRFilterSpec, IIRFilterSpec, IIRFilterCoef, FIRCoefs)
#include <OmniDSP/window.hpp>  // For WindowSpec (needed by FIR design)
#include <algorithm>           // For std::copy, std::fill, std::min, std::max
#include <cmath>  // For std::sin, std::cos, std::abs, std::pow, std::log2, std::ceil
#include <complex>   // For std::complex
#include <expected>  // For std::unexpected
#include <iostream>  // For debug messages
#include <numbers>   // For std::numbers::pi_v
#include <numeric>   // For std::inner_product, std::accumulate
#include <span>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "OmniDSP/core_types.hpp"  // Core types
#include "window.hpp"  // For default backend window generation helpers

namespace OmniDSP::default
{

  // --- Filter Design Helper Functions (Internal Linkage within BackendType)
  // --- Moved OUT of anonymous namespace to allow explicit instantiation

  //--------------------------------------------------------------------------
  // FIR Filter Design Implementation (Windowed Sinc)
  //--------------------------------------------------------------------------
  template <typename T>  // T is F32 or F64
  [[nodiscard]] OmniExpected<FIRCoefs<T>> generate_fir_filter_coeffs(
      const FIRFilterSpec& spec)
  {
    // 1. Validate Spec
    if (!spec.validate()) {
      std::cerr << "Error: Invalid FIRFilterSpec provided to "
                   "generate_fir_filter_coeffs."
                << std::endl;
      return std::unexpected(Status::InvalidArgument);
    }
    size_t num_taps = spec.order + 1;
    if (num_taps == 0) {
      std::cerr << "Error: FIR filter order must result in at least 1 tap."
                << std::endl;
      return std::unexpected(Status::InvalidArgument);
    }

    // 2. Calculate normalized frequencies
    double fn1 = spec.cutoff1 / spec.sample_rate;
    std::optional<double> fn2 = std::nullopt;
    if (spec.cutoff2.has_value()) {
      fn2 = spec.cutoff2.value() / spec.sample_rate;
    }

    // 3. Create ideal impulse response
    FIRCoefs<T> ideal_coeffs(num_taps);  // Use alias
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
          if (!fn2.has_value()) return std::unexpected(Status::InvalidArgument);
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
          if (!fn2.has_value()) return std::unexpected(Status::InvalidArgument);
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
          return std::unexpected(Status::InvalidArgument);
      }
    }

    // 4. Generate Window
    std::vector<T> window_coeffs(num_taps);  // Use vector for temporary storage
    Status window_status;
    switch (spec.window.get_type()) {
      case WindowType::Bartlett:
        window_status = bartlett_window(std::span<T>(window_coeffs));
        break;
      case WindowType::Blackman:
        window_status = blackman_window(std::span<T>(window_coeffs));
        break;
      case WindowType::Flattop:
        window_status = flattop_window(std::span<T>(window_coeffs));
        break;
      case WindowType::Hamming:
        window_status = hamming_window(std::span<T>(window_coeffs));
        break;
      case WindowType::Hann:
        window_status = hann_window(std::span<T>(window_coeffs));
        break;
      case WindowType::Rectangular:
        window_status = rectangular_window(std::span<T>(window_coeffs));
        break;
      case WindowType::Triangular:
        window_status = triangular_window(std::span<T>(window_coeffs));
        break;
      case WindowType::Gaussian:
        if (!spec.window.get_param().has_value())
          return std::unexpected(Status::InvalidArgument);
        window_status = gaussian_window(
            static_cast<T>(spec.window.get_param().value()),
            std::span<T>(window_coeffs));
        break;
      case WindowType::Kaiser:
        if (!spec.window.get_param().has_value())
          return std::unexpected(Status::InvalidArgument);
        window_status = kaiser_window(
            static_cast<T>(spec.window.get_param().value()),
            std::span<T>(window_coeffs));
        break;
      default:
        window_status = Status::InvalidArgument;
        break;
    }
    if (window_status != Status::Success) {
      std::cerr
          << "Error: Failed to generate window coefficients during FIR design."
          << std::endl;
      return std::unexpected(window_status);
    }

    // 5. Apply window
    FIRCoefs<T> final_coeffs(num_taps);  // Use alias for return type
    for (size_t n = 0; n < num_taps; ++n) {
      final_coeffs[n] = ideal_coeffs[n] * window_coeffs[n];
    }

    // 6. Normalize
    if (spec.type == FilterType::Lowpass) {
      T sum = std::accumulate(final_coeffs.begin(), final_coeffs.end(), T{0.0});
      if (std::abs(sum) > static_cast<T>(1e-9)) {
        T inv_sum = static_cast<T>(1.0) / sum;
        for (auto& coeff : final_coeffs) {
          coeff *= inv_sum;
        }
      }
    }
    return final_coeffs;  // Return the vector using the alias
  }

  //--------------------------------------------------------------------------
  // IIR Filter Design Implementation (Placeholder)
  //--------------------------------------------------------------------------
  [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>>
  generate_iir_filter_coeffs(const IIRFilterSpec& spec)
  {
    if (!spec.validate()) {
      std::cerr << "Error: Invalid IIRFilterSpec provided to "
                   "generate_iir_filter_coeffs."
                << std::endl;
      return std::unexpected(Status::InvalidArgument);
    }
    std::cerr << "Default IIR filter design not yet implemented." << std::endl;
    return std::unexpected(Status::NotImplemented);
  }

  // --- DefaultFIRFilterPlanImpl ---
  // ... (Implementation remains the same) ...
  template <typename T>
  DefaultFIRFilterPlanImpl<T>::DefaultFIRFilterPlanImpl(
      const std::vector<T>& coefficients)
      : coefficients_(coefficients),
        state_(coefficients.empty() ? 0 : coefficients.size() - 1, T{0})
  {
    if (coefficients_.empty()) {
      throw std::invalid_argument("FIR filter coefficients cannot be empty.");
    }
  }
  template <typename T>
  DefaultFIRFilterPlanImpl<T>::~DefaultFIRFilterPlanImpl() = default;
  template <typename T>
  Status DefaultFIRFilterPlanImpl<T>::execute(
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
      size_t samples_to_zero = std::min(input.size(), output.size());
      std::fill(output.begin(), output.begin() + samples_to_zero, T{0});
      if (output.size() > samples_to_zero) {
        std::fill(output.begin() + samples_to_zero, output.end(), T{0});
      }
      return Status::Success;
    }
    size_t num_taps = coefficients_.size();
    size_t state_len = state_.size();
    std::vector<T> work_buffer;
    work_buffer.reserve(state_len + input.size());
    work_buffer.insert(work_buffer.end(), state_.begin(), state_.end());
    work_buffer.insert(work_buffer.end(), input.begin(), input.end());
    for (size_t i = 0; i < input.size(); ++i) {
      T result{};
      size_t work_buffer_start_idx = state_len + i;
      for (size_t k = 0; k < num_taps; ++k) {
        size_t work_idx = work_buffer_start_idx - k;
        result += coefficients_[k] * work_buffer[work_idx];
      }
      output[i] = result;
    }
    if (state_len > 0) {
      if (work_buffer.size() >= state_len) {
        std::copy(
            work_buffer.end() - state_len, work_buffer.end(), state_.begin());
      }
      else {
        std::cerr << "FIR execute warning: Work buffer unexpectedly small ("
                  << work_buffer.size() << ") when updating state (needed "
                  << state_len << ")." << std::endl;
        std::fill(state_.begin(), state_.end(), T{0});
      }
    }
    if (output.size() > input.size()) {
      std::fill(output.begin() + input.size(), output.end(), T{0});
    }
    return Status::Success;
  }
  template <typename T>
  Status DefaultFIRFilterPlanImpl<T>::reset()
  {
    std::fill(state_.begin(), state_.end(), T{0});
    return Status::Success;
  }
  template <typename T>
  size_t DefaultFIRFilterPlanImpl<T>::get_order() const
  {
    return coefficients_.empty() ? 0 : coefficients_.size() - 1;
  }
  template <typename T>
  size_t DefaultFIRFilterPlanImpl<T>::get_num_taps() const
  {
    return coefficients_.size();
  }
  template <typename T>
  std::span<const T> DefaultFIRFilterPlanImpl<T>::get_coefficients() const
  {
    return std::span<const T>(coefficients_);
  }

  // --- DefaultIIRFilterPlanImpl ---
  // ... (Implementation remains the same) ...
  template <typename T>
  DefaultIIRFilterPlanImpl<T>::DefaultIIRFilterPlanImpl(
      const std::vector<IIRFilterCoef>& sos_coefficients)
  {
    if (sos_coefficients.empty()) {
      throw std::invalid_argument("IIR SOS coefficients cannot be empty.");
    }
    internal_coeffs_.reserve(sos_coefficients.size());
    for (const auto& sos_double : sos_coefficients) {
      InternalSOS<T> sos_t;
      sos_t.b0 = static_cast<T>(sos_double.b0);
      sos_t.b1 = static_cast<T>(sos_double.b1);
      sos_t.b2 = static_cast<T>(sos_double.b2);
      if (std::abs(sos_double.a0 - 1.0) > 1e-9) {
        throw std::runtime_error(
            "IIR SOS coefficient a0 is not 1.0 (normalization required).");
      }
      sos_t.a1 = static_cast<T>(sos_double.a1);
      sos_t.a2 = static_cast<T>(sos_double.a2);
      internal_coeffs_.push_back(sos_t);
    }
    state_.assign(internal_coeffs_.size() * 2, T{0});
  }
  template <typename T>
  DefaultIIRFilterPlanImpl<T>::~DefaultIIRFilterPlanImpl() = default;
  template <typename T>
  Status DefaultIIRFilterPlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (internal_coeffs_.empty()) {
      std::fill(output.begin(), output.end(), T{0});
      return Status::Success;
    }
    if (output.size() < input.size()) {
      return Status::SizeMismatch;
    }
    if (input.empty()) {
      size_t samples_to_zero = std::min(input.size(), output.size());
      std::fill(output.begin(), output.begin() + samples_to_zero, T{0});
      if (output.size() > samples_to_zero) {
        std::fill(output.begin() + samples_to_zero, output.end(), T{0});
      }
      return Status::Success;
    }
    size_t num_sections = internal_coeffs_.size();
    for (size_t n = 0; n < input.size(); ++n) {
      T sample_in = input[n];
      T section_in_out = sample_in;
      for (size_t k = 0; k < num_sections; ++k) {
        const auto& sos = internal_coeffs_[k];
        T& s1 = state_[k * 2 + 0];
        T& s2 = state_[k * 2 + 1];
        T y_n = section_in_out * sos.b0 + s1;
        T s1_next = section_in_out * sos.b1 - y_n * sos.a1 + s2;
        T s2_next = section_in_out * sos.b2 - y_n * sos.a2;
        s1 = s1_next;
        s2 = s2_next;
        section_in_out = y_n;
      }
      output[n] = section_in_out;
    }
    if (output.size() > input.size()) {
      std::fill(output.begin() + input.size(), output.end(), T{0});
    }
    return Status::Success;
  }
  template <typename T>
  Status DefaultIIRFilterPlanImpl<T>::reset()
  {
    std::fill(state_.begin(), state_.end(), T{0});
    return Status::Success;
  }
  template <typename T>
  size_t DefaultIIRFilterPlanImpl<T>::get_order() const
  {
    return internal_coeffs_.empty() ? 0 : internal_coeffs_.size() * 2;
  }
  template <typename T>
  size_t DefaultIIRFilterPlanImpl<T>::get_num_sections() const
  {
    return internal_coeffs_.size();
  }

  // --- Explicit Template Instantiations ---
  template class DefaultFIRFilterPlanImpl<F32>;
  template class DefaultFIRFilterPlanImpl<F64>;
  template class DefaultFIRFilterPlanImpl<C32>;
  template class DefaultFIRFilterPlanImpl<C64>;
  template std::span<const F32>
  DefaultFIRFilterPlanImpl<F32>::get_coefficients() const;
  template std::span<const F64>
  DefaultFIRFilterPlanImpl<F64>::get_coefficients() const;
  template std::span<const C32>
  DefaultFIRFilterPlanImpl<C32>::get_coefficients() const;
  template std::span<const C64>
  DefaultFIRFilterPlanImpl<C64>::get_coefficients() const;

  template class DefaultIIRFilterPlanImpl<F32>;
  template class DefaultIIRFilterPlanImpl<F64>;

  // Explicit instantiations for the design helper functions defined in this
  // file
  template OmniExpected<FIRCoefs<F32>> generate_fir_filter_coeffs<F32>(
      const FIRFilterSpec& spec);
  template OmniExpected<FIRCoefs<F64>> generate_fir_filter_coeffs<F64>(
      const FIRFilterSpec& spec);
  // No instantiation needed for non-template generate_iir_filter_coeffs

}  // namespace OmniDSP::default
