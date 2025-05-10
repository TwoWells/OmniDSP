/**
 * @file filter.cpp (Default)
 * @brief Implements Default backend FIRFilterPlanImpl, IIRFilterPlanImpl,
 * and filter design helper functions.
 */

#include "filter.hpp"  // Includes DefaultFIRFilterPlanImpl/DefaultIIRFilterPlanImpl declarations

#include <OmniDSP/core_types.hpp>  // Core types, Status, OmniExpected
#include <OmniDSP/filter.hpp>  // Public filter interfaces and specs (FIRFilterSpec, IIRFilterSpec, IIRFilterCoef, FIRCoefs)
#include <OmniDSP/window.hpp>  // For WindowSetup and the free OmniDSP::generate_window function
#include <algorithm>  // For std::copy, std::fill, std::min, std::max
#include <cmath>  // For std::sin, std::cos, std::abs, std::pow, std::log2, std::ceil
#include <complex>   // For std::complex
#include <expected>  // For std::unexpected
#include <iostream>  // For debug messages (though spdlog is preferred)
#include <numbers>   // For std::numbers::pi_v
#include <numeric>   // For std::inner_product, std::accumulate
#include <span>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "spdlog/spdlog.h"  // For logging

// The local "window.hpp" (from default directory) is no longer needed for FIR
// filter design's window generation part, as we now use the public
// OmniDSP::generate_window. It might still be needed if other functions in this
// file call Default::specific_window_functions.

namespace OmniDSP::Default {

  // --- Filter Design Helper Functions (Internal Linkage within BackendType)
  // ---

  //--------------------------------------------------------------------------
  // FIR Filter Design Implementation (Windowed Sinc)
  //--------------------------------------------------------------------------
  template <typename T>
  [[nodiscard]] OmniExpected<FIRCoefs<T>> generate_fir_filter_coeffs(
      const FIRFilterSpec&
          spec)  // FIRFilterSpec should now contain WindowSetup
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {  // Fallback if named logger isn't registered
      logger = spdlog::default_logger();
    }

    // 1. Validate Spec
    // Assuming spec.validate() is updated to work with WindowSetup if it checks
    // window validity. The WindowSetup constructor itself performs validation
    // on its parameters.
    if (!spec.validate()) {
      logger->error(
          "Invalid FIRFilterSpec provided to generate_fir_filter_coeffs.");
      return std::unexpected(Status::InvalidArgument);
    }
    size_t num_taps = spec.order + 1;
    if (num_taps == 0) {
      logger->error("FIR filter order must result in at least 1 tap.");
      return std::unexpected(Status::InvalidArgument);
    }

    // 2. Calculate normalized frequencies
    double fn1 = spec.cutoff1 / spec.sample_rate;
    std::optional<double> fn2 = std::nullopt;
    if (spec.cutoff2.has_value()) {
      fn2 = spec.cutoff2.value() / spec.sample_rate;
    }

    // 3. Create ideal impulse response
    FIRCoefs<T> ideal_coeffs(num_taps);
    const double center = static_cast<double>(spec.order) / 2.0;
    constexpr double pi = std::numbers::pi_v<double>;

    for (size_t n = 0; n < num_taps; ++n) {
      double m_double = static_cast<double>(n) - center;
      T sinc_val1;
      if (std::abs(m_double) < 1e-9) {  // Avoid division by zero at center
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
            logger->error("FIR design (Bandpass): Missing fn2 (cutoff2).");
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
            logger->error("FIR design (Bandstop): Missing fn2 (cutoff2).");
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
          logger->error("FIR design: Unknown filter type specified.");
          return std::unexpected(Status::InvalidArgument);
      }
    }

    // 4. Generate Window using the free OmniDSP::generate_window function
    //    This assumes 'spec.window_setup' is the member in FIRFilterSpec of
    //    type WindowSetup. If FIRFilterSpec still uses 'window' as member name
    //    but its type is WindowSetup, use 'spec.window'.
    std::vector<T> window_coeffs_vec(num_taps);
    std::span<T> window_coeffs_span = window_coeffs_vec;

    // Ensure the WindowSetup object within FIRFilterSpec has the correct
    // length. The FIRFilterSpec should ideally be constructed such that its
    // embedded WindowSetup already has its 'length' field set to num_taps. If
    // spec.window_setup.length is not num_taps, OmniDSP::generate_window will
    // return SizeMismatch. For this example, we assume spec.window_setup is
    // correctly configured. If FIRFilterSpec's WindowSetup member might have a
    // different length, you might need: WindowSetup local_window_setup =
    // spec.window_setup; // Copy local_window_setup.length =
    // static_cast<int>(num_taps); auto window_gen_result =
    // OmniDSP::generate_window<T>(local_window_setup, window_coeffs_span);
    // However, it's cleaner if FIRFilterSpec ensures its WindowSetup is
    // consistent.

    // Assuming spec.window_setup is the correct WindowSetup object from
    // FIRFilterSpec and its .length member is consistent with num_taps. The
    // OmniDSP::generate_window function from <OmniDSP/window.hpp> is used.
    auto window_gen_result
        = OmniDSP::generate_window<T>(spec.window_setup, window_coeffs_span);

    if (!window_gen_result) {
      // OmniDSP::generate_window already logs detailed errors.
      logger->error(
          "Failed to generate window coefficients during FIR design. Status: "
          "{}",
          static_cast<int>(window_gen_result.error()));
      return std::unexpected(window_gen_result.error());
    }
    // window_coeffs_vec now contains the generated window coefficients.

    // 5. Apply window
    FIRCoefs<T> final_coeffs(num_taps);
    for (size_t n = 0; n < num_taps; ++n) {
      final_coeffs[n]
          = ideal_coeffs[n] * window_coeffs_vec[n];  // Use window_coeffs_vec
    }

    // 6. Normalize (typically for lowpass and bandpass to ensure unity gain at
    // DC or center freq)
    //    This normalization step might need adjustment based on filter type and
    //    design goals.
    if (spec.type == FilterType::Lowpass || spec.type == FilterType::Bandpass) {
      T sum = std::accumulate(final_coeffs.begin(), final_coeffs.end(), T{0.0});
      if (std::abs(sum) > static_cast<T>(1e-9)) {  // Avoid division by zero
        T inv_sum = static_cast<T>(1.0) / sum;
        for (auto& coeff : final_coeffs) {
          coeff *= inv_sum;
        }
      }
    }
    // For Highpass, ensure sum of coeffs is close to 0 for no DC gain.
    // For Bandstop, ensure gain is 1 away from stopband and 0 in stopband.
    // The current normalization is a common simple approach.

    return final_coeffs;
  }

  //--------------------------------------------------------------------------
  // IIR Filter Design Implementation (Placeholder)
  //--------------------------------------------------------------------------
  [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>>
  generate_iir_filter_coeffs(const IIRFilterSpec& spec)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (!spec.validate()) {
      logger->error(
          "Invalid IIRFilterSpec provided to generate_iir_filter_coeffs.");
      return std::unexpected(Status::InvalidArgument);
    }
    logger->warn("Default IIR filter design not yet implemented.");
    return std::unexpected(Status::NotImplemented);
  }

  // --- FIRFilterPlanImpl ---
  // Constructor
  template <typename T>
  FIRFilterPlanImpl<T>::FIRFilterPlanImpl(const std::vector<T>& coefficients)
      : coefficients_(coefficients),
        state_(coefficients.empty() ? 0 : coefficients.size() - 1, T{0})
  {
    if (coefficients_.empty()) {
      // Consider logging this error as well.
      throw std::invalid_argument(
          "FIR filter coefficients cannot be empty for FIRFilterPlanImpl.");
    }
  }

  // Destructor
  template <typename T>
  FIRFilterPlanImpl<T>::~FIRFilterPlanImpl() = default;

  // Execute
  template <typename T>
  Status FIRFilterPlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (coefficients_.empty()) {  // Should have been caught by constructor
      std::fill(output.begin(), output.end(), T{0});
      return Status::Success;  // Or an error if this state is invalid
    }
    if (output.size() < input.size()) {
      return Status::SizeMismatch;
    }
    if (input.empty()) {  // No input to process
      // Zero out the output buffer up to its actual size.
      std::fill(output.begin(), output.end(), T{0});
      return Status::Success;
    }

    size_t num_taps = coefficients_.size();
    size_t state_len = state_.size();  // num_taps - 1

    // Efficiently process samples using state buffer
    for (size_t i = 0; i < input.size(); ++i) {
      T acc = T{0};
      // Slide new input sample into state (conceptually)
      // For direct form, state holds past inputs x[n-1], x[n-2], ...
      // Or, for transposed direct form, state holds intermediate sums.
      // The current implementation seems to use a temporary work_buffer,
      // let's optimize to use state_ directly if possible or clarify the form.

      // Assuming Direct Form FIR: state_ stores x[n-k] for k=1 to M
      // Prepend state to current input for convolution sum
      // This can be optimized by managing a circular buffer for state.
      // For now, using a temporary buffer for clarity, matching original logic.
      std::vector<T> current_block;
      current_block.reserve(num_taps);

      // Fill with state (past samples) in reverse order for convolution
      for (size_t j = 0; j < state_len; ++j) {
        current_block.push_back(state_[state_len - 1 - j]);
      }
      // Fill with current and future input samples needed for the current
      // output
      for (size_t j = 0; j < num_taps - state_len; ++j) {
        if (i + j < input.size()) {
          current_block.push_back(input[i + j]);
        }
        else {
          // This case should not be hit if logic is correct for full input
          // processing For partial last block, this might be needed if not
          // padding.
          current_block.push_back(T{0});  // Or handle boundary
        }
      }
      // The above block construction is not quite right for a sliding window.
      // Let's revert to a simpler, though potentially less performant for very
      // large inputs, version that's easier to reason about, similar to the
      // original's work_buffer.

      T current_input_sample = input[i];
      acc = coefficients_[0] * current_input_sample;
      for (size_t k = 1; k < num_taps; ++k) {
        if (k <= state_len) {  // Check k-1 because state_ is 0-indexed for past
                               // samples
          acc += coefficients_[k] * state_[k - 1];
        }
        else {
          // This branch indicates an error in state/coeff logic
          // or that the filter structure is different.
          // For a standard FIR, all coefficients multiply either current input
          // or state.
        }
      }
      // The original code's work_buffer approach for direct convolution:
      // output[i] = sum(coeffs[k] * combined_buffer[input_start + i - k])

      // Simpler direct form:
      // y[n] = b0*x[n] + b1*x[n-1] + ... + bM*x[n-M]
      // state_ stores x[n-1], x[n-2], ... x[n-M]
      // state_[0] = x[n-1], state_[1] = x[n-2]

      acc = T{0};
      acc += coefficients_[0] * current_input_sample;  // b0 * x[n]
      for (size_t k = 1; k < num_taps; ++k) {          // k from 1 to M
        acc += coefficients_[k]
               * state_[k - 1];  // bk * x[n-k] (state_[k-1] is x[n-k])
      }
      output[i] = acc;

      // Update state: shift state and insert current_input_sample
      // state_new[0] = current_input_sample (becomes x[n-1] for next iteration)
      // state_new[k] = state_old[k-1]
      if (state_len > 0) {
        for (size_t k = state_len - 1; k > 0; --k) {
          state_[k] = state_[k - 1];
        }
        state_[0] = current_input_sample;
      }
    }

    // If output buffer is larger than input, zero out remaining samples
    if (output.size() > input.size()) {
      std::fill(output.begin() + input.size(), output.end(), T{0});
    }
    return Status::Success;
  }

  // Reset
  template <typename T>
  Status FIRFilterPlanImpl<T>::reset()
  {
    std::fill(state_.begin(), state_.end(), T{0});
    return Status::Success;
  }

  // Get Order
  template <typename T>
  size_t FIRFilterPlanImpl<T>::get_order() const
  {
    return coefficients_.empty() ? 0 : coefficients_.size() - 1;
  }

  // Get Num Taps
  template <typename T>
  size_t FIRFilterPlanImpl<T>::get_num_taps() const
  {
    return coefficients_.size();
  }

  // Get Coefficients (Added as per original file structure)
  template <typename T>
  std::span<const T> FIRFilterPlanImpl<T>::get_coefficients() const
  {
    return std::span<const T>(coefficients_);
  }

  // --- IIRFilterPlanImpl ---
  // Constructor
  template <typename T>
  IIRFilterPlanImpl<T>::IIRFilterPlanImpl(
      const std::vector<IIRFilterCoef>& sos_coefficients)
  {
    if (sos_coefficients.empty()) {
      throw std::invalid_argument(
          "IIR SOS coefficients cannot be empty for IIRFilterPlanImpl.");
    }
    internal_coeffs_.reserve(sos_coefficients.size());
    for (const auto& sos_double : sos_coefficients) {
      InternalSOS<T> sos_t;
      sos_t.b0 = static_cast<T>(sos_double.b0);
      sos_t.b1 = static_cast<T>(sos_double.b1);
      sos_t.b2 = static_cast<T>(sos_double.b2);
      // It's common to normalize a0 to 1. If not, all b's and a's (except a0)
      // should be divided by a0.
      if (std::abs(sos_double.a0 - 1.0) > 1e-9) {  // Check if a0 is not 1.0
        // If a0 is significantly different from 1, normalization might be
        // needed or it's an error. For Direct Form II Transposed, a0 is usually
        // assumed to be 1. If a0 is not 1, the filter difference equation
        // changes. This implementation assumes a0=1.
        auto logger = spdlog::get("OmniDSP");
        if (!logger) {
          logger = spdlog::default_logger();
        }
        logger->warn(
            "IIR SOS coefficient a0 is {} (not 1.0). Ensure coefficients are "
            "pre-normalized if a0=1 is assumed by the form.",
            sos_double.a0);
        // If strict a0=1 is required:
        // throw std::runtime_error("IIR SOS coefficient a0 must be 1.0 (or
        // coefficients pre-normalized).");
      }
      sos_t.a1 = static_cast<T>(
          sos_double.a1);  // This is -a1 from standard form if a0=1
      sos_t.a2 = static_cast<T>(
          sos_double.a2);  // This is -a2 from standard form if a0=1
      internal_coeffs_.push_back(sos_t);
    }
    state_.assign(
        internal_coeffs_.size() * 2,
        T{0});  // 2 states per SOS section (w[n-1], w[n-2])
  }

  // Destructor
  template <typename T>
  IIRFilterPlanImpl<T>::~IIRFilterPlanImpl() = default;

  // Execute (Direct Form II Transposed for SOS)
  template <typename T>
  Status IIRFilterPlanImpl<T>::execute(
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
      std::fill(output.begin(), output.end(), T{0});
      return Status::Success;
    }

    size_t num_sections = internal_coeffs_.size();
    for (size_t n = 0; n < input.size(); ++n) {
      T section_input = input[n];  // Input to the first section
      T section_output = T{0};  // Output of the last section will be output[n]

      for (size_t k = 0; k < num_sections; ++k) {
        const auto& sos = internal_coeffs_[k];
        T& w_n_minus_1 = state_[k * 2 + 0];  // State w[n-1] for section k
        T& w_n_minus_2 = state_[k * 2 + 1];  // State w[n-2] for section k

        // y_k[n] = b0_k * w_k[n] + b1_k * w_k[n-1] + b2_k * w_k[n-2]
        // w_k[n] = x_k[n] - a1_k * w_k[n-1] - a2_k * w_k[n-2]
        // where x_k[n] is input to current section, y_k[n] is output of current
        // section

        T w_n = section_input - sos.a1 * w_n_minus_1 - sos.a2 * w_n_minus_2;
        section_output
            = sos.b0 * w_n + sos.b1 * w_n_minus_1 + sos.b2 * w_n_minus_2;

        // Update states for next sample
        w_n_minus_2 = w_n_minus_1;
        w_n_minus_1 = w_n;

        section_input
            = section_output;  // Output of this section is input to the next
      }
      output[n] = section_output;  // Final output after all sections
    }

    if (output.size() > input.size()) {
      std::fill(output.begin() + input.size(), output.end(), T{0});
    }
    return Status::Success;
  }

  // Reset
  template <typename T>
  Status IIRFilterPlanImpl<T>::reset()
  {
    std::fill(state_.begin(), state_.end(), T{0});
    return Status::Success;
  }

  // Get Order
  template <typename T>
  size_t IIRFilterPlanImpl<T>::get_order() const
  {
    // Effective order is roughly num_sections * 2 for SOS.
    // This might need a more precise definition based on how "order" is defined
    // for cascaded SOS.
    return internal_coeffs_.empty() ? 0 : internal_coeffs_.size() * 2;
  }

  // Get Num Sections
  template <typename T>
  size_t IIRFilterPlanImpl<T>::get_num_sections() const
  {
    return internal_coeffs_.size();
  }

  // --- Explicit Template Instantiations ---
  template class FIRFilterPlanImpl<F32>;
  template class FIRFilterPlanImpl<F64>;
  template class FIRFilterPlanImpl<C32>;  // FIR filters can be complex
  template class FIRFilterPlanImpl<C64>;  // FIR filters can be complex

  // get_coefficients needs to be instantiated for each type FIRFilterPlanImpl
  // is instantiated with.
  template std::span<const F32> FIRFilterPlanImpl<F32>::get_coefficients()
      const;
  template std::span<const F64> FIRFilterPlanImpl<F64>::get_coefficients()
      const;
  template std::span<const C32> FIRFilterPlanImpl<C32>::get_coefficients()
      const;
  template std::span<const C64> FIRFilterPlanImpl<C64>::get_coefficients()
      const;

  template class IIRFilterPlanImpl<F32>;  // IIR typically real
  template class IIRFilterPlanImpl<F64>;  // IIR typically real

  // Explicit instantiations for the design helper functions
  template OmniExpected<FIRCoefs<F32>> generate_fir_filter_coeffs<F32>(
      const FIRFilterSpec& spec);
  template OmniExpected<FIRCoefs<F64>> generate_fir_filter_coeffs<F64>(
      const FIRFilterSpec& spec);
  // No instantiation needed for non-template generate_iir_filter_coeffs

}  // namespace OmniDSP::Default
