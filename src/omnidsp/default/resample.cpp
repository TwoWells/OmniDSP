/**
 * @file resample.cpp (Default)
 * @brief Implements the Default backend ResamplePlanImpl class using standard
 * C++.
 */

#include "resample.hpp"  // Corresponding header for Default::ResamplePlanImpl

#include <OmniDSP/core_types.hpp>  // For Status, OmniExpected, F32, F64, FIRCoefs
#include <OmniDSP/filter.hpp>      // For FIRFilterSpec (part of ResampleSpec)
#include <OmniDSP/resample.hpp>    // For ResampleSpec definition
// Window.hpp is included by filter.hpp if FIRFilterSpec needs WindowSetup,
// or directly if ResampleSpec itself used WindowSetup directly (it doesn't
// anymore)

#include <algorithm>  // For std::copy, std::fill, std::max
#include <cassert>    // For assert macro
#include <cmath>      // For std::ceil, std::floor, std::min, std::max, std::abs
#include <iostream>   // For debug messages (though spdlog is preferred)
#include <numeric>    // For std::gcd (not needed here anymore, L/M are in spec)
#include <span>       // For std::span
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <utility>    // For std::move
#include <vector>

#include "interface/backend.hpp"  // For Abstract::Backend base class
#include "spdlog/spdlog.h"        // For logging
// "utils/resample.hpp" for calculate_resampling_factors is no longer needed
// here, as L/M are provided in ResampleSpec. "utils/filter_design.hpp" for
// design_resampling_prototype_filter is no longer needed here, as ResampleSpec
// contains the prototype_fir_spec.

namespace OmniDSP::Default {

  //--------------------------------------------------------------------------
  // ResamplePlanImpl Method Implementations
  //--------------------------------------------------------------------------

  template <typename T>
  ResamplePlanImpl<T>::ResamplePlanImpl(
      const Abstract::Backend* owner, const ResampleSpec& spec)
      : owner_backend_(owner),  // Store the owner backend
        spec_(spec),            // Store the ResampleSpec
        interpolation_factor_(spec.up_factor_L),
        decimation_factor_(spec.down_factor_M),
        prototype_coeffs_(),  // Initialize member
        filter_length_(0),
        current_phase_(0)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (!owner_backend_) {
      logger->error("Default::ResamplePlanImpl: owner_backend is null.");
      throw OmniException(  // Using OmniException as defined in core_types.hpp
          "ResamplePlanImpl requires a valid owner AbstractBackend pointer.",
          Status::InvalidArgument);
    }
    // ResampleSpec constructor performs its own validation.
    // FIRFilterSpec within spec_.prototype_fir_spec is also validated by its
    // constructor/Utils::create_spec.

    if (interpolation_factor_ == 0 || decimation_factor_ == 0) {
      logger->error(
          "Default::ResamplePlanImpl: L ({}) or M ({}) from spec is zero.",
          interpolation_factor_,
          decimation_factor_);
      throw OmniException(
          "Resampling factors L or M from spec cannot be zero.",
          Status::InvalidArgument);
    }

    design_filter();  // This will populate prototype_coeffs_ and filter_length_
                      // using spec_.prototype_fir_spec

    if (filter_length_ > 0) {
      size_t num_branch_taps = (filter_length_ + interpolation_factor_ - 1)
                               / interpolation_factor_;
      size_t state_len = (num_branch_taps > 0) ? num_branch_taps - 1 : 0;
      state_.assign(state_len, T{0});
    }
    else {
      state_.clear();
      logger->warn(
          "Default::ResamplePlanImpl: Prototype filter has zero taps after "
          "design. This might indicate an issue.");
    }

    build_polyphase_filters();  // Uses prototype_coeffs_

    logger->debug(
        "Default::ResamplePlanImpl created. IR={}, OR={}, Q={}, L={}, M={}, "
        "ProtoTaps={}, StateLen={}",
        spec_.input_rate,
        spec_.output_rate,
        spec_.quality,
        interpolation_factor_,
        decimation_factor_,
        filter_length_,
        state_.size());
  }

  template <typename T>
  ResamplePlanImpl<T>::~ResamplePlanImpl() = default;

  template <typename T>
  void ResamplePlanImpl<T>::design_filter()
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (!owner_backend_) {  // Should have been caught by constructor
      logger->critical(
          "Default::ResamplePlanImpl::design_filter: owner_backend_ is null. "
          "This should not happen.");
      throw std::logic_error("Cannot design filter without owner backend.");
    }

    // The ResampleSpec now contains the fully resolved FIRFilterSpec for the
    // prototype. We use the owner_backend_ to "generate" the coefficients from
    // this spec. The Default::Backend implementation of design_fir_filter_fXX
    // will call Default::generate_fir_filter_coeffs(spec_.prototype_fir_spec).

    OmniExpected<FIRCoefs<T>> coeffs_expected;
    if constexpr (std::is_same_v<T, float>) {
      coeffs_expected
          = owner_backend_->design_fir_filter_f32(spec_.prototype_fir_spec);
    }
    else if constexpr (std::is_same_v<T, double>) {
      coeffs_expected
          = owner_backend_->design_fir_filter_f64(spec_.prototype_fir_spec);
    }
    else {
      logger->critical(
          "Default::ResamplePlanImpl::design_filter: Unsupported data type T.");
      throw std::runtime_error(
          "Unsupported type for filter design in resampler.");
    }

    if (!coeffs_expected) {
      logger->error(
          "Default::ResamplePlanImpl::design_filter: Failed to design "
          "resampling prototype filter using owner backend. Status: {}",
          static_cast<int>(coeffs_expected.error()));
      throw OmniException(  // Using OmniException
          "Failed to design resampling prototype filter. Status: "
              + std::string(get_status_string(coeffs_expected.error())),
          coeffs_expected.error());
    }

    prototype_coeffs_ = std::move(coeffs_expected.value());

    if (prototype_coeffs_.empty()) {
      logger->error(
          "Default::ResamplePlanImpl::design_filter: Resampling prototype "
          "filter design resulted in empty coefficients.");
      throw OmniException(  // Using OmniException
          "Resampling prototype filter design resulted in empty coefficients.",
          Status::Failure);
    }

    filter_length_ = prototype_coeffs_.size();

    // Scaling factor for the prototype filter coefficients is L
    // (interpolation_factor_) This is to compensate for the gain introduced by
    // the upsampler.
    T scale_factor = static_cast<T>(interpolation_factor_);
    for (T& coeff : prototype_coeffs_) {
      coeff *= scale_factor;
    }
    logger->debug(
        "Prototype FIR filter designed for resampler. Taps: {}, First Coeff "
        "(scaled): {}",
        filter_length_,
        prototype_coeffs_.empty() ? T{0} : prototype_coeffs_[0]);
  }

  template <typename T>
  void ResamplePlanImpl<T>::build_polyphase_filters()
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (prototype_coeffs_.empty() || filter_length_ == 0
        || interpolation_factor_ == 0) {
      logger->warn(
          "Default::ResamplePlanImpl::build_polyphase_filters: Prototype "
          "coefficients are empty or factors invalid. L={}, FilterLength={}. "
          "Polyphase bank will be empty/zeroed.",
          interpolation_factor_,
          filter_length_);
      polyphase_coeffs_.clear();
      if (filter_length_ > 0 && interpolation_factor_ > 0) {
        size_t L = interpolation_factor_;
        size_t num_branch_taps = (filter_length_ + L - 1) / L;
        polyphase_coeffs_.assign(L, std::vector<T>(num_branch_taps, T{0}));
      }
      return;
    }

    size_t L = interpolation_factor_;
    size_t num_branch_taps = (filter_length_ + L - 1) / L;

    polyphase_coeffs_.assign(L, std::vector<T>(num_branch_taps, T{0}));

    for (size_t phase = 0; phase < L; ++phase) {
      for (size_t n = 0; n < num_branch_taps; ++n) {
        size_t proto_idx = phase + n * L;
        if (proto_idx < filter_length_) {
          polyphase_coeffs_[phase][n] = prototype_coeffs_[proto_idx];
        }
      }
    }
  }

  template <typename T>
  Status ResamplePlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (interpolation_factor_ == 0 || decimation_factor_ == 0
        || polyphase_coeffs_.empty()
        || (polyphase_coeffs_.empty() ? true : polyphase_coeffs_[0].empty())) {
      auto logger = spdlog::get("OmniDSP");
      if (!logger) {
        logger = spdlog::default_logger();
      }
      logger->warn(
          "Default::ResamplePlanImpl::execute: Plan not properly initialized "
          "(L/M zero or no polyphase coeffs). L={}, M={}, "
          "PolyphaseBranches={}, BranchTaps={}",
          interpolation_factor_,
          decimation_factor_,
          polyphase_coeffs_.size(),
          polyphase_coeffs_.empty() ? 0 : polyphase_coeffs_[0].size());
      return Status::NotInitialized;
    }

    size_t input_len = input.size();
    size_t output_idx = 0;
    size_t max_output_can_produce = get_output_length(input_len);
    size_t output_samples_to_write
        = std::min(max_output_can_produce, output.size());

    if (input_len == 0 && state_.empty()
        && current_phase_ == 0) {         // More precise check for no-op
      if (output_samples_to_write > 0) {  // Ensure output is zeroed if expected
        std::fill(
            output.begin(), output.begin() + output_samples_to_write, T{0});
      }
      return Status::Success;
    }
    if (output_samples_to_write == 0
        && input_len > 0) {  // If no output can be written despite input
      // This might happen if output span is 0, or if state + input is not
      // enough for even one output sample. Update state if possible.
    }

    const size_t L = interpolation_factor_;
    const size_t M = decimation_factor_;
    const size_t state_len = state_.size();
    const size_t num_branch_taps = polyphase_coeffs_[0].size();

    std::vector<T> work_buffer;
    work_buffer.reserve(state_len + input_len);
    work_buffer.insert(work_buffer.end(), state_.begin(), state_.end());
    work_buffer.insert(work_buffer.end(), input.begin(), input.end());

    size_t work_buffer_consumed_idx = 0;

    for (output_idx = 0; output_idx < output_samples_to_write; ++output_idx) {
      // Check if enough samples in work_buffer for the current polyphase filter
      // branch The "start" of the filter taps aligns with
      // work_buffer_consumed_idx + current_phase_ / L This is complex. Let's
      // simplify the view: We need 'num_branch_taps' from the conceptual
      // upsampled-then-filtered stream. The 'current_phase_' tells us which
      // polyphase filter to use. The 'work_buffer_consumed_idx' tells us how
      // many *original* input samples have been shifted into the delay line.

      if (work_buffer_consumed_idx + num_branch_taps > work_buffer.size()) {
        // Not enough total samples (state + new input) to compute this output.
        // This can happen if input_len is small.
        break;
      }

      const auto& current_filter_branch = polyphase_coeffs_[current_phase_];
      T sum{};
      for (size_t n = 0; n < num_branch_taps; ++n) {
        // The n-th tap of the current polyphase filter multiplies an input
        // sample. The input samples are taken from the work_buffer, starting at
        // work_buffer_consumed_idx. The polyphase structure means that for
        // branch `p` and tap `n`, the original coefficient was h[p + n*L]. The
        // input sample this corresponds to is x[current_input_sample_block_idx
        // - n]. state_[0] is oldest, state_[state_len-1] is newest from
        // previous block. work_buffer = [state_N, state_N-1, ..., state_1,
        // input_0, input_1, ...] For FIR filter: y[j] = sum(h[k] * x[j-k])
        // Polyphase filter branch `p`: e_p[n] = h[p + n*L]
        // Output of branch `p`: y_p[m] = sum(e_p[n] * x_upsampled[m*L - p -
        // n*L]) This is equivalent to sum(e_p[n] * x_original[m - n -
        // p/L_if_p_is_multiple_of_L]) Let's use a simpler direct convolution
        // with the current branch and the relevant part of work_buffer. The
        // state_ buffer holds previous input samples.
        // work_buffer[work_buffer_consumed_idx + n] are the effective input
        // samples for this branch.
        sum += work_buffer[work_buffer_consumed_idx + n]
               * current_filter_branch[n];
      }
      output[output_idx] = sum;

      current_phase_
          = (current_phase_ + M);  // Accumulate phase for next output sample
      size_t num_input_samples_to_advance
          = current_phase_ / L;  // Number of full input blocks to shift
      current_phase_ %= L;       // Remainder is the next phase offset
      work_buffer_consumed_idx += num_input_samples_to_advance;
    }

    // Update state_ with the remaining unprocessed samples from work_buffer
    if (state_len > 0) {
      if (work_buffer_consumed_idx >= work_buffer.size()) {
        // All samples in work_buffer (including original state) were consumed
        // or passed over.
        std::fill(state_.begin(), state_.end(), T{0});
      }
      else {
        size_t remaining_in_work_buffer
            = work_buffer.size() - work_buffer_consumed_idx;
        size_t num_to_copy_to_state
            = std::min(state_len, remaining_in_work_buffer);

        // Copy the *last* `num_to_copy_to_state` elements from the *relevant
        // part* of work_buffer The relevant part starts at
        // work_buffer_consumed_idx
        std::copy(
            work_buffer.begin() + work_buffer_consumed_idx,
            work_buffer.begin() + work_buffer_consumed_idx
                + num_to_copy_to_state,
            state_.begin());
        if (num_to_copy_to_state < state_len) {
          std::fill(state_.begin() + num_to_copy_to_state, state_.end(), T{0});
        }
      }
    }

    // Zero out remaining part of the output buffer if it's larger than what was
    // written
    if (output_idx < output.size()) {
      std::fill(output.begin() + output_idx, output.end(), T{0});
    }

    return Status::Success;
  }

  template <typename T>
  Status ResamplePlanImpl<T>::reset()
  {
    std::fill(state_.begin(), state_.end(), T{0});
    current_phase_ = 0;
    return Status::Success;
  }

  template <typename T>
  double ResamplePlanImpl<T>::get_input_rate() const
  {
    return spec_.input_rate;
  }

  template <typename T>
  double ResamplePlanImpl<T>::get_output_rate() const
  {
    return spec_.output_rate;
  }

  template <typename T>
  size_t ResamplePlanImpl<T>::get_output_length(size_t input_len) const
  {
    if (interpolation_factor_ == 0 || decimation_factor_ == 0) return 0;

    // Number of samples available including current state for processing this
    // input block
    size_t effective_input_samples = input_len + state_.size();
    if (effective_input_samples == 0 && current_phase_ == 0) return 0;

    // Calculate how many output samples can be generated
    // Each output sample consumes M/L input samples on average.
    // The current_phase_ is an accumulator for the fractional input sample
    // position. Total "phase units" available = current_phase_ (from previous
    // block) + input_len * L Number of output samples = floor
    // (total_phase_units / M)
    return (current_phase_ + input_len * interpolation_factor_)
           / decimation_factor_;
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  template class ResamplePlanImpl<float>;
  template class ResamplePlanImpl<double>;

}  // namespace OmniDSP::Default
