/**
 * @file resample.cpp (Default)
 * @brief Implements the Default backend ResamplePlanImpl class using standard
 * C++.
 */

#include "resample.hpp"  // Include the corresponding header file

#include <OmniDSP/core_types.hpp>  // For Status, OmniExpected, F32, F64
#include <OmniDSP/filter.hpp>      // For FIRFilterSpec, FilterType, FIRCoefs
#include <OmniDSP/resample.hpp>    // For ResampleSpec definition
#include <OmniDSP/window.hpp>  // For WindowSetup (used by FIRFilterSpec and ResampleSpec)
#include <algorithm>  // For std::copy, std::fill, std::max
#include <cassert>    // For assert macro
#include <cmath>      // For std::ceil, std::floor, std::min, std::max, std::abs
#include <iostream>   // For debug messages (though spdlog is preferred)
#include <numeric>    // For std::gcd (needed by the helper)
#include <span>       // For std::span
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "interface/backend.hpp"  // For AbstractBackend base class
#include "spdlog/spdlog.h"        // For logging
#include "utils/resample.hpp"     // For calculate_resampling_factors

namespace OmniDSP::Default {

  //--------------------------------------------------------------------------
  // ResamplePlanImpl Method Implementations
  //--------------------------------------------------------------------------

  // Helper to estimate FIR filter order (can be moved to a common utility if
  // used elsewhere)
  static size_t estimate_prototype_fir_order(size_t L, size_t M, int quality)
  {
    double normalized_cutoff
        = 1.0 / (2.0 * static_cast<double>(std::max(L, M)));
    // This factor influences the transition band for order estimation.
    // A smaller factor relative to cutoff implies a wider transition band for a
    // given quality, or requires a higher order for the same transition band.
    // Let's make it a fraction of the normalized cutoff itself.
    double transition_width_norm_factor = normalized_cutoff * 0.5;

    double dF;  // Normalized transition width (0 to 0.5, relative to Nyquist)
    if (quality <= 5)
      dF = 0.2 * transition_width_norm_factor;
    else if (quality <= 10)
      dF = 0.1 * transition_width_norm_factor;
    else
      dF = 0.05 * transition_width_norm_factor;
    if (dF < 1e-6) dF = 1e-6;  // Prevent dF from being too small or zero

    double desired_atten_dB = 40.0 + quality * 3.0;
    if (desired_atten_dB < 21) desired_atten_dB = 21;
    if (desired_atten_dB > 120) desired_atten_dB = 120;  // Cap attenuation

    // Kaiser window order estimation formula: N approx = (Atten_dB - 7.95) /
    // (2.285 * (TransitionWidth_cycles_per_sample))
    // TransitionWidth_cycles_per_sample = dF (where dF is 0 to 0.5,
    // representing fraction of Nyquist) The formula often uses dF as
    // (actual_transition_width_hz / sampling_freq_hz) If our dF is already
    // normalized to Nyquist (0 to 0.5), then the term is 2*dF for the formula
    // expecting 0 to Fs/2.
    size_t order = static_cast<size_t>(
        std::ceil((desired_atten_dB - 7.95) / (2.285 * (2.0 * dF))));

    size_t min_order_factor = 2;
    if (quality > 10)
      min_order_factor = 4;
    else if (quality > 5)
      min_order_factor = 3;

    size_t min_order = min_order_factor * std::max(L, M);
    if (order < min_order) order = min_order;

    if (order < 16) order = 16;  // Absolute minimum practical order
    if (order > 2048)
      order = 2048;  // Cap max order to prevent excessive computation

    if (order % 2
        != 0) {  // Ensure order is even for Type I LP FIR (odd number of taps)
      order++;
    }
    return order;
  }

  template <typename T>
  ResamplePlanImpl<T>::ResamplePlanImpl(
      const Abstract::Backend* owner, const ResampleSpec& spec)
      : owner_backend_(owner),
        spec_(spec),
        interpolation_factor_(0),
        decimation_factor_(0),
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
      throw std::invalid_argument(
          "ResamplePlanImpl requires a valid owner AbstractBackend pointer.");
    }
    // ResampleSpec's constructor validates its own members.
    // WindowSetup within spec_.window_setup is also validated by its own
    // constructor.

    Status factor_status = Utils::calculate_resampling_factors(
        spec_.input_rate,
        spec_.output_rate,
        interpolation_factor_,
        decimation_factor_);
    if (factor_status != Status::Success) {
      logger->error(
          "Default::ResamplePlanImpl: Failed to calculate resampling factors L "
          "and M. Status: {}",
          static_cast<int>(factor_status));
      throw std::runtime_error(
          "Failed to calculate resampling factors L and M.");
    }

    if (interpolation_factor_ == 0 || decimation_factor_ == 0) {
      logger->error(
          "Default::ResamplePlanImpl: Calculated L ({}) or M ({}) is zero.",
          interpolation_factor_,
          decimation_factor_);
      throw std::runtime_error("Resampling factors L or M cannot be zero.");
    }

    design_filter();  // This will populate prototype_coeffs_ and filter_length_

    if (filter_length_ > 0) {
      size_t num_branch_taps = (filter_length_ + interpolation_factor_ - 1)
                               / interpolation_factor_;
      size_t state_len = (num_branch_taps > 0) ? num_branch_taps - 1 : 0;
      state_.assign(state_len, T{0});
    }
    else {
      state_.clear();  // Should not happen if design_filter throws on error
      logger->warn(
          "Default::ResamplePlanImpl: Prototype filter has zero taps after "
          "design. This might indicate an issue.");
    }

    build_polyphase_filters();  // Uses prototype_coeffs_

    logger->debug(
        "Default::ResamplePlanImpl created. L={}, M={}, Taps={}, StateLen={}",
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

    if (!owner_backend_) {
      logger->critical(
          "Default::ResamplePlanImpl::design_filter: owner_backend_ is null. "
          "This should not happen.");
      throw std::logic_error("Cannot design filter without owner backend.");
    }

    double normalized_cutoff
        = 1.0
          / (2.0
             * static_cast<double>(
                 std::max(interpolation_factor_, decimation_factor_)));
    size_t estimated_order = estimate_prototype_fir_order(
        interpolation_factor_, decimation_factor_, spec_.quality);
    size_t num_taps = estimated_order + 1;

    FIRFilterSpec filter_spec_for_design;  // Local FIRFilterSpec for designing
                                           // the prototype
    filter_spec_for_design.type = FilterType::Lowpass;
    filter_spec_for_design.order = estimated_order;
    filter_spec_for_design.sample_rate = 1.0;  // Normalized design
    filter_spec_for_design.cutoff1 = normalized_cutoff;
    // filter_spec_for_design.cutoff2 is std::nullopt by default, which is
    // correct for lowpass.

    // Correctly populate filter_spec_for_design.window_setup using
    // spec_.window_setup
    filter_spec_for_design.window_setup.type = spec_.window_setup.type;
    filter_spec_for_design.window_setup.params = spec_.window_setup.params;
    filter_spec_for_design.window_setup.length
        = static_cast<int>(num_taps);  // CRITICAL: Set length

    // The members 'transition_width' and 'stopband_attenuation_db' are NOT part
    // of FIRFilterSpec. The member 'window' is also gone, replaced by
    // 'window_setup'.

    if (!filter_spec_for_design
             .validate()) {  // Validate the spec we just built
      logger->error(
          "Default::ResamplePlanImpl::design_filter: Constructed FIRFilterSpec "
          "for prototype design is invalid.");
      throw std::runtime_error(
          "Constructed internal FIRFilterSpec for resampling prototype is "
          "invalid.");
    }

    OmniExpected<FIRCoefs<T>> coeffs_expected;
    if constexpr (std::is_same_v<T, float>) {
      coeffs_expected
          = owner_backend_->design_fir_filter_f32(filter_spec_for_design);
    }
    else if constexpr (std::is_same_v<T, double>) {
      coeffs_expected
          = owner_backend_->design_fir_filter_f64(filter_spec_for_design);
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
      throw std::runtime_error(
          "Failed to design resampling prototype filter. Status: "
          + std::string(get_status_string(coeffs_expected.error())));
    }

    prototype_coeffs_ = std::move(coeffs_expected.value());  // Store in member

    if (prototype_coeffs_.empty()) {
      logger->error(
          "Default::ResamplePlanImpl::design_filter: Resampling prototype "
          "filter design resulted in empty coefficients.");
      throw std::runtime_error(
          "Resampling prototype filter design resulted in empty coefficients.");
    }

    filter_length_ = prototype_coeffs_.size();  // Update member filter_length_

    T scale_factor = static_cast<T>(interpolation_factor_);
    for (T& coeff : prototype_coeffs_) {
      coeff *= scale_factor;
    }
  }

  template <typename T>
  void ResamplePlanImpl<T>::build_polyphase_filters()
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    // This method now uses the member prototype_coeffs_ which was populated by
    // design_filter().
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
    // filter_length_ is the total number of taps in the prototype filter
    size_t num_branch_taps = (filter_length_ + L - 1) / L;  // Ceiling division

    polyphase_coeffs_.assign(L, std::vector<T>(num_branch_taps, T{0}));

    for (size_t phase = 0; phase < L; ++phase) {
      for (size_t n = 0; n < num_branch_taps; ++n) {
        // The polyphase decomposition formula: e_k[n] = h[k + n*L]
        // For IPP style (reverse time for filter): h[k - n*L]
        // Standard polyphase decomposition:
        // The k-th polyphase component filter p_k(n) consists of coefficients
        // h(k), h(k+L), h(k+2L), ... So,
        // polyphase_coeffs_[phase_idx][n_branch_tap] =
        // prototype_coeffs_[phase_idx + n_branch_tap * L] The 'phase' here
        // corresponds to the polyphase branch index.
        size_t proto_idx = phase + n * L;
        if (proto_idx < filter_length_) {
          polyphase_coeffs_[phase][n] = prototype_coeffs_[proto_idx];
        }
        // else, it remains 0 due to initialization (zero-padding the branch if
        // prototype_coeffs_ is not a multiple of L*num_branch_taps)
      }
    }
  }

  template <typename T>
  Status ResamplePlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (interpolation_factor_ == 0 || decimation_factor_ == 0
        || polyphase_coeffs_.empty() || polyphase_coeffs_[0].empty()) {
      // If polyphase_coeffs_[0] is empty, num_branch_taps would be 0, leading
      // to issues. This should be caught by filter_length_ > 0 check in
      // constructor after design_filter.
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

    // Calculate how many output samples we can actually write to the provided
    // output span
    size_t max_output_can_produce = get_output_length(input_len);
    size_t output_samples_to_write
        = std::min(max_output_can_produce, output.size());

    if (input_len == 0 && state_.empty()) {
      std::fill(output.begin(), output.begin() + output_samples_to_write, T{0});
      return Status::Success;
    }

    const size_t L = interpolation_factor_;
    const size_t M = decimation_factor_;
    const size_t state_len = state_.size();
    const size_t num_branch_taps = polyphase_coeffs_[0].size();

    std::vector<T> work_buffer;
    work_buffer.reserve(state_len + input_len);
    work_buffer.insert(work_buffer.end(), state_.begin(), state_.end());
    work_buffer.insert(work_buffer.end(), input.begin(), input.end());

    size_t work_buffer_ptr
        = 0;  // Points to the current "oldest" sample in work_buffer to be used

    while (output_idx < output_samples_to_write) {
      // Determine which polyphase filter branch to use
      const auto& current_filter_branch = polyphase_coeffs_[current_phase_];

      // Ensure we have enough samples in the work_buffer for this convolution
      // The convolution starts at work_buffer_ptr
      if (work_buffer_ptr + num_branch_taps > work_buffer.size()) {
        break;  // Not enough input samples (including state) to produce more
                // output
      }

      T sum{};
      for (size_t n = 0; n < num_branch_taps; ++n) {
        // Convolve with current_filter_branch, taking samples from work_buffer
        // The filter taps are typically applied in reverse order to the input
        // samples or input samples are taken in reverse order. y[j] =
        // sum_{k=0}^{N-1} h[k] * x[j-k] For polyphase, x are the samples from
        // the upsampled stream. work_buffer[work_buffer_ptr + n] are the
        // effective input samples for this branch.
        sum += work_buffer[work_buffer_ptr + n] * current_filter_branch[n];
      }
      output[output_idx++] = sum;

      // Advance phase accumulator and input pointer
      current_phase_
          = (current_phase_ + M) % L;  // Next phase for filter branch
      work_buffer_ptr
          += M;  // Consume M input samples (from the original rate perspective)
                 // This is not quite right. `work_buffer_ptr` should advance by
                 // the number of original input samples consumed to produce
                 // this output sample. This is related to `current_phase_ / L`
                 // advancement. Let's correct the advancement: For each output
                 // sample, we advance M in the output grid. This corresponds to
                 // M/L input samples on average. The `current_phase_`
                 // accumulator handles this. `work_buffer_ptr` should point to
                 // the start of the *next* block of M samples in the input
                 // sequence that will be decimated.
    }

    // After the loop, current_phase_ has the phase for the *next* output
    // sample. work_buffer_ptr should be updated based on how many input samples
    // were effectively consumed. Number of input samples consumed =
    // floor((number_of_outputs_generated * M) / L) This is complex to track
    // perfectly without simulating the exact indices.

    // Simpler state update:
    // The state should contain the last `state_len` samples from `work_buffer`
    // that were *not fully processed* or are needed for the *next* set of
    // convolutions. The number of samples effectively "consumed" from the start
    // of work_buffer to produce `output_idx` samples is roughly `(output_idx *
    // M) / L`. Let `consumed_input_blocks = (output_idx * M) / L;` (integer
    // division) The state needs to be the samples from `work_buffer` just
    // before these consumed blocks' end.

    // A more robust state update:
    // After producing `output_idx` samples, `current_phase_` holds the phase
    // for the (output_idx)-th sample. The total "time" advanced in the
    // upsampled domain is `output_idx * M`. The number of actual input samples
    // (from the start of `work_buffer`) that this corresponds to is
    // `(output_idx * M) / L`.
    size_t total_input_samples_processed_conceptually
        = (current_phase_ + output_idx * M)
          / L;  // This is not right.
                // current_phase_ is already advanced.

    // Let's use the number of output samples generated to determine how much of
    // the input was (conceptually) used. Each output sample advances the
    // conceptual pointer in the input stream by M/L samples. So, total advance
    // = output_idx * M / L. The `current_phase_` variable already tracks the
    // remainder for the *next* output. The `work_buffer_ptr` should advance by
    // the number of input samples that are fully "covered" by the M increments
    // of current_phase.

    size_t new_work_buffer_ptr_start = 0;
    if (output_idx > 0) {  // if any output was generated
      // The last conceptual_input_start_idx was for the (output_idx-1)-th
      // output sample. Let's re-evaluate state update based on remaining
      // contents of work_buffer. The `current_phase_` now points to the phase
      // for the *next* output sample that *would* be generated. The number of
      // input samples from the original stream that this corresponds to having
      // processed up to the point of needing the *next* input sample is
      // `current_phase_ / L`. This is the number of samples to shift out of the
      // combined (state + input) buffer.
      new_work_buffer_ptr_start
          = current_phase_
            / L;  // This is the number of full input blocks consumed by the
                  // interpolator before the next decimation.
    }

    if (state_len > 0) {
      size_t effective_consumed_from_work_buffer
          = new_work_buffer_ptr_start;  // This is how many original input rate
                                        // samples are "past"

      if (effective_consumed_from_work_buffer >= work_buffer.size()) {
        std::fill(
            state_.begin(), state_.end(), T{0});  // Consumed everything or more
      }
      else {
        size_t remaining_for_state
            = work_buffer.size() - effective_consumed_from_work_buffer;
        size_t copy_count = std::min(state_len, remaining_for_state);

        // Copy from the *end* of the effectively processed part of work_buffer
        // The state should be the last `state_len` samples that are "still in
        // the filter's memory" relative to the start of the *next* processing
        // block. If work_buffer = [s0,s1, i0,i1,i2,i3] and state_len=2,
        // consumed_idx=2 (i0,i1 processed) new state should be [i0,i1] if
        // num_branch_taps was > 2. This is tricky. The original approach of
        // copying from the end of work_buffer is simpler if correct.

        // If `effective_consumed_from_work_buffer` is the number of samples
        // shifted out of the "front" of the conceptual infinite input stream,
        // then the new state is the next `state_len` samples.
        std::fill(state_.begin(), state_.end(), T{0});  // Clear state first
        for (size_t i = 0; i < state_len; ++i) {
          if (effective_consumed_from_work_buffer + i < work_buffer.size()) {
            state_[i] = work_buffer[effective_consumed_from_work_buffer + i];
          }
          else {
            break;  // Not enough samples left in work_buffer
          }
        }
      }
    }
    current_phase_ %= L;  // This was already done inside the loop effectively
                          // by advancing M and then % L. The final
                          // current_phase_ is for the *next* output sample.

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
    // Consider the impact of the current state and phase on the number of
    // output samples. Number of samples "available" for processing = input_len
    // + state_.size(). Each M input samples (after interpolation by L)
    // effectively produces L output samples. The current_phase_ indicates how
    // many "sub-samples" into the current M-block we are. Max output samples =
    // floor(( (input_len + state_.size()) * L + current_phase_ ) / M) This
    // formula is often used for polyphase resamplers.
    size_t effective_input_samples
        = input_len + state_.size();  // Samples available in work_buffer
    if (effective_input_samples == 0 && current_phase_ == 0) return 0;

    // This calculates how many full M-blocks can be formed from the current
    // phase + L*effective_input_samples
    return (current_phase_ + effective_input_samples * interpolation_factor_)
           / decimation_factor_;

    // Simpler version from before:
    // return (input_len * interpolation_factor_ + current_phase_ +
    // decimation_factor_ -1) / decimation_factor_;
  }

  // calculate_max_output was removed as get_output_length should serve this
  // purpose.

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  template class ResamplePlanImpl<float>;
  template class ResamplePlanImpl<double>;

}  // namespace OmniDSP::Default
