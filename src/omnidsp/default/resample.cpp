/**
 * @file resample.cpp (Default)
 * @brief Implements the Default backend ResampleProcessorImpl class using
 * standard C++.
 */

#include "resample.hpp"  // Corresponding header for Default::ResampleProcessorImpl

#include <spdlog/fmt/ostr.h>  // For logging custom types with operator<<

#include <OmniDSP/coefs/fir_filter.hpp>  // For Coefs::FIRFilter
#include <OmniDSP/core_types.hpp>  // For OmniStatus, OmniException, F32, F64, get_status_string
#include <OmniDSP/design/fir_filter.hpp>  // For Design::FIRFilter (part of Design::Resample)
#include <OmniDSP/design/resample.hpp>  // For Design::Resample definition and its operator<<
#include <OmniDSP/params/fir_filter.hpp>  // For Params::FIRFilter (used in design_filter)
#include <algorithm>                      // For std::fill, std::min, std::copy
#include <cassert>                        // For assert
#include <cmath>                          // For std::ceil, std::abs, std::round
#include <iostream>  // For std::cerr (used by some logging)
#include <numeric>  // For std::accumulate (if used for gain normalization, not currently)
#include <span>     // For std::span
#include <sstream>  // For std::ostringstream
#include <stdexcept>  // For std::logic_error, std::runtime_error, std::invalid_argument
#include <string>   // For std::string in exceptions
#include <utility>  // For std::move
#include <vector>   // For std::vector

#include "interface/backend.hpp"  // For Abstract::Backend base class
#include "spdlog/spdlog.h"

/**
 * @namespace OmniDSP::Default
 * @brief Contains the default backend implementations for OmniDSP operations.
 */
namespace OmniDSP::Default {

  //--------------------------------------------------------------------------
  // ResampleProcessorImpl Method Implementations
  //--------------------------------------------------------------------------

  /**
   * @brief Constructor for the Default ResampleProcessorImpl.
   * @details Initializes the resampling processor based on the provided design
   * specification. This involves designing a prototype FIR filter using the
   * owner backend and then building the polyphase filter bank.
   * @param owner Pointer to the Abstract::Backend instance that owns this
   * processor. Used for internal operations like filter design. Must not be
   * null.
   * @param design_spec The fully resolved Design::Resample specification.
   * @throws OmniException if owner_backend is null, resampling factors are
   * invalid, or if the internal prototype filter design fails.
   */
  template <typename T>
  ResampleProcessorImpl<T>::ResampleProcessorImpl(
      const Abstract::Backend* owner, const Design::Resample& design_spec)
      : owner_backend_(owner),
        spec_(design_spec),  // Store the provided design spec
        interpolation_factor_(design_spec.up_factor_L),
        decimation_factor_(design_spec.down_factor_M),
        prototype_coeffs_(),  // Will be filled by design_filter
        filter_length_(0),
        current_phase_(0)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "Creating Default::ResampleProcessorImpl with Design: {}",
          design_spec);
    }

    if (!owner_backend_) {
      if (logger)
        logger->error("Default::ResampleProcessorImpl: owner_backend is null.");
      throw OmniException(
          "ResampleProcessorImpl requires a valid owner AbstractBackend "
          "pointer.",
          OmniStatus::InvalidArgument);
    }

    if (interpolation_factor_ == 0 || decimation_factor_ == 0) {
      std::ostringstream msg_stream;
      msg_stream << "Default::ResampleProcessorImpl: L ("
                 << interpolation_factor_ << ") or M (" << decimation_factor_
                 << ") from design is zero.";
      if (logger) logger->error(msg_stream.str());
      throw OmniException(msg_stream.str(), OmniStatus::InvalidArgument);
    }

    design_filter();  // This will populate prototype_coeffs_ and filter_length_

    if (filter_length_ > 0) {
      size_t num_branch_taps = (filter_length_ + interpolation_factor_ - 1)
                               / interpolation_factor_;
      size_t state_len = (num_branch_taps > 0) ? num_branch_taps - 1 : 0;
      state_.assign(state_len, T{0});
    }
    else {
      state_.clear();
      if (logger)
        logger->warn(
            "Default::ResampleProcessorImpl: Prototype filter has zero taps "
            "after design. This might indicate an issue.");
    }

    build_polyphase_filters();

    if (logger && logger->should_log(spdlog::level::debug)) {
      logger->debug(
          "Default::ResampleProcessorImpl created. IR={}, OR={}, Q={}, L={}, "
          "M={}, ProtoTaps={}, StateLen={}",
          spec_.input_rate,
          spec_.output_rate,
          spec_.quality,
          interpolation_factor_,
          decimation_factor_,
          filter_length_,
          state_.size());
    }
  }

  /**
   * @brief Destructor for ResampleProcessorImpl.
   * @tparam T Data type of the samples.
   */
  template <typename T>
  ResampleProcessorImpl<T>::~ResampleProcessorImpl()
  {
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "Default::ResampleProcessorImpl<{}> destructed.", typeid(T).name());
    }
  }

  /**
   * @brief Designs the prototype FIR filter for resampling.
   * @details This method uses the `owner_backend_` to design the FIR filter
   * based on the `prototype_fir_design` specification stored within `spec_`.
   * The resulting coefficients are scaled by the interpolation factor.
   * @tparam T Data type of the samples.
   * @throws OmniException if the owner_backend_ is null, if filter design
   * fails, or if the design results in empty coefficients.
   * @throws std::logic_error if owner_backend_ is null (should be caught by
   * constructor).
   */
  template <typename T>
  void ResampleProcessorImpl<T>::design_filter()
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (!owner_backend_) {
      // This should ideally be caught by the constructor, but defensive check
      // here.
      if (logger)
        logger->critical(
            "Default::ResampleProcessorImpl::design_filter: owner_backend_ is "
            "null. This should not happen.");
      throw std::logic_error("Cannot design filter without owner backend.");
    }

    OmniExpected<Coefs::FIRFilter<T>> coeffs_expected;
    if constexpr (std::is_same_v<T, float>) {
      coeffs_expected
          = owner_backend_->design_fir_filter_f32(spec_.prototype_fir_design);
    }
    else if constexpr (std::is_same_v<T, double>) {
      coeffs_expected
          = owner_backend_->design_fir_filter_f64(spec_.prototype_fir_design);
    }
    else {
      // This path should be unreachable due to static_assert in
      // ResampleProcessorImpl declaration
      if (logger)
        logger->critical(
            "Default::ResampleProcessorImpl::design_filter: Unsupported data "
            "type T.");
      throw OmniException(
          "Unsupported type for filter design in resampler.",
          OmniStatus::UnsupportedFeature);
    }

    if (!coeffs_expected) {
      std::ostringstream msg_stream;
      msg_stream
          << "Default::ResampleProcessorImpl::design_filter: Failed to design "
          << "resampling prototype filter using owner backend. Status: "
          << coeffs_expected.error();  // Log OmniStatus directly
      if (logger) logger->error(msg_stream.str());
      throw OmniException(msg_stream.str(), coeffs_expected.error());
    }

    prototype_coeffs_ = std::move(coeffs_expected.value());

    if (prototype_coeffs_.empty()) {
      if (logger)
        logger->error(
            "Default::ResampleProcessorImpl::design_filter: Resampling "
            "prototype filter design resulted in empty coefficients.");
      throw OmniException(
          "Resampling prototype filter design resulted in empty coefficients.",
          OmniStatus::Failure);
    }

    filter_length_ = prototype_coeffs_.size();
    T scale_factor = static_cast<T>(interpolation_factor_);
    for (T& coeff : prototype_coeffs_) {
      coeff *= scale_factor;
    }
    if (logger && logger->should_log(spdlog::level::debug)) {
      logger->debug(
          "Prototype FIR filter designed for resampler. Taps: {}, First Coeff "
          "(scaled): {}",
          filter_length_,
          prototype_coeffs_.empty() ? T{0} : prototype_coeffs_[0]);
    }
  }

  /**
   * @brief Builds the polyphase filter bank from the prototype FIR filter
   * coefficients.
   * @details Decomposes the `prototype_coeffs_` into `polyphase_coeffs_`.
   * If the prototype filter is empty or factors are invalid, the polyphase bank
   * will be empty or zeroed.
   * @tparam T Data type of the samples.
   */
  template <typename T>
  void ResampleProcessorImpl<T>::build_polyphase_filters()
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (prototype_coeffs_.empty() || filter_length_ == 0
        || interpolation_factor_ == 0) {
      if (logger)
        logger->warn(
            "Default::ResampleProcessorImpl::build_polyphase_filters: "
            "Prototype "
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
        size_t proto_idx = phase + n * L;  // Corrected indexing: phase + n * L
                                           // for polyphase decomposition
        if (proto_idx < filter_length_) {
          // The polyphase components are typically indexed as P_i[k] = h[kL +
          // i] for type 1, or P_i[k] = h[kL - i + (L-1)] for type 2 (reversed).
          // For direct application as h[phase], h[phase+L], h[phase+2L]...
          // The original code seems to imply Type 1 decomposition where
          // coefficients are picked with a stride. Let's re-verify the standard
          // polyphase decomposition. If h(n) is the prototype filter, the k-th
          // polyphase component p_k(m) = h(mM+k) For upsampling by L, then
          // filtering, then downsampling by M: The filter h(n) is decomposed
          // into L polyphase filters. p_i(m) = h(mL + i) for i = 0, ..., L-1.
          // The current implementation seems to be: polyphase_coeffs_[phase][n]
          // = prototype_coeffs_[phase + n * L]; This means for phase 0: h[0],
          // h[L], h[2L], ... For phase 1: h[1], h[1+L], h[1+2L], ... This is a
          // valid way to form polyphase components.
          polyphase_coeffs_[phase][n] = prototype_coeffs_[proto_idx];
        }
      }
    }
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "Polyphase filter bank built. Branches: {}, Taps per branch: {}",
          polyphase_coeffs_.size(),
          num_branch_taps);
    }
  }

  /**
   * @brief Executes the resampling process on a block of input samples.
   * @tparam T Data type of the samples.
   * @param input A span of constant input samples.
   * @param output A span for the resampled output samples. Must be large
   * enough.
   * @return OmniStatus::Success if the operation is successful.
   * @return OmniStatus::NotInitialized if the processor was not properly
   * initialized.
   * @return OmniStatus::Failure for other internal errors.
   */
  template <typename T>
  OmniStatus ResampleProcessorImpl<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "Default::ResampleProcessorImpl::execute called. Input size: {}, "
          "Output span size: {}",
          input.size(),
          output.size());
    }

    if (interpolation_factor_ == 0 || decimation_factor_ == 0
        || polyphase_coeffs_.empty()
        || (polyphase_coeffs_.empty() ? true : polyphase_coeffs_[0].empty())) {
      if (logger)
        logger->warn(
            "Default::ResampleProcessorImpl::execute: Plan not properly "
            "initialized (L/M zero or no polyphase coeffs). L={}, M={}, "
            "PolyphaseBranches={}, BranchTaps={}",
            interpolation_factor_,
            decimation_factor_,
            polyphase_coeffs_.size(),
            polyphase_coeffs_.empty() ? 0 : polyphase_coeffs_[0].size());
      return OmniStatus::NotInitialized;
    }

    size_t input_len = input.size();
    size_t output_idx = 0;
    size_t max_output_can_produce = get_output_length(input_len);
    size_t output_samples_to_write
        = std::min(max_output_can_produce, output.size());

    if (input_len == 0 && state_.empty() && current_phase_ == 0) {
      if (output_samples_to_write
          > 0) {  // Only fill if there's space and we'd write
        std::fill(
            output.begin(), output.begin() + output_samples_to_write, T{0});
      }
      return OmniStatus::Success;
    }

    const size_t L = interpolation_factor_;
    const size_t M = decimation_factor_;
    const size_t state_len = state_.size();
    const size_t num_branch_taps = polyphase_coeffs_[0].size();

    std::vector<T>
        work_buffer;  // This buffer holds [previous state | new input]
    work_buffer.reserve(state_len + input_len);
    work_buffer.insert(work_buffer.end(), state_.begin(), state_.end());
    work_buffer.insert(work_buffer.end(), input.begin(), input.end());

    size_t work_buffer_consumed_idx
        = 0;  // Tracks how much of the work_buffer (conceptually input samples)
              // has been processed

    for (output_idx = 0; output_idx < output_samples_to_write; ++output_idx) {
      // Check if we have enough samples in the work_buffer for the current
      // filter branch The filter branch operates on a segment of the
      // (conceptually) upsampled-then-filtered signal. The
      // work_buffer_consumed_idx points to the start of the current input
      // segment needed.
      if (work_buffer_consumed_idx + num_branch_taps > work_buffer.size()) {
        if (logger && logger->should_log(spdlog::level::debug)) {
          logger->debug(
              "Resample execute: Not enough samples in work_buffer to produce "
              "output sample {}. Work buffer size: {}, consumed: {}, needed "
              "taps: {}",
              output_idx,
              work_buffer.size(),
              work_buffer_consumed_idx,
              num_branch_taps);
        }
        break;  // Stop if not enough data for a full filter operation
      }

      const auto& current_filter_branch = polyphase_coeffs_[current_phase_];
      T sum{};
      for (size_t n = 0; n < num_branch_taps; ++n) {
        sum += work_buffer[work_buffer_consumed_idx + n]
               * current_filter_branch[n];
      }
      output[output_idx] = sum;

      // Advance phase and input consumption
      current_phase_
          = (current_phase_ + M);  // Advance by decimation factor for selecting
                                   // next polyphase filter
      size_t num_input_samples_to_advance
          = current_phase_ / L;  // Integer division: how many full input
                                 // samples are "consumed"
      current_phase_ %= L;       // Wrap phase around
      work_buffer_consumed_idx += num_input_samples_to_advance;
    }

    // Update the state_ vector with the remaining unprocessed samples from
    // work_buffer
    if (state_len > 0) {
      if (work_buffer_consumed_idx >= work_buffer.size()) {
        // All data in work_buffer (including initial state) was consumed or not
        // enough for further processing
        std::fill(state_.begin(), state_.end(), T{0});
      }
      else {
        size_t remaining_in_work_buffer
            = work_buffer.size() - work_buffer_consumed_idx;
        size_t num_to_copy_to_state
            = std::min(state_len, remaining_in_work_buffer);
        std::copy(
            work_buffer.begin() + work_buffer_consumed_idx,
            work_buffer.begin() + work_buffer_consumed_idx
                + num_to_copy_to_state,
            state_.begin());
        if (num_to_copy_to_state < state_len) {
          // If less than state_len samples were copied, zero out the rest of
          // the state
          std::fill(state_.begin() + num_to_copy_to_state, state_.end(), T{0});
        }
      }
    }

    // Zero out any remaining part of the output buffer if not all samples were
    // written
    if (output_idx < output.size()) {
      std::fill(output.begin() + output_idx, output.end(), T{0});
    }
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "Default::ResampleProcessorImpl::execute finished. Output samples "
          "written: {}",
          output_idx);
    }
    return OmniStatus::Success;
  }

  /**
   * @brief Resets the internal state of the resampler.
   * @details Clears the internal delay lines (state) and resets the current
   * polyphase filter index.
   * @tparam T Data type of the samples.
   * @return OmniStatus::Success.
   */
  template <typename T>
  OmniStatus ResampleProcessorImpl<T>::reset()
  {
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::debug)) {
      logger->debug("Default::ResampleProcessorImpl::reset called.");
    }
    std::fill(state_.begin(), state_.end(), T{0});
    current_phase_ = 0;
    return OmniStatus::Success;
  }

  /**
   * @brief Gets the input sample rate of the resampler.
   * @tparam T Data type of the samples.
   * @return The input sample rate in Hz.
   */
  template <typename T>
  double ResampleProcessorImpl<T>::get_input_rate() const
  {
    return spec_.input_rate;
  }

  /**
   * @brief Gets the output sample rate of the resampler.
   * @tparam T Data type of the samples.
   * @return The output sample rate in Hz.
   */
  template <typename T>
  double ResampleProcessorImpl<T>::get_output_rate() const
  {
    return spec_.output_rate;
  }

  /**
   * @brief Calculates the exact number of output samples that will be produced
   * for a given number of input samples.
   * @details This calculation considers the current internal state (phase) of
   * the polyphase filter.
   * @tparam T Data type of the samples.
   * @param input_len The number of input samples.
   * @return The expected number of output samples. Returns 0 if rates or
   * factors are invalid, or if input_len is 0 and the resampler is in its
   * initial state.
   */
  template <typename T>
  size_t ResampleProcessorImpl<T>::get_output_length(size_t input_len) const
  {
    if (spec_.input_rate <= 0.0 || spec_.output_rate <= 0.0
        || interpolation_factor_ == 0 || decimation_factor_ == 0) {
      return 0;
    }
    // If no input and state is clear (initial call), no output.
    if (input_len == 0 && state_.empty() && current_phase_ == 0) return 0;

    // This formula calculates how many times the decimation step M "fits" into
    // the total number of "virtual" samples after upsampling and considering
    // the current phase. (current_phase_ + input_len * L) is the total number
    // of samples at the intermediate upsampled rate. Dividing by M gives the
    // number of output samples.
    return (current_phase_ + input_len * interpolation_factor_)
           / decimation_factor_;
  }

  // Explicit template instantiations
  template class ResampleProcessorImpl<float>;
  template class ResampleProcessorImpl<double>;

}  // namespace OmniDSP::Default
