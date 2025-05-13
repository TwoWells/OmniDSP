/**
 * @file resample.cpp (Default)
 * @brief Implements the Default backend ResamplePlanImpl class using standard
 * C++.
 */

#include "resample.hpp"  // Corresponding header for Default::ResamplePlanImpl

#include <OmniDSP/coefs/fir_filter.hpp>  // For FIRCoefs
#include <OmniDSP/core_types.hpp>
#include <OmniDSP/filter.hpp>  // For Design::FIRFilter (part of Design::Resample)
#include <OmniDSP/resample.hpp>  // For Design::Resample definition
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "interface/backend.hpp"  // For Abstract::Backend base class
#include "spdlog/spdlog.h"

namespace OmniDSP::Default {

  //--------------------------------------------------------------------------
  // ResamplePlanImpl Method Implementations
  //--------------------------------------------------------------------------

  template <typename T>
  ResamplePlanImpl<T>::ResamplePlanImpl(  // TODO: Rename to
                                          // ResampleProcessorImpl
      const Abstract::Backend* owner,
      const Design::Resample& design_spec)
      : owner_backend_(owner),
        spec_(design_spec),
        interpolation_factor_(design_spec.up_factor_L),
        decimation_factor_(design_spec.down_factor_M),
        prototype_coeffs_(),
        filter_length_(0),
        current_phase_(0)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (!owner_backend_) {
      logger->error("Default::ResamplePlanImpl: owner_backend is null.");
      throw OmniException(
          "ResamplePlanImpl requires a valid owner AbstractBackend pointer.",
          Status::InvalidArgument);
    }

    if (interpolation_factor_ == 0 || decimation_factor_ == 0) {
      logger->error(
          "Default::ResamplePlanImpl: L ({}) or M ({}) from design is zero.",
          interpolation_factor_,
          decimation_factor_);
      throw OmniException(
          "Resampling factors L or M from design cannot be zero.",
          Status::InvalidArgument);
    }

    design_filter();

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

    build_polyphase_filters();

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

    if (!owner_backend_) {
      logger->critical(
          "Default::ResamplePlanImpl::design_filter: owner_backend_ is null. "
          "This should not happen.");
      throw std::logic_error("Cannot design filter without owner backend.");
    }

    OmniExpected<FIRCoefs<T>> coeffs_expected;
    if constexpr (std::is_same_v<T, float>) {
      coeffs_expected = owner_backend_->design_fir_filter_f32(
          spec_.prototype_fir_design);  // CHANGED HERE
    }
    else if constexpr (std::is_same_v<T, double>) {
      coeffs_expected = owner_backend_->design_fir_filter_f64(
          spec_.prototype_fir_design);  // CHANGED HERE
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
      throw OmniException(
          "Failed to design resampling prototype filter. Status: "
              + std::string(get_status_string(coeffs_expected.error())),
          coeffs_expected.error());
    }

    prototype_coeffs_ = std::move(coeffs_expected.value());

    if (prototype_coeffs_.empty()) {
      logger->error(
          "Default::ResamplePlanImpl::design_filter: Resampling prototype "
          "filter design resulted in empty coefficients.");
      throw OmniException(
          "Resampling prototype filter design resulted in empty coefficients.",
          Status::Failure);
    }

    filter_length_ = prototype_coeffs_.size();
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

    if (input_len == 0 && state_.empty() && current_phase_ == 0) {
      if (output_samples_to_write > 0) {
        std::fill(
            output.begin(), output.begin() + output_samples_to_write, T{0});
      }
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

    size_t work_buffer_consumed_idx = 0;

    for (output_idx = 0; output_idx < output_samples_to_write; ++output_idx) {
      if (work_buffer_consumed_idx + num_branch_taps > work_buffer.size()) {
        break;
      }

      const auto& current_filter_branch = polyphase_coeffs_[current_phase_];
      T sum{};
      for (size_t n = 0; n < num_branch_taps; ++n) {
        sum += work_buffer[work_buffer_consumed_idx + n]
               * current_filter_branch[n];
      }
      output[output_idx] = sum;

      current_phase_ = (current_phase_ + M);
      size_t num_input_samples_to_advance = current_phase_ / L;
      current_phase_ %= L;
      work_buffer_consumed_idx += num_input_samples_to_advance;
    }

    if (state_len > 0) {
      if (work_buffer_consumed_idx >= work_buffer.size()) {
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
          std::fill(state_.begin() + num_to_copy_to_state, state_.end(), T{0});
        }
      }
    }

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
    if (spec_.input_rate <= 0.0 || spec_.output_rate <= 0.0
        || interpolation_factor_ == 0 || decimation_factor_ == 0) {
      return 0;
    }
    if (input_len == 0 && state_.empty() && current_phase_ == 0) return 0;
    return (current_phase_ + input_len * interpolation_factor_)
           / decimation_factor_;
  }

  template class ResamplePlanImpl<float>;
  template class ResamplePlanImpl<double>;

}  // namespace OmniDSP::Default
