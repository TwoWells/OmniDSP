/**
 * @file iir_filter.cpp (Default)
 * @brief Implements Default backend IIRFilterProcessorImpl and IIR design
 * helper.
 */

#include "iir_filter.hpp"  // Corresponding header

#include <OmniDSP/coefs/iir_filter.hpp>  // Public Design::IIRFilter, Coefs::SOS
#include <OmniDSP/core_types.hpp>        // Core types, Status, OmniExpected
#include <OmniDSP/design/iir_filter.hpp>  // Public Design::IIRFilter, Coefs::SOS
#include <OmniDSP/types/filter.hpp>       // For FilterType enum
#include <algorithm>                      // For std::fill, std::min
#include <cmath>                          // For std::abs
#include <complex>
#include <expected>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

#include "spdlog/spdlog.h"

namespace OmniDSP::Default {

  //--------------------------------------------------------------------------
  // IIR Filter Design Implementation (Placeholder)
  //--------------------------------------------------------------------------
  [[nodiscard]] OmniExpected<Coefs::IIRFilterSOS> generate_iir_filter_coeffs(
      const Design::IIRFilter& spec)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (!spec.validate_consistency()) {
      logger->error(
          "Invalid Design::IIRFilter provided to "
          "Default::generate_iir_filter_coeffs.");
      return std::unexpected(Status::InvalidArgument);
    }
    logger->warn(
        "Default IIR filter design not yet implemented. Returning "
        "NotImplemented.");
    return std::unexpected(Status::NotImplemented);
  }

  // --- IIRFilterProcessorImpl ---
  template <typename T>
  IIRFilterProcessorImpl<T>::IIRFilterProcessorImpl(
      const Coefs::IIRFilterSOS& sos_coefficients)
  {
    if (sos_coefficients.empty()) {
      auto logger = spdlog::get("OmniDSP");
      if (!logger) logger = spdlog::default_logger();
      logger->error(
          "Default::IIRFilterProcessorImpl: SOS coefficients cannot be empty.");
      throw std::invalid_argument(
          "IIR SOS coefficients cannot be empty for IIRFilterProcessorImpl.");
    }
    internal_coeffs_.reserve(sos_coefficients.size());
    for (const auto& sos_double : sos_coefficients) {
      InternalSOS<T> sos_t;
      sos_t.b0 = static_cast<T>(sos_double.b0);
      sos_t.b1 = static_cast<T>(sos_double.b1);
      sos_t.b2 = static_cast<T>(sos_double.b2);
      if (std::abs(sos_double.a0 - 1.0) > 1e-9) {
        auto logger = spdlog::get("OmniDSP");
        if (!logger) logger = spdlog::default_logger();
        logger->warn(
            "Default IIR SOS coefficient a0 is {} (not 1.0) for section. "
            "Coefficients should be pre-normalized if a0=1 is assumed by the "
            "Direct Form II Transposed structure.",
            sos_double.a0);
      }
      sos_t.a1 = static_cast<T>(sos_double.a1);
      sos_t.a2 = static_cast<T>(sos_double.a2);
      internal_coeffs_.push_back(sos_t);
    }
    state_.assign(internal_coeffs_.size() * 2, T{0});
  }

  template <typename T>
  IIRFilterProcessorImpl<T>::~IIRFilterProcessorImpl() = default;

  template <typename T>
  Status IIRFilterProcessorImpl<T>::execute(
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
      if (!output.empty()) {
        std::fill(
            output.begin(),
            output.begin() + std::min(input.size(), output.size()),
            T{0});
      }
      return Status::Success;
    }

    size_t num_sections = internal_coeffs_.size();
    for (size_t n = 0; n < input.size(); ++n) {
      T section_input = input[n];
      T current_sample_output = T{0};

      for (size_t k = 0; k < num_sections; ++k) {
        const auto& sos = internal_coeffs_[k];
        T& w_n_minus_1 = state_[k * 2 + 0];
        T& w_n_minus_2 = state_[k * 2 + 1];

        current_sample_output = sos.b0 * section_input + w_n_minus_1;

        T s1_k_new = sos.b1 * section_input - sos.a1 * current_sample_output
                     + w_n_minus_2;
        T s2_k_new = sos.b2 * section_input - sos.a2 * current_sample_output;

        w_n_minus_1 = s1_k_new;
        w_n_minus_2 = s2_k_new;

        section_input = current_sample_output;
      }
      output[n] = current_sample_output;
    }

    if (output.size() > input.size()) {
      std::fill(output.begin() + input.size(), output.end(), T{0});
    }
    return Status::Success;
  }

  template <typename T>
  Status IIRFilterProcessorImpl<T>::reset()
  {
    std::fill(state_.begin(), state_.end(), T{0});
    return Status::Success;
  }

  template <typename T>
  size_t IIRFilterProcessorImpl<T>::get_order() const
  {
    return internal_coeffs_.empty() ? 0 : internal_coeffs_.size() * 2;
  }

  template <typename T>
  size_t IIRFilterProcessorImpl<T>::get_num_sections() const
  {
    return internal_coeffs_.size();
  }

  // Explicit template instantiations for IIRFilterProcessorImpl
  template class IIRFilterProcessorImpl<F32>;
  template class IIRFilterProcessorImpl<F64>;

}  // namespace OmniDSP::Default
