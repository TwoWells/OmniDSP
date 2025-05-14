/**
 * @file resample_design.cpp
 * @brief Implements utility functions for creating Design::Resample.
 */
#include "resample_design.hpp"  // Internal declarations for this file

#include <cmath>      // For std::ceil, std::max, std::abs
#include <limits>     // For std::numeric_limits
#include <numeric>    // For std::gcd
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>
#include <utility>  // For std::move

#include "OmniDSP/core_types.hpp"         // For OmniExpected, Status
#include "OmniDSP/design/fir_filter.hpp"  // For Design::FIRFilter
#include "OmniDSP/params/fir_filter.hpp"  // For Params::FIRFilter
#include "OmniDSP/params/resample.hpp"
#include "OmniDSP/resample.hpp"      // For Design::Resample
#include "OmniDSP/types/filter.hpp"  // For FilterType, FIRFilterDesignMethod
#include "OmniDSP/utils.hpp"  // For the public declaration of Utils::create_spec
#include "OmniDSP/window.hpp"  // For WindowSetup, WindowType

// spdlog for logging
#include "spdlog/spdlog.h"

// Include the existing resample utilities for factor calculation
// This should point to the public utility for calculating L/M if it's in
// include/OmniDSP/utils/ or if it's an internal utility, its own header.
// Assuming calculate_resampling_factors is declared in
// "src/omnidsp/utils/resample.hpp" for now based on previous context.
#include "resample.hpp"  // For calculate_resampling_factors

namespace OmniDSP::Utils::Internal {
  // Definition of the helper function, moved from static in the previous
  // version
  size_t estimate_prototype_fir_order_for_resample(
      size_t L, size_t M, int quality, double normalized_cutoff)
  {
    double transition_width_factor;
    if (quality <= 5) {
      transition_width_factor = 0.4;
    }
    else if (quality <= 10) {
      transition_width_factor = 0.2;
    }
    else {
      transition_width_factor = 0.1;
    }

    double transition_width_hz_norm
        = normalized_cutoff * transition_width_factor;
    if (transition_width_hz_norm < 1e-6) transition_width_hz_norm = 1e-6;

    double stopband_attenuation_db
        = 40.0 + (static_cast<double>(quality) / 15.0) * 60.0;
    stopband_attenuation_db
        = std::max(21.0, std::min(120.0, stopband_attenuation_db));

    size_t order = static_cast<size_t>(std::ceil(
        (stopband_attenuation_db - 7.95)
        / (2.285 * (2.0 * transition_width_hz_norm))));

    size_t min_order_from_factors = 2 * std::max(L, M) * (quality / 5 + 1);
    order = std::max(order, min_order_from_factors);

    if (order < 16) order = 16;
    if (order > 4096) order = 4096;

    if (order % 2 != 0) {
      order++;
    }
    return order;
  }
}  // namespace OmniDSP::Utils::Internal

namespace OmniDSP::Utils {

  OmniExpected<Design::Resample> create_spec(const Params::Resample& params)
  {
    // Params::Resample constructor already performed initial validation.
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    size_t L, M;
    // Assuming calculate_resampling_factors is in OmniDSP::Utils namespace
    // and declared in "src/omnidsp/utils/resample.hpp" (which might need
    // adjustment if it's a public util)
    Status factor_status = calculate_resampling_factors(
        params.input_rate_, params.output_rate_, L, M);
    if (factor_status != Status::Success) {
      logger->error(
          "Failed to calculate resampling factors L/M for IR={}, OR={}. "
          "Status: {}",
          params.input_rate_,
          params.output_rate_,
          static_cast<int>(factor_status));
      return std::unexpected(factor_status);
    }
    if (L == 0 || M == 0) {
      logger->error(
          "Calculated resampling factors L={} or M={} is zero, which is "
          "invalid.",
          L,
          M);
      return std::unexpected(Status::Failure);
    }

    logger->debug(
        "Calculated resampling factors: L={}, M={} for IR={}, OR={}",
        L,
        M,
        params.input_rate_,
        params.output_rate_);

    double normalized_prototype_cutoff
        = 1.0 / (2.0 * static_cast<double>(std::max(L, M)));

    // Call the helper from the Internal namespace
    size_t prototype_order
        = Internal::estimate_prototype_fir_order_for_resample(
            L, M, params.quality_, normalized_prototype_cutoff);
    logger->debug(
        "Estimated prototype FIR filter order: {} for L={}, M={}, Quality={}",
        prototype_order,
        L,
        M,
        params.quality_);

    double prototype_design_sample_rate = 2.0;

    Params::FIRFilter prototype_fir_params(
        FilterType::Lowpass,
        prototype_design_sample_rate,
        normalized_prototype_cutoff,
        std::nullopt,
        prototype_order,
        std::nullopt,
        std::nullopt,
        params.window_setup_,
        FIRFilterDesignMethod::WindowSinc);

    // Call the public Utils::create_spec for Params::FIRFilter
    OmniExpected<Design::FIRFilter> prototype_fir_spec_expected
        = create_spec(prototype_fir_params);

    if (!prototype_fir_spec_expected) {
      logger->error(
          "Failed to create Design::FIRFilter for resampling prototype filter. "
          "Error: {}",
          static_cast<int>(prototype_fir_spec_expected.error()));
      return std::unexpected(prototype_fir_spec_expected.error());
    }

    Design::FIRFilter final_prototype_fir_spec
        = std::move(prototype_fir_spec_expected.value());

    try {
      Design::Resample spec(
          params.input_rate_,
          params.output_rate_,
          params.quality_,
          L,
          M,
          std::move(final_prototype_fir_spec));
      return spec;
    }
    catch (const std::exception& e) {
      logger->error(
          "Exception during Design::Resample construction: {}", e.what());
      return std::unexpected(Status::Failure);
    }
  }

}  // namespace OmniDSP::Utils
