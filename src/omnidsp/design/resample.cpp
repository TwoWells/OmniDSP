/**
 * @file resample.cpp
 * @brief Implements utility functions for creating Design::Resample and
 * calculating resampling factors.
 */
#include "OmniDSP/design/resample.hpp"  // Corresponding header for Design::Resample related functions

#include <cmath>      // For std::ceil, std::max, std::abs, std::floor
#include <limits>     // For std::numeric_limits
#include <numeric>    // For std::gcd
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>
#include <utility>  // For std::move, std::optional, std::unexpected

#include "OmniDSP/core_types.hpp"  // For OmniExpected, Status, OmniStatus
#include "OmniDSP/design/fir_filter.hpp"  // For Design::FIRFilter
#include "OmniDSP/params/fir_filter.hpp"  // For Params::FIRFilter
#include "OmniDSP/params/resample.hpp"    // For Params::Resample
// #include "OmniDSP/design/resample.hpp" // Already included via corresponding
// header
#include "OmniDSP/design.hpp"  // For OmniDSP::Design::create (for FIRFilter)
#include "OmniDSP/types/filter.hpp"  // For FilterType, FIRFilterDesignMethod
#include "OmniDSP/window.hpp"        // For WindowSetup, WindowType

// spdlog for logging
#include <spdlog/fmt/ostr.h>  // Include for custom ostream operator support with spdlog

#include "spdlog/spdlog.h"

/**
 * @namespace OmniDSP::Design
 * @brief Contains structures and functions related to the design phase of DSP
 * components, translating high-level parameters into concrete specifications.
 */
namespace OmniDSP::Design {

  /**
   * @namespace OmniDSP::Design::Internal
   * @brief Contains internal helper functions for the design module, not part
   * of the public API.
   */
  namespace Internal {
    /**
     * @brief Estimates the FIR filter order for a resampling prototype filter.
     * @details This internal helper function uses heuristics and quality
     * parameters to determine a suitable order for the
     * anti-aliasing/anti-imaging filter used in resampling.
     * @param L The upsampling factor.
     * @param M The downsampling factor.
     * @param quality The quality parameter from Params::Resample (0-15).
     * @param normalized_cutoff The normalized cutoff frequency for the
     * prototype filter.
     * @return Estimated filter order (an even number, typically).
     */
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
      if (order > 4096) order = 4096;  // Cap max order

      if (order % 2 != 0) {  // Ensure even order for Type I FIR
        order++;
      }
      return order;
    }
  }  // namespace Internal

  /**
   * @brief Calculates the rational resampling factors L (upsampling) and M
   * (downsampling).
   * @param in_rate The input sample rate in Hz.
   * @param out_rate The desired output sample rate in Hz.
   * @param L Output parameter for the upsampling factor.
   * @param M Output parameter for the downsampling factor.
   * @return OmniStatus::Success if factors are calculated successfully.
   * @return OmniStatus::InvalidArgument if input/output rates are not positive.
   * @return OmniStatus::Failure if factors cannot be determined or overflow.
   */
  OmniStatus calculate_resampling_factors(
      double in_rate, double out_rate, size_t& L, size_t& M)
  {
    if (in_rate <= 0.0 || out_rate <= 0.0) {
      return OmniStatus::InvalidArgument;
    }

    const long long factor_base
        = 16777216LL;  // 2^24, a common base for rational approximation

    if (out_rate > static_cast<double>(
            std::numeric_limits<long long>::max() / factor_base)) {
      // Avoid overflow in num_approx calculation
      return OmniStatus::Failure;
    }

    long long num_approx = static_cast<long long>(
        std::floor((out_rate / in_rate) * factor_base + 0.5));

    if (num_approx == 0 && out_rate > 0)
      num_approx = 1;  // Ensure L is at least 1 if out_rate > 0
    if (num_approx < 0)
      return OmniStatus::Failure;  // Should not happen with positive rates

    long long common = std::gcd(num_approx, factor_base);

    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (common == 0) {
      if (logger)
        logger->error(
            "GCD common factor is zero in calculate_resampling_factors. "
            "num_approx={}, factor_base={}",
            num_approx,
            factor_base);
      return OmniStatus::Failure;
    }
    if (common < 0) {  // std::gcd should return non-negative
      common = -common;
    }

    long long l_long = num_approx / common;
    long long m_long = factor_base / common;

    if (l_long > static_cast<long long>(std::numeric_limits<size_t>::max())
        || m_long
               > static_cast<long long>(std::numeric_limits<size_t>::max())) {
      if (logger) logger->error("Resampling factors L/M overflow size_t.");
      return OmniStatus::Failure;  // Overflow
    }

    L = static_cast<size_t>(l_long);
    M = static_cast<size_t>(m_long);

    if (L == 0 || M == 0) {  // Should not happen if common > 0 and rates > 0
      if (out_rate > 0 && L == 0)
        L = 1;  // Ensure L is at least 1 if out_rate > 0
      if (in_rate > 0 && M == 0)
        M = 1;  // Ensure M is at least 1 if in_rate > 0
      if (L == 0 || M == 0) {
        if (logger)
          logger->error(
              "Calculated L ({}) or M ({}) is zero after adjustment.", L, M);
        return OmniStatus::Failure;
      }
    }

    return OmniStatus::Success;
  }

  /**
   * @brief Creates a fully resolved resampling design specification from input
   * parameters.
   * @details This function determines the rational resampling factors (L and
   * M), estimates the order for the prototype FIR filter based on quality
   * settings, and then creates the design specification for this prototype
   * filter.
   * @param params The input parameters for the resampling operation
   * (OmniDSP::Params::Resample).
   * @return An OmniExpected<Design::Resample> containing the resolved design
   * specification on success, or an OmniStatus error code on failure.
   * @throws No direct exceptions, errors are returned via OmniExpected.
   */
  OmniExpected<Resample> create(const Params::Resample& params)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "Attempting to create Design::Resample from Params: {}", params);
    }

    size_t L, M;
    OmniStatus factor_status = calculate_resampling_factors(
        params.input_rate_, params.output_rate_, L, M);

    if (factor_status != OmniStatus::Success) {
      if (logger)
        logger->error(
            "Failed to calculate resampling factors L/M for IR={}, OR={}. "
            "Status: {}",
            params.input_rate_,
            params.output_rate_,
            factor_status);  // Log OmniStatus directly
      return std::unexpected(factor_status);
    }
    if (L == 0 || M == 0) {  // This check might be redundant if
                             // calculate_resampling_factors is robust
      if (logger)
        logger->error(
            "Calculated resampling factors L={} or M={} is zero, which is "
            "invalid.",
            L,
            M);
      return std::unexpected(OmniStatus::Failure);
    }

    if (logger)
      logger->debug(
          "Calculated resampling factors: L={}, M={} for IR={}, OR={}",
          L,
          M,
          params.input_rate_,
          params.output_rate_);

    double normalized_prototype_cutoff
        = 1.0 / (2.0 * static_cast<double>(std::max(L, M)));

    size_t prototype_order
        = Internal::estimate_prototype_fir_order_for_resample(
            L, M, params.quality_, normalized_prototype_cutoff);
    if (logger)
      logger->debug(
          "Estimated prototype FIR filter order: {} for L={}, M={}, Quality={}",
          prototype_order,
          L,
          M,
          params.quality_);

    double prototype_design_sample_rate = 2.0;  // Normalized design rate

    Params::FIRFilter prototype_fir_params(
        FilterType::Lowpass,
        prototype_design_sample_rate,
        normalized_prototype_cutoff,
        std::nullopt,  // cutoff2_hz
        prototype_order,
        std::nullopt,  // transition_width_hz
        std::nullopt,  // stopband_attenuation_db (implicitly handled by quality
                       // in estimate_prototype_fir_order_for_resample)
        params.window_setup_,  // Pass user's window choice
        FIRFilterDesignMethod::WindowSinc);

    OmniExpected<Design::FIRFilter> prototype_fir_spec_expected
        = Design::create(
            prototype_fir_params);  // Call Design::create for FIRFilter

    if (!prototype_fir_spec_expected) {
      if (logger)
        logger->error(
            "Failed to create Design::FIRFilter for resampling prototype "
            "filter. Error: {}",
            prototype_fir_spec_expected.error());  // Log OmniStatus
      return std::unexpected(prototype_fir_spec_expected.error());
    }

    Design::FIRFilter final_prototype_fir_spec
        = std::move(prototype_fir_spec_expected.value());

    try {
      Resample spec(
          params.input_rate_,
          params.output_rate_,
          params.quality_,
          L,
          M,
          std::move(final_prototype_fir_spec));

      if (logger && logger->should_log(spdlog::level::debug)) {
        logger->debug("Successfully created Design::Resample: {}", spec);
      }
      return spec;
    }
    catch (const std::exception& e) {
      if (logger)
        logger->error(
            "Exception during Design::Resample construction: {}", e.what());
      return std::unexpected(OmniStatus::Failure);
    }
  }

}  // namespace OmniDSP::Design
