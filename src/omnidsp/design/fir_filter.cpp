/**
 * @file fir_filter.cpp
 * @brief Implements utility functions for common filter design tasks,
 * including FIR filter specification creation.
 */
#include "OmniDSP/params/fir_filter.hpp"  // For Params::FIRFilter

#include <algorithm>  // For std::max, std::min
#include <cmath>      // For std::ceil, std::log10, std::abs, std::round
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>     // For error messages
#include <vector>

#include "OmniDSP/core_types.hpp"  // For OmniExpected, Status
#include "OmniDSP/design.hpp"      // For the declaration of Design::create
#include "OmniDSP/window.hpp"      // For WindowSetup

// spdlog for logging
#include <spdlog/fmt/ostr.h>  // Include for custom ostream operator support with spdlog

#include "spdlog/spdlog.h"

// It's good practice to include the specific design type header
// if it's not fully covered by core_types or design.hpp transitively
// for the Design::FIRFilter struct definition.
#include "OmniDSP/design/fir_filter.hpp"  // For Design::FIRFilter struct

/**
 * @namespace OmniDSP::Design
 * @brief Contains structures and functions related to the design phase of DSP
 * components, translating high-level parameters into concrete specifications.
 */
namespace OmniDSP::Design {

  // Helper function to estimate FIR filter order (Kaiser's formula or similar)
  // This can be a static helper within this .cpp file or a free function in an
  // internal utils namespace if used elsewhere. For now, keep it static here.
  static size_t estimate_fir_order_from_specs(
      double sample_rate,
      double transition_width_hz,
      double stopband_attenuation_db)
  {
    if (sample_rate <= 0 || transition_width_hz <= 0) {
      if (transition_width_hz <= 0) transition_width_hz = 0.001 * sample_rate;
    }
    if (stopband_attenuation_db <= 0) stopband_attenuation_db = 21.0;

    double atten_db = std::max(21.0, stopband_attenuation_db);

    double normalized_transition_width = transition_width_hz / sample_rate;
    if (normalized_transition_width <= 0.0) {
      normalized_transition_width
          = 0.01;  // Fallback to prevent division by zero
    }

    double order_double
        = (atten_db - 7.95) / (2.285 * normalized_transition_width);
    size_t order = static_cast<size_t>(std::ceil(order_double));

    if (order % 2 != 0) {
      order++;
    }
    order = std::max(static_cast<size_t>(4), order);
    return order;
  }

  /**
   * @brief Creates a fully resolved FIR filter design specification from input
   * parameters.
   * @details This function determines the filter order if not explicitly
   * provided, based on transition width and stopband attenuation. It then
   * constructs a Design::FIRFilter object.
   * @param params The input parameters for the FIR filter design
   * (OmniDSP::Params::FIRFilter).
   * @return An OmniExpected<Design::FIRFilter> containing the resolved design
   * specification on success, or an OmniStatus error code on failure.
   * @throws No direct exceptions, errors are returned via OmniExpected.
   */
  OmniExpected<Design::FIRFilter> create(const Params::FIRFilter& params)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "Attempting to create Design::FIRFilter from Params: {}", params);
    }

    size_t final_order;

    if (params.order_.has_value()) {
      final_order = params.order_.value();
      if (logger)
        logger->debug("Using user-specified FIR order: {}", final_order);
    }
    else if (
        params.transition_width_.has_value()
        && params.stopband_attenuation_db_.has_value()) {
      if (params.sample_rate_ <= 0.0) {
        if (logger)
          logger->error("Sample rate must be positive to estimate FIR order.");
        return std::unexpected(OmniStatus::InvalidArgument);
      }
      final_order = estimate_fir_order_from_specs(
          params.sample_rate_,
          params.transition_width_.value(),
          params.stopband_attenuation_db_.value());
      if (logger)
        logger->debug(
            "Estimated FIR order: {} from characteristics.", final_order);
    }
    else {
      if (logger)
        logger->error(
            "Params::FIRFilter has insufficient information to determine "
            "filter order (missing order or characteristics).");
      return std::unexpected(OmniStatus::InvalidArgument);
    }

    if (final_order == 0) {
      if (logger) logger->error("FIR filter order cannot be zero.");
      return std::unexpected(OmniStatus::InvalidArgument);
    }
    if (final_order > 10000
        && params.design_method_ == FIRFilterDesignMethod::WindowSinc) {
      if (logger)
        logger->warn(
            "Calculated FIR filter order ({}) is very high. This may lead to "
            "long computation times or memory issues.",
            final_order);
    }

    WindowSetup final_window_setup = params.window_setup_;
    final_window_setup.length = static_cast<int>(final_order + 1);

    try {
      WindowSetup validated_window_setup(
          final_window_setup.type,
          final_window_setup.length,
          final_window_setup.params);
      final_window_setup = validated_window_setup;
    }
    catch (const std::invalid_argument& e) {
      if (logger)
        logger->error(
            "Failed to create a valid WindowSetup for Design::FIRFilter: {}. "
            "Window type: {}, Length: {}",
            e.what(),
            static_cast<int>(final_window_setup.type),
            final_window_setup.length);
      return std::unexpected(OmniStatus::InvalidArgument);
    }

    try {
      Design::FIRFilter spec(
          params.filter_type_,
          final_order,
          params.sample_rate_,
          params.cutoff1_,
          params.cutoff2_,
          final_window_setup);

      if (!spec.validate_consistency()) {
        if (logger)
          logger->error(
              "Internal consistency validation failed for created "
              "Design::FIRFilter.");
        return std::unexpected(OmniStatus::Failure);
      }
      if (logger
          && logger->should_log(
              spdlog::level::debug)) {  // Changed to debug level for successful
                                        // creation
        logger->debug("Successfully created Design::FIRFilter: {}", spec);
      }
      return spec;
    }
    catch (const std::invalid_argument& e) {
      if (logger)
        logger->error(
            "Invalid argument during Design::FIRFilter construction: {}",
            e.what());
      return std::unexpected(OmniStatus::InvalidArgument);
    }
    catch (const std::exception& e) {
      if (logger)
        logger->error(
            "Exception during Design::FIRFilter construction: {}", e.what());
      return std::unexpected(OmniStatus::Failure);
    }
  }

}  // namespace OmniDSP::Design
