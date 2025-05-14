/**
 * @file filter_design.cpp
 * @brief Implements utility functions for common filter design tasks,
 * including FIR filter specification creation.
 */
#include "filter_design.hpp"  // May contain other design utilities, or can be more specific

#include <algorithm>  // For std::max, std::min
#include <cmath>      // For std::ceil, std::log10, std::abs, std::round
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>     // For error messages
#include <vector>

#include "OmniDSP/core_types.hpp"         // For OmniExpected, Status
#include "OmniDSP/design/fir_filter.hpp"  // For Design::FIRFilter, FilterType
#include "OmniDSP/params/fir_filter.hpp"
#include "OmniDSP/types/filter.hpp"  // For Design::FIRFilter, FilterType
#include "OmniDSP/utils.hpp"   // For the declaration of Utils::create_spec
#include "OmniDSP/window.hpp"  // For WindowSetup, WindowType, WindowParams

// spdlog for logging, typically included in .cpp files
#include "spdlog/spdlog.h"
// Make sure spdlog is properly linked or made available (e.g., header-only
// version)

namespace OmniDSP::Utils {

  // Helper function to estimate FIR filter order (Kaiser's formula or similar)
  // This can be a static helper within this .cpp file or a free function in an
  // internal utils namespace.
  static size_t estimate_fir_order_from_specs(
      double sample_rate,
      double transition_width_hz,
      double stopband_attenuation_db)
  {
    if (sample_rate <= 0 || transition_width_hz <= 0
        || stopband_attenuation_db <= 0) {
      // spdlog::get("OmniDSP")->warn("Invalid inputs to
      // estimate_fir_order_from_specs."); Return a default or indicate error.
      // For now, let's assume valid inputs based on Params::FIRFilter
      // validation. Or, throw std::invalid_argument here. For simplicity, let's
      // assume Params::FIRFilter validation caught basic errors. A more robust
      // implementation would handle this.
      if (transition_width_hz <= 0)
        transition_width_hz = 0.01 * sample_rate;  // Avoid division by zero
    }

    // Ensure attenuation is at least 21dB for the formula
    double atten_db = std::max(21.0, stopband_attenuation_db);

    // Kaiser's formula for filter order N:
    // N approx = (Atten_dB - 7.95) / (2.285 *
    // (TransitionWidth_normalized_by_Fs)) TransitionWidth_normalized_by_Fs =
    // transition_width_hz / sample_rate
    double normalized_transition_width = transition_width_hz / sample_rate;
    if (normalized_transition_width
        <= 0.0) {  // Should be caught by checks above
      normalized_transition_width
          = 0.01;  // Fallback to prevent division by zero
    }

    double order_double
        = (atten_db - 7.95) / (2.285 * normalized_transition_width);
    size_t order = static_cast<size_t>(std::ceil(order_double));

    // Ensure order is odd for Type I (symmetric, even length taps = order + 1)
    // or even for Type II (symmetric, odd length taps = order + 1)
    // Most windowed sinc designs aim for Type I (linear phase, non-zero DC gain
    // for LP) which means N_taps = order + 1 should be odd, so order should be
    // even.
    if (order % 2 != 0) {
      order++;
    }
    // Ensure a minimum practical order
    if (order < 4) order = 4;  // Arbitrary minimum, adjust as needed
    // spdlog::get("OmniDSP")->debug("Estimated FIR order: {} from TW={}Hz,
    // Atten={}dB, SR={}Hz", order, transition_width_hz,
    // stopband_attenuation_db, sample_rate);
    return order;
  }

  OmniExpected<Design::FIRFilter> create_spec(const Params::FIRFilter& params)
  {
    // Params::FIRFilter constructor already performed initial validation.
    // Here, we resolve any estimations (like order) and finalize the Design.

    auto logger = spdlog::get("OmniDSP");
    if (!logger) {  // Fallback if named logger isn't registered
      logger = spdlog::default_logger();  // Or handle error
    }

    size_t final_order;

    if (params.order_.has_value()) {
      final_order = params.order_.value();
      // If order is given, characteristics might be for reference or ignored
      // for order calc. We could add a warning if characteristics are also
      // given but order is taking precedence.
      logger->debug("Using user-specified FIR order: {}", final_order);
    }
    else if (
        params.transition_width_.has_value()
        && params.stopband_attenuation_db_.has_value()) {
      // Estimate order based on characteristics
      if (params.sample_rate_
          <= 0.0) {  // Should have been caught by Params constructor
        return std::unexpected(OmniStatus::InvalidArgument);
      }
      final_order = estimate_fir_order_from_specs(
          params.sample_rate_,
          params.transition_width_.value(),
          params.stopband_attenuation_db_.value());
      logger->debug(
          "Estimated FIR order: {} from characteristics.", final_order);
    }
    else {
      // This case should have been caught by Params::FIRFilter constructor
      // validation
      logger->error(
          "Params::FIRFilter has insufficient information to determine filter "
          "order.");
      return std::unexpected(OmniStatus::InvalidArgument);
    }

    if (final_order > 10000
        && params.design_method_
               == FIRFilterDesignMethod::WindowSinc) {  // Arbitrary sanity
                                                        // check limit
      logger->warn(
          "Calculated FIR filter order ({}) is very high. This may lead to "
          "long computation times or memory issues.",
          final_order);
    }

    // The WindowSetup from params already has the type and its specific
    // parameters (e.g., beta). We now set the definitive length for the window
    // based on the final_order.
    WindowSetup final_window_setup
        = params.window_setup_;  // Copy type and params
    final_window_setup.length
        = static_cast<int>(final_order + 1);  // Set definitive length

    // Validate the newly constructed WindowSetup with its length.
    // The WindowSetup constructor itself should validate its parameters.
    try {
      WindowSetup validated_final_window_setup(
          final_window_setup.type,
          final_window_setup.length,
          final_window_setup.params  // Pass the optional params
      );
      // If it constructs without throwing, it's valid.
      final_window_setup = validated_final_window_setup;
    }
    catch (const std::invalid_argument& e) {
      logger->error(
          "Failed to create a valid WindowSetup for Design::FIRFilter: {}. "
          "Window "
          "type: {}, Length: {}, Params given: {}",
          e.what(),
          static_cast<int>(final_window_setup.type),
          final_window_setup.length,
          final_window_setup.params.has_value());
      return std::unexpected(OmniStatus::InvalidArgument);
    }

    // Construct the Design::FIRFilter
    try {
      Design::FIRFilter spec(
          params.filter_type_,
          final_order,
          params.sample_rate_,
          params.cutoff1_,
          params.cutoff2_,  // std::optional is moved/copied
          final_window_setup);
      // Perform a final consistency check on the created spec if desired
      if (!spec.validate_consistency()) {
        logger->error(
            "Internal consistency validation failed for created "
            "Design::FIRFilter.");
        return std::unexpected(OmniStatus::Failure);  // Internal error
      }
      return spec;
    }
    catch (const std::exception&
               e) {  // Catch potential errors from Design::FIRFilter
                     // constructor (e.g. asserts if enabled)
      logger->error(
          "Exception during Design::FIRFilter construction: {}", e.what());
      return std::unexpected(OmniStatus::Failure);
    }
  }

  // Explicit template instantiations for design_resampling_prototype_filter
  // (if they were in this file, but they are in their own filter_design.cpp
  // from utils) This file is now focused on Utils::create_spec for FIR.

}  // namespace OmniDSP::Utils
