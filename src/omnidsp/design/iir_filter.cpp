/**
 * @file iir_filter.cpp
 * @brief Implements utility functions for creating Design::IIRFilter from
 * Params::IIRFilter.
 */

#include "OmniDSP/params/iir_filter.hpp"  // For Params::IIRFilter

#include <spdlog/fmt/ostr.h>  // Include for custom ostream operator support with spdlog
#include <spdlog/spdlog.h>

#include <stdexcept>  // For std::invalid_argument and std::exception

#include "OmniDSP/core_types.hpp"  // For OmniExpected, Status
#include "OmniDSP/design.hpp"  // For the public declaration of Design::create
#include "OmniDSP/design/iir_filter.hpp"  // For Design::IIRFilter struct

/**
 * @namespace OmniDSP::Design
 * @brief Contains structures and functions related to the design phase of DSP
 * components, translating high-level parameters into concrete specifications.
 */
namespace OmniDSP::Design {

  /**
   * @brief Creates a fully resolved IIR filter design specification from input
   * parameters.
   * @details This function translates an OmniDSP::Params::IIRFilter object,
   * which holds user-provided parameters, into an OmniDSP::Design::IIRFilter
   * object, which represents the concrete design specification. It performs
   * validation during this process.
   * @param params The input parameters for the IIR filter design
   * (OmniDSP::Params::IIRFilter).
   * @return An OmniExpected<Design::IIRFilter> containing the resolved design
   * specification on success, or an OmniStatus error code on failure.
   * @throws No direct exceptions, errors are returned via OmniExpected.
   */
  OmniExpected<Design::IIRFilter> create(const Params::IIRFilter& params)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "Attempting to create Design::IIRFilter from Params: {}", params);
    }

    // Params::IIRFilter should have already performed its own validation in its
    // constructor. Here, we mainly construct the Design::IIRFilter object.

    try {
      // Construct Design::IIRFilter using the validated params.
      // The Design::IIRFilter constructor takes individual members.
      Design::IIRFilter design_obj(
          params.filter_type_,
          params.order_,
          params.sample_rate_,
          params.cutoff1_,
          params.cutoff2_,
          params.passband_ripple_db_,
          params.stopband_attenuation_db_,
          params.output_format_);

      // Call validate_consistency on the new Design::IIRFilter object
      if (!design_obj.validate_consistency()) {
        if (logger)
          logger->error(
              "Internal consistency validation failed for created "
              "Design::IIRFilter from Params::IIRFilter: {}",
              design_obj);  // Log the failing object
        return std::unexpected(
            OmniStatus::Failure);  // Or a more specific error
      }

      if (logger && logger->should_log(spdlog::level::debug)) {
        logger->debug("Successfully created Design::IIRFilter: {}", design_obj);
      }
      return design_obj;  // Return the Design::IIRFilter object
    }
    catch (const std::invalid_argument& e) {
      // This catch block would be relevant if the Design::IIRFilter constructor
      // itself throws std::invalid_argument
      if (logger)
        logger->error(
            "Invalid argument during Design::IIRFilter construction from "
            "Params::IIRFilter: {}",
            e.what());
      return std::unexpected(OmniStatus::InvalidArgument);
    }
    catch (const std::exception& e) {
      // Catch any other standard exceptions from constructor or validation
      if (logger)
        logger->error(
            "Exception during Design::IIRFilter construction from "
            "Params::IIRFilter: {}",
            e.what());
      return std::unexpected(OmniStatus::Failure);
    }
  }

}  // namespace OmniDSP::Design
