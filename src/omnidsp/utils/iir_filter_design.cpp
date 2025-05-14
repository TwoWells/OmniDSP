/**
 * @file iir_filter_design.cpp
 * @brief Implements utility functions for creating Design::IIRFilter from
 * Params::IIRFilter.
 */

#include <spdlog/spdlog.h>

#include <stdexcept>

#include "OmniDSP/core_types.hpp"         // For OmniExpected, Status
#include "OmniDSP/design/iir_filter.hpp"  // For Design::IIRFilter
#include "OmniDSP/params/iir_filter.hpp"
#include "OmniDSP/utils.hpp"  // For the public declaration of Utils::create_spec

namespace OmniDSP::Utils {

  // The public function's signature now returns OmniExpected<Design::IIRFilter>
  OmniExpected<Design::IIRFilter> create_spec(const Params::IIRFilter& params)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    try {
      // Construct Design::IIRFilter using the validated params.
      // The Design::IIRFilter constructor takes individual members.
      Design::IIRFilter
          design_obj(  // Changed from IIRFilterSpec to Design::IIRFilter
              params.filter_type_,
              params.order_,
              params.sample_rate_,
              params.cutoff1_,
              params.cutoff2_,
              params.passband_ripple_db_,
              params.stopband_attenuation_db_,
              params.output_format_);

      if (!design_obj
               .validate_consistency()) {  // Call validate_consistency on the
                                           // new Design::IIRFilter object
        logger->error(
            "Internal consistency validation failed for created "
            "Design::IIRFilter from Params::IIRFilter.");
        return std::unexpected(Status::Failure);
      }

      logger->debug(
          "Successfully created Design::IIRFilter from Params::IIRFilter. "
          "Type: "
          "{}, Order: {}, SR: {}",
          static_cast<int>(design_obj.type),
          design_obj.order,
          design_obj.sample_rate);
      return design_obj;  // Return the Design::IIRFilter object
    }
    catch (const std::invalid_argument& e) {
      logger->error(
          "Invalid argument during Design::IIRFilter construction from "
          "Params::IIRFilter: {}",
          e.what());
      return std::unexpected(Status::InvalidArgument);
    }
    catch (const std::exception& e) {
      logger->error(
          "Exception during Design::IIRFilter construction from "
          "Params::IIRFilter: {}",
          e.what());
      return std::unexpected(Status::Failure);
    }
  }

}  // namespace OmniDSP::Utils
