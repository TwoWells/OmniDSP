/**
 * @file resample.cpp
 * @brief Implements the constructor for ResampleParams.
 */

#include <OmniDSP/params/resample.hpp>  // Corresponding header
#include <stdexcept>                    // For std::invalid_argument
#include <string>   // For std::to_string, string concatenation
#include <utility>  // For std::move

// spdlog could be included here if detailed logging during construction is
// desired, but typically, throwing an exception is sufficient for parameter
// validation. #include "spdlog/spdlog.h"

namespace OmniDSP {

  ResampleParams::ResampleParams(
      double p_input_rate,
      double p_output_rate,
      int p_quality,
      WindowSetup p_window_setup)
      : input_rate(p_input_rate),
        output_rate(p_output_rate),
        quality(p_quality),
        window_setup(std::move(p_window_setup))  // WindowSetup is validated on
                                                 // its own construction
  {
    // auto logger = spdlog::get("OmniDSP"); // Example if logging
    // if (!logger) { logger = spdlog::default_logger(); }

    if (input_rate <= 0.0) {
      // if(logger) logger->error("ResampleParams: input_rate ({}) must be
      // positive.", input_rate);
      throw std::invalid_argument(
          "ResampleParams: input_rate (" + std::to_string(input_rate)
          + ") must be positive.");
    }
    if (output_rate <= 0.0) {
      // if(logger) logger->error("ResampleParams: output_rate ({}) must be
      // positive.", output_rate);
      throw std::invalid_argument(
          "ResampleParams: output_rate (" + std::to_string(output_rate)
          + ") must be positive.");
    }
    if (quality < 0 || quality > 15) {  // Example quality range, adjust as per
                                        // your library's design
      // if(logger) logger->error("ResampleParams: quality ({}) is out of
      // expected range [0, 15].", quality);
      throw std::invalid_argument(
          "ResampleParams: quality (" + std::to_string(quality)
          + ") is out of the typical range [0, 15].");
    }

    // The `window_setup` member is of type WindowSetup, which already validates
    // its own parameters (like window type, beta for Kaiser, etc.) upon its
    // construction. The `length` field within `p_window_setup` is explicitly
    // set to 0 in the default argument of ResampleParams' constructor,
    // signifying that the actual window length for the prototype FIR filter
    // will be determined by the `Utils::create_spec(const ResampleParams&)`
    // function based on `input_rate`, `output_rate`, and `quality`.

    // if(logger) logger->trace("ResampleParams constructed: IR={}, OR={}, Q={},
    // WinType={}", input_rate, output_rate, quality,
    // static_cast<int>(window_setup.type));
  }

}  // namespace OmniDSP
