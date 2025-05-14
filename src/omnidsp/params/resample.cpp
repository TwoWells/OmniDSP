/**
 * @file resample.cpp
 * @brief Implements the constructor and fluent setters for Params::Resample.
 */

#include <spdlog/spdlog.h>  // Include spdlog for logging

#include <OmniDSP/params/resample.hpp>  // Corresponding header
#include <OmniDSP/window.hpp>  // For WindowSetup related to default args
#include <stdexcept>           // For std::invalid_argument
#include <string>              // For std::to_string, string concatenation
#include <utility>             // For std::move

namespace OmniDSP::Params {

  // Constructor (with spdlog integration)
  Resample::Resample(
      double p_input_rate,
      double p_output_rate,
      int p_quality,
      WindowSetup p_window_setup)
      : input_rate_(p_input_rate),
        output_rate_(p_output_rate),
        quality_(p_quality),
        window_setup_(std::move(p_window_setup))
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    std::string msg;

    if (input_rate_ <= 0.0) {
      msg = "Params::Resample Constructor: input_rate_ ("
            + std::to_string(input_rate_) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (output_rate_ <= 0.0) {
      msg = "Params::Resample Constructor: output_rate ("
            + std::to_string(output_rate_) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (quality_ < 0 || quality_ > 15) {
      msg = "Params::Resample Constructor: quality (" + std::to_string(quality_)
            + ") is out of the typical range [0, 15].";
      if (logger)
        logger->error(msg);  // Log as error as it's a validation failure
      throw std::invalid_argument(msg);
    }
    // WindowSetup is validated by its own constructor.
    if (logger)
      logger->trace(
          "Params::Resample constructed: IR={}, OR={}, Q={}, WinType={}",
          input_rate_,
          output_rate_,
          quality_,
          static_cast<int>(window_setup_.type));
  }

  // --- Fluent Setter Definitions ---

  Resample& Resample::input_rate(double val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val <= 0.0) {
      std::string msg = "Params::Resample::input_rate: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    input_rate_ = val;
    if (logger) logger->trace("Params::Resample::input_rate to {}", val);
    return *this;
  }

  Resample& Resample::output_rate(double val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val <= 0.0) {
      std::string msg = "Params::Resample::output_rate: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    output_rate_ = val;
    if (logger) logger->trace("Params::Resample::output_rate to {}", val);
    return *this;
  }

  Resample& Resample::quality(int val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val < 0 || val > 15) {  // Example quality range
      std::string msg = "Params::Resample::quality: value ("
                        + std::to_string(val)
                        + ") is out of the typical range [0, 15].";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    quality_ = val;
    if (logger) logger->trace("Params::Resample::quality to {}", val);
    return *this;
  }

  Resample& Resample::window_setup(WindowSetup val)
  {
    // WindowSetup's constructor handles its own validation and logging.
    window_setup_ = std::move(val);
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger)
      logger->trace(
          "Params::Resample::window_setup to type {}",
          static_cast<int>(window_setup_.type));
    return *this;
  }

}  // namespace OmniDSP::Params
