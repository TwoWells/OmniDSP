/**
 * @file resample.cpp
 * @brief Implements the constructor and fluent setters for Params::Resample.
 */

#include "OmniDSP/params/resample.hpp"  // Corresponding header

#include <spdlog/fmt/ostr.h>  // Include for custom ostream operator support with spdlog
#include <spdlog/spdlog.h>  // Include spdlog for logging

#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::to_string, string concatenation
#include <utility>    // For std::move

#include "OmniDSP/window.hpp"  // For WindowSetup related to default args

/**
 * @namespace OmniDSP::Params
 * @brief Contains structures for specifying parameters for various DSP
 * operations before full design.
 */
namespace OmniDSP::Params {

  /**
   * @brief Constructor for Params::Resample.
   * * Initializes a Resample parameters object with validation.
   * * @param p_input_rate Input sample rate in Hz. Must be positive.
   * @param p_output_rate Desired output sample rate in Hz. Must be positive.
   * @param p_quality Quality setting for the resampling process (e.g., 0-15).
   * Higher values generally mean a sharper filter and more computation.
   * @param p_window_setup Windowing function setup for the internal prototype
   * FIR filter. The length field of this setup is typically determined
   * internally.
   * @throws std::invalid_argument if input_rate or output_rate are not
   * positive, or if quality is outside the typical range [0, 15].
   */
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
    if (logger && logger->should_log(spdlog::level::trace)) {
      // Log the entire object using the new operator<<
      logger->trace("Params::Resample constructed: {}", *this);
    }
  }

  // --- Fluent Setter Definitions ---

  /**
   * @brief Sets the input sample rate for the resampling operation.
   * @param val The new input sample rate in Hz. Must be positive.
   * @return A reference to this Params::Resample object for chaining.
   * @throws std::invalid_argument if val is not positive.
   */
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
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::Resample::input_rate updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the desired output sample rate for the resampling operation.
   * @param val The new output sample rate in Hz. Must be positive.
   * @return A reference to this Params::Resample object for chaining.
   * @throws std::invalid_argument if val is not positive.
   */
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
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::Resample::output_rate updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the quality setting for the resampling process.
   * @param val The new quality setting, typically in the range [0, 15].
   * @return A reference to this Params::Resample object for chaining.
   * @throws std::invalid_argument if val is outside the typical range [0, 15].
   */
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
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::Resample::quality updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the windowing function setup for the prototype filter design.
   * * The provided WindowSetup object's own constructor is responsible for its
   * validation. The `length` field of this WindowSetup is typically ignored by
   * the design process, as the filter order (and thus window length) is
   * determined internally based on resampling parameters and quality.
   * * @param val The new WindowSetup object.
   * @return A reference to this Params::Resample object for chaining.
   */
  Resample& Resample::window_setup(WindowSetup val)
  {
    // WindowSetup's constructor handles its own validation and logging.
    window_setup_ = std::move(val);
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::Resample::window_setup updated: {}", *this);
    }
    return *this;
  }

}  // namespace OmniDSP::Params
