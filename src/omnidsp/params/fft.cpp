/**
 * @file fft.cpp
 * @brief Implements the constructors and fluent setters for Params::FFT and
 * Params::RFFT.
 */

#include "OmniDSP/params/fft.hpp"  // Corresponding header

#include <spdlog/fmt/ostr.h>  // Include for custom ostream operator support with spdlog
#include <spdlog/spdlog.h>  // Include spdlog for logging

#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::to_string, string concatenation

/**
 * @namespace OmniDSP::Params
 * @brief Contains structures for specifying parameters for various DSP
 * operations before full design.
 */
namespace OmniDSP::Params {

  // --- Params::FFT Implementation ---

  /**
   * @brief Constructor for FFT parameters.
   * @param p_length The length of the FFT. Must be positive.
   * @throws std::invalid_argument if p_length is zero.
   */
  FFT::FFT(size_t p_length) : length_(p_length)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();  // Fallback logger
    }
    std::string msg;

    if (length_ == 0) {
      msg = "Params::FFT Constructor: length (" + std::to_string(length_)
            + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    // Further validation (e.g., power of 2) is typically backend-specific
    // and handled during Plan creation.
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::FFT constructed: {}", *this);
    }
  }

  /**
   * @brief Sets the FFT length.
   * @param val The new FFT length. Must be positive.
   * @return A reference to this Params::FFT object for chaining.
   * @throws std::invalid_argument if val is zero.
   */
  FFT& FFT::length(size_t val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val == 0) {
      std::string msg = "Params::FFT::length: value (" + std::to_string(val)
                        + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    length_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::FFT::length updated: {}", *this);
    }
    return *this;
  }

  // --- Params::RFFT Implementation ---

  /**
   * @brief Constructor for RFFT parameters.
   * @param p_length The length of the RFFT (number of real input points). Must
   * be positive.
   * @throws std::invalid_argument if p_length is zero.
   */
  RFFT::RFFT(size_t p_length) : length_(p_length)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    std::string msg;

    if (length_ == 0) {
      msg = "Params::RFFT Constructor: length (" + std::to_string(length_)
            + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    // For RFFT, some backends might require length >= 2, or even/odd
    // constraints. Basic validation (length > 0) is done here.
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::RFFT constructed: {}", *this);
    }
  }

  /**
   * @brief Sets the RFFT length (number of real input points).
   * @param val The new RFFT length. Must be positive.
   * @return A reference to this Params::RFFT object for chaining.
   * @throws std::invalid_argument if val is zero.
   */
  RFFT& RFFT::length(size_t val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val == 0) {
      std::string msg = "Params::RFFT::length: value (" + std::to_string(val)
                        + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    length_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::RFFT::length updated: {}", *this);
    }
    return *this;
  }

}  // namespace OmniDSP::Params
