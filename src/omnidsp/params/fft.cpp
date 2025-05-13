/**
 * @file fft.cpp
 * @brief Implements the constructors and fluent setters for FFTParams and
 * RFFTParams.
 */

#include "OmniDSP/params/fft.hpp"  // Corresponding header

#include <spdlog/spdlog.h>  // Include spdlog for logging

#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::to_string, string concatenation

namespace OmniDSP {

  // --- FFTParams Implementation ---

  FFTParams::FFTParams(size_t p_length) : length_(p_length)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();  // Fallback logger
    }
    std::string msg;

    if (length_ == 0) {
      msg = "FFTParams Constructor: length (" + std::to_string(length_)
            + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    // Further validation (e.g., power of 2) is typically backend-specific
    // and handled during Plan creation.
    if (logger) logger->trace("FFTParams constructed: Length={}", length_);
  }

  FFTParams& FFTParams::length(size_t val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val == 0) {
      std::string msg = "FFTParams::length: value (" + std::to_string(val)
                        + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    length_ = val;
    if (logger) logger->trace("FFTParams::length to {}", val);
    return *this;
  }

  // --- RFFTParams Implementation ---

  RFFTParams::RFFTParams(size_t p_length) : length_(p_length)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    std::string msg;

    if (length_ == 0) {
      msg = "RFFTParams Constructor: length (" + std::to_string(length_)
            + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    // For RFFT, some backends might require length >= 2, or even/odd
    // constraints. Basic validation (length > 0) is done here.
    if (logger) logger->trace("RFFTParams constructed: Length={}", length_);
  }

  RFFTParams& RFFTParams::length(size_t val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val == 0) {
      std::string msg = "RFFTParams::length: value (" + std::to_string(val)
                        + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    length_ = val;
    if (logger) logger->trace("RFFTParams::length to {}", val);
    return *this;
  }

}  // namespace OmniDSP
