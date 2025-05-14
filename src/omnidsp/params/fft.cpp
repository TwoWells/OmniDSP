/**
 * @file fft.cpp
 * @brief Implements the constructors and fluent setters for Params::FFT and
 * Params::RFFT.
 */

#include "OmniDSP/params/fft.hpp"  // Corresponding header

#include <spdlog/spdlog.h>  // Include spdlog for logging

#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::to_string, string concatenation

namespace OmniDSP::Params {

  // --- Params::FFT Implementation ---

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
    if (logger) logger->trace("Params::FFT constructed: Length={}", length_);
  }

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
    if (logger) logger->trace("Params::FFT::length to {}", val);
    return *this;
  }

  // --- Params::RFFT Implementation ---

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
    if (logger) logger->trace("Params::RFFT constructed: Length={}", length_);
  }

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
    if (logger) logger->trace("Params::RFFT::length to {}", val);
    return *this;
  }

}  // namespace OmniDSP::Params
