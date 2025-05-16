/**
 * @file window.cpp
 * @brief Implements constructors and stream operators for window specification
 * structs.
 */
#include "OmniDSP/specs/window.hpp"  // Corresponding header

#include <ostream>    // For std::ostream definitions
#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::to_string, std::string

// spdlog for logging in constructors
#include "spdlog/spdlog.h"
// fmt is included via the header for ostream_formatter, but if direct
// formatting was needed here, it would be included.

namespace OmniDSP {
  namespace Window {

    // --- Rectangular ---
    std::ostream& operator<<(std::ostream& os, const Rectangular& /*spec*/)
    {
      os << "Window::Rectangular";
      return os;
    }

    // --- Hann ---
    std::ostream& operator<<(std::ostream& os, const Hann& /*spec*/)
    {
      os << "Window::Hann";
      return os;
    }

    // --- Hamming ---
    std::ostream& operator<<(std::ostream& os, const Hamming& /*spec*/)
    {
      os << "Window::Hamming";
      return os;
    }

    // --- Blackman ---
    std::ostream& operator<<(std::ostream& os, const Blackman& /*spec*/)
    {
      os << "Window::Blackman";
      return os;
    }

    // --- Bartlett ---
    std::ostream& operator<<(std::ostream& os, const Bartlett& /*spec*/)
    {
      os << "Window::Bartlett";
      return os;
    }

    // --- Triangular ---
    std::ostream& operator<<(std::ostream& os, const Triangular& /*spec*/)
    {
      os << "Window::Triangular";
      return os;
    }

    // --- Flattop ---
    std::ostream& operator<<(std::ostream& os, const Flattop& /*spec*/)
    {
      os << "Window::Flattop";
      return os;
    }

    // --- Kaiser ---
    Kaiser::Kaiser(double b) : beta(b)
    {
      if (beta < 0.0) {
        std::string msg = "Window::Kaiser: beta parameter ("
                          + std::to_string(beta) + ") must be non-negative.";
        auto logger = spdlog::get("OmniDSP");
        if (!logger) {
          logger = spdlog::default_logger();
        }
        if (logger) {
          logger->error(msg);
        }
        throw std::invalid_argument(msg);
      }
    }

    std::ostream& operator<<(std::ostream& os, const Kaiser& spec)
    {
      os << "Window::Kaiser(beta: " << spec.beta << ")";
      return os;
    }

    // --- Gaussian ---
    Gaussian::Gaussian(double s) : sigma(s)
    {
      if (sigma <= 0.0) {
        std::string msg = "Window::Gaussian: sigma parameter ("
                          + std::to_string(sigma) + ") must be positive.";
        auto logger = spdlog::get("OmniDSP");
        if (!logger) {
          logger = spdlog::default_logger();
        }
        if (logger) {
          logger->error(msg);
        }
        throw std::invalid_argument(msg);
      }
    }

    std::ostream& operator<<(std::ostream& os, const Gaussian& spec)
    {
      os << "Window::Gaussian(sigma: " << spec.sigma << ")";
      return os;
    }

  }  // namespace Window
}  // namespace OmniDSP
