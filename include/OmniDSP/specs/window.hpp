/**
 * @file window.hpp
 * @brief Defines lightweight specification structs for various window
 * functions.
 * @details These structs act as type tags and parameter holders for configuring
 * window operations. They include validation for parameters and support for
 * easy logging via std::ostream and fmt::format (spdlog compatible).
 */
#ifndef OMNIDSP_SPECS_WINDOW_HPP
#define OMNIDSP_SPECS_WINDOW_HPP

#include <ostream>  // For std::ostream
#include <string>   // For std::string (used by Kaiser, Gaussian in .cpp)

#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT

// fmt library for spdlog compatibility and custom formatting
#include <fmt/core.h>
#include <fmt/ostream.h>

namespace OmniDSP {
  namespace Window {

    /**
     * @brief Specification for a Rectangular (or Boxcar) window.
     */
    struct OMNIDSP_EXPORT Rectangular {
      // No parameters
    };

    OMNIDSP_EXPORT std::ostream& operator<<(
        std::ostream& os, const Rectangular& spec);

    /**
     * @brief Specification for a Hann (or Hanning) window.
     */
    struct OMNIDSP_EXPORT Hann {
      // No parameters
    };

    OMNIDSP_EXPORT std::ostream& operator<<(std::ostream& os, const Hann& spec);

    /**
     * @brief Specification for a Hamming window.
     */
    struct OMNIDSP_EXPORT Hamming {
      // No parameters
    };

    OMNIDSP_EXPORT std::ostream& operator<<(
        std::ostream& os, const Hamming& spec);

    /**
     * @brief Specification for a Blackman window.
     */
    struct OMNIDSP_EXPORT Blackman {
      // No parameters
    };

    OMNIDSP_EXPORT std::ostream& operator<<(
        std::ostream& os, const Blackman& spec);

    /**
     * @brief Specification for a Bartlett (triangular, zero-endpoints) window.
     */
    struct OMNIDSP_EXPORT Bartlett {
      // No parameters
    };

    OMNIDSP_EXPORT std::ostream& operator<<(
        std::ostream& os, const Bartlett& spec);

    /**
     * @brief Specification for a Triangular window (non-zero endpoints).
     */
    struct OMNIDSP_EXPORT Triangular {
      // No parameters
    };

    OMNIDSP_EXPORT std::ostream& operator<<(
        std::ostream& os, const Triangular& spec);

    /**
     * @brief Specification for a Flattop window.
     */
    struct OMNIDSP_EXPORT Flattop {
      // No parameters
    };

    OMNIDSP_EXPORT std::ostream& operator<<(
        std::ostream& os, const Flattop& spec);

    /**
     * @brief Specification for a Kaiser window.
     */
    struct OMNIDSP_EXPORT Kaiser {
      double beta;

      /**
       * @brief Constructor for Kaiser window specification.
       * @param b The shape parameter beta. Must be non-negative.
       * @throws std::invalid_argument if beta is negative.
       */
      explicit Kaiser(double b);
    };

    OMNIDSP_EXPORT std::ostream& operator<<(
        std::ostream& os, const Kaiser& spec);

    /**
     * @brief Specification for a Gaussian window.
     */
    struct OMNIDSP_EXPORT Gaussian {
      double sigma;

      /**
       * @brief Constructor for Gaussian window specification.
       * @param s The standard deviation sigma. Must be positive.
       * @throws std::invalid_argument if sigma is not positive.
       */
      explicit Gaussian(double s);
    };

    OMNIDSP_EXPORT std::ostream& operator<<(
        std::ostream& os, const Gaussian& spec);

  }  // namespace Window
}  // namespace OmniDSP

// fmt::formatter specializations for direct logging with spdlog using {}
template <>
struct fmt::formatter<OmniDSP::Window::Rectangular> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Window::Hann> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Window::Hamming> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Window::Blackman> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Window::Bartlett> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Window::Triangular> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Window::Flattop> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Window::Kaiser> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Window::Gaussian> : fmt::ostream_formatter {};

#endif  // OMNIDSP_SPECS_WINDOW_HPP
