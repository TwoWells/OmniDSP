/**
 * @file window.hpp (types)
 * @brief Defines windowing function related enumerations for the OmniDSP
 * library.
 */
#ifndef OMNIDSP_TYPES_WINDOW_HPP
#define OMNIDSP_TYPES_WINDOW_HPP

#include <ostream>
#include <string_view>

#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT

// Include fmt headers for custom formatter specialization
#include <fmt/core.h>
#include <fmt/ostream.h>

namespace OmniDSP {
  /**
   * @namespace OmniDSP::Type
   * @brief Contains core enumerations and type definitions used throughout the
   * OmniDSP library.
   */
  namespace Type {

    /**
     * @brief Enumeration of standard window function types supported by
     * OmniDSP.
     * @note Formerly OmniDSP::WindowType.
     */
    enum class OMNIDSP_EXPORT Window {
      Rectangular,  ///< Rectangular (or Boxcar) window.
      Bartlett,     ///< Bartlett (triangular, zero-endpoints) window.
      Hann,         ///< Hann (or Hanning) window.
      Hamming,      ///< Hamming window.
      Blackman,     ///< Blackman window.
      Flattop,      ///< Flattop window, provides high amplitude accuracy.
      Gaussian,     ///< Gaussian window. Requires a "sigma" parameter.
      Kaiser,       ///< Kaiser window. Requires a "beta" parameter.
      Triangular    ///< Triangular window (non-zero endpoints, similar to
                    ///< Bartlett but can differ in definition).
    };

    /**
     * @brief Gets the string name corresponding to a Type::Window enum value.
     * @param type The window type.
     * @return A string_view representing the name of the window type.
     */
    inline std::string_view get_window_name(Window type) noexcept
    {
      switch (type) {
        case Window::Rectangular:
          return "Rectangular";
        case Window::Bartlett:
          return "Bartlett";
        case Window::Hann:
          return "Hann";
        case Window::Hamming:
          return "Hamming";
        case Window::Blackman:
          return "Blackman";
        case Window::Flattop:
          return "Flattop";
        case Window::Gaussian:
          return "Gaussian";
        case Window::Kaiser:
          return "Kaiser";
        case Window::Triangular:
          return "Triangular";
        default:
          return "Unknown Window";
      }
    }

    /**
     * @brief Overloads the << operator for easy printing/logging of
     * Type::Window.
     * @param os The output stream.
     * @param type The Type::Window enum value to print.
     * @return A reference to the output stream.
     */
    inline std::ostream& operator<<(std::ostream& os, Window type)
    {
      os << get_window_name(type);
      return os;
    }

  }  // namespace Type
}  // namespace OmniDSP

// Specialization of fmt::formatter for OmniDSP::Type::Window
template <>
struct fmt::formatter<OmniDSP::Type::Window> : fmt::ostream_formatter {};

#endif  // OMNIDSP_TYPES_WINDOW_HPP
