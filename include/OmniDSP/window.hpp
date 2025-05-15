/**
 * @file window.hpp
 * @brief Defines the WindowSetup struct for window configuration and
 * window generation utilities. The Window enum itself is now in
 * types/window.hpp.
 */

#ifndef OMNIDSP_WINDOW_HPP
#define OMNIDSP_WINDOW_HPP

#include <algorithm>  // For std::max
#include <cmath>      // For std::abs, std::cos, std::sqrt, std::pow, std::exp
#include <cstddef>    // For size_t
#include <expected>  // For std::expected and std::unexpected (used by OmniExpected)
#include <limits>     // For numeric_limits
#include <map>        // For std::map (used in WindowParams)
#include <numbers>    // For std::numbers constants
#include <optional>   // For std::optional (used in WindowSetup)
#include <ostream>    // For std::ostream (needed for operator<<)
#include <span>       // For std::span
#include <sstream>    // For std::ostringstream (to format WindowParams map)
#include <stdexcept>  // For std::invalid_argument (used in constructors)
#include <string>  // For std::string (used in WindowParams key and exceptions)
#include <string_view>
#include <utility>  // For std::move
#include <vector>   // Potentially for future generate_window returning vector

#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT
#include "OmniDSP/types/window.hpp"  // For OmniDSP::Type::Window and its operator<<
#include "core_types.hpp"  // For Status, F32, F64, OmniExpected, OmniException etc.

// Include Boost header for Bessel functions
#include "boost/math/special_functions/bessel.hpp"

// Include spdlog for logging
#include "spdlog/spdlog.h"

// Include fmt headers for custom formatter specialization
#include <fmt/core.h>     // For basic formatting
#include <fmt/ostream.h>  // Specifically for ostream_formatter

namespace OmniDSP {

  // OmniDSP::WindowType enum has been moved to OmniDSP::Type::Window in
  // types/window.hpp The get_window_type_name function has been moved and
  // renamed to OmniDSP::Type::get_window_name

  /**
   * @brief Type alias for window-specific parameters.
   *
   * This map allows passing named parameters for window types that require
   * them. For example:
   * - Kaiser window: {"beta", value}
   * - Gaussian window: {"sigma", value}
   */
  using WindowParams = std::map<std::string, double>;

  /**
   * @brief Structure to define window parameters for design or direct use.
   *
   * This structure is used to specify the configuration of a window function.
   * Validation of parameters is performed upon construction.
   */
  struct OMNIDSP_EXPORT WindowSetup {
    Type::Window
        type;    ///< The type of the window (from OmniDSP::Type::Window).
    int length;  ///< Desired length of the window. Must be non-negative.
    std::optional<WindowParams>
        params;  ///< Optional parameters specific to the window type.

    /**
     * @brief Constructor for WindowSetup.
     * @param win_type The type of the window (OmniDSP::Type::Window).
     * @param win_length Desired length of the window. Must be non-negative.
     * @param win_params Optional parameters specific to the window type.
     * Required for types like Kaiser ("beta") and Gaussian ("sigma").
     * @throws std::invalid_argument if length is negative, or if required
     * params for the specified window type are missing or invalid.
     */
    explicit WindowSetup(
        Type::Window win_type,  // Updated to use OmniDSP::Type::Window
        int win_length,
        std::optional<WindowParams> win_params = std::nullopt)
        : type(win_type), length(win_length), params(std::move(win_params))
    {
      auto logger = spdlog::get("OmniDSP");
      if (!logger) {
        logger = spdlog::default_logger();
      }

      if (length < 0) {
        std::string msg = "WindowSetup construction failed: Negative length ("
                          + std::to_string(length) + ") specified.";
        logger->error(msg);
        throw std::invalid_argument(msg);
      }

      switch (type) {
        case Type::Window::Kaiser:  // Updated to use OmniDSP::Type::Window
          if (!params) {
            std::string msg
                = "WindowSetup construction (Kaiser): Missing 'params' field.";
            logger->error(msg);
            throw std::invalid_argument(msg);
          }
          if (!params->count("beta")) {
            std::string msg
                = "WindowSetup construction (Kaiser): Missing 'beta' key in "
                  "params.";
            logger->error(msg);
            throw std::invalid_argument(msg);
          }
          {
            double beta = params->at("beta");
            if (beta < 0.0) {
              std::string msg
                  = "WindowSetup construction (Kaiser): Invalid 'beta' value ("
                    + std::to_string(beta) + "). Must be non-negative.";
              logger->error(msg);
              throw std::invalid_argument(msg);
            }
          }
          break;
        case Type::Window::Gaussian:  // Updated to use OmniDSP::Type::Window
          if (!params) {
            std::string msg
                = "WindowSetup construction (Gaussian): Missing 'params' "
                  "field.";
            logger->error(msg);
            throw std::invalid_argument(msg);
          }
          if (!params->count("sigma")) {
            std::string msg
                = "WindowSetup construction (Gaussian): Missing 'sigma' key in "
                  "params.";
            logger->error(msg);
            throw std::invalid_argument(msg);
          }
          {
            double sigma = params->at("sigma");
            if (sigma <= 0.0) {
              std::string msg
                  = "WindowSetup construction (Gaussian): Invalid 'sigma' "
                    "value ("
                    + std::to_string(sigma) + "). Must be positive.";
              logger->error(msg);
              throw std::invalid_argument(msg);
            }
          }
          break;
        case Type::Window::Rectangular:  // Updated
        case Type::Window::Bartlett:     // Updated
        case Type::Window::Hann:         // Updated
        case Type::Window::Hamming:      // Updated
        case Type::Window::Blackman:     // Updated
        case Type::Window::Flattop:      // Updated
        case Type::Window::Triangular:   // Updated
          if (params && !params->empty()) {
            logger->debug(
                "WindowSetup construction ({}): 'params' provided but not used "
                "by this window type.",
                Type::get_window_name(
                    type));  // Updated to use Type::get_window_name
          }
          break;
        default:
          // This should ideally not be reached if Type::Window is comprehensive
          // and Type::get_window_name handles all cases.
          // However, as a safeguard:
          if (Type::get_window_name(type) == "Unknown Window") {
            std::string msg
                = "WindowSetup construction: Unknown window type specified: "
                  + std::to_string(static_cast<int>(type));
            logger->error(msg);
            throw std::invalid_argument(msg);
          }
          break;
      }
    }
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of WindowSetup.
   * @param os The output stream.
   * @param setup The WindowSetup object to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, const WindowSetup& setup)
  {
    os << "WindowSetup(Type: "
       << setup.type  // Uses OmniDSP::Type::Window::operator<<
       << ", Length: " << setup.length;
    if (setup.params) {
      os << ", Params: {";
      bool first = true;
      for (const auto& pair : setup.params.value()) {
        if (!first) {
          os << ", ";
        }
        os << "\"" << pair.first << "\": " << pair.second;
        first = false;
      }
      os << "}";
    }
    os << ")";
    return os;
  }

  /**
   * @brief Generates window coefficients based on the WindowSetup structure.
   *
   * @tparam T The floating-point type (float or double) for the output
   * coefficients.
   * @param setup The WindowSetup structure configuring the window. Assumed to
   * be valid if constructed.
   * @param output The span to write the window coefficients into. Must have
   * size `setup.length`.
   * @return OmniExpected<void> indicating success or an error OmniStatus.
   */
  template <typename T_Data>  // Renamed T to T_Data for clarity
  OmniExpected<void> generate_window(
      const WindowSetup& setup, std::span<T_Data> output)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (static_cast<size_t>(setup.length) == 0) {
      if (output.empty()) {
        return {};
      }
      else {
        logger->warn(
            "generate_window: Size mismatch. Window length is 0 but output "
            "span size is {}.",
            output.size());
        return std::unexpected(OmniStatus::SizeMismatch);
      }
    }
    if (output.size() != static_cast<size_t>(setup.length)) {
      logger->warn(
          "generate_window: Size mismatch. Expected output span of size {}, "
          "got {}.",
          setup.length,
          output.size());
      return std::unexpected(OmniStatus::SizeMismatch);
    }

    double kaiser_beta = 0.0;
    if (setup.type == Type::Window::Kaiser) {  // Updated
      kaiser_beta = setup.params->at("beta");
    }

    double gaussian_sigma = 0.0;
    if (setup.type == Type::Window::Gaussian) {  // Updated
      gaussian_sigma = setup.params->at("sigma");
    }

    const double N_double = static_cast<double>(
        setup.length);  // Renamed N to N_double for clarity
    const double N_minus_1 = (N_double > 0) ? (N_double - 1.0) : 0.0;

    constexpr double pi = std::numbers::pi_v<double>;
    const double two_pi = 2.0 * pi;
    const double four_pi = 4.0 * pi;
    const double six_pi = 6.0 * pi;
    const double eight_pi = 8.0 * pi;

    for (int i = 0; i < setup.length; ++i) {
      double n_val_double
          = static_cast<double>(i);  // Renamed n_double to n_val_double
      T_Data val = T_Data{0.0};      // Use T_Data

      switch (setup.type) {
        case Type::Window::Rectangular:
          val = T_Data{1.0};
          break;                        // Updated
        case Type::Window::Bartlett:    // Updated
        case Type::Window::Triangular:  // Updated
          if (setup.length == 1) {
            val = T_Data{1.0};
          }
          else {
            double M = N_minus_1 / 2.0;
            val = static_cast<T_Data>(1.0 - std::abs(n_val_double - M) / M);
          }
          break;
        case Type::Window::Hann:  // Updated
          if (setup.length == 1) {
            val = T_Data{1.0};
          }
          else {
            val = static_cast<T_Data>(
                0.5 * (1.0 - std::cos(two_pi * n_val_double / N_minus_1)));
          }
          break;
        case Type::Window::Hamming:  // Updated
          if (setup.length == 1) {
            val = T_Data{1.0};
          }
          else {
            val = static_cast<T_Data>(
                0.54 - 0.46 * std::cos(two_pi * n_val_double / N_minus_1));
          }
          break;
        case Type::Window::Blackman:  // Updated
          if (setup.length == 1) {
            val = T_Data{1.0};
          }
          else {
            val = static_cast<T_Data>(
                0.42 - 0.50 * std::cos(two_pi * n_val_double / N_minus_1)
                + 0.08 * std::cos(four_pi * n_val_double / N_minus_1));
          }
          break;
        case Type::Window::Flattop:  // Updated
          if (setup.length == 1) {
            val = T_Data{1.0};
          }
          else {
            const double a0 = 0.21557895;
            const double a1 = 0.41663158;
            const double a2 = 0.277263158;
            const double a3 = 0.083578947;
            const double a4 = 0.006947368;
            val = static_cast<T_Data>(
                a0 - a1 * std::cos(two_pi * n_val_double / N_minus_1)
                + a2 * std::cos(four_pi * n_val_double / N_minus_1)
                - a3 * std::cos(six_pi * n_val_double / N_minus_1)
                + a4 * std::cos(eight_pi * n_val_double / N_minus_1));
          }
          break;
        case Type::Window::Kaiser: {  // Updated
          if (setup.length == 1) {
            val = T_Data{1.0};
          }
          else {
            double term_sq
                = std::pow((2.0 * n_val_double / N_minus_1 - 1.0), 2.0);
            double sqrt_arg = std::max(0.0, 1.0 - term_sq);
            double kaiser_arg = kaiser_beta * std::sqrt(sqrt_arg);
            double i0_beta = boost::math::cyl_bessel_i(0, kaiser_beta);
            val = (i0_beta > std::numeric_limits<double>::epsilon())
                      ? static_cast<T_Data>(
                            boost::math::cyl_bessel_i(0, kaiser_arg) / i0_beta)
                      : T_Data{0.0};
          }
          break;
        }
        case Type::Window::Gaussian: {  // Updated
          if (setup.length == 1) {
            val = T_Data{1.0};
          }
          else {
            double center = N_minus_1 / 2.0;
            double denom = gaussian_sigma * center;
            double term = (denom > std::numeric_limits<double>::epsilon())
                              ? (n_val_double - center) / denom
                              : 0.0;
            val = static_cast<T_Data>(std::exp(-0.5 * term * term));
          }
          break;
        }
        default:
          logger->error(
              "generate_window: Reached default case with unhandled window "
              "type: {}",
              Type::get_window_name(setup.type));  // Updated
          return std::unexpected(OmniStatus::InvalidArgument);
      }
      output[static_cast<size_t>(i)] = val;
    }
    return {};
  }

}  // namespace OmniDSP

// fmt::formatter specialization for OmniDSP::WindowSetup
// This is still needed here if WindowSetup itself is to be logged directly with
// fmt {}
template <>
struct fmt::formatter<OmniDSP::WindowSetup> : fmt::ostream_formatter {};

#endif  // OMNIDSP_WINDOW_HPP
