/**
 * @file window.hpp
 * @brief Defines windowing function types, the WindowSetup struct for
 * configuration, and window generation utilities.
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
#include <span>       // For std::span
#include <stdexcept>  // For std::invalid_argument (used in constructors)
#include <string>  // For std::string (used in WindowParams key and exceptions)
#include <string_view>  // For get_window_type_name
#include <utility>      // For std::move
#include <vector>  // Potentially for future generate_window returning vector

#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT
#include "core_types.hpp"  // For Status, F32, F64, OmniExpected, OmniException etc.

// Include Boost header for Bessel functions
#include "boost/math/special_functions/bessel.hpp"

// Include spdlog for logging
#include "spdlog/spdlog.h"

namespace OmniDSP {

  /**
   * @brief Enumeration of standard window function types supported by OmniDSP.
   */
  enum class WindowType {
    Rectangular,
    Bartlett,
    Hann,  // Also known as Hanning
    Hamming,
    Blackman,
    Flattop,
    Gaussian,   // Requires "sigma" > 0 in params
    Kaiser,     // Requires "beta" >= 0 in params
    Triangular  // Similar to Bartlett
  };

  /**
   * @brief Gets the string name corresponding to a WindowType enum value.
   * @param type The window type.
   * @return A string_view representing the name of the window type.
   */
  inline std::string_view get_window_type_name(WindowType type) noexcept
  {
    switch (type) {
      case WindowType::Rectangular:
        return "Rectangular";
      case WindowType::Bartlett:
        return "Bartlett";
      case WindowType::Hann:
        return "Hann";
      case WindowType::Hamming:
        return "Hamming";
      case WindowType::Blackman:
        return "Blackman";
      case WindowType::Flattop:
        return "Flattop";
      case WindowType::Gaussian:
        return "Gaussian";
      case WindowType::Kaiser:
        return "Kaiser";
      case WindowType::Triangular:
        return "Triangular";
      default:
        return "Unknown Window";
    }
  }

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
   *
   * Example:
   * @code
   * try {
   * OmniDSP::WindowSetup kaiser_setup(OmniDSP::WindowType::Kaiser, 128,
   * OmniDSP::WindowParams{{"beta", 4.0}});
   * OmniDSP::WindowSetup hann_setup(OmniDSP::WindowType::Hann, 64); // No
   * params needed } catch (const std::invalid_argument& e) {
   * // Handle construction failure (e.g., missing/invalid params)
   * spdlog::error("Failed to create WindowSetup: {}", e.what());
   * }
   * @endcode
   */
  struct OMNIDSP_EXPORT WindowSetup {
    WindowType type;  ///< The type of the window.
    int length;       ///< Desired length of the window. Must be non-negative.
    std::optional<WindowParams>
        params;  ///< Optional parameters specific to the window type.

    /**
     * @brief Constructor for WindowSetup.
     * @param win_type The type of the window.
     * @param win_length Desired length of the window. Must be non-negative.
     * @param win_params Optional parameters specific to the window type.
     * Required for types like Kaiser ("beta") and Gaussian ("sigma").
     * @throws std::invalid_argument if length is negative, or if required
     * params for the specified window type are missing or invalid.
     */
    explicit WindowSetup(
        WindowType win_type,
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
        case WindowType::Kaiser:
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
        case WindowType::Gaussian:
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
        case WindowType::Rectangular:
        case WindowType::Bartlett:
        case WindowType::Hann:
        case WindowType::Hamming:
        case WindowType::Blackman:
        case WindowType::Flattop:
        case WindowType::Triangular:
          if (params && !params->empty()) {
            // It's not strictly an error to provide params to a window that
            // doesn't use them, but it might indicate a user misunderstanding.
            // A debug log is appropriate.
            logger->debug(
                "WindowSetup construction ({}): 'params' provided but not used "
                "by this window type.",
                get_window_type_name(type));
          }
          break;
        default:
          if (get_window_type_name(type) == "Unknown Window") {
            std::string msg
                = "WindowSetup construction: Unknown window type specified: "
                  + std::to_string(static_cast<int>(type));
            logger->error(msg);
            throw std::invalid_argument(msg);
          }
          break;
      }
      // If we reach here, construction is successful.
      // logger->trace("WindowSetup constructed successfully: type={},
      // length={}", get_window_type_name(type), length);
    }

    // To allow for easy creation of windows without params,
    // we can add an overload or make the params parameter in the main
    // constructor default to std::nullopt (which is already done). Example:
    // WindowSetup(WindowType::Hann, 64) would use the main constructor.
  };

  /**
   * @brief Generates window coefficients based on the WindowSetup structure.
   *
   * @tparam T The floating-point type (float or double) for the output
   * coefficients.
   * @param setup The WindowSetup structure configuring the window. Assumed to
   * be valid if constructed.
   * @param output The span to write the window coefficients into. Must have
   * size `setup.length`.
   * @return OmniExpected<void> indicating success or an error Status.
   * - On success: An empty expected value.
   * - On failure: An unexpected value containing a Status code (e.g.,
   * SizeMismatch).
   */
  template <typename T>
  OmniExpected<void> generate_window(
      const WindowSetup& setup, std::span<T> output)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    // WindowSetup object is assumed to be valid by this point due to
    // constructor validation. Length non-negativity and required params
    // (Kaiser/Gaussian) are already checked.

    if (static_cast<size_t>(setup.length) == 0) {
      if (output.empty()) {
        return {};  // Success, nothing to do for zero length and empty output
      }
      else {
        logger->warn(
            "generate_window: Size mismatch. Window length is 0 but output "
            "span size is {}.",
            output.size());
        return std::unexpected(Status::SizeMismatch);
      }
    }
    if (output.size() != static_cast<size_t>(setup.length)) {
      logger->warn(
          "generate_window: Size mismatch. Expected output span of size {}, "
          "got {}.",
          setup.length,
          output.size());
      return std::unexpected(Status::SizeMismatch);
    }

    // Extract parameters (safe to assume params exist if type requires them,
    // due to constructor validation)
    double kaiser_beta = 0.0;
    if (setup.type == WindowType::Kaiser) {
      // params and "beta" key are guaranteed to exist and be valid by
      // constructor.
      kaiser_beta = setup.params->at("beta");
    }

    double gaussian_sigma = 0.0;
    if (setup.type == WindowType::Gaussian) {
      // params and "sigma" key are guaranteed to exist and be valid by
      // constructor.
      gaussian_sigma = setup.params->at("sigma");
    }

    const double N = static_cast<double>(setup.length);
    const double N_minus_1 = (N > 0) ? (N - 1.0) : 0.0;

    constexpr double pi = std::numbers::pi_v<double>;
    const double two_pi = 2.0 * pi;
    const double four_pi = 4.0 * pi;
    const double six_pi = 6.0 * pi;
    const double eight_pi = 8.0 * pi;

    for (int i = 0; i < setup.length; ++i) {
      double n_double = static_cast<double>(i);
      T val = T{0.0};

      switch (setup.type) {
        case WindowType::Rectangular:
          val = T{1.0};
          break;
        case WindowType::Bartlett:
        case WindowType::Triangular:
          if (setup.length == 1) {
            val = T{1.0};
          }
          else {
            double M = N_minus_1 / 2.0;
            val = static_cast<T>(1.0 - std::abs(n_double - M) / M);
          }
          break;
        case WindowType::Hann:
          if (setup.length == 1) {
            val = T{1.0};
          }
          else {
            val = static_cast<T>(
                0.5 * (1.0 - std::cos(two_pi * n_double / N_minus_1)));
          }
          break;
        case WindowType::Hamming:
          if (setup.length == 1) {
            val = T{1.0};
          }
          else {
            val = static_cast<T>(
                0.54 - 0.46 * std::cos(two_pi * n_double / N_minus_1));
          }
          break;
        case WindowType::Blackman:
          if (setup.length == 1) {
            val = T{1.0};
          }
          else {
            val = static_cast<T>(
                0.42 - 0.50 * std::cos(two_pi * n_double / N_minus_1)
                + 0.08 * std::cos(four_pi * n_double / N_minus_1));
          }
          break;
        case WindowType::Flattop:
          if (setup.length == 1) {
            val = T{1.0};
          }
          else {
            const double a0 = 0.21557895;
            const double a1 = 0.41663158;
            const double a2 = 0.277263158;
            const double a3 = 0.083578947;
            const double a4 = 0.006947368;
            val = static_cast<T>(
                a0 - a1 * std::cos(two_pi * n_double / N_minus_1)
                + a2 * std::cos(four_pi * n_double / N_minus_1)
                - a3 * std::cos(six_pi * n_double / N_minus_1)
                + a4 * std::cos(eight_pi * n_double / N_minus_1));
          }
          break;
        case WindowType::Kaiser: {
          if (setup.length == 1) {
            val = T{1.0};
          }
          else {
            double term_sq = std::pow((2.0 * n_double / N_minus_1 - 1.0), 2.0);
            double sqrt_arg = std::max(0.0, 1.0 - term_sq);
            double kaiser_arg = kaiser_beta * std::sqrt(sqrt_arg);
            double i0_beta = boost::math::cyl_bessel_i(0, kaiser_beta);
            val = (i0_beta > std::numeric_limits<double>::epsilon())
                      ? static_cast<T>(
                            boost::math::cyl_bessel_i(0, kaiser_arg) / i0_beta)
                      : T{0.0};
          }
          break;
        }
        case WindowType::Gaussian: {
          if (setup.length == 1) {
            val = T{1.0};
          }
          else {
            double center = N_minus_1 / 2.0;
            double denom = gaussian_sigma * center;
            double term = (denom > std::numeric_limits<double>::epsilon())
                              ? (n_double - center) / denom
                              : 0.0;
            val = static_cast<T>(std::exp(-0.5 * term * term));
          }
          break;
        }
        default:
          // This should not happen if constructor validated the type.
          logger->error(
              "generate_window: Reached default case in main switch with "
              "unknown or unvalidated window type: {}",
              static_cast<int>(setup.type));
          return std::unexpected(
              Status::InvalidArgument);  // Should be caught by constructor
      }
      output[static_cast<size_t>(i)] = val;
    }
    return {};  // Success
  }

}  // namespace OmniDSP

#endif  // OMNIDSP_WINDOW_HPP
