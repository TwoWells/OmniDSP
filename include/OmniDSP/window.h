/**
 * @file window.h
 * @brief Defines windowing function types and the WindowSpec class for
 * configuration.
 * @details Window functions are commonly used in signal processing for tasks
 * like spectral analysis (reducing spectral leakage) and FIR filter design.
 * This header provides the necessary types and structures to specify and
 * generate various standard window functions via the OmniDSP class.
 */

#ifndef OMNIDSP_WINDOW_H
#define OMNIDSP_WINDOW_H

#include <cmath>        // For std::abs needed in operator==
#include <cstddef>      // For size_t
#include <optional>     // For optional parameters like beta
#include <stdexcept>    // For invalid_argument
#include <string_view>  // For get_window_type_name

#include "core_types.h"  // For RealT

namespace OmniDSP {

  /**
   * @brief Enumeration of standard window function types supported by OmniDSP.
   * @details Each enum value corresponds to a specific mathematical window
   * function. The choice of window depends on the application requirements
   * regarding trade-offs like main lobe width, side lobe attenuation, and
   * scalloping loss.
   */
  enum class WindowType {
    Rectangular,  ///< Rectangular (Boxcar/Dirichlet) window. Equivalent to no
                  ///< window. Maximum spectral leakage.
    Bartlett,     ///< Bartlett (triangular) window with non-zero endpoints.
    Hann,     ///< Hann (or Hanning) window. Good frequency resolution, moderate
              ///< leakage.
    Hamming,  ///< Hamming window. Optimized to minimize the maximum (nearest)
              ///< side lobe. Slightly wider main lobe than Hann.
    Blackman,   ///< Blackman window. Better side lobe attenuation than
                ///< Hann/Hamming, but wider main lobe.
    Flattop,    ///< Flattop window. Very low amplitude error (flat top in
                ///< frequency domain), but very wide main lobe. Good for
                ///< amplitude measurements.
    Gaussian,   ///< Gaussian window. Requires a standard deviation parameter
                ///< (sigma). Time-frequency uncertainty principle applies.
    Kaiser,     ///< Kaiser window. Flexible window defined by a shape parameter
                ///< (beta). Can approximate many other windows. Good trade-off
                ///< control.
    Triangular  ///< Triangular window with zero endpoints (for length > 1).
                ///< Similar to Bartlett but often defined with zero endpoints.
                // Add others like Blackman-Harris, Nuttall, etc. if needed
  };

  /**
   * @brief Gets the string name corresponding to a WindowType enum value.
   * @param type The window type enum value.
   * @return A string_view representing the window name. Useful for logging or
   * display.
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
   * @brief Specification class for creating a window function.
   * @details This class encapsulates the type of window and any necessary
   * parameters (like standard deviation for Gaussian or beta for Kaiser). It is
   * used as input to OmniDSP window generation methods (e.g.,
   * `OmniDSP::hann_window`, `OmniDSP::kaiser_window`) or within other
   * specification objects like `FIRFilterSpec`. Provides static factory methods
   * for creating specs for windows that require parameters.
   * @tparam T The floating-point type (e.g., float, double) for window
   * parameters.
   */
  template <typename T> class WindowSpec {
   public:
    /**
     * @brief Default constructor. Creates a specification for a Hann window.
     */
    WindowSpec() : type_(WindowType::Hann) {}

    /**
     * @brief Constructor for basic window types that do not require parameters.
     * @param type The desired window type (e.g., Hann, Hamming, Blackman,
     * Rectangular, etc.).
     * @throws std::invalid_argument if the specified type requires parameters
     * (currently Gaussian or Kaiser). Use the static factory methods instead.
     */
    explicit WindowSpec(WindowType type) : type_(type)
    {
      if (type == WindowType::Gaussian || type == WindowType::Kaiser) {
        // Ensure users employ the static methods for parameter validation.
        throw std::invalid_argument(
            "Gaussian and Kaiser windows require parameters. Use "
            "WindowSpec::Gaussian(stddev) or WindowSpec::Kaiser(beta).");
      }
    }

    /**
     * @brief Static factory method to create a Gaussian window specification.
     * @param stddev Standard deviation (sigma) parameter controlling the window
     * width. Must be greater than 0. Smaller values result in a narrower window
     * in time domain (wider in frequency domain).
     * @return A WindowSpec object configured for a Gaussian window.
     * @throws std::invalid_argument if stddev is not positive.
     */
    static WindowSpec<T> Gaussian(RealT<T> stddev)
    {
      if (stddev <= static_cast<RealT<T>>(0.0)) {
        throw std::invalid_argument(
            "Gaussian window standard deviation must be positive.");
      }
      // Use the private constructor to set the type and parameter
      return WindowSpec<T>(WindowType::Gaussian, stddev, std::nullopt);
    }

    /**
     * @brief Static factory method to create a Kaiser window specification.
     * @param beta Shape parameter beta. Must be non-negative (>= 0).
     * Controls the trade-off between main lobe width and side lobe attenuation.
     * beta=0 yields a Rectangular window. Larger beta increases side lobe
     * attenuation but also widens the main lobe. Common values range from 4
     * to 8.
     * @return A WindowSpec object configured for a Kaiser window.
     * @throws std::invalid_argument if beta is negative.
     */
    static WindowSpec<T> Kaiser(RealT<T> beta)
    {
      if (beta < static_cast<RealT<T>>(0.0)) {
        throw std::invalid_argument(
            "Kaiser window beta parameter must be non-negative.");
      }
      // Use the private constructor to set the type and parameter
      return WindowSpec<T>(WindowType::Kaiser, std::nullopt, beta);
    }

    /**
     * @brief Gets the type of the window specified by this object.
     * @return The WindowType enum value.
     */
    WindowType get_type() const { return type_; }

    /**
     * @brief Gets the standard deviation parameter for the Gaussian window.
     * @return An optional containing the standard deviation if the type is
     * Gaussian, otherwise std::nullopt.
     */
    std::optional<RealT<T>> get_stddev() const { return stddev_; }

    /**
     * @brief Gets the beta parameter for the Kaiser window.
     * @return An optional containing the beta value if the type is Kaiser,
     * otherwise std::nullopt.
     */
    std::optional<RealT<T>> get_beta() const { return beta_; }

    /**
     * @brief Compares two WindowSpec objects for equality.
     * @details Checks if both the type and relevant parameters (if any) match.
     * Uses approximate comparison for floating-point parameters.
     * @param other The WindowSpec object to compare against.
     * @return True if the specifications are equivalent, false otherwise.
     */
    bool operator==(const WindowSpec<T>& other) const
    {
      if (type_ != other.type_) return false;
      // Only compare parameters if they are relevant for the type
      if (type_ == WindowType::Gaussian) {
        // Both must have a value and they must be approximately equal
        return stddev_.has_value() && other.stddev_.has_value()
               && (std::abs(stddev_.value() - other.stddev_.value())
                   < static_cast<RealT<T>>(1e-9));
      }
      if (type_ == WindowType::Kaiser) {
        // Both must have a value and they must be approximately equal
        return beta_.has_value() && other.beta_.has_value()
               && (std::abs(beta_.value() - other.beta_.value())
                   < static_cast<RealT<T>>(1e-9));
      }
      // Types match and no parameters needed/relevant for this type
      return true;
    }

    /**
     * @brief Compares two WindowSpec objects for inequality.
     * @param other The WindowSpec object to compare against.
     * @return True if the specifications are different, false otherwise.
     */
    bool operator!=(const WindowSpec<T>& other) const
    {
      return !(*this == other);
    }

   private:
    /**
     * @brief Private constructor used internally by static factory methods.
     * @param type The window type.
     * @param stddev Optional standard deviation for Gaussian window.
     * @param beta Optional beta parameter for Kaiser window.
     */
    explicit WindowSpec(
        WindowType type,
        std::optional<RealT<T>> stddev = std::nullopt,
        std::optional<RealT<T>> beta = std::nullopt)
        : type_(type), stddev_(stddev), beta_(beta)
    {}

    WindowType type_;  ///< The type of window function.
    std::optional<RealT<T>> stddev_
        = std::nullopt;  ///< Standard deviation (for Gaussian).
    std::optional<RealT<T>> beta_
        = std::nullopt;  ///< Shape parameter (for Kaiser).
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_WINDOW_H
