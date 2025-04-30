/**
 * @file window.hpp
 * @brief Defines windowing function types and the WindowSpec class for
 * configuration.
 */

#ifndef OMNIDSP_WINDOW_H
#define OMNIDSP_WINDOW_H

#include <algorithm>    // For std::fill, std::max
#include <cmath>        // For std::abs, std::cos, std::sqrt, std::pow, std::exp
#include <cstddef>      // For size_t
#include <limits>       // For numeric_limits
#include <numbers>      // *** ADDED: For std::numbers constants ***
#include <optional>     // For optional parameters like beta
#include <span>         // For std::span
#include <stdexcept>    // For invalid_argument
#include <string>       // For exception messages
#include <string_view>  // For get_window_type_name

#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT
#include "core_types.hpp"              // For Status, F32, F64 etc.

namespace OmniDSP {

  /**
   * @brief Enumeration of standard window function types supported by OmniDSP.
   */
  enum class WindowType {
    Rectangular,
    Bartlett,
    Hann,
    Hamming,
    Blackman,
    Flattop,
    Gaussian,  // Requires stddev > 0
    Kaiser,    // Requires beta >= 0
    Triangular
  };

  /**
   * @brief Gets the string name corresponding to a WindowType enum value.
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
   * This version is non-templated and uses double for parameters.
   */
  class OMNIDSP_EXPORT WindowSpec {  // Removed template<typename T>
   public:
    /**
     * @brief Default constructor. Creates a specification for a Rectangular
     * window.
     */
    WindowSpec() : type_(WindowType::Rectangular) {}

    /**
     * @brief Constructor for window types that do not require parameters.
     * @param type The desired window type (must NOT be Gaussian or Kaiser).
     * @throws std::invalid_argument if the specified type requires parameters.
     */
    explicit WindowSpec(WindowType type) : type_(type)
    {
      if (type == WindowType::Gaussian || type == WindowType::Kaiser) {
        throw std::invalid_argument(
             std::string(get_window_type_name(type)) +
             " window requires a parameter. Use WindowSpec(WindowType, param) "
             "or the static factory methods (Gaussian/Kaiser).");
      }
      // No parameters to store for these types
    }

    /**
     * @brief Constructor for window types that require exactly one parameter.
     * @param type The desired window type (must be Gaussian or Kaiser).
     * @param param The parameter value (stddev for Gaussian, beta for Kaiser).
     * @throws std::invalid_argument if the type is not Gaussian or Kaiser,
     * or if the parameter value is invalid (stddev <= 0, beta < 0).
     */
    WindowSpec(WindowType type, double param)
        : type_(type)  // Parameter is double
    {
      if (type == WindowType::Gaussian) {
        if (param <= 0.0) {  // Compare with double literal
          throw std::invalid_argument(
              "Gaussian window standard deviation (param) must be positive.");
        }
        param_ = param;  // Store in the single optional param_
      }
      else if (type == WindowType::Kaiser) {
        if (param < 0.0) {  // Compare with double literal
          throw std::invalid_argument(
              "Kaiser window beta parameter (param) must be non-negative.");
        }
        param_ = param;  // Store in the single optional param_
      }
      else {
        throw std::invalid_argument(
            std::string(get_window_type_name(type))
            + " window does not take a parameter. Use WindowSpec(WindowType).");
      }
    }

    /**
     * @brief Static factory method to create a Gaussian window specification.
     * @param stddev Standard deviation (sigma). Must be > 0.
     * @return A WindowSpec object configured for a Gaussian window.
     * @throws std::invalid_argument if stddev is not positive.
     */
    static WindowSpec Gaussian(double stddev)  // Parameter is double
    {
      // Validation is handled by the two-parameter constructor
      return WindowSpec(WindowType::Gaussian, stddev);
    }

    /**
     * @brief Static factory method to create a Kaiser window specification.
     * @param beta Shape parameter beta. Must be >= 0.
     * @return A WindowSpec object configured for a Kaiser window.
     * @throws std::invalid_argument if beta is negative.
     */
    static WindowSpec Kaiser(double beta)  // Parameter is double
    {
      // Validation is handled by the two-parameter constructor
      return WindowSpec(WindowType::Kaiser, beta);
    }

    /**
     * @brief Gets the type of the window specified by this object.
     */
    [[nodiscard]] WindowType get_type() const noexcept { return type_; }

    /**
     * @brief Gets the parameter value if the window type requires one (Gaussian
     * or Kaiser).
     * @return The parameter value (stddev or beta) as a double, or std::nullopt
     * if not applicable or not set.
     */
    [[nodiscard]] std::optional<double> get_param() const noexcept
    {
      return param_;
    }

    /**
     * @brief Validates the WindowSpec consistency.
     * @return True if the spec is valid (e.g., parameter present if required),
     * false otherwise.
     */
    [[nodiscard]] bool validate() const
    {
      switch (type_) {
        case WindowType::Kaiser:
        case WindowType::Gaussian:
          // These types require a parameter.
          return param_.has_value();
        case WindowType::Rectangular:
        case WindowType::Bartlett:
        case WindowType::Hann:
        case WindowType::Hamming:
        case WindowType::Blackman:
        case WindowType::Flattop:
        case WindowType::Triangular:
          // These types should not have a parameter set, but we can be lenient.
          // Stricter validation could return !param_.has_value() here.
          return true;
        default:
          return false;  // Unknown type
      }
    }

    /**
     * @brief Compares two WindowSpec objects for equality.
     */
    bool operator==(const WindowSpec& other) const noexcept
    {
      // Basic type check
      if (type_ != other.type_) return false;

      // Parameter check (only if type requires it)
      if (type_ == WindowType::Gaussian || type_ == WindowType::Kaiser) {
        // If one has value and other doesn't, they are not equal
        if (param_.has_value() != other.param_.has_value()) return false;
        // If both have values, compare them (handle potential floating point
        // issues)
        if (param_.has_value()) {
          // Use a small relative epsilon for comparison
          constexpr double epsilon = std::numeric_limits<double>::epsilon()
                                     * 100;  // Adjust multiplier as needed
          double diff = std::abs(param_.value() - other.param_.value());
          double max_val = std::max(
              std::abs(param_.value()), std::abs(other.param_.value()));
          // Handle comparison near zero separately
          if (max_val < epsilon) {
            return diff < epsilon;
          }
          return diff <= epsilon * max_val;
        }
        // If neither has value (shouldn't happen if type requires param due to
        // constructor checks, but safe)
        return true;
      }
      // Types match and no parameters needed/relevant for this type
      return true;
    }

    /**
     * @brief Compares two WindowSpec objects for inequality.
     */
    bool operator!=(const WindowSpec& other) const noexcept
    {
      return !(*this == other);
    }

   private:
    WindowType type_;
    std::optional<double> param_
        = std::nullopt;  // Single parameter for Gaussian/Kaiser
  };

  // --- Function Declarations (Example - If needed later) ---

  /**
   * @brief Generates window coefficients based on the specification.
   *
   * @tparam T The floating-point type (float or double) for the *output
   * coefficients*.
   * @param spec The non-templated window specification.
   * @param length The desired length of the window.
   * @param output The span to write the window coefficients into. Must have
   * size `length`.
   * @return Status indicating success or failure.
   */
  template <typename T>
  Status generate_window(
      const WindowSpec& spec, size_t length, std::span<T> output);

  // Add declarations for any other window-related functions...

  // --- Template Implementations (Example for generate_window - Keep details
  // internal) ---

  namespace Detail {  // Keep implementation details internal if possible

    // Helper for Kaiser window calculation
    inline double bessel_i0(double x)
    {
      double sum = 1.0;
      double term = 1.0;
      double x_half_sq = (x / 2.0) * (x / 2.0);
      int k = 1;
      // Iterate until term is negligible relative to sum or machine epsilon
      while (std::abs(term)
                 > std::abs(sum) * std::numeric_limits<double>::epsilon() * 10.0
             && k < 100) {
        term *= x_half_sq / (static_cast<double>(k) * static_cast<double>(k));
        sum += term;
        k++;
      }
      return sum;
    }

  }  // namespace Detail

  template <typename T>
  Status generate_window(
      const WindowSpec& spec, size_t length, std::span<T> output)
  {
    if (length == 0) {
      return Status::Success;  // Nothing to do
    }
    if (output.size() != length) {
      return Status::SizeMismatch;
    }
    if (!spec.validate()) {
      // This check ensures param is present if type is Kaiser/Gaussian
      return Status::InvalidArgument;
    }

    const double N = static_cast<double>(length);
    const double N_minus_1 = N - 1.0;  // Can be 0 if length is 1

    // Pre-calculate common terms using std::numbers
    constexpr double pi
        = std::numbers::pi_v<double>;  // *** Use std::numbers ***
    const double two_pi = 2.0 * pi;
    const double four_pi = 4.0 * pi;
    const double six_pi = 6.0 * pi;
    const double eight_pi = 8.0 * pi;

    for (size_t n = 0; n < length; ++n) {
      double n_double = static_cast<double>(n);
      T val = T{0.0};  // Initialize value for this point

      switch (spec.get_type()) {
        case WindowType::Rectangular:
          val = T{1.0};
          break;

        case WindowType::Bartlett:
        case WindowType::Triangular:  // Bartlett and Triangular are often the
                                      // same for odd N
          if (length == 1) {
            val = T{1.0};
          }
          else {
            // Avoid division by zero if N_minus_1 is 0 (length == 1)
            double denom = N_minus_1 / 2.0;
            val = static_cast<T>(
                1.0 - std::abs(n_double - denom) / (denom > 0 ? denom : 1.0));
          }
          break;

        case WindowType::Hann:
          if (length == 1) {
            val = T{1.0};
          }
          else {
            val = static_cast<T>(
                0.5 * (1.0 - std::cos(two_pi * n_double / N_minus_1)));
          }
          break;

        case WindowType::Hamming:
          if (length == 1) {
            val = T{1.0};
          }
          else {
            val = static_cast<T>(
                0.54 - 0.46 * std::cos(two_pi * n_double / N_minus_1));
          }
          break;

        case WindowType::Blackman:
          if (length == 1) {
            val = T{1.0};
          }
          else {
            val = static_cast<T>(
                0.42 - 0.50 * std::cos(two_pi * n_double / N_minus_1)
                + 0.08
                      * std::cos(
                          four_pi * n_double / N_minus_1));  // Use four_pi
          }
          break;

        case WindowType::Flattop:
          // Standard coefficients for flat top window (e.g., from MATLAB)
          if (length == 1) {
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
                + a2 * std::cos(four_pi * n_double / N_minus_1)  // Use four_pi
                - a3 * std::cos(six_pi * n_double / N_minus_1)   // Use six_pi
                + a4
                      * std::cos(
                          eight_pi * n_double / N_minus_1));  // Use eight_pi
          }
          break;

        case WindowType::Kaiser: {
          // Parameter should be present due to validate() check
          double beta = spec.get_param().value();
          if (length == 1) {
            val = T{1.0};
          }
          else {
            double term_sq = std::pow((2.0 * n_double / N_minus_1 - 1.0), 2.0);
            // Ensure argument to sqrt is non-negative due to potential floating
            // point issues
            double sqrt_arg = std::max(0.0, 1.0 - term_sq);
            double kaiser_arg = beta * std::sqrt(sqrt_arg);
            double i0_beta = Detail::bessel_i0(beta);
            // Avoid division by zero if i0_beta is somehow zero
            val = (i0_beta > 0)
                      ? static_cast<T>(Detail::bessel_i0(kaiser_arg) / i0_beta)
                      : T{0.0};
          }
          break;
        }

        case WindowType::Gaussian: {
          // Parameter should be present due to validate() check
          // Assuming param = stddev (sigma)
          double sigma = spec.get_param().value();
          if (length == 1) {
            val = T{1.0};
          }
          else {
            double center = N_minus_1 / 2.0;
            // Standard deviation relative to half window width is often used.
            // Let's assume the parameter 'sigma' is this relative std dev.
            // If sigma = 0.5, it reaches near zero at edges.
            // double term = (n_double - center) / (sigma * center);
            // Let's assume sigma is the parameter as defined in
            // scipy.signal.windows.gaussian where the window is exp(-0.5 * ( (n
            // - (N-1)/2) / (sigma * (N-1)/2) )**2)
            double denom = sigma * center;
            // Avoid division by zero if center is 0 (length == 1)
            double term = (denom > 0) ? (n_double - center) / denom : 0.0;
            val = static_cast<T>(std::exp(-0.5 * term * term));
          }
          break;
        }

        default:
          return Status::InvalidArgument;  // Should not happen if enum is
                                           // exhaustive
      }
      output[n] = val;
    }

    return Status::Success;
  }

}  // namespace OmniDSP

#endif  // OMNIDSP_WINDOW_H
