/**
 * @file window.h
 * @brief Defines structures for specifying window types and parameters.
 * @details Window generation itself is handled by methods within the OmniDSP
 * class (e.g., OmniDSP::hann_window). This header provides ways to specify
 * which window should be used by other components (like CQT or STFT plans).
 */

#ifndef OMNIDSP_WINDOW_H
#define OMNIDSP_WINDOW_H

#include <optional>   // For std::optional (requires C++17)
#include <stdexcept>  // For std::invalid_argument
#include <string>     // For exception messages

#include "core_types.h"  // Core types like RealT, Window enum, and now get_window_name

namespace OmniDSP {

/**
 * @brief Structure to specify a window type and its associated parameters.
 * @details Used to configure components like CQT or STFT that require
 * windowing. This struct acts as a stateless parameter object, bundling the
 * window type (from the Window enum) and any relevant parameters (like beta for
 * Kaiser or stddev for Gaussian) into a single entity. The constructors ensure
 * that initially created objects represent a valid state. The object is
 * immutable after construction (members are private with const getters).
 *
 * **Design Rationale vs. Plan Objects:**
 * Unlike stateful Plan objects (e.g., FFTPlan, CQTPlan) which encapsulate
 * backend-specific resources and are created via OmniDSP factory methods,
 * WindowSpec is simply configuration data. It does not require backend context
 * for its creation and is typically instantiated directly by the user using its
 * constructors. This approach was chosen over alternatives (like an Abstract
 * Base Class hierarchy) for simplicity, as its primary role is to provide clean
 * parameter passing to functions or Plan factories that *do* depend on the
 * OmniDSP instance's backend. The actual generation of window coefficients is
 * performed by methods on the OmniDSP instance (e.g., `dsp.kaiser_window(len,
 * beta)`), which aligns with the philosophy of centralizing backend-dependent
 * operations.
 *
 * @tparam T The underlying floating-point type (e.g., float, double) expected
 * for parameters.
 */
template <typename T>
class WindowSpec {  // Changed to class to enforce private members

 public:
  /**
   * @brief Default constructor (creates a Hann window spec).
   */
  WindowSpec() : type_(Window::Hann) {}  // Initialize private member

  /**
   * @brief Constructor for windows **without** parameters.
   * @param window_type The desired window type (e.g., Window::Hann,
   * Window::Rectangular).
   * @throws std::invalid_argument if the type requires parameters (Kaiser,
   * Gaussian).
   */
  explicit WindowSpec(Window window_type) : type_(window_type) {
    if (type_ == Window::Kaiser || type_ == Window::Gaussian) {
      // Use the get_window_name function now defined in core_types.h
      throw std::invalid_argument("Window type '" +
                                  std::string(get_window_name(type_)) +
                                  "' requires parameters. Use the constructor "
                                  "WindowSpec(Window, RealT<T>).");
    }
    // No parameters to set for other types
  }

  /**
   * @brief Constructor for windows **with** a single RealT<T> parameter (Kaiser
   * or Gaussian).
   * @param window_type The desired window type (must be Window::Kaiser or
   * Window::Gaussian).
   * @param param The parameter value (beta for Kaiser, stddev for Gaussian).
   * @throws std::invalid_argument if window_type is not Kaiser or Gaussian, or
   * if param is invalid for the type.
   */
  WindowSpec(Window window_type, RealT<T> param) : type_(window_type) {
    if (type_ == Window::Kaiser) {
      if (param < static_cast<RealT<T>>(0.0)) {
        throw std::invalid_argument(
            "Kaiser window beta parameter must be non-negative.");
      }
      beta_ = param;  // Set private member
    } else if (type_ == Window::Gaussian) {
      if (param <= static_cast<RealT<T>>(0.0)) {
        throw std::invalid_argument(
            "Gaussian window standard deviation must be positive.");
      }
      stddev_ = param;  // Set private member
    } else {
      // Use the get_window_name function now defined in core_types.h
      throw std::invalid_argument(
          "Window type '" + std::string(get_window_name(type_)) +
          "' does not take a parameter in this constructor. Use "
          "WindowSpec(Window) instead.");
    }
  }

  // --- Getters for accessing specification details ---

  /** @brief Gets the type of window function specified. */
  Window get_type() const noexcept { return type_; }

  /** @brief Gets the beta parameter if specified (only valid for Kaiser). */
  std::optional<RealT<T>> get_beta() const noexcept { return beta_; }

  /** @brief Gets the standard deviation parameter if specified (only valid for
   * Gaussian). */
  std::optional<RealT<T>> get_stddev() const noexcept { return stddev_; }

 private:
  /// @brief The type of window function. (Private)
  Window type_ = Window::Hann;

  /// @brief Optional beta parameter, required only for Window::Kaiser.
  /// (Private)
  std::optional<RealT<T>> beta_ = std::nullopt;

  /// @brief Optional standard deviation parameter, required only for
  /// Window::Gaussian. (Private)
  std::optional<RealT<T>> stddev_ = std::nullopt;

};  // End of class WindowSpec

// Note: The old standalone `generate_window` function is removed.
// Window coefficients are now generated via methods on an OmniDSP instance,
// e.g.: auto dsp = OmniDSP::create().value(); auto hann_coeffs =
// dsp.hann_window<double>(1024); auto kaiser_coeffs =
// dsp.kaiser_window<double>(1024, 8.0);
//
// WindowSpec is used like this:
// WindowSpec<double> spec1(Window::Hann);
// WindowSpec<double> spec2(Window::Kaiser, 8.0);
// WindowSpec<double> spec3(Window::Gaussian, 0.5);
// // Pass spec1, spec2, or spec3 to functions needing window configuration.
// // Access values via getters: spec2.get_type(), spec2.get_beta()

}  // namespace OmniDSP

#endif  // OMNIDSP_WINDOW_H
