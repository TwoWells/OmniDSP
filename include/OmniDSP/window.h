/**
 * @file window.h
 * @brief Public API header for windowing functions in OmniDSP.
 *
 * Declares the OmniDSP::Window class with static methods for generating
 * window coefficients or applying window functions to input signals.
 *
 * @version 2.1.0 (Added coefficient generation methods)
 * @date 2025-04-18
 */
#ifndef OMNIDSP_WINDOW_H
#define OMNIDSP_WINDOW_H

#include <cmath>  // For math functions used in window formulas (e.g., cos, sqrt, Bessel I0)
#include <cstddef>    // For size_t
#include <limits>     // For std::numeric_limits
#include <stdexcept>  // For std::invalid_argument
#include <vector>     // For std::vector return types and inputs

// Forward declare backend functions (optional, but can help clarity)
// namespace OmniDSP { namespace Backend {
//     template <typename T> std::vector<T> generate_hann_window_impl(size_t
//     length);
//     // ... other declarations ...
// }}

namespace OmniDSP {

/**
 * @brief Provides static methods for generating or applying window functions.
 *
 * Windowing functions are typically applied to input data (e.g., before an FFT)
 * to reduce spectral leakage. This class provides methods both for generating
 * the window coefficients directly and for applying them element-wise to an
 * input signal. Implementations rely on the selected backend (oneMKL,
 * Accelerate, or Stub).
 */
class Window {
 public:
  // --- Methods for APPLYING windows ---

  /**
   * @brief Applies the Hann window to the input data.
   * w[n] = 0.5 - 0.5 * cos(2*pi*n / (N-1))
   *
   * @tparam T The floating-point type (float or double).
   * @param input A vector of input data.
   * @return A vector containing the input data multiplied by the Hann window.
   * @throws std::invalid_argument If the input vector is empty.
   * @throws std::runtime_error If the backend window generation/application
   * fails.
   */
  template <typename T>
  static std::vector<T> hann(const std::vector<T>& input);

  /**
   * @brief Applies the Hamming window to the input data.
   * w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
   *
   * @tparam T The floating-point type (float or double).
   * @param input A vector of input data.
   * @return A vector containing the input data multiplied by the Hamming
   * window.
   * @throws std::invalid_argument If the input vector is empty.
   * @throws std::runtime_error If the backend window generation/application
   * fails.
   */
  template <typename T>
  static std::vector<T> hamming(const std::vector<T>& input);

  /**
   * @brief Applies the Kaiser window to the input data.
   * Defined using the zeroth-order modified Bessel function I0.
   * w[n] = I0(beta * sqrt(1 - (2n/(N-1) - 1)^2)) / I0(beta)
   *
   * @tparam T The floating-point type (float or double).
   * @param input A vector of input data.
   * @param beta  The shape parameter beta (non-negative). Typical values range
   * from 5 to 10.
   * @return A vector containing the input data multiplied by the Kaiser window.
   * @throws std::invalid_argument If the input vector is empty or beta is
   * negative.
   * @throws std::runtime_error If the backend window generation/application
   * fails.
   */
  template <typename T>
  static std::vector<T> kaiser(const std::vector<T>& input, T beta);

  /**
   * @brief Applies the Flat-top window to the input data.
   * Designed for accurate amplitude measurements in the frequency domain.
   * Uses standard 5-term coefficients.
   *
   * @tparam T The floating-point type (float or double).
   * @param input A vector of input data.
   * @return A vector containing the input data multiplied by the Flat-top
   * window.
   * @throws std::invalid_argument If the input vector is empty.
   * @throws std::runtime_error If the backend window generation/application
   * fails.
   */
  template <typename T>
  static std::vector<T> flattop(const std::vector<T>& input);

  // Add applying methods for Blackman, Bartlett etc. if they exist or are
  // needed.

  // --- Methods for GENERATING window coefficients ---

  /**
   * @brief Generates Hann window coefficients.
   * w[n] = 0.5 - 0.5 * cos(2*pi*n / (N-1))
   *
   * @tparam T The floating-point type (float or double).
   * @param length The desired number of window coefficients (N).
   * @return A vector containing the window coefficients. Returns empty if
   * length is 0.
   * @throws std::invalid_argument If length is negative (though size_t prevents
   * this).
   * @throws std::runtime_error If the backend window generation fails.
   */
  template <typename T>
  static std::vector<T> get_hann_coeffs(size_t length);

  /**
   * @brief Generates Hamming window coefficients.
   * w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
   *
   * @tparam T The floating-point type (float or double).
   * @param length The desired number of window coefficients (N).
   * @return A vector containing the window coefficients. Returns empty if
   * length is 0.
   * @throws std::invalid_argument If length is negative.
   * @throws std::runtime_error If the backend window generation fails.
   */
  template <typename T>
  static std::vector<T> get_hamming_coeffs(size_t length);

  /**
   * @brief Generates Kaiser window coefficients.
   * w[n] = I0(beta * sqrt(1 - (2n/(N-1) - 1)^2)) / I0(beta)
   *
   * @tparam T The floating-point type (float or double).
   * @param length The desired number of window coefficients (N).
   * @param beta The shape parameter beta (non-negative).
   * @return A vector containing the window coefficients. Returns empty if
   * length is 0.
   * @throws std::invalid_argument If length is negative or beta is negative.
   * @throws std::runtime_error If the backend window generation fails.
   */
  template <typename T>
  static std::vector<T> get_kaiser_coeffs(size_t length, T beta);

  /**
   * @brief Generates Flat-top window coefficients.
   * Uses standard 5-term coefficients.
   *
   * @tparam T The floating-point type (float or double).
   * @param length The desired number of window coefficients (N).
   * @return A vector containing the window coefficients. Returns empty if
   * length is 0.
   * @throws std::invalid_argument If length is negative.
   * @throws std::runtime_error If the backend window generation fails.
   */
  template <typename T>
  static std::vector<T> get_flattop_coeffs(size_t length);

  // Add get_coeffs methods for Blackman, Bartlett etc. corresponding to the
  // applying methods.

  // Note: Explicit template instantiations are defined in window.cpp
};

}  // namespace OmniDSP

#endif  // OMNIDSP_WINDOW_H
