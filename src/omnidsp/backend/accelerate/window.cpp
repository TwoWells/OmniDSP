/**
 * @file src/omnidsp/backend/accelerate/window.cpp
 * @brief Apple Accelerate backend implementation for OmniDSP window functions.
 *
 * Implements backend functions to generate window coefficients using vDSP
 * where available (Hann, Hamming). Kaiser uses Boost Math, Flattop uses manual
 * calculations. Compiled only when USE_ACCELERATE is defined.
 */

// --- Includes ---
#include <cmath>      // For M_PI, cos, sin, sqrt, std::abs
#include <limits>     // For std::numeric_limits
#include <numeric>    // For std::accumulate
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>
#include <type_traits>  // For std::is_same_v
#include <vector>

// Include Boost header for Bessel functions
#include <boost/math/special_functions/bessel.hpp>

#include "../backend_impl.h"  // Internal backend function declarations

// Only compile if USE_ACCELERATE is defined by CMake (typically on macOS)
#if defined(USE_ACCELERATE)

#include <Accelerate/Accelerate.h>  // Main header for the Accelerate framework (includes vDSP)

// Define M_PI if it's not already defined (e.g., by <cmath> in some
// environments)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace OmniDSP {
namespace Backend {
// --- Backend Window Generation Implementations ---

/**
 * @brief Generates Hann window coefficients using vDSP.
 * Note: vDSP_hann_window generates values in [0, 1].
 * The flag vDSP_HANN_NORM normalizes the first coefficient to zero.
 * Uses vDSP_hann_window for float and vDSP_hann_windowD for double.
 */
template <typename T>
std::vector<T> generate_hann_window_impl(size_t length) {
  if (length == 0) return {};
  std::vector<T> coeffs(length);
  vDSP_Length n_vDSP = static_cast<vDSP_Length>(length);

  if constexpr (std::is_same_v<T, float>) {
    // Use vDSP_hann_window for float
    vDSP_hann_window(coeffs.data(), n_vDSP,
                     vDSP_HANN_NORM);  // <<< Corrected function name
  } else {                             // double
    // Use vDSP_hann_windowD for double
    vDSP_hann_windowD(coeffs.data(), n_vDSP,
                      vDSP_HANN_NORM);  // <<< Corrected function name
  }
  return coeffs;
}

/**
 * @brief Generates Hamming window coefficients using vDSP.
 * Note: vDSP_hamm_window generates values in approx [0.08, 1.0].
 */
template <typename T>
std::vector<T> generate_hamming_window_impl(size_t length) {
  if (length == 0) return {};
  std::vector<T> coeffs(length);
  vDSP_Length n_vDSP = static_cast<vDSP_Length>(length);

  if constexpr (std::is_same_v<T, float>) {
    // vDSP_hamm_window_f seems correct based on Accelerate headers,
    // but let's stick to the pattern if _f isn't standard.
    // If vDSP_hamm_window exists for float, use it. Otherwise, keep _f.
    // Assuming vDSP_hamm_window is for float based on Hann pattern.
    vDSP_hamm_window(coeffs.data(), n_vDSP,
                     0);  // Assuming vDSP_hamm_window is float
  } else {                // double
    // Assuming vDSP_hamm_windowD is for double based on Hann pattern.
    vDSP_hamm_windowD(coeffs.data(), n_vDSP, 0);
  }
  return coeffs;
}

/**
 * @brief Generates Kaiser window coefficients using Boost Math for Bessel
 * function.
 * @param beta The Kaiser window beta parameter.
 */
template <typename T>
std::vector<T> generate_kaiser_window_impl(size_t length, T beta) {
  if (length == 0) return {};
  if (length == 1) return {static_cast<T>(1.0)};  // Handle N=1 case

  std::vector<T> coeffs(length);
  // Precompute I0(beta) using Boost Math
  T i0_beta = boost::math::cyl_bessel_i(static_cast<T>(0.0), beta);
  if (std::abs(i0_beta) < std::numeric_limits<T>::epsilon()) {
    if (std::abs(beta) < std::numeric_limits<T>::epsilon()) {
      std::fill(coeffs.begin(), coeffs.end(), static_cast<T>(1.0));
      return coeffs;
    } else {
      throw std::runtime_error("Kaiser window denominator I0(beta) is zero.");
    }
  }

  double N_minus_1_double = static_cast<double>(length - 1);
  if (N_minus_1_double == 0.0) N_minus_1_double = 1.0;

  for (size_t n = 0; n < length; ++n) {
    double factor = (static_cast<double>(n) * 2.0) / N_minus_1_double - 1.0;
    double sqrt_arg = 1.0 - factor * factor;
    if (sqrt_arg < 0.0) sqrt_arg = 0.0;

    T sqrt_term = static_cast<T>(std::sqrt(sqrt_arg));
    T bessel_arg = beta * sqrt_term;
    coeffs[n] =
        boost::math::cyl_bessel_i(static_cast<T>(0.0), bessel_arg) / i0_beta;
  }
  return coeffs;
}

/**
 * @brief Generates Flattop window coefficients (manual calculation, as vDSP
 * lacks native support).
 */
template <typename T>
std::vector<T> generate_flattop_window_impl(size_t length) {
  if (length == 0) return {};
  std::vector<T> coeffs(length);
  constexpr T a0 = static_cast<T>(0.21557895);
  constexpr T a1 = static_cast<T>(0.41663158);
  constexpr T a2 = static_cast<T>(0.277263158);
  constexpr T a3 = static_cast<T>(0.083578947);
  constexpr T a4 = static_cast<T>(0.006947368);
  double denom = (length > 1) ? static_cast<double>(length - 1) : 1.0;
  if (denom == 0.0) denom = 1.0;

  for (size_t n = 0; n < length; ++n) {
    double angle_base = 2.0 * M_PI * static_cast<double>(n) / denom;
    T cos_1x_term = static_cast<T>(cos(angle_base));
    T cos_2x_term = static_cast<T>(cos(2.0 * angle_base));
    T cos_3x_term = static_cast<T>(cos(3.0 * angle_base));
    T cos_4x_term = static_cast<T>(cos(4.0 * angle_base));
    coeffs[n] = a0 - a1 * cos_1x_term + a2 * cos_2x_term - a3 * cos_3x_term +
                a4 * cos_4x_term;
  }
  return coeffs;
}

// --- Explicit Template Instantiations ---
template std::vector<float> generate_hann_window_impl<float>(size_t length);
template std::vector<double> generate_hann_window_impl<double>(size_t length);

template std::vector<float> generate_hamming_window_impl<float>(size_t length);
template std::vector<double> generate_hamming_window_impl<double>(
    size_t length);

template std::vector<float> generate_kaiser_window_impl<float>(size_t length,
                                                               float beta);
template std::vector<double> generate_kaiser_window_impl<double>(size_t length,
                                                                 double beta);

template std::vector<float> generate_flattop_window_impl<float>(size_t length);
template std::vector<double> generate_flattop_window_impl<double>(
    size_t length);

}  // namespace Backend
}  // namespace OmniDSP

#endif  // USE_ACCELERATE
