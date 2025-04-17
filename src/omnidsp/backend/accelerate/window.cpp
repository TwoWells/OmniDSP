/**
 * @file src/omnidsp/backend/accelerate/window.cpp
 * @brief Apple Accelerate backend implementation for OmniDSP window functions.
 *
 * Implements backend functions to generate window coefficients using vDSP
 * where available (Hann, Hamming). Kaiser and Flattop use manual calculations.
 * Compiled only when USE_ACCELERATE is defined.
 */

// --- Includes ---
#include <cmath>      // For M_PI, cos, sin, sqrt, std::abs, std::cyl_bessel_i
#include <limits>     // For std::numeric_limits
#include <numeric>    // For std::accumulate
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>
#include <type_traits>  // For std::is_same_v
#include <vector>

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
 */
template <typename T>
std::vector<T> generate_hann_window_impl(size_t length) {
  if (length == 0) return {};
  std::vector<T> coeffs(length);
  vDSP_Length n_vDSP = static_cast<vDSP_Length>(length);

  if constexpr (std::is_same_v<T, float>) {
    // vDSP_hann_window_f(OutputVector, Length, Flags);
    vDSP_hann_window_f(coeffs.data(), n_vDSP, vDSP_HANN_NORM);
  } else {  // double
    // vDSP_hann_window(OutputVector, Length, Flags);
    vDSP_hann_window(coeffs.data(), n_vDSP, vDSP_HANN_NORM);
  }
  // Note: vDSP_hann_window might have slightly different scaling/endpoint
  // handling compared to the manual formula or IPP. Test comparison needed.
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
    // vDSP_hamm_window_f(OutputVector, Length, Flags=0);
    vDSP_hamm_window_f(coeffs.data(), n_vDSP,
                       0);  // Flags = 0 for standard Hamming
  } else {                  // double
    // vDSP_hamm_window(OutputVector, Length, Flags=0);
    vDSP_hamm_window(coeffs.data(), n_vDSP, 0);
  }
  // Note: vDSP_hamm_window might have slightly different scaling/endpoint
  // handling compared to the manual formula or IPP. Test comparison needed.
  return coeffs;
}

/**
 * @brief Generates Kaiser window coefficients (manual calculation, as vDSP
 * lacks native support).
 * @param beta The Kaiser window beta parameter.
 */
template <typename T>
std::vector<T> generate_kaiser_window_impl(size_t length, T beta) {
  if (length == 0) return {};
  if (length == 1) return {static_cast<T>(1.0)};  // Handle N=1 case

  std::vector<T> coeffs(length);
  // Precompute I0(beta)
  T i0_beta = std::cyl_bessel_i(static_cast<T>(0.0), beta);
  if (std::abs(i0_beta) < std::numeric_limits<T>::epsilon()) {
    // Handle potential division by zero if I0(beta) is zero
    // This is unlikely for typical beta values but possible for beta=0
    if (std::abs(beta) < std::numeric_limits<T>::epsilon()) {
      // If beta is 0, Kaiser window is effectively a rectangular window
      std::fill(coeffs.begin(), coeffs.end(), static_cast<T>(1.0));
      return coeffs;
    } else {
      // Should not happen for typical positive beta, but throw if it does
      throw std::runtime_error("Kaiser window denominator I0(beta) is zero.");
    }
  }

  double N_minus_1 = static_cast<double>(length - 1);

  for (size_t n = 0; n < length; ++n) {
    // Argument for the square root term: 1 - ( (2n / (N-1)) - 1 )^2
    double factor = (static_cast<double>(n) * 2.0) / N_minus_1 - 1.0;
    double sqrt_arg = 1.0 - factor * factor;

    // Ensure argument to sqrt is non-negative (can happen at endpoints due to
    // precision)
    if (sqrt_arg < 0.0) sqrt_arg = 0.0;

    T sqrt_term = static_cast<T>(std::sqrt(sqrt_arg));
    // Argument for Bessel function: beta * sqrt(...)
    T bessel_arg = beta * sqrt_term;
    // Calculate I0(bessel_arg) / I0(beta)
    coeffs[n] = std::cyl_bessel_i(static_cast<T>(0.0), bessel_arg) / i0_beta;
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
  // Coefficients from standard definition (e.g., SciPy)
  constexpr T a0 = static_cast<T>(0.21557895);
  constexpr T a1 = static_cast<T>(0.41663158);
  constexpr T a2 = static_cast<T>(0.277263158);
  constexpr T a3 = static_cast<T>(0.083578947);
  constexpr T a4 = static_cast<T>(0.006947368);
  // Denominator for cosine terms
  double denom = (length > 1) ? static_cast<double>(length - 1) : 1.0;

  for (size_t n = 0; n < length; ++n) {
    T cos_2pi_term = static_cast<T>(cos(2.0 * M_PI * n / denom));
    T cos_4pi_term = static_cast<T>(cos(4.0 * M_PI * n / denom));
    T cos_6pi_term = static_cast<T>(cos(6.0 * M_PI * n / denom));
    T cos_8pi_term = static_cast<T>(cos(8.0 * M_PI * n / denom));
    coeffs[n] = a0 - a1 * cos_2pi_term + a2 * cos_4pi_term - a3 * cos_6pi_term +
                a4 * cos_8pi_term;
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
