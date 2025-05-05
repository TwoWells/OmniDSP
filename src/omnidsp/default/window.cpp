/**
 * @file window.cpp (default)
 * @brief Implements Default backend window generation helper functions using
 * standard C++.
 * @details These functions write directly to the provided output span.
 */

#include "window.hpp"  // Corresponding header for declarations

#include <OmniDSP/core_types.hpp>  // For Status, F32, F64
#include <algorithm>               // For std::max
#include <cmath>     // For sin, cos, exp, sqrt, abs, M_PI (if needed)
#include <iostream>  // For potential error logging
#include <numbers>   // For std::numbers::pi_v (C++20)
#include <numeric>   // For std::fill_n
#include <span>      // For std::span
#include <stdexcept>  // For potential error conditions (although Status is preferred)
#include <vector>  // Although not returning vector, might be used internally if needed

// Include Boost Bessel function for Kaiser window
#include <boost/math/special_functions/bessel.hpp>

namespace OmniDSP::default
{

  //--------------------------------------------------------------------------
  // Default Window Generation Helper Function Implementations (Span-based)
  //--------------------------------------------------------------------------

  template <typename T>
  [[nodiscard]] Status bartlett_window(std::span<T> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;  // Nothing to do for zero length

    if (length == 1) {
      output[0] = static_cast<T>(1.0);
      return Status::Success;
    }

    const T N_minus_1 = static_cast<T>(length - 1);
    // Avoid division by zero if length was 1 (already handled, but safe)
    if (N_minus_1 <= static_cast<T>(0.0)) {
      // Should not happen due to length==1 check, but indicates logic error
      std::cerr << "Default Backend Error: Invalid length in bartlett_window."
                << std::endl;
      return Status::Failure;
    }
    const T factor = static_cast<T>(2.0) / N_minus_1;

    for (size_t n = 0; n < length; ++n) {
      // Formula: 1.0 - abs(2.0 * n / (N - 1) - 1.0)
      //        = 1.0 - abs(factor * n - 1.0)
      output[n] = static_cast<T>(1.0)
                  - std::abs(factor * static_cast<T>(n) - static_cast<T>(1.0));
    }
    return Status::Success;
  }

  template <typename T>
  [[nodiscard]] Status blackman_window(std::span<T> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0]
          = static_cast<T>(1.0);  // Or 0.0 depending on definition? Usually 1.0
      return Status::Success;
    }

    const T a0 = static_cast<T>(0.42);
    const T a1 = static_cast<T>(0.5);
    const T a2 = static_cast<T>(0.08);
    const T N_minus_1 = static_cast<T>(length - 1);
    const T factor1
        = (N_minus_1 > 0)
              ? (static_cast<T>(2.0 * std::numbers::pi_v<double>) / N_minus_1)
              : static_cast<T>(0.0);  // Use double pi
    const T factor2 = static_cast<T>(2.0) * factor1;

    for (size_t n = 0; n < length; ++n) {
      T n_T = static_cast<T>(n);
      output[n]
          = a0 - a1 * std::cos(factor1 * n_T) + a2 * std::cos(factor2 * n_T);
    }
    return Status::Success;
  }

  template <typename T>
  [[nodiscard]] Status flattop_window(std::span<T> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = static_cast<T>(1.0);  // Flat top value at center
      return Status::Success;
    }

    // Standard coefficients (from Wikipedia/SciPy)
    const T a0 = static_cast<T>(0.21557895);
    const T a1 = static_cast<T>(0.41663158);
    const T a2 = static_cast<T>(0.277263158);
    const T a3 = static_cast<T>(0.083578947);
    const T a4 = static_cast<T>(0.006947368);
    const T N_minus_1 = static_cast<T>(length - 1);
    const T factor
        = (N_minus_1 > 0)
              ? (static_cast<T>(2.0 * std::numbers::pi_v<double>) / N_minus_1)
              : static_cast<T>(0.0);  // Use double pi

    for (size_t n = 0; n < length; ++n) {
      T n_T = static_cast<T>(n);
      output[n] = a0 - a1 * std::cos(factor * n_T)
                  + a2 * std::cos(static_cast<T>(2.0) * factor * n_T)
                  - a3 * std::cos(static_cast<T>(3.0) * factor * n_T)
                  + a4 * std::cos(static_cast<T>(4.0) * factor * n_T);
    }
    return Status::Success;
  }

  template <typename T>
  [[nodiscard]] Status gaussian_window(T stddev, std::span<T> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;

    if (stddev <= static_cast<T>(0.0)) {
      std::cerr << "Default Backend Error: Gaussian window standard "
                   "deviation must be positive."
                << std::endl;
      return Status::InvalidArgument;
    }

    if (length == 1) {
      output[0] = static_cast<T>(1.0);
      return Status::Success;
    }

    const T N_minus_1 = static_cast<T>(length - 1);
    const T center = N_minus_1 / static_cast<T>(2.0);
    // Sigma definition for Gaussian window often uses (N-1)/2 in denominator
    const T sigma_term = stddev * center;
    // Avoid division by zero if center is 0 (length 1, handled above)
    if (sigma_term == static_cast<T>(0.0)) {
      std::cerr
          << "Default Backend Error: Invalid sigma term in gaussian_window."
          << std::endl;
      return Status::Failure;  // Should not happen
    }
    const T factor = static_cast<T>(-0.5) / (sigma_term * sigma_term);

    for (size_t n = 0; n < length; ++n) {
      T exponent = static_cast<T>(n) - center;
      output[n] = std::exp(factor * exponent * exponent);
    }
    return Status::Success;
  }

  template <typename T>
  [[nodiscard]] Status hamming_window(std::span<T> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = static_cast<T>(1.0);  // Or 0.08? Usually 1.0 at center
      return Status::Success;
    }

    const T a0 = static_cast<T>(0.54);
    const T a1 = static_cast<T>(0.46);  // 1.0 - a0
    const T N_minus_1 = static_cast<T>(length - 1);
    const T factor
        = (N_minus_1 > 0)
              ? (static_cast<T>(2.0 * std::numbers::pi_v<double>) / N_minus_1)
              : static_cast<T>(0.0);  // Use double pi

    for (size_t n = 0; n < length; ++n) {
      output[n] = a0 - a1 * std::cos(factor * static_cast<T>(n));
    }
    return Status::Success;
  }

  template <typename T>
  [[nodiscard]] Status hann_window(std::span<T> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = static_cast<T>(1.0);  // Or 0.0? Usually 1.0 at center
      return Status::Success;
    }

    const T N_minus_1 = static_cast<T>(length - 1);
    const T factor
        = (N_minus_1 > 0)
              ? (static_cast<T>(2.0 * std::numbers::pi_v<double>) / N_minus_1)
              : static_cast<T>(0.0);  // Use double pi

    for (size_t n = 0; n < length; ++n) {
      output[n]
          = static_cast<T>(0.5)
            * (static_cast<T>(1.0) - std::cos(factor * static_cast<T>(n)));
    }
    return Status::Success;
  }

  template <typename T>
  [[nodiscard]] Status kaiser_window(T beta, std::span<T> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;

    if (beta < static_cast<T>(0.0)) {
      std::cerr << "Default Backend Error: Kaiser window beta parameter must "
                   "be non-negative."
                << std::endl;
      return Status::InvalidArgument;
    }

    if (length == 1) {
      output[0] = static_cast<T>(1.0);
      return Status::Success;
    }

    // Use double precision for Bessel function calculation for stability
    const double beta_d = static_cast<double>(beta);
    const double bessel_i0_beta = boost::math::cyl_bessel_i(0.0, beta_d);

    if (bessel_i0_beta == 0.0) {
      // Handle beta=0 case separately (rectangular window)
      if (beta == static_cast<T>(0.0)) {
        return rectangular_window(output);
      }
      std::cerr << "Default Backend Error: Kaiser window denominator "
                   "I0(beta) is zero for non-zero beta."
                << std::endl;
      return Status::Failure;  // Mathematical issue
    }
    const T inv_denom = static_cast<T>(1.0 / bessel_i0_beta);
    const T N_minus_1 = static_cast<T>(length - 1);
    // Avoid division by zero if length is 1 (already handled)
    const T factor = (N_minus_1 > static_cast<T>(0.0))
                         ? (static_cast<T>(2.0) / N_minus_1)
                         : static_cast<T>(0.0);

    for (size_t n = 0; n < length; ++n) {
      T term = factor * static_cast<T>(n) - static_cast<T>(1.0);
      T arg_sqrt = static_cast<T>(1.0) - term * term;
      // Ensure argument to sqrt is non-negative (can happen at endpoints due
      // to precision)
      T bessel_arg = beta * std::sqrt(std::max(static_cast<T>(0.0), arg_sqrt));
      // Calculate Bessel I0 using double precision then cast
      output[n] = static_cast<T>(boost::math::cyl_bessel_i(
                      0.0, static_cast<double>(bessel_arg)))
                  * inv_denom;
    }
    return Status::Success;
  }

  template <typename T>
  [[nodiscard]] Status rectangular_window(std::span<T> output)
  {
    // Fill the span with 1.0
    std::fill_n(output.begin(), output.size(), static_cast<T>(1.0));
    return Status::Success;
  }

  template <typename T>
  [[nodiscard]] Status triangular_window(std::span<T> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = static_cast<T>(1.0);
      return Status::Success;
    }

    const T L = static_cast<T>(length);
    const T center = (L - static_cast<T>(1.0)) / static_cast<T>(2.0);
    // Use L/2 for normalization according to common definitions (e.g., SciPy)
    const T norm = L / static_cast<T>(2.0);
    if (norm == static_cast<T>(0.0)) {
      // Should not happen due to length checks
      std::cerr << "Default Backend Error: Invalid normalization factor in "
                   "triangular_window."
                << std::endl;
      return Status::Failure;
    }

    for (size_t n = 0; n < length; ++n) {
      // Formula: 1 - abs(n - (L-1)/2) / (L/2)
      output[n]
          = static_cast<T>(1.0) - std::abs(static_cast<T>(n) - center) / norm;
    }
    return Status::Success;
  }

  // --- Explicit Template Instantiations ---
  template Status bartlett_window<F32>(std::span<F32> output);
  template Status bartlett_window<F64>(std::span<F64> output);
  template Status blackman_window<F32>(std::span<F32> output);
  template Status blackman_window<F64>(std::span<F64> output);
  template Status flattop_window<F32>(std::span<F32> output);
  template Status flattop_window<F64>(std::span<F64> output);
  template Status hamming_window<F32>(std::span<F32> output);
  template Status hamming_window<F64>(std::span<F64> output);
  template Status hann_window<F32>(std::span<F32> output);
  template Status hann_window<F64>(std::span<F64> output);
  template Status rectangular_window<F32>(std::span<F32> output);
  template Status rectangular_window<F64>(std::span<F64> output);
  template Status triangular_window<F32>(std::span<F32> output);
  template Status triangular_window<F64>(std::span<F64> output);
  template Status gaussian_window<F32>(F32 stddev, std::span<F32> output);
  template Status gaussian_window<F64>(F64 stddev, std::span<F64> output);
  template Status kaiser_window<F32>(F32 beta, std::span<F32> output);
  template Status kaiser_window<F64>(F64 beta, std::span<F64> output);

}  // namespace OmniDSP::default
