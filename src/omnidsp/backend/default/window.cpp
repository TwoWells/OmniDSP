/**
 * @file window.cpp (stub)
 * @brief Implements Stub backend window generation methods for StubOmniDSPImpl
 * using standard C++.
 */

#include <cmath>        // For M_PI, sin, cos, exp, sqrt, abs etc.
#include <iostream>     // For debug/error messages
#include <numeric>      // For std::accumulate
#include <stdexcept>    // For std::invalid_argument
#include <type_traits>  // For std::is_same_v
#include <vector>

#include "OmniDSP/core_types.h"  // For Status, RealT etc.
#include "backend.h"  // Stub backend declarations (including StubOmniDSPImpl)

// Include Boost Bessel function for Kaiser window
#include <boost/math/special_functions/bessel.hpp>

// Define PI if not available from cmath
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace OmniDSP {
  namespace backend {

    //--------------------------------------------------------------------------
    // StubOmniDSPImpl - Window Generation Method Implementations (Standard C++)
    //--------------------------------------------------------------------------

    // Note: These methods are part of the StubOmniDSPImpl class.

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    StubOmniDSPImpl::bartlett_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      std::vector<RealT<T>> coeffs(length);
      RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);

      if (N_minus_1 <= static_cast<RealT<T>>(0.0)) {  // Handle length 1
        if (length == 1) coeffs[0] = static_cast<RealT<T>>(1.0);
        return coeffs;
      }

      for (size_t n = 0; n < length; ++n) {
        // Formula: 1.0 - abs(2.0 * n / (N - 1) - 1.0)
        coeffs[n]
            = 1.0 - std::abs(2.0 * static_cast<RealT<T>>(n) / N_minus_1 - 1.0);
      }
      return coeffs;
    }

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    StubOmniDSPImpl::blackman_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1) return std::vector<RealT<T>>(1, 1.0);
      std::vector<RealT<T>> coeffs(length);
      const RealT<T> a0 = 0.42;
      const RealT<T> a1 = 0.5;
      const RealT<T> a2 = 0.08;
      const RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      const RealT<T> factor1 = (N_minus_1 > 0) ? (2.0 * M_PI / N_minus_1) : 0.0;
      const RealT<T> factor2 = 2.0 * factor1;

      for (size_t n = 0; n < length; ++n) {
        RealT<T> n_T = static_cast<RealT<T>>(n);
        coeffs[n]
            = a0 - a1 * std::cos(factor1 * n_T) + a2 * std::cos(factor2 * n_T);
      }
      return coeffs;
    }

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    StubOmniDSPImpl::flattop_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1) return std::vector<RealT<T>>(1, 1.0);
      std::vector<RealT<T>> coeffs(length);
      const RealT<T> a0 = 0.21557895;
      const RealT<T> a1 = 0.41663158;
      const RealT<T> a2 = 0.277263158;
      const RealT<T> a3 = 0.083578947;
      const RealT<T> a4 = 0.006947368;
      const RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      const RealT<T> factor = (N_minus_1 > 0) ? (2.0 * M_PI / N_minus_1) : 0.0;

      for (size_t n = 0; n < length; ++n) {
        RealT<T> n_T = static_cast<RealT<T>>(n);
        coeffs[n] = a0 - a1 * std::cos(factor * n_T)
                    + a2 * std::cos(2.0 * factor * n_T)
                    - a3 * std::cos(3.0 * factor * n_T)
                    + a4 * std::cos(4.0 * factor * n_T);
      }
      return coeffs;
    }

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    StubOmniDSPImpl::gaussian_window(size_t length, RealT<T> stddev) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1) return std::vector<RealT<T>>(1, 1.0);
      if (stddev <= 0) {
        std::cerr << "Gaussian window standard deviation must be positive."
                  << std::endl;
        return std::unexpected(Status::InvalidArgument);
      }
      std::vector<RealT<T>> coeffs(length);
      RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      RealT<T> center = N_minus_1 / 2.0;
      RealT<T> sigma_term = stddev * center;
      if (sigma_term == 0.0) {
        coeffs[0] = 1.0;
        return coeffs;
      }  // Should be caught by length==1
      RealT<T> factor = -0.5 / (sigma_term * sigma_term);

      for (size_t n = 0; n < length; ++n) {
        RealT<T> exponent = static_cast<RealT<T>>(n) - center;
        coeffs[n] = std::exp(factor * exponent * exponent);
      }
      return coeffs;
    }

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    StubOmniDSPImpl::hamming_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1) return std::vector<RealT<T>>(1, 1.0);
      std::vector<RealT<T>> coeffs(length);
      const RealT<T> a0 = 0.54;
      const RealT<T> a1 = 0.46;  // 1.0 - a0
      const RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      const RealT<T> factor = (N_minus_1 > 0) ? (2.0 * M_PI / N_minus_1) : 0.0;

      for (size_t n = 0; n < length; ++n) {
        coeffs[n] = a0 - a1 * std::cos(factor * static_cast<RealT<T>>(n));
      }
      return coeffs;
    }

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    StubOmniDSPImpl::hann_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1) return std::vector<RealT<T>>(1, 1.0);
      std::vector<RealT<T>> coeffs(length);
      const RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      const RealT<T> factor = (N_minus_1 > 0) ? (2.0 * M_PI / N_minus_1) : 0.0;

      for (size_t n = 0; n < length; ++n) {
        coeffs[n] = 0.5 * (1.0 - std::cos(factor * static_cast<RealT<T>>(n)));
      }
      return coeffs;
    }

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    StubOmniDSPImpl::kaiser_window(size_t length, RealT<T> beta) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1) return std::vector<RealT<T>>(1, 1.0);
      if (beta < 0) {
        std::cerr << "Kaiser window beta parameter must be non-negative."
                  << std::endl;
        return std::unexpected(Status::InvalidArgument);
      }
      std::vector<RealT<T>> coeffs(length);

      RealT<T> bessel_i0_beta = boost::math::cyl_bessel_i(0, beta);
      if (bessel_i0_beta == 0.0) {
        std::cerr << "Kaiser window denominator I0(beta) is zero." << std::endl;
        return std::unexpected(Status::Failure);
      }
      RealT<T> inv_denom = 1.0 / bessel_i0_beta;
      RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      RealT<T> factor = (N_minus_1 > 0) ? (2.0 / N_minus_1) : 0.0;

      for (size_t n = 0; n < length; ++n) {
        RealT<T> term = factor * static_cast<RealT<T>>(n) - 1.0;
        RealT<T> arg_sqrt = 1.0 - term * term;
        // Ensure argument to sqrt is non-negative
        RealT<T> bessel_arg
            = beta * std::sqrt(std::max(static_cast<RealT<T>>(0.0), arg_sqrt));
        coeffs[n] = boost::math::cyl_bessel_i(0, bessel_arg) * inv_denom;
      }
      return coeffs;
    }

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    StubOmniDSPImpl::rectangular_window(size_t length) const
    {
      // Simple implementation using std::vector constructor
      return std::vector<RealT<T>>(length, 1.0);
    }

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    StubOmniDSPImpl::triangular_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1) return std::vector<RealT<T>>(1, 1.0);
      std::vector<RealT<T>> coeffs(length);
      RealT<T> L = static_cast<RealT<T>>(length);
      RealT<T> center = (L - 1.0) / 2.0;
      RealT<T> norm = L / 2.0;  // Normalization factor
      if (norm == 0.0)
        return std::unexpected(
            Status::Failure);  // Should be caught by length==1

      for (size_t n = 0; n < length; ++n) {
        // Formula: 1 - abs(n - (L-1)/2) / (L/2)
        coeffs[n] = 1.0 - std::abs(static_cast<RealT<T>>(n) - center) / norm;
      }
      return coeffs;
    }

    // Note: Explicit template instantiations for these methods belong in
    // src/omnidsp/backend/stub/backend.cpp where the StubOmniDSPImpl
    // class itself is instantiated.

  }  // namespace backend
}  // namespace OmniDSP
