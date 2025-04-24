/**
 * @file window.cpp (accelerate)
 * @brief Implements Accelerate backend window generation methods for
 * AccelerateOmniDSPImpl.
 */

// Only compile this file if Accelerate backend is enabled via CMake
#ifdef USE_ACCELERATE

#include <Accelerate/Accelerate.h>

#include <cmath>  // For sin, cos, exp, sqrt, etc. (needed for manual implementations)
#include <iostream>  // For debug/error messages
#include <numbers>
#include <stdexcept>    // For std::invalid_argument
#include <type_traits>  // For std::is_same_v
#include <vector>

#include "OmniDSP/core_types.h"  // For Status, RealT etc.
#include "backend.h"  // Accelerate backend declarations (including AccelerateOmniDSPImpl)

// Include Boost Bessel function for Kaiser window
#include <boost/math/special_functions/bessel.hpp>

namespace OmniDSP {
  namespace backend {

    //--------------------------------------------------------------------------
    // AccelerateOmniDSPImpl - Window Generation Method Implementations
    //--------------------------------------------------------------------------

    // Note: These methods are part of the AccelerateOmniDSPImpl class.
    // The class definition itself is typically in backend.h or backend.cpp.
    // These definitions assume they are linked correctly to that class
    // definition.

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::bartlett_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      std::vector<RealT<T>> coeffs(length);
      RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);

      // Handle edge case length = 1
      if (N_minus_1 <= static_cast<RealT<T>>(0.0)) {
        if (length == 1) coeffs[0] = static_cast<RealT<T>>(1.0);
        return coeffs;
      }

      // Generate sequence 0, 1, ..., N-1
      std::vector<RealT<T>> n_vec(length);
      RealT<T> start = 0.0;
      RealT<T> step = 1.0;
      if constexpr (std::is_same_v<T, float>) {
        vDSP_vramp(&start, &step, n_vec.data(), 1, length);
      }
      else {
        vDSP_vrampd(&start, &step, n_vec.data(), 1, length);
      }

      // Calculate Bartlett: 1.0 - abs(2.0 * n / (N - 1) - 1.0)
      RealT<T> scale = static_cast<RealT<T>>(2.0) / N_minus_1;
      RealT<T> minus_one = -1.0;
      RealT<T> one = 1.0;

      if constexpr (std::is_same_v<T, float>) {
        vDSP_vsmul(
            n_vec.data(),
            1,
            &scale,
            coeffs.data(),
            1,
            length);  // coeffs = 2n / (N-1)
        vDSP_vsadd(
            coeffs.data(),
            1,
            &minus_one,
            coeffs.data(),
            1,
            length);  // coeffs = 2n / (N-1) - 1
        vDSP_vabs(
            coeffs.data(),
            1,
            coeffs.data(),
            1,
            length);  // coeffs = |2n / (N-1) - 1|
        vDSP_vneg(
            coeffs.data(), 1, coeffs.data(), 1, length);  // coeffs = -|...|
        vDSP_vsadd(
            coeffs.data(),
            1,
            &one,
            coeffs.data(),
            1,
            length);  // coeffs = 1 - |...|
      }
      else {
        vDSP_vsmulD(n_vec.data(), 1, &scale, coeffs.data(), 1, length);
        vDSP_vsaddD(coeffs.data(), 1, &minus_one, coeffs.data(), 1, length);
        vDSP_vabsD(coeffs.data(), 1, coeffs.data(), 1, length);
        vDSP_vnegD(coeffs.data(), 1, coeffs.data(), 1, length);
        vDSP_vsaddD(coeffs.data(), 1, &one, coeffs.data(), 1, length);
      }

      return coeffs;
    }

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::blackman_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      std::vector<RealT<T>> coeffs(length);
      if constexpr (std::is_same_v<T, float>) {
        vDSP_blkman_window(
            coeffs.data(), length, 0);  // 0 for standard Blackman
      }
      else {
        vDSP_blkman_windowD(coeffs.data(), length, 0);
      }
      return coeffs;
    }

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::flattop_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      std::vector<RealT<T>> coeffs(length);
      std::vector<RealT<T>> n_vec(length);  // 0 to N-1

      // Standard coefficients (e.g., from SciPy)
      const RealT<T> a0 = 0.21557895;
      const RealT<T> a1 = 0.41663158;
      const RealT<T> a2 = 0.277263158;
      const RealT<T> a3 = 0.083578947;
      const RealT<T> a4 = 0.006947368;
      const RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      // Avoid division by zero if length is 1 (already handled, but defensive)
      const RealT<T> factor
          = (N_minus_1 > 0)
                ? (static_cast<RealT<T>>(2.0 * std::numbers::pi) / N_minus_1)
                : 0.0;

      // Generate n = 0, 1, ..., N-1
      RealT<T> start = 0.0;
      RealT<T> step = 1.0;
      if constexpr (std::is_same_v<T, float>) {
        vDSP_vramp(&start, &step, n_vec.data(), 1, length);
      }
      else {
        vDSP_vrampd(&start, &step, n_vec.data(), 1, length);
      }

      // Calculate terms using vDSP vector functions
      std::vector<RealT<T>> term1(length), term2(length), term3(length),
          term4(length);
      RealT<T> factor1 = factor;
      RealT<T> factor2 = factor * 2.0;
      RealT<T> factor3 = factor * 3.0;
      RealT<T> factor4 = factor * 4.0;

      // Calculate cosines
      int len_int = static_cast<int>(length);  // vvcos takes int* for length
      if constexpr (std::is_same_v<T, float>) {
        // Scale n first
        vDSP_vsmul(n_vec.data(), 1, &factor1, term1.data(), 1, length);
        vDSP_vsmul(n_vec.data(), 1, &factor2, term2.data(), 1, length);
        vDSP_vsmul(n_vec.data(), 1, &factor3, term3.data(), 1, length);
        vDSP_vsmul(n_vec.data(), 1, &factor4, term4.data(), 1, length);
        // Compute cosines
        vvcosf(term1.data(), term1.data(), &len_int);
        vvcosf(term2.data(), term2.data(), &len_int);
        vvcosf(term3.data(), term3.data(), &len_int);
        vvcosf(term4.data(), term4.data(), &len_int);

        // Combine terms: coeffs = a0 - a1*t1 + a2*t2 - a3*t3 + a4*t4
        vDSP_vfill(&a0, coeffs.data(), 1, length);  // Start with a0
        RealT<T> minus_a1 = -a1;
        RealT<T> minus_a3 = -a3;
        vDSP_vsma(
            term1.data(),
            1,
            &minus_a1,
            coeffs.data(),
            1,
            coeffs.data(),
            1,
            length);  // -a1*t1
        vDSP_vsma(
            term2.data(),
            1,
            &a2,
            coeffs.data(),
            1,
            coeffs.data(),
            1,
            length);  // +a2*t2
        vDSP_vsma(
            term3.data(),
            1,
            &minus_a3,
            coeffs.data(),
            1,
            coeffs.data(),
            1,
            length);  // -a3*t3
        vDSP_vsma(
            term4.data(),
            1,
            &a4,
            coeffs.data(),
            1,
            coeffs.data(),
            1,
            length);  // +a4*t4
      }
      else {  // Double precision
        vDSP_vsmulD(n_vec.data(), 1, &factor1, term1.data(), 1, length);
        vDSP_vsmulD(n_vec.data(), 1, &factor2, term2.data(), 1, length);
        vDSP_vsmulD(n_vec.data(), 1, &factor3, term3.data(), 1, length);
        vDSP_vsmulD(n_vec.data(), 1, &factor4, term4.data(), 1, length);

        vvcos(term1.data(), term1.data(), &len_int);
        vvcos(term2.data(), term2.data(), &len_int);
        vvcos(term3.data(), term3.data(), &len_int);
        vvcos(term4.data(), term4.data(), &len_int);

        vDSP_vfillD(&a0, coeffs.data(), 1, length);
        RealT<T> minus_a1 = -a1;
        RealT<T> minus_a3 = -a3;
        vDSP_vsmaD(
            term1.data(),
            1,
            &minus_a1,
            coeffs.data(),
            1,
            coeffs.data(),
            1,
            length);
        vDSP_vsmaD(
            term2.data(), 1, &a2, coeffs.data(), 1, coeffs.data(), 1, length);
        vDSP_vsmaD(
            term3.data(),
            1,
            &minus_a3,
            coeffs.data(),
            1,
            coeffs.data(),
            1,
            length);
        vDSP_vsmaD(
            term4.data(), 1, &a4, coeffs.data(), 1, coeffs.data(), 1, length);
      }

      return coeffs;
    }

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::gaussian_window(size_t length, RealT<T> stddev) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      if (stddev <= 0) {
        std::cerr << "Gaussian window standard deviation must be positive."
                  << std::endl;
        return std::unexpected(Status::InvalidArgument);
      }

      std::vector<RealT<T>> coeffs(length);
      std::vector<RealT<T>> n_vec(length);  // 0 to N-1

      RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      RealT<T> center = N_minus_1 / 2.0;
      RealT<T> sigma_term = stddev * center;  // sigma * (N-1)/2
      // Handle potential division by zero if length=1 (center=0) ->
      // sigma_term=0
      if (sigma_term == static_cast<RealT<T>>(0.0)) {
        // This case should be handled by length==1 check above, but defensive
        // check
        coeffs[0] = 1.0;
        return coeffs;
      }
      RealT<T> neg_half_inv_sigma_sq = -0.5 / (sigma_term * sigma_term);

      // Generate n = 0, 1, ..., N-1
      RealT<T> start = 0.0;
      RealT<T> step = 1.0;
      int len_int
          = static_cast<int>(length);  // vv... functions take int* length
      if constexpr (std::is_same_v<T, float>) {
        vDSP_vramp(&start, &step, n_vec.data(), 1, length);
        // Calculate: exp( neg_half_inv_sigma_sq * (n - center)^2 )
        RealT<T> neg_center = -center;
        vDSP_vsadd(
            n_vec.data(),
            1,
            &neg_center,
            coeffs.data(),
            1,
            length);  // coeffs = n - center
        vDSP_vsq(
            coeffs.data(),
            1,
            coeffs.data(),
            1,
            length);  // coeffs = (n - center)^2
        vDSP_vsmul(
            coeffs.data(),
            1,
            &neg_half_inv_sigma_sq,
            coeffs.data(),
            1,
            length);  // coeffs = factor * (...)**2
        vvexpf(coeffs.data(), coeffs.data(), &len_int);  // coeffs = exp(...)
      }
      else {  // Double
        vDSP_vrampd(&start, &step, n_vec.data(), 1, length);
        RealT<T> neg_center = -center;
        vDSP_vsaddD(n_vec.data(), 1, &neg_center, coeffs.data(), 1, length);
        vDSP_vsqD(coeffs.data(), 1, coeffs.data(), 1, length);
        vDSP_vsmulD(
            coeffs.data(), 1, &neg_half_inv_sigma_sq, coeffs.data(), 1, length);
        vvexp(coeffs.data(), coeffs.data(), &len_int);
      }

      return coeffs;
    }

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::hamming_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      std::vector<RealT<T>> coeffs(length);
      if constexpr (std::is_same_v<T, float>) {
        vDSP_hamm_window(coeffs.data(), length, 0);  // 0 for standard Hamming
      }
      else {
        vDSP_hamm_windowD(coeffs.data(), length, 0);
      }
      return coeffs;
    }

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::hann_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      std::vector<RealT<T>> coeffs(length);
      if constexpr (std::is_same_v<T, float>) {
        vDSP_hann_window(
            coeffs.data(),
            length,
            0);  // 0 for standard Hann (vDSP_HANN_NORM for energy norm)
      }
      else {
        vDSP_hann_windowD(coeffs.data(), length, 0);
      }
      return coeffs;
    }

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::kaiser_window(size_t length, RealT<T> beta) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      if (beta < 0) {
        std::cerr << "Kaiser window beta parameter must be non-negative."
                  << std::endl;
        return std::unexpected(Status::InvalidArgument);
      }

      std::vector<RealT<T>> coeffs(length);
      std::vector<RealT<T>> n_vec(length);

      // Pre-calculate denominator I0(beta) using Boost
      RealT<T> bessel_i0_beta = boost::math::cyl_bessel_i(0, beta);
      if (bessel_i0_beta == static_cast<RealT<T>>(0.0)) {
        // Handle potential division by zero if I0(beta) is zero (shouldn't
        // happen for beta >= 0)
        std::cerr << "Kaiser window denominator I0(beta) is zero." << std::endl;
        // Maybe return rectangular window in this case? Or error?
        return std::unexpected(Status::Failure);  // Indicate numerical issue
      }
      RealT<T> inv_denom = 1.0 / bessel_i0_beta;

      RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      // Avoid division by zero if length is 1 (already handled, but defensive)
      RealT<T> scale = (N_minus_1 > 0) ? (2.0 / N_minus_1) : 0.0;

      // Generate n = 0, 1, ..., N-1
      RealT<T> start = 0.0;
      RealT<T> step = 1.0;
      if constexpr (std::is_same_v<T, float>) {
        vDSP_vramp(&start, &step, n_vec.data(), 1, length);
      }
      else {
        vDSP_vrampd(&start, &step, n_vec.data(), 1, length);
      }

      // Calculate Kaiser argument: beta * sqrt(1 - (2n/(N-1) - 1)^2)
      RealT<T> minus_one = -1.0;
      RealT<T> one = 1.0;
      int len_int = static_cast<int>(length);  // vvsqrt takes int* length

      if constexpr (std::is_same_v<T, float>) {
        vDSP_vsmul(
            n_vec.data(), 1, &scale, coeffs.data(), 1, length);  // 2n/(N-1)
        vDSP_vsadd(
            coeffs.data(),
            1,
            &minus_one,
            coeffs.data(),
            1,
            length);                                            // 2n/(N-1) - 1
        vDSP_vsq(coeffs.data(), 1, coeffs.data(), 1, length);   // (term)^2
        vDSP_vneg(coeffs.data(), 1, coeffs.data(), 1, length);  // -(term)^2
        vDSP_vsadd(
            coeffs.data(),
            1,
            &one,
            coeffs.data(),
            1,
            length);  // 1 - (term)^2
        // Ensure non-negative before sqrt (due to potential floating point
        // issues at ends)
        RealT<T> zero = 0.0f;
        vDSP_vmax(coeffs.data(), 1, &zero, coeffs.data(), 1, length);
        vvsqrtf(coeffs.data(), coeffs.data(), &len_int);  // sqrt(...)
        vDSP_vsmul(
            coeffs.data(),
            1,
            &beta,
            coeffs.data(),
            1,
            length);  // beta * sqrt(...)
        // Apply Bessel I0 element-wise using Boost
        for (size_t i = 0; i < length; ++i) {
          coeffs[i] = boost::math::cyl_bessel_i(0, coeffs[i]);
        }
        vDSP_vsmul(
            coeffs.data(),
            1,
            &inv_denom,
            coeffs.data(),
            1,
            length);  // result / I0(beta)
      }
      else {  // Double
        vDSP_vsmulD(n_vec.data(), 1, &scale, coeffs.data(), 1, length);
        vDSP_vsaddD(coeffs.data(), 1, &minus_one, coeffs.data(), 1, length);
        vDSP_vsqD(coeffs.data(), 1, coeffs.data(), 1, length);
        vDSP_vnegD(coeffs.data(), 1, coeffs.data(), 1, length);
        vDSP_vsaddD(coeffs.data(), 1, &one, coeffs.data(), 1, length);
        RealT<T> zero = 0.0;
        vDSP_vmaxD(coeffs.data(), 1, &zero, coeffs.data(), 1, length);
        vvsqrt(coeffs.data(), coeffs.data(), &len_int);
        vDSP_vsmulD(coeffs.data(), 1, &beta, coeffs.data(), 1, length);
        // Apply Bessel I0 element-wise using Boost
        for (size_t i = 0; i < length; ++i) {
          coeffs[i] = boost::math::cyl_bessel_i(0, coeffs[i]);
        }
        vDSP_vsmulD(coeffs.data(), 1, &inv_denom, coeffs.data(), 1, length);
      }

      return coeffs;
    }

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::rectangular_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      std::vector<RealT<T>> coeffs(length);
      RealT<T> one = 1.0;
      if constexpr (std::is_same_v<T, float>) {
        vDSP_vfill(&one, coeffs.data(), 1, length);
      }
      else {
        vDSP_vfillD(&one, coeffs.data(), 1, length);
      }
      return coeffs;
    }

    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    AccelerateOmniDSPImpl::triangular_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      std::vector<RealT<T>> coeffs(length);
      // Formula for triangular window (non-zero endpoints for L): (2/(L+1)) * (
      // (L+1)/2 - abs(n - (L-1)/2) ) Or simpler ramp implementation
      size_t L = length;
      RealT<T> start = 0.0;
      RealT<T> one = 1.0;

      if constexpr (std::is_same_v<T, float>) {
        if (L % 2 != 0) {  // Odd length (L=2k+1 -> N=k)
          float N = (L - 1.0f) / 2.0f;
          float scale_up = 1.0f / N;
          float scale_down = -scale_up;
          vDSP_vramp(
              &start,
              &scale_up,
              coeffs.data(),
              1,
              (L + 1) / 2);  // Ramp 0 to 1
          vDSP_vramp(
              &one,
              &scale_down,
              coeffs.data() + (L + 1) / 2,
              1,
              L / 2);  // Ramp 1 to 0
        }
        else {  // Even length (L=2k -> N=k)
          float N = L / 2.0f;
          float scale_up = 1.0f / N;
          float scale_down = -scale_up;
          vDSP_vramp(
              &start,
              &scale_up,
              coeffs.data(),
              1,
              L / 2);  // Ramp 0 to (N-1)/N
          // Ramp down from 1? Definition varies. Let's make it symmetric.
          // Ramp up to 1, then down from 1, but peak is between samples.
          // Let's use formula: 1 - abs(n - (L-1)/2) / (L/2)
          std::vector<RealT<T>> n_vec(L);
          vDSP_vramp(&start, &one, n_vec.data(), 1, L);
          float center = (L - 1.0f) / 2.0f;
          float neg_center = -center;
          float inv_L_half = 1.0f / (L / 2.0f);
          vDSP_vsadd(
              n_vec.data(),
              1,
              &neg_center,
              coeffs.data(),
              1,
              L);                                            // n - center
          vDSP_vabs(coeffs.data(), 1, coeffs.data(), 1, L);  // |n - center|
          vDSP_vsmul(
              coeffs.data(),
              1,
              &inv_L_half,
              coeffs.data(),
              1,
              L);                                            // |...| / (L/2)
          vDSP_vneg(coeffs.data(), 1, coeffs.data(), 1, L);  // -|...| / (L/2)
          vDSP_vsadd(coeffs.data(), 1, &one, coeffs.data(), 1, L);  // 1 - ...
        }
      }
      else {               // Double
        if (L % 2 != 0) {  // Odd length
          double N = (L - 1.0) / 2.0;
          double scale_up = 1.0 / N;
          double scale_down = -scale_up;
          vDSP_vrampd(&start, &scale_up, coeffs.data(), 1, (L + 1) / 2);
          vDSP_vrampd(&one, &scale_down, coeffs.data() + (L + 1) / 2, 1, L / 2);
        }
        else {  // Even length
          std::vector<RealT<T>> n_vec(L);
          vDSP_vrampd(&start, &one, n_vec.data(), 1, L);
          double center = (L - 1.0) / 2.0;
          double neg_center = -center;
          double inv_L_half = 1.0 / (L / 2.0);
          vDSP_vsaddD(n_vec.data(), 1, &neg_center, coeffs.data(), 1, L);
          vDSP_vabsD(coeffs.data(), 1, coeffs.data(), 1, L);
          vDSP_vsmulD(coeffs.data(), 1, &inv_L_half, coeffs.data(), 1, L);
          vDSP_vnegD(coeffs.data(), 1, coeffs.data(), 1, L);
          vDSP_vsaddD(coeffs.data(), 1, &one, coeffs.data(), 1, L);
        }
      }

      return coeffs;
    }

    // Note: Explicit template instantiations for these methods belong in
    // src/omnidsp/backend/accelerate/backend.cpp where the
    // AccelerateOmniDSPImpl class itself is instantiated.

  }  // namespace backend
}  // namespace OmniDSP

#endif  // USE_ACCELERATE
