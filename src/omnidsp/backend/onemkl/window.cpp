/**
 * @file window.cpp (onemkl / IPP)
 * @brief Implements oneMKL backend window generation methods for
 * OneMKLOmniDSPImpl using Intel IPP.
 * @details This file provides the concrete implementations for the window
 * generation functions declared as virtual methods in OneMKLOmniDSPImpl. It
 * leverages Intel Integrated Performance Primitives (IPP) for optimized window
 * calculations where available, falling back to oneMKL Vector Math Library
 * (VML) or standard math functions for others.
 */

#include "OmniDSP/core_types.h"  // For Status, RealT etc.
#include "backend.h"  // oneMKL backend declarations (including OneMKLOmniDSPImpl)

// Include Intel IPP signal processing header
#include <ipp.h>
// More specific headers if preferred:
// #include <ipps.h> // Signal Processing

// Include MKL header for VML (used for windows not directly in IPP)
#include <mkl.h>

#include <cmath>        // For M_PI, sin, cos, exp, sqrt, abs etc.
#include <iostream>     // For debug/error messages
#include <stdexcept>    // For std::invalid_argument
#include <string>       // For error messages
#include <type_traits>  // For std::is_same_v
#include <vector>

// Boost Bessel function no longer needed for Kaiser if using ippsWinKaiser
// #include <boost/math/special_functions/bessel.hpp>

// Define PI if not available from cmath
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace OmniDSP {
  namespace backend {

    /**
     * @brief Helper function to convert IPP status codes to OmniDSP::Status.
     * @param status The IppStatus code returned by an IPP function.
     * @return The corresponding OmniDSP::Status enum value.
     */
    inline Status ipp_status_to_omnidsp_status(IppStatus status)
    {
      if (status == ippStsNoErr) {
        return Status::Success;
      }
      // Log the IPP error message for debugging
      std::cerr << "IPP Error: " << ippGetStatusString(status)
                << " (Code: " << status << ")" << std::endl;
      // Map common IPP errors to OmniDSP statuses
      if (status == ippStsNullPtrErr || status == ippStsSizeErr
          || status == ippStsStepErr || status == ippStsBadArgErr
          || status == ippStsOutOfRangeErr) {
        return Status::InvalidArgument;
      }
      if (status == ippStsMemAllocErr) {
        return Status::AllocationError;
      }
      // Add more specific mappings if needed (e.g., ippStsFIRMRFactorErr)
      return Status::BackendError;  // Generic backend error for other IPP
                                    // issues
    }

    //--------------------------------------------------------------------------
    // OneMKLOmniDSPImpl - Window Generation Method Implementations using
    // IPP/VML
    //--------------------------------------------------------------------------

    /**
     * @brief Generates Bartlett window coefficients using IPP.
     * @see OmniDSP::OmniDSP::bartlett_window
     */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    OneMKLOmniDSPImpl::bartlett_window(size_t length) const
    {
      if (length == 0)
        return std::vector<RealT<T>>();  // Empty vector for zero length
      if (length == 1)
        return std::vector<RealT<T>>(
            1, static_cast<RealT<T>>(1.0));  // Single point is 1.0
      std::vector<RealT<T>> coeffs(length);
      IppStatus status = ippStsErr;
      int len_int = static_cast<int>(length);  // IPP uses int for length

      // Use IPP Bartlett window function
      if constexpr (std::is_same_v<T, float>) {
        status = ippsWinBartlett_32f(coeffs.data(), len_int);
      }
      else {  // double
        status = ippsWinBartlett_64f(coeffs.data(), len_int);
      }

      // Check IPP status and return result or error
      if (status != ippStsNoErr) {
        return std::unexpected(ipp_status_to_omnidsp_status(status));
      }
      return coeffs;
    }

    /**
     * @brief Generates Blackman window coefficients using IPP.
     * @details Uses the standard Blackman window definition (alpha = 0.16).
     * @see OmniDSP::OmniDSP::blackman_window
     */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    OneMKLOmniDSPImpl::blackman_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      std::vector<RealT<T>> coeffs(length);
      IppStatus status = ippStsErr;
      int len_int = static_cast<int>(length);

      // Use IPP standard Blackman window function
      if constexpr (std::is_same_v<T, float>) {
        status = ippsWinBlackmanStd_32f(coeffs.data(), len_int);
      }
      else {  // double
        status = ippsWinBlackmanStd_64f(coeffs.data(), len_int);
      }

      if (status != ippStsNoErr) {
        return std::unexpected(ipp_status_to_omnidsp_status(status));
      }
      return coeffs;
    }

    /**
     * @brief Generates Flat Top window coefficients using MKL VML.
     * @details IPP does not have a direct Flat Top function, so MKL VML is used
     * for calculation.
     * @see OmniDSP::OmniDSP::flattop_window
     */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    OneMKLOmniDSPImpl::flattop_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      std::vector<RealT<T>> coeffs(length);
      std::vector<RealT<T>> n_vec(length);  // Vector for indices 0 to N-1
      MKL_LONG n_mkl
          = static_cast<MKL_LONG>(length);  // MKL VML uses MKL_LONG for length

      // Standard coefficients (e.g., from SciPy)
      const RealT<T> a0 = 0.21557895;
      const RealT<T> a1 = 0.41663158;
      const RealT<T> a2 = 0.277263158;
      const RealT<T> a3 = 0.083578947;
      const RealT<T> a4 = 0.006947368;
      const RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      // Avoid division by zero if length is 1 (already handled, but defensive)
      const RealT<T> factor
          = (N_minus_1 > 0) ? (static_cast<RealT<T>>(2.0 * M_PI) / N_minus_1)
                            : 0.0;

      // Generate n = 0, 1, ..., N-1 using VML Ramp function
      RealT<T> start = 0.0;
      RealT<T> step = 1.0;
      if constexpr (std::is_same_v<T, float>) {
        vsRamp(n_mkl, &start, &step, n_vec.data());
      }
      else {
        vdRamp(n_mkl, &start, &step, n_vec.data());
      }

      // Calculate terms using VML vector functions
      std::vector<RealT<T>> arg1(length), arg2(length), arg3(length),
          arg4(length);
      std::vector<RealT<T>> cos1(length), cos2(length), cos3(length),
          cos4(length);
      RealT<T> f1 = factor, f2 = factor * 2.0, f3 = factor * 3.0,
               f4 = factor * 4.0;

      if constexpr (std::is_same_v<T, float>) {
        // Calculate arguments for cosine functions
        vsMul(n_mkl, n_vec.data(), &f1, arg1.data());
        vsMul(n_mkl, n_vec.data(), &f2, arg2.data());
        vsMul(n_mkl, n_vec.data(), &f3, arg3.data());
        vsMul(n_mkl, n_vec.data(), &f4, arg4.data());
        // Calculate cosines
        vsCos(n_mkl, arg1.data(), cos1.data());
        vsCos(n_mkl, arg2.data(), cos2.data());
        vsCos(n_mkl, arg3.data(), cos3.data());
        vsCos(n_mkl, arg4.data(), cos4.data());

        // Combine terms: coeffs = a0 - a1*c1 + a2*c2 - a3*c3 + a4*c4
        vsMul(n_mkl, cos1.data(), &a1, arg1.data());  // arg1 = a1*c1
        vsMul(n_mkl, cos2.data(), &a2, arg2.data());  // arg2 = a2*c2
        vsMul(n_mkl, cos3.data(), &a3, arg3.data());  // arg3 = a3*c3
        vsMul(n_mkl, cos4.data(), &a4, arg4.data());  // arg4 = a4*c4

        vsSub(n_mkl, &a0, arg1.data(), coeffs.data());  // coeffs = a0 - a1*c1
        vsAdd(
            n_mkl,
            coeffs.data(),
            arg2.data(),
            coeffs.data());  // coeffs += a2*c2
        vsSub(
            n_mkl,
            coeffs.data(),
            arg3.data(),
            coeffs.data());  // coeffs -= a3*c3
        vsAdd(
            n_mkl,
            coeffs.data(),
            arg4.data(),
            coeffs.data());  // coeffs += a4*c4
      }
      else {  // Double
        vdMul(n_mkl, n_vec.data(), &f1, arg1.data());
        vdCos(n_mkl, arg1.data(), cos1.data());
        vdMul(n_mkl, n_vec.data(), &f2, arg2.data());
        vdCos(n_mkl, arg2.data(), cos2.data());
        vdMul(n_mkl, n_vec.data(), &f3, arg3.data());
        vdCos(n_mkl, arg3.data(), cos3.data());
        vdMul(n_mkl, n_vec.data(), &f4, arg4.data());
        vdCos(n_mkl, arg4.data(), cos4.data());

        vdMul(n_mkl, cos1.data(), &a1, arg1.data());
        vdMul(n_mkl, cos2.data(), &a2, arg2.data());
        vdMul(n_mkl, cos3.data(), &a3, arg3.data());
        vdMul(n_mkl, cos4.data(), &a4, arg4.data());

        vdSub(n_mkl, &a0, arg1.data(), coeffs.data());
        vdAdd(n_mkl, coeffs.data(), arg2.data(), coeffs.data());
        vdSub(n_mkl, coeffs.data(), arg3.data(), coeffs.data());
        vdAdd(n_mkl, coeffs.data(), arg4.data(), coeffs.data());
      }
      // Check VML errors? VML often uses error handlers. Assume success.
      return coeffs;
    }

    /**
     * @brief Generates Gaussian window coefficients using MKL VML.
     * @details IPP does not have a direct Gaussian function, so MKL VML is
     * used.
     * @see OmniDSP::OmniDSP::gaussian_window
     */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    OneMKLOmniDSPImpl::gaussian_window(size_t length, RealT<T> stddev) const
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
      std::vector<RealT<T>> n_vec(length);
      MKL_LONG n_mkl = static_cast<MKL_LONG>(length);

      RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      RealT<T> center = N_minus_1 / 2.0;
      RealT<T> sigma_term = stddev * center;
      if (sigma_term
          == static_cast<RealT<T>>(
              0.0)) {  // Avoid division by zero if length=1
        coeffs[0] = 1.0;
        return coeffs;
      }
      RealT<T> neg_half_inv_sigma_sq = -0.5 / (sigma_term * sigma_term);

      // Generate n = 0..N-1
      RealT<T> start = 0.0;
      RealT<T> step = 1.0;
      if constexpr (std::is_same_v<T, float>) {
        vsRamp(n_mkl, &start, &step, n_vec.data());
      }
      else {
        vdRamp(n_mkl, &start, &step, n_vec.data());
      }

      // Calculate: exp( neg_half_inv_sigma_sq * (n - center)^2 ) using VML
      std::vector<RealT<T>> temp(length);
      RealT<T> neg_center = -center;

      if constexpr (std::is_same_v<T, float>) {
        vsAdd(
            n_mkl,
            n_vec.data(),
            &neg_center,
            temp.data());                        // temp = n - center
        vsSqr(n_mkl, temp.data(), temp.data());  // temp = (n - center)^2
        vsMul(
            n_mkl,
            temp.data(),
            &neg_half_inv_sigma_sq,
            temp.data());                          // temp = factor * (...)**2
        vsExp(n_mkl, temp.data(), coeffs.data());  // coeffs = exp(...)
      }
      else {  // Double
        vdAdd(n_mkl, n_vec.data(), &neg_center, temp.data());
        vdSqr(n_mkl, temp.data(), temp.data());
        vdMul(n_mkl, temp.data(), &neg_half_inv_sigma_sq, temp.data());
        vdExp(n_mkl, temp.data(), coeffs.data());
      }
      // Check VML errors? Assume success.
      return coeffs;
    }

    /**
     * @brief Generates Hamming window coefficients using IPP.
     * @see OmniDSP::OmniDSP::hamming_window
     */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    OneMKLOmniDSPImpl::hamming_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      std::vector<RealT<T>> coeffs(length);
      IppStatus status = ippStsErr;
      int len_int = static_cast<int>(length);

      if constexpr (std::is_same_v<T, float>) {
        status = ippsWinHamming_32f(coeffs.data(), len_int);
      }
      else {  // double
        status = ippsWinHamming_64f(coeffs.data(), len_int);
      }

      if (status != ippStsNoErr) {
        return std::unexpected(ipp_status_to_omnidsp_status(status));
      }
      return coeffs;
    }

    /**
     * @brief Generates Hann window coefficients using IPP.
     * @see OmniDSP::OmniDSP::hann_window
     */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    OneMKLOmniDSPImpl::hann_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      std::vector<RealT<T>> coeffs(length);
      IppStatus status = ippStsErr;
      int len_int = static_cast<int>(length);

      if constexpr (std::is_same_v<T, float>) {
        status = ippsWinHann_32f(coeffs.data(), len_int);
      }
      else {  // double
        status = ippsWinHann_64f(coeffs.data(), len_int);
      }

      if (status != ippStsNoErr) {
        return std::unexpected(ipp_status_to_omnidsp_status(status));
      }
      return coeffs;
    }

    /**
     * @brief Generates Kaiser window coefficients using IPP.
     * @details Converts the standard beta parameter to the alpha parameter used
     * by IPP using the relationship alpha = beta / pi.
     * @see OmniDSP::OmniDSP::kaiser_window
     */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    OneMKLOmniDSPImpl::kaiser_window(size_t length, RealT<T> beta) const
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
      IppStatus status = ippStsErr;
      int len_int = static_cast<int>(length);

      // Convert beta to IPP's alpha parameter: alpha = beta / pi
      RealT<T> alpha = beta / static_cast<RealT<T>>(M_PI);

      if constexpr (std::is_same_v<T, float>) {
        status = ippsWinKaiser_32f(coeffs.data(), len_int, alpha);
      }
      else {  // double
        status = ippsWinKaiser_64f(coeffs.data(), len_int, alpha);
      }

      if (status != ippStsNoErr) {
        // IPP Kaiser might fail for certain alpha/length combinations
        std::cerr << "IPP Kaiser window generation failed. Beta/Alpha ("
                  << alpha << ") might be too large or length (" << length
                  << ") invalid for IPP." << std::endl;
        return std::unexpected(ipp_status_to_omnidsp_status(status));
      }
      return coeffs;
    }

    /**
     * @brief Generates Rectangular window coefficients (all ones) using IPP.
     * @see OmniDSP::OmniDSP::rectangular_window
     */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    OneMKLOmniDSPImpl::rectangular_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      std::vector<RealT<T>> coeffs(length);
      IppStatus status = ippStsErr;
      int len_int = static_cast<int>(length);
      RealT<T> one = 1.0;

      // Use ippsSet to fill with ones
      if constexpr (std::is_same_v<T, float>) {
        status = ippsSet_32f(one, coeffs.data(), len_int);
      }
      else {  // double
        status = ippsSet_64f(one, coeffs.data(), len_int);
      }

      if (status != ippStsNoErr) {
        // Fallback or error
        std::cerr << "IPP Set failed for rectangular window, falling back to "
                     "std::vector."
                  << std::endl;
        coeffs.assign(length, static_cast<RealT<T>>(1.0));  // Fallback
        return coeffs;
        // return std::unexpected(ipp_status_to_omnidsp_status(status));
      }
      return coeffs;
    }

    /**
     * @brief Generates Triangular window coefficients using IPP (via Bartlett).
     * @details Assumes the IPP Bartlett window definition matches the desired
     * Triangular definition.
     * @see OmniDSP::OmniDSP::triangular_window
     */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    OneMKLOmniDSPImpl::triangular_window(size_t length) const
    {
      // IPP doesn't have a distinct triangular window. The Bartlett window is
      // often considered equivalent or very similar (depending on endpoint
      // definitions). We reuse the Bartlett implementation here.
      return bartlett_window<T>(length);
    }

    // Note: Explicit template instantiations for these methods belong in
    // src/omnidsp/backend/onemkl/backend.cpp where the OneMKLOmniDSPImpl
    // class itself is instantiated.

  }  // namespace backend
}  // namespace OmniDSP
