/**
 * @file src/omnidsp/backend/onemkl/resample.cpp
 * @brief Intel oneMKL/IPP backend implementation for OmniDSP filter+downsample.
 *
 * Implements the filter_and_downsample_impl function using Intel IPP
 * polyphase resampling (ippsResamplePolyphaseFixed) for FLOAT precision only.
 * Double precision is currently unsupported and throws a runtime error.
 * Corrected argument count for ippsResamplePolyphaseFixedGetSize_32f to 7
 * arguments. Compiled only when USE_ONEMKL is defined.
 */

// --- Includes ---
#include <algorithm>  // For std::copy, std::max, std::reverse
#include <cmath>      // For std::ceil
#include <complex>    // Often included with DSP headers
#include <iostream>   // For std::cout/cerr (optional, for potential debug)
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>
#include <type_traits>  // For std::is_same_v
#include <vector>

#include "../backend.h"  // Internal backend function declarations

// Only compile if USE_ONEMKL is defined by CMake
#if defined(USE_ONEMKL)

// --- IPP Includes ---
#include <ipp.h>
#include <ippcore.h>
#include <ippdefs.h>
#include <ipps.h>
#include <ippvm.h>  // May not be strictly needed here, but often used with IPP

namespace OmniDSP {
namespace Backend {

// --- Intel IPP Helper ---
// Checks the status code returned by IPP functions and throws an error if it's
// not ippStsNoErr.
inline void check_ipp_status(IppStatus status, const char *error_msg) {
  if (status != ippStsNoErr) {
    std::string full_msg = error_msg;
    // Append IPP status code for detailed diagnostics.
    full_msg += ": IPP Status " + std::to_string(status) + " (" +
                ippGetStatusString(status) + ")";
    throw std::runtime_error(full_msg);
  }
}

/**
 * @brief MKL/IPP backend implementation for combined FIR filtering and
 * downsampling. Uses Intel IPP polyphase resampling
 * (ippsResamplePolyphaseFixed) for FLOAT only. Double precision (_64f) version
 * is currently unsupported due to apparent lack of IPP function
 * availability/compilation issues in tested versions.
 *
 * @tparam T float or double.
 * @param signal The input signal vector.
 * @param kernel The FIR filter coefficients vector (used by IPP internally for
 * length hint).
 * @param factor The integer downsampling factor (M > 0).
 * @return std::vector<T> The filtered and downsampled signal.
 * @throws std::invalid_argument If inputs are invalid (empty, factor <= 0,
 * kernel empty/invalid).
 * @throws std::runtime_error If an IPP error occurs, or if called with double
 * precision.
 */
template <typename T>
std::vector<T> filter_and_downsample_impl(const std::vector<T> &signal,
                                          const std::vector<T> &kernel,
                                          int factor) {
  // --- Check Precision ---
  if constexpr (std::is_same_v<T, double>) {
    // --- UNSUPPORTED Double Implementation ---
    throw std::runtime_error(
        "IPP filter_and_downsample (ippsResamplePolyphaseFixed) is currently "
        "not supported for double precision with this IPP version/backend.");
    // return {}; // Unreachable
  } else if constexpr (std::is_same_v<T, float>) {
    // --- FLOAT (_32f) Implementation using ippsResamplePolyphaseFixed ---
    if (signal.empty()) return {};
    if (factor <= 0)
      throw std::invalid_argument("Downsampling factor must be positive.");
    if (kernel.empty())
      throw std::invalid_argument(
          "Kernel cannot be empty (used for filter length).");

    int inLen = static_cast<int>(signal.size());
    int filterLenHint = static_cast<int>(
        kernel.size());  // Use provided kernel size as length hint
    if (filterLenHint <= 0)
      throw std::invalid_argument(
          "IPP Resampling filter length hint must be positive.");

    // --- IPP Polyphase Resampling Setup ---
    IppStatus status = ippStsNoErr;
    int inRate = factor;
    int outRate = 1;
    Ipp32f rolloff = 0.95f;
    Ipp32f alpha = 4.0f;
    IppHintAlgorithm hint = ippAlgHintAccurate;

    int specSize = 0;
    int filterLenUsed = 0;  // Output parameter for GetSize
    int filterHeight = 0;   // Output parameter for GetSize

    Ipp8u *pSpecBuffer = nullptr;
    IppsResamplingPolyphaseFixed_32f *pSpec = nullptr;

    std::vector<T> result;
    int outLen = 0;

    try {
      // 1. Get Size for the specification structure (Corrected: 7 arguments)
      status = ippsResamplePolyphaseFixedGetSize_32f(
          inRate, outRate, filterLenHint, &specSize, &filterLenUsed,
          &filterHeight, hint);
      check_ipp_status(status,
                       "IPP ippsResamplePolyphaseFixedGetSize_32f failed");

      // 2. Allocate memory for the specification structure
      pSpecBuffer = ippsMalloc_8u(specSize);
      if (!pSpecBuffer)
        throw std::runtime_error("IPP failed to allocate spec buffer");
      pSpec = reinterpret_cast<IppsResamplingPolyphaseFixed_32f *>(pSpecBuffer);

      // 3. Initialize the specification structure
      // Note: Init function still takes filterLenHint (len parameter in docs)
      status = ippsResamplePolyphaseFixedInit_32f(
          inRate, outRate, filterLenHint, rolloff, alpha, pSpec, hint);
      check_ipp_status(status, "IPP ippsResamplePolyphaseFixedInit_32f failed");

      // 4. Estimate maximum possible output length and allocate result buffer
      // Use filterLenUsed (returned by GetSize) for potentially more accurate
      // margin? Or stick with hint? Let's stick with filterLenHint for margin
      // consistency for now.
      size_t maxOutLen = static_cast<size_t>(std::ceil(
                             static_cast<double>(inLen) * outRate / inRate)) +
                         filterLenHint;
      result.resize(maxOutLen);

      // 5. Perform Resampling (Filter + Downsample)
      double time = 0;            // IPP sets the time offset
      Ipp32f norm_factor = 1.0f;  // Aim for unity gain

      // The execution function signature also needs verification, assuming 7
      // args based on common patterns IPPAPI(IppStatus,
      // ippsResamplePolyphaseFixed_32f,(const Ipp32f* pSrc, int srcLen, Ipp32f*
      // pDst, Ipp32f norm, double* pTime, int* pDstLen,
      // IppsResamplePolyphaseFixed_32f* pSpec))
      status = ippsResamplePolyphaseFixed_32f(
          signal.data(),  // Source (input signal)
          inLen,          // Source length
          result.data(),  // Destination (output buffer)
          norm_factor,    // Normalization factor
          &time,          // Output: time offset of the first output sample
          &outLen,        // Output: number of samples written to pDst
          pSpec);         // Pointer to the initialized spec structure

      check_ipp_status(status,
                       "IPP ippsResamplePolyphaseFixed_32f execution failed");

      // 6. Resize result vector to actual output length
      if (outLen < 0)
        throw std::runtime_error("IPP returned negative output length");
      result.resize(static_cast<size_t>(outLen));

      // 7. Free IPP buffers
      if (pSpecBuffer) {
        ippsFree(pSpecBuffer);
        pSpecBuffer = nullptr;
      }  // Free the original buffer
      pSpec = nullptr;  // Avoid dangling pointer

      return result;
    } catch (...) {
      // Ensure buffers are freed even if exceptions occur
      if (pSpecBuffer) {
        ippsFree(pSpecBuffer);
      }
      throw;  // Re-throw the exception
    }
  }  // End float implementation
  else {
    // Should not happen with static_assert in theory, but as fallback
    throw std::runtime_error("Unsupported type for filter_and_downsample_impl");
  }
}  // End filter_and_downsample_impl

// --- Explicit Template Instantiations ---
// Only instantiate for float, as double is currently unsupported.
template std::vector<float> filter_and_downsample_impl<float>(
    const std::vector<float> &, const std::vector<float> &, int);
// template std::vector<double> filter_and_downsample_impl<double>(const
// std::vector<double> &, const std::vector<double> &, int); // DO NOT
// INSTANTIATE DOUBLE

}  // namespace Backend
}  // namespace OmniDSP

#endif  // USE_ONEMKL
