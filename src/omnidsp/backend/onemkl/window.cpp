/**
 * @file src/omnidsp/backend/onemkl/window.cpp
 * @brief Intel oneMKL/IPP backend implementation for OmniDSP window functions.
 *
 * Implements backend functions to generate window coefficients using IPP
 * where available (Hann, Hamming, Kaiser). Flattop uses a manual calculation.
 * Corrected function signatures to void return and T* output parameter.
 * Compiled only when USE_ONEMKL is defined.
 */

// --- Includes ---
#include <cmath>      // For M_PI, cos, sin, sqrt, std::abs
#include <limits>     // For std::numeric_limits
#include <numeric>    // For std::accumulate
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>
#include <type_traits>  // For std::is_same_v
#include <vector>

#include "../backend.h"  // Internal backend function declarations

// Only compile if USE_ONEMKL is defined by CMake
#if defined(USE_ONEMKL)

#include <ipp.h>
#include <ipps.h>  // Header containing ippsWin* functions

// Define M_PI if it's not already defined (e.g., by <cmath> in some
// environments)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace OmniDSP {
namespace Backend {
namespace oneMKL {  // <<< Added oneMKL namespace wrapper

// --- Intel IPP Helper ---
// Checks the status code returned by IPP functions and throws an error if it's
// not ippStsNoErr.
inline void check_ipp_status(IppStatus status, const char* error_msg) {
  if (status != ippStsNoErr) {
    std::string full_msg = error_msg;
    // Append IPP status code for detailed diagnostics.
    full_msg += ": IPP Status " + std::to_string(status) + " (" +
                ippGetStatusString(status) + ")";
    throw std::runtime_error(full_msg);
  }
}

// --- Backend Window Generation Implementations ---

/**
 * @brief Generates Hann window coefficients using IPP.
 * Creates a source vector of 1s and applies the window to the output buffer.
 */
template <typename T>
void hann_window_impl(T* output, size_t length) {  // <<< CORRECTED SIGNATURE
  if (length == 0) return;
  // Don't create local coeffs vector
  std::vector<T> src(length, static_cast<T>(1.0));  // Source vector of 1s
  IppStatus status = ippStsNoErr;
  int len_int = static_cast<int>(length);  // IPP uses int for length

  if constexpr (std::is_same_v<T, float>) {
    // Call ippsWinHann_32f(pSrc, pDst, len) - Takes 3 args
    status =
        ippsWinHann_32f(src.data(), output, len_int);  // <<< Write to output
    check_ipp_status(status, "ippsWinHann_32f failed");
  } else {  // double
    // Call ippsWinHann_64f(pSrc, pDst, len) - Takes 3 args
    status =
        ippsWinHann_64f(src.data(), output, len_int);  // <<< Write to output
    check_ipp_status(status, "ippsWinHann_64f failed");
  }
  // No return
}

/**
 * @brief Generates Hamming window coefficients using IPP.
 * Creates a source vector of 1s and applies the window to the output buffer.
 */
template <typename T>
void hamming_window_impl(T* output, size_t length) {  // <<< CORRECTED SIGNATURE
  if (length == 0) return;
  std::vector<T> src(length, static_cast<T>(1.0));  // Source vector of 1s
  IppStatus status = ippStsNoErr;
  int len_int = static_cast<int>(length);

  if constexpr (std::is_same_v<T, float>) {
    // Call ippsWinHamming_32f(pSrc, pDst, len) - Takes 3 args
    status =
        ippsWinHamming_32f(src.data(), output, len_int);  // <<< Write to output
    check_ipp_status(status, "ippsWinHamming_32f failed");
  } else {  // double
    // Call ippsWinHamming_64f(pSrc, pDst, len) - Takes 3 args
    status =
        ippsWinHamming_64f(src.data(), output, len_int);  // <<< Write to output
    check_ipp_status(status, "ippsWinHamming_64f failed");
  }
  // No return
}

/**
 * @brief Generates Kaiser window coefficients using IPP.
 * Creates a source vector of 1s and applies the window to the output buffer.
 * @param beta The Kaiser window beta parameter.
 */
template <typename T>
void kaiser_window_impl(T* output, size_t length,
                        T beta) {  // <<< CORRECTED SIGNATURE
  if (length == 0) return;
  std::vector<T> src(length, static_cast<T>(1.0));  // Source vector of 1s
  IppStatus status = ippStsNoErr;
  int len_int = static_cast<int>(length);
  // IPP uses alpha parameter, where alpha = beta / pi
  T alpha = beta / static_cast<T>(M_PI);

  if constexpr (std::is_same_v<T, float>) {
    // Call ippsWinKaiser_32f(pSrc, pDst, len, alpha) - Takes 4 args
    status = ippsWinKaiser_32f(src.data(), output, len_int,
                               alpha);  // <<< Write to output
    check_ipp_status(status, "ippsWinKaiser_32f failed");
  } else {  // double
    // Call ippsWinKaiser_64f(pSrc, pDst, len, alpha) - Takes 4 args
    status = ippsWinKaiser_64f(src.data(), output, len_int,
                               alpha);  // <<< Write to output
    check_ipp_status(status, "ippsWinKaiser_64f failed");
  }
  // No return
}

/**
 * @brief Generates Flattop window coefficients (manual calculation, as IPP
 * lacks native support) and writes them to the output buffer.
 */
template <typename T>
void flattop_window_impl(T* output, size_t length) {  // <<< CORRECTED SIGNATURE
  if (length == 0) return;
  // Coefficients from standard definition (e.g., SciPy)
  constexpr T a0 = static_cast<T>(0.21557895);
  constexpr T a1 = static_cast<T>(0.41663158);
  constexpr T a2 = static_cast<T>(0.277263158);
  constexpr T a3 = static_cast<T>(0.083578947);
  constexpr T a4 = static_cast<T>(0.006947368);
  // Denominator for cosine terms
  double denom = (length > 1) ? static_cast<double>(length - 1) : 1.0;
  if (denom == 0.0) denom = 1.0;  // Avoid division by zero for length=1

  for (size_t n = 0; n < length; ++n) {
    double angle_base = 2.0 * M_PI * static_cast<double>(n) / denom;
    T cos_1x_term = static_cast<T>(std::cos(angle_base));
    T cos_2x_term = static_cast<T>(std::cos(2.0 * angle_base));
    T cos_3x_term = static_cast<T>(std::cos(3.0 * angle_base));
    T cos_4x_term = static_cast<T>(std::cos(4.0 * angle_base));
    // Write directly to output buffer
    output[n] = a0 - a1 * cos_1x_term + a2 * cos_2x_term - a3 * cos_3x_term +
                a4 * cos_4x_term;
  }
  // No return
}

// --- Explicit Template Instantiations --- // <<< CORRECTED SIGNATURES
template void hann_window_impl<float>(float* output, size_t length);
template void hann_window_impl<double>(double* output, size_t length);

template void hamming_window_impl<float>(float* output, size_t length);
template void hamming_window_impl<double>(double* output, size_t length);

template void kaiser_window_impl<float>(float* output, size_t length,
                                        float beta);
template void kaiser_window_impl<double>(double* output, size_t length,
                                         double beta);

template void flattop_window_impl<float>(float* output, size_t length);
template void flattop_window_impl<double>(double* output, size_t length);

}  // namespace oneMKL
}  // namespace Backend
}  // namespace OmniDSP

#endif  // USE_ONEMKL
