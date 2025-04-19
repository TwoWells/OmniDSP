/**
 * @file src/omnidsp/backend/accelerate/resample.cpp
 * @brief Apple Accelerate backend implementation for OmniDSP filter+downsample.
 *
 * Implements the filter_and_downsample_impl function using Apple's Accelerate
 * framework (vDSP), specifically the vDSP_desamp function.
 * Compiled only when USE_ACCELERATE is defined.
 */

// --- Includes ---
#include <cmath>      // For std::ceil (though vDSP calculates output length)
#include <complex>    // Often included with DSP headers
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>
#include <type_traits>  // For std::is_same_v
#include <vector>
#include <vector>  // Ensure std::vector is included

#include "../backend.h"  // Internal backend function declarations

// Only compile if USE_ACCELERATE is defined by CMake (typically on macOS)
#if defined(USE_ACCELERATE)

#include <Accelerate/Accelerate.h>  // Main header for the Accelerate framework (includes vDSP)

namespace OmniDSP {
namespace Backend {

// --- vDSP Helper Struct ---
// Use template specialization to select the correct vDSP function
// (float/double) at compile time.

// Helper for vDSP_desamp (Decimate and Sample - FIR filter + downsample)
template <typename T>
struct vDSPDesampHelper;
template <>
struct vDSPDesampHelper<float> {
  static constexpr auto desamp = vDSP_desamp;
};  // vDSP_desamp for float
template <>
struct vDSPDesampHelper<double> {
  static constexpr auto desamp = vDSP_desampD;
};  // vDSP_desampD for double

/**
 * @brief Accelerate backend implementation for combined FIR filtering and
 * downsampling. Uses vDSP_desamp which performs both operations efficiently
 * using the provided kernel.
 *
 * @tparam T float or double.
 * @param signal The input signal vector.
 * @param kernel The FIR filter coefficients vector.
 * @param factor The integer downsampling factor (M > 0).
 * @return std::vector<T> The filtered and downsampled signal.
 * @throws std::invalid_argument If inputs are invalid (empty, factor <= 0,
 * kernel empty).
 */
template <typename T>
std::vector<T> filter_and_downsample_impl(const std::vector<T> &signal,
                                          const std::vector<T> &kernel,
                                          int factor) {
  // --- Input Validation ---
  if (signal.empty() || kernel.empty()) {
    return {};  // Return empty if inputs are empty
  }
  if (factor <= 0) {
    throw std::invalid_argument("Downsampling factor must be positive.");
  }

  // vDSP uses vDSP_Length (unsigned long) for lengths
  vDSP_Length sig_len = signal.size();
  vDSP_Length ker_len = kernel.size();  // Filter length
  vDSP_Length downsample_factor =
      static_cast<vDSP_Length>(factor);  // Cast factor

  // Calculate output length for vDSP_desamp: floor((sig_len - ker_len) /
  // factor) + 1 Need careful calculation to handle potential negative numerator
  // if sig_len < ker_len
  long long numerator =
      static_cast<long long>(sig_len) - static_cast<long long>(ker_len);

  // Calculate output length, ensuring it's non-negative
  vDSP_Length out_len = 0;
  if (numerator >= 0) {
    // Note: Integer division automatically performs floor operation
    out_len = (static_cast<vDSP_Length>(numerator) / downsample_factor) + 1;
  }  // else out_len remains 0

  if (out_len == 0) {
    return {};  // No output samples possible, return empty vector
  }

  std::vector<T> result(out_len);  // Allocate result vector

  // Call vDSP_desamp (or vDSP_desampD) using the helper struct
  // Signature: vDSP_desamp(InputSignal, DecimationFactor, FilterCoeffs, Output,
  // OutputLength, FilterLength) vDSP Documentation uses: (A, D, F, C, N, L)
  vDSPDesampHelper<T>::desamp(
      signal.data(),      // Input signal (A)
      downsample_factor,  // Decimation factor (D)
      kernel.data(),      // Filter coefficients (F) - vDSP uses these directly
      result.data(),      // Output buffer (C)
      out_len,            // Number of output samples to calculate (N)
      ker_len);           // Length of filter kernel (L)

  return result;
}

// --- Explicit Template Instantiations ---
// Ensures the compiler generates code for both float and double versions.
template std::vector<float> filter_and_downsample_impl<float>(
    const std::vector<float> &, const std::vector<float> &, int);
template std::vector<double> filter_and_downsample_impl<double>(
    const std::vector<double> &, const std::vector<double> &, int);

}  // namespace Backend
}  // namespace OmniDSP

#endif  // USE_ACCELERATE
