/**
 * @file resample.h
 * @brief Public API header for resampling functions in OmniDSP.
 *
 * This header defines the platform-independent interface for performing
 * operations like filtering combined with downsampling, leveraging
 * optimized backend implementations.
 *
 * @version 1.0.0
 * @date 2025-04-15
 */

#ifndef OMNIDSP_RESAMPLE_H
#define OMNIDSP_RESAMPLE_H

#include <cstddef>  // For size_t
#include <vector>

namespace OmniDSP {

/**
 * @brief Performs combined FIR filtering and downsampling.
 *
 * Applies an FIR filter (defined by kernel coefficients) to the input signal
 * and then downsamples the result by an integer factor. The exact filtering
 * mechanism and output length depend on the selected backend.
 *
 * @tparam T The floating-point type (float or double).
 * @param signal The input signal vector.
 * @param kernel The FIR filter coefficients vector. Note: Some backends (like
 * IPP FixedPolyphase) might use the kernel's length to design an internal
 * filter rather than using these coefficients directly.
 * @param factor The integer downsampling factor (must be > 0).
 * @return std::vector<T> The filtered and downsampled signal.
 * @throws std::runtime_error If backend execution fails, is not available
 * (stub), or unsupported (e.g., double precision IPP).
 * @throws std::invalid_argument If inputs are invalid (e.g., empty
 * signal/kernel, factor <= 0).
 */
template <typename T>
std::vector<T> filter_and_downsample(const std::vector<T> &signal,
                                     const std::vector<T> &kernel, int factor);

// --- Explicit Template Instantiations (Declarations) ---
// Declare instantiations for float and double.
// The definitions that call the backend implementations reside in resample.cpp.
/** @cond OMNIDSP_INTERNAL */  // Hide from Doxygen index if desired
extern template std::vector<float> filter_and_downsample<float>(
    const std::vector<float> &signal, const std::vector<float> &kernel,
    int factor);
extern template std::vector<double> filter_and_downsample<double>(
    const std::vector<double> &signal, const std::vector<double> &kernel,
    int factor);
/** @endcond */

}  // namespace OmniDSP

#endif  // OMNIDSP_RESAMPLE_H
