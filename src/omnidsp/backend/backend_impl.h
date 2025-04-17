/**
 * @file src/omnidsp/backend/backend_impl.h
 * @brief Internal header declaring the interface for backend implementations.
 *
 * This file defines the function signatures that platform-specific backend
 * source files (e.g., onemkl/convolution.cpp, accelerate/resample.cpp) must
 * implement. It acts as the abstraction layer between the public API/core
 * library logic and the specific backend code.
 */

#ifndef OMNIDSP_BACKEND_IMPL_H
#define OMNIDSP_BACKEND_IMPL_H

#include <complex>  // Include complex as it might be needed by some backend details indirectly
#include <cstddef>  // For size_t
#include <vector>

namespace OmniDSP {
// Forward declaration for FFTPlanImpl (though not strictly needed in this
// header itself) template <typename T> struct FFTPlanImpl;

namespace Backend {

// --- Convolution / Correlation ---

/**
 * @brief Backend implementation for 1D convolution or correlation.
 *
 * @tparam T The floating-point type (float or double).
 * @param signal The input signal vector.
 * @param kernel The kernel vector.
 * @param use_correlation If true, perform correlation; otherwise, perform
 * convolution. The backend handles kernel reversal internally if needed for
 * convolution.
 * @return std::vector<T> The result of the operation (typically 'valid' mode
 * output).
 * @throws std::runtime_error If backend execution fails or is not available
 * (stub).
 * @throws std::invalid_argument If inputs are invalid (e.g., empty, or kernel
 * longer than signal).
 */
template <typename T>
std::vector<T> convolve1d_impl(const std::vector<T> &signal,
                               const std::vector<T> &kernel,
                               bool use_correlation);

// --- Resampling / Filtering + Downsampling ---

/**
 * @brief Backend implementation for combined FIR filtering and downsampling.
 *
 * Performs filtering (correlation-like, using kernel as coefficients, OR
 * using internal filter based on kernel length for some backends like IPP
 * FixedPolyphase) followed by decimation.
 *
 * @tparam T The floating-point type (float or double).
 * @param signal The input signal vector.
 * @param kernel The FIR filter coefficients vector (or used for length hint).
 * @param factor The integer downsampling factor (M > 0).
 * @return std::vector<T> The filtered and downsampled signal.
 * @throws std::runtime_error If backend execution fails, is not available
 * (stub), or unsupported (e.g., double precision IPP).
 * @throws std::invalid_argument If inputs are invalid (e.g., empty, factor <=
 * 0).
 */
template <typename T>
std::vector<T> filter_and_downsample_impl(const std::vector<T> &signal,
                                          const std::vector<T> &kernel,
                                          int factor);

// --- Window Coefficient Generation ---
// These functions generate the window coefficients only.
// The public API (`OmniDSP::Window`) applies these coefficients to the input
// signal.

/**
 * @brief Backend implementation for generating Hann window coefficients.
 * @param length The desired window length.
 * @return std::vector<T> Vector of window coefficients.
 */
template <typename T>
std::vector<T> generate_hann_window_impl(size_t length);

/**
 * @brief Backend implementation for generating Hamming window coefficients.
 * @param length The desired window length.
 * @return std::vector<T> Vector of window coefficients.
 */
template <typename T>
std::vector<T> generate_hamming_window_impl(size_t length);

/**
 * @brief Backend implementation for generating Kaiser window coefficients.
 * @param length The desired window length.
 * @param beta The Kaiser window beta parameter.
 * @return std::vector<T> Vector of window coefficients.
 */
template <typename T>
std::vector<T> generate_kaiser_window_impl(size_t length, T beta);

/**
 * @brief Backend implementation for generating Flattop window coefficients.
 * @param length The desired window length.
 * @return std::vector<T> Vector of window coefficients.
 */
template <typename T>
std::vector<T> generate_flattop_window_impl(size_t length);

// Add declarations for other backend functions here if needed in the future
// (e.g., specific matrix operations, other transforms)

}  // namespace Backend
}  // namespace OmniDSP

#endif  // OMNIDSP_BACKEND_IMPL_H
