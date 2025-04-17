/**
 * @file convolution.h
 * @brief Public API header for convolution and correlation functions in
 * OmniDSP.
 *
 * This header defines the platform-independent interface for performing
 * 1D convolution and correlation, leveraging optimized backend implementations.
 *
 * @version 1.1.0 // Version bump for API change
 * @date 2025-04-11 // Or current date
 */

#ifndef OMNIDSP_CONVOLUTION_H
#define OMNIDSP_CONVOLUTION_H

#include <complex>  // Include complex in case complex versions are added later
#include <cstddef>  // For size_t
#include <vector>

namespace OmniDSP {

/**
 * @brief Performs 1D linear convolution using optimized backends.
 *
 * Calculates y[n] = sum_{k} x[k] * h[n-k], where h is the kernel.
 * Assumes the kernel h[k] provided is the impulse response for convolution.
 * The output size currently corresponds to the 'valid' part of the convolution
 * where the kernel fully overlaps the signal, resulting in an output vector of
 * size signal.size() - kernel.size() + 1.
 * (Note: Output size behavior might need adjustment based on desired mode like
 * 'full' or 'same').
 *
 * @tparam T The floating-point type (float or double).
 * @param signal The input signal vector (x[n]).
 * @param kernel The kernel vector (impulse response h[k]).
 * @return std::vector<T> The result of the convolution.
 * @throws std::runtime_error If backend execution fails.
 * @throws std::invalid_argument If signal or kernel is empty, or if kernel is
 * longer than signal for 'valid' mode.
 */
template <typename T>
std::vector<T> convolve1d(const std::vector<T> &signal,
                          const std::vector<T> &kernel);

/**
 * @brief Performs 1D linear correlation using optimized backends.
 *
 * Calculates y[n] = sum_{k} x[n+k] * h[k], where h is the kernel.
 * This is suitable for standard FIR filtering where h[k] represents the
 * filter coefficients directly.
 * The output size currently corresponds to the 'valid' part of the correlation
 * where the kernel fully overlaps the signal, resulting in an output vector of
 * size signal.size() - kernel.size() + 1.
 * (Note: Output size behavior might need adjustment based on desired mode like
 * 'full' or 'same').
 *
 * @tparam T The floating-point type (float or double).
 * @param signal The input signal vector (x[n]).
 * @param kernel The kernel vector (e.g., FIR coefficients h[k]).
 * @return std::vector<T> The result of the correlation.
 * @throws std::runtime_error If backend execution fails.
 * @throws std::invalid_argument If signal or kernel is empty, or if kernel is
 * longer than signal for 'valid' mode.
 */
template <typename T>
std::vector<T> correlate1d(const std::vector<T> &signal,
                           const std::vector<T> &kernel);

// --- Explicit Template Instantiations (Declarations) ---
// Declare instantiations for float and double for both functions.
// The definitions that call the backend implementations should reside in a .cpp
// file.
/** @cond OMNIDSP_INTERNAL */
extern template std::vector<float> convolve1d<float>(
    const std::vector<float> &signal, const std::vector<float> &kernel);
extern template std::vector<double> convolve1d<double>(
    const std::vector<double> &signal, const std::vector<double> &kernel);

extern template std::vector<float> correlate1d<float>(
    const std::vector<float> &signal, const std::vector<float> &kernel);
extern template std::vector<double> correlate1d<double>(
    const std::vector<double> &signal, const std::vector<double> &kernel);
/** @endcond */

}  // namespace OmniDSP

#endif  // OMNIDSP_CONVOLUTION_H
