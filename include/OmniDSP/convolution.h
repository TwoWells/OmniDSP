/**
 * @file convolution.h
 * @brief Public API header for convolution and correlation functions in
 * OmniDSP.
 *
 * This header defines the platform-independent interface for performing
 * 1D convolution and correlation with different modes ('Full', 'Same',
 * 'Valid'), leveraging optimized backend implementations.
 *
 * @version 1.2.1 // Version bump for API signature change
 * @date 2025-04-18
 */

#ifndef OMNIDSP_CONVOLUTION_H
#define OMNIDSP_CONVOLUTION_H

#include <complex>  // Include complex in case complex versions are added later
#include <cstddef>  // For size_t
#include <stdexcept>  // For std::invalid_argument
#include <vector>

#include "omnidsp.h"  // For OMNIDSP_STATUS potentially, keep for consistency

namespace OmniDSP {

/**
 * @brief Enum defining the convolution/correlation mode.
 * Determines the size of the output array.
 */
enum class ConvMode {
  Full,  ///< Returns the full discrete linear convolution (size N+M-1).
  Same,  ///< Returns the central part, same size as the first input 'signal'
         ///< (size N).
  Valid  ///< Returns only parts where inputs fully overlap (size N-M+1,
         ///< requires N>=M).
};

/**
 * @brief Performs 1D linear convolution using optimized backends.
 *
 * Calculates y[n] = sum_{k} x[k] * h[n-k], where h is the kernel.
 * Assumes the kernel h[k] provided is the impulse response for convolution.
 * The output size depends on the specified mode.
 *
 * @tparam T The floating-point type (float or double).
 * @param signal The input signal vector (x[n]).
 * @param kernel The kernel vector (impulse response h[k]).
 * @param mode The convolution mode (Full, Same, Valid). Defaults to Valid.
 * @return std::vector<T> The result of the convolution.
 * @throws std::runtime_error If backend execution fails.
 * @throws std::invalid_argument If inputs are invalid or mode is unsupported by
 * backend/size combination (e.g., kernel longer than signal for 'Valid').
 */
template <typename T>
/* OMNIDSP_API removed */ std::vector<T> convolve1d(
    const std::vector<T> &signal, const std::vector<T> &kernel,
    ConvMode mode = ConvMode::Valid);

/**
 * @brief Performs 1D linear correlation using optimized backends.
 *
 * Calculates y[n] = sum_{k} x[n+k] * h[k], where h is the kernel.
 * This is suitable for standard FIR filtering where h[k] represents the
 * filter coefficients directly.
 * The output size depends on the specified mode.
 *
 * @tparam T The floating-point type (float or double).
 * @param signal The input signal vector (x[n]).
 * @param kernel The kernel vector (e.g., FIR coefficients h[k]).
 * @param mode The correlation mode (Full, Same, Valid). Defaults to Valid.
 * @return std::vector<T> The result of the correlation.
 * @throws std::runtime_error If backend execution fails.
 * @throws std::invalid_argument If inputs are invalid or mode is unsupported by
 * backend/size combination.
 */
template <typename T>
/* OMNIDSP_API removed */ std::vector<T> correlate1d(
    const std::vector<T> &signal, const std::vector<T> &kernel,
    ConvMode mode = ConvMode::Valid);

// --- Explicit Template Instantiations (Declarations) ---
// Declare instantiations for float and double for both functions.
// The definitions that call the backend implementations should reside in a .cpp
// file.
/** @cond OMNIDSP_INTERNAL */
extern template /* OMNIDSP_API removed */ std::vector<float> convolve1d<float>(
    const std::vector<float> &signal, const std::vector<float> &kernel,
    ConvMode mode);
extern template /* OMNIDSP_API removed */ std::vector<double>
convolve1d<double>(const std::vector<double> &signal,
                   const std::vector<double> &kernel, ConvMode mode);

extern template /* OMNIDSP_API removed */ std::vector<float> correlate1d<float>(
    const std::vector<float> &signal, const std::vector<float> &kernel,
    ConvMode mode);
extern template /* OMNIDSP_API removed */ std::vector<double>
correlate1d<double>(const std::vector<double> &signal,
                    const std::vector<double> &kernel, ConvMode mode);
/** @endcond */

}  // namespace OmniDSP

#endif  // OMNIDSP_CONVOLUTION_H
