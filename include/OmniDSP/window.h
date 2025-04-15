/**
 * @file window.h
 * @brief Public API header for windowing functions in OmniDSP.
 *
 * Declares the OmniDSP::Window class with static methods for applying
 * various window functions to input signals.
 *
 * @version 2.0.0 (Refactored API)
 * @date 2025-04-15
 */
#ifndef OMNIDSP_WINDOW_H
#define OMNIDSP_WINDOW_H

#include <vector>
#include <cmath>    // For std::cyl_bessel_i potentially used in documentation or examples
#include <limits>   // For std::numeric_limits potentially used in documentation or examples
#include <stdexcept> // For std::invalid_argument documentation

// Forward declare backend functions (optional, but can help clarity)
// namespace OmniDSP { namespace Backend {
//     template <typename T> std::vector<T> generate_hann_window_impl(size_t length);
//     // ... other declarations ...
// }}

namespace OmniDSP {

/**
 * @brief Provides static methods for applying window functions to signals.
 *
 * Windowing functions are applied to input data (e.g., before an FFT)
 * to reduce spectral leakage. These methods generate the appropriate
 * window coefficients using the selected backend and multiply them
 * element-wise with the input signal.
 */
class Window {
public:
    /**
     * @brief Applies the Hann window to the input data.
     * w[n] = 0.5 - 0.5 * cos(2*pi*n / (N-1))
     *
     * @tparam T The floating-point type (float or double).
     * @param input A vector of input data.
     * @return A vector containing the input data multiplied by the Hann window.
     * @throws std::invalid_argument If the input vector is empty.
     * @throws std::runtime_error If the backend window generation fails.
     */
    template <typename T>
    static std::vector<T> hann(const std::vector<T>& input);

    /**
     * @brief Applies the Hamming window to the input data.
     * w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
     *
     * @tparam T The floating-point type (float or double).
     * @param input A vector of input data.
     * @return A vector containing the input data multiplied by the Hamming window.
     * @throws std::invalid_argument If the input vector is empty.
     * @throws std::runtime_error If the backend window generation fails.
     */
    template <typename T>
    static std::vector<T> hamming(const std::vector<T>& input);

    /**
     * @brief Applies the Kaiser window to the input data.
     * Defined using the zeroth-order modified Bessel function I0.
     * w[n] = I0(beta * sqrt(1 - (2n/(N-1) - 1)^2)) / I0(beta)
     *
     * @tparam T The floating-point type (float or double).
     * @param input A vector of input data.
     * @param beta  The shape parameter beta (non-negative). Typical values range from 5 to 10.
     * @return A vector containing the input data multiplied by the Kaiser window.
     * @throws std::invalid_argument If the input vector is empty or beta is negative.
     * @throws std::runtime_error If the backend window generation fails.
     */
    template <typename T>
    static std::vector<T> kaiser(const std::vector<T>& input, T beta);

    /**
     * @brief Applies the Flat-top window to the input data.
     * Designed for accurate amplitude measurements in the frequency domain.
     * w[n] = a0 - a1*cos(2pi*n/(N-1)) + a2*cos(4pi*n/(N-1)) - a3*cos(6pi*n/(N-1)) + a4*cos(8pi*n/(N-1))
     * (Specific coefficients a0-a4 are defined in the implementation).
     *
     * @tparam T The floating-point type (float or double).
     * @param input A vector of input data.
     * @return A vector containing the input data multiplied by the Flat-top window.
     * @throws std::invalid_argument If the input vector is empty.
     * @throws std::runtime_error If the backend window generation fails.
     */
    template <typename T>
    static std::vector<T> flattop(const std::vector<T>& input);

    // Note: Explicit template instantiations are defined in window.cpp
};

} // namespace OmniDSP

#endif // OMNIDSP_WINDOW_H
