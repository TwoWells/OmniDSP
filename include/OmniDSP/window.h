#ifndef OMNIDSP_WINDOW_H
#define OMNIDSP_WINDOW_H

#include <vector>
#include <cmath>    // cos, M_PI, cyc_bessel_i
#include <limits>   // numeric_limits

namespace OmniDSP {

/**
 * @brief Provides a suite of window functions commonly used in signal processing.
 *
 * Windowing functions are applied to input data before performing a Fourier Transform
 * to reduce spectral leakage and improve the accuracy of the analysis.
 */
class Window {
public:
    /**
     * @brief Applies the Hann window to the input data.
     *
     * The Hann window is defined as:
     * w[n] = 0.5 - 0.5 * cos(2*pi*n / (N-1)) for 0 <= n <= N-1
     *
     * @tparam T The floating-point type of the data (float or double).
     * @param input A vector of input data.
     * @return A vector of windowed data.
     * @throws std::invalid_argument If the input vector is empty.
     */
    template <typename T>
    static std::vector<T> hann(const std::vector<T>& input) {
        if (input.empty()) {
            throw std::invalid_argument("Input vector cannot be empty.");
        }
        size_t N = input.size();
        std::vector<T> output(N);
        for (size_t n = 0; n < N; ++n) {
            output[n] = input[n] * static_cast<T>(0.5 - 0.5 * cos(2.0 * M_PI * n / (N - 1)));
        }
        return output;
    }

    /**
     * @brief Applies the Hamming window to the input data.
     *
     * The Hamming window is defined as:
     * w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1)) for 0 <= n <= N-1
     *
     * @tparam T The floating-point type of the data (float or double).
     * @param input A vector of input data.
     * @return A vector of windowed data.
     * @throws std::invalid_argument If the input vector is empty.
     */
    template <typename T>
    static std::vector<T> hamming(const std::vector<T>& input) {
        if (input.empty()) {
            throw std::invalid_argument("Input vector cannot be empty.");
        }
        size_t N = input.size();
        std::vector<T> output(N);
        for (size_t n = 0; n < N; ++n) {
            output[n] = input[n] * static_cast<T>(0.54 - 0.46 * cos(2.0 * M_PI * n / (N - 1)));
        }
        return output;
    }

    /**
     * @brief Applies the Kaiser window to the input data.
     *
     * The Kaiser window is defined as:
     * w[n] = I0(beta * sqrt(1 - (2n/(N-1) - 1)^2)) / I0(beta) for 0 <= n <= N-1
     * where I0 is the zeroth-order modified Bessel function of the first kind,
     * and beta is a shape parameter that controls the trade-off
     * between main lobe width and sidelobe level.
     *
     * @tparam T The floating-point type of the data (float or double).
     * @param input A vector of input data.
     * @param beta  The shape parameter of the Kaiser window.
     * A typical value is 8.6
     * @return A vector of windowed data.
     * @throws std::invalid_argument If the input vector is empty.
     */
    template <typename T>
    static std::vector<T> kaiser(const std::vector<T>& input, T beta) {
        if (input.empty()) {
            throw std::invalid_argument("Input vector cannot be empty.");
        }
        size_t N = input.size();
        std::vector<T> output(N);

        for (size_t n = 0; n < N; ++n) {
            T factor = (n * 2.0) / (N - 1) - 1.0;
            T sqrt_term = sqrt(1.0 - factor * factor);
            output[n] = input[n] * std::cyl_bessel_i(0, beta * sqrt_term) / std::cyl_bessel_i(0, beta);
        }
        return output;
    }

    /**
     * @brief Applies the Flat-top window to the input data.
     *
     * The Flat-top window is designed to have a very flat passband
     * and is often used for accurate amplitude measurements.
     * It is defined as:
     * w[n] = a0 - a1*cos(2*pi*n/(N-1)) + a2*cos(4*pi*n/(N-1)) - a3*cos(6*pi*n/(N-1)) + a4*cos(8*pi*n/(N-1))
     * for 0 <= n <= N-1
     * where a0 = 0.21557895, a1 = 0.41663158, a2 = 0.277263158, a3 = 0.083578947, and a4 = 0.006947368
     *
     * @tparam T The floating-point type of the data (float or double).
     * @param input A vector of input data.
     * @return A vector of windowed data.
     * @throws std::invalid_argument If the input vector is empty.
     */
    template <typename T>
    static std::vector<T> flattop(const std::vector<T>& input) {
        if (input.empty()) {
            throw std::invalid_argument("Input vector cannot be empty.");
        }
        size_t N = input.size();
        std::vector<T> output(N);
        constexpr T a0 = static_cast<T>(0.21557895);
        constexpr T a1 = static_cast<T>(0.41663158);
        constexpr T a2 = static_cast<T>(0.277263158);
        constexpr T a3 = static_cast<T>(0.083578947);
        constexpr T a4 = static_cast<T>(0.006947368);

        for (size_t n = 0; n < N; ++n) {
            output[n] = input[n] * static_cast<T>(a0 - a1 * cos(2.0 * M_PI * n / (N - 1)) +
                                       a2 * cos(4.0 * M_PI * n / (N - 1)) -
                                       a3 * cos(6.0 * M_PI * n / (N - 1)) +
                                       a4 * cos(8.0 * M_PI * n / (N - 1)));
        }
        return output;
    }
};

} // namespace OmniDSP

#endif