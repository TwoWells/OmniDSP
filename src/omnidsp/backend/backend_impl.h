// src/omnidsp/backend/backend_impl.h
#ifndef OMNIDSP_BACKEND_IMPL_H
#define OMNIDSP_BACKEND_IMPL_H

#include <vector>
#include <complex> // Include if needed by any backend function signature
#include <cstddef> // For size_t

namespace OmniDSP
{
    namespace Backend
    {

        /**
         * @brief Backend implementation for 1D convolution or correlation.
         *
         * @tparam T The floating-point type (float or double).
         * @param signal The input signal vector.
         * @param kernel The kernel vector.
         * @param use_correlation If true, perform correlation; otherwise, perform convolution.
         * For convolution, the backend might need to reverse the kernel internally.
         * @return std::vector<T> The result of the operation (e.g., 'valid' mode output).
         * @throws std::runtime_error If backend execution fails or is not available (stub).
         * @throws std::invalid_argument If inputs are invalid (e.g., empty, or kernel longer than signal for 'valid' mode).
         */
        template <typename T>
        std::vector<T> convolve1d_impl(const std::vector<T> &signal,
                                       const std::vector<T> &kernel,
                                       bool use_correlation);

        /**
         * @brief Backend implementation for combined FIR filtering and downsampling.
         *
         * Performs filtering (correlation-like, using kernel as coefficients) followed by decimation.
         *
         * @tparam T The floating-point type (float or double).
         * @param signal The input signal vector.
         * @param kernel The FIR filter coefficients vector.
         * @param factor The integer downsampling factor (M > 0).
         * @return std::vector<T> The filtered and downsampled signal.
         * @throws std::runtime_error If backend execution fails or is not available (stub).
         * @throws std::invalid_argument If inputs are invalid (e.g., empty, factor <= 0).
         */
        template <typename T>
        std::vector<T> filter_and_downsample_impl(const std::vector<T> &signal,
                                                  const std::vector<T> &kernel,
                                                  int factor);

        // Add declarations for backend FFT helpers here if needed in the future

    } // namespace Backend
} // namespace OmniDSP

#endif // OMNIDSP_BACKEND_IMPL_H