/**
 * @file backend.h
 * @brief Declares the functions that specific backend implementations must
 * provide.
 *
 * This header defines the signatures for functions related to windowing,
 * convolution, correlation, and resampling that are dispatched to the
 * selected backend (oneMKL, Accelerate, or Stub) via backend.cpp.
 *
 * @version 2.0.1 (Renamed from backend_impl.h)
 * @date 2025-04-18
 */
#ifndef OMNIDSP_BACKEND_H  // Renamed Header Guard
#define OMNIDSP_BACKEND_H  // Renamed Header Guard

#include <complex>
#include <cstddef>  // For size_t
#include <vector>

// Include necessary public API headers for types used in signatures
#include <OmniDSP/convolution.h>  // Defines ConvMode
#include <OmniDSP/core_types.h>   // Defines Precision, OMNIDSP_STATUS
// #include <OmniDSP/fft.h>       // Defines FFTNorm (Only needed if FFT backend
// funcs were declared here)

// --- Forward declarations for PIMPL implementation structs ---
// These structs are defined within the backend-specific .cpp files (e.g.,
// fft.cpp) They are NOT defined or used as interfaces in this header anymore.
// template <typename T> struct FFTPlanImpl;
// template <typename T> struct RFFTPlanImpl;

namespace OmniDSP {
namespace Backend {

// --- Backend Window Function Declarations ---
// These functions generate window coefficients directly.
// Implementations reside in backend/<backend_name>/window.cpp

/**
 * @brief Backend implementation for generating Hann window coefficients.
 * @param[out] output Pointer to the output buffer where coefficients will be
 * written.
 * @param length The number of coefficients to generate (size of the output
 * buffer).
 */
template <typename T>
void hann_window_impl(T* output, size_t length);

/**
 * @brief Backend implementation for generating Hamming window coefficients.
 * @param[out] output Pointer to the output buffer.
 * @param length The number of coefficients to generate.
 */
template <typename T>
void hamming_window_impl(T* output, size_t length);

/**
 * @brief Backend implementation for generating Kaiser window coefficients.
 * @param[out] output Pointer to the output buffer.
 * @param length The number of coefficients to generate.
 * @param beta The Kaiser window shape parameter beta (non-negative).
 */
template <typename T>
void kaiser_window_impl(T* output, size_t length, T beta);

/**
 * @brief Backend implementation for generating Flat-top window coefficients.
 * @param[out] output Pointer to the output buffer.
 * @param length The number of coefficients to generate.
 */
template <typename T>
void flattop_window_impl(T* output, size_t length);

// --- Backend Convolution/Correlation Function Declarations ---
// These functions perform the core computation.
// Implementations reside in backend/<backend_name>/convolution.cpp and are
// dispatched via backend.cpp

/**
 * @brief Backend implementation for 1D convolution.
 * @param signal Pointer to the input signal data.
 * @param signal_len Length of the input signal.
 * @param kernel Pointer to the kernel data.
 * @param kernel_len Length of the kernel.
 * @param mode Convolution mode (Full, Same, Valid).
 * @return std::vector<T> The convolution result.
 * @throws std::runtime_error If backend execution fails.
 * @throws std::invalid_argument If inputs/mode are invalid.
 */
template <typename T>
std::vector<T> convolve1d_impl(const T* signal, size_t signal_len,
                               const T* kernel, size_t kernel_len,
                               OmniDSP::ConvMode mode);

/**
 * @brief Backend implementation for 1D correlation.
 * @param signal Pointer to the input signal data.
 * @param signal_len Length of the input signal.
 * @param kernel Pointer to the kernel data.
 * @param kernel_len Length of the kernel.
 * @param mode Correlation mode (Full, Same, Valid).
 * @return std::vector<T> The correlation result.
 * @throws std::runtime_error If backend execution fails.
 * @throws std::invalid_argument If inputs/mode are invalid.
 */
template <typename T>
std::vector<T> correlate1d_impl(const T* signal, size_t signal_len,
                                const T* kernel, size_t kernel_len,
                                OmniDSP::ConvMode mode);

// --- Backend Resample Function Declarations ---
// Implementations reside in backend/<backend_name>/resample.cpp and are
// dispatched via backend.cpp

/**
 * @brief Backend implementation for combined FIR filtering and downsampling.
 * @tparam T float or double.
 * @param signal The input signal vector.
 * @param kernel The FIR filter coefficients vector.
 * @param factor The integer downsampling factor (must be > 0).
 * @return std::vector<T> The filtered and downsampled signal.
 * @throws std::runtime_error If backend execution fails or is unsupported.
 * @throws std::invalid_argument If inputs are invalid.
 */
template <typename T>
std::vector<T> filter_and_downsample_impl(const std::vector<T>& signal,
                                          const std::vector<T>& kernel,
                                          int factor);

}  // namespace Backend
}  // namespace OmniDSP

#endif  // OMNIDSP_BACKEND_H
