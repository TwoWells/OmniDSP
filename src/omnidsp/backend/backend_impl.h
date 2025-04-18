/**
 * @file backend_impl.h
 * @brief Defines interfaces and factory functions for backend implementations.
 */
#ifndef OMNIDSP_BACKEND_IMPL_H
#define OMNIDSP_BACKEND_IMPL_H

#include <complex>
#include <cstddef>  // For size_t
#include <memory>   // For std::unique_ptr
#include <vector>

#include "OmniDSP/convolution.h"  // Re-added: Include convolution definitions (for ConvMode)
#include "OmniDSP/omnidsp.h"  // Include base definitions (like OMNIDSP_API, OMNIDSP_STATUS, FFTNorm)

namespace OmniDSP {

// --- Forward declarations for PIMPL implementation interfaces ---
template <typename T>
class FFTPlanImpl;
template <typename T>
class RFFTPlanImpl;
template <typename T>
class WindowImpl;
// template <typename T> class ConvolutionImpl; // Keep removed for now, can be
// added if needed

// --- PIMPL Interface Definitions ---

template <typename T>
class FFTPlanImpl {
 public:
  virtual ~FFTPlanImpl() = default;
  virtual void fft(const std::complex<T>* input, std::complex<T>* output,
                   FFTNorm norm) = 0;
  virtual void ifft(const std::complex<T>* input, std::complex<T>* output,
                    FFTNorm norm) = 0;
};

template <typename T>
class RFFTPlanImpl {
 public:
  virtual ~RFFTPlanImpl() = default;
  virtual void rfft(const T* input, std::complex<T>* output, FFTNorm norm) = 0;
  virtual void irfft(const std::complex<T>* input, T* output, FFTNorm norm) = 0;
};

template <typename T>
class WindowImpl {
 public:
  virtual ~WindowImpl() = default;
  virtual void hann(T* output, size_t length) = 0;
  virtual void hamming(T* output, size_t length) = 0;
  virtual void kaiser(T* output, size_t length, T beta) = 0;
  virtual void flattop(T* output, size_t length) = 0;
  // Add others...
};

// --- Backend Namespace ---
// Contains factory functions to create backend implementations or direct
// dispatch functions.
namespace Backend {

// --- Backend Factory Functions ---
template <typename T>
std::unique_ptr<FFTPlanImpl<T>> createFFTPlan(size_t size);

template <typename T>
std::unique_ptr<RFFTPlanImpl<T>> createRFFTPlan(size_t size);

// --- Backend Window Function Implementations ---
template <typename T>
void hann_window_impl(T* output, size_t length);
template <typename T>
void hamming_window_impl(T* output, size_t length);
template <typename T>
void kaiser_window_impl(T* output, size_t length, T beta);
template <typename T>
void flattop_window_impl(T* output, size_t length);

// --- Backend Convolution Function Implementations ---
// Direct dispatch functions implemented in backend/backend.cpp

/**
 * @brief Dispatches 1D convolution to the selected backend.
 * @param signal Pointer to the input signal data.
 * @param signal_len Length of the input signal.
 * @param kernel Pointer to the kernel data.
 * @param kernel_len Length of the kernel.
 * @param mode Convolution mode (Full, Same, Valid).
 * @return std::vector<T> The convolution result.
 * @throws std::runtime_error If backend execution fails.
 * @throws std::invalid_argument If inputs/mode are invalid.
 */
template <typename T>  // <<< CORRECTED: Re-added ConvMode parameter
std::vector<T> convolve1d_impl(const T* signal, size_t signal_len,
                               const T* kernel, size_t kernel_len,
                               OmniDSP::ConvMode mode);  // Use qualified name

/**
 * @brief Dispatches 1D correlation to the selected backend.
 * @param signal Pointer to the input signal data.
 * @param signal_len Length of the input signal.
 * @param kernel Pointer to the kernel data.
 * @param kernel_len Length of the kernel.
 * @param mode Correlation mode (Full, Same, Valid).
 * @return std::vector<T> The correlation result.
 * @throws std::runtime_error If backend execution fails.
 * @throws std::invalid_argument If inputs/mode are invalid.
 */
template <typename T>  // <<< CORRECTED: Re-added ConvMode parameter
std::vector<T> correlate1d_impl(const T* signal, size_t signal_len,
                                const T* kernel, size_t kernel_len,
                                OmniDSP::ConvMode mode);  // Use qualified name

// --- Backend Resample Function Implementations ---
/**
 * @brief Dispatches filter+downsample operation to the selected backend.
 */
template <typename T>
OMNIDSP_STATUS filter_and_downsample_impl(const T* input, size_t input_len,
                                          T* output, size_t output_len,
                                          const T* kernel, size_t kernel_len,
                                          int factor);

}  // namespace Backend
}  // namespace OmniDSP

#endif  // OMNIDSP_BACKEND_IMPL_H
