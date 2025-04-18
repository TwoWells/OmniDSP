/**
 * @file backend.cpp
 * @brief Implements the backend dispatch functions.
 * Uses preprocessor flags (USE_ONEMKL, USE_ACCELERATE) set by CMake
 * to select the appropriate backend implementation at compile time by
 * conditionally defining the dispatch functions.
 */

#include <complex>
#include <memory>       // For unique_ptr
#include <stdexcept>    // For runtime_error in stubs or errors
#include <type_traits>  // For std::is_same_v
#include <vector>

#include "backend_impl.h"  // Function declarations being implemented

// --- Forward Declarations for Concrete Backend Implementations ---
// Declare the functions from the specific backend namespaces that will be
// called.

#if defined(USE_ONEMKL)
namespace OmniDSP {
namespace Backend {
namespace oneMKL {
template <typename T>
std::unique_ptr<FFTPlanImpl<T>> createFFTPlan_onemkl_impl(size_t size);
template <typename T>
std::unique_ptr<RFFTPlanImpl<T>> createRFFTPlan_onemkl_impl(size_t size);
template <typename T>
void hann_window_impl(T* output, size_t length);
template <typename T>
void hamming_window_impl(T* output, size_t length);
template <typename T>
void kaiser_window_impl(T* output, size_t length, T beta);
template <typename T>
void flattop_window_impl(T* output, size_t length);
template <typename T>
std::vector<T> convolve1d_onemkl_impl(const T* signal, size_t signal_len,
                                      const T* kernel, size_t kernel_len,
                                      ConvMode mode);
template <typename T>
std::vector<T> correlate1d_onemkl_impl(const T* signal, size_t signal_len,
                                       const T* kernel, size_t kernel_len,
                                       ConvMode mode);
template <typename T>
OMNIDSP_STATUS filter_and_downsample_impl(const T* input, size_t input_len,
                                          T* output, size_t output_len,
                                          const T* kernel, size_t kernel_len,
                                          int factor);
}  // namespace oneMKL
}  // namespace Backend
}  // namespace OmniDSP
#elif defined(USE_ACCELERATE)
namespace OmniDSP {
namespace Backend {
namespace Accelerate {
template <typename T>
std::unique_ptr<FFTPlanImpl<T>> createFFTPlan_accelerate_impl(size_t size);
template <typename T>
std::unique_ptr<RFFTPlanImpl<T>> createRFFTPlan_accelerate_impl(size_t size);
template <typename T>
void hann_window_impl(T* output, size_t length);
template <typename T>
void hamming_window_impl(T* output, size_t length);
template <typename T>
void kaiser_window_impl(T* output, size_t length, T beta);
template <typename T>
void flattop_window_impl(T* output, size_t length);
std::vector<float> convolve1d_accelerate_impl_float(const float* signal,
                                                    size_t signal_len,
                                                    const float* kernel,
                                                    size_t kernel_len,
                                                    ConvMode mode);
std::vector<double> convolve1d_accelerate_impl_double(const double* signal,
                                                      size_t signal_len,
                                                      const double* kernel,
                                                      size_t kernel_len,
                                                      ConvMode mode);
// Note: No separate correlate1d impl declared for Accelerate, assumes handled
// via convolve1d
OMNIDSP_STATUS filter_and_downsample_impl_float(const float* input,
                                                size_t input_len, float* output,
                                                size_t output_len,
                                                const float* kernel,
                                                size_t kernel_len, int factor);
OMNIDSP_STATUS filter_and_downsample_impl_double(
    const double* input, size_t input_len, double* output, size_t output_len,
    const double* kernel, size_t kernel_len, int factor);
}  // namespace Accelerate
}  // namespace Backend
}  // namespace OmniDSP
#else  // Stub backend
namespace OmniDSP {
namespace Backend {
namespace Stub {
template <typename T>
std::unique_ptr<FFTPlanImpl<T>> createFFTPlan_stub_impl(size_t size);
template <typename T>
std::unique_ptr<RFFTPlanImpl<T>> createRFFTPlan_stub_impl(size_t size);
template <typename T>
void hann_window_impl(T* output, size_t length);
template <typename T>
void hamming_window_impl(T* output, size_t length);
template <typename T>
void kaiser_window_impl(T* output, size_t length, T beta);
template <typename T>
void flattop_window_impl(T* output, size_t length);
template <typename T>
std::vector<T> convolve1d_stub_impl(const T* signal, size_t signal_len,
                                    const T* kernel, size_t kernel_len,
                                    ConvMode mode);
template <typename T>
std::vector<T> correlate1d_stub_impl(const T* signal, size_t signal_len,
                                     const T* kernel, size_t kernel_len,
                                     ConvMode mode);
template <typename T>
OMNIDSP_STATUS filter_and_downsample_impl(const T* input, size_t input_len,
                                          T* output, size_t output_len,
                                          const T* kernel, size_t kernel_len,
                                          int factor);
}  // namespace Stub
}  // namespace Backend
}  // namespace OmniDSP
#endif

namespace OmniDSP {
namespace Backend {

// --- Conditionally Defined Dispatcher Function Implementations ---
// Only one version of each function definition will exist after preprocessing.

#if defined(USE_ONEMKL)
// --- oneMKL Definitions ---
template <typename T>
std::unique_ptr<FFTPlanImpl<T>> createFFTPlan(size_t size) {
  return oneMKL::createFFTPlan_onemkl_impl<T>(size);
}
template <typename T>
std::unique_ptr<RFFTPlanImpl<T>> createRFFTPlan(size_t size) {
  return oneMKL::createRFFTPlan_onemkl_impl<T>(size);
}
template <typename T>
void hann_window_impl(T* output, size_t length) {
  oneMKL::hann_window_impl(output, length);
}
template <typename T>
void hamming_window_impl(T* output, size_t length) {
  oneMKL::hamming_window_impl(output, length);
}
template <typename T>
void kaiser_window_impl(T* output, size_t length, T beta) {
  oneMKL::kaiser_window_impl(output, length, beta);
}
template <typename T>
void flattop_window_impl(T* output, size_t length) {
  oneMKL::flattop_window_impl(output, length);
}
template <typename T>
std::vector<T> convolve1d_impl(const T* s, size_t sl, const T* k, size_t kl,
                               ConvMode m) {
  return oneMKL::convolve1d_onemkl_impl<T>(s, sl, k, kl, m);
}
template <typename T>
std::vector<T> correlate1d_impl(const T* s, size_t sl, const T* k, size_t kl,
                                ConvMode m) {
  return oneMKL::correlate1d_onemkl_impl<T>(s, sl, k, kl, m);
}
template <typename T>
OMNIDSP_STATUS filter_and_downsample_impl(const T* i, size_t il, T* o,
                                          size_t ol, const T* k, size_t kl,
                                          int f) {
  return oneMKL::filter_and_downsample_impl<T>(i, il, o, ol, k, kl, f);
}

#elif defined(USE_ACCELERATE)
// --- Accelerate Definitions ---
template <typename T>
std::unique_ptr<FFTPlanImpl<T>> createFFTPlan(size_t size) {
  return Accelerate::createFFTPlan_accelerate_impl<T>(size);
}
template <typename T>
std::unique_ptr<RFFTPlanImpl<T>> createRFFTPlan(size_t size) {
  return Accelerate::createRFFTPlan_accelerate_impl<T>(size);
}
template <typename T>
void hann_window_impl(T* output, size_t length) {
  Accelerate::hann_window_impl(output, length);
}
template <typename T>
void hamming_window_impl(T* output, size_t length) {
  Accelerate::hamming_window_impl(output, length);
}
template <typename T>
void kaiser_window_impl(T* output, size_t length, T beta) {
  Accelerate::kaiser_window_impl(output, length, beta);
}
template <typename T>
void flattop_window_impl(T* output, size_t length) {
  Accelerate::flattop_window_impl(output, length);
}
template <typename T>
std::vector<T> convolve1d_impl(const T* s, size_t sl, const T* k, size_t kl,
                               ConvMode m) {
  if constexpr (std::is_same_v<T, float>)
    return Accelerate::convolve1d_accelerate_impl_float(s, sl, k, kl, m);
  else
    return Accelerate::convolve1d_accelerate_impl_double(s, sl, k, kl, m);
}
template <typename T>
std::vector<T> correlate1d_impl(const T* s, size_t sl, const T* k, size_t kl,
                                ConvMode m) {
  // Accelerate correlation currently relies on public API reversing kernel and
  // calling convolve1d_impl
  throw std::runtime_error(
      "Backend::correlate1d_impl dispatch not implemented directly for "
      "Accelerate backend.");
}
template <typename T>
OMNIDSP_STATUS filter_and_downsample_impl(const T* i, size_t il, T* o,
                                          size_t ol, const T* k, size_t kl,
                                          int f) {
  if constexpr (std::is_same_v<T, float>)
    return Accelerate::filter_and_downsample_impl_float(i, il, o, ol, k, kl, f);
  else
    return Accelerate::filter_and_downsample_impl_double(i, il, o, ol, k, kl,
                                                         f);
}

#else  // Stub Backend Definition
// --- Stub Definitions ---
template <typename T>
std::unique_ptr<FFTPlanImpl<T>> createFFTPlan(size_t size) {
  return Stub::createFFTPlan_stub_impl<T>(size);
}
template <typename T>
std::unique_ptr<RFFTPlanImpl<T>> createRFFTPlan(size_t size) {
  return Stub::createRFFTPlan_stub_impl<T>(size);
}
template <typename T>
void hann_window_impl(T* output, size_t length) {
  Stub::hann_window_impl(output, length);
}
template <typename T>
void hamming_window_impl(T* output, size_t length) {
  Stub::hamming_window_impl(output, length);
}
template <typename T>
void kaiser_window_impl(T* output, size_t length, T beta) {
  Stub::kaiser_window_impl(output, length, beta);
}
template <typename T>
void flattop_window_impl(T* output, size_t length) {
  Stub::flattop_window_impl(output, length);
}
template <typename T>
std::vector<T> convolve1d_impl(const T* s, size_t sl, const T* k, size_t kl,
                               ConvMode m) {
  return Stub::convolve1d_stub_impl<T>(s, sl, k, kl, m);
}
template <typename T>
std::vector<T> correlate1d_impl(const T* s, size_t sl, const T* k, size_t kl,
                                ConvMode m) {
  return Stub::correlate1d_stub_impl<T>(s, sl, k, kl, m);
}
template <typename T>
OMNIDSP_STATUS filter_and_downsample_impl(const T* i, size_t il, T* o,
                                          size_t ol, const T* k, size_t kl,
                                          int f) {
  return Stub::filter_and_downsample_impl<T>(i, il, o, ol, k, kl, f);
}

#endif

// --- Explicit Template Instantiations for Dispatcher Functions ---
// These instantiate the specific versions of the dispatcher functions
// selected by the preprocessor flags above.

// FFTPlan Factory
template std::unique_ptr<FFTPlanImpl<float>> createFFTPlan<float>(size_t size);
template std::unique_ptr<FFTPlanImpl<double>> createFFTPlan<double>(
    size_t size);
// RFFTPlan Factory
template std::unique_ptr<RFFTPlanImpl<float>> createRFFTPlan<float>(
    size_t size);
template std::unique_ptr<RFFTPlanImpl<double>> createRFFTPlan<double>(
    size_t size);

// Window Dispatchers
template void hann_window_impl<float>(float* output, size_t length);
template void hann_window_impl<double>(double* output, size_t length);
template void hamming_window_impl<float>(float* output, size_t length);
template void hamming_window_impl<double>(double* output, size_t length);
template void kaiser_window_impl<float>(float* output, size_t length,
                                        float beta);
template void kaiser_window_impl<double>(double* output, size_t length,
                                         double beta);
template void flattop_window_impl<float>(float* output, size_t length);
template void flattop_window_impl<double>(double* output, size_t length);

// Convolution Dispatchers
template std::vector<float> convolve1d_impl<float>(const float*, size_t,
                                                   const float*, size_t,
                                                   ConvMode);
template std::vector<double> convolve1d_impl<double>(const double*, size_t,
                                                     const double*, size_t,
                                                     ConvMode);
template std::vector<float> correlate1d_impl<float>(const float*, size_t,
                                                    const float*, size_t,
                                                    ConvMode);
template std::vector<double> correlate1d_impl<double>(const double*, size_t,
                                                      const double*, size_t,
                                                      ConvMode);

// Resample Dispatcher
template OMNIDSP_STATUS filter_and_downsample_impl<float>(const float*, size_t,
                                                          float*, size_t,
                                                          const float*, size_t,
                                                          int);
template OMNIDSP_STATUS filter_and_downsample_impl<double>(
    const double*, size_t, double*, size_t, const double*, size_t, int);

}  // namespace Backend
}  // namespace OmniDSP
