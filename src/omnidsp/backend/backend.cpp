/**
 * @file backend.cpp
 * @brief Implements the backend dispatch functions for window, convolution, and
 * resampling. Uses preprocessor flags (USE_ONEMKL, USE_ACCELERATE) set by CMake
 * to select the appropriate backend implementation at compile time by
 * conditionally defining the dispatch functions.
 * FFT Plan creation is handled directly within the Plan classes defined in
 * backend-specific fft.cpp files.
 */

#include "backend.h"  // Use the renamed header

#include <complex>
#include <memory>  // For unique_ptr (potentially used internally by backends)
#include <stdexcept>    // For runtime_error in stubs or errors
#include <type_traits>  // For std::is_same_v
#include <vector>

// --- Forward Declarations for Concrete Backend Implementations ---
// Declare the functions from the specific backend namespaces that will be
// called by the dispatcher functions below.

#if defined(USE_ONEMKL)
namespace OmniDSP {
namespace Backend {
namespace oneMKL {
// Forward declare functions defined in onemkl/*.cpp
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
std::vector<T> filter_and_downsample_impl(const std::vector<T>& signal,
                                          const std::vector<T>& kernel,
                                          int factor);
}  // namespace oneMKL
}  // namespace Backend
}  // namespace OmniDSP
#elif defined(USE_ACCELERATE)
namespace OmniDSP {
namespace Backend {
namespace Accelerate {
// Forward declare functions defined in accelerate/*.cpp
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
// via convolve1d in the public API wrapper (convolution.cpp)
std::vector<float> filter_and_downsample_impl(const std::vector<float>& signal,
                                              const std::vector<float>& kernel,
                                              int factor);
std::vector<double> filter_and_downsample_impl(
    const std::vector<double>& signal, const std::vector<double>& kernel,
    int factor);
}  // namespace Accelerate
}  // namespace Backend
}  // namespace OmniDSP
#else  // Stub backend
namespace OmniDSP {
namespace Backend {
namespace Stub {
// Forward declare functions defined in stub/*.cpp
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
std::vector<T> filter_and_downsample_impl(const std::vector<T>& signal,
                                          const std::vector<T>& kernel,
                                          int factor);
}  // namespace Stub
}  // namespace Backend
}  // namespace OmniDSP
#endif

namespace OmniDSP {
namespace Backend {

// --- Conditionally Defined Dispatcher Function Implementations ---
// Only one version of each function definition will exist after preprocessing.
// These implement the functions declared in backend.h

#if defined(USE_ONEMKL)
// --- oneMKL Definitions ---
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
  // Note: MKL backend might have a dedicated correlate function,
  // or the public API wrapper might reverse kernel and call convolve1d_impl.
  // Assuming MKL backend provides correlate1d_onemkl_impl for now.
  return oneMKL::correlate1d_onemkl_impl<T>(s, sl, k, kl, m);
}
template <typename T>
std::vector<T> filter_and_downsample_impl(const std::vector<T>& signal,
                                          const std::vector<T>& kernel,
                                          int factor) {
  return oneMKL::filter_and_downsample_impl<T>(signal, kernel, factor);
}

#elif defined(USE_ACCELERATE)
// --- Accelerate Definitions ---
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
  // Accelerate backend implementation might be split by type
  if constexpr (std::is_same_v<T, float>) {
    // Need to cast pointers if the Accelerate function takes specific types
    return Accelerate::convolve1d_accelerate_impl_float(
        static_cast<const float*>(s), sl, static_cast<const float*>(k), kl, m);
  } else if constexpr (std::is_same_v<T, double>) {
    return Accelerate::convolve1d_accelerate_impl_double(
        static_cast<const double*>(s), sl, static_cast<const double*>(k), kl,
        m);
  } else {
    // Handle unsupported type if necessary, though templates usually prevent
    // this
    throw std::runtime_error("Unsupported type for Accelerate convolve1d_impl");
  }
}
template <typename T>
std::vector<T> correlate1d_impl(const T* s, size_t sl, const T* k, size_t kl,
                                ConvMode m) {
  // Accelerate correlation currently relies on public API reversing kernel and
  // calling convolve1d_impl. This function might not be strictly needed here
  // if the public API wrapper handles it. If a direct Accelerate correlation
  // exists, it would be called here.
  // For now, throw or delegate back to convolution with reversed kernel logic
  // (which should ideally live in the public API wrapper `convolution.cpp`).
  throw std::runtime_error(
      "Backend::correlate1d_impl dispatch not implemented directly for "
      "Accelerate backend in backend.cpp. Should be handled by public API "
      "wrapper.");
}
template <typename T>
std::vector<T> filter_and_downsample_impl(const std::vector<T>& signal,
                                          const std::vector<T>& kernel,
                                          int factor) {
  // Accelerate backend implementation is likely split by type
  if constexpr (std::is_same_v<T, float>) {
    // Need to ensure the function signature matches exactly
    return Accelerate::filter_and_downsample_impl(
        reinterpret_cast<const std::vector<float>&>(signal),
        reinterpret_cast<const std::vector<float>&>(kernel), factor);
  } else if constexpr (std::is_same_v<T, double>) {
    return Accelerate::filter_and_downsample_impl(
        reinterpret_cast<const std::vector<double>&>(signal),
        reinterpret_cast<const std::vector<double>&>(kernel), factor);
  } else {
    throw std::runtime_error(
        "Unsupported type for Accelerate filter_and_downsample_impl");
  }
}

#else
// --- Stub Definitions ---
template <typename T>
void hann_window_impl(T* output, size_t length) {
  Stub::hann_window_impl(output, length);  // Will throw
}
template <typename T>
void hamming_window_impl(T* output, size_t length) {
  Stub::hamming_window_impl(output, length);  // Will throw
}
template <typename T>
void kaiser_window_impl(T* output, size_t length, T beta) {
  Stub::kaiser_window_impl(output, length, beta);  // Will throw
}
template <typename T>
void flattop_window_impl(T* output, size_t length) {
  Stub::flattop_window_impl(output, length);  // Will throw
}
template <typename T>
std::vector<T> convolve1d_impl(const T* s, size_t sl, const T* k, size_t kl,
                               ConvMode m) {
  return Stub::convolve1d_stub_impl<T>(s, sl, k, kl, m);  // Will throw
}
template <typename T>
std::vector<T> correlate1d_impl(const T* s, size_t sl, const T* k, size_t kl,
                                ConvMode m) {
  return Stub::correlate1d_stub_impl<T>(s, sl, k, kl, m);  // Will throw
}
template <typename T>
std::vector<T> filter_and_downsample_impl(const std::vector<T>& signal,
                                          const std::vector<T>& kernel,
                                          int factor) {
  return Stub::filter_and_downsample_impl<T>(signal, kernel,
                                             factor);  // Will throw
}

#endif

// --- Explicit Template Instantiations for Dispatcher Functions ---
// These instantiate the specific versions of the dispatcher functions
// selected by the preprocessor flags above.

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
template std::vector<float> filter_and_downsample_impl<float>(
    const std::vector<float>& signal, const std::vector<float>& kernel,
    int factor);
#if !defined(USE_ONEMKL)
template std::vector<double> filter_and_downsample_impl<double>(
    const std::vector<double>& signal, const std::vector<double>& kernel,
    int factor);
#endif

}  // namespace Backend
}  // namespace OmniDSP
