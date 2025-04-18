/**
 * @file convolution.cpp
 * @brief oneMKL/IPP backend implementation for convolution and correlation.
 */
#include <ipp.h>  // Main IPP header for ipps functions

#include <algorithm>    // For std::min, std::max
#include <cmath>        // For std::floor
#include <memory>       // For std::unique_ptr
#include <stdexcept>    // For std::runtime_error, std::invalid_argument
#include <string>       // For exception messages
#include <type_traits>  // For std::is_same_v
#include <vector>

#include "../backend_impl.h"  // For ConvMode enum

namespace OmniDSP {
namespace Backend {
namespace oneMKL {

// --- Helper to check IPP status ---
namespace {  // Anonymous namespace for internal helpers
void check_ipp_status(IppStatus status, const std::string& func_name,
                      const std::string& step_name = "") {
  if (status != ippStsNoErr) {
    throw std::runtime_error("IPP Error in " + func_name +
                             (step_name.empty() ? "" : " (" + step_name + ")") +
                             ": " + std::string(ippGetStatusString(status)));
  }
}

// Helper to determine output size based on mode
size_t calculate_output_size(size_t signal_len, size_t kernel_len,
                             ConvMode mode) {
  if (signal_len == 0 || kernel_len == 0) return 0;
  switch (mode) {
    case ConvMode::Full:
      return signal_len + kernel_len - 1;
    case ConvMode::Same:
      return signal_len;
    case ConvMode::Valid:
      if (kernel_len > signal_len) return 0;
      return signal_len - kernel_len + 1;
    default:
      throw std::invalid_argument("Unknown ConvMode in calculate_output_size");
  }
}

// Helper to determine output size and lowLag for correlation based on mode
struct CorrParams {
  size_t dstLen = 0;
  int lowLag = 0;
};

CorrParams calculate_corr_params(size_t signal_len, size_t kernel_len,
                                 ConvMode mode) {
  if (signal_len == 0 || kernel_len == 0) return {0, 0};
  CorrParams params;
  long long N = static_cast<long long>(signal_len);
  long long M = static_cast<long long>(kernel_len);

  switch (mode) {
    case ConvMode::Full:
      params.lowLag = -static_cast<int>(M - 1);
      params.dstLen = static_cast<size_t>(N + M - 1);
      break;
    case ConvMode::Same:
      params.lowLag = -static_cast<int>((M - 1) / 2);
      params.dstLen = static_cast<size_t>(N);
      break;
    case ConvMode::Valid:
      if (M > N) {
        params.lowLag = 0;
        params.dstLen = 0;
      } else {
        params.lowLag = 0;
        params.dstLen = static_cast<size_t>(N - M + 1);
      }
      break;
    default:
      throw std::invalid_argument("Unknown ConvMode in calculate_corr_params");
  }
  return params;
}

// Custom deleter for ippsMalloc'd buffers
struct IPPBufferDeleter {
  void operator()(Ipp8u* ptr) const {
    if (ptr) {
      ippsFree(ptr);
    }
  }
};
using IPPBufferPtr = std::unique_ptr<Ipp8u, IPPBufferDeleter>;

}  // anonymous namespace

// --- Convolution Implementation (Templated) ---
template <typename T>
std::vector<T> convolve1d_onemkl_impl(const T* signal, size_t signal_len,
                                      const T* kernel, size_t kernel_len,
                                      ConvMode mode) {
  if (signal_len == 0 || kernel_len == 0) return {};

  size_t full_len = signal_len + kernel_len - 1;
  std::vector<T> full_result(full_len);
  IppStatus status = ippStsNoErr;
  int bufferSize = 0;
  IPPBufferPtr pBuffer = nullptr;
  IppEnum algType = ippAlgAuto;  // Use auto algorithm selection

  if constexpr (std::is_same_v<T, float>) {
    status = ippsConvolveGetBufferSize(static_cast<int>(signal_len),
                                       static_cast<int>(kernel_len), ipp32f,
                                       algType, &bufferSize);
    check_ipp_status(status, "ippsConvolveGetBufferSize (float)");
    if (bufferSize > 0) pBuffer.reset(ippsMalloc_8u(bufferSize));
    if (bufferSize > 0 && !pBuffer)
      throw std::runtime_error("Failed to allocate IPP buffer.");
    status = ippsConvolve_32f(signal, static_cast<int>(signal_len), kernel,
                              static_cast<int>(kernel_len), full_result.data(),
                              algType, pBuffer.get());
    check_ipp_status(status, "ippsConvolve_32f");
  } else if constexpr (std::is_same_v<T, double>) {
    status = ippsConvolveGetBufferSize(static_cast<int>(signal_len),
                                       static_cast<int>(kernel_len), ipp64f,
                                       algType, &bufferSize);
    check_ipp_status(status, "ippsConvolveGetBufferSize (double)");
    if (bufferSize > 0) pBuffer.reset(ippsMalloc_8u(bufferSize));
    if (bufferSize > 0 && !pBuffer)
      throw std::runtime_error("Failed to allocate IPP buffer.");
    status = ippsConvolve_64f(signal, static_cast<int>(signal_len), kernel,
                              static_cast<int>(kernel_len), full_result.data(),
                              algType, pBuffer.get());
    check_ipp_status(status, "ippsConvolve_64f");
  } else {
    throw std::runtime_error("Unsupported data type for oneMKL convolution.");
  }

  // --- Extract desired portion based on mode ---
  size_t output_size = calculate_output_size(signal_len, kernel_len, mode);
  if (output_size == 0 && mode == ConvMode::Valid) return {};

  std::vector<T> result(output_size);
  const T* full_result_ptr = full_result.data();
  T* result_ptr = result.data();

  switch (mode) {
    case ConvMode::Full:
      result = std::move(full_result);
      break;
    case ConvMode::Same: {
      size_t start_index = (kernel_len - 1) / 2;
      if (start_index + output_size <= full_len) {
        const T* start_ptr = full_result_ptr + start_index;
        if constexpr (std::is_same_v<T, float>)
          ippsCopy_32f(reinterpret_cast<const Ipp32f*>(start_ptr),
                       reinterpret_cast<Ipp32f*>(result_ptr),
                       static_cast<int>(output_size));
        else
          ippsCopy_64f(reinterpret_cast<const Ipp64f*>(start_ptr),
                       reinterpret_cast<Ipp64f*>(result_ptr),
                       static_cast<int>(output_size));
      } else {
        throw std::runtime_error(
            "oneMKL convolve1d 'Same' mode calculation error: bounds "
            "mismatch.");
      }
      break;
    }
    case ConvMode::Valid: {
      size_t start_index = kernel_len - 1;
      if (kernel_len > signal_len) return {};
      if (start_index + output_size <= full_len) {
        const T* start_ptr = full_result_ptr + start_index;
        if constexpr (std::is_same_v<T, float>)
          ippsCopy_32f(reinterpret_cast<const Ipp32f*>(start_ptr),
                       reinterpret_cast<Ipp32f*>(result_ptr),
                       static_cast<int>(output_size));
        else
          ippsCopy_64f(reinterpret_cast<const Ipp64f*>(start_ptr),
                       reinterpret_cast<Ipp64f*>(result_ptr),
                       static_cast<int>(output_size));
      } else {
        throw std::runtime_error(
            "oneMKL convolve1d 'Valid' mode calculation error: bounds "
            "mismatch.");
      }
      break;
    }
  }
  return result;
}

// --- Correlation Implementation (Templated) ---
template <typename T>
std::vector<T> correlate1d_onemkl_impl(const T* signal, size_t signal_len,
                                       const T* kernel, size_t kernel_len,
                                       ConvMode mode) {
  if (signal_len == 0 || kernel_len == 0) return {};

  CorrParams params = calculate_corr_params(signal_len, kernel_len, mode);
  if (params.dstLen == 0) return {};

  std::vector<T> result(params.dstLen);
  IppStatus status = ippStsNoErr;
  int bufferSize = 0;
  IPPBufferPtr pBuffer = nullptr;
  IppEnum algType = ippAlgAuto;  // Use auto algorithm selection

  if constexpr (std::is_same_v<T, float>) {
    // Correct call to ippsCrossCorrNormGetBufferSize for float
    status = ippsCrossCorrNormGetBufferSize(
        static_cast<int>(signal_len), static_cast<int>(kernel_len),
        static_cast<int>(params.dstLen), params.lowLag,  // Added lowLag
        ipp32f,                 // Correct dataType argument
        algType, &bufferSize);  // Correct algType argument
    check_ipp_status(status, "ippsCrossCorrNormGetBufferSize (float)");

    if (bufferSize > 0) pBuffer.reset(ippsMalloc_8u(bufferSize));
    if (bufferSize > 0 && !pBuffer)
      throw std::runtime_error(
          "Failed to allocate IPP buffer for CrossCorrNorm.");

    // Call ippsCrossCorrNorm_32f
    status = ippsCrossCorrNorm_32f(signal, static_cast<int>(signal_len), kernel,
                                   static_cast<int>(kernel_len), result.data(),
                                   static_cast<int>(params.dstLen),
                                   params.lowLag, algType, pBuffer.get());
    check_ipp_status(status, "ippsCrossCorrNorm_32f");

  } else if constexpr (std::is_same_v<T, double>) {
    // Correct call to ippsCrossCorrNormGetBufferSize for double
    status = ippsCrossCorrNormGetBufferSize(
        static_cast<int>(signal_len), static_cast<int>(kernel_len),
        static_cast<int>(params.dstLen), params.lowLag,  // Added lowLag
        ipp64f,                 // Correct dataType argument
        algType, &bufferSize);  // Correct algType argument
    check_ipp_status(status, "ippsCrossCorrNormGetBufferSize (double)");

    if (bufferSize > 0) pBuffer.reset(ippsMalloc_8u(bufferSize));
    if (bufferSize > 0 && !pBuffer)
      throw std::runtime_error(
          "Failed to allocate IPP buffer for CrossCorrNorm.");

    // Call ippsCrossCorrNorm_64f
    status = ippsCrossCorrNorm_64f(signal, static_cast<int>(signal_len), kernel,
                                   static_cast<int>(kernel_len), result.data(),
                                   static_cast<int>(params.dstLen),
                                   params.lowLag, algType, pBuffer.get());
    check_ipp_status(status, "ippsCrossCorrNorm_64f");
  } else {
    throw std::runtime_error("Unsupported data type for oneMKL correlation.");
  }

  return result;
}

// --- Explicit Template Instantiations ---
template std::vector<float> convolve1d_onemkl_impl<float>(const float*, size_t,
                                                          const float*, size_t,
                                                          ConvMode);
template std::vector<double> convolve1d_onemkl_impl<double>(const double*,
                                                            size_t,
                                                            const double*,
                                                            size_t, ConvMode);
template std::vector<float> correlate1d_onemkl_impl<float>(const float*, size_t,
                                                           const float*, size_t,
                                                           ConvMode);
template std::vector<double> correlate1d_onemkl_impl<double>(const double*,
                                                             size_t,
                                                             const double*,
                                                             size_t, ConvMode);

}  // namespace oneMKL
}  // namespace Backend
}  // namespace OmniDSP
