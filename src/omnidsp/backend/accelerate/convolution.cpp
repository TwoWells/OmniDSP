/**
 * @file convolution.cpp
 * @brief Accelerate/vDSP backend implementation for convolution.
 */
#include <Accelerate/Accelerate.h>  // Main Accelerate framework header

#include <algorithm>  // For std::reverse, std::min, std::max
#include <cmath>      // For std::floor
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "../backend_impl.h"  // For ConvMode enum

namespace OmniDSP {
namespace Backend {
namespace Accelerate {

// --- Helper to determine output size based on mode ---
size_t calculate_output_size(size_t signal_len, size_t kernel_len,
                             ConvMode mode) {
  if (signal_len == 0 || kernel_len == 0) return 0;
  switch (mode) {
    case ConvMode::Full:
      return signal_len + kernel_len - 1;
    case ConvMode::Same:
      return signal_len;
    case ConvMode::Valid:
      if (kernel_len > signal_len)
        return 0;  // Or throw? Let's return 0 for valid.
      return signal_len - kernel_len + 1;
    default:  // Should not happen
      throw std::invalid_argument("Unknown ConvMode in calculate_output_size");
  }
}

// --- Convolution Implementation ---

std::vector<float> convolve1d_accelerate_impl_float(const float* signal,
                                                    size_t signal_len,
                                                    const float* kernel,
                                                    size_t kernel_len,
                                                    ConvMode mode) {
  if (signal_len == 0 || kernel_len == 0) return {};

  // vDSP_conv calculates the 'Full' convolution result.
  size_t full_len = signal_len + kernel_len - 1;
  std::vector<float> full_result(full_len);

  // vDSP_conv requires the kernel to be reversed *by the caller*.
  std::vector<float> reversed_kernel(kernel, kernel + kernel_len);
  vDSP_vrvrs(reversed_kernel.data(), 1, kernel_len);  // Reverse kernel in-place

  // Perform full convolution using vDSP
  // vDSP_conv(Signal, SignalStride, ReversedKernel, KernelStride, Output,
  // OutputStride, OutputLength, KernelLength)
  vDSP_conv(signal, 1, reversed_kernel.data(), 1, full_result.data(), 1,
            full_len, kernel_len);

  // --- Extract desired portion based on mode ---
  size_t output_size = calculate_output_size(signal_len, kernel_len, mode);
  if (output_size == 0 && mode == ConvMode::Valid)
    return {};  // Valid mode resulted in 0 size

  std::vector<float> result(output_size);

  switch (mode) {
    case ConvMode::Full:
      // Already have the full result
      result = std::move(full_result);  // Move is efficient
      break;
    case ConvMode::Same: {
      // Extract the central part of size signal_len
      // Start index calculation (integer division)
      size_t start_index = (kernel_len - 1) / 2;
      // Check bounds (should be okay if full_len >= signal_len)
      if (start_index + output_size <= full_len) {
        std::copy(full_result.begin() + start_index,
                  full_result.begin() + start_index + output_size,
                  result.begin());
      } else {
        // This case indicates an issue with size calculations, should not
        // happen with standard modes
        throw std::runtime_error(
            "Accelerate convolve1d 'Same' mode calculation error: bounds "
            "mismatch.");
      }
      break;
    }
    case ConvMode::Valid: {
      // Extract the valid part of size signal_len - kernel_len + 1
      // Start index calculation
      size_t start_index = kernel_len - 1;
      // Check bounds (should be okay if full_len >= output_size and N >= M)
      if (kernel_len > signal_len) {  // Should have been caught by
                                      // calculate_output_size returning 0
        return {};  // Return empty vector if kernel > signal for valid mode
      }
      if (start_index + output_size <= full_len) {
        std::copy(full_result.begin() + start_index,
                  full_result.begin() + start_index + output_size,
                  result.begin());
      } else {
        throw std::runtime_error(
            "Accelerate convolve1d 'Valid' mode calculation error: bounds "
            "mismatch.");
      }
      break;
    }
  }
  return result;
}

std::vector<double> convolve1d_accelerate_impl_double(const double* signal,
                                                      size_t signal_len,
                                                      const double* kernel,
                                                      size_t kernel_len,
                                                      ConvMode mode) {
  if (signal_len == 0 || kernel_len == 0) return {};

  size_t full_len = signal_len + kernel_len - 1;
  std::vector<double> full_result(full_len);
  std::vector<double> reversed_kernel(kernel, kernel + kernel_len);
  vDSP_vrvrsD(reversed_kernel.data(), 1,
              kernel_len);  // Reverse kernel (Double)

  // Perform full convolution using vDSP (Double)
  vDSP_convD(signal, 1, reversed_kernel.data(), 1, full_result.data(), 1,
             full_len, kernel_len);

  // --- Extract desired portion based on mode ---
  size_t output_size = calculate_output_size(signal_len, kernel_len, mode);
  if (output_size == 0 && mode == ConvMode::Valid) return {};

  std::vector<double> result(output_size);

  switch (mode) {
    case ConvMode::Full:
      result = std::move(full_result);
      break;
    case ConvMode::Same: {
      size_t start_index = (kernel_len - 1) / 2;
      if (start_index + output_size <= full_len) {
        std::copy(full_result.begin() + start_index,
                  full_result.begin() + start_index + output_size,
                  result.begin());
      } else {
        throw std::runtime_error(
            "Accelerate convolve1d 'Same' mode calculation error: bounds "
            "mismatch.");
      }
      break;
    }
    case ConvMode::Valid: {
      size_t start_index = kernel_len - 1;
      if (kernel_len > signal_len) return {};
      if (start_index + output_size <= full_len) {
        std::copy(full_result.begin() + start_index,
                  full_result.begin() + start_index + output_size,
                  result.begin());
      } else {
        throw std::runtime_error(
            "Accelerate convolve1d 'Valid' mode calculation error: bounds "
            "mismatch.");
      }
      break;
    }
  }
  return result;
}

// --- Correlation Implementation ---
// Correlation is implemented by reversing the *kernel* in the public API call
// and then calling the convolve1d_impl here. So no separate correlate1d_impl
// needed here.

// --- Explicit Template Instantiations ---
// We instantiate the dispatcher functions in backend.cpp, not these specific
// impls.

}  // namespace Accelerate
}  // namespace Backend
}  // namespace OmniDSP
