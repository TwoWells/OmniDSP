/**
 * @file convolution.cpp
 * @brief Implementation file for 1D convolution and correlation functions.
 */

#include "OmniDSP/convolution.h"  // Public API declarations

#include <algorithm>  // For std::reverse
#include <stdexcept>  // For std::invalid_argument
#include <string>     // For exception messages
#include <vector>

#include "backend/backend.h"  // Backend dispatcher function declarations

namespace OmniDSP {

//--------------------------------------------------------------------------
// convolve1d
//--------------------------------------------------------------------------
template <typename T>
std::vector<T> convolve1d(
    const std::vector<T> &signal, const std::vector<T> &kernel,
    ConvMode mode /* = ConvMode::Valid */)  // Added mode parameter
{
  // --- Input Validation ---
  if (signal.empty()) {
    throw std::invalid_argument("convolve1d: Input signal cannot be empty.");
  }
  if (kernel.empty()) {
    throw std::invalid_argument("convolve1d: Kernel cannot be empty.");
  }

  // Additional validation specific to modes might be done here or in the
  // backend For example, 'Valid' mode typically requires signal_len >=
  // kernel_len
  if (mode == ConvMode::Valid && kernel.size() > signal.size()) {
    throw std::invalid_argument(
        "convolve1d: For 'Valid' mode, kernel length (" +
        std::to_string(kernel.size()) +
        ") cannot be greater than signal length (" +
        std::to_string(signal.size()) + ").");
  }

  // --- Dispatch to Backend ---
  // The backend implementation handles the actual computation based on the
  // mode.
  return Backend::convolve1d_impl<T>(signal.data(), signal.size(),
                                     kernel.data(), kernel.size(),
                                     mode);  // Pass mode to backend
}

//--------------------------------------------------------------------------
// correlate1d
//--------------------------------------------------------------------------
template <typename T>
std::vector<T> correlate1d(
    const std::vector<T> &signal, const std::vector<T> &kernel,
    ConvMode mode /* = ConvMode::Valid */)  // Added mode parameter
{
  // --- Input Validation ---
  if (signal.empty()) {
    throw std::invalid_argument("correlate1d: Input signal cannot be empty.");
  }
  if (kernel.empty()) {
    throw std::invalid_argument("correlate1d: Kernel cannot be empty.");
  }

  // Validation for 'Valid' mode
  if (mode == ConvMode::Valid && kernel.size() > signal.size()) {
    throw std::invalid_argument(
        "correlate1d: For 'Valid' mode, kernel length (" +
        std::to_string(kernel.size()) +
        ") cannot be greater than signal length (" +
        std::to_string(signal.size()) + ").");
  }

  // --- Implementation via Convolution ---
  // Correlation is equivalent to convolution with the time-reversed kernel.
  // Create a reversed copy of the kernel.
  std::vector<T> reversed_kernel = kernel;
  std::reverse(reversed_kernel.begin(), reversed_kernel.end());

  // Dispatch to the convolve1d backend implementation with the reversed kernel
  // and mode. (Assuming correlate1d_impl is not strictly necessary if this
  // approach is used)
  return Backend::convolve1d_impl<T>(signal.data(), signal.size(),
                                     reversed_kernel.data(),
                                     reversed_kernel.size(),
                                     mode);  // Pass mode to backend

  // Alternatively, if Backend::correlate1d_impl exists and is preferred:
  // return Backend::correlate1d_impl<T>(signal.data(), signal.size(),
  //                                    kernel.data(), kernel.size(),
  //                                    mode);
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
// Instantiate for float and double for all public functions.

template std::vector<float> convolve1d<float>(const std::vector<float> &signal,
                                              const std::vector<float> &kernel,
                                              ConvMode mode);  // Added mode
template std::vector<double> convolve1d<double>(
    const std::vector<double> &signal, const std::vector<double> &kernel,
    ConvMode mode);  // Added mode

template std::vector<float> correlate1d<float>(const std::vector<float> &signal,
                                               const std::vector<float> &kernel,
                                               ConvMode mode);  // Added mode
template std::vector<double> correlate1d<double>(
    const std::vector<double> &signal, const std::vector<double> &kernel,
    ConvMode mode);  // Added mode

}  // namespace OmniDSP
