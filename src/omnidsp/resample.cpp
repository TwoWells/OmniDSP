/**
 * @file resample.cpp
 * @brief Implementation file for resampling functions.
 * Updated to call the backend implementation using the correct signature.
 */

#include "OmniDSP/resample.h"  // Public API declaration

#include <cmath>  // For std::floor, std::ceil (potentially needed for validation)
#include <stdexcept>  // For std::invalid_argument, std::runtime_error
#include <string>     // For exception messages
#include <vector>

// #include "OmniDSP/omnidsp.h" // No longer needed for OMNIDSP_STATUS
#include "backend/backend.h"  // Use renamed backend header for dispatcher declarations

namespace OmniDSP {

/**
 * @brief Implementation for filter_and_downsample.
 * Performs input validation and calls the backend dispatcher.
 */
template <typename T>
std::vector<T> filter_and_downsample(const std::vector<T> &signal,
                                     const std::vector<T> &kernel, int factor) {
  // --- Input Validation ---
  if (factor <= 0) {
    throw std::invalid_argument(
        "filter_and_downsample: Downsampling factor must be positive.");
  }
  if (signal.empty()) {
    // Return empty vector if input is empty (consistent with some libraries
    // like SciPy)
    return {};
  }
  if (kernel.empty()) {
    throw std::invalid_argument(
        "filter_and_downsample: Kernel cannot be empty.");
  }
  // Backend implementations might handle this, but basic check here is good.
  // 'Valid' mode is often implicit in FIR+downsample operations,
  // potentially requiring kernel <= signal depending on backend.
  // Example check (can be refined based on backend needs):
  // if (kernel.size() > signal.size()) {
  //   throw std::invalid_argument("filter_and_downsample: Kernel length (" +
  //                               std::to_string(kernel.size()) +
  //                               ") cannot be greater than signal length (" +
  //                               std::to_string(signal.size()) +
  //                               ") for implicit 'valid' mode processing.");
  // }

  // --- Call Backend Implementation ---
  // The backend implementation now handles calculating the output size
  // and returning the result vector directly.
  // Any errors should be propagated via exceptions.
  try {
    // Corrected Call: Use the 3-argument signature declared in backend.h
    return Backend::filter_and_downsample_impl<T>(signal, kernel, factor);
  } catch (const std::exception &e) {
    // Re-throw backend errors with more context if desired
    throw std::runtime_error(
        "Error during backend filter_and_downsample execution: " +
        std::string(e.what()));
  }

  // --- Old Code Removed ---
  // Removed manual output size calculation (e.g., valid_conv_len, output_len)
  // Removed creation of output vector before calling backend
  // Removed call to the old 7-argument backend function
  // Removed status code checking
  // --- End Old Code Removed ---
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
// These declarations match the ones in resample.h
template std::vector<float> filter_and_downsample<float>(
    const std::vector<float> &signal, const std::vector<float> &kernel,
    int factor);

#if !defined(USE_ONEMKL)
template std::vector<double> filter_and_downsample<double>(
    const std::vector<double> &signal, const std::vector<double> &kernel,
    int factor);
#endif

}  // namespace OmniDSP
