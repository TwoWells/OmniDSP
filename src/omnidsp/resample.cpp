/**
 * @file resample.cpp
 * @brief Implementation file for resampling functions.
 */

#include "OmniDSP/resample.h"  // Public API declaration

#include <cmath>  // For std::floor, std::ceil (potentially needed for size calc)
#include <stdexcept>  // For std::invalid_argument, std::runtime_error
#include <string>     // For exception messages
#include <vector>

#include "OmniDSP/omnidsp.h"       // For OMNIDSP_STATUS codes
#include "backend/backend_impl.h"  // Backend dispatcher function declarations

namespace OmniDSP {

/**
 * @brief Implementation for filter_and_downsample.
 * Calculates output size, allocates memory, and calls the backend dispatcher.
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
    // like SciPy) Alternatively, throw: throw
    // std::invalid_argument("filter_and_downsample: Input signal cannot be
    // empty.");
    return {};
  }
  if (kernel.empty()) {
    throw std::invalid_argument(
        "filter_and_downsample: Kernel cannot be empty.");
  }
  // 'Valid' mode is implicit in many FIR+downsample operations, requiring
  // kernel <= signal
  if (kernel.size() > signal.size()) {
    throw std::invalid_argument("filter_and_downsample: Kernel length (" +
                                std::to_string(kernel.size()) +
                                ") cannot be greater than signal length (" +
                                std::to_string(signal.size()) +
                                ") for implicit 'valid' mode processing.");
  }

  // --- Calculate Output Size ---
  // This depends heavily on the specific convolution mode ('valid' assumed
  // here) and the exact downsampling point used by the backend (e.g., phase).
  // Output size for 'valid' convolution is N - M + 1.
  // Downsampling then takes floor((N - M + 1 - 1) / factor) + 1 ? Or similar.
  // Let's use a common formula, but acknowledge this might need adjustment
  // based on backend behavior. Formula from scipy.signal.decimate (zero-phase):
  // ceil(N / factor) - maybe too simple? Formula based on 'valid' conv then
  // downsample: floor( (N-M+1 - 1)/factor ) + 1 = floor( (N-M)/factor ) + 1
  // Let's use the valid conv based one for now.
  size_t valid_conv_len = signal.size() - kernel.size() + 1;
  if (valid_conv_len == 0) return {};  // Handle case where valid length is zero

  // Calculate output length after downsampling
  // Note: The exact rounding (floor/ceil) might depend on backend
  // implementation details Using std::floor and adding 1 is common for 0-based
  // indexing start.
  size_t output_len = static_cast<size_t>(std::floor(
                          static_cast<double>(valid_conv_len - 1) / factor)) +
                      1;

  if (output_len == 0) {
    // If calculated size is 0, return empty vector
    return {};
  }

  // --- Prepare Output & Call Backend ---
  std::vector<T> output(output_len);

  OMNIDSP_STATUS status = Backend::filter_and_downsample_impl<T>(
      signal.data(), signal.size(),  // Input signal pointer and length
      output.data(), output.size(),  // Output buffer pointer and length
      kernel.data(), kernel.size(),  // Kernel pointer and length
      factor                         // Downsampling factor
  );

  // --- Check Status and Return ---
  if (status != OMNIDSP_SUCCESS) {
    // Provide a more informative error message if possible
    throw std::runtime_error(
        "Backend::filter_and_downsample_impl failed with status code: " +
        std::to_string(status));
  }

  return output;  // Return the vector populated by the backend
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
template std::vector<float> filter_and_downsample<float>(
    const std::vector<float> &signal, const std::vector<float> &kernel,
    int factor);
template std::vector<double> filter_and_downsample<double>(
    const std::vector<double> &signal, const std::vector<double> &kernel,
    int factor);

}  // namespace OmniDSP
