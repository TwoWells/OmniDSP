/**
 * @file src/omnidsp/resample.cpp
 * @brief Implementation of public resampling functions, dispatching to
 * backends.
 */

#include <OmniDSP/resample.h>  // Public API header

#include <stdexcept>  // Potentially needed if backend throws
#include <vector>

// Include the internal header that declares the backend implementation
// functions
#include "backend/backend_impl.h"  // Adjust path if necessary

namespace OmniDSP {
// --- Public API Function Definition ---

/**
 * @brief Performs combined FIR filtering and downsampling by calling the
 * backend implementation.
 */
template <typename T>
std::vector<T> filter_and_downsample(const std::vector<T> &signal,
                                     const std::vector<T> &kernel, int factor) {
  // Call the backend implementation function declared in backend_impl.h
  return Backend::filter_and_downsample_impl(signal, kernel, factor);
}

// --- Explicit Template Instantiations (for public API) ---
// Ensure these match the declarations in resample.h.
template std::vector<float> filter_and_downsample<float>(
    const std::vector<float> &signal, const std::vector<float> &kernel,
    int factor);
template std::vector<double> filter_and_downsample<double>(
    const std::vector<double> &signal, const std::vector<double> &kernel,
    int factor);

}  // namespace OmniDSP
