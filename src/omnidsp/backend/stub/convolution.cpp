/**
 * @file src/omnidsp/backend/stub/convolution.cpp
 * @brief Stub (Error) backend implementation for OmniDSP
 * convolution/correlation.
 *
 * Provides stub implementations for backend convolution/correlation functions.
 * Compiled only when no real backend (oneMKL or Accelerate) is selected.
 * Any attempt to use these functions will result in a std::runtime_error.
 */

// --- Includes ---
#include <complex>    // Often included with DSP headers
#include <stdexcept>  // For std::runtime_error
#include <string>
#include <vector>

#include "../backend_impl.h"  // Internal backend function declarations

// Compile this only if NEITHER Accelerate nor MKL is defined by CMake
#if !defined(USE_ACCELERATE) && !defined(USE_ONEMKL)

namespace OmniDSP {
namespace Backend {

/**
 * @brief Stub implementation for 1D convolution or correlation. Throws error.
 */
template <typename T>
std::vector<T> convolve1d_impl(const std::vector<T> &signal,
                               const std::vector<T> &kernel,
                               bool use_correlation) {
  // Throw an error indicating the backend is not available.
  throw std::runtime_error(
      "OmniDSP backend (MKL/Accelerate) not selected or available during "
      "build. Cannot perform convolution/correlation.");
  // Add return statement to satisfy compiler, although unreachable
  return {};
}

// --- Explicit Template Instantiations ---
// Instantiate the stub function for both float and double.
template std::vector<float> convolve1d_impl<float>(const std::vector<float> &,
                                                   const std::vector<float> &,
                                                   bool);
template std::vector<double> convolve1d_impl<double>(
    const std::vector<double> &, const std::vector<double> &, bool);

}  // namespace Backend
}  // namespace OmniDSP

#endif  // !USE_ACCELERATE && !USE_ONEMKL
