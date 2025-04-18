/**
 * @file convolution.cpp
 * @brief Stub implementation for convolution backend.
 */
#include <stdexcept>  // For std::runtime_error
#include <string>     // For error messages
#include <vector>

#include "../backend_impl.h"  // For ConvMode enum

namespace OmniDSP {
namespace Backend {
namespace Stub {

// Helper to throw not implemented error
namespace {
void throw_stub_error(const std::string& func_name) {
  throw std::runtime_error("Stub backend: " + func_name +
                           " is not implemented.");
}
}  // anonymous namespace

/**
 * @brief Stub implementation for 1D convolution. Throws error.
 */
template <typename T>
std::vector<T> convolve1d_stub_impl(const T* signal, size_t signal_len,
                                    const T* kernel, size_t kernel_len,
                                    ConvMode mode)  // Added mode parameter
{
  throw_stub_error("convolve1d");
  // Unreachable, but prevents compiler warning about no return value
  return {};
}

/**
 * @brief Stub implementation for 1D correlation. Throws error.
 */
template <typename T>
std::vector<T> correlate1d_stub_impl(const T* signal, size_t signal_len,
                                     const T* kernel, size_t kernel_len,
                                     ConvMode mode)  // Added mode parameter
{
  throw_stub_error("correlate1d");
  return {};
}

// --- Explicit Template Instantiations ---
template std::vector<float> convolve1d_stub_impl<float>(const float*, size_t,
                                                        const float*, size_t,
                                                        ConvMode);
template std::vector<double> convolve1d_stub_impl<double>(const double*, size_t,
                                                          const double*, size_t,
                                                          ConvMode);

template std::vector<float> correlate1d_stub_impl<float>(const float*, size_t,
                                                         const float*, size_t,
                                                         ConvMode);
template std::vector<double> correlate1d_stub_impl<double>(const double*,
                                                           size_t,
                                                           const double*,
                                                           size_t, ConvMode);

}  // namespace Stub
}  // namespace Backend
}  // namespace OmniDSP
