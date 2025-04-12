/**
 * @file convolution.cpp
 * @brief Implementation of public convolution/correlation functions, dispatching to backends.
 */

#include <OmniDSP/convolution.h> // Public API header
#include <vector>
#include <stdexcept>

// Include the internal header that declares the backend implementation functions
#include "backend/backend_impl.h" // Adjust path if necessary

namespace OmniDSP
{
    // --- Public API Function Definitions ---

    /**
     * @brief Performs 1D linear convolution by calling the backend implementation.
     * The backend implementation handles kernel reversal internally if needed.
     */
    template <typename T>
    std::vector<T> convolve1d(const std::vector<T> &signal, const std::vector<T> &kernel)
    {
        // Call the backend implementation, setting use_correlation to false for convolution
        return Backend::convolve1d_impl(signal, kernel, /*use_correlation=*/false);
    }

    /**
     * @brief Performs 1D linear correlation by calling the backend implementation.
     * Suitable for FIR filtering where the kernel represents coefficients directly.
     */
    template <typename T>
    std::vector<T> correlate1d(const std::vector<T> &signal, const std::vector<T> &kernel)
    {
        // Call the backend implementation, setting use_correlation to true for correlation
        return Backend::convolve1d_impl(signal, kernel, /*use_correlation=*/true);
    }

    // --- Explicit Template Instantiations (for public API) ---
    // Ensure these match the declarations in convolution.h
    template std::vector<float> convolve1d<float>(const std::vector<float> &signal, const std::vector<float> &kernel);
    template std::vector<double> convolve1d<double>(const std::vector<double> &signal, const std::vector<double> &kernel);

    template std::vector<float> correlate1d<float>(const std::vector<float> &signal, const std::vector<float> &kernel);
    template std::vector<double> correlate1d<double>(const std::vector<double> &signal, const std::vector<double> &kernel);

} // namespace OmniDSP