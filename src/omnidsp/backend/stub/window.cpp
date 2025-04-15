/**
 * @file src/omnidsp/backend/stub/window.cpp
 * @brief Stub (Error) backend implementation for OmniDSP window functions.
 *
 * Provides stub implementations for backend window generation functions.
 * Compiled only when no real backend (oneMKL or Accelerate) is selected.
 * Any attempt to use these functions will result in a std::runtime_error.
 */

// --- Includes ---
#include <vector>
#include <cmath>       // Included for completeness, though not used by stubs
#include <stdexcept>   // For std::runtime_error
#include <string>

#include "../backend_impl.h" // Internal backend function declarations

// Compile this only if NEITHER Accelerate nor MKL is defined by CMake
#if !defined(USE_ACCELERATE) && !defined(USE_ONEMKL)

namespace OmniDSP
{
namespace Backend
{
    // --- Error Message Helper ---
    // Consistently reports that the backend is missing.
    inline void throw_backend_missing_error(const char* func_name) {
         throw std::runtime_error(std::string("OmniDSP backend (MKL/Accelerate) not selected or available during build. Cannot call ") + func_name + ".");
    }

    // --- Backend Window Generation Implementations (Stubs) ---

    template <typename T>
    std::vector<T> generate_hann_window_impl(size_t length) {
        throw_backend_missing_error("generate_hann_window_impl");
        return {}; // Unreachable
    }

    template <typename T>
    std::vector<T> generate_hamming_window_impl(size_t length) {
        throw_backend_missing_error("generate_hamming_window_impl");
        return {}; // Unreachable
    }

    template <typename T>
    std::vector<T> generate_kaiser_window_impl(size_t length, T beta) {
        throw_backend_missing_error("generate_kaiser_window_impl");
        return {}; // Unreachable
    }

    template <typename T>
    std::vector<T> generate_flattop_window_impl(size_t length) {
         throw_backend_missing_error("generate_flattop_window_impl");
        return {}; // Unreachable
    }

    // --- Explicit Template Instantiations ---
    // Instantiate the stub functions for both float and double.
    template std::vector<float> generate_hann_window_impl<float>(size_t length);
    template std::vector<double> generate_hann_window_impl<double>(size_t length);

    template std::vector<float> generate_hamming_window_impl<float>(size_t length);
    template std::vector<double> generate_hamming_window_impl<double>(size_t length);

    template std::vector<float> generate_kaiser_window_impl<float>(size_t length, float beta);
    template std::vector<double> generate_kaiser_window_impl<double>(size_t length, double beta);

    template std::vector<float> generate_flattop_window_impl<float>(size_t length);
    template std::vector<double> generate_flattop_window_impl<double>(size_t length);

} // namespace Backend
} // namespace OmniDSP

#endif // !USE_ACCELERATE && !USE_ONEMKL
