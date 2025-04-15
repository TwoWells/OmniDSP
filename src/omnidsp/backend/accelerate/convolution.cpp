/**
 * @file src/omnidsp/backend/accelerate/convolution.cpp
 * @brief Apple Accelerate backend implementation for OmniDSP convolution/correlation.
 *
 * Implements the convolve1d_impl function using Apple's Accelerate framework (vDSP),
 * utilizing vDSP_fir and vDSP_vrvrs.
 * Compiled only when USE_ACCELERATE is defined.
 */

// --- Includes ---
#include <vector>
#include <complex>     // Often included with DSP headers
#include <stdexcept>   // For std::runtime_error, std::invalid_argument
#include <string>
#include <type_traits> // For std::is_same_v
#include <algorithm>   // For std::reverse (if needed as fallback), std::copy
#include <vector>      // Ensure std::vector is included

#include "../backend_impl.h" // Internal backend function declarations

// Only compile if USE_ACCELERATE is defined by CMake (typically on macOS)
#if defined(USE_ACCELERATE)

#include <Accelerate/Accelerate.h> // Main header for the Accelerate framework (includes vDSP)

namespace OmniDSP
{
namespace Backend
{

    // --- vDSP Helper Structs ---
    // Use template specialization to select the correct vDSP function (float/double) at compile time.

    // Helper for vDSP_fir (Finite Impulse Response filter)
    template <typename T> struct vDSPFIRHelper;
    template <> struct vDSPFIRHelper<float>  { static constexpr auto fir = vDSP_fir_f; }; // vDSP_fir_f for float
    template <> struct vDSPFIRHelper<double> { static constexpr auto fir = vDSP_fir_d; }; // vDSP_fir_d for double

    // Helper for vDSP_vrvrs (Vector Reverse)
    template <typename T> struct vDSPReverseHelper;
    template <> struct vDSPReverseHelper<float>  { static constexpr auto vrvrs = vDSP_vrvrs; }; // vDSP_vrvrs for float
    template <> struct vDSPReverseHelper<double> { static constexpr auto vrvrs = vDSP_vrvrsD; }; // vDSP_vrvrsD for double

    /**
     * @brief Accelerate backend implementation for 1D convolution or correlation.
     * Uses vDSP_fir for correlation directly. For convolution, it reverses
     * the kernel using vDSP_vrvrs first, then calls vDSP_fir.
     * Always calculates the 'valid' part of the result.
     *
     * @tparam T float or double.
     * @param signal The input signal vector.
     * @param kernel The kernel vector.
     * @param use_correlation If true, perform correlation; otherwise, perform convolution.
     * @return std::vector<T> The result vector ('valid' mode).
     * @throws std::invalid_argument If inputs are invalid (empty, or kernel > signal).
     */
    template <typename T>
    std::vector<T> convolve1d_impl(const std::vector<T> &signal, const std::vector<T> &kernel, bool use_correlation)
    {
        // --- Input Validation ---
        if (signal.empty() || kernel.empty()) {
            return {}; // Return empty vector if inputs are empty
        }

        // vDSP uses vDSP_Length (unsigned long) for lengths
        vDSP_Length sig_len = signal.size();
        vDSP_Length ker_len = kernel.size();

        // Calculate output length for 'valid' mode: N_out = N_signal - N_kernel + 1
        // Use signed type for intermediate calculation to detect negative result safely.
        long long out_len_signed = static_cast<long long>(sig_len) - static_cast<long long>(ker_len) + 1;
        if (out_len_signed <= 0) {
            // Signal must be at least as long as kernel for 'valid' output.
            throw std::invalid_argument("Accelerate FIR/Conv ('valid' mode): signal length ("
                                        + std::to_string(sig_len) + ") must be >= kernel length ("
                                        + std::to_string(ker_len) + ").");
        }
        vDSP_Length out_len = static_cast<vDSP_Length>(out_len_signed);
        std::vector<T> result(out_len); // Allocate result vector

        // Make a mutable copy of the kernel because vDSP_vrvrs modifies its input,
        // and the input kernel parameter is const.
        std::vector<T> kernel_to_use = kernel;

        // --- Perform Operation ---
        if (!use_correlation) // CONVOLUTION: Reverse kernel first
        {
            if (ker_len > 0) // Need kernel to reverse
            {
                // Use the type-specific helper to call vDSP_vrvrs or vDSP_vrvrsD
                vDSPReverseHelper<T>::vrvrs(kernel_to_use.data(), /*stride=*/1, ker_len);
            }
            // Now kernel_to_use holds the time-reversed kernel.
            // Fall through to call vDSP_fir with the reversed kernel.
        }
        // else: CORRELATION: Use kernel_to_use directly (it's the original kernel).

        // Call vDSP_fir (or vDSP_fir_d) using the appropriate kernel
        // Note: vDSP_fir arguments: (Input, InputStride, FilterKernel, FilterStride, Output, OutputStride, OutputLength, FilterLength)
        // vDSP Documentation uses different variable names (Input=C, Filter=F, Output=A, N=OutputLength, M=FilterLength)
        vDSPFIRHelper<T>::fir(signal.data(),        // Input Signal C -> A in docs (Signal to filter)
                              /*InputStride=*/1,    // Stride for input signal
                              kernel_to_use.data(), // Filter F -> H in docs (Kernel coefficients)
                              /*FilterStride=*/1,   // Stride for filter kernel
                              result.data(),        // Output A -> C in docs (Result buffer)
                              /*OutputStride=*/1,   // Stride for output
                              out_len,              // Length of Output NC -> N-M+1 (Number of output samples)
                              ker_len);             // Length of Filter M -> M (Number of coefficients)

        return result; // Return the computed 'valid' part
    }

    // --- Explicit Template Instantiations ---
    // Ensures the compiler generates code for both float and double versions.
    template std::vector<float> convolve1d_impl<float>(const std::vector<float> &, const std::vector<float> &, bool);
    template std::vector<double> convolve1d_impl<double>(const std::vector<double> &, const std::vector<double> &, bool);

} // namespace Backend
} // namespace OmniDSP

#endif // USE_ACCELERATE
