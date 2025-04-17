/**
 * @file src/omnidsp/backend/accelerate/convolution.cpp
 * @brief Apple Accelerate backend implementation for OmniDSP convolution/correlation.
 *
 * Implements the convolve1d_impl function using Apple's Accelerate framework (vDSP),
 * utilizing vDSP_conv/vDSP_convD and vDSP_vrvrs/vDSP_vrvrsD.
 * Compiled only when USE_ACCELERATE is defined.
 */

// --- Includes ---
#include <vector>
#include <complex>     // Often included with DSP headers
#include <stdexcept>   // For std::runtime_error, std::invalid_argument, std::logic_error
#include <string>
#include <type_traits> // For std::is_same_v
#include <algorithm>   // For std::copy
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

    // Helper for vDSP_conv (Convolution/Cross-Correlation)
    // Selects the correct vDSP function (float/double) at compile time.
    template <typename T> struct vDSPConvHelper;
    template <> struct vDSPConvHelper<float>  { static constexpr auto conv = vDSP_conv; }; // vDSP_conv for float
    template <> struct vDSPConvHelper<double> { static constexpr auto conv = vDSP_convD; }; // vDSP_convD for double

    // Helper for vDSP_vrvrs (Vector Reverse)
    // Selects the correct vDSP function (float/double) at compile time.
    template <typename T> struct vDSPReverseHelper;
    template <> struct vDSPReverseHelper<float>  { static constexpr auto vrvrs = vDSP_vrvrs; }; // vDSP_vrvrs for float
    template <> struct vDSPReverseHelper<double> { static constexpr auto vrvrs = vDSP_vrvrsD; }; // vDSP_vrvrsD for double

    /**
     * @brief Accelerate backend implementation for 1D convolution or correlation.
     * Uses vDSP_conv/vDSP_convD. For convolution, it reverses the kernel first.
     * For correlation, it uses the kernel directly.
     * Calculates the 'valid' part of the result by extracting it from the 'full' result computed by vDSP_conv.
     *
     * @tparam T float or double.
     * @param signal The input signal vector.
     * @param kernel The kernel vector.
     * @param use_correlation If true, perform correlation; otherwise, perform convolution.
     * @return std::vector<T> The result vector ('valid' mode).
     * @throws std::invalid_argument If inputs are invalid (empty, or kernel > signal).
     * @throws std::logic_error If internal index calculation fails.
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

        // Calculate 'valid' output length: N_out = N_signal - N_kernel + 1
        // Use signed type for intermediate calculation to detect negative result safely.
        long long valid_out_len_signed = static_cast<long long>(sig_len) - static_cast<long long>(ker_len) + 1;
        if (valid_out_len_signed <= 0) {
            // Signal must be at least as long as kernel for 'valid' output.
            throw std::invalid_argument("Accelerate Conv/Corr ('valid' mode): signal length ("
                                        + std::to_string(sig_len) + ") must be >= kernel length ("
                                        + std::to_string(ker_len) + ").");
        }
        vDSP_Length valid_out_len = static_cast<vDSP_Length>(valid_out_len_signed);

        // vDSP_conv calculates the 'full' output: N_full = N_signal + N_kernel - 1
        vDSP_Length full_out_len = sig_len + ker_len - 1;
        std::vector<T> full_result(full_out_len); // Allocate buffer for the full result

        // Make a mutable copy of the kernel because vDSP_vrvrs modifies its input,
        // and the input kernel parameter is const. Also needed for potential reversal.
        std::vector<T> kernel_to_use = kernel;

        // --- Prepare Kernel for vDSP_conv ---
        // vDSP_conv performs cross-correlation. To get convolution, the *filter* (kernel)
        // argument needs to be time-reversed.
        if (!use_correlation) // CONVOLUTION: Reverse kernel first
        {
            if (ker_len > 0) // Need kernel to reverse
            {
                // Use the type-specific helper to call vDSP_vrvrs or vDSP_vrvrsD
                vDSPReverseHelper<T>::vrvrs(kernel_to_use.data(), /*stride=*/1, ker_len);
            }
            // Now kernel_to_use holds the time-reversed kernel.
        }
        // else: CORRELATION: Use kernel_to_use directly (it's the original kernel).

        // --- Perform Operation using vDSP_conv/vDSP_convD ---
        // Note: vDSP_conv arguments:
        // (InputSignal, SignalStride, FilterKernelReversedForConv, FilterStride, Output, OutputStride, OutputLength, FilterLength)
        // It computes cross-correlation. If kernel_to_use is reversed (for convolution), the result is convolution.
        // If kernel_to_use is original (for correlation), the result is correlation.
        vDSPConvHelper<T>::conv(signal.data(),        // Input Signal (C in docs)
                              /*SignalStride=*/1,
                              kernel_to_use.data(), // Filter (H in docs) - reversed if convolution, original if correlation
                              /*FilterStride=*/1,
                              full_result.data(),   // Output Buffer (A in docs) - for full result
                              /*OutputStride=*/1,
                              full_out_len,         // Length of Full Output buffer (N+M-1)
                              ker_len);             // Length of Filter (M)

        // --- Extract 'valid' part ---
        // The 'valid' part corresponds to the section of the full result where the kernel fully overlaps the signal.
        // This segment starts at index (ker_len - 1) in the 'full' result buffer.
        vDSP_Length valid_start_index = ker_len - 1;

        // Allocate the final result vector with the 'valid' size
        std::vector<T> valid_result(valid_out_len);

        // Copy the 'valid' segment from the full result buffer to the final result vector.
        if (valid_out_len > 0) // Check if there's anything to copy
        {
             // Sanity check: Ensure calculated indices are within the bounds of the full_result buffer.
             // This should always be true if the initial length validation passed.
             if (valid_start_index < full_out_len && (valid_start_index + valid_out_len) <= full_out_len)
             {
                 // Use std::copy for efficient copying
                 std::copy(full_result.begin() + valid_start_index, // Start of 'valid' segment in full_result
                           full_result.begin() + valid_start_index + valid_out_len, // End of 'valid' segment
                           valid_result.begin()); // Destination: start of the final 'valid' result vector
             } else {
                 // This case should theoretically not be reached if input validation and length calculations are correct.
                 // Throwing logic_error indicates an unexpected internal state.
                 throw std::logic_error("Internal error: Invalid indices calculated for extracting 'valid' convolution/correlation result.");
             }
        }
        // If valid_out_len is 0, valid_result is already correctly initialized as an empty vector.

        return valid_result; // Return the computed 'valid' part
    }

    // --- Explicit Template Instantiations ---
    // Ensures the compiler generates code for both float and double versions,
    // allowing the linker to find them.
    template std::vector<float> convolve1d_impl<float>(const std::vector<float> &, const std::vector<float> &, bool);
    template std::vector<double> convolve1d_impl<double>(const std::vector<double> &, const std::vector<double> &, bool);

} // namespace Backend
} // namespace OmniDSP

#endif // USE_ACCELERATE
