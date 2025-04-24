/**
 * @file filter.cpp (default)
 * @brief Implements Default backend FIRFilterPlanImpl and IIRFilterPlanImpl
 * classes.
 * @details Provides portable filter implementations using standard C++ and
 * potentially leveraging other default backend components like
 * Highway-accelerated operations if applicable (e.g., for FFT-based FIR
 * filtering).
 */

// Define HWY_TARGET_INCLUDE before including Highway headers if used (e.g., for
// FFT Conv) #ifndef HWY_TARGET_INCLUDE #define HWY_TARGET_INCLUDE
// "src/omnidsp/backend/default/filter.cpp" // Path to this file #endif #include
// "hwy/foreach_target.h" // Must be first Highway include #include
// "hwy/highway.h" #include "hwy/contrib/math/math-inl.h" #include
// "hwy/contrib/complex/complex-inl.h"

#include "OmniDSP/filter.h"  // Public filter interfaces and specs

#include <algorithm>  // For std::copy, std::fill, std::min
#include <complex>    // For std::complex
#include <iostream>   // For debug messages
#include <numeric>    // For std::inner_product (direct FIR) or other algorithms
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "OmniDSP/core_types.h"  // Core types
#include "backend.h"  // Corresponding header for Default backend declarations

// TODO: Include FFT plan headers if using FFT-based FIR filtering
// #include "OmniDSP/fft.h"
// #include "backend/default/fft.cpp" // Access to Default FFT implementation
// details?

// Highway namespace alias within the compilation unit for the active target
// HWY_BEFORE_NAMESPACE();
namespace OmniDSP {
  namespace backend {
    // namespace HWY_NAMESPACE { // Start Highway's target-specific namespace
    // (if Highway used)

    // namespace hn = hwy; // Alias for Highway types/functions

    // --- Helper Functions (if needed) ---

    // --- DefaultFIRFilterPlanImpl ---

    /**
     * @brief Constructor for the Default FIR filter plan implementation.
     * @param coefficients The FIR filter tap coefficients.
     * @throws std::invalid_argument If coefficients vector is empty.
     */
    template <typename T> DefaultFIRFilterPlanImpl<T>::DefaultFIRFilterPlanImpl(
        const std::vector<T>& coefficients)
        : coefficients_(coefficients),
          state_(
              coefficients.empty() ? 0 : coefficients.size() - 1,
              T{0})  // Initialize state buffer (size = taps - 1)
    {
      if (coefficients_.empty()) {
        // A filter with no coefficients is generally invalid.
        throw std::invalid_argument("FIR filter coefficients cannot be empty.");
      }
      std::cout << "Default FIRFilterPlanImpl created. Taps: "
                << coefficients_.size() << std::endl;  // Debug
    }

    /**
     * @brief Destructor for the Default FIR filter plan implementation.
     */
    template <typename T>
    DefaultFIRFilterPlanImpl<T>::~DefaultFIRFilterPlanImpl()
    {
      std::cout << "Default FIRFilterPlanImpl destroyed."
                << std::endl;  // Debug
    }

    /**
     * @brief Executes the FIR filter using direct convolution with state
     * management.
     * @param input The input signal segment.
     * @param output The output signal buffer. Must be at least as large as
     * input.
     * @return Status::Success on success, Status::SizeMismatch if output is too
     * small, Status::InvalidOperation if coefficients are empty.
     */
    template <typename T> Status DefaultFIRFilterPlanImpl<T>::execute(
        std::span<const T> input, std::span<T> output)
    {
      if (coefficients_.empty()) {
        std::fill(output.begin(), output.end(), T{0});
        // Return success or an error? Let's assume success as it produces
        // defined output (zeros) If empty coefficients are invalid, the
        // constructor should have thrown.
        return Status::Success;
      }
      if (output.size() < input.size()) {
        std::cerr << "FIR execute error: Output buffer (" << output.size()
                  << ") smaller than input buffer (" << input.size() << ")."
                  << std::endl;
        return Status::SizeMismatch;
      }
      if (input.empty()) {
        std::fill(
            output.begin(),
            output.begin() + output.size(),
            T{0});  // Zero out if input is empty
        return Status::Success;
      }

      // --- Filtering Logic ---
      // Choose implementation strategy:
      // 1. Direct Convolution: Simple loop, suitable for short filters.
      // Implemented below.
      // 2. FFT-based Convolution (Overlap-Add / Overlap-Save): More efficient
      // for long filters. (Placeholder)

      // --- Strategy 1: Direct Convolution with State Management ---
      size_t num_taps = coefficients_.size();
      size_t state_len = state_.size();  // Should be num_taps - 1

      // Create a temporary working buffer combining the previous state and the
      // new input This avoids complex indexing into separate state/input
      // buffers during convolution.
      std::vector<T> work_buffer;
      work_buffer.reserve(state_len + input.size());
      work_buffer.insert(work_buffer.end(), state_.begin(), state_.end());
      work_buffer.insert(work_buffer.end(), input.begin(), input.end());

      // Calculate output samples
      for (size_t i = 0; i < input.size(); ++i) {
        T result{};
        // The convolution sum for output[i] involves input samples from x[i]
        // down to x[i - num_taps + 1] In the work_buffer, this corresponds to
        // indices from (state_len + i) down to (state_len + i - num_taps + 1)
        size_t work_buffer_start_idx = state_len + i;

        // Perform the dot product (convolution sum): y[i] =
        // sum_{k=0}^{num_taps-1} h[k] * x[i-k] Where x refers to samples in the
        // conceptual infinite signal (represented by work_buffer)
        for (size_t k = 0; k < num_taps; ++k) {
          // Index in work_buffer for x[i-k] is work_buffer_start_idx - k
          size_t work_idx = work_buffer_start_idx - k;
          // We assume work_buffer is large enough (state_len + input.size())
          // and state_len = num_taps - 1, so valid indices are always accessed.
          result += coefficients_[k] * work_buffer[work_idx];
        }
        output[i] = result;
      }

      // Update the state_ vector with the last 'state_len' samples from the
      // work_buffer
      if (state_len > 0) {
        // Ensure work_buffer has enough elements (should always be true if
        // input is not empty)
        if (work_buffer.size() >= state_len) {
          std::copy(
              work_buffer.end() - state_len, work_buffer.end(), state_.begin());
        }
        else {
          // This case indicates a logic error or unexpected empty input
          // scenario
          std::cerr << "FIR execute warning: Work buffer unexpectedly small ("
                    << work_buffer.size() << ") when updating state (needed "
                    << state_len << ")." << std::endl;
          // Reset state as a fallback?
          std::fill(state_.begin(), state_.end(), T{0});
        }
      }

      // Zero out remaining output buffer if it was provided larger than input
      if (output.size() > input.size()) {
        std::fill(output.begin() + input.size(), output.end(), T{0});
      }

      // --- Strategy 2: FFT-based Convolution (Placeholder) ---
      // if (/* filter is long and FFT is beneficial */) {
      //     // TODO: Implement Overlap-Add or Overlap-Save algorithm
      //     // Requires internal FFT plan, block processing, overlap handling.
      //     std::cerr << "FFT-based FIR filtering not yet implemented in
      //     Default backend." << std::endl; return Status::NotImplemented;
      // }

      return Status::Success;
    }

    /**
     * @brief Resets the internal state of the FIR filter (delay line) to zeros.
     * @return Status::Success.
     */
    template <typename T> Status DefaultFIRFilterPlanImpl<T>::reset()
    {
      std::fill(state_.begin(), state_.end(), T{0});
      return Status::Success;
    }

    /**
     * @brief Gets the order of the FIR filter.
     * @return The filter order (number of taps - 1). Returns 0 if coefficients
     * are empty.
     */
    template <typename T> size_t DefaultFIRFilterPlanImpl<T>::get_order() const
    {
      return coefficients_.empty() ? 0 : coefficients_.size() - 1;
    }

    /**
     * @brief Gets the number of taps (coefficients) in the FIR filter.
     * @return The number of filter taps.
     */
    template <typename T>
    size_t DefaultFIRFilterPlanImpl<T>::get_num_taps() const
    {
      return coefficients_.size();
    }

    // --- DefaultIIRFilterPlanImpl ---

    /**
     * @brief Constructor for the Default IIR filter plan implementation.
     * @param sos_coefficients A vector of Second-Order Sections representing
     * the filter.
     * @throws std::invalid_argument If sos_coefficients vector is empty.
     */
    template <typename T> DefaultIIRFilterPlanImpl<T>::DefaultIIRFilterPlanImpl(
        const std::vector<SecondOrderSection<T>>& sos_coefficients)
        : sos_coeffs_(sos_coefficients)
    {
      if (sos_coeffs_.empty()) {
        throw std::invalid_argument("IIR SOS coefficients cannot be empty.");
      }
      // Initialize state buffer: 2 state variables (delays) per SOS section
      // Layout: [s1_0, s2_0, s1_1, s2_1, ..., s1_N-1, s2_N-1] where N =
      // num_sections
      state_.assign(sos_coeffs_.size() * 2, T{0});
      std::cout << "Default IIRFilterPlanImpl created. Sections: "
                << sos_coeffs_.size() << std::endl;  // Debug
    }

    /**
     * @brief Destructor for the Default IIR filter plan implementation.
     */
    template <typename T>
    DefaultIIRFilterPlanImpl<T>::~DefaultIIRFilterPlanImpl()
    {
      std::cout << "Default IIRFilterPlanImpl destroyed."
                << std::endl;  // Debug
    }

    /**
     * @brief Executes the IIR filter using Transposed Direct Form II for each
     * section.
     * @param input The input signal segment.
     * @param output The output signal buffer. Must be at least as large as
     * input.
     * @return Status::Success on success, Status::SizeMismatch if output is too
     * small, Status::InvalidOperation if coefficients are empty.
     */
    template <typename T> Status DefaultIIRFilterPlanImpl<T>::execute(
        std::span<const T> input, std::span<T> output)
    {
      if (sos_coeffs_.empty()) {
        std::fill(output.begin(), output.end(), T{0});
        return Status::Success;  // No filtering if no sections
      }
      if (output.size() < input.size()) {
        std::cerr << "IIR execute error: Output buffer (" << output.size()
                  << ") smaller than input buffer (" << input.size() << ")."
                  << std::endl;
        return Status::SizeMismatch;
      }
      if (input.empty()) {
        std::fill(
            output.begin(),
            output.begin() + output.size(),
            T{0});  // Zero out if input is empty
        return Status::Success;
      }

      // --- Filtering Logic: Transposed Direct Form II Cascade ---
      // For each input sample x[n]:
      //   input_to_section = x[n]
      //   For each section k = 0 to num_sections-1:
      //     y_n = input_to_section * b0_k + state1_k[n-1]  // Output of this
      //     section's feed-forward part state1_k[n] = input_to_section * b1_k -
      //     y_n
      //     * a1_k + state2_k[n-1] // Update state 1 state2_k[n] =
      //     input_to_section
      //     * b2_k - y_n * a2_k               // Update state 2
      //     input_to_section = y_n // Output of this section is input to next
      //   output[n] = y_n // Final output after last section

      size_t num_sections = sos_coeffs_.size();

      for (size_t n = 0; n < input.size(); ++n) {
        T sample_in = input[n];
        T section_in_out = sample_in;  // Input to the first section

        for (size_t k = 0; k < num_sections; ++k) {
          const auto& sos = sos_coeffs_[k];
          T& s1 = state_[k * 2 + 0];  // Reference to state variable 1 (delay
                                      // z^-1) for section k
          T& s2 = state_[k * 2 + 1];  // Reference to state variable 2 (delay
                                      // z^-2) for section k

          // Calculate the output of this section
          T y_n = section_in_out * sos.b0 + s1;

          // Update the states for the next time step
          T s1_next = section_in_out * sos.b1 - y_n * sos.a1 + s2;
          T s2_next = section_in_out * sos.b2 - y_n * sos.a2;

          s1 = s1_next;  // Update state 1
          s2 = s2_next;  // Update state 2

          section_in_out = y_n;  // Output of this section is input to the next
        }
        output[n] = section_in_out;  // Final output of the cascade
      }

      // Zero out remaining output buffer if it was provided larger than input
      if (output.size() > input.size()) {
        std::fill(output.begin() + input.size(), output.end(), T{0});
      }

      return Status::Success;
    }

    /**
     * @brief Resets the internal state of the IIR filter (delay elements) to
     * zeros.
     * @return Status::Success.
     */
    template <typename T> Status DefaultIIRFilterPlanImpl<T>::reset()
    {
      std::fill(state_.begin(), state_.end(), T{0});
      return Status::Success;
    }

    /**
     * @brief Gets the order of the IIR filter.
     * @details This is typically defined as the highest power of z^-1 in the
     * denominator, which for a cascade of N second-order sections is N * 2.
     * @return The filter order. Returns 0 if coefficients are empty.
     */
    template <typename T> size_t DefaultIIRFilterPlanImpl<T>::get_order() const
    {
      return sos_coeffs_.empty() ? 0 : sos_coeffs_.size() * 2;
    }

    /**
     * @brief Gets the number of second-order sections used in the filter
     * implementation.
     * @return The number of SOS sections.
     */
    template <typename T>
    size_t DefaultIIRFilterPlanImpl<T>::get_num_sections() const
    {
      return sos_coeffs_.size();
    }

    // --- Explicit Template Instantiations ---

    // DefaultFIRFilterPlanImpl
    template class DefaultFIRFilterPlanImpl<float>;
    template class DefaultFIRFilterPlanImpl<double>;
    // Complex FIR might be useful for analytic signal processing, etc.
    template class DefaultFIRFilterPlanImpl<std::complex<float>>;
    template class DefaultFIRFilterPlanImpl<std::complex<double>>;

    // DefaultIIRFilterPlanImpl (Typically only for Real types)
    template class DefaultIIRFilterPlanImpl<float>;
    template class DefaultIIRFilterPlanImpl<double>;

    // } // namespace HWY_NAMESPACE (if Highway used)
  }  // namespace backend
}  // namespace OmniDSP
// HWY_AFTER_NAMESPACE(); (if Highway used)

// #if HWY_ONCE // If Highway used
// #include "hwy/foreach_target.h" // Need this for HWY_EXPORT
// #include "hwy/highway.h"
// #include "backend.h"
// #include <vector>
// #include <complex>

// HWY_EXPORT(ExportedSimdFunction); // Export any SIMD functions defined above
// if needed externally

// // Need explicit instantiations outside the namespace for the final object
// code namespace OmniDSP { namespace backend {
//     // template class DefaultFIRFilterPlanImpl<float>; // Already done above
// }
// }
// #endif // HWY_ONCE
