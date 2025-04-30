/**
 * @file filter.cpp (default)
 * @brief Implements Default backend FIRFilterPlanImpl and IIRFilterPlanImpl
 * classes.
 * @details Provides portable filter implementations using standard C++.
 */

// Highway includes and definitions removed

#include "filter.hpp"  // Includes DefaultFIRFilterPlanImpl/DefaultIIRFilterPlanImpl declarations

#include <OmniDSP/filter.hpp>  // Public filter interfaces and specs (FIRFilterSpec, IIRFilterSpec, IIRFilterCoef)
#include <algorithm>           // For std::copy, std::fill, std::min
#include <complex>             // For std::complex
#include <iostream>            // For debug messages
#include <numeric>  // For std::inner_product (direct FIR) or other algorithms
#include <span>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "OmniDSP/core_types.hpp"  // Core types
#include "backend.hpp"  // Corresponding header for Default backend declarations (needed?)

// TODO: Include FFT plan headers if using FFT-based FIR filtering
// #include <OmniDSP/fft.hpp>
// #include "fft.hpp" // Access to Default FFT implementation details?

namespace OmniDSP {
  namespace backend {

    // --- Helper Functions (if needed) ---

    // --- DefaultFIRFilterPlanImpl ---

    /**
     * @brief Constructor for the Default FIR filter plan implementation.
     * @param coefficients The FIR filter tap coefficients.
     * @throws std::invalid_argument If coefficients vector is empty.
     */
    template <typename T>
    DefaultFIRFilterPlanImpl<T>::DefaultFIRFilterPlanImpl(
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
      // std::cout << "Default FIRFilterPlanImpl created. Taps: "
      //           << coefficients_.size() << std::endl;  // Debug
    }

    /**
     * @brief Destructor for the Default FIR filter plan implementation.
     */
    template <typename T>
    DefaultFIRFilterPlanImpl<T>::~DefaultFIRFilterPlanImpl()
    {
      // std::cout << "Default FIRFilterPlanImpl destroyed."
      //           << std::endl;  // Debug
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
    template <typename T>
    Status DefaultFIRFilterPlanImpl<T>::execute(
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
        // If input is empty, output should also be empty (or zeroed if larger)
        size_t samples_to_zero
            = std::min(input.size(), output.size());  // Should be 0
        std::fill(output.begin(), output.begin() + samples_to_zero, T{0});
        if (output.size() > samples_to_zero) {
          std::fill(output.begin() + samples_to_zero, output.end(), T{0});
        }
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
    template <typename T>
    Status DefaultFIRFilterPlanImpl<T>::reset()
    {
      std::fill(state_.begin(), state_.end(), T{0});
      return Status::Success;
    }

    /**
     * @brief Gets the order of the FIR filter.
     * @return The filter order (number of taps - 1). Returns 0 if coefficients
     * are empty.
     */
    template <typename T>
    size_t DefaultFIRFilterPlanImpl<T>::get_order() const
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
     * @param sos_coefficients A vector of IIRFilterCoef representing the filter
     * sections.
     * @throws std::invalid_argument If sos_coefficients vector is empty.
     */
    template <typename T>
    DefaultIIRFilterPlanImpl<T>::DefaultIIRFilterPlanImpl(
        const std::vector<IIRFilterCoef>&
            sos_coefficients)  // Use IIRFilterCoef
    // : sos_coeffs_(sos_coefficients) // Don't store the double version
    // directly
    {
      if (sos_coefficients.empty()) {
        throw std::invalid_argument("IIR SOS coefficients cannot be empty.");
      }

      // Convert double coefficients from IIRFilterCoef to type T for internal
      // use This avoids potential type issues during execution if T is float.
      internal_coeffs_.reserve(sos_coefficients.size());
      for (const auto& sos_double : sos_coefficients) {
        InternalSOS<T> sos_t;
        sos_t.b0 = static_cast<T>(sos_double.b0);
        sos_t.b1 = static_cast<T>(sos_double.b1);
        sos_t.b2 = static_cast<T>(sos_double.b2);
        // Assume a0 is 1.0, if not, normalization should happen here or in
        // design
        if (std::abs(sos_double.a0 - 1.0)
            > 1e-9) {  // Check if a0 is close to 1
          // Optional: Normalize coefficients by a0 here if design doesn't
          // guarantee it T a0_inv = T{1.0} / static_cast<T>(sos_double.a0);
          // sos_t.b0 *= a0_inv; sos_t.b1 *= a0_inv; sos_t.b2 *= a0_inv;
          // sos_t.a1 = static_cast<T>(sos_double.a1) * a0_inv;
          // sos_t.a2 = static_cast<T>(sos_double.a2) * a0_inv;
          // Alternatively, throw an error if a0 != 1 is unexpected
          throw std::runtime_error(
              "IIR SOS coefficient a0 is not 1.0 (normalization required).");
        }
        sos_t.a1 = static_cast<T>(sos_double.a1);
        sos_t.a2 = static_cast<T>(sos_double.a2);
        internal_coeffs_.push_back(sos_t);
      }

      // Initialize state buffer: 2 state variables (delays) per SOS section
      // Layout: [s1_0, s2_0, s1_1, s2_1, ..., s1_N-1, s2_N-1] where N =
      // num_sections
      state_.assign(internal_coeffs_.size() * 2, T{0});
      // std::cout << "Default IIRFilterPlanImpl created. Sections: "
      //           << internal_coeffs_.size() << std::endl;  // Debug
    }

    /**
     * @brief Destructor for the Default IIR filter plan implementation.
     */
    template <typename T>
    DefaultIIRFilterPlanImpl<T>::~DefaultIIRFilterPlanImpl()
    {
      // std::cout << "Default IIRFilterPlanImpl destroyed."
      //           << std::endl;  // Debug
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
    template <typename T>
    Status DefaultIIRFilterPlanImpl<T>::execute(
        std::span<const T> input, std::span<T> output)
    {
      if (internal_coeffs_.empty()) {  // Check internal coefficients
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
        // If input is empty, output should also be empty (or zeroed if larger)
        size_t samples_to_zero
            = std::min(input.size(), output.size());  // Should be 0
        std::fill(output.begin(), output.begin() + samples_to_zero, T{0});
        if (output.size() > samples_to_zero) {
          std::fill(output.begin() + samples_to_zero, output.end(), T{0});
        }
        return Status::Success;
      }

      // --- Filtering Logic: Transposed Direct Form II Cascade ---
      size_t num_sections
          = internal_coeffs_.size();  // Use internal coefficients

      for (size_t n = 0; n < input.size(); ++n) {
        T sample_in = input[n];
        T section_in_out = sample_in;  // Input to the first section

        for (size_t k = 0; k < num_sections; ++k) {
          const auto& sos = internal_coeffs_[k];  // Use internal coefficients
          T& s1 = state_[k * 2 + 0];  // Reference to state variable 1 (delay
                                      // z^-1)
          T& s2 = state_[k * 2 + 1];  // Reference to state variable 2 (delay
                                      // z^-2)

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
    template <typename T>
    Status DefaultIIRFilterPlanImpl<T>::reset()
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
    template <typename T>
    size_t DefaultIIRFilterPlanImpl<T>::get_order() const
    {
      return internal_coeffs_.empty()
                 ? 0
                 : internal_coeffs_.size() * 2;  // Use internal coefficients
    }

    /**
     * @brief Gets the number of second-order sections used in the filter
     * implementation.
     * @return The number of SOS sections.
     */
    template <typename T>
    size_t DefaultIIRFilterPlanImpl<T>::get_num_sections() const
    {
      return internal_coeffs_.size();  // Use internal coefficients
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

  }  // namespace backend
}  // namespace OmniDSP
