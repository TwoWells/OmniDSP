/**
 * @file resample.cpp (default)
 * @brief Implements the Default backend ResamplePlanImpl class using standard
 * C++.
 * @details Provides a portable resampling implementation using polyphase FIR
 * filtering. The FIR filter is designed using the owner AbstractBackend
 * instance.
 */

#include <OmniDSP/filter.hpp>  // For non-templated FIRFilterSpec, FilterType, etc.
#include <OmniDSP/resample.hpp>  // For ResampleSpec definition
#include <OmniDSP/window.hpp>    // For WindowSpec
#include <algorithm>             // For std::copy, std::fill
#include <cassert>               // For assert macro
#include <cmath>      // For std::ceil, std::floor, std::min, std::max, std::abs
#include <iostream>   // For debug messages
#include <numeric>    // For std::gcd
#include <span>       // For std::span
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "../interface/backend.hpp"  // For ResamplePlanImpl, AbstractBackend base classes
#include "backend.hpp"  // Default backend declarations (includes DefaultResamplePlanImpl declaration)

// Forward declare the DefaultResamplePlanImpl class within the namespace
namespace OmniDSP {
  namespace backend {
    template <typename T>
    class DefaultResamplePlanImpl;
  }
}  // namespace OmniDSP

// Include the header that defines ResamplePlanImpl (if not already included via
// backend.hpp) Assuming ResamplePlanImpl is defined in ../interface/backend.hpp
// or similar #include "../interface/backend.hpp"

namespace OmniDSP {
  namespace backend {

    //--------------------------------------------------------------------------
    // DefaultResamplePlanImpl Definition
    //--------------------------------------------------------------------------

    /**
     * @brief Concrete implementation of ResamplePlanImpl for the Default
     * backend. Uses standard C++ and polyphase FIR filtering.
     */
    template <typename T>  // T is real type here
    class DefaultResamplePlanImpl final : public ResamplePlanImpl<T> {
     public:
      /**
       * @brief Constructor. Calculates factors, designs filter, initializes
       * state.
       * @param owner Pointer to the AbstractBackend instance creating this
       * plan.
       * @param spec The resampling specification.
       * @throws std::invalid_argument If spec or owner is invalid.
       * @throws std::runtime_error If factor calculation or filter design
       * fails.
       */
      DefaultResamplePlanImpl(
          const AbstractBackend* owner,  // Use AbstractBackend*
          const ResampleSpec& spec);

      /** @brief Destructor. */
      ~DefaultResamplePlanImpl() override;

      // --- Interface Methods ---
      Status execute(std::span<const T> input, std::span<T> output) override;
      Status reset() override;
      double get_input_rate() const override;
      double get_output_rate() const override;
      size_t get_output_length(size_t input_length) const override;

     private:
      // --- Configuration ---
      double input_rate_;
      double output_rate_;
      size_t upsample_factor_L_;
      size_t downsample_factor_M_;
      std::vector<T> prototype_filter_coeffs_;  // The prototype FIR filter
                                                // coefficients (scaled by L)

      // --- State ---
      std::vector<T> filter_state_;  // Internal state (delay line)
      size_t current_phase_
          = 0;  // Current phase index for polyphase processing

      // --- Helper Methods (Specific to this implementation) ---
      /** @brief Calculates L and M factors. */
      static Status calculate_factors_static(
          double in_rate, double out_rate, size_t& L, size_t& M);
      /** @brief Designs the prototype filter using the owner backend. */
      Status design_filter(
          const AbstractBackend* owner,  // Use AbstractBackend*
          const ResampleSpec& spec,
          size_t L,
          size_t M);
    };

    //--------------------------------------------------------------------------
    // DefaultResamplePlanImpl Method Implementations
    //--------------------------------------------------------------------------

    template <typename T>
    DefaultResamplePlanImpl<T>::DefaultResamplePlanImpl(
        const AbstractBackend* owner,  // Use AbstractBackend*
        const ResampleSpec& spec)
        : input_rate_(spec.input_rate), output_rate_(spec.output_rate)
    {
      if (!owner) {
        throw std::invalid_argument(
            "DefaultResamplePlanImpl requires a valid owner "
            "AbstractBackend "  // Changed message
            "pointer.");
      }
      if (!spec.validate()) {
        throw std::invalid_argument("Invalid ResampleSpec provided.");
      }

      // 1. Calculate L/M factors
      Status factor_status = calculate_factors_static(
          input_rate_, output_rate_, upsample_factor_L_, downsample_factor_M_);
      if (factor_status != Status::Success) {
        throw std::runtime_error(
            "Failed to calculate resampling factors L and M.");
      }

      // 2. Design the prototype FIR filter using the owner's design method
      Status filter_status = design_filter(
          owner, spec, upsample_factor_L_, downsample_factor_M_);
      if (filter_status != Status::Success) {
        throw std::runtime_error(
            "Failed to design prototype FIR filter for resampling.");
      }

      // 3. Initialize state buffer
      size_t num_taps = prototype_filter_coeffs_.size();
      if (num_taps > 0) {
        // State length should be related to the number of taps per branch
        size_t num_branch_taps
            = (num_taps + upsample_factor_L_ - 1) / upsample_factor_L_;
        size_t state_len = (num_branch_taps > 0) ? num_branch_taps - 1 : 0;
        filter_state_.assign(state_len, T{0});
      }
      else {
        // Handle case where filter design somehow resulted in zero taps (should
        // be error earlier)
        filter_state_.clear();
        // Optionally throw or log warning
        std::cerr << "Warning: Resampler prototype filter has zero taps."
                  << std::endl;
      }

      // 4. Initialize phase
      current_phase_ = 0;

      // Debug output (optional)
      // std::cout << "Default ResamplePlanImpl created. L=" <<
      // upsample_factor_L_
      //           << ", M=" << downsample_factor_M_ << ", Taps=" << num_taps
      //           << ", StateLen=" << filter_state_.size() << std::endl;
    }

    template <typename T>
    DefaultResamplePlanImpl<T>::~DefaultResamplePlanImpl()
    {
      // std::cout << "Default ResamplePlanImpl destroyed." << std::endl;  //
      // Debug
    }

    template <typename T>
    Status DefaultResamplePlanImpl<T>::execute(
        std::span<const T> input, std::span<T> output)
    {
      if (upsample_factor_L_ == 0 || downsample_factor_M_ == 0
          || prototype_filter_coeffs_.empty()) {
        return Status::InvalidOperation;  // Plan not initialized correctly
      }

      size_t input_len = input.size();
      size_t output_len_required
          = get_output_length(input_len);  // Use estimate

      // Determine the actual number of samples we can write to the output
      // buffer
      size_t output_len_available = output.size();
      size_t output_samples_to_generate
          = std::min(output_len_required, output_len_available);

      if (output_samples_to_generate == 0 && input_len == 0
          && filter_state_.empty()) {
        // Nothing to write and nothing to process (no input, no state)
        return Status::Success;
      }
      if (output_samples_to_generate == 0
          && (input_len > 0 || !filter_state_.empty())) {
        // Output buffer is zero size, but there's work to do.
        // We might still need to update the state.
        // Or maybe return SizeMismatch? Let's proceed but write 0 samples.
      }

      // --- Polyphase Resampling Implementation (Direct FIR filtering approach)
      // ---
      const size_t L = upsample_factor_L_;
      const size_t M = downsample_factor_M_;
      const size_t num_taps = prototype_filter_coeffs_.size();
      const size_t state_len = filter_state_.size();
      const size_t num_branch_taps
          = (num_taps + L - 1) / L;  // Taps per polyphase branch

      // Ensure state length matches calculation based on branch taps (for debug
      // builds)
      assert(state_len == ((num_branch_taps > 0) ? num_branch_taps - 1 : 0));

      // Combine state and new input into a single buffer for easier processing
      std::vector<T> work_buffer;
      work_buffer.reserve(state_len + input_len);
      work_buffer.insert(
          work_buffer.end(), filter_state_.begin(), filter_state_.end());
      work_buffer.insert(work_buffer.end(), input.begin(), input.end());

      size_t output_idx = 0;
      // size_t input_samples_consumed = 0; // Not strictly needed here

      // Loop to generate output samples, up to the available output space
      while (output_idx < output_samples_to_generate) {
        // Calculate the base index in the work_buffer corresponding
        // to the start of the filter application for this output sample.
        size_t current_input_base_idx = current_phase_ / L;

        // Check if we have enough data in work_buffer to apply the filter for
        // this output sample The filter needs data up to index:
        // current_input_base_idx + num_branch_taps - 1
        if (current_input_base_idx + num_branch_taps > work_buffer.size()) {
          // Not enough input samples + state to compute this output sample.
          break;  // Stop producing output samples for this call
        }

        // Determine the polyphase filter branch index (phase)
        size_t phase_index = current_phase_ % L;

        // Apply the appropriate polyphase filter branch
        T sum{};
        for (size_t n = 0; n < num_branch_taps; ++n) {
          size_t proto_idx
              = phase_index + n * L;  // Index into prototype filter
          if (proto_idx < num_taps) {
            // Index into the work buffer (state + input)
            size_t work_idx = current_input_base_idx + n;
            sum += work_buffer[work_idx] * prototype_filter_coeffs_[proto_idx];
          }
        }

        output[output_idx] = sum;  // Store the computed output sample
        output_idx++;

        // Advance the current phase accumulator by M for the next output sample
        current_phase_ += M;

        // Track the highest index reached in the *original input* part of
        // work_buffer input_samples_consumed = std::max(input_samples_consumed,
        // (current_input_base_idx + num_branch_taps > state_len) ?
        // (current_input_base_idx + num_branch_taps - state_len) : 0);

      }  // End while loop for output samples

      // Update the filter state for the next call
      // The state consists of the last `state_len` samples from the work_buffer
      // that were involved in the calculation.
      if (state_len > 0) {
        // Calculate the starting index in work_buffer for the state needed next
        // time. This corresponds to the samples *after* the last fully consumed
        // input sample. The base index for the *next* output sample would be
        // current_phase_ / L. The state needed starts before that.
        size_t next_input_base_idx = current_phase_ / L;
        size_t state_start_idx = (next_input_base_idx >= state_len)
                                     ? (next_input_base_idx - state_len)
                                     : 0;

        // Ensure the state_start_idx is within the bounds of the work_buffer
        state_start_idx = std::min(state_start_idx, work_buffer.size());

        size_t num_available_for_state = work_buffer.size() - state_start_idx;
        size_t num_to_copy = std::min(state_len, num_available_for_state);

        if (num_to_copy > 0) {
          std::copy(
              work_buffer.begin() + state_start_idx,
              work_buffer.begin() + state_start_idx + num_to_copy,
              filter_state_.begin());
        }
        // Zero pad if fewer than state_len samples were available to copy
        if (num_to_copy < state_len) {
          std::fill(
              filter_state_.begin() + num_to_copy, filter_state_.end(), T{0});
        }
      }

      // Adjust phase accumulator for the next block start.
      // Keep the accumulated phase, but use modulo L for indexing calculations.
      // current_phase_ = current_phase_ % (L * M); // Optional bounding

      // Zero out remaining part of the *user-provided* output buffer if we
      // wrote fewer samples than the buffer size.
      if (output_idx < output.size()) {
        std::fill(output.begin() + output_idx, output.end(), T{0});
      }

      return Status::Success;
    }

    template <typename T>
    Status DefaultResamplePlanImpl<T>::reset()
    {
      std::fill(filter_state_.begin(), filter_state_.end(), T{0});
      current_phase_ = 0;
      return Status::Success;
    }

    template <typename T>
    double DefaultResamplePlanImpl<T>::get_input_rate() const
    {
      return input_rate_;
    }

    template <typename T>
    double DefaultResamplePlanImpl<T>::get_output_rate() const
    {
      return output_rate_;
    }

    template <typename T>
    size_t DefaultResamplePlanImpl<T>::get_output_length(
        size_t input_length) const
    {
      if (input_rate_ <= 0.0 || upsample_factor_L_ == 0
          || downsample_factor_M_ == 0)
        return 0;

      // Estimate based on ratio.
      double ratio = static_cast<double>(upsample_factor_L_)
                     / static_cast<double>(downsample_factor_M_);
      size_t estimated_len = static_cast<size_t>(
          std::ceil(static_cast<double>(input_length) * ratio));

      // Add a small constant (e.g., 1 or 2) to account for filter delay and
      // phase alignment edge cases. A more precise estimate might involve
      // num_taps, L, M, but a small constant is often sufficient for buffer
      // allocation purposes. Let's use +2 for safety.
      estimated_len += 2;

      return estimated_len;
    }

    // --- Helper Implementations ---

    template <typename T>
    Status DefaultResamplePlanImpl<T>::calculate_factors_static(
        double in_rate, double out_rate, size_t& L, size_t& M)
    {
      if (in_rate <= 0.0 || out_rate <= 0.0) {
        return Status::InvalidArgument;
      }
      // Use large integer base for approximation to find rational L/M
      const long long factor_base = 1LL << 32;  // Example large base
      // Calculate target ratio * base, round to nearest integer
      long long num_approx
          = static_cast<long long>((out_rate / in_rate) * factor_base + 0.5);

      if (num_approx == 0 && out_rate > 0)
        num_approx = 1;  // Avoid zero upsample factor if output rate > 0
      if (num_approx < 0)
        return Status::Failure;  // Should not happen if rates > 0

      // Find greatest common divisor (GCD) - requires <numeric> in C++17
      long long common = std::gcd(num_approx, factor_base);

      // Check for potential issues before division
      if (common == 0) {
        if (num_approx == 0 && factor_base == 0)
          return Status::Failure;  // Should not happen
        return Status::Failure;    // Unexpected GCD result
      }

      L = static_cast<size_t>(num_approx / common);
      M = static_cast<size_t>(factor_base / common);

      // Final check for zero factors
      if (L == 0 || M == 0) {
        if (out_rate > 0 && L == 0)
          L = 1;  // Ensure L is at least 1 if output rate > 0
        if (in_rate > 0 && M == 0)
          M = 1;  // Ensure M is at least 1 if input rate > 0
        if (L == 0 || M == 0)
          return Status::Failure;  // Still zero after adjustment? Error.
      }
      return Status::Success;
    }

    template <typename T>
    Status DefaultResamplePlanImpl<T>::design_filter(
        const AbstractBackend* owner,  // Use AbstractBackend*
        const ResampleSpec& spec,
        size_t L,
        size_t M)
    {
      if (!owner) return Status::InvalidArgument;

      // 1. Determine filter specifications
      double normalized_cutoff
          = 1.0 / (2.0 * static_cast<double>(std::max(L, M)));

      // Map quality to transition width / attenuation (Example values)
      // TODO: Refine these mappings based on desired performance
      // characteristics
      double transition_width_factor = 0.1;  // Relative to cutoff
      double stopband_attenuation_db = 60.0;
      if (spec.quality <= 6) {  // Low quality example
        transition_width_factor = 0.25;
        stopband_attenuation_db = 40.0;
      }
      else if (spec.quality <= 12) {  // Medium quality example
        transition_width_factor = 0.1;
        stopband_attenuation_db = 60.0;
      }
      else {  // High quality example
        transition_width_factor = 0.05;
        stopband_attenuation_db = 80.0;
      }
      double transition_width = transition_width_factor * normalized_cutoff;

      // 2. Create a non-templated FIRFilterSpec
      FIRFilterSpec filter_spec;
      filter_spec.type = FilterType::Lowpass;
      filter_spec.order = 0;  // Let design method determine order
      // *** CORRECTED: Assign double values directly ***
      filter_spec.cutoff1 = normalized_cutoff;
      filter_spec.sample_rate = 1.0;  // Normalized design
      filter_spec.transition_width = transition_width;
      filter_spec.stopband_attenuation_db = stopband_attenuation_db;
      filter_spec.window = spec.window;  // Use WindowSpec from ResampleSpec

      // 3. Call the owner's filter design method via AbstractBackend interface
      //    Need to call the correct templated version (_f32 or _f64) based on T
      OmniExpected<std::vector<T>> coeffs_expected;
      if constexpr (std::is_same_v<T, float>) {
        coeffs_expected = owner->design_fir_filter_f32(
            filter_spec);  // Pass non-templated spec
      }
      else if constexpr (std::is_same_v<T, double>) {
        coeffs_expected = owner->design_fir_filter_f64(
            filter_spec);  // Pass non-templated spec
      }
      else {
        // Should not happen due to ResamplePlan template constraint
        return Status::InvalidArgument;
      }

      if (!coeffs_expected) {
        std::cerr << "Error: Failed to design resampling filter using owner "
                     "backend. Status: "
                  << static_cast<int>(coeffs_expected.error()) << std::endl;
        return coeffs_expected.error();
      }
      if (coeffs_expected.value().empty()) {
        std::cerr
            << "Error: Resampling filter design resulted in empty coefficients."
            << std::endl;
        return Status::Failure;  // Or specific error
      }

      // 4. Store and scale the coefficients
      prototype_filter_coeffs_ = std::move(coeffs_expected.value());
      T scale_factor = static_cast<T>(L);  // Scale by upsampling factor
      for (T& coeff : prototype_filter_coeffs_) {
        coeff *= scale_factor;
      }

      // std::cout << "Designed filter with " << prototype_filter_coeffs_.size()
      // << " taps using window " << static_cast<int>(spec.window.type) <<
      // std::endl; // Debug

      return Status::Success;
    }

    //--------------------------------------------------------------------------
    // Explicit Template Instantiations
    //--------------------------------------------------------------------------
    // Instantiate templates for common types (float, double) to ensure code
    // generation.

    template class DefaultResamplePlanImpl<float>;
    template class DefaultResamplePlanImpl<double>;

  }  // namespace backend
}  // namespace OmniDSP
