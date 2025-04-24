/**
 * @file resample.cpp (default)
 * @brief Implements the Default backend ResamplePlanImpl class using standard
 * C++.
 * @details Provides a portable resampling implementation using polyphase FIR
 * filtering. The FIR filter is designed using the owner OmniDSPImpl instance.
 */

#include <algorithm>  // For std::copy, std::fill
#include <cmath>      // For std::ceil, std::floor, std::min, std::max, std::abs
#include <iostream>   // For debug messages
#include <numeric>    // For std::gcd
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "backend.h"  // Default backend declarations (includes DefaultResamplePlanImpl declaration)

// Define PI if not available from cmath
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
       * @param owner Pointer to the OmniDSPImpl instance creating this plan.
       * @param spec The resampling specification.
       * @throws std::invalid_argument If spec or owner is invalid.
       * @throws std::runtime_error If factor calculation or filter design
       * fails.
       */
      DefaultResamplePlanImpl(
          const OmniDSPImpl* owner, const ResampleSpec& spec);

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
          const OmniDSPImpl* owner,
          const ResampleSpec& spec,
          size_t L,
          size_t M);
    };

    //--------------------------------------------------------------------------
    // DefaultResamplePlanImpl Method Implementations
    //--------------------------------------------------------------------------

    template <typename T>
    DefaultResamplePlanImpl<T>::DefaultResamplePlanImpl(
        const OmniDSPImpl* owner, const ResampleSpec& spec)
        : input_rate_(spec.input_rate), output_rate_(spec.output_rate)
    {
      if (!owner) {
        throw std::invalid_argument(
            "DefaultResamplePlanImpl requires a valid owner OmniDSPImpl "
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

      // 3. Initialize state buffer (size = filter taps - 1)
      size_t num_taps = prototype_filter_coeffs_.size();
      if (num_taps > 0) {
        filter_state_.assign(num_taps - 1, T{0});
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

      std::cout << "Default ResamplePlanImpl created. L=" << upsample_factor_L_
                << ", M=" << downsample_factor_M_ << ", Taps=" << num_taps
                << std::endl;  // Debug
    }

    template <typename T>
    DefaultResamplePlanImpl<T>::~DefaultResamplePlanImpl()
    {
      std::cout << "Default ResamplePlanImpl destroyed." << std::endl;  // Debug
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

      if (output.size() < output_len_required) {
        return Status::SizeMismatch;  // Output buffer too small
      }
      if (input_len == 0) {
        std::fill(output.begin(), output.begin() + output_len_required, T{0});
        return Status::Success;  // Handle empty input
      }

      // --- Polyphase Resampling Implementation (Direct FIR filtering approach)
      // ---
      const size_t L = upsample_factor_L_;
      const size_t M = downsample_factor_M_;
      const size_t num_taps = prototype_filter_coeffs_.size();
      const size_t state_len = filter_state_.size();  // Should be num_taps - 1

      // Combine state and new input into a single buffer for easier processing
      std::vector<T> work_buffer;
      work_buffer.reserve(state_len + input_len);
      work_buffer.insert(
          work_buffer.end(), filter_state_.begin(), filter_state_.end());
      work_buffer.insert(work_buffer.end(), input.begin(), input.end());

      size_t output_idx = 0;
      const size_t num_branch_taps
          = (num_taps + L - 1) / L;  // Taps per polyphase branch

      // Loop to generate output samples
      while (output_idx < output_len_required) {
        // Calculate the index in the *input* buffer corresponding to the
        // *start* of the filter application for the current output sample. This
        // requires knowing how many input samples are consumed per output
        // sample on average (M/L), and considering the current phase offset.
        // input_idx = floor(current_output_time * input_rate)
        //           = floor( (output_idx * output_period) * input_rate )
        //           = floor( (output_idx / output_rate) * input_rate )
        //           = floor( output_idx * input_rate / output_rate )
        //           = floor( output_idx * M / L )
        // However, we need to account for the filter state and phase.

        // Alternative view: Advance time based on phase.
        // Each output sample advances the phase accumulator by M.
        // The index into the input buffer `work_buffer` needed depends on the
        // current phase and the filter tap index.

        // Calculate the index into the combined state+input buffer
        // `work_buffer` corresponding to the *first tap* of the *current*
        // polyphase filter branch.
        size_t work_buffer_base_idx
            = current_phase_ / L;  // Index of the input sample corresponding to
                                   // phase 0 of this filter application

        // Check if we have enough data in work_buffer to apply the filter
        // The last needed sample is at work_buffer_base_idx + num_branch_taps -
        // 1
        if (work_buffer_base_idx + num_branch_taps > work_buffer.size()) {
          // Not enough input samples + state to compute the full output length.
          // This can happen if output_len_required was slightly overestimated
          // or input ended. std::cerr << "Warning: Resampler execute stopping
          // early. Input/State exhausted." << std::endl;
          break;  // Stop producing output samples
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
            // The work buffer index corresponds to input sample index
            // `work_buffer_base_idx + n` shifted by the state length.
            size_t work_idx = work_buffer_base_idx + n;
            // Access coefficient h[phase + n*L] and multiply by input
            // x[floor(t/L)
            // + n]
            sum += work_buffer[work_idx] * prototype_filter_coeffs_[proto_idx];
          }
        }

        output[output_idx] = sum;  // Store the computed output sample
        output_idx++;

        // Advance the current phase by M for the next output sample
        current_phase_ += M;
      }

      // Update the filter state for the next call
      // Determine how many input samples were effectively consumed
      size_t consumed_input_samples
          = current_phase_ / L;  // Base index for the *next* block

      // The state needed for the next block consists of the last `state_len`
      // samples from the portion of `work_buffer` that was used. The relevant
      // part of work_buffer ends at index `consumed_input_samples +
      // num_branch_taps - 1`
      // ?? No. Simpler: The state should be the last `state_len` samples of the
      // *original* input that were involved. Let's find the index in the
      // *original input span* corresponding to the start of the state needed.
      // Total samples processed in work_buffer = consumed_input_samples +
      // (num_taps/L) approx?

      // Easier way: The state is the last `state_len` elements of the
      // `work_buffer` that were *potentially* accessed. The highest index
      // accessed in work_buffer was roughly `work_buffer_base_idx +
      // num_branch_taps - 1` where work_buffer_base_idx was `(current_phase_ -
      // M) / L` at the last step. Let's just copy the *last* `state_len`
      // elements from the `work_buffer`.
      size_t copy_start_idx = (work_buffer.size() >= state_len)
                                  ? (work_buffer.size() - state_len)
                                  : 0;
      size_t num_to_copy = std::min(state_len, work_buffer.size());

      if (state_len > 0 && num_to_copy > 0) {
        std::copy(
            work_buffer.begin() + copy_start_idx,
            work_buffer.begin() + copy_start_idx + num_to_copy,
            filter_state_.begin());
        // Zero pad if fewer than state_len samples were available to copy
        if (num_to_copy < state_len) {
          std::fill(
              filter_state_.begin() + num_to_copy, filter_state_.end(), T{0});
        }
      }

      // Adjust phase for the next block start, wrapping around L
      current_phase_ = current_phase_ % L;

      // Zero out remaining output buffer if we stopped early or user provided
      // larger buffer
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
      if (input_rate_ <= 0.0 || upsample_factor_L_ == 0) return 0;
      // Estimate based on ratio. Add 1 for potential ceiling effects.
      // A more precise estimate might consider filter delay, but ceil is
      // usually safe. Need to account for the state and phase. Effective input
      // length = input_length + state_len Effective output samples =
      // floor((input_length * L
      // + current_phase_) / M) ? Needs refinement.
      double ratio = output_rate_ / input_rate_;
      // Add filter length contribution heuristic? (num_taps / L) / M ?
      size_t filter_contribution
          = (prototype_filter_coeffs_.size() + L - 1)
            / L;  // Rough estimate of delay in output samples
      size_t estimated_len = static_cast<size_t>(std::ceil(
                                 static_cast<double>(input_length) * ratio))
                             + filter_contribution;

      // Alternative simpler estimate:
      // size_t estimated_len =
      // static_cast<size_t>(std::ceil(static_cast<double>(input_length) *
      // ratio)) + 1; // Add 1 for safety

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
      long long num_approx
          = static_cast<long long>(out_rate * factor_base / in_rate + 0.5);
      if (num_approx == 0) num_approx = 1;  // Avoid zero upsample factor

      // Find greatest common divisor (GCD) - requires <numeric> in C++17
      long long common = std::gcd(num_approx, factor_base);
      if (common == 0)
        return Status::Failure;  // Should not happen if rates > 0

      L = static_cast<size_t>(num_approx / common);
      M = static_cast<size_t>(factor_base / common);

      if (L == 0 || M == 0) {
        return Status::Failure;  // Calculation resulted in zero factor
      }
      return Status::Success;
    }

    template <typename T>
    Status DefaultResamplePlanImpl<T>::design_filter(
        const OmniDSPImpl* owner, const ResampleSpec& spec, size_t L, size_t M)
    {
      // Use the base class implementation provided in backend.h (via previous
      // step) This requires the implementation to be available, typically by
      // defining it in a shared .cpp file or ensuring backend.h includes its
      // definition source. Assuming the definition from the previous step is
      // accessible:
      return ResamplePlanImpl<T>::design_prototype_filter(
          owner, spec, prototype_filter_coeffs_, L, M);
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
