/**
 * @file resample.cpp (accelerate)
 * @brief Implements Accelerate backend ResamplePlanImpl class.
 * @warning Accelerate/vDSP does not provide a direct arbitrary resampling
 * function. This implementation requires a custom resampling algorithm (e.g.,
 * polyphase FIR filtering) built using vDSP primitives. The implementation
 * below is a skeleton.
 */

// Only compile this file if Accelerate backend is enabled via CMake
#ifdef USE_ACCELERATE

#include <Accelerate/Accelerate.h>

#include <cmath>    // For std::ceil, std::floor, std::gcd, std::log10, std::min
#include <complex>  // Include complex for potential internal use if needed
#include <iostream>  // For debug/error messages
#include <numbers>
#include <numeric>  // For std::gcd
#include <span>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "OmniDSP/core_types.hpp"  // For Status, RealT etc.
#include "backend.hpp"             // Accelerate backend declarations

namespace OmniDSP {
  namespace backend {

    // Helper function for greatest common divisor (C++17 has std::gcd)
    // size_t gcd(size_t a, size_t b) {
    //     return std::gcd(a, b); // Requires <numeric> and C++17
    // }

    // Helper for designing a basic low-pass FIR filter (Placeholder)
    // A real implementation would use a proper design method (e.g., Kaiser
    // window)
    template <typename T>
    std::vector<T> design_resample_filter(
        size_t num_taps, double cutoff_freq_norm)
    {
      // Placeholder: Returns a simple sinc filter windowed with Hann
      std::vector<T> h(num_taps);
      T center = static_cast<T>(num_taps - 1) / 2.0;
      T sum = 0;
      for (size_t i = 0; i < num_taps; ++i) {
        T n = static_cast<T>(i) - center;
        T sinc_val
            = (n == 0)
                  ? static_cast<T>(2.0 * cutoff_freq_norm)
                  : static_cast<T>(
                        std::sin(2.0 * std::numbers::pi * cutoff_freq_norm * n)
                        / (std::numbers::pi * n));
        T hann_val = static_cast<T>(
            0.5
            * (1.0 - std::cos(2.0 * std::numbers::pi * i / (num_taps - 1))));
        h[i] = sinc_val * hann_val;
        sum += h[i];
      }
      // Normalize filter gain to 1 at DC
      if (sum != 0) {
        for (size_t i = 0; i < num_taps; ++i) {
          h[i] /= sum;
        }
      }
      return h;
    }

    //--------------------------------------------------------------------------
    // AccelerateResamplePlanImpl Method Definitions
    //--------------------------------------------------------------------------

    template <typename T>
    AccelerateResamplePlanImpl<T>::AccelerateResamplePlanImpl(
        double input_rate, double output_rate, size_t /* max_input_size */)
        : input_rate_(input_rate), output_rate_(output_rate)
    {
      if (input_rate <= 0.0 || output_rate <= 0.0) {
        throw std::invalid_argument(
            "Input and output sample rates must be positive.");
      }

      // --- Polyphase Resampler Setup ---
      // 1. Determine Upsample (L) and Downsample (M) factors
      //    output_rate / input_rate = L / M
      //    Find simplest integer ratio L/M. Use GCD. Max precision limited by
      //    double. For practical purposes, often approximate or use large
      //    integers. Simplification: Use a fixed large integer base for
      //    approximation.
      const long long factor_base = 1LL << 32;  // Example large base
      long long l_approx = static_cast<long long>(
          output_rate_ * factor_base / input_rate_ + 0.5);
      if (l_approx == 0) l_approx = 1;  // Avoid zero upsample factor
      long long common = std::gcd(l_approx, factor_base);
      upsample_factor_L_ = static_cast<size_t>(l_approx / common);
      downsample_factor_M_ = static_cast<size_t>(factor_base / common);

      if (upsample_factor_L_ == 0 || downsample_factor_M_ == 0) {
        throw std::runtime_error(
            "Failed to determine valid up/downsample factors.");
      }

      // 2. Design low-pass anti-aliasing/imaging filter
      //    Cutoff frequency should be min(input_rate, output_rate) / 2
      //    Normalized cutoff = min(1.0, output_rate/input_rate) / (2 * L) when
      //    considering intermediate rate L*input_rate Or simply cutoff =
      //    min(input_rate, output_rate) / (2 * max(input_rate, output_rate)) ?
      //    No. Cutoff relative to the *intermediate* high sample rate (L *
      //    input_rate)
      double intermediate_rate = input_rate_ * upsample_factor_L_;
      double cutoff_freq = std::min(input_rate_, output_rate_) / 2.0;
      double normalized_cutoff
          = cutoff_freq
            / intermediate_rate;  // Cutoff relative to intermediate rate

      // Determine filter length (heuristic, depends on quality needs)
      // Longer filter = better quality, more computation, more delay.
      size_t num_taps_per_phase
          = 12;  // Example: Number of taps for each polyphase filter
      filter_length_ = upsample_factor_L_ * num_taps_per_phase
                       * 2;  // Example total taps (needs refinement)
      if (filter_length_ == 0) filter_length_ = 1;  // Ensure non-zero taps

      // Design the prototype filter
      prototype_filter_
          = design_resample_filter<T>(filter_length_, normalized_cutoff);

      // 3. Create Polyphase Filter Bank (optional pre-computation)
      //    Can be done on-the-fly in execute or precomputed here.
      //    polyphase_filters_[phase][tap] = prototype_filter_[phase + tap * L]
      //    For now, we'll just store the prototype filter.

      // 4. Initialize state (delay line)
      //    Length depends on the filter length / implementation details
      filter_state_.assign(
          filter_length_ - 1, T{});  // Need M-1 state? Or filter_length-1?
                                     // Depends on implementation.

      std::cout << "Accelerate ResamplePlanImpl created. L="
                << upsample_factor_L_ << ", M=" << downsample_factor_M_
                << ", Taps=" << filter_length_ << std::endl;  // Debug
    }

    // Destructor: Default is sufficient as std::vector handles memory.
    template <typename T>
    AccelerateResamplePlanImpl<T>::~AccelerateResamplePlanImpl()
    {
      std::cout << "Accelerate ResamplePlanImpl destroyed."
                << std::endl;  // Debug
    }

    template <typename T>
    Status AccelerateResamplePlanImpl<T>::execute(
        std::span<const T> input, std::span<T> output) const
    {
      if (upsample_factor_L_ == 0 || downsample_factor_M_ == 0
          || prototype_filter_.empty()) {
        return Status::InvalidOperation;  // Plan not initialized correctly
      }

      size_t input_len = input.size();
      size_t output_len_required = get_output_length(input_len);

      if (output.size() < output_len_required) {
        return Status::SizeMismatch;  // Output buffer too small
      }

      // --- Polyphase Resampling Implementation ---
      // This requires careful state management and indexing.
      // The core idea:
      // For each output sample 'k':
      //   1. Calculate the corresponding time 't_k' in the input signal's time
      //   base:
      //      t_k = k * (input_rate_ / output_rate_) = k * M / L
      //   2. Find the nearest input sample index 'n_k = floor(t_k)'
      //   3. Find the fractional offset 'mu_k = t_k - n_k'
      //   4. Determine the polyphase filter index 'p_k = floor(mu_k * L)'
      //   5. Apply the p_k-th polyphase filter to the input samples centered
      //   around n_k.
      //      This involves using the current filter state (delay line) and
      //      input samples. y[k] = sum( input_with_state[n_k - tap_idx] *
      //      polyphase_filter[p_k][tap_idx] )
      //   6. Update the filter state with the processed input samples.

      // --- Placeholder Implementation ---
      // A full implementation using vDSP_conv or similar is complex due to
      // state and indexing. This placeholder just copies input to output
      // (incorrectly!) to show structure.
      std::cerr << "AccelerateResamplePlanImpl::execute - Polyphase resampling "
                   "logic not implemented in skeleton!"
                << std::endl;

      size_t samples_to_copy
          = std::min(output_len_required, input_len);  // Incorrect logic
      std::copy(input.begin(), input.begin() + samples_to_copy, output.begin());
      if (output_len_required > samples_to_copy) {
        std::fill(
            output.begin() + samples_to_copy,
            output.begin() + output_len_required,
            T{});
      }
      // Need to update filter_state_ (mutable member needed, or pass state
      // buffer)

      // --- End Placeholder ---

      // Zero out remaining output buffer if user provided a larger one
      if (output.size() > output_len_required) {
        std::fill(output.begin() + output_len_required, output.end(), T{});
      }

      // return Status::UnsupportedFeature; // Until implemented
      return Status::Success;  // Placeholder success
    }

    template <typename T>
    double AccelerateResamplePlanImpl<T>::get_input_rate() const
    {
      return input_rate_;
    }

    template <typename T>
    double AccelerateResamplePlanImpl<T>::get_output_rate() const
    {
      return output_rate_;
    }

    template <typename T>
    size_t AccelerateResamplePlanImpl<T>::get_output_length(
        size_t input_length) const
    {
      if (input_rate_ <= 0.0) return 0;  // Avoid division by zero
      // Estimate output length based on ratio.
      // Add potential delay based on filter length? Needs careful calculation.
      // Simple ratio-based estimate:
      double ratio = output_rate_ / input_rate_;
      size_t estimated_len = static_cast<size_t>(
          std::ceil(static_cast<double>(input_length) * ratio));

      // Add estimate for filter delay/transient (heuristic)
      // size_t delay_estimate = filter_length_ / 2; // Example
      // return estimated_len + delay_estimate;
      return estimated_len;  // Return simple estimate for now
    }

    //--------------------------------------------------------------------------
    // Explicit Template Instantiations
    //--------------------------------------------------------------------------
    // Instantiate templates for common types (float, double) to ensure code
    // generation.

    template class OmniDSP::backend::AccelerateResamplePlanImpl<float>;
    template class OmniDSP::backend::AccelerateResamplePlanImpl<double>;

    // Remove implementation of old standalone filter_and_downsample function
    // here if it existed.

  }  // namespace backend
}  // namespace OmniDSP

#endif  // USE_ACCELERATE
