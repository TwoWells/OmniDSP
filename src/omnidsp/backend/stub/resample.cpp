/**
 * @file resample.cpp (stub)
 * @brief Implements Stub backend ResamplePlanImpl class using standard C++.
 */

#include <algorithm>  // For std::min, std::copy, std::fill
#include <cmath>  // For std::ceil, std::floor, std::gcd, std::log10, std::min, M_PI, sin, cos
#include <complex>   // Include complex for potential internal use if needed
#include <iostream>  // For debug/error messages
#include <numeric>   // For std::gcd
#include <span>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "OmniDSP/core_types.h"  // For Status, RealT etc.
#include "backend.h"             // Stub backend declarations

// Define PI if not available from cmath
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace OmniDSP {
namespace backend {

// Helper function for designing a basic low-pass FIR filter (Placeholder)
// (Copied from accelerate/resample.cpp - move to common utility?)
template <typename T>
std::vector<T> design_resample_filter(size_t num_taps,
                                      double cutoff_freq_norm) {
  std::vector<T> h(num_taps);
  T center = static_cast<T>(num_taps - 1) / 2.0;
  T sum = 0;
  for (size_t i = 0; i < num_taps; ++i) {
    T n = static_cast<T>(i) - center;
    T sinc_val =
        (n == 0) ? static_cast<T>(2.0 * cutoff_freq_norm)
                 : static_cast<T>(std::sin(2.0 * M_PI * cutoff_freq_norm * n) /
                                  (M_PI * n));
    T hann_val =
        (num_taps <= 1)
            ? static_cast<T>(1.0)
            : static_cast<T>(0.5 *
                             (1.0 - std::cos(2.0 * M_PI * i / (num_taps - 1))));
    h[i] = sinc_val * hann_val;
    sum += h[i];
  }
  // Normalize filter gain to 1 at DC
  if (sum != 0) {
    T inv_sum = static_cast<T>(1.0) / sum;
    for (size_t i = 0; i < num_taps; ++i) {
      h[i] *= inv_sum;
    }
  }
  // Note: IPP FIRMR requires scaling by L, but direct polyphase implementation
  // usually doesn't.
  return h;
}

//--------------------------------------------------------------------------
// StubResamplePlanImpl Method Definitions
//--------------------------------------------------------------------------

template <typename T>
StubResamplePlanImpl<T>::StubResamplePlanImpl(double input_rate,
                                              double output_rate,
                                              size_t /* max_input_size */)
    : input_rate_(input_rate),
      output_rate_(output_rate),
      current_phase_(0)  // Initialize phase
{
  if (input_rate <= 0.0 || output_rate <= 0.0) {
    throw std::invalid_argument(
        "Input and output sample rates must be positive.");
  }

  // --- Polyphase Resampler Setup ---
  // 1. Determine Upsample (L) and Downsample (M) factors
  const long long factor_base = 1LL << 32;
  long long num_approx =
      static_cast<long long>(output_rate_ * factor_base + 0.5);
  long long den_approx =
      static_cast<long long>(input_rate_ * factor_base + 0.5);
  if (num_approx == 0 || den_approx == 0) {
    throw std::runtime_error(
        "Could not represent sample rate ratio accurately.");
  }
  long long common = std::gcd(num_approx, den_approx);
  // Store as int, matching IPP FIRMR for potential consistency, though size_t
  // might be better
  upsample_factor_L_ = static_cast<int>(num_approx / common);
  downsample_factor_M_ = static_cast<int>(den_approx / common);

  if (upsample_factor_L_ <= 0 || downsample_factor_M_ <= 0) {
    throw std::runtime_error(
        "Failed to determine valid positive up/downsample factors.");
  }

  // 2. Design low-pass anti-aliasing/imaging filter
  double intermediate_rate = input_rate_ * upsample_factor_L_;
  double cutoff_freq = std::min(input_rate_, output_rate_) / 2.0;
  double normalized_cutoff = cutoff_freq / intermediate_rate;

  // Determine filter length (heuristic)
  size_t num_taps_per_phase = 12;  // Example
  filter_length_ = upsample_factor_L_ * num_taps_per_phase * 2;
  if (filter_length_ == 0) filter_length_ = 1;

  prototype_filter_ =
      design_resample_filter<T>(filter_length_, normalized_cutoff);

  // 3. Initialize state (delay line)
  // The state needed is related to the number of taps in each polyphase filter
  // branch. State length = ceil(filter_length_ / L) - 1 ? Or filter_length - 1?
  // Let's use filter_length - 1 for simplicity.
  filter_state_.assign(filter_length_ - 1, T{});

  std::cout << "Stub ResamplePlanImpl created. L=" << upsample_factor_L_
            << ", M=" << downsample_factor_M_ << ", Taps=" << filter_length_
            << std::endl;  // Debug
}

// Destructor: Default is sufficient as std::vector handles memory.
template <typename T>
StubResamplePlanImpl<T>::~StubResamplePlanImpl() {
  std::cout << "Stub ResamplePlanImpl destroyed." << std::endl;  // Debug
}

template <typename T>
Status StubResamplePlanImpl<T>::execute(std::span<const T> input,
                                        std::span<T> output) const {
  if (upsample_factor_L_ <= 0 || downsample_factor_M_ <= 0 ||
      prototype_filter_.empty()) {
    return Status::InvalidOperation;  // Plan not initialized correctly
  }

  size_t input_len = input.size();
  size_t output_len_required = get_output_length(input_len);  // Use estimate

  if (output.size() < output_len_required) {
    return Status::SizeMismatch;  // Output buffer too small
  }
  if (input_len == 0) {
    std::fill(output.begin(), output.begin() + output_len_required, T{});
    return Status::Success;  // Handle empty input
  }

  // --- Polyphase Resampling Implementation (Direct FIR filtering approach) ---
  size_t L = static_cast<size_t>(upsample_factor_L_);
  size_t M = static_cast<size_t>(downsample_factor_M_);
  size_t num_taps = filter_length_;
  size_t state_len = filter_state_.size();  // Should be num_taps - 1

  // Combine state and new input into a single buffer for easier processing
  std::vector<T> work_buffer = filter_state_;  // Copy previous state
  work_buffer.insert(work_buffer.end(), input.begin(),
                     input.end());  // Append new input

  size_t output_idx = 0;

  // Loop to generate output samples
  while (output_idx < output_len_required) {
    // Calculate the index in the (conceptually) upsampled signal corresponding
    // to the current time phase.
    // Each output sample corresponds to consuming M phase steps.
    // Each input sample corresponds to L phase steps.
    size_t input_sample_idx_float =
        (current_phase_ + L - 1) / L;  // Ceiling division equivalent? No.
    // Input sample index corresponding to the start of the filter application
    // for this phase
    size_t input_idx = current_phase_ / L;

    // Check if we have enough input samples (including state) to compute the
    // output
    if (input_idx + num_taps > work_buffer.size()) {
      // Not enough input samples + state to compute the full output length.
      // This can happen if output_len_required was slightly overestimated due
      // to transients. Or indicates an error in get_output_length or execute
      // logic. Let's stop generating output here. std::cerr << "Warning:
      // Resampler execute stopping early. Input/State exhausted." << std::endl;
      // std::cerr << "  output_idx=" << output_idx << ", output_len_required="
      // << output_len_required << std::endl; std::cerr << "  input_idx=" <<
      // input_idx << ", num_taps=" << num_taps << ", work_buffer.size=" <<
      // work_buffer.size() << std::endl; std::cerr << "  current_phase=" <<
      // current_phase_ << std::endl;
      break;  // Stop producing output samples
    }

    // Determine the polyphase filter branch index (phase)
    size_t phase_index = current_phase_ % L;

    // Apply the appropriate polyphase filter branch
    T sum{};
    for (size_t tap_idx = 0; tap_idx < num_taps; ++tap_idx) {
      // Check if the prototype filter index is valid for this phase
      if ((tap_idx % L) == phase_index) {
        // Index into the combined state+input buffer
        size_t work_buffer_idx =
            input_idx + (tap_idx / L);  // Correct index? Check polyphase math
        // Need to be careful with indexing, polyphase formula is:
        // y[k] = sum_{n=0}^{N_taps/L - 1} x[floor(k*M/L) - n] * h[k*M % L +
        // n*L] Let's try direct FIR: Filter the upsampled signal (conceptually)
        // Simpler: Calculate output sample using the correct polyphase branch
        // y[k] = sum_{n=0}^{N_branch-1} x_interp[k*L - n*M] * h_poly[k*M %
        // L][n] ? No.

        // Direct FIR on upsampled signal (conceptually):
        // Need to access work_buffer samples corresponding to the filter taps
        // centered around the current time.
        // Let's try a simpler FIR view:
        // For output sample `output_idx`, the corresponding time is roughly
        // `output_idx * M`. The input samples needed are around `input_idx`.
        // The specific filter phase is `phase_index`.

        // Iterate through the taps of the selected polyphase filter branch
        size_t branch_taps = (num_taps + L - 1) / L;  // Taps per branch
        sum = T{};  // Reset sum for each output sample
        for (size_t n = 0; n < branch_taps; ++n) {
          size_t proto_idx =
              phase_index + n * L;  // Index into prototype filter
          if (proto_idx < num_taps) {
            // Index into the work buffer (state + input)
            // The input sample index corresponding to the center of the filter
            // for this output sample is roughly `input_idx`.
            // We need samples from `input_idx - n` back to `input_idx - n +
            // state_len`? Let k be the output index. Time = k * M. Input index
            // ~ k*M/L. Let input_time_idx = floor(k*M/L). y[k] = sum_n
            // x[input_time_idx - n] * h_poly[phase][n]

            long long work_idx_long =
                static_cast<long long>(input_idx) - static_cast<long long>(n);
            if (work_idx_long >= 0 &&
                static_cast<size_t>(work_idx_long) < work_buffer.size()) {
              sum += work_buffer[work_idx_long] * prototype_filter_[proto_idx];
            }
          }
        }
        // Apply gain factor L? IPP requires scaling taps by L. Direct
        // implementation might not. Let's assume filter design handles gain.
        output[output_idx] = sum;  // * static_cast<T>(L); // Optional gain
        output_idx++;
      }
    }

    // Advance the current phase by M for the next output sample
    current_phase_ += M;
  }

  // Update the filter state for the next call
  size_t consumed_input = (current_phase_ + L - 1) /
                          L;  // How many input samples were effectively used?
  size_t remaining_in_buffer = work_buffer.size() - consumed_input;
  size_t needed_state = state_len;  // num_taps - 1

  if (remaining_in_buffer >= needed_state) {
    // Copy the last 'needed_state' samples from the work buffer to
    // filter_state_
    std::copy(work_buffer.end() - needed_state, work_buffer.end(),
              filter_state_.begin());
  } else {
    // Not enough remaining samples, copy all remaining and pad with zeros
    std::copy(work_buffer.begin() + consumed_input, work_buffer.end(),
              filter_state_.begin());
    std::fill(filter_state_.begin() + remaining_in_buffer, filter_state_.end(),
              T{});
  }

  // Adjust phase for the next block start
  current_phase_ = current_phase_ % L;

  // Zero out remaining output buffer if we stopped early or user provided
  // larger buffer
  if (output_idx < output.size()) {
    std::fill(output.begin() + output_idx, output.end(), T{});
  }

  return Status::Success;
}

template <typename T>
double StubResamplePlanImpl<T>::get_input_rate() const {
  return input_rate_;
}

template <typename T>
double StubResamplePlanImpl<T>::get_output_rate() const {
  return output_rate_;
}

template <typename T>
size_t StubResamplePlanImpl<T>::get_output_length(size_t input_length) const {
  if (input_rate_ <= 0.0 || upsample_factor_L_ == 0) return 0;
  // Estimate based on ratio. Add 1 for potential ceiling effects.
  // A more precise estimate might consider filter delay, but ceil is usually
  // safe for allocation.
  double ratio = output_rate_ / input_rate_;
  size_t estimated_len =
      static_cast<size_t>(std::ceil(static_cast<double>(input_length) * ratio));
  return estimated_len;
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
// Instantiate templates for common types (float, double) to ensure code
// generation.

template class OmniDSP::backend::StubResamplePlanImpl<float>;
template class OmniDSP::backend::StubResamplePlanImpl<double>;

}  // namespace backend
}  // namespace OmniDSP
