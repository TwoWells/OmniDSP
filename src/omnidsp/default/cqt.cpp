/**
 * @file cqt.cpp (Default)
 * @brief Implements the Default backend CQTPlanImpl class using a recursive
 * downsampling strategy.
 * @details Provides a portable CQT implementation using standard C++ and
 * leveraging sub-plans (FFTPlan, ResamplePlan) created via the owner
 * Backend instance, ensuring backend consistency. Processes the highest
 * octave via FFT convolution and recursively downsamples the signal to process
 * lower octaves.
 */

#include "cqt.hpp"  // Includes DefaultCQTPlanImpl declaration

#include <OmniDSP/cqt.hpp>  // Public CQTPlan interface
#include <algorithm>  // For std::transform, std::max_element, std::fill, std::min
#include <cmath>
#include <complex>
#include <functional>  // For std::function (recursive lambda)
#include <iostream>    // For debug messages
#include <limits>      // For std::numeric_limits
#include <memory>      // For std::unique_ptr
#include <numbers>
#include <numeric>  // For std::iota, std::accumulate
#include <span>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "../interface/backend.hpp"  // For AbstractBackend
#include "OmniDSP/core_types.hpp"  // Core types (includes Detail::GetComplexT)
#include "OmniDSP/fft.hpp"         // Needed for internal FFTPlan interface
#include "OmniDSP/resample.hpp"  // Needed for internal ResamplePlan interface and ResampleSpec
#include "OmniDSP/window.hpp"  // Needed for non-templated WindowSpec and window generation

namespace OmniDSP::Default {

  // Helper function to calculate the next power of two
  inline size_t next_power_of_two(size_t n)
  {
    if (n == 0) return 1;  // Smallest valid FFT size is often 1 or 2
    // Handle potential overflow if n is already close to max size_t power of 2
    // Use unsigned comparison for safety
    if (n > (std::numeric_limits<size_t>::max() / 2U)) {
      // Check if n is already the largest power of 2 representable
      if ((n > 0) && ((n & (n - 1)) == 0)) return n;
      return std::numeric_limits<size_t>::max();  // Indicate overflow or max
                                                  // possible size
    }

    size_t power = 1;
    while (power < n) {
      size_t next_power = power << 1;
      // Check for overflow during shift (if next_power becomes 0 or smaller
      // than power)
      if (next_power == 0 || next_power < power) {
        return std::numeric_limits<size_t>::max();
      }
      power = next_power;
    }
    return power;
  }

  //--------------------------------------------------------------------------
  // DefaultCQTPlanImpl Method Implementations
  //--------------------------------------------------------------------------

  template <typename T>
  DefaultCQTPlanImpl<T>::DefaultCQTPlanImpl(
      const Abstract::Backend* owner,  // Use Backend
      Real sample_rate,                // Use Real alias
      Real min_freq,                   // Use Real alias
      Real max_freq,                   // Use Real alias
      int bins_per_octave,
      const WindowSpec& window_spec)
      : owner_(owner),
        sample_rate_(sample_rate),
        min_freq_(min_freq),
        max_freq_(max_freq),
        bins_per_octave_(bins_per_octave),
        window_spec_(window_spec)
  {
    if (!owner_) {
      throw std::invalid_argument(
          "DefaultCQTPlanImpl requires a valid owner Backend pointer.");
    }
    // Basic parameter validation
    if (sample_rate <= 0 || min_freq <= 0 || max_freq <= 0
        || bins_per_octave <= 0 || min_freq >= max_freq
        || max_freq
               >= sample_rate / static_cast<Real>(2.0)) {  // Use Real alias
      throw std::invalid_argument("Invalid CQT parameters provided.");
    }
    if (!window_spec_.validate()) {
      throw std::invalid_argument("Invalid WindowSpec provided for CQT.");
    }

    // std::cout << "Default CQTPlanImpl Constructor (Recursive): Setting up
    // CQT..." << std::endl; // Debug

    // --- 1. Calculate Global CQT Parameters ---
    quality_factor_ = static_cast<Real>(1.0)
                      / (std::pow(
                             static_cast<Real>(2.0),
                             static_cast<Real>(1.0) / bins_per_octave_)
                         - static_cast<Real>(1.0));
    // Calculate total number of bins based on the full frequency range
    num_bins_ = static_cast<size_t>(
        std::ceil(bins_per_octave_ * std::log2(max_freq_ / min_freq_)));
    if (num_bins_ == 0) {
      throw std::invalid_argument("CQT parameters result in zero bins.");
    }

    center_frequencies_.resize(num_bins_);  // Use correct member name
    kernel_lengths_.resize(num_bins_);      // Use correct member name
    Real freq_ratio = std::pow(             // Use Real alias
        static_cast<Real>(2.0),
        static_cast<Real>(1.0) / bins_per_octave_);
    size_t max_kernel_len_overall = 0;  // Max length across all bins

    for (size_t k = 0; k < num_bins_; ++k) {
      center_frequencies_[k]
          = min_freq_
            * std::pow(freq_ratio, static_cast<Real>(k));  // Use Real alias
      // Check if frequency exceeds Nyquist limit for the *original* sample
      // rate
      if (center_frequencies_[k]
          >= sample_rate_ / static_cast<Real>(2.0)) {  // Use Real alias
        num_bins_ = k;  // Adjust total number of bins
        center_frequencies_.resize(num_bins_);
        kernel_lengths_.resize(num_bins_);
        if (num_bins_ == 0)
          throw std::runtime_error("No valid CQT bins below Nyquist.");
        break;  // Stop calculating bins
      }
      // Calculate kernel length based on Q factor and frequency
      kernel_lengths_[k] = static_cast<size_t>(
          std::ceil(sample_rate_ * quality_factor_ / center_frequencies_[k]));
      if (kernel_lengths_[k] == 0)
        kernel_lengths_[k] = 1;  // Ensure minimum length of 1
      if (kernel_lengths_[k] > max_kernel_len_overall) {
        max_kernel_len_overall = kernel_lengths_[k];
      }
    }

    // FFT length based on the longest kernel overall
    fft_length_
        = next_power_of_two(max_kernel_len_overall);  // Use correct member name
    if (fft_length_ == std::numeric_limits<size_t>::max()) {
      throw std::runtime_error("Required FFT length for CQT exceeds limits.");
    }

    // Hop length calculation: Base it on the lowest frequency kernel length.
    if (kernel_lengths_.empty()) {  // Handle case where num_bins_ became 0
      throw std::runtime_error(
          "No valid CQT bins found after frequency check.");
    }
    hop_length_ = std::max(
        static_cast<size_t>(1),
        kernel_lengths_[0] / 4);  // Use correct member name
    // hop_length_ = next_power_of_two(hop_length_); // Optional: force power of
    // 2

    //   std::cout << "  Num Bins (Total): " << num_bins_
    //             << ", Q: " << quality_factor_
    //             << ", Max Kernel Len: " << max_kernel_len_overall
    //             << ", FFT Len: " << fft_length_ << ", Hop Len: " <<
    //             hop_length_
    //             << std::endl; // Debug

    // --- 2. Create Sub-Plans (using owner_) ---
    // FFT Plan (for the highest octave, using the overall max FFT length)
    // Use Complex alias here for clarity, assuming FFTPlan uses complex type
    OmniExpected<std::unique_ptr<FFTPlan<Complex>>> fft_plan_result;
    if constexpr (std::is_same_v<T, float>) {
      // Assuming create_fft_plan_c32 expects size_t and returns FFTPlan<C32>
      // which is FFTPlan<Complex>
      fft_plan_result = owner_->create_fft_plan_c32(fft_length_);
    }
    else {
      // Assuming create_fft_plan_c64 expects size_t and returns FFTPlan<C64>
      // which is FFTPlan<Complex>
      fft_plan_result = owner_->create_fft_plan_c64(fft_length_);
    }

    if (!fft_plan_result) {
      throw std::runtime_error(
          "Failed to create internal FFTPlan for CQT: "
          + std::string(get_status_string(fft_plan_result.error())));
    }
    fft_plan_ = std::move(fft_plan_result.value());  // Use correct member name
    //   std::cout << "  Internal FFT Plan created via owner backend (Length: "
    //             << fft_length_ << ")." << std::endl; // Debug

    // Resample Plan (Factor 2 Downsampling)
    ResampleSpec resample_spec;
    resample_spec.input_rate = sample_rate_;
    resample_spec.output_rate = sample_rate_ / 2.0;
    resample_spec.quality = 12;
    resample_spec.window = window_spec_;

    // Use Real alias here, assuming ResamplePlan uses real type
    OmniExpected<std::unique_ptr<ResamplePlan<Real>>> resample_plan_result;
    if constexpr (std::is_same_v<T, float>) {
      // Assuming create_resample_plan_f32 expects ResampleSpec and returns
      // ResamplePlan<F32> which is ResamplePlan<Real>
      resample_plan_result = owner_->create_resample_plan_f32(resample_spec);
    }
    else {
      // Assuming create_resample_plan_f64 expects ResampleSpec and returns
      // ResamplePlan<F64> which is ResamplePlan<Real>
      resample_plan_result = owner_->create_resample_plan_f64(resample_spec);
    }

    if (!resample_plan_result) {
      throw std::runtime_error(
          "Failed to create internal ResamplePlan for CQT: "
          + std::string(get_status_string(resample_plan_result.error())));
    }
    resample_plan_
        = std::move(resample_plan_result.value());  // Use correct member name
    //   std::cout << "  Internal Resample Plan (Factor 2) created via owner
    //   backend." << std::endl; // Debug

    // --- 3. Pre-compute All CQT Kernels (FFT'd) ---
    kernels_fft_.resize(num_bins_);  // Use correct member name
    //   std::cout << "  Generating All CQT Kernels..." << std::endl; // Debug

    for (size_t k = 0; k < num_bins_; ++k) {
      size_t N_k = kernel_lengths_[k];  // Use correct member name
      if (N_k == 0) continue;

      // Generate window using the owner backend's window function
      std::vector<Real> window_coeffs(N_k);  // Use Real alias
      Status window_status;
      if constexpr (std::is_same_v<T, float>) {
        // Assuming generate_window_f32 expects WindowSpec and span<F32> which
        // is span<Real>
        window_status
            = owner_->generate_window_f32(window_spec_, window_coeffs);
      }
      else {
        // Assuming generate_window_f64 expects WindowSpec and span<F64> which
        // is span<Real>
        window_status
            = owner_->generate_window_f64(window_spec_, window_coeffs);
      }

      if (window_status != Status::Success) {
        throw std::runtime_error(
            "Failed to generate window for CQT kernel " + std::to_string(k)
            + ": " + std::string(get_status_string(window_status)));
      }

      // Generate temporal kernel
      std::vector<Complex> temporal_kernel(N_k);  // Use Complex alias
      Real freq_k
          = center_frequencies_[k];  // Use Real alias, use correct member name
      Real norm_factor               // Use Real alias
          = static_cast<Real>(1.0) / static_cast<Real>(N_k);

      for (size_t n = 0; n < N_k; ++n) {
        Real time_arg  // Use Real alias
            = static_cast<Real>(n) / sample_rate_;
        Real angle = static_cast<Real>(
                         2.0 * std::numbers::pi_v<double>)  // Use Real alias
                     * freq_k * time_arg;
        Complex complex_exp
            = {std::cos(angle), std::sin(angle)};  // Use Complex alias
        temporal_kernel[n] = window_coeffs[n] * complex_exp * norm_factor;
      }

      // Zero-pad temporal kernel to fft_length_
      temporal_kernel.resize(
          fft_length_, {0.0, 0.0});  // Use correct member name

      // FFT the padded kernel and store the conjugate
      kernels_fft_[k].resize(fft_length_);  // Use correct member name
      // Assuming fft_plan_->fft takes span<Complex> input and output
      Status status = fft_plan_->fft(
          temporal_kernel, kernels_fft_[k]);  // Use correct member name
      if (status != Status::Success) {
        throw std::runtime_error(
            "FFT failed for CQT kernel " + std::to_string(k));
      }
      // Conjugate the kernel FFT
      std::transform(
          kernels_fft_[k].begin(),  // Use correct member name
          kernels_fft_[k].end(),
          kernels_fft_[k].begin(),
          [](const Complex& c) { return std::conj(c); });  // Use Complex alias
    }
    //   std::cout << "  All CQT Kernels generated and FFT'd." << std::endl; //
    //   Debug std::cout << "Default CQTPlanImpl Constructor (Recursive): Setup
    //   complete." << std::endl; // Debug
  }

  template <typename T>
  DefaultCQTPlanImpl<T>::~DefaultCQTPlanImpl()
  {
    //   std::cout << "Default CQTPlanImpl destroyed." << std::endl; // Debug
  }

  // --- Definition using the aliases 'Real' and 'Complex' from the class scope
  // ---
  template <typename T>
  Status DefaultCQTPlanImpl<T>::execute(
      std::span<const Real> input, std::span<Complex> output) const
  {
    // *** Use this-> consistently for member access ***
    if (!this->owner_ || !this->fft_plan_ || !this->resample_plan_) {
      return Status::InvalidOperation;
    }

    size_t n_frames_total = this->get_num_output_frames(input.size());
    size_t n_bins_total = this->get_num_bins();
    size_t expected_output_size = n_bins_total * n_frames_total;

    if (output.size() < expected_output_size) {
      std::cerr << "CQT execute error: Output buffer size (" << output.size()
                << ") is smaller than required (" << expected_output_size
                << ")." << std::endl;
      return Status::SizeMismatch;
    }
    std::fill(
        output.begin(),
        output.begin() + expected_output_size,
        Complex{0, 0});  // Use Complex alias

    if (input.empty() || n_frames_total == 0 || n_bins_total == 0) {
      return Status::Success;
    }

    // --- Recursive Execution ---
    // Define the recursive lambda. Use Real and Complex aliases for types.
    // Declare the std::function object that will hold the lambda
    std::function<Status(std::span<const Real>, Real, size_t)>
        _execute_recursive;

    // *** Assign the lambda to the std::function object ***
    // Capture necessary variables including _execute_recursive by reference
    _execute_recursive
        = [this, n_frames_total, n_bins_total, &output, &_execute_recursive](
              std::span<const Real> current_input,
              Real current_sr,
              size_t /*bin_offset - unused currently*/) -> Status
    {
      // *** Use this-> consistently for member access ***
      Real current_nyquist
          = current_sr / static_cast<Real>(2.0);  // Use Real alias
      size_t start_bin = 0;
      size_t end_bin = 0;

      // Find bins for the octave below current_nyquist / 2
      while (end_bin < this->num_bins_
             && this->center_frequencies_[end_bin]
                    < current_nyquist
                          / static_cast<Real>(2.0)) {  // Use Real alias
        end_bin++;
      }
      start_bin = end_bin;  // Bins for this octave start here

      // Find bins for the current octave (up to current_nyquist)
      while (end_bin < this->num_bins_
             && this->center_frequencies_[end_bin] < current_nyquist) {
        end_bin++;
      }

      size_t num_bins_this_octave = end_bin - start_bin;

      if (num_bins_this_octave > 0) {
        // Calculate hop length scaled for the current sample rate
        size_t current_hop_length = static_cast<size_t>(std::round(
            static_cast<double>(this->hop_length_)
            * (current_sr / this->sample_rate_)));
        if (current_hop_length == 0) current_hop_length = 1;

        // Calculate number of frames for this level, ensure consistency
        size_t current_n_frames = 0;
        if (!current_input.empty() && current_hop_length > 0) {
          current_n_frames = (current_input.size() + current_hop_length - 1)
                             / current_hop_length;
        }
        if (current_n_frames == 0 && !current_input.empty())
          current_n_frames = 1;

        // Use n_frames_total captured by the outer scope
        if (current_n_frames != n_frames_total) {
          // std::cerr << "Warning: Frame count mismatch in CQT recursion ("
          //           << current_n_frames << " vs " << n_frames_total << ")"
          //           << std::endl; // Debug
          current_n_frames = n_frames_total;  // Force consistency
        }

        // Buffers for FFT processing
        std::vector<Complex> frame_buffer(
            this->fft_length_);                             // Use Complex alias
        std::vector<Complex> frame_fft(this->fft_length_);  // Use Complex alias

        for (size_t i = 0; i < current_n_frames; ++i) {
          size_t frame_start = i * current_hop_length;
          size_t frame_end
              = std::min(frame_start + this->fft_length_, current_input.size());
          size_t current_frame_len = (frame_start < current_input.size())
                                         ? (frame_end - frame_start)
                                         : 0;

          // Prepare frame buffer (real input -> complex buffer)
          std::fill(
              frame_buffer.begin(),
              frame_buffer.end(),
              Complex{0, 0});  // Use Complex alias
          for (size_t n = 0; n < current_frame_len; ++n) {
            frame_buffer[n] = Complex{
                current_input[frame_start + n], 0.0};  // Use Complex alias
          }

          // Perform FFT
          // Assuming fft_plan_->fft takes span<Complex> input and output
          Status fft_status = this->fft_plan_->fft(frame_buffer, frame_fft);
          if (fft_status != Status::Success) return fft_status;

          // Calculate CQT coefficients for this frame and octave
          for (size_t k_idx = 0; k_idx < num_bins_this_octave; ++k_idx) {
            size_t k = start_bin + k_idx;  // Global bin index
            Complex cqt_coeff = {0, 0};    // Use Complex alias
            // Perform spectral convolution (element-wise product and sum)
            for (size_t f = 0; f < this->fft_length_; ++f) {
              cqt_coeff += frame_fft[f]
                           * this->kernels_fft_[k][f];  // kernels_fft_ already
                                                        // conjugated
            }
            // Use n_bins_total captured from outer scope
            size_t output_idx = i * n_bins_total + k;

            if (output_idx < output.size()) {
              // Normalize by FFT length
              // Use fft_length_ captured from outer scope
              output[output_idx]
                  = cqt_coeff
                    / static_cast<Real>(this->fft_length_);  // Use Real alias
            }
            else {
              std::cerr << "Warning: Output index " << output_idx
                        << " out of bounds (" << output.size()
                        << ") during CQT calculation." << std::endl;
              // Decide if this should be a fatal error or just a warning
              // return Status::SizeMismatch; // Make it fatal
            }
          }
        }
      }  // End if (num_bins_this_octave > 0)

      // --- Recursive Step ---
      size_t lowest_bin_processed = start_bin;
      if (lowest_bin_processed > 0) {  // If there are lower octaves to process
        // Use resample_plan_ captured from outer scope
        this->resample_plan_->reset();  // Reset resampler state
        // Estimate downsampled length (may not be exact)
        size_t downsampled_len_est
            = this->resample_plan_->get_output_length(current_input.size());
        std::vector<Real> downsampled_signal(
            downsampled_len_est);  // Use Real alias
        std::span<Real> downsampled_span
            = downsampled_signal;  // Use Real alias

        // Perform resampling
        // Assuming resample_plan_->execute takes span<const Real> and
        // span<Real>
        Status resample_status
            = this->resample_plan_->execute(current_input, downsampled_span);
        if (resample_status != Status::Success) {
          std::cerr << "CQT execute error: Resampling failed." << std::endl;
          return resample_status;
        }
        // Adjust size if estimate was wrong (execute should return actual
        // output size) Note: ResamplePlan interface might need a way to return
        // actual output size For now, assume the span reflects the actual
        // written size if successful.
        // downsampled_signal.resize(downsampled_span.size()); // Adjust if
        // needed

        // *** Make the recursive call using the captured _execute_recursive ***
        return _execute_recursive(downsampled_signal, current_sr / 2.0, 0);
      }

      // Base case: Lowest octave processed or no bins left
      return Status::Success;
    };  // End of lambda definition assignment

    // --- Initial Call to Recursive Function ---
    Status final_status = _execute_recursive(
        input,
        this->sample_rate_,  // Use this-> for member access
        0);                  // Initial bin offset (unused)

    return final_status;
  }

  template <typename T>
  size_t DefaultCQTPlanImpl<T>::get_num_bins() const
  {
    return num_bins_;
  }

  template <typename T>
  size_t DefaultCQTPlanImpl<T>::get_num_output_frames(size_t input_length) const
  {
    if (hop_length_ == 0 || input_length == 0) {
      return 0;
    }
    // Calculate frames based on hop length
    return (input_length + hop_length_ - 1) / hop_length_;
  }

  template <typename T>
  size_t DefaultCQTPlanImpl<T>::get_hop_length() const
  {
    return hop_length_;
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  template class DefaultCQTPlanImpl<float>;
  template class DefaultCQTPlanImpl<double>;

}  // namespace OmniDSP::Default
