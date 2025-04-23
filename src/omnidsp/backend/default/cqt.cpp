/**
 * @file cqt.cpp (default)
 * @brief Implements the Default backend CQTPlanImpl class using a recursive
 * downsampling strategy.
 * @details Provides a portable CQT implementation using standard C++ and
 * leveraging sub-plans (FFTPlan, ResamplePlan) created via the owner
 * OmniDSPImpl instance, ensuring backend consistency. Processes the highest
 * octave via FFT convolution and recursively downsamples the signal to process
 * lower octaves.
 */

#include "OmniDSP/cqt.h"  // Public CQTPlan interface

#include <algorithm>  // For std::transform, std::max_element, std::fill, std::min
#include <cmath>
#include <complex>
#include <functional>  // For std::function (recursive lambda)
#include <iostream>    // For debug messages
#include <limits>      // For std::numeric_limits
#include <numeric>     // For std::iota, std::accumulate
#include <stdexcept>   // For std::runtime_error, std::invalid_argument
#include <vector>

#include "OmniDSP/core_types.h"  // Core types
#include "OmniDSP/fft.h"         // Needed for internal FFTPlan
#include "OmniDSP/resample.h"  // Needed for internal ResamplePlan and ResampleSpec
#include "OmniDSP/window.h"  // Needed for WindowSpec and window generation
#include "backend.h"  // Default backend declarations (includes DefaultCQTPlanImpl declaration)

// Define PI if not available from cmath
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace OmniDSP {
namespace backend {

// Helper function to calculate the next power of two
inline size_t next_power_of_two(size_t n) {
  if (n == 0) return 1;
  // Handle potential overflow if n is already close to max size_t power of 2
  if (n > (std::numeric_limits<size_t>::max() / 2))
    return std::numeric_limits<size_t>::max();
  size_t power = 1;
  while (power < n) {
    power <<= 1;
    if (power == 0)
      return std::numeric_limits<size_t>::max();  // Check overflow during shift
  }
  return power;
}

//--------------------------------------------------------------------------
// DefaultCQTPlanImpl Definition
//--------------------------------------------------------------------------

/**
 * @brief Concrete implementation of CQTPlanImpl for the Default backend.
 * Uses a recursive downsampling strategy with FFT-based convolution.
 */
template <typename T>  // T is real type here
class DefaultCQTPlanImpl final : public CQTPlanImpl<T> {
 public:
  /**
   * @brief Constructor. Sets up CQT parameters, kernels, and sub-plans.
   * @param owner_impl Pointer to the OmniDSPImpl instance creating this plan.
   * @param sample_rate The sample rate of the input signal in Hz.
   * @param min_freq The minimum frequency for the CQT analysis in Hz.
   * @param max_freq The maximum frequency for the CQT analysis in Hz.
   * @param bins_per_octave The number of frequency bins per octave.
   * @param window_spec Specification for the window function to use for CQT
   * kernels.
   * @throws std::invalid_argument If parameters are invalid or owner_impl is
   * null.
   * @throws std::runtime_error If creation of internal plans (e.g., FFT,
   * Resample) fails.
   */
  DefaultCQTPlanImpl(const OmniDSPImpl* owner_impl, RealT<T> sample_rate,
                     RealT<T> min_freq, RealT<T> max_freq, int bins_per_octave,
                     const WindowSpec<T>& window_spec);

  /** @brief Destructor. */
  ~DefaultCQTPlanImpl() override;

  // --- Interface Methods ---
  Status execute(std::span<const RealT<T>> input,
                 std::span<ComplexT<T>> output) const override;
  size_t get_num_bins() const override;
  size_t get_num_output_frames(size_t input_length) const override;
  size_t get_hop_length() const override;

 private:
  // --- Configuration & Parameters ---
  const OmniDSPImpl*
      owner_impl_;  // Non-owning pointer to backend implementation
  RealT<T> sample_rate_;
  RealT<T> min_freq_;
  RealT<T> max_freq_;
  int bins_per_octave_;
  WindowSpec<T> window_spec_;
  RealT<T> quality_factor_;
  size_t num_bins_;
  size_t fft_length_;
  size_t hop_length_;
  std::vector<RealT<T>> center_frequencies_;
  std::vector<size_t> kernel_lengths_;

  // --- Pre-computed Data ---
  std::vector<std::vector<ComplexT<T>>>
      kernels_fft_;  // FFT'd and conjugated kernels

  // --- Sub-Plans ---
  std::unique_ptr<FFTPlan<ComplexT<T>>> fft_plan_;
  std::unique_ptr<ResamplePlan<T>>
      resample_plan_;  // Plan for factor-2 downsampling
};

//--------------------------------------------------------------------------
// DefaultCQTPlanImpl Method Implementations
//--------------------------------------------------------------------------

template <typename T>
DefaultCQTPlanImpl<T>::DefaultCQTPlanImpl(const OmniDSPImpl* owner_impl,
                                          RealT<T> sample_rate,
                                          RealT<T> min_freq, RealT<T> max_freq,
                                          int bins_per_octave,
                                          const WindowSpec<T>& window_spec)
    : owner_impl_(owner_impl),
      sample_rate_(sample_rate),
      min_freq_(min_freq),
      max_freq_(max_freq),
      bins_per_octave_(bins_per_octave),
      window_spec_(window_spec) {
  if (!owner_impl_) {
    throw std::invalid_argument(
        "DefaultCQTPlanImpl requires a valid owner OmniDSPImpl pointer.");
  }
  // Basic parameter validation
  if (sample_rate <= 0 || min_freq <= 0 || max_freq <= 0 ||
      bins_per_octave <= 0 || min_freq >= max_freq ||
      max_freq >= sample_rate / static_cast<RealT<T>>(2.0)) {
    throw std::invalid_argument("Invalid CQT parameters provided.");
  }

  std::cout << "Default CQTPlanImpl Constructor (Recursive): Setting up CQT..."
            << std::endl;

  // --- 1. Calculate Global CQT Parameters ---
  quality_factor_ = static_cast<RealT<T>>(1.0) /
                    (std::pow(static_cast<RealT<T>>(2.0),
                              static_cast<RealT<T>>(1.0) / bins_per_octave_) -
                     static_cast<RealT<T>>(1.0));
  // Calculate total number of bins based on the full frequency range
  num_bins_ = static_cast<size_t>(
      std::ceil(bins_per_octave_ * std::log2(max_freq_ / min_freq_)));
  if (num_bins_ == 0) {
    throw std::invalid_argument("CQT parameters result in zero bins.");
  }

  center_frequencies_.resize(num_bins_);
  kernel_lengths_.resize(num_bins_);
  RealT<T> freq_ratio = std::pow(static_cast<RealT<T>>(2.0),
                                 static_cast<RealT<T>>(1.0) / bins_per_octave_);
  size_t max_kernel_len_overall = 0;  // Max length across all bins

  for (size_t k = 0; k < num_bins_; ++k) {
    center_frequencies_[k] =
        min_freq_ * std::pow(freq_ratio, static_cast<RealT<T>>(k));
    // Check if frequency exceeds Nyquist limit for the *original* sample rate
    if (center_frequencies_[k] >= sample_rate_ / static_cast<RealT<T>>(2.0)) {
      std::cerr << "Warning: CQT frequency bin " << k << " ("
                << center_frequencies_[k] << " Hz) is at or above Nyquist ("
                << sample_rate_ / 2.0 << " Hz). Clamping num_bins."
                << std::endl;
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
  fft_length_ = next_power_of_two(max_kernel_len_overall);
  if (fft_length_ == std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("Required FFT length for CQT exceeds limits.");
  }

  // Hop length calculation: Base it on the lowest frequency kernel length.
  // A smaller hop length increases temporal resolution but also computation.
  // Common choices are powers of 2, or related to kernel lengths.
  // Using kernel_lengths_[0] / 4 as a starting point.
  hop_length_ = std::max(static_cast<size_t>(1), kernel_lengths_[0] / 4);
  // Ensure hop_length is reasonable, maybe power of 2?
  // hop_length_ = next_power_of_two(hop_length_); // Optional: force power of 2

  std::cout << "  Num Bins (Total): " << num_bins_ << ", Q: " << quality_factor_
            << ", Max Kernel Len: " << max_kernel_len_overall
            << ", FFT Len: " << fft_length_ << ", Hop Len: " << hop_length_
            << std::endl;

  // --- 2. Create Sub-Plans (using owner_impl_) ---
  // FFT Plan (for the highest octave, using the overall max FFT length)
  auto fft_plan_result = owner_impl_->create_fft_plan<ComplexT<T>>(fft_length_);
  if (!fft_plan_result) {
    throw std::runtime_error(
        "Failed to create internal FFTPlan for CQT: " +
        std::string(get_status_string(fft_plan_result.error())));
  }
  fft_plan_ = std::move(fft_plan_result.value());
  std::cout << "  Internal FFT Plan created via owner backend (Length: "
            << fft_length_ << ")." << std::endl;

  // Resample Plan (Factor 2 Downsampling)
  // Create the ResampleSpec for factor-2 downsampling. Use a default quality.
  ResampleSpec resample_spec;
  resample_spec.input_rate = sample_rate_;  // Initial rate
  resample_spec.output_rate = sample_rate_ / 2.0;
  resample_spec.quality = 12;  // Default quality factor, adjust if needed
  // resample_spec.max_input_size = 0; // Optional hint, not used in current API

  auto resample_plan_result =
      owner_impl_->create_resample_plan<T>(resample_spec);
  if (!resample_plan_result) {
    throw std::runtime_error(
        "Failed to create internal ResamplePlan for CQT: " +
        std::string(get_status_string(resample_plan_result.error())));
  }
  resample_plan_ = std::move(resample_plan_result.value());
  std::cout << "  Internal Resample Plan (Factor 2) created via owner backend."
            << std::endl;

  // --- 3. Pre-compute All CQT Kernels (FFT'd) ---
  // Kernels are computed relative to the original sample rate and max FFT
  // length
  kernels_fft_.resize(num_bins_);
  std::cout << "  Generating All CQT Kernels..." << std::endl;

  for (size_t k = 0; k < num_bins_; ++k) {
    size_t N_k = kernel_lengths_[k];
    if (N_k == 0) continue;  // Should not happen if clamped correctly

    // Generate window using the owner backend's window function
    auto window_coeffs_result = owner_impl_->generate_window(window_spec_, N_k);
    if (!window_coeffs_result) {
      throw std::runtime_error(
          "Failed to generate window for CQT kernel " + std::to_string(k) +
          ": " + std::string(get_status_string(window_coeffs_result.error())));
    }
    std::vector<RealT<T>> window_coeffs =
        std::move(window_coeffs_result.value());

    // Generate temporal kernel
    std::vector<ComplexT<T>> temporal_kernel(N_k);
    RealT<T> freq_k = center_frequencies_[k];
    RealT<T> norm_factor = static_cast<RealT<T>>(1.0) /
                           static_cast<RealT<T>>(N_k);  // Normalization factor

    for (size_t n = 0; n < N_k; ++n) {
      RealT<T> time_arg =
          static_cast<RealT<T>>(n) /
          sample_rate_;  // Time relative to original sample rate
      RealT<T> angle = static_cast<RealT<T>>(2.0 * M_PI) * freq_k * time_arg;
      ComplexT<T> complex_exp = {std::cos(angle), std::sin(angle)};
      // Apply window and normalization
      temporal_kernel[n] = window_coeffs[n] * complex_exp * norm_factor;
    }

    // Zero-pad temporal kernel to fft_length_
    temporal_kernel.resize(fft_length_, {0.0, 0.0});

    // FFT the padded kernel and store the conjugate
    kernels_fft_[k].resize(fft_length_);
    Status status = fft_plan_->fft(temporal_kernel, kernels_fft_[k]);
    if (status != Status::Success) {
      throw std::runtime_error("FFT failed for CQT kernel " +
                               std::to_string(k));
    }
    // Conjugate the kernel FFT for efficient convolution via multiplication
    std::transform(kernels_fft_[k].begin(), kernels_fft_[k].end(),
                   kernels_fft_[k].begin(),
                   [](const ComplexT<T>& c) { return std::conj(c); });
  }
  std::cout << "  All CQT Kernels generated and FFT'd." << std::endl;
  std::cout << "Default CQTPlanImpl Constructor (Recursive): Setup complete."
            << std::endl;
}

template <typename T>
DefaultCQTPlanImpl<T>::~DefaultCQTPlanImpl() {
  std::cout << "Default CQTPlanImpl destroyed." << std::endl;
  // fft_plan_ and resample_plan_ unique_ptrs handle cleanup
}

template <typename T>
Status DefaultCQTPlanImpl<T>::execute(std::span<const RealT<T>> input,
                                      std::span<ComplexT<T>> output) const {
  if (!owner_impl_ || !fft_plan_ || !resample_plan_) {
    return Status::InvalidOperation;  // Plan not valid or missing sub-plans
  }

  size_t n_frames_total = get_num_output_frames(
      input.size());  // Total frames based on original hop length
  size_t n_bins_total = get_num_bins();
  size_t expected_output_size = n_bins_total * n_frames_total;

  if (output.size() < expected_output_size) {
    std::cerr << "CQT execute error: Output buffer size (" << output.size()
              << ") is smaller than required (" << expected_output_size << ")."
              << std::endl;
    return Status::SizeMismatch;
  }
  // Initialize output to zero
  std::fill(output.begin(), output.begin() + expected_output_size,
            ComplexT<T>{0, 0});

  if (input.empty() || n_frames_total == 0 || n_bins_total == 0) {
    return Status::Success;  // Handle empty cases
  }

  // --- Recursive Execution ---
  // Define the recursive lambda function
  std::function<Status(std::span<const RealT<T>>, RealT<T>, size_t)>
      _execute_recursive;
  _execute_recursive = [&](std::span<const RealT<T>> current_input,
                           RealT<T> current_sr, size_t bin_offset) -> Status {
    // 1. Identify bins for the current octave (highest frequencies for this
    // sample rate)
    RealT<T> current_nyquist = current_sr / static_cast<RealT<T>>(2.0);
    size_t start_bin = bin_offset;  // Start from the offset passed down
    size_t end_bin = start_bin;

    // Find the range of bins [start_bin, end_bin) to process at this sample
    // rate. These are the bins whose center frequencies are >= current_nyquist
    // / 2 and < current_nyquist.
    while (end_bin < num_bins_ &&
           center_frequencies_[end_bin] <
               current_nyquist / static_cast<RealT<T>>(2.0)) {
      end_bin++;  // Skip bins belonging to lower octaves
    }
    start_bin = end_bin;  // Start processing from the first bin in this octave
    while (end_bin < num_bins_ &&
           center_frequencies_[end_bin] < current_nyquist) {
      end_bin++;  // Include bins up to the current Nyquist limit
    }

    size_t num_bins_this_octave = end_bin - start_bin;

    if (num_bins_this_octave > 0) {
      // std::cout << "  Processing octave SR=" << current_sr << ", Bins " <<
      // start_bin << "-" << end_bin - 1 << std::endl; // Debug

      // 2. Calculate CQT for this octave's bins using FFT convolution
      // Hop length needs to scale with sample rate for consistent time
      // resolution
      size_t current_hop_length = static_cast<size_t>(std::round(
          static_cast<double>(hop_length_) * (current_sr / sample_rate_)));
      if (current_hop_length == 0) current_hop_length = 1;

      size_t current_n_frames =
          (current_input.size() + current_hop_length - 1) / current_hop_length;
      if (current_n_frames == 0 && !current_input.empty()) current_n_frames = 1;

      // Ensure the number of frames matches the total expected frames
      // This recursive approach might lead to slight variations if not careful.
      // For simplicity, assume current_n_frames aligns with n_frames_total for
      // now. A more robust implementation might need padding or careful frame
      // alignment.
      if (current_n_frames != n_frames_total) {
        std::cerr << "Warning: Frame count mismatch in CQT recursion ("
                  << current_n_frames << " vs " << n_frames_total << ")"
                  << std::endl;
        // Adjust current_n_frames to match total? Or handle potential output
        // buffer overflow?
        current_n_frames = n_frames_total;  // Force alignment for now
      }

      std::vector<ComplexT<T>> frame_buffer(fft_length_);
      std::vector<ComplexT<T>> frame_fft(fft_length_);

      for (size_t i = 0; i < current_n_frames; ++i) {
        size_t frame_start = i * current_hop_length;
        size_t frame_end =
            std::min(frame_start + fft_length_, current_input.size());
        size_t current_frame_len = (frame_start < current_input.size())
                                       ? (frame_end - frame_start)
                                       : 0;

        std::fill(frame_buffer.begin(), frame_buffer.end(), ComplexT<T>{0, 0});
        for (size_t n = 0; n < current_frame_len; ++n) {
          frame_buffer[n] = ComplexT<T>{current_input[frame_start + n], 0.0};
        }

        Status fft_status = fft_plan_->fft(frame_buffer, frame_fft);
        if (fft_status != Status::Success) return fft_status;

        // Multiply frame FFT by conjugated kernel FFTs for this octave
        for (size_t k_idx = 0; k_idx < num_bins_this_octave; ++k_idx) {
          size_t k = start_bin + k_idx;  // Global bin index
          ComplexT<T> cqt_coeff = {0, 0};
          // Perform element-wise multiplication and sum (dot product in
          // frequency domain)
          for (size_t f = 0; f < fft_length_; ++f) {
            cqt_coeff +=
                frame_fft[f] *
                kernels_fft_[k][f];  // kernels_fft_ is already conjugated
          }
          // Store in the final output buffer at the correct frame and bin index
          // Output layout: [bin0_frame0, bin1_frame0, ..., binK_frame0,
          // bin0_frame1, ...] (Column-major: bins vary fastest) size_t
          // output_idx = k * n_frames_total + i; // Column-major Output layout:
          // [bin0_frame0, bin0_frame1, ..., bin0_frameN, bin1_frame0, ...]
          // (Row-major: frames vary fastest)
          size_t output_idx =
              i * n_bins_total + k;  // Row-major (frame by frame - seems more
                                     // common for spectrograms)

          if (output_idx < output.size()) {
            // Apply scaling factor (1/N from IFFT implicit in convolution)
            output[output_idx] = cqt_coeff / static_cast<RealT<T>>(fft_length_);
          } else {
            std::cerr << "Warning: Output index " << output_idx
                      << " out of bounds (" << output.size()
                      << ") during CQT calculation." << std::endl;
            // Stop processing this frame? Or return error?
            return Status::SizeMismatch;  // Indicate buffer overflow
                                          // possibility
          }
        }
      }
    } else {
      // std::cout << "  No bins to process for SR=" << current_sr << std::endl;
      // // Debug
    }

    // 3. Check if lower octaves need processing (bins below start_bin)
    size_t lowest_bin_processed = start_bin;
    if (lowest_bin_processed >
        0) {  // If there are bins below the ones just processed
      // std::cout << "  Downsampling for next octave..." << std::endl; // Debug
      // 4. Downsample the signal
      // Need to reset the resampler state before processing each octave level?
      // Yes.
      resample_plan_->reset();
      size_t downsampled_len_est =
          resample_plan_->get_output_length(current_input.size());
      std::vector<RealT<T>> downsampled_signal(downsampled_len_est);
      Status resample_status =
          resample_plan_->execute(current_input, downsampled_signal);
      if (resample_status != Status::Success) {
        std::cerr << "CQT execute error: Resampling failed." << std::endl;
        return resample_status;
      }
      // Note: The actual output size of the resampler might differ slightly
      // from estimate. Using the vector's actual size after resizing might be
      // safer if execute doesn't guarantee filling. Assuming execute fills the
      // span up to the required length.

      // 5. Recursive call for the lower octave
      // The bin_offset remains 0 because we process *all remaining* bins at the
      // lower sample rate.
      return _execute_recursive(downsampled_signal, current_sr / 2.0, 0);
    }

    return Status::Success;  // Base case: No more lower octaves
  };

  // --- Initial Call to Recursive Function ---
  Status final_status = _execute_recursive(
      input, sample_rate_,
      0);  // Start with original signal, full SR, bin offset 0

  return final_status;
}

template <typename T>
size_t DefaultCQTPlanImpl<T>::get_num_bins() const {
  return num_bins_;
}

template <typename T>
size_t DefaultCQTPlanImpl<T>::get_num_output_frames(size_t input_length) const {
  if (hop_length_ == 0 || input_length == 0) {
    return 0;
  }
  // Calculate frames based on the hop length defined at the *original* sample
  // rate.
  return (input_length + hop_length_ - 1) / hop_length_;  // Ceiling division
}

template <typename T>
size_t DefaultCQTPlanImpl<T>::get_hop_length() const {
  return hop_length_;
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
template class DefaultCQTPlanImpl<float>;
template class DefaultCQTPlanImpl<double>;

}  // namespace backend
}  // namespace OmniDSP
