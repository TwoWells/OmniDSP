/**
 * @file cqt.cpp
 * @brief Implements the CQTPlan class methods, including the CQT algorithm
 * logic.
 */

#include "OmniDSP/cqt.h"  // Corresponding header

#include <cmath>
#include <complex>
#include <iostream>  // For potential debug/error messages
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/fft.h"       // Needed for internal FFTPlan
#include "OmniDSP/omnidsp.h"   // Needed for owner_dsp_ to call factories
#include "OmniDSP/resample.h"  // Needed for internal ResamplePlan (potentially)
#include "OmniDSP/window.h"    // Needed for WindowSpec (potentially)

// Define PI if not already available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace OmniDSP {

//--------------------------------------------------------------------------
// CQTPlan Method Definitions
//--------------------------------------------------------------------------

// Private Constructor Implementation
template <typename T>
CQTPlan<T>::CQTPlan(const OmniDSP* owner, RealT<T> sample_rate,
                    RealT<T> min_freq, RealT<T> max_freq,
                    int bins_per_octave /*, other params */)
    : owner_dsp_(owner),
      sample_rate_(sample_rate)
// Initialize other stored parameters here...
{
  if (!owner_dsp_) {
    throw std::invalid_argument(
        "CQTPlan must be created by a valid OmniDSP instance.");
  }
  if (sample_rate <= 0 || min_freq <= 0 || max_freq <= 0 ||
      bins_per_octave <= 0 || min_freq >= max_freq ||
      max_freq > sample_rate / 2.0) {
    throw std::invalid_argument("Invalid CQT parameters provided.");
  }

  std::cout << "CQTPlan Constructor: Setting up CQT..." << std::endl;  // Debug

  // --- 1. Calculate CQT Frequencies and Parameters ---
  // Determine the number of bins, center frequencies, Q factor, kernel lengths
  // etc. Example calculation (simplified):
  num_bins_ = static_cast<size_t>(
      std::ceil(bins_per_octave * std::log2(max_freq / min_freq)));
  // Calculate center_frequencies[num_bins_]
  // Calculate Q = 1.0 / (std::pow(2.0, 1.0 / bins_per_octave) - 1.0);
  // Calculate kernel_lengths[num_bins_] based on freq and Q, e.g.,
  // floor(sample_rate * Q / freq_k) Determine max_kernel_length Determine
  // hop_length_ (often based on lowest frequency kernel length)

  std::cout << "CQTPlan Constructor: Num Bins = " << num_bins_
            << std::endl;  // Debug
  // std::cout << "CQTPlan Constructor: Hop Length = " << hop_length_ <<
  // std::endl; // Debug

  // --- 2. Create Necessary Sub-Plans (using owner_dsp_) ---
  // Example: Create an FFT plan large enough for the longest kernel
  size_t max_kernel_len = 4096;  // Placeholder - calculate based on freqs/Q/sr
  auto fft_plan_result =
      owner_dsp_->create_fft_plan<ComplexT<T>>(max_kernel_len);
  if (!fft_plan_result) {
    throw std::runtime_error(
        "Failed to create internal FFTPlan for CQT: " +
        std::string(get_status_string(fft_plan_result.error())));
  }
  fft_plan_ = std::move(fft_plan_result.value());
  std::cout << "CQTPlan Constructor: Internal FFT Plan created."
            << std::endl;  // Debug

  // Potentially create ResamplePlan(s) if needed for kernel
  // generation/resampling

  // --- 3. Pre-compute CQT Kernels ---
  // Allocate space for kernels_
  kernels_.resize(num_bins_);
  std::cout << "CQTPlan Constructor: Generating Kernels..."
            << std::endl;  // Debug
  for (size_t k = 0; k < num_bins_; ++k) {
    // Calculate frequency freq_k for bin k
    // Calculate required length N_k for this kernel
    // Generate temporal kernel: window(N_k) * exp(2*pi*j*freq_k*t/sr)
    // (Use owner_dsp_->hann_window or other window generation methods)
    // Zero-pad kernel to max_kernel_len
    // FFT the padded kernel (using fft_plan_->fft(...))
    // Store the FFT'd kernel (potentially conjugated) in kernels_[k]
  }
  std::cout << "CQTPlan Constructor: Kernels generated." << std::endl;  // Debug

  // --- 4. Store other necessary state ---
  // e.g., hop_length_, internal buffers?

  std::cout << "CQTPlan Constructor: Setup complete." << std::endl;  // Debug
}

// Destructor: Default is sufficient if members clean up correctly.
template <typename T>
CQTPlan<T>::~CQTPlan() = default;

// Move Constructor: Default is sufficient for unique_ptr and vector members.
template <typename T>
CQTPlan<T>::CQTPlan(CQTPlan&& other) noexcept = default;

// Move Assignment Operator: Default is sufficient for unique_ptr and vector
// members.
template <typename T>
CQTPlan<T>& CQTPlan<T>::operator=(CQTPlan&& other) noexcept = default;

// --- Execute Method ---
template <typename T>
[[nodiscard]] Status CQTPlan<T>::execute(std::span<const RealT<T>> input,
                                         std::span<ComplexT<T>> output) const {
  if (!owner_dsp_ ||
      !fft_plan_) {  // Check if plan is valid (e.g., not moved from)
    return Status::InvalidOperation;
  }

  // --- 1. Validate Input/Output Sizes ---
  size_t n_frames = get_num_output_frames(input.size());
  size_t n_bins = get_num_bins();
  size_t expected_output_size = n_bins * n_frames;

  if (output.size() < expected_output_size) {
    return Status::InvalidArgument;  // Output buffer too small
  }
  if (input.empty()) {
    // Handle empty input? Return success with empty output? Zero output span?
    std::fill(output.begin(), output.begin() + expected_output_size,
              ComplexT<T>{0, 0});
    return Status::Success;
  }

  // --- 2. Prepare Internal Buffers (if needed) ---
  // e.g., buffer for FFT of input frames
  size_t fft_len = fft_plan_->get_length();
  std::vector<ComplexT<T>> frame_fft_buffer(fft_len);
  std::vector<ComplexT<T>> frame_buffer_complex(
      fft_len);  // If FFT needs complex input

  // --- 3. Process Input Frame by Frame ---
  Status status = Status::Success;
  for (size_t i = 0; i < n_frames; ++i) {
    // Calculate start sample for this frame
    size_t frame_start = i * hop_length_;
    // Determine how many samples from input correspond to the FFT length needed
    // This depends on the CQT algorithm variant (e.g., overlap-add/save)
    // For simplicity, assume we grab fft_len samples, zero-padding if needed at
    // the end
    size_t samples_to_copy = std::min(fft_len, input.size() - frame_start);

    // Copy input frame to buffer (convert to complex, zero-pad)
    std::fill(frame_buffer_complex.begin(), frame_buffer_complex.end(),
              ComplexT<T>{0, 0});
    for (size_t n = 0; n < samples_to_copy; ++n) {
      frame_buffer_complex[n] = ComplexT<T>{input[frame_start + n], 0.0};
    }

    // Perform FFT on the input frame
    status = fft_plan_->fft(frame_buffer_complex, frame_fft_buffer);
    if (status != Status::Success) {
      return status;  // Propagate FFT error
    }

    // --- 4. Multiply with Kernel FFTs ---
    for (size_t k = 0; k < n_bins; ++k) {
      // Perform element-wise multiplication: frame_fft * conj(kernel_fft)
      // Summation or specific point extraction depends on CQT variant
      ComplexT<T> cqt_coeff = {0, 0};  // Placeholder calculation
      for (size_t f = 0; f < fft_len;
           ++f) {  // Example: Simple dot product - likely incorrect for CQT
        // NOTE: This calculation is highly dependent on the chosen CQT method!
        // It usually involves multiplying the frame FFT by the (conjugated)
        // kernel FFT and summing, or directly convolving in time domain.
        // This placeholder is just illustrative.
        // A common approach involves spectral multiplication and IFFT,
        // or time-domain sparse convolution.
        // cqt_coeff += frame_fft_buffer[f] * std::conj(kernels_[k][f]); //
        // Example only
      }

      // Store result in the correct output position
      // Assuming output is [bin0_frame0, bin1_frame0, ..., binK_frame0,
      // bin0_frame1, ...] (row-major)
      output[i * n_bins + k] =
          cqt_coeff / static_cast<RealT<T>>(fft_len);  // Example scaling
    }
  }

  // Zero out any remaining part of the output buffer if needed
  if (output.size() > expected_output_size) {
    std::fill(output.begin() + expected_output_size, output.end(),
              ComplexT<T>{0, 0});
  }

  return Status::Success;  // Placeholder
}

// --- Getter Methods ---
template <typename T>
size_t CQTPlan<T>::get_num_bins() const {
  // Return the number of bins calculated in the constructor
  return num_bins_;
}

template <typename T>
size_t CQTPlan<T>::get_num_output_frames(size_t input_length) const {
  // Calculate based on input length and hop length
  if (hop_length_ == 0 || input_length == 0) {
    return 0;
  }
  // Example calculation (adjust based on exact framing logic)
  return (input_length + hop_length_ - 1) / hop_length_;  // Ceiling division
}

template <typename T>
size_t CQTPlan<T>::get_hop_length() const {
  // Return the hop length calculated in the constructor
  return hop_length_;
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
// Instantiate templates for common types (float, double) to ensure code
// generation.

template class OmniDSP::CQTPlan<float>;
template class OmniDSP::CQTPlan<double>;

}  // namespace OmniDSP
