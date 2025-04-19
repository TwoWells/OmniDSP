/**
 * @file cqt.cpp
 * @brief Implementation file for Constant-Q Transform (CQT) functions and
 * plans.
 */
#include "OmniDSP/cqt.h"  // Public header declarations

#include <algorithm>  // For std::max_element, std::min_element, std::transform, std::fill
#include <cmath>  // For std::log2, std::pow, std::ceil, std::round, std::abs
#include <complex>
#include <memory>     // For std::unique_ptr
#include <numeric>    // For std::accumulate, std::iota
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "OmniDSP/fft.h"     // Needed for FFTPlan
#include "OmniDSP/window.h"  // Needed for windowing functions

// Include backend factory or plan headers if needed directly
// #include "backend/backend_impl.h" // If using backend factory

namespace OmniDSP {

//--------------------------------------------------------------------------
// Helper Functions (Internal)
//--------------------------------------------------------------------------

// Calculates the center frequencies for CQT bins
inline std::vector<double> calculate_cqt_freqs(double fmin, int n_bins,
                                               int bins_per_octave) {
  if (fmin <= 0) throw std::invalid_argument("fmin must be positive.");
  if (n_bins <= 0) throw std::invalid_argument("n_bins must be positive.");
  if (bins_per_octave <= 0)
    throw std::invalid_argument("bins_per_octave must be positive.");

  std::vector<double> freqs(n_bins);
  double factor = std::pow(2.0, 1.0 / bins_per_octave);
  for (int i = 0; i < n_bins; ++i) {
    freqs[i] = fmin * std::pow(factor, i);
  }
  return freqs;
}

// Calculates the quality factor Q
inline double calculate_q(double sr, int bins_per_octave) {
  if (sr <= 0)
    throw std::invalid_argument("Sampling rate (sr) must be positive.");
  if (bins_per_octave <= 0)
    throw std::invalid_argument("bins_per_octave must be positive.");
  // Q = 1 / (2^(1/B) - 1)
  return 1.0 / (std::pow(2.0, 1.0 / bins_per_octave) - 1.0);
}

// Calculates required FFT length for a given frequency
inline size_t calculate_fft_length(double freq, double sr, double q) {
  if (freq <= 0)
    throw std::invalid_argument(
        "Frequency must be positive for FFT length calculation.");
  if (sr <= 0) throw std::invalid_argument("Sampling rate must be positive.");
  if (q <= 0) throw std::invalid_argument("Quality factor Q must be positive.");
  // N = ceil(Q * sr / freq)
  double N_double = std::ceil(q * sr / freq);
  if (N_double <= 0 || N_double > static_cast<double>(SIZE_MAX)) {
    throw std::overflow_error("Calculated FFT length exceeds limits.");
  }
  // Ensure power of 2 for efficiency? Often CQT doesn't require this strictly,
  // but FFT backends might be faster. Let's stick to the formula for now.
  return static_cast<size_t>(N_double);
}

//--------------------------------------------------------------------------
// CQTPlan Implementation Details (Example - may vary based on chosen method)
//--------------------------------------------------------------------------
// This is a simplified placeholder. Real CQT implementations can be complex
// (e.g., recursive, using filterbanks). This example uses a direct approach
// with varying FFT sizes per bin, which is often inefficient but illustrative.

template <typename T>
struct CQTPlanImpl {
  // Parameters stored from constructor
  double sr_;
  double fmin_;
  int n_bins_;
  int bins_per_octave_;
  Precision precision_;
  FFTNorm norm_;  // Store the desired normalization

  // Calculated values
  double q_;
  std::vector<double> cqt_freqs_;
  std::vector<size_t> fft_lengths_;  // FFT length needed for each bin
  size_t max_fft_length_;  // Maximum FFT length needed across all bins

  // FFT plan (reused if possible, or create multiple)
  // Using a single plan for the max length for simplicity here.
  // More advanced implementations might use multiple plans or other FFT
  // strategies.
  std::unique_ptr<FFTPlan<T>> fft_plan_;

  // Precomputed CQT kernels (sparse representation might be better)
  std::vector<std::vector<std::complex<T>>> cqt_kernels_fft_;

  CQTPlanImpl(double sr, double fmin, int n_bins, int bins_per_octave,
              Precision p, FFTNorm norm)
      : sr_(sr),
        fmin_(fmin),
        n_bins_(n_bins),
        bins_per_octave_(bins_per_octave),
        precision_(p),
        norm_(norm) {
    if (sr <= 0 || fmin <= 0 || n_bins <= 0 || bins_per_octave <= 0) {
      throw std::invalid_argument("Invalid CQT parameters provided.");
    }

    q_ = calculate_q(sr, bins_per_octave);
    cqt_freqs_ = calculate_cqt_freqs(fmin, n_bins, bins_per_octave);

    // Calculate required FFT lengths and find the maximum
    fft_lengths_.resize(n_bins);
    max_fft_length_ = 0;
    for (int k = 0; k < n_bins; ++k) {
      fft_lengths_[k] = calculate_fft_length(cqt_freqs_[k], sr, q_);
      if (fft_lengths_[k] > max_fft_length_) {
        max_fft_length_ = fft_lengths_[k];
      }
    }

    // --- Create FFT Plan ---
    // Create a single plan for the maximum FFT length needed.
    // The plan direction should be Forward for applying kernels.
    // The normalization mode is passed from the CQT constructor.
    try {
      fft_plan_ = std::make_unique<FFTPlan<T>>(max_fft_length_, precision_,
                                               Direction::Forward,
                                               Domain::Complex, norm_);
    } catch (const std::exception& e) {
      throw std::runtime_error("Failed to create underlying FFTPlan for CQT: " +
                               std::string(e.what()));
    }
    if (!fft_plan_) {
      throw std::runtime_error(
          "Failed to create underlying FFTPlan for CQT (null pointer).");
    }

    // --- Precompute CQT Kernels ---
    // This part is highly dependent on the chosen CQT algorithm.
    // Example: Generate time-domain kernels and FFT them.
    cqt_kernels_fft_.resize(n_bins_);
    std::vector<T> temp_window;
    std::vector<std::complex<T>> temp_kernel_time(max_fft_length_);
    std::vector<std::complex<T>> temp_kernel_fft(max_fft_length_);

    for (int k = 0; k < n_bins_; ++k) {
      size_t N_k = fft_lengths_[k];  // Length specific to this bin
      if (N_k == 0) continue;        // Skip if length is zero

      // 1. Create window (e.g., Hann) of length N_k
      // Using a simple rectangular window here for brevity
      temp_window.assign(N_k, static_cast<T>(1.0));
      // Example using Hann:
      // Window::hann(temp_window, N_k);

      // 2. Create complex sinusoid e^(2*pi*j*freq_k*t/sr) * window
      std::fill(temp_kernel_time.begin(), temp_kernel_time.end(),
                std::complex<T>(0.0, 0.0));
      T phase_factor = static_cast<T>(2.0 * M_PI * cqt_freqs_[k] / sr_);
      for (size_t n = 0; n < N_k; ++n) {
        T time_sec = static_cast<T>(n) / sr_;
        T angle = phase_factor * static_cast<T>(n);
        // Ensure window access is within bounds
        T window_val =
            (n < temp_window.size()) ? temp_window[n] : static_cast<T>(0.0);
        // Place centered in the max_fft_length buffer? Or zero-pad later?
        // Simple zero-padding approach:
        temp_kernel_time[n] = std::polar(window_val, angle) /
                              static_cast<T>(N_k);  // Normalize kernel?
      }

      // 3. Compute FFT of the zero-padded kernel using the plan
      // Ensure the plan's internal direction matches the required operation
      // (Forward) The plan was created with Forward direction.
      fft_plan_->fft(temp_kernel_time,
                     temp_kernel_fft);  // Pass only 2 arguments

      // 4. Store the FFT'd kernel (conjugate for correlation via
      // multiplication)
      cqt_kernels_fft_[k].resize(max_fft_length_);
      std::transform(temp_kernel_fft.begin(), temp_kernel_fft.end(),
                     cqt_kernels_fft_[k].begin(),
                     [](const std::complex<T>& val) { return std::conj(val); });
    }
  }

  // --- Execute Method ---
  void execute(const std::vector<std::complex<T>>& input,
               std::vector<std::complex<T>>& output) {
    if (!fft_plan_) {
      throw std::runtime_error("CQT execute called on an uninitialized plan.");
    }
    if (input.empty()) {
      output.clear();
      output.resize(n_bins_, std::complex<T>(0.0, 0.0));
      return;
    }

    // Ensure output vector has the correct size (number of CQT bins)
    if (output.size() != n_bins_) {
      output.resize(n_bins_);
    }

    // --- Main CQT Calculation (using FFT multiplication) ---
    // 1. Compute FFT of the input signal (padded to max_fft_length_)
    std::vector<std::complex<T>> input_padded = input;  // Copy input
    if (input_padded.size() < max_fft_length_) {
      input_padded.resize(max_fft_length_,
                          std::complex<T>(0.0, 0.0));  // Zero-pad
    } else if (input_padded.size() > max_fft_length_) {
      input_padded.resize(
          max_fft_length_);  // Truncate (consider warning/error)
    }

    std::vector<std::complex<T>> input_fft(max_fft_length_);
    // Use the fft_plan_ (created with Forward direction and desired norm)
    fft_plan_->fft(input_padded, input_fft);  // Pass only 2 arguments

    // 2. Multiply input FFT by precomputed conjugate kernel FFTs
    // 3. Compute IFFT of the result for each bin (or sum relevant FFT bins)
    // This step depends heavily on the specific CQT formulation.
    // A common approach sums parts of the product spectrum.
    // Simplified placeholder: just calculate dot product for illustration
    // (slow) This is NOT the efficient FFT-based method.
    for (int k = 0; k < n_bins_; ++k) {
      if (cqt_kernels_fft_[k].size() != max_fft_length_) {
        // Should not happen if precomputation was correct
        output[k] = std::complex<T>(0.0, 0.0);
        continue;
      }
      // Element-wise product and sum (dot product in frequency domain)
      std::complex<T> bin_value(0.0, 0.0);
      for (size_t i = 0; i < max_fft_length_; ++i) {
        bin_value += input_fft[i] * cqt_kernels_fft_[k][i];
      }
      output[k] = bin_value;
    }

    // Note: A proper FFT-based CQT would involve an inverse FFT here,
    // or more sophisticated spectral summing based on the kernel bandwidths.
    // The current implementation is a placeholder demonstrating the FFT call
    // fix.
  }

};  // End CQTPlanImpl struct

//--------------------------------------------------------------------------
// CQTPlan Class Method Definitions
//--------------------------------------------------------------------------

template <typename T>
CQTPlan<T>::CQTPlan(double sr, double fmin, int n_bins, int bins_per_octave,
                    Precision p, FFTNorm norm)
    : pimpl_(std::make_unique<CQTPlanImpl<T>>(sr, fmin, n_bins, bins_per_octave,
                                              p, norm)) {}

template <typename T>
CQTPlan<T>::~CQTPlan() = default;

template <typename T>
CQTPlan<T>::CQTPlan(CQTPlan&&) noexcept = default;

template <typename T>
CQTPlan<T>& CQTPlan<T>::operator=(CQTPlan&&) noexcept = default;

template <typename T>
void CQTPlan<T>::execute(const std::vector<std::complex<T>>& input,
                         std::vector<std::complex<T>>& output) {
  if (!pimpl_) {
    throw std::runtime_error(
        "CQTPlan execute called on a moved-from or uninitialized plan.");
  }
  pimpl_->execute(input, output);
}

template <typename T>
int CQTPlan<T>::getNumBins() const {
  if (!pimpl_) throw std::runtime_error("Invalid CQTPlan.");
  return pimpl_->n_bins_;
}

template <typename T>
double CQTPlan<T>::getMinFrequency() const {
  if (!pimpl_) throw std::runtime_error("Invalid CQTPlan.");
  return pimpl_->fmin_;
}

template <typename T>
double CQTPlan<T>::getSamplingRate() const {
  if (!pimpl_) throw std::runtime_error("Invalid CQTPlan.");
  return pimpl_->sr_;
}

//--------------------------------------------------------------------------
// Convenience Function
//--------------------------------------------------------------------------
template <typename T>
void cqt(const std::vector<std::complex<T>>& input,
         std::vector<std::complex<T>>& output, double sr, double fmin,
         int n_bins, int bins_per_octave, FFTNorm norm) {
  if (input.empty()) {
    output.clear();
    output.resize(n_bins, std::complex<T>(0.0, 0.0));
    return;
  }
  // Determine precision from T
  Precision prec =
      std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;

  try {
    // Create a temporary plan
    CQTPlan<T> plan(sr, fmin, n_bins, bins_per_octave, prec, norm);
    // Execute the plan
    plan.execute(input, output);
  } catch (const std::exception& e) {
    // Catch potential errors during plan creation or execution
    throw std::runtime_error("Convenience cqt failed: " +
                             std::string(e.what()));
  }
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
// Required because the definitions are in this .cpp file.

// CQTPlan Class & Methods
template class CQTPlan<float>;
template class CQTPlan<double>;
// template void CQTPlan<float>::execute(const
// std::vector<std::complex<float>>&, std::vector<std::complex<float>>&);
// template void CQTPlan<double>::execute(const
// std::vector<std::complex<double>>&, std::vector<std::complex<double>>&);
// template int CQTPlan<float>::getNumBins() const;
// template int CQTPlan<double>::getNumBins() const;
// ... other getters if instantiated explicitly ...

// Convenience Function
template void cqt<float>(const std::vector<std::complex<float>>&,
                         std::vector<std::complex<float>>&, double, double, int,
                         int, FFTNorm);
template void cqt<double>(const std::vector<std::complex<double>>&,
                          std::vector<std::complex<double>>&, double, double,
                          int, int, FFTNorm);

}  // namespace OmniDSP
