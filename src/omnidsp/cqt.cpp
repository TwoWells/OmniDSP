#include "OmniDSP/cqt.h"  // Class declaration

#include <algorithm>  // For std::max, std::max_element
#include <cmath>      // For std::pow, std::ceil, std::log2, M_PI, std::cos etc.
#include <complex>
#include <iostream>   // For std::cout, std::cerr (debugging/logging)
#include <memory>     // For std::unique_ptr, std::make_unique
#include <numeric>    // For std::iota, std::accumulate
#include <stdexcept>  // For std::invalid_argument, std::runtime_error
#include <utility>    // For std::move
#include <vector>

#include "OmniDSP/fft.h"       // For FFTPlan
#include "OmniDSP/resample.h"  // For filter_and_downsample
#include "OmniDSP/window.h"  // For OmniDSP::Window class methods (including get_coeffs)
#include "backend/backend_impl.h"  // Access backend implementations (if needed directly, unlikely here)

// Define M_PI if not defined by <cmath> (e.g., on Windows with MSVC)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace OmniDSP {

// --- Helper Functions (Internal Linkage) ---
namespace {

/**
 * @brief Calculates the smallest power of 2 greater than or equal to n.
 * Used to determine appropriate FFT lengths.
 */
int nextPowerOf2(int n) {
  if (n <= 0) return 1;
  int power = 1;
  while (power < n) {
    power *= 2;
  }
  return power;
}

}  // End anonymous namespace

// --- CQTPlan Constructor ---
/**
 * @brief Constructs and initializes the CQT Plan.
 */
template <typename T>
CQTPlan<T>::CQTPlan(T sample_rate, int n_bins, int bins_per_octave, T fmin,
                    T sparsity_threshold)
    : sr_(sample_rate),
      n_bins_(n_bins),
      bins_per_octave_(bins_per_octave),
      f_min_(fmin),
      n_octaves_(0),
      sparsity_threshold_(sparsity_threshold),
      filters_calculated_(false),
      q_(static_cast<T>(0.0)) {
  // Parameter Validation
  if (sample_rate <= 0)
    throw std::invalid_argument("CQTPlan: Sample rate must be positive.");
  if (n_bins <= 0)
    throw std::invalid_argument("CQTPlan: Number of bins must be positive.");
  if (bins_per_octave <= 0)
    throw std::invalid_argument("CQTPlan: Bins per octave must be positive.");
  if (fmin <= 0)
    throw std::invalid_argument("CQTPlan: Minimum frequency must be positive.");
  if (n_bins % bins_per_octave != 0) {
    std::cerr << "[CQTPlan WARNING] n_bins (" << n_bins
              << ") is not a multiple of bins_per_octave (" << bins_per_octave
              << "). Number of octaves might not be exact." << std::endl;
  }
  n_octaves_ = static_cast<int>(
      std::ceil(static_cast<double>(n_bins) / bins_per_octave));

  // Precompute Constants
  q_ = static_cast<T>(1.0) /
       (std::pow(static_cast<T>(2.0), static_cast<T>(1.0) / bins_per_octave_) -
        static_cast<T>(1.0));

  // Precompute FFT lengths needed for each octave
  fft_lengths_.resize(n_octaves_);
  for (int i = 0; i < n_octaves_; ++i) {
    int last_bin_in_octave = std::min((i + 1) * bins_per_octave_, n_bins_) - 1;
    T freq_k_high = f_min_ * std::pow(static_cast<T>(2.0),
                                      static_cast<T>(last_bin_in_octave) /
                                          bins_per_octave_);
    T Nk_approx = q_ * sr_ / freq_k_high;
    int Nk = static_cast<int>(std::ceil(Nk_approx));
    Nk = std::max(Nk, 1);
    fft_lengths_[i] = nextPowerOf2(Nk);
  }
}

// --- CQTPlan Destructor ---
template <typename T>
CQTPlan<T>::~CQTPlan() {}

// --- CQTPlan::calculateFilters ---
/**
 * @brief Generates the CQT filter bank (frequency domain kernels).
 */
template <typename T>
void CQTPlan<T>::calculateFilters() {
  if (filters_calculated_) return;
  std::cout << "[CQTPlan] Calculating CQT Filter Bank..." << std::endl;
  filter_bank_.assign(n_bins_, {});
  int max_fft_len = 0;
  if (!fft_lengths_.empty())
    max_fft_len = *std::max_element(fft_lengths_.begin(), fft_lengths_.end());
  if (max_fft_len <= 0)
    throw std::runtime_error("CQTPlan: Could not determine valid FFT length.");

  if (!fft_plan_ || fft_plan_->getSize() != static_cast<size_t>(max_fft_len)) {
    std::cout << "[CQTPlan] Creating FFT Plan (size " << max_fft_len
              << ") for filter calculation." << std::endl;
    try {
      fft_plan_ = std::make_unique<FFTPlan<T>>(max_fft_len);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          std::string("CQTPlan: Failed to create FFTPlan: ") + e.what());
    }
  }

  for (int k = 0; k < n_bins_; ++k) {
    int octave = k / bins_per_octave_;
    int N_fft = fft_lengths_[octave];
    T freq_k = f_min_ * std::pow(static_cast<T>(2.0),
                                 static_cast<T>(k) / bins_per_octave_);
    int Nk = static_cast<int>(std::ceil(q_ * sr_ / freq_k));
    Nk = std::max(Nk, 1);

    // 1. Create time-domain kernel
    std::vector<std::complex<T>> time_kernel(Nk);
    std::vector<T> window = OmniDSP::Window::get_hann_coeffs<T>(Nk);
    if (window.size() != static_cast<size_t>(Nk)) {
      throw std::runtime_error(
          "Window::get_hann_coeffs returned unexpected size in "
          "calculateFilters");
    }
    for (int n = 0; n < Nk; ++n) {
      T phase = static_cast<T>(2.0 * M_PI) * q_ * static_cast<T>(n) /
                static_cast<T>(Nk);
      std::complex<T> sinusoid = std::exp(std::complex<T>(0, phase));
      time_kernel[n] = window[n] * sinusoid / static_cast<T>(Nk);
    }

    // 2. Zero-pad
    std::vector<std::complex<T>> padded_kernel(N_fft, {0, 0});
    int start_idx = (N_fft - Nk) / 2;
    for (int n = 0; n < Nk; ++n) {
      if (start_idx + n >= 0 && start_idx + n < N_fft) {
        padded_kernel[start_idx + n] = time_kernel[n];
      }
    }

    // 3. Compute FFT
    if (static_cast<size_t>(N_fft) != fft_plan_->getSize()) {
      throw std::logic_error(
          "CQTPlan: calculateFilters requires FFT plan matching octave FFT "
          "length. Multi-plan support needed.");
    }
    std::vector<std::complex<T>>& freq_kernel = filter_bank_[k];
    freq_kernel.resize(N_fft);
    fft_plan_->fft(padded_kernel, freq_kernel, FFTNorm::Backward);

    // 4. Sparsify and Conjugate
    T max_abs = 0;
    for (const auto& val : freq_kernel) {
      max_abs = std::max(max_abs, std::abs(val));
    }
    if (max_abs == 0) max_abs = static_cast<T>(1.0);
    for (auto& val : freq_kernel) {
      if (std::abs(val) / max_abs <= sparsity_threshold_) val = {0, 0};
      val = std::conj(val);  // Store conjugate
    }
  }
  filters_calculated_ = true;
  std::cout << "[CQTPlan] Filter calculation complete." << std::endl;
}

// --- CQTPlan::execute ---
/**
 * @brief Computes the CQT of the input signal (basic block version).
 */
template <typename T>
void CQTPlan<T>::execute(const std::vector<std::complex<T>>& input,
                         std::vector<std::complex<T>>& output) {
  if (!filters_calculated_) calculateFilters();
  if (output.size() != static_cast<size_t>(n_bins_))
    throw std::invalid_argument("Output vector size must match n_bins");
  if (filter_bank_.size() != static_cast<size_t>(n_bins_))
    throw std::logic_error("Filter bank size mismatch");

  size_t N_fft = 0;  // Determine FFT size (assuming max length for now)
  if (fft_plan_)
    N_fft = fft_plan_->getSize();
  else if (!filter_bank_.empty() && !filter_bank_[0].empty())
    N_fft = filter_bank_[0].size();
  else
    throw std::logic_error("Cannot determine FFT size");

  if (!fft_plan_ || fft_plan_->getSize() != N_fft) {  // Ensure plan exists
    std::cout << "[CQTPlan] Creating/Resizing FFT Plan (size " << N_fft
              << ") for execution." << std::endl;
    try {
      fft_plan_ = std::make_unique<FFTPlan<T>>(N_fft);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          std::string("CQTPlan: Failed to create FFTPlan: ") + e.what());
    }
  }

  // 1. Prepare Input (Pad/Truncate)
  std::vector<std::complex<T>> processed_input = input;
  if (processed_input.size() != N_fft) {
    processed_input.resize(N_fft, {0, 0});
    if (input.size() > N_fft)
      std::cerr << "[CQTPlan WARNING] Input truncated to FFT length " << N_fft
                << std::endl;
  }

  // 2. Compute FFT of input
  std::vector<std::complex<T>> input_fft(N_fft);
  fft_plan_->fft(processed_input, input_fft, FFTNorm::Forward);

  // 3. Calculate CQT coefficients
  for (int k = 0; k < n_bins_; ++k) {
    if (filter_bank_[k].size() != N_fft)
      throw std::runtime_error("Filter/FFT size mismatch for bin " +
                               std::to_string(k));
    std::complex<T> cqt_coeff = {0, 0};
    for (size_t i = 0; i < N_fft; ++i) {
      cqt_coeff +=
          input_fft[i] * filter_bank_[k][i];  // filter_bank_ is pre-conjugated
    }
    output[k] = cqt_coeff / static_cast<T>(N_fft);  // Apply scaling
    // TODO: Apply final scaling factor
  }
}

// --- CQTPlan::setFilterBank ---
/**
 * @brief Sets a pre-computed filter bank.
 */
template <typename T>
void CQTPlan<T>::setFilterBank(
    const std::vector<std::vector<std::complex<T>>>& filter_bank) {
  if (filter_bank.size() != static_cast<size_t>(n_bins_)) {
    throw std::invalid_argument(
        "Provided filter bank size does not match n_bins");
  }
  filter_bank_ = filter_bank;
  filters_calculated_ = true;
  std::cout << "[CQTPlan] Precomputed filter bank set." << std::endl;
}

// --- Private Methods Implementations ---

template <typename T>
void CQTPlan<T>::calculateSingleOctaveCQT(
    const std::vector<std::complex<T>>& input_fft, int octave_num,
    std::vector<std::complex<T>>& output_cqt_octave) {
  // TODO: Implement recursive CQT logic for a single octave
  throw std::runtime_error("calculateSingleOctaveCQT not implemented yet.");
}

/**
 * @brief Internal helper to filter and downsample by 2 using the public API.
 * This method is intended for use within the recursive CQT calculation.
 * The output vector is passed by reference and will be modified.
 */
template <typename T>
void CQTPlan<T>::filterAndDownsampleBy2(const std::vector<T>& input,
                                        std::vector<T>& output,
                                        const std::vector<T>& filter_coeffs) {
  // This private method is named filterAndDownsampleBy2 for clarity within
  // CQTPlan's logic. It calls the public API function which handles the actual
  // operation.
  try {
    // Call the correctly named public API function from resample.h
    // Pass 2 as the downsampling factor.
    // Assign the returned vector to the output parameter.
    output = OmniDSP::filter_and_downsample<T>(input, filter_coeffs,
                                               2);  // <<< CORRECTED CALL
  } catch (const std::exception& e) {
    // Catch potential exceptions from the called function (e.g., invalid args)
    throw std::runtime_error(
        std::string("Error calling OmniDSP::filter_and_downsample<T>(..., "
                    "factor=2) within CQTPlan: ") +
        e.what());
  }
}

// --- Explicit Template Instantiation ---
template class CQTPlan<float>;
template class CQTPlan<double>;

}  // namespace OmniDSP
