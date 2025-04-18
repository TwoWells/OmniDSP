#ifndef OMNIDSP_CQT_H
#define OMNIDSP_CQT_H

#include "omnidsp.h"  // Basic types and definitions
// #include "fft.h"  // Full definition might not be needed here, only in .cpp
// Forward declaration below is sufficient for unique_ptr member.
// Keep it included though, as other parts might rely on it implicitly.
#include <cmath>
#include <complex>
#include <memory>     // Needed for std::unique_ptr
#include <stdexcept>  // For exceptions
#include <vector>

#include "fft.h"

namespace OmniDSP {

// --- Forward Declarations ---
template <typename T>
class FFTPlan;  // <<< Added Forward Declaration

/**
 * @brief Constant-Q Transform (CQT) Plan Class.
 *
 * Handles the calculation of the Constant-Q Transform, which provides
 * logarithmically spaced frequency bins. This class manages the necessary
 * FFT plans and filter calculations.
 *
 * @tparam T The floating-point type (float or double).
 */
template <typename T>
class CQTPlan {
 public:
  /**
   * @brief Constructs a CQT plan.
   *
   * @param sample_rate The sample rate of the input signal in Hz.
   * @param n_bins The total number of CQT bins to compute.
   * @param bins_per_octave The number of bins per octave.
   * @param fmin The minimum frequency for the lowest CQT bin in Hz.
   * @param sparsity_threshold Threshold for sparsifying the kernel (default
   * 0.01).
   * @throws std::invalid_argument if parameters are invalid (e.g., sample_rate
   * <= 0).
   */
  CQTPlan(T sample_rate, int n_bins, int bins_per_octave, T fmin,
          T sparsity_threshold = static_cast<T>(0.01));

  /**
   * @brief Destructor (declared in case definition exists in .cpp).
   * Default implementation is likely sufficient due to unique_ptr.
   */
  ~CQTPlan();

  /**
   * @brief Executes the CQT on an input signal block.
   *
   * Calculates the CQT coefficients for the given input signal.
   * The output format depends on the internal implementation details
   * (e.g., could be a flattened vector or require specific interpretation).
   *
   * @param input The input signal (vector of complex numbers).
   * The size might need to match specific FFT lengths used internally.
   * @param output The output vector where CQT coefficients will be stored.
   * The size must be appropriate for the number of bins (n_bins).
   */
  void execute(const std::vector<std::complex<T>>& input,
               std::vector<std::complex<T>>& output);

  /**
   * @brief (Optional) Sets a precomputed filter bank.
   * Allows providing externally computed CQT filters (e.g., loaded from a
   * file). The format must match the internal requirements.
   * @param filter_bank A vector of vectors, where each inner vector is a filter
   * kernel in the frequency domain.
   */
  void setFilterBank(
      const std::vector<std::vector<std::complex<T>>>& filter_bank);

 private:
  // --- Configuration ---
  T sr_;                  // Sample rate
  int n_bins_;            // Total number of bins
  int bins_per_octave_;   // Bins per octave
  T f_min_;               // Minimum frequency
  int n_octaves_;         // Number of octaves
  T sparsity_threshold_;  // Sparsity threshold for kernel calculation
  T q_;                   // Quality factor

  // --- Internal State ---
  bool filters_calculated_;  // Flag indicating if filters are ready
  // Store the filter bank (e.g., frequency domain kernels)
  std::vector<std::vector<std::complex<T>>> filter_bank_;

  // FFT lengths required for different octaves might vary
  std::vector<int> fft_lengths_;

  // Store FFT plans needed for calculations. Using unique_ptr for ownership.
  // Forward declaration of FFTPlan allows unique_ptr usage here.
  std::unique_ptr<FFTPlan<T>> fft_plan_;  // Line ~88

  // --- Private Methods --- (Declarations only)
  void calculateFilters();
  void calculateSingleOctaveCQT(
      const std::vector<std::complex<T>>& input_fft, int octave_num,
      std::vector<std::complex<T>>& output_cqt_octave);
  void filterAndDownsampleBy2(const std::vector<T>& input,
                              std::vector<T>& output,
                              const std::vector<T>& filter_coeffs);
};

// Explicit template instantiation declaration needed if definitions are in .cpp
extern template class CQTPlan<float>;
extern template class CQTPlan<double>;

}  // namespace OmniDSP

#endif  // OMNIDSP_CQT_H
