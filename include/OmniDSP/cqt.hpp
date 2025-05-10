/**
 * @file cqt.hpp
 * @brief Defines the interface for Constant-Q Transform (CQT) Plan objects and
 * specifications.
 */

#ifndef OMNIDSP_CQT_HPP
#define OMNIDSP_CQT_HPP

#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <string>   // For exception messages in CQTSpec constructor
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/core_types.hpp"  // For Status, OmniExpected, Utils::IsComplex_v, Utils::GetComplexType, F32, F64
#include "OmniDSP/window.hpp"  // For WindowSetup
#include "interface/backend.hpp"  // Defines Abstract::Backend and Abstract::CQTPlanImpl

// Forward declaration for CQTParams, defined in "OmniDSP/params/cqt.hpp"
namespace OmniDSP {
  struct CQTParams;
}

namespace OmniDSP {

  /**
   * @brief Specification for a single octave within a multi-rate CQT.
   * This contains parameters resolved for a specific octave's processing.
   */
  struct OMNIDSP_EXPORT CQTOctaveSpec {
    double octave_sample_rate;  ///< Effective sample rate for this octave
                                ///< (after downsampling).
    size_t fft_length;  ///< FFT length to be used for processing this octave.
    std::vector<double> bin_frequencies;  ///< Center frequencies of CQT bins
                                          ///< processed in this octave.
    std::vector<int>
        kernel_lengths_samples;  ///< Length (in samples at octave_sample_rate)
                                 ///< of the CQT analysis window for each bin in
                                 ///< this octave.
    WindowSetup window_setup_for_octave;  ///< The window setup to be used for
                                          ///< generating kernels in this
                                          ///< octave. Its `length` will be set
                                          ///< appropriately for each kernel.

    // Constructor for CQTOctaveSpec
    explicit CQTOctaveSpec(
        double p_octave_sample_rate,
        size_t p_fft_length,
        std::vector<double> p_bin_frequencies,
        std::vector<int> p_kernel_lengths_samples,
        WindowSetup p_window_setup_for_octave)
        : octave_sample_rate(p_octave_sample_rate),
          fft_length(p_fft_length),
          bin_frequencies(std::move(p_bin_frequencies)),
          kernel_lengths_samples(std::move(p_kernel_lengths_samples)),
          window_setup_for_octave(std::move(p_window_setup_for_octave))
    {
      if (octave_sample_rate <= 0)
        throw std::invalid_argument(
            "CQTOctaveSpec: octave_sample_rate must be positive.");
      if (fft_length == 0 && !bin_frequencies.empty())
        throw std::invalid_argument(
            "CQTOctaveSpec: fft_length cannot be zero if there are bins.");
      if (bin_frequencies.size() != kernel_lengths_samples.size())
        throw std::invalid_argument(
            "CQTOctaveSpec: Mismatch between number of bin frequencies and "
            "kernel lengths.");
      // WindowSetup is validated by its own constructor.
    }
  };

  /**
   * @brief Fully resolved specification for a Constant-Q Transform (CQT).
   * This struct is typically the output of `OmniDSP::Utils::create_spec(const
   * CQTParams&)` and serves as direct input for creating a `CQTPlan`.
   */
  struct OMNIDSP_EXPORT CQTSpec {
    // Original input parameters (for reference and some calculations)
    double original_sample_rate;  ///< Original sample rate of the input signal.
    double min_freq;              ///< Minimum frequency requested for CQT bins.
    double max_freq;              ///< Maximum frequency requested for CQT bins.
    int bins_per_octave;          ///< Number of CQT bins per octave.
    WindowSetup global_window_setup;  ///< The user-specified window setup (type
                                      ///< and parameters like beta). Specific
                                      ///< lengths are determined per kernel.

    // Derived parameters
    double quality_factor_q;  ///< Calculated quality factor Q = 1 /
                              ///< (2^(1/bins_per_octave) - 1).
    size_t hop_length;  ///< Hop length in samples (at original_sample_rate).
    std::vector<double> all_bin_frequencies;  ///< Sorted list of all unique CQT
                                              ///< bin center frequencies.
    int num_octaves_processed;  ///< Number of octaves that will actually be
                                ///< processed.
    std::vector<CQTOctaveSpec>
        octave_specs;  ///< Specifications for each processed octave (if
                       ///< multi-rate).

    /**
     * @brief Constructor for CQTSpec.
     * Primarily intended for internal use by `Utils::create_spec`.
     * Assumes parameters have been validated and resolved.
     */
    explicit CQTSpec(
        double p_original_sample_rate,
        double p_min_freq,
        double p_max_freq,
        int p_bins_per_octave,
        WindowSetup p_global_window_setup,
        double p_quality_factor_q,
        size_t p_hop_length,
        std::vector<double> p_all_bin_frequencies,
        int p_num_octaves_processed,
        std::vector<CQTOctaveSpec> p_octave_specs)
        : original_sample_rate(p_original_sample_rate),
          min_freq(p_min_freq),
          max_freq(p_max_freq),
          bins_per_octave(p_bins_per_octave),
          global_window_setup(std::move(p_global_window_setup)),
          quality_factor_q(p_quality_factor_q),
          hop_length(p_hop_length),
          all_bin_frequencies(std::move(p_all_bin_frequencies)),
          num_octaves_processed(p_num_octaves_processed),
          octave_specs(std::move(p_octave_specs))
    {
      // Basic internal consistency checks
      if (original_sample_rate <= 0)
        throw std::logic_error(
            "CQTSpec: original_sample_rate must be positive.");
      if (min_freq <= 0 || max_freq <= 0 || min_freq >= max_freq)
        throw std::logic_error("CQTSpec: Invalid frequency range.");
      if (bins_per_octave <= 0)
        throw std::logic_error("CQTSpec: bins_per_octave must be positive.");
      if (quality_factor_q <= 0)
        throw std::logic_error("CQTSpec: quality_factor_q must be positive.");
      if (hop_length == 0 && !all_bin_frequencies.empty())
        throw std::logic_error(
            "CQTSpec: hop_length cannot be zero if bins exist.");
      if (num_octaves_processed < 0)
        throw std::logic_error(
            "CQTSpec: num_octaves_processed cannot be negative.");
      if (static_cast<size_t>(num_octaves_processed) != octave_specs.size()
          && !all_bin_frequencies.empty()) {
        // Allow octave_specs to be empty if all_bin_frequencies is also empty
        // (e.g. no valid bins found)
        if (!octave_specs.empty() || !all_bin_frequencies.empty()) {
          throw std::logic_error(
              "CQTSpec: Mismatch between num_octaves_processed and size of "
              "octave_specs vector.");
        }
      }
      // global_window_setup is validated by its own constructor.
    }

    [[nodiscard]] size_t get_total_num_bins() const
    {
      return all_bin_frequencies.size();
    }
  };

  /**
   * @brief Plan object for executing Constant-Q Transforms (CQT).
   * @tparam T The underlying real floating-point type (e.g., F32, F64).
   */
  template <typename T>
  class OMNIDSP_EXPORT CQTPlan {
    static_assert(
        !Utils::IsComplex_v<T>, "CQTPlan requires a real type (F32 or F64).");
    using Complex = Utils::GetComplexType<T>;

   public:
    ~CQTPlan();
    CQTPlan(CQTPlan&& other) noexcept;
    CQTPlan& operator=(CQTPlan&& other) noexcept;
    CQTPlan(const CQTPlan&) = delete;
    CQTPlan& operator=(const CQTPlan&) = delete;

    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<Complex> output) const;
    size_t get_num_bins() const;  // Total number of CQT bins
    size_t get_num_output_frames(size_t input_length) const;
    size_t get_hop_length() const;  // Hop length at original sample rate

    // Factory method now takes CQTSpec
    [[nodiscard]] static OmniExpected<std::unique_ptr<CQTPlan<T>>> create(
        const Abstract::Backend& backend, const CQTSpec& spec);  // Updated

    static std::unique_ptr<CQTPlan<T>> create_from_impl(
        std::unique_ptr<Abstract::CQTPlanImpl<T>> pimpl);

   private:
    explicit CQTPlan(std::unique_ptr<Abstract::CQTPlanImpl<T>> pimpl);
    std::unique_ptr<Abstract::CQTPlanImpl<T>> pimpl_;
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_CQT_HPP
