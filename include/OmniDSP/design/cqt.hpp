/**
 * @file cqt.hpp (design)
 * @brief Defines Design::CQT and Design::CQTOctave structures for CQT design
 * specifications.
 */
#ifndef OMNIDSP_DESIGN_CQT_HPP
#define OMNIDSP_DESIGN_CQT_HPP

#include <cstddef>    // For size_t
#include <ostream>    // For std::ostream
#include <sstream>    // For std::ostringstream
#include <stdexcept>  // For std::logic_error, std::invalid_argument
#include <string>
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/core_types.hpp"  // For OMNIDSP_EXPORT
#include "OmniDSP/window.hpp"      // For WindowSetup and its operator<<

// Include fmt headers for custom formatter specialization
#include <fmt/core.h>     // For basic formatting
#include <fmt/ostream.h>  // Specifically for ostream_formatter

namespace OmniDSP::Design {

  /**
   * @brief Design specification for a single octave in a Constant-Q Transform.
   * @details Holds parameters specific to processing one octave, such as its
   * unique sample rate (if multi-rate CQT is used), FFT length,
   * bin frequencies within that octave, kernel lengths, and windowing setup.
   */
  struct OMNIDSP_EXPORT CQTOctave {
    double
        octave_sample_rate;  ///< Sample rate used for processing this octave.
    size_t
        fft_length;  ///< FFT length used for this octave's spectral processing.
    std::vector<double>
        bin_frequencies;  ///< Frequencies of the CQT bins within this octave.
    std::vector<int>
        kernel_lengths_samples;  ///< Length (in samples) of each CQT analysis
                                 ///< kernel in this octave.
    WindowSetup
        window_setup_for_octave;  ///< Windowing setup (type and parameters)
                                  ///< applied to kernels in this octave.

    /**
     * @brief Explicit constructor for CQTOctave.
     * @param p_octave_sample_rate Sample rate for this octave.
     * @param p_fft_length FFT length for this octave.
     * @param p_bin_frequencies Vector of CQT bin frequencies in this octave.
     * @param p_kernel_lengths_samples Vector of kernel lengths for each bin.
     * @param p_window_setup_for_octave Window setup for kernels in this octave.
     * @throws std::invalid_argument if sample rate is not positive, FFT length
     * is zero with bins, or if there's a mismatch between bin frequencies and
     * kernel lengths.
     */
    explicit CQTOctave(
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
            "Design::CQTOctave: octave_sample_rate must be positive.");
      if (fft_length == 0 && !bin_frequencies.empty())
        throw std::invalid_argument(
            "Design::CQTOctave: fft_length cannot be zero if there are bins.");
      if (bin_frequencies.size() != kernel_lengths_samples.size())
        throw std::invalid_argument(
            "Design::CQTOctave: Mismatch between number of bin frequencies and "
            "kernel lengths.");
    }
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of
   * Design::CQTOctave.
   * @param os The output stream.
   * @param octave The Design::CQTOctave object to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, const CQTOctave& octave)
  {
    os << "CQTOctave(SR: " << octave.octave_sample_rate
       << ", FFTLen: " << octave.fft_length
       << ", NumBins: " << octave.bin_frequencies.size()
       << ", Win: " << octave.window_setup_for_octave;
    // Optionally print first few bin freqs/kernel lengths if needed for more
    // detail
    if (!octave.bin_frequencies.empty()) {
      os << ", FirstBinFreq: " << octave.bin_frequencies[0];
    }
    if (!octave.kernel_lengths_samples.empty()) {
      os << ", FirstKernelLen: " << octave.kernel_lengths_samples[0];
    }
    os << ")";
    return os;
  }

  /**
   * @brief Fully resolved design specification for a Constant-Q Transform
   * (CQT).
   * @details This structure holds all parameters necessary to configure a CQT
   * processor, including overall parameters and a breakdown of specifications
   * for each octave if a multi-rate approach is used.
   */
  struct OMNIDSP_EXPORT CQT {
    double original_sample_rate;  ///< Original sample rate of the input signal
                                  ///< in Hz.
    double min_freq;      ///< Minimum frequency for the CQT analysis in Hz.
    double max_freq;      ///< Maximum frequency for the CQT analysis in Hz.
    int bins_per_octave;  ///< Number of CQT bins per octave.
    WindowSetup
        global_window_setup;  ///< Global windowing setup, primarily used as a
                              ///< template for octave-specific windows.
    double quality_factor_q;  ///< The calculated quality factor Q for the CQT.
    size_t
        hop_length;  ///< Hop length in samples between consecutive CQT frames.
    std::vector<double>
        all_bin_frequencies;    ///< Vector of all unique CQT bin frequencies
                                ///< across all octaves.
    int num_octaves_processed;  ///< Number of octaves that will be processed.
    std::vector<CQTOctave>
        octave_designs;  ///< Vector of CQTOctave specifications, one for each
                         ///< processed octave.

    /**
     * @brief Explicit constructor for a CQT design specification.
     * @param p_original_sample_rate Original sample rate.
     * @param p_min_freq Minimum frequency.
     * @param p_max_freq Maximum frequency.
     * @param p_bins_per_octave Bins per octave.
     * @param p_global_window_setup Global window setup.
     * @param p_quality_factor_q Calculated Q factor.
     * @param p_hop_length Hop length in samples.
     * @param p_all_bin_frequencies Vector of all bin frequencies.
     * @param p_num_octaves_processed Number of octaves to be processed.
     * @param p_octave_designs Vector of CQTOctave design specifications.
     * @throws std::logic_error if parameters are inconsistent (e.g., invalid
     * rates/frequencies, zero bins/Q-factor/hop-length, mismatch in octave
     * counts).
     */
    explicit CQT(
        double p_original_sample_rate,
        double p_min_freq,
        double p_max_freq,
        int p_bins_per_octave,
        WindowSetup p_global_window_setup,
        double p_quality_factor_q,
        size_t p_hop_length,
        std::vector<double> p_all_bin_frequencies,
        int p_num_octaves_processed,
        std::vector<CQTOctave> p_octave_designs)
        : original_sample_rate(p_original_sample_rate),
          min_freq(p_min_freq),
          max_freq(p_max_freq),
          bins_per_octave(p_bins_per_octave),
          global_window_setup(std::move(p_global_window_setup)),
          quality_factor_q(p_quality_factor_q),
          hop_length(p_hop_length),
          all_bin_frequencies(std::move(p_all_bin_frequencies)),
          num_octaves_processed(p_num_octaves_processed),
          octave_designs(std::move(p_octave_designs))
    {
      if (original_sample_rate <= 0)
        throw std::logic_error(
            "Design::CQT: original_sample_rate must be positive.");
      if (min_freq <= 0 || max_freq <= 0 || min_freq >= max_freq)
        throw std::logic_error("Design::CQT: Invalid frequency range.");
      if (bins_per_octave <= 0)
        throw std::logic_error(
            "Design::CQT: bins_per_octave must be positive.");
      if (quality_factor_q <= 0)
        throw std::logic_error(
            "Design::CQT: quality_factor_q must be positive.");
      if (hop_length == 0
          && !all_bin_frequencies
                  .empty())  // Allow hop_length=0 if no bins (vacuously true)
        throw std::logic_error(
            "Design::CQT: hop_length cannot be zero if bins exist.");
      if (num_octaves_processed < 0)
        throw std::logic_error(
            "Design::CQT: num_octaves_processed cannot be negative.");
      if (static_cast<size_t>(num_octaves_processed) != octave_designs.size()
          && !all_bin_frequencies
                  .empty()) {  // Only check if there are bins expected
        if (!octave_designs.empty()
            || !all_bin_frequencies.empty()) {  // Further refine condition
          throw std::logic_error(
              "Design::CQT: Mismatch between num_octaves_processed and size of "
              "octave_designs vector.");
        }
      }
    }

    /**
     * @brief Gets the total number of CQT bins across all octaves.
     * @return The total number of bins.
     */
    [[nodiscard]] size_t get_total_num_bins() const
    {
      return all_bin_frequencies.size();
    }
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of Design::CQT.
   * @param os The output stream.
   * @param spec The Design::CQT object to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, const CQT& spec)
  {
    os << "Design::CQT(OrigSR: " << spec.original_sample_rate
       << ", MinF: " << spec.min_freq << ", MaxF: " << spec.max_freq
       << ", Bins/Oct: " << spec.bins_per_octave
       << ", Q: " << spec.quality_factor_q << ", Hop: " << spec.hop_length
       << ", TotalBins: " << spec.get_total_num_bins()
       << ", NumOctaves: " << spec.num_octaves_processed
       << ", GlobalWin: " << spec.global_window_setup;
    if (!spec.octave_designs.empty()) {
      os << ", FirstOctaveDesign: " << spec.octave_designs[0];
    }
    os << ")";
    return os;
  }

}  // namespace OmniDSP::Design

// Specialization of fmt::formatter for Design::CQTOctave and Design::CQT
template <>
struct fmt::formatter<OmniDSP::Design::CQTOctave> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::Design::CQT> : fmt::ostream_formatter {};

#endif  // OMNIDSP_DESIGN_CQT_HPP
