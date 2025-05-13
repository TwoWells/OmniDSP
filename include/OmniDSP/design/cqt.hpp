/**
 * @file cqt.hpp (design)
 * @brief Defines Design::CQT and Design::CQTOctave structures for CQT design
 * specifications.
 */
#ifndef OMNIDSP_DESIGN_CQT_HPP
#define OMNIDSP_DESIGN_CQT_HPP

#include <cstddef>    // For size_t
#include <stdexcept>  // For std::logic_error, std::invalid_argument
#include <string>
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/core_types.hpp"  // For OMNIDSP_EXPORT
#include "OmniDSP/window.hpp"      // For WindowSetup

namespace OmniDSP::Design {

  struct OMNIDSP_EXPORT CQTOctave {
    double octave_sample_rate;
    size_t fft_length;
    std::vector<double> bin_frequencies;
    std::vector<int> kernel_lengths_samples;
    WindowSetup window_setup_for_octave;

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

  struct OMNIDSP_EXPORT CQT {
    double original_sample_rate;
    double min_freq;
    double max_freq;
    int bins_per_octave;
    WindowSetup global_window_setup;
    double quality_factor_q;
    size_t hop_length;
    std::vector<double> all_bin_frequencies;
    int num_octaves_processed;
    std::vector<CQTOctave> octave_designs;

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
      if (hop_length == 0 && !all_bin_frequencies.empty())
        throw std::logic_error(
            "Design::CQT: hop_length cannot be zero if bins exist.");
      if (num_octaves_processed < 0)
        throw std::logic_error(
            "Design::CQT: num_octaves_processed cannot be negative.");
      if (static_cast<size_t>(num_octaves_processed) != octave_designs.size()
          && !all_bin_frequencies.empty()) {
        if (!octave_designs.empty() || !all_bin_frequencies.empty()) {
          throw std::logic_error(
              "Design::CQT: Mismatch between num_octaves_processed and size of "
              "octave_designs vector.");
        }
      }
    }

    [[nodiscard]] size_t get_total_num_bins() const
    {
      return all_bin_frequencies.size();
    }
  };

}  // namespace OmniDSP::Design

#endif  // OMNIDSP_DESIGN_CQT_HPP
