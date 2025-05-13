/**
 * @file cqt_design.cpp
 * @brief Implements utility functions for creating Design::CQT from CQTParams.
 */
#include <algorithm>  // For std::sort, std::min, std::max, std::transform, std::unique
#include <bit>  // For std::bit_ceil (C++20)
#include <cmath>  // For std::log2, std::pow, std::round, std::ceil, std::abs, std::floor
#include <numeric>    // For std::iota
#include <stdexcept>  // For std::invalid_argument, std::runtime_error
#include <string>
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/core_types.hpp"  // For OmniExpected, Status
#include "OmniDSP/cqt.hpp"         // For Design::CQT, Design::CQTOctave
#include "OmniDSP/params/cqt.hpp"
#include "OmniDSP/utils.hpp"  // For the public declaration of Utils::create_spec
#include "OmniDSP/window.hpp"  // For WindowSetup
#include "spdlog/spdlog.h"

namespace OmniDSP::Utils {

  OmniExpected<Design::CQT> create_spec(const CQTParams& params)
  {
    // CQTParams constructor already performed initial validation.
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    // 1. Calculate Quality Factor (Q)
    // Q = 1 / (2^(1/B) - 1), where B is bins_per_octave
    if (params.bins_per_octave_
        <= 0) {  // Should be caught by CQTParams constructor
      return std::unexpected(Status::InvalidArgument);
    }
    double q_factor
        = 1.0
          / (std::pow(2.0, 1.0 / static_cast<double>(params.bins_per_octave_))
             - 1.0);
    if (q_factor <= 0) {
      logger->error(
          "Calculated Q factor ({}) is not positive. Bins per octave: {}",
          q_factor,
          params.bins_per_octave_);
      return std::unexpected(Status::Failure);  // Mathematical impossibility
                                                // with valid bins_per_octave
    }

    // 2. Determine all CQT bin frequencies
    std::vector<double> all_bin_frequencies;
    // int k_min =
    // static_cast<int>(std::ceil(static_cast<double>(params.bins_per_octave) *
    // std::log2(params.min_freq / params.min_freq))); // k for f_min is 0
    // relative to f_min

    // Find the lowest frequency actually used, which might be slightly
    // different from params.min_freq if we align to a reference frequency (e.g.
    // A4=440Hz) for musical applications. For simplicity, we'll start directly
    // from params.min_freq. A more advanced approach might adjust min_freq to
    // align with a base like C0.

    double current_freq = params.min_freq_;
    while (current_freq
           <= params.max_freq_
                  + 1e-6) {  // Add epsilon for floating point comparison
      if (current_freq >= params.min_freq_ - 1e-6
          && current_freq < params.sample_rate_ / 2.0) {
        all_bin_frequencies.push_back(current_freq);
      }
      current_freq
          *= std::pow(2.0, 1.0 / static_cast<double>(params.bins_per_octave_));
      if (all_bin_frequencies.size() > 10000) {  // Safety break
        logger->warn(
            "CQT bin generation exceeded 10000 bins, breaking. Check frequency "
            "range and bins_per_octave.");
        break;
      }
    }

    if (all_bin_frequencies.empty()) {
      logger->error(
          "No CQT bins generated for the given parameters. MinFreq={}, "
          "MaxFreq={}, SR={}",
          params.min_freq_,
          params.max_freq_,
          params.sample_rate_);
      return std::unexpected(Status::InvalidArgument);
    }
    std::sort(all_bin_frequencies.begin(), all_bin_frequencies.end());
    all_bin_frequencies.erase(
        std::unique(all_bin_frequencies.begin(), all_bin_frequencies.end()),
        all_bin_frequencies.end());

    // 3. Determine Hop Length (heuristic based on the lowest frequency kernel)
    // Kernel length N_k = Q * fs / f_k
    // Smallest f_k is all_bin_frequencies[0]. Longest N_k is at
    // all_bin_frequencies[0]. Hop length can be a fraction of the longest
    // kernel.
    double longest_kernel_len_approx_samples
        = q_factor * params.sample_rate_ / all_bin_frequencies[0];
    size_t hop_length = static_cast<size_t>(std::max(
        1.0,
        std::round(
            longest_kernel_len_approx_samples
            / 4.0)));  // e.g., 1/4 of lowest kernel
    // A fixed hop length (e.g., 512 samples) is also common. This could be a
    // param.

    // 4. Determine Octaves and Per-Octave Specs (for multi-rate CQT)
    std::vector<Design::CQTOctave> octave_specs;
    int num_octaves_to_process = 0;

    if (!all_bin_frequencies.empty()) {
      // double overall_max_freq_in_bins = all_bin_frequencies.back(); // Not
      // directly used in simplified logic double current_octave_min_freq =
      // all_bin_frequencies[0]; // Not directly used in simplified logic

      // Simplified single-octave processing logic for now
      // (Can be expanded for true multi-rate later)
      num_octaves_to_process = 1;  // Assume one effective processing "octave"
      std::vector<double> current_octave_bins;
      std::vector<int> current_octave_kernel_lengths;
      size_t max_kernel_len_at_orig_sr = 0;

      for (double bin_freq : all_bin_frequencies) {
        if (bin_freq
            < params.sample_rate_
                  / 2.0) {  // Ensure below Nyquist of the original sample rate
          current_octave_bins.push_back(bin_freq);
          int N_k = static_cast<int>(std::max(
              1.0, std::round(q_factor * params.sample_rate_ / bin_freq)));
          current_octave_kernel_lengths.push_back(N_k);
          if (static_cast<size_t>(N_k) > max_kernel_len_at_orig_sr) {
            max_kernel_len_at_orig_sr = N_k;
          }
        }
        else {
          logger->warn(
              "CQT bin frequency {} Hz is at or above Nyquist ({} Hz), "
              "skipping.",
              bin_freq,
              params.sample_rate_ / 2.0);
        }
      }

      if (!current_octave_bins.empty()) {
        size_t fft_len = 0;
        if (max_kernel_len_at_orig_sr > 0) {
          fft_len = std::bit_ceil(max_kernel_len_at_orig_sr);  // C++20
          // Ensure FFT is large enough for good resolution if using FFT for
          // convolution
          if (fft_len < 2 * max_kernel_len_at_orig_sr
              && max_kernel_len_at_orig_sr > 0)
            fft_len *= 2;
          if (fft_len < 32) fft_len = 32;  // Minimum practical FFT length
        }
        else {
          fft_len = 256;  // Default if no kernels (should not happen if bins
                          // exist and are valid)
        }

        WindowSetup octave_win_setup
            = params.window_setup_;  // Corrected: use params.window_setup
        // Length in octave_win_setup is per-kernel, not a single value for the
        // octave. The Design::CQTOctave's window_setup_for_octave is the
        // *template* (type, params). Actual window generation uses
        // kernel_lengths_samples[j].
        octave_win_setup.length = 0;  // Signify that length varies per kernel

        octave_specs.emplace_back(
            params.sample_rate_,  // Effective SR for this "single" octave
                                  // processing
            fft_len,
            current_octave_bins,  // Use the filtered list of bins for this
                                  // octave
            current_octave_kernel_lengths,
            octave_win_setup);
      }
      else {
        num_octaves_to_process = 0;  // No valid bins to process
        logger->warn("No valid CQT bins found below Nyquist for processing.");
      }

      if (octave_specs.empty() && !all_bin_frequencies.empty()
          && num_octaves_to_process > 0) {
        logger->error(
            "No valid processing octaves could be formed for CQT, though bins "
            "were generated.");
        return std::unexpected(Status::Failure);
      }
    }

    // 5. Construct and return Design::CQT
    try {
      Design::CQT spec(
          params.sample_rate_,
          params.min_freq_,
          params.max_freq_,
          params.bins_per_octave_,
          params.window_setup_,  // Pass the original window_setup from params
          q_factor,
          hop_length,
          all_bin_frequencies,  // This will be the filtered list if some bins
                                // were above Nyquist
          num_octaves_to_process,
          std::move(octave_specs));
      return spec;
    }
    catch (const std::invalid_argument&
               e) {  // Catch invalid_arguments from
                     // Design::CQTOctave constructor FIRST
      logger->error(
          "Invalid argument during Design::CQTOctave construction within "
          "Design::CQT: "
          "{}",
          e.what());
      return std::unexpected(Status::InvalidArgument);
    }
    catch (const std::logic_error& e) {  // Catch logic_errors from Design::CQT
                                         // constructor (more general)
      logger->error(
          "Logic error during Design::CQT construction: {}", e.what());
      return std::unexpected(Status::Failure);
    }
  }

}  // namespace OmniDSP::Utils
