/**
 * @file cqt.cpp
 * @brief Implements utility functions for creating Design::CQT from
 * Params::CQT.
 */
#include "OmniDSP/design/cqt.hpp"  // Corresponding header for Design::CQT and its operator<<

#include <spdlog/fmt/ostr.h>  // For logging custom types with operator<<

#include <algorithm>  // For std::sort, std::min, std::max, std::transform, std::unique
#include <bit>  // For std::bit_ceil (C++20)
#include <cmath>  // For std::log2, std::pow, std::round, std::ceil, std::abs, std::floor
#include <numeric>  // For std::iota
#include <sstream>  // For std::ostringstream
#include <stdexcept>  // For std::invalid_argument, std::runtime_error, std::logic_error
#include <string>   // For std::string in exceptions
#include <utility>  // For std::move
#include <vector>   // For std::vector

#include "OmniDSP/core_types.hpp"  // For OmniExpected, Status, OmniStatus and its operator<<
#include "OmniDSP/design.hpp"  // For the public declaration of Design::create
#include "OmniDSP/params/cqt.hpp"  // For Params::CQT and its operator<<
#include "OmniDSP/window.hpp"      // For WindowSetup and its operator<<
#include "spdlog/spdlog.h"

/**
 * @namespace OmniDSP::Design
 * @brief Contains structures and functions related to the design phase of DSP
 * components, translating high-level parameters into concrete specifications.
 */
namespace OmniDSP::Design {

  /**
   * @brief Creates a fully resolved Constant-Q Transform (CQT) design
   * specification from input parameters.
   * @details This function takes user-provided CQT parameters, calculates
   * necessary internal values like the Q-factor, bin frequencies, hop length,
   * and per-octave processing details (if applicable for multi-rate CQT). It
   * then constructs and returns a Design::CQT object.
   * @param params The input parameters for the CQT design
   * (OmniDSP::Params::CQT).
   * @return An OmniExpected<Design::CQT> containing the resolved design
   * specification on success, or an OmniStatus error code on failure.
   * @throws No direct exceptions; errors are returned via OmniExpected.
   */
  OmniExpected<Design::CQT> create(const Params::CQT& params)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Attempting to create Design::CQT from Params: {}", params);
    }

    // Params::CQT constructor already performed initial validation.

    // 1. Calculate Quality Factor (Q)
    if (params.bins_per_octave_ <= 0) {
      if (logger)
        logger->error(
            "Params::CQT has invalid bins_per_octave_ ({}). This should have "
            "been caught by Params constructor.",
            params.bins_per_octave_);
      return std::unexpected(OmniStatus::InvalidArgument);
    }
    double q_factor
        = 1.0
          / (std::pow(2.0, 1.0 / static_cast<double>(params.bins_per_octave_))
             - 1.0);
    if (q_factor <= 0) {
      if (logger)
        logger->error(
            "Calculated Q factor ({}) is not positive. Bins per octave: {}. "
            "Mathematical impossibility with valid bins_per_octave.",
            q_factor,
            params.bins_per_octave_);
      return std::unexpected(OmniStatus::Failure);
    }

    // 2. Determine all CQT bin frequencies
    std::vector<double> all_bin_frequencies;
    double current_freq = params.min_freq_;
    while (current_freq
           <= params.max_freq_
                  + 1e-6) {  // Add epsilon for floating point comparison
      if (current_freq
              >= params.min_freq_ - 1e-6  // Ensure it's not below min_freq due
                                          // to float issues
          && current_freq
                 < params.sample_rate_ / 2.0) {  // Ensure below Nyquist
        all_bin_frequencies.push_back(current_freq);
      }
      current_freq
          *= std::pow(2.0, 1.0 / static_cast<double>(params.bins_per_octave_));
      if (all_bin_frequencies.size() > 20000) {  // Increased safety break
        if (logger)
          logger->warn(
              "CQT bin generation exceeded 20000 bins, breaking. Check "
              "frequency range and bins_per_octave.");
        break;
      }
    }

    if (all_bin_frequencies.empty()) {
      if (logger)
        logger->error(
            "No CQT bins generated for the given parameters. MinFreq={}, "
            "MaxFreq={}, SR={}",
            params.min_freq_,
            params.max_freq_,
            params.sample_rate_);
      return std::unexpected(OmniStatus::InvalidArgument);
    }
    // Sort and unique, though the generation method should produce sorted
    // unique values already
    std::sort(all_bin_frequencies.begin(), all_bin_frequencies.end());
    all_bin_frequencies.erase(
        std::unique(all_bin_frequencies.begin(), all_bin_frequencies.end()),
        all_bin_frequencies.end());

    // 3. Determine Hop Length
    double longest_kernel_len_approx_samples
        = q_factor * params.sample_rate_ / all_bin_frequencies[0];
    size_t hop_length = static_cast<size_t>(std::max(
        1.0,
        std::round(
            longest_kernel_len_approx_samples
            / 4.0)));  // Example: 1/4 of longest kernel

    // 4. Determine Octaves and Per-Octave Specs
    std::vector<Design::CQTOctave> octave_specs;
    int num_octaves_to_process = 0;

    if (!all_bin_frequencies.empty()) {
      // Simplified single-octave processing logic for now
      num_octaves_to_process = 1;
      std::vector<double> current_octave_bins;
      std::vector<int> current_octave_kernel_lengths;
      size_t max_kernel_len_at_orig_sr = 0;

      for (double bin_freq : all_bin_frequencies) {
        if (bin_freq < params.sample_rate_ / 2.0) {
          current_octave_bins.push_back(bin_freq);
          int N_k = static_cast<int>(std::max(
              1.0, std::round(q_factor * params.sample_rate_ / bin_freq)));
          current_octave_kernel_lengths.push_back(N_k);
          if (static_cast<size_t>(N_k) > max_kernel_len_at_orig_sr) {
            max_kernel_len_at_orig_sr = N_k;
          }
        }
        else {
          if (logger)
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
          fft_len = std::bit_ceil(max_kernel_len_at_orig_sr);
          if (fft_len < 2 * max_kernel_len_at_orig_sr
              && max_kernel_len_at_orig_sr > 0)
            fft_len *= 2;
          if (fft_len < 32) fft_len = 32;
        }
        else {
          fft_len = 256;
        }

        WindowSetup octave_win_setup = params.window_setup_;
        octave_win_setup.length = 0;  // Signify that length varies per kernel

        octave_specs.emplace_back(
            params.sample_rate_,
            fft_len,
            current_octave_bins,
            current_octave_kernel_lengths,
            octave_win_setup);
      }
      else {
        num_octaves_to_process = 0;
        if (logger)
          logger->warn("No valid CQT bins found below Nyquist for processing.");
      }

      if (octave_specs.empty() && !all_bin_frequencies.empty()
          && num_octaves_to_process > 0) {
        if (logger)
          logger->error(
              "No valid processing octaves could be formed for CQT, though "
              "bins were generated.");
        return std::unexpected(OmniStatus::Failure);
      }
    }

    // 5. Construct and return Design::CQT
    try {
      Design::CQT
          spec(  // Use Design::CQT to avoid ambiguity if OmniDSP::CQT exists
              params.sample_rate_,
              params.min_freq_,
              params.max_freq_,
              params.bins_per_octave_,
              params.window_setup_,
              q_factor,
              hop_length,
              all_bin_frequencies,
              num_octaves_to_process,
              std::move(octave_specs));

      if (logger && logger->should_log(spdlog::level::debug)) {
        logger->debug("Successfully created Design::CQT: {}", spec);
      }
      return spec;
    }
    catch (const std::invalid_argument& e) {
      std::ostringstream msg_stream;
      msg_stream << "Invalid argument during Design::CQTOctave construction "
                    "within Design::CQT: "
                 << e.what();
      if (logger) logger->error(msg_stream.str());
      return std::unexpected(OmniStatus::InvalidArgument);
    }
    catch (const std::logic_error& e) {
      std::ostringstream msg_stream;
      msg_stream << "Logic error during Design::CQT construction: " << e.what();
      if (logger) logger->error(msg_stream.str());
      return std::unexpected(OmniStatus::Failure);
    }
    catch (const std::exception& e) {  // Catch any other standard exceptions
      std::ostringstream msg_stream;
      msg_stream << "Generic exception during Design::CQT construction: "
                 << e.what();
      if (logger) logger->error(msg_stream.str());
      return std::unexpected(OmniStatus::Failure);
    }
  }

}  // namespace OmniDSP::Design
