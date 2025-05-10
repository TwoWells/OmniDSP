/**
 * @file cqt.cpp
 * @brief Implements the constructor for CQTParams.
 */

#include "OmniDSP/params/cqt.hpp"  // Corresponding header

#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::to_string, string concatenation
#include <utility>    // For std::move

// spdlog could be included here if detailed logging during construction is
// desired. #include "spdlog/spdlog.h"

namespace OmniDSP {

  CQTParams::CQTParams(
      double p_sample_rate,
      double p_min_freq,
      double p_max_freq,
      int p_bins_per_octave,
      WindowSetup p_window_setup)
      : sample_rate(p_sample_rate),
        min_freq(p_min_freq),
        max_freq(p_max_freq),
        bins_per_octave(p_bins_per_octave),
        window_setup(std::move(p_window_setup))  // WindowSetup is validated on
                                                 // its own construction
  {
    // auto logger = spdlog::get("OmniDSP"); // Example if logging
    // if (!logger) { logger = spdlog::default_logger(); }

    if (sample_rate <= 0.0) {
      throw std::invalid_argument(
          "CQTParams: sample_rate (" + std::to_string(sample_rate)
          + ") must be positive.");
    }
    if (min_freq <= 0.0) {
      throw std::invalid_argument(
          "CQTParams: min_freq (" + std::to_string(min_freq)
          + ") must be positive.");
    }
    if (max_freq <= 0.0) {
      throw std::invalid_argument(
          "CQTParams: max_freq (" + std::to_string(max_freq)
          + ") must be positive.");
    }
    if (min_freq >= max_freq) {
      throw std::invalid_argument(
          "CQTParams: min_freq (" + std::to_string(min_freq)
          + ") must be greater than or equal to max_freq ("
          + std::to_string(max_freq) + ").");
    }
    double nyquist = sample_rate / 2.0;
    if (min_freq >= nyquist) {
      throw std::invalid_argument(
          "CQTParams: min_freq (" + std::to_string(min_freq)
          + ") must be less than Nyquist frequency (" + std::to_string(nyquist)
          + ").");
    }
    if (max_freq
        > nyquist) {  // max_freq can be equal to Nyquist, but practically
                      // slightly less is better. For CQT, kernels extend, so
                      // max_freq should ideally be comfortably below Nyquist.
      // Consider a warning or stricter check if max_freq is too close to
      // Nyquist, as the highest frequency CQT kernels might alias. For now, a
      // simple check: if(logger) logger->warn("CQTParams: max_freq ({}) is very
      // close to or exceeds Nyquist ({}). This might lead to aliasing for
      // highest CQT bins.", max_freq, nyquist);
    }
    if (bins_per_octave <= 0) {
      throw std::invalid_argument(
          "CQTParams: bins_per_octave (" + std::to_string(bins_per_octave)
          + ") must be positive.");
    }

    // The `window_setup` member (type WindowSetup) is validated upon its own
    // construction. The `length` field within `p_window_setup` is typically set
    // to 0 in the default argument of CQTParams' constructor, as actual CQT
    // kernel lengths are frequency-dependent and determined later by
    // `Utils::create_spec(const CQTParams&)`.
  }

}  // namespace OmniDSP
