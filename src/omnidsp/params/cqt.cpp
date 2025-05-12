/**
 * @file cqt.cpp
 * @brief Implements the constructor and fluent setters for CQTParams.
 */

#include "OmniDSP/params/cqt.hpp"  // Corresponding header

#include <spdlog/spdlog.h>  // Include spdlog for logging

#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::to_string, string concatenation
#include <utility>    // For std::move

namespace OmniDSP {

  CQTParams::CQTParams(
      double p_sample_rate,
      double p_min_freq,
      double p_max_freq,
      int p_bins_per_octave,
      WindowSetup p_window_setup)
      : sample_rate_(p_sample_rate),
        min_freq_(p_min_freq),
        max_freq_(p_max_freq),
        bins_per_octave_(p_bins_per_octave),
        window_setup_(std::move(p_window_setup))
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    std::string msg;

    if (sample_rate_ <= 0.0) {
      msg = "CQTParams Constructor: sample_rate ("
            + std::to_string(sample_rate_) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (min_freq_ <= 0.0) {
      msg = "CQTParams Constructor: min_freq (" + std::to_string(min_freq_)
            + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (max_freq_ <= 0.0) {
      msg = "CQTParams Constructor: max_freq (" + std::to_string(max_freq_)
            + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (min_freq_ >= max_freq_) {
      msg = "CQTParams Constructor: min_freq (" + std::to_string(min_freq_)
            + ") must be less than max_freq (" + std::to_string(max_freq_)
            + ").";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    double nyquist = sample_rate_ / 2.0;
    if (min_freq_ >= nyquist) {
      msg = "CQTParams Constructor: min_freq (" + std::to_string(min_freq_)
            + ") must be less than Nyquist frequency ("
            + std::to_string(nyquist) + ").";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (max_freq_ > nyquist) {
      if (logger)
        logger->warn(
            "CQTParams Constructor: max_freq ({}) is very close to or exceeds "
            "Nyquist ({}). This might lead to aliasing for highest CQT bins.",
            max_freq_,
            nyquist);
    }
    if (bins_per_octave_ <= 0) {
      msg = "CQTParams Constructor: bins_per_octave ("
            + std::to_string(bins_per_octave_) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    // WindowSetup is validated by its own constructor.
    if (logger)
      logger->trace(
          "CQTParams constructed: SR={}, MinFreq={}, MaxFreq={}, "
          "BinsPerOct={}, WinType={}",
          sample_rate_,
          min_freq_,
          max_freq_,
          bins_per_octave_,
          static_cast<int>(window_setup_.type));
  }

  // --- Fluent Setter Definitions ---

  CQTParams& CQTParams::sample_rate(double val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val <= 0.0) {
      std::string msg = "CQTParams::sample_rate: value (" + std::to_string(val)
                        + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    sample_rate_ = val;
    // Consider if re-validation against min_freq/max_freq is needed here.
    // For now, assuming individual setters validate their own direct
    // constraints. A full re-validation might be better done by calling the
    // constructor's logic or a separate validate() method if complex
    // interactions exist.
    if (logger) logger->trace("CQTParams::sample_rate to {}", val);
    return *this;
  }

  CQTParams& CQTParams::min_freq(double val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val <= 0.0) {
      std::string msg = "CQTParams::min_freq: value (" + std::to_string(val)
                        + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    // Example of cross-validation (could be more extensive or deferred)
    if (sample_rate_ > 0.0 && val >= sample_rate_ / 2.0) {
      std::string msg = "CQTParams::min_freq: value (" + std::to_string(val)
                        + ") must be less than Nyquist frequency ("
                        + std::to_string(sample_rate_ / 2.0) + ").";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (val >= max_freq_
        && max_freq_ > 0.0) {  // Check if max_freq is already set and positive
      std::string msg = "CQTParams::min_freq: value (" + std::to_string(val)
                        + ") must be less than max_freq ("
                        + std::to_string(max_freq_) + ").";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    min_freq_ = val;
    if (logger) logger->trace("CQTParams::min_freq to {}", val);
    return *this;
  }

  CQTParams& CQTParams::max_freq(double val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val <= 0.0) {
      std::string msg = "CQTParams::max_freq: value (" + std::to_string(val)
                        + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    // Example of cross-validation
    if (val <= min_freq_
        && min_freq_ > 0.0) {  // Check if min_freq is already set and positive
      std::string msg = "CQTParams::max_freq: value (" + std::to_string(val)
                        + ") must be greater than min_freq ("
                        + std::to_string(min_freq_) + ").";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (sample_rate_ > 0.0 && val > sample_rate_ / 2.0) {
      if (logger)
        logger->warn(
            "CQTParams::max_freq: value ({}) is very close to or exceeds "
            "Nyquist ({}). This might lead to aliasing.",
            val,
            sample_rate_ / 2.0);
    }
    max_freq_ = val;
    if (logger) logger->trace("CQTParams::max_freq to {}", val);
    return *this;
  }

  CQTParams& CQTParams::bins_per_octave(int val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val <= 0) {
      std::string msg = "CQTParams::bins_per_octave: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    bins_per_octave_ = val;
    if (logger) logger->trace("CQTParams::bins_per_octave to {}", val);
    return *this;
  }

  CQTParams& CQTParams::window_setup(WindowSetup val)
  {
    // WindowSetup's constructor handles its own validation and logging.
    window_setup_ = std::move(val);
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger)
      logger->trace(
          "CQTParams::window_setup to type {}",
          static_cast<int>(window_setup_.type));
    return *this;
  }

}  // namespace OmniDSP
