/**
 * @file cqt.cpp
 * @brief Implements the constructor and fluent setters for Params::CQT.
 */

#include "OmniDSP/params/cqt.hpp"  // Corresponding header

#include <spdlog/fmt/ostr.h>  // Include for custom ostream operator support with spdlog
#include <spdlog/spdlog.h>  // Include spdlog for logging

#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::to_string, string concatenation
#include <utility>    // For std::move

/**
 * @namespace OmniDSP::Params
 * @brief Contains structures for specifying parameters for various DSP
 * operations before full design.
 */
namespace OmniDSP::Params {

  /**
   * @brief Constructor for CQT parameters.
   * @details Initializes and validates parameters for specifying a Constant-Q
   * Transform.
   * @param p_sample_rate Sample rate of the input signal in Hz. Must be
   * positive.
   * @param p_min_freq Minimum frequency for CQT bins in Hz. Must be positive
   * and less than Nyquist.
   * @param p_max_freq Maximum frequency for CQT bins in Hz. Must be greater
   * than p_min_freq and less than Nyquist.
   * @param p_bins_per_octave Number of CQT bins per octave. Must be positive.
   * @param p_window_setup Windowing function setup for CQT analysis windows.
   * The `length` field is typically ignored as kernel lengths are
   * frequency-dependent.
   * @throws std::invalid_argument if parameters are inconsistent or invalid
   * (e.g., non-positive rates, invalid frequency range, non-positive
   * bins_per_octave).
   */
  CQT::CQT(
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
      msg = "Params::CQT Constructor: sample_rate ("
            + std::to_string(sample_rate_) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (min_freq_ <= 0.0) {
      msg = "Params::CQT Constructor: min_freq (" + std::to_string(min_freq_)
            + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (max_freq_ <= 0.0) {
      msg = "Params::CQT Constructor: max_freq (" + std::to_string(max_freq_)
            + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (min_freq_ >= max_freq_) {
      msg = "Params::CQT Constructor: min_freq (" + std::to_string(min_freq_)
            + ") must be less than max_freq (" + std::to_string(max_freq_)
            + ").";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    double nyquist = sample_rate_ / 2.0;
    if (min_freq_ >= nyquist) {
      msg = "Params::CQT Constructor: min_freq (" + std::to_string(min_freq_)
            + ") must be less than Nyquist frequency ("
            + std::to_string(nyquist) + ").";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (max_freq_ > nyquist) {
      if (logger)
        logger->warn(
            "Params::CQT Constructor: max_freq ({}) is very close to or "
            "exceeds "
            "Nyquist ({}). This might lead to aliasing for highest CQT bins.",
            max_freq_,
            nyquist);
    }
    if (bins_per_octave_ <= 0) {
      msg = "Params::CQT Constructor: bins_per_octave ("
            + std::to_string(bins_per_octave_) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    // WindowSetup is validated by its own constructor.
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::CQT constructed: {}", *this);
    }
  }

  // --- Fluent Setter Definitions ---

  /**
   * @brief Sets the sample rate for the CQT.
   * @param val The new sample rate in Hz. Must be positive.
   * @return A reference to this Params::CQT object for chaining.
   * @throws std::invalid_argument if val is not positive.
   */
  CQT& CQT::sample_rate(double val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val <= 0.0) {
      std::string msg = "Params::CQT::sample_rate: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    sample_rate_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::CQT::sample_rate updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the minimum frequency for the CQT bins.
   * @param val The new minimum frequency in Hz. Must be positive and less than
   * current max_freq_ and Nyquist.
   * @return A reference to this Params::CQT object for chaining.
   * @throws std::invalid_argument if val is not positive or violates frequency
   * constraints.
   */
  CQT& CQT::min_freq(double val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val <= 0.0) {
      std::string msg = "Params::CQT::min_freq: value (" + std::to_string(val)
                        + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (sample_rate_ > 0.0 && val >= sample_rate_ / 2.0) {
      std::string msg = "Params::CQT::min_freq: value (" + std::to_string(val)
                        + ") must be less than Nyquist frequency ("
                        + std::to_string(sample_rate_ / 2.0) + ").";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (val >= max_freq_ && max_freq_ > 0.0) {
      std::string msg = "Params::CQT::min_freq: value (" + std::to_string(val)
                        + ") must be less than max_freq ("
                        + std::to_string(max_freq_) + ").";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    min_freq_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::CQT::min_freq updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the maximum frequency for the CQT bins.
   * @param val The new maximum frequency in Hz. Must be positive and greater
   * than current min_freq_.
   * @return A reference to this Params::CQT object for chaining.
   * @throws std::invalid_argument if val is not positive or violates frequency
   * constraints.
   */
  CQT& CQT::max_freq(double val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val <= 0.0) {
      std::string msg = "Params::CQT::max_freq: value (" + std::to_string(val)
                        + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (val <= min_freq_ && min_freq_ > 0.0) {
      std::string msg = "Params::CQT::max_freq: value (" + std::to_string(val)
                        + ") must be greater than min_freq ("
                        + std::to_string(min_freq_) + ").";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (sample_rate_ > 0.0 && val > sample_rate_ / 2.0) {
      if (logger)
        logger->warn(
            "Params::CQT::max_freq: value ({}) is very close to or exceeds "
            "Nyquist ({}). This might lead to aliasing.",
            val,
            sample_rate_ / 2.0);
    }
    max_freq_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::CQT::max_freq updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the number of CQT bins per octave.
   * @param val The new number of bins per octave. Must be positive.
   * @return A reference to this Params::CQT object for chaining.
   * @throws std::invalid_argument if val is not positive.
   */
  CQT& CQT::bins_per_octave(int val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val <= 0) {
      std::string msg = "Params::CQT::bins_per_octave: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    bins_per_octave_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::CQT::bins_per_octave updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the windowing function setup for CQT analysis windows.
   * @param val The new WindowSetup object. Its constructor handles validation.
   * @return A reference to this Params::CQT object for chaining.
   */
  CQT& CQT::window_setup(WindowSetup val)
  {
    window_setup_ = std::move(val);
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::CQT::window_setup updated: {}", *this);
    }
    return *this;
  }

}  // namespace OmniDSP::Params
