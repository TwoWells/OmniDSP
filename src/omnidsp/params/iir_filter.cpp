/**
 * @file iir_filter.cpp
 * @brief Implements the constructor and fluent setters for Params::IIRFilter.
 */

#include "OmniDSP/params/iir_filter.hpp"  // Corresponding header

#include <spdlog/fmt/ostr.h>  // Include for custom ostream operator support with spdlog
#include <spdlog/spdlog.h>  // Include spdlog for logging

#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::to_string, string concatenation
#include <utility>    // For std::move

#include "OmniDSP/types/filter.hpp"  // For FilterType, IIRFilterFormat

/**
 * @namespace OmniDSP::Params
 * @brief Contains structures for specifying parameters for various DSP
 * operations before full design.
 */
namespace OmniDSP::Params {

  /**
   * @brief Constructor for IIRFilter parameters.
   * @details Initializes and validates parameters for designing an IIR filter.
   * @param p_type Type of the filter (e.g., Lowpass, Highpass).
   * @param p_sample_rate Sample rate of the signal in Hz. Must be positive.
   * @param p_order Desired filter order. Must be positive.
   * @param p_cutoff1 Primary cutoff frequency in Hz. Must be positive and less
   * than Nyquist.
   * @param p_cutoff2 Optional secondary cutoff frequency in Hz. Required for
   * Bandpass/Bandstop. Must be positive, less than Nyquist, and greater than
   * p_cutoff1 if specified.
   * @param p_passband_ripple_db Optional desired passband ripple in positive
   * dB. Must be positive if specified.
   * @param p_stopband_attenuation_db Optional desired stopband attenuation in
   * positive dB. Must be positive if specified.
   * @param p_output_format Desired output format for coefficients (SOS or
   * TransferFunction).
   * @throws std::invalid_argument if essential parameters are invalid (e.g.,
   * non-positive sample rate or order, invalid cutoffs, negative
   * ripple/attenuation).
   */
  IIRFilter::IIRFilter(
      FilterType p_type,
      double p_sample_rate,
      size_t p_order,
      double p_cutoff1,
      std::optional<double> p_cutoff2,
      std::optional<double> p_passband_ripple_db,
      std::optional<double> p_stopband_attenuation_db,
      IIRFilterFormat p_output_format)
      : filter_type_(p_type),
        sample_rate_(p_sample_rate),
        order_(p_order),
        cutoff1_(p_cutoff1),
        cutoff2_(std::move(p_cutoff2)),
        passband_ripple_db_(std::move(p_passband_ripple_db)),
        stopband_attenuation_db_(std::move(p_stopband_attenuation_db)),
        output_format_(p_output_format)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();  // Fallback logger
    }
    std::string msg;

    if (sample_rate_ <= 0.0) {
      msg = "Params::IIRFilter Constructor: sample_rate ("
            + std::to_string(sample_rate_) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (order_ == 0) {
      msg = "Params::IIRFilter Constructor: order (" + std::to_string(order_)
            + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (cutoff1_ <= 0.0 || cutoff1_ >= sample_rate_ / 2.0) {
      msg = "Params::IIRFilter Constructor: cutoff1 ("
            + std::to_string(cutoff1_)
            + ") is out of valid range (0, sample_rate/2 = "
            + std::to_string(sample_rate_ / 2.0) + ").";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }

    if (filter_type_ == FilterType::Bandpass
        || filter_type_ == FilterType::Bandstop) {
      if (!this->cutoff2_.has_value()) {
        msg
            = "Params::IIRFilter Constructor: cutoff2 is required for "
              "Bandpass/Bandstop filters.";
        if (logger) logger->error(msg);
        throw std::invalid_argument(msg);
      }
      if (this->cutoff2_.value() <= 0.0
          || this->cutoff2_.value() >= sample_rate_ / 2.0) {
        msg = "Params::IIRFilter Constructor: cutoff2 ("
              + std::to_string(this->cutoff2_.value())
              + ") is out of valid range (0, sample_rate/2 = "
              + std::to_string(sample_rate_ / 2.0) + ").";
        if (logger) logger->error(msg);
        throw std::invalid_argument(msg);
      }
      if (this->cutoff2_.value() <= this->cutoff1_) {
        msg = "Params::IIRFilter Constructor: For Bandpass/Bandstop, cutoff2 ("
              + std::to_string(this->cutoff2_.value())
              + ") must be greater than cutoff1 ("
              + std::to_string(this->cutoff1_) + ").";
        if (logger) logger->error(msg);
        throw std::invalid_argument(msg);
      }
    }
    else {
      if (this->cutoff2_.has_value()) {
        msg
            = "Params::IIRFilter Constructor: cutoff2 should only be specified "
              "for Bandpass/Bandstop filters.";
        if (logger)
          logger->warn(
              msg);  // Warning, as it might not be critical for some designs
      }
    }

    if (this->passband_ripple_db_.has_value()
        && this->passband_ripple_db_.value() <= 0.0) {
      msg
          = "Params::IIRFilter Constructor: passband_ripple_db, if specified, "
            "must be positive. Got "
            + std::to_string(this->passband_ripple_db_.value());
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (this->stopband_attenuation_db_.has_value()
        && this->stopband_attenuation_db_.value() <= 0.0) {
      msg
          = "Params::IIRFilter Constructor: stopband_attenuation_db, if "
            "specified, must be positive. Got "
            + std::to_string(this->stopband_attenuation_db_.value());
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }

    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::IIRFilter constructed: {}", *this);
    }
  }

  // --- Fluent Setter Definitions ---

  /**
   * @brief Sets the filter type.
   * @param val The new filter type.
   * @return A reference to this Params::IIRFilter object for chaining.
   */
  IIRFilter& IIRFilter::filter_type(FilterType val)
  {
    filter_type_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::IIRFilter::filter_type updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the sample rate.
   * @param val The new sample rate in Hz. Must be positive.
   * @return A reference to this Params::IIRFilter object for chaining.
   * @throws std::invalid_argument if val is not positive.
   */
  IIRFilter& IIRFilter::sample_rate(double val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val <= 0.0) {
      std::string msg = "Params::IIRFilter::sample_rate: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    sample_rate_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::IIRFilter::sample_rate updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the filter order.
   * @param val The new filter order. Must be positive.
   * @return A reference to this Params::IIRFilter object for chaining.
   * @throws std::invalid_argument if val is zero.
   */
  IIRFilter& IIRFilter::order(size_t val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val == 0) {
      std::string msg = "Params::IIRFilter::order: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    order_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::IIRFilter::order updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the primary cutoff frequency.
   * @param val The new primary cutoff frequency in Hz. Must be positive and
   * less than Nyquist.
   * @return A reference to this Params::IIRFilter object for chaining.
   * @throws std::invalid_argument if val is not positive or is >= Nyquist
   * frequency.
   */
  IIRFilter& IIRFilter::cutoff1(double val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val <= 0.0) {
      std::string msg = "Params::IIRFilter::cutoff1: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (sample_rate_ > 0.0 && val >= sample_rate_ / 2.0) {
      std::string msg = "Params::IIRFilter::cutoff1: value ("
                        + std::to_string(val)
                        + ") must be less than Nyquist frequency ("
                        + std::to_string(sample_rate_ / 2.0) + ").";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    cutoff1_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::IIRFilter::cutoff1 updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the secondary cutoff frequency.
   * @param val An std::optional<double> containing the new secondary cutoff
   * frequency in Hz. If specified, must be positive and less than Nyquist.
   * @return A reference to this Params::IIRFilter object for chaining.
   * @throws std::invalid_argument if val has a value that is not positive or is
   * >= Nyquist frequency.
   */
  IIRFilter& IIRFilter::cutoff2(std::optional<double> val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val.has_value()) {
      if (val.value() <= 0.0) {
        std::string msg = "Params::IIRFilter::cutoff2: value ("
                          + std::to_string(val.value())
                          + ") must be positive if specified.";
        if (logger) logger->error(msg);
        throw std::invalid_argument(msg);
      }
      if (sample_rate_ > 0.0 && val.value() >= sample_rate_ / 2.0) {
        std::string msg = "Params::IIRFilter::cutoff2: value ("
                          + std::to_string(val.value())
                          + ") must be less than Nyquist frequency ("
                          + std::to_string(sample_rate_ / 2.0)
                          + ") if specified.";
        if (logger) logger->error(msg);
        throw std::invalid_argument(msg);
      }
    }
    cutoff2_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::IIRFilter::cutoff2 updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the passband ripple.
   * @param val An std::optional<double> containing the new passband ripple in
   * positive dB. Must be positive if specified.
   * @return A reference to this Params::IIRFilter object for chaining.
   * @throws std::invalid_argument if val has a value that is not positive.
   */
  IIRFilter& IIRFilter::passband_ripple_db(std::optional<double> val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val.has_value() && val.value() <= 0.0) {
      std::string msg = "Params::IIRFilter::passband_ripple_db: value ("
                        + std::to_string(val.value())
                        + ") must be positive if specified.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    passband_ripple_db_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::IIRFilter::passband_ripple_db updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the stopband attenuation.
   * @param val An std::optional<double> containing the new stopband attenuation
   * in positive dB. Must be positive if specified.
   * @return A reference to this Params::IIRFilter object for chaining.
   * @throws std::invalid_argument if val has a value that is not positive.
   */
  IIRFilter& IIRFilter::stopband_attenuation_db(std::optional<double> val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val.has_value() && val.value() <= 0.0) {
      std::string msg = "Params::IIRFilter::stopband_attenuation_db: value ("
                        + std::to_string(val.value())
                        + ") must be positive if specified.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    stopband_attenuation_db_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "Params::IIRFilter::stopband_attenuation_db updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the desired output format for filter coefficients.
   * @param val The new IIRFilterFormat (e.g., SOS, TransferFunction).
   * @return A reference to this Params::IIRFilter object for chaining.
   */
  IIRFilter& IIRFilter::output_format(IIRFilterFormat val)
  {
    output_format_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::IIRFilter::output_format updated: {}", *this);
    }
    return *this;
  }

}  // namespace OmniDSP::Params
