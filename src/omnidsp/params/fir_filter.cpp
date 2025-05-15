/**
 * @file fir_filter.cpp
 * @brief Implements the constructor and fluent setters for FIRFilterParams.
 */

#include "OmniDSP/params/fir_filter.hpp"  // Corresponding header

#include <spdlog/fmt/ostr.h>  // Include for custom ostream operator support with spdlog
#include <spdlog/spdlog.h>  // Include spdlog for logging

#include <optional>   // For std::optional
#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::to_string, string concatenation
#include <utility>    // For std::move

#include "OmniDSP/types/filter.hpp"  // For FilterType, FIRFilterDesignMethod
#include "OmniDSP/window.hpp"        // For WindowSetup related to default args

/**
 * @namespace OmniDSP::Params
 * @brief Contains structures for specifying parameters for various DSP
 * operations before full design.
 */
namespace OmniDSP::Params {

  /**
   * @brief Constructor for FIRFilter parameters.
   * @details Initializes and validates the parameters for designing an FIR
   * filter. At least the filter order or sufficient characteristics (transition
   * width and stopband attenuation) must be provided to determine the filter
   * order.
   * @param p_type Type of the filter (e.g., Lowpass, Highpass).
   * @param p_sample_rate Sample rate of the signal in Hz. Must be positive.
   * @param p_cutoff1 Primary cutoff frequency in Hz. Must be positive and less
   * than Nyquist.
   * @param p_cutoff2 Optional secondary cutoff frequency in Hz. Required for
   * Bandpass/Bandstop. Must be positive, less than Nyquist, and greater than
   * p_cutoff1.
   * @param p_order Optional desired filter order (number of taps - 1).
   * @param p_transition_width Optional desired transition width in Hz. Used for
   * order estimation if p_order is not set. Must be positive if specified.
   * @param p_stopband_attenuation_db Optional desired stopband attenuation in
   * positive dB. Used for order estimation if p_order is not set. Must be
   * positive if specified.
   * @param p_window_setup Windowing function setup. The `length` field is
   * typically determined internally.
   * @param p_design_method Method to be used for designing the filter.
   * @throws std::invalid_argument if essential parameters are invalid (e.g.,
   * non-positive sample rate, invalid cutoffs, insufficient information for
   * order determination, invalid transition width/attenuation).
   */
  FIRFilter::FIRFilter(
      FilterType p_type,
      double p_sample_rate,
      double p_cutoff1,
      std::optional<double> p_cutoff2,
      std::optional<size_t> p_order,
      std::optional<double> p_transition_width,
      std::optional<double> p_stopband_attenuation_db,
      WindowSetup p_window_setup,
      FIRFilterDesignMethod p_design_method)
      : filter_type_(p_type),
        sample_rate_(p_sample_rate),
        cutoff1_(p_cutoff1),
        cutoff2_(std::move(p_cutoff2)),
        transition_width_(std::move(p_transition_width)),
        stopband_attenuation_db_(std::move(p_stopband_attenuation_db)),
        order_(std::move(p_order)),
        window_setup_(std::move(p_window_setup)),
        design_method_(p_design_method)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    std::string msg;

    if (sample_rate_ <= 0.0) {
      msg = "Params::FIRFilter Constructor: sample_rate must be positive. Got "
            + std::to_string(sample_rate_);
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (cutoff1_ <= 0.0 || cutoff1_ >= sample_rate_ / 2.0) {
      msg = "Params::FIRFilter Constructor: cutoff1 ("
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
            = "Params::FIRFilter Constructor: cutoff2 is required for "
              "Bandpass/Bandstop filters.";
        if (logger) logger->error(msg);
        throw std::invalid_argument(msg);
      }
      if (this->cutoff2_.value() <= 0.0
          || this->cutoff2_.value() >= sample_rate_ / 2.0) {
        msg = "Params::FIRFilter Constructor: cutoff2 ("
              + std::to_string(this->cutoff2_.value())
              + ") is out of valid range (0, sample_rate/2 = "
              + std::to_string(sample_rate_ / 2.0) + ").";
        if (logger) logger->error(msg);
        throw std::invalid_argument(msg);
      }
      if (this->cutoff2_.value() <= this->cutoff1_) {
        msg = "Params::FIRFilter Constructor: For Bandpass/Bandstop, cutoff2 ("
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
            = "Params::FIRFilter Constructor: cutoff2 should only be specified "
              "for Bandpass/Bandstop filters.";
        if (logger)
          logger->warn(msg);  // Changed to warning as it might not be critical
        // Or throw: throw std::invalid_argument(msg);
      }
    }

    if (this->transition_width_.has_value()
        && this->transition_width_.value() <= 0.0) {
      msg
          = "Params::FIRFilter Constructor: transition_width, if specified, "
            "must "
            "be positive. Got "
            + std::to_string(this->transition_width_.value());
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (this->stopband_attenuation_db_.has_value()
        && this->stopband_attenuation_db_.value() <= 0.0) {
      msg
          = "Params::FIRFilter Constructor: stopband_attenuation_db, if "
            "specified, must be positive. Got "
            + std::to_string(this->stopband_attenuation_db_.value());
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }

    if (!this->order_.has_value()
        && !(
            this->transition_width_.has_value()
            && this->stopband_attenuation_db_.has_value())) {
      msg
          = "Params::FIRFilter Constructor: Either 'order' must be specified, "
            "or "
            "both 'transition_width' and 'stopband_attenuation_db' must be "
            "specified for order estimation.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    // WindowSetup is validated by its own constructor.
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::FIRFilter constructed: {}", *this);
    }
  }

  /**
   * @brief Calculates the number of taps for the FIR filter if the order is
   * specified.
   * @return An std::optional<size_t> containing the number of taps (order + 1)
   * if order_ is set, otherwise std::nullopt.
   */
  std::optional<size_t> FIRFilter::num_taps() const
  {
    if (order_.has_value()) {
      return order_.value() + 1;
    }
    return std::nullopt;
  }

  // --- Fluent Setter Definitions ---

  /**
   * @brief Sets the filter type.
   * @param val The new filter type (e.g., Lowpass, Highpass).
   * @return A reference to this Params::FIRFilter object for chaining.
   */
  FIRFilter& FIRFilter::filter_type(FilterType val)
  {
    filter_type_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::FIRFilter::filter_type updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the sample rate.
   * @param val The new sample rate in Hz. Must be positive.
   * @return A reference to this Params::FIRFilter object for chaining.
   * @throws std::invalid_argument if val is not positive.
   */
  FIRFilter& FIRFilter::sample_rate(double val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val <= 0.0) {
      std::string msg = "Params::FIRFilter::sample_rate: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    sample_rate_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::FIRFilter::sample_rate updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the primary cutoff frequency.
   * @param val The new primary cutoff frequency in Hz. Must be positive and
   * less than Nyquist.
   * @return A reference to this Params::FIRFilter object for chaining.
   * @throws std::invalid_argument if val is not positive or is >= Nyquist
   * frequency.
   */
  FIRFilter& FIRFilter::cutoff1(double val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val <= 0.0) {
      std::string msg = "Params::FIRFilter::cutoff1: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (sample_rate_ > 0.0 && val >= sample_rate_ / 2.0) {
      std::string msg = "Params::FIRFilter::cutoff1: value ("
                        + std::to_string(val)
                        + ") must be less than Nyquist frequency ("
                        + std::to_string(sample_rate_ / 2.0) + ").";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    cutoff1_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::FIRFilter::cutoff1 updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the secondary cutoff frequency (for Bandpass/Bandstop filters).
   * @param val An std::optional<double> containing the new secondary cutoff
   * frequency in Hz. If specified, must be positive and less than Nyquist.
   * @return A reference to this Params::FIRFilter object for chaining.
   * @throws std::invalid_argument if val has a value that is not positive or is
   * >= Nyquist frequency.
   */
  FIRFilter& FIRFilter::cutoff2(std::optional<double> val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val.has_value()) {
      if (val.value() <= 0.0) {
        std::string msg = "Params::FIRFilter::cutoff2: value ("
                          + std::to_string(val.value())
                          + ") must be positive if specified.";
        if (logger) logger->error(msg);
        throw std::invalid_argument(msg);
      }
      if (sample_rate_ > 0.0 && val.value() >= sample_rate_ / 2.0) {
        std::string msg = "Params::FIRFilter::cutoff2: value ("
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
      logger->trace("Params::FIRFilter::cutoff2 updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the desired transition width for filter order estimation.
   * @param val An std::optional<double> containing the new transition width in
   * Hz. Must be positive if specified.
   * @return A reference to this Params::FIRFilter object for chaining.
   * @throws std::invalid_argument if val has a value that is not positive.
   */
  FIRFilter& FIRFilter::transition_width(std::optional<double> val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val.has_value() && val.value() <= 0.0) {
      std::string msg = "Params::FIRFilter::transition_width: value ("
                        + std::to_string(val.value())
                        + ") must be positive if specified.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    transition_width_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::FIRFilter::transition_width updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the desired stopband attenuation for filter order estimation.
   * @param val An std::optional<double> containing the new stopband attenuation
   * in positive dB. Must be positive if specified.
   * @return A reference to this Params::FIRFilter object for chaining.
   * @throws std::invalid_argument if val has a value that is not positive.
   */
  FIRFilter& FIRFilter::stopband_attenuation_db(std::optional<double> val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val.has_value() && val.value() <= 0.0) {
      std::string msg = "Params::FIRFilter::stopband_attenuation_db: value ("
                        + std::to_string(val.value())
                        + ") must be positive if specified.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    stopband_attenuation_db_ = val;
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace(
          "Params::FIRFilter::stopband_attenuation_db updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the filter order directly.
   * @param val An std::optional<size_t> containing the new filter order (number
   * of taps - 1).
   * @return A reference to this Params::FIRFilter object for chaining.
   */
  FIRFilter& FIRFilter::order(std::optional<size_t> val)
  {
    order_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::FIRFilter::order updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the windowing function setup.
   * @details The `length` field of the WindowSetup is typically determined
   * internally by the design process.
   * @param val The new WindowSetup object. Its constructor handles its own
   * validation.
   * @return A reference to this Params::FIRFilter object for chaining.
   */
  FIRFilter& FIRFilter::window_setup(WindowSetup val)
  {
    window_setup_ = std::move(val);
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::FIRFilter::window_setup updated: {}", *this);
    }
    return *this;
  }

  /**
   * @brief Sets the FIR filter design method.
   * @param val The new FIRFilterDesignMethod.
   * @return A reference to this Params::FIRFilter object for chaining.
   */
  FIRFilter& FIRFilter::design_method(FIRFilterDesignMethod val)
  {
    design_method_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (logger && logger->should_log(spdlog::level::trace)) {
      logger->trace("Params::FIRFilter::design_method updated: {}", *this);
    }
    return *this;
  }

}  // namespace OmniDSP::Params
