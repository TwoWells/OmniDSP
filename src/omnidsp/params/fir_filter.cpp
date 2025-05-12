/**
 * @file fir_filter.cpp
 * @brief Implements the constructor and fluent setters for FIRFilterParams.
 */

#include <spdlog/spdlog.h>  // Include spdlog for logging

#include <OmniDSP/params/fir_filter.hpp>  // Corresponding header
#include <OmniDSP/types/filter.hpp>  // For FilterType, FIRFilterDesignMethod
#include <OmniDSP/window.hpp>        // For WindowSetup related to default args
#include <optional>                  // For std::optional
#include <stdexcept>                 // For std::invalid_argument
#include <string>                    // For std::to_string, string concatenation
#include <utility>                   // For std::move

namespace OmniDSP {

  // Constructor (with spdlog integration)
  FIRFilterParams::FIRFilterParams(
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
      msg = "FIRFilterParams Constructor: sample_rate must be positive. Got "
            + std::to_string(sample_rate_);
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (cutoff1_ <= 0.0 || cutoff1_ >= sample_rate_ / 2.0) {
      msg = "FIRFilterParams Constructor: cutoff1 (" + std::to_string(cutoff1_)
            + ") is out of valid range (0, sample_rate/2 = "
            + std::to_string(sample_rate_ / 2.0) + ").";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }

    if (filter_type_ == FilterType::Bandpass
        || filter_type_ == FilterType::Bandstop) {
      if (!this->cutoff2_.has_value()) {
        msg
            = "FIRFilterParams Constructor: cutoff2 is required for "
              "Bandpass/Bandstop filters.";
        if (logger) logger->error(msg);
        throw std::invalid_argument(msg);
      }
      if (this->cutoff2_.value() <= 0.0
          || this->cutoff2_.value() >= sample_rate_ / 2.0) {
        msg = "FIRFilterParams Constructor: cutoff2 ("
              + std::to_string(this->cutoff2_.value())
              + ") is out of valid range (0, sample_rate/2 = "
              + std::to_string(sample_rate_ / 2.0) + ").";
        if (logger) logger->error(msg);
        throw std::invalid_argument(msg);
      }
      if (this->cutoff2_.value() <= this->cutoff1_) {
        msg = "FIRFilterParams Constructor: For Bandpass/Bandstop, cutoff2 ("
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
            = "FIRFilterParams Constructor: cutoff2 should only be specified "
              "for Bandpass/Bandstop filters.";
        if (logger)
          logger->warn(msg);  // Changed to warning as it might not be critical
        // Or throw: throw std::invalid_argument(msg);
      }
    }

    if (this->transition_width_.has_value()
        && this->transition_width_.value() <= 0.0) {
      msg
          = "FIRFilterParams Constructor: transition_width, if specified, must "
            "be positive. Got "
            + std::to_string(this->transition_width_.value());
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (this->stopband_attenuation_db_.has_value()
        && this->stopband_attenuation_db_.value() <= 0.0) {
      msg
          = "FIRFilterParams Constructor: stopband_attenuation_db, if "
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
          = "FIRFilterParams Constructor: Either 'order' must be specified, or "
            "both 'transition_width' and 'stopband_attenuation_db' must be "
            "specified for order estimation.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    // WindowSetup is validated by its own constructor.
    if (logger) logger->trace("FIRFilterParams constructed successfully.");
  }

  // Definition for num_taps
  std::optional<size_t> FIRFilterParams::num_taps() const
  {
    if (order_.has_value()) {
      return order_.value() + 1;
    }
    return std::nullopt;
  }

  // --- Fluent Setter Definitions ---

  FIRFilterParams& FIRFilterParams::filter_type(FilterType val)
  {
    filter_type_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger)
      logger->trace(
          "FIRFilterParams::filter_type to {}", static_cast<int>(val));
    // Add validation if type change invalidates cutoff2, or defer to a final
    // validate() call
    return *this;
  }

  FIRFilterParams& FIRFilterParams::sample_rate(double val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val <= 0.0) {
      std::string msg = "FIRFilterParams::sample_rate: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    sample_rate_ = val;
    if (logger) logger->trace("FIRFilterParams::sample_rate to {}", val);
    return *this;
  }

  FIRFilterParams& FIRFilterParams::cutoff1(double val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val <= 0.0) {
      std::string msg = "FIRFilterParams::cutoff1: value ("
                        + std::to_string(val) + ") must be positive.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    if (sample_rate_ > 0.0 && val >= sample_rate_ / 2.0) {
      std::string msg = "FIRFilterParams::cutoff1: value ("
                        + std::to_string(val)
                        + ") must be less than Nyquist frequency ("
                        + std::to_string(sample_rate_ / 2.0) + ").";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    cutoff1_ = val;
    if (logger) logger->trace("FIRFilterParams::cutoff1 to {}", val);
    return *this;
  }

  FIRFilterParams& FIRFilterParams::cutoff2(std::optional<double> val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val.has_value()) {
      if (val.value() <= 0.0) {
        std::string msg = "FIRFilterParams::cutoff2: value ("
                          + std::to_string(val.value())
                          + ") must be positive if specified.";
        if (logger) logger->error(msg);
        throw std::invalid_argument(msg);
      }
      if (sample_rate_ > 0.0 && val.value() >= sample_rate_ / 2.0) {
        std::string msg
            = "FIRFilterParams::cutoff2: value (" + std::to_string(val.value())
              + ") must be less than Nyquist frequency ("
              + std::to_string(sample_rate_ / 2.0) + ") if specified.";
        if (logger) logger->error(msg);
        throw std::invalid_argument(msg);
      }
      // Further validation based on type and cutoff1 could be added here or
      // deferred. Example: if (type == FilterType::Bandpass && val.value() <=
      // cutoff1) { ... }
    }
    cutoff2_ = val;
    if (logger) {
      if (val.has_value())
        logger->trace("FIRFilterParams::cutoff2 to {}", val.value());
      else
        logger->trace("FIRFilterParams::cutoff2 to nullopt");
    }
    return *this;
  }

  FIRFilterParams& FIRFilterParams::transition_width(std::optional<double> val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val.has_value() && val.value() <= 0.0) {
      std::string msg = "FIRFilterParams::transition_width: value ("
                        + std::to_string(val.value())
                        + ") must be positive if specified.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    transition_width_ = val;
    if (logger) {
      if (val.has_value())
        logger->trace("FIRFilterParams::transition_width to {}", val.value());
      else
        logger->trace("FIRFilterParams::transition_width to nullopt");
    }
    // If setting this, and order is also set, it might lead to ambiguity.
    // The constructor handles the logic that one or the other (or
    // characteristics) must be set. If order becomes set, this might become
    // irrelevant for order estimation.
    return *this;
  }

  FIRFilterParams& FIRFilterParams::stopband_attenuation_db(
      std::optional<double> val)
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (val.has_value() && val.value() <= 0.0) {
      std::string msg = "FIRFilterParams::stopband_attenuation_db: value ("
                        + std::to_string(val.value())
                        + ") must be positive if specified.";
      if (logger) logger->error(msg);
      throw std::invalid_argument(msg);
    }
    stopband_attenuation_db_ = val;
    if (logger) {
      if (val.has_value())
        logger->trace(
            "FIRFilterParams::stopband_attenuation_db to {}", val.value());
      else
        logger->trace("FIRFilterParams::stopband_attenuation_db to nullopt");
    }
    return *this;
  }

  FIRFilterParams& FIRFilterParams::order(std::optional<size_t> val)
  {
    order_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger) {
      if (val.has_value())
        logger->trace("FIRFilterParams::order to {}", val.value());
      else
        logger->trace("FIRFilterParams::order to nullopt");
    }
    // If order is set, transition_width and stopband_attenuation_db might
    // become irrelevant for order estimation.
    return *this;
  }

  FIRFilterParams& FIRFilterParams::window_setup(WindowSetup val)
  {
    // WindowSetup's constructor handles its own validation and logging.
    window_setup_ = std::move(val);
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger)
      logger->trace(
          "FIRFilterParams::window_setup to type {}",
          static_cast<int>(window_setup_.type));
    return *this;
  }

  FIRFilterParams& FIRFilterParams::design_method(FIRFilterDesignMethod val)
  {
    design_method_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    if (logger)
      logger->trace(
          "FIRFilterParams::design_method to {}", static_cast<int>(val));
    return *this;
  }

}  // namespace OmniDSP
