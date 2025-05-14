/**
 * @file iir_filter.cpp
 * @brief Implements the constructor and fluent setters for Params::IIRFilter.
 */

#include "OmniDSP/params/iir_filter.hpp"  // Corresponding header

#include <spdlog/spdlog.h>  // Include spdlog for logging

#include <OmniDSP/types/filter.hpp>  // For FilterType, IIRFilterFormat
#include <stdexcept>                 // For std::invalid_argument
#include <string>                    // For std::to_string, string concatenation
#include <utility>                   // For std::move

namespace OmniDSP::Params {

  // Constructor (with spdlog integration)
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

    if (logger) {
      logger->trace(
          "Params::IIRFilter constructed: Type={}, SR={}, Order={}, "
          "Cutoff1={}, "
          "OutputFormat={}",
          static_cast<int>(filter_type_),
          sample_rate_,
          order_,
          cutoff1_,
          static_cast<int>(output_format_));
    }
  }

  // --- Fluent Setter Definitions ---

  IIRFilter& IIRFilter::filter_type(FilterType val)
  {
    filter_type_ = val;
    // Basic re-validation or warning if type change conflicts with cutoff2
    // might be added. For simplicity, detailed cross-validation is often
    // handled when creating the full Spec.
    auto logger = spdlog::get("OmniDSP");
    if (logger)
      logger->trace(
          "Params::IIRFilter::filter_type to {}", static_cast<int>(val));
    return *this;
  }

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
    if (logger) logger->trace("Params::IIRFilter::sample_rate to {}", val);
    return *this;
  }

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
    if (logger) logger->trace("Params::IIRFilter::order to {}", val);
    return *this;
  }

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
    if (logger) logger->trace("Params::IIRFilter::cutoff1 to {}", val);
    return *this;
  }

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
    if (logger) {
      if (val.has_value())
        logger->trace("Params::IIRFilter::cutoff2 to {}", val.value());
      else
        logger->trace("Params::IIRFilter::cutoff2 to nullopt");
    }
    return *this;
  }

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
    if (logger) {
      if (val.has_value())
        logger->trace(
            "Params::IIRFilter::passband_ripple_db to {}", val.value());
      else
        logger->trace("Params::IIRFilter::passband_ripple_db to nullopt");
    }
    return *this;
  }

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
    if (logger) {
      if (val.has_value())
        logger->trace(
            "Params::IIRFilter::stopband_attenuation_db to {}", val.value());
      else
        logger->trace("Params::IIRFilter::stopband_attenuation_db to nullopt");
    }
    return *this;
  }

  IIRFilter& IIRFilter::output_format(IIRFilterFormat val)
  {
    output_format_ = val;
    auto logger = spdlog::get("OmniDSP");
    if (logger)
      logger->trace(
          "Params::IIRFilter::output_format to {}", static_cast<int>(val));
    return *this;
  }

}  // namespace OmniDSP::Params
