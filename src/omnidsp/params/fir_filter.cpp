/**
 * @file fir_filter.cpp
 * @brief Implements the constructor for FIRFilterParams.
 */

#include <OmniDSP/params/fir_filter.hpp>  // Corresponding header
#include <OmniDSP/types/filter.hpp>  // For FilterType, FIRFilterDesignMethod (needed by constructor args)
#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::to_string, string concatenation
#include <utility>    // For std::move

// Potentially include other headers if validation logic grows more complex

namespace OmniDSP {

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
      : type(p_type),
        sample_rate(p_sample_rate),
        cutoff1(p_cutoff1),
        cutoff2(std::move(p_cutoff2)),
        transition_width(std::move(p_transition_width)),
        stopband_attenuation_db(std::move(p_stopband_attenuation_db)),
        order(std::move(p_order)),
        window_setup(std::move(p_window_setup)),  // WindowSetup is validated on
                                                  // its own construction
        design_method(p_design_method)
  {
    // Perform validation
    if (sample_rate <= 0.0) {
      throw std::invalid_argument(
          "FIRFilterParams: sample_rate must be positive. Got "
          + std::to_string(sample_rate));
    }
    if (cutoff1 <= 0.0 || cutoff1 >= sample_rate / 2.0) {
      throw std::invalid_argument(
          "FIRFilterParams: cutoff1 (" + std::to_string(cutoff1)
          + ") is out of valid range (0, sample_rate/2 = "
          + std::to_string(sample_rate / 2.0) + ").");
    }

    if (type == FilterType::Bandpass || type == FilterType::Bandstop) {
      if (!this->cutoff2.has_value()) {
        throw std::invalid_argument(
            "FIRFilterParams: cutoff2 is required for Bandpass/Bandstop "
            "filters.");
      }
      if (this->cutoff2.value() <= 0.0
          || this->cutoff2.value() >= sample_rate / 2.0) {
        throw std::invalid_argument(
            "FIRFilterParams: cutoff2 (" + std::to_string(this->cutoff2.value())
            + ") is out of valid range (0, sample_rate/2 = "
            + std::to_string(sample_rate / 2.0) + ").");
      }
      if (this->cutoff2.value() <= this->cutoff1) {
        throw std::invalid_argument(
            "FIRFilterParams: For Bandpass/Bandstop, cutoff2 ("
            + std::to_string(this->cutoff2.value())
            + ") must be greater than cutoff1 (" + std::to_string(this->cutoff1)
            + ").");
      }
    }
    else {
      if (this->cutoff2.has_value()) {
        throw std::invalid_argument(
            "FIRFilterParams: cutoff2 should only be specified for "
            "Bandpass/Bandstop filters.");
      }
    }

    if (this->order.has_value() && this->order.value() == 0
        && (this->order.value() + 1) == 1) {
      // Allow order 0 (1 tap) for now.
    }

    if (this->transition_width.has_value()
        && this->transition_width.value() <= 0.0) {
      throw std::invalid_argument(
          "FIRFilterParams: transition_width, if specified, must be positive. "
          "Got "
          + std::to_string(this->transition_width.value()));
    }
    if (this->stopband_attenuation_db.has_value()
        && this->stopband_attenuation_db.value() <= 0.0) {
      throw std::invalid_argument(
          "FIRFilterParams: stopband_attenuation_db, if specified, must be "
          "positive. Got "
          + std::to_string(this->stopband_attenuation_db.value()));
    }

    // Ensure that if characteristics are not given, order must be.
    if (!this->order.has_value()
        && !(
            this->transition_width.has_value()
            && this->stopband_attenuation_db.has_value())) {
      throw std::invalid_argument(
          "FIRFilterParams: Either 'order' must be specified, or both "
          "'transition_width' and 'stopband_attenuation_db' must be specified "
          "for order estimation.");
    }

    // WindowSetup itself is validated upon its construction.
    // The window_setup.length passed to this FIRFilterParams constructor (via
    // default argument) is 0, as the actual length for window generation will
    // be derived from the filter order. The WindowSetup constructor validates
    // its own parameters (e.g. beta for Kaiser).
  }

}  // namespace OmniDSP
