/**
 * @file iir_filter.hpp (design)
 * @brief Defines the Design::IIRFilter structure for IIR filter design
 * specifications.
 */
#ifndef OMNIDSP_DESIGN_IIR_FILTER_HPP
#define OMNIDSP_DESIGN_IIR_FILTER_HPP

#include <cstddef>  // For size_t
#include <optional>
#include <ostream>  // For std::ostream
#include <sstream>  // For std::ostringstream (for optional)
#include <vector>

#include "OmniDSP/core_types.hpp"  // For OMNIDSP_EXPORT
#include "OmniDSP/types/filter.hpp"  // For FilterType, IIRFilterFormat and their operator<<

// Include fmt headers for custom formatter specialization
#include <fmt/core.h>     // For basic formatting
#include <fmt/ostream.h>  // Specifically for ostream_formatter

namespace OmniDSP::Design {

  /**
   * @brief Design specification for an IIR filter.
   * @details This structure holds all the necessary parameters that define a
   * specific IIR filter design, typically after being resolved from
   * Params::IIRFilter.
   */
  struct OMNIDSP_EXPORT IIRFilter {
    FilterType type;     ///< Type of filter (e.g., Lowpass, Highpass).
    size_t order;        ///< Filter order.
    double sample_rate;  ///< Sample rate of the signal in Hz.
    double cutoff1;      ///< Primary cutoff frequency in Hz.
    std::optional<double> cutoff2;  ///< Optional secondary cutoff frequency in
                                    ///< Hz (for Bandpass/Bandstop).
    std::optional<double>
        passband_ripple_db;  ///< Optional passband ripple in positive dB.
    std::optional<double>
        stopband_attenuation_db;  ///< Optional stopband attenuation in positive
                                  ///< dB.
    IIRFilterFormat output_format;  ///< Desired output format for coefficients
                                    ///< (SOS or TransferFunction).

    /**
     * @brief Explicit constructor for an IIRFilter design specification.
     * @param p_type Type of the filter.
     * @param p_order Filter order.
     * @param p_sample_rate Sample rate in Hz.
     * @param p_cutoff1 Primary cutoff frequency in Hz.
     * @param p_cutoff2 Optional secondary cutoff frequency in Hz.
     * @param p_passband_ripple_db Optional passband ripple in dB.
     * @param p_stopband_attenuation_db Optional stopband attenuation in dB.
     * @param p_output_format Desired coefficient output format.
     */
    explicit IIRFilter(
        FilterType p_type,
        size_t p_order,
        double p_sample_rate,
        double p_cutoff1,
        std::optional<double> p_cutoff2,
        std::optional<double> p_passband_ripple_db,
        std::optional<double> p_stopband_attenuation_db,
        IIRFilterFormat p_output_format = IIRFilterFormat::SOS)
        : type(p_type),
          order(p_order),
          sample_rate(p_sample_rate),
          cutoff1(p_cutoff1),
          cutoff2(std::move(p_cutoff2)),
          passband_ripple_db(std::move(p_passband_ripple_db)),
          stopband_attenuation_db(std::move(p_stopband_attenuation_db)),
          output_format(p_output_format)
    {}

    /**
     * @brief Validates the consistency of the IIR filter design parameters.
     * @return True if the parameters are consistent, false otherwise.
     */
    [[nodiscard]] bool validate_consistency() const
    {
      if (order == 0) return false;
      if (sample_rate <= 0.0) return false;
      if (cutoff1 <= 0.0 || cutoff1 >= 0.5 * sample_rate) return false;
      if (type == FilterType::Bandpass || type == FilterType::Bandstop) {
        if (!cutoff2.has_value() || cutoff2.value() <= 0.0
            || cutoff2.value() >= 0.5 * sample_rate
            || cutoff2.value() <= cutoff1) {
          return false;
        }
      }
      else {
        if (cutoff2.has_value()) return false;
      }
      if (passband_ripple_db.has_value() && passband_ripple_db.value() <= 0.0)
        return false;
      if (stopband_attenuation_db.has_value()
          && stopband_attenuation_db.value() <= 0.0)
        return false;
      return true;
    }
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of
   * Design::IIRFilter.
   * @param os The output stream.
   * @param spec The Design::IIRFilter object to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, const IIRFilter& spec)
  {
    os << "Design::IIRFilter(Type: "
       << spec.type  // Uses FilterType::operator<<
       << ", Order: " << spec.order << ", SR: " << spec.sample_rate
       << ", Cutoff1: " << spec.cutoff1;
    if (spec.cutoff2) {
      os << ", Cutoff2: " << spec.cutoff2.value();
    }
    else {
      os << ", Cutoff2: None";
    }
    if (spec.passband_ripple_db) {
      os << ", RippleDB: " << spec.passband_ripple_db.value();
    }
    else {
      os << ", RippleDB: None";
    }
    if (spec.stopband_attenuation_db) {
      os << ", StopAttenDB: " << spec.stopband_attenuation_db.value();
    }
    else {
      os << ", StopAttenDB: None";
    }
    os << ", Format: "
       << spec.output_format  // Uses IIRFilterFormat::operator<<
       << ")";
    return os;
  }

}  // namespace OmniDSP::Design

// Specialization of fmt::formatter for OmniDSP::Design::IIRFilter
template <>
struct fmt::formatter<OmniDSP::Design::IIRFilter> : fmt::ostream_formatter {};

#endif  // OMNIDSP_DESIGN_IIR_FILTER_HPP
