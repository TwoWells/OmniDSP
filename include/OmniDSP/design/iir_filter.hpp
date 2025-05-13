/**
 * @file iir_filter.hpp (design)
 * @brief Defines the Design::IIRFilter structure for IIR filter design
 * specifications.
 */
#ifndef OMNIDSP_DESIGN_IIR_FILTER_HPP
#define OMNIDSP_DESIGN_IIR_FILTER_HPP

#include <cstddef>  // For size_t
#include <optional>
#include <vector>

#include "OmniDSP/core_types.hpp"    // For OMNIDSP_EXPORT
#include "OmniDSP/types/filter.hpp"  // For FilterType, IIRFilterFormat

namespace OmniDSP::Design {

  /**
   * @brief Design specification for an IIR filter.
   */
  struct OMNIDSP_EXPORT IIRFilter {
    FilterType type = FilterType::Lowpass;
    size_t order = 4;
    double sample_rate = 1.0;
    double cutoff1 = 0.1;
    std::optional<double> cutoff2 = std::nullopt;
    std::optional<double> passband_ripple_db = std::nullopt;
    std::optional<double> stopband_attenuation_db = std::nullopt;
    IIRFilterFormat output_format = IIRFilterFormat::SOS;

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

}  // namespace OmniDSP::Design

#endif  // OMNIDSP_DESIGN_IIR_FILTER_HPP
