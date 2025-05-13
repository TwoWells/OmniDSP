/**
 * @file fir_filter.hpp (design)
 * @brief Defines the Design::FIRFilter structure for FIR filter design
 * specifications.
 */
#ifndef OMNIDSP_DESIGN_FIR_FILTER_HPP
#define OMNIDSP_DESIGN_FIR_FILTER_HPP

#include <cstddef>  // For size_t
#include <optional>
#include <vector>

#include "OmniDSP/core_types.hpp"    // For OMNIDSP_EXPORT
#include "OmniDSP/types/filter.hpp"  // For FilterType
#include "OmniDSP/window.hpp"        // For WindowSetup

namespace OmniDSP::Design {

  /**
   * @brief Fully resolved design specification for a Finite Impulse Response
   * (FIR) filter.
   */
  struct OMNIDSP_EXPORT FIRFilter {
    FilterType type;
    size_t order;
    double sample_rate;
    double cutoff1;
    std::optional<double> cutoff2;
    WindowSetup window_setup;

    explicit FIRFilter(
        FilterType p_type,
        size_t p_order,
        double p_sample_rate,
        double p_cutoff1,
        std::optional<double> p_cutoff2,
        WindowSetup p_window_setup)
        : type(p_type),
          order(p_order),
          sample_rate(p_sample_rate),
          cutoff1(p_cutoff1),
          cutoff2(std::move(p_cutoff2)),
          window_setup(std::move(p_window_setup))
    {
      // Basic validation can be added here or rely on Utils::create_spec
      if (static_cast<size_t>(this->window_setup.length) != (this->order + 1)
          && this->window_setup.length != 0) {
        // This indicates an internal error if Utils::create_spec was used.
        // Consider throwing std::logic_error or logging.
      }
    }

    [[nodiscard]] size_t num_taps() const { return order + 1; }

    [[nodiscard]] bool validate_consistency() const
    {
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
      if (static_cast<size_t>(window_setup.length) != (order + 1)
          && window_setup.length != 0) {
        return false;
      }
      return true;
    }
  };

}  // namespace OmniDSP::Design

#endif  // OMNIDSP_DESIGN_FIR_FILTER_HPP
