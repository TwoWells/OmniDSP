/**
 * @file fir_filter.hpp (design)
 * @brief Defines the Design::FIRFilter structure for FIR filter design
 * specifications.
 */
#ifndef OMNIDSP_DESIGN_FIR_FILTER_HPP
#define OMNIDSP_DESIGN_FIR_FILTER_HPP

#include <cstddef>  // For size_t
#include <optional>
#include <ostream>  // For std::ostream
#include <sstream>  // For std::ostringstream (for optional)
#include <vector>

#include "OmniDSP/core_types.hpp"    // For OMNIDSP_EXPORT
#include "OmniDSP/types/filter.hpp"  // For FilterType and its operator<<
#include "OmniDSP/window.hpp"        // For WindowSetup and its operator<<

// Include fmt headers for custom formatter specialization
#include <fmt/core.h>     // For basic formatting
#include <fmt/ostream.h>  // Specifically for ostream_formatter

namespace OmniDSP::Design {

  /**
   * @brief Fully resolved design specification for a Finite Impulse Response
   * (FIR) filter.
   */
  struct OMNIDSP_EXPORT FIRFilter {
    FilterType type;     ///< Type of the filter (e.g., Lowpass, Highpass).
    size_t order;        ///< Order of the filter (number of taps - 1).
    double sample_rate;  ///< Sample rate of the signal in Hz.
    double cutoff1;      ///< Primary cutoff frequency in Hz.
    std::optional<double> cutoff2;  ///< Optional secondary cutoff frequency in
                                    ///< Hz (for Bandpass/Bandstop).
    WindowSetup
        window_setup;  ///< Windowing function setup used for the design.

    /**
     * @brief Explicit constructor for a FIRFilter design specification.
     * @param p_type Type of the filter.
     * @param p_order Filter order.
     * @param p_sample_rate Sample rate in Hz.
     * @param p_cutoff1 Primary cutoff frequency in Hz.
     * @param p_cutoff2 Optional secondary cutoff frequency in Hz.
     * @param p_window_setup Window setup used for the design.
     */
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
      // Basic validation can be added here or rely on Utils::create_design
      if (static_cast<size_t>(this->window_setup.length) != (this->order + 1)
          && this->window_setup.length != 0) {
        // This indicates an internal error if Utils::create_design was used.
        // Consider throwing std::logic_error or logging.
      }
    }

    /**
     * @brief Gets the number of taps for the filter.
     * @return The number of taps (order + 1).
     */
    [[nodiscard]] size_t num_taps() const { return order + 1; }

    /**
     * @brief Validates the consistency of the FIR filter design parameters.
     * @return True if the parameters are consistent, false otherwise.
     */
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
      // The window_setup.length should match order + 1 if it was set by the
      // design process. If window_setup.length is 0, it implies it was
      // determined by order.
      if (static_cast<size_t>(window_setup.length) != (order + 1)
          && window_setup.length != 0) {
        return false;
      }
      return true;
    }
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of
   * Design::FIRFilter.
   * @param os The output stream.
   * @param spec The Design::FIRFilter object to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, const FIRFilter& spec)
  {
    os << "Design::FIRFilter(Type: "
       << spec.type  // Uses FilterType::operator<<
       << ", Order: " << spec.order << ", SR: " << spec.sample_rate
       << ", Cutoff1: " << spec.cutoff1;
    if (spec.cutoff2) {
      os << ", Cutoff2: " << spec.cutoff2.value();
    }
    else {
      os << ", Cutoff2: None";
    }
    os << ", Window: " << spec.window_setup  // Uses WindowSetup::operator<<
       << ")";
    return os;
  }

}  // namespace OmniDSP::Design

// Specialization of fmt::formatter for OmniDSP::Design::FIRFilter
template <>
struct fmt::formatter<OmniDSP::Design::FIRFilter> : fmt::ostream_formatter {};

#endif  // OMNIDSP_DESIGN_FIR_FILTER_HPP
