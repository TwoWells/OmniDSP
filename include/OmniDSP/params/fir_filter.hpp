/**
 * @file fir_filter.hpp
 * @brief Defines parameters for FIR filter design specification.
 */

#ifndef OMNIDSP_PARAMS_FIR_FILTER_HPP
#define OMNIDSP_PARAMS_FIR_FILTER_HPP

#include <optional>  // For std::optional
#include <ostream>   // For std::ostream
#include <sstream>   // For std::ostringstream (used in operator<< for optional)
#include <string>    // For std::string (used by constructor for exceptions)
#include <utility>   // For std::move
#include <vector>    // Not directly used, but common in DSP contexts

#include "OmniDSP/core_types.hpp"  // For OMNIDSP_EXPORT
#include "OmniDSP/types/filter.hpp"  // For FilterType, FIRFilterDesignMethod and their operator<<
#include "OmniDSP/window.hpp"  // For WindowSetup, and its operator<<

// Include fmt headers for custom formatter specialization
#include <fmt/core.h>     // For basic formatting
#include <fmt/ostream.h>  // Specifically for ostream_formatter

// spdlog include is deferred to .cpp as definitions are moved there.

namespace OmniDSP::Params {

  /**
   * @brief Parameters for designing a Finite Impulse Response (FIR) filter.
   *
   * This structure is typically used as input to a utility function (e.g.,
   * `OmniDSP::Utils::create_design`) which then calculates a full
   * `OmniDSP::Design::FIRFilter`.
   *
   * Construction of this object validates the provided parameters.
   * Fluent setters are available for modifying parameters after construction.
   */
  struct OMNIDSP_EXPORT FIRFilter {
    FilterType filter_type_;  ///< Type of filter (Lowpass, Highpass, etc.).
    double
        sample_rate_;  ///< Sample rate of the signal in Hz. Must be positive.
    double
        cutoff1_;  ///< Primary cutoff frequency in Hz. For Lowpass/Highpass,
                   ///< this is the only cutoff. For Bandpass/Bandstop, this is
                   ///< the lower cutoff. Must be > 0 and < sample_rate / 2.
    std::optional<double>
        cutoff2_;  ///< Secondary cutoff frequency in Hz. Required for
                   ///< Bandpass/Bandstop. Must be > 0 and < sample_rate / 2.
                   ///< For Bandpass, cutoff2 > cutoff1. For Bandstop, cutoff2 >
                   ///< cutoff1.

    // Option 1: Specify filter characteristics for order estimation
    std::optional<double>
        transition_width_;  ///< Desired transition width in Hz (e.g., for
                            ///< window method estimation).
    std::optional<double>
        stopband_attenuation_db_;  ///< Desired stopband attenuation in positive
                                   ///< dB (e.g., for window method estimation).

    // Option 2: Specify filter order directly
    std::optional<size_t>
        order_;  ///< Desired filter order (number of taps - 1). If provided,
                 ///< typically overrides estimation from
                 ///< transition_width/stopband_attenuation.

    WindowSetup
        window_setup_;  ///< Windowing function setup to be used for design
                        ///< methods like WindowSinc. The `length` field of this
                        ///< `WindowSetup` object is typically ignored by the
                        ///< `Utils::create_design` function, as the filter
                        ///< design process determines the required number of
                        ///< taps (order + 1) and uses that for the window
                        ///< generation. The user should specify window type and
                        ///< its parameters (e.g. beta for Kaiser).

    FIRFilterDesignMethod
        design_method_;  ///< Method to be used for designing the filter.

    /**
     * @brief Explicit constructor that validates parameters. Declaration only.
     * Definition is in the corresponding .cpp file.
     */
    explicit FIRFilter(
        FilterType p_type,
        double p_sample_rate,
        double p_cutoff1,
        std::optional<double> p_cutoff2 = std::nullopt,
        std::optional<size_t> p_order = std::nullopt,
        std::optional<double> p_transition_width = std::nullopt,
        std::optional<double> p_stopband_attenuation_db = std::nullopt,
        WindowSetup p_window_setup = WindowSetup{Type::Window::Hann, 0},
        FIRFilterDesignMethod p_design_method
        = FIRFilterDesignMethod::WindowSinc);

    /**
     * @brief Helper to get the number of taps if order is specified.
     * @return Number of taps (order + 1), or std::nullopt if order is not
     * specified.
     */
    [[nodiscard]] std::optional<size_t> num_taps() const;  // Definition in .cpp

    // --- Fluent Setters (Declarations) ---
    FIRFilter& filter_type(FilterType val);
    FIRFilter& sample_rate(double val);
    FIRFilter& cutoff1(double val);
    FIRFilter& cutoff2(std::optional<double> val);
    FIRFilter& transition_width(std::optional<double> val);
    FIRFilter& stopband_attenuation_db(std::optional<double> val);
    FIRFilter& order(std::optional<size_t> val);
    FIRFilter& window_setup(WindowSetup val);
    FIRFilter& design_method(FIRFilterDesignMethod val);
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of
   * Params::FIRFilter.
   * @param os The output stream.
   * @param params The Params::FIRFilter object to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, const FIRFilter& params)
  {
    os << "Params::FIRFilter(Type: "
       << params.filter_type_  // Uses FilterType::operator<<
       << ", SR: " << params.sample_rate_ << ", Cutoff1: " << params.cutoff1_;
    if (params.cutoff2_) {
      os << ", Cutoff2: " << params.cutoff2_.value();
    }
    else {
      os << ", Cutoff2: None";
    }
    if (params.transition_width_) {
      os << ", TW: " << params.transition_width_.value();
    }
    else {
      os << ", TW: None";
    }
    if (params.stopband_attenuation_db_) {
      os << ", StopAtten: " << params.stopband_attenuation_db_.value() << "dB";
    }
    else {
      os << ", StopAtten: None";
    }
    if (params.order_) {
      os << ", Order: " << params.order_.value();
    }
    else {
      os << ", Order: None (estimate)";
    }
    os << ", Window: "
       << params.window_setup_  // Relies on WindowSetup's operator<<
       << ", Method: "
       << params.design_method_  // Uses FIRFilterDesignMethod::operator<<
       << ")";
    return os;
  }

}  // namespace OmniDSP::Params

// Specialization of fmt::formatter for OmniDSP::Params::FIRFilter
template <>
struct fmt::formatter<OmniDSP::Params::FIRFilter> : fmt::ostream_formatter {};

#endif  // OMNIDSP_PARAMS_FIR_FILTER_HPP
