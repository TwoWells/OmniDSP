/**
 * @file fir_filter.hpp
 * @brief Defines parameters for FIR filter design specification.
 */

#ifndef OMNIDSP_PARAMS_FIR_FILTER_HPP
#define OMNIDSP_PARAMS_FIR_FILTER_HPP

#include <OmniDSP/core_types.hpp>    // For OMNIDSP_EXPORT
#include <OmniDSP/types/filter.hpp>  // For FilterType, FIRFilterDesignMethod
#include <OmniDSP/window.hpp>        // For WindowSetup
#include <optional>                  // For std::optional
#include <string>   // For std::string (used by constructor for exceptions)
#include <utility>  // For std::move
#include <vector>   // Not directly used, but common in DSP contexts

// spdlog include is deferred to .cpp as definitions are moved there.

namespace OmniDSP {

  /**
   * @brief Parameters for designing a Finite Impulse Response (FIR) filter.
   *
   * This structure is typically used as input to a utility function (e.g.,
   * `OmniDSP::Utils::create_spec`) which then calculates a full
   * `OmniDSP::FIRFilterSpec`.
   *
   * Construction of this object validates the provided parameters.
   * Fluent setters are available for modifying parameters after construction.
   */
  struct OMNIDSP_EXPORT FIRFilterParams {
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
                        ///< `Utils::create_spec` function, as the filter design
                        ///< process determines the required number of taps
                        ///< (order + 1) and uses that for the window
                        ///< generation. The user should specify window type and
                        ///< its parameters (e.g. beta for Kaiser).

    FIRFilterDesignMethod
        design_method_;  ///< Method to be used for designing the filter.

    /**
     * @brief Explicit constructor that validates parameters. Declaration only.
     * Definition is in the corresponding .cpp file.
     */
    explicit FIRFilterParams(
        FilterType p_type,
        double p_sample_rate,
        double p_cutoff1,
        std::optional<double> p_cutoff2 = std::nullopt,
        std::optional<size_t> p_order = std::nullopt,
        std::optional<double> p_transition_width = std::nullopt,
        std::optional<double> p_stopband_attenuation_db = std::nullopt,
        WindowSetup p_window_setup = WindowSetup{WindowType::Hann, 0},
        FIRFilterDesignMethod p_design_method
        = FIRFilterDesignMethod::WindowSinc);

    /**
     * @brief Helper to get the number of taps if order is specified.
     * @return Number of taps (order + 1), or std::nullopt if order is not
     * specified.
     */
    [[nodiscard]] std::optional<size_t> num_taps() const;  // Definition in .cpp

    // --- Fluent Setters (Declarations) ---
    FIRFilterParams& filter_type(FilterType val);
    FIRFilterParams& sample_rate(double val);
    FIRFilterParams& cutoff1(double val);
    FIRFilterParams& cutoff2(std::optional<double> val);
    FIRFilterParams& transition_width(std::optional<double> val);
    FIRFilterParams& stopband_attenuation_db(std::optional<double> val);
    FIRFilterParams& order(std::optional<size_t> val);
    FIRFilterParams& window_setup(WindowSetup val);
    FIRFilterParams& design_method(FIRFilterDesignMethod val);
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_PARAMS_FIR_FILTER_HPP
