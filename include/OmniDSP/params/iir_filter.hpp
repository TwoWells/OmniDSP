/**
 * @file iir_filter.hpp
 * @brief Defines parameters for IIR filter design specification.
 */

#ifndef OMNIDSP_PARAMS_IIR_FILTER_HPP
#define OMNIDSP_PARAMS_IIR_FILTER_HPP

#include <OmniDSP/core_types.hpp>    // For OMNIDSP_EXPORT
#include <OmniDSP/types/filter.hpp>  // For FilterType, IIRFilterFormat
#include <optional>                  // For std::optional
#include <string>   // For std::string (used by constructor for exceptions)
#include <utility>  // For std::move

// spdlog include is deferred to .cpp

namespace OmniDSP {

  /**
   * @brief Parameters for designing an Infinite Impulse Response (IIR) filter.
   *
   * This structure is typically used as input to a utility function (e.g.,
   * `OmniDSP::Utils::create_spec`) which then calculates a full
   * `OmniDSP::IIRFilterSpec`.
   *
   * Construction of this object validates the provided parameters.
   * Fluent setters are available for modifying parameters after construction.
   */
  struct OMNIDSP_EXPORT IIRFilterParams {
    FilterType filter_type_;  ///< Type of filter (Lowpass, Highpass, etc.).
    double
        sample_rate_;  ///< Sample rate of the signal in Hz. Must be positive.
    size_t order_;     ///< Desired filter order. Must be positive.
    double
        cutoff1_;  ///< Primary cutoff frequency in Hz. For Lowpass/Highpass,
                   ///< this is the only cutoff. For Bandpass/Bandstop, this is
                   ///< the lower cutoff. Must be > 0 and < sample_rate / 2.
    std::optional<double>
        cutoff2_;  ///< Secondary cutoff frequency in Hz. Required for
                   ///< Bandpass/Bandstop. Must be > 0 and < sample_rate / 2.
                   ///< For Bandpass, cutoff2 > cutoff1. For Bandstop, cutoff2 >
                   ///< cutoff1.
    std::optional<double>
        passband_ripple_db_;  ///< Optional desired passband ripple in positive
                              ///< dB (e.g., for Chebyshev I, Elliptic).
    std::optional<double>
        stopband_attenuation_db_;  ///< Optional desired stopband attenuation in
                                   ///< positive dB (e.g., for Chebyshev II,
                                   ///< Elliptic).
    IIRFilterFormat output_format_;  ///< Desired output format for coefficients
                                     ///< (SOS or TransferFunction).

    /**
     * @brief Explicit constructor that validates parameters.
     * @param p_type Type of filter.
     * @param p_sample_rate Sample rate in Hz.
     * @param p_order Filter order.
     * @param p_cutoff1 Primary cutoff frequency in Hz.
     * @param p_cutoff2 Optional secondary cutoff frequency in Hz.
     * @param p_passband_ripple_db Optional passband ripple in dB.
     * @param p_stopband_attenuation_db Optional stopband attenuation in dB.
     * @param p_output_format Desired coefficient output format. Defaults to
     * SOS.
     * @throws std::invalid_argument if parameters are inconsistent or invalid.
     */
    explicit IIRFilterParams(
        FilterType p_type,
        double p_sample_rate,
        size_t p_order,
        double p_cutoff1,
        std::optional<double> p_cutoff2 = std::nullopt,
        std::optional<double> p_passband_ripple_db = std::nullopt,
        std::optional<double> p_stopband_attenuation_db = std::nullopt,
        IIRFilterFormat p_output_format = IIRFilterFormat::SOS);

    // --- Fluent Setters (Declarations) ---
    IIRFilterParams& filter_type(FilterType val);
    IIRFilterParams& sample_rate(double val);
    IIRFilterParams& order(size_t val);
    IIRFilterParams& cutoff1(double val);
    IIRFilterParams& cutoff2(std::optional<double> val);
    IIRFilterParams& passband_ripple_db(std::optional<double> val);
    IIRFilterParams& stopband_attenuation_db(std::optional<double> val);
    IIRFilterParams& output_format(IIRFilterFormat val);
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_PARAMS_IIR_FILTER_HPP
