/**
 * @file resample.hpp
 * @brief Defines parameters for resampling operation specification.
 */

#ifndef OMNIDSP_PARAMS_RESAMPLE_HPP
#define OMNIDSP_PARAMS_RESAMPLE_HPP

#include <optional>  // Not strictly needed for current members, but good practice for Params structs
#include <ostream>  // For std::ostream (needed for operator<<)
#include <string>   // For std::string in validation messages
#include <utility>  // For std::move

#include "OmniDSP/core_types.hpp"  // For OMNIDSP_EXPORT
#include "OmniDSP/window.hpp"  // For WindowSetup, Type::Window, WindowParams, and operator<<(WindowSetup)

// Include fmt headers for custom formatter specialization
#include <fmt/core.h>     // For basic formatting
#include <fmt/ostream.h>  // Specifically for ostream_formatter

// spdlog include is deferred to .cpp as definitions are moved there.

namespace OmniDSP::Params {

  /**
   * @brief Parameters for specifying a resampling operation.
   *
   * This structure is typically used as input to a utility function
   * (e.g., `OmniDSP::Utils::create_design`) which then calculates a full
   * `OmniDSP::ResampleSpec`. The prototype anti-aliasing/anti-imaging
   * filter's characteristics (like order) will be determined based on these
   * parameters, particularly the quality setting and resampling ratio.
   *
   * Construction of this object validates the provided parameters.
   * Fluent setters are available for modifying parameters after construction.
   */
  struct OMNIDSP_EXPORT Resample {
    double input_rate_;  ///< Input sample rate in Hz. Must be positive.
    double
        output_rate_;  ///< Desired output sample rate in Hz. Must be positive.
    int quality_;  ///< Quality setting for the resampling process (e.g., 0-15).
                   ///< Higher values generally mean a sharper filter, longer
                   ///< filter kernel, and more computation.
    WindowSetup
        window_setup_;  ///< Windowing function setup to be used for designing
                        ///< the internal prototype FIR filter. The `length`
                        ///< field of this `WindowSetup` is typically ignored by
                        ///< `Utils::create_design`, as the filter order (and
                        ///< thus window length) is determined by the resampling
                        ///< parameters and quality. The user should specify the
                        ///< window type and its specific parameters (e.g., beta
                        ///< for Kaiser).

    /**
     * @brief Explicit constructor that validates parameters.
     * @param p_input_rate Input sample rate in Hz.
     * @param p_output_rate Desired output sample rate in Hz.
     * @param p_quality Quality setting (e.g., 0-15).
     * @param p_window_setup Window setup for the prototype filter design.
     * Defaults to Kaiser (beta 5.0), with length 0 as it will be determined.
     * @throws std::invalid_argument if parameters are inconsistent or invalid.
     */
    explicit Resample(
        double p_input_rate,
        double p_output_rate,
        int p_quality,
        WindowSetup p_window_setup
        = WindowSetup{Type::Window::Kaiser, 0, WindowParams{{"beta", 5.0}}});

    // --- Fluent Setters (Declarations) ---

    /**
     * @brief Sets the input sample rate.
     * @param val The new input sample rate in Hz.
     * @return A reference to this Params::Resample object.
     * @throws std::invalid_argument if val is not positive.
     */
    Resample& input_rate(double val);

    /**
     * @brief Sets the output sample rate.
     * @param val The new output sample rate in Hz.
     * @return A reference to this Params::Resample object.
     * @throws std::invalid_argument if val is not positive.
     */
    Resample& output_rate(double val);

    /**
     * @brief Sets the quality setting for resampling.
     * @param val The new quality setting (e.g., 0-15).
     * @return A reference to this Params::Resample object.
     * @throws std::invalid_argument if val is outside the accepted range.
     */
    Resample& quality(int val);

    /**
     * @brief Sets the window setup for the prototype filter.
     * @param val The new WindowSetup object. Its own constructor handles
     * validation.
     * @return A reference to this Params::Resample object.
     */
    Resample& window_setup(WindowSetup val);
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of
   * Params::Resample.
   * @param os The output stream.
   * @param params The Params::Resample object to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, const Resample& params)
  {
    os << "Params::Resample(InputRate: " << params.input_rate_
       << ", OutputRate: " << params.output_rate_
       << ", Quality: " << params.quality_ << ", Window: "
       << params.window_setup_  // Relies on WindowSetup's operator<<
       << ")";
    return os;
  }

}  // namespace OmniDSP::Params

// Specialization of fmt::formatter for OmniDSP::Params::Resample
// This allows spdlog (which uses fmtlib) to format Resample objects
// by using the operator<< we defined.
template <>
struct fmt::formatter<OmniDSP::Params::Resample> : fmt::ostream_formatter {};

#endif  // OMNIDSP_PARAMS_RESAMPLE_HPP
