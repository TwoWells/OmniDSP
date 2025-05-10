/**
 * @file resample.hpp
 * @brief Defines parameters for resampling operation specification.
 */

#ifndef OMNIDSP_PARAMS_RESAMPLE_HPP
#define OMNIDSP_PARAMS_RESAMPLE_HPP

#include <optional>  // Not strictly needed for current members, but good practice for Params structs
#include <string>  // For std::string in validation messages

#include "OmniDSP/core_types.hpp"  // For OMNIDSP_EXPORT
#include "OmniDSP/window.hpp"      // For WindowSetup, WindowType, WindowParams

namespace OmniDSP {

  /**
   * @brief Parameters for specifying a resampling operation.
   *
   * This structure is typically used as input to a utility function
   * (e.g., `OmniDSP::Utils::create_spec`) which then calculates a full
   * `OmniDSP::ResampleSpec`. The prototype anti-aliasing/anti-imaging
   * filter's characteristics (like order) will be determined based on these
   * parameters, particularly the quality setting and resampling ratio.
   *
   * Construction of this object validates the provided parameters.
   */
  struct OMNIDSP_EXPORT ResampleParams {
    double input_rate;  ///< Input sample rate in Hz. Must be positive.
    double
        output_rate;  ///< Desired output sample rate in Hz. Must be positive.
    int quality;  ///< Quality setting for the resampling process (e.g., 0-15).
                  ///< Higher values generally mean a sharper filter, longer
                  ///< filter kernel, and more computation.
    WindowSetup
        window_setup;  ///< Windowing function setup to be used for designing
                       ///< the internal prototype FIR filter. The `length`
                       ///< field of this `WindowSetup` is typically ignored by
                       ///< `Utils::create_spec`, as the filter order (and thus
                       ///< window length) is determined by the resampling
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
    explicit ResampleParams(
        double p_input_rate,
        double p_output_rate,
        int p_quality,
        // Updated default argument for p_window_setup:
        // Explicitly construct WindowParams for the std::optional<WindowParams>
        // parameter of the WindowSetup constructor.
        WindowSetup p_window_setup = WindowSetup{
            WindowType::Kaiser,
            0,
            WindowParams{{"beta", 5.0}}  // Explicitly construct WindowParams
        });

    // Add any helper methods if needed, similar to FIRFilterParams::num_taps()
    // For ResampleParams, direct helpers might be less common as most derived
    // values (like L/M factors or filter order) are calculated in
    // Utils::create_spec.
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_PARAMS_RESAMPLE_HPP
