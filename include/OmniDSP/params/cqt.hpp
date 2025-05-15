/**
 * @file cqt.hpp
 * @brief Defines parameters for Constant-Q Transform (CQT) specification.
 */

#ifndef OMNIDSP_PARAMS_CQT_HPP
#define OMNIDSP_PARAMS_CQT_HPP

#include <OmniDSP/core_types.hpp>  // For OMNIDSP_EXPORT
#include <OmniDSP/window.hpp>  // For WindowSetup, Type::Window, and its operator<<
#include <optional>            // For potential future optional parameters
#include <ostream>             // For std::ostream
#include <string>              // For std::string in validation messages
#include <utility>             // For std::move

// Include fmt headers for custom formatter specialization
#include <fmt/core.h>     // For basic formatting
#include <fmt/ostream.h>  // Specifically for ostream_formatter

namespace OmniDSP::Params {

  /**
   * @brief Parameters for specifying a Constant-Q Transform (CQT).
   *
   * This structure is used as input to `OmniDSP::Utils::create_design` to
   * generate a full `OmniDSP::CQTSpec`. It holds the primary user-configurable
   * parameters for the CQT.
   *
   * Construction of this object validates the provided parameters.
   * Fluent setters are available for modifying parameters after construction.
   */
  struct OMNIDSP_EXPORT CQT {
    double sample_rate_;  ///< Sample rate of the input signal in Hz. Must be
                          ///< positive.
    double min_freq_;     ///< Minimum frequency for the CQT bins in Hz. Must be
                          ///< positive and less than sample_rate / 2.
    double max_freq_;     ///< Maximum frequency for the CQT bins in Hz. Must be
                       ///< greater than min_freq and less than sample_rate / 2.
    int bins_per_octave_;  ///< Number of CQT bins per octave. Must be positive.
    WindowSetup
        window_setup_;  ///< Windowing function setup for the CQT analysis
                        ///< windows. The `length` field of this `WindowSetup`
                        ///< is typically ignored by `Utils::create_design`, as
                        ///< CQT kernel lengths are frequency-dependent and
                        ///< determined internally. The user should specify the
                        ///< window type and its parameters (e.g., beta for
                        ///< Kaiser).

    /**
     * @brief Explicit constructor that validates parameters.
     * @param p_sample_rate Sample rate in Hz.
     * @param p_min_freq Minimum frequency for CQT bins in Hz.
     * @param p_max_freq Maximum frequency for CQT bins in Hz.
     * @param p_bins_per_octave Number of CQT bins per octave.
     * @param p_window_setup Window setup for the CQT analysis windows. Defaults
     * to Hann window. The length of this WindowSetup is typically set to 0, as
     * actual kernel lengths are frequency-dependent.
     * @throws std::invalid_argument if parameters are inconsistent or invalid.
     */
    explicit CQT(
        double p_sample_rate,
        double p_min_freq,
        double p_max_freq,
        int p_bins_per_octave,
        WindowSetup p_window_setup = WindowSetup{Type::Window::Hann, 0});

    // --- Fluent Setters (Declarations Only) ---

    /**
     * @brief Sets the sample rate.
     * @param val The new sample rate in Hz.
     * @return A reference to this Params::CQT object.
     * @throws std::invalid_argument if val is not positive.
     */
    CQT& sample_rate(double val);

    /**
     * @brief Sets the minimum frequency.
     * @param val The new minimum frequency in Hz.
     * @return A reference to this Params::CQT object.
     * @throws std::invalid_argument if val is not positive or other frequency
     * constraints are violated.
     */
    CQT& min_freq(double val);

    /**
     * @brief Sets the maximum frequency.
     * @param val The new maximum frequency in Hz.
     * @return A reference to this Params::CQT object.
     * @throws std::invalid_argument if val is not positive or other frequency
     * constraints are violated.
     */
    CQT& max_freq(double val);

    /**
     * @brief Sets the number of bins per octave.
     * @param val The new number of bins per octave.
     * @return A reference to this Params::CQT object.
     * @throws std::invalid_argument if val is not positive.
     */
    CQT& bins_per_octave(int val);

    /**
     * @brief Sets the window setup.
     * @param val The new WindowSetup object. WindowSetup's own constructor
     * handles its validation.
     * @return A reference to this Params::CQT object.
     */
    CQT& window_setup(WindowSetup val);
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of Params::CQT.
   * @param os The output stream.
   * @param params The Params::CQT object to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, const CQT& params)
  {
    os << "Params::CQT(SR: " << params.sample_rate_
       << ", MinFreq: " << params.min_freq_ << ", MaxFreq: " << params.max_freq_
       << ", BinsPerOct: " << params.bins_per_octave_
       << ", Window: " << params.window_setup_  // Uses WindowSetup::operator<<
       << ")";
    return os;
  }

}  // namespace OmniDSP::Params

// Specialization of fmt::formatter for OmniDSP::Params::CQT
template <>
struct fmt::formatter<OmniDSP::Params::CQT> : fmt::ostream_formatter {};

#endif  // OMNIDSP_PARAMS_CQT_HPP
