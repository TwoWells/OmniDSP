/**
 * @file resample.hpp (design)
 * @brief Defines the Design::Resample structure for resampling design
 * specifications.
 */
#ifndef OMNIDSP_DESIGN_RESAMPLE_HPP
#define OMNIDSP_DESIGN_RESAMPLE_HPP

#include <cstddef>    // For size_t
#include <ostream>    // For std::ostream
#include <stdexcept>  // For std::logic_error
#include <utility>    // For std::move

#include "OmniDSP/core_types.hpp"         // For OMNIDSP_EXPORT
#include "OmniDSP/design/fir_filter.hpp"  // For Design::FIRFilter and its operator<<

// Include fmt headers for custom formatter specialization
#include <fmt/core.h>     // For basic formatting
#include <fmt/ostream.h>  // Specifically for ostream_formatter

namespace OmniDSP::Design {

  /**
   * @brief Fully resolved design specification for creating a resampling
   * processor.
   * @details This structure holds all parameters necessary to configure a
   * resampler, including the input/output rates, quality, rational conversion
   * factors, and the design of the prototype FIR filter.
   */
  struct OMNIDSP_EXPORT Resample {
    double input_rate;   ///< Input sample rate in Hz.
    double output_rate;  ///< Desired output sample rate in Hz.
    int quality;  ///< Quality setting for the resampling process (e.g., 0-15).
    size_t up_factor_L;    ///< Upsampling factor L for rational resampling.
    size_t down_factor_M;  ///< Downsampling factor M for rational resampling.
    Design::FIRFilter
        prototype_fir_design;  ///< Design specification for the prototype
                               ///< anti-aliasing/anti-imaging FIR filter.

    /**
     * @brief Explicit constructor for a Resample design specification.
     * @param p_input_rate Input sample rate in Hz.
     * @param p_output_rate Desired output sample rate in Hz.
     * @param p_quality Quality setting.
     * @param p_up_factor_L Upsampling factor L.
     * @param p_down_factor_M Downsampling factor M.
     * @param p_prototype_fir_design Design specification for the prototype FIR
     * filter.
     * @throws std::logic_error if input/output rates are not positive or L/M
     * factors are zero.
     */
    explicit Resample(
        double p_input_rate,
        double p_output_rate,
        int p_quality,
        size_t p_up_factor_L,
        size_t p_down_factor_M,
        Design::FIRFilter p_prototype_fir_design)
        : input_rate(p_input_rate),
          output_rate(p_output_rate),
          quality(p_quality),
          up_factor_L(p_up_factor_L),
          down_factor_M(p_down_factor_M),
          prototype_fir_design(std::move(p_prototype_fir_design))
    {
      if (input_rate <= 0.0 || output_rate <= 0.0) {
        throw std::logic_error(
            "Design::Resample: Input/Output rates must be positive.");
      }
      if (up_factor_L == 0 || down_factor_M == 0) {
        throw std::logic_error(
            "Design::Resample: Upsampling/Downsampling factors L/M cannot be "
            "zero.");
      }
    }
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of
   * Design::Resample.
   * @param os The output stream.
   * @param spec The Design::Resample object to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, const Resample& spec)
  {
    os << "Design::Resample(InputRate: " << spec.input_rate
       << ", OutputRate: " << spec.output_rate << ", Quality: " << spec.quality
       << ", L: " << spec.up_factor_L << ", M: " << spec.down_factor_M
       << ", PrototypeFIR: "
       << spec.prototype_fir_design  // Uses Design::FIRFilter::operator<<
       << ")";
    return os;
  }

}  // namespace OmniDSP::Design

// Specialization of fmt::formatter for OmniDSP::Design::Resample
template <>
struct fmt::formatter<OmniDSP::Design::Resample> : fmt::ostream_formatter {};

#endif  // OMNIDSP_DESIGN_RESAMPLE_HPP
