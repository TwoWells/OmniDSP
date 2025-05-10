/**
 * @file filter.hpp
 * @brief Defines common enumerations and basic types related to digital
 * filters.
 * @note This header is part of the types directory, separating type
 * definitions.
 */

#ifndef OMNIDSP_TYPES_FILTER_HPP
#define OMNIDSP_TYPES_FILTER_HPP

#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT if enums need it (usually not)

namespace OmniDSP {

  /**
   * @brief Enumeration of standard digital filter types.
   */
  enum class OMNIDSP_EXPORT FilterType {
    Lowpass,   ///< Lowpass filter
    Highpass,  ///< Highpass filter
    Bandpass,  ///< Bandpass filter
    Bandstop   ///< Bandstop (notch) filter
  };

  /**
   * @brief Enumeration of FIR filter design methods.
   */
  enum class OMNIDSP_EXPORT FIRFilterDesignMethod {
    WindowSinc,   /**< Design using the windowed-sinc method. */
    UserSpecified /**< User provides all coefficients directly (used more in
                     Coefs path). */
                  // ParksMcClellan, // Placeholder for future equiripple design
    // LeastSquares,   // Placeholder for future least-squares design
  };

  /**
   * @brief Enumeration of IIR filter coefficient formats.
   * Useful for specifying the desired output format from design functions
   * or the input format for plan creation from coefficients.
   */
  enum class OMNIDSP_EXPORT IIRFilterFormat {
    SOS, /**< Second-Order Sections. Array of biquad coefficients. Preferred for
            stability. */
    TransferFunction /**< Numerator (B) and Denominator (A) polynomials. Prone
                        to numerical issues for higher orders. */
  };

  // Add other general filter-related enums or simple type aliases here if
  // needed.

}  // namespace OmniDSP

#endif  // OMNIDSP_TYPES_FILTER_HPP
