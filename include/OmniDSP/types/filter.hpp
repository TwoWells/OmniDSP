/**
 * @file filter.hpp
 * @brief Defines common enumerations and basic types related to digital
 * filters.
 * @note This header is part of the types directory, separating type
 * definitions.
 */

#ifndef OMNIDSP_TYPES_FILTER_HPP
#define OMNIDSP_TYPES_FILTER_HPP

#include <ostream>      // For std::ostream
#include <string_view>  // For std::string_view

#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT if enums need it (usually not)

// Include fmt headers for custom formatter specialization
#include <fmt/core.h>     // For basic formatting
#include <fmt/ostream.h>  // Specifically for ostream_formatter

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
   * @brief Overloads the << operator for easy printing/logging of FilterType.
   * @param os The output stream.
   * @param type The FilterType enum value to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, FilterType type)
  {
    switch (type) {
      case FilterType::Lowpass:
        os << "Lowpass";
        break;
      case FilterType::Highpass:
        os << "Highpass";
        break;
      case FilterType::Bandpass:
        os << "Bandpass";
        break;
      case FilterType::Bandstop:
        os << "Bandstop";
        break;
      default:
        os << "Unknown FilterType";
        break;
    }
    return os;
  }

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
   * @brief Overloads the << operator for easy printing/logging of
   * FIRFilterDesignMethod.
   * @param os The output stream.
   * @param method The FIRFilterDesignMethod enum value to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(
      std::ostream& os, FIRFilterDesignMethod method)
  {
    switch (method) {
      case FIRFilterDesignMethod::WindowSinc:
        os << "WindowSinc";
        break;
      case FIRFilterDesignMethod::UserSpecified:
        os << "UserSpecified";
        break;
      default:
        os << "Unknown FIRFilterDesignMethod";
        break;
    }
    return os;
  }

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

  /**
   * @brief Overloads the << operator for easy printing/logging of
   * IIRFilterFormat.
   * @param os The output stream.
   * @param format The IIRFilterFormat enum value to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, IIRFilterFormat format)
  {
    switch (format) {
      case IIRFilterFormat::SOS:
        os << "SOS";
        break;
      case IIRFilterFormat::TransferFunction:
        os << "TransferFunction";
        break;
      default:
        os << "Unknown IIRFilterFormat";
        break;
    }
    return os;
  }

  // Removed old free functions:
  // get_filter_type_name(FilterType type)
  // get_fir_filter_design_method_name(FIRFilterDesignMethod method)

}  // namespace OmniDSP

// fmt::formatter specializations for direct logging with spdlog using {}
template <>
struct fmt::formatter<OmniDSP::FilterType> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::FIRFilterDesignMethod> : fmt::ostream_formatter {
};
template <>
struct fmt::formatter<OmniDSP::IIRFilterFormat> : fmt::ostream_formatter {};

#endif  // OMNIDSP_TYPES_FILTER_HPP
