/**
 * @file iir_filter.hpp (coefs)
 * @brief Defines coefficient structures for IIR filters.
 */
#ifndef OMNIDSP_COEFS_IIR_FILTER_HPP
#define OMNIDSP_COEFS_IIR_FILTER_HPP

#include <ostream>  // For std::ostream
#include <vector>   // For std::vector (used by IIRFilterSOS)

#include "OmniDSP/core_types.hpp"  // For OMNIDSP_EXPORT, std::vector

// Include fmt headers for custom formatter specialization
#include <fmt/core.h>     // For basic formatting
#include <fmt/ostream.h>  // Specifically for ostream_formatter

namespace OmniDSP::Coefs {

  /**
   * @brief Represents coefficients for a single second-order section (SOS) of
   * an IIR filter.
   * @details This structure holds the six coefficients that define a biquad
   * filter section. The denominator coefficient a0 is typically normalized
   * to 1.0.
   */
  struct OMNIDSP_EXPORT SOS {
    double b0 = 1.0;  ///< Numerator coefficient b0.
    double b1 = 0.0;  ///< Numerator coefficient b1.
    double b2 = 0.0;  ///< Numerator coefficient b2.
    double a0
        = 1.0;  ///< Denominator coefficient a0 (typically normalized to 1).
    double a1 = 0.0;  ///< Denominator coefficient a1.
    double a2 = 0.0;  ///< Denominator coefficient a2.
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of Coefs::SOS.
   * @param os The output stream.
   * @param sos The Coefs::SOS object to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, const SOS& sos)
  {
    os << "SOS(b0:" << sos.b0 << ", b1:" << sos.b1 << ", b2:" << sos.b2
       << ", a0:" << sos.a0 << ", a1:" << sos.a1 << ", a2:" << sos.a2 << ")";
    return os;
  }

  /**
   * @brief Type alias for a vector of Second-Order Sections, representing an
   * IIR filter.
   */
  using IIRFilterSOS = std::vector<SOS>;

}  // namespace OmniDSP::Coefs

// Specialization of fmt::formatter for OmniDSP::Coefs::SOS
template <>
struct fmt::formatter<OmniDSP::Coefs::SOS> : fmt::ostream_formatter {};

#endif  // OMNIDSP_COEFS_IIR_FILTER_HPP
