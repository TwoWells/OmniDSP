/**
 * @file iir_filter.hpp (coefs)
 * @brief Defines coefficient structures for IIR filters.
 */
#ifndef OMNIDSP_COEFS_IIR_FILTER_HPP
#define OMNIDSP_COEFS_IIR_FILTER_HPP

#include "OmniDSP/core_types.hpp"  // For OMNIDSP_EXPORT

namespace OmniDSP::Coefs {

  /**
   * @brief Represents coefficients for a single second-order section (SOS) of
   * an IIR filter.
   */
  struct OMNIDSP_EXPORT SOS {
    double b0 = 1.0;
    double b1 = 0.0;
    double b2 = 0.0;
    double a0 = 1.0;  // Typically normalized to 1 for SOS
    double a1 = 0.0;
    double a2 = 0.0;
  };

  using IIRFilterSOS = std::vector<SOS>;

}  // namespace OmniDSP::Coefs

#endif  // OMNIDSP_COEFS_IIR_HPP
