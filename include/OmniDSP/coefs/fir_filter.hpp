/**
 * @file fir_filter.hpp (coefs)
 * @brief Defines coefficient types for FIR filters.
 */
#ifndef OMNIDSP_COEFS_FIR_FILTER_HPP
#define OMNIDSP_COEFS_FIR_FILTER_HPP

#include <vector>

#include "OmniDSP/core_types.hpp"  // For F32, F64 etc. if needed for template, though std::vector<T> is generic

namespace OmniDSP::Coefs {

  /**
   * @brief Type alias for a vector of FIR filter coefficients (taps).
   * @tparam T The data type of the coefficients (e.g., F32, F64, C32, C64).
   */
  template <typename T>
  using FIRFilter = std::vector<T>;

}  // namespace OmniDSP::Coefs

#endif  // OMNIDSP_COEFS_FIR_FILTER_HPP
