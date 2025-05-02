/**
 * @file resample_utils.hpp
 * @brief Common utility functions for resampling implementations.
 */
// *** UPDATED Include Guard ***
#ifndef OMNIDSP_UTILS_RESAMPLE_UTILS_HPP
#define OMNIDSP_UTILS_RESAMPLE_UTILS_HPP

#include <OmniDSP/core_types.hpp>  // For Status
#include <cstddef>                 // For size_t

namespace OmniDSP::Utils {

  /**
   * @brief Calculates integer up/down sampling factors (L/M) for a given rate
   * change.
   * @param in_rate Input sample rate.
   * @param out_rate Desired output sample rate.
   * @param L Output reference for the upsampling factor.
   * @param M Output reference for the downsampling factor.
   * @return Status::Success if factors are calculated successfully,
   * Status::InvalidArgument if rates are invalid, Status::Failure on other
   * errors (e.g., overflow).
   */
  Status calculate_resampling_factors(
      double in_rate, double out_rate, size_t& L, size_t& M);

}  // namespace OmniDSP::Utils

// *** UPDATED Include Guard ***
#endif  // OMNIDSP_UTILS_RESAMPLE_UTILS_HPP
