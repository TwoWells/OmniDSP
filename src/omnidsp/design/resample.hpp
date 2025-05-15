/**
 * @file resample_design.hpp
 * @brief Declares internal utility functions for creating Design::Resample.
 * These are not part of the public OmniDSP::Utils API.
 */

#ifndef OMNIDSP_UTILS_INTERNAL_RESAMPLE_DESIGN_HPP
#define OMNIDSP_UTILS_INTERNAL_RESAMPLE_DESIGN_HPP

#include <cstddef>  // For size_t

#include "OmniDSP/core_types.hpp"  // For Status, OmniExpected (though not directly used by estimate_prototype_fir_order_for_resample)
#include "OmniDSP/design/resample.hpp"
#include "OmniDSP/params/resample.hpp"

namespace OmniDSP::Utils::Internal {

  /**
   * @brief Estimates the FIR filter order for a resampling prototype filter.
   * This is an internal helper function.
   *
   * @param L The upsampling factor.
   * @param M The downsampling factor.
   * @param quality The quality parameter from Params::Resample.
   * @param normalized_cutoff The normalized cutoff frequency for the prototype
   * filter.
   * @return Estimated filter order (an even number).
   */
  size_t estimate_prototype_fir_order_for_resample(
      size_t L, size_t M, int quality, double normalized_cutoff);

  // Other internal helper function declarations for resample_design.cpp could
  // go here.

}  // namespace OmniDSP::Utils::Internal

#endif  // OMNIDSP_UTILS_INTERNAL_RESAMPLE_DESIGN_HPP
