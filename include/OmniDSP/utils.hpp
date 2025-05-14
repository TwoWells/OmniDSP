/**
 * @file utils.hpp
 * @brief Declares general utility functions for the OmniDSP library,
 * such as functions to create Design objects from Params objects.
 */

#ifndef OMNIDSP_UTILS_HPP
#define OMNIDSP_UTILS_HPP

#include "core_types.hpp"  // For OmniExpected, Status
// Params - these are the inputs to create_spec
#include "params/cqt.hpp"
#include "params/fir_filter.hpp"
#include "params/iir_filter.hpp"
#include "params/resample.hpp"

// Design types - these are the outputs of create_spec
// Include from the new 'design' subdirectory
#include "OmniDSP/omnidsp_export.hpp"
#include "design/cqt.hpp"         // For Design::CQT
#include "design/fir_filter.hpp"  // For Design::FIRFilter
#include "design/iir_filter.hpp"  // For Design::IIRFilter
#include "design/resample.hpp"    // For Design::Resample

namespace OmniDSP::Utils {

  [[nodiscard]] OMNIDSP_EXPORT OmniExpected<Design::FIRFilter> create_spec(
      const Params::FIRFilter& params);

  [[nodiscard]] OMNIDSP_EXPORT OmniExpected<Design::IIRFilter> create_spec(
      const Params::IIRFilter& params);

  [[nodiscard]] OMNIDSP_EXPORT OmniExpected<Design::Resample> create_spec(
      const Params::Resample& params);

  [[nodiscard]] OMNIDSP_EXPORT OmniExpected<Design::CQT> create_spec(
      const Params::CQT& params);

}  // namespace OmniDSP::Utils

#endif  // OMNIDSP_UTILS_HPP
