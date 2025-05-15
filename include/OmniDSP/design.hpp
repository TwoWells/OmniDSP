/**
 * @file design.hpp
 * @brief Declares Design::create functions for the OmniDSP library.
 */

#ifndef OMNIDSP_DESIGN_HPP
#define OMNIDSP_DESIGN_HPP

#include "core_types.hpp"  // For OmniExpected, Status
// Params - these are the inputs to create_design
#include "params/cqt.hpp"
#include "params/fir_filter.hpp"
#include "params/iir_filter.hpp"
#include "params/resample.hpp"

// Design types - these are the outputs of create_design
// Include from the new 'design' subdirectory
#include "OmniDSP/omnidsp_export.hpp"
#include "design/cqt.hpp"         // For Design::CQT
#include "design/fir_filter.hpp"  // For Design::FIRFilter
#include "design/iir_filter.hpp"  // For Design::IIRFilter
#include "design/resample.hpp"    // For Design::Resample

namespace OmniDSP::Design {

  [[nodiscard]] OMNIDSP_EXPORT OmniExpected<Design::FIRFilter> create(
      const Params::FIRFilter& params);

  [[nodiscard]] OMNIDSP_EXPORT OmniExpected<Design::IIRFilter> create(
      const Params::IIRFilter& params);

  [[nodiscard]] OMNIDSP_EXPORT OmniExpected<Design::Resample> create(
      const Params::Resample& params);

  [[nodiscard]] OMNIDSP_EXPORT OmniExpected<Design::CQT> create(
      const Params::CQT& params);

}  // namespace OmniDSP::Design

#endif  // OMNIDSP_DESIGN_HPP
