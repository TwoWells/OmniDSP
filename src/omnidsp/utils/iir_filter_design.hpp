/**
 * @file iir_filter_design.hpp
 * @brief Declares internal utility functions for creating IIRFilterSpec.
 * These are not part of the public OmniDSP::Utils API and are for use
 * within the iir_filter_design.cpp implementation or other internal utilities.
 */

#ifndef OMNIDSP_UTILS_INTERNAL_IIR_FILTER_DESIGN_HPP
#define OMNIDSP_UTILS_INTERNAL_IIR_FILTER_DESIGN_HPP

// Include any core types or forward declarations needed by potential internal
// helpers. For now, this can be minimal as there are no internal functions
// declared yet. #include "OmniDSP/core_types.hpp" // Example if Status or
// OmniExpected were returned

// Forward declare any Params or Spec types if internal helpers were to use
// them. namespace OmniDSP {
//   struct Params::IIRFilter;
//   struct IIRFilterSpec;
// }

namespace OmniDSP::Utils::Internal {

  // Placeholder for any future internal helper function declarations related to
  // IIR filter specification creation that might be defined in
  // iir_filter_design.cpp and used by other parts of the Utils module.

  // Example of a potential internal helper (currently not in your .cpp):
  /*
  [[nodiscard]] bool check_iir_parameter_consistency_internal(
      const Params::IIRFilter& params,
      double some_other_factor
  );
  */

}  // namespace OmniDSP::Utils::Internal

#endif  // OMNIDSP_UTILS_INTERNAL_IIR_FILTER_DESIGN_HPP
