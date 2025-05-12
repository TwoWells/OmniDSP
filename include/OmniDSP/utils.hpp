/**
 * @file utils.hpp
 * @brief Declares general utility functions for the OmniDSP library,
 * such as spec creation functions.
 */

#ifndef OMNIDSP_UTILS_HPP
#define OMNIDSP_UTILS_HPP

#include "core_types.hpp"  // For OmniExpected, Status
// Params
#include "params/cqt.hpp"
#include "params/fir_filter.hpp"
#include "params/resample.hpp"
// Specs
#include "cqt.hpp"
#include "filter.hpp"
#include "resample.hpp"

namespace OmniDSP::Utils {

  // Forward declarations for Params types if not fully included above
  // struct FIRFilterParams; (already included via params/fir_filter.hpp)
  // struct ResampleParams; (already included via params/resample.hpp)
  // struct CQTParams;     (already included via params/cqt.hpp)

  /**
   * @brief Creates a fully resolved FIRFilterSpec from user-provided
   * parameters.
   *
   * This function takes `FIRFilterParams` (which should have already been
   * validated upon its own construction) and performs necessary calculations,
   * such as estimating the filter order if not explicitly provided, to produce
   * a complete `FIRFilterSpec`.
   *
   * @param params The user-defined parameters for the FIR filter.
   * @return An OmniExpected containing the `FIRFilterSpec` on success, or a
   * `Status` code on failure.
   */
  [[nodiscard]] OMNIDSP_EXPORT OmniExpected<FIRFilterSpec> create_spec(
      const FIRFilterParams& params);

  /**
   * @brief Creates a fully resolved ResampleSpec from user-provided parameters.
   *
   * This function takes `ResampleParams` and determines the internal prototype
   * FIR filter specification needed for the resampling process.
   *
   * @param params The user-defined parameters for the resampling operation.
   * @return An OmniExpected containing the `ResampleSpec` on success, or a
   * `Status` code on failure.
   */
  [[nodiscard]] OMNIDSP_EXPORT OmniExpected<ResampleSpec> create_spec(
      const ResampleParams& params);

  /**
   * @brief Creates a fully resolved CQTSpec from user-provided CQT parameters.
   *
   * This function takes `CQTParams` and calculates all necessary derived
   * parameters for the CQT, including bin frequencies, Q factor, hop length,
   * and per-octave processing details like FFT lengths and kernel sizes.
   *
   * @param params The user-defined parameters for the Constant-Q Transform.
   * @return An OmniExpected containing the `CQTSpec` on success, or a `Status`
   * code on failure.
   */
  [[nodiscard]] OMNIDSP_EXPORT OmniExpected<CQTSpec> create_spec(
      const CQTParams& params);

  // Add declarations for other Utils::create_spec overloads here as they are
  // developed e.g., for IIRFilterParams, etc.

}  // namespace OmniDSP::Utils

#endif  // OMNIDSP_UTILS_HPP
