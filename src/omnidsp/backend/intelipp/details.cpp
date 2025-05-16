/**
 * @file details.cpp
 * @brief Implements utility functions for the Intel IPP backend.
 */

#include "details.hpp"  // Corresponding header (should be details.hpp based on filename)

#include <ippcore.h>  // For IppStatus and ippGetStatusString

#include <iostream>  // For std::cerr (used for logging IPP errors)
#include <string>    // For std::string (used in error messages)

// Assuming OmniDSP::Status is defined in core_types.hpp, which should be
// included via details.hpp #include <OmniDSP/core_types.hpp> // Already
// included via details.hpp typically

namespace OmniDSP::IntelIPP::Details {

  /**
   * @brief Converts an Intel IPP status code (IppStatus) to the library's
   * generic OmniDSP::Status.
   *
   * This function maps specific IPP error codes to corresponding OmniDSP status
   * codes to provide a consistent error handling mechanism across different
   * backends. It also prints the IPP error message to std::cerr for debugging
   * purposes.
   *
   * @param status The IppStatus code returned by an Intel IPP function.
   * @return Status The equivalent OmniDSP::Status enum value.
   * Returns OmniDSP::Status::Success if IppStatus is ippStsNoErr.
   * Returns OmniDSP::Status::InvalidArgument for common IPP argument/size
   * errors. Returns OmniDSP::Status::AllocationError for ippStsMemAllocErr.
   * Returns OmniDSP::Status::InvalidOperation for ippStsContextMatchErr.
   * Returns OmniDSP::Status::BackendError for other unmapped IPP errors.
   */
  OmniStatus ipp_status_to_omnidsp_status(IppStatus status)
  {
    if (status == ippStsNoErr) {
      return OmniStatus::Success;
    }

    // Log the IPP error to standard error for easier debugging
    std::cerr << "IPP Error: " << ippGetStatusString(status)
              << " (Code: " << status << ")" << std::endl;

    // Map specific IPP errors to OmniDSP status codes
    if (status == ippStsNullPtrErr || status == ippStsSizeErr
        || status == ippStsStepErr || status == ippStsBadArgErr
        || status == ippStsOutOfRangeErr || status == ippStsFftOrderErr
        || status == ippStsFftFlagErr) {
      return OmniStatus::InvalidArgument;
    }
    if (status == ippStsMemAllocErr) {
      return OmniStatus::AllocationError;
    }
    if (status == ippStsContextMatchErr) {
      // This could indicate a misuse of an IPP state/spec object,
      // or an operation not valid for the current state.
      return OmniStatus::InvalidOperation;
    }
    // Add more specific mappings here as needed based on IPP documentation
    // and errors encountered during development.
    // For example:
    // if (status == ippStsSingularErr) return Status::NumericError; // Example
    // if (status == ippStsDivByZeroErr) return Status::NumericError; // Example

    // Default to a generic backend error if no specific mapping is found.
    return OmniStatus::BackendError;
  }

  // ... (definitions of other helpers if moved from header) ...

}  // namespace OmniDSP::IntelIPP::Details
