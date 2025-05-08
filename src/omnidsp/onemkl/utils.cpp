/**
 * @file utils.cpp (OneMKL)
 * @brief Implements utility functions for the oneMKL backend.
 */

#include "utils.hpp"  // Corresponding header

#include <mkl.h>  // For DftiErrorMessage and DFTI status constants

#include <iostream>  // For std::cerr
// No other specific headers needed for this function's definition beyond what
// utils.hpp includes for declarations.

namespace OmniDSP::OneMKL::utils {

  Status mkl_status_to_omnidsp_status(MKL_LONG mkl_status)
  {
    if (mkl_status == DFTI_NO_ERROR) {
      return Status::Success;
    }

    // Log the specific MKL error message for debugging purposes.
    // Consider making logging conditional or configurable in a real
    // application.
    std::cerr << "MKL DFTI Error: " << DftiErrorMessage(mkl_status)
              << " (Code: " << mkl_status << ")" << std::endl;

    // Map specific MKL errors to OmniDSP errors
    // This mapping can be expanded based on the MKL errors encountered
    // and how they should be represented in the OmniDSP error system.
    if (mkl_status == DFTI_MEMORY_ERROR) {
      return Status::AllocationError;
    }
    if (mkl_status == DFTI_INVALID_CONFIGURATION
        || mkl_status == DFTI_INCONSISTENT_CONFIGURATION) {
      return Status::InvalidArgument;
    }
    if (mkl_status == DFTI_NUMBER_OF_THREADS_ERROR) {
      // This could be a backend setup issue or a configuration problem.
      return Status::BackendError;
    }
    if (mkl_status == DFTI_UNIMPLEMENTED) {
      return Status::UnsupportedFeature;
    }
    // Example of other potential mappings:
    // if (mkl_status == DFTI_COMPLEX_COMPLEX_TRANSFORM_ERROR) return
    // Status::BackendError; if (mkl_status == DFTI_1D_ERROR) return
    // Status::BackendError; if (mkl_status == DFTI_LENGTH_ERROR) return
    // Status::InvalidArgument;

    // Default to a generic backend error if no specific mapping is found.
    return Status::BackendError;
  }

}  // namespace OmniDSP::OneMKL::utils
