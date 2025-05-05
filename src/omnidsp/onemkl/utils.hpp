/**
 * @file utils.hpp (onemkl)
 * @brief Utility functions for the oneMKL backend, including status code
 * conversions.
 */
// No include guards as requested

#include <mkl.h>  // For MKL_LONG and DftiErrorMessage

#include <OmniDSP/core_types.hpp>  // For OmniDSP::Status
#include <iostream>                // For std::cerr
#include <stdexcept>  // For std::logic_error (used in get_dfti_precision)
#include <string>     // For std::string
#include <type_traits>  // For std::is_floating_point_v, std::is_same_v (needed for get_dfti_precision)

// Removed: #include <ippcore.h>

namespace OmniDSP::onemkl::utils {  // Updated namespace for clarity

  // Removed: ipp_status_to_omnidsp_status function

  /**
   * @brief Converts MKL DFTI status codes to OmniDSP::Status.
   * @param status The MKL_LONG status code returned by a DFTI function.
   * @return The corresponding OmniDSP::Status enum value.
   */
  inline Status mkl_status_to_omnidsp_status(MKL_LONG status)
  {
    if (status == DFTI_NO_ERROR) {
      return Status::Success;
    }
    // Log the specific MKL error message
    std::cerr << "MKL DFTI Error: " << DftiErrorMessage(status)
              << " (Code: " << status << ")" << std::endl;

    // Map specific MKL errors to OmniDSP errors
    if (status == DFTI_MEMORY_ERROR) return Status::AllocationError;
    if (status == DFTI_INVALID_CONFIGURATION
        || status == DFTI_INCONSISTENT_CONFIGURATION)
      return Status::InvalidArgument;
    if (status == DFTI_NUMBER_OF_THREADS_ERROR)
      return Status::BackendError;  // Or maybe a config error?
    if (status == DFTI_UNIMPLEMENTED) return Status::UnsupportedFeature;
    // Add more specific mappings if needed
    // ...

    // Default to a generic backend error
    return Status::BackendError;
  }

  /**
   * @brief Helper to get the corresponding DFTI precision enum value for a real
   * type.
   * @tparam T_Real The real floating-point type (float or double).
   * @return DFTI_CONFIG_VALUE (DFTI_SINGLE or DFTI_DOUBLE).
   * @throws std::logic_error (via static_assert) if T_Real is not float or
   * double.
   */
  template <typename T_Real>  // Template on the REAL type
  constexpr DFTI_CONFIG_VALUE get_dfti_precision()
  {
    static_assert(
        std::is_floating_point_v<T_Real>,
        "get_dfti_precision requires a floating-point type");
    if constexpr (std::is_same_v<T_Real, float>) {
      return DFTI_SINGLE;
    }
    else if constexpr (std::is_same_v<T_Real, double>) {
      return DFTI_DOUBLE;
    }
    // The static_assert above ensures this part is unreachable for valid types.
  }

}  // namespace OmniDSP::onemkl::utils

// No endif for include guard
