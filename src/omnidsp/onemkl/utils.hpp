/**
 * @file details.hpp (OneMKL)
 * @brief Utility function declarations and template definitions for the oneMKL
 * backend.
 */

#ifndef OMNIDSP_ONEMKL_UTILS_HPP
#define OMNIDSP_ONEMKL_UTILS_HPP

#include <mkl.h>  // For MKL_LONG, DFTI_CONFIG_VALUE, DFTI_SINGLE, DFTI_DOUBLE

#include <OmniDSP/core_types.hpp>  // For OmniDSP::Status
#include <stdexcept>    // For std::logic_error (used in get_dfti_precision)
#include <string>       // For std::string (DftiErrorMessage might involve it)
#include <type_traits>  // For std::is_floating_point_v, std::is_same_v

namespace OmniDSP::OneMKL::Details {

  /**
   * @brief Converts MKL DFTI status codes to OmniDSP::Status.
   * @param mkl_status The MKL_LONG status code returned by a DFTI function.
   * @return The corresponding OmniDSP::Status enum value.
   */
  Status mkl_status_to_omnidsp_status(MKL_LONG mkl_status);

  /**
   * @brief Helper to get the corresponding DFTI precision enum value for a real
   * type.
   * @tparam T_Real The real floating-point type (float or double).
   * @return DFTI_CONFIG_VALUE (DFTI_SINGLE or DFTI_DOUBLE).
   * @throws std::logic_error (via static_assert) if T_Real is not float or
   * double.
   */
  template <typename T_Real>
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
    // The static_assert above ensures this part is unreachable for valid types,
    // but some compilers might warn about missing return without an else.
    // However, for constexpr, this structure is fine.
  }

}  // namespace OmniDSP::OneMKL::Details

#endif  // OMNIDSP_ONEMKL_UTILS_HPP
