/**
 * @file details.hpp (intelipp)
 * @brief Utility functions for the Intel IPP backend, including status code
 * conversions, error checking macros, and type mapping helpers.
 */

#ifndef OMNIDSP_INTELIPP_UTILS_HPP
#define OMNIDSP_INTELIPP_UTILS_HPP

#include <iostream>     // For std::cerr
#include <string>       // For std::string, std::to_string
#include <type_traits>  // For std::is_same_v

#include "OmniDSP/core_types.hpp"  // For OmniDSP::Status, OmniException, F32, C32 etc.
#include "ipp/ippcore.h"  // For IppStatus, ippGetStatusString, IppDataType, Ipp* types
#include "ipp/ipps.h"  // For IPP Spec/State types (IppsFIRSpec_*, IppsIIRState_*)

namespace OmniDSP::IntelIPP::Details {

  /**
   * @brief Converts IPP status codes to OmniDSP::Status.
   * @param status The IppStatus code returned by an IPP function.
   * @return The corresponding OmniDSP::Status enum value.
   */
  OmniStatus ipp_status_to_omnidsp_status(IppStatus status);

  // --- Type mapping from OmniDSP types to IPP types ---
  template <typename T>
  struct IPPTypeMap;
  template <>
  struct IPPTypeMap<F32> {
    using type = Ipp32f;
  };
  template <>
  struct IPPTypeMap<F64> {
    using type = Ipp64f;
  };
  template <>
  struct IPPTypeMap<C32> {
    using type = Ipp32fc;
  };
  template <>
  struct IPPTypeMap<C64> {
    using type = Ipp64fc;
  };

  template <typename T>
  using GetIPPType = typename IPPTypeMap<T>::type;

  // --- Type mapping for IPP FFT Spec structures ---
  template <typename T>
  struct IPPFFTSpecMap;
  template <>
  struct IPPFFTSpecMap<C32> {
    using type = IppsFFTSpec_C_32fc;
  };
  template <>
  struct IPPFFTSpecMap<C64> {
    using type = IppsFFTSpec_C_64fc;
  };
  template <>
  struct IPPFFTSpecMap<F32> {
    using type = IppsFFTSpec_R_32f;
  };
  template <>
  struct IPPFFTSpecMap<F64> {
    using type = IppsFFTSpec_R_64f;
  };

  template <typename T>
  using GetIPPFFTSpec = typename IPPFFTSpecMap<T>::type;

  // --- Type mapping for IPP FIR Spec structures ---
  template <typename T>
  struct IPPFIRSpecMap;
  template <>
  struct IPPFIRSpecMap<F32> {
    using type = IppsFIRSpec_32f;
  };
  template <>
  struct IPPFIRSpecMap<F64> {
    using type = IppsFIRSpec_64f;
  };
  template <>
  struct IPPFIRSpecMap<C32> {
    using type = IppsFIRSpec_32fc;
  };
  template <>
  struct IPPFIRSpecMap<C64> {
    using type = IppsFIRSpec_64fc;
  };

  template <typename T>
  using GetIPPFIRSpec = typename IPPFIRSpecMap<T>::type;

  // --- Type mapping for IPP IIR State structures ---
  template <typename T>
  struct IPPIIRStateMap;
  template <>
  struct IPPIIRStateMap<F32> {
    using type = IppsIIRState_32f;
  };
  template <>
  struct IPPIIRStateMap<F64> {
    using type = IppsIIRState_64f;
  };
  // IPP IIR is typically real only, add complex if supported/needed

  template <typename T>
  using GetIPPIIRState = typename IPPIIRStateMap<T>::type;

// --- Error Checking Macros ---

/**
 * @brief Checks an IppStatus. If not ippStsNoErr, throws OmniException.
 * @param ipp_stat The IppStatus code to check.
 * @param msg A descriptive message prefix for the exception.
 */
#define OMNI_CHECK_IPP_STATUS_THROW(ipp_stat, msg)                             \
  do {                                                                         \
    IppStatus _status = (ipp_stat);                                            \
    if (_status != ippStsNoErr) {                                              \
      /* Correct argument order (message, status) & name */                    \
      throw OmniException(                                                     \
          std::string(msg) + ": " + ippGetStatusString(_status)                \
              + " (IPP Code: " + std::to_string(_status)                       \
              + ")", /* Arg 1: msg */                                          \
          ::OmniDSP::IntelIPP::Details::ipp_status_to_omnidsp_status(          \
              _status) /* Arg 2: status */                                     \
      );                                                                       \
    }                                                                          \
  }                                                                            \
  while (0)

/**
 * @brief Checks an IppStatus. If not ippStsNoErr, converts to OmniDSP::Status
 * and returns it.
 * @param ipp_stat The IppStatus code to check.
 * @param msg A descriptive message prefix (currently unused, but could be for
 * logging).
 */
#define OMNI_CHECK_IPP_STATUS_RETURN(ipp_stat, msg)                            \
  do {                                                                         \
    IppStatus _status = (ipp_stat);                                            \
    if (_status != ippStsNoErr) {                                              \
      /* Optional: Log the error message here if desired */                    \
      /* std::cerr << msg << ": " << ippGetStatusString(_status) << std::endl; \
       */                                                                      \
      return ::OmniDSP::IntelIPP::Details::ipp_status_to_omnidsp_status(       \
          _status);                                                            \
    }                                                                          \
  }                                                                            \
  while (0)

}  // namespace OmniDSP::IntelIPP::Details

#endif  // OMNIDSP_INTELIPP_UTILS_HPP
