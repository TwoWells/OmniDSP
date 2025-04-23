/**
 * @file core_types.h
 * @brief Defines core data types, enums, and utilities used throughout OmniDSP.
 * @details This header establishes fundamental building blocks like type
 * aliases for real and complex numbers, status codes for operations, and
 * backend identifiers. It also includes the OmniExpected utility for error
 * handling based on C++23's std::expected.
 */

#ifndef OMNIDSP_CORE_TYPES_H
#define OMNIDSP_CORE_TYPES_H

#include <complex>
#include <cstddef>      // For size_t
#include <expected>     // Include for std::expected (requires C++23)
#include <string_view>  // For get_status_string

namespace OmniDSP {

// --- Basic Type Aliases ---

/**
 * @brief Alias for real floating-point types used in OmniDSP templates.
 * @details Ensures consistency in template definitions (e.g., float, double).
 * @tparam T The underlying floating-point type.
 */
template <typename T>
using RealT = T;

/**
 * @brief Alias for complex floating-point types used in OmniDSP templates.
 * @details Ensures consistency using std::complex<T>.
 * @tparam T The underlying floating-point type for real and imaginary parts.
 */
template <typename T>
using ComplexT = std::complex<T>;

// --- Core Enums ---

/**
 * @brief Status codes returned by OmniDSP functions and methods.
 * @details Provides standardized codes to indicate the outcome of operations,
 * facilitating error handling and checking.
 */
enum class Status {
  Success = 0,          ///< Operation completed successfully.
  Failure = 1,          ///< Generic or unknown failure.
  InvalidArgument = 2,  ///< Invalid input parameter provided (e.g., negative
                        ///< size, invalid enum).
  SizeMismatch =
      3,  ///< Input/output buffer sizes are incompatible for the operation.
  AllocationError = 4,    ///< Memory allocation failed during the operation.
  BackendError = 5,       ///< Error originating from a specific backend library
                          ///< (e.g., MKL error, Accelerate error).
  InvalidOperation = 6,   ///< Operation attempted in an invalid state (e.g.,
                          ///< execute on uninitialized plan, unsupported mode).
  NotImplemented = 7,     ///< Feature or specific case is not implemented yet.
  UnsupportedFeature = 8  ///< Requested feature is not supported by the
                          ///< selected backend or configuration.
  // Add more specific error codes as needed (e.g., FileError, Timeout)
};

/**
 * @brief Converts a Status enum to a human-readable string representation.
 * @param status The status code enum value.
 * @return A string_view describing the status. Useful for logging or error
 * messages.
 */
inline std::string_view get_status_string(Status status) noexcept {
  switch (status) {
    case Status::Success:
      return "Success";
    case Status::Failure:
      return "Failure";
    case Status::InvalidArgument:
      return "InvalidArgument";
    case Status::SizeMismatch:
      return "SizeMismatch";
    case Status::AllocationError:
      return "AllocationError";
    case Status::BackendError:
      return "BackendError";
    case Status::InvalidOperation:
      return "InvalidOperation";
    case Status::NotImplemented:
      return "NotImplemented";
    case Status::UnsupportedFeature:
      return "UnsupportedFeature";
    default:
      return "Unknown Status";
  }
}

/**
 * @brief Alias for std::expected using OmniDSP::Status as the error type.
 * @details Simplifies return types for functions that can fail, providing a
 * standard way to return either a value or a Status error code. Requires C++23.
 * Example Usage:
 * OmniExpected<int> result = my_function();
 * if (result) { process(*result); } else { handle_error(result.error()); }
 * @tparam T The type of the expected value on success.
 */
template <typename T>
using OmniExpected = std::expected<T, Status>;

/**
 * @brief Specifies the backend implementation library to use for computations.
 * @details Allows selecting between different optimized libraries or a default
 * portable implementation at runtime via OmniDSP::create.
 */
enum class Backend {
  Default,     ///< Portable C++ implementation with Highway SIMD acceleration.
  Accelerate,  ///< Apple Accelerate framework (macOS/iOS).
  OneMKL       ///< Intel oneMKL library.
  // Add other backends here (e.g., CUDA, OpenCL, ArmComputeLibrary)
};

/**
 * @brief Gets the string name corresponding to a Backend enum value.
 * @param backend The backend enum value.
 * @return A string_view representing the backend name.
 */
inline std::string_view get_backend_name(Backend backend) noexcept {
  switch (backend) {
    case Backend::Default:
      return "Default";
    case Backend::Accelerate:
      return "Accelerate";
    case Backend::OneMKL:
      return "oneMKL";
    default:
      return "Unknown Backend";
  }
}

// --- Utility Namespace (Optional) ---
// Could contain small helper functions or constants used across modules,
// if they don't fit better elsewhere.
namespace Detail {
// Example: Type trait to check if a type is std::complex
template <typename T>
struct is_complex : std::false_type {};
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};
template <typename T>
constexpr bool is_complex_v = is_complex<T>::value;

// Example: Get underlying real type from complex or real
template <typename T>
struct ValueType {
  using type = T;
};
template <typename T>
struct ValueType<std::complex<T>> {
  using type = T;
};
template <typename T>
using UnderlyingRealT = typename ValueType<T>::type;
}  // namespace Detail

}  // namespace OmniDSP

#endif  // OMNIDSP_CORE_TYPES_H
