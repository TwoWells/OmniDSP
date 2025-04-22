/**
 * @file core_types.h
 * @brief Defines fundamental types, enums, and aliases used throughout the
 * OmniDSP library.
 */

#ifndef OMNIDSP_CORE_TYPES_H
#define OMNIDSP_CORE_TYPES_H

#include <complex>
#include <expected>     // Include for std::expected (requires C++23)
#include <string_view>  // For get_status_string return type (optional)
#include <type_traits>  // Required for std::is_floating_point
#include <vector>

/**
 * @brief Main namespace for the OmniDSP library.
 */
namespace OmniDSP {

//-----------------------------------------------------------------------------
// Core Type Aliases
//-----------------------------------------------------------------------------

/**
 * @brief Template alias for real floating-point types (float, double, long
 * double).
 * @details Uses SFINAE (std::enable_if) to ensure T is a floating-point type.
 * This allows functions and classes to be templated on precision.
 * @tparam T The underlying floating-point type.
 */
template <typename T>
using RealT =
    typename std::enable_if<std::is_floating_point<T>::value, T>::type;

/**
 * @brief Template alias for complex types based on a real floating-point type
 * T.
 * @tparam T The underlying floating-point type for the real and imaginary
 * parts. Must satisfy std::is_floating_point.
 * @see RealT
 */
template <typename T>
using ComplexT = std::complex<RealT<T>>;

// --- Deprecated Vector Typedefs ---
// The following typedefs are removed as per the refactoring TODO:
// using VectorReal = std::vector<Real>; // Use std::vector<double> or
// std::vector<RealT<T>> directly using VectorComplex = std::vector<Complex>; //
// Use std::vector<std::complex<double>> or std::vector<ComplexT<T>> directly
// -----------------------------------

//-----------------------------------------------------------------------------
// Status Handling
//-----------------------------------------------------------------------------

/**
 * @enum Status
 * @brief Represents the status or result code of library operations.
 * @details Intended for use as the error type E in std::expected<T, E>.
 */
enum class Status {
  Success = 0,       ///< Operation completed successfully.
  Failure,           ///< Generic failure (use specific codes when possible).
  InvalidArgument,   ///< An invalid argument was provided (e.g., null pointer,
                     ///< incorrect size/value).
  InvalidOperation,  ///< The operation is not valid in the current state or
                     ///< with the given parameters.
  AllocationError,   ///< Memory allocation failed during the operation.
  BackendError,  ///< An error occurred within the selected computation backend
                 ///< (e.g., Accelerate, oneMKL).
  UnsupportedFeature,  ///< The requested feature or parameter combination is
                       ///< not supported by the current backend or build.
  SizeMismatch,        ///< Input signals or data structures have incompatible
                       ///< dimensions for the operation.
  OutOfBounds,         ///< An index or access was outside the valid range.
                       // Add more specific codes as needed...
};

/**
 * @brief Converts a Status enum value to a human-readable string.
 * @param s The Status code to convert.
 * @return A C-style string literal describing the status. Returns "Unknown
 * Status" for invalid enum values.
 * @note This function is declared noexcept as it does not throw exceptions.
 */
inline const char* get_status_string(
    Status s) noexcept {  // Use noexcept as it shouldn't throw
  switch (s) {
    case Status::Success:
      return "Success";
    case Status::Failure:
      return "Generic Failure";
    case Status::InvalidArgument:
      return "Invalid Argument";
    case Status::InvalidOperation:
      return "Invalid Operation";
    case Status::AllocationError:
      return "Allocation Error";
    case Status::BackendError:
      return "Backend Error";
    case Status::UnsupportedFeature:
      return "Unsupported Feature";
    case Status::SizeMismatch:
      return "Size Mismatch";
    case Status::OutOfBounds:
      return "Out of Bounds";
    default:
      return "Unknown Status";
  }
}

/**
 * @brief Convenience alias for std::expected using OmniDSP::Status as the error
 * type.
 * @tparam T The type of the expected value if the operation is successful.
 * @see Status
 * @see std::expected
 * @usage OmniExpected<std::vector<double>> result = some_function();
 */
template <typename T>
using OmniExpected = std::expected<T, Status>;

//-----------------------------------------------------------------------------
// Signal Properties Enums
//-----------------------------------------------------------------------------

/**
 * @enum DataType
 * @brief Distinguishes between real and complex data types used in signals.
 */
enum class DataType {
  Real,    ///< Data consists of real numbers (e.g., float, double).
  Complex  ///< Data consists of complex numbers (e.g., std::complex<float>,
           ///< std::complex<double>).
};

/**
 * @enum Domain
 * @brief Represents the domain (e.g., time, frequency) of a signal or sequence.
 */
enum class Domain {
  Time,       ///< Standard time domain representation.
  Frequency,  ///< Frequency domain representation (e.g., the result of an FFT).
  Quefrency   ///< Quefrency domain representation (e.g., the result of cepstral
              ///< analysis, typically FFT -> log magnitude -> IFFT).
};

//-----------------------------------------------------------------------------
// Processing Configuration Enums
//-----------------------------------------------------------------------------

/**
 * @enum Window
 * @brief Specifies standard window function types used in signal processing
 * (e.g., for FFTs, filter design).
 */
enum class Window {
  Bartlett,     ///< Bartlett (triangular) window.
  Blackman,     ///< Blackman window.
  Flattop,      ///< Flat top window.
  Gaussian,     ///< Gaussian window.
  Hamming,      ///< Hamming window.
  Hann,         ///< Hann (or Hanning) window.
  Kaiser,       ///< Kaiser window (often requires a beta parameter).
  Rectangular,  ///< Rectangular window (no windowing, equivalent to multiplying
                ///< by 1).
  Triangular    ///< Triangular window (similar to Bartlett, sometimes defined
                ///< slightly differently).
                // Add more as needed
};

/**
 * @enum ConvolutionMode
 * @brief Defines the size of the output for convolution or correlation
 * operations.
 * @details Determines how boundary effects are handled and the length of the
 * resulting sequence. Also applicable to correlation operations.
 */
enum class ConvolutionMode {
  /**
   * @brief Full convolution/correlation. Output length is N + M - 1, where N
   * and M are input lengths. Includes all points of the operation, including
   * boundary effects where signals partially overlap.
   */
  Full,
  /**
   * @brief 'Same' size convolution/correlation. Output length is max(N, M).
   * The output is typically centered relative to the 'full' output, matching
   * the size of the larger input.
   */
  Same,
  /**
   * @brief 'Valid' convolution/correlation. Output length is max(N, M) - min(N,
   * M) + 1. Includes only points where the signals fully overlap, discarding
   * boundary effects.
   */
  Valid
};

/**
 * @enum Backend
 * @brief Specifies the underlying computation backend to use for accelerated
 * operations.
 */
enum class Backend {
  Stub,  ///< Basic, reference implementation (portable, potentially slow).
  Accelerate,  ///< Apple's Accelerate framework (macOS/iOS optimized).
  OneMKL  ///< Intel oneAPI Math Kernel Library (optimized for Intel CPUs/GPUs).
          // Add more backends like CUDA, OpenCL etc. later
};

/**
 * @brief Gets the human-readable name of a computation backend.
 * @param backend The Backend enum value.
 * @return A C-style string literal representing the backend's name. Returns
 * "Unknown Backend" for invalid enum values.
 * @note This function is declared noexcept as it does not throw exceptions.
 */
inline const char* get_backend_name(Backend backend) noexcept {  // Use noexcept
  switch (backend) {
    case Backend::Stub:
      return "Stub";
    case Backend::Accelerate:
      return "Accelerate";
    case Backend::OneMKL:
      return "oneMKL";  // String representation
    default:
      return "Unknown Backend";  // Should not happen with enum class
  }
}

}  // namespace OmniDSP

#endif  // OMNIDSP_CORE_TYPES_H
