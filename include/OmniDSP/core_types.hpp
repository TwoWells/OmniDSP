#ifndef OMNIDSP_CORE_TYPES_HPP
#define OMNIDSP_CORE_TYPES_HPP

#include <complex>
#include <expected>  // For std::expected, std::unexpected
#include <span>
#include <stdexcept>    // Required for std::runtime_error
#include <string>       // Required for std::string
#include <string_view>  // Required for get_backend_name, get_status_string
#include <type_traits>  // Required for type traits helpers
#include <vector>

namespace OmniDSP {

  // --- Core Status Enum ---

  /**
   * @brief Status codes for OmniDSP operations.
   */
  enum class Status {
    Success = 0,          ///< Operation completed successfully.
    Failure = 1,          ///< General failure.
    InvalidArgument = 2,  ///< Invalid argument provided to a function.
    SizeMismatch = 3,     ///< Input/output sizes are incompatible.
    AllocationError = 4,  ///< Memory allocation failed.
    BackendError = 5,     ///< Error originating from the backend library (e.g.,
                          ///< MKL, IPP).
    NotInitialized = 6,   ///< Object or plan was not properly initialized.
    InvalidOperation = 7,    ///< Operation is not valid in the current state.
    UnsupportedFeature = 8,  ///< Requested feature is not supported by the
                             ///< backend or configuration.
    OutOfBounds = 9,         ///< Index or access was out of bounds.
    Timeout = 10,            ///< Operation timed out (if applicable).
    NotImplemented = 11,     ///< Feature or function is not implemented yet.
    // Add more specific codes as needed
  };

  /**
   * @brief Converts a Status enum to a human-readable string representation.
   * @param status The status code enum value.
   * @return A string_view describing the status. Useful for logging or error
   * messages.
   */
  inline std::string_view get_status_string(Status status) noexcept
  {
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
      case Status::NotInitialized:
        return "NotInitialized";
      case Status::InvalidOperation:
        return "InvalidOperation";
      case Status::NotImplemented:
        return "NotImplemented";
      case Status::UnsupportedFeature:
        return "UnsupportedFeature";
      case Status::OutOfBounds:
        return "OutOfBounds";
      case Status::Timeout:
        return "Timeout";
      default:
        return "Unknown Status";
    }
  }

  // --- Backend Selection Enum ---

  /**
   * @brief Specifies the backend implementation library to use for
   * computations.
   * @details Allows selecting between different optimized libraries or a
   * default portable implementation at runtime via OmniDSP::create.
   */
  enum class Backend {
    Default,  ///< Portable C++ implementation with potential SIMD acceleration.
    Accelerate,  ///< Apple Accelerate framework (macOS/iOS).
    OneMKL,      ///< Intel oneMKL library.
    IntelIPP     ///< Intel Integrated Performance Primitives library.
    // Add other backends here (e.g., CUDA, OpenCL, ArmComputeLibrary)
  };

  /**
   * @brief Gets the string name corresponding to a Backend enum value.
   * @param backend The backend enum value.
   * @return A string_view representing the backend name.
   */
  inline std::string_view get_backend_name(Backend backend) noexcept
  {
    switch (backend) {
      case Backend::Default:
        return "Default";
      case Backend::Accelerate:
        return "Accelerate";
      case Backend::OneMKL:
        return "oneMKL";
      case Backend::IntelIPP:
        return "IntelIPP";
      default:
        // Handle potential future additions gracefully
        return "Unknown Backend";
    }
  }

  // --- Type Aliases ---

  using F32 = float;   ///< 32-bit floating point type.
  using F64 = double;  ///< 64-bit floating point type.
  using C32
      = std::complex<F32>;  ///< Complex 32-bit float type (float complex).
  using C64
      = std::complex<F64>;  ///< Complex 64-bit float type (double complex).

  using F32Vec = std::vector<F32>;  ///< Vector of 32-bit floats.
  using F64Vec = std::vector<F64>;  ///< Vector of 64-bit floats.
  using C32Vec = std::vector<C32>;  ///< Vector of 32-bit complex floats.
  using C64Vec = std::vector<C64>;  ///< Vector of 64-bit complex floats.

  // --- OmniExpected Alias ---

  /**
   * @brief Alias for std::expected using OmniDSP::Status as the error type.
   * @tparam T The expected value type.
   */
  template <typename T>
  using OmniExpected = std::expected<T, Status>;

  // --- OmniException Definition ---

  /**
   * @brief Base exception class for OmniDSP runtime errors,
   * particularly useful for constructor failures. Aligns with OmniExpected.
   */
  class OmniException : public std::runtime_error {
   private:
    Status error_status_;  // Store the associated status code

   public:
    /**
     * @brief Construct with an error message and status code.
     * @param message A descriptive error message.
     * @param status The OmniDSP::Status code representing the error type.
     * Defaults to Status::Failure.
     */
    explicit OmniException(
        const std::string& message, Status status = Status::Failure)
        : std::runtime_error(message), error_status_(status)
    {}

    /**
     * @brief Construct with an error message (C-string) and status code.
     * @param message A descriptive error message (C-string).
     * @param status The OmniDSP::Status code representing the error type.
     * Defaults to Status::Failure.
     */
    explicit OmniException(const char* message, Status status = Status::Failure)
        : std::runtime_error(message), error_status_(status)
    {}

    /**
     * @brief Get the OmniDSP status code associated with this exception.
     * @return Status The status code.
     */
    Status get_status() const noexcept { return error_status_; }

    // what() is inherited from std::runtime_error to get the message string.
  };

  // --- Utility Type Traits ---
  namespace Utils {
    /**
     * @brief Metafunction to get the complex type corresponding to a real or
     * complex type.
     * @tparam T The input type (F32, F64, C32, C64).
     */
    template <typename T>
    struct GetComplex {   // Renamed from GetComplexType
      using type = void;  // Default to void for non-handled types
    };
    template <>
    struct GetComplex<F32> {
      using type = C32;
    };
    template <>
    struct GetComplex<F64> {
      using type = C64;
    };
    template <>
    struct GetComplex<C32> {
      using type = C32;
    };  // Complex maps to itself
    template <>
    struct GetComplex<C64> {
      using type = C64;
    };  // Complex maps to itself

    /**
     * @brief Alias template for GetComplex<T>::type.
     * @tparam T The input type.
     */
    template <typename T>
    using GetComplexType =
        typename GetComplex<T>::type;  // Renamed from GetComplexT

    /**
     * @brief Metafunction to get the real type corresponding to a real or
     * complex type.
     * @tparam T The input type (F32, F64, C32, C64).
     */
    template <typename T>
    struct GetReal {      // Renamed from GetRealType
      using type = void;  // Default to void
    };
    template <>
    struct GetReal<F32> {
      using type = F32;
    };  // Real maps to itself
    template <>
    struct GetReal<F64> {
      using type = F64;
    };  // Real maps to itself
    template <>
    struct GetReal<C32> {
      using type = F32;
    };
    template <>
    struct GetReal<C64> {
      using type = F64;
    };

    /**
     * @brief Alias template for GetReal<T>::type.
     * @tparam T The input type.
     */
    template <typename T>
    using GetRealType = typename GetReal<T>::type;  // Renamed from GetRealT

    /**
     * @brief Type trait to check if a type is OmniDSP::C32 or OmniDSP::C64.
     * @tparam T The type to check.
     */
    template <typename T>
    struct IsComplex : std::false_type {
    };  // Renamed from is_complex (PascalCase)

    template <>
    struct IsComplex<C32> : std::true_type {};  // Specialization for C32
    template <>
    struct IsComplex<C64> : std::true_type {};  // Specialization for C64

    /**
     * @brief Helper variable template for IsComplex. True if T is C32 or C64.
     * @tparam T The type to check.
     */
    template <typename T>
    inline constexpr bool IsComplex_v
        = IsComplex<T>::value;  // Renamed from is_complex_v (using _v suffix)

  }  // namespace Utils

}  // namespace OmniDSP

#endif  // OMNIDSP_CORE_TYPES_HPP
