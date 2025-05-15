#ifndef OMNIDSP_CORE_TYPES_HPP
#define OMNIDSP_CORE_TYPES_HPP

#include <complex>
#include <expected>  // For std::expected, std::unexpected
#include <ostream>   // For std::ostream (for operator<<)
#include <span>
#include <stdexcept>  // Required for std::runtime_error
#include <string>     // Required for std::string
#include <string_view>  // Required for string views if still used, though operator<< will dominate
#include <type_traits>  // Required for type traits helpers
#include <vector>

#include "OmniDSP/omnidsp_export.hpp"

// Include fmt headers for custom formatter specialization
#include <fmt/core.h>     // For basic formatting
#include <fmt/ostream.h>  // Specifically for ostream_formatter

namespace OmniDSP {

  /**
   * @brief Enumeration of core status codes for OmniDSP operations.
   */
  enum class OmniStatus {
    Success = 0,
    Failure = 1,
    InvalidArgument = 2,
    SizeMismatch = 3,
    AllocationError = 4,
    BackendError = 5,
    NotInitialized = 6,
    InvalidOperation = 7,
    UnsupportedFeature = 8,
    OutOfBounds = 9,
    Timeout = 10,
    NotImplemented = 11,
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of OmniStatus.
   * @param os The output stream.
   * @param status The OmniStatus enum value to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, OmniStatus status)
  {
    switch (status) {
      case OmniStatus::Success:
        os << "Success";
        break;
      case OmniStatus::Failure:
        os << "Failure";
        break;
      case OmniStatus::InvalidArgument:
        os << "InvalidArgument";
        break;
      case OmniStatus::SizeMismatch:
        os << "SizeMismatch";
        break;
      case OmniStatus::AllocationError:
        os << "AllocationError";
        break;
      case OmniStatus::BackendError:
        os << "BackendError";
        break;
      case OmniStatus::NotInitialized:
        os << "NotInitialized";
        break;
      case OmniStatus::InvalidOperation:
        os << "InvalidOperation";
        break;
      case OmniStatus::NotImplemented:
        os << "NotImplemented";
        break;
      case OmniStatus::UnsupportedFeature:
        os << "UnsupportedFeature";
        break;
      case OmniStatus::OutOfBounds:
        os << "OutOfBounds";
        break;
      case OmniStatus::Timeout:
        os << "Timeout";
        break;
      default:
        os << "Unknown OmniStatus";
        break;
    }
    return os;
  }

  /**
   * @brief Enumeration of available backend types in OmniDSP.
   */
  enum class BackendType { Default, Accelerate, OneMKL, IntelIPP, Dispatcher };

  /**
   * @brief Overloads the << operator for easy printing/logging of BackendType.
   * @param os The output stream.
   * @param backend The BackendType enum value to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, BackendType backend)
  {
    switch (backend) {
      case BackendType::Default:
        os << "Default";
        break;
      case BackendType::Accelerate:
        os << "Accelerate";
        break;
      case BackendType::OneMKL:
        os << "oneMKL";
        break;
      case BackendType::IntelIPP:
        os << "IntelIPP";
        break;
      case BackendType::Dispatcher:
        os << "Dispatcher";
        break;
      default:
        os << "Unknown BackendType";
        break;
    }
    return os;
  }

  /**
   * @brief Enumeration of operation categories for backend dispatching.
   */
  enum class OperationCategory {
    FFT,
    RFFT,
    Convolution,
    Correlation,
    FIRFilter,
    IIRFilter,
    Resample,
    CQT,
    Windowing,
    FilterDesign,
    GenericFallback
  };

  /**
   * @brief Overloads the << operator for easy printing/logging of
   * OperationCategory.
   * @param os The output stream.
   * @param category The OperationCategory enum value to print.
   * @return A reference to the output stream.
   */
  inline std::ostream& operator<<(std::ostream& os, OperationCategory category)
  {
    switch (category) {
      case OperationCategory::FFT:
        os << "FFT";
        break;
      case OperationCategory::RFFT:
        os << "RFFT";
        break;
      case OperationCategory::Convolution:
        os << "Convolution";
        break;
      case OperationCategory::Correlation:
        os << "Correlation";
        break;
      case OperationCategory::FIRFilter:
        os << "FIRFilter";
        break;
      case OperationCategory::IIRFilter:
        os << "IIRFilter";
        break;
      case OperationCategory::Resample:
        os << "Resample";
        break;
      case OperationCategory::CQT:
        os << "CQT";
        break;
      case OperationCategory::Windowing:
        os << "Windowing";
        break;
      case OperationCategory::FilterDesign:
        os << "FilterDesign";
        break;
      case OperationCategory::GenericFallback:
        os << "GenericFallback";
        break;
      default:
        os << "Unknown OperationCategory";
        break;
    }
    return os;
  }

  // --- Type Aliases ---
  using F32 = float;
  using F64 = double;
  using C32 = std::complex<F32>;
  using C64 = std::complex<F64>;

  using F32Vec = std::vector<F32>;
  using F64Vec = std::vector<F64>;
  using C32Vec = std::vector<C32>;
  using C64Vec = std::vector<C64>;

  // --- OmniExpected Alias ---
  template <typename T>
  using OmniExpected = std::expected<T, OmniStatus>;

  /**
   * @brief Custom exception class for OmniDSP library errors.
   * Derives from std::runtime_error and includes an OmniStatus code.
   */
  class OmniException : public std::runtime_error {
   private:
    OmniStatus error_status_;

   public:
    /**
     * @brief Constructs an OmniException with a message and status code.
     * @param message The error message.
     * @param status The OmniStatus code associated with the error. Defaults to
     * OmniStatus::Failure.
     */
    explicit OmniException(
        const std::string& message, OmniStatus status = OmniStatus::Failure)
        : std::runtime_error(message), error_status_(status)
    {}
    /**
     * @brief Constructs an OmniException with a C-string message and status
     * code.
     * @param message The error message as a C-string.
     * @param status The OmniStatus code associated with the error. Defaults to
     * OmniStatus::Failure.
     */
    explicit OmniException(
        const char* message, OmniStatus status = OmniStatus::Failure)
        : std::runtime_error(message), error_status_(status)
    {}
    /**
     * @brief Gets the OmniStatus code associated with this exception.
     * @return The OmniStatus code.
     */
    OmniStatus get_status() const noexcept { return error_status_; }
  };

  // --- Utility Type Traits ---
  /**
   * @namespace OmniDSP::Utils
   * @brief Contains utility type traits and helper functions for the OmniDSP
   * library.
   */
  namespace Utils {
    /** @brief Metafunction to get the corresponding complex type for a given
     * real or complex type. */
    template <typename T>
    struct GetComplex {
      using type
          = void;  ///< Default to void if T is not F32, F64, C32, or C64.
    };
    template <>
    struct GetComplex<F32> {
      using type = C32;  ///< float -> std::complex<float>
    };
    template <>
    struct GetComplex<F64> {
      using type = C64;  ///< double -> std::complex<double>
    };
    template <>
    struct GetComplex<C32> {
      using type = C32;  ///< std::complex<float> -> std::complex<float>
    };
    template <>
    struct GetComplex<C64> {
      using type = C64;  ///< std::complex<double> -> std::complex<double>
    };
    /** @brief Type alias for `typename GetComplex<T>::type`. */
    template <typename T>
    using GetComplexType = typename GetComplex<T>::type;

    /** @brief Metafunction to get the corresponding real type for a given real
     * or complex type. */
    template <typename T>
    struct GetReal {
      using type
          = void;  ///< Default to void if T is not F32, F64, C32, or C64.
    };
    template <>
    struct GetReal<F32> {
      using type = F32;  ///< float -> float
    };
    template <>
    struct GetReal<F64> {
      using type = F64;  ///< double -> double
    };
    template <>
    struct GetReal<C32> {
      using type = F32;  ///< std::complex<float> -> float
    };
    template <>
    struct GetReal<C64> {
      using type = F64;  ///< std::complex<double> -> double
    };
    /** @brief Type alias for `typename GetReal<T>::type`. */
    template <typename T>
    using GetRealType = typename GetReal<T>::type;

    /** @brief Type trait to check if a type is one of OmniDSP's complex types
     * (C32 or C64). */
    template <typename T>
    struct IsComplex : std::false_type {};
    template <>
    struct IsComplex<C32> : std::true_type {};
    template <>
    struct IsComplex<C64> : std::true_type {};
    /** @brief Value of the IsComplex type trait. True if T is C32 or C64, false
     * otherwise. */
    template <typename T>
    inline constexpr bool IsComplex_v = IsComplex<T>::value;
  }  // namespace Utils

}  // namespace OmniDSP

// fmt::formatter specializations for direct logging with spdlog using {}
template <>
struct fmt::formatter<OmniDSP::OmniStatus> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::BackendType> : fmt::ostream_formatter {};
template <>
struct fmt::formatter<OmniDSP::OperationCategory> : fmt::ostream_formatter {};

#endif  // OMNIDSP_CORE_TYPES_HPP
