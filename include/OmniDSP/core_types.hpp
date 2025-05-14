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

#include "OmniDSP/omnidsp_export.hpp"

namespace OmniDSP {

  // --- Core OmniStatus Enum ---
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

  inline std::string_view get_status_string(OmniStatus status) noexcept
  {
    switch (status) {
      case OmniStatus::Success:
        return "Success";
      case OmniStatus::Failure:
        return "Failure";
      case OmniStatus::InvalidArgument:
        return "InvalidArgument";
      case OmniStatus::SizeMismatch:
        return "SizeMismatch";
      case OmniStatus::AllocationError:
        return "AllocationError";
      case OmniStatus::BackendError:
        return "BackendError";
      case OmniStatus::NotInitialized:
        return "NotInitialized";
      case OmniStatus::InvalidOperation:
        return "InvalidOperation";
      case OmniStatus::NotImplemented:
        return "NotImplemented";
      case OmniStatus::UnsupportedFeature:
        return "UnsupportedFeature";
      case OmniStatus::OutOfBounds:
        return "OutOfBounds";
      case OmniStatus::Timeout:
        return "Timeout";
      default:
        return "Unknown OmniStatus";
    }
  }

  // --- BackendType Selection Enum ---
  // Added Dispatcher as a distinct backend type
  enum class BackendType { Default, Accelerate, OneMKL, IntelIPP, Dispatcher };

  inline std::string_view get_backend_name(BackendType backend) noexcept
  {
    switch (backend) {
      case BackendType::Default:
        return "Default";
      case BackendType::Accelerate:
        return "Accelerate";
      case BackendType::OneMKL:
        return "oneMKL";
      case BackendType::IntelIPP:
        return "IntelIPP";
      case BackendType::Dispatcher:
        return "Dispatcher";  // Added name for Dispatcher
      default:
        return "Unknown BackendType";
    }
  }

  // --- OperationCategory Enum ---
  // Added FilterDesign
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
    FilterDesign,  // Added FilterDesign category
    GenericFallback
  };

  // --- get_operation_category_name Function Implementation ---
  inline std::string_view get_operation_category_name(
      OperationCategory category) noexcept
  {
    switch (category) {
      case OperationCategory::FFT:
        return "FFT";
      case OperationCategory::RFFT:
        return "RFFT";
      case OperationCategory::Convolution:
        return "Convolution";
      case OperationCategory::Correlation:
        return "Correlation";
      case OperationCategory::FIRFilter:
        return "FIRFilter";
      case OperationCategory::IIRFilter:
        return "IIRFilter";
      case OperationCategory::Resample:
        return "Resample";
      case OperationCategory::CQT:
        return "CQT";
      case OperationCategory::Windowing:
        return "Windowing";
      case OperationCategory::FilterDesign:  // Added case for FilterDesign
        return "FilterDesign";
      case OperationCategory::GenericFallback:
        return "GenericFallback";
      default:
        return "Unknown OperationCategory";
    }
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

  // --- OmniException Definition ---
  class OmniException : public std::runtime_error {
   private:
    OmniStatus error_status_;

   public:
    explicit OmniException(
        const std::string& message, OmniStatus status = OmniStatus::Failure)
        : std::runtime_error(message), error_status_(status)
    {}
    explicit OmniException(
        const char* message, OmniStatus status = OmniStatus::Failure)
        : std::runtime_error(message), error_status_(status)
    {}
    OmniStatus get_status() const noexcept { return error_status_; }
  };

  // --- Utility Type Traits ---
  namespace Utils {
    template <typename T>
    struct GetComplex {
      using type = void;
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
    };
    template <>
    struct GetComplex<C64> {
      using type = C64;
    };
    template <typename T>
    using GetComplexType = typename GetComplex<T>::type;

    template <typename T>
    struct GetReal {
      using type = void;
    };
    template <>
    struct GetReal<F32> {
      using type = F32;
    };
    template <>
    struct GetReal<F64> {
      using type = F64;
    };
    template <>
    struct GetReal<C32> {
      using type = F32;
    };
    template <>
    struct GetReal<C64> {
      using type = F64;
    };
    template <typename T>
    using GetRealType = typename GetReal<T>::type;

    template <typename T>
    struct IsComplex : std::false_type {};
    template <>
    struct IsComplex<C32> : std::true_type {};
    template <>
    struct IsComplex<C64> : std::true_type {};
    template <typename T>
    inline constexpr bool IsComplex_v = IsComplex<T>::value;
  }  // namespace Utils

}  // namespace OmniDSP

#endif  // OMNIDSP_CORE_TYPES_HPP
