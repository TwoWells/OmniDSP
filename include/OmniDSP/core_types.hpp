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

  // --- Core Status Enum ---
  enum class Status {
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

  // --- BackendType Selection Enum ---
  enum class BackendType { Default, Accelerate, OneMKL, IntelIPP };

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
      default:
        return "Unknown BackendType";
    }
  }

  // --- OperationType Enum ---
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
    GenericFallback
  };

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
  using OmniExpected = std::expected<T, Status>;

  // --- OmniException Definition ---
  class OmniException : public std::runtime_error {
   private:
    Status error_status_;

   public:
    explicit OmniException(
        const std::string& message, Status status = Status::Failure)
        : std::runtime_error(message), error_status_(status)
    {}
    explicit OmniException(const char* message, Status status = Status::Failure)
        : std::runtime_error(message), error_status_(status)
    {}
    Status get_status() const noexcept { return error_status_; }
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
