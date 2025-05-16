/**
 * @file backend.cpp (OneMKL)
 * @brief Implements the OneMKL::Backend class methods using oneMKL functions.
 */

#include "backend.hpp"  // Corresponding header for oneMKL backend declarations

// Include MKL header (needed for status codes, potentially global settings)
#include <mkl.h>  // For DFTI and potentially MKL version info

// Public API headers (primarily for types if not fully covered by backend.hpp)
#include <OmniDSP/core_types.hpp>  // For Status, OmniExpected, OmniException etc.
// #include <OmniDSP/fft.hpp> // Public FFTPlan, RFFTPlan not directly created
// here anymore

// Include headers for oneMKL specific Plan *Implementations*
#include "fft.hpp"  // Defines OneMKL::FFTPlanImpl, OneMKL::RFFTPlanImpl
#include "utils.hpp"  // For oneMKL utility functions like mkl_status_to_omnidsp_status

// Standard library headers
#include <complex>
#include <expected>   // For std::unexpected
#include <iostream>   // For debug/error messages (though spdlog is preferred)
#include <memory>     // For std::make_unique, std::unique_ptr
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>     // For exception messages

// spdlog for logging
#include "spdlog/spdlog.h"

// Abstract::Backend for factory function return type
#include "interface/backend.hpp"

namespace OmniDSP::OneMKL {

  //--------------------------------------------------------------------------
  // OneMKLBackend Method Definitions
  //--------------------------------------------------------------------------

  Backend::Backend()
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    // Example: Check MKL version (optional)
    // MKLVersion Version;
    // mkl_get_version(&Version);
    // logger->info("oneMKL backend initialized: MKL Version {}.{}.{}",
    // Version.MajorVersion, Version.MinorVersion, Version.UpdateVersion);
    logger->info("oneMKL backend initialized.");
  }

  Backend::~Backend()
  {
    // auto logger = spdlog::get("OmniDSP");
    // if (logger) logger->trace("oneMKL::Backend destructed.");
  }

  BackendType Backend::get_backend() const { return BackendType::OneMKL; }

  // --- Plan Impl Factories (Overrides for FFT/RFFT) ---
  // These methods create and return the oneMKL-specific *implementation* of the
  // plans.

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>
  Backend::create_fft_plan_impl_c32(size_t length) const
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    try {
      // Attempt to create the oneMKL-specific FFTPlanImpl
      // The constructor of OneMKL::FFTPlanImpl<C32> should handle DFTI
      // descriptor creation and throw OmniException on MKL errors.
      return std::make_unique<OneMKL::FFTPlanImpl<C32>>(length);
    }
    catch (const OmniException& e) {
      logger->error(
          "oneMKL::create_fft_plan_impl_c32 (length {}) failed: {} (Status: "
          "{})",
          length,
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::bad_alloc& e) {
      logger->error(
          "oneMKL::create_fft_plan_impl_c32 (length {}) allocation failed: {}",
          length,
          e.what());
      return OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>(
          std::unexpect, OmniStatus::AllocationError);
    }
    catch (const std::exception& e) {
      logger->error(
          "oneMKL::create_fft_plan_impl_c32 (length {}) general exception: {}",
          length,
          e.what());
      return OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>(
          std::unexpect, OmniStatus::BackendError);
    }
    catch (...) {
      logger->error(
          "oneMKL::create_fft_plan_impl_c32 (length {}) unknown error.",
          length);
      return OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>
  Backend::create_fft_plan_impl_c64(size_t length) const
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    try {
      return std::make_unique<OneMKL::FFTPlanImpl<C64>>(length);
    }
    catch (const OmniException& e) {
      logger->error(
          "oneMKL::create_fft_plan_impl_c64 (length {}) failed: {} (Status: "
          "{})",
          length,
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::bad_alloc& e) {
      logger->error(
          "oneMKL::create_fft_plan_impl_c64 (length {}) allocation failed: {}",
          length,
          e.what());
      return OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>(
          std::unexpect, OmniStatus::AllocationError);
    }
    catch (const std::exception& e) {
      logger->error(
          "oneMKL::create_fft_plan_impl_c64 (length {}) general exception: {}",
          length,
          e.what());
      return OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>(
          std::unexpect, OmniStatus::BackendError);
    }
    catch (...) {
      logger->error(
          "oneMKL::create_fft_plan_impl_c64 (length {}) unknown error.",
          length);
      return OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>
  Backend::create_rfft_plan_impl_f32(size_t length) const
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    try {
      return std::make_unique<OneMKL::RFFTPlanImpl<F32>>(length);
    }
    catch (const OmniException& e) {
      logger->error(
          "oneMKL::create_rfft_plan_impl_f32 (length {}) failed: {} (Status: "
          "{})",
          length,
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::bad_alloc& e) {
      logger->error(
          "oneMKL::create_rfft_plan_impl_f32 (length {}) allocation failed: {}",
          length,
          e.what());
      return OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>(
          std::unexpect, OmniStatus::AllocationError);
    }
    catch (const std::exception& e) {
      logger->error(
          "oneMKL::create_rfft_plan_impl_f32 (length {}) general exception: {}",
          length,
          e.what());
      return OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>(
          std::unexpect, OmniStatus::BackendError);
    }
    catch (...) {
      logger->error(
          "oneMKL::create_rfft_plan_impl_f32 (length {}) unknown error.",
          length);
      return OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>
  Backend::create_rfft_plan_impl_f64(size_t length) const
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    try {
      return std::make_unique<OneMKL::RFFTPlanImpl<F64>>(length);
    }
    catch (const OmniException& e) {
      logger->error(
          "oneMKL::create_rfft_plan_impl_f64 (length {}) failed: {} (Status: "
          "{})",
          length,
          e.what(),
          static_cast<int>(e.get_status()));
      return OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>(
          std::unexpect, e.get_status());
    }
    catch (const std::bad_alloc& e) {
      logger->error(
          "oneMKL::create_rfft_plan_impl_f64 (length {}) allocation failed: {}",
          length,
          e.what());
      return OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>(
          std::unexpect, OmniStatus::AllocationError);
    }
    catch (const std::exception& e) {
      logger->error(
          "oneMKL::create_rfft_plan_impl_f64 (length {}) general exception: {}",
          length,
          e.what());
      return OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>(
          std::unexpect, OmniStatus::BackendError);
    }
    catch (...) {
      logger->error(
          "oneMKL::create_rfft_plan_impl_f64 (length {}) unknown error.",
          length);
      return OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>(
          std::unexpect, OmniStatus::Failure);
    }
  }

  // All other methods (one-off DSP operations, window generation, other plan
  // impl factories, filter design) are inherited from Default::Backend.

}  // namespace OmniDSP::OneMKL

// --- Factory function for oneMKL Backend ---
// This function is declared in "omnidsp/interface/backend.hpp"
// and defined here, in the oneMKL backend's .cpp file.
namespace OmniDSP::Abstract {
  std::unique_ptr<Backend> create_onemkl_backend()
  {
    // Use the concrete class from the OneMKL namespace
    return std::make_unique<OmniDSP::OneMKL::Backend>();
  }
}  // namespace OmniDSP::Abstract
