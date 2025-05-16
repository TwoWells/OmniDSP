/**
 * @file backend.cpp (Accelerate)
 * @brief Implements the Accelerate::Backend class methods.
 * @details This backend overrides FFT/RFFT plan implementation factories.
 * It includes fallback to Default::Backend's *Impl methods if Accelerate
 * cannot support a requested FFT length or encounters an error.
 */

#include "backend.hpp"  // Corresponding header for Accelerate backend declarations

#include <OmniDSP/omnidsp_config.hpp>  // For OMNIDSP_BACKEND_ACCELERATE_ENABLED etc.
#ifdef OMNIDSP_INTERNAL_USE_ACCELERATE  // Guard for Accelerate-specific code

// Public API headers (primarily for types if not fully covered by backend.hpp)
#include <OmniDSP/core_types.hpp>  // For Status, OmniExpected, OmniException, BackendType etc.
// #include <OmniDSP/fft.hpp> // Public FFTPlan, RFFTPlan not directly created
// here

// Standard library headers
#include <expected>  // For std::unexpected
#include <iostream>  // For std::cerr (error logging), though spdlog is preferred
#include <memory>    // For std::make_unique, std::unique_ptr
#include <stdexcept>  // For std::runtime_error, std::bad_alloc

// spdlog for logging
#include "spdlog/spdlog.h"

// Include headers for the *Accelerate* Plan *Implementations*
#include "fft.hpp"  // Defines Accelerate::FFTPlanImpl, Accelerate::RFFTPlanImpl

// Abstract::Backend for factory function return type
#include "interface/backend.hpp"

namespace OmniDSP::Accelerate {

  //--------------------------------------------------------------------------
  // Backend Method Definitions
  //--------------------------------------------------------------------------

  Backend::Backend()
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    logger->info("Accelerate backend initialized.");
    // Perform any Accelerate-specific initialization if needed.
  }

  Backend::~Backend()
  {
    // auto logger = spdlog::get("OmniDSP");
    // if (logger) logger->trace("Accelerate::Backend destructed.");
  }

  BackendType Backend::get_backend() const { return BackendType::Accelerate; }

  // --- Plan Impl Factories (FFT/RFFT Overrides Only with Fallback) ---

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>
  Backend::create_fft_plan_impl_c32(size_t length) const
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    // Check if length is supported by Accelerate vDSP DFT
    // Assuming
    // Accelerate::FFTPlanImpl<C32>::is_vdsp_dft_supported_length(length) exists
    if (Accelerate::FFTPlanImpl<C32>::is_vdsp_dft_supported_length(length)) {
      try {
        return std::make_unique<Accelerate::FFTPlanImpl<C32>>(length);
      }
      catch (const OmniException& e) {
        logger->warn(
            "Accelerate::create_fft_plan_impl_c32 (length {}) failed: {} "
            "(OmniStatus: {}). Falling back to Default::Backend impl.",
            length,
            e.what(),
            static_cast<int>(e.get_status()));
        return Default::Backend::create_fft_plan_impl_c32(
            length);  // Fallback to Default's Impl
      }
      catch (const std::bad_alloc& e) {
        logger->warn(
            "Accelerate::create_fft_plan_impl_c32 (length {}) allocation "
            "failed: {}. Falling back to Default::Backend impl.",
            length,
            e.what());
        return Default::Backend::create_fft_plan_impl_c32(length);
      }
      catch (const std::exception& e) {
        logger->warn(
            "Accelerate::create_fft_plan_impl_c32 (length {}) general "
            "exception: {}. Falling back to Default::Backend impl.",
            length,
            e.what());
        return Default::Backend::create_fft_plan_impl_c32(length);
      }
      catch (...) {
        logger->warn(
            "Accelerate::create_fft_plan_impl_c32 (length {}) unknown error. "
            "Falling back to Default::Backend impl.",
            length);
        return Default::Backend::create_fft_plan_impl_c32(length);
      }
    }
    else {
      logger->debug(
          "Accelerate backend: FFTPlan<C32> length {} not directly supported "
          "by vDSP DFT, using Default::Backend impl.",
          length);
      return Default::Backend::create_fft_plan_impl_c32(
          length);  // Fallback to Default's Impl
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>
  Backend::create_fft_plan_impl_c64(size_t length) const
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (Accelerate::FFTPlanImpl<C64>::is_vdsp_dft_supported_length(length)) {
      try {
        return std::make_unique<Accelerate::FFTPlanImpl<C64>>(length);
      }
      catch (const OmniException& e) {
        logger->warn(
            "Accelerate::create_fft_plan_impl_c64 (length {}) failed: {} "
            "(OmniStatus: {}). Falling back to Default::Backend impl.",
            length,
            e.what(),
            static_cast<int>(e.get_status()));
        return Default::Backend::create_fft_plan_impl_c64(length);
      }
      catch (const std::bad_alloc& e) {
        logger->warn(
            "Accelerate::create_fft_plan_impl_c64 (length {}) allocation "
            "failed: {}. Falling back to Default::Backend impl.",
            length,
            e.what());
        return Default::Backend::create_fft_plan_impl_c64(length);
      }
      catch (const std::exception& e) {
        logger->warn(
            "Accelerate::create_fft_plan_impl_c64 (length {}) general "
            "exception: {}. Falling back to Default::Backend impl.",
            length,
            e.what());
        return Default::Backend::create_fft_plan_impl_c64(length);
      }
      catch (...) {
        logger->warn(
            "Accelerate::create_fft_plan_impl_c64 (length {}) unknown error. "
            "Falling back to Default::Backend impl.",
            length);
        return Default::Backend::create_fft_plan_impl_c64(length);
      }
    }
    else {
      logger->debug(
          "Accelerate backend: FFTPlan<C64> length {} not directly supported "
          "by vDSP DFT, using Default::Backend impl.",
          length);
      return Default::Backend::create_fft_plan_impl_c64(length);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>
  Backend::create_rfft_plan_impl_f32(size_t length) const
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    // Assuming Accelerate::RFFTPlanImpl<F32>::is_vdsp_dft_supported_length
    // checks N (real length)
    if (Accelerate::RFFTPlanImpl<F32>::is_vdsp_dft_supported_length(length)) {
      try {
        return std::make_unique<Accelerate::RFFTPlanImpl<F32>>(length);
      }
      catch (const OmniException& e) {
        logger->warn(
            "Accelerate::create_rfft_plan_impl_f32 (length {}) failed: {} "
            "(OmniStatus: {}). Falling back to Default::Backend impl.",
            length,
            e.what(),
            static_cast<int>(e.get_status()));
        return Default::Backend::create_rfft_plan_impl_f32(length);
      }
      catch (const std::bad_alloc& e) {
        logger->warn(
            "Accelerate::create_rfft_plan_impl_f32 (length {}) allocation "
            "failed: {}. Falling back to Default::Backend impl.",
            length,
            e.what());
        return Default::Backend::create_rfft_plan_impl_f32(length);
      }
      catch (const std::exception& e) {
        logger->warn(
            "Accelerate::create_rfft_plan_impl_f32 (length {}) general "
            "exception: {}. Falling back to Default::Backend impl.",
            length,
            e.what());
        return Default::Backend::create_rfft_plan_impl_f32(length);
      }
      catch (...) {
        logger->warn(
            "Accelerate::create_rfft_plan_impl_f32 (length {}) unknown error. "
            "Falling back to Default::Backend impl.",
            length);
        return Default::Backend::create_rfft_plan_impl_f32(length);
      }
    }
    else {
      logger->debug(
          "Accelerate backend: RFFTPlan<F32> length {} not directly supported "
          "by vDSP DFT, using Default::Backend impl.",
          length);
      return Default::Backend::create_rfft_plan_impl_f32(length);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>
  Backend::create_rfft_plan_impl_f64(size_t length) const
  {
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }

    if (Accelerate::RFFTPlanImpl<F64>::is_vdsp_dft_supported_length(length)) {
      try {
        return std::make_unique<Accelerate::RFFTPlanImpl<F64>>(length);
      }
      catch (const OmniException& e) {
        logger->warn(
            "Accelerate::create_rfft_plan_impl_f64 (length {}) failed: {} "
            "(OmniStatus: {}). Falling back to Default::Backend impl.",
            length,
            e.what(),
            static_cast<int>(e.get_status()));
        return Default::Backend::create_rfft_plan_impl_f64(length);
      }
      catch (const std::bad_alloc& e) {
        logger->warn(
            "Accelerate::create_rfft_plan_impl_f64 (length {}) allocation "
            "failed: {}. Falling back to Default::Backend impl.",
            length,
            e.what());
        return Default::Backend::create_rfft_plan_impl_f64(length);
      }
      catch (const std::exception& e) {
        logger->warn(
            "Accelerate::create_rfft_plan_impl_f64 (length {}) general "
            "exception: {}. Falling back to Default::Backend impl.",
            length,
            e.what());
        return Default::Backend::create_rfft_plan_impl_f64(length);
      }
      catch (...) {
        logger->warn(
            "Accelerate::create_rfft_plan_impl_f64 (length {}) unknown error. "
            "Falling back to Default::Backend impl.",
            length);
        return Default::Backend::create_rfft_plan_impl_f64(length);
      }
    }
    else {
      logger->debug(
          "Accelerate backend: RFFTPlan<F64> length {} not directly supported "
          "by vDSP DFT, using Default::Backend impl.",
          length);
      return Default::Backend::create_rfft_plan_impl_f64(length);
    }
  }

  // All other methods are inherited from Default::Backend.

}  // namespace OmniDSP::Accelerate

// --- Factory function for Accelerate Backend ---
namespace OmniDSP::Abstract {
  std::unique_ptr<Backend> create_accelerate_backend()
  {
#ifdef OMNIDSP_INTERNAL_USE_ACCELERATE
    return std::make_unique<OmniDSP::Accelerate::Backend>();
#else
    // This case should ideally be handled by the main OmniDSP::create factory
    // by not attempting to call this if OMNIDSP_BACKEND_ACCELERATE_ENABLED is
    // false. Returning nullptr indicates the backend is not available.
    auto logger = spdlog::get("OmniDSP");
    if (!logger) {
      logger = spdlog::default_logger();
    }
    logger->warn(
        "Attempted to create Accelerate backend, but it's not "
        "enabled/supported in this build.");
    return nullptr;
#endif
  }
}  // namespace OmniDSP::Abstract

#endif  // OMNIDSP_INTERNAL_USE_ACCELERATE (Matches the guard at the top of the
        // original .cpp)
