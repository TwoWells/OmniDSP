/**
 * @file omnidsp.cpp
 * @brief Implements the non-template OmniDSP class methods (factory,
 * constructor, etc.). Template method definitions are in omnidsp.hpp.
 */

#include "OmniDSP/omnidsp.hpp"  // Corresponding header

// Include the backend interface definition (declares AbstractBackend and
// backend factory functions)
#include "interface/backend.hpp"

// *** ADDED: Include the generated configuration header ***
#include <exception>
#include <expected>   // For std::unexpected
#include <iostream>   // For debug/error messages
#include <memory>     // For std::unique_ptr
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move
#include <vector>

#include "OmniDSP/omnidsp_config.h"  // Contains OMNIDSP_HAS_BACKEND_* defines

namespace OmniDSP {

  //--------------------------------------------------------------------------
  // OmniDSP Method Definitions (Non-Template)
  //--------------------------------------------------------------------------

  // --- Factory ---
  [[nodiscard]] /* static */ OmniExpected<OmniDSP> OmniDSP::create(
      Backend backend)
  {
    std::unique_ptr<backend::AbstractBackend> pimpl = nullptr;
    try {
      switch (backend) {
        case Backend::Accelerate:
          // *** Use standard 'if' with generated config constant ***
          if constexpr (OMNIDSP_HAS_BACKEND_ACCELERATE) {  // Use constexpr if
                                                           // possible (C++17)
            pimpl = backend::create_accelerate_backend();  // Call factory
                                                           // function
          }
          else {
            std::cerr
                << "Warning: Accelerate backend requested but not compiled in."
                << std::endl;
            return std::unexpected(Status::UnsupportedFeature);
          }
          break;

        case Backend::OneMKL:
          // *** Use standard 'if' with generated config constant ***
          if constexpr (OMNIDSP_HAS_BACKEND_ONEMKL) {  // Use constexpr if
                                                       // possible (C++17)
            pimpl = backend::create_onemkl_backend();  // Call factory function
          }
          else {
            std::cerr
                << "Warning: oneMKL backend requested but not compiled in."
                << std::endl;
            return std::unexpected(Status::UnsupportedFeature);
          }
          break;

        case Backend::Default:
        default:  // Fallback to Default
                  // The default backend factory should always be available
          if constexpr (OMNIDSP_HAS_BACKEND_DEFAULT) {  // Should always be true
            pimpl = backend::create_default_backend();  // Call factory function
          }
          else {
            // This case indicates a build system error
            std::cerr << "Error: Default backend was not compiled in "
                         "(OMNIDSP_HAS_BACKEND_DEFAULT is false)."
                      << std::endl;
            return std::unexpected(Status::BackendError);
          }
          break;
      }

      if (!pimpl) {
        // This could happen if a factory function returns nullptr on error
        std::cerr << "Error: Backend factory function returned null."
                  << std::endl;
        return std::unexpected(Status::BackendError);
      }
      // Use private constructor
      OmniDSP dsp_instance(std::move(pimpl));
      return dsp_instance;
    }
    // Catch exceptions that might occur *during* factory function execution
    catch (const std::bad_alloc &e) {
      std::cerr << "Error: Memory allocation failed during backend creation: "
                << e.what() << std::endl;
      return std::unexpected(Status::AllocationError);
    }
    catch (const std::exception &e) {
      std::cerr << "Error: Exception during backend initialization: "
                << e.what() << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      std::cerr << "Error: Unknown exception during backend creation."
                << std::endl;
      return std::unexpected(Status::Failure);
    }
  }

  // --- Constructor / Destructor / Move Operations ---
  OmniDSP::OmniDSP(std::unique_ptr<backend::AbstractBackend> impl)
      : pimpl_(std::move(impl))
  {}
  OmniDSP::~OmniDSP() = default;
  OmniDSP::OmniDSP(OmniDSP &&other) noexcept = default;
  OmniDSP &OmniDSP::operator=(OmniDSP &&other) noexcept = default;

  // --- Public Member Functions (Non-Template) ---
  Backend OmniDSP::get_backend() const
  {
    if (!pimpl_)
      throw std::runtime_error("Invalid OmniDSP instance in get_backend.");
    return pimpl_->get_backend();
  }

  // --- Template Method Definitions REMOVED ---
  // (Definitions are now inline in omnidsp.hpp)

  // --- Explicit Template Instantiations REMOVED ---

}  // namespace OmniDSP
