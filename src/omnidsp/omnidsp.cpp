/**
 * @file omnidsp.cpp
 * @brief Implements the non-template OmniDSP class methods (factory,
 * constructor, etc.). Template method definitions are now in omnidsp.hpp.
 */

#include "OmniDSP/omnidsp.hpp"  // Corresponding header

// Include the backend interface definition
#include "interface/backend.hpp"  // Defines AbstractBackend

// Include concrete backend implementation headers for factory function
#include "default/backend.hpp"  // Defines DefaultBackend
#ifdef OMNIDSP_USE_ACCELERATE
#include "accelerate/backend.hpp"  // Defines AccelerateBackend
#endif
#ifdef OMNIDSP_USE_ONEMKL
#include "onemkl/backend.hpp"  // Defines OneMKLBackend
#endif

#include <exception>
#include <expected>   // For std::unexpected
#include <iostream>   // For debug/error messages
#include <memory>     // For std::unique_ptr, std::make_unique
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move
#include <vector>

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
#ifdef OMNIDSP_USE_ACCELERATE
          pimpl = std::make_unique<backend::AccelerateBackend>();
#else
          std::cerr << "Warning: Accelerate backend requested but not enabled "
                       "during build."
                    << std::endl;
          return std::unexpected(Status::UnsupportedFeature);
#endif
          break;

        case Backend::OneMKL:
#ifdef OMNIDSP_USE_ONEMKL
          pimpl = std::make_unique<backend::OneMKLBackend>();
#else
          std::cerr << "Warning: oneMKL backend requested but not enabled "
                       "during build."
                    << std::endl;
          return std::unexpected(Status::UnsupportedFeature);
#endif
          break;

        case Backend::Default:
        default:
          pimpl = std::make_unique<backend::DefaultBackend>();
          break;
      }

      if (!pimpl) {
        return std::unexpected(Status::BackendError);
      }
      // Use private constructor
      OmniDSP dsp_instance(std::move(pimpl));
      return dsp_instance;
    }
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
  // (Their definitions are now inline in omnidsp.hpp)

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations (REMOVED)
  //--------------------------------------------------------------------------
  // No longer needed because template definitions are in the header.

}  // namespace OmniDSP
