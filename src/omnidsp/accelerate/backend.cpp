/**
 * @file backend.cpp (accelerate)
 * @brief Implements the AccelerateBackend class methods.
 * @details This simplified version only overrides the FFT/RFFT plan factories,
 * inheriting all other functionality from DefaultBackend. Includes fallback
 * to DefaultBackend for unsupported FFT lengths.
 */

#include "backend.hpp"  // Corresponding header for Accelerate backend declarations

// Include headers for the public Plan classes (needed for factory return types)
#include <OmniDSP/fft.hpp>

// Include headers for the *Accelerate* Plan implementations
#include "fft.hpp"  // Defines AccelerateFFTPlanImpl, AccelerateRFFTPlanImpl

// Include necessary standard library headers
#include <expected>   // For std::unexpected
#include <iostream>   // For std::cerr (error logging)
#include <memory>     // For std::make_unique
#include <stdexcept>  // For std::runtime_error, std::bad_alloc

// Include DefaultBackend header for fallback
#include "../default/backend.hpp"

namespace OmniDSP::accelerate {

  //--------------------------------------------------------------------------
  // AccelerateBackend Method Definitions
  //--------------------------------------------------------------------------

  AccelerateBackend::AccelerateBackend()
  {
    // std::cout << "Accelerate BackendType Initialized." << std::endl; // Debug
    // message
  }

  AccelerateBackend::~AccelerateBackend()
  {
    // std::cout << "Accelerate BackendType Destroyed." << std::endl; // Debug
    // message
  }

  BackendType AccelerateBackend::get_backend() const
  {
    return BackendType::Accelerate;
  }

  // --- Plan Factories (FFT/RFFT Overrides Only with Fallback) ---

  [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C32>>>
  AccelerateBackend::create_fft_plan_c32(size_t length) const
  {
    // Check if length is supported by Accelerate vDSP DFT
    if (AccelerateFFTPlanImpl<C32>::is_vdsp_dft_supported_length(length)) {
      try {
        auto pimpl_backend
            = std::make_unique<AccelerateFFTPlanImpl<C32>>(length);
        auto plan = FFTPlan<C32>::create_from_impl(std::move(pimpl_backend));
        if (!plan) {
          return std::unexpected(Status::Failure);
        }  // Should not happen if make_unique succeeded
        return plan;
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Accelerate FFTPlan<C32> (length " << length
                  << "): " << e.what() << ". Falling back to Default."
                  << std::endl;
        // Fallback to default implementation if Accelerate fails
        return DefaultBackend::create_fft_plan_c32(length);
      }
      catch (...) {
        std::cerr << "Unknown error creating Accelerate FFTPlan<C32> (length "
                  << length << "). Falling back to Default." << std::endl;
        return DefaultBackend::create_fft_plan_c32(length);
      }
    }
    else {
      // Length not supported by Accelerate, use Default directly
      // std::cout << "Accelerate backend: FFT length " << length << " not
      // supported by vDSP DFT, using Default backend." << std::endl; // Debug
      return DefaultBackend::create_fft_plan_c32(length);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C64>>>
  AccelerateBackend::create_fft_plan_c64(size_t length) const
  {
    // Check if length is supported by Accelerate vDSP DFT
    if (AccelerateFFTPlanImpl<C64>::is_vdsp_dft_supported_length(length)) {
      try {
        auto pimpl_backend
            = std::make_unique<AccelerateFFTPlanImpl<C64>>(length);
        auto plan = FFTPlan<C64>::create_from_impl(std::move(pimpl_backend));
        if (!plan) {
          return std::unexpected(Status::Failure);
        }
        return plan;
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Accelerate FFTPlan<C64> (length " << length
                  << "): " << e.what() << ". Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_fft_plan_c64(length);
      }
      catch (...) {
        std::cerr << "Unknown error creating Accelerate FFTPlan<C64> (length "
                  << length << "). Falling back to Default." << std::endl;
        return DefaultBackend::create_fft_plan_c64(length);
      }
    }
    else {
      // Length not supported by Accelerate, use Default directly
      // std::cout << "Accelerate backend: FFT length " << length << " not
      // supported by vDSP DFT, using Default backend." << std::endl; // Debug
      return DefaultBackend::create_fft_plan_c64(length);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F32>>>
  AccelerateBackend::create_rfft_plan_f32(size_t length) const
  {
    // Check if length is supported by Accelerate vDSP DFT (checks N/2)
    if (AccelerateRFFTPlanImpl<F32>::is_vdsp_dft_supported_length(length)) {
      try {
        auto pimpl_backend
            = std::make_unique<AccelerateRFFTPlanImpl<F32>>(length);
        auto plan = RFFTPlan<F32>::create_from_impl(std::move(pimpl_backend));
        if (!plan) {
          return std::unexpected(Status::Failure);
        }
        return plan;
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Accelerate RFFTPlan<F32> (length "
                  << length << "): " << e.what() << ". Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_rfft_plan_f32(length);
      }
      catch (...) {
        std::cerr << "Unknown error creating Accelerate RFFTPlan<F32> (length "
                  << length << "). Falling back to Default." << std::endl;
        return DefaultBackend::create_rfft_plan_f32(length);
      }
    }
    else {
      // Length not supported by Accelerate, use Default directly
      // std::cout << "Accelerate backend: RFFT length " << length << " not
      // supported by vDSP DFT, using Default backend." << std::endl; // Debug
      return DefaultBackend::create_rfft_plan_f32(length);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F64>>>
  AccelerateBackend::create_rfft_plan_f64(size_t length) const
  {
    // Check if length is supported by Accelerate vDSP DFT (checks N/2)
    if (AccelerateRFFTPlanImpl<F64>::is_vdsp_dft_supported_length(length)) {
      try {
        auto pimpl_backend
            = std::make_unique<AccelerateRFFTPlanImpl<F64>>(length);
        auto plan = RFFTPlan<F64>::create_from_impl(std::move(pimpl_backend));
        if (!plan) {
          return std::unexpected(Status::Failure);
        }
        return plan;
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Accelerate RFFTPlan<F64> (length "
                  << length << "): " << e.what() << ". Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_rfft_plan_f64(length);
      }
      catch (...) {
        std::cerr << "Unknown error creating Accelerate RFFTPlan<F64> (length "
                  << length << "). Falling back to Default." << std::endl;
        return DefaultBackend::create_rfft_plan_f64(length);
      }
    }
    else {
      // Length not supported by Accelerate, use Default directly
      // std::cout << "Accelerate backend: RFFT length " << length << " not
      // supported by vDSP DFT, using Default backend." << std::endl; // Debug
      return DefaultBackend::create_rfft_plan_f64(length);
    }
  }

  // --- Other Methods REMOVED ---
  // (No implementations needed for one-off ops, windowing, filter design,
  // other plan factories, as they are inherited from DefaultBackend)

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations (Only for implemented factory methods)
  //--------------------------------------------------------------------------
  template OmniExpected<std::unique_ptr<FFTPlan<C32>>>
      AccelerateBackend::create_fft_plan_c32(size_t) const;
  template OmniExpected<std::unique_ptr<FFTPlan<C64>>>
      AccelerateBackend::create_fft_plan_c64(size_t) const;
  template OmniExpected<std::unique_ptr<RFFTPlan<F32>>>
      AccelerateBackend::create_rfft_plan_f32(size_t) const;
  template OmniExpected<std::unique_ptr<RFFTPlan<F64>>>
      AccelerateBackend::create_rfft_plan_f64(size_t) const;

}  // namespace OmniDSP::accelerate

// *** ADDED Factory Function Definition (Outside namespace) ***
// This function needs to be defined in the global OmniDSP::Abstract namespace
// as declared in interface/backend.hpp
namespace OmniDSP::Abstract {
  std::unique_ptr<Backend> create_accelerate_backend()
  {
    // Use the concrete class from the accelerate namespace
    return std::make_unique<::OmniDSP::accelerate::AccelerateBackend>();
  }
}  // namespace OmniDSP::Abstract
