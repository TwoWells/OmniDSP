/**
 * @file backend.cpp (onemkl)
 * @brief Implements the OneMKLBackend class methods using oneMKL functions.
 */

#include "backend.hpp"  // Corresponding header for oneMKL backend declarations

// Include headers for the public Plan classes (needed for factory return types)
#include <OmniDSP/convolution.hpp>
#include <OmniDSP/cqt.hpp>
#include <OmniDSP/fft.hpp>
#include <OmniDSP/filter.hpp>
#include <OmniDSP/resample.hpp>

// Include headers for the *oneMKL* Plan implementations
// #include "convolution.hpp"  // Defines OneMKLConvolutionPlanImpl,
// OneMKLCorrelationPlanImpl
#include "fft.hpp"  // Defines OneMKLFFTPlanImpl, OneMKLRFFTPlanImpl

// Include MKL header (needed for status codes, potentially global settings)
#include <mkl.h>

#include <complex>
#include <expected>   // For std::unexpected
#include <iostream>   // For debug/error messages
#include <memory>     // For std::make_unique
#include <span>       // For std::span
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>     // For exception messages
#include <vector>

// Include DefaultBackend header for fallback and base class definitions
#include "../default/backend.hpp"

// Include oneMKL utilities
#include "utils.hpp"

namespace OmniDSP::onemkl {

  //--------------------------------------------------------------------------
  // OneMKLBackend Method Definitions
  //--------------------------------------------------------------------------

  OneMKLBackend::OneMKLBackend()
  {
    // std::cout << "oneMKL Backend Initialized." << std::endl; // Debug message
  }

  OneMKLBackend::~OneMKLBackend()
  {
    // std::cout << "oneMKL Backend Destroyed." << std::endl; // Debug message
  }

  BackendType OneMKLBackend::get_backend() const { return BackendType::OneMKL; }

  // --- DSP Operations (One-off) ---
  // Inherited from DefaultBackend

  // --- Window Generation ---
  // Inherited from DefaultBackend

  // --- Plan Factories ---
  // Implementations now include fallback to DefaultBackend on error

  [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C32>>>
  OneMKLBackend::create_fft_plan_c32(size_t length) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLFFTPlanImpl<C32>>(length);
      // Constructor throws on MKL error, so if we reach here, it's likely okay
      auto plan = FFTPlan<C32>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {  // Should ideally not happen if make_unique succeeded
        std::cerr
            << "Warning: oneMKL FFTPlan<C32> (length " << length
            << ") Impl created but wrapper failed. Falling back to Default."
            << std::endl;
        return DefaultBackend::create_fft_plan_c32(length);  // Fallback
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL FFTPlan<C32> (length " << length
                << "): " << e.what() << ". Falling back to Default."
                << std::endl;
      return DefaultBackend::create_fft_plan_c32(length);  // Fallback
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL FFTPlan<C32> (length "
                << length << "). Falling back to Default." << std::endl;
      return DefaultBackend::create_fft_plan_c32(length);  // Fallback
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C64>>>
  OneMKLBackend::create_fft_plan_c64(size_t length) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLFFTPlanImpl<C64>>(length);
      auto plan = FFTPlan<C64>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr
            << "Warning: oneMKL FFTPlan<C64> (length " << length
            << ") Impl created but wrapper failed. Falling back to Default."
            << std::endl;
        return DefaultBackend::create_fft_plan_c64(length);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL FFTPlan<C64> (length " << length
                << "): " << e.what() << ". Falling back to Default."
                << std::endl;
      return DefaultBackend::create_fft_plan_c64(length);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL FFTPlan<C64> (length "
                << length << "). Falling back to Default." << std::endl;
      return DefaultBackend::create_fft_plan_c64(length);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F32>>>
  OneMKLBackend::create_rfft_plan_f32(size_t length) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLRFFTPlanImpl<F32>>(length);
      auto plan = RFFTPlan<F32>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr
            << "Warning: oneMKL RFFTPlan<F32> (length " << length
            << ") Impl created but wrapper failed. Falling back to Default."
            << std::endl;
        return DefaultBackend::create_rfft_plan_f32(length);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL RFFTPlan<F32> (length " << length
                << "): " << e.what() << ". Falling back to Default."
                << std::endl;
      return DefaultBackend::create_rfft_plan_f32(length);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL RFFTPlan<F32> (length "
                << length << "). Falling back to Default." << std::endl;
      return DefaultBackend::create_rfft_plan_f32(length);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F64>>>
  OneMKLBackend::create_rfft_plan_f64(size_t length) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLRFFTPlanImpl<F64>>(length);
      auto plan = RFFTPlan<F64>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr
            << "Warning: oneMKL RFFTPlan<F64> (length " << length
            << ") Impl created but wrapper failed. Falling back to Default."
            << std::endl;
        return DefaultBackend::create_rfft_plan_f64(length);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL RFFTPlan<F64> (length " << length
                << "): " << e.what() << ". Falling back to Default."
                << std::endl;
      return DefaultBackend::create_rfft_plan_f64(length);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL RFFTPlan<F64> (length "
                << length << "). Falling back to Default." << std::endl;
      return DefaultBackend::create_rfft_plan_f64(length);
    }
  }

  // CQT Plan Factory is NOT overridden - inherits DefaultBackend's version

  // Resample Plan Factory is NOT overridden - inherits DefaultBackend's version

  // Convolution/Correlation Plan Factory is NOT overridden - inherits
  // DefaultBackend's version

  // FIR/IIR Filter Plan Factory is NOT overridden - inherits DefaultBackend's
  // version

  // --- Filter Design ---
  // Inherited from DefaultBackend

}  // namespace OmniDSP::onemkl

// This function needs to be defined in the global OmniDSP::abstract namespace
// as declared in interface/backend.hpp
namespace OmniDSP::Abstract {
  std::unique_ptr<AbstractBackend> create_onemkl_backend()
  {
    // Use the concrete class from the onemkl namespace
    return std::make_unique<::OmniDSP::onemkl::OneMKLBackend>();
  }
}  // namespace OmniDSP::Abstract
