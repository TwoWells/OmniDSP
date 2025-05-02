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
#include "convolution.hpp"  // Defines OneMKLConvolutionPlanImpl, OneMKLCorrelationPlanImpl
#include "fft.hpp"     // Defines OneMKLFFTPlanImpl, OneMKLRFFTPlanImpl
#include "filter.hpp"  // Defines OneMKLFIRFilterPlanImpl, OneMKLIIRFilterPlanImpl
#include "resample.hpp"  // Defines OneMKLResamplePlanImpl
#include "window.hpp"  // Include declarations for oneMKL window helper functions

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

namespace OmniDSP::backend {

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

  Backend OneMKLBackend::get_backend() const { return Backend::OneMKL; }

  // --- DSP Operations (One-off) ---
  // Inherited from DefaultBackend

  // --- Window Generation ---
  // Implementations call the specific helper functions defined in
  // onemkl/window.cpp
  [[nodiscard]] Status OneMKLBackend::bartlett_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_bartlett_window_onemkl(output.first(length));
  }
  [[nodiscard]] Status OneMKLBackend::bartlett_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_bartlett_window_onemkl(output.first(length));
  }
  [[nodiscard]] Status OneMKLBackend::blackman_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_blackman_window_onemkl(output.first(length));
  }
  [[nodiscard]] Status OneMKLBackend::blackman_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_blackman_window_onemkl(output.first(length));
  }
  [[nodiscard]] Status OneMKLBackend::hamming_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_hamming_window_onemkl(output.first(length));
  }
  [[nodiscard]] Status OneMKLBackend::hamming_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_hamming_window_onemkl(output.first(length));
  }
  [[nodiscard]] Status OneMKLBackend::hann_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_hann_window_onemkl(output.first(length));
  }
  [[nodiscard]] Status OneMKLBackend::hann_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_hann_window_onemkl(output.first(length));
  }
  [[nodiscard]] Status OneMKLBackend::kaiser_window_f32(
      size_t length, double beta, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_kaiser_window_onemkl(beta, output.first(length));
  }
  [[nodiscard]] Status OneMKLBackend::kaiser_window_f64(
      size_t length, double beta, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_kaiser_window_onemkl(beta, output.first(length));
  }
  [[nodiscard]] Status OneMKLBackend::rectangular_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_rectangular_window_onemkl(output.first(length));
  }
  [[nodiscard]] Status OneMKLBackend::rectangular_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_rectangular_window_onemkl(output.first(length));
  }
  [[nodiscard]] Status OneMKLBackend::triangular_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_triangular_window_onemkl(output.first(length));
  }
  [[nodiscard]] Status OneMKLBackend::triangular_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_triangular_window_onemkl(output.first(length));
  }
  // Note: Gaussian and Flattop inherit from DefaultBackend

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

  [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<F32>>>
  OneMKLBackend::create_resample_plan_f32(const ResampleSpec& spec) const
  {
    try {
      // *** CORRECTED: Pass 'this' as the first argument ***
      auto pimpl_backend
          = std::make_unique<OneMKLResamplePlanImpl<F32>>(this, spec);
      // *** END CORRECTION ***
      auto plan = ResamplePlan<F32>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr << "Warning: oneMKL ResamplePlan<F32> Impl created but "
                     "wrapper failed. Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_resample_plan_f32(spec);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL ResamplePlan<F32>: " << e.what()
                << ". Falling back to Default." << std::endl;
      return DefaultBackend::create_resample_plan_f32(spec);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL ResamplePlan<F32>. Falling "
                   "back to Default."
                << std::endl;
      return DefaultBackend::create_resample_plan_f32(spec);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<F64>>>
  OneMKLBackend::create_resample_plan_f64(const ResampleSpec& spec) const
  {
    try {
      // *** CORRECTED: Pass 'this' as the first argument ***
      auto pimpl_backend
          = std::make_unique<OneMKLResamplePlanImpl<F64>>(this, spec);
      // *** END CORRECTION ***
      auto plan = ResamplePlan<F64>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr << "Warning: oneMKL ResamplePlan<F64> Impl created but "
                     "wrapper failed. Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_resample_plan_f64(spec);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL ResamplePlan<F64>: " << e.what()
                << ". Falling back to Default." << std::endl;
      return DefaultBackend::create_resample_plan_f64(spec);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL ResamplePlan<F64>. Falling "
                   "back to Default."
                << std::endl;
      return DefaultBackend::create_resample_plan_f64(spec);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<F32>>>
  OneMKLBackend::create_convolution_plan_f32(
      const F32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLConvolutionPlanImpl<F32>>(
          this, kernel, type, method);
      auto plan
          = ConvolutionPlan<F32>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr << "Warning: oneMKL ConvolutionPlan<F32> Impl created but "
                     "wrapper failed. Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_convolution_plan_f32(
            kernel, type, method);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL ConvolutionPlan<F32>: " << e.what()
                << ". Falling back to Default." << std::endl;
      return DefaultBackend::create_convolution_plan_f32(kernel, type, method);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL ConvolutionPlan<F32>. "
                   "Falling back to Default."
                << std::endl;
      return DefaultBackend::create_convolution_plan_f32(kernel, type, method);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<F64>>>
  OneMKLBackend::create_convolution_plan_f64(
      const F64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLConvolutionPlanImpl<F64>>(
          this, kernel, type, method);
      auto plan
          = ConvolutionPlan<F64>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr << "Warning: oneMKL ConvolutionPlan<F64> Impl created but "
                     "wrapper failed. Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_convolution_plan_f64(
            kernel, type, method);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL ConvolutionPlan<F64>: " << e.what()
                << ". Falling back to Default." << std::endl;
      return DefaultBackend::create_convolution_plan_f64(kernel, type, method);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL ConvolutionPlan<F64>. "
                   "Falling back to Default."
                << std::endl;
      return DefaultBackend::create_convolution_plan_f64(kernel, type, method);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<C32>>>
  OneMKLBackend::create_convolution_plan_c32(
      const C32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLConvolutionPlanImpl<C32>>(
          this, kernel, type, method);
      auto plan
          = ConvolutionPlan<C32>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr << "Warning: oneMKL ConvolutionPlan<C32> Impl created but "
                     "wrapper failed. Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_convolution_plan_c32(
            kernel, type, method);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL ConvolutionPlan<C32>: " << e.what()
                << ". Falling back to Default." << std::endl;
      return DefaultBackend::create_convolution_plan_c32(kernel, type, method);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL ConvolutionPlan<C32>. "
                   "Falling back to Default."
                << std::endl;
      return DefaultBackend::create_convolution_plan_c32(kernel, type, method);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<C64>>>
  OneMKLBackend::create_convolution_plan_c64(
      const C64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLConvolutionPlanImpl<C64>>(
          this, kernel, type, method);
      auto plan
          = ConvolutionPlan<C64>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr << "Warning: oneMKL ConvolutionPlan<C64> Impl created but "
                     "wrapper failed. Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_convolution_plan_c64(
            kernel, type, method);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL ConvolutionPlan<C64>: " << e.what()
                << ". Falling back to Default." << std::endl;
      return DefaultBackend::create_convolution_plan_c64(kernel, type, method);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL ConvolutionPlan<C64>. "
                   "Falling back to Default."
                << std::endl;
      return DefaultBackend::create_convolution_plan_c64(kernel, type, method);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<F32>>>
  OneMKLBackend::create_correlation_plan_f32(
      const F32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLCorrelationPlanImpl<F32>>(
          this, kernel, type, method);
      auto plan
          = CorrelationPlan<F32>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr << "Warning: oneMKL CorrelationPlan<F32> Impl created but "
                     "wrapper failed. Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_correlation_plan_f32(
            kernel, type, method);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL CorrelationPlan<F32>: " << e.what()
                << ". Falling back to Default." << std::endl;
      return DefaultBackend::create_correlation_plan_f32(kernel, type, method);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL CorrelationPlan<F32>. "
                   "Falling back to Default."
                << std::endl;
      return DefaultBackend::create_correlation_plan_f32(kernel, type, method);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<F64>>>
  OneMKLBackend::create_correlation_plan_f64(
      const F64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLCorrelationPlanImpl<F64>>(
          this, kernel, type, method);
      auto plan
          = CorrelationPlan<F64>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr << "Warning: oneMKL CorrelationPlan<F64> Impl created but "
                     "wrapper failed. Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_correlation_plan_f64(
            kernel, type, method);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL CorrelationPlan<F64>: " << e.what()
                << ". Falling back to Default." << std::endl;
      return DefaultBackend::create_correlation_plan_f64(kernel, type, method);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL CorrelationPlan<F64>. "
                   "Falling back to Default."
                << std::endl;
      return DefaultBackend::create_correlation_plan_f64(kernel, type, method);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<C32>>>
  OneMKLBackend::create_correlation_plan_c32(
      const C32Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLCorrelationPlanImpl<C32>>(
          this, kernel, type, method);
      auto plan
          = CorrelationPlan<C32>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr << "Warning: oneMKL CorrelationPlan<C32> Impl created but "
                     "wrapper failed. Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_correlation_plan_c32(
            kernel, type, method);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL CorrelationPlan<C32>: " << e.what()
                << ". Falling back to Default." << std::endl;
      return DefaultBackend::create_correlation_plan_c32(kernel, type, method);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL CorrelationPlan<C32>. "
                   "Falling back to Default."
                << std::endl;
      return DefaultBackend::create_correlation_plan_c32(kernel, type, method);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<C64>>>
  OneMKLBackend::create_correlation_plan_c64(
      const C64Vec& kernel,
      ConvolutionType type,
      ConvolutionMethod method) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLCorrelationPlanImpl<C64>>(
          this, kernel, type, method);
      auto plan
          = CorrelationPlan<C64>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr << "Warning: oneMKL CorrelationPlan<C64> Impl created but "
                     "wrapper failed. Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_correlation_plan_c64(
            kernel, type, method);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL CorrelationPlan<C64>: " << e.what()
                << ". Falling back to Default." << std::endl;
      return DefaultBackend::create_correlation_plan_c64(kernel, type, method);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL CorrelationPlan<C64>. "
                   "Falling back to Default."
                << std::endl;
      return DefaultBackend::create_correlation_plan_c64(kernel, type, method);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<F32>>>
  OneMKLBackend::create_fir_filter_plan_f32(const F32Vec& coefficients) const
  {
    try {
      auto pimpl_backend
          = std::make_unique<OneMKLFIRFilterPlanImpl<F32>>(coefficients);
      auto plan
          = FIRFilterPlan<F32>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr << "Warning: oneMKL FIRFilterPlan<F32> Impl created but "
                     "wrapper failed. Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_fir_filter_plan_f32(coefficients);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL FIRFilterPlan<F32>: " << e.what()
                << ". Falling back to Default." << std::endl;
      return DefaultBackend::create_fir_filter_plan_f32(coefficients);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL FIRFilterPlan<F32>. Falling "
                   "back to Default."
                << std::endl;
      return DefaultBackend::create_fir_filter_plan_f32(coefficients);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<F64>>>
  OneMKLBackend::create_fir_filter_plan_f64(const F64Vec& coefficients) const
  {
    try {
      auto pimpl_backend
          = std::make_unique<OneMKLFIRFilterPlanImpl<F64>>(coefficients);
      auto plan
          = FIRFilterPlan<F64>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr << "Warning: oneMKL FIRFilterPlan<F64> Impl created but "
                     "wrapper failed. Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_fir_filter_plan_f64(coefficients);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL FIRFilterPlan<F64>: " << e.what()
                << ". Falling back to Default." << std::endl;
      return DefaultBackend::create_fir_filter_plan_f64(coefficients);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL FIRFilterPlan<F64>. Falling "
                   "back to Default."
                << std::endl;
      return DefaultBackend::create_fir_filter_plan_f64(coefficients);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<C32>>>
  OneMKLBackend::create_fir_filter_plan_c32(const C32Vec& coefficients) const
  {
    try {
      auto pimpl_backend
          = std::make_unique<OneMKLFIRFilterPlanImpl<C32>>(coefficients);
      auto plan
          = FIRFilterPlan<C32>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr << "Warning: oneMKL FIRFilterPlan<C32> Impl created but "
                     "wrapper failed. Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_fir_filter_plan_c32(coefficients);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL FIRFilterPlan<C32>: " << e.what()
                << ". Falling back to Default." << std::endl;
      return DefaultBackend::create_fir_filter_plan_c32(coefficients);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL FIRFilterPlan<C32>. Falling "
                   "back to Default."
                << std::endl;
      return DefaultBackend::create_fir_filter_plan_c32(coefficients);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<C64>>>
  OneMKLBackend::create_fir_filter_plan_c64(const C64Vec& coefficients) const
  {
    try {
      auto pimpl_backend
          = std::make_unique<OneMKLFIRFilterPlanImpl<C64>>(coefficients);
      auto plan
          = FIRFilterPlan<C64>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr << "Warning: oneMKL FIRFilterPlan<C64> Impl created but "
                     "wrapper failed. Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_fir_filter_plan_c64(coefficients);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL FIRFilterPlan<C64>: " << e.what()
                << ". Falling back to Default." << std::endl;
      return DefaultBackend::create_fir_filter_plan_c64(coefficients);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL FIRFilterPlan<C64>. Falling "
                   "back to Default."
                << std::endl;
      return DefaultBackend::create_fir_filter_plan_c64(coefficients);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F32>>>
  OneMKLBackend::create_iir_filter_plan_f32(
      const std::vector<IIRFilterCoef>& sos_coefficients) const
  {
    try {
      auto pimpl_backend
          = std::make_unique<OneMKLIIRFilterPlanImpl<F32>>(sos_coefficients);
      auto plan
          = IIRFilterPlan<F32>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr << "Warning: oneMKL IIRFilterPlan<F32> Impl created but "
                     "wrapper failed. Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_iir_filter_plan_f32(sos_coefficients);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL IIRFilterPlan<F32>: " << e.what()
                << ". Falling back to Default." << std::endl;
      return DefaultBackend::create_iir_filter_plan_f32(sos_coefficients);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL IIRFilterPlan<F32>. Falling "
                   "back to Default."
                << std::endl;
      return DefaultBackend::create_iir_filter_plan_f32(sos_coefficients);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F64>>>
  OneMKLBackend::create_iir_filter_plan_f64(
      const std::vector<IIRFilterCoef>& sos_coefficients) const
  {
    try {
      auto pimpl_backend
          = std::make_unique<OneMKLIIRFilterPlanImpl<F64>>(sos_coefficients);
      auto plan
          = IIRFilterPlan<F64>::create_from_impl(std::move(pimpl_backend));
      if (!plan) {
        std::cerr << "Warning: oneMKL IIRFilterPlan<F64> Impl created but "
                     "wrapper failed. Falling back to Default."
                  << std::endl;
        return DefaultBackend::create_iir_filter_plan_f64(sos_coefficients);
      }
      return plan;
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL IIRFilterPlan<F64>: " << e.what()
                << ". Falling back to Default." << std::endl;
      return DefaultBackend::create_iir_filter_plan_f64(sos_coefficients);
    }
    catch (...) {
      std::cerr << "Unknown error creating oneMKL IIRFilterPlan<F64>. Falling "
                   "back to Default."
                << std::endl;
      return DefaultBackend::create_iir_filter_plan_f64(sos_coefficients);
    }
  }

  // --- Filter Design ---
  // Inherited from DefaultBackend

  // *** ADDED Factory Function Definition ***
  std::unique_ptr<AbstractBackend> create_onemkl_backend()
  {
    return std::make_unique<OneMKLBackend>();
  }

}  // namespace OmniDSP::backend
