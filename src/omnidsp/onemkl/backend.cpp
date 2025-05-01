/**
 * @file backend.cpp (onemkl)
 * @brief Implements the OneMKLBackend class methods using oneMKL functions.
 */

#include "backend.hpp"  // *** ADDED: Include the corresponding header ***

#ifdef OMNIDSP_USE_ONEMKL  // Compile guard

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

namespace OmniDSP::backend {

  // Helper function to check MKL status and convert to OmniDSP::Status
  inline Status mkl_status_to_omnidsp_status(MKL_LONG status)
  {
    if (status == DFTI_NO_ERROR) {
      return Status::Success;
    }
    std::cerr << "MKL DFTI Error: " << DftiErrorMessage(status)
              << " (Code: " << status << ")" << std::endl;
    if (status == DFTI_MEMORY_ERROR) return Status::AllocationError;
    if (status == DFTI_INVALID_CONFIGURATION) return Status::InvalidArgument;
    if (status == DFTI_INCONSISTENT_CONFIGURATION)
      return Status::InvalidArgument;
    if (status == DFTI_NUMBER_OF_THREADS_ERROR) return Status::BackendError;
    if (status == DFTI_UNIMPLEMENTED) return Status::UnsupportedFeature;
    return Status::BackendError;
  }

  //--------------------------------------------------------------------------
  // OneMKLBackend Method Definitions
  //--------------------------------------------------------------------------

  OneMKLBackend::OneMKLBackend()
  {
    std::cout << "oneMKL Backend Initialized." << std::endl;
  }

  OneMKLBackend::~OneMKLBackend()
  {
    std::cout << "oneMKL Backend Destroyed." << std::endl;
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
  [[nodiscard]] Status OneMKLBackend::flattop_window_f32(
      size_t length, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_flattop_window_onemkl(output.first(length));
  }
  [[nodiscard]] Status OneMKLBackend::flattop_window_f64(
      size_t length, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_flattop_window_onemkl(output.first(length));
  }
  [[nodiscard]] Status OneMKLBackend::gaussian_window_f32(
      size_t length, double stddev, std::span<F32> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_gaussian_window_onemkl(stddev, output.first(length));
  }
  [[nodiscard]] Status OneMKLBackend::gaussian_window_f64(
      size_t length, double stddev, std::span<F64> output) const
  {
    if (output.size() < length) return Status::SizeMismatch;
    return generate_gaussian_window_onemkl(stddev, output.first(length));
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

  // --- Plan Factories ---
  // Implementations remain largely the same as the previous refactoring step.

  [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C32>>>
  OneMKLBackend::create_fft_plan_c32(size_t length) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLFFTPlanImpl<C32>>(length);
      return FFTPlan<C32>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL FFTPlan<C32>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C64>>>
  OneMKLBackend::create_fft_plan_c64(size_t length) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLFFTPlanImpl<C64>>(length);
      return FFTPlan<C64>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL FFTPlan<C64>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F32>>>
  OneMKLBackend::create_rfft_plan_f32(size_t length) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLRFFTPlanImpl<F32>>(length);
      return RFFTPlan<F32>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL RFFTPlan<F32>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F64>>>
  OneMKLBackend::create_rfft_plan_f64(size_t length) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLRFFTPlanImpl<F64>>(length);
      return RFFTPlan<F64>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL RFFTPlan<F64>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
    }
  }

  // CQT Plan Factory is NOT overridden - inherits DefaultBackend's version

  [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<F32>>>
  OneMKLBackend::create_resample_plan_f32(const ResampleSpec& spec) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLResamplePlanImpl<F32>>(spec);
      return ResamplePlan<F32>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL ResamplePlan<F32>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<F64>>>
  OneMKLBackend::create_resample_plan_f64(const ResampleSpec& spec) const
  {
    try {
      auto pimpl_backend = std::make_unique<OneMKLResamplePlanImpl<F64>>(spec);
      return ResamplePlan<F64>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL ResamplePlan<F64>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
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
      return ConvolutionPlan<F32>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL ConvolutionPlan<F32>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
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
      return ConvolutionPlan<F64>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL ConvolutionPlan<F64>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
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
      return ConvolutionPlan<C32>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL ConvolutionPlan<C32>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
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
      return ConvolutionPlan<C64>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL ConvolutionPlan<C64>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
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
      return CorrelationPlan<F32>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL CorrelationPlan<F32>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
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
      return CorrelationPlan<F64>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL CorrelationPlan<F64>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
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
      return CorrelationPlan<C32>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL CorrelationPlan<C32>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
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
      return CorrelationPlan<C64>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL CorrelationPlan<C64>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<F32>>>
  OneMKLBackend::create_fir_filter_plan_f32(const F32Vec& coefficients) const
  {
    try {
      auto pimpl_backend
          = std::make_unique<OneMKLFIRFilterPlanImpl<F32>>(coefficients);
      return FIRFilterPlan<F32>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL FIRFilterPlan<F32>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<F64>>>
  OneMKLBackend::create_fir_filter_plan_f64(const F64Vec& coefficients) const
  {
    try {
      auto pimpl_backend
          = std::make_unique<OneMKLFIRFilterPlanImpl<F64>>(coefficients);
      return FIRFilterPlan<F64>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL FIRFilterPlan<F64>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<C32>>>
  OneMKLBackend::create_fir_filter_plan_c32(const C32Vec& coefficients) const
  {
    try {
      auto pimpl_backend
          = std::make_unique<OneMKLFIRFilterPlanImpl<C32>>(coefficients);
      return FIRFilterPlan<C32>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL FIRFilterPlan<C32>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<C64>>>
  OneMKLBackend::create_fir_filter_plan_c64(const C64Vec& coefficients) const
  {
    try {
      auto pimpl_backend
          = std::make_unique<OneMKLFIRFilterPlanImpl<C64>>(coefficients);
      return FIRFilterPlan<C64>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL FIRFilterPlan<C64>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F32>>>
  OneMKLBackend::create_iir_filter_plan_f32(
      const std::vector<IIRFilterCoef>& sos_coefficients) const
  {
    try {
      auto pimpl_backend
          = std::make_unique<OneMKLIIRFilterPlanImpl<F32>>(sos_coefficients);
      return IIRFilterPlan<F32>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL IIRFilterPlan<F32>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
    }
  }

  [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F64>>>
  OneMKLBackend::create_iir_filter_plan_f64(
      const std::vector<IIRFilterCoef>& sos_coefficients) const
  {
    try {
      auto pimpl_backend
          = std::make_unique<OneMKLIIRFilterPlanImpl<F64>>(sos_coefficients);
      return IIRFilterPlan<F64>::create_from_impl(std::move(pimpl_backend));
    }
    catch (const std::exception& e) {
      std::cerr << "Error creating oneMKL IIRFilterPlan<F64>: " << e.what()
                << std::endl;
      return std::unexpected(Status::BackendError);
    }
    catch (...) {
      return std::unexpected(Status::Failure);
    }
  }

  // --- Filter Design ---
  // Inherited from DefaultBackend

  // --- Explicit Template Instantiations ---
  // Plan Factories
  template OmniExpected<std::unique_ptr<FFTPlan<C32>>>
      OneMKLBackend::create_fft_plan_c32(size_t) const;
  template OmniExpected<std::unique_ptr<FFTPlan<C64>>>
      OneMKLBackend::create_fft_plan_c64(size_t) const;
  template OmniExpected<std::unique_ptr<RFFTPlan<F32>>>
      OneMKLBackend::create_rfft_plan_f32(size_t) const;
  template OmniExpected<std::unique_ptr<RFFTPlan<F64>>>
      OneMKLBackend::create_rfft_plan_f64(size_t) const;
  template OmniExpected<std::unique_ptr<ResamplePlan<F32>>>
  OneMKLBackend::create_resample_plan_f32(const ResampleSpec&) const;
  template OmniExpected<std::unique_ptr<ResamplePlan<F64>>>
  OneMKLBackend::create_resample_plan_f64(const ResampleSpec&) const;
  template OmniExpected<std::unique_ptr<ConvolutionPlan<F32>>>
  OneMKLBackend::create_convolution_plan_f32(
      const F32Vec&, ConvolutionType, ConvolutionMethod) const;
  template OmniExpected<std::unique_ptr<ConvolutionPlan<F64>>>
  OneMKLBackend::create_convolution_plan_f64(
      const F64Vec&, ConvolutionType, ConvolutionMethod) const;
  template OmniExpected<std::unique_ptr<ConvolutionPlan<C32>>>
  OneMKLBackend::create_convolution_plan_c32(
      const C32Vec&, ConvolutionType, ConvolutionMethod) const;
  template OmniExpected<std::unique_ptr<ConvolutionPlan<C64>>>
  OneMKLBackend::create_convolution_plan_c64(
      const C64Vec&, ConvolutionType, ConvolutionMethod) const;
  template OmniExpected<std::unique_ptr<CorrelationPlan<F32>>>
  OneMKLBackend::create_correlation_plan_f32(
      const F32Vec&, ConvolutionType, ConvolutionMethod) const;
  template OmniExpected<std::unique_ptr<CorrelationPlan<F64>>>
  OneMKLBackend::create_correlation_plan_f64(
      const F64Vec&, ConvolutionType, ConvolutionMethod) const;
  template OmniExpected<std::unique_ptr<CorrelationPlan<C32>>>
  OneMKLBackend::create_correlation_plan_c32(
      const C32Vec&, ConvolutionType, ConvolutionMethod) const;
  template OmniExpected<std::unique_ptr<CorrelationPlan<C64>>>
  OneMKLBackend::create_correlation_plan_c64(
      const C64Vec&, ConvolutionType, ConvolutionMethod) const;
  template OmniExpected<std::unique_ptr<FIRFilterPlan<F32>>>
  OneMKLBackend::create_fir_filter_plan_f32(const F32Vec&) const;
  template OmniExpected<std::unique_ptr<FIRFilterPlan<F64>>>
  OneMKLBackend::create_fir_filter_plan_f64(const F64Vec&) const;
  template OmniExpected<std::unique_ptr<FIRFilterPlan<C32>>>
  OneMKLBackend::create_fir_filter_plan_c32(const C32Vec&) const;
  template OmniExpected<std::unique_ptr<FIRFilterPlan<C64>>>
  OneMKLBackend::create_fir_filter_plan_c64(const C64Vec&) const;
  template OmniExpected<std::unique_ptr<IIRFilterPlan<F32>>>
  OneMKLBackend::create_iir_filter_plan_f32(
      const std::vector<IIRFilterCoef>&) const;
  template OmniExpected<std::unique_ptr<IIRFilterPlan<F64>>>
  OneMKLBackend::create_iir_filter_plan_f64(
      const std::vector<IIRFilterCoef>&) const;

}  // namespace OmniDSP::backend

#endif  // OMNIDSP_USE_ONEMKL
