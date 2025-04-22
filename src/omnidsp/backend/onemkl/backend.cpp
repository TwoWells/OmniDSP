/**
 * @file backend.cpp (onemkl)
 * @brief Implements the OneMKLOmniDSPImpl class methods using oneMKL functions.
 */

// Only compile this file if oneMKL backend is enabled via CMake
#ifdef USE_ONEMKL

#include "backend.h"  // Corresponding header for oneMKL backend declarations

// Include headers for the public Plan classes (needed for factory return types
// and constructors)
#include "OmniDSP/convolution.h"
#include "OmniDSP/cqt.h"
#include "OmniDSP/fft.h"
#include "OmniDSP/omnidsp.h"  // Needed for OmniDSP class definition (for owner pointer)
#include "OmniDSP/resample.h"
// #include "OmniDSP/filter.h" // Include when FilterPlan is added

// Include MKL header
#include <mkl.h>

#include <complex>
#include <iostream>  // For debug/error messages
#include <memory>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>     // For exception messages
#include <vector>

namespace OmniDSP {
namespace backend {

// Helper function to check MKL status and convert to OmniDSP::Status
inline Status mkl_status_to_omnidsp_status(MKL_LONG status) {
  if (status == DFTI_NO_ERROR) {
    return Status::Success;
  }
  // Add more specific mappings based on DFTI error codes if needed
  std::cerr << "MKL Error: " << DftiErrorMessage(status) << std::endl;
  // Example mappings (adjust as needed)
  if (status == DFTI_MEMORY_ERROR) return Status::AllocationError;
  if (status == DFTI_INVALID_CONFIGURATION) return Status::InvalidArgument;
  if (status == DFTI_INCONSISTENT_CONFIGURATION) return Status::InvalidArgument;
  if (status == DFTI_NUMBER_OF_THREADS_ERROR)
    return Status::BackendError;  // Or specific thread error?
  // ... other potential MKL errors ...
  return Status::BackendError;  // Generic backend error for other MKL issues
}

//--------------------------------------------------------------------------
// OneMKLOmniDSPImpl Method Definitions
//--------------------------------------------------------------------------

OneMKLOmniDSPImpl::OneMKLOmniDSPImpl() {
  // Perform any one-time setup specific to the oneMKL backend.
  // Example: Set threading mode (though often controlled by environment
  // variables) MKL_Set_Num_Threads(mkl_get_max_threads()); // Example: Use max
  // threads MKL_Set_Threading_Layer(MKL_THREADING_TBB); // Example: Use TBB

  // Example: Set DFTI default precision/domain (can also be set per descriptor)
  // DftiSetValue(nullptr, DFTI_PRECISION, DFTI_DOUBLE);
  // DftiSetValue(nullptr, DFTI_FORWARD_DOMAIN, DFTI_COMPLEX);

  std::cout << "oneMKL Backend Initialized." << std::endl;  // Debug message
}

Backend OneMKLOmniDSPImpl::get_backend() const { return Backend::OneMKL; }

// --- DSP Operations (Placeholders - Need oneMKL Implementation) ---
// These one-off functions would typically involve creating a temporary
// DFTI descriptor or VSL task, executing it, and then freeing it.

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>> OneMKLOmniDSPImpl::convolve(
    const std::vector<RealT<T>>& /*input*/,
    const std::vector<RealT<T>>& /*kernel*/, ConvolutionMode /*mode*/) const {
  // TODO: Implement 1D real convolution using oneMKL (e.g., VSL Conv/Corr task
  // or DFTI)
  std::cerr << "OneMKLOmniDSPImpl::convolve (real) not yet implemented."
            << std::endl;
  return std::unexpected(Status::UnsupportedFeature);
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<ComplexT<T>>>
OneMKLOmniDSPImpl::convolve(const std::vector<ComplexT<T>>& /*input*/,
                            const std::vector<ComplexT<T>>& /*kernel*/,
                            ConvolutionMode /*mode*/) const {
  // TODO: Implement 1D complex convolution using oneMKL (e.g., VSL Conv/Corr
  // task or DFTI)
  std::cerr << "OneMKLOmniDSPImpl::convolve (complex) not yet implemented."
            << std::endl;
  return std::unexpected(Status::UnsupportedFeature);
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>> OneMKLOmniDSPImpl::correlate(
    const std::vector<RealT<T>>& /*input*/,
    const std::vector<RealT<T>>& /*kernel*/, ConvolutionMode /*mode*/) const {
  // TODO: Implement 1D real correlation using oneMKL (e.g., VSL Conv/Corr task
  // or DFTI)
  std::cerr << "OneMKLOmniDSPImpl::correlate (real) not yet implemented."
            << std::endl;
  return std::unexpected(Status::UnsupportedFeature);
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<ComplexT<T>>>
OneMKLOmniDSPImpl::correlate(const std::vector<ComplexT<T>>& /*input*/,
                             const std::vector<ComplexT<T>>& /*kernel*/,
                             ConvolutionMode /*mode*/) const {
  // TODO: Implement 1D complex correlation using oneMKL (e.g., VSL Conv/Corr
  // task or DFTI)
  std::cerr << "OneMKLOmniDSPImpl::correlate (complex) not yet implemented."
            << std::endl;
  return std::unexpected(Status::UnsupportedFeature);
}

// --- One-off FFTs (Placeholders - Need oneMKL Implementation) ---
template <typename T>
[[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> OneMKLOmniDSPImpl::fft(
    const std::vector<ComplexT<T>>& input) const {
  // TODO: Implement one-off complex FFT using oneMKL DFTI
  // (Create descriptor, commit, compute, free)
  std::cerr << "OneMKLOmniDSPImpl::fft not yet implemented." << std::endl;
  return std::unexpected(Status::UnsupportedFeature);
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> OneMKLOmniDSPImpl::ifft(
    const std::vector<ComplexT<T>>& input) const {
  // TODO: Implement one-off complex IFFT using oneMKL DFTI
  std::cerr << "OneMKLOmniDSPImpl::ifft not yet implemented." << std::endl;
  return std::unexpected(Status::UnsupportedFeature);
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> OneMKLOmniDSPImpl::rfft(
    const std::vector<RealT<T>>& input) const {
  // TODO: Implement one-off real RFFT using oneMKL DFTI
  std::cerr << "OneMKLOmniDSPImpl::rfft not yet implemented." << std::endl;
  return std::unexpected(Status::UnsupportedFeature);
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>> OneMKLOmniDSPImpl::irfft(
    const std::vector<ComplexT<T>>& input, size_t /*output_length*/) const {
  // TODO: Implement one-off real IRFFT using oneMKL DFTI
  std::cerr << "OneMKLOmniDSPImpl::irfft not yet implemented." << std::endl;
  return std::unexpected(Status::UnsupportedFeature);
}

// --- Window Generation (Placeholders - Need oneMKL VML Implementation) ---
// oneMKL VML (Vector Math Library) provides functions for basic arithmetic,
// trigonometry, exponentials, etc., which can be used to implement windows.
// It does not have dedicated window functions like Accelerate's vDSP.

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
OneMKLOmniDSPImpl::bartlett_window(size_t /*length*/) const {
  // TODO: Implement using MKL VML functions (e.g., v?Ramp, v?Abs, v?Add, v?Sub,
  // v?LinearFrac)
  std::cerr << "OneMKLOmniDSPImpl::bartlett_window not yet implemented."
            << std::endl;
  return std::unexpected(Status::UnsupportedFeature);
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
OneMKLOmniDSPImpl::blackman_window(size_t /*length*/) const {
  // TODO: Implement using MKL VML functions (v?Ramp, v?Cos, v?Add, v?Mul, etc.)
  std::cerr << "OneMKLOmniDSPImpl::blackman_window not yet implemented."
            << std::endl;
  return std::unexpected(Status::UnsupportedFeature);
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
OneMKLOmniDSPImpl::flattop_window(size_t /*length*/) const {
  // TODO: Implement using MKL VML functions
  std::cerr << "OneMKLOmniDSPImpl::flattop_window not yet implemented."
            << std::endl;
  return std::unexpected(Status::UnsupportedFeature);
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
OneMKLOmniDSPImpl::gaussian_window(size_t /*length*/,
                                   RealT<T> /*stddev*/) const {
  // TODO: Implement using MKL VML functions (v?Ramp, v?Sub, v?Sqr, v?Exp, etc.)
  std::cerr << "OneMKLOmniDSPImpl::gaussian_window not yet implemented."
            << std::endl;
  return std::unexpected(Status::UnsupportedFeature);
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
OneMKLOmniDSPImpl::hamming_window(size_t /*length*/) const {
  // TODO: Implement using MKL VML functions
  std::cerr << "OneMKLOmniDSPImpl::hamming_window not yet implemented."
            << std::endl;
  return std::unexpected(Status::UnsupportedFeature);
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
OneMKLOmniDSPImpl::hann_window(size_t /*length*/) const {
  // TODO: Implement using MKL VML functions
  std::cerr << "OneMKLOmniDSPImpl::hann_window not yet implemented."
            << std::endl;
  return std::unexpected(Status::UnsupportedFeature);
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
OneMKLOmniDSPImpl::kaiser_window(size_t /*length*/, RealT<T> /*beta*/) const {
  // TODO: Implement using MKL VML functions and Boost Bessel function
  std::cerr << "OneMKLOmniDSPImpl::kaiser_window not yet implemented."
            << std::endl;
  return std::unexpected(Status::UnsupportedFeature);
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
OneMKLOmniDSPImpl::rectangular_window(size_t length) const {
  // TODO: Implement using MKL VML (e.g., generate constant value) or simple
  // loop
  if (length == 0) return std::vector<RealT<T>>();
  // MKL doesn't have a direct vfill like vDSP, use loop or BLAS copy from
  // scalar?
  std::vector<RealT<T>> coeffs(length, static_cast<RealT<T>>(1.0));
  return coeffs;  // Simple implementation
  // std::cerr << "OneMKLOmniDSPImpl::rectangular_window not fully optimized."
  // << std::endl; return std::unexpected(Status::UnsupportedFeature);
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
OneMKLOmniDSPImpl::triangular_window(size_t /*length*/) const {
  // TODO: Implement using MKL VML functions
  std::cerr << "OneMKLOmniDSPImpl::triangular_window not yet implemented."
            << std::endl;
  return std::unexpected(Status::UnsupportedFeature);
}

// --- Plan Factories ---
// These follow the pattern: create the OneMKL*Impl, wrap in unique_ptr, call
// public Plan constructor.

template <typename T>
[[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<T>>>
OneMKLOmniDSPImpl::create_fft_plan(size_t length) const {
  try {
    auto pimpl_backend = std::make_unique<OneMKLFFTPlanImpl<T>>(length);
    // Check MKL status from constructor if stored
    if (!pimpl_backend || pimpl_backend->mkl_status_ != DFTI_NO_ERROR) {
      return std::unexpected(mkl_status_to_omnidsp_status(
          pimpl_backend ? pimpl_backend->mkl_status_
                        : -1));  // Use -1 or specific code for null case
    }
    // Construct public Plan object using private constructor (OmniDSP is
    // friend)
    FFTPlan<T>* plan_ptr = new FFTPlan<T>(std::move(pimpl_backend));
    return std::unique_ptr<FFTPlan<T>>(plan_ptr);

  } catch (const std::bad_alloc&) {
    return std::unexpected(Status::AllocationError);
  } catch (const std::exception& e) {
    std::cerr << "Error creating oneMKL FFTPlanImpl: " << e.what() << std::endl;
    return std::unexpected(Status::BackendError);
  } catch (...) {
    return std::unexpected(Status::Failure);
  }
}

template <typename T>
[[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<T>>>
OneMKLOmniDSPImpl::create_rfft_plan(size_t length) const {
  try {
    auto pimpl_backend = std::make_unique<OneMKLRFFTPlanImpl<T>>(length);
    if (!pimpl_backend || pimpl_backend->mkl_status_ != DFTI_NO_ERROR) {
      return std::unexpected(mkl_status_to_omnidsp_status(
          pimpl_backend ? pimpl_backend->mkl_status_ : -1));
    }
    RFFTPlan<T>* plan_ptr = new RFFTPlan<T>(std::move(pimpl_backend));
    return std::unique_ptr<RFFTPlan<T>>(plan_ptr);
  } catch (const std::bad_alloc&) {
    return std::unexpected(Status::AllocationError);
  } catch (const std::exception& e) {
    std::cerr << "Error creating oneMKL RFFTPlanImpl: " << e.what()
              << std::endl;
    return std::unexpected(Status::BackendError);
  } catch (...) {
    return std::unexpected(Status::Failure);
  }
}

template <typename T>
[[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<T>>>
OneMKLOmniDSPImpl::create_cqt_plan(
    const OmniDSP* owner,  // Pass owner for sub-plan creation
    RealT<T> sample_rate, RealT<T> min_freq, RealT<T> max_freq,
    int bins_per_octave) const {
  // CQTPlan doesn't use Pimpl directly. It needs the OmniDSP owner
  // to create its internal FFT/Resample plans using the *correct* backend
  // (oneMKL). We just call the public CQTPlan constructor (which is friended by
  // OmniDSP).
  try {
    // The CQTPlan constructor itself will call back to owner->create_fft_plan
    // etc., which will resolve to the oneMKL implementations via the owner
    // object.
    CQTPlan<T>* plan_ptr =
        new CQTPlan<T>(owner, sample_rate, min_freq, max_freq, bins_per_octave);
    return std::unique_ptr<CQTPlan<T>>(plan_ptr);

  } catch (const std::bad_alloc&) {
    return std::unexpected(Status::AllocationError);
  } catch (const std::exception& e) {
    // Catch errors from CQT setup or internal plan creation
    std::cerr << "Error creating CQTPlan (oneMKL backend context): " << e.what()
              << std::endl;
    return std::unexpected(
        Status::Failure);  // Or BackendError if sub-plan creation failed
  } catch (...) {
    return std::unexpected(Status::Failure);
  }
}

template <typename T>
[[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<T>>>
OneMKLOmniDSPImpl::create_resample_plan(double input_rate, double output_rate,
                                        size_t max_input_size) const {
  try {
    auto pimpl_backend = std::make_unique<OneMKLResamplePlanImpl<T>>(
        input_rate, output_rate, max_input_size);
    // Add status check if OneMKLResamplePlanImpl constructor can fail
    // detectably
    ResamplePlan<T>* plan_ptr = new ResamplePlan<T>(std::move(pimpl_backend));
    return std::unique_ptr<ResamplePlan<T>>(plan_ptr);
  } catch (const std::bad_alloc&) {
    return std::unexpected(Status::AllocationError);
  } catch (const std::exception& e) {
    std::cerr << "Error creating oneMKL ResamplePlanImpl: " << e.what()
              << std::endl;
    return std::unexpected(Status::BackendError);
  } catch (...) {
    return std::unexpected(Status::Failure);
  }
}

template <typename T>
[[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
OneMKLOmniDSPImpl::create_convolution_plan(const std::vector<T>& kernel,
                                           ConvolutionMode mode) const {
  try {
    auto pimpl_backend =
        std::make_unique<OneMKLConvolutionPlanImpl<T>>(kernel, mode);
    // Add status check if constructor can fail detectably
    if (!pimpl_backend || pimpl_backend->mkl_status_ !=
                              DFTI_NO_ERROR) {  // Example check if using DFTI
      return std::unexpected(mkl_status_to_omnidsp_status(
          pimpl_backend ? pimpl_backend->mkl_status_ : -1));
    }
    ConvolutionPlan<T>* plan_ptr =
        new ConvolutionPlan<T>(std::move(pimpl_backend));
    return std::unique_ptr<ConvolutionPlan<T>>(plan_ptr);
  } catch (const std::bad_alloc&) {
    return std::unexpected(Status::AllocationError);
  } catch (const std::exception& e) {
    std::cerr << "Error creating oneMKL ConvolutionPlanImpl: " << e.what()
              << std::endl;
    return std::unexpected(Status::BackendError);
  } catch (...) {
    return std::unexpected(Status::Failure);
  }
}

template <typename T>
[[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
OneMKLOmniDSPImpl::create_correlation_plan(const std::vector<T>& kernel,
                                           ConvolutionMode mode) const {
  try {
    auto pimpl_backend =
        std::make_unique<OneMKLCorrelationPlanImpl<T>>(kernel, mode);
    // Add status check if constructor can fail detectably
    if (!pimpl_backend || pimpl_backend->mkl_status_ !=
                              DFTI_NO_ERROR) {  // Example check if using DFTI
      return std::unexpected(mkl_status_to_omnidsp_status(
          pimpl_backend ? pimpl_backend->mkl_status_ : -1));
    }
    CorrelationPlan<T>* plan_ptr =
        new CorrelationPlan<T>(std::move(pimpl_backend));
    return std::unique_ptr<CorrelationPlan<T>>(plan_ptr);
  } catch (const std::bad_alloc&) {
    return std::unexpected(Status::AllocationError);
  } catch (const std::exception& e) {
    std::cerr << "Error creating oneMKL CorrelationPlanImpl: " << e.what()
              << std::endl;
    return std::unexpected(Status::BackendError);
  } catch (...) {
    return std::unexpected(Status::Failure);
  }
}

// Add implementations for FilterPlan factory and filter design methods when
// added

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
// Instantiate templates for common types (float, double) for OneMKLOmniDSPImpl
// methods

// Define types for brevity
using float_c = OmniDSP::ComplexT<float>;
using double_c = OmniDSP::ComplexT<double>;

// DSP Operations
template OmniExpected<std::vector<float>> OneMKLOmniDSPImpl::convolve(
    const std::vector<float>&, const std::vector<float>&,
    ConvolutionMode) const;
template OmniExpected<std::vector<double>> OneMKLOmniDSPImpl::convolve(
    const std::vector<double>&, const std::vector<double>&,
    ConvolutionMode) const;
template OmniExpected<std::vector<float_c>> OneMKLOmniDSPImpl::convolve(
    const std::vector<float_c>&, const std::vector<float_c>&,
    ConvolutionMode) const;
template OmniExpected<std::vector<double_c>> OneMKLOmniDSPImpl::convolve(
    const std::vector<double_c>&, const std::vector<double_c>&,
    ConvolutionMode) const;

template OmniExpected<std::vector<float>> OneMKLOmniDSPImpl::correlate(
    const std::vector<float>&, const std::vector<float>&,
    ConvolutionMode) const;
template OmniExpected<std::vector<double>> OneMKLOmniDSPImpl::correlate(
    const std::vector<double>&, const std::vector<double>&,
    ConvolutionMode) const;
template OmniExpected<std::vector<float_c>> OneMKLOmniDSPImpl::correlate(
    const std::vector<float_c>&, const std::vector<float_c>&,
    ConvolutionMode) const;
template OmniExpected<std::vector<double_c>> OneMKLOmniDSPImpl::correlate(
    const std::vector<double_c>&, const std::vector<double_c>&,
    ConvolutionMode) const;

// One-off FFTs
template OmniExpected<std::vector<float_c>> OneMKLOmniDSPImpl::fft(
    const std::vector<float_c>&) const;
template OmniExpected<std::vector<double_c>> OneMKLOmniDSPImpl::fft(
    const std::vector<double_c>&) const;
template OmniExpected<std::vector<float_c>> OneMKLOmniDSPImpl::ifft(
    const std::vector<float_c>&) const;
template OmniExpected<std::vector<double_c>> OneMKLOmniDSPImpl::ifft(
    const std::vector<double_c>&) const;
template OmniExpected<std::vector<float_c>> OneMKLOmniDSPImpl::rfft(
    const std::vector<float>&) const;
template OmniExpected<std::vector<double_c>> OneMKLOmniDSPImpl::rfft(
    const std::vector<double>&) const;
template OmniExpected<std::vector<float>> OneMKLOmniDSPImpl::irfft(
    const std::vector<float_c>&, size_t) const;
template OmniExpected<std::vector<double>> OneMKLOmniDSPImpl::irfft(
    const std::vector<double_c>&, size_t) const;

// Window Generation
template OmniExpected<std::vector<float>> OneMKLOmniDSPImpl::bartlett_window(
    size_t) const;
template OmniExpected<std::vector<double>> OneMKLOmniDSPImpl::bartlett_window(
    size_t) const;
template OmniExpected<std::vector<float>> OneMKLOmniDSPImpl::blackman_window(
    size_t) const;
template OmniExpected<std::vector<double>> OneMKLOmniDSPImpl::blackman_window(
    size_t) const;
template OmniExpected<std::vector<float>> OneMKLOmniDSPImpl::flattop_window(
    size_t) const;
template OmniExpected<std::vector<double>> OneMKLOmniDSPImpl::flattop_window(
    size_t) const;
template OmniExpected<std::vector<float>> OneMKLOmniDSPImpl::gaussian_window(
    size_t, float) const;
template OmniExpected<std::vector<double>> OneMKLOmniDSPImpl::gaussian_window(
    size_t, double) const;
template OmniExpected<std::vector<float>> OneMKLOmniDSPImpl::hamming_window(
    size_t) const;
template OmniExpected<std::vector<double>> OneMKLOmniDSPImpl::hamming_window(
    size_t) const;
template OmniExpected<std::vector<float>> OneMKLOmniDSPImpl::hann_window(
    size_t) const;
template OmniExpected<std::vector<double>> OneMKLOmniDSPImpl::hann_window(
    size_t) const;
template OmniExpected<std::vector<float>> OneMKLOmniDSPImpl::kaiser_window(
    size_t, float) const;
template OmniExpected<std::vector<double>> OneMKLOmniDSPImpl::kaiser_window(
    size_t, double) const;
template OmniExpected<std::vector<float>> OneMKLOmniDSPImpl::rectangular_window(
    size_t) const;
template OmniExpected<std::vector<double>>
    OneMKLOmniDSPImpl::rectangular_window(size_t) const;
template OmniExpected<std::vector<float>> OneMKLOmniDSPImpl::triangular_window(
    size_t) const;
template OmniExpected<std::vector<double>> OneMKLOmniDSPImpl::triangular_window(
    size_t) const;

// Plan Factories
template OmniExpected<std::unique_ptr<FFTPlan<float_c>>>
    OneMKLOmniDSPImpl::create_fft_plan(size_t) const;
template OmniExpected<std::unique_ptr<FFTPlan<double_c>>>
    OneMKLOmniDSPImpl::create_fft_plan(size_t) const;

template OmniExpected<std::unique_ptr<RFFTPlan<float>>>
    OneMKLOmniDSPImpl::create_rfft_plan(size_t) const;
template OmniExpected<std::unique_ptr<RFFTPlan<double>>>
    OneMKLOmniDSPImpl::create_rfft_plan(size_t) const;

template OmniExpected<std::unique_ptr<CQTPlan<float>>>
OneMKLOmniDSPImpl::create_cqt_plan(const OmniDSP*, float, float, float,
                                   int) const;
template OmniExpected<std::unique_ptr<CQTPlan<double>>>
OneMKLOmniDSPImpl::create_cqt_plan(const OmniDSP*, double, double, double,
                                   int) const;

template OmniExpected<std::unique_ptr<ResamplePlan<float>>>
OneMKLOmniDSPImpl::create_resample_plan(double, double, size_t) const;
template OmniExpected<std::unique_ptr<ResamplePlan<double>>>
OneMKLOmniDSPImpl::create_resample_plan(double, double, size_t) const;

template OmniExpected<std::unique_ptr<ConvolutionPlan<float>>>
OneMKLOmniDSPImpl::create_convolution_plan(const std::vector<float>&,
                                           ConvolutionMode) const;
template OmniExpected<std::unique_ptr<ConvolutionPlan<double>>>
OneMKLOmniDSPImpl::create_convolution_plan(const std::vector<double>&,
                                           ConvolutionMode) const;
template OmniExpected<std::unique_ptr<ConvolutionPlan<float_c>>>
OneMKLOmniDSPImpl::create_convolution_plan(const std::vector<float_c>&,
                                           ConvolutionMode) const;
template OmniExpected<std::unique_ptr<ConvolutionPlan<double_c>>>
OneMKLOmniDSPImpl::create_convolution_plan(const std::vector<double_c>&,
                                           ConvolutionMode) const;

template OmniExpected<std::unique_ptr<CorrelationPlan<float>>>
OneMKLOmniDSPImpl::create_correlation_plan(const std::vector<float>&,
                                           ConvolutionMode) const;
template OmniExpected<std::unique_ptr<CorrelationPlan<double>>>
OneMKLOmniDSPImpl::create_correlation_plan(const std::vector<double>&,
                                           ConvolutionMode) const;
template OmniExpected<std::unique_ptr<CorrelationPlan<float_c>>>
OneMKLOmniDSPImpl::create_correlation_plan(const std::vector<float_c>&,
                                           ConvolutionMode) const;
template OmniExpected<std::unique_ptr<CorrelationPlan<double_c>>>
OneMKLOmniDSPImpl::create_correlation_plan(const std::vector<double_c>&,
                                           ConvolutionMode) const;

}  // namespace backend
}  // namespace OmniDSP

#endif  // USE_ONEMKL
