/**
 * @file backend.cpp (stub)
 * @brief Implements the StubOmniDSPImpl class methods using standard C++.
 */

#include "backend.h"  // Corresponding header for Stub backend declarations

// Include headers for the public Plan classes (needed for factory return types
// and constructors)
#include "OmniDSP/convolution.h"
#include "OmniDSP/cqt.h"
#include "OmniDSP/fft.h"
#include "OmniDSP/omnidsp.h"  // Needed for OmniDSP class definition (for owner pointer)
#include "OmniDSP/resample.h"
#include "OmniDSP/window.h"  // For WindowSpec
// #include "OmniDSP/filter.h" // Include when FilterPlan is added

// Include standard library headers needed for stub implementations
#include <algorithm>  // For std::reverse, std::copy, std::fill
#include <cmath>      // For M_PI, sin, cos, exp, sqrt, abs etc.
#include <complex>
#include <iostream>  // For debug/error messages
#include <memory>
#include <numeric>    // For std::accumulate, std::gcd
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>     // For exception messages
#include <vector>

// Include Boost Bessel function for Kaiser window
#include <boost/math/special_functions/bessel.hpp>

// Define PI if not available from cmath
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace OmniDSP {
namespace backend {

// Helper function (move to common utility?)
inline Status check_status(bool condition, Status error_status,
                           const std::string& error_message = "") {
  if (!condition) {
    if (!error_message.empty()) {
      std::cerr << "Stub Backend Error: " << error_message << std::endl;
    }
    return error_status;
  }
  return Status::Success;
}

//--------------------------------------------------------------------------
// StubOmniDSPImpl Method Definitions
//--------------------------------------------------------------------------

StubOmniDSPImpl::StubOmniDSPImpl() {
  // No specific setup needed for the stub backend
  std::cout << "Stub Backend Initialized." << std::endl;  // Debug message
}

Backend StubOmniDSPImpl::get_backend() const { return Backend::Stub; }

// --- DSP Operations ---
// Implementations often delegate to the corresponding Plan for consistency,
// or provide a direct basic loop-based implementation.

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>> StubOmniDSPImpl::convolve(
    const std::vector<RealT<T>>& input, const std::vector<RealT<T>>& kernel,
    ConvolutionMode mode) const {
  try {
    // Create a temporary plan and execute it
    StubConvolutionPlanImpl<RealT<T>> plan(kernel, mode);
    size_t output_len = plan.get_output_length(input.size());
    std::vector<RealT<T>> output(output_len);
    Status status = plan.execute(input, output);
    if (status != Status::Success) {
      return std::unexpected(status);
    }
    return output;
  } catch (const std::exception& e) {
    std::cerr << "Stub convolve (real) error: " << e.what() << std::endl;
    return std::unexpected(Status::Failure);  // Or more specific error
  }
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> StubOmniDSPImpl::convolve(
    const std::vector<ComplexT<T>>& input,
    const std::vector<ComplexT<T>>& kernel, ConvolutionMode mode) const {
  try {
    StubConvolutionPlanImpl<ComplexT<T>> plan(kernel, mode);
    size_t output_len = plan.get_output_length(input.size());
    std::vector<ComplexT<T>> output(output_len);
    Status status = plan.execute(input, output);
    if (status != Status::Success) {
      return std::unexpected(status);
    }
    return output;
  } catch (const std::exception& e) {
    std::cerr << "Stub convolve (complex) error: " << e.what() << std::endl;
    return std::unexpected(Status::Failure);
  }
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>> StubOmniDSPImpl::correlate(
    const std::vector<RealT<T>>& input, const std::vector<RealT<T>>& kernel,
    ConvolutionMode mode) const {
  try {
    StubCorrelationPlanImpl<RealT<T>> plan(kernel, mode);
    size_t output_len = plan.get_output_length(input.size());
    std::vector<RealT<T>> output(output_len);
    Status status = plan.execute(input, output);
    if (status != Status::Success) {
      return std::unexpected(status);
    }
    return output;
  } catch (const std::exception& e) {
    std::cerr << "Stub correlate (real) error: " << e.what() << std::endl;
    return std::unexpected(Status::Failure);
  }
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> StubOmniDSPImpl::correlate(
    const std::vector<ComplexT<T>>& input,
    const std::vector<ComplexT<T>>& kernel, ConvolutionMode mode) const {
  try {
    StubCorrelationPlanImpl<ComplexT<T>> plan(kernel, mode);
    size_t output_len = plan.get_output_length(input.size());
    std::vector<ComplexT<T>> output(output_len);
    Status status = plan.execute(input, output);
    if (status != Status::Success) {
      return std::unexpected(status);
    }
    return output;
  } catch (const std::exception& e) {
    std::cerr << "Stub correlate (complex) error: " << e.what() << std::endl;
    return std::unexpected(Status::Failure);
  }
}

// --- One-off FFTs ---
template <typename T>
[[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> StubOmniDSPImpl::fft(
    const std::vector<ComplexT<T>>& input) const {
  try {
    StubFFTPlanImpl<ComplexT<T>> plan(input.size());
    std::vector<ComplexT<T>> output(input.size());
    Status status = plan.fft(input, output);
    if (status != Status::Success) {
      return std::unexpected(status);
    }
    return output;
  } catch (const std::exception& e) {
    std::cerr << "Stub fft error: " << e.what() << std::endl;
    return std::unexpected(Status::Failure);
  }
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> StubOmniDSPImpl::ifft(
    const std::vector<ComplexT<T>>& input) const {
  try {
    StubFFTPlanImpl<ComplexT<T>> plan(input.size());
    std::vector<ComplexT<T>> output(input.size());
    Status status = plan.ifft(input, output);
    if (status != Status::Success) {
      return std::unexpected(status);
    }
    // Apply 1/N scaling for IFFT consistency? Or leave to user?
    // Let's apply scaling here for the one-off function.
    if (!output.empty()) {
      RealT<typename T::value_type> scale =
          1.0 / static_cast<RealT<typename T::value_type>>(output.size());
      for (auto& val : output) {
        val *= scale;
      }
    }
    return output;
  } catch (const std::exception& e) {
    std::cerr << "Stub ifft error: " << e.what() << std::endl;
    return std::unexpected(Status::Failure);
  }
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<ComplexT<T>>> StubOmniDSPImpl::rfft(
    const std::vector<RealT<T>>& input) const {
  try {
    StubRFFTPlanImpl<RealT<T>> plan(input.size());
    size_t output_len = (input.size() / 2) + 1;
    std::vector<ComplexT<T>> output(output_len);
    Status status = plan.rfft(input, output);
    if (status != Status::Success) {
      return std::unexpected(status);
    }
    return output;
  } catch (const std::exception& e) {
    std::cerr << "Stub rfft error: " << e.what() << std::endl;
    return std::unexpected(Status::Failure);
  }
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>> StubOmniDSPImpl::irfft(
    const std::vector<ComplexT<T>>& input, size_t output_length) const {
  try {
    // Need to infer original length N from input size (N/2 + 1)
    if (input.empty()) {
      if (output_length == 0)
        return std::vector<RealT<T>>();
      else
        return std::unexpected(Status::InvalidArgument);  // Cannot infer N
    }
    size_t N = (input.size() - 1) * 2;
    if (N == 0 && input.size() == 1) N = 1;  // Handle N=1 case
    if (N != output_length) {
      std::cerr << "Stub irfft error: output_length (" << output_length
                << ") does not match length (N=" << N
                << ") inferred from input size (" << input.size() << ")."
                << std::endl;
      return std::unexpected(Status::InvalidArgument);
    }

    StubRFFTPlanImpl<RealT<T>> plan(output_length);
    std::vector<RealT<T>> output(output_length);
    Status status = plan.irfft(input, output);
    if (status != Status::Success) {
      return std::unexpected(status);
    }
    // Apply 1/N scaling?
    if (!output.empty()) {
      RealT<T> scale = 1.0 / static_cast<RealT<T>>(output.size());
      for (auto& val : output) {
        val *= scale;
      }
    }
    return output;
  } catch (const std::exception& e) {
    std::cerr << "Stub irfft error: " << e.what() << std::endl;
    return std::unexpected(Status::Failure);
  }
}

// --- Window Generation ---
// These directly implement the standard formulas using std::math functions.

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
StubOmniDSPImpl::bartlett_window(size_t length) const {
  if (length == 0) return std::vector<RealT<T>>();
  std::vector<RealT<T>> coeffs(length);
  RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
  if (N_minus_1 <= 0.0) {  // Handle length 1
    if (length == 1) coeffs[0] = 1.0;
    return coeffs;
  }
  for (size_t n = 0; n < length; ++n) {
    coeffs[n] =
        1.0 - std::abs(2.0 * static_cast<RealT<T>>(n) / N_minus_1 - 1.0);
  }
  return coeffs;
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
StubOmniDSPImpl::blackman_window(size_t length) const {
  if (length == 0) return std::vector<RealT<T>>();
  if (length == 1) return std::vector<RealT<T>>(1, 1.0);
  std::vector<RealT<T>> coeffs(length);
  const RealT<T> a0 = 0.42;
  const RealT<T> a1 = 0.5;
  const RealT<T> a2 = 0.08;
  const RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
  const RealT<T> factor1 = (N_minus_1 > 0) ? (2.0 * M_PI / N_minus_1) : 0.0;
  const RealT<T> factor2 = 2.0 * factor1;
  for (size_t n = 0; n < length; ++n) {
    RealT<T> n_T = static_cast<RealT<T>>(n);
    coeffs[n] =
        a0 - a1 * std::cos(factor1 * n_T) + a2 * std::cos(factor2 * n_T);
  }
  return coeffs;
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
StubOmniDSPImpl::flattop_window(size_t length) const {
  if (length == 0) return std::vector<RealT<T>>();
  if (length == 1) return std::vector<RealT<T>>(1, 1.0);
  std::vector<RealT<T>> coeffs(length);
  const RealT<T> a0 = 0.21557895;
  const RealT<T> a1 = 0.41663158;
  const RealT<T> a2 = 0.277263158;
  const RealT<T> a3 = 0.083578947;
  const RealT<T> a4 = 0.006947368;
  const RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
  const RealT<T> factor = (N_minus_1 > 0) ? (2.0 * M_PI / N_minus_1) : 0.0;
  for (size_t n = 0; n < length; ++n) {
    RealT<T> n_T = static_cast<RealT<T>>(n);
    coeffs[n] =
        a0 - a1 * std::cos(factor * n_T) + a2 * std::cos(2.0 * factor * n_T) -
        a3 * std::cos(3.0 * factor * n_T) + a4 * std::cos(4.0 * factor * n_T);
  }
  return coeffs;
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
StubOmniDSPImpl::gaussian_window(size_t length, RealT<T> stddev) const {
  if (length == 0) return std::vector<RealT<T>>();
  if (length == 1) return std::vector<RealT<T>>(1, 1.0);
  if (stddev <= 0) return std::unexpected(Status::InvalidArgument);
  std::vector<RealT<T>> coeffs(length);
  RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
  RealT<T> center = N_minus_1 / 2.0;
  RealT<T> sigma_term = stddev * center;
  if (sigma_term == 0.0) {
    coeffs[0] = 1.0;
    return coeffs;
  }  // Should be caught by length==1
  RealT<T> factor = -0.5 / (sigma_term * sigma_term);
  for (size_t n = 0; n < length; ++n) {
    RealT<T> n_T = static_cast<RealT<T>>(n);
    RealT<T> exponent = n_T - center;
    coeffs[n] = std::exp(factor * exponent * exponent);
  }
  return coeffs;
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
StubOmniDSPImpl::hamming_window(size_t length) const {
  if (length == 0) return std::vector<RealT<T>>();
  if (length == 1) return std::vector<RealT<T>>(1, 1.0);
  std::vector<RealT<T>> coeffs(length);
  const RealT<T> a0 = 0.54;
  const RealT<T> a1 = 0.46;  // 1.0 - a0
  const RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
  const RealT<T> factor = (N_minus_1 > 0) ? (2.0 * M_PI / N_minus_1) : 0.0;
  for (size_t n = 0; n < length; ++n) {
    coeffs[n] = a0 - a1 * std::cos(factor * static_cast<RealT<T>>(n));
  }
  return coeffs;
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>> StubOmniDSPImpl::hann_window(
    size_t length) const {
  if (length == 0) return std::vector<RealT<T>>();
  if (length == 1) return std::vector<RealT<T>>(1, 1.0);
  std::vector<RealT<T>> coeffs(length);
  const RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
  const RealT<T> factor = (N_minus_1 > 0) ? (2.0 * M_PI / N_minus_1) : 0.0;
  for (size_t n = 0; n < length; ++n) {
    coeffs[n] = 0.5 * (1.0 - std::cos(factor * static_cast<RealT<T>>(n)));
  }
  return coeffs;
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
StubOmniDSPImpl::kaiser_window(size_t length, RealT<T> beta) const {
  if (length == 0) return std::vector<RealT<T>>();
  if (length == 1) return std::vector<RealT<T>>(1, 1.0);
  if (beta < 0) return std::unexpected(Status::InvalidArgument);
  std::vector<RealT<T>> coeffs(length);
  RealT<T> bessel_i0_beta = boost::math::cyl_bessel_i(0, beta);
  if (bessel_i0_beta == 0.0) return std::unexpected(Status::Failure);
  RealT<T> inv_denom = 1.0 / bessel_i0_beta;
  RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
  RealT<T> factor = (N_minus_1 > 0) ? (2.0 / N_minus_1) : 0.0;
  for (size_t n = 0; n < length; ++n) {
    RealT<T> term = factor * static_cast<RealT<T>>(n) - 1.0;
    RealT<T> arg_sqrt = 1.0 - term * term;
    RealT<T> bessel_arg =
        beta * std::sqrt(std::max(static_cast<RealT<T>>(0.0), arg_sqrt));
    coeffs[n] = boost::math::cyl_bessel_i(0, bessel_arg) * inv_denom;
  }
  return coeffs;
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
StubOmniDSPImpl::rectangular_window(size_t length) const {
  return std::vector<RealT<T>>(length, 1.0);
}

template <typename T>
[[nodiscard]] OmniExpected<std::vector<RealT<T>>>
StubOmniDSPImpl::triangular_window(size_t length) const {
  if (length == 0) return std::vector<RealT<T>>();
  if (length == 1) return std::vector<RealT<T>>(1, 1.0);
  std::vector<RealT<T>> coeffs(length);
  RealT<T> L = static_cast<RealT<T>>(length);
  RealT<T> center = (L - 1.0) / 2.0;
  RealT<T> norm = L / 2.0;  // Normalization factor
  if (norm == 0.0)
    return std::unexpected(Status::Failure);  // Should be caught by length==1
  for (size_t n = 0; n < length; ++n) {
    coeffs[n] = 1.0 - std::abs(static_cast<RealT<T>>(n) - center) / norm;
  }
  return coeffs;
}

// --- Plan Factories ---
template <typename T>
[[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<T>>>
StubOmniDSPImpl::create_fft_plan(size_t length) const {
  try {
    auto pimpl_backend = std::make_unique<StubFFTPlanImpl<T>>(length);
    // Construct public Plan object using private constructor (OmniDSP is
    // friend)
    FFTPlan<T>* plan_ptr = new FFTPlan<T>(std::move(pimpl_backend));
    return std::unique_ptr<FFTPlan<T>>(plan_ptr);
  } catch (const std::exception& e) {
    std::cerr << "Error creating Stub FFTPlanImpl: " << e.what() << std::endl;
    return std::unexpected(Status::Failure);  // Or AllocationError
  } catch (...) {
    return std::unexpected(Status::Failure);
  }
}

template <typename T>
[[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<T>>>
StubOmniDSPImpl::create_rfft_plan(size_t length) const {
  try {
    auto pimpl_backend = std::make_unique<StubRFFTPlanImpl<T>>(length);
    RFFTPlan<T>* plan_ptr = new RFFTPlan<T>(std::move(pimpl_backend));
    return std::unique_ptr<RFFTPlan<T>>(plan_ptr);
  } catch (const std::exception& e) {
    std::cerr << "Error creating Stub RFFTPlanImpl: " << e.what() << std::endl;
    return std::unexpected(Status::Failure);
  } catch (...) {
    return std::unexpected(Status::Failure);
  }
}

template <typename T>
[[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<T>>>
StubOmniDSPImpl::create_cqt_plan(const OmniDSP* owner, RealT<T> sample_rate,
                                 RealT<T> min_freq, RealT<T> max_freq,
                                 int bins_per_octave) const {
  // CQTPlan constructor needs the owner to create internal plans using the
  // *stub* backend
  try {
    CQTPlan<T>* plan_ptr =
        new CQTPlan<T>(owner, sample_rate, min_freq, max_freq, bins_per_octave);
    return std::unique_ptr<CQTPlan<T>>(plan_ptr);
  } catch (const std::exception& e) {
    std::cerr << "Error creating Stub CQTPlan: " << e.what() << std::endl;
    return std::unexpected(Status::Failure);
  } catch (...) {
    return std::unexpected(Status::Failure);
  }
}

template <typename T>
[[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<T>>>
StubOmniDSPImpl::create_resample_plan(double input_rate, double output_rate,
                                      size_t max_input_size) const {
  try {
    auto pimpl_backend = std::make_unique<StubResamplePlanImpl<T>>(
        input_rate, output_rate, max_input_size);
    ResamplePlan<T>* plan_ptr = new ResamplePlan<T>(std::move(pimpl_backend));
    return std::unique_ptr<ResamplePlan<T>>(plan_ptr);
  } catch (const std::exception& e) {
    std::cerr << "Error creating Stub ResamplePlanImpl: " << e.what()
              << std::endl;
    return std::unexpected(Status::Failure);
  } catch (...) {
    return std::unexpected(Status::Failure);
  }
}

template <typename T>
[[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
StubOmniDSPImpl::create_convolution_plan(const std::vector<T>& kernel,
                                         ConvolutionMode mode) const {
  try {
    auto pimpl_backend =
        std::make_unique<StubConvolutionPlanImpl<T>>(kernel, mode);
    ConvolutionPlan<T>* plan_ptr =
        new ConvolutionPlan<T>(std::move(pimpl_backend));
    return std::unique_ptr<ConvolutionPlan<T>>(plan_ptr);
  } catch (const std::exception& e) {
    std::cerr << "Error creating Stub ConvolutionPlanImpl: " << e.what()
              << std::endl;
    return std::unexpected(Status::Failure);
  } catch (...) {
    return std::unexpected(Status::Failure);
  }
}

template <typename T>
[[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
StubOmniDSPImpl::create_correlation_plan(const std::vector<T>& kernel,
                                         ConvolutionMode mode) const {
  try {
    auto pimpl_backend =
        std::make_unique<StubCorrelationPlanImpl<T>>(kernel, mode);
    CorrelationPlan<T>* plan_ptr =
        new CorrelationPlan<T>(std::move(pimpl_backend));
    return std::unique_ptr<CorrelationPlan<T>>(plan_ptr);
  } catch (const std::exception& e) {
    std::cerr << "Error creating Stub CorrelationPlanImpl: " << e.what()
              << std::endl;
    return std::unexpected(Status::Failure);
  } catch (...) {
    return std::unexpected(Status::Failure);
  }
}

// Add implementations for FilterPlan factory and filter design methods when
// added

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
// Instantiate templates for common types (float, double) for StubOmniDSPImpl
// methods

// Define types for brevity
using float_c = OmniDSP::ComplexT<float>;
using double_c = OmniDSP::ComplexT<double>;

// DSP Operations
template OmniExpected<std::vector<float>> StubOmniDSPImpl::convolve(
    const std::vector<float>&, const std::vector<float>&,
    ConvolutionMode) const;
template OmniExpected<std::vector<double>> StubOmniDSPImpl::convolve(
    const std::vector<double>&, const std::vector<double>&,
    ConvolutionMode) const;
template OmniExpected<std::vector<float_c>> StubOmniDSPImpl::convolve(
    const std::vector<float_c>&, const std::vector<float_c>&,
    ConvolutionMode) const;
template OmniExpected<std::vector<double_c>> StubOmniDSPImpl::convolve(
    const std::vector<double_c>&, const std::vector<double_c>&,
    ConvolutionMode) const;

template OmniExpected<std::vector<float>> StubOmniDSPImpl::correlate(
    const std::vector<float>&, const std::vector<float>&,
    ConvolutionMode) const;
template OmniExpected<std::vector<double>> StubOmniDSPImpl::correlate(
    const std::vector<double>&, const std::vector<double>&,
    ConvolutionMode) const;
template OmniExpected<std::vector<float_c>> StubOmniDSPImpl::correlate(
    const std::vector<float_c>&, const std::vector<float_c>&,
    ConvolutionMode) const;
template OmniExpected<std::vector<double_c>> StubOmniDSPImpl::correlate(
    const std::vector<double_c>&, const std::vector<double_c>&,
    ConvolutionMode) const;

// One-off FFTs
template OmniExpected<std::vector<float_c>> StubOmniDSPImpl::fft(
    const std::vector<float_c>&) const;
template OmniExpected<std::vector<double_c>> StubOmniDSPImpl::fft(
    const std::vector<double_c>&) const;
template OmniExpected<std::vector<float_c>> StubOmniDSPImpl::ifft(
    const std::vector<float_c>&) const;
template OmniExpected<std::vector<double_c>> StubOmniDSPImpl::ifft(
    const std::vector<double_c>&) const;
template OmniExpected<std::vector<float_c>> StubOmniDSPImpl::rfft(
    const std::vector<float>&) const;
template OmniExpected<std::vector<double_c>> StubOmniDSPImpl::rfft(
    const std::vector<double>&) const;
template OmniExpected<std::vector<float>> StubOmniDSPImpl::irfft(
    const std::vector<float_c>&, size_t) const;
template OmniExpected<std::vector<double>> StubOmniDSPImpl::irfft(
    const std::vector<double_c>&, size_t) const;

// Window Generation
template OmniExpected<std::vector<float>> StubOmniDSPImpl::bartlett_window(
    size_t) const;
template OmniExpected<std::vector<double>> StubOmniDSPImpl::bartlett_window(
    size_t) const;
template OmniExpected<std::vector<float>> StubOmniDSPImpl::blackman_window(
    size_t) const;
template OmniExpected<std::vector<double>> StubOmniDSPImpl::blackman_window(
    size_t) const;
template OmniExpected<std::vector<float>> StubOmniDSPImpl::flattop_window(
    size_t) const;
template OmniExpected<std::vector<double>> StubOmniDSPImpl::flattop_window(
    size_t) const;
template OmniExpected<std::vector<float>> StubOmniDSPImpl::gaussian_window(
    size_t, float) const;
template OmniExpected<std::vector<double>> StubOmniDSPImpl::gaussian_window(
    size_t, double) const;
template OmniExpected<std::vector<float>> StubOmniDSPImpl::hamming_window(
    size_t) const;
template OmniExpected<std::vector<double>> StubOmniDSPImpl::hamming_window(
    size_t) const;
template OmniExpected<std::vector<float>> StubOmniDSPImpl::hann_window(
    size_t) const;
template OmniExpected<std::vector<double>> StubOmniDSPImpl::hann_window(
    size_t) const;
template OmniExpected<std::vector<float>> StubOmniDSPImpl::kaiser_window(
    size_t, float) const;
template OmniExpected<std::vector<double>> StubOmniDSPImpl::kaiser_window(
    size_t, double) const;
template OmniExpected<std::vector<float>> StubOmniDSPImpl::rectangular_window(
    size_t) const;
template OmniExpected<std::vector<double>> StubOmniDSPImpl::rectangular_window(
    size_t) const;
template OmniExpected<std::vector<float>> StubOmniDSPImpl::triangular_window(
    size_t) const;
template OmniExpected<std::vector<double>> StubOmniDSPImpl::triangular_window(
    size_t) const;

// Plan Factories
template OmniExpected<std::unique_ptr<FFTPlan<float_c>>>
    StubOmniDSPImpl::create_fft_plan(size_t) const;
template OmniExpected<std::unique_ptr<FFTPlan<double_c>>>
    StubOmniDSPImpl::create_fft_plan(size_t) const;

template OmniExpected<std::unique_ptr<RFFTPlan<float>>>
    StubOmniDSPImpl::create_rfft_plan(size_t) const;
template OmniExpected<std::unique_ptr<RFFTPlan<double>>>
    StubOmniDSPImpl::create_rfft_plan(size_t) const;

template OmniExpected<std::unique_ptr<CQTPlan<float>>>
StubOmniDSPImpl::create_cqt_plan(const OmniDSP*, float, float, float,
                                 int) const;
template OmniExpected<std::unique_ptr<CQTPlan<double>>>
StubOmniDSPImpl::create_cqt_plan(const OmniDSP*, double, double, double,
                                 int) const;

template OmniExpected<std::unique_ptr<ResamplePlan<float>>>
StubOmniDSPImpl::create_resample_plan(double, double, size_t) const;
template OmniExpected<std::unique_ptr<ResamplePlan<double>>>
StubOmniDSPImpl::create_resample_plan(double, double, size_t) const;

template OmniExpected<std::unique_ptr<ConvolutionPlan<float>>>
StubOmniDSPImpl::create_convolution_plan(const std::vector<float>&,
                                         ConvolutionMode) const;
template OmniExpected<std::unique_ptr<ConvolutionPlan<double>>>
StubOmniDSPImpl::create_convolution_plan(const std::vector<double>&,
                                         ConvolutionMode) const;
template OmniExpected<std::unique_ptr<ConvolutionPlan<float_c>>>
StubOmniDSPImpl::create_convolution_plan(const std::vector<float_c>&,
                                         ConvolutionMode) const;
template OmniExpected<std::unique_ptr<ConvolutionPlan<double_c>>>
StubOmniDSPImpl::create_convolution_plan(const std::vector<double_c>&,
                                         ConvolutionMode) const;

template OmniExpected<std::unique_ptr<CorrelationPlan<float>>>
StubOmniDSPImpl::create_correlation_plan(const std::vector<float>&,
                                         ConvolutionMode) const;
template OmniExpected<std::unique_ptr<CorrelationPlan<double>>>
StubOmniDSPImpl::create_correlation_plan(const std::vector<double>&,
                                         ConvolutionMode) const;
template OmniExpected<std::unique_ptr<CorrelationPlan<float_c>>>
StubOmniDSPImpl::create_correlation_plan(const std::vector<float_c>&,
                                         ConvolutionMode) const;
template OmniExpected<std::unique_ptr<CorrelationPlan<double_c>>>
StubOmniDSPImpl::create_correlation_plan(const std::vector<double_c>&,
                                         ConvolutionMode) const;

}  // namespace backend
}  // namespace OmniDSP
