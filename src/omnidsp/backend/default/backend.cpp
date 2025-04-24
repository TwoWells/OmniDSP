/**
 * @file backend.cpp (default)
 * @brief Implements the DefaultOmniDSPImpl class methods using standard C++.
 * @details This file provides the base implementations for DSP operations,
 * window generation, and plan factories. Optimized backends inherit from
 * DefaultOmniDSPImpl and override methods where applicable.
 */

#include "backend.h"  // Corresponding header for Default backend declarations

// Include headers for the public Plan classes (needed for factory return types
// and Pimpl access/construction)
#include "OmniDSP/convolution.h"
#include "OmniDSP/cqt.h"  // Include CQTPlan header
#include "OmniDSP/fft.h"
#include "OmniDSP/filter.h"  // Include FilterPlan header
#include "OmniDSP/omnidsp.h"  // Needed only if DefaultCQTPlanImpl needs OmniDSP* owner
#include "OmniDSP/resample.h"
#include "OmniDSP/window.h"

// Include standard library headers needed for default implementations
#include <algorithm>  // For std::reverse, std::copy, std::fill, std::max, std::min
#include <cmath>      // For sin, cos, exp, sqrt, abs, log2, ceil, floor etc.
#include <complex>
#include <iostream>  // For debug/error messages
#include <memory>    // For std::unique_ptr, std::make_unique
#include <numbers>
#include <numeric>    // For std::accumulate, std::gcd
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>     // For exception messages
#include <utility>    // For std::move
#include <vector>

// Include Boost Bessel function for Kaiser window (needed by default impl)
#include <boost/math/special_functions/bessel.hpp>

namespace OmniDSP {
  namespace backend {

    /**
     * @brief Helper function to check a condition and return a status.
     * @param condition The boolean condition to check.
     * @param error_status The Status to return if the condition is false.
     * @param error_message An optional error message to print to stderr if the
     * condition is false.
     * @return Status::Success if condition is true, otherwise error_status.
     */
    inline Status check_status(
        bool condition,
        Status error_status,
        const std::string& error_message = "")
    {
      if (!condition) {
        if (!error_message.empty()) {
          std::cerr << "Default Backend Error: " << error_message << std::endl;
        }
        return error_status;
      }
      return Status::Success;
    }

    //--------------------------------------------------------------------------
    // DefaultOmniDSPImpl Method Definitions
    //--------------------------------------------------------------------------

    /**
     * @brief Constructor for the Default backend implementation.
     */
    DefaultOmniDSPImpl::DefaultOmniDSPImpl()
    {
      // No specific setup needed for the default backend
      std::cout << "Default Backend Initialized."
                << std::endl;  // Debug message
    }

    /**
     * @brief Destructor for the Default backend implementation.
     */
    DefaultOmniDSPImpl::~DefaultOmniDSPImpl()
    {
      std::cout << "Default Backend Destroyed." << std::endl;  // Debug message
    }

    /**
     * @brief Gets the backend type identifier.
     * @return Backend::Default.
     */
    Backend DefaultOmniDSPImpl::get_backend() const { return Backend::Default; }

    // --- DSP Operations ---
    // (convolve, correlate implementations...)
    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    DefaultOmniDSPImpl::convolve(
        const std::vector<RealT<T>>& input,
        const std::vector<RealT<T>>& kernel,
        ConvolutionMode mode) const
    {
      try {
        DefaultConvolutionPlanImpl<RealT<T>> plan(kernel, mode);
        size_t output_len = plan.get_output_length(input.size());
        std::vector<RealT<T>> output(output_len);
        Status status = plan.execute(input, output);
        if (status != Status::Success) return std::unexpected(status);
        return output;
      }
      catch (const std::exception& e) {
        std::cerr << "Default convolve (real) error: " << e.what() << std::endl;
        return std::unexpected(Status::Failure);
      }
    }
    template <typename T> [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>>
    DefaultOmniDSPImpl::convolve(
        const std::vector<ComplexT<T>>& input,
        const std::vector<ComplexT<T>>& kernel,
        ConvolutionMode mode) const
    {
      try {
        DefaultConvolutionPlanImpl<ComplexT<T>> plan(kernel, mode);
        size_t output_len = plan.get_output_length(input.size());
        std::vector<ComplexT<T>> output(output_len);
        Status status = plan.execute(input, output);
        if (status != Status::Success) return std::unexpected(status);
        return output;
      }
      catch (const std::exception& e) {
        std::cerr << "Default convolve (complex) error: " << e.what()
                  << std::endl;
        return std::unexpected(Status::Failure);
      }
    }
    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    DefaultOmniDSPImpl::correlate(
        const std::vector<RealT<T>>& input,
        const std::vector<RealT<T>>& kernel,
        ConvolutionMode mode) const
    {
      try {
        DefaultCorrelationPlanImpl<RealT<T>> plan(kernel, mode);
        size_t output_len = plan.get_output_length(input.size());
        std::vector<RealT<T>> output(output_len);
        Status status = plan.execute(input, output);
        if (status != Status::Success) return std::unexpected(status);
        return output;
      }
      catch (const std::exception& e) {
        std::cerr << "Default correlate (real) error: " << e.what()
                  << std::endl;
        return std::unexpected(Status::Failure);
      }
    }
    template <typename T> [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>>
    DefaultOmniDSPImpl::correlate(
        const std::vector<ComplexT<T>>& input,
        const std::vector<ComplexT<T>>& kernel,
        ConvolutionMode mode) const
    {
      try {
        DefaultCorrelationPlanImpl<ComplexT<T>> plan(kernel, mode);
        size_t output_len = plan.get_output_length(input.size());
        std::vector<ComplexT<T>> output(output_len);
        Status status = plan.execute(input, output);
        if (status != Status::Success) return std::unexpected(status);
        return output;
      }
      catch (const std::exception& e) {
        std::cerr << "Default correlate (complex) error: " << e.what()
                  << std::endl;
        return std::unexpected(Status::Failure);
      }
    }

    // --- One-off FFTs ---
    // (fft, ifft, rfft, irfft implementations...)
    template <typename T> [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>>
    DefaultOmniDSPImpl::fft(const std::vector<ComplexT<T>>& input) const
    {
      try {
        DefaultFFTPlanImpl<ComplexT<T>> plan(input.size());
        std::vector<ComplexT<T>> output(input.size());
        Status status = plan.fft(input, output);
        if (status != Status::Success) return std::unexpected(status);
        return output;
      }
      catch (const std::exception& e) {
        std::cerr << "Default fft error: " << e.what() << std::endl;
        return std::unexpected(Status::Failure);
      }
    }
    template <typename T> [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>>
    DefaultOmniDSPImpl::ifft(const std::vector<ComplexT<T>>& input) const
    {
      try {
        DefaultFFTPlanImpl<ComplexT<T>> plan(input.size());
        std::vector<ComplexT<T>> output(input.size());
        Status status = plan.ifft(input, output);
        if (status != Status::Success) return std::unexpected(status);
        if (!output.empty()) {
          using RealType = typename T::value_type;
          RealType scale = static_cast<RealType>(1.0)
                           / static_cast<RealType>(output.size());
          for (auto& val : output) {
            val *= scale;
          }
        }
        return output;
      }
      catch (const std::exception& e) {
        std::cerr << "Default ifft error: " << e.what() << std::endl;
        return std::unexpected(Status::Failure);
      }
    }
    template <typename T> [[nodiscard]] OmniExpected<std::vector<ComplexT<T>>>
    DefaultOmniDSPImpl::rfft(const std::vector<RealT<T>>& input) const
    {
      try {
        DefaultRFFTPlanImpl<RealT<T>> plan(input.size());
        size_t output_len = (input.empty()) ? 0 : (input.size() / 2) + 1;
        std::vector<ComplexT<T>> output(output_len);
        Status status = plan.rfft(input, output);
        if (status != Status::Success) return std::unexpected(status);
        return output;
      }
      catch (const std::exception& e) {
        std::cerr << "Default rfft error: " << e.what() << std::endl;
        return std::unexpected(Status::Failure);
      }
    }
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<RealT<T>>> DefaultOmniDSPImpl::irfft(
        const std::vector<ComplexT<T>>& input, size_t output_length) const
    {
      try {
        if (output_length == 0) {
          if (input.empty())
            return std::vector<RealT<T>>();
          else
            return std::unexpected(Status::InvalidArgument);
        }
        size_t expected_input_len = (output_length / 2) + 1;
        if (input.size() != expected_input_len) {
          std::cerr << "Default irfft error: Input size (" << input.size()
                    << ") does not match expected size (" << expected_input_len
                    << ") for output length " << output_length << "."
                    << std::endl;
          return std::unexpected(Status::SizeMismatch);
        }
        DefaultRFFTPlanImpl<RealT<T>> plan(output_length);
        std::vector<RealT<T>> output(output_length);
        Status status = plan.irfft(input, output);
        if (status != Status::Success) return std::unexpected(status);
        if (!output.empty()) {
          RealT<T> scale = static_cast<RealT<T>>(1.0)
                           / static_cast<RealT<T>>(output.size());
          for (auto& val : output) {
            val *= scale;
          }
        }
        return output;
      }
      catch (const std::exception& e) {
        std::cerr << "Default irfft error: " << e.what() << std::endl;
        return std::unexpected(Status::Failure);
      }
    }

    // --- Window Generation ---
    // (bartlett_window, blackman_window, etc. implementations...)
    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    DefaultOmniDSPImpl::bartlett_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      std::vector<RealT<T>> coeffs(length);
      RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      if (N_minus_1 <= static_cast<RealT<T>>(0.0)) {
        if (length == 1) coeffs[0] = static_cast<RealT<T>>(1.0);
        return coeffs;
      }
      for (size_t n = 0; n < length; ++n) {
        coeffs[n] = static_cast<RealT<T>>(1.0)
                    - std::abs(
                        static_cast<RealT<T>>(2.0) * static_cast<RealT<T>>(n)
                            / N_minus_1
                        - static_cast<RealT<T>>(1.0));
      }
      return coeffs;
    }
    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    DefaultOmniDSPImpl::blackman_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      std::vector<RealT<T>> coeffs(length);
      const RealT<T> a0 = static_cast<RealT<T>>(0.42);
      const RealT<T> a1 = static_cast<RealT<T>>(0.5);
      const RealT<T> a2 = static_cast<RealT<T>>(0.08);
      const RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      const RealT<T> factor1
          = (N_minus_1 > static_cast<RealT<T>>(0.0))
                ? (static_cast<RealT<T>>(2.0 * std::numbers::pi) / N_minus_1)
                : static_cast<RealT<T>>(0.0);
      const RealT<T> factor2 = static_cast<RealT<T>>(2.0) * factor1;
      for (size_t n = 0; n < length; ++n) {
        RealT<T> n_T = static_cast<RealT<T>>(n);
        coeffs[n]
            = a0 - a1 * std::cos(factor1 * n_T) + a2 * std::cos(factor2 * n_T);
      }
      return coeffs;
    }
    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    DefaultOmniDSPImpl::flattop_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      std::vector<RealT<T>> coeffs(length);
      const RealT<T> a0 = static_cast<RealT<T>>(0.21557895);
      const RealT<T> a1 = static_cast<RealT<T>>(0.41663158);
      const RealT<T> a2 = static_cast<RealT<T>>(0.277263158);
      const RealT<T> a3 = static_cast<RealT<T>>(0.083578947);
      const RealT<T> a4 = static_cast<RealT<T>>(0.006947368);
      const RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      const RealT<T> factor
          = (N_minus_1 > static_cast<RealT<T>>(0.0))
                ? (static_cast<RealT<T>>(2.0 * std::numbers::pi) / N_minus_1)
                : static_cast<RealT<T>>(0.0);
      for (size_t n = 0; n < length; ++n) {
        RealT<T> n_T = static_cast<RealT<T>>(n);
        coeffs[n] = a0 - a1 * std::cos(factor * n_T)
                    + a2 * std::cos(static_cast<RealT<T>>(2.0) * factor * n_T)
                    - a3 * std::cos(static_cast<RealT<T>>(3.0) * factor * n_T)
                    + a4 * std::cos(static_cast<RealT<T>>(4.0) * factor * n_T);
      }
      return coeffs;
    }
    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    DefaultOmniDSPImpl::gaussian_window(size_t length, RealT<T> stddev) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      if (stddev <= static_cast<RealT<T>>(0.0)) {
        std::cerr << "Gaussian window standard deviation must be positive."
                  << std::endl;
        return std::unexpected(Status::InvalidArgument);
      }
      std::vector<RealT<T>> coeffs(length);
      RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      RealT<T> center = N_minus_1 / static_cast<RealT<T>>(2.0);
      RealT<T> sigma_term = stddev * center;
      if (sigma_term == static_cast<RealT<T>>(0.0)) {
        coeffs[0] = static_cast<RealT<T>>(1.0);
        return coeffs;
      }
      RealT<T> factor = static_cast<RealT<T>>(-0.5) / (sigma_term * sigma_term);
      for (size_t n = 0; n < length; ++n) {
        RealT<T> exponent = static_cast<RealT<T>>(n) - center;
        coeffs[n] = std::exp(factor * exponent * exponent);
      }
      return coeffs;
    }
    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    DefaultOmniDSPImpl::hamming_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      std::vector<RealT<T>> coeffs(length);
      const RealT<T> a0 = static_cast<RealT<T>>(0.54);
      const RealT<T> a1 = static_cast<RealT<T>>(0.46);
      const RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      const RealT<T> factor
          = (N_minus_1 > static_cast<RealT<T>>(0.0))
                ? (static_cast<RealT<T>>(2.0 * std::numbers::pi) / N_minus_1)
                : static_cast<RealT<T>>(0.0);
      for (size_t n = 0; n < length; ++n) {
        coeffs[n] = a0 - a1 * std::cos(factor * static_cast<RealT<T>>(n));
      }
      return coeffs;
    }
    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    DefaultOmniDSPImpl::hann_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      std::vector<RealT<T>> coeffs(length);
      const RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      const RealT<T> factor
          = (N_minus_1 > static_cast<RealT<T>>(0.0))
                ? (static_cast<RealT<T>>(2.0 * std::numbers::pi) / N_minus_1)
                : static_cast<RealT<T>>(0.0);
      for (size_t n = 0; n < length; ++n) {
        coeffs[n] = static_cast<RealT<T>>(0.5)
                    * (static_cast<RealT<T>>(1.0)
                       - std::cos(factor * static_cast<RealT<T>>(n)));
      }
      return coeffs;
    }
    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    DefaultOmniDSPImpl::kaiser_window(size_t length, RealT<T> beta) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      if (beta < static_cast<RealT<T>>(0.0)) {
        std::cerr << "Kaiser window beta parameter must be non-negative."
                  << std::endl;
        return std::unexpected(Status::InvalidArgument);
      }
      std::vector<RealT<T>> coeffs(length);
      RealT<T> bessel_i0_beta
          = boost::math::cyl_bessel_i(static_cast<RealT<T>>(0.0), beta);
      if (bessel_i0_beta == static_cast<RealT<T>>(0.0)) {
        std::cerr << "Kaiser window denominator I0(beta) is zero." << std::endl;
        return std::unexpected(Status::Failure);
      }
      RealT<T> inv_denom = static_cast<RealT<T>>(1.0) / bessel_i0_beta;
      RealT<T> N_minus_1 = static_cast<RealT<T>>(length - 1);
      RealT<T> factor = (N_minus_1 > static_cast<RealT<T>>(0.0))
                            ? (static_cast<RealT<T>>(2.0) / N_minus_1)
                            : static_cast<RealT<T>>(0.0);
      for (size_t n = 0; n < length; ++n) {
        RealT<T> term
            = factor * static_cast<RealT<T>>(n) - static_cast<RealT<T>>(1.0);
        RealT<T> arg_sqrt = static_cast<RealT<T>>(1.0) - term * term;
        RealT<T> bessel_arg
            = beta * std::sqrt(std::max(static_cast<RealT<T>>(0.0), arg_sqrt));
        coeffs[n]
            = boost::math::cyl_bessel_i(static_cast<RealT<T>>(0.0), bessel_arg)
              * inv_denom;
      }
      return coeffs;
    }
    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    DefaultOmniDSPImpl::rectangular_window(size_t length) const
    {
      return std::vector<RealT<T>>(length, static_cast<RealT<T>>(1.0));
    }
    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    DefaultOmniDSPImpl::triangular_window(size_t length) const
    {
      if (length == 0) return std::vector<RealT<T>>();
      if (length == 1)
        return std::vector<RealT<T>>(1, static_cast<RealT<T>>(1.0));
      std::vector<RealT<T>> coeffs(length);
      RealT<T> L = static_cast<RealT<T>>(length);
      RealT<T> center
          = (L - static_cast<RealT<T>>(1.0)) / static_cast<RealT<T>>(2.0);
      RealT<T> norm = L / static_cast<RealT<T>>(2.0);
      if (norm == static_cast<RealT<T>>(0.0))
        return std::unexpected(Status::Failure);
      for (size_t n = 0; n < length; ++n) {
        coeffs[n] = static_cast<RealT<T>>(1.0)
                    - std::abs(static_cast<RealT<T>>(n) - center) / norm;
      }
      return coeffs;
    }

    // --- Plan Factories ---
    // (create_fft_plan, create_rfft_plan, create_resample_plan,
    // create_convolution_plan, create_correlation_plan implementations...)
    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<T>>>
    DefaultOmniDSPImpl::create_fft_plan(size_t length) const
    {
      try {
        auto pimpl_backend = std::make_unique<DefaultFFTPlanImpl<T>>(length);
        FFTPlan<T>* plan_ptr = new FFTPlan<T>(std::move(pimpl_backend));
        return std::unique_ptr<FFTPlan<T>>(plan_ptr);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Default FFTPlanImpl: " << e.what()
                  << std::endl;
        return std::unexpected(Status::Failure);
      }
      catch (...) {
        return std::unexpected(Status::Failure);
      }
    }
    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<T>>>
    DefaultOmniDSPImpl::create_rfft_plan(size_t length) const
    {
      try {
        auto pimpl_backend = std::make_unique<DefaultRFFTPlanImpl<T>>(length);
        RFFTPlan<T>* plan_ptr = new RFFTPlan<T>(std::move(pimpl_backend));
        return std::unique_ptr<RFFTPlan<T>>(plan_ptr);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Default RFFTPlanImpl: " << e.what()
                  << std::endl;
        return std::unexpected(Status::Failure);
      }
      catch (...) {
        return std::unexpected(Status::Failure);
      }
    }
    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<T>>>
    DefaultOmniDSPImpl::create_resample_plan(
        double input_rate, double output_rate, size_t max_input_size) const
    {
      try {
        auto pimpl_backend = std::make_unique<DefaultResamplePlanImpl<T>>(
            input_rate, output_rate, max_input_size);
        ResamplePlan<T>* plan_ptr
            = new ResamplePlan<T>(std::move(pimpl_backend));
        return std::unique_ptr<ResamplePlan<T>>(plan_ptr);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Default ResamplePlanImpl: " << e.what()
                  << std::endl;
        return std::unexpected(Status::Failure);
      }
      catch (...) {
        return std::unexpected(Status::Failure);
      }
    }
    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
    DefaultOmniDSPImpl::create_convolution_plan(
        const std::vector<T>& kernel, ConvolutionMode mode) const
    {
      try {
        auto pimpl_backend
            = std::make_unique<DefaultConvolutionPlanImpl<T>>(kernel, mode);
        ConvolutionPlan<T>* plan_ptr
            = new ConvolutionPlan<T>(std::move(pimpl_backend));
        return std::unique_ptr<ConvolutionPlan<T>>(plan_ptr);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Default ConvolutionPlanImpl: " << e.what()
                  << std::endl;
        return std::unexpected(Status::Failure);
      }
      catch (...) {
        return std::unexpected(Status::Failure);
      }
    }
    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
    DefaultOmniDSPImpl::create_correlation_plan(
        const std::vector<T>& kernel, ConvolutionMode mode) const
    {
      try {
        auto pimpl_backend
            = std::make_unique<DefaultCorrelationPlanImpl<T>>(kernel, mode);
        CorrelationPlan<T>* plan_ptr
            = new CorrelationPlan<T>(std::move(pimpl_backend));
        return std::unique_ptr<CorrelationPlan<T>>(plan_ptr);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Default CorrelationPlanImpl: " << e.what()
                  << std::endl;
        return std::unexpected(Status::Failure);
      }
      catch (...) {
        return std::unexpected(Status::Failure);
      }
    }

    // --- Filter Plan Factories --- Added

    /**
     * @brief Creates a default FIR filter plan implementation.
     * @tparam T The data type (real or complex).
     * @param coefficients The FIR filter coefficients vector.
     * @return An OmniExpected containing a unique_ptr to the public
     * FIRFilterPlan on success, or a Status error code.
     */
    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<T>>>
    DefaultOmniDSPImpl::create_fir_filter_plan(
        const std::vector<T>& coefficients) const
    {
      try {
        auto pimpl_backend
            = std::make_unique<DefaultFIRFilterPlanImpl<T>>(coefficients);
        // Requires FIRFilterPlan to have a constructor taking
        // unique_ptr<FIRFilterPlanImpl<T>> and OmniDSP (or DefaultOmniDSPImpl)
        // to be a friend.
        FIRFilterPlan<T>* plan_ptr
            = new FIRFilterPlan<T>(std::move(pimpl_backend));
        return std::unique_ptr<FIRFilterPlan<T>>(plan_ptr);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Default FIRFilterPlanImpl: " << e.what()
                  << std::endl;
        return std::unexpected(
            Status::Failure);  // Or AllocationError, InvalidArgument
      }
      catch (...) {
        return std::unexpected(Status::Failure);
      }
    }

    /**
     * @brief Creates a default IIR filter plan implementation.
     * @tparam T The data type (typically real: float or double).
     * @param sos_coefficients A vector of Second-Order Sections representing
     * the IIR filter.
     * @return An OmniExpected containing a unique_ptr to the public
     * IIRFilterPlan on success, or a Status error code.
     */
    template <typename T>
    [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<T>>>
    DefaultOmniDSPImpl::create_iir_filter_plan(
        const std::vector<SecondOrderSection<T>>& sos_coefficients) const
    {
      try {
        auto pimpl_backend
            = std::make_unique<DefaultIIRFilterPlanImpl<T>>(sos_coefficients);
        // Requires IIRFilterPlan to have a constructor taking
        // unique_ptr<IIRFilterPlanImpl<T>> and OmniDSP (or DefaultOmniDSPImpl)
        // to be a friend.
        IIRFilterPlan<T>* plan_ptr
            = new IIRFilterPlan<T>(std::move(pimpl_backend));
        return std::unique_ptr<IIRFilterPlan<T>>(plan_ptr);
      }
      catch (const std::exception& e) {
        std::cerr << "Error creating Default IIRFilterPlanImpl: " << e.what()
                  << std::endl;
        return std::unexpected(
            Status::Failure);  // Or AllocationError, InvalidArgument
      }
      catch (...) {
        return std::unexpected(Status::Failure);
      }
    }

    // --- Filter Design Methods ---

    /**
     * @brief Designs an FIR filter using the windowed-sinc method.
     * @tparam T Real floating-point type (float or double).
     * @param spec The FIR filter specification.
     * @return An OmniExpected containing the filter coefficients on success, or
     * a Status error code.
     */
    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    DefaultOmniDSPImpl::design_fir_filter(const FIRFilterSpec<T>& spec) const
    {
      // 1. Validate Spec
      if (!spec.validate()) {
        return std::unexpected(Status::InvalidArgument);
      }
      size_t num_taps = spec.order + 1;
      if (num_taps == 0) {
        return std::unexpected(
            Status::InvalidArgument);  // Order must be >= 0 -> taps >= 1
      }

      // 2. Calculate normalized frequencies (0 to 0.5)
      RealT<T> fn1 = spec.cutoff1 / spec.sample_rate;
      std::optional<RealT<T>> fn2 = std::nullopt;
      if (spec.cutoff2.has_value()) {
        fn2 = spec.cutoff2.value() / spec.sample_rate;
      }

      // 3. Create ideal impulse response (centered around (num_taps-1)/2)
      std::vector<RealT<T>> ideal_coeffs(num_taps);
      RealT<T> center = static_cast<RealT<T>>(spec.order)
                        / static_cast<RealT<T>>(2.0);  // Center index

      for (size_t n = 0; n < num_taps; ++n) {
        RealT<T> m = static_cast<RealT<T>>(n) - center;
        RealT<T> sinc_val
            = (m == static_cast<RealT<T>>(0.0))
                  ? static_cast<RealT<T>>(1.0)
                  : std::sin(
                        static_cast<RealT<T>>(2.0 * std::numbers::pi) * fn1 * m)
                        / (static_cast<RealT<T>>(std::numbers::pi) * m);

        switch (spec.type) {
          case FilterType::Lowpass:
            ideal_coeffs[n] = static_cast<RealT<T>>(2.0) * fn1 * sinc_val;
            break;
          case FilterType::Highpass:
            ideal_coeffs[n]
                = ((m == static_cast<RealT<T>>(0.0))
                       ? static_cast<RealT<T>>(1.0)
                       : std::sin(static_cast<RealT<T>>(std::numbers::pi) * m)
                             / (static_cast<RealT<T>>(std::numbers::pi)
                                * m))  // Sinc(n) for delta
                  - static_cast<RealT<T>>(2.0) * fn1 * sinc_val;
            break;
          case FilterType::Bandpass: {
            if (!fn2.has_value())
              return std::unexpected(
                  Status::InvalidArgument);  // Should be caught by validate
            RealT<T> sinc_val2
                = (m == static_cast<RealT<T>>(0.0))
                      ? static_cast<RealT<T>>(1.0)
                      : std::sin(
                            static_cast<RealT<T>>(2.0 * std::numbers::pi)
                            * fn2.value() * m)
                            / (static_cast<RealT<T>>(std::numbers::pi) * m);
            ideal_coeffs[n]
                = static_cast<RealT<T>>(2.0) * fn2.value() * sinc_val2
                  - static_cast<RealT<T>>(2.0) * fn1 * sinc_val;
            break;
          }
          case FilterType::Bandstop: {
            if (!fn2.has_value())
              return std::unexpected(Status::InvalidArgument);
            RealT<T> sinc_val2
                = (m == static_cast<RealT<T>>(0.0))
                      ? static_cast<RealT<T>>(1.0)
                      : std::sin(
                            static_cast<RealT<T>>(2.0 * std::numbers::pi)
                            * fn2.value() * m)
                            / (static_cast<RealT<T>>(std::numbers::pi) * m);
            ideal_coeffs[n]
                = ((m == static_cast<RealT<T>>(0.0))
                       ? static_cast<RealT<T>>(1.0)
                       : std::sin(static_cast<RealT<T>>(std::numbers::pi) * m)
                             / (static_cast<RealT<T>>(std::numbers::pi)
                                * m))  // Sinc(n) for delta
                  - (static_cast<RealT<T>>(2.0) * fn2.value() * sinc_val2
                     - static_cast<RealT<T>>(2.0) * fn1
                           * sinc_val);  // Subtract bandpass
            break;
          }
          default:
            return std::unexpected(
                Status::InvalidArgument);  // Unknown filter type
        }
      }

      // 4. Generate Window using the helper method
      auto window_result = generate_window(spec.window, num_taps);
      if (!window_result) {
        return std::unexpected(window_result.error());
      }
      const auto& window_coeffs = window_result.value();

      // 5. Apply window
      std::vector<RealT<T>> final_coeffs(num_taps);
      for (size_t n = 0; n < num_taps; ++n) {
        final_coeffs[n] = ideal_coeffs[n] * window_coeffs[n];
      }

      // 6. Normalize (optional, but common for lowpass/bandpass gain)
      // For lowpass, normalize sum to 1 (DC gain)
      // For bandpass, normalize gain at center frequency? More complex.
      // Let's normalize lowpass for now.
      if (spec.type == FilterType::Lowpass) {
        RealT<T> sum = std::accumulate(
            final_coeffs.begin(),
            final_coeffs.end(),
            static_cast<RealT<T>>(0.0));
        if (std::abs(sum) > static_cast<RealT<T>>(
                1e-9)) {  // Avoid division by zero/small number
          RealT<T> inv_sum = static_cast<RealT<T>>(1.0) / sum;
          for (auto& coeff : final_coeffs) {
            coeff *= inv_sum;
          }
        }
      }
      // TODO: Consider normalization for other filter types if needed.

      return final_coeffs;
    }

    /**
     * @brief Designs an IIR filter (Placeholder - Not Implemented).
     * @tparam T Real floating-point type (float or double).
     * @param spec The IIR filter specification.
     * @return Status::NotImplemented.
     */
    template <typename T>
    [[nodiscard]] OmniExpected<std::vector<SecondOrderSection<T>>>
    DefaultOmniDSPImpl::design_iir_filter(const IIRFilterSpec<T>& spec) const
    {
      // TODO: Implement IIR filter design algorithms (Butterworth, Chebyshev
      // I/II, Elliptic) This involves complex steps like:
      // 1. Pre-warping frequencies for bilinear transform.
      // 2. Designing analog prototype filter (lowpass).
      // 3. Transforming prototype to target type (lowpass, highpass, bandpass,
      // bandstop).
      // 4. Applying bilinear transform to get digital filter ZPK (zeros, poles,
      // gain).
      // 5. Converting ZPK to SOS (Second-Order Sections) for stability.
      std::cerr << "Default IIR filter design not yet implemented."
                << std::endl;
      return std::unexpected(Status::NotImplemented);
    }

    /**
     * @brief Generates window coefficients based on the specification.
     * @details Calls the appropriate specific window generation method of this
     * class.
     * @tparam T Real floating-point type.
     * @param spec The window specification.
     * @param length The desired window length.
     * @return An OmniExpected containing the window coefficients on success, or
     * a Status error code.
     */
    template <typename T> [[nodiscard]] OmniExpected<std::vector<RealT<T>>>
    DefaultOmniDSPImpl::generate_window(
        const WindowSpec<T>& spec, size_t length) const
    {
      // Dispatch to the specific window generation method based on
      // spec.get_type()
      switch (spec.get_type()) {
        case WindowType::Bartlett:
          return bartlett_window<T>(length);
        case WindowType::Blackman:
          return blackman_window<T>(length);
        case WindowType::Flattop:
          return flattop_window<T>(length);
        case WindowType::Hamming:
          return hamming_window<T>(length);
        case WindowType::Hann:
          return hann_window<T>(length);
        case WindowType::Rectangular:
          return rectangular_window<T>(length);
        case WindowType::Triangular:
          return triangular_window<T>(length);
        case WindowType::Gaussian:
          if (!spec.get_stddev().has_value())
            return std::unexpected(Status::InvalidArgument);
          return gaussian_window<T>(length, spec.get_stddev().value());
        case WindowType::Kaiser:
          if (!spec.get_beta().has_value())
            return std::unexpected(Status::InvalidArgument);
          return kaiser_window<T>(length, spec.get_beta().value());
        default:
          return std::unexpected(
              Status::InvalidArgument);  // Unknown window type
      }
    }

    //--------------------------------------------------------------------------
    // Explicit Template Instantiations
    //--------------------------------------------------------------------------
    // Instantiate templates for common types (float, double) for
    // DefaultOmniDSPImpl methods

    // Define types for brevity
    using float_c = OmniDSP::ComplexT<float>;
    using double_c = OmniDSP::ComplexT<double>;

    // DSP Operations
    template OmniExpected<std::vector<float>> DefaultOmniDSPImpl::convolve(
        const std::vector<float>&,
        const std::vector<float>&,
        ConvolutionMode) const;
    template OmniExpected<std::vector<double>> DefaultOmniDSPImpl::convolve(
        const std::vector<double>&,
        const std::vector<double>&,
        ConvolutionMode) const;
    template OmniExpected<std::vector<float_c>> DefaultOmniDSPImpl::convolve(
        const std::vector<float_c>&,
        const std::vector<float_c>&,
        ConvolutionMode) const;
    template OmniExpected<std::vector<double_c>> DefaultOmniDSPImpl::convolve(
        const std::vector<double_c>&,
        const std::vector<double_c>&,
        ConvolutionMode) const;

    template OmniExpected<std::vector<float>> DefaultOmniDSPImpl::correlate(
        const std::vector<float>&,
        const std::vector<float>&,
        ConvolutionMode) const;
    template OmniExpected<std::vector<double>> DefaultOmniDSPImpl::correlate(
        const std::vector<double>&,
        const std::vector<double>&,
        ConvolutionMode) const;
    template OmniExpected<std::vector<float_c>> DefaultOmniDSPImpl::correlate(
        const std::vector<float_c>&,
        const std::vector<float_c>&,
        ConvolutionMode) const;
    template OmniExpected<std::vector<double_c>> DefaultOmniDSPImpl::correlate(
        const std::vector<double_c>&,
        const std::vector<double_c>&,
        ConvolutionMode) const;

    // One-off FFTs
    template OmniExpected<std::vector<float_c>> DefaultOmniDSPImpl::fft(
        const std::vector<float_c>&) const;
    template OmniExpected<std::vector<double_c>> DefaultOmniDSPImpl::fft(
        const std::vector<double_c>&) const;
    template OmniExpected<std::vector<float_c>> DefaultOmniDSPImpl::ifft(
        const std::vector<float_c>&) const;
    template OmniExpected<std::vector<double_c>> DefaultOmniDSPImpl::ifft(
        const std::vector<double_c>&) const;
    template OmniExpected<std::vector<float_c>> DefaultOmniDSPImpl::rfft(
        const std::vector<float>&) const;
    template OmniExpected<std::vector<double_c>> DefaultOmniDSPImpl::rfft(
        const std::vector<double>&) const;
    template OmniExpected<std::vector<float>> DefaultOmniDSPImpl::irfft(
        const std::vector<float_c>&, size_t) const;
    template OmniExpected<std::vector<double>> DefaultOmniDSPImpl::irfft(
        const std::vector<double_c>&, size_t) const;

    // Window Generation
    template OmniExpected<std::vector<float>>
        DefaultOmniDSPImpl::bartlett_window(size_t) const;
    template OmniExpected<std::vector<double>>
        DefaultOmniDSPImpl::bartlett_window(size_t) const;
    template OmniExpected<std::vector<float>>
        DefaultOmniDSPImpl::blackman_window(size_t) const;
    template OmniExpected<std::vector<double>>
        DefaultOmniDSPImpl::blackman_window(size_t) const;
    template OmniExpected<std::vector<float>>
        DefaultOmniDSPImpl::flattop_window(size_t) const;
    template OmniExpected<std::vector<double>>
        DefaultOmniDSPImpl::flattop_window(size_t) const;
    template OmniExpected<std::vector<float>>
    DefaultOmniDSPImpl::gaussian_window(size_t, float) const;
    template OmniExpected<std::vector<double>>
    DefaultOmniDSPImpl::gaussian_window(size_t, double) const;
    template OmniExpected<std::vector<float>>
        DefaultOmniDSPImpl::hamming_window(size_t) const;
    template OmniExpected<std::vector<double>>
        DefaultOmniDSPImpl::hamming_window(size_t) const;
    template OmniExpected<std::vector<float>> DefaultOmniDSPImpl::hann_window(
        size_t) const;
    template OmniExpected<std::vector<double>> DefaultOmniDSPImpl::hann_window(
        size_t) const;
    template OmniExpected<std::vector<float>> DefaultOmniDSPImpl::kaiser_window(
        size_t, float) const;
    template OmniExpected<std::vector<double>>
    DefaultOmniDSPImpl::kaiser_window(size_t, double) const;
    template OmniExpected<std::vector<float>>
        DefaultOmniDSPImpl::rectangular_window(size_t) const;
    template OmniExpected<std::vector<double>>
        DefaultOmniDSPImpl::rectangular_window(size_t) const;
    template OmniExpected<std::vector<float>>
        DefaultOmniDSPImpl::triangular_window(size_t) const;
    template OmniExpected<std::vector<double>>
        DefaultOmniDSPImpl::triangular_window(size_t) const;

    // Plan Factories
    template OmniExpected<std::unique_ptr<FFTPlan<float_c>>>
        DefaultOmniDSPImpl::create_fft_plan(size_t) const;
    template OmniExpected<std::unique_ptr<FFTPlan<double_c>>>
        DefaultOmniDSPImpl::create_fft_plan(size_t) const;

    template OmniExpected<std::unique_ptr<RFFTPlan<float>>>
        DefaultOmniDSPImpl::create_rfft_plan(size_t) const;
    template OmniExpected<std::unique_ptr<RFFTPlan<double>>>
        DefaultOmniDSPImpl::create_rfft_plan(size_t) const;

    // CQT Plan Factory
    template OmniExpected<std::unique_ptr<CQTPlan<float>>>
    DefaultOmniDSPImpl::create_cqt_plan(
        const OmniDSP*, float, float, float, int) const;
    template OmniExpected<std::unique_ptr<CQTPlan<double>>>
    DefaultOmniDSPImpl::create_cqt_plan(
        const OmniDSP*, double, double, double, int) const;

    // Resample Plan Factories
    template OmniExpected<std::unique_ptr<ResamplePlan<float>>>
    DefaultOmniDSPImpl::create_resample_plan(double, double, size_t) const;
    template OmniExpected<std::unique_ptr<ResamplePlan<double>>>
    DefaultOmniDSPImpl::create_resample_plan(double, double, size_t) const;

    // Convolution Plan Factories
    template OmniExpected<std::unique_ptr<ConvolutionPlan<float>>>
    DefaultOmniDSPImpl::create_convolution_plan(
        const std::vector<float>&, ConvolutionMode) const;
    template OmniExpected<std::unique_ptr<ConvolutionPlan<double>>>
    DefaultOmniDSPImpl::create_convolution_plan(
        const std::vector<double>&, ConvolutionMode) const;
    template OmniExpected<std::unique_ptr<ConvolutionPlan<float_c>>>
    DefaultOmniDSPImpl::create_convolution_plan(
        const std::vector<float_c>&, ConvolutionMode) const;
    template OmniExpected<std::unique_ptr<ConvolutionPlan<double_c>>>
    DefaultOmniDSPImpl::create_convolution_plan(
        const std::vector<double_c>&, ConvolutionMode) const;

    // Correlation Plan Factories
    template OmniExpected<std::unique_ptr<CorrelationPlan<float>>>
    DefaultOmniDSPImpl::create_correlation_plan(
        const std::vector<float>&, ConvolutionMode) const;
    template OmniExpected<std::unique_ptr<CorrelationPlan<double>>>
    DefaultOmniDSPImpl::create_correlation_plan(
        const std::vector<double>&, ConvolutionMode) const;
    template OmniExpected<std::unique_ptr<CorrelationPlan<float_c>>>
    DefaultOmniDSPImpl::create_correlation_plan(
        const std::vector<float_c>&, ConvolutionMode) const;
    template OmniExpected<std::unique_ptr<CorrelationPlan<double_c>>>
    DefaultOmniDSPImpl::create_correlation_plan(
        const std::vector<double_c>&, ConvolutionMode) const;

    // Filter Plan Factories
    template OmniExpected<std::unique_ptr<FIRFilterPlan<float>>>
    DefaultOmniDSPImpl::create_fir_filter_plan(const std::vector<float>&) const;
    template OmniExpected<std::unique_ptr<FIRFilterPlan<double>>>
    DefaultOmniDSPImpl::create_fir_filter_plan(
        const std::vector<double>&) const;
    template OmniExpected<std::unique_ptr<FIRFilterPlan<float_c>>>
    DefaultOmniDSPImpl::create_fir_filter_plan(
        const std::vector<float_c>&) const;
    template OmniExpected<std::unique_ptr<FIRFilterPlan<double_c>>>
    DefaultOmniDSPImpl::create_fir_filter_plan(
        const std::vector<double_c>&) const;

    template OmniExpected<std::unique_ptr<IIRFilterPlan<float>>>
    DefaultOmniDSPImpl::create_iir_filter_plan(
        const std::vector<SecondOrderSection<float>>&) const;
    template OmniExpected<std::unique_ptr<IIRFilterPlan<double>>>
    DefaultOmniDSPImpl::create_iir_filter_plan(
        const std::vector<SecondOrderSection<double>>&) const;

    // Filter Design Methods
    template OmniExpected<std::vector<float>>
    DefaultOmniDSPImpl::design_fir_filter(const FIRFilterSpec<float>&) const;
    template OmniExpected<std::vector<double>>
    DefaultOmniDSPImpl::design_fir_filter(const FIRFilterSpec<double>&) const;

    template OmniExpected<std::vector<SecondOrderSection<float>>>
    DefaultOmniDSPImpl::design_iir_filter(const IIRFilterSpec<float>&) const;
    template OmniExpected<std::vector<SecondOrderSection<double>>>
    DefaultOmniDSPImpl::design_iir_filter(const IIRFilterSpec<double>&) const;

    // Window Generation Helper
    template OmniExpected<std::vector<float>>
    DefaultOmniDSPImpl::generate_window(const WindowSpec<float>&, size_t) const;
    template OmniExpected<std::vector<double>>
    DefaultOmniDSPImpl::generate_window(
        const WindowSpec<double>&, size_t) const;

  }  // namespace backend
}  // namespace OmniDSP
