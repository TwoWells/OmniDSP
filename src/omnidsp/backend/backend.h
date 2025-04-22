/**
 * @file backend.h
 * @brief Defines the abstract interfaces for backend implementations (Pimpl
 * pattern).
 * @details Concrete backend implementations (Stub, Accelerate, oneMKL) must
 * inherit from the abstract base classes defined here (OmniDSPImpl, *PlanImpl).
 */

#ifndef OMNIDSP_BACKEND_H
#define OMNIDSP_BACKEND_H

#include <complex>
#include <cstddef>  // For size_t
#include <memory>   // For std::unique_ptr
#include <span>     // For std::span (requires C++20)
#include <vector>

#include "OmniDSP/core_types.h"  // Core types, enums, OmniExpected

// Forward declare public Plan classes from the OmniDSP namespace
namespace OmniDSP {
template <typename T>
class FFTPlan;
template <typename T>
class RFFTPlan;
template <typename T>
class CQTPlan;  // Although CQTPlan doesn't use Pimpl, OmniDSPImpl might still
                // need it
template <typename T>
class ResamplePlan;
template <typename T>
class ConvolutionPlan;
template <typename T>
class CorrelationPlan;
// Forward declare FilterPlan when added
// template<typename T> class FilterPlan;
}  // namespace OmniDSP

namespace OmniDSP {
namespace backend {

//--------------------------------------------------------------------------
// Plan Implementation Interfaces (Abstract Base Classes)
//--------------------------------------------------------------------------

/**
 * @brief Abstract base class defining the interface for complex FFT Plan
 * implementations (Pimpl).
 * @tparam T The complex floating-point type (e.g., std::complex<float>).
 */
template <typename T>
class FFTPlanImpl {
 public:
  virtual ~FFTPlanImpl() = default;
  virtual Status fft(std::span<const T> input, std::span<T> output) const = 0;
  virtual Status ifft(std::span<const T> input, std::span<T> output) const = 0;
  virtual size_t get_length() const = 0;
};

/**
 * @brief Abstract base class defining the interface for real FFT Plan
 * implementations (Pimpl).
 * @tparam T The real floating-point type (e.g., float, double).
 */
template <typename T>
class RFFTPlanImpl {
 public:
  virtual ~RFFTPlanImpl() = default;
  virtual Status rfft(std::span<const RealT<T>> input,
                      std::span<ComplexT<T>> output) const = 0;
  virtual Status irfft(std::span<const ComplexT<T>> input,
                       std::span<RealT<T>> output) const = 0;
  virtual size_t get_length() const = 0;
};

/**
 * @brief Abstract base class defining the interface for Resample Plan
 * implementations (Pimpl).
 * @tparam T The floating-point type (e.g., float, double).
 */
template <typename T>
class ResamplePlanImpl {
 public:
  virtual ~ResamplePlanImpl() = default;
  virtual Status execute(std::span<const T> input,
                         std::span<T> output) const = 0;
  virtual double get_input_rate() const = 0;
  virtual double get_output_rate() const = 0;
  virtual size_t get_output_length(size_t input_length) const = 0;
};

/**
 * @brief Abstract base class defining the interface for Convolution Plan
 * implementations (Pimpl).
 * @tparam T The data type (e.g., float, std::complex<float>).
 */
template <typename T>
class ConvolutionPlanImpl {
 public:
  virtual ~ConvolutionPlanImpl() = default;
  virtual Status execute(std::span<const T> input,
                         std::span<T> output) const = 0;
  virtual size_t get_kernel_length() const = 0;
  virtual ConvolutionMode get_mode() const = 0;
  virtual size_t get_output_length(size_t input_length) const = 0;
};

/**
 * @brief Abstract base class defining the interface for Correlation Plan
 * implementations (Pimpl).
 * @tparam T The data type (e.g., float, std::complex<float>).
 */
template <typename T>
class CorrelationPlanImpl {
 public:
  virtual ~CorrelationPlanImpl() = default;
  virtual Status execute(std::span<const T> input,
                         std::span<T> output) const = 0;
  virtual size_t get_template_length() const = 0;
  virtual ConvolutionMode get_mode() const = 0;
  virtual size_t get_output_length(size_t input_length) const = 0;
};

// Add FilterPlanImpl when IIR filtering is added
// template<typename T>
// class FilterPlanImpl { ... };

//--------------------------------------------------------------------------
// Main Backend Implementation Interface (Abstract Base Class)
//--------------------------------------------------------------------------

/**
 * @brief Abstract base class defining the interface for backend implementations
 * (Pimpl).
 * @details Concrete backend implementations (Stub, Accelerate, oneMKL) inherit
 * from this class. Provides virtual methods corresponding to the public API of
 * the main OmniDSP class.
 */
class OmniDSPImpl {
 public:
  /** @brief Virtual destructor is required for base classes with virtual
   * functions. */
  virtual ~OmniDSPImpl() = default;

  /** @brief Gets the backend type associated with this implementation. */
  virtual Backend get_backend() const = 0;

  // === Virtual methods corresponding to OmniDSP public API ===

  // --- DSP Operations ---
  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> convolve(
      const std::vector<RealT<T>>& input, const std::vector<RealT<T>>& kernel,
      ConvolutionMode mode) const = 0;

  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<ComplexT<T>>> convolve(
      const std::vector<ComplexT<T>>& input,
      const std::vector<ComplexT<T>>& kernel, ConvolutionMode mode) const = 0;

  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> correlate(
      const std::vector<RealT<T>>& input, const std::vector<RealT<T>>& kernel,
      ConvolutionMode mode) const = 0;

  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<ComplexT<T>>> correlate(
      const std::vector<ComplexT<T>>& input,
      const std::vector<ComplexT<T>>& kernel, ConvolutionMode mode) const = 0;

  // --- One-off FFTs ---
  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<ComplexT<T>>> fft(
      const std::vector<ComplexT<T>>& input) const = 0;

  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<ComplexT<T>>> ifft(
      const std::vector<ComplexT<T>>& input) const = 0;

  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<ComplexT<T>>> rfft(
      const std::vector<RealT<T>>& input) const = 0;

  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> irfft(
      const std::vector<ComplexT<T>>& input, size_t output_length) const = 0;

  // --- Window Generation ---
  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> bartlett_window(
      size_t length) const = 0;
  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> blackman_window(
      size_t length) const = 0;
  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> flattop_window(
      size_t length) const = 0;
  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> gaussian_window(
      size_t length, RealT<T> stddev) const = 0;
  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> hamming_window(
      size_t length) const = 0;
  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> hann_window(
      size_t length) const = 0;
  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> kaiser_window(
      size_t length, RealT<T> beta) const = 0;
  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> rectangular_window(
      size_t length) const = 0;
  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> triangular_window(
      size_t length) const = 0;

  // --- Plan Factories ---
  // Note: These return the public Plan types wrapping the backend-specific Impl
  // types.
  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::unique_ptr<FFTPlan<T>>>
  create_fft_plan(size_t length) const = 0;

  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::unique_ptr<RFFTPlan<T>>>
  create_rfft_plan(size_t length) const = 0;

  // CQTPlan doesn't use Pimpl directly, but the factory logic might differ per
  // backend if backend-specific FFT/Resample plans influence CQT creation.
  // However, the simplest approach is likely a non-virtual helper in
  // OmniDSP::create or a single implementation here that calls the CQTPlan
  // constructor. Let's keep it virtual for now, allowing backend overrides if
  // needed.
  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::unique_ptr<CQTPlan<T>>>
  create_cqt_plan(const OmniDSP* owner,  // Pass owner for sub-plan creation
                  RealT<T> sample_rate, RealT<T> min_freq, RealT<T> max_freq,
                  int bins_per_octave) const = 0;

  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::unique_ptr<ResamplePlan<T>>>
  create_resample_plan(double input_rate, double output_rate,
                       size_t max_input_size) const = 0;

  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
  create_convolution_plan(const std::vector<T>& kernel,
                          ConvolutionMode mode) const = 0;  // Added

  template <typename T>
  [[nodiscard]] virtual OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
  create_correlation_plan(const std::vector<T>& kernel,
                          ConvolutionMode mode) const = 0;  // Added

  // Add virtual methods for FilterPlan factory when added to OmniDSP
  // template<typename T>
  // [[nodiscard]] virtual OmniExpected<std::unique_ptr<FilterPlan<T>>>
  // create_iir_filter_plan(...) const = 0;

  // Add virtual methods for filter design if they need backend specifics
  // template<typename T>
  // [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>>
  // design_fir_filter(...) const = 0; template<typename T>
  // [[nodiscard]] virtual OmniExpected<std::pair<...>> design_iir_filter(...)
  // const = 0;

};  // class OmniDSPImpl

}  // namespace backend
}  // namespace OmniDSP

#endif  // OMNIDSP_BACKEND_H
