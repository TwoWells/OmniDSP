/**
 * @file backend.h
 * @brief Defines the abstract interfaces for backend implementations (Pimpl
 * pattern).
 * @details Concrete backend implementations (Default, Accelerate, oneMKL) must
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
#include "OmniDSP/filter.h"  // Include FIRFilterSpec, IIRFilterSpec, SecondOrderSection
#include "OmniDSP/resample.h"  // Include ResampleSpec
#include "OmniDSP/window.h"    // Include WindowSpec

// Forward declare public Plan classes from the OmniDSP namespace
namespace OmniDSP {
  template <typename T> class FFTPlan;
  template <typename T> class RFFTPlan;
  template <typename T> class CQTPlan;
  template <typename T> class ResamplePlan;  // Forward declaration
  template <typename T> class ConvolutionPlan;
  template <typename T> class CorrelationPlan;
  template <typename T> class FIRFilterPlan;
  template <typename T> class IIRFilterPlan;
  class OmniDSP;
}  // namespace OmniDSP

namespace OmniDSP {
  namespace backend {

    //--------------------------------------------------------------------------
    // Plan Implementation Interfaces (Abstract Base Classes)
    //--------------------------------------------------------------------------

    // FFTPlanImpl definition...
    template <typename T>  // T is complex type here
    class FFTPlanImpl {
     public:
      virtual ~FFTPlanImpl() = default;
      virtual Status fft(std::span<const T> input, std::span<T> output) const
          = 0;
      virtual Status ifft(std::span<const T> input, std::span<T> output) const
          = 0;
      virtual size_t get_length() const = 0;
    };

    // RFFTPlanImpl definition...
    template <typename T>  // T is real type here
    class RFFTPlanImpl {
     public:
      virtual ~RFFTPlanImpl() = default;
      virtual Status rfft(
          std::span<const RealT<T>> input, std::span<ComplexT<T>> output) const
          = 0;
      virtual Status irfft(
          std::span<const ComplexT<T>> input, std::span<RealT<T>> output) const
          = 0;
      virtual size_t get_length() const = 0;
    };

    // ResamplePlanImpl definition...
    template <typename T> class ResamplePlanImpl {
     public:
      virtual ~ResamplePlanImpl() = default;
      virtual Status execute(
          std::span<const T> input,
          std::span<T> output) const
          = 0;  // Made const
      virtual double get_input_rate() const = 0;
      virtual double get_output_rate() const = 0;
      virtual size_t get_output_length(size_t input_length) const = 0;
      virtual Status reset() = 0;  // Added reset method
    };

    // ConvolutionPlanImpl definition...
    template <typename T> class ConvolutionPlanImpl {
     public:
      virtual ~ConvolutionPlanImpl() = default;
      virtual Status execute(
          std::span<const T> input, std::span<T> output) const
          = 0;
      virtual size_t get_kernel_length() const = 0;
      virtual ConvolutionMode get_mode() const = 0;
      virtual size_t get_output_length(size_t input_length) const = 0;
    };

    // CorrelationPlanImpl definition...
    template <typename T> class CorrelationPlanImpl {
     public:
      virtual ~CorrelationPlanImpl() = default;
      virtual Status execute(
          std::span<const T> input, std::span<T> output) const
          = 0;
      virtual size_t get_template_length() const = 0;
      virtual ConvolutionMode get_mode() const = 0;
      virtual size_t get_output_length(size_t input_length) const = 0;
    };

    // FIRFilterPlanImpl definition...
    template <typename T> class FIRFilterPlanImpl {
     public:
      virtual ~FIRFilterPlanImpl() = default;
      virtual Status execute(std::span<const T> input, std::span<T> output) = 0;
      virtual Status reset() = 0;
      virtual size_t get_order() const = 0;
      virtual size_t get_num_taps() const = 0;
    };

    // IIRFilterPlanImpl definition...
    template <typename T> class IIRFilterPlanImpl {
     public:
      virtual ~IIRFilterPlanImpl() = default;
      virtual Status execute(std::span<const T> input, std::span<T> output) = 0;
      virtual Status reset() = 0;
      virtual size_t get_order() const = 0;
      virtual size_t get_num_sections() const = 0;
    };

    // CQTPlanImpl definition...
    template <typename T>  // T is real type here
    class CQTPlanImpl {
     public:
      virtual ~CQTPlanImpl() = default;
      virtual Status execute(
          std::span<const RealT<T>> input, std::span<ComplexT<T>> output) const
          = 0;
      virtual size_t get_num_bins() const = 0;
      virtual size_t get_num_output_frames(size_t input_length) const = 0;
      virtual size_t get_hop_length() const = 0;
    };

    //--------------------------------------------------------------------------
    // Main Backend Implementation Interface (Abstract Base Class)
    //--------------------------------------------------------------------------

    /**
     * @brief Abstract base class defining the interface for backend
     * implementations (Pimpl).
     */
    class OmniDSPImpl {
     public:
      virtual ~OmniDSPImpl() = default;
      virtual Backend get_backend() const = 0;

      // === Virtual methods corresponding to OmniDSP public API ===
      // (Convolution, Correlation, FFTs, Windows...)
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> convolve(
          const std::vector<RealT<T>>& input,
          const std::vector<RealT<T>>& kernel,
          ConvolutionMode mode) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<ComplexT<T>>> convolve(
          const std::vector<ComplexT<T>>& input,
          const std::vector<ComplexT<T>>& kernel,
          ConvolutionMode mode) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> correlate(
          const std::vector<RealT<T>>& input,
          const std::vector<RealT<T>>& kernel,
          ConvolutionMode mode) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<ComplexT<T>>> correlate(
          const std::vector<ComplexT<T>>& input,
          const std::vector<ComplexT<T>>& kernel,
          ConvolutionMode mode) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<ComplexT<T>>> fft(
          const std::vector<ComplexT<T>>& input) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<ComplexT<T>>> ifft(
          const std::vector<ComplexT<T>>& input) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<ComplexT<T>>> rfft(
          const std::vector<RealT<T>>& input) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> irfft(
          const std::vector<ComplexT<T>>& input, size_t output_length) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> bartlett_window(
          size_t length) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> blackman_window(
          size_t length) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> flattop_window(
          size_t length) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> gaussian_window(
          size_t length, RealT<T> stddev) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> hamming_window(
          size_t length) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> hann_window(
          size_t length) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> kaiser_window(
          size_t length, RealT<T> beta) const
          = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>>
      rectangular_window(size_t length) const = 0;
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>>
      triangular_window(size_t length) const = 0;

      // --- Plan Factories ---
      template <typename T>  // T is complex type
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<FFTPlan<T>>>
      create_fft_plan(size_t length) const = 0;

      template <typename T>  // T is real type
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<RFFTPlan<T>>>
      create_rfft_plan(size_t length) const = 0;

      template <typename T>  // T is real type
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CQTPlan<T>>>
      create_cqt_plan(
          RealT<T> sample_rate,
          RealT<T> min_freq,
          RealT<T> max_freq,
          int bins_per_octave,
          const WindowSpec<T>& window_spec = WindowSpec<T>()) const
          = 0;

      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<ResamplePlan<T>>>
      create_resample_plan(const ResampleSpec& spec) const = 0;

      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
      create_convolution_plan(
          const std::vector<T>& kernel, ConvolutionMode mode) const
          = 0;

      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
      create_correlation_plan(
          const std::vector<T>& kernel, ConvolutionMode mode) const
          = 0;

      // Filter Plan Factories
      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<FIRFilterPlan<T>>>
      create_fir_filter_plan(const std::vector<T>& coefficients) const = 0;

      template <typename T>  // T is typically real
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<IIRFilterPlan<T>>>
      create_iir_filter_plan(
          const std::vector<SecondOrderSection<T>>& sos_coefficients) const
          = 0;

      // --- Internal Helper for CQTPlanImpl Creation ---
      template <typename T>  // T is real type
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CQTPlanImpl<T>>>
      create_cqt_plan_impl(
          RealT<T> sample_rate,
          RealT<T> min_freq,
          RealT<T> max_freq,
          int bins_per_octave,
          const WindowSpec<T>& window_spec) const
          = 0;

      // --- Internal Helper for Filter Design --- //
      /**
       * @brief Designs an FIR filter based on the specification.
       * @note This is intended for internal use (e.g., by ResamplePlanImpl).
       * @tparam T Real floating-point type.
       * @param spec The FIR filter specification.
       * @return An OmniExpected containing the filter coefficients on success,
       * or a Status error code.
       */
      template <typename T>  // T is real type
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>>
      design_fir_filter(const FIRFilterSpec<T>& spec) const
          = 0;  // <<< Already present

      /**
       * @brief Designs an IIR filter based on the specification.
       * @note This is intended for internal use.
       * @tparam T Real floating-point type.
       * @param spec The IIR filter specification.
       * @return An OmniExpected containing the SOS coefficients on success, or
       * a Status error code.
       */
      template <typename T>  // T is real type
      [[nodiscard]] virtual OmniExpected<std::vector<SecondOrderSection<T>>>
      design_iir_filter(const IIRFilterSpec<T>& spec) const = 0;

      /**
       * @brief Generates window coefficients based on the specification.
       * @note This is intended for internal use (e.g., by CQTPlanImpl,
       * design_fir_filter).
       * @tparam T Real floating-point type.
       * @param spec The window specification.
       * @param length The desired window length.
       * @return An OmniExpected containing the window coefficients on success,
       * or a Status error code.
       */
      template <typename T>  // T is real type
      [[nodiscard]] virtual OmniExpected<std::vector<RealT<T>>> generate_window(
          const WindowSpec<T>& spec, size_t length) const
          = 0;

    };  // class OmniDSPImpl

  }  // namespace backend
}  // namespace OmniDSP

#endif  // OMNIDSP_BACKEND_H
