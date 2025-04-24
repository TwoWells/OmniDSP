/**
 * @file backend.h
 * @brief Defines the abstract interfaces for backend implementations (Pimpl
 * pattern).
 * @details Concrete backend implementations (Default, Accelerate, oneMKL) must
 * inherit from the abstract base classes defined here (OmniDSPImpl, *PlanImpl).
 * These classes define the contract that each backend must fulfill.
 */

#ifndef OMNIDSP_BACKEND_H
#define OMNIDSP_BACKEND_H

#include <complex>
#include <cstddef>  // For size_t
#include <memory>   // For std::unique_ptr
#include <span>     // For std::span (requires C++20)
#include <vector>

// Include core library types and specifications
#include "OmniDSP/convolution.h"
#include "OmniDSP/core_types.h"
#include "OmniDSP/filter.h"
#include "OmniDSP/resample.h"
#include "OmniDSP/window.h"

// Forward declare public Plan classes from the OmniDSP namespace
// These are the user-facing classes whose implementation is hidden.
namespace OmniDSP {
  template <typename T>
  class FFTPlan;
  template <typename T>
  class RFFTPlan;
  template <typename T>
  class CQTPlan;
  template <typename T>
  class ResamplePlan;
  template <typename T>
  class ConvolutionPlan;
  template <typename T>
  class CorrelationPlan;
  template <typename T>
  class FIRFilterPlan;
  template <typename T>
  class IIRFilterPlan;
  class OmniDSP;  // Forward declare the main user-facing OmniDSP class
}  // namespace OmniDSP

namespace OmniDSP {
  namespace backend {

    // Forward declare the main backend implementation base class
    class OmniDSPImpl;

    //--------------------------------------------------------------------------
    // Plan Implementation Interfaces (Abstract Base Classes)
    // These define the internal implementation interface for each Plan type.
    //--------------------------------------------------------------------------

    /** @brief Interface for complex-to-complex FFT plan implementations. */
    template <typename T>  // T is complex type here (e.g., std::complex<float>)
    class FFTPlanImpl {
     public:
      virtual ~FFTPlanImpl() = default;
      /** @brief Executes the forward complex FFT. */
      virtual Status fft(std::span<const T> input, std::span<T> output) const
          = 0;
      /** @brief Executes the inverse complex FFT. */
      virtual Status ifft(std::span<const T> input, std::span<T> output) const
          = 0;
      /** @brief Gets the FFT length. */
      virtual size_t get_length() const = 0;
    };

    /** @brief Interface for real-to-complex/complex-to-real FFT plan
     * implementations. */
    template <typename T>  // T is real type here (e.g., float)
    class RFFTPlanImpl {
     public:
      virtual ~RFFTPlanImpl() = default;
      /** @brief Executes the forward real-to-complex FFT. */
      virtual Status rfft(
          std::span<const RealT<T>> input, std::span<ComplexT<T>> output) const
          = 0;
      /** @brief Executes the inverse complex-to-real FFT. */
      virtual Status irfft(
          std::span<const ComplexT<T>> input, std::span<RealT<T>> output) const
          = 0;
      /** @brief Gets the logical FFT length (number of real points). */
      virtual size_t get_length() const = 0;
    };

    /** @brief Interface for resampling plan implementations. */
    template <typename T>  // T is real type here
    class ResamplePlanImpl {
     public:
      virtual ~ResamplePlanImpl() = default;

      /**
       * @brief Executes the resampling operation using internal state.
       * @param input A span representing the input signal buffer.
       * @param output A span representing the output buffer for the resampled
       * signal.
       * @return Status::Success on success, or an error code on failure.
       */
      virtual Status execute(
          std::span<const T> input,
          std::span<T> output)
          = 0;  // Non-const for state update

      /**
       * @brief Resets the internal state of the resampler (e.g., filter delay
       * lines, phase).
       * @return Status::Success on success, or an error code if resetting
       * fails.
       */
      virtual Status reset() = 0;

      // Getters remain const
      /** @brief Gets the input sample rate configured for this plan. */
      virtual double get_input_rate() const = 0;
      /** @brief Gets the output sample rate configured for this plan. */
      virtual double get_output_rate() const = 0;
      /** @brief Estimates the required output length for a given input length.
       */
      virtual size_t get_output_length(size_t input_length) const = 0;
    };

    /** @brief Interface for convolution plan implementations. */
    template <typename T>
    class ConvolutionPlanImpl {
     public:
      virtual ~ConvolutionPlanImpl() = default;
      /** @brief Executes the convolution operation. */
      virtual Status execute(
          std::span<const T> input, std::span<T> output) const
          = 0;
      /** @brief Gets the length of the kernel used by this plan. */
      virtual size_t get_kernel_length() const = 0;
      /** @brief Gets the convolution type (boundary handling mode). */
      virtual ConvolutionMode get_mode() const = 0;
      /** @brief Calculates the expected output length for a given input length.
       */
      virtual size_t get_output_length(size_t input_length) const = 0;
    };

    /** @brief Interface for correlation plan implementations. */
    template <typename T>
    class CorrelationPlanImpl {
     public:
      virtual ~CorrelationPlanImpl() = default;
      /** @brief Executes the correlation operation. */
      virtual Status execute(
          std::span<const T> input, std::span<T> output) const
          = 0;
      /** @brief Gets the length of the template (kernel) used by this plan. */
      virtual size_t get_template_length() const = 0;
      /** @brief Gets the correlation type (boundary handling mode). */
      virtual ConvolutionMode get_mode() const = 0;
      /** @brief Calculates the expected output length for a given input length.
       */
      virtual size_t get_output_length(size_t input_length) const = 0;
    };

    /** @brief Interface for FIR filter plan implementations. */
    template <typename T>
    class FIRFilterPlanImpl {
     public:
      virtual ~FIRFilterPlanImpl() = default;
      /** @brief Applies the FIR filter to an input signal segment, updating
       * state.
       */
      virtual Status execute(std::span<const T> input, std::span<T> output) = 0;
      /** @brief Resets the internal filter state (delay line). */
      virtual Status reset() = 0;
      /** @brief Gets the filter order (number of taps - 1). */
      virtual size_t get_order() const = 0;
      /** @brief Gets the number of filter taps (coefficients). */
      virtual size_t get_num_taps() const = 0;
    };

    /** @brief Interface for IIR filter plan implementations. */
    template <typename T>
    class IIRFilterPlanImpl {
     public:
      virtual ~IIRFilterPlanImpl() = default;
      /** @brief Applies the IIR filter to an input signal segment, updating
       * state.
       */
      virtual Status execute(std::span<const T> input, std::span<T> output) = 0;
      /** @brief Resets the internal filter state (delay elements). */
      virtual Status reset() = 0;
      /** @brief Gets the filter order. */
      virtual size_t get_order() const = 0;
      /** @brief Gets the number of second-order sections used. */
      virtual size_t get_num_sections() const = 0;
    };

    /** @brief Interface for Constant-Q Transform (CQT) plan implementations. */
    template <typename T>  // T is real type here
    class CQTPlanImpl {
     public:
      virtual ~CQTPlanImpl() = default;
      /** @brief Executes the Constant-Q Transform. */
      virtual Status execute(
          std::span<const RealT<T>> input, std::span<ComplexT<T>> output) const
          = 0;
      /** @brief Gets the number of frequency bins. */
      virtual size_t get_num_bins() const = 0;
      /** @brief Calculates the number of output time frames for a given input
       * length. */
      virtual size_t get_num_output_frames(size_t input_length) const = 0;
      /** @brief Gets the hop length between CQT frames (in samples). */
      virtual size_t get_hop_length() const = 0;
    };

    //--------------------------------------------------------------------------
    // Main Backend Implementation Interface (Abstract Base Class)
    //--------------------------------------------------------------------------

    /**
     * @brief Abstract base class defining the interface for backend
     * implementations (Pimpl). This class defines the complete set of
     * operations that a concrete backend (like Default, Accelerate, oneMKL)
     * must provide. It acts as the internal interface used by the public
     * OmniDSP class.
     */
    class OmniDSPImpl {
     public:
      virtual ~OmniDSPImpl() = default;
      /** @brief Returns the identifier for this backend implementation. */
      virtual Backend get_backend() const = 0;

      // === Virtual methods corresponding to OmniDSP public API ===

      // --- DSP Operations (One-off) ---
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

      // --- Window Generation (One-off) ---
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

      template <typename T>
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<FIRFilterPlan<T>>>
      create_fir_filter_plan(const std::vector<T>& coefficients) const = 0;

      template <typename T>  // T is typically real
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<IIRFilterPlan<T>>>
      create_iir_filter_plan(
          const std::vector<SecondOrderSection<T>>& sos_coefficients) const
          = 0;

      // --- Internal Helper for CQTPlanImpl Creation ---
      // Allows CQTPlan constructor to create a backend-specific CQTPlanImpl
      // without knowing the concrete backend type.
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
      design_fir_filter(const FIRFilterSpec<T>& spec) const = 0;

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
