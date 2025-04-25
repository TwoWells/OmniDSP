/**
 * @file backend.h
 * @brief Defines the abstract base class interface for backend implementations.
 * @details Concrete backend implementations (DefaultBackend, AccelerateBackend,
 * OneMKLBackend) must inherit from the abstract base classes defined here
 * (AbstractBackend, *PlanImpl).
 */

#ifndef OMNIDSP_BACKEND_H
#define OMNIDSP_BACKEND_H

#include <complex>
#include <cstddef>  // For size_t
#include <memory>   // For std::unique_ptr
#include <span>     // For std::span (requires C++20)
#include <vector>

// Include core library types and specifications
#include "OmniDSP/convolution.h"  // Includes ConvolutionType
#include "OmniDSP/core_types.h"  // Includes Status, Backend, OmniExpected, F32, C32, F64, C64, Vec aliases etc.
#include "OmniDSP/filter.h"  // Includes FIRFilterSpec, IIRFilterSpec, SecondOrderSection
#include "OmniDSP/resample.h"  // Includes ResampleSpec
#include "OmniDSP/window.h"    // Includes WindowSpec, WindowType

// Forward declare public Plan classes from the OmniDSP namespace
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
  class OmniDSP;
}  // namespace OmniDSP

namespace OmniDSP {
  namespace backend {

    // Forward declare the main backend implementation base class
    class AbstractBackend;  // Renamed from OmniDSPImpl

    //--------------------------------------------------------------------------
    // Plan Implementation Interfaces (Abstract Base Classes)
    //--------------------------------------------------------------------------
    // FFTPlanImpl, RFFTPlanImpl, ResamplePlanImpl, ConvolutionPlanImpl,
    // CorrelationPlanImpl, FIRFilterPlanImpl, IIRFilterPlanImpl, CQTPlanImpl
    // declarations remain the same as before...

    /** @brief Interface for complex-to-complex FFT plan implementations. */
    template <typename T>  // T is complex type here (e.g., C32, C64)
    class FFTPlanImpl {
     public:
      virtual ~FFTPlanImpl() = default;
      virtual Status fft(std::span<const T> input, std::span<T> output) const
          = 0;
      virtual Status ifft(std::span<const T> input, std::span<T> output) const
          = 0;
      virtual size_t get_length() const = 0;
    };

    /** @brief Interface for real-to-complex/complex-to-real FFT plan
     * implementations. */
    template <typename T>  // T is real type here (e.g., F32, F64)
    class RFFTPlanImpl {
     public:
      virtual ~RFFTPlanImpl() = default;
      virtual Status rfft(
          std::span<const T> input, std::span<ComplexT<T>> output) const
          = 0;
      virtual Status irfft(
          std::span<const ComplexT<T>> input, std::span<T> output) const
          = 0;
      virtual size_t get_length() const = 0;
    };

    /** @brief Interface for resampling plan implementations. */
    template <typename T>  // T is real type here (F32, F64)
    class ResamplePlanImpl {
     public:
      virtual ~ResamplePlanImpl() = default;
      virtual Status execute(std::span<const T> input, std::span<T> output) = 0;
      virtual Status reset() = 0;
      virtual double get_input_rate() const = 0;
      virtual double get_output_rate() const = 0;
      virtual size_t get_output_length(size_t input_length) const = 0;
    };

    /** @brief Interface for convolution plan implementations. */
    template <typename T>  // T can be F32, F64, C32, C64
    class ConvolutionPlanImpl {
     public:
      virtual ~ConvolutionPlanImpl() = default;
      virtual Status execute(
          std::span<const T> input, std::span<T> output) const
          = 0;
      virtual size_t get_kernel_length() const = 0;
      virtual ConvolutionType get_type() const = 0;
      virtual size_t get_output_length(size_t input_length) const = 0;
    };

    /** @brief Interface for correlation plan implementations. */
    template <typename T>  // T can be F32, F64, C32, C64
    class CorrelationPlanImpl {
     public:
      virtual ~CorrelationPlanImpl() = default;
      virtual Status execute(
          std::span<const T> input, std::span<T> output) const
          = 0;
      virtual size_t get_template_length() const = 0;
      virtual ConvolutionType get_type() const = 0;
      virtual size_t get_output_length(size_t input_length) const = 0;
    };

    /** @brief Interface for FIR filter plan implementations. */
    template <typename T>  // T can be F32, F64, C32, C64
    class FIRFilterPlanImpl {
     public:
      virtual ~FIRFilterPlanImpl() = default;
      virtual Status execute(std::span<const T> input, std::span<T> output) = 0;
      virtual Status reset() = 0;
      virtual size_t get_order() const = 0;
      virtual size_t get_num_taps() const = 0;
    };

    /** @brief Interface for IIR filter plan implementations. */
    template <typename T>  // T is typically real (F32, F64)
    class IIRFilterPlanImpl {
     public:
      virtual ~IIRFilterPlanImpl() = default;
      virtual Status execute(std::span<const T> input, std::span<T> output) = 0;
      virtual Status reset() = 0;
      virtual size_t get_order() const = 0;
      virtual size_t get_num_sections() const = 0;
    };

    /** @brief Interface for Constant-Q Transform (CQT) plan implementations. */
    template <typename T>  // T is real type here (F32, F64)
    class CQTPlanImpl {
     public:
      virtual ~CQTPlanImpl() = default;
      virtual Status execute(
          std::span<const T> input, std::span<ComplexT<T>> output) const
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
     * implementations. Replaces OmniDSPImpl.
     * One-off operations are virtual (not pure) and should have default
     * implementations provided in DefaultBackend. Plan factories and design
     * functions are pure virtual.
     */
    class AbstractBackend {  // Renamed from OmniDSPImpl
     public:
      virtual ~AbstractBackend() = default;
      /** @brief Returns the identifier for this backend implementation. */
      virtual Backend get_backend() const = 0;  // Pure virtual

      // === Virtual methods corresponding to OmniDSP public API ===

      // --- DSP Operations (One-off) ---
      // These are NOT pure virtual; DefaultBackend provides implementations.
      // Backends override these ONLY if a direct optimized call exists.
      [[nodiscard]] virtual OmniExpected<F32Vec> convolve_f32(
          const F32Vec& input,
          const F32Vec& kernel,
          ConvolutionType mode) const;
      [[nodiscard]] virtual OmniExpected<F64Vec> convolve_f64(
          const F64Vec& input,
          const F64Vec& kernel,
          ConvolutionType mode) const;
      [[nodiscard]] virtual OmniExpected<C32Vec> convolve_c32(
          const C32Vec& input,
          const C32Vec& kernel,
          ConvolutionType mode) const;
      [[nodiscard]] virtual OmniExpected<C64Vec> convolve_c64(
          const C64Vec& input,
          const C64Vec& kernel,
          ConvolutionType mode) const;

      [[nodiscard]] virtual OmniExpected<F32Vec> correlate_f32(
          const F32Vec& input,
          const F32Vec& kernel,
          ConvolutionType mode) const;
      [[nodiscard]] virtual OmniExpected<F64Vec> correlate_f64(
          const F64Vec& input,
          const F64Vec& kernel,
          ConvolutionType mode) const;
      [[nodiscard]] virtual OmniExpected<C32Vec> correlate_c32(
          const C32Vec& input,
          const C32Vec& kernel,
          ConvolutionType mode) const;
      [[nodiscard]] virtual OmniExpected<C64Vec> correlate_c64(
          const C64Vec& input,
          const C64Vec& kernel,
          ConvolutionType mode) const;

      [[nodiscard]] virtual OmniExpected<C32Vec> fft_c32(
          const C32Vec& input) const;
      [[nodiscard]] virtual OmniExpected<C64Vec> fft_c64(
          const C64Vec& input) const;

      [[nodiscard]] virtual OmniExpected<C32Vec> ifft_c32(
          const C32Vec& input) const;
      [[nodiscard]] virtual OmniExpected<C64Vec> ifft_c64(
          const C64Vec& input) const;

      [[nodiscard]] virtual OmniExpected<C32Vec> rfft_f32(
          const F32Vec& input) const;
      [[nodiscard]] virtual OmniExpected<C64Vec> rfft_f64(
          const F64Vec& input) const;

      [[nodiscard]] virtual OmniExpected<F32Vec> irfft_c32(
          const C32Vec& input, size_t output_length) const;
      [[nodiscard]] virtual OmniExpected<F64Vec> irfft_c64(
          const C64Vec& input, size_t output_length) const;

      // --- Window Generation (One-off) ---
      // Keep these pure virtual as window generation is fundamental.
      [[nodiscard]] virtual Status bartlett_window_f32(
          size_t length, std::span<F32> output) const
          = 0;
      [[nodiscard]] virtual Status bartlett_window_f64(
          size_t length, std::span<F64> output) const
          = 0;
      [[nodiscard]] virtual Status blackman_window_f32(
          size_t length, std::span<F32> output) const
          = 0;
      [[nodiscard]] virtual Status blackman_window_f64(
          size_t length, std::span<F64> output) const
          = 0;
      [[nodiscard]] virtual Status flattop_window_f32(
          size_t length, std::span<F32> output) const
          = 0;
      [[nodiscard]] virtual Status flattop_window_f64(
          size_t length, std::span<F64> output) const
          = 0;
      [[nodiscard]] virtual Status gaussian_window_f32(
          size_t length, double stddev, std::span<F32> output) const
          = 0;
      [[nodiscard]] virtual Status gaussian_window_f64(
          size_t length, double stddev, std::span<F64> output) const
          = 0;
      [[nodiscard]] virtual Status hamming_window_f32(
          size_t length, std::span<F32> output) const
          = 0;
      [[nodiscard]] virtual Status hamming_window_f64(
          size_t length, std::span<F64> output) const
          = 0;
      [[nodiscard]] virtual Status hann_window_f32(
          size_t length, std::span<F32> output) const
          = 0;
      [[nodiscard]] virtual Status hann_window_f64(
          size_t length, std::span<F64> output) const
          = 0;
      [[nodiscard]] virtual Status kaiser_window_f32(
          size_t length, double beta, std::span<F32> output) const
          = 0;
      [[nodiscard]] virtual Status kaiser_window_f64(
          size_t length, double beta, std::span<F64> output) const
          = 0;
      [[nodiscard]] virtual Status rectangular_window_f32(
          size_t length, std::span<F32> output) const
          = 0;
      [[nodiscard]] virtual Status rectangular_window_f64(
          size_t length, std::span<F64> output) const
          = 0;
      [[nodiscard]] virtual Status triangular_window_f32(
          size_t length, std::span<F32> output) const
          = 0;
      [[nodiscard]] virtual Status triangular_window_f64(
          size_t length, std::span<F64> output) const
          = 0;

      // --- Plan Factories ---
      // MUST be implemented by each backend. Pure virtual.
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<FFTPlan<C32>>>
      create_fft_plan_c32(size_t length) const = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<FFTPlan<C64>>>
      create_fft_plan_c64(size_t length) const = 0;

      [[nodiscard]] virtual OmniExpected<std::unique_ptr<RFFTPlan<F32>>>
      create_rfft_plan_f32(size_t length) const = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<RFFTPlan<F64>>>
      create_rfft_plan_f64(size_t length) const = 0;

      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CQTPlan<F32>>>
      create_cqt_plan_f32(
          F32 sample_rate,
          F32 min_freq,
          F32 max_freq,
          int bins_per_octave,
          const WindowSpec<F32>& window_spec) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CQTPlan<F64>>>
      create_cqt_plan_f64(
          F64 sample_rate,
          F64 min_freq,
          F64 max_freq,
          int bins_per_octave,
          const WindowSpec<F64>& window_spec) const
          = 0;

      [[nodiscard]] virtual OmniExpected<std::unique_ptr<ResamplePlan<F32>>>
      create_resample_plan_f32(const ResampleSpec& spec) const = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<ResamplePlan<F64>>>
      create_resample_plan_f64(const ResampleSpec& spec) const = 0;

      [[nodiscard]] virtual OmniExpected<std::unique_ptr<ConvolutionPlan<F32>>>
      create_convolution_plan_f32(
          const F32Vec& kernel, ConvolutionType mode) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<ConvolutionPlan<F64>>>
      create_convolution_plan_f64(
          const F64Vec& kernel, ConvolutionType mode) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<ConvolutionPlan<C32>>>
      create_convolution_plan_c32(
          const C32Vec& kernel, ConvolutionType mode) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<ConvolutionPlan<C64>>>
      create_convolution_plan_c64(
          const C64Vec& kernel, ConvolutionType mode) const
          = 0;

      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CorrelationPlan<F32>>>
      create_correlation_plan_f32(
          const F32Vec& kernel, ConvolutionType mode) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CorrelationPlan<F64>>>
      create_correlation_plan_f64(
          const F64Vec& kernel, ConvolutionType mode) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CorrelationPlan<C32>>>
      create_correlation_plan_c32(
          const C32Vec& kernel, ConvolutionType mode) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CorrelationPlan<C64>>>
      create_correlation_plan_c64(
          const C64Vec& kernel, ConvolutionType mode) const
          = 0;

      [[nodiscard]] virtual OmniExpected<std::unique_ptr<FIRFilterPlan<F32>>>
      create_fir_filter_plan_f32(const F32Vec& coefficients) const = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<FIRFilterPlan<F64>>>
      create_fir_filter_plan_f64(const F64Vec& coefficients) const = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<FIRFilterPlan<C32>>>
      create_fir_filter_plan_c32(const C32Vec& coefficients) const = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<FIRFilterPlan<C64>>>
      create_fir_filter_plan_c64(const C64Vec& coefficients) const = 0;

      [[nodiscard]] virtual OmniExpected<std::unique_ptr<IIRFilterPlan<F32>>>
      create_iir_filter_plan_f32(
          const std::vector<SecondOrderSection<F32>>& sos_coefficients) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<IIRFilterPlan<F64>>>
      create_iir_filter_plan_f64(
          const std::vector<SecondOrderSection<F64>>& sos_coefficients) const
          = 0;

      // --- Filter Design ---
      // MUST be implemented by each backend. Pure virtual.
      [[nodiscard]] virtual OmniExpected<F32Vec> design_fir_filter_f32(
          const FIRFilterSpec<F32>& spec) const
          = 0;
      [[nodiscard]] virtual OmniExpected<F64Vec> design_fir_filter_f64(
          const FIRFilterSpec<F64>& spec) const
          = 0;

      [[nodiscard]] virtual OmniExpected<std::vector<SecondOrderSection<F32>>>
      design_iir_filter_f32(const IIRFilterSpec<F32>& spec) const = 0;
      [[nodiscard]] virtual OmniExpected<std::vector<SecondOrderSection<F64>>>
      design_iir_filter_f64(const IIRFilterSpec<F64>& spec) const = 0;

      // --- Internal Helpers ---
      // MUST be implemented by each backend. Pure virtual.
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CQTPlanImpl<F32>>>
      create_cqt_plan_impl_f32(
          F32 sample_rate,
          F32 min_freq,
          F32 max_freq,
          int bins_per_octave,
          const WindowSpec<F32>& window_spec) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CQTPlanImpl<F64>>>
      create_cqt_plan_impl_f64(
          F64 sample_rate,
          F64 min_freq,
          F64 max_freq,
          int bins_per_octave,
          const WindowSpec<F64>& window_spec) const
          = 0;

      // Helper to generate windows internally (e.g., for filter design)
      // Returns vector directly, unlike the public API version. Pure virtual.
      [[nodiscard]] virtual OmniExpected<F32Vec> generate_window_vec_f32(
          const WindowSpec<F32>& spec, size_t length) const
          = 0;
      [[nodiscard]] virtual OmniExpected<F64Vec> generate_window_vec_f64(
          const WindowSpec<F64>& spec, size_t length) const
          = 0;

    };  // class AbstractBackend

  }  // namespace backend
}  // namespace OmniDSP

#endif  // OMNIDSP_BACKEND_H
