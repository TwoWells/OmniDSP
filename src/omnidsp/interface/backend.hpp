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
// #include <OmniDSP/omnidsp.hpp>   // Avoid including the main header here if
// possible
#include <OmniDSP/convolution.hpp>  // Includes ConvolutionType, ConvolutionMethod
#include <OmniDSP/core_types.hpp>  // Includes Status, Backend, OmniExpected, F32, C32, F64, C64, Vec aliases, GetComplexT etc.
#include <OmniDSP/filter.hpp>  // Includes FIRFilterSpec, IIRFilterSpec, IIRFilterCoef
#include <OmniDSP/resample.hpp>  // Includes ResampleSpec
#include <OmniDSP/window.hpp>    // Includes WindowSpec, WindowType

// Forward declare public Plan classes from the OmniDSP namespace
// (These might not be strictly necessary here, but good practice)
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
  class OmniDSPImpl;  // Forward declare if needed by AbstractBackend (e.g.,
                      // friend?)
}  // namespace OmniDSP

namespace OmniDSP {
  namespace backend {

    // Forward declare the main backend implementation base class
    class AbstractBackend;

    //--------------------------------------------------------------------------
    // Plan Implementation Interfaces (Abstract Base Classes)
    //--------------------------------------------------------------------------
    // These define the common interface for *all* backend implementations of a
    // specific plan type.

    /** @brief Interface for complex-to-complex FFT plan implementations. */
    template <typename T>  // T is complex type here (e.g., C32, C64)
    class FFTPlanImpl {
     public:
      virtual ~FFTPlanImpl() = default;
      /** @brief Executes the forward FFT. */
      [[nodiscard]] virtual Status fft(
          std::span<const T> input, std::span<T> output) const
          = 0;
      /** @brief Executes the inverse FFT. */
      [[nodiscard]] virtual Status ifft(
          std::span<const T> input, std::span<T> output) const
          = 0;
      /** @brief Gets the length (size) of the FFT. */
      virtual size_t get_length() const = 0;
    };

    /** @brief Interface for real-to-complex/complex-to-real FFT plan
     * implementations. */
    template <typename T>  // T is real type here (e.g., F32, F64)
    class RFFTPlanImpl {
     public:
      using Complex = Detail::GetComplexT<T>;  // Use type trait from core_types
      virtual ~RFFTPlanImpl() = default;
      /** @brief Executes the forward real-to-complex FFT. */
      [[nodiscard]] virtual Status rfft(
          std::span<const T> input, std::span<Complex> output) const
          = 0;
      /** @brief Executes the inverse complex-to-real FFT. */
      [[nodiscard]] virtual Status irfft(
          std::span<const Complex> input, std::span<T> output) const
          = 0;
      /** @brief Gets the length (size) of the real data. */
      virtual size_t get_length() const = 0;
    };

    /** @brief Interface for resampling plan implementations. */
    template <typename T>  // T is real type here (F32, F64)
    class ResamplePlanImpl {
     public:
      virtual ~ResamplePlanImpl() = default;
      /** @brief Processes a chunk of input data. */
      [[nodiscard]] virtual Status execute(
          std::span<const T> input, std::span<T> output)
          = 0;  // Not const, stateful
      /** @brief Resets the internal state of the resampler. */
      [[nodiscard]] virtual Status reset() = 0;  // Not const, stateful
      /** @brief Gets the input sample rate. */
      virtual double get_input_rate() const = 0;
      /** @brief Gets the output sample rate. */
      virtual double get_output_rate() const = 0;
      /** @brief Calculates the expected output length for a given input length.
       */
      virtual size_t get_output_length(size_t input_length) const = 0;
    };

    /** @brief Interface for convolution plan implementations. */
    template <typename T>  // T can be F32, F64, C32, C64
    class ConvolutionPlanImpl {
     public:
      virtual ~ConvolutionPlanImpl() = default;
      /** @brief Executes the convolution operation. */
      [[nodiscard]] virtual Status execute(
          std::span<const T> input, std::span<T> output) const
          = 0;
      /** @brief Gets the length of the convolution kernel. */
      virtual size_t get_kernel_length() const = 0;
      /** @brief Gets the convolution type (boundary handling). */
      virtual ConvolutionType get_type() const = 0;
      /** @brief Calculates the expected output length for a given input length.
       */
      virtual size_t get_output_length(size_t input_length) const = 0;
      /** @brief Gets the convolution kernel used by the plan. */
      virtual std::span<const T> get_kernel() const = 0;
      /** @brief Gets the convolution method hint used when creating this plan.
       */
      virtual ConvolutionMethod get_method() const = 0;
    };

    /** @brief Interface for correlation plan implementations. */
    template <typename T>  // T can be F32, F64, C32, C64
    class CorrelationPlanImpl {
     public:
      virtual ~CorrelationPlanImpl() = default;
      /** @brief Executes the correlation operation. */
      [[nodiscard]] virtual Status execute(
          std::span<const T> input, std::span<T> output) const
          = 0;
      /** @brief Gets the length of the correlation template. */
      virtual size_t get_template_length() const = 0;
      /** @brief Gets the correlation type (boundary handling). */
      virtual ConvolutionType get_type() const = 0;
      /** @brief Calculates the expected output length for a given input length.
       */
      virtual size_t get_output_length(size_t input_length) const = 0;
      /** @brief Gets the correlation template used by the plan. */
      virtual std::span<const T> get_template() const = 0;
      /** @brief Gets the correlation method hint used when creating this plan.
       */
      virtual ConvolutionMethod get_method() const = 0;
    };

    /** @brief Interface for FIR filter plan implementations. */
    template <typename T>  // T can be F32, F64, C32, C64
    class FIRFilterPlanImpl {
     public:
      virtual ~FIRFilterPlanImpl() = default;
      /** @brief Processes a chunk of input data. */
      [[nodiscard]] virtual Status execute(
          std::span<const T> input, std::span<T> output)
          = 0;  // Not const, stateful
      /** @brief Resets the internal state (delay line) of the filter. */
      [[nodiscard]] virtual Status reset() = 0;  // Not const, stateful
      /** @brief Gets the filter order (length - 1). */
      virtual size_t get_order() const = 0;
      /** @brief Gets the number of filter coefficients (taps). */
      virtual size_t get_num_taps() const = 0;
      // virtual std::span<const T> get_coefficients() const = 0; // Consider
      // adding?
    };

    /** @brief Interface for IIR filter plan implementations. */
    template <typename T>  // T is typically real (F32, F64)
    class IIRFilterPlanImpl {
     public:
      virtual ~IIRFilterPlanImpl() = default;
      /** @brief Processes a chunk of input data. */
      [[nodiscard]] virtual Status execute(
          std::span<const T> input, std::span<T> output)
          = 0;  // Not const, stateful
      /** @brief Resets the internal state (delay lines) of the filter. */
      [[nodiscard]] virtual Status reset() = 0;  // Not const, stateful
      /** @brief Gets the filter order (maximum of numerator/denominator
       * orders). */
      virtual size_t get_order() const = 0;
      /** @brief Gets the number of second-order sections. */
      virtual size_t get_num_sections() const = 0;
      // virtual const std::vector<IIRFilterCoef>& get_sections() const = 0; //
      // Consider adding? Use IIRFilterCoef
    };

    /** @brief Interface for Constant-Q Transform (CQT) plan implementations. */
    template <typename T>  // T is real type here (F32, F64)
    class CQTPlanImpl {
     public:
      using Complex = Detail::GetComplexT<T>;  // Use type trait from core_types
      virtual ~CQTPlanImpl() = default;
      /** @brief Executes the CQT on an input signal. */
      [[nodiscard]] virtual Status execute(
          std::span<const T> input, std::span<Complex> output) const
          = 0;
      /** @brief Gets the number of frequency bins in the CQT output. */
      virtual size_t get_num_bins() const = 0;
      /** @brief Calculates the number of output time frames for a given input
       * length. */
      virtual size_t get_num_output_frames(size_t input_length) const = 0;
      /** @brief Gets the hop length (number of samples between frames). */
      virtual size_t get_hop_length() const = 0;
      // Add other relevant getters? e.g., get_sample_rate(), get_min_freq(),
      // get_max_freq() etc.
    };

    //--------------------------------------------------------------------------
    // Main Backend Implementation Interface (Abstract Base Class)
    //--------------------------------------------------------------------------

    /**
     * @brief Abstract base class defining the interface for backend
     * implementations. Defines the factory methods for creating Plan objects
     * and potentially optimized one-off operations.
     */
    class AbstractBackend {
     public:
      virtual ~AbstractBackend() = default;
      /** @brief Returns the identifier for this backend implementation. */
      virtual Backend get_backend() const = 0;  // Pure virtual

      // === Virtual methods corresponding to OmniDSP public API ===

      // --- DSP Operations (One-off) ---
      [[nodiscard]] virtual OmniExpected<F32Vec> convolve_f32(
          const F32Vec& input,
          const F32Vec& kernel,
          ConvolutionType type,
          ConvolutionMethod method = ConvolutionMethod::Auto) const
          = 0;
      [[nodiscard]] virtual OmniExpected<F64Vec> convolve_f64(
          const F64Vec& input,
          const F64Vec& kernel,
          ConvolutionType type,
          ConvolutionMethod method = ConvolutionMethod::Auto) const
          = 0;
      [[nodiscard]] virtual OmniExpected<C32Vec> convolve_c32(
          const C32Vec& input,
          const C32Vec& kernel,
          ConvolutionType type,
          ConvolutionMethod method = ConvolutionMethod::Auto) const
          = 0;
      [[nodiscard]] virtual OmniExpected<C64Vec> convolve_c64(
          const C64Vec& input,
          const C64Vec& kernel,
          ConvolutionType type,
          ConvolutionMethod method = ConvolutionMethod::Auto) const
          = 0;
      [[nodiscard]] virtual OmniExpected<F32Vec> correlate_f32(
          const F32Vec& input,
          const F32Vec& kernel,
          ConvolutionType type,
          ConvolutionMethod method = ConvolutionMethod::Auto) const
          = 0;
      [[nodiscard]] virtual OmniExpected<F64Vec> correlate_f64(
          const F64Vec& input,
          const F64Vec& kernel,
          ConvolutionType type,
          ConvolutionMethod method = ConvolutionMethod::Auto) const
          = 0;
      [[nodiscard]] virtual OmniExpected<C32Vec> correlate_c32(
          const C32Vec& input,
          const C32Vec& kernel,
          ConvolutionType type,
          ConvolutionMethod method = ConvolutionMethod::Auto) const
          = 0;
      [[nodiscard]] virtual OmniExpected<C64Vec> correlate_c64(
          const C64Vec& input,
          const C64Vec& kernel,
          ConvolutionType type,
          ConvolutionMethod method = ConvolutionMethod::Auto) const
          = 0;
      [[nodiscard]] virtual OmniExpected<C32Vec> fft_c32(
          const C32Vec& input) const
          = 0;
      [[nodiscard]] virtual OmniExpected<C64Vec> fft_c64(
          const C64Vec& input) const
          = 0;
      [[nodiscard]] virtual OmniExpected<C32Vec> ifft_c32(
          const C32Vec& input) const
          = 0;
      [[nodiscard]] virtual OmniExpected<C64Vec> ifft_c64(
          const C64Vec& input) const
          = 0;
      [[nodiscard]] virtual OmniExpected<C32Vec> rfft_f32(
          const F32Vec& input) const
          = 0;
      [[nodiscard]] virtual OmniExpected<C64Vec> rfft_f64(
          const F64Vec& input) const
          = 0;
      [[nodiscard]] virtual OmniExpected<F32Vec> irfft_c32(
          const C32Vec& input, size_t output_length) const
          = 0;
      [[nodiscard]] virtual OmniExpected<F64Vec> irfft_c64(
          const C64Vec& input, size_t output_length) const
          = 0;

      // --- Window Generation (One-off) ---
      // Public API methods taking span output
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
      // These return the PUBLIC Plan interfaces (e.g., FFTPlan<C32>)
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
          const WindowSpec& window_spec) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CQTPlan<F64>>>
      create_cqt_plan_f64(
          F64 sample_rate,
          F64 min_freq,
          F64 max_freq,
          int bins_per_octave,
          const WindowSpec& window_spec) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<ResamplePlan<F32>>>
      create_resample_plan_f32(const ResampleSpec& spec) const = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<ResamplePlan<F64>>>
      create_resample_plan_f64(const ResampleSpec& spec) const = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<ConvolutionPlan<F32>>>
      create_convolution_plan_f32(
          const F32Vec& kernel,
          ConvolutionType type,
          ConvolutionMethod method = ConvolutionMethod::Auto) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<ConvolutionPlan<F64>>>
      create_convolution_plan_f64(
          const F64Vec& kernel,
          ConvolutionType type,
          ConvolutionMethod method = ConvolutionMethod::Auto) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<ConvolutionPlan<C32>>>
      create_convolution_plan_c32(
          const C32Vec& kernel,
          ConvolutionType type,
          ConvolutionMethod method = ConvolutionMethod::Auto) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<ConvolutionPlan<C64>>>
      create_convolution_plan_c64(
          const C64Vec& kernel,
          ConvolutionType type,
          ConvolutionMethod method = ConvolutionMethod::Auto) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CorrelationPlan<F32>>>
      create_correlation_plan_f32(
          const F32Vec& kernel,
          ConvolutionType type,
          ConvolutionMethod method = ConvolutionMethod::Auto) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CorrelationPlan<F64>>>
      create_correlation_plan_f64(
          const F64Vec& kernel,
          ConvolutionType type,
          ConvolutionMethod method = ConvolutionMethod::Auto) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CorrelationPlan<C32>>>
      create_correlation_plan_c32(
          const C32Vec& kernel,
          ConvolutionType type,
          ConvolutionMethod method = ConvolutionMethod::Auto) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<CorrelationPlan<C64>>>
      create_correlation_plan_c64(
          const C64Vec& kernel,
          ConvolutionType type,
          ConvolutionMethod method = ConvolutionMethod::Auto) const
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
          const std::vector<IIRFilterCoef>& sos_coefficients) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::unique_ptr<IIRFilterPlan<F64>>>
      create_iir_filter_plan_f64(
          const std::vector<IIRFilterCoef>& sos_coefficients) const
          = 0;

      // --- Filter Design ---
      [[nodiscard]] virtual OmniExpected<F32Vec> design_fir_filter_f32(
          const FIRFilterSpec& spec) const
          = 0;
      [[nodiscard]] virtual OmniExpected<F64Vec> design_fir_filter_f64(
          const FIRFilterSpec& spec) const
          = 0;
      [[nodiscard]] virtual OmniExpected<std::vector<IIRFilterCoef>>
      design_iir_filter_f32(const IIRFilterSpec& spec) const = 0;
      [[nodiscard]] virtual OmniExpected<std::vector<IIRFilterCoef>>
      design_iir_filter_f64(const IIRFilterSpec& spec) const = 0;

      // --- Internal Helpers ---
      // *** REMOVED: Internal implementation factories are NOT part of the
      // abstract interface ***

      // Internal window generation helper (UPDATED SIGNATURE)
      // Pure virtual, MUST be implemented by each backend.
      // Writes directly to the output span.
      [[nodiscard]] virtual Status generate_window_f32(
          const WindowSpec& spec, std::span<F32> output) const
          = 0;
      [[nodiscard]] virtual Status generate_window_f64(
          const WindowSpec& spec, std::span<F64> output) const
          = 0;

     protected:
      // Add protected default constructor if needed by derived classes
      AbstractBackend() = default;

    };  // class AbstractBackend

  }  // namespace backend
}  // namespace OmniDSP

#endif  // OMNIDSP_BACKEND_H
