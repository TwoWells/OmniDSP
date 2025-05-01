/**
 * @file backend.hpp
 * @brief Defines the abstract base class interface for backend implementations.
 * @details Concrete backend implementations (DefaultBackend, AccelerateBackend,
 * OneMKLBackend) must inherit from the abstract base classes defined here
 * (AbstractBackend, *PlanImpl).
 */

#ifndef OMNIDSP_BACKEND_HPP  // Changed guard name convention
#define OMNIDSP_BACKEND_HPP

#include <complex>
#include <cstddef>  // For size_t
#include <memory>   // For std::unique_ptr
#include <span>     // For std::span (requires C++20)
#include <vector>

// Include core library types and specifications
#include <OmniDSP/convolution.hpp>  // Includes ConvolutionType, ConvolutionMethod
#include <OmniDSP/core_types.hpp>  // Includes Status, Backend, OmniExpected, F32, C32, F64, C64, Vec aliases, GetComplexT etc.
#include <OmniDSP/filter.hpp>  // Includes FIRFilterSpec, IIRFilterSpec, IIRFilterCoef
#include <OmniDSP/resample.hpp>  // Includes ResampleSpec
#include <OmniDSP/window.hpp>    // Includes WindowSpec, WindowType

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
  class OmniDSPImpl;
}  // namespace OmniDSP

namespace OmniDSP {
  namespace backend {

    class AbstractBackend;

    //--------------------------------------------------------------------------
    // Plan Implementation Interfaces (Abstract Base Classes)
    //--------------------------------------------------------------------------

    /** @brief Interface for complex-to-complex FFT plan implementations. */
    template <typename T>
    class FFTPlanImpl {
     public:
      virtual ~FFTPlanImpl() = default;
      [[nodiscard]] virtual Status fft(
          std::span<const T> input, std::span<T> output) const
          = 0;
      [[nodiscard]] virtual Status ifft(
          std::span<const T> input, std::span<T> output) const
          = 0;
      virtual size_t get_length() const = 0;
    };

    /** @brief Interface for real-to-complex/complex-to-real FFT plan
     * implementations. */
    template <typename T>
    class RFFTPlanImpl {
     public:
      using Complex = Detail::GetComplexT<T>;
      virtual ~RFFTPlanImpl() = default;
      [[nodiscard]] virtual Status rfft(
          std::span<const T> input, std::span<Complex> output) const
          = 0;
      [[nodiscard]] virtual Status irfft(
          std::span<const Complex> input, std::span<T> output) const
          = 0;
      virtual size_t get_length() const = 0;
    };

    /** @brief Interface for resampling plan implementations. */
    template <typename T>
    class ResamplePlanImpl {
     public:
      virtual ~ResamplePlanImpl() = default;
      [[nodiscard]] virtual Status execute(
          std::span<const T> input, std::span<T> output)
          = 0;
      [[nodiscard]] virtual Status reset() = 0;
      virtual double get_input_rate() const = 0;
      virtual double get_output_rate() const = 0;
      virtual size_t get_output_length(size_t input_length) const = 0;
    };

    /** @brief Interface for convolution plan implementations. */
    template <typename T>
    class ConvolutionPlanImpl {
     public:
      virtual ~ConvolutionPlanImpl() = default;
      [[nodiscard]] virtual Status execute(
          std::span<const T> input, std::span<T> output) const
          = 0;
      virtual size_t get_kernel_length() const = 0;
      virtual ConvolutionType get_type() const = 0;
      virtual size_t get_output_length(size_t input_length) const = 0;
      // *** ADDED pure virtual functions ***
      [[nodiscard]] virtual std::span<const T> get_kernel() const = 0;
      [[nodiscard]] virtual ConvolutionMethod get_method() const = 0;
    };

    /** @brief Interface for correlation plan implementations. */
    template <typename T>
    class CorrelationPlanImpl {
     public:
      virtual ~CorrelationPlanImpl() = default;
      [[nodiscard]] virtual Status execute(
          std::span<const T> input, std::span<T> output) const
          = 0;
      virtual size_t get_template_length() const = 0;
      virtual ConvolutionType get_type() const = 0;
      virtual size_t get_output_length(size_t input_length) const = 0;
      // *** ADDED pure virtual functions ***
      [[nodiscard]] virtual std::span<const T> get_template() const = 0;
      [[nodiscard]] virtual ConvolutionMethod get_method() const = 0;
    };

    /** @brief Interface for FIR filter plan implementations. */
    template <typename T>
    class FIRFilterPlanImpl {
     public:
      virtual ~FIRFilterPlanImpl() = default;
      [[nodiscard]] virtual Status execute(
          std::span<const T> input, std::span<T> output)
          = 0;
      [[nodiscard]] virtual Status reset() = 0;
      virtual size_t get_order() const = 0;
      virtual size_t get_num_taps() const = 0;
      // [[nodiscard]] virtual std::span<const T> get_coefficients() const = 0;
      // // Optional
    };

    /** @brief Interface for IIR filter plan implementations. */
    template <typename T>
    class IIRFilterPlanImpl {
     public:
      virtual ~IIRFilterPlanImpl() = default;
      [[nodiscard]] virtual Status execute(
          std::span<const T> input, std::span<T> output)
          = 0;
      [[nodiscard]] virtual Status reset() = 0;
      virtual size_t get_order() const = 0;
      virtual size_t get_num_sections() const = 0;
      // virtual const std::vector<IIRFilterCoef>& get_sections() const = 0; //
      // Optional
    };

    /** @brief Interface for Constant-Q Transform (CQT) plan implementations. */
    template <typename T>
    class CQTPlanImpl {
     public:
      using Complex = Detail::GetComplexT<T>;
      virtual ~CQTPlanImpl() = default;
      [[nodiscard]] virtual Status execute(
          std::span<const T> input, std::span<Complex> output) const
          = 0;
      virtual size_t get_num_bins() const = 0;
      virtual size_t get_num_output_frames(size_t input_length) const = 0;
      virtual size_t get_hop_length() const = 0;
    };

    //--------------------------------------------------------------------------
    // Main Backend Implementation Interface (Abstract Base Class)
    //--------------------------------------------------------------------------
    class AbstractBackend {
     public:
      virtual ~AbstractBackend() = default;
      virtual Backend get_backend() const = 0;

      // === Virtual methods corresponding to OmniDSP public API ===
      // ... (One-off operations, Window Generation, Plan Factories, Filter
      // Design) ...
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
      // Internal window generation helper
      [[nodiscard]] virtual Status generate_window_f32(
          const WindowSpec& spec, std::span<F32> output) const
          = 0;
      [[nodiscard]] virtual Status generate_window_f64(
          const WindowSpec& spec, std::span<F64> output) const
          = 0;

     protected:
      AbstractBackend() = default;

    };  // class AbstractBackend

  }  // namespace backend
}  // namespace OmniDSP

#endif  // OMNIDSP_BACKEND_HPP
