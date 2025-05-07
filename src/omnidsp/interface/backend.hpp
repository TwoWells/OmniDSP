/**
 * @file backend.hpp
 * @brief Defines the abstract base class interface for backend implementations.
 */

#ifndef OMNIDSP_BACKEND_HPP
#define OMNIDSP_BACKEND_HPP

#include <complex>
#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <vector>

// Include core library types and specifications
#include <OmniDSP/convolution.hpp>
#include <OmniDSP/core_types.hpp>
#include <OmniDSP/filter.hpp>
#include <OmniDSP/resample.hpp>
#include <OmniDSP/window.hpp>

// Forward declare public Plan classes
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

namespace OmniDSP::Abstract {

  class AbstractBackend;

  //--------------------------------------------------------------------------
  // Plan Implementation Interfaces (Abstract Base Classes)
  //--------------------------------------------------------------------------
  // ... (PlanImpl interfaces remain the same) ...
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
  template <typename T>
  class RFFTPlanImpl {
   public:
    using Complex = Utils::GetComplexType<T>; /* *** UPDATED Namespace *** */
    virtual ~RFFTPlanImpl() = default;
    [[nodiscard]] virtual Status rfft(
        std::span<const T> input, std::span<Complex> output) const
        = 0;
    [[nodiscard]] virtual Status irfft(
        std::span<const Complex> input, std::span<T> output) const
        = 0;
    virtual size_t get_length() const = 0;
  };
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
    [[nodiscard]] virtual std::span<const T> get_kernel() const = 0;
    [[nodiscard]] virtual ConvolutionMethod get_method() const = 0;
  };
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
    [[nodiscard]] virtual std::span<const T> get_template() const = 0;
    [[nodiscard]] virtual ConvolutionMethod get_method() const = 0;
  };
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
  };
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
  };
  template <typename T>
  class CQTPlanImpl {
   public:
    using Complex = Utils::GetComplexType<T>; /* *** UPDATED Namespace *** */
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
    // ... (Virtual function declarations remain the same) ...
   public:
    virtual ~AbstractBackend() = default;
    virtual BackendType get_backend() const = 0;
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
    [[nodiscard]] virtual OmniExpected<FIRCoefs<F32>> design_fir_filter_f32(
        const FIRFilterSpec& spec) const
        = 0;
    [[nodiscard]] virtual OmniExpected<FIRCoefs<F64>> design_fir_filter_f64(
        const FIRFilterSpec& spec) const
        = 0;
    [[nodiscard]] virtual OmniExpected<std::vector<IIRFilterCoef>>
    design_iir_filter_f32(const IIRFilterSpec& spec) const = 0;
    [[nodiscard]] virtual OmniExpected<std::vector<IIRFilterCoef>>
    design_iir_filter_f64(const IIRFilterSpec& spec) const = 0;
    [[nodiscard]] virtual Status generate_window_f32(
        const WindowSpec& spec, std::span<F32> output) const
        = 0;
    [[nodiscard]] virtual Status generate_window_f64(
        const WindowSpec& spec, std::span<F64> output) const
        = 0;

   protected:
    AbstractBackend() = default;
  };

  // --- Backend Factory Function Declarations ---
  // These functions are defined in the respective backend's .cpp file
  // and are called by OmniDSP::create.

  // Always available
  std::unique_ptr<AbstractBackend> create_default_backend();

  // Conditionally available based on CMake definitions
  // (These declarations are fine even if the backend isn't compiled,
  // as long as they aren't *called* without the definition being linked)
  std::unique_ptr<AbstractBackend> create_accelerate_backend();
  std::unique_ptr<AbstractBackend> create_onemkl_backend();
  std::unique_ptr<AbstractBackend> create_intelipp_backend();

  // Add declarations for other potential backends here...

}  // namespace OmniDSP::Abstract

#endif  // OMNIDSP_BACKEND_HPP
