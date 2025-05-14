/**
 * @file backend.hpp
 * @brief Defines the abstract base class interface for backend implementations.
 */

#ifndef OMNIDSP_ABSTRACT_BACKEND_HPP
#define OMNIDSP_ABSTRACT_BACKEND_HPP

#include <complex>
#include <cstddef>
#include <expected>
#include <memory>
#include <span>
#include <vector>

// Core types are fundamental and are included directly.
#include <OmniDSP/core_types.hpp>  // For Status, OmniExpected, F32, C32, C32Vec, C64Vec, BackendType, Utils::GetComplexType etc.
#include <OmniDSP/types/convolution.hpp>  // For ConvolutionType, ConvolutionMethod (used by Params::Correlation)
#include <OmniDSP/window.hpp>  // For WindowSetup (used in some design method params if any, or by Design structs)

// Parameter types (needed for method parameters)
#include <OmniDSP/params/convolution.hpp>  // For Params::Correlation

// Design specifications (needed for method parameters)
#include <OmniDSP/design/cqt.hpp>         // Defines Design::CQT
#include <OmniDSP/design/fir_filter.hpp>  // Defines Design::FIRFilter
#include <OmniDSP/design/iir_filter.hpp>  // Defines Design::IIRFilter
#include <OmniDSP/design/resample.hpp>    // Defines Design::Resample

// Coefficient types (may be returned by design methods)
#include <OmniDSP/coefs/fir_filter.hpp>  // Defines Coefs::FIRFilter
#include <OmniDSP/coefs/iir_filter.hpp>

namespace OmniDSP::Abstract {

  //--------------------------------------------------------------------------
  // Plan / Processor Implementation Interfaces (Abstract Base Classes)
  //--------------------------------------------------------------------------
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
    using Complex = Utils::GetComplexType<T>;
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
  class ResampleProcessorImpl {
   public:
    virtual ~ResampleProcessorImpl() = default;
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
  class FIRFilterProcessorImpl {
   public:
    virtual ~FIRFilterProcessorImpl() = default;
    [[nodiscard]] virtual Status execute(
        std::span<const T> input, std::span<T> output)
        = 0;
    [[nodiscard]] virtual Status reset() = 0;
    virtual size_t get_order() const = 0;
    virtual size_t get_num_taps() const = 0;
  };

  template <typename T>
  class IIRFilterProcessorImpl {
   public:
    virtual ~IIRFilterProcessorImpl() = default;
    [[nodiscard]] virtual Status execute(
        std::span<const T> input, std::span<T> output)
        = 0;
    [[nodiscard]] virtual Status reset() = 0;
    virtual size_t get_order() const = 0;
    virtual size_t get_num_sections() const = 0;
  };

  template <typename T>
  class CQTProcessorImpl {
   public:
    using Complex = Utils::GetComplexType<T>;
    virtual ~CQTProcessorImpl() = default;
    [[nodiscard]] virtual Status execute(
        std::span<const T> input, std::span<Complex> output)
        const  // Typically CQT execute is stateful if overlapping frames
        = 0;
    [[nodiscard]] virtual Status reset() = 0;  // Added reset method
    virtual size_t get_num_bins() const = 0;
    virtual size_t get_num_output_frames(size_t input_length) const = 0;
    virtual size_t get_hop_length() const = 0;
  };

  class Backend {
   public:
    virtual ~Backend() = default;
    virtual BackendType get_backend() const = 0;

    // One-off DSP Operations
    [[nodiscard]] virtual OmniExpected<F32Vec> convolve_f32(
        const F32Vec&, const F32Vec&, ConvolutionType, ConvolutionMethod) const
        = 0;
    [[nodiscard]] virtual OmniExpected<F64Vec> convolve_f64(
        const F64Vec&, const F64Vec&, ConvolutionType, ConvolutionMethod) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C32Vec> convolve_c32(
        const C32Vec&, const C32Vec&, ConvolutionType, ConvolutionMethod) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C64Vec> convolve_c64(
        const C64Vec&, const C64Vec&, ConvolutionType, ConvolutionMethod) const
        = 0;
    [[nodiscard]] virtual OmniExpected<F32Vec> correlate_f32(
        const F32Vec&, const F32Vec&, ConvolutionType, ConvolutionMethod) const
        = 0;
    [[nodiscard]] virtual OmniExpected<F64Vec> correlate_f64(
        const F64Vec&, const F64Vec&, ConvolutionType, ConvolutionMethod) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C32Vec> correlate_c32(
        const C32Vec&, const C32Vec&, ConvolutionType, ConvolutionMethod) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C64Vec> correlate_c64(
        const C64Vec&, const C64Vec&, ConvolutionType, ConvolutionMethod) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C32Vec> fft_c32(const C32Vec&) const = 0;
    [[nodiscard]] virtual OmniExpected<C64Vec> fft_c64(const C64Vec&) const = 0;
    [[nodiscard]] virtual OmniExpected<C32Vec> ifft_c32(const C32Vec&) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C64Vec> ifft_c64(const C64Vec&) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C32Vec> rfft_f32(const F32Vec&) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C64Vec> rfft_f64(const F64Vec&) const
        = 0;
    [[nodiscard]] virtual OmniExpected<F32Vec> irfft_c32(
        const C32Vec&, size_t) const
        = 0;
    [[nodiscard]] virtual OmniExpected<F64Vec> irfft_c64(
        const C64Vec&, size_t) const
        = 0;

    // Window Generation Methods
    [[nodiscard]] virtual Status bartlett_window_f32(
        size_t, std::span<F32>) const
        = 0;
    [[nodiscard]] virtual Status bartlett_window_f64(
        size_t, std::span<F64>) const
        = 0;
    [[nodiscard]] virtual Status blackman_window_f32(
        size_t, std::span<F32>) const
        = 0;
    [[nodiscard]] virtual Status blackman_window_f64(
        size_t, std::span<F64>) const
        = 0;
    [[nodiscard]] virtual Status flattop_window_f32(
        size_t, std::span<F32>) const
        = 0;
    [[nodiscard]] virtual Status flattop_window_f64(
        size_t, std::span<F64>) const
        = 0;
    [[nodiscard]] virtual Status gaussian_window_f32(
        size_t, double, std::span<F32>) const
        = 0;
    [[nodiscard]] virtual Status gaussian_window_f64(
        size_t, double, std::span<F64>) const
        = 0;
    [[nodiscard]] virtual Status hamming_window_f32(
        size_t, std::span<F32>) const
        = 0;
    [[nodiscard]] virtual Status hamming_window_f64(
        size_t, std::span<F64>) const
        = 0;
    [[nodiscard]] virtual Status hann_window_f32(size_t, std::span<F32>) const
        = 0;
    [[nodiscard]] virtual Status hann_window_f64(size_t, std::span<F64>) const
        = 0;
    [[nodiscard]] virtual Status kaiser_window_f32(
        size_t, double, std::span<F32>) const
        = 0;
    [[nodiscard]] virtual Status kaiser_window_f64(
        size_t, double, std::span<F64>) const
        = 0;
    [[nodiscard]] virtual Status rectangular_window_f32(
        size_t, std::span<F32>) const
        = 0;
    [[nodiscard]] virtual Status rectangular_window_f64(
        size_t, std::span<F64>) const
        = 0;
    [[nodiscard]] virtual Status triangular_window_f32(
        size_t, std::span<F32>) const
        = 0;
    [[nodiscard]] virtual Status triangular_window_f64(
        size_t, std::span<F64>) const
        = 0;

    // Plan Impl / Processor Impl Factory Methods
    [[nodiscard]] virtual OmniExpected<std::unique_ptr<FFTPlanImpl<C32>>>
    create_fft_plan_impl_c32(size_t length) const = 0;
    [[nodiscard]] virtual OmniExpected<std::unique_ptr<FFTPlanImpl<C64>>>
    create_fft_plan_impl_c64(size_t length) const = 0;
    [[nodiscard]] virtual OmniExpected<std::unique_ptr<RFFTPlanImpl<F32>>>
    create_rfft_plan_impl_f32(size_t length) const = 0;
    [[nodiscard]] virtual OmniExpected<std::unique_ptr<RFFTPlanImpl<F64>>>
    create_rfft_plan_impl_f64(size_t length) const = 0;

    [[nodiscard]] virtual OmniExpected<std::unique_ptr<CQTProcessorImpl<F32>>>
    create_cqt_processor_impl_f32(const Design::CQT& design) const = 0;
    [[nodiscard]] virtual OmniExpected<std::unique_ptr<CQTProcessorImpl<F64>>>
    create_cqt_processor_impl_f64(const Design::CQT& design) const = 0;

    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<ResampleProcessorImpl<F32>>>
    create_resample_processor_impl_f32(const Design::Resample& design) const
        = 0;
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<ResampleProcessorImpl<F64>>>
    create_resample_processor_impl_f64(const Design::Resample& design) const
        = 0;

    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<ConvolutionPlanImpl<F32>>>
    create_convolution_plan_impl_f32(
        const Params::Convolution&
            params,  // Changed to use Params::Convolution
        std::span<const F32> kernel_coeffs) const  // Changed to span and name
        = 0;
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<ConvolutionPlanImpl<F64>>>
    create_convolution_plan_impl_f64(
        const Params::Convolution&
            params,  // Changed to use Params::Convolution
        std::span<const F64> kernel_coeffs) const  // Changed to span and name
        = 0;
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<ConvolutionPlanImpl<C32>>>
    create_convolution_plan_impl_c32(
        const Params::Convolution&
            params,  // Changed to use Params::Convolution
        std::span<const C32> kernel_coeffs) const  // Changed to span and name
        = 0;
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<ConvolutionPlanImpl<C64>>>
    create_convolution_plan_impl_c64(
        const Params::Convolution&
            params,  // Changed to use Params::Convolution
        std::span<const C64> kernel_coeffs) const  // Changed to span and name
        = 0;

    // Updated create_correlation_plan_impl signatures
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<CorrelationPlanImpl<F32>>>
    create_correlation_plan_impl_f32(
        const Params::Correlation& params,
        std::span<const F32> template_coeffs) const
        = 0;
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<CorrelationPlanImpl<F64>>>
    create_correlation_plan_impl_f64(
        const Params::Correlation& params,
        std::span<const F64> template_coeffs) const
        = 0;
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<CorrelationPlanImpl<C32>>>
    create_correlation_plan_impl_c32(
        const Params::Correlation& params,
        std::span<const C32> template_coeffs) const
        = 0;
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<CorrelationPlanImpl<C64>>>
    create_correlation_plan_impl_c64(
        const Params::Correlation& params,
        std::span<const C64> template_coeffs) const
        = 0;

    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<FIRFilterProcessorImpl<F32>>>
    create_fir_filter_processor_impl_f32(
        const Coefs::FIRFilter<F32>& coefficients) const
        = 0;  // Assuming Coefs::FIRFilter<T> holds std::vector<T> or similar
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<FIRFilterProcessorImpl<F64>>>
    create_fir_filter_processor_impl_f64(
        const Coefs::FIRFilter<F64>& coefficients) const
        = 0;
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<FIRFilterProcessorImpl<C32>>>
    create_fir_filter_processor_impl_c32(
        const Coefs::FIRFilter<C32>& coefficients) const
        = 0;
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<FIRFilterProcessorImpl<C64>>>
    create_fir_filter_processor_impl_c64(
        const Coefs::FIRFilter<C64>& coefficients) const
        = 0;

    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<IIRFilterProcessorImpl<F32>>>
    create_iir_filter_processor_impl_f32(
        const Coefs::IIRFilterSOS& sos_coefficients) const
        = 0;
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<IIRFilterProcessorImpl<F64>>>
    create_iir_filter_processor_impl_f64(
        const Coefs::IIRFilterSOS& sos_coefficients) const
        = 0;

    // Filter Design
    // For real coefficients
    [[nodiscard]] virtual OmniExpected<Coefs::FIRFilter<F32>>
    design_fir_filter_f32(const Design::FIRFilter& design) const = 0;
    [[nodiscard]] virtual OmniExpected<Coefs::FIRFilter<F64>>
    design_fir_filter_f64(const Design::FIRFilter& design) const = 0;

    // For complex coefficients
    [[nodiscard]] virtual OmniExpected<Coefs::FIRFilter<C32>>
    design_fir_filter_c32(const Design::FIRFilter& design) const = 0;
    [[nodiscard]] virtual OmniExpected<Coefs::FIRFilter<C64>>
    design_fir_filter_c64(const Design::FIRFilter& design) const = 0;

    // For IIR filters (real coefficients)
    [[nodiscard]] virtual OmniExpected<Coefs::IIRFilterSOS>
    design_iir_filter_f32(const Design::IIRFilter& design) const = 0;
    [[nodiscard]] virtual OmniExpected<Coefs::IIRFilterSOS>
    design_iir_filter_f64(const Design::IIRFilter& design) const = 0;

   protected:
    Backend() = default;
  };

  // Factory function declarations
  std::unique_ptr<Backend> create_default_backend();
  std::unique_ptr<Backend> create_accelerate_backend();
  std::unique_ptr<Backend> create_onemkl_backend();
  std::unique_ptr<Backend> create_intelipp_backend();

}  // namespace OmniDSP::Abstract

#endif  // OMNIDSP_ABSTRACT_BACKEND_HPP
