/**
 * @file backend.hpp (dispatcher)
 * @brief Defines the Dispatcher::Backend class, which delegates operations
 * to other configured backend implementations based on OperationCategory.
 */

#ifndef OMNIDSP_DISPATCHER_BACKEND_HPP
#define OMNIDSP_DISPATCHER_BACKEND_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "OmniDSP/core_types.hpp"
#include "OmniDSP/types/convolution.hpp"
#include "interface/backend.hpp"

// Ensure full definitions for types used in method signatures are available
#include "OmniDSP/coefs/fir_filter.hpp"
#include "OmniDSP/coefs/iir_filter.hpp"
#include "OmniDSP/design/cqt.hpp"
#include "OmniDSP/design/fir_filter.hpp"
#include "OmniDSP/design/iir_filter.hpp"
#include "OmniDSP/design/resample.hpp"
// WindowSetup is included via core_types or other public headers if needed by
// interface

namespace OmniDSP {
  // OperationCategory enum assumed to be available (e.g., via core_types.hpp)
}

namespace OmniDSP::Dispatcher {

  class Backend final : public Abstract::Backend {
   public:
    Backend(
        std::unique_ptr<Abstract::Backend> primary_backend,
        std::map<OperationCategory, std::shared_ptr<Abstract::Backend>>
            backend_overrides);

    ~Backend() override = default;

    BackendType get_backend() const override;

    // --- One-off DSP Operations ---
    [[nodiscard]] OmniExpected<F32Vec> convolve_f32(
        const F32Vec& input,
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const override;
    [[nodiscard]] OmniExpected<F64Vec> convolve_f64(
        const F64Vec& input,
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const override;
    [[nodiscard]] OmniExpected<C32Vec> convolve_c32(
        const C32Vec& input,
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const override;
    [[nodiscard]] OmniExpected<C64Vec> convolve_c64(
        const C64Vec& input,
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const override;
    [[nodiscard]] OmniExpected<F32Vec> correlate_f32(
        const F32Vec& input,
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const override;
    [[nodiscard]] OmniExpected<F64Vec> correlate_f64(
        const F64Vec& input,
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const override;
    [[nodiscard]] OmniExpected<C32Vec> correlate_c32(
        const C32Vec& input,
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const override;
    [[nodiscard]] OmniExpected<C64Vec> correlate_c64(
        const C64Vec& input,
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const override;
    [[nodiscard]] OmniExpected<C32Vec> fft_c32(
        const C32Vec& input) const override;
    [[nodiscard]] OmniExpected<C64Vec> fft_c64(
        const C64Vec& input) const override;
    [[nodiscard]] OmniExpected<C32Vec> ifft_c32(
        const C32Vec& input) const override;
    [[nodiscard]] OmniExpected<C64Vec> ifft_c64(
        const C64Vec& input) const override;
    [[nodiscard]] OmniExpected<C32Vec> rfft_f32(
        const F32Vec& input) const override;
    [[nodiscard]] OmniExpected<C64Vec> rfft_f64(
        const F64Vec& input) const override;
    [[nodiscard]] OmniExpected<F32Vec> irfft_c32(
        const C32Vec& input, size_t output_length) const override;
    [[nodiscard]] OmniExpected<F64Vec> irfft_c64(
        const C64Vec& input, size_t output_length) const override;

    // --- Specific Window Generation Methods ---
    [[nodiscard]] Status bartlett_window_f32(
        size_t length, std::span<F32> output) const override;
    [[nodiscard]] Status bartlett_window_f64(
        size_t length, std::span<F64> output) const override;
    [[nodiscard]] Status blackman_window_f32(
        size_t length, std::span<F32> output) const override;
    [[nodiscard]] Status blackman_window_f64(
        size_t length, std::span<F64> output) const override;
    [[nodiscard]] Status flattop_window_f32(
        size_t length, std::span<F32> output) const override;
    [[nodiscard]] Status flattop_window_f64(
        size_t length, std::span<F64> output) const override;
    [[nodiscard]] Status gaussian_window_f32(
        size_t length, double stddev, std::span<F32> output) const override;
    [[nodiscard]] Status gaussian_window_f64(
        size_t length, double stddev, std::span<F64> output) const override;
    [[nodiscard]] Status hamming_window_f32(
        size_t length, std::span<F32> output) const override;
    [[nodiscard]] Status hamming_window_f64(
        size_t length, std::span<F64> output) const override;
    [[nodiscard]] Status hann_window_f32(
        size_t length, std::span<F32> output) const override;
    [[nodiscard]] Status hann_window_f64(
        size_t length, std::span<F64> output) const override;
    [[nodiscard]] Status kaiser_window_f32(
        size_t length, double beta, std::span<F32> output) const override;
    [[nodiscard]] Status kaiser_window_f64(
        size_t length, double beta, std::span<F64> output) const override;
    [[nodiscard]] Status rectangular_window_f32(
        size_t length, std::span<F32> output) const override;
    [[nodiscard]] Status rectangular_window_f64(
        size_t length, std::span<F64> output) const override;
    [[nodiscard]] Status triangular_window_f32(
        size_t length, std::span<F32> output) const override;
    [[nodiscard]] Status triangular_window_f64(
        size_t length, std::span<F64> output) const override;

    // --- Plan Impl / Processor Impl Factory Methods ---
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>
    create_fft_plan_impl_c32(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>
    create_fft_plan_impl_c64(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>
    create_rfft_plan_impl_f32(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>
    create_rfft_plan_impl_f64(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<F32>>>
    create_cqt_processor_impl_f32(const Design::CQT& design) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<F64>>>
    create_cqt_processor_impl_f64(const Design::CQT& design) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::ResampleProcessorImpl<F32>>>
    create_resample_processor_impl_f32(
        const Design::Resample& design) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::ResampleProcessorImpl<F64>>>
    create_resample_processor_impl_f64(
        const Design::Resample& design) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::ConvolutionPlanImpl<F32>>>
    create_convolution_plan_impl_f32(
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::ConvolutionPlanImpl<F64>>>
    create_convolution_plan_impl_f64(
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::ConvolutionPlanImpl<C32>>>
    create_convolution_plan_impl_c32(
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::ConvolutionPlanImpl<C64>>>
    create_convolution_plan_impl_c64(
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::CorrelationPlanImpl<F32>>>
    create_correlation_plan_impl_f32(
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::CorrelationPlanImpl<F64>>>
    create_correlation_plan_impl_f64(
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::CorrelationPlanImpl<C32>>>
    create_correlation_plan_impl_c32(
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::CorrelationPlanImpl<C64>>>
    create_correlation_plan_impl_c64(
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::FIRFilterProcessorImpl<F32>>>
    create_fir_filter_processor_impl_f32(
        const F32Vec& coefficients) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::FIRFilterProcessorImpl<F64>>>
    create_fir_filter_processor_impl_f64(
        const F64Vec& coefficients) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::FIRFilterProcessorImpl<C32>>>
    create_fir_filter_processor_impl_c32(
        const C32Vec& coefficients) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::FIRFilterProcessorImpl<C64>>>
    create_fir_filter_processor_impl_c64(
        const C64Vec& coefficients) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::IIRFilterProcessorImpl<F32>>>
    create_iir_filter_processor_impl_f32(
        const Coefs::IIRFilterSOS& sos_coefficients) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::IIRFilterProcessorImpl<F64>>>
    create_iir_filter_processor_impl_f64(
        const Coefs::IIRFilterSOS& sos_coefficients) const override;

    // --- Filter Design ---
    // Real coefficient FIR filter design
    [[nodiscard]] OmniExpected<Coefs::FIRFilter<F32>> design_fir_filter_f32(
        const Design::FIRFilter& design) const override;
    [[nodiscard]] OmniExpected<Coefs::FIRFilter<F64>> design_fir_filter_f64(
        const Design::FIRFilter& design) const override;

    // Complex coefficient FIR filter design
    [[nodiscard]] OmniExpected<Coefs::FIRFilter<C32>> design_fir_filter_c32(
        const Design::FIRFilter& design) const override;
    [[nodiscard]] OmniExpected<Coefs::FIRFilter<C64>> design_fir_filter_c64(
        const Design::FIRFilter& design) const override;

    // IIR filter design
    [[nodiscard]] OmniExpected<Coefs::IIRFilterSOS> design_iir_filter_f32(
        const Design::IIRFilter& design) const override;
    [[nodiscard]] OmniExpected<Coefs::IIRFilterSOS> design_iir_filter_f64(
        const Design::IIRFilter& design) const override;

   private:
    std::unique_ptr<Abstract::Backend> primary_backend_;
    std::map<OperationCategory, std::shared_ptr<Abstract::Backend>> overrides_;
    Abstract::Backend* select_backend(OperationCategory category) const;
  };

}  // namespace OmniDSP::Dispatcher

#endif  // OMNIDSP_DISPATCHER_BACKEND_HPP
