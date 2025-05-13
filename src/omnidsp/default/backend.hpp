/**
 * @file backend.hpp (Default)
 * @brief Defines the Default backend class, inheriting from Abstract::Backend.
 */

#ifndef OMNIDSP_DEFAULT_BACKEND_HPP
#define OMNIDSP_DEFAULT_BACKEND_HPP

#include <spdlog/spdlog.h>  // For logging (optional here, mainly in .cpp)

#include <OmniDSP/core_types.hpp>  // For F32, F64, C32Vec, C64Vec, Status, OmniExpected etc.
#include <OmniDSP/design/fir_filter.hpp>  // For Design::FIRFilter
#include <OmniDSP/window.hpp>  // For WindowSetup (used in implementations)

#include "../interface/backend.hpp"  // Abstract::Backend

namespace OmniDSP::Default {

  class Backend : public Abstract::Backend {
   public:
    Backend();
    ~Backend() override;

    BackendType get_backend() const override;

    // --- DSP Operations (One-Off Implementations) ---
    [[nodiscard]] OmniExpected<F32Vec> convolve_f32(
        const F32Vec& input,
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const override;
    [[nodiscard]] OmniExpected<F64Vec> convolve_f64(
        const F64Vec& input,
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const override;
    [[nodiscard]] OmniExpected<C32Vec> convolve_c32(
        const C32Vec& input,
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const override;
    [[nodiscard]] OmniExpected<C64Vec> convolve_c64(
        const C64Vec& input,
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const override;
    [[nodiscard]] OmniExpected<F32Vec> correlate_f32(
        const F32Vec& input,
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const override;
    [[nodiscard]] OmniExpected<F64Vec> correlate_f64(
        const F64Vec& input,
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const override;
    [[nodiscard]] OmniExpected<C32Vec> correlate_c32(
        const C32Vec& input,
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const override;
    [[nodiscard]] OmniExpected<C64Vec> correlate_c64(
        const C64Vec& input,
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const override;
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

    // --- Window Generation Overrides ---
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

    // --- Plan Factories ---
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>
    create_fft_plan_impl_c32(size_t length) const;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>
    create_fft_plan_impl_c64(size_t length) const;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>
    create_rfft_plan_impl_f32(size_t length) const;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>
    create_rfft_plan_impl_f64(size_t length) const;

    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<F32>>>
    create_cqt_processor_impl_f32(const Design::CQT& spec) const;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::CQTProcessorImpl<F64>>>
    create_cqt_processor_impl_f64(const Design::CQT& spec) const;

    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::ResampleProcessorImpl<F32>>>
    create_resample_processor_impl_f32(const Design::Resample& spec) const;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::ResampleProcessorImpl<F64>>>
    create_resample_processor_impl_f64(const Design::Resample& spec) const;

    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::ConvolutionPlanImpl<F32>>>
    create_convolution_plan_impl_f32(
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::ConvolutionPlanImpl<F64>>>
    create_convolution_plan_impl_f64(
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::ConvolutionPlanImpl<C32>>>
    create_convolution_plan_impl_c32(
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::ConvolutionPlanImpl<C64>>>
    create_convolution_plan_impl_c64(
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const;

    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::CorrelationPlanImpl<F32>>>
    create_correlation_plan_impl_f32(
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::CorrelationPlanImpl<F64>>>
    create_correlation_plan_impl_f64(
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::CorrelationPlanImpl<C32>>>
    create_correlation_plan_impl_c32(
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::CorrelationPlanImpl<C64>>>
    create_correlation_plan_impl_c64(
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method) const;

    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::FIRFilterProcessorImpl<F32>>>
    create_fir_filter_processor_impl_f32(const F32Vec& coefficients) const;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::FIRFilterProcessorImpl<F64>>>
    create_fir_filter_processor_impl_f64(const F64Vec& coefficients) const;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::FIRFilterProcessorImpl<C32>>>
    create_fir_filter_processor_impl_c32(const C32Vec& coefficients) const;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::FIRFilterProcessorImpl<C64>>>
    create_fir_filter_processor_impl_c64(const C64Vec& coefficients) const;

    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::IIRFilterProcessorImpl<F32>>>
    create_iir_filter_processor_impl_f32(
        const std::vector<IIRFilterCoef>& sos_coefficients) const;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::IIRFilterProcessorImpl<F64>>>
    create_iir_filter_processor_impl_f64(
        const std::vector<IIRFilterCoef>& sos_coefficients) const;

    // --- Filter Design ---
    // For real coefficients
    [[nodiscard]] OmniExpected<FIRCoefs<F32>> design_fir_filter_f32(
        const Design::FIRFilter& spec) const override;
    [[nodiscard]] OmniExpected<FIRCoefs<F64>> design_fir_filter_f64(
        const Design::FIRFilter& spec) const override;

    // For complex coefficients (NEWLY ADDED based on Abstract::Backend
    // requirements)
    [[nodiscard]] OmniExpected<FIRCoefs<C32>> design_fir_filter_c32(
        const Design::FIRFilter& spec) const override;
    [[nodiscard]] OmniExpected<FIRCoefs<C64>> design_fir_filter_c64(
        const Design::FIRFilter& spec) const override;

    // For IIR filters (real coefficients)
    [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>>
    design_iir_filter_f32(const Design::IIRFilter& spec) const override;
    [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>>
    design_iir_filter_f64(const Design::IIRFilter& spec) const override;

   private:
    // Helper for window generation (if any, or remove if not needed)
  };

}  // namespace OmniDSP::Default

#endif  // OMNIDSP_DEFAULT_BACKEND_HPP
