/**
 * @file backend.hpp (Default)
 * @brief Declares the concrete Default backend implementation class.
 */

#ifndef OMNIDSP_DEFAULT_BACKEND_HPP
#define OMNIDSP_DEFAULT_BACKEND_HPP

#include <OmniDSP/convolution.hpp>
#include <OmniDSP/core_types.hpp>
#include <OmniDSP/cqt.hpp>
#include <OmniDSP/fft.hpp>
#include <OmniDSP/filter.hpp>
#include <OmniDSP/resample.hpp>
#include <OmniDSP/window.hpp>
#include <memory>
#include <span>
#include <vector>

#include "../interface/backend.hpp"

namespace OmniDSP::Default {
  // Forward declare Plan implementations
  template <typename T>
  class FFTPlanImpl;
  template <typename T>
  class RFFTPlanImpl;
  template <typename T>
  class CQTPlanImpl;
  template <typename T>
  class ResamplePlanImpl;
  template <typename T>
  class ConvolutionPlanImpl;
  template <typename T>
  class CorrelationPlanImpl;
  template <typename T>
  class FIRFilterPlanImpl;
  template <typename T>
  class IIRFilterPlanImpl;

  class DefaultBackend : public ::OmniDSP::Abstract::Backend {
   public:
    DefaultBackend();
    ~DefaultBackend() override;

    ::OmniDSP::BackendType get_backend() const override;

    // --- One-off Operations ---
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

    // --- Window Generation ---
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
    [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C32>>>
    create_fft_plan_c32(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C64>>>
    create_fft_plan_c64(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F32>>>
    create_rfft_plan_f32(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F64>>>
    create_rfft_plan_f64(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<F32>>>
    create_cqt_plan_f32(
        F32 sample_rate,
        F32 min_freq,
        F32 max_freq,
        int bins_per_octave,
        const WindowSpec& window_spec) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<F64>>>
    create_cqt_plan_f64(
        F64 sample_rate,
        F64 min_freq,
        F64 max_freq,
        int bins_per_octave,
        const WindowSpec& window_spec) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<F32>>>
    create_resample_plan_f32(const ResampleSpec& spec) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<F64>>>
    create_resample_plan_f64(const ResampleSpec& spec) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<F32>>>
    create_convolution_plan_f32(
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<F64>>>
    create_convolution_plan_f64(
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<C32>>>
    create_convolution_plan_c32(
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<C64>>>
    create_convolution_plan_c64(
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<F32>>>
    create_correlation_plan_f32(
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<F64>>>
    create_correlation_plan_f64(
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<C32>>>
    create_correlation_plan_c32(
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<C64>>>
    create_correlation_plan_c64(
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<F32>>>
    create_fir_filter_plan_f32(const F32Vec& coefficients) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<F64>>>
    create_fir_filter_plan_f64(const F64Vec& coefficients) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<C32>>>
    create_fir_filter_plan_c32(const C32Vec& coefficients) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<C64>>>
    create_fir_filter_plan_c64(const C64Vec& coefficients) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F32>>>
    create_iir_filter_plan_f32(
        const std::vector<IIRFilterCoef>& sos_coefficients) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F64>>>
    create_iir_filter_plan_f64(
        const std::vector<IIRFilterCoef>& sos_coefficients) const override;

    // --- Filter Design ---
    // *** UPDATED Return Type for FIR ***
    [[nodiscard]] OmniExpected<FIRCoefs<F32>> design_fir_filter_f32(
        const FIRFilterSpec& spec) const override;
    [[nodiscard]] OmniExpected<FIRCoefs<F64>> design_fir_filter_f64(
        const FIRFilterSpec& spec) const override;
    [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>>
    design_iir_filter_f32(const IIRFilterSpec& spec) const override;
    [[nodiscard]] OmniExpected<std::vector<IIRFilterCoef>>
    design_iir_filter_f64(const IIRFilterSpec& spec) const override;

    // --- Internal window generation helper override ---
    [[nodiscard]] Status generate_window_f32(
        const WindowSpec& spec, std::span<F32> output) const override;
    [[nodiscard]] Status generate_window_f64(
        const WindowSpec& spec, std::span<F64> output) const override;

   private:
    // --- Internal Implementation Factories (Specific to DefaultBackend) ---
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>
    create_fft_plan_impl_c32(size_t length) const;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>
    create_fft_plan_impl_c64(size_t length) const;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>
    create_rfft_plan_impl_f32(size_t length) const;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>
    create_rfft_plan_impl_f64(size_t length) const;
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
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::CQTPlanImpl<F32>>>
    create_cqt_plan_impl_f32(
        F32 sample_rate,
        F32 min_freq,
        F32 max_freq,
        int bins_per_octave,
        const WindowSpec& window_spec) const;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::CQTPlanImpl<F64>>>
    create_cqt_plan_impl_f64(
        F64 sample_rate,
        F64 min_freq,
        F64 max_freq,
        int bins_per_octave,
        const WindowSpec& window_spec) const;
  };

  std::unique_ptr<Abstract::Backend> Abstract::create_default_backend();

}  // namespace OmniDSP::Default

#endif  // OMNIDSP_DEFAULT_BACKEND_HPP
