/**
 * @file backend.h (default)
 * @brief Declares the concrete Default backend implementation class.
 * @details This class inherits from the abstract interface AbstractBackend
 * and provides portable implementations, potentially using Google Highway
 * for SIMD acceleration where applicable. It implements all pure virtual
 * functions from the base class.
 */

#ifndef OMNIDSP_DEFAULT_BACKEND_H
#define OMNIDSP_DEFAULT_BACKEND_H

#include <memory>  // For std::unique_ptr

#include "../interface/backend.hpp"  // Include the base AbstractBackend interface

namespace OmniDSP {
  namespace backend {

    // Forward declare the concrete Plan implementations for the Default backend
    template <typename T>
    class DefaultFFTPlanImpl;
    template <typename T>
    class DefaultRFFTPlanImpl;
    template <typename T>
    class DefaultCQTPlanImpl;
    template <typename T>
    class DefaultResamplePlanImpl;
    template <typename T>
    class DefaultConvolutionPlanImpl;
    template <typename T>
    class DefaultCorrelationPlanImpl;
    template <typename T>
    class DefaultFIRFilterPlanImpl;
    template <typename T>
    class DefaultIIRFilterPlanImpl;

    // --- Concrete Default Backend Implementation ---
    // Inherits from the abstract AbstractBackend interface
    class DefaultBackend final
        : public AbstractBackend {  // Renamed and inherits from AbstractBackend
     public:
      // --- Constructor / Destructor ---
      DefaultBackend();            // Renamed
      ~DefaultBackend() override;  // Renamed

      // --- Core ---
      Backend get_backend() const override;

      // --- Override ALL pure virtual functions from AbstractBackend ---
      // --- Plus, optionally provide implementations for non-pure virtuals (or
      // rely on AbstractBackend's defaults if they existed) ---
      // --- Since AbstractBackend now has virtual defaults for one-offs, we
      // don't *need* to declare them here unless we want to change the default
      // ---
      // --- We DO need to declare overrides for the PURE virtuals ---

      // Window Generation (Pure Virtual in Base)
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

      // Plan Factories (Pure Virtual in Base)
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
          const WindowSpec<F32>& window_spec) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlan<F64>>>
      create_cqt_plan_f64(
          F64 sample_rate,
          F64 min_freq,
          F64 max_freq,
          int bins_per_octave,
          const WindowSpec<F64>& window_spec) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<F32>>>
      create_resample_plan_f32(const ResampleSpec& spec) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<F64>>>
      create_resample_plan_f64(const ResampleSpec& spec) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<F32>>>
      create_convolution_plan_f32(
          const F32Vec& kernel, ConvolutionType mode) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<F64>>>
      create_convolution_plan_f64(
          const F64Vec& kernel, ConvolutionType mode) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<C32>>>
      create_convolution_plan_c32(
          const C32Vec& kernel, ConvolutionType mode) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<C64>>>
      create_convolution_plan_c64(
          const C64Vec& kernel, ConvolutionType mode) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<F32>>>
      create_correlation_plan_f32(
          const F32Vec& kernel, ConvolutionType mode) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<F64>>>
      create_correlation_plan_f64(
          const F64Vec& kernel, ConvolutionType mode) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<C32>>>
      create_correlation_plan_c32(
          const C32Vec& kernel, ConvolutionType mode) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<C64>>>
      create_correlation_plan_c64(
          const C64Vec& kernel, ConvolutionType mode) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<F32>>>
      create_fir_filter_plan_f32(const F32Vec& coefficients) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<F64>>>
      create_fir_filter_plan_f64(const F64Vec& coefficients) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<C32>>>
      create_fir_filter_plan_c32(const C32Vec& coefficients) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<C64>>>
      create_fir_filter_plan_c64(const C64Vec& coefficients) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F32>>>
      create_iir_filter_plan_f32(const std::vector<SecondOrderSection<F32>>&
                                     sos_coefficients) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F64>>>
      create_iir_filter_plan_f64(const std::vector<SecondOrderSection<F64>>&
                                     sos_coefficients) const override;

      // Filter Design (Pure Virtual in Base)
      [[nodiscard]] OmniExpected<F32Vec> design_fir_filter_f32(
          const FIRFilterSpec<F32>& spec) const override;
      [[nodiscard]] OmniExpected<F64Vec> design_fir_filter_f64(
          const FIRFilterSpec<F64>& spec) const override;
      [[nodiscard]] OmniExpected<std::vector<SecondOrderSection<F32>>>
      design_iir_filter_f32(const IIRFilterSpec<F32>& spec) const override;
      [[nodiscard]] OmniExpected<std::vector<SecondOrderSection<F64>>>
      design_iir_filter_f64(const IIRFilterSpec<F64>& spec) const override;

      // Internal Helpers (Pure Virtual in Base)
      [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlanImpl<F32>>>
      create_cqt_plan_impl_f32(
          F32 sample_rate,
          F32 min_freq,
          F32 max_freq,
          int bins_per_octave,
          const WindowSpec<F32>& window_spec) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<CQTPlanImpl<F64>>>
      create_cqt_plan_impl_f64(
          F64 sample_rate,
          F64 min_freq,
          F64 max_freq,
          int bins_per_octave,
          const WindowSpec<F64>& window_spec) const override;
      [[nodiscard]] OmniExpected<F32Vec> generate_window_vec_f32(
          const WindowSpec<F32>& spec, size_t length) const override;
      [[nodiscard]] OmniExpected<F64Vec> generate_window_vec_f64(
          const WindowSpec<F64>& spec, size_t length) const override;

     private:
      // Any private members specific to the default backend's state (if any)
    };

  }  // namespace backend
}  // namespace OmniDSP

#endif  // OMNIDSP_DEFAULT_BACKEND_H
