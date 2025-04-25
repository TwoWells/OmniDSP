/**
 * @file backend.h (accelerate)
 * @brief Declares the concrete Accelerate backend implementation class.
 * @details This class inherits from DefaultBackend and overrides functions
 * where Accelerate framework provides optimized implementations.
 */

#ifndef OMNIDSP_ACCELERATE_BACKEND_H
#define OMNIDSP_ACCELERATE_BACKEND_H

#ifdef OMNIDSP_USE_ACCELERATE  // << CHECK THIS FLAG NAME is correct

#include <Accelerate/Accelerate.h>  // Main Accelerate framework header

#include <complex>
#include <memory>  // For std::unique_ptr
#include <vector>

#include "../default/backend.h"  // Inherit from DefaultBackend

namespace OmniDSP {
  namespace backend {

    // Forward declare Accelerate Plan Impls (declarations might be here or
    // included)
    template <typename T>
    class AccelerateFFTPlanImpl;
    template <typename T>
    class AccelerateRFFTPlanImpl;
    template <typename T>
    class AccelerateResamplePlanImpl;
    template <typename T>
    class AccelerateConvolutionPlanImpl;
    template <typename T>
    class AccelerateCorrelationPlanImpl;
    template <typename T>
    class AccelerateFIRFilterPlanImpl;
    template <typename T>
    class AccelerateIIRFilterPlanImpl;
    // No AccelerateCQTPlanImpl needed if using default CQT logic

    //--------------------------------------------------------------------------
    // Accelerate Main Backend Implementation Class
    //--------------------------------------------------------------------------

    /** @brief Concrete Accelerate backend implementation. Inherits from
     * DefaultBackend. */
    class AccelerateBackend final
        : public DefaultBackend {  // Renamed and inherits from DefaultBackend
     public:
      // --- Constructor / Destructor ---
      AccelerateBackend();            // Renamed
      ~AccelerateBackend() override;  // Renamed

      // --- Core ---
      Backend get_backend() const override;

      // --- Override functions optimized by Accelerate ---
      // --- If a function is NOT overridden, the DefaultBackend implementation
      // is used ---

      // Window Generation (Override functions implemented efficiently in vDSP)
      [[nodiscard]] Status bartlett_window_f32(
          size_t length, std::span<F32> output) const override;
      [[nodiscard]] Status bartlett_window_f64(
          size_t length, std::span<F64> output) const override;
      [[nodiscard]] Status blackman_window_f32(
          size_t length, std::span<F32> output) const override;
      [[nodiscard]] Status blackman_window_f64(
          size_t length, std::span<F64> output) const override;
      [[nodiscard]] Status hamming_window_f32(
          size_t length, std::span<F32> output) const override;
      [[nodiscard]] Status hamming_window_f64(
          size_t length, std::span<F64> output) const override;
      [[nodiscard]] Status hann_window_f32(
          size_t length, std::span<F32> output) const override;
      [[nodiscard]] Status hann_window_f64(
          size_t length, std::span<F64> output) const override;
      [[nodiscard]] Status rectangular_window_f32(
          size_t length, std::span<F32> output) const override;
      [[nodiscard]] Status rectangular_window_f64(
          size_t length, std::span<F64> output) const override;
      // NOTE: flattop, gaussian, kaiser, triangular are NOT overridden here,
      // will use DefaultBackend impl.

      // Plan Factories (MUST override to return Accelerate plans)
      [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C32>>>
      create_fft_plan_c32(size_t length) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C64>>>
      create_fft_plan_c64(size_t length) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F32>>>
      create_rfft_plan_f32(size_t length) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F64>>>
      create_rfft_plan_f64(size_t length) const override;
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
      [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F32>>>
      create_iir_filter_plan_f32(const std::vector<SecondOrderSection<F32>>&
                                     sos_coefficients) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F64>>>
      create_iir_filter_plan_f64(const std::vector<SecondOrderSection<F64>>&
                                     sos_coefficients) const override;

      // Internal Helpers (Override if needed)
      [[nodiscard]] OmniExpected<F32Vec> generate_window_vec_f32(
          const WindowSpec<F32>& spec, size_t length) const override;
      [[nodiscard]] OmniExpected<F64Vec> generate_window_vec_f64(
          const WindowSpec<F64>& spec, size_t length) const override;

      // NOTE: One-off operations (convolve_*, fft_*, etc.) are NOT overridden
      // here. They will use the DefaultBackend implementation (which
      // creates/uses plans via the overridden factories above). Only override
      // them if Accelerate has a *direct* one-off function that's faster.

      // NOTE: Filter design methods are NOT overridden, will use DefaultBackend
      // impl. NOTE: CQT Plan factory and impl helper are NOT overridden, will
      // use DefaultBackend impl (which will correctly use the overridden
      // Accelerate FFT/Resample plans).

     private:
      // Any private members specific to the Accelerate backend's state (if any)
    };

  }  // namespace backend
}  // namespace OmniDSP

#endif  // OMNIDSP_USE_ACCELERATE
#endif  // OMNIDSP_ACCELERATE_BACKEND_H
