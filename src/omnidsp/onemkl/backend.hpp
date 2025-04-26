/**
 * @file backend.h (onemkl)
 * @brief Declares the concrete oneMKL backend implementation class.
 * @details This class inherits from DefaultBackend and overrides functions
 * where oneMKL (DFTI, VML, IPP) provides optimized implementations.
 */

#ifndef OMNIDSP_ONEMKL_BACKEND_H
#define OMNIDSP_ONEMKL_BACKEND_H

// Only compile this file if oneMKL backend is enabled via CMake
#ifdef OMNIDSP_USE_ONEMKL  // << CHECK THIS FLAG NAME is correct

#include <ipps.h>  // Include IPP explicitly for ResamplePlanImpl etc.
#include <mkl.h>   // Main oneMKL header (includes DFTI, VML, etc.)

#include <complex>
#include <memory>  // For std::unique_ptr
#include <vector>

#include "../default/backend.hpp"  // Inherit from DefaultBackend

namespace OmniDSP {
  namespace backend {

    // Forward declare oneMKL Plan Impls (declarations might be here or
    // included)
    template <typename T>
    class OneMKLFFTPlanImpl;
    template <typename T>
    class OneMKLRFFTPlanImpl;
    template <typename T>
    class OneMKLResamplePlanImpl;
    template <typename T>
    class OneMKLConvolutionPlanImpl;
    template <typename T>
    class OneMKLCorrelationPlanImpl;
    template <typename T>
    class OneMKLFIRFilterPlanImpl;
    template <typename T>
    class OneMKLIIRFilterPlanImpl;
    // No OneMKLCQTPlanImpl needed if using default CQT logic

    //--------------------------------------------------------------------------
    // oneMKL Main Backend Implementation Class
    //--------------------------------------------------------------------------

    /** @brief Concrete oneMKL backend implementation. Inherits from
     * DefaultBackend. */
    class OneMKLBackend final
        : public DefaultBackend {  // Renamed and inherits from DefaultBackend
     public:
      // --- Constructor / Destructor ---
      OneMKLBackend();            // Renamed
      ~OneMKLBackend() override;  // Renamed

      // --- Core ---
      Backend get_backend() const override;

      // --- Override functions optimized by oneMKL (DFTI, IPP, VML) ---
      // --- If a function is NOT overridden, the DefaultBackend implementation
      // is used ---

      // Window Generation (Override functions implemented efficiently in
      // IPP/VML)
      [[nodiscard]] Status bartlett_window_f32(
          size_t length, std::span<F32> output) const override;
      [[nodiscard]] Status bartlett_window_f64(
          size_t length, std::span<F64> output) const override;
      [[nodiscard]] Status blackman_window_f32(
          size_t length, std::span<F32> output) const override;
      [[nodiscard]] Status blackman_window_f64(
          size_t length, std::span<F64> output) const override;
      [[nodiscard]] Status flattop_window_f32(
          size_t length, std::span<F32> output) const override;  // Use VML
      [[nodiscard]] Status flattop_window_f64(
          size_t length, std::span<F64> output) const override;  // Use VML
      [[nodiscard]] Status gaussian_window_f32(
          size_t length,
          double stddev,
          std::span<F32> output) const override;  // Use VML
      [[nodiscard]] Status gaussian_window_f64(
          size_t length,
          double stddev,
          std::span<F64> output) const override;  // Use VML
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
          size_t length, std::span<F32> output) const override;  // Use IPP/VML
      [[nodiscard]] Status rectangular_window_f64(
          size_t length, std::span<F64> output) const override;  // Use IPP/VML
      [[nodiscard]] Status triangular_window_f32(
          size_t length,
          std::span<F32> output) const override;  // Reuse Bartlett
      [[nodiscard]] Status triangular_window_f64(
          size_t length,
          std::span<F64> output) const override;  // Reuse Bartlett

      // Plan Factories (MUST override to return oneMKL plans)
      [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C32>>>
      create_fft_plan_c32(size_t length) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<FFTPlan<C64>>>
      create_fft_plan_c64(size_t length) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F32>>>
      create_rfft_plan_f32(size_t length) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<RFFTPlan<F64>>>
      create_rfft_plan_f64(size_t length) const override;
      [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<F32>>>
      create_resample_plan_f32(
          const ResampleSpec& spec) const override;  // Uses IPP
      [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<F64>>>
      create_resample_plan_f64(
          const ResampleSpec& spec) const override;  // Uses IPP
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
      // Override FIR/IIR plan factories if oneMKL implementations are provided
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

      // Internal Helpers (Override if needed, otherwise default is used)
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

      // Filter Design (Override if oneMKL provides optimized design functions)
      // [[nodiscard]] OmniExpected<F32Vec> design_fir_filter_f32(const
      // FIRFilterSpec<F32>& spec) const override;
      // [[nodiscard]] OmniExpected<F64Vec> design_fir_filter_f64(const
      // FIRFilterSpec<F64>& spec) const override;
      // [[nodiscard]] OmniExpected<std::vector<SecondOrderSection<F32>>>
      // design_iir_filter_f32(const IIRFilterSpec<F32>& spec) const override;
      // [[nodiscard]] OmniExpected<std::vector<SecondOrderSection<F64>>>
      // design_iir_filter_f64(const IIRFilterSpec<F64>& spec) const override;

      // NOTE: One-off operations (convolve_*, fft_*, etc.) are NOT overridden
      // here by default. They will use the DefaultBackend implementation (which
      // creates/uses plans via the overridden factories above). Only override
      // them if oneMKL has a *direct* one-off function that's faster.

      // NOTE: CQT Plan factory and impl helper are NOT overridden, will use
      // DefaultBackend impl (which will correctly use the overridden oneMKL
      // FFT/Resample plans).

     private:
      // Any private members specific to the oneMKL backend's state (if any)
    };

  }  // namespace backend
}  // namespace OmniDSP

#endif  // OMNIDSP_USE_ONEMKL
#endif  // OMNIDSP_ONEMKL_BACKEND_H
