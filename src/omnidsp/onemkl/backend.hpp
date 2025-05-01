/**
 * @file backend.hpp (onemkl)
 * @brief Declares the concrete oneMKL backend implementation class.
 * @details This class inherits from DefaultBackend and overrides functions
 * where oneMKL (DFTI, VML, IPP) provides optimized implementations.
 */

#ifndef OMNIDSP_ONEMKL_BACKEND_HPP
#define OMNIDSP_ONEMKL_BACKEND_HPP

// Only compile this file if oneMKL backend is enabled via CMake
#ifdef OMNIDSP_USE_ONEMKL  // << Ensure this CMake flag name is correct

#include <memory>  // For std::unique_ptr
#include <span>    // For std::span
#include <vector>  // For vector types

#include "../default/backend.hpp"  // Inherit from DefaultBackend
// Include necessary types referenced in method signatures
#include <OmniDSP/convolution.hpp>
#include <OmniDSP/core_types.hpp>
#include <OmniDSP/cqt.hpp>
#include <OmniDSP/fft.hpp>
#include <OmniDSP/filter.hpp>
#include <OmniDSP/resample.hpp>
#include <OmniDSP/window.hpp>

// Include the headers declaring the oneMKL Plan implementations
#include "convolution.hpp"
#include "fft.hpp"
#include "filter.hpp"  // *** ADDED: Include the new filter header ***
#include "resample.hpp"
#include "window.hpp"

namespace OmniDSP::backend {

  //--------------------------------------------------------------------------
  // oneMKL Main Backend Implementation Class
  //--------------------------------------------------------------------------

  /** @brief Concrete oneMKL backend implementation. Inherits from
   * DefaultBackend. */
  class OneMKLBackend final : public DefaultBackend {
   public:
    // --- Constructor / Destructor ---
    OneMKLBackend();
    ~OneMKLBackend() override;

    // --- Core ---
    // MUST override to identify this backend
    Backend get_backend() const override;

    // --- Override functions optimized by oneMKL (DFTI, IPP, VML) ---
    // --- If a function is NOT overridden, the DefaultBackend implementation is
    // used ---

    // Window Generation (Override functions implemented efficiently in IPP/VML)
    // Assumes implementations exist in onemkl/window.cpp
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
    create_fir_filter_plan_f32(
        const F32Vec& coefficients) const override;  // Uses IPP
    [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<F64>>>
    create_fir_filter_plan_f64(
        const F64Vec& coefficients) const override;  // Uses IPP
    [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<C32>>>
    create_fir_filter_plan_c32(
        const C32Vec& coefficients) const override;  // Uses IPP
    [[nodiscard]] OmniExpected<std::unique_ptr<FIRFilterPlan<C64>>>
    create_fir_filter_plan_c64(
        const C64Vec& coefficients) const override;  // Uses IPP
    [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F32>>>
    create_iir_filter_plan_f32(
        const std::vector<IIRFilterCoef>& sos_coefficients)
        const override;  // Uses IPP
    [[nodiscard]] OmniExpected<std::unique_ptr<IIRFilterPlan<F64>>>
    create_iir_filter_plan_f64(
        const std::vector<IIRFilterCoef>& sos_coefficients)
        const override;  // Uses IPP

    // NOTE: One-off operations (convolve_*, fft_*, etc.) are NOT overridden.
    // They will inherit the DefaultBackend implementation, which internally
    // uses the Plan factories overridden above.

    // NOTE: Filter design methods are NOT overridden. They will inherit
    // the DefaultBackend implementation.

    // NOTE: CQT Plan factory is NOT overridden. It will inherit the
    // DefaultBackend implementation, which will correctly use the overridden
    // oneMKL FFT/Resample plans created by *this* backend instance.

   private:
    // Any private members specific to the oneMKL backend's state (if any)
  };

}  // namespace OmniDSP::backend

#endif  // OMNIDSP_USE_ONEMKL
#endif  // OMNIDSP_ONEMKL_BACKEND_HPP
