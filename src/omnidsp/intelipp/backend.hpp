/**
 * @file backend.hpp (IntelIPP)
 * @brief Declares the concrete Intel IPP backend implementation class.
 * @details This class inherits from DefaultBackend and overrides functions
 * where Intel IPP provides optimized implementations (FFT, windowing,
 * resampling, filtering). Convolution/Correlation and CQT functions are
 * inherited.
 */

#ifndef OMNIDSP_INTELIPP_BACKEND_HPP
#define OMNIDSP_INTELIPP_BACKEND_HPP

#include <memory>  // For std::unique_ptr
#include <span>    // For std::span
#include <vector>  // For vector types

// *** Inherit from DefaultBackend instead of AbstractBackend ***
#include "../default/backend.hpp"

// Include necessary types referenced in method signatures
#include <OmniDSP/core_types.hpp>
#include <OmniDSP/fft.hpp>  // Needed for overridden FFT plans
#include <OmniDSP/filter.hpp>
#include <OmniDSP/resample.hpp>
#include <OmniDSP/window.hpp>
// Note: convolution.hpp and cqt.hpp are included via default/backend.hpp

// Include the headers declaring the IntelIPP Plan implementations
#include "fft.hpp"         // Defines IntelIPPFFTPlanImpl etc.
#include "fir_filter.hpp"  // Defines IntelIPPFIRFilterPlanImpl etc.
#include "iir_filter.hpp"  // Defines IntelIPPFIRFilterPlanImpl etc.
#include "resample.hpp"    // Defines IntelIPPResamplePlanImpl
#include "window.hpp"      // Defines IntelIPP window helpers (if overridden)

namespace OmniDSP::IntelIPP {

  //--------------------------------------------------------------------------
  // IntelIPP Main Backend Implementation Class
  //--------------------------------------------------------------------------

  /**
   * @brief Concrete Intel IPP backend implementation. Inherits from
   * DefaultBackend.
   */
  class Backend final : public Default::DefaultBackend {
   public:
    // --- Constructor / Destructor ---
    Backend();
    // Destructor override is inherited from DefaultBackend (which overrides
    // AbstractBackend)
    ~Backend() override;

    // --- Core ---
    // MUST override to identify this backend
    BackendType get_backend() const override;

    // --- Override functions optimized by Intel IPP ---
    // --- If a function is NOT overridden, the DefaultBackend implementation is
    // used ---

    // Window Generation (Override functions implemented efficiently in IPP)
    [[nodiscard]] Status bartlett_window_f32(
        size_t length,
        std::span<F32> output) const override;  // IPP specialized
    [[nodiscard]] Status bartlett_window_f64(
        size_t length,
        std::span<F64> output) const override;  // IPP specialized
    [[nodiscard]] Status blackman_window_f32(
        size_t length,
        std::span<F32> output) const override;  // IPP specialized
    [[nodiscard]] Status blackman_window_f64(
        size_t length,
        std::span<F64> output) const override;  // IPP specialized
    // [[nodiscard]] Status flattop_window_f32(...) const override; // NOT
    // specialized - Inherited from DefaultBackend
    // [[nodiscard]] Status flattop_window_f64(...) const override; // NOT
    // specialized - Inherited from DefaultBackend
    // [[nodiscard]] Status gaussian_window_f32(...) const override; // NOT
    // specialized - Inherited from DefaultBackend
    // [[nodiscard]] Status gaussian_window_f64(...) const override; // NOT
    // specialized - Inherited from DefaultBackend
    [[nodiscard]] Status hamming_window_f32(
        size_t length,
        std::span<F32> output) const override;  // IPP specialized
    [[nodiscard]] Status hamming_window_f64(
        size_t length,
        std::span<F64> output) const override;  // IPP specialized
    [[nodiscard]] Status hann_window_f32(size_t length, std::span<F32> output)
        const override;  // IPP specialized
    [[nodiscard]] Status hann_window_f64(size_t length, std::span<F64> output)
        const override;  // IPP specialized
    [[nodiscard]] Status kaiser_window_f32(
        size_t length,
        double beta,
        std::span<F32> output) const override;  // IPP specialized
    [[nodiscard]] Status kaiser_window_f64(
        size_t length,
        double beta,
        std::span<F64> output) const override;  // IPP specialized
    [[nodiscard]] Status rectangular_window_f32(
        size_t length,
        std::span<F32> output) const override;  // IPP specialized
    [[nodiscard]] Status rectangular_window_f64(
        size_t length,
        std::span<F64> output) const override;  // IPP specialized
    [[nodiscard]] Status triangular_window_f32(
        size_t length, std::span<F32> output)
        const override;  // IPP specialized (via Bartlett)
    [[nodiscard]] Status triangular_window_f64(
        size_t length, std::span<F64> output)
        const override;  // IPP specialized (via Bartlett)

    // Plan Factories (MUST override to return IntelIPP plans where applicable)
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

    // NOTE: Convolution/Correlation/CQT plan factories are NOT overridden.
    // They will inherit the DefaultBackend implementation, which will
    // use the overridden IntelIPP FFT/Resample plans created by *this* backend.

    // NOTE: One-off operations (convolve_*, fft_*, correlate_*) are NOT
    // declared here. They are inherited directly from DefaultBackend.

    // NOTE: Filter design methods are NOT declared here.
    // They are inherited directly from DefaultBackend.

   private:
    // Any private members specific to the IntelIPP backend's state (if any)
  };

  // Factory function declaration (definition in .cpp file)
  // This still returns a pointer to the ABSTRACT base, which is correct.
  std::unique_ptr<Abstract::Backend> create_intelipp_backend();

}  // namespace OmniDSP::IntelIPP

#endif  // OMNIDSP_INTELIPP_BACKEND_HPP
