/**
 * @file backend.hpp (IntelIPP)
 * @brief Declares the concrete Intel IPP backend implementation class.
 * @details This class inherits from Default::Backend and overrides functions
 * where Intel IPP provides optimized implementations (FFT, specific windowing,
 * resampling, filtering). Other functionalities are inherited from
 * Default::Backend.
 */

#ifndef OMNIDSP_INTELIPP_BACKEND_HPP
#define OMNIDSP_INTELIPP_BACKEND_HPP

#include <memory>  // For std::unique_ptr
#include <span>    // For std::span
#include <vector>  // For vector types

#include "../default/backend.hpp"  // Inherits from Default::Backend

// Include necessary types referenced in method signatures
#include <OmniDSP/core_types.hpp>
#include <OmniDSP/fft.hpp>  // For public FFTPlan, RFFTPlan (though we return Impl)
#include <OmniDSP/fir_filter.hpp>  // For public FIRFilterPlan, IIRFilterPlan, IIRFilterCoef
#include <OmniDSP/iir_filter.hpp>  // For public FIRFilterPlan, IIRFilterPlan, IIRFilterCoef
#include <OmniDSP/resample.hpp>  // For public ResamplePlan, Design::Resample
#include <OmniDSP/window.hpp>    // For WindowSetup, specific window params

// Forward declare or include Abstract Plan Impl types (needed for return types)
// These are defined in "omnidsp/interface/backend.hpp"
// #include "omnidsp/interface/backend.hpp" // Already included via
// default/backend.hpp

// Include the headers declaring the IntelIPP Plan *Implementations*
// These define classes like IntelIPP::FFTPlanImpl<T>
#include "fft.hpp"  // Defines IntelIPP::FFTPlanImpl, IntelIPP::RFFTPlanImpl
#include "fir_filter.hpp"  // Defines IntelIPP::FIRFilterPlanImpl
#include "iir_filter.hpp"  // Defines IntelIPP::IIRFilterPlanImpl
#include "resample.hpp"    // Defines IntelIPP::ResamplePlanImpl
// "window.hpp" for IntelIPP might contain helpers if specific window overrides
// call them.

namespace OmniDSP::IntelIPP {

  //--------------------------------------------------------------------------
  // IntelIPP Main Backend Implementation Class
  //--------------------------------------------------------------------------

  /**
   * @brief Concrete Intel IPP backend implementation. Inherits from
   * Default::Backend.
   */
  class Backend final : public Default::Backend {
   public:
    // --- Constructor / Destructor ---
    Backend();
    ~Backend() override;

    // --- Core ---
    BackendType get_backend() const override;

    // --- Override functions optimized by Intel IPP ---
    // If a function is NOT overridden, the Default::Backend implementation is
    // used.

    // Window Generation (Override functions implemented efficiently in IPP)
    // These override the specific virtual window functions from
    // Abstract::Backend (which Default::Backend also overrides).
    [[nodiscard]] Status bartlett_window_f32(
        size_t length, std::span<F32> output) const override;
    [[nodiscard]] Status bartlett_window_f64(
        size_t length, std::span<F64> output) const override;
    [[nodiscard]] Status blackman_window_f32(
        size_t length, std::span<F32> output) const override;
    [[nodiscard]] Status blackman_window_f64(
        size_t length, std::span<F64> output) const override;
    // Flattop and Gaussian are inherited from Default::Backend as per original
    // file.
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

    // --- Plan Impl Factories (Overrides) ---
    // These methods override the corresponding pure virtual
    // `create_*_plan_impl_*` methods from Abstract::Backend (which
    // Default::Backend provides default implementations for). They return
    // unique_ptr to the backend-specific *Impl objects.
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C32>>>
    create_fft_plan_impl_c32(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::FFTPlanImpl<C64>>>
    create_fft_plan_impl_c64(size_t length) const override;

    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F32>>>
    create_rfft_plan_impl_f32(size_t length) const override;
    [[nodiscard]] OmniExpected<std::unique_ptr<Abstract::RFFTPlanImpl<F64>>>
    create_rfft_plan_impl_f64(size_t length) const override;

    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::ResampleProcessorImpl<F32>>>
    create_resample_plan_impl_f32(const Design::Resample& spec) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::ResampleProcessorImpl<F64>>>
    create_resample_plan_impl_f64(const Design::Resample& spec) const override;

    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::FIRFilterProcessorImpl<F32>>>
    create_fir_filter_plan_impl_f32(const F32Vec& coefficients) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::FIRFilterProcessorImpl<F64>>>
    create_fir_filter_plan_impl_f64(const F64Vec& coefficients) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::FIRFilterProcessorImpl<C32>>>
    create_fir_filter_plan_impl_c32(const C32Vec& coefficients) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::FIRFilterProcessorImpl<C64>>>
    create_fir_filter_plan_impl_c64(const C64Vec& coefficients) const override;

    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::IIRFilterProcessorImpl<F32>>>
    create_iir_filter_plan_impl_f32(
        const std::vector<IIRFilterCoef>& sos_coefficients) const override;
    [[nodiscard]] OmniExpected<
        std::unique_ptr<Abstract::IIRFilterProcessorImpl<F64>>>
    create_iir_filter_plan_impl_f64(
        const std::vector<IIRFilterCoef>& sos_coefficients) const override;

    // NOTE: create_cqt_plan_impl_*, create_convolution_plan_impl_*,
    // create_correlation_plan_impl_* are NOT overridden here, so they will be
    // inherited from Default::Backend. If IntelIPP had specialized versions,
    // they would be overridden here.

    // NOTE: One-off operations (convolve_*, fft_*, correlate_*) and filter
    // design methods are inherited from Default::Backend as they are not
    // overridden here.

   private:
    // Any private members specific to the IntelIPP backend's state (if any)
  };

  // Factory function declaration (definition in .cpp file)
  // This still returns a pointer to the ABSTRACT base, which is correct.
  // This function is typically declared in "omnidsp/interface/backend.hpp"
  // and defined in the intelipp backend's .cpp file.
  // For completeness, if it's specific to this backend's header:
  // std::unique_ptr<Abstract::Backend> create_intelipp_backend();

}  // namespace OmniDSP::IntelIPP

#endif  // OMNIDSP_INTELIPP_BACKEND_HPP
