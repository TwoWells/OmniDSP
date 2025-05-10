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
#include <OmniDSP/core_types.hpp>  // For Status, OmniExpected, F32, C32, BackendType etc.

// Forward declarations for Spec and Plan types used in the interface.
// The full definitions will be included in the .cpp files of concrete backends
// or in the public API headers that use this interface (like Plan headers).
namespace OmniDSP {
  // From core_types.hpp (already included, but good to be aware of what's used)
  // enum class Status;
  // template <typename T> class OmniExpected;
  // using F32 = float; ...

  // From window.hpp
  struct WindowSetup;  // Forward declaration

  // From filter.hpp
  struct FIRFilterSpec;  // Forward declaration
  struct IIRFilterSpec;  // Forward declaration
  struct IIRFilterCoef;  // Forward declaration
  template <typename T>
  using FIRCoefs
      = std::vector<T>;  // Alias can be here if T is known (it is via F32/F64)
                         // Or FIRCoefs itself would need to be forward declared
                         // if it's a class template. Since it's a using alias,
                         // this is fine.

  // From resample.hpp
  struct ResampleSpec;  // Forward declaration

  // From convolution.hpp
  enum class ConvolutionType;    // Forward declaration
  enum class ConvolutionMethod;  // Forward declaration

  // Public Plan classes (forward declared as they appear in return types of
  // OmniDSP facade, but Abstract::Backend primarily deals with Impl classes for
  // creation)
  template <typename T>
  class FFTPlan;
  template <typename T>
  class RFFTPlan;
  template <typename T>
  class CQTPlan;
  template <typename T>
  class ResamplePlan;
  template <typename T>
  class ConvolutionPlan;
  template <typename T>
  class CorrelationPlan;
  template <typename T>
  class FIRFilterPlan;
  template <typename T>
  class IIRFilterPlan;

}  // namespace OmniDSP

namespace OmniDSP::Abstract {

  //--------------------------------------------------------------------------
  // Plan Implementation Interfaces (Abstract Base Classes)
  // These define the private PIMPL interfaces for Plan objects.
  // They use types from core_types.hpp and forward-declared Spec types.
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
  class ResamplePlanImpl {
   public:
    virtual ~ResamplePlanImpl() = default;
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
    virtual ConvolutionType get_type() const = 0;  // Uses forward-declared enum
    virtual size_t get_output_length(size_t input_length) const = 0;
    [[nodiscard]] virtual std::span<const T> get_kernel() const = 0;
    [[nodiscard]] virtual ConvolutionMethod get_method() const
        = 0;  // Uses forward-declared enum
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
  class FIRFilterPlanImpl {
   public:
    virtual ~FIRFilterPlanImpl() = default;
    [[nodiscard]] virtual Status execute(
        std::span<const T> input, std::span<T> output)
        = 0;
    [[nodiscard]] virtual Status reset() = 0;
    virtual size_t get_order() const = 0;
    virtual size_t get_num_taps() const = 0;
  };

  template <typename T>
  class IIRFilterPlanImpl {
   public:
    virtual ~IIRFilterPlanImpl() = default;
    [[nodiscard]] virtual Status execute(
        std::span<const T> input, std::span<T> output)
        = 0;
    [[nodiscard]] virtual Status reset() = 0;
    virtual size_t get_order() const = 0;
    virtual size_t get_num_sections() const = 0;
  };

  template <typename T>
  class CQTPlanImpl {
   public:
    using Complex = Utils::GetComplexType<T>;
    virtual ~CQTPlanImpl() = default;
    [[nodiscard]] virtual Status execute(
        std::span<const T> input, std::span<Complex> output) const
        = 0;
    virtual size_t get_num_bins() const = 0;
    virtual size_t get_num_output_frames(size_t input_length) const = 0;
    virtual size_t get_hop_length() const = 0;
  };

  //--------------------------------------------------------------------------
  // Main Backend Implementation Interface (Abstract Base Class)
  // All methods are pure virtual.
  //--------------------------------------------------------------------------
  class Backend {
   public:
    virtual ~Backend() = default;
    virtual BackendType get_backend() const = 0;

    // --- One-off DSP Operations ---
    [[nodiscard]] virtual OmniExpected<F32Vec> convolve_f32(
        const F32Vec& input,
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const
        = 0;
    [[nodiscard]] virtual OmniExpected<F64Vec> convolve_f64(
        const F64Vec& input,
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C32Vec> convolve_c32(
        const C32Vec& input,
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C64Vec> convolve_c64(
        const C64Vec& input,
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const
        = 0;

    [[nodiscard]] virtual OmniExpected<F32Vec> correlate_f32(
        const F32Vec& input,
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const
        = 0;
    [[nodiscard]] virtual OmniExpected<F64Vec> correlate_f64(
        const F64Vec& input,
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C32Vec> correlate_c32(
        const C32Vec& input,
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C64Vec> correlate_c64(
        const C64Vec& input,
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const
        = 0;

    [[nodiscard]] virtual OmniExpected<C32Vec> fft_c32(
        const C32Vec& input) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C64Vec> fft_c64(
        const C64Vec& input) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C32Vec> ifft_c32(
        const C32Vec& input) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C64Vec> ifft_c64(
        const C64Vec& input) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C32Vec> rfft_f32(
        const F32Vec& input) const
        = 0;
    [[nodiscard]] virtual OmniExpected<C64Vec> rfft_f64(
        const F64Vec& input) const
        = 0;
    [[nodiscard]] virtual OmniExpected<F32Vec> irfft_c32(
        const C32Vec& input, size_t output_length) const
        = 0;
    [[nodiscard]] virtual OmniExpected<F64Vec> irfft_c64(
        const C64Vec& input, size_t output_length) const
        = 0;

    // --- Specific Window Generation Methods (Pure Virtual) ---
    [[nodiscard]] virtual Status bartlett_window_f32(
        size_t length, std::span<F32> output) const
        = 0;
    [[nodiscard]] virtual Status bartlett_window_f64(
        size_t length, std::span<F64> output) const
        = 0;
    [[nodiscard]] virtual Status blackman_window_f32(
        size_t length, std::span<F32> output) const
        = 0;
    [[nodiscard]] virtual Status blackman_window_f64(
        size_t length, std::span<F64> output) const
        = 0;
    [[nodiscard]] virtual Status flattop_window_f32(
        size_t length, std::span<F32> output) const
        = 0;
    [[nodiscard]] virtual Status flattop_window_f64(
        size_t length, std::span<F64> output) const
        = 0;
    [[nodiscard]] virtual Status gaussian_window_f32(
        size_t length, double stddev, std::span<F32> output) const
        = 0;
    [[nodiscard]] virtual Status gaussian_window_f64(
        size_t length, double stddev, std::span<F64> output) const
        = 0;
    [[nodiscard]] virtual Status hamming_window_f32(
        size_t length, std::span<F32> output) const
        = 0;
    [[nodiscard]] virtual Status hamming_window_f64(
        size_t length, std::span<F64> output) const
        = 0;
    [[nodiscard]] virtual Status hann_window_f32(
        size_t length, std::span<F32> output) const
        = 0;
    [[nodiscard]] virtual Status hann_window_f64(
        size_t length, std::span<F64> output) const
        = 0;
    [[nodiscard]] virtual Status kaiser_window_f32(
        size_t length, double beta, std::span<F32> output) const
        = 0;
    [[nodiscard]] virtual Status kaiser_window_f64(
        size_t length, double beta, std::span<F64> output) const
        = 0;
    [[nodiscard]] virtual Status rectangular_window_f32(
        size_t length, std::span<F32> output) const
        = 0;
    [[nodiscard]] virtual Status rectangular_window_f64(
        size_t length, std::span<F64> output) const
        = 0;
    [[nodiscard]] virtual Status triangular_window_f32(
        size_t length, std::span<F32> output) const
        = 0;
    [[nodiscard]] virtual Status triangular_window_f64(
        size_t length, std::span<F64> output) const
        = 0;

    // --- Plan Impl Factory Methods ---
    [[nodiscard]] virtual OmniExpected<std::unique_ptr<FFTPlanImpl<C32>>>
    create_fft_plan_impl_c32(size_t length) const = 0;
    [[nodiscard]] virtual OmniExpected<std::unique_ptr<FFTPlanImpl<C64>>>
    create_fft_plan_impl_c64(size_t length) const = 0;

    [[nodiscard]] virtual OmniExpected<std::unique_ptr<RFFTPlanImpl<F32>>>
    create_rfft_plan_impl_f32(size_t length) const = 0;
    [[nodiscard]] virtual OmniExpected<std::unique_ptr<RFFTPlanImpl<F64>>>
    create_rfft_plan_impl_f64(size_t length) const = 0;

    [[nodiscard]] virtual OmniExpected<std::unique_ptr<CQTPlanImpl<F32>>>
    create_cqt_plan_impl_f32(
        F32 sample_rate,
        F32 min_freq,
        F32 max_freq,
        int bins_per_octave,
        const WindowSetup& window_setup) const
        = 0;
    [[nodiscard]] virtual OmniExpected<std::unique_ptr<CQTPlanImpl<F64>>>
    create_cqt_plan_impl_f64(
        F64 sample_rate,
        F64 min_freq,
        F64 max_freq,
        int bins_per_octave,
        const WindowSetup& window_setup) const
        = 0;

    [[nodiscard]] virtual OmniExpected<std::unique_ptr<ResamplePlanImpl<F32>>>
    create_resample_plan_impl_f32(const ResampleSpec& spec) const = 0;
    [[nodiscard]] virtual OmniExpected<std::unique_ptr<ResamplePlanImpl<F64>>>
    create_resample_plan_impl_f64(const ResampleSpec& spec) const = 0;

    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<ConvolutionPlanImpl<F32>>>
    create_convolution_plan_impl_f32(
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const
        = 0;
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<ConvolutionPlanImpl<F64>>>
    create_convolution_plan_impl_f64(
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const
        = 0;
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<ConvolutionPlanImpl<C32>>>
    create_convolution_plan_impl_c32(
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const
        = 0;
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<ConvolutionPlanImpl<C64>>>
    create_convolution_plan_impl_c64(
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const
        = 0;

    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<CorrelationPlanImpl<F32>>>
    create_correlation_plan_impl_f32(
        const F32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const
        = 0;
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<CorrelationPlanImpl<F64>>>
    create_correlation_plan_impl_f64(
        const F64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const
        = 0;
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<CorrelationPlanImpl<C32>>>
    create_correlation_plan_impl_c32(
        const C32Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const
        = 0;
    [[nodiscard]] virtual OmniExpected<
        std::unique_ptr<CorrelationPlanImpl<C64>>>
    create_correlation_plan_impl_c64(
        const C64Vec& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto) const
        = 0;

    [[nodiscard]] virtual OmniExpected<std::unique_ptr<FIRFilterPlanImpl<F32>>>
    create_fir_filter_plan_impl_f32(const F32Vec& coefficients) const = 0;
    [[nodiscard]] virtual OmniExpected<std::unique_ptr<FIRFilterPlanImpl<F64>>>
    create_fir_filter_plan_impl_f64(const F64Vec& coefficients) const = 0;
    [[nodiscard]] virtual OmniExpected<std::unique_ptr<FIRFilterPlanImpl<C32>>>
    create_fir_filter_plan_impl_c32(const C32Vec& coefficients) const = 0;
    [[nodiscard]] virtual OmniExpected<std::unique_ptr<FIRFilterPlanImpl<C64>>>
    create_fir_filter_plan_impl_c64(const C64Vec& coefficients) const = 0;

    [[nodiscard]] virtual OmniExpected<std::unique_ptr<IIRFilterPlanImpl<F32>>>
    create_iir_filter_plan_impl_f32(
        const std::vector<IIRFilterCoef>& sos_coefficients) const
        = 0;
    [[nodiscard]] virtual OmniExpected<std::unique_ptr<IIRFilterPlanImpl<F64>>>
    create_iir_filter_plan_impl_f64(
        const std::vector<IIRFilterCoef>& sos_coefficients) const
        = 0;

    // --- Filter Design ---
    [[nodiscard]] virtual OmniExpected<FIRCoefs<F32>> design_fir_filter_f32(
        const FIRFilterSpec& spec) const
        = 0;
    [[nodiscard]] virtual OmniExpected<FIRCoefs<F64>> design_fir_filter_f64(
        const FIRFilterSpec& spec) const
        = 0;
    [[nodiscard]] virtual OmniExpected<std::vector<IIRFilterCoef>>
    design_iir_filter_f32(const IIRFilterSpec& spec) const = 0;
    [[nodiscard]] virtual OmniExpected<std::vector<IIRFilterCoef>>
    design_iir_filter_f64(const IIRFilterSpec& spec) const = 0;

   protected:
    Backend() = default;
  };

  // --- Backend Factory Function Declarations ---
  std::unique_ptr<Backend> create_default_backend();
  std::unique_ptr<Backend> create_accelerate_backend();
  std::unique_ptr<Backend> create_onemkl_backend();
  std::unique_ptr<Backend> create_intelipp_backend();

}  // namespace OmniDSP::Abstract

#endif  // OMNIDSP_ABSTRACT_BACKEND_HPP
