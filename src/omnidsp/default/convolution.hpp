#ifndef OMNIDSP_DEFAULT_CONVOLUTION_HPP
#define OMNIDSP_DEFAULT_CONVOLUTION_HPP

#include <OmniDSP/convolution.hpp>  // For ConvolutionType, ConvolutionMethod
#include <OmniDSP/core_types.hpp>   // For Status, F32, C32, etc.
#include <complex>
#include <memory>     // For std::unique_ptr
#include <span>       // For std::span
#include <stdexcept>  // For std::logic_error
#include <variant>    // Include for std::variant
#include <vector>

#include "../interface/backend.hpp"  // Defines ConvolutionPlanImpl/CorrelationPlanImpl and AbstractBackend

// Forward declare FFT plan impl BASE classes used internally
// These are defined in interface/backend.hpp
namespace OmniDSP::backend {
  template <typename T_Complex>
  class FFTPlanImpl;
  template <typename T_Real>
  class RFFTPlanImpl;
}  // namespace OmniDSP::backend

namespace OmniDSP::backend {

  /**
   * @brief Default backend implementation for Convolution Plan (Standard C++).
   * @details Uses FFT-based convolution internally, leveraging a provided FFT
   * plan implementation.
   * @tparam T Data type (F32, F64, C32, C64).
   */
  template <typename T>
  class DefaultConvolutionPlanImpl final : public ConvolutionPlanImpl<T> {
    // Define complex type corresponding to T
    using T_Complex = Detail::GetComplexT<T>;
    // Define real type corresponding to T
    using T_Real = Detail::GetRealT<T>;

   public:  // <-- Make variant public
    // Define the variant type to hold a pointer to the appropriate base FFT
    // plan implementation
    using FFTPlanImplVariant = std::variant<
        std::unique_ptr<FFTPlanImpl<T_Complex>>,  // For complex T
        std::unique_ptr<RFFTPlanImpl<T_Real>>     // For real T
        >;

    // public constructor and methods remain here
   public:
    /**
     * @brief Constructor.
     * @param fft_plan_variant Variant holding the unique_ptr to the appropriate
     * FFT plan implementation.
     * @param kernel The convolution kernel.
     * @param type The convolution boundary type (Full, Same, Valid).
     * @param method The requested convolution method (Auto, Direct, FFT).
     * Currently only FFT is implemented here.
     * @throws std::invalid_argument if kernel is empty or fft_plan_variant
     * holds null.
     * @throws std::length_error if calculated FFT length overflows.
     * @throws std::runtime_error on internal errors (e.g., allocation).
     */
    DefaultConvolutionPlanImpl(
        FFTPlanImplVariant&& fft_plan_variant,
        const std::vector<T>& kernel,
        ConvolutionType type,
        ConvolutionMethod method);
    ~DefaultConvolutionPlanImpl() override = default;  // Defined here

    Status execute(
        std::span<const T> input, std::span<T> output) const override;
    size_t get_kernel_length() const override;
    ConvolutionType get_type() const override;
    ConvolutionMethod get_method() const override;
    size_t get_output_length(size_t input_length) const override;
    std::span<const T> get_kernel() const override;

    // Disable copy/move operations
    DefaultConvolutionPlanImpl(const DefaultConvolutionPlanImpl&) = delete;
    DefaultConvolutionPlanImpl& operator=(const DefaultConvolutionPlanImpl&)
        = delete;
    DefaultConvolutionPlanImpl(DefaultConvolutionPlanImpl&&) = delete;
    DefaultConvolutionPlanImpl& operator=(DefaultConvolutionPlanImpl&&)
        = delete;

   private:
    ConvolutionType type_;
    ConvolutionMethod method_;
    std::vector<T> original_kernel_;
    size_t kernel_length_;
    size_t fft_length_;  // Pre-calculated FFT length based on kernel

    // Store the FFT of the (padded, reversed) kernel
    std::vector<T_Complex> kernel_fft_;

    // Single variant to hold the appropriate FFT plan implementation pointer
    FFTPlanImplVariant fft_plan_impl_variant_;  // Now stores the passed-in plan

    // Temporary Buffers
    mutable std::unique_ptr<T[]> input_padded_;
    mutable std::unique_ptr<T_Complex[]> input_fft_;
    mutable std::unique_ptr<T_Complex[]> product_fft_;
    mutable std::unique_ptr<T[]> result_ifft_;
  };

  /**
   * @brief Default backend implementation for Correlation Plan (Standard C++).
   * @details Uses FFT-based correlation internally, leveraging a provided FFT
   * plan implementation.
   * @tparam T Data type (F32, F64, C32, C64).
   */
  template <typename T>
  class DefaultCorrelationPlanImpl final : public CorrelationPlanImpl<T> {
    // Define complex type corresponding to T
    using T_Complex = Detail::GetComplexT<T>;
    // Define real type corresponding to T
    using T_Real = Detail::GetRealT<T>;

   public:  // <-- Make variant public
    // Define the variant type to hold a pointer to the appropriate base FFT
    // plan implementation
    using FFTPlanImplVariant = std::variant<
        std::unique_ptr<FFTPlanImpl<T_Complex>>,  // For complex T
        std::unique_ptr<RFFTPlanImpl<T_Real>>     // For real T
        >;

    // public constructor and methods remain here
   public:
    /**
     * @brief Constructor.
     * @param fft_plan_variant Variant holding the unique_ptr to the appropriate
     * FFT plan implementation.
     * @param kernel The correlation kernel (template).
     * @param type The correlation boundary type (Full, Same, Valid).
     * @param method The requested correlation method (Auto, Direct, FFT).
     * Currently only FFT is implemented here.
     * @throws std::invalid_argument if kernel is empty or fft_plan_variant
     * holds null.
     * @throws std::length_error if calculated FFT length overflows.
     * @throws std::runtime_error on internal errors (e.g., allocation).
     */
    DefaultCorrelationPlanImpl(
        FFTPlanImplVariant&& fft_plan_variant,
        const std::vector<T>& kernel,
        ConvolutionType type,
        ConvolutionMethod method);
    ~DefaultCorrelationPlanImpl() override = default;  // Defined here

    Status execute(
        std::span<const T> input, std::span<T> output) const override;
    size_t get_template_length() const override;
    ConvolutionType get_type() const override;
    ConvolutionMethod get_method() const override;
    size_t get_output_length(size_t input_length) const override;
    std::span<const T> get_template() const override;

    // Disable copy/move operations
    DefaultCorrelationPlanImpl(const DefaultCorrelationPlanImpl&) = delete;
    DefaultCorrelationPlanImpl& operator=(const DefaultCorrelationPlanImpl&)
        = delete;
    DefaultCorrelationPlanImpl(DefaultCorrelationPlanImpl&&) = delete;
    DefaultCorrelationPlanImpl& operator=(DefaultCorrelationPlanImpl&&)
        = delete;

   private:
    ConvolutionType type_;
    ConvolutionMethod method_;
    std::vector<T> original_kernel_;  // Stores the original template
    size_t kernel_length_;
    size_t fft_length_;

    // Store the CONJUGATE of the FFT of the (padded) kernel
    std::vector<T_Complex> kernel_fft_conj_;

    // Single variant to hold the appropriate FFT plan implementation pointer
    FFTPlanImplVariant fft_plan_impl_variant_;  // Now stores the passed-in plan

    // Temporary Buffers
    mutable std::unique_ptr<T[]> input_padded_;
    mutable std::unique_ptr<T_Complex[]> input_fft_;
    mutable std::unique_ptr<T_Complex[]> product_fft_;
    mutable std::unique_ptr<T[]> result_ifft_;
  };

  // --- Explicit Template Instantiations (Declaration) ---
  extern template class DefaultConvolutionPlanImpl<F32>;
  extern template class DefaultConvolutionPlanImpl<F64>;
  extern template class DefaultConvolutionPlanImpl<C32>;
  extern template class DefaultConvolutionPlanImpl<C64>;

  extern template class DefaultCorrelationPlanImpl<F32>;
  extern template class DefaultCorrelationPlanImpl<F64>;
  extern template class DefaultCorrelationPlanImpl<C32>;
  extern template class DefaultCorrelationPlanImpl<C64>;

  // --- Backend Factory Function Declarations (Implementation in backend.cpp)
  // --- These declarations remain unchanged (match AbstractBackend)
  [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlanImpl<F32>>>
  create_default_convolution_plan_impl_f32(
      const F32Vec& kernel, ConvolutionType type, ConvolutionMethod method);
  [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlanImpl<F64>>>
  create_default_convolution_plan_impl_f64(
      const F64Vec& kernel, ConvolutionType type, ConvolutionMethod method);
  [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlanImpl<C32>>>
  create_default_convolution_plan_impl_c32(
      const C32Vec& kernel, ConvolutionType type, ConvolutionMethod method);
  [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlanImpl<C64>>>
  create_default_convolution_plan_impl_c64(
      const C64Vec& kernel, ConvolutionType type, ConvolutionMethod method);

  [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlanImpl<F32>>>
  create_default_correlation_plan_impl_f32(
      const F32Vec& kernel, ConvolutionType type, ConvolutionMethod method);
  [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlanImpl<F64>>>
  create_default_correlation_plan_impl_f64(
      const F64Vec& kernel, ConvolutionType type, ConvolutionMethod method);
  [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlanImpl<C32>>>
  create_default_correlation_plan_impl_c32(
      const C32Vec& kernel, ConvolutionType type, ConvolutionMethod method);
  [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlanImpl<C64>>>
  create_default_correlation_plan_impl_c64(
      const C64Vec& kernel, ConvolutionType type, ConvolutionMethod method);

}  // namespace OmniDSP::backend

#endif  // OMNIDSP_DEFAULT_CONVOLUTION_HPP
