/**
 * @file convolution.hpp (onemkl)
 * @brief Declares the oneMKL backend ConvolutionPlanImpl and
 * CorrelationPlanImpl classes.
 */
#ifndef OMNIDSP_ONEMKL_CONVOLUTION_HPP
#define OMNIDSP_ONEMKL_CONVOLUTION_HPP

#include <mkl.h>  // MKL types might be needed (MKL_LONG, status codes)

#include <OmniDSP/convolution.hpp>  // For ConvolutionType, ConvolutionMethod
#include <OmniDSP/core_types.hpp>
#include <complex>
#include <memory>  // For std::unique_ptr
#include <span>
#include <variant>  // For std::variant
#include <vector>

#include "../interface/backend.hpp"  // Base PlanImpl interfaces

// Forward declare the specific oneMKL FFT plan implementations used internally
namespace OmniDSP::backend {
  template <typename T_Complex>
  class OneMKLFFTPlanImpl;
  template <typename T_Real>
  class OneMKLRFFTPlanImpl;
  class AbstractBackend;  // Forward declare AbstractBackend
}  // namespace OmniDSP::backend

namespace OmniDSP::backend {

  /**
   * @brief oneMKL implementation for convolution plans. Uses internal oneMKL
   * FFT plans.
   * @tparam T Data type (F32, F64, C32, C64).
   */
  template <typename T>  // T can be F32, F64, C32, C64
  class OneMKLConvolutionPlanImpl final : public ConvolutionPlanImpl<T> {
    // *** UPDATED Namespace ***
    using T_Complex = Utils::GetComplexT<T>;
    using T_Real = Utils::GetRealT<T>;
    using FFTPlanImplVariant = std::variant<
        std::monostate,
        std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>,
        std::unique_ptr<OneMKLRFFTPlanImpl<T_Real>>>;

   public:
    OneMKLConvolutionPlanImpl(
        const AbstractBackend* owner,
        const std::vector<T>& kernel,
        ConvolutionType type,
        ConvolutionMethod method);

    ~OneMKLConvolutionPlanImpl() override;

    OneMKLConvolutionPlanImpl(const OneMKLConvolutionPlanImpl&) = delete;
    OneMKLConvolutionPlanImpl& operator=(const OneMKLConvolutionPlanImpl&)
        = delete;
    OneMKLConvolutionPlanImpl(OneMKLConvolutionPlanImpl&&) = delete;
    OneMKLConvolutionPlanImpl& operator=(OneMKLConvolutionPlanImpl&&) = delete;

    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) const override;
    size_t get_kernel_length() const override
    {
      return kernel_length_;
    }  // Define inline
    ConvolutionType get_type() const override
    {
      return type_;
    }  // Define inline
    size_t get_output_length(size_t input_length)
        const override;  // Keep definition in cpp (more complex)
    [[nodiscard]] std::span<const T> get_kernel() const override
    {
      return std::span<const T>(original_kernel_);
    }
    [[nodiscard]] ConvolutionMethod get_method() const override
    {
      return method_;
    }

    MKL_LONG mkl_status_;

   private:
    ConvolutionType type_;
    ConvolutionMethod method_;
    std::vector<T> original_kernel_;
    size_t kernel_length_;
    size_t fft_length_;
    FFTPlanImplVariant fft_plan_impl_variant_;
    std::vector<T_Complex> kernel_fft_;
    std::unique_ptr<T[]> input_padded_;
    std::unique_ptr<T_Complex[]> input_fft_;
    std::unique_ptr<T_Complex[]> product_fft_;
    std::unique_ptr<T[]> result_ifft_;
  };

  /**
   * @brief oneMKL implementation for correlation plans. Uses internal oneMKL
   * FFT plans.
   * @tparam T Data type (F32, F64, C32, C64).
   */
  template <typename T>  // T can be F32, F64, C32, C64
  class OneMKLCorrelationPlanImpl final : public CorrelationPlanImpl<T> {
    // *** UPDATED Namespace ***
    using T_Complex = Utils::GetComplexT<T>;
    using T_Real = Utils::GetRealT<T>;
    using FFTPlanImplVariant = std::variant<
        std::monostate,
        std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>,
        std::unique_ptr<OneMKLRFFTPlanImpl<T_Real>>>;

   public:
    OneMKLCorrelationPlanImpl(
        const AbstractBackend* owner,
        const std::vector<T>& kernel,
        ConvolutionType type,
        ConvolutionMethod method);

    ~OneMKLCorrelationPlanImpl() override;

    OneMKLCorrelationPlanImpl(const OneMKLCorrelationPlanImpl&) = delete;
    OneMKLCorrelationPlanImpl& operator=(const OneMKLCorrelationPlanImpl&)
        = delete;
    OneMKLCorrelationPlanImpl(OneMKLCorrelationPlanImpl&&) = delete;
    OneMKLCorrelationPlanImpl& operator=(OneMKLCorrelationPlanImpl&&) = delete;

    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) const override;
    size_t get_template_length() const override
    {
      return kernel_length_;
    }  // Define inline
    ConvolutionType get_type() const override
    {
      return type_;
    }  // Define inline
    size_t get_output_length(
        size_t input_length) const override;  // Keep definition in cpp
    [[nodiscard]] std::span<const T> get_template() const override
    {
      return std::span<const T>(original_kernel_);
    }
    [[nodiscard]] ConvolutionMethod get_method() const override
    {
      return method_;
    }

    MKL_LONG mkl_status_;

   private:
    ConvolutionType type_;
    ConvolutionMethod method_;
    std::vector<T> original_kernel_;  // Store original template
    size_t kernel_length_;            // Template length
    size_t fft_length_;
    FFTPlanImplVariant fft_plan_impl_variant_;
    std::vector<T_Complex> kernel_fft_conj_;  // Precomputed CONJUGATED FFT
    std::unique_ptr<T[]> input_padded_;
    std::unique_ptr<T_Complex[]> input_fft_;
    std::unique_ptr<T_Complex[]> product_fft_;
    std::unique_ptr<T[]> result_ifft_;
  };

}  // namespace OmniDSP::backend

#endif  // OMNIDSP_ONEMKL_CONVOLUTION_HPP
