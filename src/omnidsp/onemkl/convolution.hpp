/**
 * @file convolution.hpp (onemkl)
 * @brief Declares the oneMKL backend ConvolutionPlanImpl and
 * CorrelationPlanImpl classes.
 */
#ifndef OMNIDSP_ONEMKL_CONVOLUTION_HPP
#define OMNIDSP_ONEMKL_CONVOLUTION_HPP

#ifdef OMNIDSP_USE_ONEMKL  // Compile guard

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
    using T_Complex = Detail::GetComplexT<T>;
    using T_Real = Detail::GetRealT<T>;
    // Variant to hold either a complex or real FFT plan implementation pointer
    using FFTPlanImplVariant = std::variant<
        std::monostate,  // Default state
        std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>,
        std::unique_ptr<OneMKLRFFTPlanImpl<T_Real>>>;

   public:
    /**
     * @brief Constructor. Creates internal FFT plan and pre-computes kernel
     * FFT.
     * @param owner Pointer to the creating AbstractBackend (needed for FFT plan
     * creation).
     * @param kernel The convolution kernel.
     * @param type The convolution type (Full, Same, Valid).
     * @param method The preferred convolution method (Direct, FFT, Auto -
     * influences internal choices).
     * @throws std::invalid_argument If kernel is empty or owner is null.
     * @throws std::runtime_error If internal FFT plan creation or kernel FFT
     * fails.
     * @throws std::bad_alloc If memory allocation fails.
     */
    OneMKLConvolutionPlanImpl(
        const AbstractBackend* owner,  // Needs owner to create sub-plans
        const std::vector<T>& kernel,
        ConvolutionType type,
        ConvolutionMethod method);

    ~OneMKLConvolutionPlanImpl() override;

    // --- Deleted Copy/Move ---
    OneMKLConvolutionPlanImpl(const OneMKLConvolutionPlanImpl&) = delete;
    OneMKLConvolutionPlanImpl& operator=(const OneMKLConvolutionPlanImpl&)
        = delete;
    OneMKLConvolutionPlanImpl(OneMKLConvolutionPlanImpl&&) = delete;
    OneMKLConvolutionPlanImpl& operator=(OneMKLConvolutionPlanImpl&&) = delete;

    // --- Interface Methods ---
    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) const override;
    size_t get_kernel_length() const override;
    ConvolutionType get_type() const override;
    size_t get_output_length(size_t input_length) const override;
    std::span<const T> get_kernel() const override;
    ConvolutionMethod get_method() const override;

    // Public member to allow backend factory to check status after construction
    // (Set based on internal FFT plan creation status)
    MKL_LONG mkl_status_;

   private:
    ConvolutionType type_;
    ConvolutionMethod method_;
    std::vector<T> original_kernel_;  // Store original kernel
    size_t kernel_length_;
    size_t fft_length_;  // FFT length used internally
    FFTPlanImplVariant
        fft_plan_impl_variant_;  // Holds the FFT or RFFT plan impl
    std::vector<T_Complex>
        kernel_fft_;  // Precomputed FFT of the (reversed) kernel

    // Temporary buffers (consider making unique_ptrs or resizing vectors in
    // execute) Using unique_ptrs to avoid large stack allocation
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
    using T_Complex = Detail::GetComplexT<T>;
    using T_Real = Detail::GetRealT<T>;
    // Variant to hold either a complex or real FFT plan implementation pointer
    using FFTPlanImplVariant = std::variant<
        std::monostate,  // Default state
        std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>,
        std::unique_ptr<OneMKLRFFTPlanImpl<T_Real>>>;

   public:
    /**
     * @brief Constructor. Creates internal FFT plan and pre-computes conjugated
     * kernel FFT.
     * @param owner Pointer to the creating AbstractBackend (needed for FFT plan
     * creation).
     * @param kernel The correlation template (kernel).
     * @param type The correlation type (Full, Same, Valid).
     * @param method The preferred correlation method (Direct, FFT, Auto -
     * influences internal choices).
     * @throws std::invalid_argument If kernel is empty or owner is null.
     * @throws std::runtime_error If internal FFT plan creation or kernel FFT
     * fails.
     * @throws std::bad_alloc If memory allocation fails.
     */
    OneMKLCorrelationPlanImpl(
        const AbstractBackend* owner,  // Needs owner to create sub-plans
        const std::vector<T>& kernel,
        ConvolutionType type,
        ConvolutionMethod method);

    ~OneMKLCorrelationPlanImpl() override;

    // --- Deleted Copy/Move ---
    OneMKLCorrelationPlanImpl(const OneMKLCorrelationPlanImpl&) = delete;
    OneMKLCorrelationPlanImpl& operator=(const OneMKLCorrelationPlanImpl&)
        = delete;
    OneMKLCorrelationPlanImpl(OneMKLCorrelationPlanImpl&&) = delete;
    OneMKLCorrelationPlanImpl& operator=(OneMKLCorrelationPlanImpl&&) = delete;

    // --- Interface Methods ---
    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) const override;
    size_t get_template_length() const override;
    ConvolutionType get_type() const override;
    size_t get_output_length(size_t input_length) const override;
    std::span<const T> get_template() const override;
    ConvolutionMethod get_method() const override;

    // Public member to allow backend factory to check status after construction
    MKL_LONG mkl_status_;

   private:
    ConvolutionType type_;
    ConvolutionMethod method_;
    std::vector<T> original_kernel_;  // Store original template
    size_t kernel_length_;            // Template length
    size_t fft_length_;
    FFTPlanImplVariant
        fft_plan_impl_variant_;  // Holds the FFT or RFFT plan impl
    std::vector<T_Complex>
        kernel_fft_conj_;  // Precomputed CONJUGATED FFT of the template

    // Temporary buffers
    std::unique_ptr<T[]> input_padded_;
    std::unique_ptr<T_Complex[]> input_fft_;
    std::unique_ptr<T_Complex[]> product_fft_;
    std::unique_ptr<T[]> result_ifft_;
  };

}  // namespace OmniDSP::backend

#endif  // OMNIDSP_USE_ONEMKL
#endif  // OMNIDSP_ONEMKL_CONVOLUTION_HPP
