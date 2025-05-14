#ifndef OMNIDSP_DEFAULT_CONVOLUTION_HPP
#define OMNIDSP_DEFAULT_CONVOLUTION_HPP

#include <OmniDSP/core_types.hpp>  // For Status, F32, C32, etc., Utils::*
#include <OmniDSP/params/convolution.hpp>  // For Params::Convolution, Params::Correlation
#include <complex>
#include <memory>     // For std::unique_ptr
#include <span>       // For std::span
#include <stdexcept>  // For std::logic_error, std::invalid_argument
#include <variant>    // Include for std::variant
#include <vector>

#include "../interface/backend.hpp"  // Defines Abstract::ConvolutionPlanImpl/CorrelationPlanImpl

// Forward declare FFT plan impl BASE classes used internally
// These are defined in interface/backend.hpp
// (Already forward-declared or included via backend.hpp)

namespace OmniDSP::Default {

  /**
   * @brief Default backend implementation for Convolution Plan (Standard C++).
   * @details Uses FFT-based convolution internally, leveraging a provided FFT
   * plan implementation.
   * @tparam T Data type (F32, F64, C32, C64).
   */
  template <typename T>
  class ConvolutionPlanImpl final : public Abstract::ConvolutionPlanImpl<T> {
    using T_Complex = Utils::GetComplexType<T>;
    using T_Real = Utils::GetRealType<T>;

   public:
    // Define the variant type to hold a pointer to the appropriate base FFT
    // plan implementation
    using FFTPlanImplVariant = std::variant<
        std::unique_ptr<Abstract::FFTPlanImpl<T_Complex>>,  // For complex T
        std::unique_ptr<Abstract::RFFTPlanImpl<T_Real>>     // For real T
        >;

   public:
    /**
     * @brief Constructor.
     * @param fft_plan_variant Variant holding the unique_ptr to the appropriate
     * FFT plan implementation.
     * @param params The convolution parameters.
     * @param kernel_coeffs The convolution kernel coefficients.
     * @throws std::invalid_argument if kernel_coeffs is empty, fft_plan_variant
     * holds null, or if the provided FFT plan's length is insufficient for
     * the operation defined by params and kernel_coeffs.
     * @throws std::runtime_error on internal errors (e.g., allocation).
     */
    ConvolutionPlanImpl(
        FFTPlanImplVariant&& fft_plan_variant,
        const Params::Convolution& params,
        std::span<const T> kernel_coeffs);

    ~ConvolutionPlanImpl() override = default;

    OmniStatus execute(
        std::span<const T> input, std::span<T> output) const override;
    size_t get_kernel_length() const override;
    ConvolutionType get_type() const override;
    ConvolutionMethod get_method()
        const override;  // Will return method_hint from params
    size_t get_output_length(size_t input_length) const override;
    std::span<const T> get_kernel() const override;

    // Disable copy/move operations
    ConvolutionPlanImpl(const ConvolutionPlanImpl&) = delete;
    ConvolutionPlanImpl& operator=(const ConvolutionPlanImpl&) = delete;
    ConvolutionPlanImpl(ConvolutionPlanImpl&&) = delete;
    ConvolutionPlanImpl& operator=(ConvolutionPlanImpl&&) = delete;

   private:
    ConvolutionType type_;
    ConvolutionMethod method_hint_;  // Stores the hint from Params::Convolution
    std::vector<T>
        original_kernel_;  // Stores a copy of the kernel coefficients
    size_t kernel_length_;
    size_t fft_length_;        // FFT length derived from the provided FFT plan
    size_t max_input_length_;  // Stored from params for validation if needed

    std::vector<T_Complex> kernel_fft_;
    FFTPlanImplVariant fft_plan_impl_variant_;

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
  class CorrelationPlanImpl final : public Abstract::CorrelationPlanImpl<T> {
    using T_Complex = Utils::GetComplexType<T>;
    using T_Real = Utils::GetRealType<T>;

   public:
    using FFTPlanImplVariant = std::variant<
        std::unique_ptr<Abstract::FFTPlanImpl<T_Complex>>,
        std::unique_ptr<Abstract::RFFTPlanImpl<T_Real>>>;

   public:
    /**
     * @brief Constructor.
     * @param fft_plan_variant Variant holding the unique_ptr to the appropriate
     * FFT plan implementation.
     * @param params The correlation parameters.
     * @param template_coeffs The correlation template coefficients.
     * @throws std::invalid_argument if template_coeffs is empty,
     * fft_plan_variant holds null, or if the provided FFT plan's length is
     * insufficient for the operation defined by params and template_coeffs.
     * @throws std::runtime_error on internal errors (e.g., allocation).
     */
    CorrelationPlanImpl(
        FFTPlanImplVariant&& fft_plan_variant,
        const Params::Correlation& params,
        std::span<const T> template_coeffs);

    ~CorrelationPlanImpl() override = default;

    OmniStatus execute(
        std::span<const T> input, std::span<T> output) const override;
    size_t get_template_length() const override;
    ConvolutionType get_type() const override;
    ConvolutionMethod get_method()
        const override;  // Will return method_hint from params
    size_t get_output_length(size_t input_length) const override;
    std::span<const T> get_template() const override;

    // Disable copy/move operations
    CorrelationPlanImpl(const CorrelationPlanImpl&) = delete;
    CorrelationPlanImpl& operator=(const CorrelationPlanImpl&) = delete;
    CorrelationPlanImpl(CorrelationPlanImpl&&) = delete;
    CorrelationPlanImpl& operator=(CorrelationPlanImpl&&) = delete;

   private:
    ConvolutionType type_;
    ConvolutionMethod method_hint_;  // Stores the hint from Params::Correlation
    std::vector<T>
        original_template_;  // Stores a copy of the template coefficients
    size_t template_length_;
    size_t fft_length_;        // FFT length derived from the provided FFT plan
    size_t max_input_length_;  // Stored from params for validation if needed

    std::vector<T_Complex>
        template_fft_conj_;  // Stores conjugate of template FFT
    FFTPlanImplVariant fft_plan_impl_variant_;

    // Temporary Buffers
    mutable std::unique_ptr<T[]> input_padded_;
    mutable std::unique_ptr<T_Complex[]> input_fft_;
    mutable std::unique_ptr<T_Complex[]> product_fft_;
    mutable std::unique_ptr<T[]> result_ifft_;
  };

  // --- Explicit Template Instantiations (Declaration) ---
  // These are typically done in the .cpp file if not header-only.
  // If you need them exported from a DLL, they might be here.
  // For now, assuming they will be in the .cpp file.
  // extern template class ConvolutionPlanImpl<F32>;
  // ... and so on for other types and CorrelationPlanImpl

}  // namespace OmniDSP::Default

#endif  // OMNIDSP_DEFAULT_CONVOLUTION_HPP
