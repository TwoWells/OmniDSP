/**
 * @file convolution.hpp
 * @brief Defines the public API for Convolution and Correlation Plan objects.
 */

#ifndef OMNIDSP_CONVOLUTION_HPP
#define OMNIDSP_CONVOLUTION_HPP

#include <complex>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>  // For std::move
#include <vector>  // Required for std::vector in old signature, kept for safety, review if removable

#include "OmniDSP/core_types.hpp"  // Includes Status, OmniExpected, F32, C32 etc.
#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT on explicit instantiations
#include "OmniDSP/types/convolution.hpp"  // For ConvolutionType, ConvolutionMethod (used by Params)

// Include Abstract::Backend for the static create method
#include "interface/backend.hpp"  // Defines Abstract::Backend

// Include Params types for Convolution and Correlation for the static create
// method
#include "OmniDSP/params/convolution.hpp"  // For Params::Convolution, Params::Correlation

// Forward declare Abstract Impl classes
namespace OmniDSP::Abstract {
  template <typename T>
  class ConvolutionPlanImpl;
  template <typename T>
  class CorrelationPlanImpl;
}  // namespace OmniDSP::Abstract

namespace OmniDSP {

  /**
   * @brief Plan object for executing Convolution operations.
   * @tparam T The data type (e.g., F32, F64, C32, C64).
   */
  template <typename T>
  class ConvolutionPlan {  // OMNIDSP_EXPORT removed from class template
   public:
    ~ConvolutionPlan();
    ConvolutionPlan(ConvolutionPlan&& other) noexcept;
    ConvolutionPlan& operator=(ConvolutionPlan&& other) noexcept;
    ConvolutionPlan(const ConvolutionPlan&) = delete;
    ConvolutionPlan& operator=(const ConvolutionPlan&) = delete;

    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) const;
    size_t get_kernel_length() const;
    ConvolutionType get_type() const;
    ConvolutionMethod get_method_hint() const;
    size_t get_output_length(size_t input_length) const;
    std::span<const T> get_kernel() const;

    /**
     * @brief Creates a ConvolutionPlan using a specific backend and parameters.
     * @param backend The backend to use for creating the plan.
     * @param params The parameters for the convolution operation.
     * @param kernel_coeffs The convolution kernel coefficients.
     * @return An OmniExpected containing a unique_ptr to the ConvolutionPlan or
     * an error Status.
     */
    [[nodiscard]] static OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
    create(
        const Abstract::Backend& backend,
        const Params::Convolution& params,
        std::span<const T> kernel_coeffs);

    // Declaration only for create_from_impl
    static std::unique_ptr<ConvolutionPlan<T>> create_from_impl(
        std::unique_ptr<Abstract::ConvolutionPlanImpl<T>> pimpl);

   private:
    explicit ConvolutionPlan(
        std::unique_ptr<Abstract::ConvolutionPlanImpl<T>> pimpl);
    std::unique_ptr<Abstract::ConvolutionPlanImpl<T>> pimpl_;
  };

  /**
   * @brief Plan object for executing Correlation operations.
   * @tparam T The data type (e.g., F32, F64, C32, C64).
   */
  template <typename T>
  class CorrelationPlan {  // OMNIDSP_EXPORT removed from class template
   public:
    ~CorrelationPlan();
    CorrelationPlan(CorrelationPlan&& other) noexcept;
    CorrelationPlan& operator=(CorrelationPlan&& other) noexcept;
    CorrelationPlan(const CorrelationPlan&) = delete;
    CorrelationPlan& operator=(const CorrelationPlan&) = delete;

    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) const;
    size_t get_template_length() const;
    ConvolutionType get_type() const;
    ConvolutionMethod get_method_hint() const;
    size_t get_output_length(size_t input_length) const;
    std::span<const T> get_template() const;

    /**
     * @brief Creates a CorrelationPlan using a specific backend and parameters.
     * @param backend The backend to use for creating the plan.
     * @param params The parameters for the correlation operation.
     * @param template_coeffs The correlation template coefficients.
     * @return An OmniExpected containing a unique_ptr to the CorrelationPlan or
     * an error Status.
     */
    [[nodiscard]] static OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
    create(
        const Abstract::Backend& backend,
        const Params::Correlation& params,
        std::span<const T> template_coeffs);

    // Declaration only for create_from_impl
    static std::unique_ptr<CorrelationPlan<T>> create_from_impl(
        std::unique_ptr<Abstract::CorrelationPlanImpl<T>> pimpl);

   private:
    explicit CorrelationPlan(
        std::unique_ptr<Abstract::CorrelationPlanImpl<T>> pimpl);
    std::unique_ptr<Abstract::CorrelationPlanImpl<T>> pimpl_;
  };
}  // namespace OmniDSP

#endif  // OMNIDSP_CONVOLUTION_HPP
