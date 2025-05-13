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
#include <vector>

#include "OmniDSP/core_types.hpp"  // Includes Status, OmniExpected, F32, C32 etc.
#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT on explicit instantiations
#include "OmniDSP/types/convolution.hpp"

// Include Abstract::Backend for the static create method
#include "interface/backend.hpp"  // Defines Abstract::Backend

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

    [[nodiscard]] static OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
    create(
        const Abstract::Backend& backend,
        const std::vector<T>& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto);

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

    [[nodiscard]] static OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
    create(
        const Abstract::Backend& backend,
        const std::vector<T>& kernel,
        ConvolutionType type,
        ConvolutionMethod method = ConvolutionMethod::Auto);

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
