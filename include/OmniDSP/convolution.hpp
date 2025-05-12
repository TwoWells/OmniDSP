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
#include <string_view>  // For get_convolution_type_name etc. if they were here (now in core_types)
#include <utility>  // For std::move
#include <vector>

#include "OmniDSP/core_types.hpp"  // Includes Status, OmniExpected, F32, C32,
#include "OmniDSP/omnidsp_export.hpp"
// AND NOW ConvolutionType, ConvolutionMethod
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

  // ConvolutionType and ConvolutionMethod enums, and their get_*_name functions
  // have been MOVED to core_types.hpp

  /**
   * @brief Plan object for executing Convolution operations.
   * @tparam T The data type (e.g., F32, F64, C32, C64).
   */
  template <typename T>
  class OMNIDSP_EXPORT ConvolutionPlan {
   public:
    ~ConvolutionPlan();
    ConvolutionPlan(ConvolutionPlan&& other) noexcept;
    ConvolutionPlan& operator=(ConvolutionPlan&& other) noexcept;
    ConvolutionPlan(const ConvolutionPlan&) = delete;
    ConvolutionPlan& operator=(const ConvolutionPlan&) = delete;

    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) const;
    size_t get_kernel_length() const;
    ConvolutionType get_type() const;           // Enum now from core_types.hpp
    ConvolutionMethod get_method_hint() const;  // Enum now from core_types.hpp
    size_t get_output_length(size_t input_length) const;
    std::span<const T> get_kernel() const;

    [[nodiscard]] static OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
    create(
        const Abstract::Backend& backend,
        const std::vector<T>& kernel,
        ConvolutionType type,  // Enum now from core_types.hpp
        ConvolutionMethod method
        = ConvolutionMethod::Auto  // Enum now from core_types.hpp
    )
    {
      OmniExpected<std::unique_ptr<Abstract::ConvolutionPlanImpl<T>>>
          pimpl_expected;
      if constexpr (std::is_same_v<T, F32>) {
        pimpl_expected
            = backend.create_convolution_plan_impl_f32(kernel, type, method);
      }
      else if constexpr (std::is_same_v<T, F64>) {
        pimpl_expected
            = backend.create_convolution_plan_impl_f64(kernel, type, method);
      }
      else if constexpr (std::is_same_v<T, C32>) {
        pimpl_expected
            = backend.create_convolution_plan_impl_c32(kernel, type, method);
      }
      else if constexpr (std::is_same_v<T, C64>) {
        pimpl_expected
            = backend.create_convolution_plan_impl_c64(kernel, type, method);
      }
      else {
        return std::unexpected(Status::UnsupportedFeature);
      }

      if (!pimpl_expected) {
        return std::unexpected(pimpl_expected.error());
      }

      auto plan = ConvolutionPlan<T>::create_from_impl(
          std::move(pimpl_expected.value()));
      if (!plan) {
        return std::unexpected(Status::Failure);
      }
      return plan;
    }

    static std::unique_ptr<ConvolutionPlan<T>> create_from_impl(
        std::unique_ptr<Abstract::ConvolutionPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      return std::unique_ptr<ConvolutionPlan<T>>(
          new ConvolutionPlan<T>(std::move(pimpl)));
    }

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
  class OMNIDSP_EXPORT CorrelationPlan {
   public:
    ~CorrelationPlan();
    CorrelationPlan(CorrelationPlan&& other) noexcept;
    CorrelationPlan& operator=(CorrelationPlan&& other) noexcept;
    CorrelationPlan(const CorrelationPlan&) = delete;
    CorrelationPlan& operator=(const CorrelationPlan&) = delete;

    [[nodiscard]] Status execute(
        std::span<const T> input, std::span<T> output) const;
    size_t get_template_length() const;
    ConvolutionType get_type() const;           // Enum now from core_types.hpp
    ConvolutionMethod get_method_hint() const;  // Enum now from core_types.hpp
    size_t get_output_length(size_t input_length) const;
    std::span<const T> get_template() const;

    [[nodiscard]] static OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
    create(
        const Abstract::Backend& backend,
        const std::vector<T>& kernel,
        ConvolutionType type,  // Enum now from core_types.hpp
        ConvolutionMethod method
        = ConvolutionMethod::Auto  // Enum now from core_types.hpp
    )
    {
      OmniExpected<std::unique_ptr<Abstract::CorrelationPlanImpl<T>>>
          pimpl_expected;
      if constexpr (std::is_same_v<T, F32>) {
        pimpl_expected
            = backend.create_correlation_plan_impl_f32(kernel, type, method);
      }
      else if constexpr (std::is_same_v<T, F64>) {
        pimpl_expected
            = backend.create_correlation_plan_impl_f64(kernel, type, method);
      }
      else if constexpr (std::is_same_v<T, C32>) {
        pimpl_expected
            = backend.create_correlation_plan_impl_c32(kernel, type, method);
      }
      else if constexpr (std::is_same_v<T, C64>) {
        pimpl_expected
            = backend.create_correlation_plan_impl_c64(kernel, type, method);
      }
      else {
        return std::unexpected(Status::UnsupportedFeature);
      }

      if (!pimpl_expected) {
        return std::unexpected(pimpl_expected.error());
      }

      auto plan = CorrelationPlan<T>::create_from_impl(
          std::move(pimpl_expected.value()));
      if (!plan) {
        return std::unexpected(Status::Failure);
      }
      return plan;
    }

    static std::unique_ptr<CorrelationPlan<T>> create_from_impl(
        std::unique_ptr<Abstract::CorrelationPlanImpl<T>> pimpl)
    {
      if (!pimpl) {
        return nullptr;
      }
      return std::unique_ptr<CorrelationPlan<T>>(
          new CorrelationPlan<T>(std::move(pimpl)));
    }

   private:
    explicit CorrelationPlan(
        std::unique_ptr<Abstract::CorrelationPlanImpl<T>> pimpl);
    std::unique_ptr<Abstract::CorrelationPlanImpl<T>> pimpl_;
  };

}  // namespace OmniDSP

#endif  // OMNIDSP_CONVOLUTION_HPP
