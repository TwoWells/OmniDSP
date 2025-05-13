/**
 * @file convolution.cpp
 * @brief Implements the ConvolutionPlan and CorrelationPlan class methods,
 * forwarding calls to backend implementations.
 */

#include "OmniDSP/convolution.hpp"  // Corresponding header

// Include the backend interface definition which declares the Impl classes
#include <expected>  // For std::unexpected
#include <memory>    // For std::unique_ptr
#include <numeric>  // For std::max, std::min (potentially used in get_output_length)
#include <span>
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move
#include <vector>

#include "backend.hpp"  // Defines abstract::ConvolutionPlanImpl, abstract::CorrelationPlanImpl

// Include core types for F32, F64 etc. used in instantiations
#include <OmniDSP/core_types.hpp>

namespace OmniDSP {

  //--------------------------------------------------------------------------
  // ConvolutionPlan Method Definitions
  //--------------------------------------------------------------------------

  // Static create_from_impl Definition
  template <typename T>
  std::unique_ptr<ConvolutionPlan<T>> ConvolutionPlan<T>::create_from_impl(
      std::unique_ptr<Abstract::ConvolutionPlanImpl<T>> pimpl)
  {
    if (!pimpl) {
      return nullptr;
    }
    // Calls the private constructor of ConvolutionPlan<T>
    // Using `new` directly as make_unique cannot access private constructor.
    // The std::unique_ptr will manage the memory.
    return std::unique_ptr<ConvolutionPlan<T>>(
        new ConvolutionPlan<T>(std::move(pimpl)));
  }

  // Static create Definition
  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<ConvolutionPlan<T>>>
  ConvolutionPlan<T>::create(
      const Abstract::Backend& backend,
      const std::vector<T>& kernel,
      ConvolutionType type,
      ConvolutionMethod method)
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
      return std::unexpected(Status::Failure);  // Or AllocationError
    }
    return plan;
  }

  template <typename T>
  ConvolutionPlan<T>::ConvolutionPlan(
      std::unique_ptr<Abstract::ConvolutionPlanImpl<T>> pimpl)
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "ConvolutionPlan created with null implementation.");
    }
  }

  template <typename T>
  ConvolutionPlan<T>::~ConvolutionPlan() = default;

  template <typename T>
  ConvolutionPlan<T>::ConvolutionPlan(ConvolutionPlan&& other) noexcept
      = default;

  template <typename T>
  ConvolutionPlan<T>& ConvolutionPlan<T>::operator=(
      ConvolutionPlan&& other) noexcept
      = default;

  template <typename T>
  [[nodiscard]] Status ConvolutionPlan<T>::execute(
      std::span<const T> input, std::span<T> output) const
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    return pimpl_->execute(input, output);
  }

  template <typename T>
  size_t ConvolutionPlan<T>::get_kernel_length() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ConvolutionPlan instance in get_kernel_length.");
    }
    return pimpl_->get_kernel_length();
  }

  template <typename T>
  ConvolutionType ConvolutionPlan<T>::get_type() const
  {
    if (!pimpl_) {
      throw std::runtime_error("Invalid ConvolutionPlan instance in get_type.");
    }
    return pimpl_->get_type();
  }

  template <typename T>
  ConvolutionMethod ConvolutionPlan<T>::get_method_hint() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ConvolutionPlan instance in get_method_hint.");
    }
    return pimpl_->get_method();
  }

  template <typename T>
  size_t ConvolutionPlan<T>::get_output_length(size_t input_length) const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ConvolutionPlan instance in get_output_length.");
    }
    return pimpl_->get_output_length(input_length);
  }

  template <typename T>
  std::span<const T> ConvolutionPlan<T>::get_kernel() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ConvolutionPlan instance in get_kernel.");
    }
    return pimpl_->get_kernel();
  }

  //--------------------------------------------------------------------------
  // CorrelationPlan Method Definitions
  //--------------------------------------------------------------------------

  // Static create_from_impl Definition
  template <typename T>
  std::unique_ptr<CorrelationPlan<T>> CorrelationPlan<T>::create_from_impl(
      std::unique_ptr<Abstract::CorrelationPlanImpl<T>> pimpl)
  {
    if (!pimpl) {
      return nullptr;
    }
    return std::unique_ptr<CorrelationPlan<T>>(
        new CorrelationPlan<T>(std::move(pimpl)));
  }

  // Static create Definition
  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<CorrelationPlan<T>>>
  CorrelationPlan<T>::create(
      const Abstract::Backend& backend,
      const std::vector<T>& kernel,
      ConvolutionType type,
      ConvolutionMethod method)
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
      return std::unexpected(Status::Failure);  // Or AllocationError
    }
    return plan;
  }

  template <typename T>
  CorrelationPlan<T>::CorrelationPlan(
      std::unique_ptr<Abstract::CorrelationPlanImpl<T>> pimpl)
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "CorrelationPlan created with null implementation.");
    }
  }

  template <typename T>
  CorrelationPlan<T>::~CorrelationPlan() = default;

  template <typename T>
  CorrelationPlan<T>::CorrelationPlan(CorrelationPlan&& other) noexcept
      = default;

  template <typename T>
  CorrelationPlan<T>& CorrelationPlan<T>::operator=(
      CorrelationPlan&& other) noexcept
      = default;

  template <typename T>
  [[nodiscard]] Status CorrelationPlan<T>::execute(
      std::span<const T> input, std::span<T> output) const
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    return pimpl_->execute(input, output);
  }

  template <typename T>
  size_t CorrelationPlan<T>::get_template_length() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid CorrelationPlan instance in get_template_length.");
    }
    return pimpl_->get_template_length();
  }

  template <typename T>
  ConvolutionType CorrelationPlan<T>::get_type() const
  {
    if (!pimpl_) {
      throw std::runtime_error("Invalid CorrelationPlan instance in get_type.");
    }
    return pimpl_->get_type();
  }

  template <typename T>
  ConvolutionMethod CorrelationPlan<T>::get_method_hint() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid CorrelationPlan instance in get_method_hint.");
    }
    return pimpl_->get_method();
  }

  template <typename T>
  size_t CorrelationPlan<T>::get_output_length(size_t input_length) const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid CorrelationPlan instance in get_output_length.");
    }
    return pimpl_->get_output_length(input_length);
  }

  template <typename T>
  std::span<const T> CorrelationPlan<T>::get_template() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid CorrelationPlan instance in get_template.");
    }
    return pimpl_->get_template();
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------

  template class OMNIDSP_EXPORT ConvolutionPlan<F32>;
  template class OMNIDSP_EXPORT ConvolutionPlan<F64>;
  template class OMNIDSP_EXPORT ConvolutionPlan<C32>;
  template class OMNIDSP_EXPORT ConvolutionPlan<C64>;

  template class OMNIDSP_EXPORT CorrelationPlan<F32>;
  template class OMNIDSP_EXPORT CorrelationPlan<F64>;
  template class OMNIDSP_EXPORT CorrelationPlan<C32>;
  template class OMNIDSP_EXPORT CorrelationPlan<C64>;

  // Explicitly instantiate static members for export
  // For ConvolutionPlan
  template OMNIDSP_EXPORT std::unique_ptr<ConvolutionPlan<F32>>
  ConvolutionPlan<F32>::create_from_impl(
      std::unique_ptr<Abstract::ConvolutionPlanImpl<F32>> pimpl);
  template OMNIDSP_EXPORT std::unique_ptr<ConvolutionPlan<F64>>
  ConvolutionPlan<F64>::create_from_impl(
      std::unique_ptr<Abstract::ConvolutionPlanImpl<F64>> pimpl);
  template OMNIDSP_EXPORT std::unique_ptr<ConvolutionPlan<C32>>
  ConvolutionPlan<C32>::create_from_impl(
      std::unique_ptr<Abstract::ConvolutionPlanImpl<C32>> pimpl);
  template OMNIDSP_EXPORT std::unique_ptr<ConvolutionPlan<C64>>
  ConvolutionPlan<C64>::create_from_impl(
      std::unique_ptr<Abstract::ConvolutionPlanImpl<C64>> pimpl);

  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<ConvolutionPlan<F32>>>
  ConvolutionPlan<F32>::create(
      const Abstract::Backend&,
      const std::vector<F32>&,
      ConvolutionType,
      ConvolutionMethod);
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<ConvolutionPlan<F64>>>
  ConvolutionPlan<F64>::create(
      const Abstract::Backend&,
      const std::vector<F64>&,
      ConvolutionType,
      ConvolutionMethod);
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<ConvolutionPlan<C32>>>
  ConvolutionPlan<C32>::create(
      const Abstract::Backend&,
      const std::vector<C32>&,
      ConvolutionType,
      ConvolutionMethod);
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<ConvolutionPlan<C64>>>
  ConvolutionPlan<C64>::create(
      const Abstract::Backend&,
      const std::vector<C64>&,
      ConvolutionType,
      ConvolutionMethod);

  // For CorrelationPlan
  template OMNIDSP_EXPORT std::unique_ptr<CorrelationPlan<F32>>
  CorrelationPlan<F32>::create_from_impl(
      std::unique_ptr<Abstract::CorrelationPlanImpl<F32>> pimpl);
  template OMNIDSP_EXPORT std::unique_ptr<CorrelationPlan<F64>>
  CorrelationPlan<F64>::create_from_impl(
      std::unique_ptr<Abstract::CorrelationPlanImpl<F64>> pimpl);
  template OMNIDSP_EXPORT std::unique_ptr<CorrelationPlan<C32>>
  CorrelationPlan<C32>::create_from_impl(
      std::unique_ptr<Abstract::CorrelationPlanImpl<C32>> pimpl);
  template OMNIDSP_EXPORT std::unique_ptr<CorrelationPlan<C64>>
  CorrelationPlan<C64>::create_from_impl(
      std::unique_ptr<Abstract::CorrelationPlanImpl<C64>> pimpl);

  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<CorrelationPlan<F32>>>
  CorrelationPlan<F32>::create(
      const Abstract::Backend&,
      const std::vector<F32>&,
      ConvolutionType,
      ConvolutionMethod);
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<CorrelationPlan<F64>>>
  CorrelationPlan<F64>::create(
      const Abstract::Backend&,
      const std::vector<F64>&,
      ConvolutionType,
      ConvolutionMethod);
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<CorrelationPlan<C32>>>
  CorrelationPlan<C32>::create(
      const Abstract::Backend&,
      const std::vector<C32>&,
      ConvolutionType,
      ConvolutionMethod);
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<CorrelationPlan<C64>>>
  CorrelationPlan<C64>::create(
      const Abstract::Backend&,
      const std::vector<C64>&,
      ConvolutionType,
      ConvolutionMethod);

}  // namespace OmniDSP
