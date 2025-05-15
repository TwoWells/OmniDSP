/**
 * @file convolution.cpp
 * @brief Implements the ConvolutionPlan and CorrelationPlan class methods,
 * forwarding calls to backend implementations.
 */

#include "OmniDSP/plan/convolution.hpp"  // Corresponding header

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
// Include Params types for Convolution and Correlation
#include <OmniDSP/params/convolution.hpp>  // For Params::Convolution, Params::Correlation

namespace OmniDSP::Plan {

  //--------------------------------------------------------------------------
  // Plan::Convolution Method Definitions
  //--------------------------------------------------------------------------

  // Static create_from_impl Definition
  template <typename T>
  std::unique_ptr<Convolution<T>> Convolution<T>::create_from_impl(
      std::unique_ptr<Abstract::ConvolutionPlanImpl<T>> pimpl)
  {
    if (!pimpl) {
      return nullptr;
    }
    // Calls the private constructor of ConvolutionPlan<T>
    // Using `new` directly as make_unique cannot access private constructor.
    // The std::unique_ptr will manage the memory.
    return std::unique_ptr<Convolution<T>>(
        new Convolution<T>(std::move(pimpl)));
  }

  // Static create Definition - Updated Signature
  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<Convolution<T>>>
  Convolution<T>::create(
      const Abstract::Backend& backend,
      const Params::Convolution& params,
      std::span<const T> kernel_coeffs)  // Changed parameters
  {
    OmniExpected<std::unique_ptr<Abstract::ConvolutionPlanImpl<T>>>
        pimpl_expected;
    if constexpr (std::is_same_v<T, F32>) {
      pimpl_expected = backend.create_convolution_plan_impl_f32(
          params, kernel_coeffs);  // Updated call
    }
    else if constexpr (std::is_same_v<T, F64>) {
      pimpl_expected = backend.create_convolution_plan_impl_f64(
          params, kernel_coeffs);  // Updated call
    }
    else if constexpr (std::is_same_v<T, C32>) {
      pimpl_expected = backend.create_convolution_plan_impl_c32(
          params, kernel_coeffs);  // Updated call
    }
    else if constexpr (std::is_same_v<T, C64>) {
      pimpl_expected = backend.create_convolution_plan_impl_c64(
          params, kernel_coeffs);  // Updated call
    }
    else {
      return std::unexpected(OmniStatus::UnsupportedFeature);
    }

    if (!pimpl_expected) {
      return std::unexpected(pimpl_expected.error());
    }

    auto plan
        = Convolution<T>::create_from_impl(std::move(pimpl_expected.value()));
    if (!plan) {
      return std::unexpected(OmniStatus::Failure);  // Or AllocationError
    }
    return plan;
  }

  template <typename T>
  Convolution<T>::Convolution(
      std::unique_ptr<Abstract::ConvolutionPlanImpl<T>> pimpl)
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Plan::Convolution created with null implementation.");
    }
  }

  template <typename T>
  Convolution<T>::~Convolution() = default;

  template <typename T>
  Convolution<T>::Convolution(Convolution&& other) noexcept = default;

  template <typename T>
  Convolution<T>& Convolution<T>::operator=(Convolution&& other) noexcept
      = default;

  template <typename T>
  [[nodiscard]] OmniStatus Convolution<T>::execute(
      std::span<const T> input, std::span<T> output) const
  {
    if (!pimpl_) {
      return OmniStatus::InvalidOperation;
    }
    return pimpl_->execute(input, output);
  }

  template <typename T>
  size_t Convolution<T>::get_kernel_length() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Plan::Convolution instance in get_kernel_length.");
    }
    return pimpl_->get_kernel_length();
  }

  template <typename T>
  ConvolutionType Convolution<T>::get_type() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Plan::Convolution instance in get_type.");
    }
    return pimpl_->get_type();
  }

  template <typename T>
  ConvolutionMethod Convolution<T>::get_method_hint() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Plan::Convolution instance in get_method_hint.");
    }
    return pimpl_->get_method();
  }

  template <typename T>
  size_t Convolution<T>::get_output_length(size_t input_length) const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Plan::Convolution instance in get_output_length.");
    }
    return pimpl_->get_output_length(input_length);
  }

  template <typename T>
  std::span<const T> Plan::Convolution<T>::get_kernel() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Plan::Convolution instance in get_kernel.");
    }
    return pimpl_->get_kernel();
  }

  //--------------------------------------------------------------------------
  // Plan::Correlation Method Definitions
  //--------------------------------------------------------------------------

  // Static create_from_impl Definition
  template <typename T>
  std::unique_ptr<Correlation<T>> Correlation<T>::create_from_impl(
      std::unique_ptr<Abstract::CorrelationPlanImpl<T>> pimpl)
  {
    if (!pimpl) {
      return nullptr;
    }
    return std::unique_ptr<Correlation<T>>(
        new Correlation<T>(std::move(pimpl)));
  }

  // Static create Definition - Updated Signature
  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<Correlation<T>>>
  Correlation<T>::create(
      const Abstract::Backend& backend,
      const Params::Correlation& params,   // Changed parameter
      std::span<const T> template_coeffs)  // Changed parameter
  {
    OmniExpected<std::unique_ptr<Abstract::CorrelationPlanImpl<T>>>
        pimpl_expected;
    if constexpr (std::is_same_v<T, F32>) {
      pimpl_expected = backend.create_correlation_plan_impl_f32(
          params, template_coeffs);  // Updated call
    }
    else if constexpr (std::is_same_v<T, F64>) {
      pimpl_expected = backend.create_correlation_plan_impl_f64(
          params, template_coeffs);  // Updated call
    }
    else if constexpr (std::is_same_v<T, C32>) {
      pimpl_expected = backend.create_correlation_plan_impl_c32(
          params, template_coeffs);  // Updated call
    }
    else if constexpr (std::is_same_v<T, C64>) {
      pimpl_expected = backend.create_correlation_plan_impl_c64(
          params, template_coeffs);  // Updated call
    }
    else {
      return std::unexpected(OmniStatus::UnsupportedFeature);
    }

    if (!pimpl_expected) {
      return std::unexpected(pimpl_expected.error());
    }

    auto plan
        = Correlation<T>::create_from_impl(std::move(pimpl_expected.value()));
    if (!plan) {
      return std::unexpected(OmniStatus::Failure);  // Or AllocationError
    }
    return plan;
  }

  template <typename T>
  Correlation<T>::Correlation(
      std::unique_ptr<Abstract::CorrelationPlanImpl<T>> pimpl)
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Plan::Correlation created with null implementation.");
    }
  }

  template <typename T>
  Correlation<T>::~Correlation() = default;

  template <typename T>
  Correlation<T>::Correlation(Correlation&& other) noexcept = default;

  template <typename T>
  Correlation<T>& Correlation<T>::operator=(Correlation&& other) noexcept
      = default;

  template <typename T>
  [[nodiscard]] OmniStatus Correlation<T>::execute(
      std::span<const T> input, std::span<T> output) const
  {
    if (!pimpl_) {
      return OmniStatus::InvalidOperation;
    }
    return pimpl_->execute(input, output);
  }

  template <typename T>
  size_t Correlation<T>::get_template_length() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Plan::Correlation instance in get_template_length.");
    }
    return pimpl_->get_template_length();
  }

  template <typename T>
  ConvolutionType Correlation<T>::get_type() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Plan::Correlation instance in get_type.");
    }
    return pimpl_->get_type();
  }

  template <typename T>
  ConvolutionMethod Correlation<T>::get_method_hint() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Plan::Correlation instance in get_method_hint.");
    }
    return pimpl_->get_method();
  }

  template <typename T>
  size_t Correlation<T>::get_output_length(size_t input_length) const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Plan::Correlation instance in get_output_length.");
    }
    return pimpl_->get_output_length(input_length);
  }

  template <typename T>
  std::span<const T> Correlation<T>::get_template() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Plan::Correlation instance in get_template.");
    }
    return pimpl_->get_template();
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------

  template class OMNIDSP_EXPORT Convolution<F32>;
  template class OMNIDSP_EXPORT Convolution<F64>;
  template class OMNIDSP_EXPORT Convolution<C32>;
  template class OMNIDSP_EXPORT Convolution<C64>;

  template class OMNIDSP_EXPORT Correlation<F32>;
  template class OMNIDSP_EXPORT Correlation<F64>;
  template class OMNIDSP_EXPORT Correlation<C32>;
  template class OMNIDSP_EXPORT Correlation<C64>;

  // Explicitly instantiate static members for export
  // For Plan::Convolution
  template OMNIDSP_EXPORT std::unique_ptr<Convolution<F32>>
  Convolution<F32>::create_from_impl(
      std::unique_ptr<Abstract::ConvolutionPlanImpl<F32>> pimpl);
  template OMNIDSP_EXPORT std::unique_ptr<Convolution<F64>>
  Convolution<F64>::create_from_impl(
      std::unique_ptr<Abstract::ConvolutionPlanImpl<F64>> pimpl);
  template OMNIDSP_EXPORT std::unique_ptr<Convolution<C32>>
  Convolution<C32>::create_from_impl(
      std::unique_ptr<Abstract::ConvolutionPlanImpl<C32>> pimpl);
  template OMNIDSP_EXPORT std::unique_ptr<Convolution<C64>>
  Convolution<C64>::create_from_impl(
      std::unique_ptr<Abstract::ConvolutionPlanImpl<C64>> pimpl);

  // Updated signatures for explicit instantiations of ::create
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Convolution<F32>>>
  Convolution<F32>::create(
      const Abstract::Backend&,
      const Params::Convolution&,  // Updated
      std::span<const F32>);       // Updated
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Convolution<F64>>>
  Convolution<F64>::create(
      const Abstract::Backend&,
      const Params::Convolution&,  // Updated
      std::span<const F64>);       // Updated
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Convolution<C32>>>
  Convolution<C32>::create(
      const Abstract::Backend&,
      const Params::Convolution&,  // Updated
      std::span<const C32>);       // Updated
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Convolution<C64>>>
  Convolution<C64>::create(
      const Abstract::Backend&,
      const Params::Convolution&,  // Updated
      std::span<const C64>);       // Updated

  // For Plan::Correlation
  template OMNIDSP_EXPORT std::unique_ptr<Correlation<F32>>
  Correlation<F32>::create_from_impl(
      std::unique_ptr<Abstract::CorrelationPlanImpl<F32>> pimpl);
  template OMNIDSP_EXPORT std::unique_ptr<Correlation<F64>>
  Correlation<F64>::create_from_impl(
      std::unique_ptr<Abstract::CorrelationPlanImpl<F64>> pimpl);
  template OMNIDSP_EXPORT std::unique_ptr<Correlation<C32>>
  Correlation<C32>::create_from_impl(
      std::unique_ptr<Abstract::CorrelationPlanImpl<C32>> pimpl);
  template OMNIDSP_EXPORT std::unique_ptr<Correlation<C64>>
  Correlation<C64>::create_from_impl(
      std::unique_ptr<Abstract::CorrelationPlanImpl<C64>> pimpl);

  // Updated signatures for explicit instantiations of ::create
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Correlation<F32>>>
  Correlation<F32>::create(
      const Abstract::Backend&,
      const Params::Correlation&,  // Updated
      std::span<const F32>);       // Updated
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Correlation<F64>>>
  Correlation<F64>::create(
      const Abstract::Backend&,
      const Params::Correlation&,  // Updated
      std::span<const F64>);       // Updated
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Correlation<C32>>>
  Correlation<C32>::create(
      const Abstract::Backend&,
      const Params::Correlation&,  // Updated
      std::span<const C32>);       // Updated
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<Correlation<C64>>>
  Correlation<C64>::create(
      const Abstract::Backend&,
      const Params::Correlation&,  // Updated
      std::span<const C64>);       // Updated

}  // namespace OmniDSP::Plan
