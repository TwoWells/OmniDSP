/**
 * @file convolution.cpp
 * @brief Implements the ConvolutionPlan and CorrelationPlan class methods,
 * forwarding calls to backend implementations.
 */

#include <OmniDSP/convolution.hpp>  // Corresponding header

// Include the backend interface definition which declares the Impl classes
#include <memory>  // For std::unique_ptr
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

  template <typename T>
  ConvolutionPlan<T>::ConvolutionPlan(
      std::unique_ptr<abstract::ConvolutionPlanImpl<T>> pimpl)
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
    // Add size checks for robustness?
    // size_t expected_len = get_output_length(input.size());
    // if (output.size() < expected_len) return Status::SizeMismatch;
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

  // *** ADDED Definition ***
  template <typename T>
  ConvolutionMethod ConvolutionPlan<T>::get_method_hint() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ConvolutionPlan instance in get_method_hint.");
    }
    // Assumes ConvolutionPlanImpl has a get_method() method
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

  // *** ADDED Definition ***
  template <typename T>
  std::span<const T> ConvolutionPlan<T>::get_kernel() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ConvolutionPlan instance in get_kernel.");
    }
    // Assumes ConvolutionPlanImpl has a get_kernel() method
    return pimpl_->get_kernel();
  }

  //--------------------------------------------------------------------------
  // CorrelationPlan Method Definitions
  //--------------------------------------------------------------------------

  template <typename T>
  CorrelationPlan<T>::CorrelationPlan(
      std::unique_ptr<abstract::CorrelationPlanImpl<T>> pimpl)
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
    // Add size checks?
    // size_t expected_len = get_output_length(input.size());
    // if (output.size() < expected_len) return Status::SizeMismatch;
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

  // *** ADDED Definition ***
  template <typename T>
  ConvolutionMethod CorrelationPlan<T>::get_method_hint() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid CorrelationPlan instance in get_method_hint.");
    }
    // Assumes CorrelationPlanImpl has a get_method() method
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

  // *** ADDED Definition ***
  template <typename T>
  std::span<const T> CorrelationPlan<T>::get_template() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid CorrelationPlan instance in get_template.");
    }
    // Assumes CorrelationPlanImpl has a get_template() method
    return pimpl_->get_template();
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------

  // ConvolutionPlan Instantiations
  template class ConvolutionPlan<F32>;
  template class ConvolutionPlan<F64>;
  template class ConvolutionPlan<C32>;
  template class ConvolutionPlan<C64>;

  // CorrelationPlan Instantiations
  template class CorrelationPlan<F32>;
  template class CorrelationPlan<F64>;
  template class CorrelationPlan<C32>;
  template class CorrelationPlan<C64>;

}  // namespace OmniDSP
