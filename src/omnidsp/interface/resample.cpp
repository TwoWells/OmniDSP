/**
 * @file resample.cpp
 * @brief Implements the ResamplePlan class methods, forwarding calls to backend
 * implementations (Pimpl pattern).
 */

#include <OmniDSP/core_types.hpp>  // For Status, OmniExpected, F32, F64, Utils::IsComplex_v etc.
#include <OmniDSP/omnidsp_export.hpp>  // For OMNIDSP_EXPORT
#include <OmniDSP/resample.hpp>  // Corresponding header (now contains declaration of create)
#include <expected>  // For std::unexpected
#include <memory>    // For std::unique_ptr
#include <span>
#include <stdexcept>    // For std::runtime_error
#include <type_traits>  // for std::is_same_v
#include <utility>      // For std::move, std::unexpect

#include "backend.hpp"  // Defines abstract::ResamplePlanImpl and Abstract::Backend

namespace OmniDSP {

  //--------------------------------------------------------------------------
  // ResamplePlan Method Definitions
  //--------------------------------------------------------------------------

  // Definition of the static create method
  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<ResamplePlan<T>>>
  ResamplePlan<T>::create(
      const Abstract::Backend& backend, const ResampleSpec& spec)
  {
    OmniExpected<std::unique_ptr<Abstract::ResamplePlanImpl<T>>> pimpl_expected;
    if constexpr (std::is_same_v<T, F32>) {
      pimpl_expected = backend.create_resample_plan_impl_f32(spec);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      pimpl_expected = backend.create_resample_plan_impl_f64(spec);
    }
    else {
      // This case should ideally be caught by the static_assert in ResamplePlan
      // declaration if T is constrained there. For safety, returning error.
      return std::unexpected(Status::UnsupportedFeature);
    }

    if (!pimpl_expected) {
      return std::unexpected(pimpl_expected.error());
    }

    // create_from_impl is static and defined in the header, so it can be called
    // here.
    auto plan
        = ResamplePlan<T>::create_from_impl(std::move(pimpl_expected.value()));
    if (!plan) {
      // This could happen if 'new ResamplePlan<T>(...)' in create_from_impl
      // fails
      return std::unexpected(Status::Failure);  // Or AllocationError
    }
    return plan;
  }

  // Destructor
  template <typename T>
  ResamplePlan<T>::~ResamplePlan() = default;

  // Move constructor
  template <typename T>
  ResamplePlan<T>::ResamplePlan(ResamplePlan&& other) noexcept = default;

  // Move assignment operator
  template <typename T>
  ResamplePlan<T>& ResamplePlan<T>::operator=(ResamplePlan&& other) noexcept
      = default;

  // Execute
  template <typename T>
  [[nodiscard]] Status ResamplePlan<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    return pimpl_->execute(input, output);
  }

  // Reset
  template <typename T>
  Status ResamplePlan<T>::reset()
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    return pimpl_->reset();
  }

  // get_input_rate
  template <typename T>
  double ResamplePlan<T>::get_input_rate() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ResamplePlan instance in get_input_rate.");
    }
    return pimpl_->get_input_rate();
  }

  // get_output_rate
  template <typename T>
  double ResamplePlan<T>::get_output_rate() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ResamplePlan instance in get_output_rate.");
    }
    return pimpl_->get_output_rate();
  }

  // get_output_length
  template <typename T>
  size_t ResamplePlan<T>::get_output_length(size_t input_length) const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ResamplePlan instance in get_output_length.");
    }
    return pimpl_->get_output_length(input_length);
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  template class OMNIDSP_EXPORT ResamplePlan<F32>;
  template class OMNIDSP_EXPORT ResamplePlan<F64>;

  // Explicitly instantiate the static create method for F32 and F64.
  // This ensures their definitions are compiled into the DLL.
  // The OMNIDSP_EXPORT here might be redundant if the class instantiation
  // already exports it, but it can be more explicit for MSVC.
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<ResamplePlan<F32>>>
  ResamplePlan<F32>::create(
      const Abstract::Backend& backend, const ResampleSpec& spec);
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<ResamplePlan<F64>>>
  ResamplePlan<F64>::create(
      const Abstract::Backend& backend, const ResampleSpec& spec);

}  // namespace OmniDSP
