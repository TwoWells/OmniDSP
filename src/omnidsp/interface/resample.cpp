/**
 * @file resample.cpp
 * @brief Implements the ResampleProcessor class methods, forwarding calls to
 * backend implementations (Pimpl pattern).
 */

#include <OmniDSP/core_types.hpp>
#include <OmniDSP/omnidsp_export.hpp>
#include <OmniDSP/resample.hpp>
#include <expected>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "backend.hpp"  // Defines ::OmniDSP::Abstract::ResamplePlanImpl and ::OmniDSP::Abstract::Backend

namespace OmniDSP {

  //--------------------------------------------------------------------------
  // ResampleProcessor Method Definitions
  //--------------------------------------------------------------------------

  // Private constructor definition is NOW IN THE HEADER (resample.hpp)
  // template <typename T>
  // ResampleProcessor<T>::ResampleProcessor(std::unique_ptr<::OmniDSP::Abstract::ResampleProcessorImpl<T>>
  // pimpl)
  //     : pimpl_(std::move(pimpl)) {
  //   if (!pimpl_) {
  //     throw std::runtime_error(
  //         "ResampleProcessor constructed with null implementation pointer.");
  //   }
  // }

  // Static create method definition
  template <typename T>
  [[nodiscard]] OmniExpected<std::unique_ptr<ResampleProcessor<T>>>
  ResampleProcessor<T>::create(
      const ::OmniDSP::Abstract::Backend& backend,
      const Design::Resample& design)
  {
    OmniExpected<std::unique_ptr<::OmniDSP::Abstract::ResampleProcessorImpl<T>>>
        pimpl_expected;
    if constexpr (std::is_same_v<T, F32>) {
      pimpl_expected = backend.create_resample_processor_impl_f32(design);
    }
    else if constexpr (std::is_same_v<T, F64>) {
      pimpl_expected = backend.create_resample_processor_impl_f64(design);
    }
    else {
      return std::unexpected(Status::UnsupportedFeature);
    }

    if (!pimpl_expected) {
      return std::unexpected(pimpl_expected.error());
    }

    // create_from_impl is static and inline in the header, so it's directly
    // usable
    auto plan = ResampleProcessor<T>::create_from_impl(
        std::move(pimpl_expected.value()));
    if (!plan) {
      // This could happen if 'new ResampleProcessor<T>(...)' in
      // create_from_impl fails (e.g. bad_alloc) or if pimpl_expected.value()
      // was somehow null after a successful expected.
      return std::unexpected(Status::Failure);
    }
    return plan;
  }

  template <typename T>
  ResampleProcessor<T>::~ResampleProcessor() = default;

  template <typename T>
  ResampleProcessor<T>::ResampleProcessor(ResampleProcessor&& other) noexcept
      = default;

  template <typename T>
  ResampleProcessor<T>& ResampleProcessor<T>::operator=(
      ResampleProcessor&& other) noexcept
      = default;

  template <typename T>
  [[nodiscard]] Status ResampleProcessor<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    return pimpl_->execute(input, output);
  }

  template <typename T>
  Status ResampleProcessor<T>::reset()
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    return pimpl_->reset();
  }

  template <typename T>
  double ResampleProcessor<T>::get_input_rate() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ResampleProcessor instance in get_input_rate.");
    }
    return pimpl_->get_input_rate();
  }

  template <typename T>
  double ResampleProcessor<T>::get_output_rate() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ResampleProcessor instance in get_output_rate.");
    }
    return pimpl_->get_output_rate();
  }

  template <typename T>
  size_t ResampleProcessor<T>::get_output_length(size_t input_length) const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ResampleProcessor instance in get_output_length.");
    }
    return pimpl_->get_output_length(input_length);
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  template class OMNIDSP_EXPORT ResampleProcessor<F32>;
  template class OMNIDSP_EXPORT ResampleProcessor<F64>;

  // Explicitly instantiate the static create method
  // The OMNIDSP_EXPORT here ensures visibility if this is part of a shared
  // library.
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<ResampleProcessor<F32>>>
  ResampleProcessor<F32>::create(
      const ::OmniDSP::Abstract::Backend&, const Design::Resample&);
  template OMNIDSP_EXPORT OmniExpected<std::unique_ptr<ResampleProcessor<F64>>>
  ResampleProcessor<F64>::create(
      const ::OmniDSP::Abstract::Backend&, const Design::Resample&);

}  // namespace OmniDSP
