/**
 * @file iir_filter.cpp
 * @brief Implements the IIRFilterProcessor class methods,
 * forwarding calls to backend implementations (Pimpl pattern).
 * TODO: Rename to IIRFilterProcessor if these become stateful.
 */

#include "OmniDSP/processor/iir_filter.hpp"  // Corresponding header for IIRFilterPlan

// Include the backend interface definition which declares the Impl classes
#include <memory>     // For std::unique_ptr
#include <span>       // For std::span
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move
#include <vector>     // For explicit template instantiations

#include "backend.hpp"  // Defines Abstract::IIRFilterPlanImpl
// TODO: This should become Abstract::IIRFilterProcessorImpl
#include "OmniDSP/core_types.hpp"  // For F32, F64

namespace OmniDSP::Processor {

  //--------------------------------------------------------------------------
  // Processor::IIRFilter Method Definitions
  //--------------------------------------------------------------------------

  /**
   * @brief Private constructor used by OmniDSP factory methods.
   */
  template <typename T>
  IIRFilter<T>::IIRFilter(std::unique_ptr<Abstract::IIRFilterProcessorImpl<T>>
                              pimpl)  // Adjust Impl type
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Processor::IIRFilter cannot be created with a null "
          "implementation pointer.");
    }
  }

  /**
   * @brief Destructor.
   */
  template <typename T>
  IIRFilter<T>::~IIRFilter() = default;

  /**
   * @brief Move constructor.
   */
  template <typename T>
  IIRFilter<T>::IIRFilter(IIRFilter&& other) noexcept = default;

  /**
   * @brief Move assignment operator.
   */
  template <typename T>
  IIRFilter<T>& IIRFilter<T>::operator=(IIRFilter&& other) noexcept = default;

  /**
   * @brief Applies the IIR filter (cascade of SOS) to an input signal.
   */
  template <typename T>
  [[nodiscard]] OmniStatus IIRFilter<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (!pimpl_) {
      return OmniStatus::InvalidOperation;
    }
    return pimpl_->execute(input, output);
  }

  /**
   * @brief Resets the internal state of the filter (delay elements).
   */
  template <typename T>
  OmniStatus IIRFilter<T>::reset()
  {
    if (!pimpl_) {
      return OmniStatus::InvalidOperation;
    }
    return pimpl_->reset();
  }

  /**
   * @brief Gets the order of the IIR filter.
   */
  template <typename T>
  size_t IIRFilter<T>::get_order() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Processor::IIRFilter instance: Implementation "
          "pointer is "
          "null in get_order.");
    }
    return pimpl_->get_order();
  }

  /**
   * @brief Gets the number of second-order sections used in the filter.
   */
  template <typename T>
  size_t IIRFilter<T>::get_num_sections() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Processor::IIRFilter instance: Implementation "
          "pointer is "
          "null in get_num_sections.");
    }
    return pimpl_->get_num_sections();
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations for Processor::IIRFilter
  //--------------------------------------------------------------------------

  template class OMNIDSP_EXPORT IIRFilter<F32>;
  template class OMNIDSP_EXPORT IIRFilter<F64>;

  // The static create_from_impl method is defined inline in the header,
  // so it does not need separate explicit instantiation here.

}  // namespace OmniDSP::Processor
