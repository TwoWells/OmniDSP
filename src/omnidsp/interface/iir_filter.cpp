/**
 * @file iir_filter.cpp
 * @brief Implements the IIRFilterProcessor class methods,
 * forwarding calls to backend implementations (Pimpl pattern).
 * TODO: Rename to IIRFilterProcessor if these become stateful.
 */

#include "OmniDSP/iir_filter.hpp"  // Corresponding header for IIRFilterPlan

// Include the backend interface definition which declares the Impl classes
#include <memory>     // For std::unique_ptr
#include <span>       // For std::span
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move
#include <vector>     // For explicit template instantiations

#include "backend.hpp"  // Defines Abstract::IIRFilterPlanImpl
// TODO: This should become Abstract::IIRFilterProcessorImpl
#include "OmniDSP/core_types.hpp"  // For F32, F64

namespace OmniDSP {

  //--------------------------------------------------------------------------
  // IIRFilterProcessor Method Definitions
  // TODO: Rename to IIRFilterProcessor if stateful
  //--------------------------------------------------------------------------

  /**
   * @brief Private constructor used by OmniDSP factory methods.
   */
  template <typename T>
  IIRFilterProcessor<T>::IIRFilterProcessor(  // Or IIRFilterProcessor
      std::unique_ptr<Abstract::IIRFilterProcessorImpl<T>>
          pimpl)  // Adjust Impl type
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "IIRFilterProcessor/Processor cannot be created with a null "
          "implementation pointer.");
    }
  }

  /**
   * @brief Destructor.
   */
  template <typename T>
  IIRFilterProcessor<T>::~IIRFilterProcessor()
      = default;  // Or IIRFilterProcessor

  /**
   * @brief Move constructor.
   */
  template <typename T>
  IIRFilterProcessor<T>::IIRFilterProcessor(IIRFilterProcessor&& other) noexcept
      = default;  // Or IIRFilterProcessor

  /**
   * @brief Move assignment operator.
   */
  template <typename T>
  IIRFilterProcessor<T>& IIRFilterProcessor<T>::operator=(
      IIRFilterProcessor&& other) noexcept
      = default;  // Or IIRFilterProcessor

  /**
   * @brief Applies the IIR filter (cascade of SOS) to an input signal.
   */
  template <typename T>
  [[nodiscard]] OmniStatus
  IIRFilterProcessor<T>::execute(  // Or IIRFilterProcessor
      std::span<const T> input,
      std::span<T> output)
  {  // Potentially non-const if Processor
    if (!pimpl_) {
      return OmniStatus::InvalidOperation;
    }
    return pimpl_->execute(input, output);
  }

  /**
   * @brief Resets the internal state of the filter (delay elements).
   */
  template <typename T>
  OmniStatus IIRFilterProcessor<T>::reset()
  {  // Or IIRFilterProcessor
    if (!pimpl_) {
      return OmniStatus::InvalidOperation;
    }
    return pimpl_->reset();
  }

  /**
   * @brief Gets the order of the IIR filter.
   */
  template <typename T>
  size_t IIRFilterProcessor<T>::get_order() const
  {  // Or IIRFilterProcessor
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid IIRFilterProcessor/Processor instance: Implementation "
          "pointer is "
          "null in get_order.");
    }
    return pimpl_->get_order();
  }

  /**
   * @brief Gets the number of second-order sections used in the filter.
   */
  template <typename T>
  size_t IIRFilterProcessor<T>::get_num_sections() const
  {  // Or IIRFilterProcessor
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid IIRFilterProcessor/Processor instance: Implementation "
          "pointer is "
          "null in get_num_sections.");
    }
    return pimpl_->get_num_sections();
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations for IIRFilterProcessor
  //--------------------------------------------------------------------------
  // IIRFilterProcessor (Typically Real)
  // TODO: Rename to IIRFilterProcessor if stateful
  template class OMNIDSP_EXPORT IIRFilterProcessor<F32>;
  template class OMNIDSP_EXPORT IIRFilterProcessor<F64>;

  // The static create_from_impl method is defined inline in the header,
  // so it does not need separate explicit instantiation here.

}  // namespace OmniDSP
