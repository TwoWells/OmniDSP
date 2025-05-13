/**
 * @file fir_filter.cpp
 * @brief Implements the FIRFilterProcessor class methods,
 * forwarding calls to backend implementations (Pimpl pattern).
 * TODO: Rename to FIRFilterProcessor if these become stateful.
 */

#include "OmniDSP/fir_filter.hpp"  // Corresponding header for FIRFilterPlan

// Include the backend interface definition which declares the Impl classes
#include <memory>     // For std::unique_ptr
#include <span>       // For std::span
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move
#include <vector>     // For explicit template instantiations

#include "backend.hpp"  // Defines Abstract::FIRFilterPlanImpl
// TODO: This should become Abstract::FIRFilterProcessorImpl
#include "OmniDSP/core_types.hpp"  // For F32, F64, C32, C64

namespace OmniDSP {

  //--------------------------------------------------------------------------
  // FIRFilterProcessor Method Definitions
  // TODO: Rename to FIRFilterProcessor if stateful
  //--------------------------------------------------------------------------

  /**
   * @brief Private constructor used by OmniDSP factory methods.
   * Takes ownership of the backend-specific implementation object.
   * @param pimpl A unique_ptr to the backend-specific implementation.
   * @throws std::runtime_error if pimpl is null.
   */
  template <typename T>
  FIRFilterProcessor<T>::FIRFilterProcessor(  // Or FIRFilterProcessor
      std::unique_ptr<Abstract::FIRFilterProcessorImpl<T>>
          pimpl)  // Adjust Impl type
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "FIRFilterProcessor/Processor cannot be created with a null "
          "implementation pointer.");
    }
  }

  /**
   * @brief Destructor.
   */
  template <typename T>
  FIRFilterProcessor<T>::~FIRFilterProcessor()
      = default;  // Or FIRFilterProcessor

  /**
   * @brief Move constructor.
   */
  template <typename T>
  FIRFilterProcessor<T>::FIRFilterProcessor(FIRFilterProcessor&& other) noexcept
      = default;  // Or FIRFilterProcessor

  /**
   * @brief Move assignment operator.
   */
  template <typename T>
  FIRFilterProcessor<T>& FIRFilterProcessor<T>::operator=(
      FIRFilterProcessor&& other) noexcept
      = default;  // Or FIRFilterProcessor

  /**
   * @brief Applies the FIR filter to an input signal.
   */
  template <typename T>
  [[nodiscard]] Status FIRFilterProcessor<T>::execute(  // Or FIRFilterProcessor
      std::span<const T> input,
      std::span<T> output)
  {  // Potentially non-const if Processor
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    return pimpl_->execute(input, output);
  }

  /**
   * @brief Resets the internal state of the filter.
   */
  template <typename T>
  Status FIRFilterProcessor<T>::reset()
  {  // Or FIRFilterProcessor
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    return pimpl_->reset();
  }

  /**
   * @brief Gets the order of the FIR filter.
   */
  template <typename T>
  size_t FIRFilterProcessor<T>::get_order() const
  {  // Or FIRFilterProcessor
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid FIRFilterProcessor/Processor instance: Implementation "
          "pointer is "
          "null in get_order.");
    }
    return pimpl_->get_order();
  }

  /**
   * @brief Gets the number of taps (coefficients) in the FIR filter.
   */
  template <typename T>
  size_t FIRFilterProcessor<T>::get_num_taps() const
  {  // Or FIRFilterProcessor
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid FIRFilterProcessor/Processor instance: Implementation "
          "pointer is "
          "null in get_num_taps.");
    }
    return pimpl_->get_num_taps();
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations for FIRFilterProcessor
  //--------------------------------------------------------------------------
  // TODO: Rename to FIRFilterProcessor if stateful
  template class OMNIDSP_EXPORT FIRFilterProcessor<F32>;
  template class OMNIDSP_EXPORT FIRFilterProcessor<F64>;
  template class OMNIDSP_EXPORT FIRFilterProcessor<C32>;
  template class OMNIDSP_EXPORT FIRFilterProcessor<C64>;

  // The static create_from_impl method is defined inline in the header,
  // so it does not need separate explicit instantiation here.

}  // namespace OmniDSP
