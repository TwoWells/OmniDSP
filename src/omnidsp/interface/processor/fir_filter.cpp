/**
 * @file fir_filter.cpp
 * @brief Implements the Processor::FIRFilter class methods,
 * forwarding calls to backend implementations (Pimpl pattern).
 */

#include "OmniDSP/processor/fir_filter.hpp"

// Include the backend interface definition which declares the Impl classes
#include <memory>     // For std::unique_ptr
#include <span>       // For std::span
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move
#include <vector>     // For explicit template instantiations

#include "OmniDSP/core_types.hpp"  // For F32, F64, C32, C64
#include "backend.hpp"

namespace OmniDSP::Processor {

  //--------------------------------------------------------------------------
  // Processor::FIRFilterProcessor Method Definitions
  //--------------------------------------------------------------------------

  /**
   * @brief Private constructor used by OmniDSP factory methods.
   * Takes ownership of the backend-specific implementation object.
   * @param pimpl A unique_ptr to the backend-specific implementation.
   * @throws std::runtime_error if pimpl is null.
   */
  template <typename T>
  FIRFilter<T>::FIRFilter(std::unique_ptr<Abstract::FIRFilterProcessorImpl<T>>
                              pimpl)  // Adjust Impl type
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Processor::FIRFilter cannot be created with a null "
          "implementation pointer.");
    }
  }

  /**
   * @brief Destructor.
   */
  template <typename T>
  FIRFilter<T>::~FIRFilter() = default;

  /**
   * @brief Move constructor.
   */
  template <typename T>
  FIRFilter<T>::FIRFilter(FIRFilter&& other) noexcept = default;

  /**
   * @brief Move assignment operator.
   */
  template <typename T>
  FIRFilter<T>& FIRFilter<T>::operator=(FIRFilter&& other) noexcept = default;

  /**
   * @brief Applies the FIR filter to an input signal.
   */
  template <typename T>
  [[nodiscard]] OmniStatus FIRFilter<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (!pimpl_) {
      return OmniStatus::InvalidOperation;
    }
    return pimpl_->execute(input, output);
  }

  /**
   * @brief Resets the internal state of the filter.
   */
  template <typename T>
  OmniStatus FIRFilter<T>::reset()
  {
    if (!pimpl_) {
      return OmniStatus::InvalidOperation;
    }
    return pimpl_->reset();
  }

  /**
   * @brief Gets the order of the FIR filter.
   */
  template <typename T>
  size_t FIRFilter<T>::get_order() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Processor::FIRFilter instance: Implementation "
          "pointer is "
          "null in get_order.");
    }
    return pimpl_->get_order();
  }

  /**
   * @brief Gets the number of taps (coefficients) in the FIR filter.
   */
  template <typename T>
  size_t FIRFilter<T>::get_num_taps() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid Processor::FIRFilter instance: Implementation "
          "pointer is "
          "null in get_num_taps.");
    }
    return pimpl_->get_num_taps();
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations for Processor::FIRFilter
  //--------------------------------------------------------------------------

  template class OMNIDSP_EXPORT FIRFilter<F32>;
  template class OMNIDSP_EXPORT FIRFilter<F64>;
  template class OMNIDSP_EXPORT FIRFilter<C32>;
  template class OMNIDSP_EXPORT FIRFilter<C64>;

  // The static create_from_impl method is defined inline in the header,
  // so it does not need separate explicit instantiation here.

}  // namespace OmniDSP::Processor
