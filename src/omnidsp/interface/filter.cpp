/**
 * @file filter.cpp
 * @brief Implements the FIRFilterPlan and IIRFilterPlan class methods,
 * forwarding calls to backend implementations (Pimpl pattern).
 * TODO: Rename to FIRFilterProcessor and IIRFilterProcessor if these become
 * stateful.
 */

#include "OmniDSP/filter.hpp"  // Corresponding header

// Include the backend interface definition which declares the Impl classes
#include <memory>     // For std::unique_ptr
#include <span>       // For std::span
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move
#include <vector>     // For explicit template instantiations

#include "backend.hpp"  // Defines Abstract::FIRFilterPlanImpl, Abstract::IIRFilterPlanImpl
// TODO: These should become Abstract::FIRFilterProcessorImpl etc.
#include "OmniDSP/core_types.hpp"      // For F32, F64, C32, C64
#include "OmniDSP/omnidsp_export.hpp"  // For OMNIDSP_EXPORT

namespace OmniDSP {

  //--------------------------------------------------------------------------
  // FIRFilterPlan Method Definitions
  // TODO: Rename to FIRFilterProcessor if stateful
  //--------------------------------------------------------------------------

  /**
   * @brief Private constructor used by OmniDSP factory methods.
   * Takes ownership of the backend-specific implementation object.
   * @param pimpl A unique_ptr to the backend-specific implementation.
   * @throws std::runtime_error if pimpl is null.
   */
  template <typename T>
  FIRFilterPlan<T>::FIRFilterPlan(  // Or FIRFilterProcessor
      std::unique_ptr<Abstract::FIRFilterPlanImpl<T>>
          pimpl)  // Adjust Impl type
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "FIRFilterPlan/Processor cannot be created with a null "
          "implementation pointer.");
    }
  }

  /**
   * @brief Destructor.
   */
  template <typename T>
  FIRFilterPlan<T>::~FIRFilterPlan() = default;  // Or FIRFilterProcessor

  /**
   * @brief Move constructor.
   */
  template <typename T>
  FIRFilterPlan<T>::FIRFilterPlan(FIRFilterPlan&& other) noexcept
      = default;  // Or FIRFilterProcessor

  /**
   * @brief Move assignment operator.
   */
  template <typename T>
  FIRFilterPlan<T>& FIRFilterPlan<T>::operator=(FIRFilterPlan&& other) noexcept
      = default;  // Or FIRFilterProcessor

  /**
   * @brief Applies the FIR filter to an input signal.
   */
  template <typename T>
  [[nodiscard]] Status FIRFilterPlan<T>::execute(  // Or FIRFilterProcessor
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
  Status FIRFilterPlan<T>::reset()
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
  size_t FIRFilterPlan<T>::get_order() const
  {  // Or FIRFilterProcessor
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid FIRFilterPlan/Processor instance: Implementation pointer is "
          "null in get_order.");
    }
    return pimpl_->get_order();
  }

  /**
   * @brief Gets the number of taps (coefficients) in the FIR filter.
   */
  template <typename T>
  size_t FIRFilterPlan<T>::get_num_taps() const
  {  // Or FIRFilterProcessor
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid FIRFilterPlan/Processor instance: Implementation pointer is "
          "null in get_num_taps.");
    }
    return pimpl_->get_num_taps();
  }

  //--------------------------------------------------------------------------
  // IIRFilterPlan Method Definitions
  // TODO: Rename to IIRFilterProcessor if stateful
  //--------------------------------------------------------------------------

  /**
   * @brief Private constructor used by OmniDSP factory methods.
   */
  template <typename T>
  IIRFilterPlan<T>::IIRFilterPlan(  // Or IIRFilterProcessor
      std::unique_ptr<Abstract::IIRFilterPlanImpl<T>>
          pimpl)  // Adjust Impl type
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "IIRFilterPlan/Processor cannot be created with a null "
          "implementation pointer.");
    }
  }

  /**
   * @brief Destructor.
   */
  template <typename T>
  IIRFilterPlan<T>::~IIRFilterPlan() = default;  // Or IIRFilterProcessor

  /**
   * @brief Move constructor.
   */
  template <typename T>
  IIRFilterPlan<T>::IIRFilterPlan(IIRFilterPlan&& other) noexcept
      = default;  // Or IIRFilterProcessor

  /**
   * @brief Move assignment operator.
   */
  template <typename T>
  IIRFilterPlan<T>& IIRFilterPlan<T>::operator=(IIRFilterPlan&& other) noexcept
      = default;  // Or IIRFilterProcessor

  /**
   * @brief Applies the IIR filter (cascade of SOS) to an input signal.
   */
  template <typename T>
  [[nodiscard]] Status IIRFilterPlan<T>::execute(  // Or IIRFilterProcessor
      std::span<const T> input,
      std::span<T> output)
  {  // Potentially non-const if Processor
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    return pimpl_->execute(input, output);
  }

  /**
   * @brief Resets the internal state of the filter (delay elements).
   */
  template <typename T>
  Status IIRFilterPlan<T>::reset()
  {  // Or IIRFilterProcessor
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    return pimpl_->reset();
  }

  /**
   * @brief Gets the order of the IIR filter.
   */
  template <typename T>
  size_t IIRFilterPlan<T>::get_order() const
  {  // Or IIRFilterProcessor
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid IIRFilterPlan/Processor instance: Implementation pointer is "
          "null in get_order.");
    }
    return pimpl_->get_order();
  }

  /**
   * @brief Gets the number of second-order sections used in the filter.
   */
  template <typename T>
  size_t IIRFilterPlan<T>::get_num_sections() const
  {  // Or IIRFilterProcessor
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid IIRFilterPlan/Processor instance: Implementation pointer is "
          "null in get_num_sections.");
    }
    return pimpl_->get_num_sections();
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------

  // FIRFilterPlan (Real and Complex)
  // TODO: Rename to FIRFilterProcessor if stateful
  template class OMNIDSP_EXPORT FIRFilterPlan<F32>;
  template class OMNIDSP_EXPORT FIRFilterPlan<F64>;
  template class OMNIDSP_EXPORT FIRFilterPlan<C32>;
  template class OMNIDSP_EXPORT FIRFilterPlan<C64>;

  // IIRFilterPlan (Typically Real)
  // TODO: Rename to IIRFilterProcessor if stateful
  template class OMNIDSP_EXPORT IIRFilterPlan<F32>;
  template class OMNIDSP_EXPORT IIRFilterPlan<F64>;

  // The static create_from_impl methods are defined inline in the header,
  // so they do not need separate explicit instantiation here.

}  // namespace OmniDSP
