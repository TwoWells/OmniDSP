/**
 * @file filter.cpp
 * @brief Implements the FIRFilterPlan and IIRFilterPlan class methods,
 * forwarding calls to backend implementations (Pimpl pattern).
 */

#include "OmniDSP/filter.hpp"  // Corresponding header

// Include the backend interface definition which declares the Impl classes
#include <memory>     // For std::unique_ptr
#include <span>       // For std::span
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move
#include <vector>     // For explicit template instantiations

#include "backend.hpp"

namespace OmniDSP {

  //--------------------------------------------------------------------------
  // FIRFilterPlan Method Definitions
  //--------------------------------------------------------------------------

  /**
   * @brief Private constructor used by OmniDSP factory methods.
   * Takes ownership of the backend-specific implementation object.
   * @param pimpl A unique_ptr to the backend-specific implementation.
   * @throws std::runtime_error if pimpl is null.
   */
  template <typename T>
  FIRFilterPlan<T>::FIRFilterPlan(
      std::unique_ptr<Abstract::FIRFilterPlanImpl<T>> pimpl)
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "FIRFilterPlan cannot be created with a null implementation "
          "pointer.");
    }
  }

  /**
   * @brief Destructor.
   * The unique_ptr pimpl_ automatically deletes the managed implementation
   * object.
   */
  template <typename T>
  FIRFilterPlan<T>::~FIRFilterPlan() = default;

  /**
   * @brief Move constructor.
   * Transfers ownership of the implementation pointer.
   */
  template <typename T>
  FIRFilterPlan<T>::FIRFilterPlan(FIRFilterPlan&& other) noexcept = default;

  /**
   * @brief Move assignment operator.
   * Transfers ownership of the implementation pointer.
   */
  template <typename T>
  FIRFilterPlan<T>& FIRFilterPlan<T>::operator=(FIRFilterPlan&& other) noexcept
      = default;

  /**
   * @brief Applies the FIR filter to an input signal by calling the backend
   * implementation.
   * @param input A span representing the input signal segment.
   * @param output A span representing the output buffer. Must be large enough
   * to hold the corresponding output segment.
   * @return Status::Success on success, or an error code on failure.
   * Returns Status::InvalidOperation if the plan's implementation is missing.
   */
  template <typename T>
  [[nodiscard]] Status FIRFilterPlan<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    // Forward the call to the backend implementation.
    // The Impl execute might modify state, but the public interface is const.
    // We assume the Impl handles state correctly (e.g., mutable members).
    // If Impl::execute were const, this const_cast wouldn't be needed.
    // Since FIR filter state IS modified, Impl::execute must be non-const.
    return pimpl_->execute(input, output);
  }

  /**
   * @brief Resets the internal state of the filter by calling the backend
   * implementation.
   * @return Status::Success on success, or an error code if resetting fails.
   * Returns Status::InvalidOperation if the plan's implementation is missing.
   */
  template <typename T>
  Status FIRFilterPlan<T>::reset()
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    // Forward the call to the backend implementation.
    return pimpl_->reset();
  }

  /**
   * @brief Gets the order of the FIR filter by calling the backend
   * implementation.
   * @return The filter order (number of taps - 1).
   * @throws std::runtime_error if the plan's implementation is missing.
   */
  template <typename T>
  size_t FIRFilterPlan<T>::get_order() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid FIRFilterPlan instance: Implementation pointer is null in "
          "get_order.");
    }
    return pimpl_->get_order();
  }

  /**
   * @brief Gets the number of taps (coefficients) in the FIR filter by calling
   * the backend implementation.
   * @return The number of filter taps.
   * @throws std::runtime_error if the plan's implementation is missing.
   */
  template <typename T>
  size_t FIRFilterPlan<T>::get_num_taps() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid FIRFilterPlan instance: Implementation pointer is null in "
          "get_num_taps.");
    }
    return pimpl_->get_num_taps();
  }

  //--------------------------------------------------------------------------
  // IIRFilterPlan Method Definitions
  //--------------------------------------------------------------------------

  /**
   * @brief Private constructor used by OmniDSP factory methods.
   * Takes ownership of the backend-specific implementation object.
   * @param pimpl A unique_ptr to the backend-specific implementation.
   * @throws std::runtime_error if pimpl is null.
   */
  template <typename T>
  IIRFilterPlan<T>::IIRFilterPlan(
      std::unique_ptr<Abstract::IIRFilterPlanImpl<T>> pimpl)
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "IIRFilterPlan cannot be created with a null implementation "
          "pointer.");
    }
  }

  /**
   * @brief Destructor.
   * The unique_ptr pimpl_ automatically deletes the managed implementation
   * object.
   */
  template <typename T>
  IIRFilterPlan<T>::~IIRFilterPlan() = default;

  /**
   * @brief Move constructor.
   * Transfers ownership of the implementation pointer.
   */
  template <typename T>
  IIRFilterPlan<T>::IIRFilterPlan(IIRFilterPlan&& other) noexcept = default;

  /**
   * @brief Move assignment operator.
   * Transfers ownership of the implementation pointer.
   */
  template <typename T>
  IIRFilterPlan<T>& IIRFilterPlan<T>::operator=(IIRFilterPlan&& other) noexcept
      = default;

  /**
   * @brief Applies the IIR filter (cascade of SOS) to an input signal by
   * calling the backend implementation.
   * @param input A span representing the input signal segment.
   * @param output A span representing the output buffer. Must be large enough
   * to hold the corresponding output segment.
   * @return Status::Success on success, or an error code on failure.
   * Returns Status::InvalidOperation if the plan's implementation is missing.
   */
  template <typename T>
  [[nodiscard]] Status IIRFilterPlan<T>::execute(
      std::span<const T> input, std::span<T> output)
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    // Forward the call to the backend implementation.
    return pimpl_->execute(input, output);
  }

  /**
   * @brief Resets the internal state of the filter (delay elements) by calling
   * the backend implementation.
   * @return Status::Success on success, or an error code if resetting fails.
   * Returns Status::InvalidOperation if the plan's implementation is missing.
   */
  template <typename T>
  Status IIRFilterPlan<T>::reset()
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    // Forward the call to the backend implementation.
    return pimpl_->reset();
  }

  /**
   * @brief Gets the order of the IIR filter by calling the backend
   * implementation.
   * @return The filter order.
   * @throws std::runtime_error if the plan's implementation is missing.
   */
  template <typename T>
  size_t IIRFilterPlan<T>::get_order() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid IIRFilterPlan instance: Implementation pointer is null in "
          "get_order.");
    }
    return pimpl_->get_order();
  }

  /**
   * @brief Gets the number of second-order sections used in the filter
   * implementation by calling the backend implementation.
   * @return The number of SOS sections.
   * @throws std::runtime_error if the plan's implementation is missing.
   */
  template <typename T>
  size_t IIRFilterPlan<T>::get_num_sections() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid IIRFilterPlan instance: Implementation pointer is null in "
          "get_num_sections.");
    }
    return pimpl_->get_num_sections();
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  // Instantiate templates for common types to ensure code generation.

  // FIRFilterPlan (Real and Complex)
  template class FIRFilterPlan<F32>;
  template class FIRFilterPlan<F64>;
  template class FIRFilterPlan<C32>;
  template class FIRFilterPlan<C64>;

  // IIRFilterPlan (Typically Real)
  template class IIRFilterPlan<F32>;
  template class IIRFilterPlan<F64>;

}  // namespace OmniDSP
