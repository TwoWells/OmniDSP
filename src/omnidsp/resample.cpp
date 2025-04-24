/**
 * @file resample.cpp
 * @brief Implements the ResamplePlan class methods, forwarding calls to backend
 * implementations (Pimpl pattern).
 */

#include "OmniDSP/resample.h"  // Corresponding header

// Include the backend interface definition which declares ResamplePlanImpl
#include <memory>  // For std::unique_ptr
#include <span>
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move

#include "backend/backend.h"

namespace OmniDSP {

  //--------------------------------------------------------------------------
  // ResamplePlan Method Definitions
  //--------------------------------------------------------------------------

  /**
   * @brief Private constructor used by OmniDSP factory methods.
   * Takes ownership of the backend-specific implementation object.
   * @param pimpl A unique_ptr to the backend-specific implementation.
   * @throws std::runtime_error if pimpl is null.
   */
  template <typename T>
  ResamplePlan<T>::ResamplePlan(
      std::unique_ptr<backend::ResamplePlanImpl<T>> pimpl)
      : pimpl_(std::move(pimpl))
  {
    // Constructor body can be empty, initialization is done via member
    // initializer list
    if (!pimpl_) {
      // This should ideally be caught by the factory creating the Impl,
      // but check here as a safeguard.
      throw std::runtime_error(
          "ResamplePlan cannot be created with a null implementation pointer.");
    }
  }

  /**
   * @brief Destructor.
   * The unique_ptr pimpl_ automatically deletes the managed implementation
   * object.
   */
  template <typename T>
  ResamplePlan<T>::~ResamplePlan() = default;

  /**
   * @brief Move constructor.
   * Transfers ownership of the implementation pointer.
   */
  template <typename T>
  ResamplePlan<T>::ResamplePlan(ResamplePlan&& other) noexcept = default;

  /**
   * @brief Move assignment operator.
   * Transfers ownership of the implementation pointer.
   */
  template <typename T>
  ResamplePlan<T>& ResamplePlan<T>::operator=(ResamplePlan&& other) noexcept
      = default;

  /**
   * @brief Executes the pre-planned resampling operation by calling the backend
   * implementation.
   * @param input A span representing the input signal buffer.
   * @param output A span representing the output buffer for the resampled
   * signal.
   * @return Status::Success on success, or an error code on failure.
   * Returns Status::InvalidOperation if the plan's implementation is missing.
   */
  template <typename T>
  [[nodiscard]] Status ResamplePlan<T>::execute(
      std::span<const T> input, std::span<T> output) const
  {
    if (!pimpl_) {
      // Should not happen if constructor validates, but defensive check
      return Status::InvalidOperation;
    }
    // Forward the call to the actual backend implementation.
    // Note: The Impl execute is non-const, but the public API execute is const.
    // This requires the pimpl_ pointer itself to be const, but the object it
    // points to can have its non-const execute method called. If the Impl
    // execute needs to modify the Impl state, the Impl members must be mutable.
    // Let's cast away constness here, assuming the Impl execute modifies
    // mutable state. A potentially cleaner design might make the public execute
    // non-const if state modification is expected. However, following the
    // pattern of FFTPlan etc., we keep public execute const.
    return const_cast<backend::ResamplePlanImpl<T>*>(pimpl_.get())
        ->execute(input, output);
  }

  /**
   * @brief Resets the internal state of the resampler by calling the backend
   * implementation.
   * @return Status::Success on success, or an error code if resetting fails.
   * Returns Status::InvalidOperation if the plan's implementation is missing.
   */
  template <typename T>
  Status ResamplePlan<T>::reset()
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    // Forward the call to the actual backend implementation.
    return pimpl_->reset();
  }

  /**
   * @brief Gets the input sample rate by calling the backend implementation.
   * @return The input sample rate in Hz.
   * @throws std::runtime_error if the plan's implementation is missing.
   */
  template <typename T>
  double ResamplePlan<T>::get_input_rate() const
  {
    if (!pimpl_) {
      // Throwing here because returning 0.0 could be misleading.
      throw std::runtime_error(
          "Invalid ResamplePlan instance: Implementation pointer is null in "
          "get_input_rate.");
    }
    return pimpl_->get_input_rate();
  }

  /**
   * @brief Gets the output sample rate by calling the backend implementation.
   * @return The output sample rate in Hz.
   * @throws std::runtime_error if the plan's implementation is missing.
   */
  template <typename T>
  double ResamplePlan<T>::get_output_rate() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ResamplePlan instance: Implementation pointer is null in "
          "get_output_rate.");
    }
    return pimpl_->get_output_rate();
  }

  /**
   * @brief Estimates the required output length by calling the backend
   * implementation.
   * @param input_length The length of the input signal.
   * @return An estimated length for the output signal buffer.
   * @throws std::runtime_error if the plan's implementation is missing.
   */
  template <typename T>
  size_t ResamplePlan<T>::get_output_length(size_t input_length) const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ResamplePlan instance: Implementation pointer is null in "
          "get_output_length.");
    }
    // Forward the calculation to the implementation, which knows about filter
    // delays etc.
    return pimpl_->get_output_length(input_length);
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  // Instantiate templates for common types (float, double) to ensure code
  // generation for the public ResamplePlan class.

  template class OmniDSP::ResamplePlan<float>;
  template class OmniDSP::ResamplePlan<double>;

}  // namespace OmniDSP
