/**
 * @file resample.cpp
 * @brief Implements the ResamplePlan class methods, forwarding calls to backend
 * implementations (Pimpl pattern).
 */

#include <OmniDSP/resample.hpp>  // Corresponding header

// Include the backend interface definition which declares ResamplePlanImpl
#include <memory>  // For std::unique_ptr
#include <span>
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move

#include "backend.hpp"  // Defines backend::ResamplePlanImpl

// Include core types for aliases F32, F64 used in instantiations
#include <OmniDSP/core_types.hpp>

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
  template <typename T>  // T is REAL type
  ResamplePlan<T>::ResamplePlan(
      std::unique_ptr<backend::ResamplePlanImpl<T>> pimpl)
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "ResamplePlan cannot be created with a null implementation pointer.");
    }
  }

  /**
   * @brief Destructor.
   */
  template <typename T>
  ResamplePlan<T>::~ResamplePlan() = default;

  /**
   * @brief Move constructor.
   */
  template <typename T>
  ResamplePlan<T>::ResamplePlan(ResamplePlan&& other) noexcept = default;

  /**
   * @brief Move assignment operator.
   */
  template <typename T>
  ResamplePlan<T>& ResamplePlan<T>::operator=(ResamplePlan&& other) noexcept
      = default;

  /**
   * @brief Executes the pre-planned resampling operation by calling the backend
   * implementation. Modifies the internal state of the plan.
   * @param input A span representing the input signal buffer.
   * @param output A span representing the output buffer for the resampled
   * signal.
   * @return Status::Success on success, or an error code on failure.
   */
  template <typename T>  // T is REAL type
  [[nodiscard]] Status ResamplePlan<T>::execute(
      std::span<const T> input, std::span<T> output) /* *** REMOVED const *** */
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    // Add size checks for robustness? get_output_length provides an estimate.
    // IPP's execute function handles output size internally. Let's rely on
    // that.
    // size_t expected_output_size = get_output_length(input.size());
    // if (output.size() < expected_output_size && !input.empty()) {
    //     // Maybe return SizeMismatch or just let the backend handle it?
    // }

    // Forward the call to the actual backend implementation.
    // The backend implementation's execute IS non-const.
    return pimpl_->execute(input, output);
  }

  /**
   * @brief Resets the internal state of the resampler by calling the backend
   * implementation.
   * @return Status::Success on success, or an error code if resetting fails.
   */
  template <typename T>  // T is REAL type
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
   */
  template <typename T>  // T is REAL type
  double ResamplePlan<T>::get_input_rate() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ResamplePlan instance in get_input_rate.");
    }
    return pimpl_->get_input_rate();
  }

  /**
   * @brief Gets the output sample rate by calling the backend implementation.
   */
  template <typename T>  // T is REAL type
  double ResamplePlan<T>::get_output_rate() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ResamplePlan instance in get_output_rate.");
    }
    return pimpl_->get_output_rate();
  }

  /**
   * @brief Estimates the required output length by calling the backend
   * implementation.
   */
  template <typename T>  // T is REAL type
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

  template class ResamplePlan<F32>;  // float
  template class ResamplePlan<F64>;  // double

}  // namespace OmniDSP
