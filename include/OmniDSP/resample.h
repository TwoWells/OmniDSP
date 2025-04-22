/**
 * @file resample.h
 * @brief Defines the interface for Resample Plan objects.
 * @details Plan objects are created via OmniDSP::create_resample_plan and
 * provide optimized execution for resampling signals between specified input
 * and output rates.
 */

#ifndef OMNIDSP_RESAMPLE_H
#define OMNIDSP_RESAMPLE_H

#include <cmath>    // For std::ceil (used in placeholder getter)
#include <cstddef>  // For size_t
#include <memory>   // For std::unique_ptr (Pimpl)
#include <span>     // For input/output views (requires C++20)
#include <vector>   // For std::vector (potentially used internally)

#include "core_types.h"  // Core types like RealT, Status

namespace OmniDSP {

// Forward declare the main OmniDSP class for friend declaration
class OmniDSP;

// Forward declarations for implementation classes (Pimpl idiom)
namespace backend {
template <typename T>
class ResamplePlanImpl;
}  // namespace backend

/**
 * @brief Plan object for executing signal resampling operations.
 * @details This class encapsulates the pre-computed data (e.g., polyphase
 * filter coefficients) and backend-specific context required for efficient
 * resampling. Instances are created via OmniDSP::create_resample_plan. This
 * class is non-copyable but movable.
 * @tparam T The underlying floating-point type (e.g., float, double). Typically
 * used with RealT<T>.
 */
template <typename T>
class ResamplePlan {
  // Friend declaration allows OmniDSP factory methods to call the private
  // constructor
  friend class OmniDSP;

 public:
  /**
   * @brief Destructor. Cleans up implementation resources.
   * @details Defined in the source file to handle Pimpl destruction.
   */
  ~ResamplePlan();

  /**
   * @brief Move constructor.
   * @details Defined in the source file.
   */
  ResamplePlan(ResamplePlan&& other) noexcept;

  /**
   * @brief Move assignment operator.
   * @details Defined in the source file.
   */
  ResamplePlan& operator=(ResamplePlan&& other) noexcept;

  /** @brief Deleted copy constructor. ResamplePlan instances are non-copyable.
   */
  ResamplePlan(const ResamplePlan&) = delete;
  /** @brief Deleted copy assignment operator. ResamplePlan instances are
   * non-copyable. */
  ResamplePlan& operator=(const ResamplePlan&) = delete;

  /**
   * @brief Executes the pre-planned resampling operation on an input signal.
   * @param input A span representing the input signal buffer.
   * @param output A span representing the output buffer for the resampled
   * signal. The required size can be estimated via get_output_length().
   * @return Status::Success on success, or an error code on failure.
   * @note The output span must be pre-allocated with sufficient size before
   * calling execute. The exact output size might depend on internal filter
   * delays and boundary handling.
   */
  [[nodiscard]] Status execute(std::span<const T> input,
                               std::span<T> output) const;

  /**
   * @brief Gets the input sample rate configured for this plan.
   * @return The input sample rate in Hz.
   */
  double get_input_rate() const;

  /**
   * @brief Gets the output sample rate configured for this plan.
   * @return The output sample rate in Hz.
   */
  double get_output_rate() const;

  /**
   * @brief Estimates the required output length for a given input length based
   * on the resampling ratio.
   * @details The actual output length might vary slightly due to filter delays
   * and implementation details. This provides a suitable size for allocating
   * the output buffer.
   * @param input_length The length of the input signal.
   * @return An estimated length for the output signal buffer.
   */
  size_t get_output_length(size_t input_length) const;

 private:
  /**
   * @brief Private constructor, called by OmniDSP factory methods.
   * @param pimpl A unique_ptr to the backend-specific implementation object.
   */
  explicit ResamplePlan(std::unique_ptr<backend::ResamplePlanImpl<T>> pimpl);

  /**
   * @brief Pointer to the implementation object (Pimpl idiom).
   */
  std::unique_ptr<backend::ResamplePlanImpl<T>> pimpl_;
};

// --- Template Implementations (Definitions) ---
// Definitions for template class methods that depend on the Impl class
// (constructor, destructor, move ops, execute, getters) MUST be provided
// in a .cpp file where the ResamplePlanImpl<T> is fully defined.

// Example placeholders for header compilation (real definitions MUST be in
// .cpp):
template <typename T>
double ResamplePlan<T>::get_input_rate() const {
  // return pimpl_ ? pimpl_->get_input_rate() : 0.0; // Requires Impl definition
  return 0.0;
}
template <typename T>
double ResamplePlan<T>::get_output_rate() const {
  // return pimpl_ ? pimpl_->get_output_rate() : 0.0; // Requires Impl
  // definition
  return 0.0;
}
template <typename T>
size_t ResamplePlan<T>::get_output_length(size_t input_length) const {
  // double ratio = get_output_rate() / get_input_rate(); // Needs real getters
  // if (ratio == 0.0) return 0; // Avoid division by zero if rates are invalid
  // // Estimate output length, potentially add padding for filter delay
  // return static_cast<size_t>(std::ceil(static_cast<double>(input_length) *
  // ratio)); // Requires Impl definition Placeholder:
  double in_rate = get_input_rate();
  double out_rate = get_output_rate();
  if (in_rate <= 0.0) return 0;  // Avoid division by zero
  return static_cast<size_t>(
      std::ceil(static_cast<double>(input_length) * (out_rate / in_rate)));
}

}  // namespace OmniDSP

#endif  // OMNIDSP_RESAMPLE_H
