/**
 * @file convolution.h
 * @brief Defines interfaces for Convolution and Correlation Plan objects.
 * @details Plan objects are created via OmniDSP factory methods (e.g.,
 * OmniDSP::create_convolution_plan) and provide optimized execution for
 * repeated convolution or correlation operations, typically with a fixed
 * kernel/template.
 */

#ifndef OMNIDSP_CONVOLUTION_H
#define OMNIDSP_CONVOLUTION_H

#include <complex>  // For std::complex
#include <cstddef>  // For size_t
#include <memory>   // For std::unique_ptr (Pimpl)
#include <span>     // For input/output views (requires C++20)
#include <vector>   // For std::vector (potentially used internally)

#include "core_types.h"  // Core types like RealT, ComplexT, Status, ConvolutionMode

namespace OmniDSP {

// Forward declare the main OmniDSP class for friend declaration
class OmniDSP;

// Forward declarations for implementation classes (Pimpl idiom)
namespace backend {
template <typename T>
class ConvolutionPlanImpl;
template <typename T>
class CorrelationPlanImpl;
}  // namespace backend

/**
 * @brief Plan object for executing 1D Convolutions.
 * @details This class encapsulates the pre-computed data (e.g., FFT of the
 * kernel) and backend-specific context required for efficient convolution
 * execution, especially when convolving multiple input signals with the same
 * kernel. Instances are created via an OmniDSP factory method (to be added).
 * This class is non-copyable but movable.
 * @tparam T The data type of the signals (e.g., RealT<float>,
 * ComplexT<double>).
 */
template <typename T>
class ConvolutionPlan {
  // Friend declaration allows OmniDSP factory methods to call the private
  // constructor
  friend class OmniDSP;

 public:
  /**
   * @brief Destructor. Cleans up implementation resources.
   * @details Defined in the source file to handle Pimpl destruction.
   */
  ~ConvolutionPlan();

  /**
   * @brief Move constructor.
   * @details Defined in the source file.
   */
  ConvolutionPlan(ConvolutionPlan&& other) noexcept;

  /**
   * @brief Move assignment operator.
   * @details Defined in the source file.
   */
  ConvolutionPlan& operator=(ConvolutionPlan&& other) noexcept;

  /** @brief Deleted copy constructor. ConvolutionPlan instances are
   * non-copyable. */
  ConvolutionPlan(const ConvolutionPlan&) = delete;
  /** @brief Deleted copy assignment operator. ConvolutionPlan instances are
   * non-copyable. */
  ConvolutionPlan& operator=(const ConvolutionPlan&) = delete;

  /**
   * @brief Executes the pre-planned convolution on an input signal.
   * @param input A span representing the input signal buffer.
   * @param output A span representing the output buffer for the convolution
   * result. The required size depends on the mode and can be obtained via
   * get_output_length().
   * @return Status::Success on success, or an error code on failure.
   * @note The output span must be pre-allocated with sufficient size before
   * calling execute.
   */
  [[nodiscard]] Status execute(std::span<const T> input,
                               std::span<T> output) const;

  /**
   * @brief Gets the length of the kernel used in this convolution plan.
   * @return The length of the kernel.
   */
  size_t get_kernel_length() const;

  /**
   * @brief Gets the convolution mode (Full, Same, Valid) configured for this
   * plan.
   * @return The ConvolutionMode enum value.
   */
  ConvolutionMode get_mode() const;

  /**
   * @brief Calculates the required output length for a given input length based
   * on the plan's kernel length and mode.
   * @param input_length The length of the input signal.
   * @return The expected length of the output signal.
   */
  size_t get_output_length(size_t input_length) const;

 private:
  /**
   * @brief Private constructor, called by OmniDSP factory methods.
   * @param pimpl A unique_ptr to the backend-specific implementation object.
   */
  explicit ConvolutionPlan(
      std::unique_ptr<backend::ConvolutionPlanImpl<T>> pimpl);

  /**
   * @brief Pointer to the implementation object (Pimpl idiom).
   */
  std::unique_ptr<backend::ConvolutionPlanImpl<T>> pimpl_;
};

/**
 * @brief Plan object for executing 1D Cross-Correlations.
 * @details This class encapsulates the pre-computed data (e.g., FFT of the
 * template) and backend-specific context required for efficient correlation
 * execution, especially when correlating multiple input signals with the same
 * template. Instances are created via an OmniDSP factory method (to be added).
 * This class is non-copyable but movable.
 * @tparam T The data type of the signals (e.g., RealT<float>,
 * ComplexT<double>).
 */
template <typename T>
class CorrelationPlan {
  // Friend declaration allows OmniDSP factory methods to call the private
  // constructor
  friend class OmniDSP;

 public:
  /**
   * @brief Destructor. Cleans up implementation resources.
   * @details Defined in the source file to handle Pimpl destruction.
   */
  ~CorrelationPlan();

  /**
   * @brief Move constructor.
   * @details Defined in the source file.
   */
  CorrelationPlan(CorrelationPlan&& other) noexcept;

  /**
   * @brief Move assignment operator.
   * @details Defined in the source file.
   */
  CorrelationPlan& operator=(CorrelationPlan&& other) noexcept;

  /** @brief Deleted copy constructor. CorrelationPlan instances are
   * non-copyable. */
  CorrelationPlan(const CorrelationPlan&) = delete;
  /** @brief Deleted copy assignment operator. CorrelationPlan instances are
   * non-copyable. */
  CorrelationPlan& operator=(const CorrelationPlan&) = delete;

  /**
   * @brief Executes the pre-planned cross-correlation on an input signal.
   * @param input A span representing the input signal buffer.
   * @param output A span representing the output buffer for the correlation
   * result. The required size depends on the mode and can be obtained via
   * get_output_length().
   * @return Status::Success on success, or an error code on failure.
   * @note The output span must be pre-allocated with sufficient size before
   * calling execute.
   */
  [[nodiscard]] Status execute(std::span<const T> input,
                               std::span<T> output) const;

  /**
   * @brief Gets the length of the template (kernel) used in this correlation
   * plan.
   * @return The length of the template.
   */
  size_t get_template_length() const;

  /**
   * @brief Gets the correlation mode (Full, Same, Valid) configured for this
   * plan.
   * @return The ConvolutionMode enum value.
   */
  ConvolutionMode get_mode() const;

  /**
   * @brief Calculates the required output length for a given input length based
   * on the plan's template length and mode.
   * @param input_length The length of the input signal.
   * @return The expected length of the output signal.
   */
  size_t get_output_length(size_t input_length) const;

 private:
  /**
   * @brief Private constructor, called by OmniDSP factory methods.
   * @param pimpl A unique_ptr to the backend-specific implementation object.
   */
  explicit CorrelationPlan(
      std::unique_ptr<backend::CorrelationPlanImpl<T>> pimpl);

  /**
   * @brief Pointer to the implementation object (Pimpl idiom).
   */
  std::unique_ptr<backend::CorrelationPlanImpl<T>> pimpl_;
};

// --- Template Implementations (Definitions) ---
// Definitions for template class methods that depend on the Impl class
// (constructors, destructors, move ops, execute, getters) MUST be provided
// in a .cpp file where the Impl classes are fully defined.

// Example placeholders for header compilation (real definitions MUST be in
// .cpp):
template <typename T>
size_t ConvolutionPlan<T>::get_kernel_length() const {
  return 0;
}
template <typename T>
ConvolutionMode ConvolutionPlan<T>::get_mode() const {
  return ConvolutionMode::Full;
}  // Placeholder default
template <typename T>
size_t ConvolutionPlan<T>::get_output_length(size_t /*input_length*/) const {
  return 0;
}

template <typename T>
size_t CorrelationPlan<T>::get_template_length() const {
  return 0;
}
template <typename T>
ConvolutionMode CorrelationPlan<T>::get_mode() const {
  return ConvolutionMode::Full;
}  // Placeholder default
template <typename T>
size_t CorrelationPlan<T>::get_output_length(size_t /*input_length*/) const {
  return 0;
}

}  // namespace OmniDSP

#endif  // OMNIDSP_CONVOLUTION_H
