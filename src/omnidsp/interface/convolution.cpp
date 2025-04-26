/**
 * @file convolution.cpp
 * @brief Implements the ConvolutionPlan and CorrelationPlan class methods,
 * forwarding calls to backend implementations.
 */

#include "OmniDSP/convolution.hpp"  // Corresponding header

// Include Pimpl interface definition
// #include "backend/backend.hpp"

#include <memory>  // For std::unique_ptr
#include <numeric>  // For std::max, std::min (potentially used in get_output_length)
#include <span>
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move
#include <vector>

namespace OmniDSP {
  namespace backend {

    /**
     * @brief Abstract base class defining the interface for Convolution Plan
     * implementations (Pimpl).
     * @tparam T The data type (e.g., float, std::complex<float>).
     */
    template <typename T>
    class ConvolutionPlanImpl {
     public:
      virtual ~ConvolutionPlanImpl() = default;
      virtual Status execute(
          std::span<const T> input, std::span<T> output) const
          = 0;
      virtual size_t get_kernel_length() const = 0;
      virtual ConvolutionType get_type() const = 0;
      virtual size_t get_output_length(size_t input_length) const = 0;
    };

    /**
     * @brief Abstract base class defining the interface for Correlation Plan
     * implementations (Pimpl).
     * @tparam T The data type (e.g., float, std::complex<float>).
     */
    template <typename T>
    class CorrelationPlanImpl {
     public:
      virtual ~CorrelationPlanImpl() = default;
      virtual Status execute(
          std::span<const T> input, std::span<T> output) const
          = 0;
      virtual size_t get_template_length() const
          = 0;  // Renamed from kernel for clarity
      virtual ConvolutionType get_type() const = 0;
      virtual size_t get_output_length(size_t input_length) const = 0;
    };

  }  // namespace backend

  //--------------------------------------------------------------------------
  // ConvolutionPlan Method Definitions
  //--------------------------------------------------------------------------

  template <typename T>
  ConvolutionPlan<T>::ConvolutionPlan(
      std::unique_ptr<backend::ConvolutionPlanImpl<T>> pimpl)
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "ConvolutionPlan created with null implementation.");
    }
  }

  template <typename T>
  ConvolutionPlan<T>::~ConvolutionPlan() = default;

  template <typename T>
  ConvolutionPlan<T>::ConvolutionPlan(ConvolutionPlan&& other) noexcept
      = default;

  template <typename T>
  ConvolutionPlan<T>& ConvolutionPlan<T>::operator=(
      ConvolutionPlan&& other) noexcept
      = default;

  template <typename T>
  [[nodiscard]] Status ConvolutionPlan<T>::execute(
      std::span<const T> input, std::span<T> output) const
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    // TODO: Add size checks for output based on get_output_length?
    // size_t expected_len = get_output_length(input.size());
    // if (output.size() < expected_len) return Status::SizeMismatch;
    return pimpl_->execute(input, output);
  }

  template <typename T>
  size_t ConvolutionPlan<T>::get_kernel_length() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ConvolutionPlan instance in get_kernel_length.");
    }
    return pimpl_->get_kernel_length();
  }

  template <typename T>
  ConvolutionType ConvolutionPlan<T>::get_type() const
  {
    if (!pimpl_) {
      throw std::runtime_error("Invalid ConvolutionPlan instance in get_type.");
    }
    return pimpl_->get_type();
  }

  template <typename T>
  size_t ConvolutionPlan<T>::get_output_length(size_t input_length) const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid ConvolutionPlan instance in get_output_length.");
    }
    // Forward calculation to implementation, which knows the kernel length and
    // mode.
    return pimpl_->get_output_length(input_length);
  }

  //--------------------------------------------------------------------------
  // CorrelationPlan Method Definitions
  //--------------------------------------------------------------------------

  template <typename T>
  CorrelationPlan<T>::CorrelationPlan(
      std::unique_ptr<backend::CorrelationPlanImpl<T>> pimpl)
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "CorrelationPlan created with null implementation.");
    }
  }

  template <typename T>
  CorrelationPlan<T>::~CorrelationPlan() = default;

  template <typename T>
  CorrelationPlan<T>::CorrelationPlan(CorrelationPlan&& other) noexcept
      = default;

  template <typename T>
  CorrelationPlan<T>& CorrelationPlan<T>::operator=(
      CorrelationPlan&& other) noexcept
      = default;

  template <typename T>
  [[nodiscard]] Status CorrelationPlan<T>::execute(
      std::span<const T> input, std::span<T> output) const
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    // TODO: Add size checks for output based on get_output_length?
    // size_t expected_len = get_output_length(input.size());
    // if (output.size() < expected_len) return Status::SizeMismatch;
    return pimpl_->execute(input, output);
  }

  template <typename T>
  size_t CorrelationPlan<T>::get_template_length() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid CorrelationPlan instance in get_template_length.");
    }
    return pimpl_->get_template_length();
  }

  template <typename T>
  ConvolutionType CorrelationPlan<T>::get_type() const
  {
    if (!pimpl_) {
      throw std::runtime_error("Invalid CorrelationPlan instance in get_type.");
    }
    return pimpl_->get_type();
  }

  template <typename T>
  size_t CorrelationPlan<T>::get_output_length(size_t input_length) const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Invalid CorrelationPlan instance in get_output_length.");
    }
    // Forward calculation to implementation.
    return pimpl_->get_output_length(input_length);
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  // Instantiate templates for common types (float, double, complex) to ensure
  // code generation.

  // ConvolutionPlan Instantiations
  template class ConvolutionPlan<F32>;
  template class ConvolutionPlan<F64>;
  template class ConvolutionPlan<C32>;
  template class ConvolutionPlan<C64>;

  // CorrelationPlan Instantiations
  template class CorrelationPlan<F32>;
  template class CorrelationPlan<F64>;
  template class CorrelationPlan<C32>;
  template class CorrelationPlan<C64>;

}  // namespace OmniDSP
