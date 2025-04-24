/**
 * @file convolution.cpp
 * @brief Implements the ConvolutionPlan and CorrelationPlan class methods,
 * forwarding calls to backend implementations.
 */

#include "OmniDSP/convolution.h"  // Corresponding header

// Include Pimpl interface definition (defined below or in a separate backend.h)
// #include "backend/backend.h" // May contain base class definitions if
// separated

// Include concrete backend implementation headers (Placeholders - needed for
// unique_ptr deletion) #include "backend/stub/stub_convolution.h" #ifdef
// USE_ACCELERATE #include "backend/accelerate/accelerate_convolution.h" #endif
// #ifdef USE_ONEMKL
// #include "backend/onemkl/onemkl_convolution.h"
// #endif

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
      virtual ConvolutionMode get_mode() const = 0;
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
      virtual ConvolutionMode get_mode() const = 0;
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
  ConvolutionMode ConvolutionPlan<T>::get_mode() const
  {
    if (!pimpl_) {
      throw std::runtime_error("Invalid ConvolutionPlan instance in get_mode.");
    }
    return pimpl_->get_mode();
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
  ConvolutionMode CorrelationPlan<T>::get_mode() const
  {
    if (!pimpl_) {
      throw std::runtime_error("Invalid CorrelationPlan instance in get_mode.");
    }
    return pimpl_->get_mode();
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

  // Define complex types for brevity
  using float_c = OmniDSP::ComplexT<float>;
  using double_c = OmniDSP::ComplexT<double>;

  // ConvolutionPlan Instantiations
  template class OmniDSP::ConvolutionPlan<float>;
  template class OmniDSP::ConvolutionPlan<double>;
  template class OmniDSP::ConvolutionPlan<float_c>;
  template class OmniDSP::ConvolutionPlan<double_c>;

  // CorrelationPlan Instantiations
  template class OmniDSP::CorrelationPlan<float>;
  template class OmniDSP::CorrelationPlan<double>;
  template class OmniDSP::CorrelationPlan<float_c>;
  template class OmniDSP::CorrelationPlan<double_c>;

  // Remove implementation of old standalone convolve1d/correlate1d functions
  // here if they existed.

}  // namespace OmniDSP
