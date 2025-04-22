/**
 * @file resample.cpp
 * @brief Implements the ResamplePlan class methods, forwarding calls to backend
 * implementations.
 */

#include "OmniDSP/resample.h"  // Corresponding header

// Include Pimpl interface definition (defined below or in a separate backend.h)
// #include "backend/backend.h" // May contain base class definitions if
// separated

// Include concrete backend implementation headers (Placeholders - needed for
// unique_ptr deletion) #include "backend/stub/stub_resample.h" #ifdef
// USE_ACCELERATE #include "backend/accelerate/accelerate_resample.h" #endif
// #ifdef USE_ONEMKL
// #include "backend/onemkl/onemkl_resample.h"
// #endif

#include <cmath>   // For std::ceil
#include <memory>  // For std::unique_ptr
#include <span>
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move
#include <vector>

namespace OmniDSP {
namespace backend {

/**
 * @brief Abstract base class defining the interface for Resample Plan
 * implementations (Pimpl).
 * @tparam T The floating-point type (e.g., float, double).
 */
template <typename T>
class ResamplePlanImpl {
 public:
  virtual ~ResamplePlanImpl() = default;
  virtual Status execute(std::span<const T> input,
                         std::span<T> output) const = 0;
  virtual double get_input_rate() const = 0;
  virtual double get_output_rate() const = 0;
  virtual size_t get_output_length(size_t input_length) const = 0;
};

// Define explicit instantiations for the Impl interfaces if needed,
// though usually not required for abstract classes.

}  // namespace backend

//--------------------------------------------------------------------------
// ResamplePlan Method Definitions
//--------------------------------------------------------------------------

template <typename T>
ResamplePlan<T>::ResamplePlan(
    std::unique_ptr<backend::ResamplePlanImpl<T>> pimpl)
    : pimpl_(std::move(pimpl)) {
  // Constructor body can be empty, initialization is done via member
  // initializer list
  if (!pimpl_) {
    // This should ideally be caught by the factory creating the Impl
    throw std::runtime_error("ResamplePlan created with null implementation.");
  }
}

// Destructor: Needs definition in the .cpp file where ResamplePlanImpl is
// complete. Default implementation is usually sufficient if ResamplePlanImpl
// has a virtual destructor.
template <typename T>
ResamplePlan<T>::~ResamplePlan() = default;

// Move Constructor: Default implementation is sufficient for unique_ptr.
template <typename T>
ResamplePlan<T>::ResamplePlan(ResamplePlan&& other) noexcept = default;

// Move Assignment Operator: Default implementation is sufficient for
// unique_ptr.
template <typename T>
ResamplePlan<T>& ResamplePlan<T>::operator=(ResamplePlan&& other) noexcept =
    default;

template <typename T>
[[nodiscard]] Status ResamplePlan<T>::execute(std::span<const T> input,
                                              std::span<T> output) const {
  if (!pimpl_) {
    // Should not happen if constructor validates, but defensive check
    return Status::InvalidOperation;  // Or another appropriate error
  }
  // TODO: Add size checks for output based on get_output_length?
  // size_t expected_len = get_output_length(input.size());
  // if (output.size() < expected_len) return Status::SizeMismatch; // Or
  // InvalidArgument?
  return pimpl_->execute(input, output);
}

template <typename T>
double ResamplePlan<T>::get_input_rate() const {
  if (!pimpl_) {
    // Throw? Return 0? A plan should always have an impl if constructed.
    throw std::runtime_error(
        "Invalid ResamplePlan instance in get_input_rate.");
  }
  return pimpl_->get_input_rate();
}

template <typename T>
double ResamplePlan<T>::get_output_rate() const {
  if (!pimpl_) {
    throw std::runtime_error(
        "Invalid ResamplePlan instance in get_output_rate.");
  }
  return pimpl_->get_output_rate();
}

template <typename T>
size_t ResamplePlan<T>::get_output_length(size_t input_length) const {
  if (!pimpl_) {
    throw std::runtime_error(
        "Invalid ResamplePlan instance in get_output_length.");
  }
  // Forward the calculation to the implementation, which knows about filter
  // delays etc.
  return pimpl_->get_output_length(input_length);
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
// Instantiate templates for common types (float, double) to ensure code
// generation.

template class OmniDSP::ResamplePlan<float>;
template class OmniDSP::ResamplePlan<double>;

// Remove implementation of old standalone filter_and_downsample function here
// if it existed.

}  // namespace OmniDSP
