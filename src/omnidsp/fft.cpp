/**
 * @file fft.cpp
 * @brief Implements the FFTPlan and RFFTPlan class methods, forwarding calls to
 * backend implementations.
 */

#include "OmniDSP/fft.h"  // Corresponding header

// Include Pimpl interface definitions (defined below or in a separate
// backend.h) #include "backend/backend.h" // May contain base class definitions
// if separated

// Include concrete backend implementation headers (Placeholders - needed for
// unique_ptr deletion) Although not strictly needed for this file's compilation
// if using default destructor, good practice to ensure Impl destructors are
// known. Alternatively, define Plan destructors explicitly here without needing
// these includes. #include "backend/stub/stub_fft.h" #ifdef USE_ACCELERATE
// #include "backend/accelerate/accelerate_fft.h"
// #endif
// #ifdef USE_ONEMKL
// #include "backend/onemkl/onemkl_fft.h"
// #endif

#include <memory>  // For std::unique_ptr
#include <span>
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move
#include <vector>

namespace OmniDSP {
namespace backend {

/**
 * @brief Abstract base class defining the interface for complex FFT Plan
 * implementations (Pimpl).
 * @tparam T The floating-point type (e.g., float, double).
 */
template <typename T>
class FFTPlanImpl {
 public:
  virtual ~FFTPlanImpl() = default;
  virtual Status fft(std::span<const ComplexT<T>> input,
                     std::span<ComplexT<T>> output) const = 0;
  virtual Status ifft(std::span<const ComplexT<T>> input,
                      std::span<ComplexT<T>> output) const = 0;
  virtual size_t get_length() const = 0;
};

/**
 * @brief Abstract base class defining the interface for real FFT Plan
 * implementations (Pimpl).
 * @tparam T The floating-point type (e.g., float, double).
 */
template <typename T>
class RFFTPlanImpl {
 public:
  virtual ~RFFTPlanImpl() = default;
  virtual Status rfft(std::span<const RealT<T>> input,
                      std::span<ComplexT<T>> output) const = 0;
  virtual Status irfft(std::span<const ComplexT<T>> input,
                       std::span<RealT<T>> output) const = 0;
  virtual size_t get_length() const = 0;
};

// Define explicit instantiations for the Impl interfaces if needed,
// though usually not required for abstract classes.

}  // namespace backend

//--------------------------------------------------------------------------
// FFTPlan Method Definitions
//--------------------------------------------------------------------------

template <typename T>
FFTPlan<T>::FFTPlan(std::unique_ptr<backend::FFTPlanImpl<T>> pimpl)
    : pimpl_(std::move(pimpl)) {
  // Constructor body can be empty, initialization is done via member
  // initializer list
  if (!pimpl_) {
    // This should ideally be caught by the factory creating the Impl
    throw std::runtime_error("FFTPlan created with null implementation.");
  }
}

// Destructor: Needs definition in the .cpp file where FFTPlanImpl is complete.
// Default implementation is usually sufficient if FFTPlanImpl has a virtual
// destructor.
template <typename T>
FFTPlan<T>::~FFTPlan() = default;

// Move Constructor: Default implementation is sufficient for unique_ptr.
template <typename T>
FFTPlan<T>::FFTPlan(FFTPlan&& other) noexcept = default;

// Move Assignment Operator: Default implementation is sufficient for
// unique_ptr.
template <typename T>
FFTPlan<T>& FFTPlan<T>::operator=(FFTPlan&& other) noexcept = default;

template <typename T>
[[nodiscard]] Status FFTPlan<T>::fft(std::span<const ComplexT<T>> input,
                                     std::span<ComplexT<T>> output) const {
  if (!pimpl_) {
    // Should not happen if constructor validates, but defensive check
    return Status::InvalidOperation;  // Or another appropriate error
  }
  // TODO: Add size checks: input.size() == get_length(), output.size() ==
  // get_length() ?
  return pimpl_->fft(input, output);
}

template <typename T>
[[nodiscard]] Status FFTPlan<T>::ifft(std::span<const ComplexT<T>> input,
                                      std::span<ComplexT<T>> output) const {
  if (!pimpl_) {
    return Status::InvalidOperation;
  }
  // TODO: Add size checks: input.size() == get_length(), output.size() ==
  // get_length() ?
  return pimpl_->ifft(input, output);
}

template <typename T>
size_t FFTPlan<T>::get_length() const {
  if (!pimpl_) {
    return 0;  // Or throw? A plan should always have an impl if constructed.
  }
  return pimpl_->get_length();
}

//--------------------------------------------------------------------------
// RFFTPlan Method Definitions
//--------------------------------------------------------------------------

template <typename T>
RFFTPlan<T>::RFFTPlan(std::unique_ptr<backend::RFFTPlanImpl<T>> pimpl)
    : pimpl_(std::move(pimpl)) {
  if (!pimpl_) {
    throw std::runtime_error("RFFTPlan created with null implementation.");
  }
}

// Destructor: Needs definition in the .cpp file where RFFTPlanImpl is complete.
template <typename T>
RFFTPlan<T>::~RFFTPlan() = default;

// Move Constructor: Default implementation is sufficient for unique_ptr.
template <typename T>
RFFTPlan<T>::RFFTPlan(RFFTPlan&& other) noexcept = default;

// Move Assignment Operator: Default implementation is sufficient for
// unique_ptr.
template <typename T>
RFFTPlan<T>& RFFTPlan<T>::operator=(RFFTPlan&& other) noexcept = default;

template <typename T>
[[nodiscard]] Status RFFTPlan<T>::rfft(std::span<const RealT<T>> input,
                                       std::span<ComplexT<T>> output) const {
  if (!pimpl_) {
    return Status::InvalidOperation;
  }
  // TODO: Add size checks: input.size() == get_length(), output.size() ==
  // get_length()/2 + 1 ?
  return pimpl_->rfft(input, output);
}

template <typename T>
[[nodiscard]] Status RFFTPlan<T>::irfft(std::span<const ComplexT<T>> input,
                                        std::span<RealT<T>> output) const {
  if (!pimpl_) {
    return Status::InvalidOperation;
  }
  // TODO: Add size checks: output.size() == get_length(), input.size() ==
  // get_length()/2 + 1 ?
  return pimpl_->irfft(input, output);
}

template <typename T>
size_t RFFTPlan<T>::get_length() const {
  if (!pimpl_) {
    return 0;
  }
  return pimpl_->get_length();
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
// Instantiate templates for common types (float, double) to ensure code
// generation.

// Define complex types for brevity
using float_c = OmniDSP::ComplexT<float>;
using double_c = OmniDSP::ComplexT<double>;

// FFTPlan Instantiations
template class OmniDSP::FFTPlan<float_c>;
template class OmniDSP::FFTPlan<double_c>;

// RFFTPlan Instantiations
template class OmniDSP::RFFTPlan<float>;
template class OmniDSP::RFFTPlan<double>;

// Note: If method definitions were not defaulted (e.g., custom move logic),
// they would also need explicit instantiations if defined in the .cpp file.
// Since we used =default for destructor/move ops, instantiating the class is
// sufficient. The execute/getter methods are implicitly instantiated with the
// class.

}  // namespace OmniDSP
