/**
 * @file fft.cpp
 * @brief Implements the public FFTPlan and RFFTPlan class methods, forwarding
 * calls to the backend implementation pointer (pimpl).
 */

#include <OmniDSP/fft.hpp>  // Corresponding header for public Plan classes

// Include the backend interface definition which declares the Impl interfaces
#include <memory>  // For std::unique_ptr
#include <span>
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move
#include <vector>

#include "backend.hpp"

// Include core types for aliases like C32, F32 etc. used in instantiations
#include <OmniDSP/core_types.hpp>

namespace OmniDSP {

  //--------------------------------------------------------------------------
  // FFTPlan Method Definitions
  //--------------------------------------------------------------------------

  // Constructor - Takes the backend implementation pointer
  template <typename T>  // T is Complex type (C32, C64)
  FFTPlan<T>::FFTPlan(std::unique_ptr<Abstract::FFTPlanImpl<T>> pimpl)
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      // This should ideally be caught by the factory creating the Impl
      throw std::runtime_error("FFTPlan created with null implementation.");
    }
  }

  // Destructor: Needs definition in the .cpp file where FFTPlanImpl is
  // complete. Default implementation is usually sufficient if FFTPlanImpl has a
  // virtual destructor.
  template <typename T>
  FFTPlan<T>::~FFTPlan() = default;

  // Move Constructor: Default implementation is sufficient for unique_ptr.
  template <typename T>
  FFTPlan<T>::FFTPlan(FFTPlan&& other) noexcept = default;

  // Move Assignment Operator: Default implementation is sufficient for
  // unique_ptr.
  template <typename T>
  FFTPlan<T>& FFTPlan<T>::operator=(FFTPlan&& other) noexcept = default;

  // fft Method - Uses template parameter T
  template <typename T>  // T is Complex type (C32, C64)
  [[nodiscard]] Status FFTPlan<T>::fft(
      std::span<const T> input, std::span<T> output) const
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    // Add size checks against get_length() for robustness
    if (input.size() != get_length() || output.size() != get_length()) {
      return Status::SizeMismatch;
    }
    return pimpl_->fft(input, output);
  }

  // ifft Method - Uses template parameter T
  template <typename T>  // T is Complex type (C32, C64)
  [[nodiscard]] Status FFTPlan<T>::ifft(
      std::span<const T> input, std::span<T> output) const
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    // Add size checks against get_length() for robustness
    if (input.size() != get_length() || output.size() != get_length()) {
      return Status::SizeMismatch;
    }
    // Note: Scaling (1/N) is typically handled by the user or backend
    // implementation
    return pimpl_->ifft(input, output);
  }

  // get_length Method
  template <typename T>  // T is Complex type (C32, C64)
  size_t FFTPlan<T>::get_length() const
  {
    if (!pimpl_) {
      // Or throw? Plan should be valid if constructed.
      throw std::runtime_error(
          "Attempted to get length from an invalid FFTPlan.");
    }
    return pimpl_->get_length();
  }

  //--------------------------------------------------------------------------
  // RFFTPlan Method Definitions
  //--------------------------------------------------------------------------

  // Constructor - Takes the backend implementation pointer
  template <typename T>  // T is REAL type (F32, F64)
  RFFTPlan<T>::RFFTPlan(std::unique_ptr<Abstract::RFFTPlanImpl<T>> pimpl)
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error("RFFTPlan created with null implementation.");
    }
  }

  // Destructor: Needs definition in the .cpp file where RFFTPlanImpl is
  // complete.
  template <typename T>
  RFFTPlan<T>::~RFFTPlan() = default;

  // Move Constructor: Default implementation is sufficient for unique_ptr.
  template <typename T>
  RFFTPlan<T>::RFFTPlan(RFFTPlan&& other) noexcept = default;

  // Move Assignment Operator: Default implementation is sufficient for
  // unique_ptr.
  template <typename T>
  RFFTPlan<T>& RFFTPlan<T>::operator=(RFFTPlan&& other) noexcept = default;

  // rfft Method - Uses T (Real) and ComplexT<T>
  template <typename T>  // T is REAL type (F32, F64)
  [[nodiscard]] Status RFFTPlan<T>::rfft(
      std::span<const T> input,
      std::span<Complex> output) const  // Use Complex alias
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    // Add size checks against get_length() for robustness
    size_t N = get_length();
    if (N == 0 && (input.empty() && output.empty()))
      return Status::Success;  // Allow empty case
    if (N == 0 || input.size() != N || output.size() != (N / 2 + 1)) {
      return Status::SizeMismatch;
    }
    return pimpl_->rfft(input, output);
  }

  // irfft Method - Uses ComplexT<T> and T (Real)
  template <typename T>  // T is REAL type (F32, F64)
  [[nodiscard]] Status RFFTPlan<T>::irfft(
      std::span<const Complex> input,
      std::span<T> output) const  // Use Complex alias
  {
    if (!pimpl_) {
      return Status::InvalidOperation;
    }
    // Add size checks against get_length() for robustness
    size_t N = get_length();
    if (N == 0 && (input.empty() && output.empty()))
      return Status::Success;  // Allow empty case
    if (N == 0 || output.size() != N || input.size() != (N / 2 + 1)) {
      return Status::SizeMismatch;
    }
    // Note: Scaling (1/N) is typically handled by the user or backend
    // implementation
    return pimpl_->irfft(input, output);
  }

  // get_length Method
  template <typename T>  // T is REAL type (F32, F64)
  size_t RFFTPlan<T>::get_length() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Attempted to get length from an invalid RFFTPlan.");
    }
    return pimpl_->get_length();
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  // Instantiate templates for common types (float, double) to ensure code
  // generation for the public Plan classes.

  // FFTPlan Instantiations (for Complex types C32, C64)
  template class FFTPlan<C32>;
  template class FFTPlan<C64>;

  // RFFTPlan Instantiations (for Real types F32, F64)
  template class RFFTPlan<F32>;
  template class RFFTPlan<F64>;

}  // namespace OmniDSP
