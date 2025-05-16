/**
 * @file fft.cpp
 * @brief Implements the public Plan::FFT and Plan::RFFT class methods,
 * forwarding calls to the backend implementation pointer (pimpl).
 */

#include <OmniDSP/plan/fft.hpp>  // Corresponding header for public Plan classes

// Include the backend interface definition which declares the Impl interfaces
#include <memory>  // For std::unique_ptr
#include <span>
#include <stdexcept>  // For std::runtime_error
#include <utility>    // For std::move
#include <vector>

#include "backend.hpp"

// Include core types for aliases like C32, F32 etc. used in instantiations
#include <OmniDSP/core_types.hpp>

namespace OmniDSP::Plan {

  //--------------------------------------------------------------------------
  // Plan::FFT Method Definitions
  //--------------------------------------------------------------------------

  // Constructor - Takes the backend implementation pointer
  template <typename T>  // T is Complex type (C32, C64)
  FFT<T>::FFT(std::unique_ptr<Abstract::FFTPlanImpl<T>> pimpl)
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      // This should ideally be caught by the factory creating the Impl
      throw std::runtime_error("Plan::FFT created with null implementation.");
    }
  }

  // Destructor: Needs definition in the .cpp file where FFTPlanImpl is
  // complete. Default implementation is usually sufficient if FFTPlanImpl has a
  // virtual destructor.
  template <typename T>
  FFT<T>::~FFT() = default;

  // Move Constructor: Default implementation is sufficient for unique_ptr.
  template <typename T>
  FFT<T>::FFT(FFT&& other) noexcept = default;

  // Move Assignment Operator: Default implementation is sufficient for
  // unique_ptr.
  template <typename T>
  FFT<T>& FFT<T>::operator=(FFT&& other) noexcept = default;

  // fft Method - Uses template parameter T
  template <typename T>  // T is Complex type (C32, C64)
  [[nodiscard]] OmniStatus FFT<T>::fft(
      std::span<const T> input, std::span<T> output) const
  {
    if (!pimpl_) {
      return OmniStatus::InvalidOperation;
    }
    // Add size checks against get_length() for robustness
    if (input.size() != get_length() || output.size() != get_length()) {
      return OmniStatus::SizeMismatch;
    }
    return pimpl_->fft(input, output);
  }

  // ifft Method - Uses template parameter T
  template <typename T>  // T is Complex type (C32, C64)
  [[nodiscard]] OmniStatus FFT<T>::ifft(
      std::span<const T> input, std::span<T> output) const
  {
    if (!pimpl_) {
      return OmniStatus::InvalidOperation;
    }
    // Add size checks against get_length() for robustness
    if (input.size() != get_length() || output.size() != get_length()) {
      return OmniStatus::SizeMismatch;
    }
    // Note: Scaling (1/N) is typically handled by the user or backend
    // implementation
    return pimpl_->ifft(input, output);
  }

  // get_length Method
  template <typename T>  // T is Complex type (C32, C64)
  size_t FFT<T>::get_length() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Attempted to get length from an invalid Plan::FFT.");
    }
    return pimpl_->get_length();
  }

  //--------------------------------------------------------------------------
  // Plan::RFFT Method Definitions
  //--------------------------------------------------------------------------

  // Constructor - Takes the backend implementation pointer
  template <typename T>  // T is REAL type (F32, F64)
  RFFT<T>::RFFT(std::unique_ptr<Abstract::RFFTPlanImpl<T>> pimpl)
      : pimpl_(std::move(pimpl))
  {
    if (!pimpl_) {
      throw std::runtime_error("Plan::RFFT created with null implementation.");
    }
  }

  // Destructor: Needs definition in the .cpp file where RFFTPlanImpl is
  // complete.
  template <typename T>
  RFFT<T>::~RFFT() = default;

  // Move Constructor: Default implementation is sufficient for unique_ptr.
  template <typename T>
  RFFT<T>::RFFT(RFFT&& other) noexcept = default;

  // Move Assignment Operator: Default implementation is sufficient for
  // unique_ptr.
  template <typename T>
  RFFT<T>& RFFT<T>::operator=(RFFT&& other) noexcept = default;

  // rfft Method - Uses T (Real) and ComplexT<T>
  template <typename T>  // T is REAL type (F32, F64)
  [[nodiscard]] OmniStatus RFFT<T>::rfft(
      std::span<const T> input,
      std::span<Complex> output) const  // Use Complex alias
  {
    if (!pimpl_) {
      return OmniStatus::InvalidOperation;
    }
    // Add size checks against get_length() for robustness
    size_t N = get_length();
    if (N == 0 && (input.empty() && output.empty()))
      return OmniStatus::Success;  // Allow empty case
    if (N == 0 || input.size() != N || output.size() != (N / 2 + 1)) {
      return OmniStatus::SizeMismatch;
    }
    return pimpl_->rfft(input, output);
  }

  // irfft Method - Uses ComplexT<T> and T (Real)
  template <typename T>  // T is REAL type (F32, F64)
  [[nodiscard]] OmniStatus RFFT<T>::irfft(
      std::span<const Complex> input,
      std::span<T> output) const  // Use Complex alias
  {
    if (!pimpl_) {
      return OmniStatus::InvalidOperation;
    }
    // Add size checks against get_length() for robustness
    size_t N = get_length();
    if (N == 0 && (input.empty() && output.empty()))
      return OmniStatus::Success;  // Allow empty case
    if (N == 0 || output.size() != N || input.size() != (N / 2 + 1)) {
      return OmniStatus::SizeMismatch;
    }
    // Note: Scaling (1/N) is typically handled by the user or backend
    // implementation
    return pimpl_->irfft(input, output);
  }

  // get_length Method
  template <typename T>  // T is REAL type (F32, F64)
  size_t RFFT<T>::get_length() const
  {
    if (!pimpl_) {
      throw std::runtime_error(
          "Attempted to get length from an invalid Plan::RFFT.");
    }
    return pimpl_->get_length();
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  // Instantiate templates for common types (float, double) to ensure code
  // generation for the public Plan classes.

  // Plan::FFT Instantiations (for Complex types C32, C64)
  template class FFT<C32>;
  template class FFT<C64>;

  // Plan::RFFT Instantiations (for Real types F32, F64)
  template class RFFT<F32>;
  template class RFFT<F64>;

}  // namespace OmniDSP::Plan
