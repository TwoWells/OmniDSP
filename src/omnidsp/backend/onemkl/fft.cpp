/**
 * @file fft.cpp (OneMKL)
 * @brief Implements oneMKL backend FFTPlanImpl and RFFTPlanImpl classes using
 * DFTI.
 */

#include "fft.hpp"  // Include the corresponding header file

// Include MKL header for DFTI
#include <mkl.h>

#include <complex>
#include <iostream>  // For debug/error messages
#include <limits>    // For numeric_limits
#include <span>
#include <stdexcept>    // For std::runtime_error, std::invalid_argument
#include <string>       // For exception messages
#include <type_traits>  // For std::is_same_v
#include <vector>

// Include core types to ensure Utils::* traits are available if needed by
// helpers
#include <OmniDSP/core_types.hpp>

// *** ADDED: Include the new utility header ***
#include "utils.hpp"

namespace OmniDSP::OneMKL {

  // *** REMOVED mkl_status_to_omnidsp_status helper (now in Details.hpp) ***

  // *** REMOVED get_dfti_precision helper (now in Details.hpp) ***

  //--------------------------------------------------------------------------
  // FFTPlanImpl Method Definitions (Complex FFT)
  //--------------------------------------------------------------------------

  // Constructor Implementation
  template <typename T>  // T is Complex (C32, C64)
  FFTPlanImpl<T>::FFTPlanImpl(size_t length)
      : length_(length), mkl_status_(DFTI_NO_ERROR)
  {
    if (length == 0) {
      throw std::invalid_argument("FFT length cannot be zero.");
    }

    using RealType = typename T::value_type;
    // *** Call helper from Details.hpp ***
    DFTI_CONFIG_VALUE precision;
    try {
      precision = Details::get_dfti_precision<RealType>();
    }
    catch (const std::logic_error&
               e) {  // Catch potential error if static_assert fails somehow
      throw std::invalid_argument(
          "Unsupported precision type for oneMKL FFTPlan.");
    }

    mkl_status_ = DftiCreateDescriptor(
        &descriptor_handle_,
        precision,
        DFTI_COMPLEX,
        1,
        static_cast<MKL_LONG>(length_));
    if (mkl_status_ != DFTI_NO_ERROR) {
      std::cerr << "MKL Error during DftiCreateDescriptor: "
                << DftiErrorMessage(mkl_status_) << std::endl;
      descriptor_handle_ = nullptr;
      throw std::runtime_error(
          "Failed to create oneMKL DFTI descriptor (complex).");
    }

    mkl_status_
        = DftiSetValue(descriptor_handle_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (mkl_status_ != DFTI_NO_ERROR) {
      DftiFreeDescriptor(&descriptor_handle_);
      descriptor_handle_ = nullptr;
      throw std::runtime_error(
          "Failed to set DFTI_PLACEMENT for oneMKL FFTPlan.");
    }

    mkl_status_ = DftiCommitDescriptor(descriptor_handle_);
    if (mkl_status_ != DFTI_NO_ERROR) {
      DftiFreeDescriptor(&descriptor_handle_);
      descriptor_handle_ = nullptr;
      throw std::runtime_error(
          "Failed to commit oneMKL DFTI descriptor (complex).");
    }
  }

  // Destructor Implementation
  template <typename T>
  FFTPlanImpl<T>::~FFTPlanImpl()
  {
    if (descriptor_handle_) {
      MKL_LONG free_status = DftiFreeDescriptor(&descriptor_handle_);
      if (free_status != DFTI_NO_ERROR) {
        std::cerr
            << "Warning: Failed to free oneMKL DFTI descriptor (complex): "
            << DftiErrorMessage(free_status) << std::endl;
      }
    }
  }

  // fft Method Implementation
  template <typename T>
  OmniStatus FFTPlanImpl<T>::fft(
      std::span<const T> input, std::span<T> output) const
  {
    if (!descriptor_handle_) return OmniStatus::InvalidOperation;
    if (input.size() != length_ || output.size() != length_)
      return OmniStatus::SizeMismatch;
    if (length_ == 0) return OmniStatus::Success;

    MKL_LONG status = DftiComputeForward(
        descriptor_handle_,
        const_cast<void*>(static_cast<const void*>(input.data())),
        const_cast<void*>(static_cast<void*>(output.data())));
    // *** Use helper from Details.hpp ***
    return Details::mkl_status_to_omnidsp_status(status);
  }

  // ifft Method Implementation
  template <typename T>
  OmniStatus FFTPlanImpl<T>::ifft(
      std::span<const T> input, std::span<T> output) const
  {
    if (!descriptor_handle_) return OmniStatus::InvalidOperation;
    if (input.size() != length_ || output.size() != length_)
      return OmniStatus::SizeMismatch;
    if (length_ == 0) return OmniStatus::Success;

    MKL_LONG status = DftiComputeBackward(
        descriptor_handle_,
        const_cast<void*>(static_cast<const void*>(input.data())),
        const_cast<void*>(static_cast<void*>(output.data())));
    // *** Use helper from Details.hpp ***
    return Details::mkl_status_to_omnidsp_status(status);
  }

  // get_length Method Implementation
  template <typename T>
  size_t FFTPlanImpl<T>::get_length() const
  {
    return length_;
  }

  //--------------------------------------------------------------------------
  // RFFTPlanImpl Method Definitions (Real FFT)
  //--------------------------------------------------------------------------

  // Constructor Implementation
  template <typename T>  // T is REAL type
  RFFTPlanImpl<T>::RFFTPlanImpl(size_t length)
      : length_(length), mkl_status_(DFTI_NO_ERROR)
  {
    if (length == 0) {
      throw std::invalid_argument("RFFT length cannot be zero.");
    }
    if (length < 2 && length != 0) {
      throw std::invalid_argument("oneMKL RFFTPlan requires length >= 2.");
    }

    DFTI_CONFIG_VALUE precision;
    try {
      // *** Call helper from Details.hpp ***
      precision = Details::get_dfti_precision<T>();  // T is already Real here
    }
    catch (const std::logic_error& e) {
      throw std::invalid_argument(
          "Unsupported precision type for oneMKL RFFTPlan.");
    }

    mkl_status_ = DftiCreateDescriptor(
        &descriptor_handle_,
        precision,
        DFTI_REAL,
        1,
        static_cast<MKL_LONG>(length_));
    if (mkl_status_ != DFTI_NO_ERROR) {
      std::cerr << "MKL Error during DftiCreateDescriptor: "
                << DftiErrorMessage(mkl_status_) << std::endl;
      descriptor_handle_ = nullptr;
      throw std::runtime_error(
          "Failed to create oneMKL DFTI descriptor (real).");
    }

    mkl_status_ = DftiSetValue(
        descriptor_handle_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    if (mkl_status_ != DFTI_NO_ERROR) {
      DftiFreeDescriptor(&descriptor_handle_);
      descriptor_handle_ = nullptr;
      throw std::runtime_error(
          "Failed to set DFTI_CONJUGATE_EVEN_STORAGE for oneMKL RFFTPlan.");
    }

    mkl_status_
        = DftiSetValue(descriptor_handle_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (mkl_status_ != DFTI_NO_ERROR) {
      DftiFreeDescriptor(&descriptor_handle_);
      descriptor_handle_ = nullptr;
      throw std::runtime_error(
          "Failed to set DFTI_PLACEMENT for oneMKL RFFTPlan.");
    }

    input_strides_[0] = 0;
    output_strides_[0] = 0;
    mkl_status_
        = DftiSetValue(descriptor_handle_, DFTI_INPUT_STRIDES, input_strides_);
    if (mkl_status_ != DFTI_NO_ERROR) {
      DftiFreeDescriptor(&descriptor_handle_);
      descriptor_handle_ = nullptr;
      throw std::runtime_error("Failed to set DFTI_INPUT_STRIDES (real).");
    }
    mkl_status_ = DftiSetValue(
        descriptor_handle_, DFTI_OUTPUT_STRIDES, output_strides_);
    if (mkl_status_ != DFTI_NO_ERROR) {
      DftiFreeDescriptor(&descriptor_handle_);
      descriptor_handle_ = nullptr;
      throw std::runtime_error("Failed to set DFTI_OUTPUT_STRIDES (real).");
    }

    mkl_status_ = DftiCommitDescriptor(descriptor_handle_);
    if (mkl_status_ != DFTI_NO_ERROR) {
      DftiFreeDescriptor(&descriptor_handle_);
      descriptor_handle_ = nullptr;
      throw std::runtime_error(
          "Failed to commit oneMKL DFTI descriptor (real).");
    }
  }

  // Destructor Implementation
  template <typename T>
  RFFTPlanImpl<T>::~RFFTPlanImpl()
  {
    if (descriptor_handle_) {
      MKL_LONG free_status = DftiFreeDescriptor(&descriptor_handle_);
      if (free_status != DFTI_NO_ERROR) {
        std::cerr << "Warning: Failed to free oneMKL DFTI descriptor (real): "
                  << DftiErrorMessage(free_status) << std::endl;
      }
    }
  }

  // rfft Method Implementation
  template <typename T>
  OmniStatus RFFTPlanImpl<T>::rfft(
      std::span<const T> input, std::span<Complex> output) const
  {
    if (!descriptor_handle_) return OmniStatus::InvalidOperation;
    size_t N = length_;
    size_t output_size_expected = (N / 2) + 1;
    if (N == 0 && input.empty() && output.empty()) return OmniStatus::Success;
    if (input.size() != N || output.size() != output_size_expected)
      return OmniStatus::SizeMismatch;

    MKL_LONG status = DftiComputeForward(
        descriptor_handle_,
        const_cast<void*>(static_cast<const void*>(input.data())),
        const_cast<void*>(static_cast<void*>(output.data())));
    // *** Use helper from Details.hpp ***
    return Details::mkl_status_to_omnidsp_status(status);
  }

  // irfft Method Implementation
  template <typename T>
  OmniStatus RFFTPlanImpl<T>::irfft(
      std::span<const Complex> input, std::span<T> output) const
  {
    if (!descriptor_handle_) return OmniStatus::InvalidOperation;
    size_t N = length_;
    size_t input_size_expected = (N / 2) + 1;
    if (N == 0 && input.empty() && output.empty()) return OmniStatus::Success;
    if (input.size() != input_size_expected || output.size() != N)
      return OmniStatus::SizeMismatch;

    MKL_LONG status = DftiComputeBackward(
        descriptor_handle_,
        const_cast<void*>(static_cast<const void*>(input.data())),
        const_cast<void*>(static_cast<void*>(output.data())));
    // *** Use helper from Details.hpp ***
    return Details::mkl_status_to_omnidsp_status(status);
  }

  // get_length Method Implementation
  template <typename T>
  size_t RFFTPlanImpl<T>::get_length() const
  {
    return length_;
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  template class FFTPlanImpl<C32>;
  template class FFTPlanImpl<C64>;
  template class RFFTPlanImpl<F32>;
  template class RFFTPlanImpl<F64>;

}  // namespace OmniDSP::OneMKL
