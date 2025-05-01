/**
 * @file fft.cpp (onemkl)
 * @brief Implements oneMKL backend FFTPlanImpl and RFFTPlanImpl classes using
 * DFTI.
 */

#include "fft.hpp"  // Include the corresponding header file

#ifdef OMNIDSP_USE_ONEMKL  // Compile guard

// Include MKL header for DFTI
#include <mkl.h>

#include <complex>
#include <iostream>  // For debug/error messages
#include <limits>    // For numeric_limits
#include <span>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>     // For exception messages
#include <vector>

// Include core types to ensure Detail::* traits are available if needed by
// helpers
#include <OmniDSP/core_types.hpp>

namespace OmniDSP::backend {

  // Helper function from onemkl/backend.cpp (or move to common utility)
  // Ensure this helper is available, either defined here, in backend.cpp, or a
  // common header
  inline Status mkl_status_to_omnidsp_status(MKL_LONG status)
  {
    if (status == DFTI_NO_ERROR) {
      return Status::Success;
    }
    // Log the MKL error message for debugging purposes
    std::cerr << "MKL DFTI Error: " << DftiErrorMessage(status)
              << " (Code: " << status << ")" << std::endl;
    if (status == DFTI_MEMORY_ERROR) return Status::AllocationError;
    if (status == DFTI_INVALID_CONFIGURATION) return Status::InvalidArgument;
    if (status == DFTI_INCONSISTENT_CONFIGURATION)
      return Status::InvalidArgument;
    if (status == DFTI_NUMBER_OF_THREADS_ERROR) return Status::BackendError;
    if (status == DFTI_UNIMPLEMENTED) return Status::UnsupportedFeature;
    // Add more specific mappings if needed
    return Status::BackendError;  // Generic backend error for other MKL issues
  }

  // Helper to get DFTI precision enum value
  template <typename T>
  constexpr DFTI_CONFIG_VALUE get_dfti_precision()
  {
    if constexpr (std::is_same_v<T, float>) return DFTI_SINGLE;
    if constexpr (std::is_same_v<T, double>) return DFTI_DOUBLE;
    return DFTI_PRECISION_ERROR;  // Indicate error if type is wrong
  }

  //--------------------------------------------------------------------------
  // OneMKLFFTPlanImpl Method Definitions (Complex FFT)
  //--------------------------------------------------------------------------

  // Constructor Implementation
  template <typename T>
  OneMKLFFTPlanImpl<T>::OneMKLFFTPlanImpl(size_t length)
      : length_(length), mkl_status_(DFTI_NO_ERROR)
  {
    if (length == 0) {
      throw std::invalid_argument("FFT length cannot be zero.");
    }
    // Check if length is a power of two (optional, but good practice for some
    // algos) if ((length & (length - 1)) != 0) {
    //     throw std::invalid_argument("oneMKL FFTPlan currently requires
    //     power-of-two length.");
    // }

    using RealType = typename T::value_type;
    DFTI_CONFIG_VALUE precision = get_dfti_precision<RealType>();
    if (precision == DFTI_PRECISION_ERROR) {
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
      // Set handle to null explicitly on failure
      descriptor_handle_ = nullptr;
      throw std::runtime_error(
          "Failed to create oneMKL DFTI descriptor (complex).");
    }

    // Set configuration (e.g., out-of-place)
    mkl_status_
        = DftiSetValue(descriptor_handle_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (mkl_status_ != DFTI_NO_ERROR) {
      DftiFreeDescriptor(&descriptor_handle_);  // Clean up
      descriptor_handle_ = nullptr;
      throw std::runtime_error(
          "Failed to set DFTI_PLACEMENT for oneMKL FFTPlan.");
    }

    // Commit the descriptor
    mkl_status_ = DftiCommitDescriptor(descriptor_handle_);
    if (mkl_status_ != DFTI_NO_ERROR) {
      DftiFreeDescriptor(&descriptor_handle_);
      descriptor_handle_ = nullptr;
      throw std::runtime_error(
          "Failed to commit oneMKL DFTI descriptor (complex).");
    }

    // std::cout << "oneMKL FFTPlanImpl created for length " << length_ <<
    // std::endl; // Debug
  }

  // Destructor Implementation
  template <typename T>
  OneMKLFFTPlanImpl<T>::~OneMKLFFTPlanImpl()
  {
    if (descriptor_handle_) {
      MKL_LONG free_status = DftiFreeDescriptor(&descriptor_handle_);
      if (free_status != DFTI_NO_ERROR) {
        std::cerr
            << "Warning: Failed to free oneMKL DFTI descriptor (complex): "
            << DftiErrorMessage(free_status) << std::endl;
      }
      else {
        // std::cout << "oneMKL FFTPlanImpl destroyed for length " << length_ <<
        // std::endl; // Debug
      }
    }
  }

  // fft Method Implementation
  template <typename T>
  Status OneMKLFFTPlanImpl<T>::fft(
      std::span<const T> input, std::span<T> output) const
  {
    if (!descriptor_handle_)
      return Status::InvalidOperation;  // Plan not initialized
    if (input.size() != length_ || output.size() != length_) {
      return Status::SizeMismatch;
    }
    if (length_ == 0) return Status::Success;  // Nothing to do

    // MKL uses void* for data pointers
    MKL_LONG status = DftiComputeForward(
        descriptor_handle_,
        const_cast<void*>(static_cast<const void*>(input.data())),  // Input
        const_cast<void*>(static_cast<void*>(output.data()))        // Output
    );

    return mkl_status_to_omnidsp_status(status);
  }

  // ifft Method Implementation
  template <typename T>
  Status OneMKLFFTPlanImpl<T>::ifft(
      std::span<const T> input, std::span<T> output) const
  {
    if (!descriptor_handle_) return Status::InvalidOperation;
    if (input.size() != length_ || output.size() != length_) {
      return Status::SizeMismatch;
    }
    if (length_ == 0) return Status::Success;  // Nothing to do

    MKL_LONG status = DftiComputeBackward(
        descriptor_handle_,
        const_cast<void*>(static_cast<const void*>(input.data())),  // Input
        const_cast<void*>(static_cast<void*>(output.data()))        // Output
    );

    // Note: MKL default backward transform is unnormalized (no 1/N factor).
    // The user is responsible for scaling the output if needed.

    return mkl_status_to_omnidsp_status(status);
  }

  // get_length Method Implementation
  template <typename T>
  size_t OneMKLFFTPlanImpl<T>::get_length() const
  {
    return length_;
  }

  //--------------------------------------------------------------------------
  // OneMKLRFFTPlanImpl Method Definitions (Real FFT)
  //--------------------------------------------------------------------------

  // Constructor Implementation
  template <typename T>
  OneMKLRFFTPlanImpl<T>::OneMKLRFFTPlanImpl(size_t length)
      : length_(length), mkl_status_(DFTI_NO_ERROR)
  {
    if (length == 0) {
      throw std::invalid_argument("RFFT length cannot be zero.");
    }
    if (length < 2
        && length != 0) {  // MKL DFTI requires N>=2 for real transforms
      throw std::invalid_argument("oneMKL RFFTPlan requires length >= 2.");
    }
    // Power of two check might be needed depending on MKL version/config, but
    // often not strictly required if ((length & (length - 1)) != 0) {
    //     throw std::invalid_argument("oneMKL RFFTPlan currently requires
    //     power-of-two length.");
    // }

    DFTI_CONFIG_VALUE precision = get_dfti_precision<T>();
    if (precision == DFTI_PRECISION_ERROR) {
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

    // Configure storage format (CCE)
    mkl_status_ = DftiSetValue(
        descriptor_handle_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    if (mkl_status_ != DFTI_NO_ERROR) {
      DftiFreeDescriptor(&descriptor_handle_);
      descriptor_handle_ = nullptr;
      throw std::runtime_error(
          "Failed to set DFTI_CONJUGATE_EVEN_STORAGE for oneMKL RFFTPlan.");
    }

    // Configure placement
    mkl_status_
        = DftiSetValue(descriptor_handle_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (mkl_status_ != DFTI_NO_ERROR) {
      DftiFreeDescriptor(&descriptor_handle_);
      descriptor_handle_ = nullptr;
      throw std::runtime_error(
          "Failed to set DFTI_PLACEMENT for oneMKL RFFTPlan.");
    }

    // Set strides (typically 0 for single 1D transform)
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

    // Commit descriptor
    mkl_status_ = DftiCommitDescriptor(descriptor_handle_);
    if (mkl_status_ != DFTI_NO_ERROR) {
      DftiFreeDescriptor(&descriptor_handle_);
      descriptor_handle_ = nullptr;
      throw std::runtime_error(
          "Failed to commit oneMKL DFTI descriptor (real).");
    }

    // std::cout << "oneMKL RFFTPlanImpl created for length " << length_ <<
    // std::endl; // Debug
  }

  // Destructor Implementation
  template <typename T>
  OneMKLRFFTPlanImpl<T>::~OneMKLRFFTPlanImpl()
  {
    if (descriptor_handle_) {
      MKL_LONG free_status = DftiFreeDescriptor(&descriptor_handle_);
      if (free_status != DFTI_NO_ERROR) {
        std::cerr << "Warning: Failed to free oneMKL DFTI descriptor (real): "
                  << DftiErrorMessage(free_status) << std::endl;
      }
      else {
        // std::cout << "oneMKL RFFTPlanImpl destroyed for length " << length_
        // << std::endl; // Debug
      }
    }
  }

  // rfft Method Implementation
  template <typename T>
  Status OneMKLRFFTPlanImpl<T>::rfft(
      std::span<const T> input, std::span<Complex> output) const
  {
    if (!descriptor_handle_) return Status::InvalidOperation;

    size_t N = length_;
    size_t output_size_expected = (N / 2) + 1;

    if (input.size() != N || output.size() != output_size_expected) {
      return Status::SizeMismatch;
    }
    if (N == 0) return Status::Success;  // Nothing to do

    MKL_LONG status = DftiComputeForward(
        descriptor_handle_,
        const_cast<void*>(
            static_cast<const void*>(input.data())),  // Real input
        const_cast<void*>(
            static_cast<void*>(output.data()))  // Complex output (CCE format)
    );

    return mkl_status_to_omnidsp_status(status);
  }

  // irfft Method Implementation
  template <typename T>
  Status OneMKLRFFTPlanImpl<T>::irfft(
      std::span<const Complex> input, std::span<T> output) const
  {
    if (!descriptor_handle_) return Status::InvalidOperation;

    size_t N = length_;
    size_t input_size_expected = (N / 2) + 1;

    if (input.size() != input_size_expected || output.size() != N) {
      return Status::SizeMismatch;
    }
    if (N == 0) return Status::Success;  // Nothing to do

    MKL_LONG status = DftiComputeBackward(
        descriptor_handle_,
        const_cast<void*>(static_cast<const void*>(
            input.data())),  // Complex input (CCE format)
        const_cast<void*>(static_cast<void*>(output.data()))  // Real output
    );

    // Note: MKL default backward transform for real data is scaled by 1/N.

    return mkl_status_to_omnidsp_status(status);
  }

  // get_length Method Implementation
  template <typename T>
  size_t OneMKLRFFTPlanImpl<T>::get_length() const
  {
    return length_;
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  // Define complex types for brevity
  using float_c = OmniDSP::C32;
  using double_c = OmniDSP::C64;

  // OneMKLFFTPlanImpl Instantiations
  template class OmniDSP::backend::OneMKLFFTPlanImpl<float_c>;
  template class OmniDSP::backend::OneMKLFFTPlanImpl<double_c>;

  // OneMKLRFFTPlanImpl Instantiations
  template class OmniDSP::backend::OneMKLRFFTPlanImpl<float>;
  template class OmniDSP::backend::OneMKLRFFTPlanImpl<double>;

}  // namespace OmniDSP::backend

#endif  // OMNIDSP_USE_ONEMKL
