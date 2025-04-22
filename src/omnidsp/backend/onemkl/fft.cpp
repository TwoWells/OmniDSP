/**
 * @file fft.cpp (onemkl)
 * @brief Implements oneMKL backend FFTPlanImpl and RFFTPlanImpl classes using
 * DFTI.
 */

// Only compile this file if oneMKL backend is enabled via CMake
#ifdef USE_ONEMKL

#include "OmniDSP/core_types.h"  // For Status, RealT, ComplexT etc.
#include "backend.h"             // oneMKL backend declarations

// Include MKL header for DFTI
#include <mkl.h>

#include <complex>
#include <iostream>  // For debug/error messages
#include <span>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>     // For exception messages
#include <vector>

namespace OmniDSP {
namespace backend {

// Helper function from onemkl/backend.cpp (or move to common utility)
inline Status mkl_status_to_omnidsp_status(MKL_LONG status) {
  if (status == DFTI_NO_ERROR) {
    return Status::Success;
  }
  // Log the MKL error message for debugging purposes
  std::cerr << "MKL DFTI Error: " << DftiErrorMessage(status)
            << " (Code: " << status << ")" << std::endl;
  if (status == DFTI_MEMORY_ERROR) return Status::AllocationError;
  if (status == DFTI_INVALID_CONFIGURATION) return Status::InvalidArgument;
  if (status == DFTI_INCONSISTENT_CONFIGURATION) return Status::InvalidArgument;
  if (status == DFTI_NUMBER_OF_THREADS_ERROR) return Status::BackendError;
  if (status == DFTI_UNIMPLEMENTED) return Status::UnsupportedFeature;
  // Add more specific mappings if needed
  return Status::BackendError;  // Generic backend error for other MKL issues
}

// Helper to get DFTI precision enum value
template <typename T>
constexpr DFTI_CONFIG_VALUE get_dfti_precision() {
  if constexpr (std::is_same_v<T, float>) return DFTI_SINGLE;
  if constexpr (std::is_same_v<T, double>) return DFTI_DOUBLE;
  // Add long double if needed and supported
  // static_assert(false, "Unsupported precision for DFTI"); // Or handle
  // differently
  return DFTI_PRECISION_ERROR;  // Indicate error if type is wrong
}

//--------------------------------------------------------------------------
// OneMKLFFTPlanImpl Method Definitions (Complex FFT)
//--------------------------------------------------------------------------

template <typename T>
OneMKLFFTPlanImpl<T>::OneMKLFFTPlanImpl(size_t length)
    : length_(length), mkl_status_(DFTI_NO_ERROR) {
  if (length == 0) {
    throw std::invalid_argument("FFT length cannot be zero.");
  }

  // Determine precision based on the complex value type
  using ValueType = typename T::value_type;
  DFTI_CONFIG_VALUE precision = get_dfti_precision<ValueType>();
  if (precision == DFTI_PRECISION_ERROR) {
    throw std::invalid_argument(
        "Unsupported precision type for oneMKL FFTPlan.");
  }

  // Create descriptor for 1D complex FFT
  mkl_status_ =
      DftiCreateDescriptor(&descriptor_handle_, precision, DFTI_COMPLEX, 1,
                           static_cast<MKL_LONG>(length_));
  if (mkl_status_ != DFTI_NO_ERROR) {
    std::cerr << "MKL Error during DftiCreateDescriptor: "
              << DftiErrorMessage(mkl_status_) << std::endl;
    throw std::runtime_error(
        "Failed to create oneMKL DFTI descriptor (complex).");
  }

  // Set configuration (e.g., out-of-place)
  mkl_status_ =
      DftiSetValue(descriptor_handle_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  if (mkl_status_ != DFTI_NO_ERROR) {
    DftiFreeDescriptor(
        &descriptor_handle_);  // Clean up partially created descriptor
    throw std::runtime_error(
        "Failed to set DFTI_PLACEMENT for oneMKL FFTPlan.");
  }
  // Set other parameters as needed (e.g., DFTI_FORWARD_SCALE,
  // DFTI_BACKWARD_SCALE) Default scaling: forward is 1.0, backward is 1.0/N

  // Commit the descriptor
  mkl_status_ = DftiCommitDescriptor(descriptor_handle_);
  if (mkl_status_ != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&descriptor_handle_);
    throw std::runtime_error(
        "Failed to commit oneMKL DFTI descriptor (complex).");
  }

  std::cout << "oneMKL FFTPlanImpl created for length " << length_
            << std::endl;  // Debug
}

template <typename T>
OneMKLFFTPlanImpl<T>::~OneMKLFFTPlanImpl() {
  if (descriptor_handle_) {
    mkl_status_ = DftiFreeDescriptor(&descriptor_handle_);
    if (mkl_status_ != DFTI_NO_ERROR) {
      std::cerr << "Warning: Failed to free oneMKL DFTI descriptor (complex): "
                << DftiErrorMessage(mkl_status_) << std::endl;
    } else {
      std::cout << "oneMKL FFTPlanImpl destroyed for length " << length_
                << std::endl;  // Debug
    }
  }
}

template <typename T>
Status OneMKLFFTPlanImpl<T>::fft(std::span<const T> input,
                                 std::span<T> output) const {
  if (!descriptor_handle_)
    return Status::InvalidOperation;  // Plan not initialized
  if (input.size() != length_ || output.size() != length_) {
    return Status::SizeMismatch;
  }

  // MKL uses void* for data pointers
  mkl_status_ = DftiComputeForward(
      descriptor_handle_,
      const_cast<void*>(static_cast<const void*>(input.data())),
      const_cast<void*>(static_cast<void*>(output.data())));

  return mkl_status_to_omnidsp_status(mkl_status_);
}

template <typename T>
Status OneMKLFFTPlanImpl<T>::ifft(std::span<const T> input,
                                  std::span<T> output) const {
  if (!descriptor_handle_) return Status::InvalidOperation;
  if (input.size() != length_ || output.size() != length_) {
    return Status::SizeMismatch;
  }

  mkl_status_ = DftiComputeBackward(
      descriptor_handle_,
      const_cast<void*>(static_cast<const void*>(input.data())),
      const_cast<void*>(static_cast<void*>(output.data())));

  // Note: MKL default backward transform is unnormalized (no 1/N factor).
  // The user is responsible for scaling the output if needed.

  return mkl_status_to_omnidsp_status(mkl_status_);
}

template <typename T>
size_t OneMKLFFTPlanImpl<T>::get_length() const {
  return length_;
}

//--------------------------------------------------------------------------
// OneMKLRFFTPlanImpl Method Definitions (Real FFT)
//--------------------------------------------------------------------------

template <typename T>
OneMKLRFFTPlanImpl<T>::OneMKLRFFTPlanImpl(size_t length)
    : length_(length), mkl_status_(DFTI_NO_ERROR) {
  if (length == 0) {
    throw std::invalid_argument("RFFT length cannot be zero.");
  }

  DFTI_CONFIG_VALUE precision = get_dfti_precision<T>();
  if (precision == DFTI_PRECISION_ERROR) {
    throw std::invalid_argument(
        "Unsupported precision type for oneMKL RFFTPlan.");
  }

  // Create descriptor for 1D real FFT
  mkl_status_ = DftiCreateDescriptor(&descriptor_handle_, precision, DFTI_REAL,
                                     1, static_cast<MKL_LONG>(length_));
  if (mkl_status_ != DFTI_NO_ERROR) {
    std::cerr << "MKL Error during DftiCreateDescriptor: "
              << DftiErrorMessage(mkl_status_) << std::endl;
    throw std::runtime_error("Failed to create oneMKL DFTI descriptor (real).");
  }

  // Set configuration for real FFTs (Conjugate-Even Complex Storage)
  // This stores the N/2+1 complex results in a complex array.
  mkl_status_ = DftiSetValue(descriptor_handle_, DFTI_CONJUGATE_EVEN_STORAGE,
                             DFTI_COMPLEX_COMPLEX);
  if (mkl_status_ != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&descriptor_handle_);
    throw std::runtime_error(
        "Failed to set DFTI_CONJUGATE_EVEN_STORAGE for oneMKL RFFTPlan.");
  }

  mkl_status_ =
      DftiSetValue(descriptor_handle_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  if (mkl_status_ != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&descriptor_handle_);
    throw std::runtime_error(
        "Failed to set DFTI_PLACEMENT for oneMKL RFFTPlan.");
  }

  // Set strides (usually 0 for single 1D transform)
  // Input (real) stride
  input_strides_[0] = 0;  // Distance between start of consecutive real
                          // sequences (only 1 sequence)
  // Output (complex CCE format) stride
  output_strides_[0] =
      0;  // Distance between start of consecutive complex sequences
  mkl_status_ =
      DftiSetValue(descriptor_handle_, DFTI_INPUT_STRIDES, input_strides_);
  if (mkl_status_ != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&descriptor_handle_);
    throw std::runtime_error("Failed to set DFTI_INPUT_STRIDES (real).");
  }
  mkl_status_ =
      DftiSetValue(descriptor_handle_, DFTI_OUTPUT_STRIDES, output_strides_);
  if (mkl_status_ != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&descriptor_handle_);
    throw std::runtime_error("Failed to set DFTI_OUTPUT_STRIDES (real).");
  }

  // Set number of transforms and distances (usually 1 and 0 for single
  // transform)
  mkl_status_ = DftiSetValue(descriptor_handle_, DFTI_NUMBER_OF_TRANSFORMS, 1);
  if (mkl_status_ != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&descriptor_handle_);
    throw std::runtime_error("Failed to set DFTI_NUMBER_OF_TRANSFORMS (real).");
  }
  mkl_status_ = DftiSetValue(descriptor_handle_, DFTI_INPUT_DISTANCE, 0);
  if (mkl_status_ != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&descriptor_handle_);
    throw std::runtime_error("Failed to set DFTI_INPUT_DISTANCE (real).");
  }
  mkl_status_ = DftiSetValue(descriptor_handle_, DFTI_OUTPUT_DISTANCE, 0);
  if (mkl_status_ != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&descriptor_handle_);
    throw std::runtime_error("Failed to set DFTI_OUTPUT_DISTANCE (real).");
  }

  // Set scaling factors (defaults are 1.0 for forward, 1.0/N for backward)
  // DftiSetValue(descriptor_handle_, DFTI_FORWARD_SCALE, 1.0);
  // DftiSetValue(descriptor_handle_, DFTI_BACKWARD_SCALE, 1.0 /
  // static_cast<double>(length_));

  // Commit the descriptor
  mkl_status_ = DftiCommitDescriptor(descriptor_handle_);
  if (mkl_status_ != DFTI_NO_ERROR) {
    DftiFreeDescriptor(&descriptor_handle_);
    throw std::runtime_error("Failed to commit oneMKL DFTI descriptor (real).");
  }

  std::cout << "oneMKL RFFTPlanImpl created for length " << length_
            << std::endl;  // Debug
}

template <typename T>
OneMKLRFFTPlanImpl<T>::~OneMKLRFFTPlanImpl() {
  if (descriptor_handle_) {
    mkl_status_ = DftiFreeDescriptor(&descriptor_handle_);
    if (mkl_status_ != DFTI_NO_ERROR) {
      std::cerr << "Warning: Failed to free oneMKL DFTI descriptor (real): "
                << DftiErrorMessage(mkl_status_) << std::endl;
    } else {
      std::cout << "oneMKL RFFTPlanImpl destroyed for length " << length_
                << std::endl;  // Debug
    }
  }
}

template <typename T>
Status OneMKLRFFTPlanImpl<T>::rfft(std::span<const RealT<T>> input,
                                   std::span<ComplexT<T>> output) const {
  if (!descriptor_handle_) return Status::InvalidOperation;

  size_t N = length_;
  size_t output_size_expected = (N / 2) + 1;

  if (input.size() != N || output.size() != output_size_expected) {
    return Status::SizeMismatch;
  }

  // Input is RealT<T>*, Output is ComplexT<T>* (which MKL handles with
  // COMPLEX_COMPLEX storage)
  mkl_status_ = DftiComputeForward(
      descriptor_handle_,
      const_cast<void*>(static_cast<const void*>(input.data())),
      const_cast<void*>(static_cast<void*>(output.data())));

  return mkl_status_to_omnidsp_status(mkl_status_);
}

template <typename T>
Status OneMKLRFFTPlanImpl<T>::irfft(std::span<const ComplexT<T>> input,
                                    std::span<RealT<T>> output) const {
  if (!descriptor_handle_) return Status::InvalidOperation;

  size_t N = length_;
  size_t input_size_expected = (N / 2) + 1;

  if (input.size() != input_size_expected || output.size() != N) {
    return Status::SizeMismatch;
  }

  // Input is ComplexT<T>*, Output is RealT<T>*
  mkl_status_ = DftiComputeBackward(
      descriptor_handle_,
      const_cast<void*>(static_cast<const void*>(input.data())),
      const_cast<void*>(static_cast<void*>(output.data())));

  // Note: MKL default backward transform is scaled by 1/N.
  // If different scaling is needed, set DFTI_BACKWARD_SCALE during setup.

  return mkl_status_to_omnidsp_status(mkl_status_);
}

template <typename T>
size_t OneMKLRFFTPlanImpl<T>::get_length() const {
  return length_;
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
// Instantiate templates for common types (float, double) to ensure code
// generation.

// Define complex types for brevity
using float_c = OmniDSP::ComplexT<float>;
using double_c = OmniDSP::ComplexT<double>;

// OneMKLFFTPlanImpl Instantiations
template class OmniDSP::backend::OneMKLFFTPlanImpl<float_c>;
template class OmniDSP::backend::OneMKLFFTPlanImpl<double_c>;

// OneMKLRFFTPlanImpl Instantiations
template class OmniDSP::backend::OneMKLRFFTPlanImpl<float>;
template class OmniDSP::backend::OneMKLRFFTPlanImpl<double>;

}  // namespace backend
}  // namespace OmniDSP

#endif  // USE_ONEMKL
