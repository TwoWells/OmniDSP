/**
 * @file convolution.cpp (onemkl)
 * @brief Implements oneMKL backend ConvolutionPlanImpl and CorrelationPlanImpl
 * classes, using internal oneMKL FFT plans (DFTI) for FFT operations.
 */

#include "OmniDSP/core_types.h"  // For Status, ConvolutionMode etc.
#include "backend.h"             // oneMKL backend declarations

// Include MKL header
#include <mkl.h>

#include <algorithm>  // For std::reverse, std::copy
#include <cmath>      // For log2, ceil, etc.
#include <complex>
#include <iostream>  // For debug/error messages
#include <memory>    // For std::unique_ptr
#include <numeric>   // For std::max, std::min
#include <span>
#include <stdexcept>    // For std::runtime_error, std::invalid_argument
#include <type_traits>  // For std::is_same_v
#include <vector>

namespace OmniDSP {
namespace backend {

// Helper function from onemkl/backend.cpp (or move to common utility)
inline Status mkl_status_to_omnidsp_status(MKL_LONG status) {
  if (status == DFTI_NO_ERROR) {
    return Status::Success;
  }
  std::cerr << "MKL Error: " << DftiErrorMessage(status) << std::endl;
  if (status == DFTI_MEMORY_ERROR) return Status::AllocationError;
  if (status == DFTI_INVALID_CONFIGURATION) return Status::InvalidArgument;
  if (status == DFTI_INCONSISTENT_CONFIGURATION) return Status::InvalidArgument;
  if (status == DFTI_NUMBER_OF_THREADS_ERROR) return Status::BackendError;
  return Status::BackendError;
}

// Helper function to calculate next power of two (move to common utility?)
inline size_t next_power_of_two(size_t n) {
  if (n == 0) return 1;
  size_t result = 1;
  while (result < n) {
    result <<= 1;
    if (result == 0) return size_t(-1);  // Overflow
  }
  return result;
}

// Forward declare the concrete FFT plan impls needed internally
// Assumes these are defined in onemkl/fft.cpp or similar
template <typename T_Complex>
class OneMKLFFTPlanImpl;  // Expects complex type
template <typename T_Real>
class OneMKLRFFTPlanImpl;  // Expects real type

// Helper type traits (move to common utility?)
namespace Detail {
template <typename T>
struct is_complex : std::false_type {};
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};
template <typename T>
constexpr bool is_complex_v = is_complex<T>::value;

template <typename T>
struct ValueType {
  using type = T;
};
template <typename T>
struct ValueType<std::complex<T>> {
  using type = T;
};
}  // namespace Detail

//--------------------------------------------------------------------------
// OneMKLConvolutionPlanImpl Method Definitions
//--------------------------------------------------------------------------

template <typename T>
OneMKLConvolutionPlanImpl<T>::OneMKLConvolutionPlanImpl(
    const std::vector<T>& kernel, ConvolutionMode mode)
    : mode_(mode),
      kernel_length_(kernel.size()),
      mkl_status_(DFTI_NO_ERROR)  // Initialize MKL status
{
  if (kernel_length_ == 0) {
    throw std::invalid_argument("Convolution kernel cannot be empty.");
  }

  // --- Strategy: FFT-based Convolution using internal oneMKL FFT Plan ---
  fft_length_ = next_power_of_two(kernel_length_ * 2);  // Placeholder strategy
  if (fft_length_ == size_t(-1)) {
    throw std::length_error(
        "Calculated FFT length for convolution plan exceeds limits.");
  }

  // --- Create internal FFT plan ---
  try {
    if constexpr (Detail::is_complex_v<T>) {  // T is std::complex<Real>
      internal_fft_plan_ = std::make_unique<OneMKLFFTPlanImpl<T>>(fft_length_);
      // Check status from FFT plan constructor
      if (internal_fft_plan_->mkl_status_ != DFTI_NO_ERROR) {
        mkl_status_ = internal_fft_plan_->mkl_status_;  // Store error
        throw std::runtime_error(
            "Failed to create internal oneMKL FFT plan for ConvolutionPlan.");
      }
    } else {  // T is Real (float or double)
      internal_rfft_plan_ =
          std::make_unique<OneMKLRFFTPlanImpl<T>>(fft_length_);
      if (internal_rfft_plan_->mkl_status_ != DFTI_NO_ERROR) {
        mkl_status_ = internal_rfft_plan_->mkl_status_;
        throw std::runtime_error(
            "Failed to create internal oneMKL RFFT plan for ConvolutionPlan.");
      }
    }
  } catch (const std::exception& e) {
    // Catch potential bad_alloc or runtime_error from FFT plan creation
    mkl_status_ = DFTI_MEMORY_ERROR;  // Assume allocation error or setup error
    throw std::runtime_error(
        "Failed to create internal FFT plan for ConvolutionPlan: " +
        std::string(e.what()));
  } catch (...) {
    mkl_status_ = DFTI_INTERNAL_ERROR;  // Generic internal error
    throw std::runtime_error(
        "Unknown error creating internal FFT plan for ConvolutionPlan.");
  }

  // --- Prepare and FFT the kernel using the internal plan ---
  std::vector<T> padded_kernel = kernel;
  std::reverse(padded_kernel.begin(),
               padded_kernel.end());       // Reverse for convolution
  padded_kernel.resize(fft_length_, T{});  // Zero-pad

  Status fft_status = Status::Success;
  if constexpr (Detail::is_complex_v<T>) {  // Complex Kernel -> Complex FFT
    kernel_fft_.resize(fft_length_);
    fft_status = internal_fft_plan_->fft(padded_kernel, kernel_fft_);
  } else {                        // Real Kernel -> Real FFT
    using Complex = ComplexT<T>;  // Get the corresponding complex type
    kernel_fft_.resize(fft_length_ / 2 + 1);
    fft_status = internal_rfft_plan_->rfft(padded_kernel, kernel_fft_);
  }

  if (fft_status != Status::Success) {
    // Store the MKL status if available from the failed FFT operation
    // (Assuming FFTPlanImpl methods might update mkl_status_ on failure)
    // mkl_status_ = internal_fft_plan_ ? internal_fft_plan_->mkl_status_ :
    // internal_rfft_plan_->mkl_status_;
    throw std::runtime_error(
        "Failed to compute FFT of kernel during ConvolutionPlan creation. "
        "Status: " +
        std::string(get_status_string(fft_status)));
  }

  std::cout << "oneMKL ConvPlanImpl created. Kernel Len: " << kernel_length_
            << ", FFT Len: " << fft_length_ << std::endl;  // Debug
}

template <typename T>
OneMKLConvolutionPlanImpl<T>::~OneMKLConvolutionPlanImpl() {
  // unique_ptr members (internal_fft_plan_ / internal_rfft_plan_) handle
  // cleanup automatically This includes freeing the DFTI descriptor via the FFT
  // plan destructors.
  std::cout << "oneMKL ConvPlanImpl destroyed." << std::endl;  // Debug
}

template <typename T>
Status OneMKLConvolutionPlanImpl<T>::execute(std::span<const T> input,
                                             std::span<T> output) const {
  // Check if plan was properly initialized
  if constexpr (Detail::is_complex_v<T>) {
    if (!internal_fft_plan_) return Status::InvalidOperation;
  } else {
    if (!internal_rfft_plan_) return Status::InvalidOperation;
  }

  size_t input_len = input.size();
  size_t full_output_len = input_len + kernel_length_ - 1;
  size_t current_fft_len = fft_length_;

  // --- FFT Length Handling ---
  size_t required_fft_len_for_linear = next_power_of_two(full_output_len);
  if (current_fft_len < required_fft_len_for_linear) {
    std::cerr << "Warning: Convolution plan's internal FFT length ("
              << current_fft_len
              << ") is too small for full linear convolution with input size ("
              << input_len
              << "). Result will be circular. Overlap-Add/Save not implemented "
                 "in skeleton."
              << std::endl;
  }
  if (input_len > current_fft_len) {
    std::cerr << "Error: Input size (" << input_len
              << ") exceeds convolution plan's internal FFT length ("
              << current_fft_len << "). Overlap-Add/Save not implemented."
              << std::endl;
    return Status::InvalidArgument;
  }

  // --- Allocate temporary buffers ---
  std::vector<T> input_padded(current_fft_len, T{});
  std::copy(input.begin(), input.end(), input_padded.begin());

  using Complex = ComplexT<typename Detail::ValueType<T>::type>;
  std::vector<Complex> input_fft(current_fft_len);
  std::vector<Complex> product_fft(current_fft_len);
  std::vector<T> result_full_padded(current_fft_len);

  // --- Perform FFT on input using internal plan ---
  Status status;
  if constexpr (Detail::is_complex_v<T>) {
    input_fft.resize(current_fft_len);
    status = internal_fft_plan_->fft(input_padded, input_fft);
  } else {
    input_fft.resize(current_fft_len / 2 + 1);
    status = internal_rfft_plan_->rfft(input_padded, input_fft);
  }
  if (status != Status::Success) return status;

  // --- Multiply FFTs (Input * Kernel) using MKL VML ---
  product_fft.resize(input_fft.size());
  using Value = typename Detail::ValueType<T>::type;
  MKL_LONG n_mul = static_cast<MKL_LONG>(
      product_fft.size());  // Number of complex elements to multiply

  if constexpr (std::is_same_v<Value, float>) {
    vcMul(n_mul, reinterpret_cast<const MKL_Complex8*>(input_fft.data()),
          reinterpret_cast<const MKL_Complex8*>(kernel_fft_.data()),
          reinterpret_cast<MKL_Complex8*>(product_fft.data()));
  } else {  // double
    vzMul(n_mul, reinterpret_cast<const MKL_Complex16*>(input_fft.data()),
          reinterpret_cast<const MKL_Complex16*>(kernel_fft_.data()),
          reinterpret_cast<MKL_Complex16*>(product_fft.data()));
  }
  // TODO: Check VML status? VML usually doesn't return status codes directly,
  // but might have error handlers or check input validity.

  // --- Inverse FFT using internal plan ---
  result_full_padded.resize(current_fft_len);
  if constexpr (Detail::is_complex_v<T>) {
    status = internal_fft_plan_->ifft(product_fft, result_full_padded);
  } else {
    status = internal_rfft_plan_->irfft(product_fft, result_full_padded);
  }
  if (status != Status::Success) return status;

  // --- Extract Correct Output based on Mode ---
  size_t output_len = get_output_length(input_len);
  if (output.size() < output_len) {
    return Status::SizeMismatch;
  }
  if (full_output_len > result_full_padded.size()) {
    return Status::Failure;  // Error in length calculation or FFT
  }

  switch (mode_) {
    case ConvolutionMode::Full:
      std::copy(result_full_padded.begin(),
                result_full_padded.begin() + full_output_len, output.begin());
      break;
    case ConvolutionMode::Same: {
      size_t start = (kernel_length_ - 1) / 2;
      if (start + input_len > full_output_len) return Status::Failure;
      std::copy(result_full_padded.begin() + start,
                result_full_padded.begin() + start + input_len, output.begin());
    } break;
    case ConvolutionMode::Valid: {
      size_t start = kernel_length_ - 1;
      size_t valid_len =
          (input_len >= kernel_length_) ? (input_len - kernel_length_ + 1) : 0;
      if (valid_len > output_len) return Status::Failure;
      if (start + valid_len > full_output_len) return Status::Failure;
      if (valid_len > 0) {
        std::copy(result_full_padded.begin() + start,
                  result_full_padded.begin() + start + valid_len,
                  output.begin());
      }
    } break;
    default:
      return Status::InvalidArgument;  // Should not happen
  }

  if (output.size() > output_len) {
    std::fill(output.begin() + output_len, output.end(), T{});
  }

  return Status::Success;
}

template <typename T>
size_t OneMKLConvolutionPlanImpl<T>::get_kernel_length() const {
  return kernel_length_;
}

template <typename T>
ConvolutionMode OneMKLConvolutionPlanImpl<T>::get_mode() const {
  return mode_;
}

template <typename T>
size_t OneMKLConvolutionPlanImpl<T>::get_output_length(
    size_t input_length) const {
  if (kernel_length_ == 0) return 0;
  switch (mode_) {
    case ConvolutionMode::Full:
      if (input_length > SIZE_MAX - kernel_length_ + 1) return SIZE_MAX;
      return input_length + kernel_length_ - 1;
    case ConvolutionMode::Same:
      return input_length;
    case ConvolutionMode::Valid:
      return (input_length >= kernel_length_)
                 ? (input_length - kernel_length_ + 1)
                 : 0;
    default:
      return 0;
  }
}

//--------------------------------------------------------------------------
// OneMKLCorrelationPlanImpl Method Definitions
//--------------------------------------------------------------------------

template <typename T>
OneMKLCorrelationPlanImpl<T>::OneMKLCorrelationPlanImpl(
    const std::vector<T>& kernel, ConvolutionMode mode)
    : mode_(mode), template_length_(kernel.size()), mkl_status_(DFTI_NO_ERROR) {
  if (template_length_ == 0) {
    throw std::invalid_argument("Correlation template cannot be empty.");
  }
  fft_length_ =
      next_power_of_two(template_length_ * 2);  // Placeholder strategy
  if (fft_length_ == size_t(-1)) {
    throw std::length_error(
        "Calculated FFT length for correlation plan exceeds limits.");
  }

  // --- Create internal FFT plan ---
  try {
    if constexpr (Detail::is_complex_v<T>) {
      internal_fft_plan_ = std::make_unique<OneMKLFFTPlanImpl<T>>(fft_length_);
      if (internal_fft_plan_->mkl_status_ != DFTI_NO_ERROR) {
        mkl_status_ = internal_fft_plan_->mkl_status_;
        throw std::runtime_error(
            "Failed to create internal oneMKL FFT plan for CorrelationPlan.");
      }
    } else {
      internal_rfft_plan_ =
          std::make_unique<OneMKLRFFTPlanImpl<T>>(fft_length_);
      if (internal_rfft_plan_->mkl_status_ != DFTI_NO_ERROR) {
        mkl_status_ = internal_rfft_plan_->mkl_status_;
        throw std::runtime_error(
            "Failed to create internal oneMKL RFFT plan for CorrelationPlan.");
      }
    }
  } catch (const std::exception& e) {
    mkl_status_ = DFTI_MEMORY_ERROR;
    throw std::runtime_error(
        "Failed to create internal FFT plan for CorrelationPlan: " +
        std::string(e.what()));
  } catch (...) {
    mkl_status_ = DFTI_INTERNAL_ERROR;
    throw std::runtime_error(
        "Unknown error creating internal FFT plan for CorrelationPlan.");
  }

  // --- Prepare and FFT the template (NO reversal) ---
  std::vector<T> padded_template = kernel;
  padded_template.resize(fft_length_, T{});  // Zero-pad

  Status fft_status = Status::Success;
  using Complex = ComplexT<typename Detail::ValueType<T>::type>;
  std::vector<Complex> temp_template_fft;  // Temporary storage

  if constexpr (Detail::is_complex_v<T>) {
    temp_template_fft.resize(fft_length_);
    fft_status = internal_fft_plan_->fft(padded_template, temp_template_fft);
  } else {
    temp_template_fft.resize(fft_length_ / 2 + 1);
    fft_status = internal_rfft_plan_->rfft(padded_template, temp_template_fft);
  }

  if (fft_status != Status::Success) {
    throw std::runtime_error(
        "Failed to compute FFT of template during CorrelationPlan creation. "
        "Status: " +
        std::string(get_status_string(fft_status)));
  }

  // --- CONJUGATE the result and store in template_fft_conj_ ---
  template_fft_conj_ = temp_template_fft;  // Copy
  using Value = typename Detail::ValueType<T>::type;
  MKL_LONG n_conj = static_cast<MKL_LONG>(template_fft_conj_.size());
  if constexpr (std::is_same_v<Value, float>) {
    vcConj(n_conj,
           reinterpret_cast<const MKL_Complex8*>(template_fft_conj_.data()),
           reinterpret_cast<MKL_Complex8*>(
               template_fft_conj_.data()));  // In-place conjugate
  } else {                                   // double
    vzConj(n_conj,
           reinterpret_cast<const MKL_Complex16*>(template_fft_conj_.data()),
           reinterpret_cast<MKL_Complex16*>(
               template_fft_conj_.data()));  // In-place conjugate
  }
  // TODO: Check VML status?

  std::cout << "oneMKL CorrPlanImpl created. Template Len: " << template_length_
            << ", FFT Len: " << fft_length_ << std::endl;  // Debug
}

template <typename T>
OneMKLCorrelationPlanImpl<T>::~OneMKLCorrelationPlanImpl() {
  // unique_ptr members handle cleanup
  std::cout << "oneMKL CorrPlanImpl destroyed." << std::endl;  // Debug
}

template <typename T>
Status OneMKLCorrelationPlanImpl<T>::execute(std::span<const T> input,
                                             std::span<T> output) const {
  // Check if plan was properly initialized
  if constexpr (Detail::is_complex_v<T>) {
    if (!internal_fft_plan_) return Status::InvalidOperation;
  } else {
    if (!internal_rfft_plan_) return Status::InvalidOperation;
  }

  size_t input_len = input.size();
  size_t full_output_len = input_len + template_length_ - 1;
  size_t current_fft_len = fft_length_;

  // --- FFT Length Handling ---
  size_t required_fft_len_for_linear = next_power_of_two(full_output_len);
  if (current_fft_len < required_fft_len_for_linear) {
    std::cerr << "Warning: Correlation plan's internal FFT length ("
              << current_fft_len
              << ") is too small for full linear correlation with input size ("
              << input_len
              << "). Result will be circular. Overlap-Add/Save not implemented "
                 "in skeleton."
              << std::endl;
  }
  if (input_len > current_fft_len) {
    std::cerr << "Error: Input size (" << input_len
              << ") exceeds correlation plan's internal FFT length ("
              << current_fft_len << "). Overlap-Add/Save not implemented."
              << std::endl;
    return Status::InvalidArgument;
  }

  // --- Allocate temporary buffers ---
  std::vector<T> input_padded(current_fft_len, T{});
  std::copy(input.begin(), input.end(), input_padded.begin());

  using Complex = ComplexT<typename Detail::ValueType<T>::type>;
  std::vector<Complex> input_fft(current_fft_len);
  std::vector<Complex> product_fft(current_fft_len);
  std::vector<T> result_full_padded(current_fft_len);

  // --- Perform FFT on input ---
  Status status;
  if constexpr (Detail::is_complex_v<T>) {
    input_fft.resize(current_fft_len);
    status = internal_fft_plan_->fft(input_padded, input_fft);
  } else {
    input_fft.resize(current_fft_len / 2 + 1);
    status = internal_rfft_plan_->rfft(input_padded, input_fft);
  }
  if (status != Status::Success) return status;

  // --- Multiply FFTs (Input * Conjugated(Template)) ---
  product_fft.resize(input_fft.size());
  using Value = typename Detail::ValueType<T>::type;
  MKL_LONG n_mul = static_cast<MKL_LONG>(product_fft.size());
  if constexpr (std::is_same_v<Value, float>) {
    vcMul(n_mul, reinterpret_cast<const MKL_Complex8*>(input_fft.data()),
          reinterpret_cast<const MKL_Complex8*>(
              template_fft_conj_.data()),  // Use pre-conjugated template FFT
          reinterpret_cast<MKL_Complex8*>(product_fft.data()));
  } else {  // double
    vzMul(n_mul, reinterpret_cast<const MKL_Complex16*>(input_fft.data()),
          reinterpret_cast<const MKL_Complex16*>(
              template_fft_conj_.data()),  // Use pre-conjugated template FFT
          reinterpret_cast<MKL_Complex16*>(product_fft.data()));
  }

  // --- Inverse FFT ---
  result_full_padded.resize(current_fft_len);
  if constexpr (Detail::is_complex_v<T>) {
    status = internal_fft_plan_->ifft(product_fft, result_full_padded);
  } else {
    status = internal_rfft_plan_->irfft(product_fft, result_full_padded);
  }
  if (status != Status::Success) return status;

  // --- Extract Correct Output based on Mode ---
  size_t output_len = get_output_length(input_len);
  if (output.size() < output_len) {
    return Status::SizeMismatch;
  }
  if (full_output_len > result_full_padded.size()) {
    return Status::Failure;
  }

  // Correlation output indices differ slightly from convolution for 'same' and
  // 'valid'
  switch (mode_) {
    case ConvolutionMode::Full:
      std::copy(result_full_padded.begin(),
                result_full_padded.begin() + full_output_len, output.begin());
      break;
    case ConvolutionMode::Same: {
      size_t start =
          (full_output_len > input_len) ? (full_output_len - input_len) / 2 : 0;
      size_t count = std::min(input_len, full_output_len);
      if (start + count > full_output_len) return Status::Failure;
      std::copy(result_full_padded.begin() + start,
                result_full_padded.begin() + start + count, output.begin());
    } break;
    case ConvolutionMode::Valid: {
      size_t start = 0;  // Valid correlation starts at index 0
      size_t valid_len = (input_len >= template_length_)
                             ? (input_len - template_length_ + 1)
                             : 0;
      if (valid_len > output_len) return Status::Failure;
      if (start + valid_len > full_output_len) return Status::Failure;
      if (valid_len > 0) {
        std::copy(result_full_padded.begin() + start,
                  result_full_padded.begin() + start + valid_len,
                  output.begin());
      }
    } break;
    default:
      return Status::InvalidArgument;
  }

  if (output.size() > output_len) {
    std::fill(output.begin() + output_len, output.end(), T{});
  }

  return Status::Success;
}

template <typename T>
size_t OneMKLCorrelationPlanImpl<T>::get_template_length() const {
  return template_length_;
}

template <typename T>
ConvolutionMode OneMKLCorrelationPlanImpl<T>::get_mode() const {
  return mode_;
}

template <typename T>
size_t OneMKLCorrelationPlanImpl<T>::get_output_length(
    size_t input_length) const {
  // Same logic as convolution output length
  if (template_length_ == 0) return 0;
  switch (mode_) {
    case ConvolutionMode::Full:
      if (input_length > SIZE_MAX - template_length_ + 1) return SIZE_MAX;
      return input_length + template_length_ - 1;
    case ConvolutionMode::Same:
      return input_length;
    case ConvolutionMode::Valid:
      return (input_length >= template_length_)
                 ? (input_length - template_length_ + 1)
                 : 0;
    default:
      return 0;
  }
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------

// Define complex types for brevity
using float_c = OmniDSP::ComplexT<float>;
using double_c = OmniDSP::ComplexT<double>;

// OneMKLConvolutionPlanImpl Instantiations
template class OmniDSP::backend::OneMKLConvolutionPlanImpl<float>;
template class OmniDSP::backend::OneMKLConvolutionPlanImpl<double>;
template class OmniDSP::backend::OneMKLConvolutionPlanImpl<float_c>;
template class OmniDSP::backend::OneMKLConvolutionPlanImpl<double_c>;

// OneMKLCorrelationPlanImpl Instantiations
template class OmniDSP::backend::OneMKLCorrelationPlanImpl<float>;
template class OmniDSP::backend::OneMKLCorrelationPlanImpl<double>;
template class OmniDSP::backend::OneMKLCorrelationPlanImpl<float_c>;
template class OmniDSP::backend::OneMKLCorrelationPlanImpl<double_c>;

}  // namespace backend
}  // namespace OmniDSP
