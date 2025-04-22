/**
 * @file convolution.cpp (accelerate)
 * @brief Implements Accelerate backend ConvolutionPlanImpl and
 * CorrelationPlanImpl classes, using internal
 * AccelerateFFTPlanImpl/AccelerateRFFTPlanImpl for FFT operations.
 */

// Only compile this file if Accelerate backend is enabled via CMake
#ifdef USE_ACCELERATE

#include <Accelerate/Accelerate.h>

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

#include "OmniDSP/core_types.h"  // For Status, ConvolutionMode etc.
#include "backend.h"             // Accelerate backend declarations

namespace OmniDSP {
namespace backend {

// Helper from fft.cpp (or move to a common utility header)
inline bool is_power_of_two(size_t n) {
  return (n > 0) && ((n & (n - 1)) == 0);
}
// Helper to calculate next power of two
inline size_t next_power_of_two(size_t n) {
  if (n == 0) return 1;
  // Handle potential overflow if n is already close to max size_t power of 2
  size_t result = 1;
  while (result < n) {
    result <<= 1;
    if (result == 0) return size_t(-1);  // Indicate overflow or too large
  }
  return result;
  // Alternative using bit manipulation (often faster):
  // if (n == 0) return 1;
  // n--;
  // n |= n >> 1;
  // n |= n >> 2;
  // n |= n >> 4;
  // n |= n >> 8;
  // n |= n >> 16;
  // #if SIZE_MAX > 0xFFFFFFFF // If size_t is 64-bit
  // n |= n >> 32;
  // #endif
  // n++;
  // return n;
}

// Forward declare the concrete FFT plan impls needed internally
// These should be defined in accelerate/fft.cpp (or a header included by it)
template <typename T_Complex>
class AccelerateFFTPlanImpl;  // Expects complex type
template <typename T_Real>
class AccelerateRFFTPlanImpl;  // Expects real type

// Helper type traits (can be moved to a common utility header)
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
// AccelerateConvolutionPlanImpl Method Definitions
//--------------------------------------------------------------------------

template <typename T>
AccelerateConvolutionPlanImpl<T>::AccelerateConvolutionPlanImpl(
    const std::vector<T>& kernel, ConvolutionMode mode)
    : mode_(mode), kernel_length_(kernel.size()) {
  if (kernel_length_ == 0) {
    throw std::invalid_argument("Convolution kernel cannot be empty.");
  }

  // --- Strategy: FFT-based Convolution using internal FFT Plan ---
  // Determine FFT length. A common choice for linear convolution is N+M-1,
  // padded to the next power of two for FFT efficiency.
  // For a plan, we might need a fixed block size or max input length.
  // Placeholder: Choose FFT length based on kernel size * 2 (common for
  // overlap-add/save blocks)
  fft_length_ = next_power_of_two(kernel_length_ * 2);
  if (fft_length_ == size_t(-1)) {  // Check for overflow from next_power_of_two
    throw std::length_error(
        "Calculated FFT length for convolution plan exceeds limits.");
  }

  // --- Create internal FFT plan ---
  try {
    if constexpr (Detail::is_complex_v<T>) {  // T is std::complex<Real>
      internal_fft_plan_ =
          std::make_unique<AccelerateFFTPlanImpl<T>>(fft_length_);
    } else {  // T is Real (float or double)
      internal_rfft_plan_ =
          std::make_unique<AccelerateRFFTPlanImpl<T>>(fft_length_);
    }
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "Failed to create internal FFT plan for ConvolutionPlan: " +
        std::string(e.what()));
  } catch (...) {
    throw std::runtime_error(
        "Unknown error creating internal FFT plan for ConvolutionPlan.");
  }

  // --- Prepare and FFT the kernel using the internal plan ---
  std::vector<T> padded_kernel = kernel;
  std::reverse(padded_kernel.begin(),
               padded_kernel.end());       // Reverse for convolution
  padded_kernel.resize(fft_length_, T{});  // Zero-pad

  Status fft_status;
  if constexpr (Detail::is_complex_v<T>) {  // Complex Kernel -> Complex FFT
    kernel_fft_.resize(fft_length_);
    fft_status = internal_fft_plan_->fft(padded_kernel, kernel_fft_);
  } else {                        // Real Kernel -> Real FFT
    using Complex = ComplexT<T>;  // Get the corresponding complex type
    kernel_fft_.resize(fft_length_ / 2 + 1);
    fft_status = internal_rfft_plan_->rfft(padded_kernel, kernel_fft_);
  }

  if (fft_status != Status::Success) {
    // Clean up internal plan if FFT failed? unique_ptr handles it if exception
    // propagates.
    throw std::runtime_error(
        "Failed to compute FFT of kernel during ConvolutionPlan creation. "
        "Status: " +
        std::string(get_status_string(fft_status)));
  }

  std::cout << "Accelerate ConvPlanImpl created. Kernel Len: " << kernel_length_
            << ", FFT Len: " << fft_length_ << std::endl;  // Debug
}

template <typename T>
AccelerateConvolutionPlanImpl<T>::~AccelerateConvolutionPlanImpl() {
  // unique_ptr members (internal_fft_plan_ / internal_rfft_plan_) handle
  // cleanup automatically
  std::cout << "Accelerate ConvPlanImpl destroyed." << std::endl;  // Debug
}

template <typename T>
Status AccelerateConvolutionPlanImpl<T>::execute(std::span<const T> input,
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
  // This skeleton assumes the input fits within the plan's FFT length,
  // or that overlap-add/save is handled externally or internally.
  // A production implementation would need OLA/OLS here if input_len >
  // block_size.
  size_t required_fft_len_for_linear = next_power_of_two(full_output_len);
  if (current_fft_len < required_fft_len_for_linear) {
    std::cerr << "Warning: Convolution plan's internal FFT length ("
              << current_fft_len
              << ") is too small for full linear convolution with input size ("
              << input_len
              << "). Result will be circular. Overlap-Add/Save not implemented "
                 "in skeleton."
              << std::endl;
    // Depending on requirements, could return error or proceed with circular
    // result. Let's proceed for now, but this highlights limitation of fixed
    // FFT length in plan.
  }
  if (input_len > current_fft_len) {
    std::cerr << "Error: Input size (" << input_len
              << ") exceeds convolution plan's internal FFT length ("
              << current_fft_len
              << "). Overlap-Add/Save not implemented in skeleton."
              << std::endl;
    return Status::InvalidArgument;
  }

  // --- Allocate temporary buffers ---
  std::vector<T> input_padded(current_fft_len, T{});
  // Copy input, padding with zeros
  std::copy(input.begin(), input.end(), input_padded.begin());

  // Buffers for FFT results (always complex)
  using Complex =
      ComplexT<typename Detail::ValueType<T>::type>;  // Get underlying complex
                                                      // type
  std::vector<Complex> input_fft(
      current_fft_len);  // Allocate full size for simplicity, adjust if RFFT
  std::vector<Complex> product_fft(current_fft_len);
  std::vector<T> result_full_padded(current_fft_len);  // Output of IFFT

  // --- Perform FFT on input using internal plan ---
  Status status;
  if constexpr (Detail::is_complex_v<T>) {
    input_fft.resize(current_fft_len);  // Ensure correct size
    status = internal_fft_plan_->fft(input_padded, input_fft);
  } else {                                      // Real Input
    input_fft.resize(current_fft_len / 2 + 1);  // RFFT output size
    status = internal_rfft_plan_->rfft(input_padded, input_fft);
  }
  if (status != Status::Success) {
    std::cerr << "Convolution execute failed during input FFT. Status: "
              << get_status_string(status) << std::endl;
    return status;
  }

  // --- Multiply FFTs (Input * Kernel) ---
  product_fft.resize(input_fft.size());  // Match size of input_fft (N or N/2+1)
  // Determine underlying value type (float or double) for vDSP calls
  using Value = typename Detail::ValueType<T>::type;
  if constexpr (std::is_same_v<Value, float>) {
    vDSP_zvmul(reinterpret_cast<const DSPComplex*>(input_fft.data()), 1,
               reinterpret_cast<const DSPComplex*>(kernel_fft_.data()), 1,
               reinterpret_cast<DSPComplex*>(product_fft.data()), 1,
               product_fft.size(), 1);
  } else {  // double
    vDSP_zvmulD(reinterpret_cast<const DSPDoubleComplex*>(input_fft.data()), 1,
                reinterpret_cast<const DSPDoubleComplex*>(kernel_fft_.data()),
                1, reinterpret_cast<DSPDoubleComplex*>(product_fft.data()), 1,
                product_fft.size(), 1);
  }

  // --- Inverse FFT using internal plan ---
  result_full_padded.resize(current_fft_len);  // Ensure correct size
  if constexpr (Detail::is_complex_v<T>) {
    status = internal_fft_plan_->ifft(product_fft, result_full_padded);
  } else {  // Real Result
    status = internal_rfft_plan_->irfft(product_fft, result_full_padded);
  }
  if (status != Status::Success) {
    std::cerr << "Convolution execute failed during inverse FFT. Status: "
              << get_status_string(status) << std::endl;
    return status;
  }

  // --- Extract Correct Output based on Mode ---
  size_t output_len = get_output_length(input_len);
  if (output.size() < output_len) {
    std::cerr << "Convolution execute error: Output buffer size ("
              << output.size() << ") is smaller than required (" << output_len
              << ")." << std::endl;
    return Status::SizeMismatch;
  }

  // Check if the calculated full output length fits within the IFFT result
  // buffer
  if (full_output_len > result_full_padded.size()) {
    std::cerr << "Convolution execute internal error: Full output length "
                 "exceeds FFT buffer size."
              << std::endl;
    return Status::Failure;  // Should not happen if FFT length is sufficient
  }

  switch (mode_) {
    case ConvolutionMode::Full:
      std::copy(result_full_padded.begin(),
                result_full_padded.begin() + full_output_len, output.begin());
      break;
    case ConvolutionMode::Same: {
      size_t start = (kernel_length_ - 1) / 2;
      // Check bounds carefully
      if (start + input_len >
          full_output_len) {  // Check against actual linear conv length
        std::cerr << "Convolution execute internal error: 'Same' mode "
                     "calculation error."
                  << std::endl;
        return Status::Failure;
      }
      std::copy(result_full_padded.begin() + start,
                result_full_padded.begin() + start + input_len, output.begin());
    } break;
    case ConvolutionMode::Valid: {
      size_t start = kernel_length_ - 1;
      size_t valid_len =
          (input_len >= kernel_length_) ? (input_len - kernel_length_ + 1) : 0;
      if (valid_len >
          output_len) {  // Check against expected output len for mode
        std::cerr << "Convolution execute internal error: 'Valid' mode "
                     "calculation error."
                  << std::endl;
        return Status::Failure;
      }
      // Check bounds against actual linear conv length
      if (start + valid_len > full_output_len) {
        std::cerr
            << "Convolution execute internal error: 'Valid' mode bounds error."
            << std::endl;
        return Status::Failure;
      }
      if (valid_len > 0) {
        std::copy(result_full_padded.begin() + start,
                  result_full_padded.begin() + start + valid_len,
                  output.begin());
      }
    } break;
    default:  // Should not happen with enum class
      std::cerr << "Convolution execute error: Unknown convolution mode."
                << std::endl;
      return Status::InvalidArgument;
  }

  // Zero out remaining output buffer if user provided a larger one
  if (output.size() > output_len) {
    std::fill(output.begin() + output_len, output.end(), T{});
  }

  return Status::Success;
}

template <typename T>
size_t AccelerateConvolutionPlanImpl<T>::get_kernel_length() const {
  return kernel_length_;
}

template <typename T>
ConvolutionMode AccelerateConvolutionPlanImpl<T>::get_mode() const {
  return mode_;
}

template <typename T>
size_t AccelerateConvolutionPlanImpl<T>::get_output_length(
    size_t input_length) const {
  if (kernel_length_ == 0) return 0;  // Should be caught by constructor
  switch (mode_) {
    case ConvolutionMode::Full:
      // Need to handle potential overflow if input_length is huge
      if (input_length > SIZE_MAX - kernel_length_ + 1) return SIZE_MAX;
      return input_length + kernel_length_ - 1;
    case ConvolutionMode::Same:
      return input_length;
    case ConvolutionMode::Valid:
      return (input_length >= kernel_length_)
                 ? (input_length - kernel_length_ + 1)
                 : 0;
    default:  // Should not happen
      return 0;
  }
}

//--------------------------------------------------------------------------
// AccelerateCorrelationPlanImpl Method Definitions
//--------------------------------------------------------------------------

template <typename T>
AccelerateCorrelationPlanImpl<T>::AccelerateCorrelationPlanImpl(
    const std::vector<T>& kernel, ConvolutionMode mode)
    : mode_(mode), template_length_(kernel.size()) {
  if (template_length_ == 0) {
    throw std::invalid_argument("Correlation template cannot be empty.");
  }
  // Similar setup as Convolution, using internal FFT plan
  fft_length_ =
      next_power_of_two(template_length_ * 2);  // Placeholder strategy
  if (fft_length_ == size_t(-1)) {
    throw std::length_error(
        "Calculated FFT length for correlation plan exceeds limits.");
  }

  // --- Create internal FFT plan ---
  try {
    if constexpr (Detail::is_complex_v<T>) {
      internal_fft_plan_ =
          std::make_unique<AccelerateFFTPlanImpl<T>>(fft_length_);
    } else {
      internal_rfft_plan_ =
          std::make_unique<AccelerateRFFTPlanImpl<T>>(fft_length_);
    }
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "Failed to create internal FFT plan for CorrelationPlan: " +
        std::string(e.what()));
  } catch (...) {
    throw std::runtime_error(
        "Unknown error creating internal FFT plan for CorrelationPlan.");
  }

  // --- Prepare and FFT the template (NO reversal) ---
  std::vector<T> padded_template = kernel;
  padded_template.resize(fft_length_, T{});  // Zero-pad

  Status fft_status;
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

  // --- CONJUGATE the result and store in template_fft_ ---
  template_fft_ = temp_template_fft;  // Copy
  using Value = typename Detail::ValueType<T>::type;
  if constexpr (std::is_same_v<Value, float>) {
    vDSP_zvconj(reinterpret_cast<const DSPComplex*>(template_fft_.data()), 1,
                reinterpret_cast<DSPComplex*>(template_fft_.data()), 1,
                template_fft_.size());
  } else {  // double
    vDSP_zvconjD(
        reinterpret_cast<const DSPDoubleComplex*>(template_fft_.data()), 1,
        reinterpret_cast<DSPDoubleComplex*>(template_fft_.data()), 1,
        template_fft_.size());
  }

  std::cout << "Accelerate CorrPlanImpl created. Template Len: "
            << template_length_ << ", FFT Len: " << fft_length_
            << std::endl;  // Debug
}

template <typename T>
AccelerateCorrelationPlanImpl<T>::~AccelerateCorrelationPlanImpl() {
  // unique_ptr members handle cleanup
  std::cout << "Accelerate CorrPlanImpl destroyed." << std::endl;  // Debug
}

template <typename T>
Status AccelerateCorrelationPlanImpl<T>::execute(std::span<const T> input,
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
  if (status != Status::Success) {
    std::cerr << "Correlation execute failed during input FFT. Status: "
              << get_status_string(status) << std::endl;
    return status;
  }

  // --- Multiply FFTs (Input * Conjugated(Template)) ---
  product_fft.resize(input_fft.size());
  using Value = typename Detail::ValueType<T>::type;
  if constexpr (std::is_same_v<Value, float>) {
    vDSP_zvmul(reinterpret_cast<const DSPComplex*>(input_fft.data()), 1,
               reinterpret_cast<const DSPComplex*>(template_fft_.data()), 1,
               reinterpret_cast<DSPComplex*>(product_fft.data()), 1,
               product_fft.size(), 1);
  } else {  // double
    vDSP_zvmulD(reinterpret_cast<const DSPDoubleComplex*>(input_fft.data()), 1,
                reinterpret_cast<const DSPDoubleComplex*>(template_fft_.data()),
                1, reinterpret_cast<DSPDoubleComplex*>(product_fft.data()), 1,
                product_fft.size(), 1);
  }

  // --- Inverse FFT ---
  result_full_padded.resize(current_fft_len);
  if constexpr (Detail::is_complex_v<T>) {
    status = internal_fft_plan_->ifft(product_fft, result_full_padded);
  } else {
    status = internal_rfft_plan_->irfft(product_fft, result_full_padded);
  }
  if (status != Status::Success) {
    std::cerr << "Correlation execute failed during inverse FFT. Status: "
              << get_status_string(status) << std::endl;
    return status;
  }

  // --- Extract Correct Output based on Mode ---
  size_t output_len = get_output_length(input_len);
  if (output.size() < output_len) {
    std::cerr << "Correlation execute error: Output buffer size ("
              << output.size() << ") is smaller than required (" << output_len
              << ")." << std::endl;
    return Status::SizeMismatch;
  }

  // Check if the calculated full output length fits within the IFFT result
  // buffer
  if (full_output_len > result_full_padded.size()) {
    std::cerr << "Correlation execute internal error: Full output length "
                 "exceeds FFT buffer size."
              << std::endl;
    return Status::Failure;  // Should not happen if FFT length is sufficient
  }

  // Correlation output indices differ slightly from convolution for 'same' and
  // 'valid'
  switch (mode_) {
    case ConvolutionMode::Full:
      std::copy(result_full_padded.begin(),
                result_full_padded.begin() + full_output_len, output.begin());
      break;
    case ConvolutionMode::Same: {
      // Center the output around the input length
      size_t start =
          (full_output_len > input_len) ? (full_output_len - input_len) / 2 : 0;
      size_t count =
          std::min(input_len, full_output_len);  // Number of elements to copy
      if (start + count > full_output_len) {     // Bounds check
        std::cerr
            << "Correlation execute internal error: 'Same' mode bounds error."
            << std::endl;
        return Status::Failure;
      }
      std::copy(result_full_padded.begin() + start,
                result_full_padded.begin() + start + count, output.begin());
    } break;
    case ConvolutionMode::Valid: {
      // Valid part starts at index 0 for correlation
      size_t start = 0;
      size_t valid_len = (input_len >= template_length_)
                             ? (input_len - template_length_ + 1)
                             : 0;
      if (valid_len > output_len) {
        std::cerr << "Correlation execute internal error: 'Valid' mode "
                     "calculation error."
                  << std::endl;
        return Status::Failure;
      }
      if (start + valid_len > full_output_len) {  // Check bounds
        std::cerr
            << "Correlation execute internal error: 'Valid' mode bounds error."
            << std::endl;
        return Status::Failure;
      }
      if (valid_len > 0) {
        std::copy(result_full_padded.begin() + start,
                  result_full_padded.begin() + start + valid_len,
                  output.begin());
      }
    } break;
    default:
      std::cerr << "Correlation execute error: Unknown convolution mode."
                << std::endl;
      return Status::InvalidArgument;
  }

  if (output.size() > output_len) {
    std::fill(output.begin() + output_len, output.end(), T{});
  }

  return Status::Success;
}

template <typename T>
size_t AccelerateCorrelationPlanImpl<T>::get_template_length() const {
  return template_length_;
}

template <typename T>
ConvolutionMode AccelerateCorrelationPlanImpl<T>::get_mode() const {
  return mode_;
}

template <typename T>
size_t AccelerateCorrelationPlanImpl<T>::get_output_length(
    size_t input_length) const {
  // Same logic as convolution output length
  if (template_length_ == 0) return 0;  // Should be caught by constructor
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

// AccelerateConvolutionPlanImpl Instantiations
template class OmniDSP::backend::AccelerateConvolutionPlanImpl<float>;
template class OmniDSP::backend::AccelerateConvolutionPlanImpl<double>;
template class OmniDSP::backend::AccelerateConvolutionPlanImpl<float_c>;
template class OmniDSP::backend::AccelerateConvolutionPlanImpl<double_c>;

// AccelerateCorrelationPlanImpl Instantiations
template class OmniDSP::backend::AccelerateCorrelationPlanImpl<float>;
template class OmniDSP::backend::AccelerateCorrelationPlanImpl<double>;
template class OmniDSP::backend::AccelerateCorrelationPlanImpl<float_c>;
template class OmniDSP::backend::AccelerateCorrelationPlanImpl<double_c>;

}  // namespace backend
}  // namespace OmniDSP

#endif  // USE_ACCELERATE
