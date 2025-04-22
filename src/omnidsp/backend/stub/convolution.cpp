/**
 * @file convolution.cpp (stub)
 * @brief Implements Stub backend ConvolutionPlanImpl and CorrelationPlanImpl
 * classes using standard C++.
 */

#include <algorithm>  // For std::reverse, std::copy, std::fill
#include <cmath>      // For std::abs
#include <complex>
#include <iostream>  // For debug/error messages
#include <numeric>   // For std::max, std::min
#include <span>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "OmniDSP/core_types.h"  // For Status, ConvolutionMode etc.
#include "backend.h"             // Stub backend declarations

namespace OmniDSP {
namespace backend {

//--------------------------------------------------------------------------
// StubConvolutionPlanImpl Method Definitions
//--------------------------------------------------------------------------

template <typename T>
StubConvolutionPlanImpl<T>::StubConvolutionPlanImpl(
    const std::vector<T>& kernel, ConvolutionMode mode)
    : mode_(mode), kernel_length_(kernel.size()), reversed_kernel_(kernel) {
  if (kernel_length_ == 0) {
    throw std::invalid_argument("Convolution kernel cannot be empty.");
  }
  // Pre-reverse the kernel for easier direct convolution calculation
  std::reverse(reversed_kernel_.begin(), reversed_kernel_.end());
  std::cout << "Stub ConvPlanImpl created. Kernel Len: " << kernel_length_
            << std::endl;  // Debug
}

template <typename T>
Status StubConvolutionPlanImpl<T>::execute(std::span<const T> input,
                                           std::span<T> output) const {
  size_t input_len = input.size();
  size_t output_len_expected = get_output_length(input_len);

  if (output.size() < output_len_expected) {
    return Status::SizeMismatch;
  }
  if (input_len == 0 || kernel_length_ == 0) {
    std::fill(output.begin(), output.begin() + output_len_expected, T{});
    return Status::Success;  // Nothing to do
  }

  // Direct convolution implementation: y[n] = sum(x[k] * h[n-k])
  // Since kernel h is pre-reversed (h_rev[k] = h[M-1-k]),
  // y[n] = sum(x[k] * h_rev[k-n+M-1]) - this seems complex.
  // Alternative: y[n] = sum(x[n-k] * h[k])
  // Let h be the original kernel (not reversed).
  // Let h_rev be the reversed kernel stored in reversed_kernel_.
  // y[n] = sum_k x[k] * h[n-k] = sum_j x[n-j] * h[j]
  // Using reversed kernel: y[n] = sum_k x[k] * h_rev[M-1-(n-k)] = sum_k x[k] *
  // h_rev[k-n+M-1]

  // Let's implement sum_j x[n-j] * h[j] using the *original* kernel orientation
  // for clarity, even though we stored the reversed one.
  // We can access the original kernel via reversed_kernel_[M-1-j].
  std::vector<T> kernel = reversed_kernel_;  // Get a copy
  std::reverse(kernel.begin(),
               kernel.end());  // Reverse it back to original orientation h[j]

  size_t M = kernel_length_;
  size_t N = input_len;
  size_t full_len = N + M - 1;

  // Determine start and end indices for output based on mode
  size_t out_start_idx = 0;
  size_t out_end_idx = 0;  // Exclusive end index

  switch (mode_) {
    case ConvolutionMode::Full:
      out_start_idx = 0;
      out_end_idx = full_len;
      break;
    case ConvolutionMode::Same:
      out_start_idx = (M - 1) / 2;
      out_end_idx = out_start_idx + N;
      break;
    case ConvolutionMode::Valid:
      out_start_idx = M - 1;
      out_end_idx = N;  // N - M + 1 outputs, last index is N-1
      break;
    default:
      return Status::InvalidArgument;  // Should not happen
  }

  // Calculate convolution for the required output range
  for (size_t n_out = 0; n_out < output_len_expected; ++n_out) {
    size_t n = n_out + out_start_idx;  // Index in the 'full' convolution result
    T sum{};  // Initialize sum to zero for the current output sample

    // Summation: y[n] = sum_j x[n-j] * h[j]
    size_t j_start = (n >= N) ? (n - (N - 1)) : 0;  // Max(0, n - N + 1)
    size_t j_end = (n < M) ? (n + 1) : M;           // Min(M, n + 1)

    for (size_t j = j_start; j < j_end; ++j) {
      if (n >= j) {                       // Ensure input index n-j is valid
        sum += input[n - j] * kernel[j];  // Use original kernel h[j]
      }
    }
    output[n_out] = sum;
  }

  // Zero out remaining output buffer if user provided a larger one
  if (output.size() > output_len_expected) {
    std::fill(output.begin() + output_len_expected, output.end(), T{});
  }

  return Status::Success;
}

template <typename T>
size_t StubConvolutionPlanImpl<T>::get_kernel_length() const {
  return kernel_length_;
}

template <typename T>
ConvolutionMode StubConvolutionPlanImpl<T>::get_mode() const {
  return mode_;
}

template <typename T>
size_t StubConvolutionPlanImpl<T>::get_output_length(
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
// StubCorrelationPlanImpl Method Definitions
//--------------------------------------------------------------------------

template <typename T>
StubCorrelationPlanImpl<T>::StubCorrelationPlanImpl(
    const std::vector<T>& kernel, ConvolutionMode mode)
    : mode_(mode),
      template_length_(kernel.size()),
      template_(kernel)  // Store template directly
{
  if (template_length_ == 0) {
    throw std::invalid_argument("Correlation template cannot be empty.");
  }
  std::cout << "Stub CorrPlanImpl created. Template Len: " << template_length_
            << std::endl;  // Debug
}

template <typename T>
Status StubCorrelationPlanImpl<T>::execute(std::span<const T> input,
                                           std::span<T> output) const {
  size_t input_len = input.size();
  size_t output_len_expected = get_output_length(input_len);

  if (output.size() < output_len_expected) {
    return Status::SizeMismatch;
  }
  if (input_len == 0 || template_length_ == 0) {
    std::fill(output.begin(), output.begin() + output_len_expected, T{});
    return Status::Success;  // Nothing to do
  }

  size_t M = template_length_;
  size_t N = input_len;
  size_t full_len = N + M - 1;

  // Determine start and end indices for output based on mode
  // Correlation indices relative to 'full' result differ from convolution
  size_t full_start_idx =
      0;  // Index corresponding to output[0] in the full result
  size_t out_count = output_len_expected;

  switch (mode_) {
    case ConvolutionMode::Full:
      full_start_idx = 0;
      break;
    case ConvolutionMode::Same:
      // Center alignment for correlation
      full_start_idx = (full_len > N) ? (full_len - N) / 2 : 0;
      break;
    case ConvolutionMode::Valid:
      // Valid correlation starts at index 0 of full result
      full_start_idx = 0;
      break;
    default:
      return Status::InvalidArgument;
  }

  // Direct correlation implementation: y[n] = sum_k x[n+k] * conj(h[k])
  // Index 'n' here ranges over the output indices relative to the 'full'
  // result. The actual output buffer index is n_out = n - full_start_idx.
  for (size_t n_out = 0; n_out < out_count; ++n_out) {
    size_t n =
        n_out + full_start_idx;  // Index in the 'full' correlation result space
    T sum{};

    // Sum over k where both x[n+k] and h[k] are valid
    size_t k_start = 0;
    size_t k_end = M;  // Loop over template length

    for (size_t k = k_start; k < k_end; ++k) {
      // Calculate index into input signal: idx = n + k - (M - 1)
      // This aligns the start of the correlation properly.
      // Example: n=0 (first output of 'full'), k=0 -> idx = 0 + 0 - M + 1 = 1-M
      // (invalid) Example: n=M-1 (first output of 'valid'), k=0 -> idx =
      // M-1+0-M+1 = 0 (valid start) Example: n=M-1, k=M-1 -> idx = M-1+M-1-M+1
      // = M-1 (valid end for kernel) Example: n=N-1 (last index for 'same'),
      // k=0 -> idx = N-1+0-M+1 = N-M (valid if N>=M) Example: n=N-1, k=M-1 ->
      // idx = N-1+M-1-M+1 = N-1 (valid end) Example: n=N+M-2 (last output of
      // 'full'), k=M-1 -> idx = N+M-2+M-1-M+1 = N+M-2 (invalid)

      // Let's try a different indexing: y[m] = sum_k x[k] * conj(h[k-m])
      // where m ranges appropriately.

      // Easier: y[n_out] = sum_k x[k] * conj(h[k - n]) where n is lag
      // Let n_lag = n - (M - 1) range from -(M-1) to N-1 for full correlation
      long long n_lag =
          static_cast<long long>(n) - static_cast<long long>(M - 1);

      for (size_t k = 0; k < N; ++k) {  // Iterate over input signal x[k]
        long long h_idx = static_cast<long long>(k) - n_lag;
        if (h_idx >= 0 && static_cast<size_t>(h_idx) <
                              M) {  // Check if template index is valid
          if constexpr (Detail::is_complex_v<T>) {
            sum += input[k] * std::conj(template_[h_idx]);
          } else {
            sum += input[k] * template_[h_idx];  // Conjugate is no-op for real
          }
        }
      }
    }
    output[n_out] = sum;
  }

  // Zero out remaining output buffer if user provided a larger one
  if (output.size() > output_len_expected) {
    std::fill(output.begin() + output_len_expected, output.end(), T{});
  }

  return Status::Success;
}

template <typename T>
size_t StubCorrelationPlanImpl<T>::get_template_length() const {
  return template_length_;
}

template <typename T>
ConvolutionMode StubCorrelationPlanImpl<T>::get_mode() const {
  return mode_;
}

template <typename T>
size_t StubCorrelationPlanImpl<T>::get_output_length(
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

// StubConvolutionPlanImpl Instantiations
template class OmniDSP::backend::StubConvolutionPlanImpl<float>;
template class OmniDSP::backend::StubConvolutionPlanImpl<double>;
template class OmniDSP::backend::StubConvolutionPlanImpl<float_c>;
template class OmniDSP::backend::StubConvolutionPlanImpl<double_c>;

// StubCorrelationPlanImpl Instantiations
template class OmniDSP::backend::StubCorrelationPlanImpl<float>;
template class OmniDSP::backend::StubCorrelationPlanImpl<double>;
template class OmniDSP::backend::StubCorrelationPlanImpl<float_c>;
template class OmniDSP::backend::StubCorrelationPlanImpl<double_c>;

}  // namespace backend
}  // namespace OmniDSP
