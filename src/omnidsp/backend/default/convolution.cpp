/**
 * @file convolution.cpp (default)
 * @brief Implements Default backend ConvolutionPlanImpl and CorrelationPlanImpl
 * classes using standard C++ and FFT-based approach via DefaultFFTPlanImpl,
 * with Highway SIMD acceleration for frequency-domain multiplication and IFFT
 * scaling.
 * @details Provides portable implementations using FFT for
 * convolution/correlation, leveraging the Highway-accelerated Default FFT plans
 * and Highway for the element-wise complex multiplication and scaling steps.
 */

// Define HWY_TARGET_INCLUDE before including Highway headers
#ifndef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "src/omnidsp/backend/default/convolution.cpp"  // Path to this file
#endif

#include <algorithm>  // For std::reverse, std::copy, std::fill, std::max, std::min
#include <cmath>      // For std::abs, std::log2, std::ceil
#include <complex>
#include <iostream>  // For debug/error messages
#include <memory>    // For std::unique_ptr, std::make_unique
#include <numeric>   // For std::max, std::min
#include <span>
#include <stdexcept>    // For std::runtime_error, std::invalid_argument
#include <type_traits>  // For std::is_same_v
#include <vector>

#include "OmniDSP/core_types.h"  // For Status, ConvolutionMode etc.
#include "backend.h"  // Corresponding header for Default backend declarations
#include "hwy/contrib/complex/complex-inl.h"  // For complex operations
#include "hwy/contrib/math/math-inl.h"
#include "hwy/foreach_target.h"  // Must be first Highway include
#include "hwy/highway.h"

// Define complex types for brevity used internally
using float_c = OmniDSP::ComplexT<float>;
using double_c = OmniDSP::ComplexT<double>;

// Highway namespace alias within the compilation unit for the active target
HWY_BEFORE_NAMESPACE();
namespace OmniDSP {
namespace backend {
namespace HWY_NAMESPACE {  // Start Highway's target-specific namespace

namespace hn = hwy;  // Alias for Highway types/functions within this namespace

//--------------------------------------------------------------------------
// Highway-Accelerated Operations
//--------------------------------------------------------------------------

/**
 * @brief Performs element-wise complex multiplication using Highway SIMD.
 * @details Computes `out[i] = a[i] * b[i]` for complex vectors a and b.
 * This function lives inside the HWY_NAMESPACE.
 * @tparam T The underlying real floating-point type (float or double).
 * @param a_ptr Pointer to the first complex input array.
 * @param b_ptr Pointer to the second complex input array.
 * @param out_ptr Pointer to the complex output array.
 * @param size The number of complex elements in the arrays.
 */
template <typename T>
HWY_NOINLINE void ComplexMultiply_HWY(const std::complex<T>* HWY_RESTRICT a_ptr,
                                      const std::complex<T>* HWY_RESTRICT b_ptr,
                                      std::complex<T>* HWY_RESTRICT out_ptr,
                                      size_t size) {
  const hn::ScalableTag<T> d;
  using CplxV = hn::Vec2<decltype(d)>;  // Represents complex vector

  size_t i = 0;
  // Vectorized loop
  for (; i + hn::Lanes(d) <= size; i += hn::Lanes(d)) {
    const CplxV a_vec =
        hn::LoadInterleaved2(d, reinterpret_cast<const T*>(a_ptr + i));
    const CplxV b_vec =
        hn::LoadInterleaved2(d, reinterpret_cast<const T*>(b_ptr + i));
    const CplxV result_vec = hn::ComplexMul(a_vec, b_vec);
    hn::StoreInterleaved2(result_vec, d, reinterpret_cast<T*>(out_ptr + i));
  }
  // Scalar remainder loop
  for (; i < size; ++i) {
    out_ptr[i] = a_ptr[i] * b_ptr[i];
  }
}

/**
 * @brief Scales elements of an array by a scalar factor using Highway SIMD.
 * @details Computes `data[i] *= scale` in-place. Handles both real and complex
 * types. This function lives inside the HWY_NAMESPACE.
 * @tparam T Data type (float, double, std::complex<float>,
 * std::complex<double>).
 * @param data_ptr Pointer to the data array (modified in-place).
 * @param size The number of elements in the array.
 * @param scale The scalar factor to multiply by.
 */
template <typename T>
HWY_NOINLINE void ScaleArray_HWY(T* HWY_RESTRICT data_ptr, size_t size,
                                 typename Detail::UnderlyingReal<T> scale) {
  using RealT = typename Detail::UnderlyingReal<T>;
  const hn::ScalableTag<RealT> d;
  const auto vscale = hn::Set(d, scale);

  size_t i = 0;
  if constexpr (Detail::is_complex_v<T>) {
    using CplxV = hn::Vec2<decltype(d)>;
    // Vectorized loop for complex
    for (; i + hn::Lanes(d) <= size; i += hn::Lanes(d)) {
      CplxV data_vec =
          hn::LoadInterleaved2(d, reinterpret_cast<RealT*>(data_ptr + i));
      // Scale both real and imaginary parts by the real scalar
      data_vec.val[0] = hn::Mul(data_vec.val[0], vscale);  // Real part
      data_vec.val[1] = hn::Mul(data_vec.val[1], vscale);  // Imaginary part
      hn::StoreInterleaved2(data_vec, d,
                            reinterpret_cast<RealT*>(data_ptr + i));
    }
  } else {
    using V = hn::Vec<decltype(d)>;
    // Vectorized loop for real
    for (; i + hn::Lanes(d) <= size; i += hn::Lanes(d)) {
      V data_vec =
          hn::LoadU(d, data_ptr + i);  // Use LoadU for potentially unaligned
      data_vec = hn::Mul(data_vec, vscale);
      hn::StoreU(data_vec, d, data_ptr + i);  // Use StoreU
    }
  }

  // Scalar remainder loop
  for (; i < size; ++i) {
    data_ptr[i] *= scale;
  }
}

// Explicit instantiations for exported functions
void ComplexMultiply_F32_HWY(const float_c* a, const float_c* b, float_c* out,
                             size_t size) {
  ComplexMultiply_HWY<float>(a, b, out, size);
}
void ComplexMultiply_F64_HWY(const double_c* a, const double_c* b,
                             double_c* out, size_t size) {
  ComplexMultiply_HWY<double>(a, b, out, size);
}
void ScaleReal_F32_HWY(float* data, size_t size, float scale) {
  ScaleArray_HWY<float>(data, size, scale);
}
void ScaleReal_F64_HWY(double* data, size_t size, double scale) {
  ScaleArray_HWY<double>(data, size, scale);
}
void ScaleComplex_F32_HWY(float_c* data, size_t size, float scale) {
  ScaleArray_HWY<float_c>(data, size, scale);
}
void ScaleComplex_F64_HWY(double_c* data, size_t size, double scale) {
  ScaleArray_HWY<double_c>(data, size, scale);
}

}  // namespace HWY_NAMESPACE
}  // namespace backend
}  // namespace OmniDSP
HWY_AFTER_NAMESPACE();

//==========================================================================
// Exported Wrapper Functions & Dispatch Logic (Compiled Once)
//==========================================================================
/**
 * @brief This block is compiled only once, regardless of the number of Highway
 * targets.
 * @details Contains exported functions for dynamic dispatch and helpers.
 * Headers needed by this block must be included again within the #if HWY_ONCE
 * scope.
 */
#if HWY_ONCE

#include <complex>
#include <span>
#include <stdexcept>  // For Plan constructors
#include <vector>

#include "OmniDSP/core_types.h"
#include "backend.h"               // Include again for HWY_ONCE block
#include "hwy/dynamic_dispatch.h"  // Include again for HWY_ONCE block

namespace OmniDSP {
namespace backend {

// Define types for brevity
using float_c = OmniDSP::ComplexT<float>;
using double_c = OmniDSP::ComplexT<double>;

// --- Exported Wrappers for Complex Multiplication ---
void MultiplyComplex_F32_Export(const float_c* a, const float_c* b,
                                float_c* out, size_t size) {
  HWY_NAMESPACE::ComplexMultiply_F32_HWY(a, b, out, size);
}
HWY_EXPORT(MultiplyComplex_F32_Export);
void MultiplyComplex_F64_Export(const double_c* a, const double_c* b,
                                double_c* out, size_t size) {
  HWY_NAMESPACE::ComplexMultiply_F64_HWY(a, b, out, size);
}
HWY_EXPORT(MultiplyComplex_F64_Export);

// --- Exported Wrappers for Scaling ---
void ScaleReal_F32_Export(float* data, size_t size, float scale) {
  HWY_NAMESPACE::ScaleReal_F32_HWY(data, size, scale);
}
HWY_EXPORT(ScaleReal_F32_Export);
void ScaleReal_F64_Export(double* data, size_t size, double scale) {
  HWY_NAMESPACE::ScaleReal_F64_HWY(data, size, scale);
}
HWY_EXPORT(ScaleReal_F64_Export);
void ScaleComplex_F32_Export(float_c* data, size_t size, float scale) {
  HWY_NAMESPACE::ScaleComplex_F32_HWY(data, size, scale);
}
HWY_EXPORT(ScaleComplex_F32_Export);
void ScaleComplex_F64_Export(double_c* data, size_t size, double scale) {
  HWY_NAMESPACE::ScaleComplex_F64_HWY(data, size, scale);
}
HWY_EXPORT(ScaleComplex_F64_Export);

// --- Dispatcher Functions for Complex Multiplication ---
Status DispatchMultiplyComplex(std::span<const float_c> a,
                               std::span<const float_c> b,
                               std::span<float_c> out) {
  if (a.size() != b.size() || a.size() != out.size())
    return Status::SizeMismatch;
  if (a.empty()) return Status::Success;
  auto func = HWY_DYNAMIC_DISPATCH(MultiplyComplex_F32_Export);
  func(a.data(), b.data(), out.data(), a.size());
  return Status::Success;
}
Status DispatchMultiplyComplex(std::span<const double_c> a,
                               std::span<const double_c> b,
                               std::span<double_c> out) {
  if (a.size() != b.size() || a.size() != out.size())
    return Status::SizeMismatch;
  if (a.empty()) return Status::Success;
  auto func = HWY_DYNAMIC_DISPATCH(MultiplyComplex_F64_Export);
  func(a.data(), b.data(), out.data(), a.size());
  return Status::Success;
}

// --- Dispatcher Functions for Scaling ---
Status DispatchScale(std::span<float> data, float scale) {
  if (data.empty()) return Status::Success;
  auto func = HWY_DYNAMIC_DISPATCH(ScaleReal_F32_Export);
  func(data.data(), data.size(), scale);
  return Status::Success;
}
Status DispatchScale(std::span<double> data, double scale) {
  if (data.empty()) return Status::Success;
  auto func = HWY_DYNAMIC_DISPATCH(ScaleReal_F64_Export);
  func(data.data(), data.size(), scale);
  return Status::Success;
}
Status DispatchScale(std::span<float_c> data, float scale) {
  if (data.empty()) return Status::Success;
  auto func = HWY_DYNAMIC_DISPATCH(ScaleComplex_F32_Export);
  func(data.data(), data.size(), scale);
  return Status::Success;
}
Status DispatchScale(std::span<double_c> data, double scale) {
  if (data.empty()) return Status::Success;
  auto func = HWY_DYNAMIC_DISPATCH(ScaleComplex_F64_Export);
  func(data.data(), data.size(), scale);
  return Status::Success;
}

//--------------------------------------------------------------------------
// Helper Functions (Standard C++)
//--------------------------------------------------------------------------
inline size_t next_power_of_two(size_t n) {
  if (n == 0) return 1;
  if ((n > 0) && ((n & (n - 1)) == 0)) return n;
  if (n > (SIZE_MAX / 2)) return SIZE_MAX;
  size_t power =
      static_cast<size_t>(std::ceil(std::log2(static_cast<double>(n))));
  size_t result = static_cast<size_t>(1) << power;
  if (result == 0 && power > 0) return SIZE_MAX;
  return result;
}
namespace Detail { /* Type traits helpers... */
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
template <typename T>
using UnderlyingReal = typename ValueType<T>::type;
template <typename T>
using CorrespondingComplex = ComplexT<UnderlyingReal<T>>;
}  // namespace Detail

//--------------------------------------------------------------------------
// DefaultConvolutionPlanImpl Method Definitions (FFT-based)
//--------------------------------------------------------------------------
template <typename T>
DefaultConvolutionPlanImpl<T>::DefaultConvolutionPlanImpl(
    const std::vector<T>& kernel, ConvolutionMode mode)
    : mode_(mode), kernel_length_(kernel.size()) {
  // Constructor logic remains the same (create FFT plan, FFT kernel)
  if (kernel_length_ == 0)
    throw std::invalid_argument("Convolution kernel cannot be empty.");
  fft_length_ = next_power_of_two(kernel_length_ * 2);
  if (fft_length_ == SIZE_MAX)
    throw std::length_error("FFT length exceeds limits.");
  if constexpr (!Detail::is_complex_v<T>) {
    if (fft_length_ < 2 && fft_length_ != 0) fft_length_ = 2;
  }
  try {
    if constexpr (Detail::is_complex_v<T>)
      internal_fft_plan_ = std::make_unique<DefaultFFTPlanImpl<T>>(fft_length_);
    else
      internal_rfft_plan_ =
          std::make_unique<DefaultRFFTPlanImpl<T>>(fft_length_);
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed internal FFT plan creation: " +
                             std::string(e.what()));
  }
  std::vector<T> padded_kernel = kernel;
  std::reverse(padded_kernel.begin(), padded_kernel.end());
  padded_kernel.resize(fft_length_, T{});
  Status fft_status;
  using ComplexOutputType = Detail::CorrespondingComplex<T>;
  std::vector<ComplexOutputType> temp_fft_output;
  if constexpr (Detail::is_complex_v<T>) {
    kernel_fft_.resize(fft_length_);
    temp_fft_output.resize(fft_length_);
    fft_status = internal_fft_plan_->fft(padded_kernel, temp_fft_output);
  } else {
    kernel_fft_.resize(fft_length_ / 2 + 1);
    temp_fft_output.resize(fft_length_ / 2 + 1);
    fft_status = internal_rfft_plan_->rfft(padded_kernel, temp_fft_output);
  }
  if (fft_status != Status::Success)
    throw std::runtime_error("Failed FFT of kernel: " +
                             std::string(get_status_string(fft_status)));
  kernel_fft_ = std::move(temp_fft_output);
}
template <typename T>
DefaultConvolutionPlanImpl<T>::~DefaultConvolutionPlanImpl() = default;

template <typename T>
Status DefaultConvolutionPlanImpl<T>::execute(std::span<const T> input,
                                              std::span<T> output) const {
  if constexpr (Detail::is_complex_v<T>) {
    if (!internal_fft_plan_) return Status::InvalidOperation;
  } else {
    if (!internal_rfft_plan_) return Status::InvalidOperation;
  }

  size_t input_len = input.size();
  size_t output_len_expected = get_output_length(input_len);
  size_t full_conv_len = input_len + kernel_length_ - 1;

  if (output.size() < output_len_expected) return Status::SizeMismatch;
  std::fill(output.begin(), output.begin() + output_len_expected, T{});
  if (input_len == 0 || kernel_length_ == 0) return Status::Success;
  if (fft_length_ < full_conv_len) { /* Warning */
  }
  if (input_len > fft_length_) return Status::InvalidArgument;

  std::vector<T> input_padded(fft_length_, T{});
  std::copy(input.begin(), input.end(), input_padded.begin());

  using ComplexType = Detail::CorrespondingComplex<T>;
  std::vector<ComplexType> input_fft;
  std::vector<ComplexType> product_fft;
  std::vector<T> result_full_padded(fft_length_);

  Status status;
  if constexpr (Detail::is_complex_v<T>) {
    input_fft.resize(fft_length_);
    status = internal_fft_plan_->fft(input_padded, input_fft);
  } else {
    input_fft.resize(fft_length_ / 2 + 1);
    status = internal_rfft_plan_->rfft(input_padded, input_fft);
  }
  if (status != Status::Success) return status;

  // Multiply FFTs using Highway Dispatch
  product_fft.resize(input_fft.size());
  status = DispatchMultiplyComplex(std::span<const ComplexType>(input_fft),
                                   std::span<const ComplexType>(kernel_fft_),
                                   std::span<ComplexType>(product_fft));
  if (status != Status::Success) return status;

  // Inverse FFT
  if constexpr (Detail::is_complex_v<T>) {
    status = internal_fft_plan_->ifft(product_fft, result_full_padded);
  } else {
    status = internal_rfft_plan_->irfft(product_fft, result_full_padded);
  }
  if (status != Status::Success) return status;

  // Apply IFFT Scaling (1/N) using Highway Dispatch
  if (fft_length_ > 0) {
    using RealType = Detail::UnderlyingReal<T>;
    RealType scale =
        static_cast<RealType>(1.0) / static_cast<RealType>(fft_length_);
    // Dispatch based on whether result_full_padded is Real or Complex
    status = DispatchScale(std::span<T>(result_full_padded), scale);
    if (status != Status::Success) {
      std::cerr << "DefaultConvolutionPlanImpl::execute failed during Highway "
                   "scaling."
                << std::endl;
      return status;
    }
  }

  // Extract Correct Output based on Mode (Scalar logic)
  size_t full_start_idx = 0;
  size_t count = output_len_expected;
  switch (mode_) { /* Set full_start_idx based on mode */
    case ConvolutionMode::Full:
      full_start_idx = 0;
      break;
    case ConvolutionMode::Same:
      full_start_idx = (kernel_length_ - 1) / 2;
      break;
    case ConvolutionMode::Valid:
      full_start_idx = kernel_length_ - 1;
      break;
  }
  size_t copy_end_idx = full_start_idx + count;
  if (copy_end_idx > fft_length_) { /* Warning & adjust count */
    copy_end_idx = fft_length_;
    count = (full_start_idx < fft_length_) ? (fft_length_ - full_start_idx) : 0;
  }
  count = std::min({count, output.size(), output_len_expected});
  if (count > 0 && full_start_idx < fft_length_) {
    std::copy(result_full_padded.begin() + full_start_idx,
              result_full_padded.begin() + full_start_idx + count,
              output.begin());
  }
  if (output.size() > output_len_expected) {
    std::fill(output.begin() + output_len_expected, output.end(), T{});
  }
  return Status::Success;
}
template <typename T>
size_t DefaultConvolutionPlanImpl<T>::get_kernel_length() const {
  return kernel_length_;
}
template <typename T>
ConvolutionMode DefaultConvolutionPlanImpl<T>::get_mode() const {
  return mode_;
}
template <typename T>
size_t DefaultConvolutionPlanImpl<T>::get_output_length(
    size_t input_length) const {
  if (kernel_length_ == 0) return 0;
  switch (mode_) { /* Calculate length based on mode */
    case ConvolutionMode::Full:
      return (input_length > SIZE_MAX - kernel_length_ + 1)
                 ? SIZE_MAX
                 : input_length + kernel_length_ - 1;
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
// DefaultCorrelationPlanImpl Method Definitions (FFT-based)
//--------------------------------------------------------------------------
template <typename T>
DefaultCorrelationPlanImpl<T>::DefaultCorrelationPlanImpl(
    const std::vector<T>& kernel, ConvolutionMode mode)
    : mode_(mode), template_length_(kernel.size()) {
  // Constructor logic remains the same (create FFT plan, FFT template, store
  // conjugate)
  if (template_length_ == 0)
    throw std::invalid_argument("Correlation template cannot be empty.");
  fft_length_ = next_power_of_two(template_length_ * 2);
  if (fft_length_ == SIZE_MAX)
    throw std::length_error("FFT length exceeds limits.");
  if constexpr (!Detail::is_complex_v<T>) {
    if (fft_length_ < 2 && fft_length_ != 0) fft_length_ = 2;
  }
  try {
    if constexpr (Detail::is_complex_v<T>)
      internal_fft_plan_ = std::make_unique<DefaultFFTPlanImpl<T>>(fft_length_);
    else
      internal_rfft_plan_ =
          std::make_unique<DefaultRFFTPlanImpl<T>>(fft_length_);
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed internal FFT plan creation: " +
                             std::string(e.what()));
  }
  std::vector<T> padded_template = kernel;
  padded_template.resize(fft_length_, T{});
  Status fft_status;
  using ComplexOutputType = Detail::CorrespondingComplex<T>;
  std::vector<ComplexOutputType> temp_fft_output;
  if constexpr (Detail::is_complex_v<T>) {
    temp_fft_output.resize(fft_length_);
    fft_status = internal_fft_plan_->fft(padded_template, temp_fft_output);
  } else {
    temp_fft_output.resize(fft_length_ / 2 + 1);
    fft_status = internal_rfft_plan_->rfft(padded_template, temp_fft_output);
  }
  if (fft_status != Status::Success)
    throw std::runtime_error("Failed FFT of template: " +
                             std::string(get_status_string(fft_status)));
  template_fft_conj_.resize(temp_fft_output.size());
  for (size_t i = 0; i < temp_fft_output.size(); ++i)
    template_fft_conj_[i] = std::conj(temp_fft_output[i]);
}
template <typename T>
DefaultCorrelationPlanImpl<T>::~DefaultCorrelationPlanImpl() = default;

template <typename T>
Status DefaultCorrelationPlanImpl<T>::execute(std::span<const T> input,
                                              std::span<T> output) const {
  if constexpr (Detail::is_complex_v<T>) {
    if (!internal_fft_plan_) return Status::InvalidOperation;
  } else {
    if (!internal_rfft_plan_) return Status::InvalidOperation;
  }

  size_t input_len = input.size();
  size_t output_len_expected = get_output_length(input_len);
  size_t full_corr_len = input_len + template_length_ - 1;

  if (output.size() < output_len_expected) return Status::SizeMismatch;
  std::fill(output.begin(), output.begin() + output_len_expected, T{});
  if (input_len == 0 || template_length_ == 0) return Status::Success;
  if (fft_length_ < full_corr_len) { /* Warning */
  }
  if (input_len > fft_length_) return Status::InvalidArgument;

  std::vector<T> input_padded(fft_length_, T{});
  std::copy(input.begin(), input.end(), input_padded.begin());

  using ComplexType = Detail::CorrespondingComplex<T>;
  std::vector<ComplexType> input_fft;
  std::vector<ComplexType> product_fft;
  std::vector<T> result_full_padded(fft_length_);

  Status status;
  if constexpr (Detail::is_complex_v<T>) {
    input_fft.resize(fft_length_);
    status = internal_fft_plan_->fft(input_padded, input_fft);
  } else {
    input_fft.resize(fft_length_ / 2 + 1);
    status = internal_rfft_plan_->rfft(input_padded, input_fft);
  }
  if (status != Status::Success) return status;

  // Multiply FFTs using Highway Dispatch
  product_fft.resize(input_fft.size());
  status = DispatchMultiplyComplex(
      std::span<const ComplexType>(input_fft),
      std::span<const ComplexType>(template_fft_conj_),  // Use pre-conjugated
      std::span<ComplexType>(product_fft));
  if (status != Status::Success) return status;

  // Inverse FFT
  if constexpr (Detail::is_complex_v<T>) {
    status = internal_fft_plan_->ifft(product_fft, result_full_padded);
  } else {
    status = internal_rfft_plan_->irfft(product_fft, result_full_padded);
  }
  if (status != Status::Success) return status;

  // Apply IFFT Scaling (1/N) using Highway Dispatch
  if (fft_length_ > 0) {
    using RealType = Detail::UnderlyingReal<T>;
    RealType scale =
        static_cast<RealType>(1.0) / static_cast<RealType>(fft_length_);
    status = DispatchScale(std::span<T>(result_full_padded), scale);
    if (status != Status::Success) {
      std::cerr << "DefaultCorrelationPlanImpl::execute failed during Highway "
                   "scaling."
                << std::endl;
      return status;
    }
  }

  // Extract Correct Output based on Mode (Scalar logic)
  size_t full_start_idx = 0;
  size_t count = output_len_expected;
  switch (mode_) { /* Set full_start_idx based on mode */
    case ConvolutionMode::Full:
      full_start_idx = 0;
      break;
    case ConvolutionMode::Same:
      full_start_idx =
          (full_corr_len > input_len) ? (full_corr_len - input_len) / 2 : 0;
      break;
    case ConvolutionMode::Valid:
      full_start_idx = 0;
      break;
  }
  size_t copy_end_idx = full_start_idx + count;
  if (copy_end_idx > fft_length_) { /* Warning & adjust count */
    copy_end_idx = fft_length_;
    count = (full_start_idx < fft_length_) ? (fft_length_ - full_start_idx) : 0;
  }
  count = std::min({count, output.size(), output_len_expected});
  if (count > 0 && full_start_idx < fft_length_) {
    std::copy(result_full_padded.begin() + full_start_idx,
              result_full_padded.begin() + full_start_idx + count,
              output.begin());
  }
  if (output.size() > output_len_expected) {
    std::fill(output.begin() + output_len_expected, output.end(), T{});
  }
  return Status::Success;
}
template <typename T>
size_t DefaultCorrelationPlanImpl<T>::get_template_length() const {
  return template_length_;
}
template <typename T>
ConvolutionMode DefaultCorrelationPlanImpl<T>::get_mode() const {
  return mode_;
}
template <typename T>
size_t DefaultCorrelationPlanImpl<T>::get_output_length(
    size_t input_length) const {
  if (template_length_ == 0) return 0;
  switch (mode_) { /* Calculate length based on mode */
    case ConvolutionMode::Full:
      return (input_length > SIZE_MAX - template_length_ + 1)
                 ? SIZE_MAX
                 : input_length + template_length_ - 1;
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
template class OmniDSP::backend::DefaultConvolutionPlanImpl<float>;
template class OmniDSP::backend::DefaultConvolutionPlanImpl<double>;
template class OmniDSP::backend::DefaultConvolutionPlanImpl<float_c>;
template class OmniDSP::backend::DefaultConvolutionPlanImpl<double_c>;
template class OmniDSP::backend::DefaultCorrelationPlanImpl<float>;
template class OmniDSP::backend::DefaultCorrelationPlanImpl<double>;
template class OmniDSP::backend::DefaultCorrelationPlanImpl<float_c>;
template class OmniDSP::backend::DefaultCorrelationPlanImpl<double_c>;

}  // namespace backend
}  // namespace OmniDSP

#endif  // HWY_ONCE
