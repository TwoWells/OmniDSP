#include "convolution.hpp"  // Plan class declarations for this backend

// Include necessary standard library headers
#include <algorithm>  // For std::reverse, std::copy, std::fill, std::max, std::min
#include <cassert>  // For assert
#include <cmath>    // For std::log2, std::ceil, std::abs, std::conj
#include <complex>
#include <cstddef>  // For size_t
#include <cstring>  // For std::memcpy
#include <limits>   // For std::numeric_limits
#include <memory>   // For std::unique_ptr, std::make_unique
#include <span>     // For std::span
#include <stdexcept>  // For std::runtime_error, std::invalid_argument, std::length_error
#include <string>  // For std::string in exceptions
#include <type_traits>  // For std::is_same_v, std::is_base_of_v, std::decay_t, std::remove_pointer_t
#include <variant>  // Include for std::visit, std::holds_alternative, std::monostate, std::get_if
#include <vector>

// Include OmniDSP headers
#include <OmniDSP/convolution.hpp>  // For ConvolutionType, ConvolutionMethod
#include <OmniDSP/core_types.hpp>   // For Status, F32, C32, etc., Utils::*

#include "../interface/backend.hpp"  // For AbstractBackend and FFTPlanImpl base classes (FFTPlanImpl, RFFTPlanImpl)

namespace OmniDSP::Default {

  //--------------------------------------------------------------------------
  // Helper Functions (Standard C++) - Keep from original file
  //--------------------------------------------------------------------------
  namespace convolution_detail {  // Internal namespace

    // Calculates the next power of two >= n.
    // Returns 0 if n is 0 or if the next power of two would overflow size_t.
    inline size_t next_power_of_two(size_t n)
    {
      if (n == 0) return 0;  // Return 0 for input 0
      // Check if n is already a power of 2
      if ((n > 0) && ((n & (n - 1)) == 0)) {
        return n;
      }
      // Check for potential overflow before calculation
      // If n is already >= half of max size_t, the next power of 2 might
      // overflow
      if (n > (std::numeric_limits<size_t>::max() / 2U)) {
        // Check if n itself is the max power of 2 representable
        if (n == (~(std::numeric_limits<size_t>::max() >> 1))) return n;
        return 0;  // Indicate overflow otherwise
      }
      // Efficient bit manipulation way to find next power of 2
      size_t power = 1;
      while (power < n) {
        // Check for overflow during shift
        if (power > (std::numeric_limits<size_t>::max() / 2U)) return 0;
        power <<= 1;
      }
      return power;
    }

    // Helper for element-wise complex multiplication (Standard C++)
    template <typename T_Complex>
    void complex_multiply(
        std::span<const T_Complex> a,  // Expects const span
        std::span<const T_Complex> b,  // Expects const span
        std::span<T_Complex> out)      // Expects non-const span
    {
      assert(
          a.size() == b.size() && a.size() == out.size()
          && "Span sizes must match for complex_multiply");
      size_t n = a.size();
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
      }
    }

    // Helper for element-wise complex multiplication with conjugate (Standard
    // C++) out = a * conj(b)
    template <typename T_Complex>
    void complex_multiply_conj(
        std::span<const T_Complex> a,  // Expects const span
        std::span<const T_Complex> b,  // Expects const span
        std::span<T_Complex> out)      // Expects non-const span
    {
      assert(
          a.size() == b.size() && a.size() == out.size()
          && "Span sizes must match for complex_multiply_conj");
      size_t n = a.size();
      for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * std::conj(b[i]);
      }
    }

  }  // namespace convolution_detail

  //--------------------------------------------------------------------------
  // ConvolutionPlanImpl Method Definitions (FFT-based, Standard C++)
  //--------------------------------------------------------------------------

  template <typename T>  // T can be F32, F64, C32, C64
  ConvolutionPlanImpl<T>::ConvolutionPlanImpl(
      FFTPlanImplVariant&& fft_plan_variant,  // Accept the FFT plan variant
      const std::vector<T>& kernel,
      ConvolutionType type,
      ConvolutionMethod method)
      : type_(type),
        method_(method),
        original_kernel_(kernel),
        kernel_length_(kernel.size()),
        fft_length_(0),  // Will be set below
        fft_plan_impl_variant_(
            std::move(fft_plan_variant))  // Store the passed-in plan
  {
    if (kernel_length_ == 0) {
      throw std::invalid_argument("Convolution kernel cannot be empty.");
    }

    // Define complex type
    using T_Complex = Utils::GetComplexType<T>;
    // Define real type
    using T_Real = Utils::GetRealType<T>;
    // Define the specific plan pointer types we expect in the variant
    using FFTPlanPtr = std::unique_ptr<Abstract::FFTPlanImpl<T_Complex>>;
    using RFFTPlanPtr = std::unique_ptr<Abstract::RFFTPlanImpl<T_Real>>;

    Status fft_status = Status::Failure;
    size_t kernel_fft_size = 0;

    // Prepare the kernel for FFT: Pad and Reverse
    std::vector<T> padded_kernel = original_kernel_;
    std::reverse(
        padded_kernel.begin(), padded_kernel.end());  // Reverse for convolution

    // --- Use if constexpr to separate logic for Real and Complex T ---
    if constexpr (Utils::IsComplex_v<T>) {
      // --- T is Complex (C32, C64) ---
      // Expect FFTPlanPtr in the variant
      const auto* fft_plan_ptr_ptr
          = std::get_if<FFTPlanPtr>(&fft_plan_impl_variant_);
      if (!fft_plan_ptr_ptr || !*fft_plan_ptr_ptr) {
        throw std::invalid_argument(
            "Incorrect FFT plan type provided for complex convolution "
            "(expected FFTPlanImpl).");
      }
      auto& plan_impl_ptr = *fft_plan_ptr_ptr;  // Get the unique_ptr

      fft_length_ = plan_impl_ptr->get_length();
      if (fft_length_ < kernel_length_) {
        throw std::invalid_argument(
            "Provided FFT plan length is too small for the kernel.");
      }

      padded_kernel.resize(fft_length_, T{});  // Pad with zeros

      kernel_fft_size = fft_length_;
      kernel_fft_.resize(kernel_fft_size);
      const T_Complex* kernel_input_ptr = reinterpret_cast<const T_Complex*>(
          padded_kernel.data());  // T is T_Complex
      std::span<const T_Complex> kernel_input_span(
          kernel_input_ptr, fft_length_);
      T_Complex* kernel_output_ptr = kernel_fft_.data();
      std::span<T_Complex> kernel_output_span(
          kernel_output_ptr, kernel_fft_size);
      fft_status = plan_impl_ptr->fft(kernel_input_span, kernel_output_span);
    }
    else {
      // --- T is Real (F32, F64) ---
      // Expect RFFTPlanPtr in the variant
      const auto* rfft_plan_ptr_ptr
          = std::get_if<RFFTPlanPtr>(&fft_plan_impl_variant_);
      if (!rfft_plan_ptr_ptr || !*rfft_plan_ptr_ptr) {
        throw std::invalid_argument(
            "Incorrect FFT plan type provided for real convolution (expected "
            "RFFTPlanImpl).");
      }
      auto& plan_impl_ptr = *rfft_plan_ptr_ptr;  // Get the unique_ptr

      fft_length_ = plan_impl_ptr->get_length();
      if (fft_length_ < 2) {
        throw std::runtime_error("Internal RFFT plan length must be >= 2.");
      }
      if (fft_length_ < kernel_length_) {
        throw std::invalid_argument(
            "Provided FFT plan length is too small for the kernel.");
      }

      padded_kernel.resize(fft_length_, T{});  // Pad with zeros

      kernel_fft_size = fft_length_ / 2 + 1;
      kernel_fft_.resize(kernel_fft_size);
      const T_Real* kernel_input_ptr = padded_kernel.data();  // T is T_Real
      std::span<const T_Real> kernel_input_span(kernel_input_ptr, fft_length_);
      T_Complex* kernel_output_ptr = kernel_fft_.data();
      std::span<T_Complex> kernel_output_span(
          kernel_output_ptr, kernel_fft_size);
      fft_status = plan_impl_ptr->rfft(kernel_input_span, kernel_output_span);
    }

    if (fft_status != Status::Success) {
      throw std::runtime_error(
          "Failed FFT of convolution kernel. Status: "
          + std::to_string(static_cast<int>(fft_status)));
    }

    // Allocate temporary buffers used during execute
    try {
      input_padded_ = std::make_unique<T[]>(fft_length_);
      input_fft_ = std::make_unique<T_Complex[]>(
          kernel_fft_size);  // Use calculated kernel_fft_size
      product_fft_ = std::make_unique<T_Complex[]>(
          kernel_fft_size);  // Use calculated kernel_fft_size
      result_ifft_ = std::make_unique<T[]>(fft_length_);
    }
    catch (const std::bad_alloc& e) {
      throw std::runtime_error(
          "Failed to allocate temporary buffers for convolution plan: "
          + std::string(e.what()));
    }
    if (!input_padded_ || !input_fft_ || !product_fft_ || !result_ifft_) {
      throw std::runtime_error(
          "Failed to allocate temporary buffers for convolution plan "
          "(unexpected).");
    }
  }

  // Execute method
  template <typename T>
  Status ConvolutionPlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output) const
  {
    const size_t signal_len = input.size();
    const size_t output_len_expected = get_output_length(signal_len);

    // --- Input Validation ---
    if (output.size() < output_len_expected) {
      return Status::SizeMismatch;
    }

    // Define complex/real types and plan pointer types
    using T_Complex = Utils::GetComplexType<T>;
    using T_Real = Utils::GetRealType<T>;
    using FFTPlanPtr = std::unique_ptr<Abstract::FFTPlanImpl<T_Complex>>;
    using RFFTPlanPtr = std::unique_ptr<Abstract::RFFTPlanImpl<T_Real>>;

    // Check for empty input/kernel early
    if (signal_len == 0 || kernel_length_ == 0) {
      std::fill(output.begin(), output.begin() + output_len_expected, T{});
      if (output.size() > output_len_expected) {
        std::fill(output.begin() + output_len_expected, output.end(), T{});
      }
      return Status::Success;
    }

    // --- FFT Length Check ---
    size_t required_fft_len = signal_len + kernel_length_ - 1;
    if (fft_length_ < required_fft_len) {
      return Status::InvalidArgument;  // Plan too small for this input
    }

    // --- Prepare Buffers ---
    std::memcpy(input_padded_.get(), input.data(), signal_len * sizeof(T));
    std::fill(
        input_padded_.get() + signal_len,
        input_padded_.get() + fft_length_,
        T{});

    size_t input_fft_size = kernel_fft_.size();
    T_Complex* input_fft_ptr = input_fft_.get();
    std::span<T_Complex> input_fft_span(input_fft_ptr, input_fft_size);
    T_Complex* product_fft_ptr = product_fft_.get();
    std::span<T_Complex> product_fft_span(product_fft_ptr, kernel_fft_.size());
    T* result_ifft_ptr = result_ifft_.get();
    std::span<T> result_ifft_span(result_ifft_ptr, fft_length_);

    Status status = Status::Failure;

    // --- Use if constexpr to separate logic for Real and Complex T ---
    if constexpr (Utils::IsComplex_v<T>) {
      // --- T is Complex ---
      // Retrieve the expected FFTPlanPtr
      const auto* fft_plan_ptr_ptr
          = std::get_if<FFTPlanPtr>(&fft_plan_impl_variant_);
      if (!fft_plan_ptr_ptr || !*fft_plan_ptr_ptr)
        return Status::InvalidOperation;  // Should not happen if constructor
                                          // succeeded
      auto& plan_impl_ptr = *fft_plan_ptr_ptr;

      // 1. FFT Input
      const T_Complex* input_padded_ptr
          = reinterpret_cast<const T_Complex*>(input_padded_.get());
      std::span<const T_Complex> input_padded_complex_span(
          input_padded_ptr, fft_length_);
      status = plan_impl_ptr->fft(input_padded_complex_span, input_fft_span);
      if (status != Status::Success) {
        return status;
      }

      // 2. Multiply FFTs
      const T_Complex* kernel_fft_ptr = kernel_fft_.data();
      std::span<const T_Complex> kernel_fft_const_span(
          kernel_fft_ptr, kernel_fft_.size());
      convolution_detail::complex_multiply(
          std::span<const T_Complex>(input_fft_span),
          kernel_fft_const_span,
          product_fft_span);

      // 3. IFFT Result
      const T_Complex* product_ptr = product_fft_.get();
      std::span<const T_Complex> product_const_span(
          product_ptr, product_fft_span.size());
      T_Complex* result_ptr
          = reinterpret_cast<T_Complex*>(result_ifft_ptr);  // T is T_Complex
      std::span<T_Complex> result_complex_span(result_ptr, fft_length_);
      status = plan_impl_ptr->ifft(product_const_span, result_complex_span);
      if (status != Status::Success) {
        return status;
      }
    }
    else {
      // --- T is Real ---
      // Retrieve the expected RFFTPlanPtr
      const auto* rfft_plan_ptr_ptr
          = std::get_if<RFFTPlanPtr>(&fft_plan_impl_variant_);
      if (!rfft_plan_ptr_ptr || !*rfft_plan_ptr_ptr)
        return Status::InvalidOperation;  // Should not happen if constructor
                                          // succeeded
      auto& plan_impl_ptr = *rfft_plan_ptr_ptr;

      // 1. RFFT Input
      const T_Real* input_padded_ptr = input_padded_.get();  // T is T_Real
      std::span<const T_Real> input_padded_real_span(
          input_padded_ptr, fft_length_);
      status = plan_impl_ptr->rfft(input_padded_real_span, input_fft_span);
      if (status != Status::Success) {
        return status;
      }

      // 2. Multiply FFTs
      const T_Complex* kernel_fft_ptr = kernel_fft_.data();
      std::span<const T_Complex> kernel_fft_const_span(
          kernel_fft_ptr, kernel_fft_.size());
      convolution_detail::complex_multiply(
          std::span<const T_Complex>(input_fft_span),
          kernel_fft_const_span,
          product_fft_span);

      // 3. IRFFT Result
      const T_Complex* product_ptr = product_fft_.get();
      std::span<const T_Complex> product_const_span(
          product_ptr, product_fft_span.size());
      // result_ifft_span is already std::span<T> where T is T_Real
      status = plan_impl_ptr->irfft(product_const_span, result_ifft_span);
      if (status != Status::Success) {
        return status;
      }
    }

    // --- Extract Correct Output based on Type ---
    // (Extraction logic remains the same)
    size_t full_start_idx = 0;
    size_t count = output_len_expected;

    switch (type_) {
      case ConvolutionType::Full:
        full_start_idx = 0;
        count = signal_len + kernel_length_ - 1;
        break;
      case ConvolutionType::Same:
        full_start_idx = (kernel_length_ - 1) / 2;
        count = signal_len;
        break;
      case ConvolutionType::Valid:
        full_start_idx = kernel_length_ - 1;
        count = (signal_len >= kernel_length_)
                    ? (signal_len - kernel_length_ + 1)
                    : 0;
        break;
      default:
        return Status::InvalidArgument;
    }

    assert(count == output_len_expected);

    size_t linear_conv_len = signal_len + kernel_length_ - 1;
    if (full_start_idx >= linear_conv_len) {
      count = 0;
    }
    else if (full_start_idx + count > linear_conv_len) {
      count = linear_conv_len - full_start_idx;
    }

    count = std::min(count, output.size());

    if (count > 0) {
      if (full_start_idx + count <= fft_length_) {
        std::memcpy(
            output.data(),
            result_ifft_.get() + full_start_idx,
            count * sizeof(T));
      }
      else {
        size_t safe_count = (full_start_idx < fft_length_)
                                ? (fft_length_ - full_start_idx)
                                : 0;
        safe_count = std::min(safe_count, count);
        if (safe_count > 0) {
          std::memcpy(
              output.data(),
              result_ifft_.get() + full_start_idx,
              safe_count * sizeof(T));
        }
        if (output_len_expected > safe_count) {
          std::fill(
              output.begin() + safe_count,
              output.begin() + output_len_expected,
              T{});
        }
      }
    }

    if (count < output_len_expected) {
      std::fill(
          output.begin() + count, output.begin() + output_len_expected, T{});
    }
    if (output.size() > output_len_expected) {
      std::fill(output.begin() + output_len_expected, output.end(), T{});
    }

    return Status::Success;
  }

  // --- Getters (remain the same) ---
  template <typename T>
  size_t ConvolutionPlanImpl<T>::get_kernel_length() const
  {
    return kernel_length_;
  }
  template <typename T>
  ConvolutionType ConvolutionPlanImpl<T>::get_type() const
  {
    return type_;
  }
  template <typename T>
  ConvolutionMethod ConvolutionPlanImpl<T>::get_method() const
  {
    return method_;
  }
  template <typename T>
  size_t ConvolutionPlanImpl<T>::get_output_length(size_t input_length) const
  {
    if (kernel_length_ == 0)
      return (
          type_ == ConvolutionType::Same ? input_length
                                         : 0);  // Handle zero kernel length
    switch (type_) {
      case ConvolutionType::Full:
        // Check for potential overflow before adding
        if (input_length
            > std::numeric_limits<size_t>::max() - kernel_length_ + 1) {
          throw std::length_error(
              "Output length calculation overflow (Full mode)");
        }
        return input_length + kernel_length_ - 1;
      case ConvolutionType::Same:
        return input_length;
      case ConvolutionType::Valid:
        return (input_length >= kernel_length_)
                   ? (input_length - kernel_length_ + 1)
                   : 0;
      default:
        throw std::logic_error(
            "Invalid ConvolutionType encountered in get_output_length");  // Should
                                                                          // not
                                                                          // happen
    }
  }
  template <typename T>
  std::span<const T> ConvolutionPlanImpl<T>::get_kernel() const
  {
    // Return span of the original, non-padded, non-reversed kernel
    return std::span<const T>(original_kernel_);
  }

  //--------------------------------------------------------------------------
  // CorrelationPlanImpl Method Definitions (FFT-based, Standard C++)
  //--------------------------------------------------------------------------

  template <typename T>  // T can be F32, F64, C32, C64
  CorrelationPlanImpl<T>::CorrelationPlanImpl(
      FFTPlanImplVariant&& fft_plan_variant,  // Accept the FFT plan variant
      const std::vector<T>& kernel,  // This is the 'template' for correlation
      ConvolutionType
          type,  // Using ConvolutionType enum for mode (Full, Same, Valid)
      ConvolutionMethod method)  // Using ConvolutionMethod enum
      : type_(type),
        method_(method),
        original_kernel_(kernel),       // Store the original template
        kernel_length_(kernel.size()),  // Length of the template
        fft_length_(0),                 // Will be set below
        fft_plan_impl_variant_(
            std::move(fft_plan_variant))  // Store the passed-in plan
  {
    if (kernel_length_ == 0) {
      throw std::invalid_argument(
          "Correlation kernel (template) cannot be empty.");
    }

    // Define complex/real types and plan pointer types
    using T_Complex = Utils::GetComplexType<T>;
    using T_Real = Utils::GetRealType<T>;
    using FFTPlanPtr = std::unique_ptr<Abstract::FFTPlanImpl<T_Complex>>;
    using RFFTPlanPtr = std::unique_ptr<Abstract::RFFTPlanImpl<T_Real>>;

    Status fft_status = Status::Failure;
    size_t kernel_fft_size = 0;
    std::vector<T_Complex>
        temp_kernel_fft;  // Temp buffer for FFT result before conjugation

    // Prepare the kernel for FFT: Pad (NO reverse for correlation)
    std::vector<T> padded_kernel = original_kernel_;

    // --- Use if constexpr to separate logic for Real and Complex T ---
    if constexpr (Utils::IsComplex_v<T>) {
      // --- T is Complex ---
      // Expect FFTPlanPtr
      const auto* fft_plan_ptr_ptr
          = std::get_if<FFTPlanPtr>(&fft_plan_impl_variant_);
      if (!fft_plan_ptr_ptr || !*fft_plan_ptr_ptr) {
        throw std::invalid_argument(
            "Incorrect FFT plan type provided for complex correlation "
            "(expected FFTPlanImpl).");
      }
      auto& plan_impl_ptr = *fft_plan_ptr_ptr;

      fft_length_ = plan_impl_ptr->get_length();
      if (fft_length_ < kernel_length_) {
        throw std::invalid_argument(
            "Provided FFT plan length is too small for the template.");
      }

      padded_kernel.resize(fft_length_, T{});  // Pad

      kernel_fft_size = fft_length_;
      temp_kernel_fft.resize(kernel_fft_size);
      const T_Complex* kernel_input_ptr
          = reinterpret_cast<const T_Complex*>(padded_kernel.data());
      std::span<const T_Complex> kernel_input_span(
          kernel_input_ptr, fft_length_);
      T_Complex* kernel_output_ptr = temp_kernel_fft.data();
      std::span<T_Complex> kernel_output_span(
          kernel_output_ptr, kernel_fft_size);
      fft_status = plan_impl_ptr->fft(kernel_input_span, kernel_output_span);
    }
    else {
      // --- T is Real ---
      // Expect RFFTPlanPtr
      const auto* rfft_plan_ptr_ptr
          = std::get_if<RFFTPlanPtr>(&fft_plan_impl_variant_);
      if (!rfft_plan_ptr_ptr || !*rfft_plan_ptr_ptr) {
        throw std::invalid_argument(
            "Incorrect FFT plan type provided for real correlation (expected "
            "RFFTPlanImpl).");
      }
      auto& plan_impl_ptr = *rfft_plan_ptr_ptr;

      fft_length_ = plan_impl_ptr->get_length();
      if (fft_length_ < 2) {
        throw std::runtime_error(
            "Internal RFFT plan length must be >= 2 for correlation.");
      }
      if (fft_length_ < kernel_length_) {
        throw std::invalid_argument(
            "Provided FFT plan length is too small for the template.");
      }

      padded_kernel.resize(fft_length_, T{});  // Pad

      kernel_fft_size = fft_length_ / 2 + 1;
      temp_kernel_fft.resize(kernel_fft_size);
      const T_Real* kernel_input_ptr = padded_kernel.data();
      std::span<const T_Real> kernel_input_span(kernel_input_ptr, fft_length_);
      T_Complex* kernel_output_ptr = temp_kernel_fft.data();
      std::span<T_Complex> kernel_output_span(
          kernel_output_ptr, kernel_fft_size);
      fft_status = plan_impl_ptr->rfft(kernel_input_span, kernel_output_span);
    }

    if (fft_status != Status::Success) {
      throw std::runtime_error(
          "Failed FFT of correlation kernel (template). Status: "
          + std::to_string(static_cast<int>(fft_status)));
    }

    // Store the CONJUGATE of the kernel's FFT for correlation
    kernel_fft_conj_.resize(kernel_fft_size);
    for (size_t i = 0; i < kernel_fft_size; ++i) {
      kernel_fft_conj_[i] = std::conj(temp_kernel_fft[i]);
    }

    // Allocate temporary buffers
    try {
      input_padded_ = std::make_unique<T[]>(fft_length_);
      input_fft_ = std::make_unique<T_Complex[]>(
          kernel_fft_size);  // Use kernel_fft_size
      product_fft_ = std::make_unique<T_Complex[]>(
          kernel_fft_size);  // Use kernel_fft_size
      result_ifft_ = std::make_unique<T[]>(fft_length_);
    }
    catch (const std::bad_alloc& e) {
      throw std::runtime_error(
          "Failed to allocate temporary buffers for correlation plan: "
          + std::string(e.what()));
    }
    if (!input_padded_ || !input_fft_ || !product_fft_ || !result_ifft_) {
      throw std::runtime_error(
          "Failed to allocate temporary buffers for correlation plan "
          "(unexpected).");
    }
  }

  // Execute method
  template <typename T>
  Status CorrelationPlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output) const
  {
    const size_t signal_len = input.size();
    const size_t output_len_expected = get_output_length(signal_len);

    // --- Input Validation ---
    if (output.size() < output_len_expected) {
      return Status::SizeMismatch;
    }

    // Define complex/real types and plan pointer types
    using T_Complex = Utils::GetComplexType<T>;
    using T_Real = Utils::GetRealType<T>;
    using FFTPlanPtr = std::unique_ptr<Abstract::FFTPlanImpl<T_Complex>>;
    using RFFTPlanPtr = std::unique_ptr<Abstract::RFFTPlanImpl<T_Real>>;

    // Check for empty input/kernel early
    if (signal_len == 0 || kernel_length_ == 0) {
      std::fill(output.begin(), output.begin() + output_len_expected, T{});
      if (output.size() > output_len_expected) {
        std::fill(output.begin() + output_len_expected, output.end(), T{});
      }
      return Status::Success;
    }

    // --- FFT Length Check ---
    size_t required_fft_len = signal_len + kernel_length_ - 1;
    if (fft_length_ < required_fft_len) {
      return Status::InvalidArgument;
    }  // Plan too small

    // --- Prepare Buffers ---
    std::memcpy(input_padded_.get(), input.data(), signal_len * sizeof(T));
    std::fill(
        input_padded_.get() + signal_len,
        input_padded_.get() + fft_length_,
        T{});

    size_t input_fft_size = kernel_fft_conj_.size();
    T_Complex* input_fft_ptr = input_fft_.get();
    std::span<T_Complex> input_fft_span(input_fft_ptr, input_fft_size);
    T_Complex* product_fft_ptr = product_fft_.get();
    std::span<T_Complex> product_fft_span(
        product_fft_ptr, kernel_fft_conj_.size());
    T* result_ifft_ptr = result_ifft_.get();
    std::span<T> result_ifft_span(result_ifft_ptr, fft_length_);

    Status status = Status::Failure;

    // --- Use if constexpr to separate logic for Real and Complex T ---
    if constexpr (Utils::IsComplex_v<T>) {
      // --- T is Complex ---
      // Retrieve the expected FFTPlanPtr
      const auto* fft_plan_ptr_ptr
          = std::get_if<FFTPlanPtr>(&fft_plan_impl_variant_);
      if (!fft_plan_ptr_ptr || !*fft_plan_ptr_ptr)
        return Status::InvalidOperation;
      auto& plan_impl_ptr = *fft_plan_ptr_ptr;

      // 1. FFT Input
      const T_Complex* input_padded_ptr
          = reinterpret_cast<const T_Complex*>(input_padded_.get());
      std::span<const T_Complex> input_padded_complex_span(
          input_padded_ptr, fft_length_);
      status = plan_impl_ptr->fft(input_padded_complex_span, input_fft_span);
      if (status != Status::Success) {
        return status;
      }

      // 2. Multiply FFTs
      const T_Complex* kernel_fft_conj_ptr = kernel_fft_conj_.data();
      std::span<const T_Complex> kernel_fft_conj_span(
          kernel_fft_conj_ptr, kernel_fft_conj_.size());
      convolution_detail::complex_multiply(
          std::span<const T_Complex>(input_fft_span),
          kernel_fft_conj_span,
          product_fft_span);

      // 3. IFFT Result
      const T_Complex* product_ptr = product_fft_.get();
      std::span<const T_Complex> product_const_span(
          product_ptr, product_fft_span.size());
      T_Complex* result_ptr = reinterpret_cast<T_Complex*>(result_ifft_ptr);
      std::span<T_Complex> result_complex_span(result_ptr, fft_length_);
      status = plan_impl_ptr->ifft(product_const_span, result_complex_span);
      if (status != Status::Success) {
        return status;
      }
    }
    else {
      // --- T is Real ---
      // Retrieve the expected RFFTPlanPtr
      const auto* rfft_plan_ptr_ptr
          = std::get_if<RFFTPlanPtr>(&fft_plan_impl_variant_);
      if (!rfft_plan_ptr_ptr || !*rfft_plan_ptr_ptr)
        return Status::InvalidOperation;
      auto& plan_impl_ptr = *rfft_plan_ptr_ptr;

      // 1. RFFT Input
      const T_Real* input_padded_ptr = input_padded_.get();
      std::span<const T_Real> input_padded_real_span(
          input_padded_ptr, fft_length_);
      status = plan_impl_ptr->rfft(input_padded_real_span, input_fft_span);
      if (status != Status::Success) {
        return status;
      }

      // 2. Multiply FFTs
      const T_Complex* kernel_fft_conj_ptr = kernel_fft_conj_.data();
      std::span<const T_Complex> kernel_fft_conj_span(
          kernel_fft_conj_ptr, kernel_fft_conj_.size());
      convolution_detail::complex_multiply(
          std::span<const T_Complex>(input_fft_span),
          kernel_fft_conj_span,
          product_fft_span);

      // 3. IRFFT Result
      const T_Complex* product_ptr = product_fft_.get();
      std::span<const T_Complex> product_const_span(
          product_ptr, product_fft_span.size());
      // result_ifft_span is already std::span<T> where T is T_Real
      status = plan_impl_ptr->irfft(product_const_span, result_ifft_span);
      if (status != Status::Success) {
        return status;
      }
    }

    // --- Extract Correct Output based on Type ---
    // (Extraction logic remains the same)
    size_t full_start_idx = 0;
    size_t count = output_len_expected;

    switch (type_) {
      case ConvolutionType::Full:
        full_start_idx = 0;
        count = signal_len + kernel_length_ - 1;
        break;
      case ConvolutionType::Same:
        full_start_idx = (kernel_length_ - 1) / 2;
        count = signal_len;
        break;
      case ConvolutionType::Valid:
        full_start_idx
            = 0;  // Note: Valid correlation starts at index 0 of full result
        count = (signal_len >= kernel_length_)
                    ? (signal_len - kernel_length_ + 1)
                    : 0;
        break;
      default:
        return Status::InvalidArgument;
    }

    assert(count == output_len_expected);

    size_t linear_corr_len = signal_len + kernel_length_ - 1;
    if (full_start_idx >= linear_corr_len) {
      count = 0;
    }
    else if (full_start_idx + count > linear_corr_len) {
      count = linear_corr_len - full_start_idx;
    }

    count = std::min(count, output.size());

    if (count > 0) {
      if (full_start_idx + count <= fft_length_) {
        std::memcpy(
            output.data(),
            result_ifft_.get() + full_start_idx,
            count * sizeof(T));
      }
      else {
        size_t safe_count = (full_start_idx < fft_length_)
                                ? (fft_length_ - full_start_idx)
                                : 0;
        safe_count = std::min(safe_count, count);
        if (safe_count > 0) {
          std::memcpy(
              output.data(),
              result_ifft_.get() + full_start_idx,
              safe_count * sizeof(T));
        }
        if (output_len_expected > safe_count) {
          std::fill(
              output.begin() + safe_count,
              output.begin() + output_len_expected,
              T{});
        }
      }
    }

    if (count < output_len_expected) {
      std::fill(
          output.begin() + count, output.begin() + output_len_expected, T{});
    }
    if (output.size() > output_len_expected) {
      std::fill(output.begin() + output_len_expected, output.end(), T{});
    }

    return Status::Success;
  }

  // --- Getters (remain the same) ---
  template <typename T>
  size_t CorrelationPlanImpl<T>::get_template_length() const
  {
    return kernel_length_;
  }
  template <typename T>
  ConvolutionType CorrelationPlanImpl<T>::get_type() const
  {
    return type_;
  }
  template <typename T>
  ConvolutionMethod CorrelationPlanImpl<T>::get_method() const
  {
    return method_;
  }
  template <typename T>
  size_t CorrelationPlanImpl<T>::get_output_length(size_t input_length) const
  {
    // Same logic as convolution for output length calculation
    if (kernel_length_ == 0)
      return (type_ == ConvolutionType::Same ? input_length : 0);
    switch (type_) {
      case ConvolutionType::Full:
        if (input_length
            > std::numeric_limits<size_t>::max() - kernel_length_ + 1) {
          throw std::length_error(
              "Output length calculation overflow (Full mode)");
        }
        return input_length + kernel_length_ - 1;
      case ConvolutionType::Same:
        return input_length;
      case ConvolutionType::Valid:
        return (input_length >= kernel_length_)
                   ? (input_length - kernel_length_ + 1)
                   : 0;
      default:
        throw std::logic_error(
            "Invalid ConvolutionType encountered in get_output_length");
    }
  }
  template <typename T>
  std::span<const T> CorrelationPlanImpl<T>::get_template() const
  {
    // Return span of the original, non-padded template
    return std::span<const T>(original_kernel_);
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  // Instantiate the plan implementation classes for supported types
  template class ConvolutionPlanImpl<F32>;
  template class ConvolutionPlanImpl<F64>;
  template class ConvolutionPlanImpl<C32>;
  template class ConvolutionPlanImpl<C64>;

  template class CorrelationPlanImpl<F32>;
  template class CorrelationPlanImpl<F64>;
  template class CorrelationPlanImpl<C32>;
  template class CorrelationPlanImpl<C64>;

}  // namespace OmniDSP::Default
