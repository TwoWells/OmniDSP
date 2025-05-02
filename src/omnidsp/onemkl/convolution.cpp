/**
 * @file convolution.cpp (onemkl)
 * @brief Implements oneMKL backend ConvolutionPlanImpl and CorrelationPlanImpl
 * classes, using internal oneMKL FFT plans (DFTI) for FFT operations.
 */

#include "convolution.hpp"  // Include the corresponding header file

// Include MKL header for VML and DFTI status codes/helpers
#include <mkl.h>

#include <algorithm>  // For std::reverse, std::copy, std::min
#include <cassert>    // For assert
#include <cmath>      // For log2, ceil, etc. (used by next_power_of_two)
#include <complex>
#include <cstring>   // For std::memcpy
#include <iostream>  // For debug/error messages
#include <limits>    // For std::numeric_limits
#include <memory>    // For std::unique_ptr, std::make_unique
#include <numeric>   // For std::max, std::min (if needed)
#include <span>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument, std::length_error
#include <type_traits>  // For std::is_same_v
#include <variant>      // For std::variant
#include <vector>

// Include necessary OmniDSP headers
#include <OmniDSP/convolution.hpp>
#include <OmniDSP/core_types.hpp>

#include "../interface/backend.hpp"  // For AbstractBackend
#include "fft.hpp"  // Include header for OneMKLFFTPlanImpl, OneMKLRFFTPlanImpl

// *** ADDED: Include the new utility header ***
#include "utils.hpp"

namespace OmniDSP::backend {

  // *** REMOVED mkl_status_to_omnidsp_status helper (now in utils.hpp) ***

  // Helper function to calculate next power of two
  namespace convolution_detail {
    inline size_t next_power_of_two(size_t n)
    {
      // ... (implementation) ...
      if (n == 0)
        return 1;  // Return 1 for 0 input? Or 0? FFT usually needs >=1
      if ((n > 0) && ((n & (n - 1)) == 0)) {
        return n;
      }
      if (n > (std::numeric_limits<size_t>::max() / 2U)) {
        if (n == (~(std::numeric_limits<size_t>::max() >> 1))) return n;
        return 0;
      }
      size_t power = 1;
      while (power < n) {
        if (power > (std::numeric_limits<size_t>::max() / 2U)) return 0;
        power <<= 1;
      }
      return power;
    }
  }  // namespace convolution_detail

  //--------------------------------------------------------------------------
  // OneMKLConvolutionPlanImpl Method Definitions
  //--------------------------------------------------------------------------

  template <typename T>
  OneMKLConvolutionPlanImpl<T>::OneMKLConvolutionPlanImpl(
      const AbstractBackend* owner,
      const std::vector<T>& kernel,
      ConvolutionType type,
      ConvolutionMethod method)
      : type_(type),
        method_(method),
        original_kernel_(kernel),
        kernel_length_(kernel.size()),
        fft_length_(0),
        mkl_status_(DFTI_NO_ERROR)
  {
    if (!owner) {
      throw std::invalid_argument("OneMKLConvolutionPlanImpl requires owner.");
    }
    if (kernel_length_ == 0) {
      throw std::invalid_argument("Convolution kernel empty.");
    }

    // Determine required FFT length
    size_t min_fft_len
        = 1 + kernel_length_ - 1;  // Simplified, assume min input len 1
    min_fft_len = std::max(min_fft_len, kernel_length_);
    fft_length_ = convolution_detail::next_power_of_two(min_fft_len);
    if (fft_length_ == 0) {
      throw std::length_error("FFT length overflow during plan creation.");
    }
    if constexpr (!Utils::is_complex_v<T>) {
      if (fft_length_ < 2) fft_length_ = 2;
    }  // RFFT needs length >= 2

    // Create the appropriate internal FFT plan
    Status fft_plan_status = Status::Failure;
    if constexpr (Utils::is_complex_v<T>) {
      try {
        fft_plan_impl_variant_
            .emplace<std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>(
                std::make_unique<OneMKLFFTPlanImpl<T_Complex>>(fft_length_));
        auto* plan_ptr_ptr
            = std::get_if<std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>(
                &fft_plan_impl_variant_);
        if (plan_ptr_ptr && *plan_ptr_ptr) {
          mkl_status_ = (*plan_ptr_ptr)->mkl_status_;
          if (mkl_status_ == DFTI_NO_ERROR) fft_plan_status = Status::Success;
        }
      }
      catch (...) {
        fft_plan_status = Status::Failure;
      }
    }
    else {
      try {
        fft_plan_impl_variant_
            .emplace<std::unique_ptr<OneMKLRFFTPlanImpl<T_Real>>>(
                std::make_unique<OneMKLRFFTPlanImpl<T_Real>>(fft_length_));
        auto* plan_ptr_ptr
            = std::get_if<std::unique_ptr<OneMKLRFFTPlanImpl<T_Real>>>(
                &fft_plan_impl_variant_);
        if (plan_ptr_ptr && *plan_ptr_ptr) {
          mkl_status_ = (*plan_ptr_ptr)->mkl_status_;
          if (mkl_status_ == DFTI_NO_ERROR) fft_plan_status = Status::Success;
        }
      }
      catch (...) {
        fft_plan_status = Status::Failure;
      }
    }
    if (fft_plan_status != Status::Success) {
      throw std::runtime_error(
          "Failed internal FFT plan creation for convolution. MKL Status: "
          + std::to_string(mkl_status_));
    }

    // Prepare the kernel: Reverse, Pad, FFT
    std::vector<T> padded_kernel = original_kernel_;
    std::reverse(padded_kernel.begin(), padded_kernel.end());
    padded_kernel.resize(fft_length_, T{0});  // Pad with zeros

    Status kernel_fft_status = Status::Failure;
    size_t kernel_fft_size = 0;

    // --- Kernel FFT Calculation (Outside Visit) ---
    if constexpr (Utils::is_complex_v<T>) {
      // T is Complex: Use FFTPlanImpl
      auto* plan_ptr_ptr
          = std::get_if<std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>(
              &fft_plan_impl_variant_);
      if (plan_ptr_ptr && *plan_ptr_ptr) {
        kernel_fft_size = fft_length_;
        kernel_fft_.resize(kernel_fft_size);
        const T_Complex* p_in
            = reinterpret_cast<const T_Complex*>(padded_kernel.data());
        kernel_fft_status
            = (*plan_ptr_ptr)->fft({p_in, fft_length_}, kernel_fft_);
      }
      else {
        kernel_fft_status = Status::InvalidOperation;  // Should not happen
      }
    }
    else {
      // T is Real: Use RFFTPlanImpl
      auto* plan_ptr_ptr
          = std::get_if<std::unique_ptr<OneMKLRFFTPlanImpl<T_Real>>>(
              &fft_plan_impl_variant_);
      if (plan_ptr_ptr && *plan_ptr_ptr) {
        kernel_fft_size = fft_length_ / 2 + 1;
        kernel_fft_.resize(kernel_fft_size);
        const T_Real* p_in = padded_kernel.data();  // T is T_Real here
        kernel_fft_status
            = (*plan_ptr_ptr)->rfft({p_in, fft_length_}, kernel_fft_);
      }
      else {
        kernel_fft_status = Status::InvalidOperation;  // Should not happen
      }
    }
    // --- End Kernel FFT Calculation ---

    if (kernel_fft_status != Status::Success) {
      throw std::runtime_error(
          "Failed FFT of convolution kernel. Status: "
          + std::to_string(static_cast<int>(kernel_fft_status)));
    }

    // Allocate temporary buffers
    try {
      input_padded_ = std::make_unique<T[]>(fft_length_);
      input_fft_ = std::make_unique<T_Complex[]>(
          kernel_fft_size);  // Size matches kernel_fft_
      product_fft_ = std::make_unique<T_Complex[]>(
          kernel_fft_size);  // Size matches kernel_fft_
      result_ifft_ = std::make_unique<T[]>(fft_length_);
    }
    catch (const std::bad_alloc& e) {
      throw std::runtime_error(
          "Buffer allocation failed for convolution plan: "
          + std::string(e.what()));
    }
    if (!input_padded_ || !input_fft_ || !product_fft_ || !result_ifft_) {
      throw std::runtime_error(
          "Buffer allocation failed for convolution plan (unexpected).");
    }
  }

  template <typename T>
  OneMKLConvolutionPlanImpl<T>::~OneMKLConvolutionPlanImpl() = default;

  template <typename T>
  Status OneMKLConvolutionPlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output) const
  {
    if (std::holds_alternative<std::monostate>(fft_plan_impl_variant_)) {
      return Status::InvalidOperation;  // Plan not properly initialized
    }

    const size_t input_len = input.size();
    const size_t output_len_expected = get_output_length(input_len);
    const size_t full_conv_len = (input_len > 0 && kernel_length_ > 0)
                                     ? (input_len + kernel_length_ - 1)
                                     : 0;
    const size_t current_fft_len = fft_length_;  // Use the length from the plan

    if (output.size() < output_len_expected) {
      return Status::SizeMismatch;
    }
    if (input_len == 0 || kernel_length_ == 0) {
      std::fill(output.begin(), output.begin() + output_len_expected, T{0});
      if (output.size() > output_len_expected) {
        std::fill(output.begin() + output_len_expected, output.end(), T{0});
      }
      return Status::Success;
    }

    // Check if the precomputed FFT length is sufficient for this input size
    if (current_fft_len < full_conv_len) {
      std::cerr << "Warning: OneMKLConvolutionPlan FFT length ("
                << current_fft_len
                << ") is too small for required linear convolution length ("
                << full_conv_len
                << "). Result will be incorrect (circular convolution)."
                << std::endl;
      // return Status::InvalidArgument; // Or proceed with warning
    }
    // Ensure input length itself doesn't exceed FFT length (for padding)
    if (input_len > current_fft_len) {
      return Status::InvalidArgument;  // Input longer than FFT plan allows
    }

    // 1. Pad Input Signal
    std::memcpy(input_padded_.get(), input.data(), input_len * sizeof(T));
    std::fill(
        input_padded_.get() + input_len,
        input_padded_.get() + current_fft_len,
        T{0});

    // Prepare spans for buffers
    size_t input_fft_size
        = kernel_fft_.size();  // Size of the complex FFT result
    T_Complex* input_fft_ptr = input_fft_.get();
    std::span<T_Complex> input_fft_span(input_fft_ptr, input_fft_size);
    T_Complex* product_fft_ptr = product_fft_.get();
    std::span<T_Complex> product_fft_span(product_fft_ptr, kernel_fft_.size());
    T* result_ifft_ptr = result_ifft_.get();
    std::span<T> result_ifft_span(result_ifft_ptr, current_fft_len);

    Status status = Status::Failure;

    // 2. Perform FFT on Input
    if constexpr (Utils::is_complex_v<T>) {
      auto* plan_ptr_ptr
          = std::get_if<std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>(
              &fft_plan_impl_variant_);
      if (!plan_ptr_ptr || !*plan_ptr_ptr) return Status::InvalidOperation;
      const T_Complex* p_in
          = reinterpret_cast<const T_Complex*>(input_padded_.get());
      status = (*plan_ptr_ptr)->fft({p_in, current_fft_len}, input_fft_span);
    }
    else {
      auto* plan_ptr_ptr
          = std::get_if<std::unique_ptr<OneMKLRFFTPlanImpl<T_Real>>>(
              &fft_plan_impl_variant_);
      if (!plan_ptr_ptr || !*plan_ptr_ptr) return Status::InvalidOperation;
      const T_Real* p_in = input_padded_.get();
      status = (*plan_ptr_ptr)->rfft({p_in, current_fft_len}, input_fft_span);
    }
    if (status != Status::Success) return status;

    // 3. Multiply FFTs (Input FFT * Kernel FFT) using VML
    MKL_LONG n_mul = static_cast<MKL_LONG>(product_fft_span.size());
    if constexpr (std::is_same_v<T_Real, float>) {  // Check underlying real
                                                    // type
      vcMul(
          n_mul,
          reinterpret_cast<const MKL_Complex8*>(input_fft_ptr),
          reinterpret_cast<const MKL_Complex8*>(kernel_fft_.data()),
          reinterpret_cast<MKL_Complex8*>(product_fft_ptr));
    }
    else {  // double
      vzMul(
          n_mul,
          reinterpret_cast<const MKL_Complex16*>(input_fft_ptr),
          reinterpret_cast<const MKL_Complex16*>(kernel_fft_.data()),
          reinterpret_cast<MKL_Complex16*>(product_fft_ptr));
    }

    // 4. Perform IFFT
    const T_Complex* product_ptr = product_fft_.get();
    std::span<const T_Complex> product_const_span(
        product_ptr, product_fft_span.size());
    if constexpr (Utils::is_complex_v<T>) {
      auto* plan_ptr_ptr
          = std::get_if<std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>(
              &fft_plan_impl_variant_);
      if (!plan_ptr_ptr || !*plan_ptr_ptr) return Status::InvalidOperation;
      T_Complex* result_ptr = reinterpret_cast<T_Complex*>(result_ifft_ptr);
      std::span<T_Complex> result_complex_span(result_ptr, current_fft_len);
      status = (*plan_ptr_ptr)->ifft(product_const_span, result_complex_span);
    }
    else {
      auto* plan_ptr_ptr
          = std::get_if<std::unique_ptr<OneMKLRFFTPlanImpl<T_Real>>>(
              &fft_plan_impl_variant_);
      if (!plan_ptr_ptr || !*plan_ptr_ptr) return Status::InvalidOperation;
      status = (*plan_ptr_ptr)->irfft(product_const_span, result_ifft_span);
    }
    if (status != Status::Success) return status;

    // 5. Extract Correct Output based on Type
    size_t full_start_idx = 0;
    size_t count = output_len_expected;

    switch (type_) {
      case ConvolutionType::Full:
        full_start_idx = 0;
        count = full_conv_len;  // Use calculated full length
        break;
      case ConvolutionType::Same:
        full_start_idx = (kernel_length_ - 1) / 2;
        count = input_len;
        break;
      case ConvolutionType::Valid:
        full_start_idx = kernel_length_ - 1;
        count = (input_len >= kernel_length_) ? (input_len - kernel_length_ + 1)
                                              : 0;
        break;
      default:
        return Status::InvalidArgument;  // Should not happen
    }

    // Ensure count matches expected output length derived from
    // get_output_length
    assert(
        count == output_len_expected && "Output length calculation mismatch");

    // Check bounds before copying
    size_t linear_conv_len_available = std::min(
        full_conv_len, current_fft_len);  // How much of the linear conv is
                                          // actually in the buffer
    if (full_start_idx >= linear_conv_len_available) {
      count = 0;  // Start index is beyond available data
    }
    else if (full_start_idx + count > linear_conv_len_available) {
      count = linear_conv_len_available
              - full_start_idx;  // Adjust count to available data
      if (count > output.size())
        count = output.size();  // Cannot copy more than output buffer size
      std::cerr << "Warning: Convolution result truncated due to FFT length."
                << std::endl;
    }

    // Ensure count does not exceed output buffer size
    count = std::min(count, output.size());

    // Copy the relevant part of the IFFT result
    if (count > 0) {
      std::memcpy(
          output.data(),
          result_ifft_.get() + full_start_idx,
          count * sizeof(T));
    }

    // Zero pad the rest of the output buffer if necessary
    if (count < output_len_expected) {
      std::fill(
          output.begin() + count, output.begin() + output_len_expected, T{0});
    }
    // Zero pad any extra space in the user-provided buffer beyond the expected
    // length
    if (output.size() > output_len_expected) {
      std::fill(output.begin() + output_len_expected, output.end(), T{0});
    }

    return Status::Success;
  }

  template <typename T>
  size_t OneMKLConvolutionPlanImpl<T>::get_output_length(
      size_t input_length) const
  {
    if (kernel_length_ == 0)
      return (type_ == ConvolutionType::Same ? input_length : 0);
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
        // Should not happen if type_ is validated in constructor
        throw std::logic_error(
            "Invalid ConvolutionType encountered in get_output_length");
    }
  }

  //--------------------------------------------------------------------------
  // OneMKLCorrelationPlanImpl Method Definitions
  //--------------------------------------------------------------------------
  template <typename T>
  OneMKLCorrelationPlanImpl<T>::OneMKLCorrelationPlanImpl(
      const AbstractBackend* owner,
      const std::vector<T>& kernel,  // Template
      ConvolutionType type,
      ConvolutionMethod method)
      : type_(type),
        method_(method),
        original_kernel_(kernel),
        kernel_length_(kernel.size()),  // Template length
        fft_length_(0),
        mkl_status_(DFTI_NO_ERROR)
  {
    if (!owner) {
      throw std::invalid_argument("OneMKLCorrelationPlanImpl requires owner.");
    }
    if (kernel_length_ == 0) {
      throw std::invalid_argument("Correlation template empty.");
    }

    // Determine FFT length
    size_t min_fft_len = 1 + kernel_length_ - 1;  // Simplified
    min_fft_len = std::max(min_fft_len, kernel_length_);
    fft_length_ = convolution_detail::next_power_of_two(min_fft_len);
    if (fft_length_ == 0) {
      throw std::length_error("FFT length overflow.");
    }
    if constexpr (!Utils::is_complex_v<T>) {
      if (fft_length_ < 2) fft_length_ = 2;
    }

    // Create internal FFT plan
    Status fft_plan_status = Status::Failure;
    if constexpr (Utils::is_complex_v<T>) {
      try {
        fft_plan_impl_variant_
            .emplace<std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>(
                std::make_unique<OneMKLFFTPlanImpl<T_Complex>>(fft_length_));
        auto* p = std::get_if<std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>(
            &fft_plan_impl_variant_);
        if (p && *p) {
          mkl_status_ = (*p)->mkl_status_;
          if (mkl_status_ == DFTI_NO_ERROR) fft_plan_status = Status::Success;
        }
      }
      catch (...) {
        fft_plan_status = Status::Failure;
      }
    }
    else {
      try {
        fft_plan_impl_variant_
            .emplace<std::unique_ptr<OneMKLRFFTPlanImpl<T_Real>>>(
                std::make_unique<OneMKLRFFTPlanImpl<T_Real>>(fft_length_));
        auto* p = std::get_if<std::unique_ptr<OneMKLRFFTPlanImpl<T_Real>>>(
            &fft_plan_impl_variant_);
        if (p && *p) {
          mkl_status_ = (*p)->mkl_status_;
          if (mkl_status_ == DFTI_NO_ERROR) fft_plan_status = Status::Success;
        }
      }
      catch (...) {
        fft_plan_status = Status::Failure;
      }
    }
    if (fft_plan_status != Status::Success) {
      throw std::runtime_error(
          "Failed internal FFT plan creation for correlation.");
    }

    // Prepare the template: Pad (NO REVERSAL)
    std::vector<T> padded_template = original_kernel_;
    padded_template.resize(fft_length_, T{0});

    Status kernel_fft_status = Status::Failure;
    size_t kernel_fft_size = 0;
    std::vector<T_Complex>
        temp_kernel_fft;  // Temporary storage for FFT before conjugation

    // FFT the padded template
    // --- Template FFT Calculation (Outside Visit) ---
    if constexpr (Utils::is_complex_v<T>) {
      auto* plan_ptr_ptr
          = std::get_if<std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>(
              &fft_plan_impl_variant_);
      if (plan_ptr_ptr && *plan_ptr_ptr) {
        kernel_fft_size = fft_length_;
        temp_kernel_fft.resize(kernel_fft_size);
        const T_Complex* p_in
            = reinterpret_cast<const T_Complex*>(padded_template.data());
        kernel_fft_status
            = (*plan_ptr_ptr)->fft({p_in, fft_length_}, temp_kernel_fft);
      }
      else {
        kernel_fft_status = Status::InvalidOperation;
      }
    }
    else {
      auto* plan_ptr_ptr
          = std::get_if<std::unique_ptr<OneMKLRFFTPlanImpl<T_Real>>>(
              &fft_plan_impl_variant_);
      if (plan_ptr_ptr && *plan_ptr_ptr) {
        kernel_fft_size = fft_length_ / 2 + 1;
        temp_kernel_fft.resize(kernel_fft_size);
        const T_Real* p_in = padded_template.data();
        kernel_fft_status
            = (*plan_ptr_ptr)->rfft({p_in, fft_length_}, temp_kernel_fft);
      }
      else {
        kernel_fft_status = Status::InvalidOperation;
      }
    }
    // --- End Template FFT Calculation ---

    if (kernel_fft_status != Status::Success) {
      throw std::runtime_error("Failed FFT of correlation template.");
    }

    // Store the CONJUGATE of the template's FFT
    kernel_fft_conj_.resize(kernel_fft_size);
    MKL_LONG n_conj = static_cast<MKL_LONG>(kernel_fft_size);
    if constexpr (std::is_same_v<T_Real, float>) {  // Check underlying real
                                                    // type
      vcConj(
          n_conj,
          reinterpret_cast<const MKL_Complex8*>(temp_kernel_fft.data()),
          reinterpret_cast<MKL_Complex8*>(kernel_fft_conj_.data()));
    }
    else {  // double
      vzConj(
          n_conj,
          reinterpret_cast<const MKL_Complex16*>(temp_kernel_fft.data()),
          reinterpret_cast<MKL_Complex16*>(kernel_fft_conj_.data()));
    }

    // Allocate temporary buffers
    try {
      input_padded_ = std::make_unique<T[]>(fft_length_);
      input_fft_ = std::make_unique<T_Complex[]>(kernel_fft_size);
      product_fft_ = std::make_unique<T_Complex[]>(kernel_fft_size);
      result_ifft_ = std::make_unique<T[]>(fft_length_);
    }
    catch (const std::bad_alloc& e) {
      throw std::runtime_error(
          "Buffer allocation failed for correlation plan.");
    }
    if (!input_padded_ || !input_fft_ || !product_fft_ || !result_ifft_) {
      throw std::runtime_error(
          "Buffer allocation failed for correlation plan.");
    }
  }

  template <typename T>
  OneMKLCorrelationPlanImpl<T>::~OneMKLCorrelationPlanImpl() = default;

  template <typename T>
  Status OneMKLCorrelationPlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output) const
  {
    if (std::holds_alternative<std::monostate>(fft_plan_impl_variant_)) {
      return Status::InvalidOperation;
    }
    const size_t input_len = input.size();
    const size_t output_len_expected = get_output_length(input_len);
    const size_t full_corr_len = (input_len > 0 && kernel_length_ > 0)
                                     ? (input_len + kernel_length_ - 1)
                                     : 0;
    const size_t current_fft_len = fft_length_;

    if (output.size() < output_len_expected) {
      return Status::SizeMismatch;
    }
    if (input_len == 0 || kernel_length_ == 0) {
      std::fill(output.begin(), output.begin() + output_len_expected, T{0});
      if (output.size() > output_len_expected) {
        std::fill(output.begin() + output_len_expected, output.end(), T{0});
      }
      return Status::Success;
    }
    if (current_fft_len < full_corr_len) { /* Warning */
    }
    if (input_len > current_fft_len) {
      return Status::InvalidArgument;
    }

    std::memcpy(input_padded_.get(), input.data(), input_len * sizeof(T));
    std::fill(
        input_padded_.get() + input_len,
        input_padded_.get() + current_fft_len,
        T{0});

    size_t input_fft_size = kernel_fft_conj_.size();
    T_Complex* input_fft_ptr = input_fft_.get();
    std::span<T_Complex> input_fft_span(input_fft_ptr, input_fft_size);
    T_Complex* product_fft_ptr = product_fft_.get();
    std::span<T_Complex> product_fft_span(
        product_fft_ptr, kernel_fft_conj_.size());
    T* result_ifft_ptr = result_ifft_.get();
    std::span<T> result_ifft_span(result_ifft_ptr, current_fft_len);

    Status status = Status::Failure;

    // FFT Input
    if constexpr (Utils::is_complex_v<T>) {
      auto* plan_ptr_ptr
          = std::get_if<std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>(
              &fft_plan_impl_variant_);
      if (!plan_ptr_ptr || !*plan_ptr_ptr) return Status::InvalidOperation;
      const T_Complex* p_in
          = reinterpret_cast<const T_Complex*>(input_padded_.get());
      status = (*plan_ptr_ptr)->fft({p_in, current_fft_len}, input_fft_span);
    }
    else {
      auto* plan_ptr_ptr
          = std::get_if<std::unique_ptr<OneMKLRFFTPlanImpl<T_Real>>>(
              &fft_plan_impl_variant_);
      if (!plan_ptr_ptr || !*plan_ptr_ptr) return Status::InvalidOperation;
      const T_Real* p_in = input_padded_.get();
      status = (*plan_ptr_ptr)->rfft({p_in, current_fft_len}, input_fft_span);
    }
    if (status != Status::Success) return status;

    // Multiply FFTs (Input FFT * CONJ(Kernel FFT))
    MKL_LONG n_mul = static_cast<MKL_LONG>(product_fft_span.size());
    if constexpr (std::is_same_v<T_Real, float>) {  // Check underlying real
                                                    // type
      vcMul(
          n_mul,
          reinterpret_cast<const MKL_Complex8*>(input_fft_ptr),
          reinterpret_cast<const MKL_Complex8*>(kernel_fft_conj_.data()),
          reinterpret_cast<MKL_Complex8*>(product_fft_ptr));
    }
    else {  // double
      vzMul(
          n_mul,
          reinterpret_cast<const MKL_Complex16*>(input_fft_ptr),
          reinterpret_cast<const MKL_Complex16*>(kernel_fft_conj_.data()),
          reinterpret_cast<MKL_Complex16*>(product_fft_ptr));
    }

    // IFFT Result
    const T_Complex* product_ptr = product_fft_.get();
    std::span<const T_Complex> product_const_span(
        product_ptr, product_fft_span.size());
    if constexpr (Utils::is_complex_v<T>) {
      auto* plan_ptr_ptr
          = std::get_if<std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>(
              &fft_plan_impl_variant_);
      if (!plan_ptr_ptr || !*plan_ptr_ptr) return Status::InvalidOperation;
      T_Complex* result_ptr = reinterpret_cast<T_Complex*>(result_ifft_ptr);
      std::span<T_Complex> result_complex_span(result_ptr, current_fft_len);
      status = (*plan_ptr_ptr)->ifft(product_const_span, result_complex_span);
    }
    else {
      auto* plan_ptr_ptr
          = std::get_if<std::unique_ptr<OneMKLRFFTPlanImpl<T_Real>>>(
              &fft_plan_impl_variant_);
      if (!plan_ptr_ptr || !*plan_ptr_ptr) return Status::InvalidOperation;
      status = (*plan_ptr_ptr)->irfft(product_const_span, result_ifft_span);
    }
    if (status != Status::Success) return status;

    // Extract output (same logic as convolution, but check Valid mode start
    // index)
    size_t full_start_idx = 0;
    size_t count = output_len_expected;

    switch (type_) {
      case ConvolutionType::Full:
        full_start_idx = 0;
        count = full_corr_len;
        break;
      case ConvolutionType::Same:
        full_start_idx = (kernel_length_ - 1) / 2;
        count = input_len;
        break;
      case ConvolutionType::Valid:
        full_start_idx
            = 0;  // Note: Valid correlation starts at index 0 of full result
        count = (input_len >= kernel_length_) ? (input_len - kernel_length_ + 1)
                                              : 0;
        break;
      default:
        return Status::InvalidArgument;
    }

    assert(count == output_len_expected);
    size_t linear_corr_len_available = std::min(full_corr_len, current_fft_len);
    if (full_start_idx >= linear_corr_len_available) {
      count = 0;
    }
    else if (full_start_idx + count > linear_corr_len_available) {
      count = linear_corr_len_available - full_start_idx;
      if (count > output.size()) count = output.size();
      std::cerr << "Warning: Correlation result truncated." << std::endl;
    }
    count = std::min(count, output.size());

    if (count > 0) {
      std::memcpy(
          output.data(),
          result_ifft_.get() + full_start_idx,
          count * sizeof(T));
    }

    if (count < output_len_expected) {
      std::fill(
          output.begin() + count, output.begin() + output_len_expected, T{0});
    }
    if (output.size() > output_len_expected) {
      std::fill(output.begin() + output_len_expected, output.end(), T{0});
    }
    return Status::Success;
  }

  template <typename T>
  size_t OneMKLCorrelationPlanImpl<T>::get_output_length(
      size_t input_length) const
  {
    // Same logic as convolution
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

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  template class OneMKLConvolutionPlanImpl<F32>;
  template class OneMKLConvolutionPlanImpl<F64>;
  template class OneMKLConvolutionPlanImpl<C32>;
  template class OneMKLConvolutionPlanImpl<C64>;

  template class OneMKLCorrelationPlanImpl<F32>;
  template class OneMKLCorrelationPlanImpl<F64>;
  template class OneMKLCorrelationPlanImpl<C32>;
  template class OneMKLCorrelationPlanImpl<C64>;

}  // namespace OmniDSP::backend
