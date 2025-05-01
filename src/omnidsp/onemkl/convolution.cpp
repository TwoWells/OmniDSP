/**
 * @file convolution.cpp (onemkl)
 * @brief Implements oneMKL backend ConvolutionPlanImpl and CorrelationPlanImpl
 * classes, using internal oneMKL FFT plans (DFTI) for FFT operations.
 */

#include "convolution.hpp"  // Include the corresponding header file

#ifdef OMNIDSP_USE_ONEMKL  // Compile guard

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

namespace OmniDSP::backend {

  // Helper function from onemkl/backend.cpp (or move to common utility)
  // Ensure this helper is available
  inline Status mkl_status_to_omnidsp_status(MKL_LONG status)
  {
    if (status == DFTI_NO_ERROR) {
      return Status::Success;
    }
    std::cerr << "MKL DFTI Error: " << DftiErrorMessage(status)
              << " (Code: " << status << ")" << std::endl;
    if (status == DFTI_MEMORY_ERROR) return Status::AllocationError;
    if (status == DFTI_INVALID_CONFIGURATION) return Status::InvalidArgument;
    if (status == DFTI_INCONSISTENT_CONFIGURATION)
      return Status::InvalidArgument;
    if (status == DFTI_NUMBER_OF_THREADS_ERROR) return Status::BackendError;
    if (status == DFTI_UNIMPLEMENTED) return Status::UnsupportedFeature;
    return Status::BackendError;
  }

  // Helper function to calculate next power of two (move to common utility?)
  // Copied from default/convolution.cpp - ensure consistency or use common
  // header
  namespace convolution_detail {  // Internal namespace
    inline size_t next_power_of_two(size_t n)
    {
      if (n == 0) return 0;  // Return 0 for input 0
      if ((n > 0) && ((n & (n - 1)) == 0)) {
        return n;
      }
      if (n > (std::numeric_limits<size_t>::max() / 2U)) {
        if (n == (~(std::numeric_limits<size_t>::max() >> 1))) return n;
        return 0;  // Indicate overflow
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
      const AbstractBackend* owner,  // Needs owner to create sub-plans
      const std::vector<T>& kernel,
      ConvolutionType type,
      ConvolutionMethod method)
      : type_(type),
        method_(method),  // Store method hint
        original_kernel_(kernel),
        kernel_length_(kernel.size()),
        fft_length_(0),             // Will be set below
        mkl_status_(DFTI_NO_ERROR)  // Initialize MKL status
  {
    if (!owner) {
      throw std::invalid_argument(
          "OneMKLConvolutionPlanImpl requires a valid owner AbstractBackend "
          "pointer.");
    }
    if (kernel_length_ == 0) {
      throw std::invalid_argument("Convolution kernel cannot be empty.");
    }

    // --- Determine FFT length (using a simple heuristic for now) ---
    // A more sophisticated approach might consider input length hints if
    // available or use different lengths based on 'method' hint.
    size_t min_fft_len
        = kernel_length_ * 2 - 1;  // Minimum length for linear convolution
    fft_length_ = convolution_detail::next_power_of_two(min_fft_len);
    if (fft_length_ == 0) {  // Check for overflow from next_power_of_two
      throw std::length_error(
          "Calculated FFT length for convolution plan exceeds limits.");
    }
    if constexpr (!Detail::is_complex_v<T>) {  // Real FFT
      if (fft_length_ < 2) fft_length_ = 2;  // MKL DFTI requires N>=2 for real
    }

    // --- Create internal FFT plan using the owner backend ---
    // This ensures we use the oneMKL FFT plan if called from OneMKLBackend
    Status fft_plan_status = Status::Failure;
    if constexpr (Detail::is_complex_v<T>) {  // Complex data -> Complex FFT
                                              // plan
      auto plan_expected = owner->create_fft_plan_c32(
          fft_length_);  // Assuming T is C32 or C64
      if (!plan_expected) {
        fft_plan_status = plan_expected.error();
      }
      else {
        // Move the concrete OneMKLFFTPlanImpl pointer into the variant
        // We need to downcast or know the concrete type returned by the owner
        // Assuming owner IS OneMKLBackend here, which is slightly fragile.
        // A better approach might involve dynamic_cast or a type query on the
        // backend. For now, assume owner->create_fft_plan_c32 returns a plan
        // wrapping OneMKLFFTPlanImpl. This requires the public Plan's pimpl()
        // accessor or similar.
        // *** This part is tricky without modifying public Plan API or using
        // dynamic_cast ***
        // *** Simplified approach: Recreate the plan directly here (less ideal)
        // ***
        try {
          fft_plan_impl_variant_
              .emplace<std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>(
                  std::make_unique<OneMKLFFTPlanImpl<T_Complex>>(fft_length_));
          // Check the status of the newly created plan
          auto* plan_ptr_ptr
              = std::get_if<std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>(
                  &fft_plan_impl_variant_);
          if (plan_ptr_ptr && *plan_ptr_ptr) {
            mkl_status_ = (*plan_ptr_ptr)->mkl_status_;
            if (mkl_status_ == DFTI_NO_ERROR) fft_plan_status = Status::Success;
          }
        }
        catch (...) { /* handle exceptions */
          fft_plan_status = Status::Failure;
        }
      }
    }
    else {  // Real data -> Real FFT plan
      auto plan_expected = owner->create_rfft_plan_f32(
          fft_length_);  // Assuming T is F32 or F64
      if (!plan_expected) {
        fft_plan_status = plan_expected.error();
      }
      else {
        // *** Similar casting/recreation issue as above ***
        // *** Simplified approach: Recreate the plan directly here ***
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
        catch (...) { /* handle exceptions */
          fft_plan_status = Status::Failure;
        }
      }
    }

    if (fft_plan_status != Status::Success) {
      throw std::runtime_error(
          "Failed to create internal FFT plan for ConvolutionPlan. Status: "
          + std::to_string(static_cast<int>(fft_plan_status)));
    }

    // --- Prepare and FFT the kernel using the internal plan ---
    std::vector<T> padded_kernel = original_kernel_;
    std::reverse(
        padded_kernel.begin(), padded_kernel.end());  // Reverse for convolution
    padded_kernel.resize(fft_length_, T{});           // Zero-pad

    Status kernel_fft_status = Status::Failure;
    size_t kernel_fft_size = 0;

    // Visit the variant to call the correct FFT function
    std::visit(
        [&](auto&& plan_ptr)
        {
          using PlanPtrT = std::decay_t<decltype(plan_ptr)>;
          if constexpr (!std::is_same_v<
                            PlanPtrT,
                            std::monostate>) {  // Check if variant holds a
                                                // valid plan
            if (!plan_ptr) return;  // Should not happen if status was Success

            if constexpr (std::is_same_v<
                              PlanPtrT,
                              std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>) {
              kernel_fft_size = fft_length_;
              kernel_fft_.resize(kernel_fft_size);
              const T_Complex* kernel_input_ptr
                  = reinterpret_cast<const T_Complex*>(padded_kernel.data());
              std::span<const T_Complex> kernel_input_span(
                  kernel_input_ptr, fft_length_);
              kernel_fft_status = plan_ptr->fft(
                  kernel_input_span, std::span<T_Complex>(kernel_fft_));
            }
            else if constexpr (std::is_same_v<
                                   PlanPtrT,
                                   std::unique_ptr<
                                       OneMKLRFFTPlanImpl<T_Real>>>) {
              kernel_fft_size = fft_length_ / 2 + 1;
              kernel_fft_.resize(kernel_fft_size);
              const T_Real* kernel_input_ptr = padded_kernel.data();
              std::span<const T_Real> kernel_input_span(
                  kernel_input_ptr, fft_length_);
              kernel_fft_status = plan_ptr->rfft(
                  kernel_input_span, std::span<T_Complex>(kernel_fft_));
            }
          }
        },
        fft_plan_impl_variant_);

    if (kernel_fft_status != Status::Success) {
      throw std::runtime_error(
          "Failed to compute FFT of kernel during ConvolutionPlan creation. "
          "Status: "
          + std::string(get_status_string(kernel_fft_status)));
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
          "Failed to allocate temporary buffers for convolution plan: "
          + std::string(e.what()));
    }
    if (!input_padded_ || !input_fft_ || !product_fft_ || !result_ifft_) {
      throw std::runtime_error(
          "Failed to allocate temporary buffers for convolution plan "
          "(unexpected).");
    }

    // std::cout << "oneMKL ConvPlanImpl created. Kernel Len: " <<
    // kernel_length_ << ", FFT Len: " << fft_length_ << std::endl; // Debug
  }

  // Destructor Implementation
  template <typename T>
  OneMKLConvolutionPlanImpl<T>::~OneMKLConvolutionPlanImpl()
  {
    // unique_ptr members handle cleanup automatically
    // std::cout << "oneMKL ConvPlanImpl destroyed." << std::endl; // Debug
  }

  // Execute Method Implementation
  template <typename T>
  Status OneMKLConvolutionPlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output) const
  {
    // --- Input Validation and Setup ---
    if (std::holds_alternative<std::monostate>(fft_plan_impl_variant_)) {
      return Status::InvalidOperation;  // Plan not initialized correctly
    }
    const size_t input_len = input.size();
    const size_t output_len_expected = get_output_length(input_len);
    const size_t full_conv_len = input_len + kernel_length_ - 1;
    const size_t current_fft_len = fft_length_;

    if (output.size() < output_len_expected) {
      return Status::SizeMismatch;
    }
    if (input_len == 0 || kernel_length_ == 0) {  // Handle empty input/kernel
      std::fill(output.begin(), output.begin() + output_len_expected, T{0});
      if (output.size() > output_len_expected) {
        std::fill(output.begin() + output_len_expected, output.end(), T{0});
      }
      return Status::Success;
    }
    if (current_fft_len < full_conv_len) {
      std::cerr
          << "Warning: Convolution plan's internal FFT length ("
          << current_fft_len
          << ") is too small for full linear convolution with input size ("
          << input_len << "). Result will be circular." << std::endl;
      // Proceeding with circular convolution
    }
    if (input_len > current_fft_len) {
      std::cerr << "Error: Input size (" << input_len
                << ") exceeds convolution plan's internal FFT length ("
                << current_fft_len << "). Overlap-Add/Save not implemented."
                << std::endl;
      return Status::InvalidArgument;  // Or SizeMismatch?
    }

    // --- Prepare Buffers ---
    std::memcpy(input_padded_.get(), input.data(), input_len * sizeof(T));
    std::fill(
        input_padded_.get() + input_len,
        input_padded_.get() + current_fft_len,
        T{0});

    size_t input_fft_size = kernel_fft_.size();
    T_Complex* input_fft_ptr = input_fft_.get();
    std::span<T_Complex> input_fft_span(input_fft_ptr, input_fft_size);
    T_Complex* product_fft_ptr = product_fft_.get();
    std::span<T_Complex> product_fft_span(product_fft_ptr, kernel_fft_.size());
    T* result_ifft_ptr = result_ifft_.get();
    std::span<T> result_ifft_span(result_ifft_ptr, current_fft_len);

    Status status = Status::Failure;

    // --- FFT Input, Multiply, IFFT using Visitor ---
    std::visit(
        [&](auto&& plan_ptr)
        {
          using PlanPtrT = std::decay_t<decltype(plan_ptr)>;
          if constexpr (!std::is_same_v<PlanPtrT, std::monostate>) {
            if (!plan_ptr) {
              status = Status::InvalidOperation;
              return;
            }  // Plan pointer is null

            // 1. FFT Input
            if constexpr (std::is_same_v<
                              PlanPtrT,
                              std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>) {
              const T_Complex* input_padded_ptr
                  = reinterpret_cast<const T_Complex*>(input_padded_.get());
              std::span<const T_Complex> input_padded_complex_span(
                  input_padded_ptr, current_fft_len);
              status = plan_ptr->fft(input_padded_complex_span, input_fft_span);
            }
            else if constexpr (std::is_same_v<
                                   PlanPtrT,
                                   std::unique_ptr<
                                       OneMKLRFFTPlanImpl<T_Real>>>) {
              const T_Real* input_padded_ptr = input_padded_.get();
              std::span<const T_Real> input_padded_real_span(
                  input_padded_ptr, current_fft_len);
              status = plan_ptr->rfft(input_padded_real_span, input_fft_span);
            }
            if (status != Status::Success) return;  // Exit visitor on error

            // 2. Multiply FFTs (Input * Kernel) using MKL VML
            MKL_LONG n_mul = static_cast<MKL_LONG>(product_fft_span.size());
            if constexpr (std::is_same_v<T_Real, float>) {
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
            // VML status check? Assume success for now.

            // 3. Inverse FFT
            const T_Complex* product_ptr = product_fft_.get();
            std::span<const T_Complex> product_const_span(
                product_ptr, product_fft_span.size());
            if constexpr (std::is_same_v<
                              PlanPtrT,
                              std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>) {
              T_Complex* result_ptr
                  = reinterpret_cast<T_Complex*>(result_ifft_ptr);
              std::span<T_Complex> result_complex_span(
                  result_ptr, current_fft_len);
              status = plan_ptr->ifft(product_const_span, result_complex_span);
            }
            else if constexpr (std::is_same_v<
                                   PlanPtrT,
                                   std::unique_ptr<
                                       OneMKLRFFTPlanImpl<T_Real>>>) {
              // result_ifft_span is already std::span<T> where T is T_Real
              status = plan_ptr->irfft(product_const_span, result_ifft_span);
            }
            // status is now set by the IFFT/IRFFT call
          }
          else {
            status = Status::InvalidOperation;  // Variant holds monostate
          }
        },
        fft_plan_impl_variant_);

    if (status != Status::Success) return status;

    // --- Extract Correct Output based on Type ---
    // (Extraction logic remains the same as default/convolution.cpp)
    size_t full_start_idx = 0;
    size_t count = output_len_expected;

    switch (type_) {
      case ConvolutionType::Full:
        full_start_idx = 0;
        count = input_len + kernel_length_ - 1;
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
        return Status::InvalidArgument;
    }

    assert(count == output_len_expected);

    size_t linear_conv_len = input_len + kernel_length_ - 1;
    if (full_start_idx >= linear_conv_len) {
      count = 0;
    }
    else if (full_start_idx + count > linear_conv_len) {
      count = linear_conv_len - full_start_idx;
    }

    count = std::min(count, output.size());

    if (count > 0) {
      if (full_start_idx + count <= current_fft_len) {
        std::memcpy(
            output.data(),
            result_ifft_.get() + full_start_idx,
            count * sizeof(T));
      }
      else {
        // This case indicates an issue, either FFT length was too small or
        // indexing error
        size_t safe_count = (full_start_idx < current_fft_len)
                                ? (current_fft_len - full_start_idx)
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
              T{0});
        }
        std::cerr << "Warning: Convolution result truncated due to "
                     "insufficient FFT length or indexing issue."
                  << std::endl;
      }
    }

    // Zero out remaining part of user buffer if necessary
    if (count < output_len_expected) {
      std::fill(
          output.begin() + count, output.begin() + output_len_expected, T{0});
    }
    if (output.size() > output_len_expected) {
      std::fill(output.begin() + output_len_expected, output.end(), T{0});
    }

    return Status::Success;
  }

  // Getters
  template <typename T>
  size_t OneMKLConvolutionPlanImpl<T>::get_kernel_length() const
  {
    return kernel_length_;
  }
  template <typename T>
  ConvolutionType OneMKLConvolutionPlanImpl<T>::get_type() const
  {
    return type_;
  }
  template <typename T>
  ConvolutionMethod OneMKLConvolutionPlanImpl<T>::get_method() const
  {
    return method_;
  }
  template <typename T>
  size_t OneMKLConvolutionPlanImpl<T>::get_output_length(
      size_t input_length) const
  {
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
  std::span<const T> OneMKLConvolutionPlanImpl<T>::get_kernel() const
  {
    return std::span<const T>(original_kernel_);
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
      throw std::invalid_argument(
          "OneMKLCorrelationPlanImpl requires a valid owner AbstractBackend "
          "pointer.");
    }
    if (kernel_length_ == 0) {
      throw std::invalid_argument("Correlation template cannot be empty.");
    }

    // --- Determine FFT length ---
    size_t min_fft_len = kernel_length_ * 2 - 1;
    fft_length_ = convolution_detail::next_power_of_two(min_fft_len);
    if (fft_length_ == 0) {
      throw std::length_error(
          "Calculated FFT length for correlation plan exceeds limits.");
    }
    if constexpr (!Detail::is_complex_v<T>) {  // Real FFT
      if (fft_length_ < 2) fft_length_ = 2;
    }

    // --- Create internal FFT plan ---
    Status fft_plan_status = Status::Failure;
    if constexpr (Detail::is_complex_v<T>) {
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
          "Failed to create internal FFT plan for CorrelationPlan. Status: "
          + std::to_string(static_cast<int>(fft_plan_status)));
    }

    // --- Prepare and FFT the template (NO reversal) ---
    std::vector<T> padded_template = original_kernel_;
    padded_template.resize(fft_length_, T{});  // Zero-pad

    Status kernel_fft_status = Status::Failure;
    size_t kernel_fft_size = 0;
    std::vector<T_Complex> temp_kernel_fft;  // Temp storage

    std::visit(
        [&](auto&& plan_ptr)
        {
          using PlanPtrT = std::decay_t<decltype(plan_ptr)>;
          if constexpr (!std::is_same_v<PlanPtrT, std::monostate>) {
            if (!plan_ptr) return;
            if constexpr (std::is_same_v<
                              PlanPtrT,
                              std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>) {
              kernel_fft_size = fft_length_;
              temp_kernel_fft.resize(kernel_fft_size);
              const T_Complex* kernel_input_ptr
                  = reinterpret_cast<const T_Complex*>(padded_template.data());
              std::span<const T_Complex> kernel_input_span(
                  kernel_input_ptr, fft_length_);
              kernel_fft_status = plan_ptr->fft(
                  kernel_input_span, std::span<T_Complex>(temp_kernel_fft));
            }
            else if constexpr (std::is_same_v<
                                   PlanPtrT,
                                   std::unique_ptr<
                                       OneMKLRFFTPlanImpl<T_Real>>>) {
              kernel_fft_size = fft_length_ / 2 + 1;
              temp_kernel_fft.resize(kernel_fft_size);
              const T_Real* kernel_input_ptr = padded_template.data();
              std::span<const T_Real> kernel_input_span(
                  kernel_input_ptr, fft_length_);
              kernel_fft_status = plan_ptr->rfft(
                  kernel_input_span, std::span<T_Complex>(temp_kernel_fft));
            }
          }
        },
        fft_plan_impl_variant_);

    if (kernel_fft_status != Status::Success) {
      throw std::runtime_error(
          "Failed to compute FFT of template during CorrelationPlan creation. "
          "Status: "
          + std::string(get_status_string(kernel_fft_status)));
    }

    // --- CONJUGATE the result and store in kernel_fft_conj_ ---
    kernel_fft_conj_.resize(kernel_fft_size);
    MKL_LONG n_conj = static_cast<MKL_LONG>(kernel_fft_size);
    if constexpr (std::is_same_v<T_Real, float>) {
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
    // VML status check? Assume success.

    // Allocate temporary buffers
    try {
      input_padded_ = std::make_unique<T[]>(fft_length_);
      input_fft_ = std::make_unique<T_Complex[]>(kernel_fft_size);
      product_fft_ = std::make_unique<T_Complex[]>(kernel_fft_size);
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

    // std::cout << "oneMKL CorrPlanImpl created. Template Len: " <<
    // kernel_length_ << ", FFT Len: " << fft_length_ << std::endl; // Debug
  }

  // Destructor Implementation
  template <typename T>
  OneMKLCorrelationPlanImpl<T>::~OneMKLCorrelationPlanImpl()
  {
    // unique_ptr members handle cleanup
    // std::cout << "oneMKL CorrPlanImpl destroyed." << std::endl; // Debug
  }

  // Execute Method Implementation
  template <typename T>
  Status OneMKLCorrelationPlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output) const
  {
    // --- Input Validation and Setup ---
    if (std::holds_alternative<std::monostate>(fft_plan_impl_variant_)) {
      return Status::InvalidOperation;
    }
    const size_t input_len = input.size();
    const size_t output_len_expected = get_output_length(input_len);
    const size_t full_corr_len = input_len + kernel_length_ - 1;
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
    if (current_fft_len < full_corr_len) {
      std::cerr
          << "Warning: Correlation plan's internal FFT length ("
          << current_fft_len
          << ") is too small for full linear correlation with input size ("
          << input_len << "). Result will be circular." << std::endl;
    }
    if (input_len > current_fft_len) {
      std::cerr << "Error: Input size (" << input_len
                << ") exceeds correlation plan's internal FFT length ("
                << current_fft_len << "). Overlap-Add/Save not implemented."
                << std::endl;
      return Status::InvalidArgument;
    }

    // --- Prepare Buffers ---
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

    // --- FFT Input, Multiply, IFFT using Visitor ---
    std::visit(
        [&](auto&& plan_ptr)
        {
          using PlanPtrT = std::decay_t<decltype(plan_ptr)>;
          if constexpr (!std::is_same_v<PlanPtrT, std::monostate>) {
            if (!plan_ptr) {
              status = Status::InvalidOperation;
              return;
            }

            // 1. FFT Input
            if constexpr (std::is_same_v<
                              PlanPtrT,
                              std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>) {
              const T_Complex* input_padded_ptr
                  = reinterpret_cast<const T_Complex*>(input_padded_.get());
              std::span<const T_Complex> input_padded_complex_span(
                  input_padded_ptr, current_fft_len);
              status = plan_ptr->fft(input_padded_complex_span, input_fft_span);
            }
            else if constexpr (std::is_same_v<
                                   PlanPtrT,
                                   std::unique_ptr<
                                       OneMKLRFFTPlanImpl<T_Real>>>) {
              const T_Real* input_padded_ptr = input_padded_.get();
              std::span<const T_Real> input_padded_real_span(
                  input_padded_ptr, current_fft_len);
              status = plan_ptr->rfft(input_padded_real_span, input_fft_span);
            }
            if (status != Status::Success) return;

            // 2. Multiply FFTs (Input * Conjugated(Template)) using MKL VML
            MKL_LONG n_mul = static_cast<MKL_LONG>(product_fft_span.size());
            if constexpr (std::is_same_v<T_Real, float>) {
              vcMul(
                  n_mul,
                  reinterpret_cast<const MKL_Complex8*>(input_fft_ptr),
                  reinterpret_cast<const MKL_Complex8*>(
                      kernel_fft_conj_.data()),
                  reinterpret_cast<MKL_Complex8*>(product_fft_ptr));
            }
            else {  // double
              vzMul(
                  n_mul,
                  reinterpret_cast<const MKL_Complex16*>(input_fft_ptr),
                  reinterpret_cast<const MKL_Complex16*>(
                      kernel_fft_conj_.data()),
                  reinterpret_cast<MKL_Complex16*>(product_fft_ptr));
            }
            // VML status check? Assume success.

            // 3. Inverse FFT
            const T_Complex* product_ptr = product_fft_.get();
            std::span<const T_Complex> product_const_span(
                product_ptr, product_fft_span.size());
            if constexpr (std::is_same_v<
                              PlanPtrT,
                              std::unique_ptr<OneMKLFFTPlanImpl<T_Complex>>>) {
              T_Complex* result_ptr
                  = reinterpret_cast<T_Complex*>(result_ifft_ptr);
              std::span<T_Complex> result_complex_span(
                  result_ptr, current_fft_len);
              status = plan_ptr->ifft(product_const_span, result_complex_span);
            }
            else if constexpr (std::is_same_v<
                                   PlanPtrT,
                                   std::unique_ptr<
                                       OneMKLRFFTPlanImpl<T_Real>>>) {
              status = plan_ptr->irfft(product_const_span, result_ifft_span);
            }
            // status is now set by the IFFT/IRFFT call
          }
          else {
            status = Status::InvalidOperation;
          }
        },
        fft_plan_impl_variant_);

    if (status != Status::Success) return status;

    // --- Extract Correct Output based on Type ---
    // (Extraction logic remains the same as default/convolution.cpp)
    size_t full_start_idx = 0;
    size_t count = output_len_expected;

    switch (type_) {
      case ConvolutionType::Full:
        full_start_idx = 0;
        count = input_len + kernel_length_ - 1;
        break;
      case ConvolutionType::Same:
        full_start_idx = (kernel_length_ - 1) / 2;
        count = input_len;
        break;
      case ConvolutionType::Valid:
        full_start_idx = 0;  // Valid correlation starts at index 0
        count = (input_len >= kernel_length_) ? (input_len - kernel_length_ + 1)
                                              : 0;
        break;
      default:
        return Status::InvalidArgument;
    }

    assert(count == output_len_expected);

    size_t linear_corr_len = input_len + kernel_length_ - 1;
    if (full_start_idx >= linear_corr_len) {
      count = 0;
    }
    else if (full_start_idx + count > linear_corr_len) {
      count = linear_corr_len - full_start_idx;
    }

    count = std::min(count, output.size());

    if (count > 0) {
      if (full_start_idx + count <= current_fft_len) {
        std::memcpy(
            output.data(),
            result_ifft_.get() + full_start_idx,
            count * sizeof(T));
      }
      else {
        size_t safe_count = (full_start_idx < current_fft_len)
                                ? (current_fft_len - full_start_idx)
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
              T{0});
        }
        std::cerr << "Warning: Correlation result truncated due to "
                     "insufficient FFT length or indexing issue."
                  << std::endl;
      }
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

  // Getters
  template <typename T>
  size_t OneMKLCorrelationPlanImpl<T>::get_template_length() const
  {
    return kernel_length_;
  }
  template <typename T>
  ConvolutionType OneMKLCorrelationPlanImpl<T>::get_type() const
  {
    return type_;
  }
  template <typename T>
  ConvolutionMethod OneMKLCorrelationPlanImpl<T>::get_method() const
  {
    return method_;
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
  template <typename T>
  std::span<const T> OneMKLCorrelationPlanImpl<T>::get_template() const
  {
    return std::span<const T>(original_kernel_);
  }

  template <typename T>
  ConvolutionMethod OneMKLConvolutionPlanImpl<T>::get_method() const
  {
    return method_;
  }

  template <typename T>
  ConvolutionMethod OneMKLCorrelationPlanImpl<T>::get_method() const
  {
    return method_;
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  // OneMKLConvolutionPlanImpl Instantiations
  template class OmniDSP::backend::OneMKLConvolutionPlanImpl<F32>;
  template class OmniDSP::backend::OneMKLConvolutionPlanImpl<F64>;
  template class OmniDSP::backend::OneMKLConvolutionPlanImpl<C32>;
  template class OmniDSP::backend::OneMKLConvolutionPlanImpl<C64>;

  // OneMKLCorrelationPlanImpl Instantiations
  template class OmniDSP::backend::OneMKLCorrelationPlanImpl<F32>;
  template class OmniDSP::backend::OneMKLCorrelationPlanImpl<F64>;
  template class OmniDSP::backend::OneMKLCorrelationPlanImpl<C32>;
  template class OmniDSP::backend::OneMKLCorrelationPlanImpl<C64>;

}  // namespace OmniDSP::backend

#endif  // OMNIDSP_USE_ONEMKL
