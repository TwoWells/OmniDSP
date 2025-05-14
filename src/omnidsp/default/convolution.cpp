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
#include <OmniDSP/core_types.hpp>  // For Status, F32, C32, etc., Utils::*
#include <OmniDSP/params/convolution.hpp>  // For Params::Convolution, Params::Correlation

#include "../interface/backend.hpp"  // For AbstractBackend and FFTPlanImpl base classes

namespace OmniDSP::Default {

  //--------------------------------------------------------------------------
  // Helper Functions (Standard C++) - Kept from original file
  //--------------------------------------------------------------------------
  namespace convolution_detail {  // Internal namespace

    // Calculates the next power of two >= n.
    // Returns 0 if n is 0 or if the next power of two would overflow size_t.
    inline size_t next_power_of_two(size_t n)
    {
      if (n == 0)
        return 1;  // FFT of length 0 is usually not meaningful, min length
                   // often 1 or 2
      if ((n > 0) && ((n & (n - 1)) == 0)) {
        return n;
      }
      if (n > (std::numeric_limits<size_t>::max() / 2U)) {
        if (n == (~(std::numeric_limits<size_t>::max() >> 1))) return n;
        // spdlog is not available here, consider alternative logging or error
        // propagation if needed
        return 0;  // Indicate overflow otherwise
      }
      size_t power = 1;
      while (power < n) {
        if (power > (std::numeric_limits<size_t>::max() / 2U)) {
          return 0;  // Overflow
        }
        power <<= 1;
      }
      return power;
    }

    // Helper for element-wise complex multiplication (Standard C++)
    template <typename T_Complex_Func_Arg>
    void complex_multiply(
        std::span<const T_Complex_Func_Arg> a,
        std::span<const T_Complex_Func_Arg> b,
        std::span<T_Complex_Func_Arg> out)
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
    // C++) out = a * conj(b) This function is not used in the provided code,
    // but kept for completeness if needed later.
    template <typename T_Complex_Func_Arg>
    void complex_multiply_conj(
        std::span<const T_Complex_Func_Arg> a,
        std::span<const T_Complex_Func_Arg> b,
        std::span<T_Complex_Func_Arg> out)
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

  template <typename T>
  ConvolutionPlanImpl<T>::ConvolutionPlanImpl(
      FFTPlanImplVariant&& fft_plan_variant,
      const Params::Convolution& params,
      std::span<const T> kernel_coeffs)
      : type_(params.type_),
        method_hint_(params.method_hint_),
        original_kernel_(kernel_coeffs.begin(), kernel_coeffs.end()),
        kernel_length_(kernel_coeffs.size()),
        fft_length_(0),  // Will be set based on the provided FFT plan
        max_input_length_(params.max_input_length_),
        fft_plan_impl_variant_(std::move(fft_plan_variant))
  {
    if (kernel_length_ == 0
        && params.max_input_length_
               > 0) {  // Allow empty kernel if max_input_length is also 0
                       // (vacuously true convolution)
      // If kernel is empty but we expect input, this is usually an issue or
      // results in all zeros. Depending on strictness, this could be an error
      // or handled by execute returning zeros. For now, allow construction,
      // execute will handle output.
    }
    if (kernel_length_ == 0 && params.max_input_length_ == 0) {
      // Both kernel and max input are zero, effectively a no-op.
      // fft_length_ will remain 0, buffers won't be allocated.
      // This is acceptable; execute will handle it.
    }

    using T_Complex_Local
        = Utils::GetComplexType<T>;  // Use a distinct name to avoid ambiguity
    using T_Real_Local = Utils::GetRealType<T>;
    using FFTPlanPtr = std::unique_ptr<Abstract::FFTPlanImpl<T_Complex_Local>>;
    using RFFTPlanPtr = std::unique_ptr<Abstract::RFFTPlanImpl<T_Real_Local>>;

    Status fft_status = Status::Failure;
    size_t kernel_fft_size = 0;

    // Determine fft_length_ from the provided plan
    if constexpr (Utils::IsComplex_v<T>) {
      const auto* fft_plan_ptr_ptr
          = std::get_if<FFTPlanPtr>(&fft_plan_impl_variant_);
      if (!fft_plan_ptr_ptr || !*fft_plan_ptr_ptr) {
        if (method_hint_ == ConvolutionMethod::FFT
            || (method_hint_ == ConvolutionMethod::Auto
                && kernel_length_ > 0)) {  // FFT needed but no plan
          throw std::invalid_argument(
              "Null or incorrect FFT plan type provided for complex "
              "convolution when FFT method is indicated.");
        }
        // If method is Direct or Auto (and kernel is empty), no FFT plan is
        // strictly needed by this constructor. fft_length_ will remain 0. The
        // execute method must handle this.
      }
      else {
        fft_length_ = (*fft_plan_ptr_ptr)->get_length();
      }
    }
    else {  // T is Real
      const auto* rfft_plan_ptr_ptr
          = std::get_if<RFFTPlanPtr>(&fft_plan_impl_variant_);
      if (!rfft_plan_ptr_ptr || !*rfft_plan_ptr_ptr) {
        if (method_hint_ == ConvolutionMethod::FFT
            || (method_hint_ == ConvolutionMethod::Auto
                && kernel_length_ > 0)) {
          throw std::invalid_argument(
              "Null or incorrect FFT plan type provided for real convolution "
              "when FFT method is indicated.");
        }
      }
      else {
        fft_length_ = (*rfft_plan_ptr_ptr)->get_length();
      }
    }

    // If fft_length_ is still 0 but FFT is hinted and lengths are non-zero,
    // it's an issue. This should be caught by the backend when creating the FFT
    // plan variant if params.max_input_length_ was used. Here, we primarily
    // validate the fft_length_ obtained *from* the variant.
    if (method_hint_ == ConvolutionMethod::FFT && fft_length_ == 0
        && (max_input_length_ > 0 || kernel_length_ > 0)) {
      throw std::invalid_argument(
          "Provided FFT plan has zero length, but FFT method was specified for "
          "non-empty operation.");
    }

    size_t min_required_fft_length = 0;
    if (max_input_length_ > 0
        || kernel_length_
               > 0) {  // Only calculate if there's something to convolve
      min_required_fft_length = max_input_length_ + kernel_length_ - 1;
      if (min_required_fft_length == 0
          && (max_input_length_ > 0 || kernel_length_ > 0)) {
        min_required_fft_length = std::max(max_input_length_, kernel_length_);
      }
    }

    if (fft_length_ > 0
        && fft_length_ < min_required_fft_length) {  // Only check if fft_length
                                                     // is positive
      throw std::invalid_argument(
          "Provided FFT plan length (" + std::to_string(fft_length_)
          + ") is too small for the operation. Minimum required: "
          + std::to_string(min_required_fft_length)
          + " (based on max_input_length: " + std::to_string(max_input_length_)
          + ", kernel_length: " + std::to_string(kernel_length_) + ").");
    }

    if (fft_length_ > 0
        && kernel_length_
               > 0) {  // Only proceed with kernel FFT if lengths are valid
      std::vector<T> padded_reversed_kernel = original_kernel_;
      std::reverse(
          padded_reversed_kernel.begin(), padded_reversed_kernel.end());
      padded_reversed_kernel.resize(fft_length_, T{});

      if constexpr (Utils::IsComplex_v<T>) {
        const auto& plan_impl_ptr_variant
            = std::get_if<FFTPlanPtr>(&fft_plan_impl_variant_);
        if (!plan_impl_ptr_variant
            || !(*plan_impl_ptr_variant)) { /* Handled by earlier checks if FFT
                                               was required */
        }
        else {
          const auto& plan_impl_ptr = *plan_impl_ptr_variant;
          kernel_fft_size = fft_length_;
          kernel_fft_.resize(kernel_fft_size);
          const T_Complex_Local* kernel_input_ptr
              = reinterpret_cast<const T_Complex_Local*>(
                  padded_reversed_kernel.data());
          std::span<const T_Complex_Local> kernel_input_span(
              kernel_input_ptr, fft_length_);
          fft_status = plan_impl_ptr->fft(
              kernel_input_span,
              std::span<T_Complex_Local>(kernel_fft_.data(), kernel_fft_size));
        }
      }
      else {  // T is Real
        const auto& plan_impl_ptr_variant
            = std::get_if<RFFTPlanPtr>(&fft_plan_impl_variant_);
        if (!plan_impl_ptr_variant || !(*plan_impl_ptr_variant)) { /* Handled */
        }
        else {
          const auto& plan_impl_ptr = *plan_impl_ptr_variant;
          if (fft_length_ < 2) {  // RFFT of length 0 or 1 is problematic
            throw std::invalid_argument(
                "RFFT plan length must be >= 2 for non-empty kernel.");
          }
          kernel_fft_size = fft_length_ / 2 + 1;
          kernel_fft_.resize(kernel_fft_size);
          const T_Real_Local* kernel_input_ptr = padded_reversed_kernel.data();
          std::span<const T_Real_Local> kernel_input_span(
              kernel_input_ptr, fft_length_);
          fft_status = plan_impl_ptr->rfft(
              kernel_input_span,
              std::span<T_Complex_Local>(kernel_fft_.data(), kernel_fft_size));
        }
      }

      if (fft_status != Status::Success) {
        throw std::runtime_error(
            "Failed FFT of convolution kernel. Status: "
            + std::to_string(static_cast<int>(fft_status)));
      }
    }
    else if (kernel_length_ > 0 && method_hint_ == ConvolutionMethod::FFT) {
      // Kernel exists, FFT was hinted, but fft_length ended up 0 (e.g. from bad
      // FFT plan from backend)
      throw std::runtime_error(
          "Cannot perform FFT-based convolution: FFT length is zero despite "
          "non-empty kernel and FFT hint.");
    }

    try {
      if (fft_length_ > 0) {
        input_padded_ = std::make_unique<T[]>(fft_length_);
        input_fft_ = std::make_unique<T_Complex_Local[]>(
            kernel_fft_size);  // Use kernel_fft_size which matches FFT output
        product_fft_ = std::make_unique<T_Complex_Local[]>(kernel_fft_size);
        result_ifft_ = std::make_unique<T[]>(fft_length_);

        if (!input_padded_ || !input_fft_ || !product_fft_ || !result_ifft_) {
          throw std::runtime_error(
              "Failed to allocate temporary buffers for convolution plan "
              "(unexpected null).");
        }
      }
    }
    catch (const std::bad_alloc& e) {
      throw std::runtime_error(
          "Failed to allocate temporary buffers for convolution plan: "
          + std::string(e.what()));
    }
  }

  template <typename T>
  Status ConvolutionPlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output) const
  {
    const size_t signal_len = input.size();
    const size_t output_len_expected = get_output_length(signal_len);

    if (output.size() < output_len_expected) {
      return Status::SizeMismatch;
    }

    if (signal_len > max_input_length_ && max_input_length_ > 0) {
      return Status::InvalidArgument;
    }

    using T_Complex_Local = Utils::GetComplexType<T>;
    using T_Real_Local = Utils::GetRealType<T>;
    using FFTPlanPtr = std::unique_ptr<Abstract::FFTPlanImpl<T_Complex_Local>>;
    using RFFTPlanPtr = std::unique_ptr<Abstract::RFFTPlanImpl<T_Real_Local>>;

    if ((signal_len == 0 && kernel_length_ == 0)
        || (type_ == ConvolutionType::Valid && signal_len < kernel_length_)) {
      std::fill(output.begin(), output.begin() + output_len_expected, T{});
      if (output.size() > output_len_expected) {
        std::fill(output.begin() + output_len_expected, output.end(), T{});
      }
      return Status::Success;
    }
    if (kernel_length_ == 0) {  // If kernel is empty, output is all zeros (or
                                // depends on definition for 'full'/'same')
      std::fill(output.begin(), output.begin() + output_len_expected, T{});
      if (output.size() > output_len_expected) {
        std::fill(output.begin() + output_len_expected, output.end(), T{});
      }
      return Status::Success;
    }

    size_t required_fft_len_for_current_input = signal_len + kernel_length_ - 1;
    if (required_fft_len_for_current_input == 0
        && (signal_len > 0 || kernel_length_ > 0)) {
      required_fft_len_for_current_input = std::max(signal_len, kernel_length_);
    }

    if (fft_length_ < required_fft_len_for_current_input) {
      return Status::InvalidArgument;
    }

    if (fft_length_ == 0) {
      // This case implies direct convolution should happen, or it's an error if
      // FFT was expected. For now, assuming if fft_length_ is 0, it's a no-op
      // or direct (not implemented here). If method_hint_ was FFT, constructor
      // should have errored or setup fft_length_.
      if (method_hint_ == ConvolutionMethod::FFT)
        return Status::InvalidOperation;  // FFT plan not available
      // Fallback to direct or error for Auto if direct not implemented
      return Status::NotImplemented;  // Direct convolution not implemented here
    }
    if (!input_padded_ || !input_fft_ || !product_fft_ || !result_ifft_) {
      return Status::NotInitialized;  // Buffers not allocated
    }

    std::memcpy(input_padded_.get(), input.data(), signal_len * sizeof(T));
    std::fill(
        input_padded_.get() + signal_len,
        input_padded_.get() + fft_length_,
        T{});

    size_t input_fft_size
        = kernel_fft_
              .size();  // This is the size of the complex FFT output array
    std::span<T_Complex_Local> input_fft_span(input_fft_.get(), input_fft_size);
    std::span<T_Complex_Local> product_fft_span(
        product_fft_.get(), input_fft_size);  // Same size
    std::span<T> result_ifft_span(result_ifft_.get(), fft_length_);

    Status status = Status::Failure;

    if constexpr (Utils::IsComplex_v<T>) {
      const auto* plan_ptr_variant
          = std::get_if<FFTPlanPtr>(&fft_plan_impl_variant_);
      if (!plan_ptr_variant || !(*plan_ptr_variant))
        return Status::NotInitialized;  // FFT plan not available
      const auto& plan_impl_ptr = *plan_ptr_variant;

      std::span<const T_Complex_Local> input_padded_complex_span(
          reinterpret_cast<const T_Complex_Local*>(input_padded_.get()),
          fft_length_);
      status = plan_impl_ptr->fft(input_padded_complex_span, input_fft_span);
      if (status != Status::Success) return status;

      convolution_detail::complex_multiply<
          T_Complex_Local>(  // Explicit template argument
          input_fft_span,    // Implicitly converts to std::span<const
                             // T_Complex_Local>
          std::span<const T_Complex_Local>(
              kernel_fft_.data(), kernel_fft_.size()),
          product_fft_span);

      std::span<T_Complex_Local> result_complex_span(
          reinterpret_cast<T_Complex_Local*>(result_ifft_.get()), fft_length_);
      status = plan_impl_ptr->ifft(product_fft_span, result_complex_span);
      if (status != Status::Success) return status;
    }
    else {  // T is Real
      const auto* plan_ptr_variant
          = std::get_if<RFFTPlanPtr>(&fft_plan_impl_variant_);
      if (!plan_ptr_variant || !(*plan_ptr_variant))
        return Status::NotInitialized;
      const auto& plan_impl_ptr = *plan_ptr_variant;

      std::span<const T_Real_Local> input_padded_real_span(
          input_padded_.get(), fft_length_);
      status = plan_impl_ptr->rfft(input_padded_real_span, input_fft_span);
      if (status != Status::Success) return status;

      convolution_detail::complex_multiply<
          T_Complex_Local>(  // Explicit template argument
          input_fft_span,    // Implicitly converts to std::span<const
                             // T_Complex_Local>
          std::span<const T_Complex_Local>(
              kernel_fft_.data(), kernel_fft_.size()),
          product_fft_span);

      status = plan_impl_ptr->irfft(product_fft_span, result_ifft_span);
      if (status != Status::Success) return status;
    }

    size_t full_start_idx = 0;
    size_t count = output_len_expected;

    switch (type_) {
      case ConvolutionType::Full:
        full_start_idx = 0;
        break;
      case ConvolutionType::Same:
        full_start_idx = (kernel_length_ > 0) ? (kernel_length_ - 1) / 2 : 0;
        break;
      case ConvolutionType::Valid:
        full_start_idx = (kernel_length_ > 0) ? (kernel_length_ - 1) : 0;
        break;
      default:
        return Status::InvalidArgument;
    }

    size_t linear_conv_len = (signal_len > 0 && kernel_length_ > 0)
                                 ? (signal_len + kernel_length_ - 1)
                                 : (signal_len == 0 && kernel_length_ == 0
                                        ? 0
                                        : std::max(signal_len, kernel_length_));

    if (full_start_idx >= linear_conv_len && count > 0) {
      count = 0;
    }
    else if (full_start_idx + count > linear_conv_len) {
      count = linear_conv_len - full_start_idx;
    }

    count = std::min(count, output.size());
    count = std::min(count, output_len_expected);

    if (count > 0) {
      if (full_start_idx < fft_length_
          && full_start_idx + count <= fft_length_) {
        std::memcpy(
            output.data(),
            result_ifft_.get() + full_start_idx,
            count * sizeof(T));
      }
      else {
        std::fill(output.begin(), output.begin() + count, T{});
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
    return method_hint_;
  }

  template <typename T>
  size_t ConvolutionPlanImpl<T>::get_output_length(size_t input_length) const
  {
    if (kernel_length_ == 0 && input_length == 0) return 0;
    if (kernel_length_ == 0) {  // Kernel is empty
      if (type_ == ConvolutionType::Full || type_ == ConvolutionType::Same)
        return input_length;
      return 0;  // Valid
    }
    if (input_length == 0) {  // Input is empty
      if (type_ == ConvolutionType::Full)
        return kernel_length_ - 1;  // Or kernel_length_ if definition differs
      return 0;                     // Same or Valid
    }

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
        throw std::logic_error("Invalid ConvolutionType in get_output_length");
    }
  }

  template <typename T>
  std::span<const T> ConvolutionPlanImpl<T>::get_kernel() const
  {
    return std::span<const T>(original_kernel_);
  }

  //--------------------------------------------------------------------------
  // CorrelationPlanImpl Method Definitions
  //--------------------------------------------------------------------------

  template <typename T>
  CorrelationPlanImpl<T>::CorrelationPlanImpl(
      FFTPlanImplVariant&& fft_plan_variant,
      const Params::Correlation& params,
      std::span<const T> template_coeffs)
      : type_(params.type_),
        method_hint_(params.method_hint_),
        original_template_(template_coeffs.begin(), template_coeffs.end()),
        template_length_(template_coeffs.size()),
        fft_length_(0),
        max_input_length_(params.max_input_length_),
        fft_plan_impl_variant_(std::move(fft_plan_variant))
  {
    if (template_length_ == 0 && params.max_input_length_ > 0) {
      // Allow construction, execute will handle output.
    }
    if (template_length_ == 0 && params.max_input_length_ == 0) {
      // No-op, fft_length_ remains 0.
    }

    using T_Complex_Local = Utils::GetComplexType<T>;
    using T_Real_Local = Utils::GetRealType<T>;
    using FFTPlanPtr = std::unique_ptr<Abstract::FFTPlanImpl<T_Complex_Local>>;
    using RFFTPlanPtr = std::unique_ptr<Abstract::RFFTPlanImpl<T_Real_Local>>;

    Status fft_status = Status::Failure;
    size_t template_fft_size = 0;

    if constexpr (Utils::IsComplex_v<T>) {
      const auto* fft_plan_ptr_ptr
          = std::get_if<FFTPlanPtr>(&fft_plan_impl_variant_);
      if (!fft_plan_ptr_ptr || !*fft_plan_ptr_ptr) {
        if (method_hint_ == ConvolutionMethod::FFT
            || (method_hint_ == ConvolutionMethod::Auto
                && template_length_ > 0)) {
          throw std::invalid_argument(
              "Null or incorrect FFT plan type provided for complex "
              "correlation when FFT method is indicated.");
        }
      }
      else {
        fft_length_ = (*fft_plan_ptr_ptr)->get_length();
      }
    }
    else {  // T is Real
      const auto* rfft_plan_ptr_ptr
          = std::get_if<RFFTPlanPtr>(&fft_plan_impl_variant_);
      if (!rfft_plan_ptr_ptr || !*rfft_plan_ptr_ptr) {
        if (method_hint_ == ConvolutionMethod::FFT
            || (method_hint_ == ConvolutionMethod::Auto
                && template_length_ > 0)) {
          throw std::invalid_argument(
              "Null or incorrect FFT plan type provided for real correlation "
              "when FFT method is indicated.");
        }
      }
      else {
        fft_length_ = (*rfft_plan_ptr_ptr)->get_length();
      }
    }
    if (method_hint_ == ConvolutionMethod::FFT && fft_length_ == 0
        && (max_input_length_ > 0 || template_length_ > 0)) {
      throw std::invalid_argument(
          "Provided FFT plan has zero length, but FFT method was specified for "
          "non-empty operation.");
    }

    size_t min_required_fft_length = 0;
    if (max_input_length_ > 0 || template_length_ > 0) {
      min_required_fft_length = max_input_length_ + template_length_ - 1;
      if (min_required_fft_length == 0
          && (max_input_length_ > 0 || template_length_ > 0)) {
        min_required_fft_length = std::max(max_input_length_, template_length_);
      }
    }

    if (fft_length_ > 0 && fft_length_ < min_required_fft_length) {
      throw std::invalid_argument(
          "Provided FFT plan length (" + std::to_string(fft_length_)
          + ") is too small for the operation. Minimum required: "
          + std::to_string(min_required_fft_length)
          + " (based on max_input_length: " + std::to_string(max_input_length_)
          + ", template_length: " + std::to_string(template_length_) + ").");
    }

    if (fft_length_ > 0 && template_length_ > 0) {
      std::vector<T> padded_template = original_template_;
      padded_template.resize(
          fft_length_, T{});  // NO reverse for correlation template

      std::vector<T_Complex_Local> temp_template_fft_buffer;

      if constexpr (Utils::IsComplex_v<T>) {
        const auto& plan_impl_ptr_variant
            = std::get_if<FFTPlanPtr>(&fft_plan_impl_variant_);
        if (!plan_impl_ptr_variant || !(*plan_impl_ptr_variant)) { /* Handled */
        }
        else {
          const auto& plan_impl_ptr = *plan_impl_ptr_variant;
          template_fft_size = fft_length_;
          temp_template_fft_buffer.resize(template_fft_size);
          const T_Complex_Local* template_input_ptr
              = reinterpret_cast<const T_Complex_Local*>(
                  padded_template.data());
          std::span<const T_Complex_Local> template_input_span(
              template_input_ptr, fft_length_);
          fft_status = plan_impl_ptr->fft(
              template_input_span,
              std::span<T_Complex_Local>(
                  temp_template_fft_buffer.data(), template_fft_size));
        }
      }
      else {  // T is Real
        const auto& plan_impl_ptr_variant
            = std::get_if<RFFTPlanPtr>(&fft_plan_impl_variant_);
        if (!plan_impl_ptr_variant || !(*plan_impl_ptr_variant)) { /* Handled */
        }
        else {
          const auto& plan_impl_ptr = *plan_impl_ptr_variant;
          if (fft_length_ < 2) {
            throw std::invalid_argument(
                "RFFT plan length must be >= 2 for non-empty template.");
          }
          template_fft_size = fft_length_ / 2 + 1;
          temp_template_fft_buffer.resize(template_fft_size);
          const T_Real_Local* template_input_ptr = padded_template.data();
          std::span<const T_Real_Local> template_input_span(
              template_input_ptr, fft_length_);
          fft_status = plan_impl_ptr->rfft(
              template_input_span,
              std::span<T_Complex_Local>(
                  temp_template_fft_buffer.data(), template_fft_size));
        }
      }

      if (fft_status != Status::Success) {
        throw std::runtime_error(
            "Failed FFT of correlation template. Status: "
            + std::to_string(static_cast<int>(fft_status)));
      }

      template_fft_conj_.resize(template_fft_size);
      for (size_t i = 0; i < template_fft_size; ++i) {
        template_fft_conj_[i] = std::conj(temp_template_fft_buffer[i]);
      }
    }
    else if (template_length_ > 0 && method_hint_ == ConvolutionMethod::FFT) {
      throw std::runtime_error(
          "Cannot perform FFT-based correlation: FFT length is zero despite "
          "non-empty template and FFT hint.");
    }

    try {
      if (fft_length_ > 0) {
        input_padded_ = std::make_unique<T[]>(fft_length_);
        input_fft_ = std::make_unique<T_Complex_Local[]>(
            template_fft_size);  // Use template_fft_size
        product_fft_ = std::make_unique<T_Complex_Local[]>(template_fft_size);
        result_ifft_ = std::make_unique<T[]>(fft_length_);
        if (!input_padded_ || !input_fft_ || !product_fft_ || !result_ifft_) {
          throw std::runtime_error(
              "Failed to allocate temporary buffers for correlation plan "
              "(unexpected null).");
        }
      }
    }
    catch (const std::bad_alloc& e) {
      throw std::runtime_error(
          "Failed to allocate temporary buffers for correlation plan: "
          + std::string(e.what()));
    }
  }

  template <typename T>
  Status CorrelationPlanImpl<T>::execute(
      std::span<const T> input, std::span<T> output) const
  {
    const size_t signal_len = input.size();
    const size_t output_len_expected = get_output_length(signal_len);

    if (output.size() < output_len_expected) {
      return Status::SizeMismatch;
    }
    if (signal_len > max_input_length_ && max_input_length_ > 0) {
      return Status::InvalidArgument;
    }

    using T_Complex_Local = Utils::GetComplexType<T>;
    using T_Real_Local = Utils::GetRealType<T>;
    using FFTPlanPtr = std::unique_ptr<Abstract::FFTPlanImpl<T_Complex_Local>>;
    using RFFTPlanPtr = std::unique_ptr<Abstract::RFFTPlanImpl<T_Real_Local>>;

    if ((signal_len == 0 && template_length_ == 0)
        || (type_ == ConvolutionType::Valid && signal_len < template_length_)) {
      std::fill(output.begin(), output.begin() + output_len_expected, T{});
      if (output.size() > output_len_expected) {
        std::fill(output.begin() + output_len_expected, output.end(), T{});
      }
      return Status::Success;
    }
    if (template_length_ == 0) {
      std::fill(output.begin(), output.begin() + output_len_expected, T{});
      if (output.size() > output_len_expected) {
        std::fill(output.begin() + output_len_expected, output.end(), T{});
      }
      return Status::Success;
    }

    size_t required_fft_len_for_current_input
        = signal_len + template_length_ - 1;
    if (required_fft_len_for_current_input == 0
        && (signal_len > 0 || template_length_ > 0)) {
      required_fft_len_for_current_input
          = std::max(signal_len, template_length_);
    }

    if (fft_length_ < required_fft_len_for_current_input) {
      return Status::InvalidArgument;
    }
    if (fft_length_ == 0) {
      if (method_hint_ == ConvolutionMethod::FFT)
        return Status::InvalidOperation;
      return Status::NotImplemented;
    }
    if (!input_padded_ || !input_fft_ || !product_fft_ || !result_ifft_) {
      return Status::NotInitialized;
    }

    std::memcpy(input_padded_.get(), input.data(), signal_len * sizeof(T));
    std::fill(
        input_padded_.get() + signal_len,
        input_padded_.get() + fft_length_,
        T{});

    size_t input_fft_size
        = template_fft_conj_.size();  // Size of complex FFT output
    std::span<T_Complex_Local> input_fft_span(input_fft_.get(), input_fft_size);
    std::span<T_Complex_Local> product_fft_span(
        product_fft_.get(), input_fft_size);
    std::span<T> result_ifft_span(result_ifft_.get(), fft_length_);

    Status status = Status::Failure;

    if constexpr (Utils::IsComplex_v<T>) {
      const auto* plan_ptr_variant
          = std::get_if<FFTPlanPtr>(&fft_plan_impl_variant_);
      if (!plan_ptr_variant || !(*plan_ptr_variant))
        return Status::NotInitialized;
      const auto& plan_impl_ptr = *plan_ptr_variant;

      std::span<const T_Complex_Local> input_padded_complex_span(
          reinterpret_cast<const T_Complex_Local*>(input_padded_.get()),
          fft_length_);
      status = plan_impl_ptr->fft(input_padded_complex_span, input_fft_span);
      if (status != Status::Success) return status;

      convolution_detail::complex_multiply<
          T_Complex_Local>(  // Explicit template argument
          input_fft_span,    // Implicitly converts to std::span<const
                             // T_Complex_Local>
          std::span<const T_Complex_Local>(
              template_fft_conj_.data(), template_fft_conj_.size()),
          product_fft_span);

      std::span<T_Complex_Local> result_complex_span(
          reinterpret_cast<T_Complex_Local*>(result_ifft_.get()), fft_length_);
      status = plan_impl_ptr->ifft(product_fft_span, result_complex_span);
      if (status != Status::Success) return status;
    }
    else {  // T is Real
      const auto* plan_ptr_variant
          = std::get_if<RFFTPlanPtr>(&fft_plan_impl_variant_);
      if (!plan_ptr_variant || !(*plan_ptr_variant))
        return Status::NotInitialized;
      const auto& plan_impl_ptr = *plan_ptr_variant;

      std::span<const T_Real_Local> input_padded_real_span(
          input_padded_.get(), fft_length_);
      status = plan_impl_ptr->rfft(input_padded_real_span, input_fft_span);
      if (status != Status::Success) return status;

      convolution_detail::complex_multiply<
          T_Complex_Local>(  // Explicit template argument
          input_fft_span,    // Implicitly converts to std::span<const
                             // T_Complex_Local>
          std::span<const T_Complex_Local>(
              template_fft_conj_.data(), template_fft_conj_.size()),
          product_fft_span);

      status = plan_impl_ptr->irfft(product_fft_span, result_ifft_span);
      if (status != Status::Success) return status;
    }

    size_t full_start_idx = 0;
    size_t count = output_len_expected;

    switch (type_) {
      case ConvolutionType::Full:
        full_start_idx = 0;
        break;
      case ConvolutionType::Same:
        full_start_idx
            = (template_length_ > 0) ? (template_length_ - 1) / 2 : 0;
        break;
      case ConvolutionType::Valid:
        full_start_idx = (template_length_ > 0) ? (template_length_ - 1) : 0;
        if (signal_len < template_length_) full_start_idx = 0;
        break;
      default:
        return Status::InvalidArgument;
    }

    size_t linear_corr_len
        = (signal_len > 0 && template_length_ > 0)
              ? (signal_len + template_length_ - 1)
              : (signal_len == 0 && template_length_ == 0
                     ? 0
                     : std::max(signal_len, template_length_));

    if (full_start_idx >= linear_corr_len && count > 0) {
      count = 0;
    }
    else if (full_start_idx + count > linear_corr_len) {
      count = linear_corr_len - full_start_idx;
    }

    count = std::min(count, output.size());
    count = std::min(count, output_len_expected);

    if (count > 0) {
      if (full_start_idx < fft_length_
          && full_start_idx + count <= fft_length_) {
        std::memcpy(
            output.data(),
            result_ifft_.get() + full_start_idx,
            count * sizeof(T));
      }
      else {
        std::fill(output.begin(), output.begin() + count, T{});
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

  template <typename T>
  size_t CorrelationPlanImpl<T>::get_template_length() const
  {
    return template_length_;
  }

  template <typename T>
  ConvolutionType CorrelationPlanImpl<T>::get_type() const
  {
    return type_;
  }

  template <typename T>
  ConvolutionMethod CorrelationPlanImpl<T>::get_method() const
  {
    return method_hint_;
  }

  template <typename T>
  size_t CorrelationPlanImpl<T>::get_output_length(size_t input_length) const
  {
    if (template_length_ == 0 && input_length == 0) return 0;
    if (template_length_ == 0) {
      if (type_ == ConvolutionType::Full || type_ == ConvolutionType::Same)
        return input_length;
      return 0;
    }
    if (input_length == 0) {
      if (type_ == ConvolutionType::Full)
        return template_length_ - 1;  // Or template_length_
      return 0;
    }

    switch (type_) {
      case ConvolutionType::Full:
        if (input_length
            > std::numeric_limits<size_t>::max() - template_length_ + 1) {
          throw std::length_error(
              "Output length calculation overflow (Full mode)");
        }
        return input_length + template_length_ - 1;
      case ConvolutionType::Same:
        return input_length;
      case ConvolutionType::Valid:
        return (input_length >= template_length_)
                   ? (input_length - template_length_ + 1)
                   : 0;
      default:
        throw std::logic_error("Invalid ConvolutionType in get_output_length");
    }
  }

  template <typename T>
  std::span<const T> CorrelationPlanImpl<T>::get_template() const
  {
    return std::span<const T>(original_template_);
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  template class ConvolutionPlanImpl<F32>;
  template class ConvolutionPlanImpl<F64>;
  template class ConvolutionPlanImpl<C32>;
  template class ConvolutionPlanImpl<C64>;

  template class CorrelationPlanImpl<F32>;
  template class CorrelationPlanImpl<F64>;
  template class CorrelationPlanImpl<C32>;
  template class CorrelationPlanImpl<C64>;

}  // namespace OmniDSP::Default
