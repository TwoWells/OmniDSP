/**
 * @file fft.cpp (accelerate)
 * @brief Implements Accelerate backend FFTPlanImpl and RFFTPlanImpl classes
 * using vDSP DFT.
 */

#include "fft.hpp"  // Corresponding header file

#include <Accelerate/Accelerate.h>  // vDSP header

#include <cmath>     // For std::log2, std::pow
#include <iostream>  // For error logging
#include <limits>    // For numeric_limits
#include <stdexcept>
#include <vector>

// Include core types to ensure Utils::* traits are available if needed by
// helpers
#include <OmniDSP/core_types.hpp>

namespace OmniDSP::accelerate {

  //--------------------------------------------------------------------------
  // Helper Functions
  //--------------------------------------------------------------------------
  namespace {  // Anonymous namespace for internal helpers

    // Check if a length 'L' (N for complex, N/2 for real) is supported by
    // vDSP_DFT_Interleaved_CreateSetup Uses the constant defined in the header.
    bool is_vdsp_dft_setup_length_supported(size_t L_param)
    {
      if (L_param > MAX_VDSP_DFT_INTERLEAVED_SETUP_LENGTH) return false;
      if (L_param < 8 && L_param != 0)
        return false;  // Minimum supported length for f=1 is 8 (n=3)

      // Check factors f=1, 3, 5, 9, 15 and powers of 2
      for (int n = 2; n <= 12; ++n) {
        size_t p2 = static_cast<size_t>(1) << n;
        if (p2 == 0) break;  // Avoid overflow if n is too large for size_t
        if (L_param == p2 && n >= 3) return true;       // f=1 (min n=3 -> L=8)
        if (n <= 8 && L_param == 3 * p2) return true;   // f=3
        if (n <= 7 && L_param == 5 * p2) return true;   // f=5
        if (n <= 7 && L_param == 9 * p2) return true;   // f=9 (3*3)
        if (n <= 7 && L_param == 15 * p2) return true;  // f=15 (3*5)
      }
      return false;
    }

  }  // anonymous namespace

  //--------------------------------------------------------------------------
  // AccelerateFFTPlanImpl Method Definitions (Complex FFT)
  //--------------------------------------------------------------------------

  // Static length validation helper
  template <typename T>
  [[nodiscard]] bool AccelerateFFTPlanImpl<T>::is_vdsp_dft_supported_length(
      size_t N_complex)
  {
    // For complex FFT, the 'Length' parameter to CreateSetup is N_complex
    return is_vdsp_dft_setup_length_supported(N_complex);
  }

  // Constructor
  template <typename T>
  AccelerateFFTPlanImpl<T>::AccelerateFFTPlanImpl(size_t length)
      : length_(length)
  {
    if (length_ == 0) {
      fft_setup_forward_ = nullptr;
      fft_setup_inverse_ = nullptr;
      return;  // Allow zero-length plan
    }

    // Check if length is supported BEFORE trying to create setup
    if (!is_vdsp_dft_supported_length(length_)) {
      // This constructor should only be called if the length IS supported,
      // based on the logic in AccelerateBackend::create_fft_plan_c*.
      // Throwing here indicates a logic error in the factory.
      throw std::logic_error(
          "AccelerateFFTPlanImpl constructor called with unsupported length: "
          + std::to_string(length_));
    }

    using Real = typename T::value_type;
    vDSP_Length vL = static_cast<vDSP_Length>(length_);

    if constexpr (std::is_same_v<Real, float>) {
      fft_setup_forward_ = vDSP_DFT_Interleaved_CreateSetup(
          nullptr, vL, vDSP_DFT_FORWARD, vDSP_DFT_Interleaved_ComplextoComplex);
      fft_setup_inverse_ = vDSP_DFT_Interleaved_CreateSetup(
          fft_setup_forward_,
          vL,
          vDSP_DFT_INVERSE,
          vDSP_DFT_Interleaved_ComplextoComplex);
    }
    else {  // double
      fft_setup_forward_ = vDSP_DFT_Interleaved_CreateSetupD(
          nullptr, vL, vDSP_DFT_FORWARD, vDSP_DFT_Interleaved_ComplextoComplex);
      fft_setup_inverse_ = vDSP_DFT_Interleaved_CreateSetupD(
          fft_setup_forward_,
          vL,
          vDSP_DFT_INVERSE,
          vDSP_DFT_Interleaved_ComplextoComplex);
    }

    if (!fft_setup_forward_ || !fft_setup_inverse_) {
      vDSP_DFT_Interleaved_DestroySetup(fft_setup_forward_);
      vDSP_DFT_Interleaved_DestroySetup(
          fft_setup_inverse_);  // Safe even if it's same as forward or null
      fft_setup_forward_ = nullptr;
      fft_setup_inverse_ = nullptr;
      throw std::runtime_error(
          "Accelerate FFT: Failed to create vDSP_DFT_Interleaved setup.");
    }
  }

  // Destructor
  template <typename T>
  AccelerateFFTPlanImpl<T>::~AccelerateFFTPlanImpl()
  {
    if constexpr (std::is_same_v<typename T::value_type, float>) {
      vDSP_DFT_Interleaved_DestroySetup(fft_setup_forward_);
      // Don't destroy inverse if it shares memory (created with non-null
      // previous) vDSP_DFT_Interleaved_DestroySetup(fft_setup_inverse_);
    }
    else {
      vDSP_DFT_Interleaved_DestroySetupD(fft_setup_forward_);
      // vDSP_DFT_Interleaved_DestroySetupD(fft_setup_inverse_);
    }
  }

  // fft Method
  template <typename T>
  Status AccelerateFFTPlanImpl<T>::fft(
      std::span<const T> input, std::span<T> output) const
  {
    if (!fft_setup_forward_)
      return length_ == 0 ? Status::Success : Status::InvalidOperation;
    if (input.size() != length_ || output.size() != length_)
      return Status::SizeMismatch;
    if (length_ == 0) return Status::Success;

    using Real = typename T::value_type;

    if constexpr (std::is_same_v<Real, float>) {
      vDSP_DFT_Interleaved_Execute(
          fft_setup_forward_,
          reinterpret_cast<const DSPComplex*>(input.data()),
          reinterpret_cast<DSPComplex*>(output.data()));
    }
    else {
      vDSP_DFT_Interleaved_ExecuteD(
          fft_setup_forward_,
          reinterpret_cast<const DSPDoubleComplex*>(input.data()),
          reinterpret_cast<DSPDoubleComplex*>(output.data()));
    }
    return Status::Success;
  }

  // ifft Method
  template <typename T>
  Status AccelerateFFTPlanImpl<T>::ifft(
      std::span<const T> input, std::span<T> output) const
  {
    if (!fft_setup_inverse_)
      return length_ == 0 ? Status::Success : Status::InvalidOperation;
    if (input.size() != length_ || output.size() != length_)
      return Status::SizeMismatch;
    if (length_ == 0) return Status::Success;

    using Real = typename T::value_type;

    if constexpr (std::is_same_v<Real, float>) {
      vDSP_DFT_Interleaved_Execute(
          fft_setup_inverse_,
          reinterpret_cast<const DSPComplex*>(input.data()),
          reinterpret_cast<DSPComplex*>(output.data()));
    }
    else {
      vDSP_DFT_Interleaved_ExecuteD(
          fft_setup_inverse_,
          reinterpret_cast<const DSPDoubleComplex*>(input.data()),
          reinterpret_cast<DSPDoubleComplex*>(output.data()));
    }
    // NOTE: No automatic 1/N scaling by vDSP
    return Status::Success;
  }

  // get_length Method
  template <typename T>
  size_t AccelerateFFTPlanImpl<T>::get_length() const
  {
    return length_;
  }

  //--------------------------------------------------------------------------
  // AccelerateRFFTPlanImpl Method Definitions (Real FFT)
  //--------------------------------------------------------------------------

  // Static length validation helper
  template <typename T>  // T is REAL type
  [[nodiscard]] bool AccelerateRFFTPlanImpl<T>::is_vdsp_dft_supported_length(
      size_t N_real)
  {
    if (N_real < 2 || (N_real % 2 != 0))
      return false;  // N must be even and >= 2
    // For real FFT, the 'Length' parameter to CreateSetup is N/2
    return is_vdsp_dft_setup_length_supported(N_real / 2);
  }

  // Constructor
  template <typename T>  // T is REAL type
  AccelerateRFFTPlanImpl<T>::AccelerateRFFTPlanImpl(size_t length)
      : length_(length)
  {
    if (length_ == 0) {
      fft_setup_forward_ = nullptr;
      fft_setup_inverse_ = nullptr;
      return;  // Allow zero-length plan
    }
    if (!is_vdsp_dft_supported_length(length_)) {
      // This constructor should only be called if the length IS supported,
      // based on the logic in AccelerateBackend::create_rfft_plan_f*.
      throw std::logic_error(
          "AccelerateRFFTPlanImpl constructor called with unsupported length: "
          + std::to_string(length_));
    }

    vDSP_Length vL_half = static_cast<vDSP_Length>(length_ / 2);

    if constexpr (std::is_same_v<T, float>) {
      fft_setup_forward_ = vDSP_DFT_Interleaved_CreateSetup(
          nullptr,
          vL_half,
          vDSP_DFT_FORWARD,
          vDSP_DFT_Interleaved_RealtoComplex);
      fft_setup_inverse_ = vDSP_DFT_Interleaved_CreateSetup(
          fft_setup_forward_,
          vL_half,
          vDSP_DFT_INVERSE,
          vDSP_DFT_Interleaved_RealtoComplex);
    }
    else {  // double
      fft_setup_forward_ = vDSP_DFT_Interleaved_CreateSetupD(
          nullptr,
          vL_half,
          vDSP_DFT_FORWARD,
          vDSP_DFT_Interleaved_RealtoComplex);
      fft_setup_inverse_ = vDSP_DFT_Interleaved_CreateSetupD(
          fft_setup_forward_,
          vL_half,
          vDSP_DFT_INVERSE,
          vDSP_DFT_Interleaved_RealtoComplex);
    }

    if (!fft_setup_forward_ || !fft_setup_inverse_) {
      vDSP_DFT_Interleaved_DestroySetup(fft_setup_forward_);
      vDSP_DFT_Interleaved_DestroySetup(fft_setup_inverse_);
      fft_setup_forward_ = nullptr;
      fft_setup_inverse_ = nullptr;
      throw std::runtime_error(
          "Accelerate RFFT: Failed to create vDSP_DFT_Interleaved setup.");
    }
  }

  // Destructor
  template <typename T>
  AccelerateRFFTPlanImpl<T>::~AccelerateRFFTPlanImpl()
  {
    if constexpr (std::is_same_v<T, float>) {
      vDSP_DFT_Interleaved_DestroySetup(fft_setup_forward_);
      // vDSP_DFT_Interleaved_DestroySetup(fft_setup_inverse_); // Don't destroy
      // shared
    }
    else {
      vDSP_DFT_Interleaved_DestroySetupD(fft_setup_forward_);
      // vDSP_DFT_Interleaved_DestroySetupD(fft_setup_inverse_); // Don't
      // destroy shared
    }
  }

  // rfft Method
  template <typename T>  // T is REAL type
  Status AccelerateRFFTPlanImpl<T>::rfft(
      std::span<const T> input, std::span<Complex> output) const
  {
    if (!fft_setup_forward_)
      return length_ == 0 ? Status::Success : Status::InvalidOperation;
    size_t N = length_;
    size_t N_over_2 = N / 2;
    size_t output_size_expected = N_over_2 + 1;

    if (N == 0 && input.empty() && output.empty()) return Status::Success;
    if (input.size() != N || output.size() != output_size_expected) {
      return Status::SizeMismatch;
    }

    // Input is T*, Output is Complex* (packed format)
    if constexpr (std::is_same_v<T, float>) {
      vDSP_DFT_Interleaved_Execute(
          fft_setup_forward_,
          reinterpret_cast<const float*>(
              input.data()),  // Treat real input as interleaved
          reinterpret_cast<float*>(output.data())  // Output packed complex
      );
    }
    else {  // double
      vDSP_DFT_Interleaved_ExecuteD(
          fft_setup_forward_,
          reinterpret_cast<const double*>(input.data()),
          reinterpret_cast<double*>(output.data()));
    }
    return Status::Success;
  }

  // irfft Method
  template <typename T>  // T is REAL type
  Status AccelerateRFFTPlanImpl<T>::irfft(
      std::span<const Complex> input, std::span<T> output) const
  {
    if (!fft_setup_inverse_)
      return length_ == 0 ? Status::Success : Status::InvalidOperation;
    size_t N = length_;
    size_t N_over_2 = N / 2;
    size_t input_size_expected = N_over_2 + 1;

    if (N == 0 && input.empty() && output.empty()) return Status::Success;
    if (input.size() != input_size_expected || output.size() != N) {
      return Status::SizeMismatch;
    }

    // Input is Complex* (packed format), Output is T* (real)
    if constexpr (std::is_same_v<T, float>) {
      vDSP_DFT_Interleaved_Execute(
          fft_setup_inverse_,
          reinterpret_cast<const float*>(input.data()),  // Input packed complex
          reinterpret_cast<float*>(
              output.data())  // Output real (interleaved conceptually)
      );
    }
    else {  // double
      vDSP_DFT_Interleaved_ExecuteD(
          fft_setup_inverse_,
          reinterpret_cast<const double*>(input.data()),
          reinterpret_cast<double*>(output.data()));
    }
    // NOTE: No automatic 1/N scaling by vDSP
    return Status::Success;
  }

  // get_length Method
  template <typename T>  // T is REAL type
  size_t AccelerateRFFTPlanImpl<T>::get_length() const
  {
    return length_;
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  template class AccelerateFFTPlanImpl<C32>;
  template class AccelerateFFTPlanImpl<C64>;
  template class AccelerateRFFTPlanImpl<F32>;
  template class AccelerateRFFTPlanImpl<F64>;

}  // namespace OmniDSP::accelerate
