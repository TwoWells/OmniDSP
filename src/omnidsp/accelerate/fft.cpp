/**
 * @file fft.cpp (Accelerate)
 * @brief Implements Accelerate backend FFTPlanImpl and RFFTPlanImpl classes
 * using vDSP DFT.
 */

#include <OmniDSP/omnidsp_config.hpp>
#ifdef OMNIDSP_INTERNAL_USE_ACCELERATE

#include <Accelerate/Accelerate.h>  // vDSP header

#include <cmath>     // For std::log2, std::pow
#include <iostream>  // For error logging
#include <limits>    // For numeric_limits
#include <stdexcept>  // For std::runtime_error, std::logic_error, std::invalid_argument
#include <vector>  // For std::vector

#include "fft.hpp"  // Corresponding header file

// Include core types to ensure Utils::* traits are available
#include <OmniDSP/core_types.hpp>

namespace OmniDSP::Accelerate {

  //--------------------------------------------------------------------------
  // Helper Functions
  //--------------------------------------------------------------------------
  namespace {  // Anonymous namespace for internal helpers

    // Check if a length 'L' (N for complex, N/2 for real) is supported by
    // vDSP_DFT_Interleaved_CreateSetup Uses the constant defined in the header.
    bool is_vdsp_dft_setup_length_supported(size_t L_param)
    {
      if (L_param > MAX_VDSP_DFT_INTERLEAVED_SETUP_LENGTH) return false;

      if (L_param == 0) return true;  // Allow zero-length for empty plans

      // According to vDSP_DFT_Interleaved_CreateSetup documentation,
      // the Length parameter (which is N for complex FFT, or N/2 for N-point
      // real FFT) must be f * 2^(n-1) where f is 1, 3, or 5, and n >= 4.
      // Smallest L_param: f=1, n=4 => 1 * 2^3 = 8.
      if (L_param < 8) return false;

      // More precise check based on factors and powers of 2.
      // This is a simplified check; the ultimate test is if CreateSetup
      // succeeds.
      for (int n_exp = 3;
           (static_cast<size_t>(1) << n_exp) <= L_param && n_exp < 20;
           ++n_exp) {  // n-1 in formula, so n_exp = n-1
        size_t p2_n_minus_1 = static_cast<size_t>(1) << n_exp;
        if (L_param == 1 * p2_n_minus_1) return true;
        if (L_param == 3 * p2_n_minus_1) return true;
        if (L_param == 5 * p2_n_minus_1) return true;
      }
      // Fallback for other possible valid non-power-of-2 lengths if the above
      // doesn't catch all. However, the f * 2^(n-1) rule is quite specific. If
      // it's a large power of two, it's likely fine.
      if ((L_param & (L_param - 1)) == 0)
        return true;  // It's a power of 2 and >= 8

      return false;
    }

  }  // anonymous namespace

  //--------------------------------------------------------------------------
  // FFTPlanImpl Method Definitions (Complex FFT)
  //--------------------------------------------------------------------------

  // Static length validation helper
  template <typename T>
  [[nodiscard]] bool FFTPlanImpl<T>::is_vdsp_dft_supported_length(
      size_t N_complex)
  {
    // For complex FFT, the 'Length' parameter to CreateSetup is N_complex
    return is_vdsp_dft_setup_length_supported(N_complex);
  }

  // Constructor
  template <typename T>
  FFTPlanImpl<T>::FFTPlanImpl(size_t length) : length_(length)
  {
    if (length_ == 0) {
      fft_setup_forward_ = nullptr;
      fft_setup_inverse_ = nullptr;
      return;  // Allow zero-length plan
    }

    if (!is_vdsp_dft_supported_length(length_)) {
      throw std::invalid_argument(
          "Accelerate FFTPlanImpl: FFT length " + std::to_string(length_)
          + " is not supported by vDSP_DFT_Interleaved_CreateSetup.");
    }

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
      if constexpr (std::is_same_v<Real, float>) {
        if (fft_setup_forward_)
          vDSP_DFT_Interleaved_DestroySetup(fft_setup_forward_);
        if (fft_setup_inverse_ && fft_setup_inverse_ != fft_setup_forward_) {
          vDSP_DFT_Interleaved_DestroySetup(fft_setup_inverse_);
        }
        else if (!fft_setup_forward_ && fft_setup_inverse_) {
          vDSP_DFT_Interleaved_DestroySetup(fft_setup_inverse_);
        }
      }
      else {
        if (fft_setup_forward_)
          vDSP_DFT_Interleaved_DestroySetupD(fft_setup_forward_);
        if (fft_setup_inverse_ && fft_setup_inverse_ != fft_setup_forward_) {
          vDSP_DFT_Interleaved_DestroySetupD(fft_setup_inverse_);
        }
        else if (!fft_setup_forward_ && fft_setup_inverse_) {
          vDSP_DFT_Interleaved_DestroySetupD(fft_setup_inverse_);
        }
      }
      fft_setup_forward_ = nullptr;
      fft_setup_inverse_ = nullptr;
      throw std::runtime_error(
          "Accelerate FFT: Failed to create vDSP_DFT_Interleaved setup(s).");
    }
  }

  // Destructor
  template <typename T>
  FFTPlanImpl<T>::~FFTPlanImpl()
  {
    if (fft_setup_forward_) {
      if constexpr (std::is_same_v<Real, float>) {
        vDSP_DFT_Interleaved_DestroySetup(fft_setup_forward_);
      }
      else {
        vDSP_DFT_Interleaved_DestroySetupD(fft_setup_forward_);
      }
    }
  }

  // fft Method
  template <typename T>
  Status FFTPlanImpl<T>::fft(
      std::span<const T> input, std::span<T> output) const
  {
    if (!fft_setup_forward_)
      return length_ == 0 ? Status::Success : Status::InvalidOperation;
    if (input.size() != length_ || output.size() != length_)
      return Status::SizeMismatch;
    if (length_ == 0) return Status::Success;

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
  Status FFTPlanImpl<T>::ifft(
      std::span<const T> input, std::span<T> output) const
  {
    if (!fft_setup_inverse_)
      return length_ == 0 ? Status::Success : Status::InvalidOperation;
    if (input.size() != length_ || output.size() != length_)
      return Status::SizeMismatch;
    if (length_ == 0) return Status::Success;

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
    return Status::Success;
  }

  // get_length Method
  template <typename T>
  size_t FFTPlanImpl<T>::get_length() const
  {
    return length_;
  }

  //--------------------------------------------------------------------------
  // RFFTPlanImpl Method Definitions (Real FFT)
  //--------------------------------------------------------------------------

  // Static length validation helper
  template <typename T>  // T is REAL type
  [[nodiscard]] bool RFFTPlanImpl<T>::is_vdsp_dft_supported_length(
      size_t N_real)
  {
    if (N_real == 0) return true;
    if (N_real < 2)
      return false;  // Real FFT of length N implies N/2 complex FFT. N/2 must
                     // be >=1. And N/2 must meet
                     // is_vdsp_dft_setup_length_supported criteria. Smallest
                     // N/2 is 8, so smallest N_real is 16.
    if (N_real < 16) return false;
    if (N_real % 2 != 0) return false;
    return is_vdsp_dft_setup_length_supported(N_real / 2);
  }

  // Constructor
  template <typename T>  // T is REAL type
  RFFTPlanImpl<T>::RFFTPlanImpl(size_t length) : length_(length)
  {
    if (length_ == 0) {
      fft_setup_forward_ = nullptr;
      fft_setup_inverse_ = nullptr;
      temp_packed_buffer_.clear();
      return;
    }
    if (!is_vdsp_dft_supported_length(length_)) {
      throw std::invalid_argument(
          "RFFTPlanImpl: Real FFT length " + std::to_string(length_)
          + " (N/2=" + std::to_string(length_ / 2)
          + ") is not supported by vDSP_DFT_Interleaved_CreateSetup.");
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
      if constexpr (std::is_same_v<T, float>) {
        if (fft_setup_forward_)
          vDSP_DFT_Interleaved_DestroySetup(fft_setup_forward_);
        if (fft_setup_inverse_ && fft_setup_inverse_ != fft_setup_forward_) {
          vDSP_DFT_Interleaved_DestroySetup(fft_setup_inverse_);
        }
        else if (!fft_setup_forward_ && fft_setup_inverse_) {
          vDSP_DFT_Interleaved_DestroySetup(fft_setup_inverse_);
        }
      }
      else {
        if (fft_setup_forward_)
          vDSP_DFT_Interleaved_DestroySetupD(fft_setup_forward_);
        if (fft_setup_inverse_ && fft_setup_inverse_ != fft_setup_forward_) {
          vDSP_DFT_Interleaved_DestroySetupD(fft_setup_inverse_);
        }
        else if (!fft_setup_forward_ && fft_setup_inverse_) {
          vDSP_DFT_Interleaved_DestroySetupD(fft_setup_inverse_);
        }
      }
      fft_setup_forward_ = nullptr;
      fft_setup_inverse_ = nullptr;
      throw std::runtime_error(
          "Accelerate RFFT: Failed to create vDSP_DFT_Interleaved setup(s).");
    }

    try {
      if (length_ > 0) {  // Only resize if length is non-zero
        temp_packed_buffer_.resize(length_);
      }
    }
    catch (const std::bad_alloc& e) {
      if constexpr (std::is_same_v<T, float>) {
        if (fft_setup_forward_)
          vDSP_DFT_Interleaved_DestroySetup(fft_setup_forward_);
      }
      else {
        if (fft_setup_forward_)
          vDSP_DFT_Interleaved_DestroySetupD(fft_setup_forward_);
      }
      fft_setup_forward_ = nullptr;
      fft_setup_inverse_ = nullptr;
      throw std::runtime_error(
          "Accelerate RFFT: Failed to allocate temporary packed buffer: "
          + std::string(e.what()));
    }
  }

  // Destructor
  template <typename T>
  RFFTPlanImpl<T>::~RFFTPlanImpl()
  {
    if (fft_setup_forward_) {
      if constexpr (std::is_same_v<T, float>) {
        vDSP_DFT_Interleaved_DestroySetup(fft_setup_forward_);
      }
      else {
        vDSP_DFT_Interleaved_DestroySetupD(fft_setup_forward_);
      }
    }
  }

  // rfft Method
  template <typename T>  // T is REAL type
  Status RFFTPlanImpl<T>::rfft(
      std::span<const T> input, std::span<Complex> output) const
  {
    if (!fft_setup_forward_)
      return length_ == 0 ? Status::Success : Status::InvalidOperation;

    size_t N = length_;
    size_t N_half_plus_1 = N / 2 + 1;

    if (N == 0 && input.empty() && output.empty()) return Status::Success;
    if (input.size() != N || output.size() != N_half_plus_1) {
      return Status::SizeMismatch;
    }
    if (N > 0 && temp_packed_buffer_.size() != N) {
      std::cerr << "RFFTPlanImpl::rfft internal error: temp_packed_buffer_ not "
                   "sized correctly."
                << std::endl;
      return Status::Failure;
    }

    T* p_packed_dst = temp_packed_buffer_.data();

    if constexpr (std::is_same_v<T, float>) {
      vDSP_DFT_Interleaved_Execute(
          fft_setup_forward_,
          reinterpret_cast<const DSPComplex*>(input.data()),
          reinterpret_cast<DSPComplex*>(p_packed_dst));
    }
    else {
      vDSP_DFT_Interleaved_ExecuteD(
          fft_setup_forward_,
          reinterpret_cast<const DSPDoubleComplex*>(input.data()),
          reinterpret_cast<DSPDoubleComplex*>(p_packed_dst));
    }

    output[0] = Complex(p_packed_dst[0], static_cast<T>(0.0));

    for (size_t k = 1; k < N / 2; ++k) {
      output[k] = Complex(p_packed_dst[2 * k], p_packed_dst[2 * k + 1]);
    }

    if (N % 2 == 0) {
      if (N / 2 < output.size()) {
        output[N / 2] = Complex(p_packed_dst[1], static_cast<T>(0.0));
      }
      else {
        std::cerr << "RFFTPlanImpl::rfft error: Nyquist index out of bounds "
                     "for output span."
                  << std::endl;
        return Status::Failure;
      }
    }
    if (N % 2 != 0 && N > 1) {
      size_t k_last = N / 2;
      if (k_last < output.size() && (2 * k_last + 1) < N) {
        output[k_last]
            = Complex(p_packed_dst[2 * k_last], p_packed_dst[2 * k_last + 1]);
      }
      else if (k_last < output.size() && (2 * k_last) < N) {
        std::cerr << "RFFTPlanImpl::rfft warning: Unexpected packing for odd "
                     "N, only real part for last element."
                  << std::endl;
        output[k_last] = Complex(p_packed_dst[2 * k_last], static_cast<T>(0.0));
      }
      else {
        std::cerr << "RFFTPlanImpl::rfft error: Index out of bounds for last "
                     "element of odd N."
                  << std::endl;
      }
    }
    return Status::Success;
  }

  // irfft Method
  template <typename T>  // T is REAL type
  Status RFFTPlanImpl<T>::irfft(
      std::span<const Complex> input, std::span<T> output_span) const
  {
    if (!fft_setup_inverse_)
      return length_ == 0 ? Status::Success : Status::InvalidOperation;

    size_t N = length_;
    size_t N_half_plus_1 = N / 2 + 1;

    if (N == 0 && input.empty() && output_span.empty()) return Status::Success;
    if (input.size() != N_half_plus_1 || output_span.size() != N) {
      return Status::SizeMismatch;
    }
    if (N > 0 && temp_packed_buffer_.size() != N) {
      std::cerr << "RFFTPlanImpl::irfft internal error: temp_packed_buffer_ "
                   "not sized correctly."
                << std::endl;
      return Status::Failure;
    }

    T* p_packed_src = temp_packed_buffer_.data();

    p_packed_src[0] = input[0].real();

    for (size_t k = 1; k < N / 2; ++k) {
      p_packed_src[2 * k] = input[k].real();
      p_packed_src[2 * k + 1] = input[k].imag();
    }

    if (N % 2 == 0) {
      if (N / 2 < input.size()) {
        p_packed_src[1] = input[N / 2].real();
      }
      else {
        std::cerr << "RFFTPlanImpl::irfft error: Nyquist index out of bounds "
                     "for input span."
                  << std::endl;
        return Status::Failure;
      }
    }
    if (N % 2 != 0 && N > 1) {
      size_t k_last = N / 2;
      if (k_last < input.size() && (2 * k_last + 1) < N) {
        p_packed_src[2 * k_last] = input[k_last].real();
        p_packed_src[2 * k_last + 1] = input[k_last].imag();
      }
      else if (k_last < input.size() && (2 * k_last) < N) {
        std::cerr << "RFFTPlanImpl::irfft warning: Unexpected packing for odd "
                     "N input, only real part for last element."
                  << std::endl;
        p_packed_src[2 * k_last] = input[k_last].real();
        if ((2 * k_last + 1) < N)
          p_packed_src[2 * k_last + 1] = static_cast<T>(0.0);
      }
      else {
        std::cerr << "RFFTPlanImpl::irfft error: Index out of bounds for "
                     "packing last element of odd N."
                  << std::endl;
      }
    }

    if constexpr (std::is_same_v<T, float>) {
      vDSP_DFT_Interleaved_Execute(
          fft_setup_inverse_,
          reinterpret_cast<const DSPComplex*>(p_packed_src),
          reinterpret_cast<DSPComplex*>(output_span.data()));
    }
    else {
      vDSP_DFT_Interleaved_ExecuteD(
          fft_setup_inverse_,
          reinterpret_cast<const DSPDoubleComplex*>(p_packed_src),
          reinterpret_cast<DSPDoubleComplex*>(output_span.data()));
    }
    return Status::Success;
  }

  // get_length Method
  template <typename T>  // T is REAL type
  size_t RFFTPlanImpl<T>::get_length() const
  {
    return length_;
  }

  //--------------------------------------------------------------------------
  // Explicit Template Instantiations
  //--------------------------------------------------------------------------
  template class FFTPlanImpl<C32>;
  template class FFTPlanImpl<C64>;
  template class RFFTPlanImpl<F32>;
  template class RFFTPlanImpl<F64>;

}  // namespace OmniDSP::Accelerate

#endif  // OMNIDSP_INTERNAL_USE_ACCELERATE
