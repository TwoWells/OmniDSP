/**
 * @file fft.cpp (accelerate)
 * @brief Implements Accelerate backend FFTPlanImpl and RFFTPlanImpl classes.
 * Both FFTPlanImpl and RFFTPlanImpl use the modern interleaved DFT functions.
 */

// Only compile this file if Accelerate backend is enabled via CMake
#ifdef USE_ACCELERATE

#include <Accelerate/Accelerate.h>

#include <cmath>  // For log2, pow
#include <complex>
#include <cstdlib>   // For malloc/free (if needed, though vector preferred)
#include <iostream>  // For debug/error messages
#include <set>       // For checking supported lengths
#include <span>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "backend.h"  // Accelerate backend declarations

namespace OmniDSP {
namespace backend {

// Helper function to check if a number is a power of two
// No longer strictly needed for validation, but kept for potential use
inline bool is_power_of_two(size_t n) {
  return (n > 0) && ((n & (n - 1)) == 0);
}

// Helper function to check if a length is supported by Accelerate interleaved
// DFT Based on documentation: f * 2^n where f={1,3,5,9,15} and n is within
// limits. Max length 4096. This checks the 'Length' parameter passed to
// CreateSetup, which is N for C2C and N/2 for R2C.
bool is_accelerate_interleaved_dft_length_supported(size_t length_param) {
  if (length_param == 0 || length_param > 4096) {  // Max N or N/2 is 4096
    return false;
  }
  // Check against the specific allowed lengths derived from the f * 2^n formula
  static const std::set<size_t> supported_lengths = {
      // f = 1 (n=3..12 -> 8..4096)
      8,
      16,
      32,
      64,
      128,
      256,
      512,
      1024,
      2048,
      4096,
      // f = 3 (n=2..8 -> 12..768)
      12,
      24,
      48,
      96,
      192,
      384,
      768,
      // f = 5 (n=2..7 -> 20..640)
      20,
      40,
      80,
      160,
      320,
      640,
      // f = 9 (n=2..7 -> 36..1152)
      36,
      72,
      144,
      288,
      576,
      1152,
      // f = 15 (n=2..7 -> 60..1920)
      60,
      120,
      240,
      480,
      960,
      1920,
  };
  return supported_lengths.count(length_param);
}

//--------------------------------------------------------------------------
// AccelerateFFTPlanImpl Method Definitions (Complex FFT - Interleaved)
//--------------------------------------------------------------------------

template <typename T>
AccelerateFFTPlanImpl<T>::AccelerateFFTPlanImpl(size_t length)
    : length_(length) {
  if (length == 0) {
    throw std::invalid_argument("FFT length cannot be zero.");
  }
  // Validate against supported lengths for interleaved DFT (Length param is N
  // for C2C)
  if (!is_accelerate_interleaved_dft_length_supported(length_)) {
    throw std::invalid_argument("Unsupported FFT length " +
                                std::to_string(length_) +
                                " for Accelerate interleaved DFT.");
  }

  // Use interleaved DFT setup for Complex-to-Complex
  if constexpr (std::is_same_v<T, std::complex<float>>) {
    setup_ = vDSP_DFT_Interleaved_CreateSetup(
        nullptr, static_cast<vDSP_Length>(length_),
        vDSP_DFT_FORWARD,  // Direction doesn't matter for setup
        vDSP_DFT_Interleaved_ComplexToComplex);
  } else {  // double
    setup_ = vDSP_DFT_Interleaved_CreateSetupD(
        nullptr, static_cast<vDSP_Length>(length_), vDSP_DFT_FORWARD,
        vDSP_DFT_Interleaved_ComplexToComplex);
  }

  if (!setup_) {
    throw std::runtime_error(
        "Failed to create Accelerate interleaved DFTSetup for length " +
        std::to_string(length_));
  }

  std::cout << "Accelerate FFTPlanImpl (Interleaved) created for length "
            << length_ << std::endl;  // Debug
}

template <typename T>
AccelerateFFTPlanImpl<T>::~AccelerateFFTPlanImpl() {
  if (setup_) {
    if constexpr (std::is_same_v<T, std::complex<float>>) {
      vDSP_DFT_Interleaved_DestroySetup(setup_);
    } else {
      vDSP_DFT_Interleaved_DestroySetupD(setup_);
    }
    std::cout << "Accelerate FFTPlanImpl (Interleaved) destroyed for length "
              << length_ << std::endl;  // Debug
  }
}

template <typename T>
Status AccelerateFFTPlanImpl<T>::fft(std::span<const T> input,
                                     std::span<T> output) const {
  if (!setup_) return Status::InvalidOperation;
  if (input.size() != length_ || output.size() != length_) {
    return Status::SizeMismatch;
  }

  // Execute directly on interleaved data
  if constexpr (std::is_same_v<T, std::complex<float>>) {
    vDSP_DFT_Interleaved_Execute(
        setup_, reinterpret_cast<const DSPComplex*>(input.data()),
        reinterpret_cast<DSPComplex*>(output.data()), vDSP_DFT_FORWARD);
  } else {  // double
    vDSP_DFT_Interleaved_ExecuteD(
        setup_, reinterpret_cast<const DSPDoubleComplex*>(input.data()),
        reinterpret_cast<DSPDoubleComplex*>(output.data()), vDSP_DFT_FORWARD);
  }

  return Status::Success;
}

template <typename T>
Status AccelerateFFTPlanImpl<T>::ifft(std::span<const T> input,
                                      std::span<T> output) const {
  if (!setup_) return Status::InvalidOperation;
  if (input.size() != length_ || output.size() != length_) {
    return Status::SizeMismatch;
  }

  // Execute directly on interleaved data
  if constexpr (std::is_same_v<T, std::complex<float>>) {
    vDSP_DFT_Interleaved_Execute(
        setup_, reinterpret_cast<const DSPComplex*>(input.data()),
        reinterpret_cast<DSPComplex*>(output.data()), vDSP_DFT_INVERSE);
  } else {  // double
    vDSP_DFT_Interleaved_ExecuteD(
        setup_, reinterpret_cast<const DSPDoubleComplex*>(input.data()),
        reinterpret_cast<DSPDoubleComplex*>(output.data()), vDSP_DFT_INVERSE);
  }

  // Inverse DFTs are typically unscaled by 1/N. User responsibility.
  return Status::Success;
}

template <typename T>
size_t AccelerateFFTPlanImpl<T>::get_length() const {
  return length_;
}

//--------------------------------------------------------------------------
// AccelerateRFFTPlanImpl Method Definitions (Real FFT - Interleaved)
//--------------------------------------------------------------------------

template <typename T>
AccelerateRFFTPlanImpl<T>::AccelerateRFFTPlanImpl(size_t length)
    : length_(length) {
  if (length == 0 || length % 2 != 0) {
    // Real FFT requires even length N for N/2 complex values
    throw std::invalid_argument(
        "RFFT length must be non-zero and even for Accelerate backend.");
  }
  size_t length_over_2 = length_ / 2;

  // Validate N/2 against supported lengths for interleaved DFT
  if (!is_accelerate_interleaved_dft_length_supported(length_over_2)) {
    throw std::invalid_argument(
        "Unsupported RFFT length " + std::to_string(length_) +
        " (N/2=" + std::to_string(length_over_2) +
        " is unsupported) for Accelerate interleaved DFT.");
  }

  // Use interleaved DFT setup for Real-to-Complex
  if constexpr (std::is_same_v<T, float>) {
    setup_ = vDSP_DFT_Interleaved_CreateSetup(
        nullptr, static_cast<vDSP_Length>(length_over_2),
        vDSP_DFT_FORWARD,  // Direction irrelevant for setup
        vDSP_DFT_Interleaved_RealToComplex);
  } else {  // double
    setup_ = vDSP_DFT_Interleaved_CreateSetupD(
        nullptr, static_cast<vDSP_Length>(length_over_2), vDSP_DFT_FORWARD,
        vDSP_DFT_Interleaved_RealToComplex);
  }

  if (!setup_) {
    throw std::runtime_error(
        "Failed to create Accelerate interleaved DFTSetup (RealToComplex) for "
        "length " +
        std::to_string(length_));
  }

  // No need for temp_split_complex_ buffer anymore
  std::cout << "Accelerate RFFTPlanImpl (Interleaved) created for length "
            << length_ << std::endl;  // Debug
}

template <typename T>
AccelerateRFFTPlanImpl<T>::~AccelerateRFFTPlanImpl() {
  if (setup_) {
    if constexpr (std::is_same_v<T, float>) {
      vDSP_DFT_Interleaved_DestroySetup(setup_);
    } else {
      vDSP_DFT_Interleaved_DestroySetupD(setup_);
    }
    std::cout << "Accelerate RFFTPlanImpl (Interleaved) destroyed for length "
              << length_ << std::endl;  // Debug
  }
  // No need to free temp_split_complex_
}

template <typename T>
Status AccelerateRFFTPlanImpl<T>::rfft(std::span<const RealT<T>> input,
                                       std::span<ComplexT<T>> output) const {
  if (!setup_) return Status::InvalidOperation;

  size_t N = length_;
  size_t N_over_2 = N / 2;
  size_t output_size_expected = N_over_2 + 1;

  if (input.size() != N || output.size() != output_size_expected) {
    return Status::SizeMismatch;
  }

  // Allocate temporary buffer for the packed interleaved output (size N/2
  // complex)
  std::vector<ComplexT<T>> temp_packed_output(N_over_2);

  // Execute Real->Packed Complex FFT
  if constexpr (std::is_same_v<T, float>) {
    // Input is cast from Real* to const DSPComplex* (N reals treated as N/2
    // complex) Output is written to temp_packed_output (N/2 complex)
    vDSP_DFT_Interleaved_Execute(
        setup_, reinterpret_cast<const DSPComplex*>(input.data()),
        reinterpret_cast<DSPComplex*>(temp_packed_output.data()),
        vDSP_DFT_FORWARD);
  } else {  // double
    vDSP_DFT_Interleaved_ExecuteD(
        setup_, reinterpret_cast<const DSPDoubleComplex*>(input.data()),
        reinterpret_cast<DSPDoubleComplex*>(temp_packed_output.data()),
        vDSP_DFT_FORWARD);
  }

  // Unpack the N/2 complex results into the standard N/2 + 1 complex output
  // format DC component is real part of first element
  output[0] = {temp_packed_output[0].real(), 0.0};
  // Components 1 to N/2 - 1
  for (size_t k = 1; k < N_over_2; ++k) {
    output[k] = temp_packed_output[k];
  }
  // Nyquist component is imaginary part of first element
  output[N_over_2] = {temp_packed_output[0].imag(), 0.0};

  // Scaling: vDSP real DFTs often have implicit scaling. Check docs.
  // Typically might need multiplication by 0.5? Let's assume unscaled for now.

  return Status::Success;
}

template <typename T>
Status AccelerateRFFTPlanImpl<T>::irfft(std::span<const ComplexT<T>> input,
                                        std::span<RealT<T>> output) const {
  if (!setup_) return Status::InvalidOperation;

  size_t N = length_;
  size_t N_over_2 = N / 2;
  size_t input_size_expected = N_over_2 + 1;

  if (input.size() != input_size_expected || output.size() != N) {
    return Status::SizeMismatch;
  }

  // Allocate temporary buffer for the packed interleaved input (size N/2
  // complex)
  std::vector<ComplexT<T>> temp_packed_input(N_over_2);

  // Pack the standard N/2 + 1 complex input into the N/2 packed format
  temp_packed_input[0] = {input[0].real(),
                          input[N_over_2].real()};  // DC and Nyquist packed
  for (size_t k = 1; k < N_over_2; ++k) {
    temp_packed_input[k] = input[k];
  }

  // Execute Packed Complex->Real IFFT
  // Input is temp_packed_input (N/2 complex)
  // Output is cast from Real* to DSPComplex* (N reals treated as N/2 complex)
  if constexpr (std::is_same_v<T, float>) {
    vDSP_DFT_Interleaved_Execute(
        setup_, reinterpret_cast<const DSPComplex*>(temp_packed_input.data()),
        reinterpret_cast<DSPComplex*>(output.data()), vDSP_DFT_INVERSE);
  } else {  // double
    vDSP_DFT_Interleaved_ExecuteD(
        setup_,
        reinterpret_cast<const DSPDoubleComplex*>(temp_packed_input.data()),
        reinterpret_cast<DSPDoubleComplex*>(output.data()), vDSP_DFT_INVERSE);
  }

  // Scaling: Inverse real DFTs often need scaling by 1/N or 1/(2N). Check docs.
  // Leave unscaled for now, user responsibility.

  return Status::Success;
}

template <typename T>
size_t AccelerateRFFTPlanImpl<T>::get_length() const {
  return length_;
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------

// Define complex types for brevity
using float_c = OmniDSP::ComplexT<float>;
using double_c = OmniDSP::ComplexT<double>;

// AccelerateFFTPlanImpl Instantiations
template class OmniDSP::backend::AccelerateFFTPlanImpl<float_c>;
template class OmniDSP::backend::AccelerateFFTPlanImpl<double_c>;

// AccelerateRFFTPlanImpl Instantiations
template class OmniDSP::backend::AccelerateRFFTPlanImpl<float>;
template class OmniDSP::backend::AccelerateRFFTPlanImpl<double>;

}  // namespace backend
}  // namespace OmniDSP

#endif  // USE_ACCELERATE
