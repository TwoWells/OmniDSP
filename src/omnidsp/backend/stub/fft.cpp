/**
 * @file fft.cpp (stub)
 * @brief Implements Stub backend FFTPlanImpl and RFFTPlanImpl classes using
 * standard C++.
 */

#include <algorithm>  // For std::reverse, std::swap
#include <cmath>      // For M_PI, sin, cos, log2, pow
#include <complex>
#include <iostream>  // For debug/error messages
#include <numeric>   // For std::iota
#include <span>
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <vector>

#include "OmniDSP/core_types.h"  // For Status, RealT, ComplexT etc.
#include "backend.h"             // Stub backend declarations

// Define PI if not available from cmath
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace OmniDSP {
namespace backend {

// Helper function (move to common utility?)
inline bool is_power_of_two(size_t n) {
  return (n > 0) && ((n & (n - 1)) == 0);
}

//--------------------------------------------------------------------------
// StubFFTPlanImpl Method Definitions (Complex FFT)
//--------------------------------------------------------------------------

// Helper function for bit reversal permutation
inline size_t reverse_bits(size_t n, size_t num_bits) {
  size_t reversed_n = 0;
  for (size_t i = 0; i < num_bits; ++i) {
    if ((n >> i) & 1) {
      reversed_n |= (1 << (num_bits - 1 - i));
    }
  }
  return reversed_n;
}

// Basic Iterative Cooley-Tukey FFT Implementation (Radix-2)
template <typename T>
void cooley_tukey_fft(std::span<T> data, bool inverse,
                      const std::vector<T>& twiddles,
                      const std::vector<size_t>& bit_reverse_indices) {
  size_t N = data.size();
  if (N == 0 || N == 1) return;  // Nothing to do for empty or single element

  // 1. Bit-reversal permutation
  for (size_t i = 0; i < N; ++i) {
    size_t reversed_i = bit_reverse_indices[i];
    // Swap only if i < reversed_i to avoid swapping twice
    if (i < reversed_i) {
      std::swap(data[i], data[reversed_i]);
    }
  }

  // 2. Iterative butterfly stages
  size_t twiddle_idx = 0;
  for (size_t len = 2; len <= N;
       len <<= 1) {  // len = block size (2, 4, 8, ...)
    size_t half_len = len >> 1;
    for (size_t i = 0; i < N; i += len) {  // i = start index of block
      for (size_t j = 0; j < half_len;
           ++j) {  // j = index within the half-block
        // Get twiddle factor W_N^k (or W_N^-k for inverse)
        // Twiddles are precomputed as exp(-2*pi*j*k/N)
        T twiddle = twiddles[twiddle_idx +
                             j];  // Index depends on how twiddles were stored
        if (inverse) {
          twiddle = std::conj(twiddle);  // Use conjugate for inverse FFT
        }

        size_t idx1 = i + j;
        size_t idx2 = i + j + half_len;

        T u = data[idx1];
        T v = data[idx2] * twiddle;

        data[idx1] = u + v;
        data[idx2] = u - v;
      }
    }
    // Move to the next set of twiddle factors for the next stage
    // This assumes twiddles are stored stage by stage
    twiddle_idx += half_len;  // Or calculate based on len/N
  }
}

template <typename T>
StubFFTPlanImpl<T>::StubFFTPlanImpl(size_t length) : length_(length) {
  if (length == 0) return;  // Allow empty plan? Or throw? Let's allow for now.

  // Stub implementation often uses simple Radix-2 Cooley-Tukey
  if (!is_power_of_two(length_)) {
    throw std::invalid_argument(
        "Stub FFT implementation currently requires power-of-two lengths.");
  }

  // Precompute bit reversal indices
  size_t num_bits =
      static_cast<size_t>(std::log2(static_cast<double>(length_)));
  bit_reverse_indices_.resize(length_);
  for (size_t i = 0; i < length_; ++i) {
    bit_reverse_indices_[i] = reverse_bits(i, num_bits);
  }

  // Precompute twiddle factors: W_N^k = exp(-2*pi*j*k/N)
  // We need factors for different stages: k/len where len = 2, 4, ..., N
  // Total twiddles needed: N/2 + N/4 + ... + 1 = N - 1 ? No, just N/2 unique
  // ones. Store them stage by stage? Or just the N/2 unique ones? Let's store
  // the N/2 unique W_N^k factors (k=0 to N/2-1) and index correctly later. Or
  // store them as needed by the iterative algorithm above (N/2 total).
  twiddle_factors_.resize(length_ / 2);
  using ValueType = typename T::value_type;  // float or double
  for (size_t k = 0; k < length_ / 2; ++k) {
    ValueType angle = -2.0 * M_PI * static_cast<ValueType>(k) /
                      static_cast<ValueType>(length_);
    twiddle_factors_[k] = T{std::cos(angle), std::sin(angle)};
  }
  std::cout << "Stub FFTPlanImpl created for length " << length_
            << std::endl;  // Debug
}

template <typename T>
Status StubFFTPlanImpl<T>::fft(std::span<const T> input,
                               std::span<T> output) const {
  if (input.size() != length_ || output.size() != length_) {
    return Status::SizeMismatch;
  }
  if (length_ == 0) return Status::Success;  // Handle empty case

  // Copy input to output for in-place algorithm (or modify algorithm for
  // out-of-place)
  std::copy(input.begin(), input.end(), output.begin());

  // Perform FFT
  cooley_tukey_fft(output, false, twiddle_factors_, bit_reverse_indices_);

  return Status::Success;
}

template <typename T>
Status StubFFTPlanImpl<T>::ifft(std::span<const T> input,
                                std::span<T> output) const {
  if (input.size() != length_ || output.size() != length_) {
    return Status::SizeMismatch;
  }
  if (length_ == 0) return Status::Success;

  // Copy input to output for in-place algorithm
  std::copy(input.begin(), input.end(), output.begin());

  // Perform Inverse FFT
  cooley_tukey_fft(output, true, twiddle_factors_, bit_reverse_indices_);

  // Note: Stub IFFT is unscaled by 1/N.
  // Scaling is applied in the one-off StubOmniDSPImpl::ifft function.

  return Status::Success;
}

template <typename T>
size_t StubFFTPlanImpl<T>::get_length() const {
  return length_;
}

//--------------------------------------------------------------------------
// StubRFFTPlanImpl Method Definitions (Real FFT)
//--------------------------------------------------------------------------

template <typename T>
StubRFFTPlanImpl<T>::StubRFFTPlanImpl(size_t length) : length_(length) {
  if (length == 0) return;
  if (!is_power_of_two(length_)) {
    throw std::invalid_argument(
        "Stub RFFT implementation currently requires power-of-two lengths.");
  }
  if (length_ < 2 && length_ != 0) {  // Need N>=2 for N/2 FFT
    throw std::invalid_argument("Stub RFFT length must be >= 2.");
  }

  // Precompute factors for the N/2 complex FFT used internally
  size_t N_over_2 = length_ / 2;
  size_t num_bits =
      static_cast<size_t>(std::log2(static_cast<double>(N_over_2)));
  bit_reverse_indices_.resize(N_over_2);
  for (size_t i = 0; i < N_over_2; ++i) {
    bit_reverse_indices_[i] = reverse_bits(i, num_bits);
  }

  twiddle_factors_.resize(N_over_2 / 2);  // Twiddles for N/2 FFT
  for (size_t k = 0; k < N_over_2 / 2; ++k) {
    RealT<T> angle = -2.0 * M_PI * static_cast<RealT<T>>(k) /
                     static_cast<RealT<T>>(N_over_2);
    twiddle_factors_[k] = ComplexT<T>{std::cos(angle), std::sin(angle)};
  }

  std::cout << "Stub RFFTPlanImpl created for length " << length_
            << std::endl;  // Debug
}

template <typename T>
Status StubRFFTPlanImpl<T>::rfft(std::span<const RealT<T>> input,
                                 std::span<ComplexT<T>> output) const {
  size_t N = length_;
  if (N == 0) return Status::Success;
  size_t N_over_2 = N / 2;
  size_t output_size_expected = N_over_2 + 1;

  if (input.size() != N || output.size() != output_size_expected) {
    return Status::SizeMismatch;
  }

  // 1. Pack N real inputs into N/2 complex inputs: y[k] = x[2k] + j*x[2k+1]
  std::vector<ComplexT<T>> packed_input(N_over_2);
  for (size_t k = 0; k < N_over_2; ++k) {
    packed_input[k] = ComplexT<T>{input[2 * k], input[2 * k + 1]};
  }

  // 2. Compute N/2 complex FFT on packed_input
  std::vector<ComplexT<T>> packed_fft(N_over_2);
  // Use the complex FFT helper with N/2 factors
  cooley_tukey_fft<ComplexT<T>>(packed_input, false, twiddle_factors_,
                                bit_reverse_indices_);
  packed_fft =
      packed_input;  // Result is now in packed_fft after in-place modification

  // 3. Unpack the N/2 complex FFT result into N/2 + 1 RFFT output
  //    Requires twiddle factors W_N^k = exp(-2*pi*j*k/N) for k = 0 to N/2
  output[0] = ComplexT<T>{packed_fft[0].real() + packed_fft[0].imag(),
                          0.0};  // DC component

  for (size_t k = 1; k <= N / 4; ++k) {  // Iterate up to N/4
    ComplexT<T> fft_k = packed_fft[k];
    ComplexT<T> fft_N_minus_k = packed_fft[N_over_2 - k];  // Use symmetry

    RealT<T> angle =
        -2.0 * M_PI * static_cast<RealT<T>>(k) / static_cast<RealT<T>>(N);
    ComplexT<T> W_N_k = {std::cos(angle), std::sin(angle)};

    ComplexT<T> term1 = fft_k + std::conj(fft_N_minus_k);
    ComplexT<T> term2 = fft_k - std::conj(fft_N_minus_k);
    ComplexT<T> term2_twiddle =
        ComplexT<T>{0.0, -1.0} * term2 * W_N_k;  // -j * term2 * W_N^k

    output[k] = 0.5 * (term1 + term2_twiddle);
    // Use conjugate symmetry for the upper half (not needed for standard RFFT
    // output)
    if (k < N_over_2) {  // Avoid overwriting Nyquist if N is even
      output[N_over_2 - k] = 0.5 * std::conj(term1 - term2_twiddle);
    }
  }

  // Handle Nyquist frequency (k = N/2) for even N
  // It's stored in the imaginary part of the DC component of the packed FFT
  output[N_over_2] =
      ComplexT<T>{packed_fft[0].real() - packed_fft[0].imag(), 0.0};

  return Status::Success;
}

template <typename T>
Status StubRFFTPlanImpl<T>::irfft(std::span<const ComplexT<T>> input,
                                  std::span<RealT<T>> output) const {
  size_t N = length_;
  if (N == 0) return Status::Success;
  size_t N_over_2 = N / 2;
  size_t input_size_expected = N_over_2 + 1;

  if (input.size() != input_size_expected || output.size() != N) {
    return Status::SizeMismatch;
  }

  // 1. Pack the N/2 + 1 RFFT input into N/2 complex values for IFFT
  std::vector<ComplexT<T>> packed_ifft_input(N_over_2);
  // Use formula derived from RFFT unpacking (solve for packed_fft[k])
  // packed_fft[k] = output[k] + conj(output[N/2-k]) + j*W_N^k*(output[k] -
  // conj(output[N/2-k])) This seems overly complex. Easier way: Reconstruct the
  // full N-point complex spectrum using conjugate symmetry.
  std::vector<ComplexT<T>> full_spectrum(N);
  full_spectrum[0] = input[0];  // DC
  for (size_t k = 1; k < N_over_2; ++k) {
    full_spectrum[k] = input[k];
    full_spectrum[N - k] = std::conj(input[k]);  // Conjugate symmetry
  }
  full_spectrum[N_over_2] = input[N_over_2];  // Nyquist

  // 2. Perform N-point complex IFFT on the reconstructed spectrum
  //    Need a temporary complex FFT plan for size N
  try {
    StubFFTPlanImpl<ComplexT<T>> temp_cfft_plan(N);
    std::vector<ComplexT<T>> complex_output(N);
    Status status = temp_cfft_plan.ifft(full_spectrum, complex_output);
    if (status != Status::Success) return status;

    // 3. The result should be purely real (imaginary parts close to zero)
    //    Copy the real parts to the output buffer.
    for (size_t i = 0; i < N; ++i) {
      // Optional: Check if imag part is acceptably small?
      // if (std::abs(complex_output[i].imag()) > 1e-9) { // Tolerance check
      //     std::cerr << "Warning: Non-zero imaginary part in IRFFT result at
      //     index " << i << std::endl;
      // }
      output[i] = complex_output[i].real();
    }

  } catch (const std::exception& e) {
    std::cerr << "Error during temporary CFFT plan creation in IRFFT: "
              << e.what() << std::endl;
    return Status::Failure;
  } catch (...) {
    return Status::Failure;
  }

  // Note: Stub IRFFT is unscaled by 1/N.
  // Scaling is applied in the one-off StubOmniDSPImpl::irfft function.

  return Status::Success;
}

template <typename T>
size_t StubRFFTPlanImpl<T>::get_length() const {
  return length_;
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations
//--------------------------------------------------------------------------
// Instantiate templates for common types (float, double) to ensure code
// generation.

// Define complex types for brevity
using float_c = OmniDSP::ComplexT<float>;
using double_c = OmniDSP::ComplexT<double>;

// StubFFTPlanImpl Instantiations
template class OmniDSP::backend::StubFFTPlanImpl<float_c>;
template class OmniDSP::backend::StubFFTPlanImpl<double_c>;

// StubRFFTPlanImpl Instantiations
template class OmniDSP::backend::StubRFFTPlanImpl<float>;
template class OmniDSP::backend::StubRFFTPlanImpl<double>;

}  // namespace backend
}  // namespace OmniDSP
