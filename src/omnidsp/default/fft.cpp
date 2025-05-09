#include "fft.hpp"  // Plan class definitions for this backend

// Include necessary standard library headers
#include <algorithm>  // For std::copy, std::swap
#include <cassert>    // For assert
#include <cmath>      // For std::polar, std::cos, std::sin
#include <complex>
#include <cstring>    // For std::memcpy
#include <memory>     // For std::unique_ptr, std::make_unique
#include <numbers>    // For std::numbers::pi_v
#include <span>       // For std::span
#include <stdexcept>  // For std::logic_error, std::invalid_argument
#include <vector>

// Include core_types to ensure Utils::GetComplexType is available
#include <OmniDSP/core_types.hpp>

namespace OmniDSP::Default {

  //-----------------------------------------------------------------------------
  // Internal Helper Functions (Standard C++)
  //-----------------------------------------------------------------------------

  namespace fft_detail {  // Internal namespace for implementation details

    // Bit reversal permutation
    template <typename T_Complex>
    void bit_reverse_permutation(T_Complex* data, size_t n)
    {
      if (n <= 1) return;
      size_t j = 0;
      for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while (j >= bit) {
          j -= bit;
          bit >>= 1;
        }
        j += bit;
        if (i < j) {
          std::swap(data[i], data[j]);
        }
      }
    }

    // Function to compute twiddle factors (W_N^k) for iterative FFT
    template <bool Inverse, typename T_Complex>
    std::vector<T_Complex> compute_twiddles(size_t n)
    {
      using T = typename T_Complex::value_type;
      if (n == 0) return {};
      // Iterative FFT needs N/2 twiddle factors W_N^0 to W_N^(N/2 - 1)
      std::vector<T_Complex> twiddles(n / 2);
      const T angle_sign = Inverse
                               ? static_cast<T>(2.0 * std::numbers::pi_v<T>)
                               : static_cast<T>(-2.0 * std::numbers::pi_v<T>);
      for (size_t k = 0; k < n / 2; ++k) {
        twiddles[k] = std::polar(
            static_cast<T>(1.0),
            angle_sign * static_cast<T>(k) / static_cast<T>(n));
      }
      return twiddles;
    }

    // Helper for iterative FFT execution (assumes buffer is allocated)
    // This helper function itself does not need to be const, as it modifies the
    // temp buffer. The const-ness applies to the Plan methods calling it.
    template <bool Inverse, typename T_Complex>
    void execute_fft_iterative(
        std::span<const T_Complex> input,
        std::span<T_Complex> output,
        size_t n,
        const std::vector<T_Complex>&
            twiddles,            // Precomputed W_N^k for k=0..N/2-1
        T_Complex* temp_buffer)  // Pre-allocated temporary buffer of size N
    {
      using T = typename T_Complex::value_type;
      assert(temp_buffer != nullptr && "Temporary buffer must be provided");
      assert(input.size() >= n && "Input span too small");
      assert(output.size() >= n && "Output span too small");
      assert(twiddles.size() >= n / 2 && "Twiddle factor vector too small");

      // 1. Copy input to temp buffer
      std::memcpy(temp_buffer, input.data(), n * sizeof(T_Complex));

      // 2. Bit reverse permutation in-place
      bit_reverse_permutation(temp_buffer, n);

      // 3. Iterative Cooley-Tukey stages
      for (size_t len = 2; len <= n;
           len <<= 1) {  // len = size of sub-FFTs being combined
        size_t half_len = len >> 1;
        size_t twiddle_idx_step
            = n / len;  // Step through precomputed W_N^k factors
        for (size_t i = 0; i < n;
             i += len) {  // Start index of each sub-FFT pair
          for (size_t j = 0; j < half_len; ++j) {  // Index within the sub-FFT
            // Use precomputed twiddle: W_len^j = W_N^(j * N/len) = twiddles[j *
            // twiddle_idx_step]
            const T_Complex& w = twiddles[j * twiddle_idx_step];
            T_Complex u = temp_buffer[i + j];
            T_Complex v = temp_buffer[i + j + half_len];
            temp_buffer[i + j] = u + w * v;
            temp_buffer[i + j + half_len] = u - w * v;
          }
        }
      }

      // 4. Apply scaling if inverse and copy to output
      if constexpr (Inverse) {
        T scale = static_cast<T>(1.0) / static_cast<T>(n);
        for (size_t i = 0; i < n; ++i) {
          output[i] = temp_buffer[i] * scale;
        }
      }
      else {
        std::memcpy(output.data(), temp_buffer, n * sizeof(T_Complex));
      }
    }

  }  // namespace fft_detail

  //-----------------------------------------------------------------------------
  // Plan Class Implementations (Standard C++)
  //-----------------------------------------------------------------------------

  // --- FFTPlanImpl Implementation ---
  template <typename T_Complex>
  FFTPlanImpl<T_Complex>::FFTPlanImpl(size_t n) : n_(n)
  {
    using T = typename T_Complex::value_type;  // Scalar type
    if (n == 0 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("FFT size must be a positive power of 2.");
    }
    // Precompute twiddles for both directions (only need N/2 factors for
    // iterative)
    forward_twiddles_ = fft_detail::compute_twiddles<false, T_Complex>(n_);
    inverse_twiddles_ = fft_detail::compute_twiddles<true, T_Complex>(n_);

    // Allocate temporary buffer using standard C++
    try {
      temp_buffer_ = std::make_unique<T_Complex[]>(n_);
    }
    catch (const std::bad_alloc& e) {
      // Consider logging the error here
      throw std::runtime_error(
          "Failed to allocate memory for FFT buffer: " + std::string(e.what()));
    }
    if (!temp_buffer_) {  // Double check allocation (make_unique throws on
                          // failure)
      throw std::runtime_error(
          "Failed to allocate memory for FFT buffer (unexpected).");
    }
  }

  template <typename T_Complex>
  Status FFTPlanImpl<T_Complex>::fft(
      std::span<const T_Complex> input, std::span<T_Complex> output) const
  {
    if (input.size() < n_ || output.size() < n_) {
      return Status::SizeMismatch;
    }
    try {
      // Pass const member data (twiddles) and mutable temp buffer to helper
      fft_detail::execute_fft_iterative<false>(
          input, output, n_, forward_twiddles_, temp_buffer_.get());
    }
    catch (const std::exception& e) {
      // Log error e.what()
      return Status::Failure;  // Or a more specific error
    }
    return Status::Success;
  }

  template <typename T_Complex>
  Status FFTPlanImpl<T_Complex>::ifft(
      std::span<const T_Complex> input, std::span<T_Complex> output) const
  {
    if (input.size() < n_ || output.size() < n_) {
      return Status::SizeMismatch;
    }
    try {
      // Use inverse twiddles and apply scaling inside helper
      fft_detail::execute_fft_iterative<true>(
          input, output, n_, inverse_twiddles_, temp_buffer_.get());
    }
    catch (const std::exception& e) {
      // Log error e.what()
      return Status::Failure;  // Or a more specific error
    }
    return Status::Success;
  }

  template <typename T_Complex>
  size_t FFTPlanImpl<T_Complex>::get_length() const
  {
    return n_;
  }

  // --- RFFTPlanImpl Implementation ---
  template <typename T_Real>
  RFFTPlanImpl<T_Real>::RFFTPlanImpl(size_t n)
      : n_(n), cfft_plan_(n / 2)  // Initialize CFFT plan for size N/2
  {
    // Use the alias defined in the header (matches base class)
    using T_Complex = Utils::GetComplexType<T_Real>;
    if (n < 2 || (n & (n - 1)) != 0) {
      throw std::invalid_argument("RFFT size must be a power of 2 and >= 2.");
    }
    half_n_ = n / 2;

    // Allocate temporary buffer for packed N/2 complex data
    try {
      packed_buffer_ = std::make_unique<T_Complex[]>(half_n_);
    }
    catch (const std::bad_alloc& e) {
      // Log error e.what()
      throw std::runtime_error(
          "Failed to allocate memory for RFFT packed buffer: "
          + std::string(e.what()));
    }
    if (!packed_buffer_) {  // Double check allocation
      throw std::runtime_error(
          "Failed to allocate memory for RFFT packed buffer (unexpected).");
    }
  }

  // FIX: Use T_Complex alias for the output span type
  template <typename T_Real>
  Status RFFTPlanImpl<T_Real>::rfft(
      std::span<const T_Real> input, std::span<T_Complex> output) const
  {
    // using T_Complex = std::complex<T_Real>; // Original incorrect definition
    // Use the alias defined in the header (matches base class)
    using T_Complex = Utils::GetComplexType<T_Real>;
    size_t expected_output_size = half_n_ + 1;
    if (input.size() < n_ || output.size() < expected_output_size) {
      return Status::SizeMismatch;
    }

    // 1. Perform N/2-point complex FFT on the real input treated as interleaved
    // complex pairs. The underlying cfft_plan_.fft is const, so this call is
    // valid within a const method.
    Status status = cfft_plan_.fft(
        // Treat real input span as complex span of half length
        std::span<const T_Complex>(
            reinterpret_cast<const T_Complex*>(input.data()), half_n_),
        // Output to the mutable packed_buffer_
        std::span<T_Complex>(packed_buffer_.get(), half_n_));
    if (status != Status::Success) {
      return status;  // Propagate error from underlying CFFT
    }

    // 2. Unpacking Stage: Convert N/2 complex result (X = packed_buffer_) to
    // N/2+1 RFFT result (Y = output) This part reads from packed_buffer_ and
    // writes to output, which is fine for a const method.
    T_Complex* X = packed_buffer_.get();  // Input for unpacking
    T_Complex* Y = output.data();         // Output destination

    // Handle DC (k=0) and Nyquist (k=N/2) components separately
    Y[0] = T_Complex(X[0].real() + X[0].imag(), 0.0);        // Y[0] is real
    Y[half_n_] = T_Complex(X[0].real() - X[0].imag(), 0.0);  // Y[N/2] is real

    // Process remaining frequencies (k = 1 to N/2 - 1)
    const T_Real scale_half = static_cast<T_Real>(0.5);
    const T_Real angle_unit = static_cast<T_Real>(
        -2.0 * std::numbers::pi_v<T_Real> / n_);  // Angle step for W_N^k

    for (size_t k = 1; k < half_n_; ++k) {  // Loop from 1 to N/2 - 1
      T_Complex Xk = X[k];
      T_Complex XNk = std::conj(X[half_n_ - k]);  // X[N/2 - k]*
      // Twiddle factor W_N^k = exp(-j * 2*pi*k / N)
      T_Complex W = std::polar(
          static_cast<T_Real>(1.0), angle_unit * static_cast<T_Real>(k));
      // Calculate terms based on formula: Y[k] = 0.5 * [ (Xk + XNk) - j*W*(Xk -
      // XNk) ]
      T_Complex sum = Xk + XNk;
      T_Complex diff = Xk - XNk;
      T_Complex jW = T_Complex(0.0, -1.0) * W;  // -j * W
      // Combine: Y[k] = 0.5 * (sum + jW * diff)
      Y[k] = scale_half * (sum + jW * diff);
    }
    return Status::Success;
  }

  // FIX: Use T_Complex alias for the input span type
  template <typename T_Real>
  Status RFFTPlanImpl<T_Real>::irfft(
      std::span<const T_Complex> input, std::span<T_Real> output) const
  {
    // using T_Complex = std::complex<T_Real>; // Original incorrect definition
    // Use the alias defined in the header (matches base class)
    using T_Complex = Utils::GetComplexType<T_Real>;
    size_t expected_input_size = half_n_ + 1;
    if (input.size() < expected_input_size || output.size() < n_) {
      return Status::SizeMismatch;
    }

    // 1. Packing Stage: Convert N/2+1 RFFT input (Y = input) to N/2 complex
    // spectrum (X = packed_buffer_) for CIFFT This reads from input and writes
    // to the mutable packed_buffer_. This is okay within a const method *if*
    // packed_buffer_ is treated as conceptually mutable state used internally
    // for calculation.
    const T_Complex* Y = input.data();    // Input for packing
    T_Complex* X = packed_buffer_.get();  // Output destination for packing

    // Handle k=0 component for X[0] using Y[0] and Y[N/2]
    T_Real y0_real = Y[0].real();         // Y[0] is real
    T_Real yn2_real = Y[half_n_].real();  // Y[N/2] is real
    X[0] = static_cast<T_Real>(0.5)
           * T_Complex(y0_real + yn2_real, y0_real - yn2_real);

    // Process remaining frequencies (k = 1 to N/2 - 1) to compute
    // X[1]...X[N/2-1]
    const T_Real scale_half = static_cast<T_Real>(0.5);
    const T_Real angle_unit = static_cast<T_Real>(
        2.0 * std::numbers::pi_v<T_Real>
        / n_);  // Angle step for W_N^-k = exp(j*2*pi*k/N)

    for (size_t k = 1; k < half_n_; ++k) {  // Loop 1 to N/2 - 1
      T_Complex Yk = Y[k];
      T_Complex YNk = std::conj(Y[half_n_ - k]);  // Y[N/2 - k]*
      // Twiddle factor W_N^-k = exp(j * 2*pi*k / N)
      T_Complex W_inv = std::polar(
          static_cast<T_Real>(1.0), angle_unit * static_cast<T_Real>(k));
      // Calculate terms based on formula: X[k] = 0.5 * [ (Yk + YNk) +
      // j*W_inv*(Yk - YNk) ]
      T_Complex sum = Yk + YNk;
      T_Complex diff = Yk - YNk;
      T_Complex jW_inv = T_Complex(0.0, 1.0) * W_inv;  // j * W_inv
      // Combine: X[k] = 0.5 * (sum + jW_inv * diff)
      X[k] = scale_half * (sum + jW_inv * diff);
    }

    // 2. Perform N/2-point inverse complex FFT on the packed spectrum X
    // (packed_buffer_). The underlying cfft_plan_.ifft is const, so this call
    // is valid.
    Status status = cfft_plan_.ifft(
        // Input is the packed buffer
        std::span<const T_Complex>(packed_buffer_.get(), half_n_),
        // Treat the real output span as a complex span for the IFFT output
        std::span<T_Complex>(
            reinterpret_cast<T_Complex*>(output.data()), half_n_));
    if (status != Status::Success) {
      return status;  // Propagate error from underlying CIFFT
    }

    // Scaling is handled correctly: Packing (0.5) * CIFFT (1/(N/2)) = 1/N.
    // The output buffer now contains the final N real samples.

    return Status::Success;
  }

  template <typename T_Real>
  size_t RFFTPlanImpl<T_Real>::get_length() const
  {
    return n_;  // Return the original real length N
  }

  // Explicit template instantiations for float and double based types
  // Need to instantiate with the complex types for FFTPlanImpl
  template class FFTPlanImpl<C32>;  // std::complex<float>
  template class FFTPlanImpl<C64>;  // std::complex<double>
  // Need to instantiate with the real types for RFFTPlanImpl
  template class RFFTPlanImpl<F32>;  // float
  template class RFFTPlanImpl<F64>;  // double

}  // namespace OmniDSP::Default
