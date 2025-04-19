/**
 * @file src/omnidsp/backend/accelerate/fft.cpp
 * @brief Apple Accelerate backend implementation for OmniDSP FFTPlan.
 *
 * Implements the FFTPlanImpl structure and FFTPlan methods using
 * Apple's Accelerate framework (vDSP) routines (vDSP_DFT for C2C, vDSP_fft for
 * Real). Compiled only when USE_ACCELERATE is defined.
 */

// --- Includes ---
#include <OmniDSP/omnidsp.h>  // Public API header

#include <algorithm>  // For std::copy
#include <cmath>      // For std::sqrt, std::log2
#include <complex>
#include <limits>     // For std::numeric_limits
#include <memory>     // For std::unique_ptr
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>
#include <type_traits>  // For std::is_same_v, std::conditional
#include <vector>

#include "../backend.h"  // Internal backend function declarations (though not directly used here)

// Only compile if USE_ACCELERATE is defined by CMake (typically on macOS)
#if defined(USE_ACCELERATE)

#include <Accelerate/Accelerate.h>  // Main header for the Accelerate framework (includes vDSP)

namespace OmniDSP {

// --- FFTPlanImpl Definition (Accelerate Backend) ---
// This struct holds the Accelerate-specific data and logic for an FFT plan.
template <typename T>
struct FFTPlanImpl {
  // --- Members ---
  size_t length = 0;  // FFT Length (N)
  size_t complex_length =
      0;  // Length of complex spectrum (N for C2C, N/2+1 for Real)
  Direction direction = Direction::Forward;  // Transform direction
  Precision precision = Precision::SINGLE;   // float or double
  Domain domain = Domain::Complex;           // Complex or Real
  NormMode norm_mode = NormMode::Backward;   // Normalization mode
  T forward_scale = 1.0;   // Scaling factor applied to forward transform
  T backward_scale = 1.0;  // Scaling factor applied to inverse transform

  // Accelerate-specific handles and buffers
  void *setup_handle_c2c = nullptr;  // Opaque handle for vDSP_DFT (C2C FFTs,
                                     // type varies by precision)
  // FFTSetup/FFTSetupD handle for vDSP_fft (Real FFTs)
  typename std::conditional<std::is_same_v<T, float>, FFTSetup, FFTSetupD>::type
      setup_handle_real = nullptr;
  std::vector<T> real_buffer;  // Temporary buffer for real part in Real FFTs
                               // (split complex)
  std::vector<T> imag_buffer;  // Temporary buffer for imag part in Real FFTs
                               // (split complex)
  vDSP_Stride stride = 1;      // Data stride (usually 1 for contiguous arrays)

  // --- Helper: Calculate log2(N) for FFTSetup ---
  // vDSP FFTSetup requires the length N as log2(N).
  // It also currently requires N to be a power of 2 for Real domain FFTs using
  // vDSP_fft_zrip. Marked const because it does not modify the object state.
  vDSP_Length get_log2n(size_t n) const  // <<< Added const here
  {
    if (n == 0) throw std::invalid_argument("FFT length cannot be zero.");
    // Check if n is a power of 2 (handles n=1 correctly)
    bool is_power_of_two = (n > 0) && ((n & (n - 1)) == 0);
    if (!is_power_of_two) {
      // vDSP_DFT (C2C) handles non-power-of-2 lengths.
      if (domain == Domain::Real) {
        // vDSP_fft_zrip (Real FFT) requires power-of-2.
        throw std::runtime_error(
            "Accelerate FFTSetup (for Real domain) currently requires "
            "power-of-2 length. Got N=" +
            std::to_string(n));
      }
      // For C2C, no log2n needed for vDSP_DFT setup, return 0.
      return 0;
    }
    // Only calculate and return log2n if needed (for Real domain power-of-2
    // lengths) Use cast to vDSP_Length which is typically unsigned long
    return (domain == Domain::Real && n != 0)
               ? static_cast<vDSP_Length>(std::log2(static_cast<double>(n)))
               : 0;
  }

  // --- Constructor ---
  // Sets up the FFT plan based on user parameters.
  FFTPlanImpl(size_t len, Precision prec, Direction dir, Domain dom,
              NormMode norm) {
    if (len == 0) {
      throw std::invalid_argument("FFT length N cannot be zero.");
    }
    length = len;
    direction = dir;
    precision = prec;
    domain = dom;
    norm_mode = norm;

    // Calculate scaling factors based on normalization mode
    T scaleN = static_cast<T>(length);
    T scaleSqrtN = std::sqrt(scaleN);
    if (scaleN == static_cast<T>(0.0))  // Defensive check
    {
      throw std::runtime_error("ScaleN is zero, length N must be > 0");
    }
    if (scaleSqrtN == static_cast<T>(0.0) &&
        norm_mode == NormMode::Ortho)  // Check for N=0 if using Ortho
    {
      throw std::runtime_error(
          "ScaleSqrtN is zero for Ortho, length N must be > 0");
    }

    // --- Complex Domain Setup (using vDSP_DFT) ---
    if (dom == Domain::Complex) {
      complex_length = length;  // Input and output size N
      // Determine scaling factors based on mode (vDSP_DFT does NOT scale
      // internally)
      switch (norm_mode) {
        case NormMode::Backward:  // Default: IFFT scaled by 1/N
          forward_scale = 1.0;
          backward_scale = 1.0 / scaleN;
          break;
        case NormMode::Ortho:  // Unitary: FFT and IFFT scaled by 1/sqrt(N)
          forward_scale = 1.0 / scaleSqrtN;
          backward_scale = 1.0 / scaleSqrtN;
          break;
        case NormMode::Forward:  // FFT scaled by 1/N
          forward_scale = 1.0 / scaleN;
          backward_scale = 1.0;
          break;
      }

      // Determine vDSP direction enum
      vDSP_DFT_Direction vdsp_dir =
          (dir == Direction::Forward) ? vDSP_DFT_FORWARD : vDSP_DFT_INVERSE;

      // Create the appropriate vDSP_DFT setup handle (out-of-place complex)
      if constexpr (std::is_same_v<T, float>)  // SINGLE precision
        setup_handle_c2c = vDSP_DFT_zop_CreateSetup(nullptr, length, vdsp_dir);
      else  // DOUBLE precision
        setup_handle_c2c = vDSP_DFT_zop_CreateSetupD(nullptr, length, vdsp_dir);

      if (!setup_handle_c2c)  // Check if setup creation failed
        throw std::runtime_error(
            "Failed to create Accelerate vDSP_DFT (C2C) setup. Check "
            "length/parameters.");
    }
    // --- Real Domain Setup (using vDSP_fft_zrip) ---
    else  // Domain::Real
    {
      complex_length = length / 2 + 1;  // Real FFT output size (CCE format)
      // vDSP Real FFT (vDSP_fft_zrip) implicitly scales the IFFT by 2/N.
      // We need to calculate external scaling factors to match the desired
      // NormMode.
      T implicit_inv_scale = static_cast<T>(2.0) / scaleN;
      T factor = static_cast<T>(1.0) /
                 implicit_inv_scale;  // Factor to multiply vDSP result by to
                                      // get unscaled
      T factorSqrt = std::sqrt(factor);

      if (factorSqrt == static_cast<T>(0.0) && norm_mode == NormMode::Ortho &&
          scaleN > 0)  // Defensive check
      {
        throw std::runtime_error(
            "FactorSqrt is zero for Ortho (Real), length N must be > 0");
      }

      // Determine scaling factors based on mode
      switch (norm_mode) {
        case NormMode::Backward:   // RFFT unscaled(1.0), IRFFT scaled by 1/N
          forward_scale = factor;  // Apply N/2 to get unscaled forward
          backward_scale =
              (1.0 / scaleN) *
              factor;  // Apply (1/N) * (N/2) = 1/2 to get 1/N overall inverse
          break;
        case NormMode::Ortho:  // RFFT & IRFFT scaled by 1/sqrt(N)
          forward_scale =
              (1.0 / scaleSqrtN) * factor;  // Apply (1/sqrtN) * (N/2)
          backward_scale =
              (1.0 / scaleSqrtN) * factor;  // Apply (1/sqrtN) * (N/2)
          break;
        case NormMode::Forward:  // RFFT scaled by 1/N, IRFFT unscaled(1.0)
          forward_scale = (1.0 / scaleN) * factor;  // Apply (1/N) * (N/2) = 1/2
          backward_scale = factor;  // Apply N/2 to get unscaled inverse
          break;
      }

      // Get log2(N) and create FFTSetup handle (throws if N not power-of-2)
      vDSP_Length log2n = get_log2n(length);   // Call the const version
      if constexpr (std::is_same_v<T, float>)  // SINGLE precision
        setup_handle_real = vDSP_create_fftsetup(log2n, kFFTRadix2);
      else  // DOUBLE precision
        setup_handle_real = vDSP_create_fftsetupD(log2n, kFFTRadix2);

      if (!setup_handle_real)  // Check if setup creation failed
        throw std::runtime_error(
            "Failed to create Accelerate FFTSetup (Real) setup. Is length a "
            "power of 2?");

      // Allocate temporary buffers needed for split complex format used by
      // vDSP_fft_zrip Size is N/2 because vDSP packs DC and Nyquist into the
      // split format cleverly.
      real_buffer.resize(length / 2);
      imag_buffer.resize(length / 2);
    }
  }

  // --- Destructor ---
  // Releases the Accelerate setup handles.
  ~FFTPlanImpl() {
    if (setup_handle_c2c)  // If C2C handle exists
    {
      if constexpr (std::is_same_v<T, float>)
        vDSP_DFT_DestroySetup(
            reinterpret_cast<vDSP_DFT_Setup>(setup_handle_c2c));
      else
        vDSP_DFT_DestroySetupD(
            reinterpret_cast<vDSP_DFT_SetupD>(setup_handle_c2c));
      setup_handle_c2c = nullptr;
    }
    if (setup_handle_real)  // If Real FFT handle exists
    {
      if constexpr (std::is_same_v<T, float>)
        vDSP_destroy_fftsetup(setup_handle_real);
      else
        vDSP_destroy_fftsetupD(setup_handle_real);
      setup_handle_real = nullptr;
    }
  }

  // --- Packing Helper: Complex Conjugate Even (CCE) to vDSP Split Complex ---
  // Converts standard complex array (size N/2+1) to vDSP split format for
  // IRFFT. Marked const as it only reads complex_input and writes to
  // split_output_void (passed by pointer) and doesn't modify member variables.
  void pack_complex_to_split(const std::complex<T> *complex_input,
                             void *split_output_void) const {
    const size_t N = length;
    const size_t Nc = complex_length;
    if (Nc == 0 || N == 0) return;

    if constexpr (std::is_same_v<T, float>)  // SINGLE precision
    {
      DSPSplitComplex *split =
          reinterpret_cast<DSPSplitComplex *>(split_output_void);
      split->realp[0] = complex_input[0].real();  // DC component
      // Nyquist component (real part of last element) goes into imagp[0] for N
      // even
      split->imagp[0] =
          (N % 2 == 0 && Nc > N / 2) ? complex_input[N / 2].real() : 0.0f;
      for (size_t k = 1; k < N / 2; ++k) {  // Components 1 to N/2 - 1
        split->realp[k] = complex_input[k].real();
        split->imagp[k] = complex_input[k].imag();
      }
    } else  // DOUBLE precision
    {
      DSPDoubleSplitComplex *split =
          reinterpret_cast<DSPDoubleSplitComplex *>(split_output_void);
      split->realp[0] = complex_input[0].real();  // DC component
      // Nyquist component (real part of last element) goes into imagp[0] for N
      // even
      split->imagp[0] =
          (N % 2 == 0 && Nc > N / 2) ? complex_input[N / 2].real() : 0.0;
      for (size_t k = 1; k < N / 2; ++k) {  // Components 1 to N/2 - 1
        split->realp[k] = complex_input[k].real();
        split->imagp[k] = complex_input[k].imag();
      }
    }
  }

  // --- Unpacking Helper: vDSP Split Complex to Complex Conjugate Even (CCE)
  // --- Converts vDSP split format (from RFFT) to standard complex array (size
  // N/2+1). Marked const as it only reads split_input_void and writes to
  // cce_output (passed by pointer) and doesn't modify member variables.
  void unpack_split_to_complex(const void *split_input_void,
                               std::complex<T> *cce_output) const {
    const size_t N = length;
    const size_t Nc = complex_length;  // Should be N/2 + 1
    if (Nc == 0 || N == 0) return;

    if constexpr (std::is_same_v<T, float>)  // SINGLE precision
    {
      const DSPSplitComplex *split =
          reinterpret_cast<const DSPSplitComplex *>(split_input_void);
      cce_output[0] =
          std::complex<T>(split->realp[0], 0.0f);  // DC component from realp[0]
      for (size_t k = 1; k < N / 2; ++k) {         // Components 1 to N/2 - 1
        cce_output[k] = std::complex<T>(split->realp[k], split->imagp[k]);
      }
      if (N % 2 == 0 &&
          Nc > N / 2) {  // Nyquist component (if N is even) from imagp[0]
        cce_output[N / 2] = std::complex<T>(split->imagp[0], 0.0f);
      }
      // If N is odd, Nc = (N-1)/2 + 1 = N/2 + 0.5 + 1 -> Nc = N/2 + 1.5 ? No.
      // Nc=(N+1)/2 If N is odd, the loop goes up to k < N/2, e.g. N=5, k<2.5 ->
      // k=1, 2. Nc=3. cce_output[0], cce_output[1], cce_output[2] are filled.
      // Correct.
    } else  // DOUBLE precision
    {
      const DSPDoubleSplitComplex *split =
          reinterpret_cast<const DSPDoubleSplitComplex *>(split_input_void);
      cce_output[0] = std::complex<T>(split->realp[0], 0.0);  // DC component
      for (size_t k = 1; k < N / 2; ++k) {  // Components 1 to N/2 - 1
        cce_output[k] = std::complex<T>(split->realp[k], split->imagp[k]);
      }
      if (N % 2 == 0 && Nc > N / 2) {  // Nyquist component
        cce_output[N / 2] = std::complex<T>(split->imagp[0], 0.0);
      }
    }
  }

  // --- Rule of 5/3: Non-Copyable ---
  FFTPlanImpl(const FFTPlanImpl &) = delete;
  FFTPlanImpl &operator=(const FFTPlanImpl &) = delete;
  // Move constructor/assignment defaulted in omnidsp.h

  // --- Execute Methods (FFT) ---
  // These methods are const because they operate using the pre-configured plan
  // (handles) and potentially temporary buffers, but don't modify the plan's
  // configuration itself.

  // Execute Complex-to-Complex FFT (Out-of-Place) using vDSP_DFT
  void execute_c2c_oop(const std::complex<T> *input,
                       std::complex<T> *output) const {
    if (!setup_handle_c2c)
      throw std::runtime_error("C2C FFT plan not initialized.");

    // vDSP_DFT expects separate real and imaginary pointers for interleaved
    // complex data.
    if constexpr (std::is_same_v<T, float>) {
      vDSP_DFT_Execute(
          reinterpret_cast<vDSP_DFT_Setup>(setup_handle_c2c),
          reinterpret_cast<const float *>(input),  // Real part of input
          reinterpret_cast<const float *>(input) +
              1,                              // Imag part of input (stride 2)
          reinterpret_cast<float *>(output),  // Real part of output
          reinterpret_cast<float *>(output) +
              1);  // Imag part of output (stride 2)
      // Apply external scaling if necessary (vDSP_DFT does not scale)
      T scale_factor =
          (direction == Direction::Forward) ? forward_scale : backward_scale;
      if (std::abs(scale_factor - 1.0f) >
          std::numeric_limits<float>::epsilon()) {
        // Scale the interleaved complex output (length * 2 floats)
        vDSP_vsmul(reinterpret_cast<const float *>(output), 1, &scale_factor,
                   reinterpret_cast<float *>(output), 1, length * 2);
      }
    } else  // Double precision
    {
      vDSP_DFT_ExecuteD(
          reinterpret_cast<vDSP_DFT_SetupD>(setup_handle_c2c),
          reinterpret_cast<const double *>(input),
          reinterpret_cast<const double *>(input) + 1,  // Stride 2
          reinterpret_cast<double *>(output),
          reinterpret_cast<double *>(output) + 1);  // Stride 2
      // Apply external scaling
      T scale_factor =
          (direction == Direction::Forward) ? forward_scale : backward_scale;
      if (std::abs(scale_factor - 1.0) >
          std::numeric_limits<double>::epsilon()) {
        // Scale the interleaved complex output (length * 2 doubles)
        vDSP_vsmulD(reinterpret_cast<const double *>(output), 1, &scale_factor,
                    reinterpret_cast<double *>(output), 1, length * 2);
      }
    }
  }

  // Execute Complex-to-Complex FFT (In-Place - Simulated)
  void execute_c2c_ip(std::complex<T> *data) const {
    if (!setup_handle_c2c)
      throw std::runtime_error("C2C FFT plan not initialized.");
    // vDSP_DFT does not support true in-place. Simulate using OOP.
    // Create a temporary copy of the input data
    std::vector<std::complex<T>> temp_input(data, data + length);
    // Execute out-of-place from the temporary input back to the original buffer
    execute_c2c_oop(temp_input.data(), data);
    // Note: Original implementation copied output to a temp vector then back.
    // This revised approach avoids one copy if execute_c2c_oop writes directly
    // to 'data'.
  }

  // Execute Real-to-Complex FFT (Out-of-Place) using vDSP_fft_zrip
  void execute_rfft_oop(const T *real_input,
                        std::complex<T> *complex_output) const {
    if (!setup_handle_real)
      throw std::runtime_error("Real FFT plan not initialized.");
    if (domain != Domain::Real || direction != Direction::Forward) {
      throw std::runtime_error(
          "execute_rfft called on incorrect plan type (must be Real/Forward).");
    }

    // vDSP_fft_zrip operates in-place on a split complex buffer.
    // We need mutable access to the internal real/imag buffers for this.
    // Use const_cast carefully, as we know vDSP_fft_zrip uses them as scratch
    // space.
    T *real_buf_ptr = const_cast<T *>(real_buffer.data());
    T *imag_buf_ptr = const_cast<T *>(imag_buffer.data());

    if constexpr (std::is_same_v<T, float>) {
      DSPSplitComplex split_buffer = {real_buf_ptr, imag_buf_ptr};
      // Convert real input to split complex format (writes to split_buffer)
      vDSP_ctoz(reinterpret_cast<const DSPComplex *>(real_input), 2,
                &split_buffer, 1, length / 2);
      // Perform FFT (modifies split_buffer in-place)
      vDSP_fft_zrip(setup_handle_real, &split_buffer, stride, get_log2n(length),
                    kFFTDirection_Forward);
      // Unpack result from split_buffer to standard CCE complex format (writes
      // to complex_output)
      unpack_split_to_complex(&split_buffer, complex_output);
      // Apply external scaling (reads and writes complex_output)
      if (std::abs(forward_scale - 1.0f) >
          std::numeric_limits<float>::epsilon()) {
        // Scale the interleaved complex output (complex_length * 2 floats)
        vDSP_vsmul(reinterpret_cast<float *>(complex_output), 1, &forward_scale,
                   reinterpret_cast<float *>(complex_output), 1,
                   complex_length * 2);
      }
    } else  // Double precision
    {
      DSPDoubleSplitComplex split_buffer = {real_buf_ptr, imag_buf_ptr};
      vDSP_ctozD(reinterpret_cast<const DSPDoubleComplex *>(real_input), 2,
                 &split_buffer, 1, length / 2);
      vDSP_fft_zripD(setup_handle_real, &split_buffer, stride,
                     get_log2n(length), kFFTDirection_Forward);
      unpack_split_to_complex(&split_buffer, complex_output);
      // Apply external scaling
      if (std::abs(forward_scale - 1.0) >
          std::numeric_limits<double>::epsilon()) {
        // Scale the interleaved complex output (complex_length * 2 doubles)
        vDSP_vsmulD(reinterpret_cast<double *>(complex_output), 1,
                    &forward_scale, reinterpret_cast<double *>(complex_output),
                    1, complex_length * 2);
      }
    }
  }

  // Execute Complex-to-Real Inverse FFT (Out-of-Place) using vDSP_fft_zrip
  void execute_irfft_oop(const std::complex<T> *complex_input,
                         T *real_output) const {
    if (!setup_handle_real)
      throw std::runtime_error("Real FFT plan not initialized.");
    if (domain != Domain::Real || direction != Direction::Inverse) {
      throw std::runtime_error(
          "execute_irfft called on incorrect plan type (must be "
          "Real/Inverse).");
    }

    // Need mutable access to internal buffers. Use const_cast carefully.
    T *real_buf_ptr = const_cast<T *>(real_buffer.data());
    T *imag_buf_ptr = const_cast<T *>(imag_buffer.data());

    // Pack CCE complex input into split complex format.
    if constexpr (std::is_same_v<T, float>) {
      DSPSplitComplex split_buffer = {real_buf_ptr, imag_buf_ptr};
      // Pack input into split_buffer
      pack_complex_to_split(complex_input, &split_buffer);
      // Perform inverse FFT (modifies split_buffer in-place)
      vDSP_fft_zrip(setup_handle_real, &split_buffer, stride, get_log2n(length),
                    kFFTDirection_Inverse);
      // Apply external scaling (modifies split_buffer before unpacking)
      // Note: vDSP inverse FFT implicitly scales by 2/N. backward_scale
      // includes adjustment for this.
      if (std::abs(backward_scale - 1.0f) >
          std::numeric_limits<float>::epsilon()) {
        vDSP_vsmul(split_buffer.realp, 1, &backward_scale, split_buffer.realp,
                   1, length / 2);
        vDSP_vsmul(split_buffer.imagp, 1, &backward_scale, split_buffer.imagp,
                   1, length / 2);
      }
      // Convert resulting split complex back to interleaved real output (writes
      // to real_output)
      vDSP_ztoc(&split_buffer, 1, reinterpret_cast<DSPComplex *>(real_output),
                2, length / 2);
    } else  // Double precision
    {
      DSPDoubleSplitComplex split_buffer = {real_buf_ptr, imag_buf_ptr};
      pack_complex_to_split(complex_input, &split_buffer);
      vDSP_fft_zripD(setup_handle_real, &split_buffer, stride,
                     get_log2n(length), kFFTDirection_Inverse);
      // Apply external scaling
      if (std::abs(backward_scale - 1.0) >
          std::numeric_limits<double>::epsilon()) {
        vDSP_vsmulD(split_buffer.realp, 1, &backward_scale, split_buffer.realp,
                    1, length / 2);
        vDSP_vsmulD(split_buffer.imagp, 1, &backward_scale, split_buffer.imagp,
                    1, length / 2);
      }
      vDSP_ztocD(&split_buffer, 1,
                 reinterpret_cast<DSPDoubleComplex *>(real_output), 2,
                 length / 2);
    }
  }
};  // End FFTPlanImpl struct

// --- Explicit Instantiations for FFTPlanImpl ---
template struct FFTPlanImpl<float>;
template struct FFTPlanImpl<double>;

// --- FFTPlan Method Definitions ---
// Link public FFTPlan methods to the Accelerate PIMPL implementation.

template <typename T>
FFTPlan<T>::FFTPlan(size_t l, Precision p, Direction d, Domain dom, NormMode n)
    : pimpl_(std::make_unique<FFTPlanImpl<T>>(l, p, d, dom, n)) {}

template <typename T>
FFTPlan<T>::~FFTPlan() = default;

// Move constructor/assignment defaulted in header

// Execute methods are const as they delegate to the const pimpl methods
template <typename T>
void FFTPlan<T>::execute(const std::complex<T> *i, std::complex<T> *o) const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
  pimpl_->execute_c2c_oop(i, o);
}
template <typename T>
void FFTPlan<T>::execute(std::complex<T> *d) const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
  pimpl_->execute_c2c_ip(d);
}
template <typename T>
void FFTPlan<T>::execute_rfft(const T *ri, std::complex<T> *co) const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
  pimpl_->execute_rfft_oop(ri, co);
}
template <typename T>
void FFTPlan<T>::execute_irfft(const std::complex<T> *ci, T *ro) const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
  pimpl_->execute_irfft_oop(ci, ro);
}

// Getters are const as they delegate to the const pimpl methods or return
// members
template <typename T>
size_t FFTPlan<T>::getLength() const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
  return pimpl_->length;
}
template <typename T>
size_t FFTPlan<T>::getComplexLength() const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
  return pimpl_->complex_length;
}
template <typename T>
Direction FFTPlan<T>::getDirection() const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
  return pimpl_->direction;
}
template <typename T>
Precision FFTPlan<T>::getPrecision() const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
  return pimpl_->precision;
}
template <typename T>
Domain FFTPlan<T>::getDomain() const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
  return pimpl_->domain;
}
template <typename T>
NormMode FFTPlan<T>::getNormMode() const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
  return pimpl_->norm_mode;
}

// --- Explicit Template Instantiations for FFTPlan Class ---
// Required when definitions are in a .cpp file.
// OMNIDSP_EXPORT is needed if building a shared library/DLL.
template class OMNIDSP_EXPORT OmniDSP::FFTPlan<float>;
template class OMNIDSP_EXPORT OmniDSP::FFTPlan<double>;

}  // namespace OmniDSP

#endif  // USE_ACCELERATE
