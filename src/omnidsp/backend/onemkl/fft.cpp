/**
 * @file src/omnidsp/backend/onemkl/fft.cpp
 * @brief Intel oneMKL backend implementation for OmniDSP FFTPlan and RFFTPlan.
 *
 * Implements the FFTPlanImpl/RFFTPlanImpl structures and FFTPlan/RFFTPlan
 * methods using Intel oneMKL DFTI routines. Compiled only when USE_ONEMKL is
 * defined. This version aligns with the updated fft.h API using overloaded plan
 * methods.
 */

// --- Includes ---
// Public API headers first
#include <OmniDSP/core_types.h>  // For OmniDSP::Precision
#include <OmniDSP/fft.h>         // Public Plan class declarations & Enums

#include <algorithm>  // For std::copy
#include <cmath>      // For std::sqrt
#include <complex>
#include <memory>     // For std::unique_ptr
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>
#include <type_traits>  // For std::is_same_v, std::conditional
#include <vector>

// Only compile if USE_ONEMKL is defined by CMake
#if defined(USE_ONEMKL)

#include <mkl.h>  // Main MKL header (includes DFTI)

namespace OmniDSP {

// --- MKL DFTI Helper ---
// Checks the status code returned by MKL DFTI functions and throws an error if
// it's not DFTI_NO_ERROR.
inline void check_mkl_status(MKL_LONG status, const char *error_msg) {
  if (!DftiErrorClass(status, DFTI_NO_ERROR)) {
    std::string full_msg = error_msg;
    // Append MKL status code and error message for detailed diagnostics.
    full_msg += ": MKL DFTI Status " + std::to_string(status) + " - " +
                DftiErrorMessage(status);
    throw std::runtime_error(full_msg);
  }
}

// --- FFTPlanImpl Definition (MKL Backend for C2C) ---
// Implementation details for FFTPlan (Complex-to-Complex).
template <typename T>
struct FFTPlanImpl {
  // --- Members ---
  size_t length = 0;                         // FFT Length (N)
  Direction direction = Direction::Forward;  // Primary direction plan supports
  Precision precision = Precision::SINGLE;   // float or double
  FFTNorm norm_mode = FFTNorm::Backward;     // Normalization mode
  DFTI_DESCRIPTOR_HANDLE handle = nullptr;   // MKL DFTI descriptor handle

  // --- Constructor ---
  FFTPlanImpl(size_t len, Precision prec, Direction dir, Domain dom,
              FFTNorm norm) {
    if (len == 0) throw std::invalid_argument("FFT length N cannot be zero.");
    // Verify template type matches precision enum
    if constexpr (std::is_same_v<T, float>) {
      if (prec != Precision::SINGLE)
        throw std::invalid_argument("Precision mismatch for float FFTPlan.");
    } else if constexpr (std::is_same_v<T, double>) {
      if (prec != Precision::DOUBLE)
        throw std::invalid_argument("Precision mismatch for double FFTPlan.");
    } else {
      throw std::invalid_argument("Unsupported template type for FFTPlan.");
    }
    // Verify domain is Complex for FFTPlan
    if (dom != Domain::Complex) {
      throw std::invalid_argument("FFTPlan requires Domain::Complex.");
    }

    length = len;
    direction = dir;  // Store the primary direction this plan is created for
    precision = prec;
    norm_mode = norm;  // Store the chosen normalization

    // Calculate scaling factors based on normalization mode
    T scaleN = static_cast<T>(length);
    T scaleSqrtN = std::sqrt(scaleN);
    T forward_scale = 1.0;
    T backward_scale = 1.0;

    // MKL applies the scale factor during computation.
    // We set DFTI_FORWARD_SCALE and DFTI_BACKWARD_SCALE accordingly.
    switch (norm_mode) {
      case FFTNorm::Backward:  // Forward unscaled, Inverse scaled by 1/N
        forward_scale = 1.0;
        backward_scale = (scaleN > 0) ? (1.0 / scaleN) : 1.0;
        break;
      case FFTNorm::Ortho:  // Forward and Inverse scaled by 1/sqrt(N)
        if (scaleSqrtN <= static_cast<T>(0.0))
          throw std::runtime_error("Cannot use Ortho norm with length 0 or 1.");
        forward_scale = 1.0 / scaleSqrtN;
        backward_scale = 1.0 / scaleSqrtN;
        break;
      case FFTNorm::Forward:  // Forward scaled by 1/N, Inverse unscaled
        if (scaleN <= static_cast<T>(0.0))
          throw std::runtime_error("Cannot use Forward norm with length 0.");
        forward_scale = 1.0 / scaleN;
        backward_scale = 1.0;
        break;
    }

    // --- MKL DFTI Descriptor Creation and Configuration ---
    MKL_LONG status;
    DFTI_CONFIG_VALUE mkl_domain_type = DFTI_COMPLEX;
    DFTI_CONFIG_VALUE mkl_prec_type =
        (prec == Precision::SINGLE) ? DFTI_SINGLE : DFTI_DOUBLE;

    // 1. Create Descriptor
    status = DftiCreateDescriptor(&handle, mkl_prec_type, mkl_domain_type, 1,
                                  (MKL_LONG)len);
    check_mkl_status(status, "MKL DftiCreateDescriptor failed (C2C)");

    // 2. Configure Descriptor
    // Set scaling factors
    using MKLScaleType = typename std::conditional<std::is_same_v<T, float>,
                                                   float, double>::type;
    status = DftiSetValue(handle, DFTI_FORWARD_SCALE,
                          static_cast<MKLScaleType>(forward_scale));
    check_mkl_status(status,
                     "MKL DftiSetValue DFTI_FORWARD_SCALE failed (C2C)");
    status = DftiSetValue(handle, DFTI_BACKWARD_SCALE,
                          static_cast<MKLScaleType>(backward_scale));
    check_mkl_status(status,
                     "MKL DftiSetValue DFTI_BACKWARD_SCALE failed (C2C)");

    // Set placement (always out-of-place for this implementation)
    status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    check_mkl_status(
        status, "MKL DftiSetValue DFTI_PLACEMENT failed for NOT_INPLACE (C2C)");

    // Set storage format (standard complex interleaved)
    status = DftiSetValue(handle, DFTI_COMPLEX_STORAGE, DFTI_COMPLEX_COMPLEX);
    check_mkl_status(status,
                     "MKL DftiSetValue DFTI_COMPLEX_STORAGE failed (C2C)");

    // 3. Commit Descriptor
    status = DftiCommitDescriptor(handle);
    check_mkl_status(status, "MKL DftiCommitDescriptor failed (C2C)");
  }

  // --- Destructor ---
  ~FFTPlanImpl() {
    if (handle) {
      // DftiFreeDescriptor returns a status, but we usually ignore it in
      // destructors as throwing from a destructor is problematic.
      (void)DftiFreeDescriptor(
          &handle);  // Cast to void to suppress unused result warning
      handle = nullptr;
    }
  }

  // --- Rule of 5/3: Non-Copyable ---
  FFTPlanImpl(const FFTPlanImpl &) = delete;
  FFTPlanImpl &operator=(const FFTPlanImpl &) = delete;
  // Move constructor/assignment defined for the public class FFTPlan

  // --- Internal Execute Methods (using MKL) ---
  // These are called by the public FFTPlan methods

  // Execute Complex-to-Complex Forward (Out-of-Place)
  void execute_fft_oop(const std::complex<T> *input,
                       std::complex<T> *output) const {
    if (!handle)
      throw std::runtime_error("MKL FFTPlan handle is null (execute_fft_oop).");
    // MKL input buffer can be const, but API requires non-const void*
    MKL_LONG status = DftiComputeForward(
        handle, const_cast<std::complex<T> *>(input), output);
    check_mkl_status(status, "MKL DftiComputeForward (C2C OOP) failed");
  }

  // Execute Complex-to-Complex Inverse (Out-of-Place)
  void execute_ifft_oop(const std::complex<T> *input,
                        std::complex<T> *output) const {
    if (!handle)
      throw std::runtime_error(
          "MKL FFTPlan handle is null (execute_ifft_oop).");
    // MKL input buffer can be const, but API requires non-const void*
    MKL_LONG status = DftiComputeBackward(
        handle, const_cast<std::complex<T> *>(input), output);
    check_mkl_status(status, "MKL DftiComputeBackward (C2C OOP) failed");
  }
};

// --- RFFTPlanImpl Definition (MKL Backend for R2C/C2R) ---
// Implementation details for RFFTPlan (Real transforms).
template <typename T>
struct RFFTPlanImpl {
  // --- Members ---
  size_t length = 0;          // Real FFT Length (N)
  size_t complex_length = 0;  // Length of complex spectrum (N/2 + 1)
  Direction direction = Direction::Forward;  // Primary direction plan supports
  Precision precision = Precision::SINGLE;   // float or double
  FFTNorm norm_mode = FFTNorm::Backward;     // Normalization mode
  DFTI_DESCRIPTOR_HANDLE handle = nullptr;   // MKL DFTI descriptor handle

  // --- Constructor ---
  RFFTPlanImpl(size_t len, Precision prec, Direction dir, Domain dom,
               FFTNorm norm) {
    if (len == 0) throw std::invalid_argument("RFFT length N cannot be zero.");
    // Verify template type matches precision enum
    if constexpr (std::is_same_v<T, float>) {
      if (prec != Precision::SINGLE)
        throw std::invalid_argument("Precision mismatch for float RFFTPlan.");
    } else if constexpr (std::is_same_v<T, double>) {
      if (prec != Precision::DOUBLE)
        throw std::invalid_argument("Precision mismatch for double RFFTPlan.");
    } else {
      throw std::invalid_argument("Unsupported template type for RFFTPlan.");
    }
    // Verify domain is Real for RFFTPlan
    if (dom != Domain::Real) {
      throw std::invalid_argument("RFFTPlan requires Domain::Real.");
    }

    length = len;
    complex_length = len / 2 + 1;
    direction = dir;  // Store the primary direction
    precision = prec;
    norm_mode = norm;  // Store the chosen normalization

    // Calculate scaling factors (same logic as C2C)
    T scaleN = static_cast<T>(length);
    T scaleSqrtN = std::sqrt(scaleN);
    T forward_scale = 1.0;
    T backward_scale = 1.0;

    switch (norm_mode) {
      case FFTNorm::Backward:
        forward_scale = 1.0;
        backward_scale = (scaleN > 0) ? (1.0 / scaleN) : 1.0;
        break;
      case FFTNorm::Ortho:
        if (scaleSqrtN <= static_cast<T>(0.0))
          throw std::runtime_error("Cannot use Ortho norm with length 0 or 1.");
        forward_scale = 1.0 / scaleSqrtN;
        backward_scale = 1.0 / scaleSqrtN;
        break;
      case FFTNorm::Forward:
        if (scaleN <= static_cast<T>(0.0))
          throw std::runtime_error("Cannot use Forward norm with length 0.");
        forward_scale = 1.0 / scaleN;
        backward_scale = 1.0;
        break;
    }

    // --- MKL DFTI Descriptor Creation and Configuration ---
    MKL_LONG status;
    DFTI_CONFIG_VALUE mkl_domain_type = DFTI_REAL;  // Real domain for RFFTPlan
    DFTI_CONFIG_VALUE mkl_prec_type =
        (prec == Precision::SINGLE) ? DFTI_SINGLE : DFTI_DOUBLE;

    // 1. Create Descriptor
    status = DftiCreateDescriptor(&handle, mkl_prec_type, mkl_domain_type, 1,
                                  (MKL_LONG)len);
    check_mkl_status(status, "MKL DftiCreateDescriptor failed (Real)");

    // 2. Configure Descriptor
    // Set scaling factors
    using MKLScaleType = typename std::conditional<std::is_same_v<T, float>,
                                                   float, double>::type;
    status = DftiSetValue(handle, DFTI_FORWARD_SCALE,
                          static_cast<MKLScaleType>(forward_scale));
    check_mkl_status(status,
                     "MKL DftiSetValue DFTI_FORWARD_SCALE failed (Real)");
    status = DftiSetValue(handle, DFTI_BACKWARD_SCALE,
                          static_cast<MKLScaleType>(backward_scale));
    check_mkl_status(status,
                     "MKL DftiSetValue DFTI_BACKWARD_SCALE failed (Real)");

    // Configure storage format for real transforms (use standard CCE packed
    // format) DFTI_COMPLEX_COMPLEX means the complex output (size N/2+1) is
    // stored as a standard complex array.
    status =
        DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    check_mkl_status(
        status, "MKL DftiSetValue DFTI_CONJUGATE_EVEN_STORAGE failed (Real)");

    // Set placement (always out-of-place)
    status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    check_mkl_status(
        status,
        "MKL DftiSetValue DFTI_PLACEMENT failed for NOT_INPLACE (Real)");

    // 3. Commit Descriptor
    status = DftiCommitDescriptor(handle);
    check_mkl_status(status, "MKL DftiCommitDescriptor failed (Real)");
  }

  // --- Destructor ---
  ~RFFTPlanImpl() {
    if (handle) {
      (void)DftiFreeDescriptor(&handle);
      handle = nullptr;
    }
  }

  // --- Rule of 5/3: Non-Copyable ---
  RFFTPlanImpl(const RFFTPlanImpl &) = delete;
  RFFTPlanImpl &operator=(const RFFTPlanImpl &) = delete;
  // Move constructor/assignment defined for the public class RFFTPlan

  // --- Internal Execute Methods (using MKL) ---
  // These are called by the public RFFTPlan methods

  // Real-to-Complex Out-of-Place (RFFT)
  void execute_rfft_oop(const T *real_input,
                        std::complex<T> *complex_output) const {
    if (!handle)
      throw std::runtime_error(
          "MKL RFFTPlan handle is null (execute_rfft_oop).");
    // MKL input buffer can be const, but API requires non-const void*
    MKL_LONG status =
        DftiComputeForward(handle, const_cast<T *>(real_input), complex_output);
    check_mkl_status(status, "MKL DftiComputeForward (RFFT OOP) failed");
  }

  // Complex-to-Real Out-of-Place (IRFFT)
  void execute_irfft_oop(const std::complex<T> *complex_input,
                         T *real_output) const {
    if (!handle)
      throw std::runtime_error(
          "MKL RFFTPlan handle is null (execute_irfft_oop).");
    // MKL input buffer can be const, but API requires non-const void*
    MKL_LONG status = DftiComputeBackward(
        handle, const_cast<std::complex<T> *>(complex_input), real_output);
    check_mkl_status(status, "MKL DftiComputeBackward (IRFFT OOP) failed");
  }
};

// --- Explicit Instantiations for Impl Structs ---
// These MUST come AFTER the full definitions.
template struct FFTPlanImpl<float>;
template struct FFTPlanImpl<double>;
template struct RFFTPlanImpl<float>;
template struct RFFTPlanImpl<double>;

// --- FFTPlan Method Definitions ---
// Bridge the public API class to the PIMPL implementation

template <typename T>
FFTPlan<T>::FFTPlan(size_t l, Precision p, Direction d, Domain dom, FFTNorm n)
    : pimpl_(std::make_unique<FFTPlanImpl<T>>(l, p, d, dom, n)) {}

template <typename T>
FFTPlan<T>::~FFTPlan() = default;
template <typename T>
FFTPlan<T>::FFTPlan(FFTPlan &&) noexcept = default;
template <typename T>
FFTPlan<T> &FFTPlan<T>::operator=(FFTPlan &&) noexcept = default;

template <typename T>
void FFTPlan<T>::fft(const std::vector<std::complex<T>> &input,
                     std::vector<std::complex<T>> &output) {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
  // Check if the plan was created for the correct direction
  if (pimpl_->direction != Direction::Forward) {
    throw std::runtime_error(
        "fft() called on a plan not created with Direction::Forward.");
  }
  // Check input size
  if (input.size() != pimpl_->length) {
    throw std::invalid_argument("Input vector size (" +
                                std::to_string(input.size()) +
                                ") does not match FFT plan length (" +
                                std::to_string(pimpl_->length) + ").");
  }
  // Ensure output vector is the correct size
  if (output.size() != pimpl_->length) {
    try {
      output.resize(pimpl_->length);
    } catch (const std::bad_alloc &e) {
      throw std::runtime_error(
          "Failed to resize output vector in FFTPlan::fft: " +
          std::string(e.what()));
    }
  }
  // Call the backend implementation's execute method
  pimpl_->execute_fft_oop(input.data(), output.data());
}

template <typename T>
void FFTPlan<T>::ifft(const std::vector<std::complex<T>> &input,
                      std::vector<std::complex<T>> &output) {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
  // Check if the plan was created for the correct direction
  if (pimpl_->direction != Direction::Inverse) {
    throw std::runtime_error(
        "ifft() called on a plan not created with Direction::Inverse.");
  }
  // Check input size
  if (input.size() != pimpl_->length) {
    throw std::invalid_argument("Input vector size (" +
                                std::to_string(input.size()) +
                                ") does not match FFT plan length (" +
                                std::to_string(pimpl_->length) + ").");
  }
  // Ensure output vector is the correct size
  if (output.size() != pimpl_->length) {
    try {
      output.resize(pimpl_->length);
    } catch (const std::bad_alloc &e) {
      throw std::runtime_error(
          "Failed to resize output vector in FFTPlan::ifft: " +
          std::string(e.what()));
    }
  }
  // Call the backend implementation's execute method
  pimpl_->execute_ifft_oop(input.data(), output.data());
}

template <typename T>
size_t FFTPlan<T>::getSize() const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
  return pimpl_->length;
}

// --- RFFTPlan Method Definitions ---
// Bridge the public API class to the PIMPL implementation

template <typename T>
RFFTPlan<T>::RFFTPlan(size_t l, Precision p, Direction d, Domain dom, FFTNorm n)
    : pimpl_(std::make_unique<RFFTPlanImpl<T>>(l, p, d, dom, n)) {}

template <typename T>
RFFTPlan<T>::~RFFTPlan() = default;
template <typename T>
RFFTPlan<T>::RFFTPlan(RFFTPlan &&) noexcept = default;
template <typename T>
RFFTPlan<T> &RFFTPlan<T>::operator=(RFFTPlan &&) noexcept = default;

template <typename T>
void RFFTPlan<T>::rfft(const std::vector<T> &input,
                       std::vector<std::complex<T>> &output) {
  if (!pimpl_)
    throw std::runtime_error("Invalid RFFTPlan (moved-from or uninitialized).");
  // Check direction
  if (pimpl_->direction != Direction::Forward) {
    throw std::runtime_error(
        "rfft() called on a plan not created with Direction::Forward.");
  }
  // Check input size
  if (input.size() != pimpl_->length) {
    throw std::invalid_argument("Input vector size (" +
                                std::to_string(input.size()) +
                                ") does not match RFFT plan length (" +
                                std::to_string(pimpl_->length) + ").");
  }
  // Ensure output vector is the correct size (N/2 + 1)
  if (output.size() != pimpl_->complex_length) {
    try {
      output.resize(pimpl_->complex_length);
    } catch (const std::bad_alloc &e) {
      throw std::runtime_error(
          "Failed to resize output vector in RFFTPlan::rfft: " +
          std::string(e.what()));
    }
  }
  // Call backend implementation
  pimpl_->execute_rfft_oop(input.data(), output.data());
}

template <typename T>
void RFFTPlan<T>::irfft(const std::vector<std::complex<T>> &input,
                        std::vector<T> &output) {
  if (!pimpl_)
    throw std::runtime_error("Invalid RFFTPlan (moved-from or uninitialized).");
  // Check direction
  if (pimpl_->direction != Direction::Inverse) {
    throw std::runtime_error(
        "irfft() called on a plan not created with Direction::Inverse.");
  }
  // Check input size (N/2 + 1)
  if (input.size() != pimpl_->complex_length) {
    throw std::invalid_argument("Input vector size (" +
                                std::to_string(input.size()) +
                                ") does not match RFFT plan complex length (" +
                                std::to_string(pimpl_->complex_length) + ").");
  }
  // Ensure output vector is the correct size (N)
  if (output.size() != pimpl_->length) {
    try {
      output.resize(pimpl_->length);
    } catch (const std::bad_alloc &e) {
      throw std::runtime_error(
          "Failed to resize output vector in RFFTPlan::irfft: " +
          std::string(e.what()));
    }
  }
  // Call backend implementation
  pimpl_->execute_irfft_oop(input.data(), output.data());
}

template <typename T>
size_t RFFTPlan<T>::getSize() const {
  if (!pimpl_)
    throw std::runtime_error("Invalid RFFTPlan (moved-from or uninitialized).");
  return pimpl_->length;  // Return the real length N
}

// --- Explicit Template Instantiations for Public Plan Classes & Methods ---
// These ensure the classes and methods are generated for float and double.
// OMNIDSP_EXPORT might be needed for shared libs/DLLs if defined elsewhere.
#ifndef OMNIDSP_EXPORT
#define OMNIDSP_EXPORT  // Define as empty if not building DLLs or handled
                        // differently
#endif

// FFTPlan Class & Methods
template class OMNIDSP_EXPORT OmniDSP::FFTPlan<float>;
template class OMNIDSP_EXPORT OmniDSP::FFTPlan<double>;
template void OMNIDSP_EXPORT
FFTPlan<float>::fft(const std::vector<std::complex<float>> &,
                    std::vector<std::complex<float>> &);
template void OMNIDSP_EXPORT
FFTPlan<double>::fft(const std::vector<std::complex<double>> &,
                     std::vector<std::complex<double>> &);
template void OMNIDSP_EXPORT
FFTPlan<float>::ifft(const std::vector<std::complex<float>> &,
                     std::vector<std::complex<float>> &);
template void OMNIDSP_EXPORT
FFTPlan<double>::ifft(const std::vector<std::complex<double>> &,
                      std::vector<std::complex<double>> &);
template size_t OMNIDSP_EXPORT FFTPlan<float>::getSize() const;
template size_t OMNIDSP_EXPORT FFTPlan<double>::getSize() const;

// RFFTPlan Class & Methods
template class OMNIDSP_EXPORT OmniDSP::RFFTPlan<float>;
template class OMNIDSP_EXPORT OmniDSP::RFFTPlan<double>;
template void OMNIDSP_EXPORT RFFTPlan<float>::rfft(
    const std::vector<float> &, std::vector<std::complex<float>> &);
template void OMNIDSP_EXPORT RFFTPlan<double>::rfft(
    const std::vector<double> &, std::vector<std::complex<double>> &);
template void OMNIDSP_EXPORT RFFTPlan<float>::irfft(
    const std::vector<std::complex<float>> &, std::vector<float> &);
template void OMNIDSP_EXPORT RFFTPlan<double>::irfft(
    const std::vector<std::complex<double>> &, std::vector<double> &);
template size_t OMNIDSP_EXPORT RFFTPlan<float>::getSize() const;
template size_t OMNIDSP_EXPORT RFFTPlan<double>::getSize() const;

}  // namespace OmniDSP

#endif  // USE_ONEMKL
