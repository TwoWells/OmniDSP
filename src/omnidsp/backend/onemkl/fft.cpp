/**
 * @file src/omnidsp/backend/onemkl/fft.cpp
 * @brief Intel oneMKL backend implementation for OmniDSP FFTPlan.
 *
 * Implements the FFTPlanImpl structure and FFTPlan methods using
 * Intel oneMKL DFTI routines. Compiled only when USE_ONEMKL is defined.
 */

// --- Includes ---
#include <OmniDSP/omnidsp.h>  // Public API header

#include <algorithm>  // For std::copy
#include <cmath>      // For std::sqrt
#include <complex>
#include <iomanip>    // For std::setprecision (optional)
#include <iostream>   // For std::cout/cerr (optional, for potential debug)
#include <memory>     // For std::unique_ptr
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>
#include <type_traits>  // For std::is_same_v, std::conditional
#include <vector>

#include "../backend_impl.h"  // Internal backend function declarations (though not directly used here)

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

// --- FFTPlanImpl Definition (MKL Backend) ---
// This struct holds the MKL-specific data (DFTI descriptor) and logic for
// FFTPlan.
template <typename T>
struct FFTPlanImpl {
  // --- Members ---
  size_t length = 0;  // FFT Length (N)
  size_t complex_length =
      0;  // Length of complex spectrum (N for C2C, N/2+1 for Real)
  Direction direction = Direction::FORWARD;  // Transform direction
  Precision precision = Precision::SINGLE;   // float or double
  Domain domain = Domain::COMPLEX;           // COMPLEX or REAL
  NormMode norm_mode = NormMode::BACKWARD;   // Normalization mode
  T forward_scale = 1.0;   // Scaling factor applied to forward transform
  T backward_scale = 1.0;  // Scaling factor applied to inverse transform
  DFTI_DESCRIPTOR_HANDLE handle = nullptr;  // MKL DFTI descriptor handle

  // --- Constructor ---
  // Creates and configures the MKL DFTI descriptor based on plan parameters.
  FFTPlanImpl(size_t len, Precision prec, Direction dir, Domain dom,
              NormMode norm) {
    if (len == 0) throw std::invalid_argument("FFT length N cannot be zero.");
    length = len;
    direction = dir;
    precision = prec;
    domain = dom;
    norm_mode = norm;
    // Calculate complex length based on domain
    complex_length = (dom == Domain::REAL) ? len / 2 + 1 : len;

    // Calculate scaling factors based on normalization mode
    T scaleN = static_cast<T>(length);
    T scaleSqrtN = std::sqrt(scaleN);
    // Basic validation for scaling factors
    if (scaleN == static_cast<T>(0.0))
      throw std::runtime_error("ScaleN is zero (length must be > 0)");
    if (scaleSqrtN == static_cast<T>(0.0) && norm_mode == NormMode::ORTHO)
      throw std::runtime_error(
          "ScaleSqrtN is zero for ORTHO (length must be > 0)");

    switch (norm_mode) {
      case NormMode::BACKWARD:  // Forward unscaled, Inverse scaled by 1/N
        forward_scale = 1.0;
        backward_scale = 1.0 / scaleN;
        break;
      case NormMode::ORTHO:  // Forward and Inverse scaled by 1/sqrt(N)
        forward_scale = 1.0 / scaleSqrtN;
        backward_scale = 1.0 / scaleSqrtN;
        break;
      case NormMode::FORWARD:  // Forward scaled by 1/N, Inverse unscaled
        forward_scale = 1.0 / scaleN;
        backward_scale = 1.0;
        break;
    }

    // --- MKL DFTI Descriptor Creation and Configuration ---
    MKL_LONG status;
    // Determine MKL domain and precision types
    DFTI_CONFIG_VALUE domain_type =
        (dom == Domain::REAL) ? DFTI_REAL : DFTI_COMPLEX;
    DFTI_CONFIG_VALUE prec_type =
        (prec == Precision::SINGLE) ? DFTI_SINGLE : DFTI_DOUBLE;

    // Create the descriptor
    status =
        DftiCreateDescriptor(&handle, prec_type, domain_type, 1, (MKL_LONG)len);
    check_mkl_status(status, "MKL DftiCreateDescriptor failed");

    // Set scaling factors
    // Note: MKL uses float/double directly for scale values, not complex
    using MKLScaleType = typename std::conditional<std::is_same_v<T, float>,
                                                   float, double>::type;
    status = DftiSetValue(handle, DFTI_FORWARD_SCALE,
                          static_cast<MKLScaleType>(forward_scale));
    check_mkl_status(status, "MKL DftiSetValue DFTI_FORWARD_SCALE failed");
    status = DftiSetValue(handle, DFTI_BACKWARD_SCALE,
                          static_cast<MKLScaleType>(backward_scale));
    check_mkl_status(status, "MKL DftiSetValue DFTI_BACKWARD_SCALE failed");

    // Configure storage format for real transforms (use standard CCE)
    if (dom == Domain::REAL) {
      status = DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE,
                            DFTI_COMPLEX_COMPLEX);
      check_mkl_status(status,
                       "MKL DftiSetValue DFTI_CONJUGATE_EVEN_STORAGE failed");
    }

    // Configure for out-of-place transforms initially
    // In-place execution is handled by simulating with a temporary buffer if
    // needed.
    status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    check_mkl_status(status,
                     "MKL DftiSetValue DFTI_PLACEMENT failed for NOT_INPLACE");

    // Commit the descriptor configuration
    status = DftiCommitDescriptor(handle);
    check_mkl_status(status, "MKL DftiCommitDescriptor failed");
  }

  // --- Destructor ---
  // Frees the MKL DFTI descriptor.
  ~FFTPlanImpl() {
    if (handle) {
      // DftiFreeDescriptor returns a status, but we usually ignore it in
      // destructors.
      (void)DftiFreeDescriptor(&handle);
      handle = nullptr;  // Prevent double-free
    }
  }

  // --- Rule of 5/3: Non-Copyable ---
  FFTPlanImpl(const FFTPlanImpl &) = delete;
  FFTPlanImpl &operator=(const FFTPlanImpl &) = delete;
  // Move constructor/assignment are defaulted in omnidsp.h for the public class

  // --- Execute Methods ---

  // Complex-to-Complex Out-of-Place
  void execute_c2c_oop(const std::complex<T> *input,
                       std::complex<T> *output) const {
    if (!handle)
      throw std::runtime_error("MKL FFTPlan handle is null (execute_c2c_oop).");
    MKL_LONG status;
    if (direction == Direction::FORWARD) {
      // MKL input buffer can be const, but API requires non-const void*
      status = DftiComputeForward(handle, const_cast<std::complex<T> *>(input),
                                  output);
    } else {  // INVERSE
      status = DftiComputeBackward(handle, const_cast<std::complex<T> *>(input),
                                   output);
    }
    check_mkl_status(status,
                     "MKL DftiComputeForward/Backward (C2C OOP) failed");
  }

  // Complex-to-Complex In-Place (Simulated)
  void execute_c2c_ip(std::complex<T> *data) const {
    if (!handle)
      throw std::runtime_error("MKL FFTPlan handle is null (execute_c2c_ip).");
    MKL_LONG status;
    // MKL DFTI doesn't guarantee true in-place for all configurations.
    // Simulate using the out-of-place version with a temporary buffer for
    // safety.
    std::vector<std::complex<T>> temp_output(length);
    if (direction == Direction::FORWARD) {
      status = DftiComputeForward(handle, data, temp_output.data());
    } else {  // INVERSE
      status = DftiComputeBackward(handle, data, temp_output.data());
    }
    check_mkl_status(
        status, "MKL DftiComputeForward/Backward (C2C IP simulation) failed");
    // Copy result back to the original buffer
    std::copy(temp_output.begin(), temp_output.end(), data);
  }

  // Real-to-Complex Out-of-Place (RFFT)
  void execute_rfft_oop(const T *real_input,
                        std::complex<T> *complex_output) const {
    if (!handle)
      throw std::runtime_error(
          "MKL FFTPlan handle is null (execute_rfft_oop).");
    if (domain != Domain::REAL || direction != Direction::FORWARD) {
      throw std::runtime_error(
          "execute_rfft called on incorrect plan type (must be REAL/FORWARD).");
    }
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
          "MKL FFTPlan handle is null (execute_irfft_oop).");
    if (domain != Domain::REAL || direction != Direction::INVERSE) {
      throw std::runtime_error(
          "execute_irfft called on incorrect plan type (must be "
          "REAL/INVERSE).");
    }
    // MKL input buffer can be const, but API requires non-const void*
    MKL_LONG status = DftiComputeBackward(
        handle, const_cast<std::complex<T> *>(complex_input), real_output);
    check_mkl_status(status, "MKL DftiComputeBackward (IRFFT OOP) failed");
  }
};  // End FFTPlanImpl struct

// --- Explicit Instantiations for FFTPlanImpl ---
// Ensures the compiler generates code for both float and double versions.
template struct FFTPlanImpl<float>;
template struct FFTPlanImpl<double>;

// --- FFTPlan Method Definitions ---
// These definitions link the public FFTPlan class methods to the PIMPL
// implementation.

template <typename T>
FFTPlan<T>::FFTPlan(size_t l, Precision p, Direction d, Domain dom, NormMode n)
    : pimpl_(std::make_unique<FFTPlanImpl<T>>(l, p, d, dom, n)) {}

template <typename T>
FFTPlan<T>::~FFTPlan() = default;  // unique_ptr handles pimpl_ deletion

// Move constructor/assignment are defaulted in the header (omnidsp.h)

// Execute methods forward calls to the implementation.
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

// Getters forward calls to the implementation.
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
// These ensure the public FFTPlan class itself is instantiated and exported
// correctly. OMNIDSP_EXPORT is needed here if building a shared library/DLL.
template class OMNIDSP_EXPORT OmniDSP::FFTPlan<float>;
template class OMNIDSP_EXPORT OmniDSP::FFTPlan<double>;

}  // namespace OmniDSP

#endif  // USE_ONEMKL
