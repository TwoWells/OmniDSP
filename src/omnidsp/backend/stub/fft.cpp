/**
 * @file src/omnidsp/backend/stub/fft.cpp
 * @brief Stub (Error) backend implementation for OmniDSP FFTPlan.
 *
 * Provides stub implementations for FFTPlanImpl and FFTPlan methods.
 * Compiled only when no real backend (oneMKL or Accelerate) is selected.
 * Any attempt to create or use an FFTPlan will result in a std::runtime_error.
 */

// --- Includes ---
#include <OmniDSP/omnidsp.h>  // Public API header

#include <cmath>  // For dummy scale calculation in stub FFTPlanImpl
#include <complex>
#include <memory>     // For std::unique_ptr
#include <stdexcept>  // For std::runtime_error
#include <string>
#include <type_traits>  // For std::is_same_v
#include <vector>

#include "../backend.h"  // Internal backend function declarations (not directly used here)

// Compile this only if NEITHER Accelerate nor MKL is defined by CMake
#if !defined(USE_ACCELERATE) && !defined(USE_ONEMKL)

namespace OmniDSP {

// --- FFTPlanImpl Definition (Stub) ---
// This struct holds placeholder members but its constructor and methods throw
// errors.
template <typename T>
struct FFTPlanImpl {
  // --- Members (Stored but not functionally used) ---
  size_t length = 0;
  size_t complex_length = 0;
  Direction direction = Direction::Forward;
  Precision precision = Precision::SINGLE;
  Domain domain = Domain::Complex;
  FFTNorm norm_mode = FFTNorm::Backward;
  T forward_scale = 1.0;
  T backward_scale = 1.0;

  // --- Constructor (Throws Error) ---
  FFTPlanImpl(size_t len, Precision prec, Direction dir, Domain dom,
              FFTNorm norm) {
    // Immediately throw an error upon attempted creation.
    std::string error_msg =
        "OmniDSP backend (MKL/Accelerate) not selected or available during "
        "build. ";
    error_msg += "Cannot create FFTPlan.";
    throw std::runtime_error(error_msg);

    // --- Unreachable code ---
    // Initialize members to silence potential compiler warnings, although
    // the exception above means this code will never actually execute.
    length = len;
    complex_length = (dom == Domain::Real) ? len / 2 + 1 : len;
    direction = dir;
    precision = prec;
    domain = dom;
    norm_mode = norm;
    // Dummy scale calculations
    T scaleN = static_cast<T>(length);
    T scaleSqrtN = std::sqrt(scaleN);
    switch (norm_mode) {
      case FFTNorm::Backward:
        forward_scale = 1.0;
        backward_scale = 1.0 / scaleN;
        break;
      case FFTNorm::Ortho:
        forward_scale = 1.0 / scaleSqrtN;
        backward_scale = 1.0 / scaleSqrtN;
        break;
      case FFTNorm::Forward:
        forward_scale = 1.0 / scaleN;
        backward_scale = 1.0;
        break;
    }
    // --- End Unreachable code ---
  }

  // --- Destructor ---
  ~FFTPlanImpl() = default;  // Default destructor is sufficient

  // --- Rule of 5/3: Non-Copyable ---
  FFTPlanImpl(const FFTPlanImpl &) = delete;
  FFTPlanImpl &operator=(const FFTPlanImpl &) = delete;
  // Move constructor/assignment defaulted in omnidsp.h for the public class

  // --- Execute Methods (Throw Error) ---
  // These methods are defined but will throw if ever called (which shouldn't
  // happen if the constructor throws as intended).
  void execute_c2c_oop(const std::complex<T> *, std::complex<T> *) const {
    throw std::runtime_error(
        "OmniDSP backend not available (stub execute_c2c_oop called).");
  }
  void execute_c2c_ip(std::complex<T> *) const {
    throw std::runtime_error(
        "OmniDSP backend not available (stub execute_c2c_ip called).");
  }
  void execute_rfft_oop(const T *, std::complex<T> *) const {
    throw std::runtime_error(
        "OmniDSP backend not available (stub execute_rfft_oop called).");
  }
  void execute_irfft_oop(const std::complex<T> *, T *) const {
    throw std::runtime_error(
        "OmniDSP backend not available (stub execute_irfft_oop called).");
  }
};  // End FFTPlanImpl struct

// --- Explicit Instantiations for FFTPlanImpl (Stub) ---
template struct FFTPlanImpl<float>;
template struct FFTPlanImpl<double>;

// --- FFTPlan Method Definitions (Stub) ---
// These definitions link the public FFTPlan class methods to the stub PIMPL
// implementation. The constructor will call the throwing FFTPlanImpl
// constructor.

template <typename T>
FFTPlan<T>::FFTPlan(size_t l, Precision p, Direction d, Domain dom, FFTNorm n)
    : pimpl_(std::make_unique<FFTPlanImpl<T>>(l, p, d, dom, n)) {
}  // This line throws

template <typename T>
FFTPlan<T>::~FFTPlan() = default;  // unique_ptr handles pimpl_ deletion

// Move constructor/assignment are defaulted in the header (omnidsp.h)

// Execute methods will likely never be reached if constructor throws,
// but define them to call the throwing implementation methods.
template <typename T>
void FFTPlan<T>::execute(const std::complex<T> *i, std::complex<T> *o) const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (stub - moved-from?).");
  pimpl_->execute_c2c_oop(i, o);
}
template <typename T>
void FFTPlan<T>::execute(std::complex<T> *d) const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (stub - moved-from?).");
  pimpl_->execute_c2c_ip(d);
}
template <typename T>
void FFTPlan<T>::execute_rfft(const T *ri, std::complex<T> *co) const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (stub - moved-from?).");
  pimpl_->execute_rfft_oop(ri, co);
}
template <typename T>
void FFTPlan<T>::execute_irfft(const std::complex<T> *ci, T *ro) const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (stub - moved-from?).");
  pimpl_->execute_irfft_oop(ci, ro);
}

// Getters might be called on a moved-from object if not careful, add checks.
// They will return default values if called after the constructor throws (which
// shouldn't happen).
template <typename T>
size_t FFTPlan<T>::getLength() const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (stub - moved-from?).");
  return pimpl_->length;  // Returns 0 if constructor failed
}
template <typename T>
size_t FFTPlan<T>::getComplexLength() const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (stub - moved-from?).");
  return pimpl_->complex_length;  // Returns 0 if constructor failed
}
template <typename T>
Direction FFTPlan<T>::getDirection() const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (stub - moved-from?).");
  return pimpl_->direction;
}
template <typename T>
Precision FFTPlan<T>::getPrecision() const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (stub - moved-from?).");
  return pimpl_->precision;
}
template <typename T>
Domain FFTPlan<T>::getDomain() const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (stub - moved-from?).");
  return pimpl_->domain;
}
template <typename T>
FFTNorm FFTPlan<T>::getFFTNorm() const {
  if (!pimpl_)
    throw std::runtime_error("Invalid FFTPlan (stub - moved-from?).");
  return pimpl_->norm_mode;
}

// --- Explicit Template Instantiations for FFTPlan Class ---
// These ensure the public FFTPlan class itself is instantiated and exported
// correctly even when linking against the stub implementation. OMNIDSP_EXPORT
// is needed here if building a shared library/DLL.
template class OMNIDSP_EXPORT OmniDSP::FFTPlan<float>;
template class OMNIDSP_EXPORT OmniDSP::FFTPlan<double>;

}  // namespace OmniDSP

#endif  // !USE_ACCELERATE && !USE_ONEMKL
