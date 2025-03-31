// src/fft_impl_stub.cpp

#include "omnifft.h" // Contains FFTPlanImpl forward decl, FFTPlan decl, enums

// Compile this only if NEITHER Accelerate nor MKL is used by CMake defines
#if !defined(USE_ACCELERATE) && !defined(USE_ONEMKL)

#include <stdexcept>
#include <complex>
#include <memory>      // For std::unique_ptr, std::make_unique
#include <vector>
#include <type_traits> // For std::is_same_v
#include <cmath>       // For std::floor

namespace OmniFFT {

// --- Stub FFTPlanImpl Definition (Error Implementation) ---
template <typename T>
struct FFTPlanImpl {
    size_t length = 0;
    size_t complex_length = 0;
    Direction direction = Direction::FORWARD;
    Precision precision = Precision::SINGLE;
    Domain domain = Domain::COMPLEX;

    // Constructor: Immediately throws
    FFTPlanImpl(size_t len, Precision prec, Direction dir, Domain dom) {
        throw std::runtime_error("FFT library backend not selected/available during build. "
                                 "Cannot create FFTPlan. Please ensure oneMKL or Accelerate is found and enabled.");
        // Initialize members anyway to silence potential warnings
        length = len;
        direction = dir;
        precision = prec;
        domain = dom;
        complex_length = (dom == Domain::REAL) ? len / 2 + 1 : 0;
    }

    // Destructor: Nothing to do
    ~FFTPlanImpl() = default;

    // Execute methods: Throw if somehow called
    void execute_c2c_oop(const std::complex<T>* input, std::complex<T>* output) const {
         throw std::runtime_error("FFT backend not available (stub execute called).");
    }
     void execute_c2c_ip(std::complex<T>* data) const {
         throw std::runtime_error("FFT backend not available (stub execute called).");
     }
     void execute_r2c_oop(const T* real_input, std::complex<T>* complex_output) const {
         throw std::runtime_error("FFT backend not available (stub execute called).");
    }
     void execute_c2r_oop(const std::complex<T>* complex_input, T* real_output) const {
         throw std::runtime_error("FFT backend not available (stub execute called).");
    }

     // --- Move Semantics (default is sufficient) ---
     FFTPlanImpl(FFTPlanImpl&&) noexcept = default;
     FFTPlanImpl& operator=(FFTPlanImpl&&) noexcept = default;

     // --- Delete Copy Semantics ---
     FFTPlanImpl(const FFTPlanImpl&) = delete;
     FFTPlanImpl& operator=(const FFTPlanImpl&) = delete;
};


// --- FFTPlan Method Definitions ---

template <typename T>
FFTPlan<T>::FFTPlan(size_t length, Precision precision, Direction direction, Domain domain)
    // This will call the throwing FFTPlanImpl constructor
    : pimpl_(std::make_unique<FFTPlanImpl<T>>(length, precision, direction, domain)) {}

template <typename T> FFTPlan<T>::~FFTPlan() = default;
template <typename T> FFTPlan<T>::FFTPlan(FFTPlan&& other) noexcept = default;
template <typename T> FFTPlan<T>& FFTPlan<T>::operator=(FFTPlan<T>&& other) noexcept = default;

// C2C Execute Methods
template <typename T>
void FFTPlan<T>::execute(const std::complex<T>* input, std::complex<T>* output) const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (init likely failed).");
    // Call would throw anyway if pimpl_ existed
    pimpl_->execute_c2c_oop(input, output);
}

template <typename T>
void FFTPlan<T>::execute(std::complex<T>* data) const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (init likely failed).");
    // Call would throw anyway if pimpl_ existed
    pimpl_->execute_c2c_ip(data);
}

// R2C / C2R Execute Methods
template <typename T>
void FFTPlan<T>::execute_r2c(const T* real_input, std::complex<T>* complex_output) const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (init likely failed).");
     // Call would throw anyway if pimpl_ existed
     pimpl_->execute_r2c_oop(real_input, complex_output);
}

template <typename T>
void FFTPlan<T>::execute_c2r(const std::complex<T>* complex_input, T* real_output) const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (init likely failed).");
     // Call would throw anyway if pimpl_ existed
     pimpl_->execute_c2r_oop(complex_input, real_output);
}

// Getters
template <typename T> size_t FFTPlan<T>::getLength() const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (init likely failed).");
    return pimpl_->length;
}
template <typename T> size_t FFTPlan<T>::getComplexLength() const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (init likely failed).");
    return pimpl_->complex_length;
}
template <typename T> Direction FFTPlan<T>::getDirection() const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (init likely failed).");
     return pimpl_->direction;
}
template <typename T> Precision FFTPlan<T>::getPrecision() const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (init likely failed).");
     return pimpl_->precision;
}
template <typename T> Domain FFTPlan<T>::getDomain() const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (init likely failed).");
     return pimpl_->domain;
}


// --- Explicit Instantiations ---
template struct FFTPlanImpl<float>;
template struct FFTPlanImpl<double>;
template class FFTPlan<float>;
template class FFTPlan<double>;

} // namespace OmniFFT

#endif // !USE_ACCELERATE && !USE_ONEMKL