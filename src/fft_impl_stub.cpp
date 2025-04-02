/**
 * @file fft_impl_stub.cpp
 * @brief Stub (Error) backend implementation for OmniDSP.
 *
 * This file contains a stub implementation of the FFTPlanImpl class.
 * It is compiled only when no real backend (oneMKL or Accelerate) is
 * selected by the CMake build system (i.e., neither USE_ONEMKL nor
 * USE_ACCELERATE is defined).
 *
 * Any attempt to create an FFTPlan when this backend is active will
 * result in a std::runtime_error being thrown, indicating that a
 * functional FFT backend was not available during the build.
 *
 * @version 1.0.0
 * @date 2025-03-31
 */

#include <OmniDSP/omnidsp.h> // Public API header

// Compile this only if NEITHER Accelerate nor MKL is defined by CMake
#if !defined(USE_ACCELERATE) && !defined(USE_ONEMKL)

#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <complex>    // For std::complex (used in method signatures)
#include <memory>     // For std::unique_ptr, std::make_unique
#include <vector>     // Used in method signatures
#include <type_traits>// For std::is_same_v
#include <cmath>      // For std::sqrt (needed for dummy scale calculation)
#include <string>     // For exception messages

namespace OmniDSP {

/**
 * @internal
 * @brief Stub backend implementation that throws errors upon use.
 *
 * This struct provides the necessary interface for FFTPlanImpl, but its
 * constructor throws an exception, preventing successful FFTPlan creation
 * when no functional backend is available. Execute methods also throw
 * if they were somehow reached (which shouldn't happen if construction fails).
 * @tparam T float or double
 */
template <typename T>
struct FFTPlanImpl {
    // --- Configuration Members (Stored but not used functionally) ---
    size_t length = 0;        ///< N: Number of points in the real domain.
    size_t complex_length = 0;///< Nc = N/2 + 1: Number of complex points for REAL domain.
    Direction direction = Direction::FORWARD; ///< Transform direction.
    Precision precision = Precision::SINGLE; ///< Data precision.
    Domain domain = Domain::COMPLEX;       ///< Transform domain (COMPLEX or REAL).
    NormMode norm_mode = NormMode::BACKWARD; ///< Normalization mode.

    // --- Dummy Scaling Factors ---
    T forward_scale = 1.0;
    T backward_scale = 1.0;

    // --- No Backend Handles ---

    /**
     * @brief Stub constructor: Throws std::runtime_error immediately.
     * @param len Length 'N' of the transform.
     * @param prec Precision (SINGLE or DOUBLE).
     * @param dir Direction (FORWARD or INVERSE).
     * @param dom Domain (COMPLEX or REAL).
     * @param norm Normalization mode (BACKWARD, ORTHO, FORWARD).
     * @throws std::runtime_error Always throws to indicate no backend is available.
     */
    FFTPlanImpl(size_t len, Precision prec, Direction dir, Domain dom, NormMode norm) {
        // Construct error message before throwing
        std::string error_msg = "OmniDSP backend not selected/available during build. ";
        error_msg += "Cannot create FFTPlan. Please ensure a supported backend (oneMKL or Accelerate) is found during CMake configuration and enabled.";
        throw std::runtime_error(error_msg);

        // Unreachable code: Initialize members anyway to potentially silence compiler warnings
        // about uninitialized members, although the throw prevents reaching here.
        length = len; direction = dir; precision = prec; domain = dom; norm_mode = norm;
        complex_length = (dom == Domain::REAL) ? len / 2 + 1 : len;
        // Dummy scale calculation (unreachable)
        T scaleN = static_cast<T>(length);
        T scaleSqrtN = std::sqrt(scaleN);
        if (scaleN != static_cast<T>(0.0)) {
            switch (norm_mode) {
                case NormMode::BACKWARD: forward_scale = 1.0; backward_scale = 1.0 / scaleN; break;
                case NormMode::ORTHO:    if (scaleSqrtN != static_cast<T>(0.0)) { forward_scale = 1.0 / scaleSqrtN; backward_scale = 1.0 / scaleSqrtN; } break;
                case NormMode::FORWARD:  forward_scale = 1.0 / scaleN; backward_scale = 1.0; break;
            }
        }
    }

    /**
     * @brief Stub destructor (does nothing).
     */
    ~FFTPlanImpl() = default;

    // --- Stub Execute Methods (All Throw) ---

    /** @brief Stub: Throws runtime_error. */
    void execute_c2c_oop(const std::complex<T>* input, std::complex<T>* output) const {
         throw std::runtime_error("OmniDSP backend not available (stub execute_c2c_oop called).");
    }
     /** @brief Stub: Throws runtime_error. */
     void execute_c2c_ip(std::complex<T>* data) const {
         throw std::runtime_error("OmniDSP backend not available (stub execute_c2c_ip called).");
     }
     /** @brief Stub: Throws runtime_error. */
     void execute_rfft_oop(const T* real_input, std::complex<T>* complex_output) const {
         throw std::runtime_error("OmniDSP backend not available (stub execute_rfft_oop called).");
    }
     /** @brief Stub: Throws runtime_error. */
     void execute_irfft_oop(const std::complex<T>* complex_input, T* real_output) const {
         throw std::runtime_error("OmniDSP backend not available (stub execute_irfft_oop called).");
    }

     // --- Rule of 5/3: Move Semantics (Default is sufficient) ---
     /** @brief Default move constructor. */
     FFTPlanImpl(FFTPlanImpl&&) noexcept = default;
     /** @brief Default move assignment operator. */
     FFTPlanImpl& operator=(FFTPlanImpl&&) noexcept = default;

     // --- Delete Copy Semantics ---
     /** @brief Deleted copy constructor. */
     FFTPlanImpl(const FFTPlanImpl&) = delete;
     /** @brief Deleted copy assignment operator. */
     FFTPlanImpl& operator=(const FFTPlanImpl&) = delete;
};


// --- FFTPlan Method Definitions ---
// Defined here where FFTPlanImpl is complete (although unusable).
// The FFTPlan constructor will fail because the FFTPlanImpl constructor throws.

/** @brief FFTPlan constructor definition (attempts to create throwing Impl). */
template <typename T>
FFTPlan<T>::FFTPlan(size_t length, Precision precision, Direction direction, Domain domain, NormMode norm)
    // This call to std::make_unique will invoke the FFTPlanImpl constructor, which throws.
    : pimpl_(std::make_unique<FFTPlanImpl<T>>(length, precision, direction, domain, norm)) {}

/** @brief FFTPlan destructor definition. */
template <typename T> FFTPlan<T>::~FFTPlan() = default; // Default is OK

/** @brief FFTPlan move constructor definition. */
template <typename T> FFTPlan<T>::FFTPlan(FFTPlan&& other) noexcept = default; // Default is OK

/** @brief FFTPlan move assignment definition. */
template <typename T> FFTPlan<T>& FFTPlan<T>::operator=(FFTPlan<T>&& other) noexcept = default; // Default is OK

// Execute methods: Check Pimpl (likely null or invalid) and forward to Impl (which throws).
// These will likely throw due to the null check before even reaching the Impl call.
/** @brief FFTPlan C2C execute OOP definition. */
template <typename T>
void FFTPlan<T>::execute(const std::complex<T>* input, std::complex<T>* output) const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (likely backend init failed).");
    pimpl_->execute_c2c_oop(input, output);
}
/** @brief FFTPlan C2C execute IP definition. */
template <typename T>
void FFTPlan<T>::execute(std::complex<T>* data) const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (likely backend init failed).");
    pimpl_->execute_c2c_ip(data);
}
/** @brief FFTPlan RFFT execute definition. */
template <typename T>
void FFTPlan<T>::execute_rfft(const T* real_input, std::complex<T>* complex_output) const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (likely backend init failed).");
     pimpl_->execute_rfft_oop(real_input, complex_output);
}
/** @brief FFTPlan IRFFT execute definition. */
template <typename T>
void FFTPlan<T>::execute_irfft(const std::complex<T>* complex_input, T* real_output) const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (likely backend init failed).");
     pimpl_->execute_irfft_oop(complex_input, real_output);
}

// Getters: Check Pimpl (likely null or invalid) and return member from Impl.
/** @brief FFTPlan getLength definition. */
template <typename T> size_t FFTPlan<T>::getLength() const { if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (likely backend init failed)."); return pimpl_->length; }
/** @brief FFTPlan getComplexLength definition. */
template <typename T> size_t FFTPlan<T>::getComplexLength() const { if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (likely backend init failed)."); return pimpl_->complex_length; }
/** @brief FFTPlan getDirection definition. */
template <typename T> Direction FFTPlan<T>::getDirection() const { if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (likely backend init failed)."); return pimpl_->direction; }
/** @brief FFTPlan getPrecision definition. */
template <typename T> Precision FFTPlan<T>::getPrecision() const { if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (likely backend init failed)."); return pimpl_->precision; }
/** @brief FFTPlan getDomain definition. */
template <typename T> Domain FFTPlan<T>::getDomain() const { if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (likely backend init failed)."); return pimpl_->domain; }
/** @brief FFTPlan getNormMode definition. */
template <typename T> NormMode FFTPlan<T>::getNormMode() const { if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (likely backend init failed)."); return pimpl_->norm_mode; }


// --- Explicit Instantiations ---
// Instantiate the stub implementation class template for float and double.
template struct FFTPlanImpl<float>;
template struct FFTPlanImpl<double>;

// Instantiate the public wrapper class template for float and double using the stub definitions.
template class FFTPlan<float>;
template class FFTPlan<double>;

} // namespace OmniDSP

#endif // !USE_ACCELERATE && !USE_ONEMKL