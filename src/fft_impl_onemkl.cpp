/**
 * @file fft_impl_onemkl.cpp
 * @brief Intel oneMKL backend implementation for OmniFFT.
 *
 * This file contains the concrete implementation of the FFTPlanImpl class
 * using Intel's oneAPI Math Kernel Library (oneMKL) DFTI routines. It is
 * compiled only when USE_ONEMKL is defined by the CMake build system.
 */

#include <OmniFFT/omnifft.h> // Public API header (defines FFTPlan, enums, FFTPlanImpl forward decl)

#if defined(USE_ONEMKL) // Only compile contents if oneMKL backend is selected

#include <mkl_dfti.h> // oneMKL DFTI interface header
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <complex>    // For std::complex
#include <memory>     // For std::unique_ptr, std::make_unique
#include <vector>     // Potentially needed for helper functions if added
#include <algorithm>  // Potentially needed for helper functions if added
#include <type_traits>// For std::is_same_v, std::conditional
#include <string>     // For std::string in error messages
#include <cmath>      // For std::sqrt

namespace OmniFFT {

/**
 * @internal
 * @brief Helper function to check MKL status codes and throw runtime_error on failure.
 * @param status The MKL status code returned by a DFTI function.
 * @param error_msg A descriptive message prefix for the exception.
 * @throws std::runtime_error If the status indicates an error.
 */
inline void check_mkl_status(MKL_LONG status, const char* error_msg) {
    // DFTI_NO_ERROR is typically 0, DftiErrorClass checks against other error classes.
    if (status != DFTI_NO_ERROR && !DftiErrorClass(status, DFTI_NO_ERROR)) {
        std::string full_msg = error_msg;
        full_msg += ": ";
        full_msg += DftiErrorMessage(status); // Get MKL error string
        throw std::runtime_error(full_msg);
    }
}

/**
 * @internal
 * @brief oneMKL backend implementation details for FFTPlan.
 *
 * This struct holds the MKL-specific descriptor handle and implements
 * the FFT execution using MKL DFTI functions based on the parameters
 * provided during FFTPlan construction.
 * @tparam T float or double
 */
template <typename T>
struct FFTPlanImpl {
    // Store configuration parameters
    size_t length = 0;        ///< N: Number of points in the real domain.
    size_t complex_length = 0;///< Nc = N/2 + 1: Number of complex points for REAL domain.
    Direction direction = Direction::FORWARD; ///< Transform direction.
    Precision precision = Precision::SINGLE; ///< Data precision.
    Domain domain = Domain::COMPLEX;       ///< Transform domain (COMPLEX or REAL).
    NormMode norm_mode = NormMode::BACKWARD; ///< Normalization mode.

    // Store calculated scale factors (for reference, MKL uses internal factors set via API)
    T forward_scale = 1.0;
    T backward_scale = 1.0;

    // MKL specific handle
    DFTI_DESCRIPTOR_HANDLE handle = nullptr; ///< The core MKL DFTI descriptor.

    /**
     * @brief Constructs and configures the oneMKL DFTI descriptor.
     * @param len Length 'N' of the transform.
     * @param prec Precision (SINGLE or DOUBLE).
     * @param dir Direction (FORWARD or INVERSE).
     * @param dom Domain (COMPLEX or REAL).
     * @param norm Normalization mode (BACKWARD, ORTHO, FORWARD).
     */
    FFTPlanImpl(size_t len, Precision prec, Direction dir, Domain dom, NormMode norm) :
        length(len), direction(dir), precision(prec), domain(dom), norm_mode(norm)
    {
         if (length == 0) { throw std::invalid_argument("FFT length N cannot be zero."); }
         // Calculate complex length for REAL domain transforms
         if (domain == Domain::REAL) { complex_length = length / 2 + 1; }
         else { complex_length = length; } // For C2C, complex length is N

        // Calculate desired abstract scale factors based on norm mode
        T scaleN = static_cast<T>(length);
        T scaleSqrtN = std::sqrt(scaleN);

        // Ensure scaleN and scaleSqrtN are not zero before division
        if (scaleN == static_cast<T>(0.0)) {
             throw std::runtime_error("Cannot calculate scaling factors for N=0");
        }
         if (scaleSqrtN == static_cast<T>(0.0) && norm_mode == NormMode::ORTHO) {
             throw std::runtime_error("Cannot use ORTHO norm mode for N=0");
         }


        switch (norm_mode) {
            case NormMode::BACKWARD: // fwd=1, bwd=1/N
                forward_scale = static_cast<T>(1.0);
                backward_scale = static_cast<T>(1.0) / scaleN;
                break;
            case NormMode::ORTHO: // fwd=1/sqrt(N), bwd=1/sqrt(N)
                forward_scale = static_cast<T>(1.0) / scaleSqrtN;
                backward_scale = static_cast<T>(1.0) / scaleSqrtN;
                break;
            case NormMode::FORWARD: // fwd=1/N, bwd=1
                forward_scale = static_cast<T>(1.0) / scaleN;
                backward_scale = static_cast<T>(1.0);
                break;
        }

        // --- MKL Descriptor Setup ---
        MKL_LONG status;
        DFTI_CONFIG_VALUE domain_type = (domain == Domain::REAL) ? DFTI_REAL : DFTI_COMPLEX;
        DFTI_CONFIG_VALUE prec_type = (precision == Precision::SINGLE) ? DFTI_SINGLE : DFTI_DOUBLE;

        // 1. Create Descriptor
        status = DftiCreateDescriptor(&handle, prec_type, domain_type, 1, (MKL_LONG)length);
        check_mkl_status(status, "Failed to create oneMKL DFTI descriptor");

        // 2. Set Scaling Factors explicitly for MKL
        // MKL requires float for DFTI_SINGLE precision scales, double otherwise
        using MKLScaleType = typename std::conditional<std::is_same_v<T, float>, float, double>::type;
        status = DftiSetValue(handle, DFTI_FORWARD_SCALE, static_cast<MKLScaleType>(forward_scale));
        check_mkl_status(status, "Failed to set MKL forward scale");
        status = DftiSetValue(handle, DFTI_BACKWARD_SCALE, static_cast<MKLScaleType>(backward_scale));
        check_mkl_status(status, "Failed to set MKL backward scale");

        // 3. Configure Storage (for REAL transforms)
        if (domain == Domain::REAL) {
            // Use standard Complex Conjugate Even storage compatible with std::complex<T> output
             status = DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
             check_mkl_status(status, "Failed to set MKL conjugate even storage");
        }

        // 4. Set Placement (defaulting to out-of-place for simplicity in this API version)
        // MKL supports in-place for C2C and R2C/C2R, but requires careful buffer management.
        status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        check_mkl_status(status, "Failed to set oneMKL DFTI placement to out-of-place");

        // 5. Commit Descriptor - MKL performs setup/optimizations here
        status = DftiCommitDescriptor(handle);
        check_mkl_status(status, "Failed to commit oneMKL DFTI descriptor");
    }

    /**
     * @brief Destroys the MKL DFTI descriptor and frees resources.
     */
    ~FFTPlanImpl() {
        if (handle) {
            // Errors during free are usually ignored or logged, not thrown from destructor
            (void)DftiFreeDescriptor(&handle);
        }
    }

    /** @brief Implements C2C out-of-place execution. */
    void execute_c2c_oop(const std::complex<T>* input, std::complex<T>* output) const {
        MKL_LONG status;
        if (direction == Direction::FORWARD) {
            // MKL DFTI requires non-const input pointer even for out-of-place
            status = DftiComputeForward(handle, const_cast<std::complex<T>*>(input), output);
        } else { // INVERSE
            status = DftiComputeBackward(handle, const_cast<std::complex<T>*>(input), output);
        }
         check_mkl_status(status, "Failed to execute oneMKL C2C DFTI compute (OOP)");
    }

    /** @brief Implements C2C in-place execution. */
    void execute_c2c_ip(std::complex<T>* data) const {
        // Ensure plan is configured for in-place execution
        MKL_LONG current_placement;
        DftiGetValue(handle, DFTI_PLACEMENT, &current_placement);
        if (current_placement != DFTI_INPLACE) {
             // Reconfigure - NOTE: This modifies state in a const method via handle.
             // Potential issue if called concurrently on the same FFTPlan. Consider non-const execute_inplace.
             MKL_LONG status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_INPLACE);
             check_mkl_status(status, "Failed to set MKL placement to in-place");
             status = DftiCommitDescriptor(handle); // Re-commit needed after changing placement
             check_mkl_status(status, "Failed to re-commit MKL descriptor for in-place");
        }

        MKL_LONG status;
        if (direction == Direction::FORWARD) {
            status = DftiComputeForward(handle, data);
        } else { // INVERSE
            status = DftiComputeBackward(handle, data);
        }
        check_mkl_status(status, "Failed to execute oneMKL C2C DFTI compute (in-place)");
    }

    /** @brief Implements R2C forward out-of-place execution. */
    void execute_rfft_oop(const T* real_input, std::complex<T>* complex_output) const {
         // Assumes descriptor configured for REAL domain, CCE storage, and OOP.
         // MKL requires non-const input pointer.
         MKL_LONG status = DftiComputeForward(handle, const_cast<T*>(real_input), complex_output);
         check_mkl_status(status, "Failed to execute oneMKL RFFT (R2C) compute");
    }

    /** @brief Implements C2R inverse out-of-place execution. */
    void execute_irfft_oop(const std::complex<T>* complex_input, T* real_output) const {
         // Assumes descriptor configured for REAL domain, CCE storage, and OOP.
         // MKL requires non-const input pointer.
         MKL_LONG status = DftiComputeBackward(handle, const_cast<std::complex<T>*>(complex_input), real_output);
         check_mkl_status(status, "Failed to execute oneMKL IRFFT (C2R) compute");
    }

     // --- Rule of 5/3: Move Semantics ---
     /** @brief Move constructor. */
     FFTPlanImpl(FFTPlanImpl&& other) noexcept :
         length(other.length),
         complex_length(other.complex_length),
         direction(other.direction),
         precision(other.precision),
         domain(other.domain),
         norm_mode(other.norm_mode),
         forward_scale(other.forward_scale),
         backward_scale(other.backward_scale),
         handle(other.handle) // Transfer ownership of handle
     {
        // Ensure the moved-from object doesn't own the handle anymore
        other.handle = nullptr;
     }

     /** @brief Move assignment operator. */
     FFTPlanImpl& operator=(FFTPlanImpl&& other) noexcept {
          if (this != &other) {
            // Release existing resource before overwriting
            if (handle) { (void)DftiFreeDescriptor(&handle); }
            // Transfer ownership from other
            length = other.length;
            complex_length = other.complex_length;
            direction = other.direction;
            precision = other.precision;
            domain = other.domain;
            norm_mode = other.norm_mode;
            forward_scale = other.forward_scale;
            backward_scale = other.backward_scale;
            handle = other.handle;
            // Nullify the source handle
            other.handle = nullptr;
         }
         return *this;
     }

     // --- Delete Copy Semantics ---
     /** @brief Deleted copy constructor. */
     FFTPlanImpl(const FFTPlanImpl&) = delete;
     /** @brief Deleted copy assignment operator. */
     FFTPlanImpl& operator=(const FFTPlanImpl&) = delete;
};


// --- FFTPlan Method Definitions ---
// These are defined here where FFTPlanImpl is fully defined.

/** @brief FFTPlan constructor definition (forwards to Impl). */
template <typename T>
FFTPlan<T>::FFTPlan(size_t length, Precision precision, Direction direction, Domain domain, NormMode norm)
    : pimpl_(std::make_unique<FFTPlanImpl<T>>(length, precision, direction, domain, norm)) {}

/** @brief FFTPlan destructor definition. */
template <typename T> FFTPlan<T>::~FFTPlan() = default; // Default works because Impl is defined

/** @brief FFTPlan move constructor definition. */
template <typename T> FFTPlan<T>::FFTPlan(FFTPlan&& other) noexcept = default; // Default works

/** @brief FFTPlan move assignment definition. */
template <typename T> FFTPlan<T>& FFTPlan<T>::operator=(FFTPlan<T>&& other) noexcept = default; // Default works

// Execute methods: Check preconditions and forward to Impl
template <typename T>
void FFTPlan<T>::execute(const std::complex<T>* input, std::complex<T>* output) const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (moved from or init failed).");
    if (pimpl_->domain != Domain::COMPLEX) {
        throw std::runtime_error("Execute (C2C) called on FFTPlan created for REAL domain.");
    }
    pimpl_->execute_c2c_oop(input, output);
}

template <typename T>
void FFTPlan<T>::execute(std::complex<T>* data) const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (moved from or init failed).");
    if (pimpl_->domain != Domain::COMPLEX) {
        throw std::runtime_error("Execute (C2C in-place) called on FFTPlan created for REAL domain.");
    }
    pimpl_->execute_c2c_ip(data);
}

template <typename T>
void FFTPlan<T>::execute_rfft(const T* real_input, std::complex<T>* complex_output) const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (moved from or init failed).");
     if (pimpl_->domain != Domain::REAL || pimpl_->direction != Direction::FORWARD) {
        throw std::runtime_error("execute_rfft requires Domain::REAL and Direction::FORWARD.");
     }
     pimpl_->execute_rfft_oop(real_input, complex_output);
}

template <typename T>
void FFTPlan<T>::execute_irfft(const std::complex<T>* complex_input, T* real_output) const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (moved from or init failed).");
     if (pimpl_->domain != Domain::REAL || pimpl_->direction != Direction::INVERSE) {
        throw std::runtime_error("execute_irfft requires Domain::REAL and Direction::INVERSE.");
     }
     pimpl_->execute_irfft_oop(complex_input, real_output);
}

// Getters: Check Pimpl and return value from Impl
template <typename T> size_t FFTPlan<T>::getLength() const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (moved from or init failed)."); return pimpl_->length;
}
template <typename T> size_t FFTPlan<T>::getComplexLength() const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (moved from or init failed)."); return pimpl_->complex_length;
}
template <typename T> Direction FFTPlan<T>::getDirection() const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (moved from or init failed)."); return pimpl_->direction;
}
template <typename T> Precision FFTPlan<T>::getPrecision() const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (moved from or init failed)."); return pimpl_->precision;
}
template <typename T> Domain FFTPlan<T>::getDomain() const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (moved from or init failed)."); return pimpl_->domain;
}
template <typename T> NormMode FFTPlan<T>::getNormMode() const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (moved from or init failed)."); return pimpl_->norm_mode;
}


// --- Explicit Instantiations ---
// Instantiate the implementation class template for float and double.
template struct FFTPlanImpl<float>;
template struct FFTPlanImpl<double>;

// Instantiate the public wrapper class template for float and double.
// This forces the compiler to generate code for all FFTPlan methods.
template class FFTPlan<float>;
template class FFTPlan<double>;

} // namespace OmniFFT

#endif // USE_ONEMKL