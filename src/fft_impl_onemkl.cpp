// src/fft_impl_onemkl.cpp

#include "omnifft.h" // Contains FFTPlanImpl forward decl, FFTPlan decl, enums

#if defined(USE_ONEMKL) // Guard ensures this compiles only when MKL is selected

#include <mkl.h>
#include <stdexcept>
#include <complex>
#include <memory>      // For std::unique_ptr, std::make_unique
#include <vector>
#include <algorithm>
#include <type_traits> // For std::is_same_v
#include <string>      // For std::string in error messages
#include <cmath>       // For std::floor

namespace OmniFFT {

// Helper to check MKL status
inline void check_mkl_status(MKL_LONG status, const char* error_msg) {
    if (status != DFTI_NO_ERROR && !DftiErrorClass(status, DFTI_NO_ERROR)) {
        std::string full_msg = error_msg;
        full_msg += ": ";
        full_msg += DftiErrorMessage(status);
        throw std::runtime_error(full_msg);
    }
}

// --- oneMKL FFTPlanImpl Definition ---
template <typename T>
struct FFTPlanImpl {
    size_t length = 0;        // N: Real domain length
    size_t complex_length = 0;// Nc = N/2 + 1
    Direction direction = Direction::FORWARD;
    Precision precision = Precision::SINGLE;
    Domain domain = Domain::COMPLEX; // Store the domain

    DFTI_DESCRIPTOR_HANDLE handle = nullptr;
    // In-place configuration state not strictly needed for R2C/C2R if always out-of-place
    // bool is_in_place = false; // MKL R2C/C2R does support in-place, but API is simpler out-of-place

    // Constructor
    FFTPlanImpl(size_t len, Precision prec, Direction dir, Domain dom) :
        length(len), direction(dir), precision(prec), domain(dom)
    {
         if (length == 0) {
            throw std::invalid_argument("FFT length cannot be zero.");
        }
        if (domain == Domain::REAL) {
            complex_length = length / 2 + 1;
        } else {
            complex_length = 0; // Not applicable for C2C
        }

        MKL_LONG status;
        DFTI_CONFIG_VALUE domain_type = (domain == Domain::REAL) ? DFTI_REAL : DFTI_COMPLEX;
        DFTI_CONFIG_VALUE prec_type = (precision == Precision::SINGLE) ? DFTI_SINGLE : DFTI_DOUBLE;

        status = DftiCreateDescriptor(&handle, prec_type, domain_type, 1, (MKL_LONG)length);
        check_mkl_status(status, "Failed to create oneMKL DFTI descriptor");

        // Configure storage for REAL domain transforms to use standard complex layout
        if (domain == Domain::REAL) {
            // Use standard CCE storage (array of std::complex<T>) for the complex spectrum
             status = DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
             check_mkl_status(status, "Failed to set MKL conjugate even storage");
             // MKL defaults typically place real/imag parts contiguously, matching std::complex
        }

        // Set placement - R2C/C2R often used out-of-place for simplicity with this API
        // Note: MKL does support in-place for R2C/C2R, but requires careful buffer handling/overlap.
        status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        check_mkl_status(status, "Failed to set oneMKL DFTI placement to out-of-place");

        // Note: MKL scaling for R2C/C2R: Forward=1.0, Backward=1.0/N (default, usually desired)

        status = DftiCommitDescriptor(handle);
        check_mkl_status(status, "Failed to commit oneMKL DFTI descriptor");
    }

    // Destructor
    ~FFTPlanImpl() {
        if (handle) {
            (void)DftiFreeDescriptor(&handle);
        }
    }

    // Execute C2C Out-of-Place
    void execute_c2c_oop(const std::complex<T>* input, std::complex<T>* output) const {
        MKL_LONG status;
        if (direction == Direction::FORWARD) {
            status = DftiComputeForward(handle, const_cast<std::complex<T>*>(input), output);
        } else {
            status = DftiComputeBackward(handle, const_cast<std::complex<T>*>(input), output);
        }
         check_mkl_status(status, "Failed to execute oneMKL C2C DFTI compute");
    }

    // Execute C2C In-Place
    void execute_c2c_ip(std::complex<T>* data) const {
        // Need to reconfigure for in-place if not already configured
        MKL_LONG current_placement;
        DftiGetValue(handle, DFTI_PLACEMENT, &current_placement);
        if (current_placement != DFTI_INPLACE) {
             MKL_LONG status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_INPLACE);
             check_mkl_status(status, "Failed to set MKL placement to in-place");
             status = DftiCommitDescriptor(handle);
             check_mkl_status(status, "Failed to re-commit MKL descriptor for in-place");
             // Note: Modifying const object state requires care - potential race if called concurrently
             // A non-const method or separate plans might be safer.
        }

        MKL_LONG status;
        if (direction == Direction::FORWARD) {
            status = DftiComputeForward(handle, data);
        } else {
            status = DftiComputeBackward(handle, data);
        }
        check_mkl_status(status, "Failed to execute oneMKL C2C DFTI compute (in-place)");
    }

    // Execute R2C Forward Out-of-Place
    void execute_r2c_oop(const T* real_input, std::complex<T>* complex_output) const {
         // Assumes DFTI_REAL descriptor, DFTI_CONJUGATE_EVEN_STORAGE=DFTI_COMPLEX_COMPLEX
         // Assumes out-of-place configuration
         MKL_LONG status = DftiComputeForward(handle, const_cast<T*>(real_input), complex_output);
         check_mkl_status(status, "Failed to execute oneMKL R2C DFTI compute");
    }

    // Execute C2R Inverse Out-of-Place
    void execute_c2r_oop(const std::complex<T>* complex_input, T* real_output) const {
         // Assumes DFTI_REAL descriptor, DFTI_CONJUGATE_EVEN_STORAGE=DFTI_COMPLEX_COMPLEX
         // Assumes out-of-place configuration
         MKL_LONG status = DftiComputeBackward(handle, const_cast<std::complex<T>*>(complex_input), real_output);
         check_mkl_status(status, "Failed to execute oneMKL C2R DFTI compute");
    }

     // --- Move Semantics ---
     FFTPlanImpl(FFTPlanImpl&& other) noexcept :
         length(other.length),
         complex_length(other.complex_length),
         direction(other.direction),
         precision(other.precision),
         domain(other.domain),
         handle(other.handle)
     {
        other.handle = nullptr;
     }

     FFTPlanImpl& operator=(FFTPlanImpl&& other) noexcept {
          if (this != &other) {
            if (handle) { (void)DftiFreeDescriptor(&handle); }
            length = other.length;
            complex_length = other.complex_length;
            direction = other.direction;
            precision = other.precision;
            domain = other.domain;
            handle = other.handle;
            other.handle = nullptr;
         }
         return *this;
     }

     // --- Delete Copy Semantics ---
     FFTPlanImpl(const FFTPlanImpl&) = delete;
     FFTPlanImpl& operator=(const FFTPlanImpl&) = delete;
};


// --- FFTPlan Method Definitions ---

template <typename T>
FFTPlan<T>::FFTPlan(size_t length, Precision precision, Direction direction, Domain domain)
    // Pass all arguments to the Impl constructor
    : pimpl_(std::make_unique<FFTPlanImpl<T>>(length, precision, direction, domain)) {}

template <typename T> FFTPlan<T>::~FFTPlan() = default;
template <typename T> FFTPlan<T>::FFTPlan(FFTPlan&& other) noexcept = default;
template <typename T> FFTPlan<T>& FFTPlan<T>::operator=(FFTPlan<T>&& other) noexcept = default;

// C2C Execute Methods
template <typename T>
void FFTPlan<T>::execute(const std::complex<T>* input, std::complex<T>* output) const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid.");
    if (pimpl_->domain != Domain::COMPLEX) {
        throw std::runtime_error("Execute (C2C) called on FFTPlan created for REAL domain.");
    }
    pimpl_->execute_c2c_oop(input, output);
}

template <typename T>
void FFTPlan<T>::execute(std::complex<T>* data) const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid.");
    if (pimpl_->domain != Domain::COMPLEX) {
        throw std::runtime_error("Execute (C2C in-place) called on FFTPlan created for REAL domain.");
    }
    pimpl_->execute_c2c_ip(data);
}

// R2C / C2R Execute Methods
template <typename T>
void FFTPlan<T>::execute_r2c(const T* real_input, std::complex<T>* complex_output) const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid.");
     if (pimpl_->domain != Domain::REAL || pimpl_->direction != Direction::FORWARD) {
        throw std::runtime_error("execute_r2c called on FFTPlan with incorrect domain or direction.");
     }
     pimpl_->execute_r2c_oop(real_input, complex_output);
}

template <typename T>
void FFTPlan<T>::execute_c2r(const std::complex<T>* complex_input, T* real_output) const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid.");
     if (pimpl_->domain != Domain::REAL || pimpl_->direction != Direction::INVERSE) {
        throw std::runtime_error("execute_c2r called on FFTPlan with incorrect domain or direction.");
     }
     pimpl_->execute_c2r_oop(complex_input, real_output);
}

// Getters
template <typename T> size_t FFTPlan<T>::getLength() const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid."); return pimpl_->length;
}
template <typename T> size_t FFTPlan<T>::getComplexLength() const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid."); return pimpl_->complex_length;
}
template <typename T> Direction FFTPlan<T>::getDirection() const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid."); return pimpl_->direction;
}
template <typename T> Precision FFTPlan<T>::getPrecision() const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid."); return pimpl_->precision;
}
template <typename T> Domain FFTPlan<T>::getDomain() const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid."); return pimpl_->domain;
}


// --- Explicit Instantiations ---
template struct FFTPlanImpl<float>;
template struct FFTPlanImpl<double>;
template class FFTPlan<float>;
template class FFTPlan<double>;

} // namespace OmniFFT

#endif // USE_ONEMKL