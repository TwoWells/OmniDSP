// src/fft_impl_accelerate.cpp

#include "fft_lib.h" // Contains FFTPlanImpl forward decl, FFTPlan decl, enums

#if defined(USE_ACCELERATE) // Guard ensures this compiles only when Accelerate is selected

#include <Accelerate/Accelerate.h>
#include <stdexcept>
#include <complex>
#include <memory>      // For std::unique_ptr, std::make_unique
#include <vector>
#include <algorithm>   // For std::copy
#include <type_traits> // For std::is_same_v
#include <cmath>       // For std::log2, std::ceil

namespace CrossPlatformFFT {

// --- Accelerate FFTPlanImpl Definition ---
template <typename T>
struct FFTPlanImpl {
    size_t length = 0;        // N: Real domain length
    size_t complex_length = 0;// Nc = N/2 + 1
    Direction direction = Direction::FORWARD;
    Precision precision = Precision::SINGLE;
    Domain domain = Domain::COMPLEX;

    // Handles for different domains (DFT for C2C, FFT for R2C/C2R)
    void* setup_handle_c2c = nullptr; // vDSP_DFT_Setup or vDSP_DFT_SetupD
    FFTSetup setup_handle_real = nullptr; // FFTSetup or FFTSetupD (older API for real FFT)

    // Buffers needed for Accelerate real FFT packing/unpacking
    std::vector<T> real_buffer; // Holds split complex real part
    std::vector<T> imag_buffer; // Holds split complex imag part
    vDSP_Stride stride = 1;     // Usually 1 for packed data

    // Helper to get log2(N) needed for vDSP_create_fftsetup
    vDSP_Length get_log2n(size_t n) {
        if (n == 0) throw std::invalid_argument("FFT length cannot be zero.");
        // Check if n is a power of 2 (required by vDSP_create_fftsetup)
        if ((n & (n - 1)) != 0 && n != 1) { // Allow N=1
             // If not power of 2, vDSP_create_fftsetup will return NULL.
             // We could pad here, but let's throw for now.
             throw std::runtime_error("Accelerate FFTSetup currently requires power-of-2 lengths for REAL domain.");
        }
        // Calculate log2(n). Use std::log2 and handle n=1 case.
        return (n == 1) ? 0 : static_cast<vDSP_Length>(std::log2(static_cast<double>(n)));
    }

    // Constructor
    FFTPlanImpl(size_t len, Precision prec, Direction dir, Domain dom) :
        length(len), direction(dir), precision(prec), domain(dom)
    {
         if (length == 0) {
            throw std::invalid_argument("FFT length cannot be zero.");
        }

        if (domain == Domain::COMPLEX) {
            complex_length = 0; // Not applicable
            vDSP_DFT_Direction vdsp_dir = (direction == Direction::FORWARD) ?
                                           vDSP_DFT_FORWARD : vDSP_DFT_INVERSE;
            if constexpr (std::is_same_v<T, float>) {
                setup_handle_c2c = vDSP_DFT_zop_CreateSetup(nullptr, length, vdsp_dir);
            } else {
                setup_handle_c2c = vDSP_DFT_zrop_CreateSetupD(nullptr, length, vdsp_dir);
            }
            if (!setup_handle_c2c) {
                throw std::runtime_error("Failed to create Accelerate vDSP_DFT (C2C) setup.");
            }
        } else { // Domain::REAL
            complex_length = length / 2 + 1;
            vDSP_Length log2n = get_log2n(length); // Throws if not power of 2

            if constexpr (std::is_same_v<T, float>) {
                setup_handle_real = vDSP_create_fftsetup(log2n, kFFTRadix2); // Using older FFT API
            } else {
                setup_handle_real = vDSP_create_fftsetupD(log2n, kFFTRadix2);
            }
            if (!setup_handle_real) {
                 throw std::runtime_error("Failed to create Accelerate FFTSetup (REAL) setup. Is length a power of 2?");
            }
            // Allocate split complex buffers
            real_buffer.resize(length / 2); // Size N/2
            imag_buffer.resize(length / 2); // Size N/2
        }
    }

    // Destructor
    ~FFTPlanImpl() {
        if (setup_handle_c2c) {
             if constexpr (std::is_same_v<T, float>) vDSP_DFT_DestroySetup(reinterpret_cast<vDSP_DFT_Setup>(setup_handle_c2c));
             else vDSP_DFT_DestroySetupD(reinterpret_cast<vDSP_DFT_SetupD>(setup_handle_c2c));
        }
        if (setup_handle_real) {
             if constexpr (std::is_same_v<T, float>) vDSP_destroy_fftsetup(reinterpret_cast<FFTSetup>(setup_handle_real));
             else vDSP_destroy_fftsetupD(reinterpret_cast<FFTSetupD>(setup_handle_real));
        }
    }

    // --- Packing/Unpacking Helpers for Accelerate Real FFT ---

    // Pack std::complex[Nc] CCE format into Accelerate DSPSplitComplex[N/2] format
    void pack_complex_to_split(const std::complex<T>* cce_input, DSPSplitComplex* split_output) const {
        const size_t N = length;
        const size_t Nc = complex_length; // N/2 + 1
        if (Nc == 0) return; // Should not happen if domain is REAL

        if constexpr (std::is_same_v<T, float>) {
            DSPSplitComplex* split = reinterpret_cast<DSPSplitComplex*>(split_output);
            split->realp[0] = cce_input[0].real(); // DC component
            split->imagp[0] = (N % 2 == 0) ? cce_input[Nc - 1].real() : 0.0f; // Nyquist component (if N is even)

            for (size_t k = 1; k < Nc - (N % 2 == 0 ? 1 : 0); ++k) { // Up to N/2 (exclusive if N even)
                split->realp[k] = cce_input[k].real();
                split->imagp[k] = cce_input[k].imag();
            }
        } else { // double
            DSPDoubleSplitComplex* split = reinterpret_cast<DSPDoubleSplitComplex*>(split_output);
            split->realp[0] = cce_input[0].real(); // DC component
            split->imagp[0] = (N % 2 == 0) ? cce_input[Nc - 1].real() : 0.0; // Nyquist component (if N is even)

             for (size_t k = 1; k < Nc - (N % 2 == 0 ? 1 : 0); ++k) {
                split->realp[k] = cce_input[k].real();
                split->imagp[k] = cce_input[k].imag();
            }
        }
    }

    // Unpack Accelerate DSPSplitComplex[N/2] format into std::complex[Nc] CCE format
    void unpack_split_to_complex(const DSPSplitComplex* split_input, std::complex<T>* cce_output) const {
        const size_t N = length;
        const size_t Nc = complex_length; // N/2 + 1
        if (Nc == 0) return;

        if constexpr (std::is_same_v<T, float>) {
            const DSPSplitComplex* split = reinterpret_cast<const DSPSplitComplex*>(split_input);
            cce_output[0] = std::complex<T>(split->realp[0], 0.0f); // DC component

             for (size_t k = 1; k < N / 2; ++k) { // Components 1 to N/2-1
                cce_output[k] = std::complex<T>(split->realp[k], split->imagp[k]);
            }
            if (N % 2 == 0) { // Nyquist component for even N
                 cce_output[N / 2] = std::complex<T>(split->imagp[0], 0.0f);
            }
        } else { // double
            const DSPDoubleSplitComplex* split = reinterpret_cast<const DSPDoubleSplitComplex*>(split_input);
            cce_output[0] = std::complex<T>(split->realp[0], 0.0); // DC component

             for (size_t k = 1; k < N / 2; ++k) {
                cce_output[k] = std::complex<T>(split->realp[k], split->imagp[k]);
            }
             if (N % 2 == 0) { // Nyquist component for even N
                 cce_output[N / 2] = std::complex<T>(split->imagp[0], 0.0);
            }
        }
         // Note: This assumes standard CCE format where complex_output[0] is DC,
         // complex_output[N/2] is Nyquist (if N even), and others are complex pairs.
    }


    // --- Execute Methods ---

    // Execute C2C Out-of-Place
    void execute_c2c_oop(const std::complex<T>* input, std::complex<T>* output) const {
       if (!setup_handle_c2c) throw std::runtime_error("Invalid FFTPlan state for C2C.");
        if constexpr (std::is_same_v<T, float>) {
            const float* realp = reinterpret_cast<const float*>(input);
            const float* imagp = realp + 1;
            float* out_realp = reinterpret_cast<float*>(output);
            float* out_imagp = out_realp + 1;
            vDSP_DFT_Execute(reinterpret_cast<vDSP_DFT_Setup>(setup_handle_c2c),
                             realp, imagp, out_realp, out_imagp);
        } else { // double
            const double* realp = reinterpret_cast<const double*>(input);
            const double* imagp = realp + 1;
            double* out_realp = reinterpret_cast<double*>(output);
            double* out_imagp = out_realp + 1;
            vDSP_DFT_ExecuteD(reinterpret_cast<vDSP_DFT_SetupD>(setup_handle_c2c),
                              realp, imagp, out_realp, out_imagp);
        }
    }

    // Execute C2C In-Place (Simulated)
    void execute_c2c_ip(std::complex<T>* data) const {
        if (!setup_handle_c2c) throw std::runtime_error("Invalid FFTPlan state for C2C.");
        std::vector<std::complex<T>> temp_output(length);
        execute_c2c_oop(data, temp_output.data());
        std::copy(temp_output.begin(), temp_output.end(), data);
    }

    // Execute R2C Forward Out-of-Place
    void execute_r2c_oop(const T* real_input, std::complex<T>* complex_output) const {
         if (!setup_handle_real) throw std::runtime_error("Invalid FFTPlan state for REAL domain.");
         const size_t N = length;
         if (N == 0) return;
         vDSP_Length log2n = get_log2n(N);

         // Need mutable buffers for split complex data
         DSPSplitComplex split_data;
         split_data.realp = real_buffer.data();
         split_data.imagp = imag_buffer.data();

         // 1. Convert real input to split complex format (packed)
         if constexpr (std::is_same_v<T, float>) {
              vDSP_ctoz(reinterpret_cast<const DSPComplex*>(real_input), 2, &split_data, 1, N / 2);
         } else { // double
              vDSP_ctozD(reinterpret_cast<const DSPDoubleComplex*>(real_input), 2, reinterpret_cast<DSPDoubleSplitComplex*>(&split_data), 1, N / 2);
         }

         // 2. Perform in-place FFT on the split complex data
         if constexpr (std::is_same_v<T, float>) {
              vDSP_fft_zrip(reinterpret_cast<FFTSetup>(setup_handle_real), &split_data, stride, log2n, kFFTDirection_Forward);
         } else { // double
              vDSP_fft_zripD(reinterpret_cast<FFTSetupD>(setup_handle_real), reinterpret_cast<DSPDoubleSplitComplex*>(&split_data), stride, log2n, kFFTDirection_Forward);
         }

         // 3. Unpack the split complex result into standard complex output format
         unpack_split_to_complex(&split_data, complex_output);

         // Apply scaling: vDSP_fft_zrip forward results need scaling by 1/2 ? Check docs.
         // vDSP documentation often implies results are scaled by 1/2 for forward.
         // Let's apply 0.5 scaling factor
         if constexpr (std::is_same_v<T, float>) {
             float scale = 0.5f;
             vDSP_vsmul(reinterpret_cast<float*>(complex_output), 1, &scale, reinterpret_cast<float*>(complex_output), 1, complex_length * 2); // Multiply real and imag parts
         } else {
             double scale = 0.5;
             vDSP_vsmulD(reinterpret_cast<double*>(complex_output), 1, &scale, reinterpret_cast<double*>(complex_output), 1, complex_length * 2);
         }

    }

    // Execute C2R Inverse Out-of-Place
    void execute_c2r_oop(const std::complex<T>* complex_input, T* real_output) const {
        if (!setup_handle_real) throw std::runtime_error("Invalid FFTPlan state for REAL domain.");
        const size_t N = length;
        if (N == 0) return;
        vDSP_Length log2n = get_log2n(N);

        // Need mutable buffers for split complex data
        DSPSplitComplex split_data;
        split_data.realp = real_buffer.data();
        split_data.imagp = imag_buffer.data();

        // 1. Pack standard complex input format into split complex format
        pack_complex_to_split(complex_input, &split_data);

        // Apply scaling BEFORE inverse FFT: vDSP_fft_zrip inverse expects input scaled by 1/2 ? Check docs.
        // Yes, docs suggest scaling input by 0.5 for inverse.
        if constexpr (std::is_same_v<T, float>) {
             float scale = 0.5f;
             vDSP_vsmul(split_data.realp, 1, &scale, split_data.realp, 1, N / 2);
             vDSP_vsmul(split_data.imagp, 1, &scale, split_data.imagp, 1, N / 2);
        } else {
             double scale = 0.5;
             vDSP_vsmulD(split_data.realp, 1, &scale, split_data.realp, 1, N / 2);
             vDSP_vsmulD(split_data.imagp, 1, &scale, split_data.imagp, 1, N / 2);
        }

        // 2. Perform in-place Inverse FFT on the split complex data
         if constexpr (std::is_same_v<T, float>) {
              vDSP_fft_zrip(reinterpret_cast<FFTSetup>(setup_handle_real), &split_data, stride, log2n, kFFTDirection_Inverse);
         } else { // double
              vDSP_fft_zripD(reinterpret_cast<FFTSetupD>(setup_handle_real), reinterpret_cast<DSPDoubleSplitComplex*>(&split_data), stride, log2n, kFFTDirection_Inverse);
         }

         // 3. Convert the resulting split complex data back to real output
         if constexpr (std::is_same_v<T, float>) {
              vDSP_ztoc(&split_data, 1, reinterpret_cast<DSPComplex*>(real_output), 2, N / 2);
         } else { // double
              vDSP_ztocD(reinterpret_cast<DSPDoubleSplitComplex*>(&split_data), 1, reinterpret_cast<DSPDoubleComplex*>(real_output), 2, N / 2);
         }
         // Note: The combined scaling of 0.5 before inverse results in an overall 1/N scaling
         // because the FFT pair inherently involves a factor of N. Check vDSP docs carefully.
         // Unlike MKL, Accelerate often requires manual scaling management.
    }

     // --- Move Semantics ---
     FFTPlanImpl(FFTPlanImpl&& other) noexcept :
         length(other.length),
         complex_length(other.complex_length),
         direction(other.direction),
         precision(other.precision),
         domain(other.domain),
         setup_handle_c2c(other.setup_handle_c2c),
         setup_handle_real(other.setup_handle_real),
         real_buffer(std::move(other.real_buffer)),
         imag_buffer(std::move(other.imag_buffer)),
         stride(other.stride)
     {
        other.setup_handle_c2c = nullptr;
        other.setup_handle_real = nullptr;
     }

     FFTPlanImpl& operator=(FFTPlanImpl&& other) noexcept {
          if (this != &other) {
            // Release existing resources
            if (setup_handle_c2c) {
                 if constexpr (std::is_same_v<T, float>) vDSP_DFT_DestroySetup(reinterpret_cast<vDSP_DFT_Setup>(setup_handle_c2c));
                 else vDSP_DFT_DestroySetupD(reinterpret_cast<vDSP_DFT_SetupD>(setup_handle_c2c));
            }
             if (setup_handle_real) {
                 if constexpr (std::is_same_v<T, float>) vDSP_destroy_fftsetup(reinterpret_cast<FFTSetup>(setup_handle_real));
                 else vDSP_destroy_fftsetupD(reinterpret_cast<FFTSetupD>(setup_handle_real));
            }
            // Transfer ownership
            length = other.length;
            complex_length = other.complex_length;
            direction = other.direction;
            precision = other.precision;
            domain = other.domain;
            setup_handle_c2c = other.setup_handle_c2c;
            setup_handle_real = other.setup_handle_real;
            real_buffer = std::move(other.real_buffer);
            imag_buffer = std::move(other.imag_buffer);
            stride = other.stride;
            // Nullify source
            other.setup_handle_c2c = nullptr;
            other.setup_handle_real = nullptr;
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

} // namespace CrossPlatformFFT

#endif // USE_ACCELERATE