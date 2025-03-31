/**
 * @file fft_impl_accelerate.cpp
 * @brief Apple Accelerate backend implementation for OmniFFT.
 *
 * This file contains the concrete implementation of the FFTPlanImpl class
 * using Apple's Accelerate framework (specifically vDSP functions). It is
 * compiled only when USE_ACCELERATE is defined by the CMake build system.
 *
 * Note: The current implementation for Domain::REAL transforms using Accelerate
 * requires the FFT length 'N' to be a power of 2 due to the underlying
 * vDSP_create_fftsetup function. Manual scaling is applied based on NormMode.
 *
 * @version 1.0.0
 * @date 2025-03-31
 */

#include <OmniFFT/omnifft.h> // Public API header

#if defined(USE_ACCELERATE) // Only compile contents if Accelerate backend is selected

#include <Accelerate/Accelerate.h> // Main Accelerate header (includes vDSP)
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <complex>    // For std::complex
#include <memory>     // For std::unique_ptr, std::make_unique
#include <vector>     // For internal buffers
#include <algorithm>  // For std::copy (used in simulated in-place C2C)
#include <type_traits>// For std::is_same_v
#include <cmath>      // For std::sqrt, std::log2, std::ceil

namespace OmniFFT {

/**
 * @internal
 * @brief Accelerate backend implementation details for FFTPlan.
 *
 * This struct holds the Accelerate-specific setup handles and buffers,
 * and implements the FFT execution using vDSP functions. It handles the
 * differences between complex (using vDSP_DFT) and real (using vDSP_fft) APIs,
 * including necessary data packing/unpacking and manual scaling for real transforms
 * and normalization modes.
 *
 * @tparam T float or double
 */
template <typename T>
struct FFTPlanImpl {
    // --- Configuration Members ---
    size_t length = 0;        ///< N: Number of points in the real domain.
    size_t complex_length = 0;///< Nc = N/2 + 1: Number of complex points for REAL domain.
    Direction direction = Direction::FORWARD; ///< Transform direction.
    Precision precision = Precision::SINGLE; ///< Data precision.
    Domain domain = Domain::COMPLEX;       ///< Transform domain (COMPLEX or REAL).
    NormMode norm_mode = NormMode::BACKWARD; ///< Normalization mode.

    // --- Calculated Scaling Factors ---
    // These are applied manually after vDSP calls
    T forward_scale = 1.0;
    T backward_scale = 1.0;

    // --- Accelerate Handles ---
    // Use void* for C2C handle to easily store float/double versions
    void* setup_handle_c2c = nullptr;     ///< vDSP_DFT_Setup or vDSP_DFT_SetupD for C2C
    // Use specific type for REAL handle (older API)
    typename std::conditional<std::is_same_v<T, float>, FFTSetup, FFTSetupD>::type
        setup_handle_real = nullptr;      ///< FFTSetup or FFTSetupD for R2C/C2R

    // --- Buffers for R2C/C2R (Split Complex Format) ---
    std::vector<T> real_buffer; ///< Temporary buffer for real parts of split complex data. Size N/2.
    std::vector<T> imag_buffer; ///< Temporary buffer for imag parts of split complex data. Size N/2.
    vDSP_Stride stride = 1;     ///< Stride for accessing split complex buffers (usually 1).

    /**
     * @internal @brief Helper to calculate log2(N) required by vDSP_create_fftsetup.
     * @param n The length N.
     * @return vDSP_Length Base-2 logarithm of n.
     * @throws std::runtime_error if n is not a power of 2 (and n != 1).
     * @throws std::invalid_argument if n is zero.
     */
    vDSP_Length get_log2n(size_t n) {
        if (n == 0) throw std::invalid_argument("FFT length cannot be zero.");
        // vDSP_create_fftsetup requires N to be a power of 2. Allow N=1.
        bool is_power_of_two = (n > 0) && ((n & (n - 1)) == 0);
        if (!is_power_of_two && n != 1) {
             throw std::runtime_error("Accelerate FFTSetup (for REAL domain) currently requires length N to be a power of 2.");
        }
        // Use std::log2, handle N=1 edge case (log2(1)=0)
        return (n == 1) ? 0 : static_cast<vDSP_Length>(std::log2(static_cast<double>(n)));
    }

    /**
     * @brief Constructs and configures the Accelerate setup handles.
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

        // Calculate required manual scaling factors based on NormMode and Domain
        T scaleN = static_cast<T>(length);
        T scaleSqrtN = std::sqrt(scaleN);
        if (scaleN == static_cast<T>(0.0)) { throw std::runtime_error("Cannot calculate scaling factors for N=0"); }
        if (scaleSqrtN == static_cast<T>(0.0) && norm_mode == NormMode::ORTHO) { throw std::runtime_error("Cannot use ORTHO norm mode for N=0"); }


        if (domain == Domain::COMPLEX) {
            complex_length = length; // N complex points for C2C
            // C2C with vDSP_DFT: Assume raw IFFT(FFT(x)) = x (no inherent scaling).
            // Therefore, apply the desired normalization scales directly.
            switch (norm_mode) {
                case NormMode::BACKWARD: forward_scale = 1.0; backward_scale = 1.0 / scaleN; break;
                case NormMode::ORTHO:    forward_scale = 1.0 / scaleSqrtN; backward_scale = 1.0 / scaleSqrtN; break;
                case NormMode::FORWARD:  forward_scale = 1.0 / scaleN; backward_scale = 1.0; break;
            }
            // Create C2C setup using vDSP_DFT (newer API)
            vDSP_DFT_Direction vdsp_dir = (direction == Direction::FORWARD) ? vDSP_DFT_FORWARD : vDSP_DFT_INVERSE;
            if constexpr (std::is_same_v<T, float>) setup_handle_c2c = vDSP_DFT_zop_CreateSetup(nullptr, length, vdsp_dir);
            else setup_handle_c2c = vDSP_DFT_zrop_CreateSetupD(nullptr, length, vdsp_dir);
            if (!setup_handle_c2c) throw std::runtime_error("Failed to create Accelerate vDSP_DFT (C2C) setup.");

        } else { // Domain::REAL
            complex_length = length / 2 + 1;
            // R2C/C2R with vDSP_fft_zrip: Assume raw IFFT(FFT(x)) = x * (N/2).
            // We derived the needed scaling relationship: scale_f * scale_b = 2.0 / N
            T factor = (scaleN == 0) ? static_cast<T>(1.0) : static_cast<T>(2.0) / scaleN; // Handle N=0? Length check prevents this.
            T factorSqrt = std::sqrt(factor);
            if (factorSqrt == static_cast<T>(0.0) && norm_mode == NormMode::ORTHO) { throw std::runtime_error("Cannot use ORTHO norm mode for N=0"); }


            switch (norm_mode) {
                case NormMode::BACKWARD: forward_scale = 1.0; backward_scale = factor; break; // scale_f=1 -> scale_b=2/N
                case NormMode::ORTHO:    forward_scale = factorSqrt; backward_scale = factorSqrt; break; // scale_f=sqrt(2/N) -> scale_b=sqrt(2/N)
                case NormMode::FORWARD:  forward_scale = factor; backward_scale = 1.0; break; // scale_b=1 -> scale_f=2/N
            }
            // Create R2C/C2R setup using vDSP_fft (older API, requires power-of-2 N)
            vDSP_Length log2n = get_log2n(length); // Throws if N not power-of-2 (or 1)
            if constexpr (std::is_same_v<T, float>) setup_handle_real = vDSP_create_fftsetup(log2n, kFFTRadix2);
            else setup_handle_real = vDSP_create_fftsetupD(log2n, kFFTRadix2);
            if (!setup_handle_real) throw std::runtime_error("Failed to create Accelerate FFTSetup (REAL) setup. Is length a power of 2?");
            // Allocate buffers for split complex format used by vDSP_fft_zrip
            real_buffer.resize(length / 2); // Size N/2
            imag_buffer.resize(length / 2); // Size N/2
        }
    }

    /**
     * @brief Destroys the Accelerate setup handles.
     */
    ~FFTPlanImpl() {
        // Destroy C2C handle if created
        if (setup_handle_c2c) {
             if constexpr (std::is_same_v<T, float>) vDSP_DFT_DestroySetup(reinterpret_cast<vDSP_DFT_Setup>(setup_handle_c2c));
             else vDSP_DFT_DestroySetupD(reinterpret_cast<vDSP_DFT_SetupD>(setup_handle_c2c));
        }
        // Destroy REAL handle if created
        if (setup_handle_real) {
             if constexpr (std::is_same_v<T, float>) vDSP_destroy_fftsetup(setup_handle_real);
             else vDSP_destroy_fftsetupD(setup_handle_real);
        }
    }

    // --- Packing/Unpacking Helpers for Accelerate Real FFT (Split Complex) ---

    /**
     * @internal @brief Packs std::complex[Nc] (CCE) into Accelerate DSPSplitComplex[N/2].
     * @param cce_input Input array [size Nc = N/2 + 1] with Hermitian symmetry.
     * @param split_output Output split complex structure with pointers to buffers of size N/2.
     */
    void pack_complex_to_split(const std::complex<T>* cce_input, void* split_output_void) const {
        const size_t N = length;
        const size_t Nc = complex_length; // N/2 + 1
        if (Nc == 0 || N == 0) return;

        if constexpr (std::is_same_v<T, float>) {
            DSPSplitComplex* split = reinterpret_cast<DSPSplitComplex*>(split_output_void);
            split->realp[0] = cce_input[0].real();                          // DC component
            split->imagp[0] = (N % 2 == 0 && Nc > 1) ? cce_input[Nc - 1].real() : 0.0f; // Nyquist component (if N is even and exists)
            // Components from 1 up to N/2 - 1
            for (size_t k = 1; k < N / 2; ++k) {
                split->realp[k] = cce_input[k].real();
                split->imagp[k] = cce_input[k].imag();
            }
        } else { // double
            DSPDoubleSplitComplex* split = reinterpret_cast<DSPDoubleSplitComplex*>(split_output_void);
            split->realp[0] = cce_input[0].real();
            split->imagp[0] = (N % 2 == 0 && Nc > 1) ? cce_input[Nc - 1].real() : 0.0;
            for (size_t k = 1; k < N / 2; ++k) {
                split->realp[k] = cce_input[k].real();
                split->imagp[k] = cce_input[k].imag();
            }
        }
    }

    /**
     * @internal @brief Unpacks Accelerate DSPSplitComplex[N/2] into std::complex[Nc] (CCE).
     * @param split_input_void Input split complex structure with pointers to buffers of size N/2.
     * @param cce_output Output array [size Nc = N/2 + 1].
     */
    void unpack_split_to_complex(const void* split_input_void, std::complex<T>* cce_output) const {
        const size_t N = length;
        const size_t Nc = complex_length; // N/2 + 1
        if (Nc == 0 || N == 0) return;

        if constexpr (std::is_same_v<T, float>) {
            const DSPSplitComplex* split = reinterpret_cast<const DSPSplitComplex*>(split_input_void);
            cce_output[0] = std::complex<T>(split->realp[0], 0.0f); // DC component
            // Components from 1 up to N/2 - 1
             for (size_t k = 1; k < N / 2; ++k) {
                cce_output[k] = std::complex<T>(split->realp[k], split->imagp[k]);
            }
            // Nyquist component for even N (stored in imagp[0])
            if (N % 2 == 0 && Nc > 1) {
                 cce_output[N / 2] = std::complex<T>(split->imagp[0], 0.0f);
            }
        } else { // double
            const DSPDoubleSplitComplex* split = reinterpret_cast<const DSPDoubleSplitComplex*>(split_input_void);
            cce_output[0] = std::complex<T>(split->realp[0], 0.0);
             for (size_t k = 1; k < N / 2; ++k) {
                cce_output[k] = std::complex<T>(split->realp[k], split->imagp[k]);
            }
             if (N % 2 == 0 && Nc > 1) {
                 cce_output[N / 2] = std::complex<T>(split->imagp[0], 0.0);
            }
        }
    }


    // --- Execute Methods ---

    /** @brief Implements C2C out-of-place execution with manual scaling. */
    void execute_c2c_oop(const std::complex<T>* input, std::complex<T>* output) const {
       if (!setup_handle_c2c) throw std::runtime_error("Invalid FFTPlan state for C2C (Accelerate).");
        // Determine scale factor based on direction and norm mode
        T scale = (direction == Direction::FORWARD) ? forward_scale : backward_scale;

        // Perform raw FFT/IFFT using vDSP_DFT
        if constexpr (std::is_same_v<T, float>) {
            // vDSP_DFT needs separate real/imag pointers for zop
            vDSP_DFT_Execute(reinterpret_cast<vDSP_DFT_Setup>(setup_handle_c2c),
                             reinterpret_cast<const float*>(input),       // Input real start
                             reinterpret_cast<const float*>(input) + 1,   // Input imag start
                             reinterpret_cast<float*>(output),      // Output real start
                             reinterpret_cast<float*>(output) + 1); // Output imag start
            // Apply manual scaling if not 1.0
            if (scale != 1.0f) {
                // Scale interleaved complex output (real and imag parts)
                vDSP_vsmul(reinterpret_cast<const float*>(output), 1, &scale, reinterpret_cast<float*>(output), 1, length * 2);
            }
        } else { // double
            vDSP_DFT_ExecuteD(reinterpret_cast<vDSP_DFT_SetupD>(setup_handle_c2c),
                              reinterpret_cast<const double*>(input),
                              reinterpret_cast<const double*>(input) + 1,
                              reinterpret_cast<double*>(output),
                              reinterpret_cast<double*>(output) + 1);
            if (scale != 1.0) {
                vDSP_vsmulD(reinterpret_cast<const double*>(output), 1, &scale, reinterpret_cast<double*>(output), 1, length * 2);
            }
        }
    }

    /** @brief Implements C2C in-place execution (simulated). */
    void execute_c2c_ip(std::complex<T>* data) const {
        if (!setup_handle_c2c) throw std::runtime_error("Invalid FFTPlan state for C2C (Accelerate).");
        // vDSP_DFT_zop requires separate output, so simulate in-place
        std::vector<std::complex<T>> temp_output(length);
        execute_c2c_oop(data, temp_output.data()); // Execute OOP, applies scaling
        std::copy(temp_output.begin(), temp_output.end(), data); // Copy scaled result back
    }

    /** @brief Implements R2C forward out-of-place execution with manual scaling. */
    void execute_rfft_oop(const T* real_input, std::complex<T>* complex_output) const {
         if (!setup_handle_real) throw std::runtime_error("Invalid FFTPlan state for REAL domain (Accelerate).");
         const size_t N = length;
         if (N == 0) return;
         vDSP_Length log2n = get_log2n(N); // Already checked for power-of-2
         T scale = forward_scale;          // Scaling factor for this transform direction/norm

         // Use internal buffers for split complex format (mutable)
         typename std::conditional<std::is_same_v<T, float>, DSPSplitComplex, DSPDoubleSplitComplex>::type split_data;
         split_data.realp = real_buffer.data();
         split_data.imagp = imag_buffer.data();

         // 1. Convert real input to split complex format (vDSP_ctoz packs real input)
         if constexpr (std::is_same_v<T, float>) vDSP_ctoz(reinterpret_cast<const DSPComplex*>(real_input), 2, &split_data, 1, N / 2);
         else vDSP_ctozD(reinterpret_cast<const DSPDoubleComplex*>(real_input), 2, &split_data, 1, N / 2);

         // 2. Perform in-place FFT on the packed split complex data using vDSP_fft_zrip
         if constexpr (std::is_same_v<T, float>) vDSP_fft_zrip(setup_handle_real, &split_data, stride, log2n, kFFTDirection_Forward);
         else vDSP_fft_zripD(setup_handle_real, &split_data, stride, log2n, kFFTDirection_Forward);

         // 3. Unpack the resulting split complex data into standard CCE complex output format
         unpack_split_to_complex(&split_data, complex_output);

         // 4. Apply the calculated manual scaling factor to the complex output
         if (scale != static_cast<T>(1.0)) {
             size_t complex_elements_to_scale = complex_length * 2; // real+imag parts
             if constexpr (std::is_same_v<T, float>) vDSP_vsmul(reinterpret_cast<float*>(complex_output), 1, &scale, reinterpret_cast<float*>(complex_output), 1, complex_elements_to_scale);
             else vDSP_vsmulD(reinterpret_cast<double*>(complex_output), 1, &scale, reinterpret_cast<double*>(complex_output), 1, complex_elements_to_scale);
         }
    }

    /** @brief Implements C2R inverse out-of-place execution with manual scaling. */
    void execute_irfft_oop(const std::complex<T>* complex_input, T* real_output) const {
        if (!setup_handle_real) throw std::runtime_error("Invalid FFTPlan state for REAL domain (Accelerate).");
        const size_t N = length;
        if (N == 0) return;
        vDSP_Length log2n = get_log2n(N);
        T scale = backward_scale; // Scaling factor for this transform direction/norm

        // Use internal buffers for split complex format (mutable)
        typename std::conditional<std::is_same_v<T, float>, DSPSplitComplex, DSPDoubleSplitComplex>::type split_data;
        split_data.realp = real_buffer.data();
        split_data.imagp = imag_buffer.data();

        // 1. Pack standard CCE complex input (Hermitian assumed) into split complex format
        pack_complex_to_split(complex_input, &split_data);

        // 2. Perform in-place Inverse FFT on the packed split complex data using vDSP_fft_zrip
         if constexpr (std::is_same_v<T, float>) vDSP_fft_zrip(setup_handle_real, &split_data, stride, log2n, kFFTDirection_Inverse);
         else vDSP_fft_zripD(setup_handle_real, &split_data, stride, log2n, kFFTDirection_Inverse);

         // 3. Convert the resulting split complex data back to real output using vDSP_ztoc
         if constexpr (std::is_same_v<T, float>) vDSP_ztoc(&split_data, 1, reinterpret_cast<DSPComplex*>(real_output), 2, N / 2);
         else vDSP_ztocD(&split_data, 1, reinterpret_cast<DSPDoubleComplex*>(real_output), 2, N / 2);

         // 4. Apply the calculated manual scaling factor to the final real output
         if (scale != static_cast<T>(1.0)) {
              if constexpr (std::is_same_v<T, float>) vDSP_vsmul(real_output, 1, &scale, real_output, 1, N);
              else vDSP_vsmulD(real_output, 1, &scale, real_output, 1, N);
         }
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
         setup_handle_c2c(other.setup_handle_c2c),
         setup_handle_real(other.setup_handle_real),
         real_buffer(std::move(other.real_buffer)), // Move buffers
         imag_buffer(std::move(other.imag_buffer)),
         stride(other.stride)
     {
        // Ensure moved-from object doesn't own handles
        other.setup_handle_c2c = nullptr;
        other.setup_handle_real = nullptr;
     }

     /** @brief Move assignment operator. */
     FFTPlanImpl& operator=(FFTPlanImpl&& other) noexcept {
         if (this != &other) {
            // Release existing resources
            // (Call destructors for handles)
             this->~FFTPlanImpl(); // Call own destructor carefully, or manually destroy handles
            // if (setup_handle_c2c) { /* destroy c2c */ }
            // if (setup_handle_real) { /* destroy real */ }

            // Transfer ownership from other
            length = other.length;
            complex_length = other.complex_length;
            direction = other.direction;
            precision = other.precision;
            domain = other.domain;
            norm_mode = other.norm_mode;
            forward_scale = other.forward_scale;
            backward_scale = other.backward_scale;
            setup_handle_c2c = other.setup_handle_c2c;
            setup_handle_real = other.setup_handle_real;
            real_buffer = std::move(other.real_buffer); // Move buffers
            imag_buffer = std::move(other.imag_buffer);
            stride = other.stride;

            // Nullify source handles
            other.setup_handle_c2c = nullptr;
            other.setup_handle_real = nullptr;
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
// Defined here where FFTPlanImpl is complete.

/** @brief FFTPlan constructor definition (forwards to Impl). */
template <typename T>
FFTPlan<T>::FFTPlan(size_t length, Precision precision, Direction direction, Domain domain, NormMode norm)
    : pimpl_(std::make_unique<FFTPlanImpl<T>>(length, precision, direction, domain, norm)) {}

/** @brief FFTPlan destructor definition. */
template <typename T> FFTPlan<T>::~FFTPlan() = default;
/** @brief FFTPlan move constructor definition. */
template <typename T> FFTPlan<T>::FFTPlan(FFTPlan&& other) noexcept = default;
/** @brief FFTPlan move assignment definition. */
template <typename T> FFTPlan<T>& FFTPlan<T>::operator=(FFTPlan<T>&& other) noexcept = default;

/** @brief FFTPlan C2C execute OOP definition. */
template <typename T>
void FFTPlan<T>::execute(const std::complex<T>* input, std::complex<T>* output) const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (moved from or init failed).");
    if (pimpl_->domain != Domain::COMPLEX) throw std::runtime_error("Execute (C2C) called on FFTPlan for REAL domain.");
    pimpl_->execute_c2c_oop(input, output);
}

/** @brief FFTPlan C2C execute IP definition. */
template <typename T>
void FFTPlan<T>::execute(std::complex<T>* data) const {
    if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (moved from or init failed).");
    if (pimpl_->domain != Domain::COMPLEX) throw std::runtime_error("Execute (C2C in-place) called on FFTPlan for REAL domain.");
    pimpl_->execute_c2c_ip(data);
}

/** @brief FFTPlan RFFT execute definition. */
template <typename T>
void FFTPlan<T>::execute_rfft(const T* real_input, std::complex<T>* complex_output) const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (moved from or init failed).");
     if (pimpl_->domain != Domain::REAL || pimpl_->direction != Direction::FORWARD) throw std::runtime_error("execute_rfft requires Domain::REAL and Direction::FORWARD.");
     pimpl_->execute_rfft_oop(real_input, complex_output);
}

/** @brief FFTPlan IRFFT execute definition. */
template <typename T>
void FFTPlan<T>::execute_irfft(const std::complex<T>* complex_input, T* real_output) const {
     if (!pimpl_) throw std::runtime_error("FFTPlan is not valid (moved from or init failed).");
     if (pimpl_->domain != Domain::REAL || pimpl_->direction != Direction::INVERSE) throw std::runtime_error("execute_irfft requires Domain::REAL and Direction::INVERSE.");
     pimpl_->execute_irfft_oop(complex_input, real_output);
}

// Getters: Check Pimpl and return value from Impl
/** @brief FFTPlan getLength definition. */
template <typename T> size_t FFTPlan<T>::getLength() const { if (!pimpl_) throw std::runtime_error("FFTPlan is not valid."); return pimpl_->length; }
/** @brief FFTPlan getComplexLength definition. */
template <typename T> size_t FFTPlan<T>::getComplexLength() const { if (!pimpl_) throw std::runtime_error("FFTPlan is not valid."); return pimpl_->complex_length; }
/** @brief FFTPlan getDirection definition. */
template <typename T> Direction FFTPlan<T>::getDirection() const { if (!pimpl_) throw std::runtime_error("FFTPlan is not valid."); return pimpl_->direction; }
/** @brief FFTPlan getPrecision definition. */
template <typename T> Precision FFTPlan<T>::getPrecision() const { if (!pimpl_) throw std::runtime_error("FFTPlan is not valid."); return pimpl_->precision; }
/** @brief FFTPlan getDomain definition. */
template <typename T> Domain FFTPlan<T>::getDomain() const { if (!pimpl_) throw std::runtime_error("FFTPlan is not valid."); return pimpl_->domain; }
/** @brief FFTPlan getNormMode definition. */
template <typename T> NormMode FFTPlan<T>::getNormMode() const { if (!pimpl_) throw std::runtime_error("FFTPlan is not valid."); return pimpl_->norm_mode; }


// --- Explicit Instantiations ---
// Instantiate the implementation class template for float and double.
template struct FFTPlanImpl<float>;
template struct FFTPlanImpl<double>;

// Instantiate the public wrapper class template for float and double.
template class FFTPlan<float>;
template class FFTPlan<double>;

} // namespace OmniFFT

#endif // USE_ACCELERATE