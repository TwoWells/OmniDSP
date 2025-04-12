/**
 * @file accelerate.cpp
 * @brief Apple Accelerate backend implementation for OmniDSP (FFT, FIR, Conv/Corr, Desamp).
 *
 * Implements FFTPlanImpl and backend functions using Apple's Accelerate framework (vDSP),
 * utilizing specialized FIR and decimation functions where available.
 * Compiled only when USE_ACCELERATE is defined.
 */

// --- Includes ---

// Standard C++ Headers
#include <vector>
#include <complex>
#include <memory>
#include <stdexcept>
#include <type_traits> // For std::is_same_v, std::conditional
#include <string>
#include <cmath>     // For std::sqrt, std::log2
#include <algorithm> // For std::copy, std::fill, std::min

// Public Project Headers (If needed by backend implementation details)
#include <OmniDSP/omnidsp.h> // For FFTPlan definition etc. potentially

// Internal Project Headers
#include "backend_impl.h" // Include the internal header LAST

// --- Conditionally Compile Backend Implementation ---

#if defined(USE_ACCELERATE) // Only compile this code if USE_ACCELERATE is defined (e.g., by CMake on macOS)

// Backend-specific includes GO HERE:
#include <Accelerate/Accelerate.h> // Main header for the Accelerate framework (includes vDSP)

namespace OmniDSP
{

    // --- FFTPlanImpl Definition (Structure) ---
    // This struct holds the backend-specific data and logic for an FFT plan using Accelerate.
    template <typename T> // Template parameter T for float or double precision
    struct FFTPlanImpl
    {
        // --- Members ---
        size_t length = 0;                        // FFT Length (N)
        size_t complex_length = 0;                // Length of complex spectrum (N for C2C, N/2+1 for Real)
        Direction direction = Direction::FORWARD; // Transform direction
        Precision precision = Precision::SINGLE;  // float or double
        Domain domain = Domain::COMPLEX;          // COMPLEX or REAL
        NormMode norm_mode = NormMode::BACKWARD;  // Normalization mode
        T forward_scale = 1.0;                    // Scaling factor applied to forward transform
        T backward_scale = 1.0;                   // Scaling factor applied to inverse transform

        // Accelerate-specific handles and buffers
        void *setup_handle_c2c = nullptr; // Opaque handle for vDSP_DFT (C2C FFTs, type varies by precision)
        // FFTSetup/FFTSetupD handle for vDSP_fft (Real FFTs)
        typename std::conditional<std::is_same_v<T, float>, FFTSetup, FFTSetupD>::type
            setup_handle_real = nullptr;
        std::vector<T> real_buffer; // Temporary buffer for real part in Real FFTs
        std::vector<T> imag_buffer; // Temporary buffer for imag part in Real FFTs
        vDSP_Stride stride = 1;     // Data stride (usually 1 for contiguous arrays)

        // --- Helper: Calculate log2(N) for FFTSetup ---
        // vDSP FFTSetup requires the length N as log2(N).
        // It also currently requires N to be a power of 2 for REAL domain FFTs.
        vDSP_Length get_log2n(size_t n)
        {
            if (n == 0)
                throw std::invalid_argument("FFT length cannot be zero.");
            // Check if n is a power of 2 (except for n=1)
            bool is_power_of_two = (n > 0) && ((n & (n - 1)) == 0);
            if (!is_power_of_two && n != 1)
            {
                // vDSP_DFT (C2C) handles non-power-of-2 lengths.
                if (domain == Domain::REAL)
                {
                    // vDSP_fft_zrip (Real FFT) requires power-of-2.
                    throw std::runtime_error("Accelerate FFTSetup (for REAL domain) currently requires power-of-2 length. Got N=" + std::to_string(n));
                }
                // For C2C, no log2n needed for vDSP_DFT setup.
            }
            // Only calculate and return log2n if needed (for REAL domain power-of-2 lengths)
            return (domain == Domain::REAL && is_power_of_two && n != 0) ? static_cast<vDSP_Length>(std::log2(static_cast<double>(n))) : 0;
        }

        // --- Constructor ---
        // Sets up the FFT plan based on user parameters.
        FFTPlanImpl(size_t len, Precision prec, Direction dir, Domain dom, NormMode norm)
        {
            if (len == 0)
            {
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
            if (scaleN == static_cast<T>(0.0)) // Defensive check
            {
                throw std::runtime_error("ScaleN is zero, length N must be > 0");
            }
            if (scaleSqrtN == static_cast<T>(0.0) && norm_mode == NormMode::ORTHO) // Check for N=0 if using ORTHO
            {
                throw std::runtime_error("ScaleSqrtN is zero for ORTHO, length N must be > 0");
            }

            // --- Complex Domain Setup ---
            if (dom == Domain::COMPLEX)
            {
                complex_length = length; // Input and output size N
                // Determine scaling factors based on mode
                switch (norm_mode)
                {
                case NormMode::BACKWARD: // Default: IFFT scaled by 1/N
                    forward_scale = 1.0;
                    backward_scale = 1.0 / scaleN;
                    break;
                case NormMode::ORTHO: // Unitary: FFT and IFFT scaled by 1/sqrt(N)
                    forward_scale = 1.0 / scaleSqrtN;
                    backward_scale = 1.0 / scaleSqrtN;
                    break;
                case NormMode::FORWARD: // FFT scaled by 1/N
                    forward_scale = 1.0 / scaleN;
                    backward_scale = 1.0;
                    break;
                }

                // Determine vDSP direction enum
                vDSP_DFT_Direction vdsp_dir = (dir == Direction::FORWARD) ? vDSP_DFT_FORWARD : vDSP_DFT_INVERSE;

                // Create the appropriate vDSP_DFT setup handle (out-of-place complex)
                if constexpr (std::is_same_v<T, float>)                                       // SINGLE precision
                    setup_handle_c2c = vDSP_DFT_zop_CreateSetup(nullptr, length, vdsp_dir);   // zop = complex out-of-place
                else                                                                          // DOUBLE precision
                    setup_handle_c2c = vDSP_DFT_zrop_CreateSetupD(nullptr, length, vdsp_dir); // zrop = complex real out-of-place (D for double)

                if (!setup_handle_c2c) // Check if setup creation failed
                    throw std::runtime_error("Failed to create Accelerate vDSP_DFT (C2C) setup. Check length/parameters.");
            }
            // --- Real Domain Setup ---
            else // Domain::REAL
            {
                complex_length = length / 2 + 1; // Real FFT output size
                // vDSP Real FFT (vDSP_fft_zrip) implicitly scales the IFFT by 2/N (or N/2 depending on view).
                // We need to calculate external scaling factors to match the desired NormMode.
                T factor = static_cast<T>(2.0) / scaleN;                                             // Base factor for vDSP scaling -> desired scale
                T factorSqrt = std::sqrt(factor);                                                    // Factor for ORTHO mode
                if (factorSqrt == static_cast<T>(0.0) && norm_mode == NormMode::ORTHO && scaleN > 0) // Defensive check for N=0 edge case with sqrt
                {
                    throw std::runtime_error("FactorSqrt is zero for ORTHO (REAL), length N must be > 0");
                }

                // Determine scaling factors based on mode
                switch (norm_mode)
                {
                case NormMode::BACKWARD: // RFFT unscaled, IRFFT scaled by 1/N (achieved by factor=2/N * N/2)
                    forward_scale = 1.0;
                    backward_scale = factor; // Apply the 2/N factor on inverse
                    break;
                case NormMode::ORTHO:            // RFFT & IRFFT scaled by 1/sqrt(N) (requires external sqrt(2/N) * sqrt(N/2))
                    forward_scale = factorSqrt;  // Apply sqrt(2/N) externally
                    backward_scale = factorSqrt; // Apply sqrt(2/N) externally
                    break;
                case NormMode::FORWARD:     // RFFT scaled by 1/N, IRFFT unscaled
                    forward_scale = factor; // Apply 2/N factor on forward
                    backward_scale = 1.0;
                    break;
                }

                // Get log2(N) and create FFTSetup handle (throws if N not power-of-2)
                vDSP_Length log2n = get_log2n(length);
                if constexpr (std::is_same_v<T, float>)                          // SINGLE precision
                    setup_handle_real = vDSP_create_fftsetup(log2n, kFFTRadix2); // Use Radix2 for standard FFT
                else                                                             // DOUBLE precision
                    setup_handle_real = vDSP_create_fftsetupD(log2n, kFFTRadix2);

                if (!setup_handle_real) // Check if setup creation failed
                    throw std::runtime_error("Failed to create Accelerate FFTSetup (REAL) setup. Is length a power of 2?");

                // Allocate temporary buffers needed for split complex format used by vDSP_fft_zrip
                // Size is N/2 because vDSP packs DC and Nyquist into the split format cleverly.
                real_buffer.resize(length / 2);
                imag_buffer.resize(length / 2);
            }
        }

        // --- Destructor ---
        // Releases the Accelerate setup handles.
        ~FFTPlanImpl()
        {
            if (setup_handle_c2c) // If C2C handle exists
            {
                if constexpr (std::is_same_v<T, float>)
                    vDSP_DFT_DestroySetup(reinterpret_cast<vDSP_DFT_Setup>(setup_handle_c2c));
                else
                    vDSP_DFT_DestroySetupD(reinterpret_cast<vDSP_DFT_SetupD>(setup_handle_c2c));
            }
            if (setup_handle_real) // If Real FFT handle exists
            {
                if constexpr (std::is_same_v<T, float>)
                    vDSP_destroy_fftsetup(setup_handle_real); // Destroy FFTSetup
                else
                    vDSP_destroy_fftsetupD(setup_handle_real); // Destroy FFTSetupD
            }
        }

        // --- Packing Helper: Complex Conjugate Even (CCE) to vDSP Split Complex ---
        // Converts the standard complex array format (size N/2+1, Hermitian symmetry assumed)
        // used by OmniDSP for Real FFT output into the 'split complex' format required by
        // vDSP_fft_zrip for inverse transforms.
        // Assumes `complex_input` has size `complex_length` (N/2+1).
        // `split_output_void` must point to a DSPSplitComplex (float) or DSPDoubleSplitComplex (double).
        void pack_complex_to_split(const std::complex<T> *complex_input, void *split_output_void) const
        {
            const size_t N = length;          // Original real length
            const size_t Nc = complex_length; // Complex length (N/2+1)
            if (Nc == 0 || N == 0)
                return; // Nothing to pack

            if constexpr (std::is_same_v<T, float>) // SINGLE precision
            {
                DSPSplitComplex *split = reinterpret_cast<DSPSplitComplex *>(split_output_void);
                // Pack DC component (index 0) into realp[0]
                split->realp[0] = complex_input[0].real();
                // Pack Nyquist component (index Nc-1 = N/2) into imagp[0] if N is even
                split->imagp[0] = (N % 2 == 0 && Nc > 1) ? complex_input[Nc - 1].real() : 0.0f;
                // Pack remaining components (1 to N/2 - 1)
                for (size_t k = 1; k < N / 2; ++k)
                {
                    split->realp[k] = complex_input[k].real();
                    split->imagp[k] = complex_input[k].imag();
                }
            }
            else // DOUBLE precision
            {
                DSPDoubleSplitComplex *split = reinterpret_cast<DSPDoubleSplitComplex *>(split_output_void);
                // Pack DC component
                split->realp[0] = complex_input[0].real();
                // Pack Nyquist component
                split->imagp[0] = (N % 2 == 0 && Nc > 1) ? complex_input[Nc - 1].real() : 0.0;
                // Pack remaining components
                for (size_t k = 1; k < N / 2; ++k)
                {
                    split->realp[k] = complex_input[k].real();
                    split->imagp[k] = complex_input[k].imag();
                }
            }
        }

        // --- Unpacking Helper: vDSP Split Complex to Complex Conjugate Even (CCE) ---
        // Converts the 'split complex' format produced by vDSP_fft_zrip (forward transform)
        // into the standard OmniDSP complex array format (size N/2+1).
        // `split_input_void` must point to a DSPSplitComplex (float) or DSPDoubleSplitComplex (double).
        // `cce_output` must have size `complex_length` (N/2+1).
        void unpack_split_to_complex(const void *split_input_void, std::complex<T> *cce_output) const
        {
            const size_t N = length;          // Original real length
            const size_t Nc = complex_length; // Complex length (N/2+1)
            if (Nc == 0 || N == 0)
                return; // Nothing to unpack

            if constexpr (std::is_same_v<T, float>) // SINGLE precision
            {
                const DSPSplitComplex *split = reinterpret_cast<const DSPSplitComplex *>(split_input_void);
                // Unpack DC component (realp[0])
                cce_output[0] = std::complex<T>(split->realp[0], 0.0f);
                // Unpack components from 1 to N/2 - 1
                for (size_t k = 1; k < N / 2; ++k)
                {
                    cce_output[k] = std::complex<T>(split->realp[k], split->imagp[k]);
                }
                // Unpack Nyquist component (imagp[0]) if N is even
                if (N % 2 == 0 && Nc > 1)
                { // Check Nc > 1 ensures output array is large enough
                    cce_output[N / 2] = std::complex<T>(split->imagp[0], 0.0f);
                }
            }
            else // DOUBLE precision
            {
                const DSPDoubleSplitComplex *split = reinterpret_cast<const DSPDoubleSplitComplex *>(split_input_void);
                // Unpack DC component
                cce_output[0] = std::complex<T>(split->realp[0], 0.0);
                // Unpack components from 1 to N/2 - 1
                for (size_t k = 1; k < N / 2; ++k)
                {
                    cce_output[k] = std::complex<T>(split->realp[k], split->imagp[k]);
                }
                // Unpack Nyquist component if N is even
                if (N % 2 == 0 && Nc > 1)
                {
                    cce_output[N / 2] = std::complex<T>(split->imagp[0], 0.0);
                }
            }
        }

        // --- Rule of 5/3: Move Semantics ---
        // Enable moving FFTPlanImpl objects (transfer ownership of handles/buffers)
        // but disable copying (as handles are not trivially copyable).
        FFTPlanImpl(FFTPlanImpl &&other) noexcept
            : length(other.length), complex_length(other.complex_length), direction(other.direction),
              precision(other.precision), domain(other.domain), norm_mode(other.norm_mode),
              forward_scale(other.forward_scale), backward_scale(other.backward_scale),
              setup_handle_c2c(other.setup_handle_c2c), setup_handle_real(other.setup_handle_real),
              real_buffer(std::move(other.real_buffer)), imag_buffer(std::move(other.imag_buffer)),
              stride(other.stride)
        {
            // Nullify handles in the moved-from object to prevent double-free
            other.setup_handle_c2c = nullptr;
            other.setup_handle_real = nullptr;
        }
        FFTPlanImpl &operator=(FFTPlanImpl &&other) noexcept
        {
            if (this != &other)
            {
                // Release existing resources first
                if (setup_handle_c2c)
                { /* destroy c2c */
                }
                if (setup_handle_real)
                { /* destroy real */
                }
                // Swap members
                std::swap(length, other.length);
                std::swap(complex_length, other.complex_length);
                // ... swap all other members ...
                std::swap(setup_handle_c2c, other.setup_handle_c2c);
                std::swap(setup_handle_real, other.setup_handle_real);
                std::swap(real_buffer, other.real_buffer);
                std::swap(imag_buffer, other.imag_buffer);
            }
            return *this;
        }
        FFTPlanImpl(const FFTPlanImpl &) = delete;            // Disable copy constructor
        FFTPlanImpl &operator=(const FFTPlanImpl &) = delete; // Disable copy assignment

        // --- Execute Methods (FFT) ---

        // Execute Complex-to-Complex FFT (Out-of-Place)
        void execute_c2c_oop(const std::complex<T> *input, std::complex<T> *output) const
        {
            if (!setup_handle_c2c)
                throw std::runtime_error("C2C FFT plan not initialized.");
            // vDSP_DFT expects separate real and imaginary pointers.
            if constexpr (std::is_same_v<T, float>)
            {
                vDSP_DFT_Execute(reinterpret_cast<vDSP_DFT_Setup>(setup_handle_c2c),
                                 reinterpret_cast<const float *>(input),     // Real part of input
                                 reinterpret_cast<const float *>(input) + 1, // Imag part of input (interleaved)
                                 reinterpret_cast<float *>(output),          // Real part of output
                                 reinterpret_cast<float *>(output) + 1);     // Imag part of output (interleaved)
                // Apply external scaling if needed (vDSP_DFT doesn't scale itself)
                if (std::abs(forward_scale - 1.0f) > std::numeric_limits<float>::epsilon() && direction == Direction::FORWARD)
                {
                    vDSP_vsmul(reinterpret_cast<const float *>(output), 2, &forward_scale, reinterpret_cast<float *>(output), 2, length);
                }
                else if (std::abs(backward_scale - 1.0f) > std::numeric_limits<float>::epsilon() && direction == Direction::INVERSE)
                {
                    vDSP_vsmul(reinterpret_cast<const float *>(output), 2, &backward_scale, reinterpret_cast<float *>(output), 2, length);
                }
            }
            else
            {
                vDSP_DFT_ExecuteD(reinterpret_cast<vDSP_DFT_SetupD>(setup_handle_c2c),
                                  reinterpret_cast<const double *>(input),     // Real part of input
                                  reinterpret_cast<const double *>(input) + 1, // Imag part of input (interleaved)
                                  reinterpret_cast<double *>(output),          // Real part of output
                                  reinterpret_cast<double *>(output) + 1);     // Imag part of output (interleaved)
                // Apply external scaling
                if (std::abs(forward_scale - 1.0) > std::numeric_limits<double>::epsilon() && direction == Direction::FORWARD)
                {
                    vDSP_vsmulD(reinterpret_cast<const double *>(output), 2, &forward_scale, reinterpret_cast<double *>(output), 2, length);
                }
                else if (std::abs(backward_scale - 1.0) > std::numeric_limits<double>::epsilon() && direction == Direction::INVERSE)
                {
                    vDSP_vsmulD(reinterpret_cast<const double *>(output), 2, &backward_scale, reinterpret_cast<double *>(output), 2, length);
                }
            }
        }

        // Execute Complex-to-Complex FFT (In-Place)
        // vDSP_DFT does not support true in-place, so we simulate it using the OOP version and a copy.
        void execute_c2c_ip(std::complex<T> *data) const
        {
            if (!setup_handle_c2c)
                throw std::runtime_error("C2C FFT plan not initialized.");
            // Allocate temporary output buffer
            std::vector<std::complex<T>> temp_output(length);
            // Perform out-of-place transform
            execute_c2c_oop(data, temp_output.data());
            // Copy result back to original buffer
            std::copy(temp_output.begin(), temp_output.end(), data);
        }

        // Execute Real-to-Complex FFT (Out-of-Place)
        // Input: real_input (size N)
        // Output: complex_output (size N/2+1, CCE format)
        void execute_rfft_oop(const T *real_input, std::complex<T> *complex_output) const
        {
            if (!setup_handle_real)
                throw std::runtime_error("Real FFT plan not initialized.");
            // vDSP requires input in 'split complex' format even for real input.
            // We use vDSP_ctoz to convert the real input into this format.
            if constexpr (std::is_same_v<T, float>)
            {
                DSPSplitComplex split_input = {real_buffer.data(), imag_buffer.data()};
                // Convert real input to split complex format (real parts in realp, imag parts in imagp)
                vDSP_ctoz(reinterpret_cast<const DSPComplex *>(real_input), 2, // Source (treat real as complex with imag=0), stride 2
                          &split_input, 1, length / 2);                        // Destination split complex, stride 1, N/2 elements

                // Perform in-place complex FFT on the split complex data
                vDSP_fft_zrip(setup_handle_real, &split_input, stride, get_log2n(length), kFFTDirection_Forward);

                // Unpack the resulting split complex format into standard CCE complex format
                unpack_split_to_complex(&split_input, complex_output);

                // Apply external scaling if needed
                if (std::abs(forward_scale - 1.0f) > std::numeric_limits<float>::epsilon())
                {
                    // Scale the output complex array (note: requires careful handling of strides if not interleaved)
                    // Assuming complex_output is standard interleaved std::complex<T>
                    vDSP_vsmul(reinterpret_cast<float *>(complex_output), 1, &forward_scale, reinterpret_cast<float *>(complex_output), 1, complex_length * 2); // Multiply real and imag parts
                }
            }
            else
            {
                DSPDoubleSplitComplex split_input = {real_buffer.data(), imag_buffer.data()};
                vDSP_ctozD(reinterpret_cast<const DSPDoubleComplex *>(real_input), 2,
                           &split_input, 1, length / 2);

                vDSP_fft_zripD(setup_handle_real, &split_input, stride, get_log2n(length), kFFTDirection_Forward);

                unpack_split_to_complex(&split_input, complex_output);

                // Apply external scaling
                if (std::abs(forward_scale - 1.0) > std::numeric_limits<double>::epsilon())
                {
                    vDSP_vsmulD(reinterpret_cast<double *>(complex_output), 1, &forward_scale, reinterpret_cast<double *>(complex_output), 1, complex_length * 2);
                }
            }
        }

        // Execute Complex-to-Real Inverse FFT (Out-of-Place)
        // Input: complex_input (size N/2+1, CCE format, Hermitian symmetry assumed)
        // Output: real_output (size N)
        void execute_irfft_oop(const std::complex<T> *complex_input, T *real_output) const
        {
            if (!setup_handle_real)
                throw std::runtime_error("Real FFT plan not initialized.");
            // Pack the input CCE complex data into the required split complex format.
            if constexpr (std::is_same_v<T, float>)
            {
                DSPSplitComplex split_packed = {real_buffer.data(), imag_buffer.data()};
                pack_complex_to_split(complex_input, &split_packed);

                // Perform in-place inverse FFT on the split complex data
                vDSP_fft_zrip(setup_handle_real, &split_packed, stride, get_log2n(length), kFFTDirection_Inverse);

                // Apply external scaling FIRST (vDSP inverse FFT has implicit 2/N scaling)
                if (std::abs(backward_scale - 1.0f) > std::numeric_limits<float>::epsilon())
                {
                    // Scale the packed real/imag parts before converting back
                    vDSP_vsmul(split_packed.realp, 1, &backward_scale, split_packed.realp, 1, length / 2);
                    vDSP_vsmul(split_packed.imagp, 1, &backward_scale, split_packed.imagp, 1, length / 2);
                }

                // Convert the resulting split complex data back to interleaved real output
                vDSP_ztoc(&split_packed, 1,                               // Source split complex, stride 1
                          reinterpret_cast<DSPComplex *>(real_output), 2, // Destination real (treat as complex), stride 2
                          length / 2);                                    // N/2 elements
            }
            else
            {
                DSPDoubleSplitComplex split_packed = {real_buffer.data(), imag_buffer.data()};
                pack_complex_to_split(complex_input, &split_packed);

                vDSP_fft_zripD(setup_handle_real, &split_packed, stride, get_log2n(length), kFFTDirection_Inverse);

                // Apply external scaling
                if (std::abs(backward_scale - 1.0) > std::numeric_limits<double>::epsilon())
                {
                    vDSP_vsmulD(split_packed.realp, 1, &backward_scale, split_packed.realp, 1, length / 2);
                    vDSP_vsmulD(split_packed.imagp, 1, &backward_scale, split_packed.imagp, 1, length / 2);
                }

                vDSP_ztocD(&split_packed, 1,
                           reinterpret_cast<DSPDoubleComplex *>(real_output), 2,
                           length / 2);
            }
        }
    }; // End FFTPlanImpl struct

    // Explicit Instantiations for FFTPlanImpl (required by compiler)
    template struct FFTPlanImpl<float>;
    template struct FFTPlanImpl<double>;

    // --- FFTPlan Method Definitions ---
    // These are the public methods of FFTPlan<T> that users interact with.
    // They mostly forward calls to the private implementation (pimpl_).

    template <typename T>
    FFTPlan<T>::FFTPlan(size_t l, Precision p, Direction d, Domain dom, NormMode n)
        : pimpl_(std::make_unique<FFTPlanImpl<T>>(l, p, d, dom, n)) {} // Create the implementation object

    template <typename T>
    FFTPlan<T>::~FFTPlan() = default; // Default destructor (unique_ptr handles pimpl_ cleanup)

    template <typename T>
    FFTPlan<T>::FFTPlan(FFTPlan &&other) noexcept = default; // Default move constructor

    template <typename T>
    FFTPlan<T> &FFTPlan<T>::operator=(FFTPlan<T> &&other) noexcept = default; // Default move assignment

    // Execute C2C OOP
    template <typename T>
    void FFTPlan<T>::execute(const std::complex<T> *i, std::complex<T> *o) const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
        pimpl_->execute_c2c_oop(i, o); // Forward to implementation
    }

    // Execute C2C IP
    template <typename T>
    void FFTPlan<T>::execute(std::complex<T> *d) const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
        pimpl_->execute_c2c_ip(d); // Forward to implementation
    }

    // Execute RFFT OOP
    template <typename T>
    void FFTPlan<T>::execute_rfft(const T *ri, std::complex<T> *co) const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
        pimpl_->execute_rfft_oop(ri, co); // Forward to implementation
    }

    // Execute IRFFT OOP
    template <typename T>
    void FFTPlan<T>::execute_irfft(const std::complex<T> *ci, T *ro) const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
        pimpl_->execute_irfft_oop(ci, ro); // Forward to implementation
    }

    // --- Getters ---
    template <typename T>
    size_t FFTPlan<T>::getLength() const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan");
        return pimpl_->length;
    }
    template <typename T>
    size_t FFTPlan<T>::getComplexLength() const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan");
        return pimpl_->complex_length;
    }
    template <typename T>
    Direction FFTPlan<T>::getDirection() const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan");
        return pimpl_->direction;
    }
    template <typename T>
    Precision FFTPlan<T>::getPrecision() const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan");
        return pimpl_->precision;
    }
    template <typename T>
    Domain FFTPlan<T>::getDomain() const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan");
        return pimpl_->domain;
    }
    template <typename T>
    NormMode FFTPlan<T>::getNormMode() const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan");
        return pimpl_->norm_mode;
    }

    // Explicit Instantiations for FFTPlan class (required by compiler)
    template class FFTPlan<float>;
    template class FFTPlan<double>;

    // === Backend Implementation for FIR, Conv/Corr, Desamp ===

    namespace Backend
    {
        // Contains the Accelerate-specific implementations called by public API functions.

        // --- vDSP Helper Structs ---
        // These structs use template specialization to select the correct vDSP function
        // (float or double version) based on the template type T at compile time.

        // Helper for vDSP_fir (Finite Impulse Response filter)
        template <typename T>
        struct vDSPFIRHelper;
        template <>
        struct vDSPFIRHelper<float>
        {
            static constexpr auto fir = vDSP_fir_f;
        }; // vDSP_fir_f for float
        template <>
        struct vDSPFIRHelper<double>
        {
            static constexpr auto fir = vDSP_fir_d;
        }; // vDSP_fir_d for double

        // Helper for vDSP_vrvrs (Vector Reverse)
        template <typename T>
        struct vDSPReverseHelper;
        template <>
        struct vDSPReverseHelper<float>
        {
            static constexpr auto vrvrs = vDSP_vrvrs;
        }; // vDSP_vrvrs for float
        template <>
        struct vDSPReverseHelper<double>
        {
            static constexpr auto vrvrs = vDSP_vrvrsD;
        }; // vDSP_vrvrsD for double

        // Helper for vDSP_desamp (Decimate and Sample - FIR filter + downsample)
        template <typename T>
        struct vDSPDesampHelper;
        template <>
        struct vDSPDesampHelper<float>
        {
            static constexpr auto desamp = vDSP_desamp;
        }; // vDSP_desamp for float
        template <>
        struct vDSPDesampHelper<double>
        {
            static constexpr auto desamp = vDSP_desampD;
        }; // vDSP_desampD for double

        // --- Backend Implementation Functions ---

        /**
         * @brief Accelerate backend implementation for 1D convolution or correlation.
         * Uses vDSP_fir for correlation directly. For convolution, it reverses
         * the kernel using vDSP_vrvrs first, then calls vDSP_fir.
         * Always calculates the 'valid' part of the result.
         */
        template <typename T>
        std::vector<T> convolve1d_impl(const std::vector<T> &signal, const std::vector<T> &kernel, bool use_correlation)
        {
            // Input validation
            if (signal.empty() || kernel.empty())
                return {}; // Return empty vector if inputs are empty

            vDSP_Length sig_len = signal.size();
            vDSP_Length ker_len = kernel.size();

            // Calculate output length for 'valid' mode
            // N_out = N_signal - N_kernel + 1
            long long out_len_signed = static_cast<long long>(sig_len) - static_cast<long long>(ker_len) + 1;
            if (out_len_signed <= 0)
            {
                // Signal must be at least as long as kernel for 'valid' output.
                throw std::invalid_argument("Accelerate FIR/Conv ('valid' mode): signal length (" + std::to_string(sig_len) + ") must be >= kernel length (" + std::to_string(ker_len) + ").");
            }
            vDSP_Length out_len = static_cast<vDSP_Length>(out_len_signed);
            std::vector<T> result(out_len); // Allocate result vector

            std::vector<T> kernel_to_use = kernel; // Make a mutable copy of the kernel

            // --- Perform Operation ---
            if (!use_correlation) // CONVOLUTION: Reverse kernel first
            {
                if (ker_len > 0) // Need kernel to reverse
                {
                    // Use the type-specific helper to call vDSP_vrvrs or vDSP_vrvrsD
                    vDSPReverseHelper<T>::vrvrs(kernel_to_use.data(), /*stride=*/1, ker_len);
                }
                // Now kernel_to_use holds the time-reversed kernel.
                // Fall through to call vDSP_fir with the reversed kernel.
            }
            // else: CORRELATION: Use kernel_to_use directly (it's the original kernel).

            // Call vDSP_fir (or vDSP_fir_d) using the appropriate kernel
            // Note: vDSP_fir arguments are slightly unconventional:
            // vDSP_fir(InputSignal, InputStride, FilterKernel, FilterStride, Output, OutputStride, OutputLength, FilterLength)
            vDSPFIRHelper<T>::fir(signal.data(),        // Input Signal C -> A in docs (Signal to filter)
                                  /*InputStride=*/1,    // Stride for input signal
                                  kernel_to_use.data(), // Filter F -> H in docs (Kernel coefficients)
                                  /*FilterStride=*/1,   // Stride for filter kernel
                                  result.data(),        // Output A -> C in docs (Result buffer)
                                  /*OutputStride=*/1,   // Stride for output
                                  out_len,              // Length of Output NC -> N-M+1 (Number of output samples)
                                  ker_len);             // Length of Filter M -> M (Number of coefficients)

            return result; // Return the computed 'valid' part
        }

        /**
         * @brief Accelerate backend implementation for combined FIR filtering and downsampling.
         * Uses vDSP_desamp which performs both operations efficiently.
         */
        template <typename T>
        std::vector<T> filter_and_downsample_impl(const std::vector<T> &signal, const std::vector<T> &kernel, int factor)
        {
            // Input validation
            if (signal.empty() || kernel.empty())
                return {}; // Return empty if inputs are empty
            if (factor <= 0)
                throw std::invalid_argument("Downsampling factor must be positive.");

            vDSP_Length sig_len = signal.size();
            vDSP_Length ker_len = kernel.size();
            vDSP_Length downsample_factor = static_cast<vDSP_Length>(factor); // Cast factor

            // Calculate output length for vDSP_desamp: floor((sig_len - ker_len) / factor) + 1
            // Need careful calculation to handle potential negative numerator if sig_len < ker_len
            long long numerator = static_cast<long long>(sig_len) - static_cast<long long>(ker_len);

            // Calculate output length, ensuring it's non-negative
            vDSP_Length out_len = 0;
            if (numerator >= 0)
            {
                out_len = (static_cast<vDSP_Length>(numerator) / downsample_factor) + 1;
            } // else out_len remains 0

            if (out_len == 0)
                return {}; // No output samples possible, return empty vector

            std::vector<T> result(out_len); // Allocate result vector

            // Call vDSP_desamp (or vDSP_desampD) using the helper struct
            // vDSP_desamp(InputSignal, DecimationFactor, FilterCoeffs, Output, OutputLength, FilterLength)
            vDSPDesampHelper<T>::desamp(signal.data(),     // Input signal (A)
                                        downsample_factor, // Decimation factor (D)
                                        kernel.data(),     // Filter coefficients (F)
                                        result.data(),     // Output buffer (C)
                                        out_len,           // Number of output samples to calculate (N)
                                        ker_len);          // Length of filter kernel (L)

            return result;
        }

        // --- Explicit Instantiations for Backend functions ---
        // These ensure the compiler generates code for both float and double versions.
        template std::vector<float> convolve1d_impl<float>(const std::vector<float> &, const std::vector<float> &, bool);
        template std::vector<double> convolve1d_impl<double>(const std::vector<double> &, const std::vector<double> &, bool);

        template std::vector<float> filter_and_downsample_impl<float>(const std::vector<float> &, const std::vector<float> &, int);
        template std::vector<double> filter_and_downsample_impl<double>(const std::vector<double> &, const std::vector<double> &, int);

    } // namespace Backend
} // namespace OmniDSP

#endif // USE_ACCELERATE