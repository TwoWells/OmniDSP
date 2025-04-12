/**
 * @file onemkl.cpp
 * @brief Intel oneMKL backend implementation for OmniDSP (FFT, Conv/Corr, Filter+Downsample).
 *
 * Implements FFTPlanImpl and backend functions using Intel oneMKL.
 * Compiled only when USE_ONEMKL is defined.
 *
 * Version 1.0.4: Using VSL_CONV_MODE_AUTO instead of VSL_CONV_MODE_DIRECT.
 * Using MKL Conv for Math Conv, MKL Corr for Math Corr based on naming docs.
 */

// --- Includes ---
#include <vector>
#include <complex>
#include <memory>        // For std::unique_ptr
#include <stdexcept>     // For std::runtime_error, std::invalid_argument
#include <type_traits>   // For std::is_same_v, std::conditional
#include <string>
#include <cmath>         // For std::sqrt
#include <algorithm>     // For std::copy, std::max, std::reverse
#include <iostream>      // Include iostream for std::cout/cerr
#include <iomanip>       // Include iomanip for std::setprecision
#include <OmniDSP/omnidsp.h> // Public API header
#include "backend_impl.h" // Internal backend function declarations

#if defined(USE_ONEMKL) // Only compile if USE_ONEMKL is defined

#include <mkl.h>             // Main MKL header
#include <mkl_vsl.h>         // VSL header (likely included by mkl.h, but explicit for clarity)

namespace OmniDSP
{
    // --- MKL Helpers ---
    /**
     * @brief Checks the status code returned by an MKL DFTI function.
     * Throws a std::runtime_error if the status indicates an error.
     */
    inline void check_mkl_status(MKL_LONG status, const char *error_msg)
    {
        // DFTI uses MKL_LONG status codes. Check if it's an error (not DFTI_NO_ERROR).
        if (!DftiErrorClass(status, DFTI_NO_ERROR)) // DftiErrorClass returns true for errors
        {
            std::string full_msg = error_msg;
            // Append the MKL status number and the corresponding error message string.
            full_msg += ": MKL DFTI Status " + std::to_string(status) + " - " + DftiErrorMessage(status);
            throw std::runtime_error(full_msg);
        }
    }
    /**
     * @brief Checks the status code returned by an MKL VSL function.
     * Throws a std::runtime_error if the status indicates an error.
     */
    inline void check_mkl_status(int status, const char *error_msg)
    {
        // VSL typically uses int status codes. Check if it's not VSL_STATUS_OK (usually 0).
        if (status != VSL_STATUS_OK)
        {
            std::string full_msg = error_msg;
            // Append the VSL status number.
            full_msg += ": MKL VSL Status " + std::to_string(status);
            throw std::runtime_error(full_msg);
        }
    }

    // --- FFTPlanImpl Definition (MKL Backend) ---
    // Full definition of the implementation class for FFTPlan using MKL DFTI.
    template <typename T>
    struct FFTPlanImpl
    {
        // --- Members ---
        size_t length = 0;             // FFT Length (N)
        size_t complex_length = 0;     // Length of complex spectrum (N for C2C, N/2+1 for Real)
        Direction direction = Direction::FORWARD; // Transform direction
        Precision precision = Precision::SINGLE;  // float or double
        Domain domain = Domain::COMPLEX;        // COMPLEX or REAL
        NormMode norm_mode = NormMode::BACKWARD; // Normalization mode
        T forward_scale = 1.0;                   // Scaling factor for forward FFT
        T backward_scale = 1.0;                  // Scaling factor for inverse FFT
        DFTI_DESCRIPTOR_HANDLE handle = nullptr;   // MKL DFTI descriptor

        // --- Constructor ---
        // Creates and configures the MKL DFTI descriptor.
        FFTPlanImpl(size_t len, Precision prec, Direction dir, Domain dom, NormMode norm)
        {
            if (len == 0) throw std::invalid_argument("FFT length N cannot be zero.");
            length = len; direction = dir; precision = prec; domain = dom; norm_mode = norm;
            complex_length = (dom == Domain::REAL) ? len / 2 + 1 : len;

            // Calculate scaling factors
            T scaleN = static_cast<T>(length); T scaleSqrtN = std::sqrt(scaleN);
            if (scaleN == static_cast<T>(0.0)) throw std::runtime_error("ScaleN is zero");
            if (scaleSqrtN == static_cast<T>(0.0) && norm_mode == NormMode::ORTHO) throw std::runtime_error("ScaleSqrtN is zero for ORTHO");
            switch (norm_mode) {
                case NormMode::BACKWARD: forward_scale = 1.0; backward_scale = 1.0 / scaleN; break;
                case NormMode::ORTHO: forward_scale = 1.0 / scaleSqrtN; backward_scale = 1.0 / scaleSqrtN; break;
                case NormMode::FORWARD: forward_scale = 1.0 / scaleN; backward_scale = 1.0; break;
            }

            // Create MKL DFTI Descriptor
            MKL_LONG status;
            DFTI_CONFIG_VALUE domain_type = (dom == Domain::REAL) ? DFTI_REAL : DFTI_COMPLEX;
            DFTI_CONFIG_VALUE prec_type = (prec == Precision::SINGLE) ? DFTI_SINGLE : DFTI_DOUBLE;
            status = DftiCreateDescriptor(&handle, prec_type, domain_type, 1, (MKL_LONG)len);
            check_mkl_status(status, "MKL DftiCreateDescriptor failed");

            // Configure Descriptor
            using MKLScaleType = typename std::conditional<std::is_same_v<T, float>, float, double>::type;
            status = DftiSetValue(handle, DFTI_FORWARD_SCALE, static_cast<MKLScaleType>(forward_scale));
            check_mkl_status(status, "MKL DftiSetValue DFTI_FORWARD_SCALE failed");
            status = DftiSetValue(handle, DFTI_BACKWARD_SCALE, static_cast<MKLScaleType>(backward_scale));
            check_mkl_status(status, "MKL DftiSetValue DFTI_BACKWARD_SCALE failed");
            if (dom == Domain::REAL) {
                // Use standard complex storage for real FFT results (N/2+1 complex points)
                status = DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
                check_mkl_status(status, "MKL DftiSetValue DFTI_CONJUGATE_EVEN_STORAGE failed");
            }
            // Configure for out-of-place transforms by default
            status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            check_mkl_status(status, "MKL DftiSetValue DFTI_PLACEMENT failed for NOT_INPLACE");

            // Commit Descriptor
            status = DftiCommitDescriptor(handle);
            check_mkl_status(status, "MKL DftiCommitDescriptor failed");
        }

        // --- Destructor ---
        ~FFTPlanImpl() {
            if (handle) { (void)DftiFreeDescriptor(&handle); }
        }

        // --- Rule of 5/3: Move Semantics ---
        FFTPlanImpl(FFTPlanImpl &&other) noexcept
            : length(other.length), complex_length(other.complex_length), direction(other.direction),
              precision(other.precision), domain(other.domain), norm_mode(other.norm_mode),
              forward_scale(other.forward_scale), backward_scale(other.backward_scale), handle(other.handle)
        { other.handle = nullptr; }
        FFTPlanImpl &operator=(FFTPlanImpl &&other) noexcept {
             if (this != &other) {
                 if (handle) { (void)DftiFreeDescriptor(&handle); }
                 length = other.length; complex_length = other.complex_length; direction = other.direction;
                 precision = other.precision; domain = other.domain; norm_mode = other.norm_mode;
                 forward_scale = other.forward_scale; backward_scale = other.backward_scale;
                 handle = other.handle;
                 other.handle = nullptr;
             }
             return *this;
        }
        FFTPlanImpl(const FFTPlanImpl &) = delete;
        FFTPlanImpl &operator=(const FFTPlanImpl &) = delete;

        // --- Execute Methods (FFT) ---
        // Execute Complex-to-Complex FFT (Out-of-Place)
        void execute_c2c_oop(const std::complex<T> *input, std::complex<T> *output) const {
             if (!handle) throw std::runtime_error("MKL FFTPlan handle is null.");
             MKL_LONG status;
             if (direction == Direction::FORWARD) {
                 status = DftiComputeForward(handle, const_cast<std::complex<T>*>(input), output);
             } else {
                 status = DftiComputeBackward(handle, const_cast<std::complex<T>*>(input), output);
             }
             check_mkl_status(status, "MKL DftiComputeForward/Backward (C2C OOP) failed");
        }

        // Execute Complex-to-Complex FFT (In-Place)
        void execute_c2c_ip(std::complex<T> *data) const {
             if (!handle) throw std::runtime_error("MKL FFTPlan handle is null.");
             MKL_LONG status;
             // Assume plan is configured DFTI_NOT_INPLACE and simulate IP
             std::vector<std::complex<T>> temp_output(length);
             if (direction == Direction::FORWARD) {
                 status = DftiComputeForward(handle, data, temp_output.data());
                 check_mkl_status(status, "MKL DftiComputeForward (C2C IP simulation) failed");
             } else { // INVERSE
                 status = DftiComputeBackward(handle, data, temp_output.data());
                 check_mkl_status(status, "MKL DftiComputeBackward (C2C IP simulation) failed");
             }
             std::copy(temp_output.begin(), temp_output.end(), data);
        }

        // Execute Real-to-Complex FFT (Out-of-Place)
        void execute_rfft_oop(const T *real_input, std::complex<T> *complex_output) const {
            if (!handle) throw std::runtime_error("MKL FFTPlan handle is null.");
            if (domain != Domain::REAL || direction != Direction::FORWARD) {
                throw std::runtime_error("execute_rfft called on incorrect plan type (domain/direction).");
            }
            MKL_LONG status = DftiComputeForward(handle, const_cast<T*>(real_input), complex_output);
            check_mkl_status(status, "MKL DftiComputeForward (RFFT OOP) failed");
        }

        // Execute Complex-to-Real Inverse FFT (Out-of-Place)
        void execute_irfft_oop(const std::complex<T> *complex_input, T *real_output) const {
            if (!handle) throw std::runtime_error("MKL FFTPlan handle is null.");
            if (domain != Domain::REAL || direction != Direction::INVERSE) {
                 throw std::runtime_error("execute_irfft called on incorrect plan type (domain/direction).");
            }
            MKL_LONG status = DftiComputeBackward(handle, const_cast<std::complex<T>*>(complex_input), real_output);
            check_mkl_status(status, "MKL DftiComputeBackward (IRFFT OOP) failed");
        }
    }; // End FFTPlanImpl struct definition

    // Explicit Instantiations for FFTPlanImpl
    template struct FFTPlanImpl<float>;
    template struct FFTPlanImpl<double>;

    // --- FFTPlan Method Definitions ---
    // These implement the public FFTPlan class methods by forwarding to the pimpl_
    template <typename T>
    FFTPlan<T>::FFTPlan(size_t l, Precision p, Direction d, Domain dom, NormMode n)
        : pimpl_(std::make_unique<FFTPlanImpl<T>>(l, p, d, dom, n)) {}

    template <typename T>
    FFTPlan<T>::~FFTPlan() = default;

    template <typename T>
    FFTPlan<T>::FFTPlan(FFTPlan &&other) noexcept = default;

    template <typename T>
    FFTPlan<T> &FFTPlan<T>::operator=(FFTPlan<T> &&other) noexcept = default;

    template <typename T>
    void FFTPlan<T>::execute(const std::complex<T> *i, std::complex<T> *o) const {
        if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
        pimpl_->execute_c2c_oop(i, o);
    }

    template <typename T>
    void FFTPlan<T>::execute(std::complex<T> *d) const {
        if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
        pimpl_->execute_c2c_ip(d);
    }

    template <typename T>
    void FFTPlan<T>::execute_rfft(const T *ri, std::complex<T> *co) const {
        if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
        pimpl_->execute_rfft_oop(ri, co);
    }

    template <typename T>
    void FFTPlan<T>::execute_irfft(const std::complex<T> *ci, T *ro) const {
        if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
        pimpl_->execute_irfft_oop(ci, ro);
    }

    // --- Getters ---
    template <typename T> size_t FFTPlan<T>::getLength() const {
        if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
        return pimpl_->length;
    }
    template <typename T> size_t FFTPlan<T>::getComplexLength() const {
        if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
        return pimpl_->complex_length;
    }
    template <typename T> Direction FFTPlan<T>::getDirection() const {
        if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
        return pimpl_->direction;
    }
    template <typename T> Precision FFTPlan<T>::getPrecision() const {
        if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
        return pimpl_->precision;
    }
    template <typename T> Domain FFTPlan<T>::getDomain() const {
        if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
        return pimpl_->domain;
    }
    template <typename T> NormMode FFTPlan<T>::getNormMode() const {
        if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from or uninitialized).");
        return pimpl_->norm_mode;
    }

    // Explicit Instantiations for FFTPlan class
    template class FFTPlan<float>;
    template class FFTPlan<double>;

    // === Backend Implementation for Conv/Corr and Filter+Downsample ===
    namespace Backend
    {
        /**
         * @brief MKL backend implementation for 1D convolution or correlation using VSL.
         * Calculates the 'valid' part using AUTO mode.
         * Selects MKL Convolution function for mathematical Convolution.
         * Selects MKL Correlation function for mathematical Correlation.
         * Kernel is NEVER reversed explicitly.
         * Explicitly sets start index to 0 for both conv and corr.
         * Swapped kernel/signal order in Exec calls based on Intel/ESSL examples.
         */
        template <typename T>
        std::vector<T> convolve1d_impl(const std::vector<T> &signal, const std::vector<T> &kernel, bool use_correlation)
        {
            if (signal.empty() || kernel.empty()) { return {}; }

            // MKL function parameters nx, ny, nz correspond to lengths of
            // first sequence (x), second sequence (y), and result (z).
            // Based on ESSL/Intel examples, kernel is MKL's x, signal is MKL's y.
            int nx = static_cast<int>(kernel.size()); // Kernel length (MKL's x)
            int ny = static_cast<int>(signal.size()); // Signal length (MKL's y)
            int nz = ny - nx + 1; // Output length for 'valid' part (sig_len - ker_len + 1)

            if (nz <= 0) {
                 // For 'valid' mode equivalent, signal must be >= kernel.
                 // If using FFT/AUTO mode, MKL might handle this, but let's keep the check
                 // as the *intent* is to match SciPy's 'valid' mode.
                throw std::invalid_argument("MKL Conv/Corr ('valid' mode equivalent): signal length (" + std::to_string(ny) + ") must be >= kernel length (" + std::to_string(nx) + ").");
            }
            std::vector<T> result(nz);

            // --- Use AUTO mode ---
            const int mode = VSL_CONV_MODE_AUTO; // Let MKL choose algorithm

            VSLConvTaskPtr task = nullptr; // Use VSLConvTaskPtr for both Conv and Corr tasks
            int status = VSL_STATUS_OK;

            // Kernel is always passed as original (un-reversed)
            const std::vector<T>& kernel_ptr_for_mkl = kernel;

            // Strides
            MKL_INT stride_kernel = 1; // Stride for kernel (MKL's x)
            MKL_INT stride_signal = 1; // Stride for signal (MKL's y)
            MKL_INT stride_result = 1; // Stride for result (MKL's z)
            // --- FIX: Set start index based on 'valid' mode equivalent ---
            // For 'valid' mode, the first output corresponds to the kernel
            // fully overlapping the beginning of the signal.
            // In MKL's definition (kernel=h, signal=x):
            // Conv (Math Corr): y(i) = sum[ h(k) * x(i+k) ]. First output (i=0) needs x[0]..x[nx-1]. Start index=0.
            // Corr (Math Conv): y(i) = sum[ h(k) * x(i-k) ]. First output (i=0) needs x[0]..x[-(nx-1)]? No, MKL handles this.
            // For 'valid' output y[0..nz-1], the corresponding signal indices used are different for conv/corr.
            // However, MKL VSL documentation for SetStart suggests iy0 is the index in the *mathematical*
            // result sequence (full convolution/correlation) where the output should begin.
            // For 'valid' mode, the output starts at index 'nx-1' in the full convolution result,
            // and index '0' in the full correlation result (using typical definitions).
            // Since MKL Conv = Math Corr and MKL Corr = Math Conv, we set:
            // - start_index = 0 for MKL Conv (Math Corr)
            // - start_index = nx - 1 for MKL Corr (Math Conv)
            MKL_INT start_index = use_correlation ? 0 : (nx - 1);
            // --- End FIX ---


            // Determine which MKL function set to use based on the desired *mathematical* operation
            // AND the MKL naming convention documentation.
            bool call_mkl_convolution = !use_correlation; // Call MKL Conv for Math Conv
            bool call_mkl_correlation = use_correlation;  // Call MKL Corr for Math Corr

            try
            {
                 // --- Debug Prints Active ---
                 std::cout << std::fixed << std::setprecision(9);
                 std::cout << "[DEBUG] MKL Backend: Requesting " << (use_correlation ? "Correlation" : "Convolution")
                           << " (Math Def). Calling MKL " << (call_mkl_convolution ? "Conv" : "Corr") << " function." << std::endl;
                 std::cout << "  [DEBUG] Mode=AUTO nx(ker)=" << nx << " ny(sig)=" << ny << " nz(out)=" << nz << std::endl; // Changed mode in log
                 std::cout << "  [DEBUG] StartIndex=" << start_index << std::endl; // Log the calculated start index
                 std::cout << "  [DEBUG] Signal(MKL y)(" << signal.size() << "): "; for(T v : signal) std::cout << v << " "; std::cout << std::endl;
                 std::cout << "  [DEBUG] Kernel(MKL x)" << "(" << kernel_ptr_for_mkl.size() << "): "; for(T v : kernel_ptr_for_mkl) std::cout << v << " "; std::cout << std::endl;
                 // --- End Debug Print ---

                if constexpr (std::is_same_v<T, float>) {
                    if (call_mkl_convolution) { // Mathematical Convolution -> Use MKL Convolution
                        status = vslsConvNewTask1D(&task, mode, nx, ny, nz); check_mkl_status(status, "MKL vslsConvNewTask1D failed");
                        status = vslConvSetStart(task, &start_index); check_mkl_status(status, "MKL vslConvSetStart failed");
                        status = vslsConvExec(task, kernel_ptr_for_mkl.data(), &stride_kernel, signal.data(), &stride_signal, result.data(), &stride_result);
                        check_mkl_status(status, "MKL vslsConvExec failed");
                    } else { // Mathematical Correlation -> Use MKL Correlation
                        status = vslsCorrNewTask1D(&task, mode, nx, ny, nz); check_mkl_status(status, "MKL vslsCorrNewTask1D failed");
                        status = vslCorrSetStart(task, &start_index); check_mkl_status(status, "MKL vslCorrSetStart failed");
                        status = vslsCorrExec(task, kernel_ptr_for_mkl.data(), &stride_kernel, signal.data(), &stride_signal, result.data(), &stride_result);
                        check_mkl_status(status, "MKL vslsCorrExec failed");
                    }
                } else { // DOUBLE
                    if (call_mkl_convolution) { // Mathematical Convolution -> Use MKL Convolution
                        status = vsldConvNewTask1D(&task, mode, nx, ny, nz); check_mkl_status(status, "MKL vsldConvNewTask1D failed");
                        status = vslConvSetStart(task, &start_index); check_mkl_status(status, "MKL vslConvSetStart failed");
                        status = vsldConvExec(task, kernel_ptr_for_mkl.data(), &stride_kernel, signal.data(), &stride_signal, result.data(), &stride_result);
                        check_mkl_status(status, "MKL vsldConvExec failed");
                    } else { // Mathematical Correlation -> Use MKL Correlation
                        status = vsldCorrNewTask1D(&task, mode, nx, ny, nz); check_mkl_status(status, "MKL vsldCorrNewTask1D failed");
                        status = vslCorrSetStart(task, &start_index); check_mkl_status(status, "MKL vslCorrSetStart failed");
                        status = vsldCorrExec(task, kernel_ptr_for_mkl.data(), &stride_kernel, signal.data(), &stride_signal, result.data(), &stride_result);
                        check_mkl_status(status, "MKL vsldCorrExec failed");
                    }
                }

                 // --- Debug Prints Active ---
                 std::cout << "  [DEBUG] Result(" << result.size() << "): "; for(T v : result) std::cout << v << " "; std::cout << std::endl << std::endl;
                 // --- End Debug Print ---

                // Use vslConvDeleteTask for both Conv and Corr tasks
                if (task) { status = vslConvDeleteTask(&task); check_mkl_status(status, "MKL vslConvDeleteTask failed"); task = nullptr; }
                return result;
            } catch (...) { if (task) { (void)vslConvDeleteTask(&task); } throw; }
            return {};
        }

        /**
         * @brief MKL backend implementation for combined FIR filtering and downsampling by factor.
         * Performs FIR filtering (mathematical correlation) using MKL VSL Correlation functions,
         * followed by manual decimation. Uses AUTO mode for the underlying correlation.
         */
        template <typename T>
        std::vector<T> filter_and_downsample_impl(const std::vector<T> &signal, const std::vector<T> &kernel, int factor)
        {
             if (signal.empty() || kernel.empty()) { return {}; }
             if (factor <= 0) { throw std::invalid_argument("Downsampling factor must be positive."); }
            // Step 1: Perform FIR filtering using the mathematical correlation implementation
            // (which calls MKL's Correlation functions with AUTO mode).
            std::vector<T> filtered_signal = convolve1d_impl(signal, kernel, /*use_correlation=*/true);
            // Step 2: Manually decimate the filtered signal.
            if (filtered_signal.empty()) { return {}; }
            size_t filtered_len = filtered_signal.size();
            // Corrected decimation length calculation
            size_t decimated_len = (filtered_len > 0) ? ((filtered_len -1) / factor + 1) : 0;
            std::vector<T> result(decimated_len);
            for (size_t i = 0; i < decimated_len; ++i) {
                size_t index_in_filtered = i * factor;
                 // No need to check bounds here as decimated_len ensures index is valid
                result[i] = filtered_signal[index_in_filtered];
            }
            return result; // Return decimated result
        }

        // --- Explicit Instantiations for Backend functions ---
        template std::vector<float> convolve1d_impl<float>(const std::vector<float> &, const std::vector<float> &, bool);
        template std::vector<double> convolve1d_impl<double>(const std::vector<double> &, const std::vector<double> &, bool);
        template std::vector<float> filter_and_downsample_impl<float>(const std::vector<float> &, const std::vector<float> &, int);
        template std::vector<double> filter_and_downsample_impl<double>(const std::vector<double> &, const std::vector<double> &, int);

    } // namespace Backend
} // namespace OmniDSP

#endif // USE_ONEMKL
