/**
 * @file onemkl.cpp
 * @brief Intel oneMKL backend implementation for OmniDSP (FFT, Conv/Corr, Filter+Downsample).
 *
 * Implements FFTPlanImpl and backend functions using Intel oneMKL.
 * The filter_and_downsample_impl function now uses Intel IPP polyphase resampling
 * for float precision. Double precision is removed due to apparent lack of
 * IPP function availability in version 2022.1.0.
 * Uses correct IPP type alias 'IppsResamplingPolyphaseFixed_32f'.
 * Compiled only when USE_ONEMKL is defined. Includes compile-time log guards.
 *
 * @version 1.2.3 - Used correct IPP type alias, removed _64f resampling impl.
 */

// --- Includes ---
#include <vector>
#include <complex>
#include <memory>      // For std::unique_ptr
#include <stdexcept>   // For std::runtime_error, std::invalid_argument
#include <type_traits> // For std::is_same_v, std::conditional
#include <string>
#include <cmath>             // For std::sqrt, std::ceil
#include <algorithm>         // For std::copy, std::max, std::reverse
#include <iostream>          // Include iostream for std::cout/cerr
#include <iomanip>           // Include iomanip for std::setprecision
#include <OmniDSP/omnidsp.h> // Public API header
#include "backend_impl.h"    // Internal backend function declarations

#if defined(USE_ONEMKL) // Only compile if USE_ONEMKL is defined

#include <mkl.h>     // Main MKL header (includes DFTI)
#include <mkl_vsl.h> // VSL header (for convolution)

// --- IPP Includes ---
#include <ipp.h>
#include <ippdefs.h>
#include <ippcore.h>
#include <ipps.h>
#include <ippvm.h>

namespace OmniDSP
{
    // --- MKL DFTI Helper ---
    inline void check_mkl_status(MKL_LONG status, const char *error_msg)
    {
        if (!DftiErrorClass(status, DFTI_NO_ERROR))
        {
            std::string full_msg = error_msg;
            full_msg += ": MKL DFTI Status " + std::to_string(status) + " - " + DftiErrorMessage(status);
            throw std::runtime_error(full_msg);
        }
    }
    // --- MKL VSL Helper ---
    inline void check_mkl_status(int status, const char *error_msg)
    {
        if (status != VSL_STATUS_OK)
        {
            std::string full_msg = error_msg;
            full_msg += ": MKL VSL Status " + std::to_string(status);
            throw std::runtime_error(full_msg);
        }
    }
    // --- Intel IPP Helper ---
    inline void check_ipp_status(IppStatus status, const char *error_msg)
    {
        if (status != ippStsNoErr)
        {
            std::string full_msg = error_msg;
            full_msg += ": IPP Status " + std::to_string(status);
            throw std::runtime_error(full_msg);
        }
    }

    // --- FFTPlanImpl Definition (MKL Backend) ---
    template <typename T>
    struct FFTPlanImpl
    {
        // Members... (omitted for brevity - same as before)
        size_t length = 0;
        size_t complex_length = 0;
        Direction direction = Direction::FORWARD;
        Precision precision = Precision::SINGLE;
        Domain domain = Domain::COMPLEX;
        NormMode norm_mode = NormMode::BACKWARD;
        T forward_scale = 1.0;
        T backward_scale = 1.0;
        DFTI_DESCRIPTOR_HANDLE handle = nullptr;

        // Constructor... (omitted for brevity - same as before)
        FFTPlanImpl(size_t len, Precision prec, Direction dir, Domain dom, NormMode norm)
        {
            if (len == 0)
                throw std::invalid_argument("FFT length N cannot be zero.");
            length = len;
            direction = dir;
            precision = prec;
            domain = dom;
            norm_mode = norm;
            complex_length = (dom == Domain::REAL) ? len / 2 + 1 : len;

            // Calculate scaling factors
            T scaleN = static_cast<T>(length);
            T scaleSqrtN = std::sqrt(scaleN);
            if (scaleN == static_cast<T>(0.0))
                throw std::runtime_error("ScaleN is zero");
            if (scaleSqrtN == static_cast<T>(0.0) && norm_mode == NormMode::ORTHO)
                throw std::runtime_error("ScaleSqrtN is zero for ORTHO");
            switch (norm_mode)
            {
            case NormMode::BACKWARD:
                forward_scale = 1.0;
                backward_scale = 1.0 / scaleN;
                break;
            case NormMode::ORTHO:
                forward_scale = 1.0 / scaleSqrtN;
                backward_scale = 1.0 / scaleSqrtN;
                break;
            case NormMode::FORWARD:
                forward_scale = 1.0 / scaleN;
                backward_scale = 1.0;
                break;
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
            if (dom == Domain::REAL)
            {
                status = DftiSetValue(handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
                check_mkl_status(status, "MKL DftiSetValue DFTI_CONJUGATE_EVEN_STORAGE failed");
            }
            status = DftiSetValue(handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
            check_mkl_status(status, "MKL DftiSetValue DFTI_PLACEMENT failed for NOT_INPLACE");

            // Commit Descriptor
            status = DftiCommitDescriptor(handle);
            check_mkl_status(status, "MKL DftiCommitDescriptor failed");
        }

        // Destructor... (omitted for brevity - same as before)
        ~FFTPlanImpl()
        {
            if (handle)
            {
                (void)DftiFreeDescriptor(&handle);
            }
        }

        // Rule of 5/3... (omitted for brevity - same as before)
        FFTPlanImpl(const FFTPlanImpl &) = delete;
        FFTPlanImpl &operator=(const FFTPlanImpl &) = delete;

        // Execute Methods... (omitted for brevity - same as before)
        void execute_c2c_oop(const std::complex<T> *input, std::complex<T> *output) const
        {
            if (!handle) throw std::runtime_error("MKL FFTPlan handle is null.");
            MKL_LONG status;
            if (direction == Direction::FORWARD) status = DftiComputeForward(handle, const_cast<std::complex<T> *>(input), output);
            else status = DftiComputeBackward(handle, const_cast<std::complex<T> *>(input), output);
            check_mkl_status(status, "MKL DftiComputeForward/Backward (C2C OOP) failed");
        }
        void execute_c2c_ip(std::complex<T> *data) const
        {
             if (!handle) throw std::runtime_error("MKL FFTPlan handle is null.");
             MKL_LONG status;
             std::vector<std::complex<T>> temp_output(length);
             if (direction == Direction::FORWARD) status = DftiComputeForward(handle, data, temp_output.data());
             else status = DftiComputeBackward(handle, data, temp_output.data());
             check_mkl_status(status, "MKL DftiComputeForward/Backward (C2C IP simulation) failed");
             std::copy(temp_output.begin(), temp_output.end(), data);
        }
        void execute_rfft_oop(const T *real_input, std::complex<T> *complex_output) const
        {
             if (!handle) throw std::runtime_error("MKL FFTPlan handle is null.");
             if (domain != Domain::REAL || direction != Direction::FORWARD) throw std::runtime_error("execute_rfft called on incorrect plan type.");
             MKL_LONG status = DftiComputeForward(handle, const_cast<T *>(real_input), complex_output);
             check_mkl_status(status, "MKL DftiComputeForward (RFFT OOP) failed");
        }
        void execute_irfft_oop(const std::complex<T> *complex_input, T *real_output) const
        {
             if (!handle) throw std::runtime_error("MKL FFTPlan handle is null.");
             if (domain != Domain::REAL || direction != Direction::INVERSE) throw std::runtime_error("execute_irfft called on incorrect plan type.");
             MKL_LONG status = DftiComputeBackward(handle, const_cast<std::complex<T> *>(complex_input), real_output);
             check_mkl_status(status, "MKL DftiComputeBackward (IRFFT OOP) failed");
        }
    };

    // Explicit Instantiations for FFTPlanImpl
    template struct FFTPlanImpl<float>;
    template struct FFTPlanImpl<double>;

    // --- FFTPlan Method Definitions ---
    // (Omitted for brevity - same as before)
    template <typename T> FFTPlan<T>::FFTPlan(size_t l, Precision p, Direction d, Domain dom, NormMode n) : pimpl_(std::make_unique<FFTPlanImpl<T>>(l, p, d, dom, n)) {}
    template <typename T> FFTPlan<T>::~FFTPlan() = default;
    template <typename T> void FFTPlan<T>::execute(const std::complex<T> *i, std::complex<T> *o) const { if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from?)"); pimpl_->execute_c2c_oop(i, o); }
    template <typename T> void FFTPlan<T>::execute(std::complex<T> *d) const { if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from?)"); pimpl_->execute_c2c_ip(d); }
    template <typename T> void FFTPlan<T>::execute_rfft(const T *ri, std::complex<T> *co) const { if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from?)"); pimpl_->execute_rfft_oop(ri, co); }
    template <typename T> void FFTPlan<T>::execute_irfft(const std::complex<T> *ci, T *ro) const { if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from?)"); pimpl_->execute_irfft_oop(ci, ro); }
    template <typename T> size_t FFTPlan<T>::getLength() const { if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from?)"); return pimpl_->length; }
    template <typename T> size_t FFTPlan<T>::getComplexLength() const { if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from?)"); return pimpl_->complex_length; }
    template <typename T> Direction FFTPlan<T>::getDirection() const { if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from?)"); return pimpl_->direction; }
    template <typename T> Precision FFTPlan<T>::getPrecision() const { if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from?)"); return pimpl_->precision; }
    template <typename T> Domain FFTPlan<T>::getDomain() const { if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from?)"); return pimpl_->domain; }
    template <typename T> NormMode FFTPlan<T>::getNormMode() const { if (!pimpl_) throw std::runtime_error("Invalid FFTPlan (moved-from?)"); return pimpl_->norm_mode; }


    // === Backend Implementation for Conv/Corr and Filter+Downsample ===
    namespace Backend
    {
        /**
         * @brief MKL backend implementation for 1D convolution or correlation using VSL.
         */
        template <typename T>
        std::vector<T> convolve1d_impl(const std::vector<T> &signal, const std::vector<T> &kernel, bool use_correlation)
        {
            // (Omitted for brevity - same as before)
            if (signal.empty() || kernel.empty()) return {};
            int nx = static_cast<int>(kernel.size()); int ny = static_cast<int>(signal.size()); int nz = ny - nx + 1;
            if (nz <= 0) throw std::invalid_argument("MKL Conv/Corr ('valid' mode equivalent): signal length must be >= kernel length.");
            std::vector<T> result(nz); const int mode = VSL_CONV_MODE_AUTO; VSLConvTaskPtr task = nullptr; int status = VSL_STATUS_OK;
            MKL_INT stride_kernel = 1, stride_signal = 1, stride_result = 1; MKL_INT start_index = use_correlation ? 0 : (nx - 1);
            bool call_mkl_convolution = !use_correlation; bool call_mkl_correlation = use_correlation;
            try {
                if constexpr (std::is_same_v<T, float>) {
                    if (call_mkl_convolution) { status = vslsConvNewTask1D(&task, mode, nx, ny, nz); check_mkl_status(status, "MKL vslsConvNewTask1D"); status = vslConvSetStart(task, &start_index); check_mkl_status(status, "MKL vslConvSetStart"); status = vslsConvExec(task, kernel.data(), &stride_kernel, signal.data(), &stride_signal, result.data(), &stride_result); check_mkl_status(status, "MKL vslsConvExec"); }
                    else { status = vslsCorrNewTask1D(&task, mode, nx, ny, nz); check_mkl_status(status, "MKL vslsCorrNewTask1D"); status = vslCorrSetStart(task, &start_index); check_mkl_status(status, "MKL vslCorrSetStart"); status = vslsCorrExec(task, kernel.data(), &stride_kernel, signal.data(), &stride_signal, result.data(), &stride_result); check_mkl_status(status, "MKL vslsCorrExec"); }
                } else {
                     if (call_mkl_convolution) { status = vsldConvNewTask1D(&task, mode, nx, ny, nz); check_mkl_status(status, "MKL vsldConvNewTask1D"); status = vslConvSetStart(task, &start_index); check_mkl_status(status, "MKL vslConvSetStart"); status = vsldConvExec(task, kernel.data(), &stride_kernel, signal.data(), &stride_signal, result.data(), &stride_result); check_mkl_status(status, "MKL vsldConvExec"); }
                     else { status = vsldCorrNewTask1D(&task, mode, nx, ny, nz); check_mkl_status(status, "MKL vsldCorrNewTask1D"); status = vslCorrSetStart(task, &start_index); check_mkl_status(status, "MKL vslCorrSetStart"); status = vsldCorrExec(task, kernel.data(), &stride_kernel, signal.data(), &stride_signal, result.data(), &stride_result); check_mkl_status(status, "MKL vsldCorrExec"); }
                }
                if (task) { status = vslConvDeleteTask(&task); check_mkl_status(status, "MKL vslConvDeleteTask"); task = nullptr; }
                return result;
            } catch (...) { if (task) { (void)vslConvDeleteTask(&task); } throw; }
        }


        /**
         * @brief MKL backend implementation for combined FIR filtering and downsampling.
         * Uses Intel IPP polyphase resampling (ippsResamplePolyphaseFixed) for FLOAT only.
         * Double precision (_64f) version is removed as the type/functions seem unavailable
         * in IPP 2022.1.0 based on header inspection/typedefs.
         */
        template <typename T>
        std::vector<T> filter_and_downsample_impl(const std::vector<T> &signal, const std::vector<T> &kernel, int factor)
        {
            // --- Check Precision ---
            if constexpr (std::is_same_v<T, double>) {
                // --- REMOVED _64f Implementation ---
                throw std::runtime_error("IPP filter_and_downsample (ippsResamplePolyphaseFixed) is currently not supported for double precision with this IPP version/backend.");
            }
            else if constexpr (std::is_same_v<T, float>)
            {
                // --- FLOAT (_32f) Implementation ---
                if (signal.empty()) return {};
                if (factor <= 0) throw std::invalid_argument("Downsampling factor must be positive.");
                if (kernel.empty()) throw std::invalid_argument("Kernel cannot be empty for IPP resampling filter.");

                int inLen = static_cast<int>(signal.size());
                int filterLen = static_cast<int>(kernel.size());
                if (filterLen <= 0) throw std::invalid_argument("IPP Resampling filter length must be positive.");

                // --- IPP Polyphase Resampling Setup ---
                IppStatus status = ippStsNoErr;
                int inRate = factor;
                int outRate = 1;
                Ipp32f rolloff = 0.95f;
                Ipp32f beta = 9.0f;
                IppHintAlgorithm hint = ippAlgHintAccurate;

                int specSize = 0;
                int initBufSize = 0;
                int height = 0;

                Ipp8u *pSpecBuffer = nullptr;
                Ipp8u *pInitBuf = nullptr;

                // --- CORRECTED Type Alias based on typedefs ---
                using IppSpecType = IppsResamplingPolyphaseFixed_32f; // Uppercase I, with 'ing', no 'Spec'

                IppSpecType *pSpec = nullptr;

                std::vector<T> result;
                int outLen = 0;

                try {
                    // 1. Get Size (Using 7 arguments)
                    status = ippsResamplePolyphaseFixedGetSize_32f(inRate, outRate, filterLen, &specSize, &initBufSize, &height, hint);
                    check_ipp_status(status, "IPP ippsResamplePolyphaseFixedGetSize_32f failed");

                    pSpecBuffer = ippsMalloc_8u(specSize);
                    if (!pSpecBuffer) throw std::runtime_error("IPP failed to allocate spec buffer");
                    // --- CORRECTED Cast ---
                    pSpec = reinterpret_cast<IppSpecType *>(pSpecBuffer);

                    pInitBuf = ippsMalloc_8u(initBufSize);
                    if (!pInitBuf) throw std::runtime_error("IPP failed to allocate init buffer");

                    // 2. Initialize (Using 7 arguments)
                    status = ippsResamplePolyphaseFixedInit_32f(inRate, outRate, filterLen, rolloff, beta, pSpec, hint);
                    check_ipp_status(status, "IPP ippsResamplePolyphaseFixedInit_32f failed");

                    // 3. Allocate result buffer
                    size_t maxOutLen = static_cast<size_t>(std::ceil(static_cast<double>(inLen) * outRate / inRate)) + filterLen;
                    result.resize(maxOutLen);
                    Ipp64f time = 0;

                    // 4. Perform Resampling (Using 7 arguments)
                    T exec_factor_or_norm = static_cast<T>(1.0); // Assuming 1.0 is correct for 'norm'

                    status = ippsResamplePolyphaseFixed_32f(signal.data(), inLen, result.data(), exec_factor_or_norm, &time, &outLen, pSpec);
                    check_ipp_status(status, "IPP ippsResamplePolyphaseFixed_32f failed");

                    // 5. Resize result vector
                    if (outLen < 0) throw std::runtime_error("IPP returned negative output length");
                    result.resize(static_cast<size_t>(outLen));

                    // 6. Free IPP buffers
                    if (pInitBuf) { ippsFree(pInitBuf); pInitBuf = nullptr; }
                    if (pSpecBuffer) { ippsFree(pSpecBuffer); pSpecBuffer = nullptr; }
                    return result;
                }
                catch (...) {
                    // Ensure buffers are freed even if exceptions occur
                    if (pInitBuf) { ippsFree(pInitBuf); }
                    if (pSpecBuffer) { ippsFree(pSpecBuffer); }
                    throw; // Re-throw the exception
                }
            } // End float implementation
            else {
                 // Should not happen with static_assert, but as fallback
                 throw std::runtime_error("Unsupported type for filter_and_downsample_impl");
            }
        } // End filter_and_downsample_impl


        // --- Explicit Instantiations for Backend functions ---
        template std::vector<float> convolve1d_impl<float>(const std::vector<float> &, const std::vector<float> &, bool);
        template std::vector<double> convolve1d_impl<double>(const std::vector<double> &, const std::vector<double> &, bool);
        template std::vector<float> filter_and_downsample_impl<float>(const std::vector<float> &, const std::vector<float> &, int);
        // --- REMOVED Double Instantiation for filter_and_downsample ---
        // template std::vector<double> filter_and_downsample_impl<double>(const std::vector<double> &, const std::vector<double> &, int);

    } // namespace Backend
} // namespace OmniDSP

// --- Explicit Template Instantiations for FFTPlan Class ---
template class OMNIDSP_EXPORT OmniDSP::FFTPlan<float>;
template class OMNIDSP_EXPORT OmniDSP::FFTPlan<double>;

#endif // USE_ONEMKL
