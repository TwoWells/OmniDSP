/**
 * @file src/omnidsp/backend/onemkl/window.cpp
 * @brief Intel oneMKL/IPP backend implementation for OmniDSP window functions.
 *
 * Implements backend functions to generate window coefficients using IPP
 * where available (Hann, Hamming, Kaiser). Flattop uses a manual calculation.
 * Corrected Hann/Hamming/Kaiser calls to use correct arguments based on ipps.h.
 * Compiled only when USE_ONEMKL is defined.
 */

// --- Includes ---
#include <vector>
#include <cmath>       // For M_PI, cos, sin, sqrt, std::abs, std::cyl_bessel_i
#include <stdexcept>   // For std::runtime_error, std::invalid_argument
#include <string>
#include <numeric>     // For std::accumulate
#include <limits>      // For std::numeric_limits
#include <type_traits> // For std::is_same_v

#include "../backend_impl.h" // Internal backend function declarations

// Only compile if USE_ONEMKL is defined by CMake
#if defined(USE_ONEMKL)

#include <ipp.h>
#include <ipps.h> // Header containing ippsWin* functions

// Define M_PI if it's not already defined (e.g., by <cmath> in some environments)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace OmniDSP
{
namespace Backend
{

    // --- Intel IPP Helper ---
    // Checks the status code returned by IPP functions and throws an error if it's not ippStsNoErr.
    inline void check_ipp_status(IppStatus status, const char *error_msg)
    {
        if (status != ippStsNoErr)
        {
            std::string full_msg = error_msg;
            // Append IPP status code for detailed diagnostics.
            full_msg += ": IPP Status " + std::to_string(status) + " (" + ippGetStatusString(status) + ")";
            throw std::runtime_error(full_msg);
        }
    }

    // --- Backend Window Generation Implementations ---

    /**
     * @brief Generates Hann window coefficients using IPP.
     * Creates a source vector of 1s and applies the window to it.
     */
    template <typename T>
    std::vector<T> generate_hann_window_impl(size_t length) {
        if (length == 0) return {};
        std::vector<T> coeffs(length);
        std::vector<T> src(length, static_cast<T>(1.0)); // Source vector of 1s
        IppStatus status = ippStsNoErr;
        int len_int = static_cast<int>(length); // IPP uses int for length

        if constexpr (std::is_same_v<T, float>) {
            // Call ippsWinHann_32f(pSrc, pDst, len) - Takes 3 args
            status = ippsWinHann_32f(src.data(), coeffs.data(), len_int);
            check_ipp_status(status, "ippsWinHann_32f failed");
        } else { // double
            // Call ippsWinHann_64f(pSrc, pDst, len) - Takes 3 args
            status = ippsWinHann_64f(src.data(), coeffs.data(), len_int);
            check_ipp_status(status, "ippsWinHann_64f failed");
        }
        return coeffs;
    }

    /**
     * @brief Generates Hamming window coefficients using IPP.
     * Creates a source vector of 1s and applies the window to it.
     */
    template <typename T>
    std::vector<T> generate_hamming_window_impl(size_t length) {
         if (length == 0) return {};
        std::vector<T> coeffs(length);
        std::vector<T> src(length, static_cast<T>(1.0)); // Source vector of 1s
        IppStatus status = ippStsNoErr;
        int len_int = static_cast<int>(length);

        if constexpr (std::is_same_v<T, float>) {
            // Call ippsWinHamming_32f(pSrc, pDst, len) - Takes 3 args
            status = ippsWinHamming_32f(src.data(), coeffs.data(), len_int);
            check_ipp_status(status, "ippsWinHamming_32f failed");
        } else { // double
            // Call ippsWinHamming_64f(pSrc, pDst, len) - Takes 3 args
            status = ippsWinHamming_64f(src.data(), coeffs.data(), len_int);
            check_ipp_status(status, "ippsWinHamming_64f failed");
        }
        return coeffs;
    }

    /**
     * @brief Generates Kaiser window coefficients using IPP.
     * Creates a source vector of 1s and applies the window to it.
     * @param beta The Kaiser window beta parameter.
     */
    template <typename T>
    std::vector<T> generate_kaiser_window_impl(size_t length, T beta) {
        if (length == 0) return {};
        std::vector<T> coeffs(length);
        std::vector<T> src(length, static_cast<T>(1.0)); // Source vector of 1s
        IppStatus status = ippStsNoErr;
        int len_int = static_cast<int>(length);
        // IPP uses alpha parameter, where alpha = beta / pi
        T alpha = beta / static_cast<T>(M_PI);

        if constexpr (std::is_same_v<T, float>) {
            // Call ippsWinKaiser_32f(pSrc, pDst, len, alpha) - Takes 4 args
            status = ippsWinKaiser_32f(src.data(), coeffs.data(), len_int, alpha);
            check_ipp_status(status, "ippsWinKaiser_32f failed");
        } else { // double
            // Call ippsWinKaiser_64f(pSrc, pDst, len, alpha) - Takes 4 args
            status = ippsWinKaiser_64f(src.data(), coeffs.data(), len_int, alpha);
            check_ipp_status(status, "ippsWinKaiser_64f failed");
        }
        return coeffs;
    }

    /**
     * @brief Generates Flattop window coefficients (manual calculation, as IPP lacks native support).
     */
    template <typename T>
    std::vector<T> generate_flattop_window_impl(size_t length) {
         if (length == 0) return {};
        std::vector<T> coeffs(length);
        // Coefficients from standard definition (e.g., SciPy)
        constexpr T a0 = static_cast<T>(0.21557895);
        constexpr T a1 = static_cast<T>(0.41663158);
        constexpr T a2 = static_cast<T>(0.277263158);
        constexpr T a3 = static_cast<T>(0.083578947);
        constexpr T a4 = static_cast<T>(0.006947368);
        // Denominator for cosine terms
        double denom = (length > 1) ? static_cast<double>(length - 1) : 1.0;

        for (size_t n = 0; n < length; ++n) {
            T cos_2pi_term = static_cast<T>(cos(2.0 * M_PI * n / denom));
            T cos_4pi_term = static_cast<T>(cos(4.0 * M_PI * n / denom));
            T cos_6pi_term = static_cast<T>(cos(6.0 * M_PI * n / denom));
            T cos_8pi_term = static_cast<T>(cos(8.0 * M_PI * n / denom));
            coeffs[n] = a0 - a1 * cos_2pi_term + a2 * cos_4pi_term - a3 * cos_6pi_term + a4 * cos_8pi_term;
        }
        return coeffs;
    }


    // --- Explicit Template Instantiations ---
    template std::vector<float> generate_hann_window_impl<float>(size_t length);
    template std::vector<double> generate_hann_window_impl<double>(size_t length);

    template std::vector<float> generate_hamming_window_impl<float>(size_t length);
    template std::vector<double> generate_hamming_window_impl<double>(size_t length);

    template std::vector<float> generate_kaiser_window_impl<float>(size_t length, float beta);
    template std::vector<double> generate_kaiser_window_impl<double>(size_t length, double beta);

    template std::vector<float> generate_flattop_window_impl<float>(size_t length);
    template std::vector<double> generate_flattop_window_impl<double>(size_t length);


} // namespace Backend
} // namespace OmniDSP

#endif // USE_ONEMKL
