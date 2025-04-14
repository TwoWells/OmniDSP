/**
 * @file stub.cpp // Renamed for clarity
 * @brief Stub (Error) backend implementation for OmniDSP.
 *
 * Provides stub implementations for FFTPlanImpl and backend conv/corr/filter+downsample
 * functions. Compiled only when no real backend (oneMKL or Accelerate) is
 * selected. Any attempt to use OmniDSP functionality will result in a
 * std::runtime_error being thrown.
 */

// --- Includes ---
#include <OmniDSP/omnidsp.h> // Public API header
#include <vector>
#include <complex>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <cmath> // For dummy scale calculation in stub FFTPlanImpl

// --- Conditionally Compile Backend Implementation ---

// Compile this only if NEITHER Accelerate nor MKL is defined by CMake
#if !defined(USE_ACCELERATE) && !defined(USE_ONEMKL)

// Internal backend header include (for function declarations)
#include "backend_impl.h"

namespace OmniDSP
{

    // --- FFTPlanImpl Definition (Stub) ---
    template <typename T>
    struct FFTPlanImpl
    {
        // --- Members (Stored but not functionally used) ---
        size_t length = 0;
        size_t complex_length = 0;
        Direction direction = Direction::FORWARD;
        Precision precision = Precision::SINGLE;
        Domain domain = Domain::COMPLEX;
        NormMode norm_mode = NormMode::BACKWARD;
        T forward_scale = 1.0;
        T backward_scale = 1.0;

        // --- Constructor (Throws Error) ---
        FFTPlanImpl(size_t len, Precision prec, Direction dir, Domain dom, NormMode norm)
        {
            std::string error_msg = "OmniDSP backend not selected/available during build. ";
            error_msg += "Cannot create FFTPlan.";
            throw std::runtime_error(error_msg);

            // Unreachable, but initialize to silence potential warnings
            length = len;
            complex_length = (dom == Domain::REAL) ? len / 2 + 1 : len;
            // ... (dummy scale calculations) ...
        }

        // --- Destructor ---
        ~FFTPlanImpl() = default;

        // --- Rule of 5/3: Move Semantics ---
        // Move constructor and assignment are defaulted in omnidsp.h
        // Definitions are REMOVED from here to avoid C2995 error.
        FFTPlanImpl(const FFTPlanImpl &) = delete;
        FFTPlanImpl &operator=(const FFTPlanImpl &) = delete;

        // --- Execute Methods (Throw Error) ---
        void execute_c2c_oop(const std::complex<T> *, std::complex<T> *) const
        {
            throw std::runtime_error("OmniDSP backend not available (stub execute_c2c_oop called).");
        }
        void execute_c2c_ip(std::complex<T> *) const
        {
            throw std::runtime_error("OmniDSP backend not available (stub execute_c2c_ip called).");
        }
        void execute_rfft_oop(const T *, std::complex<T> *) const
        {
            throw std::runtime_error("OmniDSP backend not available (stub execute_rfft_oop called).");
        }
        void execute_irfft_oop(const std::complex<T> *, T *) const
        {
            throw std::runtime_error("OmniDSP backend not available (stub execute_irfft_oop called).");
        }
    };
    // Explicit Instantiations for FFTPlanImpl stub
    template struct FFTPlanImpl<float>;
    template struct FFTPlanImpl<double>;

    // --- FFTPlan Method Definitions (Stub) ---
    // These definitions use the throwing FFTPlanImpl constructor/methods
    template <typename T>
    FFTPlan<T>::FFTPlan(size_t l, Precision p, Direction d, Domain dom, NormMode n)
        : pimpl_(std::make_unique<FFTPlanImpl<T>>(l, p, d, dom, n)) {} // This line throws

    template <typename T>
    FFTPlan<T>::~FFTPlan() = default;

    // Move constructor/assignment are defaulted in the header (omnidsp.h)

    // Execute methods will likely never be reached if constructor throws, but define anyway
    template <typename T>
    void FFTPlan<T>::execute(const std::complex<T> *i, std::complex<T> *o) const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan (stub)");
        pimpl_->execute_c2c_oop(i, o);
    }
    template <typename T>
    void FFTPlan<T>::execute(std::complex<T> *d) const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan (stub)");
        pimpl_->execute_c2c_ip(d);
    }
    template <typename T>
    void FFTPlan<T>::execute_rfft(const T *ri, std::complex<T> *co) const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan (stub)");
        pimpl_->execute_rfft_oop(ri, co);
    }
    template <typename T>
    void FFTPlan<T>::execute_irfft(const std::complex<T> *ci, T *ro) const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan (stub)");
        pimpl_->execute_irfft_oop(ci, ro);
    }
    // Getters might be called on a moved-from object if not careful, add checks
    template <typename T>
    size_t FFTPlan<T>::getLength() const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan (stub)");
        return pimpl_->length;
    }
    template <typename T>
    size_t FFTPlan<T>::getComplexLength() const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan (stub)");
        return pimpl_->complex_length;
    }
    template <typename T>
    Direction FFTPlan<T>::getDirection() const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan (stub)");
        return pimpl_->direction;
    }
    template <typename T>
    Precision FFTPlan<T>::getPrecision() const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan (stub)");
        return pimpl_->precision;
    }
    template <typename T>
    Domain FFTPlan<T>::getDomain() const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan (stub)");
        return pimpl_->domain;
    }
    template <typename T>
    NormMode FFTPlan<T>::getNormMode() const
    {
        if (!pimpl_)
            throw std::runtime_error("Invalid FFTPlan (stub)");
        return pimpl_->norm_mode;
    }

    // Explicit Instantiations for FFTPlan class (already done in omnidsp.cpp)
    // template class FFTPlan<float>; // Defined in omnidsp.cpp
    // template class FFTPlan<double>; // Defined in omnidsp.cpp

    // --- Backend Conv/Corr/Filter+Downsample Implementation (Stub) ---
    namespace Backend
    {

        template <typename T>
        std::vector<T> convolve1d_impl(const std::vector<T> &signal, const std::vector<T> &kernel, bool use_correlation)
        {
            throw std::runtime_error("OmniDSP backend not available (stub convolve1d_impl called).");
            // Add return statement to satisfy compiler, although unreachable
            return {};
        }

        template <typename T>
        std::vector<T> filter_and_downsample_impl(const std::vector<T> &signal, const std::vector<T> &kernel, int factor)
        {
            throw std::runtime_error("OmniDSP backend not available (stub filter_and_downsample_impl called).");
            return {};
        }

        // Explicit Instantiations for FFTPlan class (needed when definition is here)
        template class OMNIDSP_EXPORT OmniDSP::FFTPlan<float>;
        template class OMNIDSP_EXPORT OmniDSP::FFTPlan<double>;

        // --- Explicit Instantiations for Backend Stubs ---
        template std::vector<float> convolve1d_impl<float>(const std::vector<float> &, const std::vector<float> &, bool);
        template std::vector<double> convolve1d_impl<double>(const std::vector<double> &, const std::vector<double> &, bool);

        template std::vector<float> filter_and_downsample_impl<float>(const std::vector<float> &, const std::vector<float> &, int);
        template std::vector<double> filter_and_downsample_impl<double>(const std::vector<double> &, const std::vector<double> &, int);

    } // namespace Backend
} // namespace OmniDSP

#endif // !USE_ACCELERATE && !USE_ONEMKL