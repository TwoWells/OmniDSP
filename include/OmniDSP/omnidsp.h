/**
 * @file omnidsp.h
 * @brief Public API header for the OmniDSP library.
 *
 * This header defines the primary interface for performing Fast Fourier Transforms
 * using the OmniDSP library, which abstracts over different high-performance
 * backends like Intel oneMKL and Apple Accelerate.
 *
 * @version 1.0.0
 * @date 2025-03-31
 */

#ifndef OMNIDSP_H
#define OMNIDSP_H

#include <complex>      // For std::complex
#include <vector>       // For convenience functions using std::vector
#include <memory>       // For std::unique_ptr (Pimpl idiom)
#include <stdexcept>    // For std::runtime_error, std::invalid_argument
#include <cstddef>      // For size_t
#include <type_traits>  // For static_assert, std::is_same_v
#include <cmath>        // For std::cyl_bessel_i

/**
 * @brief Main namespace for the OmniDSP library.
 */
namespace OmniDSP {

/**
 * @brief Specifies the direction of the Fourier Transform.
 */
enum class Direction {
    /** Forward Transform (e.g., time to frequency).
     * - For Domain::COMPLEX: Standard Complex FFT.
     * - For Domain::REAL: Real-to-Complex FFT (RFFT). */
    FORWARD,
    /** Inverse Transform (e.g., frequency to time).
     * - For Domain::COMPLEX: Standard Inverse Complex FFT.
     * - For Domain::REAL: Complex-to-Real Inverse FFT (IRFFT). */
    INVERSE
};

/**
 * @brief Specifies the floating-point precision for calculations.
 */
enum class Precision {
    SINGLE, ///< Use float precision.
    DOUBLE  ///< Use double precision.
};

/**
 * @brief Specifies the domain of the input/output signals.
 */
enum class Domain {
    /** Complex-to-Complex (C2C) transform. Input and output are complex. */
    COMPLEX,
    /** Real-valued transform.
     * - Direction::FORWARD: Real-to-Complex (R2C / RFFT).
     * - Direction::INVERSE: Complex-to-Real (C2R / IRFFT). */
    REAL
};

/**
 * @brief Specifies the normalization/scaling mode applied to the transforms.
 * This controls the placement and value of scaling factors (like 1/N or 1/sqrt(N)).
 */
enum class NormMode {
    /**
     * @brief Forward transform is unscaled, Inverse transform is scaled by 1/N.
     * This is the most common convention in signal processing, ensuring IFFT(FFT(x)) = x.
     * (Default mode).
     */
    BACKWARD,
    /**
     * @brief Forward and Inverse transforms are both scaled by 1/sqrt(N).
     * This makes the transform unitary (preserves L2 norm / energy according to Parseval's theorem).
     * IFFT(FFT(x)) = x.
     */
    ORTHO,
    /**
     * @brief Forward transform is scaled by 1/N, Inverse transform is unscaled.
     * Less common, but available. IFFT(FFT(x)) = x.
     */
    FORWARD
};


/**
 * @brief Provides a suite of window functions commonly used in signal processing.
 *
 * Windowing functions are applied to input data before performing a Fourier Transform
 * to reduce spectral leakage and improve the accuracy of the analysis.
 */
class Window {
public:
    /**
     * @brief Applies the Hann window to the input data.
     *
     * The Hann window is defined as:
     * w[n] = 0.5 - 0.5 * cos(2*pi*n / (N-1)) for 0 <= n <= N-1
     *
     * @tparam T The floating-point type of the data (float or double).
     * @param input A vector of input data.
     * @return A vector of windowed data.
     * @throws std::invalid_argument If the input vector is empty.
     */
    template <typename T>
    static std::vector<T> hann(const std::vector<T>& input) {
        if (input.empty()) {
            throw std::invalid_argument("Input vector cannot be empty.");
        }
        size_t N = input.size();
        std::vector<T> output(N);
        for (size_t n = 0; n < N; ++n) {
            output[n] = input[n] * static_cast<T>(0.5 - 0.5 * cos(2.0 * M_PI * n / (N - 1)));
        }
        return output;
    }

    /**
     * @brief Applies the Hamming window to the input data.
     *
     * The Hamming window is defined as:
     * w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1)) for 0 <= n <= N-1
     *
     * @tparam T The floating-point type of the data (float or double).
     * @param input A vector of input data.
     * @return A vector of windowed data.
     * @throws std::invalid_argument If the input vector is empty.
     */
    template <typename T>
    static std::vector<T> hamming(const std::vector<T>& input) {
        if (input.empty()) {
            throw std::invalid_argument("Input vector cannot be empty.");
        }
        size_t N = input.size();
        std::vector<T> output(N);
        for (size_t n = 0; n < N; ++n) {
            output[n] = input[n] * static_cast<T>(0.54 - 0.46 * cos(2.0 * M_PI * n / (N - 1)));
        }
        return output;
    }

    /**
     * @brief Applies the Kaiser window to the input data.
     *
     * The Kaiser window is defined as:
     * w[n] = I0(beta * sqrt(1 - (2n/(N-1) - 1)^2)) / I0(beta) for 0 <= n <= N-1
     * where I0 is the zeroth-order modified Bessel function of the first kind,
     * and beta is a shape parameter that controls the trade-off
     * between main lobe width and sidelobe level.
     *
     * @tparam T The floating-point type of the data (float or double).
     * @param input A vector of input data.
     * @param beta  The shape parameter of the Kaiser window.
     * A typical value is 8.6
     * @return A vector of windowed data.
     * @throws std::invalid_argument If the input vector is empty.
     */
    template <typename T>
    static std::vector<T> kaiser(const std::vector<T>& input, T beta) {
        if (input.empty()) {
            throw std::invalid_argument("Input vector cannot be empty.");
        }
        size_t N = input.size();
        std::vector<T> output(N);

        for (size_t n = 0; n < N; ++n) {
            T factor = (n * 2.0) / (N - 1) - 1.0;
            T sqrt_term = sqrt(1.0 - factor * factor);
            output[n] = input[n] * std::cyl_bessel_i(0, beta * sqrt_term) / std::cyl_bessel_i(0, beta);
        }
        return output;
    }

    /**
     * @brief Applies the Flat-top window to the input data.
     *
     * The Flat-top window is designed to have a very flat passband
     * and is often used for accurate amplitude measurements.
     * It is defined as:
     * w[n] = a0 - a1*cos(2*pi*n/(N-1)) + a2*cos(4*pi*n/(N-1)) - a3*cos(6*pi*n/(N-1)) + a4*cos(8*pi*n/(N-1))
     * for 0 <= n <= N-1
     * where a0 = 0.21557895, a1 = 0.41663158, a2 = 0.277263158, a3 = 0.083578947, and a4 = 0.006947368
     *
     * @tparam T The floating-point type of the data (float or double).
     * @param input A vector of input data.
     * @return A vector of windowed data.
     * @throws std::invalid_argument If the input vector is empty.
     */
    template <typename T>
    static std::vector<T> flattop(const std::vector<T>& input) {
        if (input.empty()) {
            throw std::invalid_argument("Input vector cannot be empty.");
        }
        size_t N = input.size();
        std::vector<T> output(N);
        constexpr T a0 = static_cast<T>(0.21557895);
        constexpr T a1 = static_cast<T>(0.41663158);
        constexpr T a2 = static_cast<T>(0.277263158);
        constexpr T a3 = static_cast<T>(0.083578947);
        constexpr T a4 = static_cast<T>(0.006947368);

        for (size_t n = 0; n < N; ++n) {
            output[n] = input[n] * static_cast<T>(a0 - a1 * cos(2.0 * M_PI * n / (N - 1)) +
                                       a2 * cos(4.0 * M_PI * n / (N - 1)) -
                                       a3 * cos(6.0 * M_PI * n / (N - 1)) +
                                       a4 * cos(8.0 * M_PI * n / (N - 1)));
        }
        return output;
    }
};


// Forward declaration for the implementation class (Pimpl idiom)
// Do not use this class directly.
template <typename T>
class FFTPlanImpl;

/**
 * @brief Manages a pre-calculated plan for efficient FFT execution.
 *
 * Creating an FFTPlan object involves setup overhead specific to the chosen backend
 * (oneMKL or Accelerate) and the transform parameters (length, domain, etc.).
 * Reusing an existing FFTPlan object for multiple transforms of the same type
 * is generally much faster than creating a new plan for each transform.
 *
 * The template parameter T determines the precision (float or double).
 *
 * This class is move-only (not copyable).
 *
 * @tparam T The floating-point type (float or double).
 */
template <typename T>
class FFTPlan {
    // Ensure T is either float or double at compile time
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "FFTPlan only supports float or double precision.");

public:
    /**
     * @brief Constructs and initializes an FFT plan for specified parameters.
     * @param length The size 'N' of the transform.
     * - For Domain::COMPLEX: The number of complex input/output points.
     * - For Domain::REAL: The number of real input/output points (the corresponding complex spectrum size will be N/2 + 1).
     * - Note for Accelerate REAL domain: N must currently be a power of 2.
     * @param precision The floating point precision (Precision::SINGLE or Precision::DOUBLE). This must match the template parameter T.
     * @param direction The direction of the transform (Direction::FORWARD or Direction::INVERSE).
     * @param domain The domain of the transform (Domain::COMPLEX or Domain::REAL).
     * @param norm The normalization mode (NormMode::BACKWARD, NormMode::ORTHO, NormMode::FORWARD). Defaults to BACKWARD.
     * @throws std::runtime_error If backend initialization fails (e.g., unsupported parameters, memory allocation error, backend library unavailable or not found during build).
     * @throws std::invalid_argument If length is zero.
     */
    FFTPlan(size_t length, Precision precision, Direction direction, Domain domain,
            NormMode norm = NormMode::BACKWARD);

    /**
     * @brief Destructor. Releases any backend resources associated with the plan.
     */
    ~FFTPlan();

    // --- Rule of Five: Move-Only ---
    /** @brief Deleted copy constructor. FFTPlan is not copyable. */
    FFTPlan(const FFTPlan&) = delete;
    /** @brief Deleted copy assignment operator. FFTPlan is not copyable. */
    FFTPlan& operator=(const FFTPlan&) = delete;
    /** @brief Move constructor. Transfers ownership of the plan resources. */
    FFTPlan(FFTPlan&&) noexcept;
    /** @brief Move assignment operator. Transfers ownership of the plan resources. */
    FFTPlan& operator=(FFTPlan&&) noexcept;

    // --- Execution Methods ---

    /**
     * @brief Executes a Complex-to-Complex (C2C) FFT out-of-place.
     * Requires the plan to have been created with Domain::COMPLEX.
     * Input and output arrays must point to valid memory buffers each capable of holding `getLength()` complex elements.
     * The specific scaling applied depends on the `NormMode` the plan was created with.
     *
     * @param input Pointer to the input complex data array [size N].
     * @param output Pointer to the output complex data array [size N].
     * @throws std::runtime_error If execution fails or the plan's domain is not Domain::COMPLEX.
     * @throws std::invalid_argument If input or output pointers are null.
     */
    void execute(const std::complex<T>* input, std::complex<T>* output) const;

    /**
     * @brief Executes a Complex-to-Complex (C2C) FFT in-place.
     * Requires the plan to have been created with Domain::COMPLEX.
     * The data array must point to a valid memory buffer capable of holding `getLength()` complex elements.
     * The input data in the buffer will be overwritten with the transform result.
     * The specific scaling applied depends on the `NormMode` the plan was created with.
     * Note: Backend support for in-place operations might vary or involve internal copies.
     *
     * @param data Pointer to the complex data array [size N] (input and output).
     * @throws std::runtime_error If execution fails or the plan's domain is not Domain::COMPLEX.
     * @throws std::invalid_argument If the data pointer is null.
     */
    void execute(std::complex<T>* data) const;

    /**
     * @brief Executes a Real-to-Complex (RFFT) Forward transform out-of-place.
     * Requires the plan to have been created with Domain::REAL and Direction::FORWARD.
     * The real input array must have size `getLength()` (N).
     * The complex output array must have size `getComplexLength()` (N/2 + 1).
     * The specific scaling applied depends on the `NormMode` the plan was created with.
     *
     * @param real_input Pointer to the input real data array [size N].
     * @param complex_output Pointer to the output complex spectrum array [size N/2 + 1].
     * @throws std::runtime_error If execution fails or the plan's domain/direction is not Domain::REAL/Direction::FORWARD.
     * @throws std::invalid_argument If input or output pointers are null.
     */
    void execute_rfft(const T* real_input, std::complex<T>* complex_output) const;

    /**
     * @brief Executes an Inverse Real FFT (Complex-to-Real - IRFFT) transform out-of-place.
     * Requires the plan to have been created with Domain::REAL and Direction::INVERSE.
     * The complex input array must have size `getComplexLength()` (N/2 + 1).
     * The real output array must have size `getLength()` (N).
     * IMPORTANT: For the output to be purely real, the input complex spectrum must possess
     * Hermitian symmetry (i.e., `input[k] == conj(input[N-k])` considering wrap-around,
     * and real DC/Nyquist components). This is naturally satisfied if the input is the
     * result of a previous RFFT (`execute_rfft`) on a real signal. Behavior is undefined
     * if the input does not meet this condition.
     * The specific scaling applied depends on the `NormMode` the plan was created with.
     *
     * @param complex_input Pointer to the input complex spectrum array [size N/2 + 1] (Hermitian symmetry assumed).
     * @param real_output Pointer to the output real data array [size N].
     * @throws std::runtime_error If execution fails or the plan's domain/direction is not Domain::REAL/Direction::INVERSE.
     * @throws std::invalid_argument If input or output pointers are null.
     */
    void execute_irfft(const std::complex<T>* complex_input, T* real_output) const;


    // --- Getters ---

    /**
     * @brief Gets the length 'N' associated with the plan.
     * For Domain::COMPLEX plans, this is the number of complex points.
     * For Domain::REAL plans, this is the number of real points.
     * @return size_t The length N.
     */
    size_t getLength() const;

    /**
     * @brief Gets the length (Nc = N/2 + 1) of the complex spectrum for Domain::REAL plans.
     * Returns 0 for Domain::COMPLEX plans where this concept doesn't apply directly.
     * @return size_t The complex length Nc, or 0.
     */
    size_t getComplexLength() const;

    /**
     * @brief Gets the transform direction configured for this plan.
     * @return Direction (FORWARD or INVERSE).
     */
    Direction getDirection() const;

    /**
     * @brief Gets the floating-point precision configured for this plan.
     * @return Precision (SINGLE or DOUBLE).
     */
    Precision getPrecision() const;

    /**
     * @brief Gets the transform domain configured for this plan.
     * @return Domain (COMPLEX or REAL).
     */
    Domain getDomain() const;

    /**
     * @brief Gets the normalization mode configured for this plan.
     * @return NormMode (BACKWARD, ORTHO, or FORWARD).
     */
    NormMode getNormMode() const;

private:
    /** @brief Pointer to the internal, backend-specific implementation. */
    std::unique_ptr<FFTPlanImpl<T>> pimpl_;
};

// --- Convenience Functions ---
// These functions provide a simpler interface using std::vector for common cases.
// They internally create, use, and destroy an FFTPlan object.
// All convenience functions currently use the default NormMode::BACKWARD.

/** @brief Performs C2C forward FFT (out-of-place) using NormMode::BACKWARD. */
template <typename T> void fft(const std::vector<std::complex<T>>& input, std::vector<std::complex<T>>& output);
/** @brief Performs C2C inverse FFT (out-of-place) using NormMode::BACKWARD. */
template <typename T> void ifft(const std::vector<std::complex<T>>& input, std::vector<std::complex<T>>& output);
/** @brief Performs C2C forward FFT (in-place) using NormMode::BACKWARD. */
template <typename T> void fft_inplace(std::vector<std::complex<T>>& data);
/** @brief Performs C2C inverse FFT (in-place) using NormMode::BACKWARD. */
template <typename T> void ifft_inplace(std::vector<std::complex<T>>& data);
/** @brief Performs R2C forward FFT (out-of-place) using NormMode::BACKWARD. */
template <typename T> void rfft(const std::vector<T>& real_input, std::vector<std::complex<T>>& complex_output);
/** @brief Performs C2R inverse FFT (out-of-place) using NormMode::BACKWARD. Input must have Hermitian symmetry. */
template <typename T> void irfft(const std::vector<std::complex<T>>& complex_input, std::vector<T>& real_output);


// --- Explicit Template Instantiations (Declarations) ---
// These tell the compiler that the full definition (instantiation) of these templates
// will be provided in exactly one implementation file (linked later), preventing
// duplicate symbols and potentially speeding up compilation in user code.

/** @cond OMNIDSP_INTERNAL */ // Hide implementation detail from Doxygen index
extern template class FFTPlan<float>;
extern template class FFTPlan<double>;

extern template void fft<float>(const std::vector<std::complex<float>>&, std::vector<std::complex<float>>&);
extern template void fft<double>(const std::vector<std::complex<double>>&, std::vector<std::complex<double>>&);
extern template void ifft<float>(const std::vector<std::complex<float>>&, std::vector<std::complex<float>>&);
extern template void ifft<double>(const std::vector<std::complex<double>>&, std::vector<std::complex<double>>&);
extern template void fft_inplace<float>(std::vector<std::complex<float>>&);
extern template void fft_inplace<double>(std::vector<std::complex<double>>&);
extern template void ifft_inplace<float>(std::vector<std::complex<float>>&);
extern template void ifft_inplace<double>(std::vector<std::complex<double>>&);
extern template void rfft<float>(const std::vector<float>&, std::vector<std::complex<float>>&);
extern template void rfft<double>(const std::vector<double>&, std::vector<std::complex<double>>&);
extern template void irfft<float>(const std::vector<std::complex<float>>&, std::vector<float>&);
extern template void irfft<double>(const std::vector<std::complex<double>>&, std::vector<double>&);
/** @endcond */

} // namespace OmniDSP

#endif // OMNIDSP_H