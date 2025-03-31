/**
 * @file omnifft.cpp
 * @brief Implementation of convenience functions for the OmniFFT library.
 *
 * This file provides std::vector-based wrapper functions for common FFT operations.
 * These functions simplify common use cases by managing the creation and execution
 * of an underlying FFTPlan object. They currently use the default NormMode::BACKWARD
 * normalization.
 *
 * @version 1.0.0
 * @date 2025-03-31
 */

#include <OmniFFT/omnifft.h> // Main public header
#include <vector>            // For std::vector parameters
#include <complex>           // For std::complex type
#include <stdexcept>         // For potential exceptions from FFTPlan
#include <type_traits>       // For std::is_same_v
#include <cmath>             // For irfft size calculation if needed (currently just arithmetic)

// NOTE: FFTPlan<T> class methods (constructor, destructor, execute*, getters)
//       are defined and instantiated in the conditionally compiled
//       fft_impl_onemkl.cpp, fft_impl_accelerate.cpp, or fft_impl_stub.cpp files.
//       This file only defines and instantiates the convenience functions below.

namespace OmniFFT {

// --- Implementation of Convenience Functions ---

/**
 * @brief Performs a Complex-to-Complex forward FFT (out-of-place) using NormMode::BACKWARD.
 * Creates, executes, and destroys an FFTPlan internally.
 * @tparam T float or double.
 * @param input Input complex vector.
 * @param output Output complex vector (will be resized to match input size).
 * @throws std::runtime_error If backend FFTPlan creation or execution fails.
 */
template <typename T>
void fft(const std::vector<std::complex<T>>& input, std::vector<std::complex<T>>& output) {
    if (input.empty()) {
        output.clear();
        return;
    }
    const size_t N = input.size();
    output.resize(N);
    // Determine precision from template type T
    const Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;

    // Create and execute plan using default NormMode::BACKWARD
    FFTPlan<T> plan(N, prec, Direction::FORWARD, Domain::COMPLEX, NormMode::BACKWARD);
    plan.execute(input.data(), output.data());
}

/**
 * @brief Performs a Complex-to-Complex inverse FFT (out-of-place) using NormMode::BACKWARD.
 * Creates, executes, and destroys an FFTPlan internally. Result is scaled by 1/N if using oneMKL,
 * check backend notes for Accelerate scaling.
 * @tparam T float or double.
 * @param input Input complex vector.
 * @param output Output complex vector (will be resized to match input size).
 * @throws std::runtime_error If backend FFTPlan creation or execution fails.
 */
template <typename T>
void ifft(const std::vector<std::complex<T>>& input, std::vector<std::complex<T>>& output) {
     if (input.empty()) {
        output.clear();
        return;
    }
    const size_t N = input.size();
    output.resize(N);
    const Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;

    // Create and execute plan using default NormMode::BACKWARD
    FFTPlan<T> plan(N, prec, Direction::INVERSE, Domain::COMPLEX, NormMode::BACKWARD);
    plan.execute(input.data(), output.data());
}

/**
 * @brief Performs a Complex-to-Complex forward FFT (in-place) using NormMode::BACKWARD.
 * Creates, executes, and destroys an FFTPlan internally. Input vector `data` is modified.
 * @tparam T float or double.
 * @param data Input/Output complex vector (modified in place).
 * @throws std::runtime_error If backend FFTPlan creation or execution fails.
 */
template <typename T>
void fft_inplace(std::vector<std::complex<T>>& data) {
     if (data.empty()) return;
     const size_t N = data.size();
     const Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;

     // Create and execute plan using default NormMode::BACKWARD
     FFTPlan<T> plan(N, prec, Direction::FORWARD, Domain::COMPLEX, NormMode::BACKWARD);
     plan.execute(data.data());
}

/**
 * @brief Performs a Complex-to-Complex inverse FFT (in-place) using NormMode::BACKWARD.
 * Creates, executes, and destroys an FFTPlan internally. Input vector `data` is modified.
 * Result is scaled by 1/N if using oneMKL, check backend notes for Accelerate scaling.
 * @tparam T float or double.
 * @param data Input/Output complex vector (modified in place).
 * @throws std::runtime_error If backend FFTPlan creation or execution fails.
 */
template <typename T>
void ifft_inplace(std::vector<std::complex<T>>& data) {
     if (data.empty()) return;
     const size_t N = data.size();
     const Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;

     // Create and execute plan using default NormMode::BACKWARD
     FFTPlan<T> plan(N, prec, Direction::INVERSE, Domain::COMPLEX, NormMode::BACKWARD);
     plan.execute(data.data());
}

/**
 * @brief Performs a Real-to-Complex forward FFT (out-of-place) using NormMode::BACKWARD.
 * Creates, executes, and destroys an FFTPlan internally.
 * @tparam T float or double.
 * @param real_input Input real vector of size N.
 * @param complex_output Output complex vector (will be resized to N/2 + 1).
 * @throws std::runtime_error If backend FFTPlan creation or execution fails (e.g., N not power-of-2 for Accelerate REAL).
 */
template <typename T>
void rfft(const std::vector<T>& real_input, std::vector<std::complex<T>>& complex_output) {
    if (real_input.empty()) {
        complex_output.clear();
        return;
    }
    const size_t N = real_input.size();
    const size_t Nc = N / 2 + 1; // Complex output size
    complex_output.resize(Nc);
    const Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;

    // Create and execute plan using default NormMode::BACKWARD
    FFTPlan<T> plan(N, prec, Direction::FORWARD, Domain::REAL, NormMode::BACKWARD);
    plan.execute_rfft(real_input.data(), complex_output.data());
}

/**
 * @brief Performs a Complex-to-Real inverse FFT (out-of-place) using NormMode::BACKWARD.
 * Creates, executes, and destroys an FFTPlan internally. Input complex vector must have
 * Hermitian symmetry for output to be purely real. Result is scaled by 1/N if using oneMKL,
 * check backend notes for Accelerate scaling.
 * @tparam T float or double.
 * @param complex_input Input complex vector of size Nc = N/2 + 1 (Hermitian symmetry assumed).
 * @param real_output Output real vector (will be resized to N = 2*(Nc-1) or 1 if Nc=1).
 * @throws std::runtime_error If backend FFTPlan creation or execution fails.
 */
template <typename T>
void irfft(const std::vector<std::complex<T>>& complex_input, std::vector<T>& real_output) {
    if (complex_input.empty()) {
        real_output.clear();
        return;
    }
    const size_t Nc = complex_input.size();
    if (Nc < 1) { // Should not happen if complex_input wasn't empty, but defensive check
        real_output.clear();
        return;
    }

    // Infer original real length N from complex length Nc = N/2 + 1
    // If Nc = 1 (DC only), N could be 0 or 1. Assume N=1.
    const size_t N = (Nc == 1) ? 1 : 2 * (Nc - 1);
    real_output.resize(N);

    // Handle N=1 case directly (plan creation might fail otherwise or be overkill)
    // Requires NormMode::BACKWARD which has scale=1/N=1.0 for N=1
    if (Nc == 1 && N == 1) {
         real_output[0] = complex_input[0].real(); // Assumes scale is 1.0
         return;
    }

    const Precision prec = std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;

    // Create and execute plan using default NormMode::BACKWARD
    FFTPlan<T> plan(N, prec, Direction::INVERSE, Domain::REAL, NormMode::BACKWARD);
    plan.execute_irfft(complex_input.data(), real_output.data());
}


// --- Explicit Template Instantiations (Definition) ---
// Instantiates the convenience functions for float and double.
// The FFTPlan<T> class itself is instantiated in the fft_impl_*.cpp files.

// C2C Functions
template void fft<float>(const std::vector<std::complex<float>>&, std::vector<std::complex<float>>&);
template void fft<double>(const std::vector<std::complex<double>>&, std::vector<std::complex<double>>&);
template void ifft<float>(const std::vector<std::complex<float>>&, std::vector<std::complex<float>>&);
template void ifft<double>(const std::vector<std::complex<double>>&, std::vector<std::complex<double>>&);
template void fft_inplace<float>(std::vector<std::complex<float>>&);
template void fft_inplace<double>(std::vector<std::complex<double>>&);
template void ifft_inplace<float>(std::vector<std::complex<float>>&);
template void ifft_inplace<double>(std::vector<std::complex<double>>&);

// R2C / C2R Functions
template void rfft<float>(const std::vector<float>&, std::vector<std::complex<float>>&);
template void rfft<double>(const std::vector<double>&, std::vector<std::complex<double>>&);
template void irfft<float>(const std::vector<std::complex<float>>&, std::vector<float>&);
template void irfft<double>(const std::vector<std::complex<double>>&, std::vector<double>&);

} // namespace OmniFFT