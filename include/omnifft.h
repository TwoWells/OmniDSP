#ifndef OMNIFFT_H
#define OMNIFFT_H

#include <complex>
#include <vector>
#include <memory> // For std::unique_ptr
#include <stdexcept> // For exceptions
#include <cstddef> // For size_t

namespace OmniFFT {

// Enum for transform direction
enum class Direction {
    FORWARD, // C2C: FFT, R2C: Real FFT
    INVERSE  // C2C: IFFT, C2R: Inverse Real FFT (produces real output)
};

// Enum for precision
enum class Precision {
    SINGLE, // float
    DOUBLE  // double
};

// Enum for transform domain
enum class Domain {
    COMPLEX, // Complex-to-Complex (C2C) transform
    REAL     // Real-to-Complex (R2C) / Complex-to-Real (C2R) transform
};

// Forward declaration for the implementation class (Pimpl)
template <typename T>
class FFTPlanImpl;

// --- FFTPlan Class ---
// Manages pre-computed data (plan) for efficient repeated FFTs.
template <typename T>
class FFTPlan {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "FFTPlan only supports float or double precision.");

public:
    /**
     * @brief Constructs an FFT plan.
     * @param length The size 'N' of the transform. For Domain::COMPLEX, this is the number of complex input/output points. For Domain::REAL, this is the number of real input/output points (the corresponding complex spectrum size will be N/2 + 1).
     * @param precision The floating point precision (SINGLE or DOUBLE).
     * @param direction The direction of the transform (FORWARD or INVERSE).
     * @param domain The domain of the transform (COMPLEX or REAL).
     * @throws std::runtime_error If initialization fails (e.g., unsupported size, memory allocation error, backend unavailable).
     * @throws std::invalid_argument If length is zero.
     */
    FFTPlan(size_t length, Precision precision, Direction direction, Domain domain);

    /**
     * @brief Destructor (handles cleanup of backend resources).
     */
    ~FFTPlan();

    // Delete copy constructor and assignment operator
    FFTPlan(const FFTPlan&) = delete;
    FFTPlan& operator=(const FFTPlan&) = delete;

    // Allow move construction and assignment
    FFTPlan(FFTPlan&&) noexcept;
    FFTPlan& operator=(FFTPlan&&) noexcept;

    /**
     * @brief Executes a Complex-to-Complex (C2C) FFT out-of-place.
     * Requires plan to be created with Domain::COMPLEX.
     * Input and output arrays must have size `getLength()`.
     * @param input Pointer to the input complex data array.
     * @param output Pointer to the output complex data array.
     * @throws std::runtime_error If execution fails or plan domain is not COMPLEX.
     * @throws std::invalid_argument If input or output is null.
     */
    void execute(const std::complex<T>* input, std::complex<T>* output) const;

    /**
     * @brief Executes a Complex-to-Complex (C2C) FFT in-place.
     * Requires plan to be created with Domain::COMPLEX.
     * Data array must have size `getLength()`.
     * @param data Pointer to the complex data array (will be overwritten).
     * @throws std::runtime_error If execution fails or plan domain is not COMPLEX.
     * @throws std::invalid_argument If data is null.
     */
    void execute(std::complex<T>* data) const;

    /**
     * @brief Executes a Real-to-Complex (R2C) FFT Forward transform.
     * Requires plan to be created with Domain::REAL and Direction::FORWARD.
     * Real input array must have size `getLength()` (N).
     * Complex output array must have size `getComplexLength()` (N/2 + 1).
     * @param real_input Pointer to the input real data array.
     * @param complex_output Pointer to the output complex spectrum array.
     * @throws std::runtime_error If execution fails or plan domain/direction is incorrect.
     * @throws std::invalid_argument If input or output is null.
     */
    void execute_r2c(const T* real_input, std::complex<T>* complex_output) const;

    /**
     * @brief Executes a Complex-to-Real (C2R) FFT Inverse transform.
     * Requires plan to be created with Domain::REAL and Direction::INVERSE.
     * Complex input array must have size `getComplexLength()` (N/2 + 1) and satisfy Hermitian symmetry.
     * Real output array must have size `getLength()` (N).
     * @param complex_input Pointer to the input complex spectrum array.
     * @param real_output Pointer to the output real data array.
     * @throws std::runtime_error If execution fails or plan domain/direction is incorrect.
     * @throws std::invalid_argument If input or output is null.
     */
    void execute_c2r(const std::complex<T>* complex_input, T* real_output) const;


    /**
     * @brief Gets the length 'N' of the real signal domain associated with the plan.
     * For Domain::COMPLEX, this is also the complex signal length.
     * For Domain::REAL, the corresponding complex spectrum length is N/2 + 1.
     */
    size_t getLength() const;

    /**
     * @brief Gets the length (N/2 + 1) of the complex spectrum for Domain::REAL plans.
     * Returns 0 for Domain::COMPLEX plans.
     */
    size_t getComplexLength() const;

    /**
     * @brief Gets the direction of the FFT plan.
     */
    Direction getDirection() const;

    /**
     * @brief Gets the precision of the FFT plan.
     */
    Precision getPrecision() const;

    /**
     * @brief Gets the domain (COMPLEX or REAL) of the FFT plan.
     */
    Domain getDomain() const;


private:
    std::unique_ptr<FFTPlanImpl<T>> pimpl_; // Pointer to implementation
};

// --- Convenience Functions ---

// C2C Forward (Out-of-place)
template <typename T>
void fft(const std::vector<std::complex<T>>& input, std::vector<std::complex<T>>& output);

// C2C Inverse (Out-of-place)
template <typename T>
void ifft(const std::vector<std::complex<T>>& input, std::vector<std::complex<T>>& output);

// C2C Forward (In-place)
template <typename T>
void fft_inplace(std::vector<std::complex<T>>& data);

// C2C Inverse (In-place)
template <typename T>
void ifft_inplace(std::vector<std::complex<T>>& data);

// R2C Forward (Out-of-place)
template <typename T>
void fft_r2c(const std::vector<T>& real_input, std::vector<std::complex<T>>& complex_output);

// C2R Inverse (Out-of-place)
template <typename T>
void ifft_c2r(const std::vector<std::complex<T>>& complex_input, std::vector<T>& real_output);


// --- Explicit Template Instantiations (Declarations) ---
// Declares that explicit instantiations will be provided elsewhere (in impl files)
extern template class FFTPlan<float>;
extern template class FFTPlan<double>;

// Declarations for convenience functions (instantiated in fft_lib.cpp)
extern template void fft<float>(const std::vector<std::complex<float>>&, std::vector<std::complex<float>>&);
extern template void fft<double>(const std::vector<std::complex<double>>&, std::vector<std::complex<double>>&);
extern template void ifft<float>(const std::vector<std::complex<float>>&, std::vector<std::complex<float>>&);
extern template void ifft<double>(const std::vector<std::complex<double>>&, std::vector<std::complex<double>>&);
extern template void fft_inplace<float>(std::vector<std::complex<float>>&);
extern template void fft_inplace<double>(std::vector<std::complex<double>>&);
extern template void ifft_inplace<float>(std::vector<std::complex<float>>&);
extern template void ifft_inplace<double>(std::vector<std::complex<double>>&);
// New declarations
extern template void fft_r2c<float>(const std::vector<float>&, std::vector<std::complex<float>>&);
extern template void fft_r2c<double>(const std::vector<double>&, std::vector<std::complex<double>>&);
extern template void ifft_c2r<float>(const std::vector<std::complex<float>>&, std::vector<float>&);
extern template void ifft_c2r<double>(const std::vector<std::complex<double>>&, std::vector<double>&);


} // namespace OmniFFT

#endif // OMNIFFOMNIFFT