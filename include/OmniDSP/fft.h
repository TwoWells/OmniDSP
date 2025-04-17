/**
 * @file fft.h
 * @brief Public API header for FFT functionality in the OmniDSP library.
 *
 * This header defines the primary interface for performing Fast Fourier
 * Transforms, including the FFTPlan class and convenience functions.
 *
 * @version 1.0.2 (Added missing omnidsp.h include)
 * @date 2025-04-17
 */

#ifndef OMNIDSP_FFT_H
#define OMNIDSP_FFT_H

// --- Include necessary headers ---
#include <complex>  // For std::complex
#include <cstddef>  // For size_t
#include <memory>   // For std::unique_ptr (Pimpl idiom)
#include <stdexcept>  // For std::runtime_error, std::invalid_argument documentation
#include <type_traits>  // For static_assert, std::is_same_v
#include <vector>       // For std::vector

// Include the generated export header for DLL symbol handling
#include <OmniDSP/omnidsp_export.h>  // Adjust path/name if EXPORT_FILE_NAME was used in CMake

// <<< Added: Include the main header which defines core enums >>>
#include <OmniDSP/omnidsp.h>  // Defines Precision, Direction, Domain, NormMode

namespace OmniDSP {
// --- FFTPlan ---

// Forward declaration for the implementation class (Pimpl idiom)
template <typename T>
struct FFTPlanImpl;  // Use struct for PIMPL forward declaration

/**
 * @brief Manages a pre-calculated plan for efficient FFT execution.
 * (See omnidsp.h or main documentation for Enum definitions)
 * @tparam T The floating-point type (float or double).
 */
template <typename T>
class OMNIDSP_EXPORT FFTPlan {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "FFTPlan only supports float or double precision.");

 public:
  /**
   * @brief Constructs and initializes an FFT plan.
   * @param length The length of the FFT (N).
   * @param precision The precision (SINGLE or DOUBLE).
   * @param direction The transform direction (FORWARD or INVERSE).
   * @param domain The domain (COMPLEX or REAL).
   * @param norm The normalization mode (default BACKWARD).
   * @throws std::invalid_argument If length is zero.
   * @throws std::runtime_error If plan creation fails in the backend.
   */
  FFTPlan(size_t length, Precision precision, Direction direction,
          Domain domain, NormMode norm = NormMode::BACKWARD);

  /** @brief Destructor. */
  ~FFTPlan();

  // Move-Only semantics enforced
  FFTPlan(const FFTPlan &) = delete;
  FFTPlan &operator=(const FFTPlan &) = delete;
  FFTPlan(FFTPlan &&) noexcept = default;  // Default move constructor
  FFTPlan &operator=(FFTPlan &&) noexcept = default;  // Default move assignment

  // --- Execution Methods ---

  /**
   * @brief Executes a complex-to-complex FFT (out-of-place).
   * @param input Pointer to the input complex data (size N).
   * @param output Pointer to the output complex data buffer (size N).
   * @throws std::runtime_error If plan is invalid or execution fails.
   */
  void execute(const std::complex<T> *input, std::complex<T> *output) const;

  /**
   * @brief Executes a complex-to-complex FFT (in-place).
   * @param data Pointer to the complex data buffer (size N), modified in place.
   * @throws std::runtime_error If plan is invalid or execution fails.
   */
  void execute(std::complex<T> *data) const;

  /**
   * @brief Executes a real-to-complex FFT (RFFT, out-of-place).
   * Requires the plan to be created with Domain::REAL and Direction::FORWARD.
   * @param real_input Pointer to the input real data (size N).
   * @param complex_output Pointer to the output complex data buffer (size N/2 +
   * 1).
   * @throws std::runtime_error If plan is invalid, not a REAL/FORWARD plan, or
   * execution fails.
   */
  void execute_rfft(const T *real_input, std::complex<T> *complex_output) const;

  /**
   * @brief Executes a complex-to-real inverse FFT (IRFFT, out-of-place).
   * Requires the plan to be created with Domain::REAL and Direction::INVERSE.
   * @param complex_input Pointer to the input complex data (size N/2 + 1).
   * @param real_output Pointer to the output real data buffer (size N).
   * @throws std::runtime_error If plan is invalid, not a REAL/INVERSE plan, or
   * execution fails.
   */
  void execute_irfft(const std::complex<T> *complex_input,
                     T *real_output) const;

  // --- Getters ---
  // Provide access to the plan's configuration parameters.

  /** @brief Gets the FFT length (N) the plan was configured for. */
  size_t getLength() const;
  /** @brief Gets the length of the complex spectrum output (N for C2C, N/2+1
   * for Real). */
  size_t getComplexLength() const;
  /** @brief Gets the transform direction (FORWARD or INVERSE). */
  Direction getDirection() const;
  /** @brief Gets the precision (SINGLE or DOUBLE). */
  Precision getPrecision() const;
  /** @brief Gets the domain (COMPLEX or REAL). */
  Domain getDomain() const;
  /** @brief Gets the normalization mode used by the plan. */
  NormMode getNormMode() const;

 private:
  // Pointer to the implementation details (Pimpl idiom)
  std::unique_ptr<FFTPlanImpl<T>> pimpl_;
};

// --- Convenience Functions (Declarations) ---
// Provide simpler interfaces for common FFT operations using std::vector.
// These typically create and manage an FFTPlan internally.

/**
 * @brief Performs an out-of-place complex-to-complex forward FFT.
 * Uses default normalization (NormMode::BACKWARD).
 * @tparam T float or double.
 * @param input Input complex vector.
 * @param output Output complex vector (resized appropriately).
 */
template <typename T>
void fft(const std::vector<std::complex<T>> &input,
         std::vector<std::complex<T>> &output);

/**
 * @brief Performs an out-of-place complex-to-complex inverse FFT.
 * Uses default normalization (NormMode::BACKWARD, scales by 1/N).
 * @tparam T float or double.
 * @param input Input complex vector (spectrum).
 * @param output Output complex vector (time-domain, resized appropriately).
 */
template <typename T>
void ifft(const std::vector<std::complex<T>> &input,
          std::vector<std::complex<T>> &output);

/**
 * @brief Performs an in-place complex-to-complex forward FFT.
 * Uses default normalization (NormMode::BACKWARD).
 * @tparam T float or double.
 * @param data Complex vector modified in-place.
 */
template <typename T>
void fft_inplace(std::vector<std::complex<T>> &data);

/**
 * @brief Performs an in-place complex-to-complex inverse FFT.
 * Uses default normalization (NormMode::BACKWARD, scales by 1/N).
 * @tparam T float or double.
 * @param data Complex vector (spectrum) modified in-place to time-domain.
 */
template <typename T>
void ifft_inplace(std::vector<std::complex<T>> &data);

/**
 * @brief Performs an out-of-place real-to-complex forward FFT (RFFT).
 * Uses default normalization (NormMode::BACKWARD). Output size is N/2 + 1.
 * @tparam T float or double.
 * @param real_input Input real vector (size N).
 * @param complex_output Output complex vector (spectrum, resized to N/2 + 1).
 */
template <typename T>
void rfft(const std::vector<T> &real_input,
          std::vector<std::complex<T>> &complex_output);

/**
 * @brief Performs an out-of-place complex-to-real inverse FFT (IRFFT).
 * Uses default normalization (NormMode::BACKWARD). Input size is N/2 + 1.
 * @tparam T float or double.
 * @param complex_input Input complex vector (spectrum, size N/2 + 1).
 * @param real_output Output real vector (time-domain, resized to N).
 */
template <typename T>
void irfft(const std::vector<std::complex<T>> &complex_input,
           std::vector<T> &real_output);

// --- Explicit Template Instantiations (Declarations) ---
// Needed when template definitions are in a separate .cpp file to ensure
// symbols are generated.

/** @cond OMNIDSP_INTERNAL */  // Hide from Doxygen index
// Instantiations for FFTPlan class
extern template class OMNIDSP_EXPORT FFTPlan<float>;
extern template class OMNIDSP_EXPORT FFTPlan<double>;

// Instantiations for convenience functions
extern template OMNIDSP_EXPORT void fft<float>(
    const std::vector<std::complex<float>> &,
    std::vector<std::complex<float>> &);
extern template OMNIDSP_EXPORT void fft<double>(
    const std::vector<std::complex<double>> &,
    std::vector<std::complex<double>> &);
extern template OMNIDSP_EXPORT void ifft<float>(
    const std::vector<std::complex<float>> &,
    std::vector<std::complex<float>> &);
extern template OMNIDSP_EXPORT void ifft<double>(
    const std::vector<std::complex<double>> &,
    std::vector<std::complex<double>> &);
extern template OMNIDSP_EXPORT void fft_inplace<float>(
    std::vector<std::complex<float>> &);
extern template OMNIDSP_EXPORT void fft_inplace<double>(
    std::vector<std::complex<double>> &);
extern template OMNIDSP_EXPORT void ifft_inplace<float>(
    std::vector<std::complex<float>> &);
extern template OMNIDSP_EXPORT void ifft_inplace<double>(
    std::vector<std::complex<double>> &);
extern template OMNIDSP_EXPORT void rfft<float>(
    const std::vector<float> &, std::vector<std::complex<float>> &);
extern template OMNIDSP_EXPORT void rfft<double>(
    const std::vector<double> &, std::vector<std::complex<double>> &);
extern template OMNIDSP_EXPORT void irfft<float>(
    const std::vector<std::complex<float>> &, std::vector<float> &);
extern template OMNIDSP_EXPORT void irfft<double>(
    const std::vector<std::complex<double>> &, std::vector<double> &);
/** @endcond */

}  // namespace OmniDSP

#endif  // OMNIDSP_FFT_H
