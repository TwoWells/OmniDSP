/**
 * @file fft.h
 * @brief Public API header for FFT functionality in the OmniDSP library.
 *
 * This header defines the primary interface for performing Fast Fourier
 * Transforms, including the FFTPlan class and convenience functions. It relies
 * on core enums being defined elsewhere (e.g., in omnidsp.h).
 *
 * @version 1.0.1 (Corrected Enum location)
 * @date 2025-04-15
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

// Note: This header assumes the core OmniDSP Enums (Direction, Precision,
// Domain, NormMode)
//       are defined before this header is included, typically by including
//       <OmniDSP/omnidsp.h>.

namespace OmniDSP {
// --- FFTPlan ---

// Forward declaration for the implementation class (Pimpl idiom)
template <typename T>
class FFTPlanImpl;

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
   */
  FFTPlan(size_t length, Precision precision, Direction direction,
          Domain domain, NormMode norm = NormMode::BACKWARD);

  /** @brief Destructor. */
  ~FFTPlan();

  // Move-Only
  FFTPlan(const FFTPlan &) = delete;
  FFTPlan &operator=(const FFTPlan &) = delete;
  FFTPlan(FFTPlan &&) noexcept = default;
  FFTPlan &operator=(FFTPlan &&) noexcept = default;

  // Execution Methods
  void execute(const std::complex<T> *input, std::complex<T> *output) const;
  void execute(std::complex<T> *data) const;
  void execute_rfft(const T *real_input, std::complex<T> *complex_output) const;
  void execute_irfft(const std::complex<T> *complex_input,
                     T *real_output) const;

  // Getters
  size_t getLength() const;
  size_t getComplexLength() const;
  Direction getDirection() const;
  Precision getPrecision() const;
  Domain getDomain() const;
  NormMode getNormMode() const;

 private:
  std::unique_ptr<FFTPlanImpl<T>> pimpl_;
};

// --- Convenience Functions (Declarations) ---

template <typename T>
void fft(const std::vector<std::complex<T>> &input,
         std::vector<std::complex<T>> &output);
template <typename T>
void ifft(const std::vector<std::complex<T>> &input,
          std::vector<std::complex<T>> &output);
template <typename T>
void fft_inplace(std::vector<std::complex<T>> &data);
template <typename T>
void ifft_inplace(std::vector<std::complex<T>> &data);
template <typename T>
void rfft(const std::vector<T> &real_input,
          std::vector<std::complex<T>> &complex_output);
template <typename T>
void irfft(const std::vector<std::complex<T>> &complex_input,
           std::vector<T> &real_output);

// --- Explicit Template Instantiations (Declarations) ---

/** @cond OMNIDSP_INTERNAL */
extern template class OMNIDSP_EXPORT FFTPlan<float>;
extern template class OMNIDSP_EXPORT FFTPlan<double>;

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
