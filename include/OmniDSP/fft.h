/**
 * @file fft.h
 * @brief Public API header for FFT and related transforms (IFFT, RFFT, IRFFT).
 * Defines plan classes, convenience functions, and related enums.
 */
#ifndef OMNIDSP_FFT_H
#define OMNIDSP_FFT_H

#include <complex>
#include <cstddef>    // For size_t
#include <memory>     // For std::unique_ptr (used in PIMPL)
#include <stdexcept>  // For std::invalid_argument in docs
#include <vector>

#include "omnidsp.h"  // Basic types (like OMNIDSP_STATUS) and namespace definition

namespace OmniDSP {

// --- FFT-Specific Enums ---

/** @brief Specifies the direction of the Fourier Transform. */
enum class Direction { FORWARD, INVERSE };

/** @brief Specifies the domain of the input/output signals for FFT. */
enum class Domain { COMPLEX, REAL };

/**
 * @brief Specifies the normalization/scaling mode applied to FFTs.
 */
enum class FFTNorm {
  Backward,  ///< No scaling on forward FFT, 1/N scaling on inverse FFT.
  Forward,   ///< 1/N scaling on forward FFT, no scaling on inverse FFT.
  Ortho      ///< 1/sqrt(N) scaling on both forward and inverse FFT.
};

// --- Forward declaration of implementation classes (PIMPL pattern) ---
template <typename T>
class FFTPlanImpl;
template <typename T>
class RFFTPlanImpl;

/**
 * @brief Plan class for Complex-to-Complex FFT and IFFT.
 *
 * Pre-calculates factors and allocates resources for efficient FFT computation
 * on signals of a fixed size. Uses the PIMPL idiom to hide backend details.
 *
 * @tparam T Floating-point type (float or double).
 */
template <typename T>
class FFTPlan {
 public:
  /**
   * @brief Constructs an FFT plan for a given signal size.
   * @param size The length N of the complex signal (number of complex samples).
   * Must be > 0.
   * @throws std::runtime_error If plan creation fails in the backend.
   * @throws std::invalid_argument If size is 0.
   */
  explicit FFTPlan(size_t size);

  /** @brief Destructor (handles implementation object cleanup). */
  ~FFTPlan();

  // Disable copy operations
  FFTPlan(const FFTPlan&) = delete;
  FFTPlan& operator=(const FFTPlan&) = delete;
  // Enable move operations (needed for unique_ptr member)
  FFTPlan(FFTPlan&&) noexcept;
  FFTPlan& operator=(FFTPlan&&) noexcept;

  /**
   * @brief Computes the forward Fast Fourier Transform (FFT).
   * Transforms a complex signal from the time domain to the frequency domain.
   * @param input Complex input vector (size must match plan size).
   * @param output Complex output vector (size must match plan size).
   * @param norm Normalization mode to apply. Defaults to Forward.
   * @throws std::runtime_error If execution fails in the backend or plan is
   * invalid.
   * @throws std::invalid_argument If input/output sizes don't match plan size.
   */
  void fft(const std::vector<std::complex<T>>& input,
           std::vector<std::complex<T>>& output,
           FFTNorm norm = FFTNorm::Forward);

  /**
   * @brief Computes the inverse Fast Fourier Transform (IFFT).
   * Transforms a complex signal from the frequency domain to the time domain.
   * @param input Complex input vector (size must match plan size).
   * @param output Complex output vector (size must match plan size).
   * @param norm Normalization mode to apply. Defaults to Backward.
   * @throws std::runtime_error If execution fails in the backend or plan is
   * invalid.
   * @throws std::invalid_argument If input/output sizes don't match plan size.
   */
  void ifft(const std::vector<std::complex<T>>& input,
            std::vector<std::complex<T>>& output,
            FFTNorm norm = FFTNorm::Backward);

  /**
   * @brief Gets the size (FFT length N) this plan was created for.
   * @return size_t The size of the transform (number of complex samples).
   */
  size_t getSize() const;

 private:
  size_t size_;  // Stores the size N the plan was created for
  std::unique_ptr<FFTPlanImpl<T>> pimpl_;  // Pointer to implementation
};

/**
 * @brief Plan class for Real-to-Complex (RFFT) and Complex-to-Real (IRFFT).
 *
 * Pre-calculates factors and allocates resources for efficient RFFT/IRFFT
 * computation on real signals of a fixed size N. The complex output/input
 * size is N/2 + 1. Uses the PIMPL idiom.
 *
 * @tparam T Floating-point type (float or double).
 */
template <typename T>
class RFFTPlan {
 public:
  /**
   * @brief Constructs an RFFT/IRFFT plan for a given REAL signal size.
   * @param size The length N of the real signal (number of real samples). Must
   * be > 0. The corresponding complex spectrum size will be N/2 + 1.
   * @throws std::runtime_error If plan creation fails in the backend.
   * @throws std::invalid_argument If size is 0.
   */
  explicit RFFTPlan(size_t size);

  /** @brief Destructor (handles implementation object cleanup). */
  ~RFFTPlan();

  // Disable copy operations
  RFFTPlan(const RFFTPlan&) = delete;
  RFFTPlan& operator=(const RFFTPlan&) = delete;
  // Enable move operations
  RFFTPlan(RFFTPlan&&) noexcept;
  RFFTPlan& operator=(RFFTPlan&&) noexcept;

  /**
   * @brief Computes the forward Real-to-Complex FFT (RFFT).
   * Transforms a real signal from the time domain to the complex frequency
   * domain. Output contains only the non-redundant part of the spectrum (size
   * N/2 + 1).
   * @param input Real input vector (size N must match plan size).
   * @param output Complex output vector (size must be N/2 + 1).
   * @param norm Normalization mode to apply. Defaults to Forward.
   * @throws std::runtime_error If execution fails in the backend or plan is
   * invalid.
   * @throws std::invalid_argument If input/output sizes are incorrect.
   */
  void rfft(const std::vector<T>& input, std::vector<std::complex<T>>& output,
            FFTNorm norm = FFTNorm::Forward);

  /**
   * @brief Computes the inverse Complex-to-Real FFT (IRFFT).
   * Transforms a complex frequency spectrum (non-redundant part, size N/2 + 1)
   * to a real time domain signal (size N).
   * @param input Complex input vector (size must be N/2 + 1, where N is the
   * plan size).
   * @param output Real output vector (size must be N, the plan size).
   * @param norm Normalization mode to apply. Defaults to Backward.
   * @throws std::runtime_error If execution fails in the backend or plan is
   * invalid.
   * @throws std::invalid_argument If input/output sizes are incorrect.
   */
  void irfft(const std::vector<std::complex<T>>& input, std::vector<T>& output,
             FFTNorm norm = FFTNorm::Backward);

  /**
   * @brief Gets the real signal size N this plan was created for.
   * @return size_t The size N of the real transform.
   */
  size_t getSize() const;

 private:
  size_t size_;  // Stores the real size N the plan was created for
  std::unique_ptr<RFFTPlanImpl<T>> pimpl_;  // Pointer to implementation
};

// --- Convenience Functions (Stateless) ---

/**
 * @brief Computes FFT using a temporary plan (less efficient for repeated
 * calls). Uses Forward normalization by default. Output vector is resized if
 * necessary.
 * @param input Complex input vector.
 * @param output Complex output vector.
 * @param norm Normalization mode.
 * @throws std::runtime_error If temporary plan creation or execution fails.
 * @throws std::invalid_argument If input is empty.
 */
template <typename T>
void fft(const std::vector<std::complex<T>>& input,
         std::vector<std::complex<T>>& output, FFTNorm norm = FFTNorm::Forward);

/**
 * @brief Computes IFFT using a temporary plan.
 * Uses Backward normalization by default. Output vector is resized if
 * necessary.
 * @param input Complex input vector.
 * @param output Complex output vector.
 * @param norm Normalization mode.
 * @throws std::runtime_error If temporary plan creation or execution fails.
 * @throws std::invalid_argument If input is empty.
 */
template <typename T>
void ifft(const std::vector<std::complex<T>>& input,
          std::vector<std::complex<T>>& output,
          FFTNorm norm = FFTNorm::Backward);

/**
 * @brief Computes RFFT using a temporary plan.
 * Uses Forward normalization by default. Output size must be input.size()/2
 * + 1. Output vector is resized if necessary.
 * @param input Real input vector.
 * @param output Complex output vector.
 * @param norm Normalization mode.
 * @throws std::runtime_error If temporary plan creation or execution fails.
 * @throws std::invalid_argument If input is empty.
 */
template <typename T>
void rfft(const std::vector<T>& input, std::vector<std::complex<T>>& output,
          FFTNorm norm = FFTNorm::Forward);

/**
 * @brief Computes IRFFT using a temporary plan.
 * Uses Backward normalization by default. Output size N must be specified.
 * Input size must be N/2 + 1. Output vector is resized if necessary.
 * @param input Complex input vector (size N/2 + 1).
 * @param output Real output vector (will be resized to N).
 * @param N The desired length of the real output signal. Must be > 0.
 * @param norm Normalization mode.
 * @throws std::runtime_error If temporary plan creation or execution fails.
 * @throws std::invalid_argument If input size is incorrect for output size N,
 * or if N=0.
 */
template <typename T>
void irfft(const std::vector<std::complex<T>>& input, std::vector<T>& output,
           size_t N,  // Must provide original real signal length N
           FFTNorm norm = FFTNorm::Backward);

}  // namespace OmniDSP

#endif  // OMNIDSP_FFT_H
