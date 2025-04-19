/**
 * @file fft.h
 * @brief Public API header for FFT and related transforms (IFFT, RFFT, IRFFT).
 * Defines plan classes, convenience functions (including in-place overloads),
 * and related enums within the OmniDSP namespace.
 * @version 2.1.0 (Added in-place overloads for fft/ifft)
 * @date 2025-04-18
 */
#ifndef OMNIDSP_FFT_H
#define OMNIDSP_FFT_H

// --- Standard Includes ---
#include <complex>
#include <cstddef>    // For size_t
#include <memory>     // For std::unique_ptr
#include <stdexcept>  // For std::invalid_argument in docs
#include <vector>

// --- Project Includes ---
#include "core_types.h"  // Include core types (Precision) - NOT omnidsp.h

namespace OmniDSP {  // <<< Namespace opened for this header's content

// --- FFT-Specific Enums ---

/** @brief Specifies the direction of the Fourier Transform. */
enum class Direction {
  FORWARD, /**< Forward Transform (e.g., time to frequency). */
  INVERSE  /**< Inverse Transform (e.g., frequency to time). */
};

/** @brief Specifies the domain of the input/output signals. */
enum class Domain {
  COMPLEX, /**< Complex-to-Complex (C2C) transform. */
  REAL     /**< Real-valued transform (R2C/C2R). */
};

/** @brief Specifies the normalization/scaling mode applied to the transforms.
 */
enum class FFTNorm {
  /** @brief Forward unscaled, Inverse scaled by 1/N. */
  BACKWARD,
  /** @brief Forward scaled by 1/N, Inverse unscaled. */
  FORWARD,
  /** @brief Forward and Inverse scaled by 1/sqrt(N). Unitary. */
  ORTHO
};

// --- Forward declaration of implementation classes (PIMPL pattern) ---
// These are internal details hidden from the public API user.
template <typename T>
struct FFTPlanImpl;
template <typename T>
struct RFFTPlanImpl;

// --- FFTPlan Class (Complex-to-Complex) ---

/**
 * @brief Manages a pre-calculated plan for efficient Complex-to-Complex (C2C)
 * FFT execution.
 *
 * Use this class when performing multiple FFTs or IFFTs of the same size,
 * precision, direction, domain, and normalization. Creating the plan involves
 * setup overhead, but subsequent executions using the plan are faster than
 * using the stateless convenience functions.
 *
 * @tparam T The floating-point type (float or double).
 */
template <typename T>
class FFTPlan {
 public:
  /**
   * @brief Constructs an FFT plan for C2C transforms.
   * @param length The size 'N' of the transform. Must be > 0.
   * @param precision The desired floating-point precision (Precision::SINGLE or
   * Precision::DOUBLE).
   * @param direction The primary transform direction (Direction::FORWARD or
   * Direction::INVERSE) this plan is optimized for.
   * @param domain The domain (must be Domain::COMPLEX for FFTPlan).
   * @param norm The normalization mode (FFTNorm::BACKWARD, FFTNorm::FORWARD, or
   * FFTNorm::ORTHO).
   * @throws std::invalid_argument If length is 0, precision/domain mismatch, or
   * invalid enum values.
   * @throws std::runtime_error If the backend fails to create the plan (e.g.,
   * unsupported length, memory allocation failure).
   */
  FFTPlan(size_t length, Precision precision, Direction direction,
          Domain domain, FFTNorm norm);

  /** @brief Destructor, releases backend resources. */
  ~FFTPlan();

  // --- Rule of 5/3: Move-only ---
  /** @brief Move constructor. */
  FFTPlan(FFTPlan &&) noexcept;
  /** @brief Move assignment operator. */
  FFTPlan &operator=(FFTPlan &&) noexcept;
  /** @brief Deleted copy constructor. */
  FFTPlan(const FFTPlan &) = delete;
  /** @brief Deleted copy assignment operator. */
  FFTPlan &operator=(const FFTPlan &) = delete;

  /**
   * @brief Executes the forward FFT (out-of-place) using the plan.
   * Requires the plan to be created with Direction::FORWARD.
   * @param input The input complex signal vector (size must match plan length).
   * @param output The output complex spectrum vector (will be resized if
   * necessary, size must match plan length).
   * @throws std::runtime_error If called on a plan not created for FORWARD
   * direction, or if backend execution fails.
   * @throws std::invalid_argument If input size doesn't match plan length.
   * @throws std::bad_alloc If resizing output fails.
   */
  void fft(const std::vector<std::complex<T>> &input,
           std::vector<std::complex<T>> &output);

  /**
   * @brief Executes the inverse FFT (out-of-place) using the plan.
   * Requires the plan to be created with Direction::INVERSE.
   * @param input The input complex spectrum vector (size must match plan
   * length).
   * @param output The output complex signal vector (will be resized if
   * necessary, size must match plan length).
   * @throws std::runtime_error If called on a plan not created for INVERSE
   * direction, or if backend execution fails.
   * @throws std::invalid_argument If input size doesn't match plan length.
   * @throws std::bad_alloc If resizing output fails.
   */
  void ifft(const std::vector<std::complex<T>> &input,
            std::vector<std::complex<T>> &output);

  // --- Getters ---
  /** @brief Gets the length 'N' of the transform this plan was created for. */
  size_t getSize() const;
  // Add other getters if needed (getDirection, getPrecision, getDomain,
  // getNormMode)

 private:
  /** @brief Pointer to the internal implementation details (PIMPL pattern). */
  std::unique_ptr<FFTPlanImpl<T>> pimpl_;
};

// --- RFFTPlan Class (Real-to-Complex / Complex-to-Real) ---

/**
 * @brief Manages a pre-calculated plan for efficient Real FFT execution
 * (RFFT/IRFFT).
 *
 * Use this class for transforms involving real signals (R2C or C2R).
 *
 * @tparam T The floating-point type (float or double).
 */
template <typename T>
class RFFTPlan {
 public:
  /**
   * @brief Constructs an FFT plan for Real transforms (R2C/C2R).
   * @param length The size 'N' of the real signal. Must be > 0.
   * @param precision The desired floating-point precision (Precision::SINGLE or
   * Precision::DOUBLE).
   * @param direction The primary transform direction (Direction::FORWARD for
   * RFFT, Direction::INVERSE for IRFFT).
   * @param domain The domain (must be Domain::REAL for RFFTPlan).
   * @param norm The normalization mode (FFTNorm::BACKWARD, FFTNorm::FORWARD, or
   * FFTNorm::ORTHO).
   * @throws std::invalid_argument If length is 0, precision/domain mismatch, or
   * invalid enum values.
   * @throws std::runtime_error If the backend fails to create the plan (e.g.,
   * unsupported length, memory allocation failure).
   */
  RFFTPlan(size_t length, Precision precision, Direction direction,
           Domain domain, FFTNorm norm);

  /** @brief Destructor, releases backend resources. */
  ~RFFTPlan();

  // --- Rule of 5/3: Move-only ---
  /** @brief Move constructor. */
  RFFTPlan(RFFTPlan &&) noexcept;
  /** @brief Move assignment operator. */
  RFFTPlan &operator=(RFFTPlan &&) noexcept;
  /** @brief Deleted copy constructor. */
  RFFTPlan(const RFFTPlan &) = delete;
  /** @brief Deleted copy assignment operator. */
  RFFTPlan &operator=(const RFFTPlan &) = delete;

  /**
   * @brief Executes the forward Real-to-Complex FFT (RFFT) using the plan.
   * Requires the plan to be created with Direction::FORWARD.
   * @param input The input real signal vector (size N, must match plan length).
   * @param output The output complex spectrum vector (will be resized if
   * necessary, size N/2 + 1).
   * @throws std::runtime_error If called on a plan not created for FORWARD
   * direction, or if backend execution fails.
   * @throws std::invalid_argument If input size doesn't match plan length.
   * @throws std::bad_alloc If resizing output fails.
   */
  void rfft(const std::vector<T> &input, std::vector<std::complex<T>> &output);

  /**
   * @brief Executes the inverse Complex-to-Real FFT (IRFFT) using the plan.
   * Requires the plan to be created with Direction::INVERSE.
   * Input spectrum must possess Hermitian symmetry for the output to be purely
   * real.
   * @param input The input complex spectrum vector (size N/2 + 1, where N is
   * the plan length).
   * @param output The output real signal vector (will be resized if necessary,
   * size N).
   * @throws std::runtime_error If called on a plan not created for INVERSE
   * direction, or if backend execution fails.
   * @throws std::invalid_argument If input size is incorrect for plan length N.
   * @throws std::bad_alloc If resizing output fails.
   */
  void irfft(const std::vector<std::complex<T>> &input, std::vector<T> &output);

  // --- Getters ---
  /** @brief Gets the length 'N' of the real signal this plan was created for.
   */
  size_t getSize() const;
  // Add other getters if needed

 private:
  /** @brief Pointer to the internal implementation details (PIMPL pattern). */
  std::unique_ptr<RFFTPlanImpl<T>> pimpl_;
};

// --- Convenience Functions (Stateless) ---
// These functions create temporary plans internally. Use Plan objects for
// repeated transforms.

// -- Out-of-Place Complex FFT/IFFT --
/**
 * @brief Performs an out-of-place complex-to-complex forward FFT.
 * Creates a temporary plan internally.
 * @tparam T float or double.
 * @param input Input complex signal vector.
 * @param output Output complex spectrum vector (resized if necessary).
 * @param norm Normalization mode (default: FORWARD).
 * @throws std::runtime_error If plan creation or execution fails.
 */
template <typename T>
void fft(const std::vector<std::complex<T>> &input,
         std::vector<std::complex<T>> &output, FFTNorm norm = FFTNorm::FORWARD);

/**
 * @brief Performs an out-of-place complex-to-complex inverse FFT.
 * Creates a temporary plan internally.
 * @tparam T float or double.
 * @param input Input complex spectrum vector.
 * @param output Output complex signal vector (resized if necessary).
 * @param norm Normalization mode (default: BACKWARD).
 * @throws std::runtime_error If plan creation or execution fails.
 */
template <typename T>
void ifft(const std::vector<std::complex<T>> &input,
          std::vector<std::complex<T>> &output,
          FFTNorm norm = FFTNorm::BACKWARD);

// -- In-Place Complex FFT/IFFT (Overloads) --
/**
 * @brief Performs an in-place complex-to-complex forward FFT.
 * Creates a temporary plan internally. Modifies the input vector directly.
 * @tparam T float or double.
 * @param data Input/output vector containing the complex signal. Modified in
 * place.
 * @param norm Normalization mode (default: FORWARD).
 * @throws std::runtime_error If plan creation or execution fails.
 */
template <typename T>
void fft(std::vector<std::complex<T>> &data, FFTNorm norm = FFTNorm::FORWARD);

/**
 * @brief Performs an in-place complex-to-complex inverse FFT.
 * Creates a temporary plan internally. Modifies the input vector directly.
 * @tparam T float or double.
 * @param data Input/output vector containing the complex spectrum. Modified in
 * place.
 * @param norm Normalization mode (default: BACKWARD).
 * @throws std::runtime_error If plan creation or execution fails.
 */
template <typename T>
void ifft(std::vector<std::complex<T>> &data, FFTNorm norm = FFTNorm::BACKWARD);

// -- Out-of-Place Real FFT/IFFT --
/**
 * @brief Performs an out-of-place real-to-complex forward FFT (RFFT).
 * Creates a temporary plan internally. Output size is N/2 + 1.
 * @tparam T float or double.
 * @param input Input real signal vector (size N).
 * @param output Output complex spectrum vector (resized if necessary, size N/2
 * + 1).
 * @param norm Normalization mode (default: FORWARD).
 * @throws std::runtime_error If plan creation or execution fails.
 */
template <typename T>
void rfft(const std::vector<T> &input, std::vector<std::complex<T>> &output,
          FFTNorm norm = FFTNorm::FORWARD);

/**
 * @brief Performs an out-of-place complex-to-real inverse FFT (IRFFT).
 * Creates a temporary plan internally. Input size must be N/2 + 1.
 * @tparam T float or double.
 * @param input Input complex spectrum vector (size N/2 + 1). Must have
 * Hermitian symmetry.
 * @param output Output real signal vector (resized if necessary, size N).
 * @param N The desired length of the output real signal.
 * @param norm Normalization mode (default: BACKWARD).
 * @throws std::runtime_error If plan creation or execution fails.
 * @throws std::invalid_argument If input size doesn't match N/2 + 1.
 */
template <typename T>
void irfft(const std::vector<std::complex<T>> &input, std::vector<T> &output,
           size_t N, FFTNorm norm = FFTNorm::BACKWARD);

}  // namespace OmniDSP

#endif  // OMNIDSP_FFT_H
