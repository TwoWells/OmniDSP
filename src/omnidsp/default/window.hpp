/**
 * @file window.hpp (Default)
 * @brief Declares internal helper functions for window generation for the
 * Default backend.
 * @details These functions now write directly to the provided output span.
 */

#ifndef OMNIDSP_DEFAULT_WINDOW_HPP
#define OMNIDSP_DEFAULT_WINDOW_HPP

#include <OmniDSP/core_types.hpp>  // Status, F32, F64
#include <cstddef>                 // For size_t
#include <span>                    // For std::span

namespace OmniDSP::Default {

  // --- Default Window Generation Helper Functions (Span-based Output) ---
  // These implement the actual window calculations used by
  // Backend::generate_window_*

  /**
   * @brief Generates a Bartlett (triangular, zero-endpoints) window.
   * @tparam T Data type (F32 or F64).
   * @param output The span to write the window values into. Size determines
   * window length.
   * @return Status::Success on success, Status::InvalidArgument if output is
   * empty.
   */
  template <typename T>  // T = F32 or F64
  [[nodiscard]] Status bartlett_window(std::span<T> output);

  /** @brief Generates a Blackman window. */
  template <typename T>  // T = F32 or F64
  [[nodiscard]] Status blackman_window(std::span<T> output);

  /** @brief Generates a Flat top window. */
  template <typename T>  // T = F32 or F64
  [[nodiscard]] Status flattop_window(std::span<T> output);

  /** @brief Generates a Hamming window. */
  template <typename T>  // T = F32 or F64
  [[nodiscard]] Status hamming_window(std::span<T> output);

  /** @brief Generates a Hann (Hanning) window. */
  template <typename T>  // T = F32 or F64
  [[nodiscard]] Status hann_window(std::span<T> output);

  /** @brief Generates a Rectangular window (fills with ones). */
  template <typename T>  // T = F32 or F64
  [[nodiscard]] Status rectangular_window(std::span<T> output);

  /** @brief Generates a Triangular window (non-zero endpoints). */
  template <typename T>  // T = F32 or F64
  [[nodiscard]] Status triangular_window(std::span<T> output);

  /**
   * @brief Generates a Gaussian window.
   * @tparam T Data type (F32 or F64).
   * @param stddev Standard deviation (sigma) of the Gaussian curve. Must be
   * positive.
   * @param output The span to write the window values into. Size determines
   * window length.
   * @return Status::Success on success, Status::InvalidArgument if output is
   * empty or stddev <= 0.
   */
  template <typename T>  // T = F32 or F64
  [[nodiscard]] Status gaussian_window(T stddev, std::span<T> output);

  /**
   * @brief Generates a Kaiser window.
   * @tparam T Data type (F32 or F64).
   * @param beta Shape parameter (beta >= 0). beta=0 is rectangular.
   * @param output The span to write the window values into. Size determines
   * window length.
   * @return Status::Success on success, Status::InvalidArgument if output is
   * empty or beta < 0.
   */
  template <typename T>  // T = F32 or F64
  [[nodiscard]] Status kaiser_window(T beta, std::span<T> output);

  // --- Explicit Template Instantiations (Declaration) ---
  extern template Status bartlett_window<F32>(std::span<F32> output);
  extern template Status bartlett_window<F64>(std::span<F64> output);
  extern template Status blackman_window<F32>(std::span<F32> output);
  extern template Status blackman_window<F64>(std::span<F64> output);
  extern template Status flattop_window<F32>(std::span<F32> output);
  extern template Status flattop_window<F64>(std::span<F64> output);
  extern template Status hamming_window<F32>(std::span<F32> output);
  extern template Status hamming_window<F64>(std::span<F64> output);
  extern template Status hann_window<F32>(std::span<F32> output);
  extern template Status hann_window<F64>(std::span<F64> output);
  extern template Status rectangular_window<F32>(std::span<F32> output);
  extern template Status rectangular_window<F64>(std::span<F64> output);
  extern template Status triangular_window<F32>(std::span<F32> output);
  extern template Status triangular_window<F64>(std::span<F64> output);
  extern template Status gaussian_window<F32>(
      F32 stddev, std::span<F32> output);
  extern template Status gaussian_window<F64>(
      F64 stddev, std::span<F64> output);
  extern template Status kaiser_window<F32>(F32 beta, std::span<F32> output);
  extern template Status kaiser_window<F64>(F64 beta, std::span<F64> output);

}  // namespace OmniDSP::Default

#endif  // OMNIDSP_DEFAULT_WINDOW_HPP
