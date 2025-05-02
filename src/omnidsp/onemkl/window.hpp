/**
 * @file window.hpp (onemkl)
 * @brief Declares oneMKL backend helper functions for window generation using
 * IPP/VML.
 */

#ifndef OMNIDSP_ONEMKL_WINDOW_HPP
#define OMNIDSP_ONEMKL_WINDOW_HPP

#include <OmniDSP/core_types.hpp>  // For Status, F32, F64
#include <span>                    // For std::span

namespace OmniDSP::backend {

  // --- oneMKL/IPP Window Generation Helper Function Declarations ---
  // These functions perform the actual computation using IPP/VML.
  // They are called by the corresponding virtual methods in OneMKLBackend.

  [[nodiscard]] Status generate_bartlett_window_onemkl(std::span<F32> output);
  [[nodiscard]] Status generate_bartlett_window_onemkl(std::span<F64> output);

  [[nodiscard]] Status generate_blackman_window_onemkl(std::span<F32> output);
  [[nodiscard]] Status generate_blackman_window_onemkl(std::span<F64> output);

  // *** REMOVED Flattop declarations ***
  // [[nodiscard]] Status generate_flattop_window_onemkl(std::span<F32> output);
  // [[nodiscard]] Status generate_flattop_window_onemkl(std::span<F64> output);

  // *** REMOVED Gaussian declarations ***
  // [[nodiscard]] Status generate_gaussian_window_onemkl(
  //     double stddev, std::span<F32> output);
  // [[nodiscard]] Status generate_gaussian_window_onemkl(
  //     double stddev, std::span<F64> output);

  [[nodiscard]] Status generate_hamming_window_onemkl(std::span<F32> output);
  [[nodiscard]] Status generate_hamming_window_onemkl(std::span<F64> output);

  [[nodiscard]] Status generate_hann_window_onemkl(std::span<F32> output);
  [[nodiscard]] Status generate_hann_window_onemkl(std::span<F64> output);

  [[nodiscard]] Status generate_kaiser_window_onemkl(
      double beta, std::span<F32> output);
  [[nodiscard]] Status generate_kaiser_window_onemkl(
      double beta, std::span<F64> output);

  [[nodiscard]] Status generate_rectangular_window_onemkl(
      std::span<F32> output);
  [[nodiscard]] Status generate_rectangular_window_onemkl(
      std::span<F64> output);

  [[nodiscard]] Status generate_triangular_window_onemkl(std::span<F32> output);
  [[nodiscard]] Status generate_triangular_window_onemkl(std::span<F64> output);

}  // namespace OmniDSP::backend

#endif  // OMNIDSP_ONEMKL_WINDOW_HPP
