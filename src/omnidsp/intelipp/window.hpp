/**
 * @file window.hpp (intelipp)
 * @brief Declares Intel IPP backend helper functions for window generation.
 */

#ifndef OMNIDSP_INTELIPP_WINDOW_HPP
#define OMNIDSP_INTELIPP_WINDOW_HPP

#include <OmniDSP/core_types.hpp>  // For Status, F32, F64
#include <span>                    // For std::span

// Changed namespace to intelipp
namespace OmniDSP::intelipp {

  // --- Intel IPP Window Generation Helper Function Declarations ---
  // These functions perform the actual computation using IPP.
  // They are called by the corresponding virtual methods in IntelIPPBackend.

  // Renamed functions from _onemkl to _intelipp
  [[nodiscard]] Status generate_bartlett_window_intelipp(std::span<F32> output);
  [[nodiscard]] Status generate_bartlett_window_intelipp(std::span<F64> output);

  [[nodiscard]] Status generate_blackman_window_intelipp(std::span<F32> output);
  [[nodiscard]] Status generate_blackman_window_intelipp(std::span<F64> output);

  [[nodiscard]] Status generate_hamming_window_intelipp(std::span<F32> output);
  [[nodiscard]] Status generate_hamming_window_intelipp(std::span<F64> output);

  [[nodiscard]] Status generate_hann_window_intelipp(std::span<F32> output);
  [[nodiscard]] Status generate_hann_window_intelipp(std::span<F64> output);

  [[nodiscard]] Status generate_kaiser_window_intelipp(
      double beta, std::span<F32> output);
  [[nodiscard]] Status generate_kaiser_window_intelipp(
      double beta, std::span<F64> output);

  [[nodiscard]] Status generate_rectangular_window_intelipp(
      std::span<F32> output);
  [[nodiscard]] Status generate_rectangular_window_intelipp(
      std::span<F64> output);

  [[nodiscard]] Status generate_triangular_window_intelipp(
      std::span<F32> output);
  [[nodiscard]] Status generate_triangular_window_intelipp(
      std::span<F64> output);

}  // namespace OmniDSP::intelipp

#endif  // OMNIDSP_INTELIPP_WINDOW_HPP
