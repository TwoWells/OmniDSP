/**
 * @file window.cpp (onemkl / IPP / VML)
 * @brief Implements oneMKL backend window generation helper functions using
 * Intel IPP/VML.
 */

#include "window.hpp"  // Include the new header file

// Include Intel IPP signal processing header
#include <ipps.h>
// Include IPP core header for ippGetStatusString
#include <ippcore.h>
// Include MKL header for VML (used for windows not directly in IPP)
#include <mkl.h>

#include <cmath>      // For std::abs used by VML Gaussian
#include <iostream>   // For debug/error messages
#include <numbers>    // For std::numbers::pi_v
#include <numeric>    // For std::fill_n
#include <span>       // For std::span
#include <stdexcept>  // For potential error conditions
#include <string>     // For error messages
#include <vector>     // Needed for IPP window temp input and VML windows

// Include core types for Status and type aliases F32, F64
#include <OmniDSP/core_types.hpp>

// *** ADDED: Include the new utility header ***
#include "utils.hpp"

namespace OmniDSP::backend {

  // *** REMOVED ipp_status_to_omnidsp_status helper (now in utils.hpp) ***

  //--------------------------------------------------------------------------
  // oneMKL Window Generation Helper Function Implementations using IPP/VML
  //--------------------------------------------------------------------------

  // --- Bartlett ---
  [[nodiscard]] Status generate_bartlett_window_onemkl(std::span<F32> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0f;
      return Status::Success;
    }
    // *** Corrected: Use 3-argument version with temp input of ones ***
    std::vector<F32> temp_input(length, 1.0f);
    IppStatus status = ippsWinBartlett_32f(
        temp_input.data(), output.data(), static_cast<int>(length));
    // *** Use helper from utils.hpp ***
    return ipp_status_to_omnidsp_status(status);
  }
  [[nodiscard]] Status generate_bartlett_window_onemkl(std::span<F64> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0;
      return Status::Success;
    }
    // *** Corrected: Use 3-argument version with temp input of ones ***
    std::vector<F64> temp_input(length, 1.0);
    IppStatus status = ippsWinBartlett_64f(
        temp_input.data(), output.data(), static_cast<int>(length));
    // *** Use helper from utils.hpp ***
    return ipp_status_to_omnidsp_status(status);
  }

  // --- Blackman ---
  [[nodiscard]] Status generate_blackman_window_onemkl(std::span<F32> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0f;
      return Status::Success;
    }
    // *** Corrected: Use 3-argument version with temp input of ones ***
    std::vector<F32> temp_input(length, 1.0f);
    IppStatus status = ippsWinBlackmanStd_32f(
        temp_input.data(), output.data(), static_cast<int>(length));
    // *** Use helper from utils.hpp ***
    return ipp_status_to_omnidsp_status(status);
  }
  [[nodiscard]] Status generate_blackman_window_onemkl(std::span<F64> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0;
      return Status::Success;
    }
    // *** Corrected: Use 3-argument version with temp input of ones ***
    std::vector<F64> temp_input(length, 1.0);
    IppStatus status = ippsWinBlackmanStd_64f(
        temp_input.data(), output.data(), static_cast<int>(length));
    // *** Use helper from utils.hpp ***
    return ipp_status_to_omnidsp_status(status);
  }

  // --- Hamming ---
  [[nodiscard]] Status generate_hamming_window_onemkl(std::span<F32> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0f;
      return Status::Success;
    }
    // *** Corrected: Use 3-argument version with temp input of ones ***
    std::vector<F32> temp_input(length, 1.0f);
    IppStatus status = ippsWinHamming_32f(
        temp_input.data(), output.data(), static_cast<int>(length));
    // *** Use helper from utils.hpp ***
    return ipp_status_to_omnidsp_status(status);
  }
  [[nodiscard]] Status generate_hamming_window_onemkl(std::span<F64> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0;
      return Status::Success;
    }
    // *** Corrected: Use 3-argument version with temp input of ones ***
    std::vector<F64> temp_input(length, 1.0);
    IppStatus status = ippsWinHamming_64f(
        temp_input.data(), output.data(), static_cast<int>(length));
    // *** Use helper from utils.hpp ***
    return ipp_status_to_omnidsp_status(status);
  }

  // --- Hann ---
  [[nodiscard]] Status generate_hann_window_onemkl(std::span<F32> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0f;
      return Status::Success;
    }
    // *** Corrected: Use 3-argument version with temp input of ones ***
    std::vector<F32> temp_input(length, 1.0f);
    IppStatus status = ippsWinHann_32f(
        temp_input.data(), output.data(), static_cast<int>(length));
    // *** Use helper from utils.hpp ***
    return ipp_status_to_omnidsp_status(status);
  }
  [[nodiscard]] Status generate_hann_window_onemkl(std::span<F64> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0;
      return Status::Success;
    }
    // *** Corrected: Use 3-argument version with temp input of ones ***
    std::vector<F64> temp_input(length, 1.0);
    IppStatus status = ippsWinHann_64f(
        temp_input.data(), output.data(), static_cast<int>(length));
    // *** Use helper from utils.hpp ***
    return ipp_status_to_omnidsp_status(status);
  }

  // --- Kaiser ---
  [[nodiscard]] Status generate_kaiser_window_onemkl(
      double beta, std::span<F32> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (beta < 0.0) return Status::InvalidArgument;
    if (length == 1) {
      output[0] = 1.0f;
      return Status::Success;
    }

    Ipp32f alpha = static_cast<Ipp32f>(beta / std::numbers::pi_v<double>);
    // *** Corrected: Use 3-argument version with temp input of ones ***
    std::vector<F32> temp_input(length, 1.0f);
    IppStatus status = ippsWinKaiser_32f(
        temp_input.data(), output.data(), static_cast<int>(length), alpha);
    // *** Use helper from utils.hpp ***
    return ipp_status_to_omnidsp_status(status);
  }
  [[nodiscard]] Status generate_kaiser_window_onemkl(
      double beta, std::span<F64> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (beta < 0.0) return Status::InvalidArgument;
    if (length == 1) {
      output[0] = 1.0;
      return Status::Success;
    }

    Ipp64f alpha = static_cast<Ipp64f>(beta / std::numbers::pi_v<double>);
    // *** Corrected: Use 3-argument version with temp input of ones ***
    std::vector<F64> temp_input(length, 1.0);
    IppStatus status = ippsWinKaiser_64f(
        temp_input.data(), output.data(), static_cast<int>(length), alpha);
    // *** Use helper from utils.hpp ***
    return ipp_status_to_omnidsp_status(status);
  }

  // --- Rectangular ---
  [[nodiscard]] Status generate_rectangular_window_onemkl(std::span<F32> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    // ippsSet takes 3 arguments: value, buffer, length
    IppStatus status
        = ippsSet_32f(1.0f, output.data(), static_cast<int>(length));
    // *** Use helper from utils.hpp ***
    return ipp_status_to_omnidsp_status(status);
  }
  [[nodiscard]] Status generate_rectangular_window_onemkl(std::span<F64> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    // ippsSet takes 3 arguments: value, buffer, length
    IppStatus status
        = ippsSet_64f(1.0, output.data(), static_cast<int>(length));
    // *** Use helper from utils.hpp ***
    return ipp_status_to_omnidsp_status(status);
  }

  // --- Triangular ---
  [[nodiscard]] Status generate_triangular_window_onemkl(std::span<F32> output)
  {
    return generate_bartlett_window_onemkl(output);  // Reuse Bartlett
  }
  [[nodiscard]] Status generate_triangular_window_onemkl(std::span<F64> output)
  {
    return generate_bartlett_window_onemkl(output);  // Reuse Bartlett
  }

}  // namespace OmniDSP::backend
