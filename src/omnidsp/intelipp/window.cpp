/**
 * @file window.cpp (intelipp)
 * @brief Implements Intel IPP backend window generation helper functions.
 */

#include "window.hpp"  // Corresponding header

// Include Intel IPP signal processing header
#include <ipps.h>
// Include IPP core header for ippGetStatusString (needed by Utils)
#include <ippcore.h>

#include <iostream>   // For debug/error messages
#include <numbers>    // For std::numbers::pi_v
#include <numeric>    // Not strictly needed now ippsSet is used for Rectangular
#include <span>       // For std::span
#include <stdexcept>  // For potential error conditions
#include <string>     // For error messages
#include <vector>     // Needed for IPP window temp input

// Include core types for Status and type aliases F32, F64
#include <OmniDSP/core_types.hpp>

// Include the intelipp utility header
#include "details.hpp"

// Changed namespace to intelipp
namespace OmniDSP::IntelIPP {

  //--------------------------------------------------------------------------
  // Intel IPP Window Generation Helper Function Implementations
  //--------------------------------------------------------------------------

  // --- Bartlett ---
  // Renamed function
  [[nodiscard]] Status generate_bartlett_window_intelipp(std::span<F32> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0f;
      return Status::Success;
    }
    // IPP Bartlett needs dummy input
    std::vector<F32> temp_input(length, 1.0f);
    IppStatus status = ippsWinBartlett_32f(
        temp_input.data(), output.data(), static_cast<int>(length));
    // Use helper from intelipp/Utils.hpp
    return Details::ipp_status_to_omnidsp_status(status);
  }
  // Renamed function
  [[nodiscard]] Status generate_bartlett_window_intelipp(std::span<F64> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0;
      return Status::Success;
    }
    // IPP Bartlett needs dummy input
    std::vector<F64> temp_input(length, 1.0);
    IppStatus status = ippsWinBartlett_64f(
        temp_input.data(), output.data(), static_cast<int>(length));
    // Use helper from intelipp/Utils.hpp
    return Details::ipp_status_to_omnidsp_status(status);
  }

  // --- Blackman ---
  // Renamed function
  [[nodiscard]] Status generate_blackman_window_intelipp(std::span<F32> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0f;
      return Status::Success;
    }
    // IPP Blackman needs dummy input
    std::vector<F32> temp_input(length, 1.0f);
    // Using standard Blackman, Opt might also be available
    IppStatus status = ippsWinBlackmanStd_32f(
        temp_input.data(), output.data(), static_cast<int>(length));
    // Use helper from intelipp/Utils.hpp
    return Details::ipp_status_to_omnidsp_status(status);
  }
  // Renamed function
  [[nodiscard]] Status generate_blackman_window_intelipp(std::span<F64> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0;
      return Status::Success;
    }
    // IPP Blackman needs dummy input
    std::vector<F64> temp_input(length, 1.0);
    // Using standard Blackman, Opt might also be available
    IppStatus status = ippsWinBlackmanStd_64f(
        temp_input.data(), output.data(), static_cast<int>(length));
    // Use helper from intelipp/Utils.hpp
    return Details::ipp_status_to_omnidsp_status(status);
  }

  // --- Hamming ---
  // Renamed function
  [[nodiscard]] Status generate_hamming_window_intelipp(std::span<F32> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0f;
      return Status::Success;
    }
    // IPP Hamming needs dummy input
    std::vector<F32> temp_input(length, 1.0f);
    IppStatus status = ippsWinHamming_32f(
        temp_input.data(), output.data(), static_cast<int>(length));
    // Use helper from intelipp/Utils.hpp
    return Details::ipp_status_to_omnidsp_status(status);
  }
  // Renamed function
  [[nodiscard]] Status generate_hamming_window_intelipp(std::span<F64> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0;
      return Status::Success;
    }
    // IPP Hamming needs dummy input
    std::vector<F64> temp_input(length, 1.0);
    IppStatus status = ippsWinHamming_64f(
        temp_input.data(), output.data(), static_cast<int>(length));
    // Use helper from intelipp/Utils.hpp
    return Details::ipp_status_to_omnidsp_status(status);
  }

  // --- Hann ---
  // Renamed function
  [[nodiscard]] Status generate_hann_window_intelipp(std::span<F32> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0f;
      return Status::Success;
    }
    // IPP Hann needs dummy input
    std::vector<F32> temp_input(length, 1.0f);
    IppStatus status = ippsWinHann_32f(
        temp_input.data(), output.data(), static_cast<int>(length));
    // Use helper from intelipp/Utils.hpp
    return Details::ipp_status_to_omnidsp_status(status);
  }
  // Renamed function
  [[nodiscard]] Status generate_hann_window_intelipp(std::span<F64> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0;
      return Status::Success;
    }
    // IPP Hann needs dummy input
    std::vector<F64> temp_input(length, 1.0);
    IppStatus status = ippsWinHann_64f(
        temp_input.data(), output.data(), static_cast<int>(length));
    // Use helper from intelipp/Utils.hpp
    return Details::ipp_status_to_omnidsp_status(status);
  }

  // --- Kaiser ---
  // Renamed function
  [[nodiscard]] Status generate_kaiser_window_intelipp(
      double beta, std::span<F32> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    // IPP uses alpha = beta / pi
    if (beta < 0.0) return Status::InvalidArgument;
    if (length == 1) {
      output[0] = 1.0f;
      return Status::Success;
    }

    Ipp32f alpha = static_cast<Ipp32f>(beta / std::numbers::pi_v<double>);
    // IPP Kaiser needs dummy input
    std::vector<F32> temp_input(length, 1.0f);
    IppStatus status = ippsWinKaiser_32f(
        temp_input.data(), output.data(), static_cast<int>(length), alpha);
    // Use helper from intelipp/Utils.hpp
    return Details::ipp_status_to_omnidsp_status(status);
  }
  // Renamed function
  [[nodiscard]] Status generate_kaiser_window_intelipp(
      double beta, std::span<F64> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    // IPP uses alpha = beta / pi
    if (beta < 0.0) return Status::InvalidArgument;
    if (length == 1) {
      output[0] = 1.0;
      return Status::Success;
    }

    Ipp64f alpha = static_cast<Ipp64f>(beta / std::numbers::pi_v<double>);
    // IPP Kaiser needs dummy input
    std::vector<F64> temp_input(length, 1.0);
    IppStatus status = ippsWinKaiser_64f(
        temp_input.data(), output.data(), static_cast<int>(length), alpha);
    // Use helper from intelipp/Utils.hpp
    return Details::ipp_status_to_omnidsp_status(status);
  }

  // --- Rectangular ---
  // Renamed function
  [[nodiscard]] Status generate_rectangular_window_intelipp(
      std::span<F32> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    // Use ippsSet to fill with 1.0
    IppStatus status
        = ippsSet_32f(1.0f, output.data(), static_cast<int>(length));
    // Use helper from intelipp/Utils.hpp
    return Details::ipp_status_to_omnidsp_status(status);
  }
  // Renamed function
  [[nodiscard]] Status generate_rectangular_window_intelipp(
      std::span<F64> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    // Use ippsSet to fill with 1.0
    IppStatus status
        = ippsSet_64f(1.0, output.data(), static_cast<int>(length));
    // Use helper from intelipp/Utils.hpp
    return Details::ipp_status_to_omnidsp_status(status);
  }

  // --- Triangular ---
  // Renamed function
  [[nodiscard]] Status generate_triangular_window_intelipp(
      std::span<F32> output)
  {
    // IPP Bartlett is equivalent to Triangular
    return generate_bartlett_window_intelipp(output);
  }
  // Renamed function
  [[nodiscard]] Status generate_triangular_window_intelipp(
      std::span<F64> output)
  {
    // IPP Bartlett is equivalent to Triangular
    return generate_bartlett_window_intelipp(output);
  }

}  // namespace OmniDSP::IntelIPP

// No endif for include guard
