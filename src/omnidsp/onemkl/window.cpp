/**
 * @file window.cpp (onemkl / IPP)
 * @brief Implements oneMKL backend window generation helper functions using
 * Intel IPP/VML.
 */

#include "window.hpp"  // Include the corresponding header file

#ifdef OMNIDSP_USE_ONEMKL  // Compile guard

// Include Intel IPP signal processing header
#include <ipps.h>
// Include MKL header for VML (used for windows not directly in IPP)
#include <mkl.h>

#include <algorithm>  // For std::max needed by Kaiser helper
#include <cmath>      // For std::max needed in Kaiser
#include <iostream>   // For debug/error messages
#include <numbers>    // For std::numbers::pi_v
#include <numeric>    // For std::fill_n
#include <span>       // For std::span
#include <stdexcept>  // For potential error conditions
#include <string>     // For error messages
#include <vector>     // Only needed if VML implementations use temp vectors

// Include Boost Bessel function for Kaiser window
#include <boost/math/special_functions/bessel.hpp>

// Include core types for Status and type aliases F32, F64
#include <OmniDSP/core_types.hpp>

namespace OmniDSP::backend {

  /**
   * @brief Helper function to convert IPP status codes to OmniDSP::Status.
   * @param status The IppStatus code returned by an IPP function.
   * @return The corresponding OmniDSP::Status enum value.
   */
  inline Status ipp_status_to_omnidsp_status(IppStatus status)
  {
    if (status == ippStsNoErr) {
      return Status::Success;
    }
    std::cerr << "IPP Error: " << ippGetStatusString(status)
              << " (Code: " << status << ")" << std::endl;
    if (status == ippStsNullPtrErr || status == ippStsSizeErr
        || status == ippStsStepErr || status == ippStsBadArgErr
        || status == ippStsOutOfRangeErr) {
      return Status::InvalidArgument;
    }
    if (status == ippStsMemAllocErr) {
      return Status::AllocationError;
    }
    return Status::BackendError;
  }

  //--------------------------------------------------------------------------
  // oneMKL Window Generation Helper Function Implementations using IPP/VML
  //--------------------------------------------------------------------------
  // Note: These are now free functions within the namespace.

  // --- Bartlett ---
  [[nodiscard]] Status generate_bartlett_window_onemkl(std::span<F32> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0f;
      return Status::Success;
    }
    IppStatus status
        = ippsWinBartlett_32f(output.data(), static_cast<int>(length));
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
    IppStatus status
        = ippsWinBartlett_64f(output.data(), static_cast<int>(length));
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
    IppStatus status
        = ippsWinBlackmanStd_32f(output.data(), static_cast<int>(length));
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
    IppStatus status
        = ippsWinBlackmanStd_64f(output.data(), static_cast<int>(length));
    return ipp_status_to_omnidsp_status(status);
  }

  // --- Flat Top (using VML) ---
  [[nodiscard]] Status generate_flattop_window_onemkl(std::span<F32> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0f;
      return Status::Success;
    }

    const MKL_LONG n_mkl = static_cast<MKL_LONG>(length);
    const float a0 = 0.21557895f;
    const float a1 = 0.41663158f;
    const float a2 = 0.277263158f;
    const float a3 = 0.083578947f;
    const float a4 = 0.006947368f;
    const float N_minus_1 = static_cast<float>(length - 1);
    const float factor
        = (N_minus_1 > 0.0f)
              ? (static_cast<float>(2.0 * std::numbers::pi_v<double>)
                 / N_minus_1)
              : 0.0f;

    std::vector<float> n_vec(length), arg1(length), arg2(length), arg3(length),
        arg4(length);
    std::vector<float> cos1(length), cos2(length), cos3(length), cos4(length);
    float start = 0.0f, step = 1.0f;

    vsRamp(n_mkl, &start, &step, n_vec.data());

    float f1 = factor, f2 = factor * 2.0f, f3 = factor * 3.0f,
          f4 = factor * 4.0f;
    vsMul(n_mkl, n_vec.data(), &f1, arg1.data());
    vsCos(n_mkl, arg1.data(), cos1.data());
    vsMul(n_mkl, n_vec.data(), &f2, arg2.data());
    vsCos(n_mkl, arg2.data(), cos2.data());
    vsMul(n_mkl, n_vec.data(), &f3, arg3.data());
    vsCos(n_mkl, arg3.data(), cos3.data());
    vsMul(n_mkl, n_vec.data(), &f4, arg4.data());
    vsCos(n_mkl, arg4.data(), cos4.data());

    vsMul(n_mkl, cos1.data(), &a1, arg1.data());
    vsMul(n_mkl, cos2.data(), &a2, arg2.data());
    vsMul(n_mkl, cos3.data(), &a3, arg3.data());
    vsMul(n_mkl, cos4.data(), &a4, arg4.data());

    vsSub(n_mkl, &a0, arg1.data(), output.data());
    vsAdd(n_mkl, output.data(), arg2.data(), output.data());
    vsSub(n_mkl, output.data(), arg3.data(), output.data());
    vsAdd(n_mkl, output.data(), arg4.data(), output.data());

    return Status::Success;
  }
  [[nodiscard]] Status generate_flattop_window_onemkl(std::span<F64> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (length == 1) {
      output[0] = 1.0;
      return Status::Success;
    }

    const MKL_LONG n_mkl = static_cast<MKL_LONG>(length);
    const double a0 = 0.21557895;
    const double a1 = 0.41663158;
    const double a2 = 0.277263158;
    const double a3 = 0.083578947;
    const double a4 = 0.006947368;
    const double N_minus_1 = static_cast<double>(length - 1);
    const double factor = (N_minus_1 > 0.0)
                              ? (2.0 * std::numbers::pi_v<double> / N_minus_1)
                              : 0.0;

    std::vector<double> n_vec(length), arg1(length), arg2(length), arg3(length),
        arg4(length);
    std::vector<double> cos1(length), cos2(length), cos3(length), cos4(length);
    double start = 0.0, step = 1.0;

    vdRamp(n_mkl, &start, &step, n_vec.data());

    double f1 = factor, f2 = factor * 2.0, f3 = factor * 3.0, f4 = factor * 4.0;
    vdMul(n_mkl, n_vec.data(), &f1, arg1.data());
    vdCos(n_mkl, arg1.data(), cos1.data());
    vdMul(n_mkl, n_vec.data(), &f2, arg2.data());
    vdCos(n_mkl, arg2.data(), cos2.data());
    vdMul(n_mkl, n_vec.data(), &f3, arg3.data());
    vdCos(n_mkl, arg3.data(), cos3.data());
    vdMul(n_mkl, n_vec.data(), &f4, arg4.data());
    vdCos(n_mkl, arg4.data(), cos4.data());

    vdMul(n_mkl, cos1.data(), &a1, arg1.data());
    vdMul(n_mkl, cos2.data(), &a2, arg2.data());
    vdMul(n_mkl, cos3.data(), &a3, arg3.data());
    vdMul(n_mkl, cos4.data(), &a4, arg4.data());

    vdSub(n_mkl, &a0, arg1.data(), output.data());
    vdAdd(n_mkl, output.data(), arg2.data(), output.data());
    vdSub(n_mkl, output.data(), arg3.data(), output.data());
    vdAdd(n_mkl, output.data(), arg4.data(), output.data());

    return Status::Success;
  }

  // --- Gaussian (using VML) ---
  [[nodiscard]] Status generate_gaussian_window_onemkl(
      double stddev, std::span<F32> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (stddev <= 0.0) return Status::InvalidArgument;
    if (length == 1) {
      output[0] = 1.0f;
      return Status::Success;
    }

    const MKL_LONG n_mkl = static_cast<MKL_LONG>(length);
    const float N_minus_1 = static_cast<float>(length - 1);
    const float center = N_minus_1 / 2.0f;
    const float sigma_term = static_cast<float>(stddev) * center;
    if (sigma_term == 0.0f) return Status::Failure;
    const float factor = -0.5f / (sigma_term * sigma_term);

    std::vector<float> n_vec(length), temp(length);
    float start = 0.0f, step = 1.0f, neg_center = -center;

    vsRamp(n_mkl, &start, &step, n_vec.data());
    vsAdd(n_mkl, n_vec.data(), &neg_center, temp.data());
    vsSqr(n_mkl, temp.data(), temp.data());
    vsMul(n_mkl, temp.data(), &factor, temp.data());
    vsExp(n_mkl, temp.data(), output.data());

    return Status::Success;
  }
  [[nodiscard]] Status generate_gaussian_window_onemkl(
      double stddev, std::span<F64> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    if (stddev <= 0.0) return Status::InvalidArgument;
    if (length == 1) {
      output[0] = 1.0;
      return Status::Success;
    }

    const MKL_LONG n_mkl = static_cast<MKL_LONG>(length);
    const double N_minus_1 = static_cast<double>(length - 1);
    const double center = N_minus_1 / 2.0;
    const double sigma_term = stddev * center;
    if (sigma_term == 0.0) return Status::Failure;
    const double factor = -0.5 / (sigma_term * sigma_term);

    std::vector<double> n_vec(length), temp(length);
    double start = 0.0, step = 1.0, neg_center = -center;

    vdRamp(n_mkl, &start, &step, n_vec.data());
    vdAdd(n_mkl, n_vec.data(), &neg_center, temp.data());
    vdSqr(n_mkl, temp.data(), temp.data());
    vdMul(n_mkl, temp.data(), &factor, temp.data());
    vdExp(n_mkl, temp.data(), output.data());

    return Status::Success;
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
    IppStatus status
        = ippsWinHamming_32f(output.data(), static_cast<int>(length));
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
    IppStatus status
        = ippsWinHamming_64f(output.data(), static_cast<int>(length));
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
    IppStatus status = ippsWinHann_32f(output.data(), static_cast<int>(length));
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
    IppStatus status = ippsWinHann_64f(output.data(), static_cast<int>(length));
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
    IppStatus status
        = ippsWinKaiser_32f(output.data(), static_cast<int>(length), alpha);
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
    IppStatus status
        = ippsWinKaiser_64f(output.data(), static_cast<int>(length), alpha);
    return ipp_status_to_omnidsp_status(status);
  }

  // --- Rectangular ---
  [[nodiscard]] Status generate_rectangular_window_onemkl(std::span<F32> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    IppStatus status
        = ippsSet_32f(1.0f, output.data(), static_cast<int>(length));
    return ipp_status_to_omnidsp_status(status);
  }
  [[nodiscard]] Status generate_rectangular_window_onemkl(std::span<F64> output)
  {
    const size_t length = output.size();
    if (length == 0) return Status::Success;
    IppStatus status
        = ippsSet_64f(1.0, output.data(), static_cast<int>(length));
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

#endif  // OMNIDSP_USE_ONEMKL
