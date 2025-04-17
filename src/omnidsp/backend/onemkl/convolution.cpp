/**
 * @file src/omnidsp/backend/onemkl/convolution.cpp
 * @brief Intel oneMKL backend implementation for OmniDSP
 * convolution/correlation.
 *
 * Implements the convolve1d_impl function using Intel oneMKL VSL routines.
 * Compiled only when USE_ONEMKL is defined.
 */

// --- Includes ---
#include <complex>  // Although not used directly, often included with DSP headers
#include <memory>     // For smart pointers if needed (not directly here)
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <string>
#include <type_traits>  // For std::is_same_v
#include <vector>

#include "../backend_impl.h"  // Internal backend function declarations

// Only compile if USE_ONEMKL is defined by CMake
#if defined(USE_ONEMKL)

#include <mkl.h>      // Main MKL header
#include <mkl_vsl.h>  // VSL header (for convolution/correlation)

namespace OmniDSP {
namespace Backend {

// --- MKL VSL Helper ---
// Checks the status code returned by MKL VSL functions and throws an error if
// it's not VSL_STATUS_OK.
inline void check_mkl_status(int status, const char *error_msg) {
  if (status != VSL_STATUS_OK) {
    std::string full_msg = error_msg;
    // Append MKL VSL status code for detailed diagnostics.
    full_msg += ": MKL VSL Status " + std::to_string(status);
    // Note: VSL doesn't have a direct equivalent of DftiErrorMessage.
    // You might need to consult MKL documentation for status code meanings.
    throw std::runtime_error(full_msg);
  }
}

/**
 * @brief MKL backend implementation for 1D convolution or correlation using
 * VSL. Calculates the 'valid' part of the linear convolution/correlation.
 *
 * @tparam T float or double.
 * @param signal The input signal vector.
 * @param kernel The kernel vector.
 * @param use_correlation If true, perform correlation; otherwise, perform
 * convolution.
 * @return std::vector<T> The result vector.
 * @throws std::invalid_argument If inputs are invalid (empty, or kernel >
 * signal).
 * @throws std::runtime_error If an MKL VSL error occurs.
 */
template <typename T>
std::vector<T> convolve1d_impl(const std::vector<T> &signal,
                               const std::vector<T> &kernel,
                               bool use_correlation) {
  // --- Input Validation ---
  if (signal.empty() || kernel.empty()) {
    // Return empty vector for empty inputs, consistent with public API
    // expectation
    return {};
  }

  // VSL uses 'int' for dimensions
  int nx = static_cast<int>(kernel.size());  // Kernel length (MKL xlen)
  int ny = static_cast<int>(signal.size());  // Signal length (MKL ylen)
  // Calculate output length for 'valid' mode equivalent
  int nz = ny - nx + 1;  // Result length (MKL zlen)

  if (nz <= 0) {
    throw std::invalid_argument(
        "MKL Conv/Corr ('valid' mode equivalent): signal length (" +
        std::to_string(ny) + ") must be >= kernel length (" +
        std::to_string(nx) + ").");
  }

  std::vector<T> result(nz);  // Allocate result vector

  // --- MKL VSL Setup ---
  const int mode = VSL_CONV_MODE_AUTO;  // Let MKL choose FFT or direct
  VSLConvTaskPtr task = nullptr;        // Task descriptor
  int status = VSL_STATUS_OK;           // Status variable

  // Strides are typically 1 for contiguous std::vector data
  MKL_INT stride_kernel = 1, stride_signal = 1, stride_result = 1;
  // Start index for 'valid' mode equivalent (depends on conv vs corr)
  // For convolution (use_correlation=false), start index is nx-1.
  // For correlation (use_correlation=true), start index is 0.
  MKL_INT start_index = use_correlation ? 0 : (nx - 1);

  // Flags to clarify which MKL function branch to take
  bool call_mkl_convolution = !use_correlation;
  bool call_mkl_correlation = use_correlation;

  try {
    // --- Create and Execute VSL Task ---
    if constexpr (std::is_same_v<T, float>) {  // SINGLE PRECISION
      if (call_mkl_convolution) {
        // Create convolution task
        status = vslsConvNewTask1D(&task, mode, nx, ny, nz);
        check_mkl_status(status, "MKL vslsConvNewTask1D failed");
        // Set start index for 'valid' mode
        status = vslConvSetStart(task, &start_index);
        check_mkl_status(status, "MKL vslConvSetStart failed");
        // Execute convolution
        status =
            vslsConvExec(task, kernel.data(), &stride_kernel, signal.data(),
                         &stride_signal, result.data(), &stride_result);
        check_mkl_status(status, "MKL vslsConvExec failed");
      } else {  // Correlation
        // Create correlation task
        status = vslsCorrNewTask1D(&task, mode, nx, ny, nz);
        check_mkl_status(status, "MKL vslsCorrNewTask1D failed");
        // Set start index for 'valid' mode
        status = vslCorrSetStart(task, &start_index);
        check_mkl_status(status, "MKL vslCorrSetStart failed");
        // Execute correlation
        status =
            vslsCorrExec(task, kernel.data(), &stride_kernel, signal.data(),
                         &stride_signal, result.data(), &stride_result);
        check_mkl_status(status, "MKL vslsCorrExec failed");
      }
    } else {  // DOUBLE PRECISION (T = double)
      if (call_mkl_convolution) {
        // Create convolution task
        status = vsldConvNewTask1D(&task, mode, nx, ny, nz);
        check_mkl_status(status, "MKL vsldConvNewTask1D failed");
        // Set start index for 'valid' mode
        status = vslConvSetStart(task, &start_index);
        check_mkl_status(status, "MKL vslConvSetStart failed");
        // Execute convolution
        status =
            vsldConvExec(task, kernel.data(), &stride_kernel, signal.data(),
                         &stride_signal, result.data(), &stride_result);
        check_mkl_status(status, "MKL vsldConvExec failed");
      } else {  // Correlation
        // Create correlation task
        status = vsldCorrNewTask1D(&task, mode, nx, ny, nz);
        check_mkl_status(status, "MKL vsldCorrNewTask1D failed");
        // Set start index for 'valid' mode
        status = vslCorrSetStart(task, &start_index);
        check_mkl_status(status, "MKL vslCorrSetStart failed");
        // Execute correlation
        status =
            vsldCorrExec(task, kernel.data(), &stride_kernel, signal.data(),
                         &stride_signal, result.data(), &stride_result);
        check_mkl_status(status, "MKL vsldCorrExec failed");
      }
    }

    // --- Clean up VSL Task ---
    if (task) {
      status = vslConvDeleteTask(
          &task);  // Use vslConvDeleteTask for both conv/corr tasks
      check_mkl_status(status, "MKL vslConvDeleteTask failed");
      task = nullptr;  // Prevent double deletion in case of exception below
    }

    return result;

  } catch (...) {
    // Ensure task is deleted even if an exception occurs during
    // checks/execution
    if (task) {
      (void)vslConvDeleteTask(&task);  // Ignore status in exception handler
    }
    throw;  // Re-throw the original exception
  }
}

// --- Explicit Template Instantiations ---
// Ensures the compiler generates code for both float and double versions.
template std::vector<float> convolve1d_impl<float>(const std::vector<float> &,
                                                   const std::vector<float> &,
                                                   bool);
template std::vector<double> convolve1d_impl<double>(
    const std::vector<double> &, const std::vector<double> &, bool);

}  // namespace Backend
}  // namespace OmniDSP

#endif  // USE_ONEMKL
