/**
 * @file window.cpp
 * @brief Implementation file for windowing functions in OmniDSP.
 *
 * Implements the static methods declared in OmniDSP::Window (window.h).
 * It dispatches calls to the appropriate backend implementation.
 * Also implements the coefficient generation methods.
 *
 * @version 2.1.1 (Fix template issue in helper)
 * @date 2025-04-18
 */

#include "OmniDSP/window.h"  // Class declaration

#include <iostream>  // For potential debug/error messages
#include <numeric>  // For std::iota (potentially useful but not currently used)
#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::string
#include <vector>

#include "backend/backend_impl.h"  // Backend function declarations

namespace OmniDSP {

//-----------------------------------------------------------------------------
// Internal Helper (if needed, e.g., for validation)
//-----------------------------------------------------------------------------
namespace {  // Anonymous namespace for internal linkage

// Helper to validate input vector for applying functions
template <typename T>
void validate_input(const std::vector<T>& input, const char* func_name) {
  if (input.empty()) {
    throw std::invalid_argument(std::string(func_name) +
                                ": Input vector cannot be empty.");
  }
}

// Helper to validate parameters for coefficient generation requests
// This function does NOT depend on the template type T.
void validate_coeffs_request(
    size_t length,
    const char* func_name) {  // <<< CORRECTED: Removed template<typename T>
  // Length 0 is handled by returning empty vector in the calling function.
  // size_t prevents negative lengths, so no check needed here.
  // Add other checks if necessary (e.g., max length).
}

// Helper to validate parameters for Kaiser window coefficient generation
// This function DOES depend on T for the beta parameter.
template <typename T>
void validate_kaiser_coeffs_params(size_t length, T beta,
                                   const char* func_name) {
  validate_coeffs_request(length,
                          func_name);  // Calls the non-template version now
  if (beta < static_cast<T>(0.0)) {
    throw std::invalid_argument(
        std::string(func_name) +
        ": Kaiser window beta parameter cannot be negative.");
  }
}

// Helper to validate parameters for applying Kaiser window
template <typename T>
void validate_kaiser_params(const std::vector<T>& input, T beta,
                            const char* func_name) {
  validate_input(input, func_name);  // Validate input vector first
  if (beta < static_cast<T>(0.0)) {
    throw std::invalid_argument(
        std::string(func_name) +
        ": Kaiser window beta parameter cannot be negative.");
  }
}

}  // anonymous namespace

//-----------------------------------------------------------------------------
// Public Static Methods - APPLYING Windows
//-----------------------------------------------------------------------------

template <typename T>
std::vector<T> Window::hann(const std::vector<T>& input) {
  validate_input(input, "Window::hann");
  size_t length = input.size();
  std::vector<T> output(length);
  // Assuming backend impl generates coefficients into output
  Backend::hann_window_impl(output.data(), length);
  // Apply the window element-wise
  for (size_t i = 0; i < length; ++i) {
    output[i] *= input[i];
  }
  return output;
}

template <typename T>
std::vector<T> Window::hamming(const std::vector<T>& input) {
  validate_input(input, "Window::hamming");
  size_t length = input.size();
  std::vector<T> output(length);
  Backend::hamming_window_impl(output.data(), length);
  for (size_t i = 0; i < length; ++i) {
    output[i] *= input[i];
  }
  return output;
}

template <typename T>
std::vector<T> Window::kaiser(const std::vector<T>& input, T beta) {
  validate_kaiser_params(input, beta,
                         "Window::kaiser");  // Use specific Kaiser validator
  size_t length = input.size();
  std::vector<T> output(length);
  Backend::kaiser_window_impl(output.data(), length, beta);
  for (size_t i = 0; i < length; ++i) {
    output[i] *= input[i];
  }
  return output;
}

template <typename T>
std::vector<T> Window::flattop(const std::vector<T>& input) {
  validate_input(input, "Window::flattop");
  size_t length = input.size();
  std::vector<T> output(length);
  Backend::flattop_window_impl(output.data(), length);
  for (size_t i = 0; i < length; ++i) {
    output[i] *= input[i];
  }
  return output;
}

// Add implementations for Blackman, Bartlett etc. if declared in header

//-----------------------------------------------------------------------------
// Public Static Methods - GENERATING Window Coefficients
//-----------------------------------------------------------------------------

template <typename T>
std::vector<T> Window::get_hann_coeffs(size_t length) {
  validate_coeffs_request(
      length, "Window::get_hann_coeffs");  // Call non-template helper
  if (length == 0) return {};
  std::vector<T> ones(length, static_cast<T>(1.0));
  // Call applying function which implicitly generates coeffs and multiplies by
  // 1
  return Window::hann<T>(ones);
}

template <typename T>
std::vector<T> Window::get_hamming_coeffs(size_t length) {
  validate_coeffs_request(
      length, "Window::get_hamming_coeffs");  // Call non-template helper
  if (length == 0) return {};
  std::vector<T> ones(length, static_cast<T>(1.0));
  return Window::hamming<T>(ones);
}

template <typename T>
std::vector<T> Window::get_kaiser_coeffs(size_t length, T beta) {
  // Use the specific Kaiser validator which remains a template
  validate_kaiser_coeffs_params<T>(length, beta, "Window::get_kaiser_coeffs");
  if (length == 0) return {};
  std::vector<T> ones(length, static_cast<T>(1.0));
  return Window::kaiser<T>(ones, beta);
}

template <typename T>
std::vector<T> Window::get_flattop_coeffs(size_t length) {
  validate_coeffs_request(
      length, "Window::get_flattop_coeffs");  // Call non-template helper
  if (length == 0) return {};
  std::vector<T> ones(length, static_cast<T>(1.0));
  return Window::flattop<T>(ones);
}

// Add implementations for get_blackman_coeffs, get_bartlett_coeffs etc.

//-----------------------------------------------------------------------------
// Explicit Template Instantiations
//-----------------------------------------------------------------------------

// Instantiations for APPLYING methods
template std::vector<float> Window::hann(const std::vector<float>& input);
template std::vector<double> Window::hann(const std::vector<double>& input);
template std::vector<float> Window::hamming(const std::vector<float>& input);
template std::vector<double> Window::hamming(const std::vector<double>& input);
template std::vector<float> Window::kaiser(const std::vector<float>& input,
                                           float beta);
template std::vector<double> Window::kaiser(const std::vector<double>& input,
                                            double beta);
template std::vector<float> Window::flattop(const std::vector<float>& input);
template std::vector<double> Window::flattop(const std::vector<double>& input);
// Add instantiations for other applying methods if they exist

// Instantiations for GENERATING methods
template std::vector<float> Window::get_hann_coeffs(size_t length);
template std::vector<double> Window::get_hann_coeffs(size_t length);
template std::vector<float> Window::get_hamming_coeffs(size_t length);
template std::vector<double> Window::get_hamming_coeffs(size_t length);
template std::vector<float> Window::get_kaiser_coeffs(size_t length,
                                                      float beta);
template std::vector<double> Window::get_kaiser_coeffs(size_t length,
                                                       double beta);
template std::vector<float> Window::get_flattop_coeffs(size_t length);
template std::vector<double> Window::get_flattop_coeffs(size_t length);
// Add instantiations for other get_coeffs methods if they exist

}  // namespace OmniDSP
