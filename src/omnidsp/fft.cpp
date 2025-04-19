/**
 * @file fft.cpp
 * @brief Implementation file for FFT convenience functions.
 * Delegates plan creation and execution to backend implementations.
 * Includes definitions for both out-of-place and in-place overloads.
 */

// --- Includes ---
// Best practice: Include the corresponding header first.
#include "OmniDSP/fft.h"  // Public header declarations (defines FFTNorm etc.)

// Include other necessary standard library headers
#include <complex>
#include <memory>       // For std::unique_ptr (used indirectly by Plans)
#include <stdexcept>    // For std::runtime_error, std::invalid_argument
#include <string>       // For exception messages
#include <type_traits>  // For std::is_same_v
#include <vector>

// Include necessary OmniDSP headers
#include <OmniDSP/core_types.h>  // For Precision enum

// NOTE: We don't include backend_impl.h here. The Plan classes handle backend
// interaction.

namespace OmniDSP {

//--------------------------------------------------------------------------
// Convenience Functions (Stateless) Implementation
//--------------------------------------------------------------------------
// These create temporary plans internally using the public Plan constructors.

// --- Out-of-Place Functions ---

template <typename T>
void fft(const std::vector<std::complex<T>>& input,
         std::vector<std::complex<T>>& output, FFTNorm norm) {
  if (input.empty()) {
    output.clear();
    return;  // Handle empty input
  }
  size_t size = input.size();
  // Ensure output is correct size (allow resizing)
  if (output.size() != size) {
    try {
      output.resize(size);
    } catch (const std::bad_alloc& e) {
      throw std::runtime_error("Failed to resize output vector in fft(oop): " +
                               std::string(e.what()));
    }
  }

  // Determine precision from type T
  Precision prec =
      std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;

  // Create a temporary plan for forward FFT
  try {
    // Use the multi-parameter constructor declared in fft.h
    FFTPlan<T> plan(size, prec, Direction::FORWARD, Domain::COMPLEX, norm);
    plan.fft(input, output);  // Call the plan's method
  } catch (const std::exception& e) {
    // Catch potential errors during plan creation or execution
    throw std::runtime_error("Convenience fft (oop) failed: " +
                             std::string(e.what()));
  }
}

template <typename T>
void ifft(const std::vector<std::complex<T>>& input,
          std::vector<std::complex<T>>& output, FFTNorm norm) {
  if (input.empty()) {
    output.clear();
    return;
  }
  size_t size = input.size();
  // Ensure output is correct size (allow resizing)
  if (output.size() != size) {
    try {
      output.resize(size);
    } catch (const std::bad_alloc& e) {
      throw std::runtime_error("Failed to resize output vector in ifft(oop): " +
                               std::string(e.what()));
    }
  }

  Precision prec =
      std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;

  // Create a temporary plan for inverse FFT
  try {
    // Use the multi-parameter constructor declared in fft.h
    FFTPlan<T> plan(size, prec, Direction::INVERSE, Domain::COMPLEX, norm);
    plan.ifft(input, output);  // Call the plan's method
  } catch (const std::exception& e) {
    throw std::runtime_error("Convenience ifft (oop) failed: " +
                             std::string(e.what()));
  }
}

template <typename T>
void rfft(const std::vector<T>& input, std::vector<std::complex<T>>& output,
          FFTNorm norm) {
  if (input.empty()) {
    output.clear();
    return;
  }
  size_t size = input.size();
  size_t expected_output_size = size / 2 + 1;
  // Ensure output is correct size (allow resizing)
  if (output.size() != expected_output_size) {
    try {
      output.resize(expected_output_size);
    } catch (const std::bad_alloc& e) {
      throw std::runtime_error("Failed to resize output vector in rfft: " +
                               std::string(e.what()));
    }
  }

  Precision prec =
      std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;

  // Create a temporary plan for forward RFFT
  try {
    // Use the multi-parameter constructor declared in fft.h
    RFFTPlan<T> plan(size, prec, Direction::FORWARD, Domain::REAL, norm);
    plan.rfft(input, output);  // Call the plan's method
  } catch (const std::exception& e) {
    throw std::runtime_error("Convenience rfft failed: " +
                             std::string(e.what()));
  }
}

template <typename T>
void irfft(const std::vector<std::complex<T>>& input, std::vector<T>& output,
           size_t N,  // N = target real output size
           FFTNorm norm) {
  if (N == 0) {  // Handle N=0 case
    output.clear();
    if (!input.empty()) {
      throw std::invalid_argument(
          "irfft: Input must be empty if target size N is 0.");
    }
    return;
  }
  size_t expected_input_size = N / 2 + 1;
  if (input.size() != expected_input_size) {
    throw std::invalid_argument("irfft: Input vector size (" +
                                std::to_string(input.size()) +
                                ") does not match expected size (" +
                                std::to_string(expected_input_size) +
                                ") for output size N=" + std::to_string(N));
  }
  // Ensure output is correct size (allow resizing)
  if (output.size() != N) {
    try {
      output.resize(N);
    } catch (const std::bad_alloc& e) {
      throw std::runtime_error("Failed to resize output vector in irfft: " +
                               std::string(e.what()));
    }
  }

  Precision prec =
      std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;

  // Create a temporary plan for inverse IRFFT
  try {
    // Use the multi-parameter constructor declared in fft.h
    RFFTPlan<T> plan(N, prec, Direction::INVERSE, Domain::REAL, norm);
    plan.irfft(input, output);  // Call the plan's method
  } catch (const std::exception& e) {
    throw std::runtime_error("Convenience irfft failed: " +
                             std::string(e.what()));
  }
}

// --- In-Place Functions (Overloads) --- ADDED DEFINITIONS ---

template <typename T>
void fft(std::vector<std::complex<T>>& data, FFTNorm norm) {
  if (data.empty()) {
    return;  // Handle empty input
  }
  size_t size = data.size();
  Precision prec =
      std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;

  // Create a temporary plan for forward FFT
  try {
    // Simulate in-place using the out-of-place plan method
    FFTPlan<T> plan(size, prec, Direction::FORWARD, Domain::COMPLEX, norm);
    // Create a temporary copy for the input to the OOP call
    std::vector<std::complex<T>> temp_input = data;
    plan.fft(temp_input, data);  // Execute OOP, writing result back into 'data'
  } catch (const std::exception& e) {
    // Catch potential errors during plan creation or execution
    throw std::runtime_error("Convenience fft (in-place) failed: " +
                             std::string(e.what()));
  }
}

template <typename T>
void ifft(std::vector<std::complex<T>>& data, FFTNorm norm) {
  if (data.empty()) {
    return;  // Handle empty input
  }
  size_t size = data.size();
  Precision prec =
      std::is_same_v<T, float> ? Precision::SINGLE : Precision::DOUBLE;

  // Create a temporary plan for inverse FFT
  try {
    // Simulate in-place using the out-of-place plan method
    FFTPlan<T> plan(size, prec, Direction::INVERSE, Domain::COMPLEX, norm);
    // Create a temporary copy for the input to the OOP call
    std::vector<std::complex<T>> temp_input = data;
    plan.ifft(temp_input,
              data);  // Execute OOP, writing result back into 'data'
  } catch (const std::exception& e) {
    // Catch potential errors during plan creation or execution
    throw std::runtime_error("Convenience ifft (in-place) failed: " +
                             std::string(e.what()));
  }
}

//--------------------------------------------------------------------------
// Explicit Template Instantiations (ALL Convenience Functions)
//--------------------------------------------------------------------------

// Out-of-Place Convenience Functions
template void fft<float>(const std::vector<std::complex<float>>&,
                         std::vector<std::complex<float>>&, FFTNorm);
template void fft<double>(const std::vector<std::complex<double>>&,
                          std::vector<std::complex<double>>&, FFTNorm);
template void ifft<float>(const std::vector<std::complex<float>>&,
                          std::vector<std::complex<float>>&, FFTNorm);
template void ifft<double>(const std::vector<std::complex<double>>&,
                           std::vector<std::complex<double>>&, FFTNorm);
template void rfft<float>(const std::vector<float>&,
                          std::vector<std::complex<float>>&, FFTNorm);
template void rfft<double>(const std::vector<double>&,
                           std::vector<std::complex<double>>&, FFTNorm);
template void irfft<float>(const std::vector<std::complex<float>>&,
                           std::vector<float>&, size_t, FFTNorm);
template void irfft<double>(const std::vector<std::complex<double>>&,
                            std::vector<double>&, size_t, FFTNorm);

// In-Place Convenience Function Overloads (Corrected Instantiations)
template void fft<float>(std::vector<std::complex<float>>& data, FFTNorm norm);
template void fft<double>(std::vector<std::complex<double>>& data,
                          FFTNorm norm);
template void ifft<float>(std::vector<std::complex<float>>& data, FFTNorm norm);
template void ifft<double>(std::vector<std::complex<double>>& data,
                           FFTNorm norm);

}  // namespace OmniDSP
