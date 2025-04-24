/**
 * @file fft.cpp
 * @brief Example usage of OmniDSP for Complex-to-Complex FFTs.
 *
 * Demonstrates C2C transforms using the FFTPlan class (OOP and IP)
 * and convenience functions (OOP and IP). Uses Backward normalization.
 */

#include <OmniDSP/omnidsp.h>  // Includes necessary headers like fft.h, core_types.h

#include <algorithm>  // For std::min, std::max
#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <vector>

int main()
{
  // --- Configuration ---
  const size_t N = 16;  // FFT Size
  using Real = double;
  using Complex = std::complex<Real>;
  const auto CURRENT_PRECISION
      = OmniDSP::Precision::Double;  // Assuming Precision::Double is defined

  std::cout << "OmniDSP C2C FFT Example (N=" << N << ")" << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  // --- Prepare Input Data ---
  std::vector<Complex> input_signal_c2c(N);
  double frequency = 3.0;  // Example frequency for FFT input
  for (size_t i = 0; i < N; ++i) {
    Real real_val = std::cos(
        2.0 * std::numbers::pi * frequency * static_cast<Real>(i) / N);
    Real imag_val = std::sin(
        1.5 * std::numbers::pi * frequency * static_cast<Real>(i)
        / N);  // Different frequency for imag part
    input_signal_c2c[i] = Complex(real_val, imag_val);
  }

  std::cout << "Input Complex Signal:" << std::endl;
  for (size_t i = 0; i < std::min((size_t)5, N); ++i)
    std::cout << input_signal_c2c[i] << " ";
  if (N > 5) std::cout << "...";
  std::cout << std::endl << std::endl;

  // --- Execute DSP functions within a try-catch block ---
  try {
    // ==========================================================
    // Method 1: Using FFTPlan for C2C (Out-of-Place)
    // ==========================================================
    std::cout << "--- Method 1: Using FFTPlan (C2C OOP) ---" << std::endl;
    OmniDSP::FFTPlan<Real> forward_plan_c2c(
        N,
        CURRENT_PRECISION,
        OmniDSP::Direction::Forward,
        OmniDSP::Domain::Complex,
        OmniDSP::FFTNorm::Backward);  // Using Backward norm as specified
    OmniDSP::FFTPlan<Real> inverse_plan_c2c(
        N,
        CURRENT_PRECISION,
        OmniDSP::Direction::Inverse,
        OmniDSP::Domain::Complex,
        OmniDSP::FFTNorm::Backward);  // Using Backward norm as specified

    std::vector<Complex> spectrum_c2c_plan(N);
    std::vector<Complex> reconstructed_c2c_plan(N);

    // CORRECTED: Use .fft() method and pass vectors by reference
    // forward_plan_c2c.execute(input_signal_c2c.data(),
    // spectrum_c2c_plan.data()); // Incorrect
    forward_plan_c2c.fft(input_signal_c2c, spectrum_c2c_plan);  // Correct

    std::cout << "Spectrum (Plan OOP):" << std::endl;
    for (size_t i = 0; i < std::min((size_t)5, N); ++i)
      std::cout << spectrum_c2c_plan[i] << " ";
    if (N > 5) std::cout << "...";
    std::cout << std::endl;

    // CORRECTED: Use .ifft() method and pass vectors by reference
    // inverse_plan_c2c.execute(spectrum_c2c_plan.data(),
    // reconstructed_c2c_plan.data()); // Incorrect
    inverse_plan_c2c.ifft(
        spectrum_c2c_plan,
        reconstructed_c2c_plan);  // Correct

    std::cout << "\nReconstructed (Plan OOP):" << std::endl;
    for (size_t i = 0; i < std::min((size_t)5, N); ++i)
      std::cout << reconstructed_c2c_plan[i] << " ";
    if (N > 5) std::cout << "...";
    std::cout << std::endl;

    Real max_diff_oop = 0.0;
    for (size_t i = 0; i < N; ++i)
      max_diff_oop = std::max(
          max_diff_oop,
          std::abs(input_signal_c2c[i] - reconstructed_c2c_plan[i]));
    std::cout << "Max Difference (OOP): " << max_diff_oop << std::endl
              << std::endl;

    // ==========================================================
    // Method 2: Using Convenience Functions (C2C OOP)
    // ==========================================================
    std::cout << "--- Method 2: Using Convenience Functions (C2C OOP) ---"
              << std::endl;
    std::vector<Complex> spectrum_conv_c2c;  // Output vector will be resized
    std::vector<Complex>
        reconstructed_conv_c2c;  // Output vector will be resized

    // Assuming convenience functions take vector references and handle norm
    // correctly
    OmniDSP::fft(
        input_signal_c2c,
        spectrum_conv_c2c,
        OmniDSP::FFTNorm::Backward);  // Specify norm if needed
    OmniDSP::ifft(
        spectrum_conv_c2c,
        reconstructed_conv_c2c,
        OmniDSP::FFTNorm::Backward);  // Specify norm if needed

    std::cout << "Spectrum (Convenience Func):" << std::endl;
    for (size_t i = 0; i < std::min((size_t)5, N); ++i)
      std::cout << spectrum_conv_c2c[i] << " ";
    if (N > 5) std::cout << "...";
    std::cout << std::endl;

    std::cout << "\nReconstructed (Convenience Func):" << std::endl;
    for (size_t i = 0; i < std::min((size_t)5, N); ++i)
      std::cout << reconstructed_conv_c2c[i] << " ";
    if (N > 5) std::cout << "...";
    std::cout << std::endl << std::endl;

    // ==========================================================
    // Method 3: Using In-place Transform (C2C)
    // ==========================================================
    std::cout << "--- Method 3: Using In-place Transform (C2C) ---"
              << std::endl;
    std::vector<Complex> data_inplace_c2c
        = input_signal_c2c;  // Copy input data

    // Assuming in-place functions exist and take vector references
    // Also assuming they use a default normalization or have overloads to
    // specify it. Add normalization argument if required by the function
    // signature.
    OmniDSP::fft(
        data_inplace_c2c /*, OmniDSP::FFTNorm::Backward */);  // Add norm if
                                                              // needed
    std::cout << "Spectrum (In-place Forward):" << std::endl;
    for (size_t i = 0; i < std::min((size_t)5, N); ++i)
      std::cout << data_inplace_c2c[i] << " ";
    if (N > 5) std::cout << "...";
    std::cout << std::endl;

    OmniDSP::ifft(
        data_inplace_c2c /*, OmniDSP::FFTNorm::Backward */);  // Add norm if
                                                              // needed
    std::cout << "\nReconstructed (In-place Inverse):" << std::endl;
    for (size_t i = 0; i < std::min((size_t)5, N); ++i)
      std::cout << data_inplace_c2c[i] << " ";
    if (N > 5) std::cout << "...";
    std::cout << std::endl;

    Real max_diff_ip = 0.0;
    for (size_t i = 0; i < N; ++i)
      max_diff_ip = std::max(
          max_diff_ip, std::abs(input_signal_c2c[i] - data_inplace_c2c[i]));
    std::cout << "Max Difference (In-Place): " << max_diff_ip << std::endl
              << std::endl;
  }
  catch (const std::exception& e) {
    std::cerr << "\nFATAL ERROR during C2C FFT operation: " << e.what()
              << std::endl;
    return 1;
  }

  std::cout << "OmniDSP C2C FFT Example Finished Successfully." << std::endl;
  return 0;
}
