/**
 * @file windows.cpp
 * @brief Example usage of the OmniDSP Window class.
 *
 * Demonstrates applying different window functions to a signal.
 * Updated includes after header refactoring.
 */

#include <OmniDSP/omnidsp.hpp>
#include <OmniDSP/window.hpp>
#include <cmath>    // For sin
#include <iomanip>  // For std::setprecision
#include <iostream>
#include <vector>

// Helper to print vector contents
template <typename T>
void print_vector(const std::string& name, const std::vector<T>& vec)
{
  std::cout << name << ": [";
  bool first = true;
  for (const auto& val : vec) {
    if (!first) std::cout << ", ";
    std::cout << std::fixed << std::setprecision(4) << val;
    first = false;
  }
  std::cout << "]" << std::endl;
}

int main()
{
  using Real = double;  // Use double for examples

  std::cout << "OmniDSP Windowing Example" << std::endl;
  std::cout << "-------------------------" << std::endl;

  try {
    // Generate a simple sine wave signal
    size_t signal_length = 10;
    std::vector<Real> input_signal(signal_length);
    double freq = 1.0;  // Arbitrary frequency for shape
    double sample_rate
        = static_cast<double>(signal_length);  // Make one cycle fit
    for (size_t i = 0; i < signal_length; ++i) {
      input_signal[i] = std::sin(
          2.0 * std::numbers::pi * freq * static_cast<double>(i) / sample_rate);
    }
    print_vector("Input Signal ", input_signal);
    std::cout << std::endl;

    // Apply Hann window
    std::cout << "Applying Hann Window..." << std::endl;
    std::vector<Real> hann_output = OmniDSP::Window::hann(input_signal);
    print_vector("Hann Output  ", hann_output);
    std::cout << std::endl;

    // Apply Hamming window
    std::cout << "Applying Hamming Window..." << std::endl;
    std::vector<Real> hamming_output = OmniDSP::Window::hamming(input_signal);
    print_vector("Hamming Output", hamming_output);
    std::cout << std::endl;

    // Apply Kaiser window
    Real kaiser_beta = 8.6;  // Common beta value
    std::cout << "Applying Kaiser Window (beta=" << kaiser_beta << ")..."
              << std::endl;
    std::vector<Real> kaiser_output
        = OmniDSP::Window::kaiser(input_signal, kaiser_beta);
    print_vector("Kaiser Output ", kaiser_output);
    std::cout << std::endl;

    // Apply Flattop window
    std::cout << "Applying Flattop Window..." << std::endl;
    std::vector<Real> flattop_output = OmniDSP::Window::flattop(input_signal);
    print_vector("Flattop Output", flattop_output);
    std::cout << std::endl;
  }
  catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "OmniDSP Windowing Example Finished Successfully." << std::endl;
  return 0;
}
