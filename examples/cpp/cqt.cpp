/**
 * @file cqt.cpp
 * @brief Example demonstrating the Constant-Q Transform (CQT) using OmniDSP.
 */

// Ensure all potentially relevant headers are included
#include <OmniDSP/core_types.hpp>  // Defines Precision
#include <OmniDSP/cqt.hpp>         // Primary header for CQTPlan
#include <OmniDSP/fft.hpp>         // Defines FFTNorm
#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
#include <stdexcept>  // Include for std::exception
#include <vector>

// Define the Real type alias (assuming double precision for this example)
using Real = double;
using Complex = std::complex<Real>;

// --- Helper Function (Example) ---
// Generates a simple sine wave
std::vector<Real> generateSineWave(
    double freq, double sampleRate, double duration)
{
  size_t numSamples = static_cast<size_t>(sampleRate * duration);
  std::vector<Real> signal(numSamples);
  Real angularFreq = 2.0 * std::numbers::pi * freq / sampleRate;
  for (size_t i = 0; i < numSamples; ++i) {
    signal[i] = std::sin(angularFreq * i);
  }
  return signal;
}

// --- Main Function ---
int main()
{
  // --- Parameters ---
  // Example parameters for CQT calculation
  const double sr = 44100.0;  // Sample rate (Hz)
  const double fmin = 27.5;   // Minimum frequency (Hz) - A0 note
  const double fmax
      = 20000.0;  // Maximum frequency (Hz) - Approximate (Note: fmax isn't
                  // directly used in this CQTProcessor constructor)
  const int bins_per_octave
      = 12;  // Number of bins per octave (e.g., 12 for semitones)
  const double duration = 1.0;             // Duration of the signal in seconds
  const int n_bins = 7 * bins_per_octave;  // Example: 7 octaves

  // --- Generate Input Signal ---
  // Example: A sine wave at 440 Hz (A4 note)
  std::vector<Real> inputSignal = generateSineWave(440.0, sr, duration);
  std::cout << "Generated a sine wave with " << inputSignal.size()
            << " samples." << std::endl;

  // --- Create CQT Plan ---
  try {
    std::cout << "Creating CQT Plan..." << std::endl;

    // Construct the CQTProcessor object
    // Using definitions from fft.h (immersive fft_h_updated) and core_types.h:
    // - Precision is likely an enum class (OmniDSP::Precision::Double)
    // - FFTNorm is an enum class with member 'Ortho' (OmniDSP::FFTNorm::Ortho)
    OmniDSP::CQTProcessor<Real> cqtPlan(
        sr,                          // Sample rate (double)
        fmin,                        // Minimum frequency (double)
        n_bins,                      // Total number of CQT bins (int)
        bins_per_octave,             // Bins per octave (int)
        OmniDSP::Precision::Double,  // Precision enum
        OmniDSP::FFTNorm::Ortho      // FFT normalization enum (Using ALL_CAPS
                                     // version from fft.h)
    );

    std::cout << "CQT Plan created successfully." << std::endl;

    // --- Execute CQT ---
    std::cout << "Executing CQT..." << std::endl;
    // The execute method likely takes the input signal and returns the CQT
    // result. The exact return type depends on the CQTProcessor implementation
    // (e.g., std::vector<std::vector<Complex>>). This is a placeholder for the
    // actual execution call. Example: Get the expected output size first size_t
    // num_frames = cqtPlan.get_num_frames(inputSignal.size());
    // std::vector<std::vector<Complex>> cqtResult(n_bins,
    // std::vector<Complex>(num_frames)); cqtPlan.execute(inputSignal,
    // cqtResult); // Replace with actual execute signature

    // --- Process Results (Example) ---
    // std::cout << "CQT executed. Result dimensions (example): " <<
    // cqtResult.size() << " x " << (cqtResult.empty() ? 0 :
    // cqtResult[0].size()) << std::endl; Add code here to analyze or print the
    // cqtResult

    std::cout << "CQT example finished." << std::endl;
  }
  catch (const std::exception& e) {
    std::cerr << "Error during CQT processing: " << e.what() << std::endl;
    return 1;  // Indicate error
  }
  catch (...) {
    std::cerr << "An unknown error occurred during CQT processing."
              << std::endl;
    return 1;  // Indicate error
  }

  return 0;  // Indicate success
}
