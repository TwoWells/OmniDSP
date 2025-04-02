/**
 * @file cqt.cpp
 * @brief Example usage of the OmniDSP CQTPlan class.
 *
 * Demonstrates performing a Constant Q Transform using the efficient
 * frequency-domain implementation.
 */

#include <cmath>
#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>
#include <algorithm> // For std::min
#include <OmniDSP/omnidsp.h>
#include <OmniDSP/windows.h>
#include <OmniDSP/cqt.h> // Include the CQT header


int main() {
    using Real = double;
    using Complex = std::complex<Real>;

    std::cout << "OmniDSP CQT Example" << std::endl;
    std::cout << "-------------------" << std::endl;

    try {
        // ==========================================================
        // CQT Example
        // ==========================================================
        std::cout << "--- Setting up and Running CQT ---" << std::endl;

        // CQT parameters
        double cqt_sample_rate = 44100.0;
        double cqt_lowest_freq = 55.0;    // A1 note
        double cqt_highest_freq = 1760.0; // A6 note
        int cqt_bins_per_octave = 12;     // Semitones

        // Define the window function to be used (Hann in this case)
        auto hann_window_func = [](const std::vector<Real>& win_in) {
            // This lambda adapts the static Window::hann function
            // The CQTPlan constructor will call this with appropriate lengths internally
             return OmniDSP::Window::hann(win_in);
        };

        // Create CQTPlan
        std::cout << "Creating CQT Plan..." << std::endl;
        OmniDSP::CQTPlan<Real> cqt_plan(cqt_sample_rate, cqt_lowest_freq,
                                        cqt_highest_freq, cqt_bins_per_octave,
                                        hann_window_func);
        std::cout << "CQT Plan created (Num Bins: " << cqt_plan.getNumBins()
                  << ", Internal FFT Length: " << cqt_plan.getFFTLength() << ")" << std::endl;


        // Generate a test signal (e.g., a simple chord A3 + A4)
        size_t cqt_signal_len = 8192; // Longer signal for better CQT resolution
        std::cout << "Generating test signal (Length: " << cqt_signal_len << ")..." << std::endl;
        std::vector<Real> cqt_input_signal(cqt_signal_len);
        double freq1 = 220.0; // A3
        double freq2 = 440.0; // A4
        for (size_t i = 0; i < cqt_input_signal.size(); ++i) {
            double time = static_cast<double>(i) / cqt_sample_rate;
            cqt_input_signal[i] = 0.5 * std::cos(2.0 * M_PI * freq1 * time) +
                                  0.5 * std::cos(2.0 * M_PI * freq2 * time);
        }

        // Execute CQT
        std::cout << "Executing CQT..." << std::endl;
        std::vector<Complex> cqt_output; // Output is now vector<Complex>
        cqt_plan.execute(cqt_input_signal, cqt_output);
        std::cout << "CQT Execution complete." << std::endl;


        // Print some CQT output (magnitudes for first few bins)
        size_t num_bins_to_print = std::min((size_t)15, cqt_output.size());
        std::cout << "\nCQT Output Magnitudes (first " << num_bins_to_print << " bins of " << cqt_output.size() << "):" << std::endl;
        for (size_t i = 0; i < num_bins_to_print; ++i) {
             // Print magnitude of the single complex coefficient cqt_output[i]
             // Calculate approximate center frequency for context
             double bin_freq = cqt_plan.getLowestFrequency() * std::pow(2.0, static_cast<double>(i) / cqt_plan.getBinsPerOctave());
             std::cout << "Bin " << i << " (~" << static_cast<int>(std::round(bin_freq)) << " Hz): " << std::abs(cqt_output[i]) << std::endl;
        }
        if (cqt_output.size() > num_bins_to_print) std::cout << "..." << std::endl;


    } catch (const std::exception& e) {
        std::cerr << "\nFATAL ERROR during CQT operation: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nOmniDSP CQT Example Finished Successfully." << std::endl;
    return 0;
}