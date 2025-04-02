#include <gtest/gtest.h>
#include <OmniDSP/omnidsp.h> // Use correct include path
#include <OmniDSP/cqt.h>     // Include CQT header
#include <vector>
#include <complex>
#include <limits>
#include <cmath>
#include <string> // For std::to_string in error messages

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Test Fixture ---
class FFT_CQT_Test : public ::testing::Test {
protected:
    // Test parameters
    const double sample_rate = 44100.0; // Standard audio sample rate
    const double lowest_freq = 20.0;    // Lowest human-audible frequency
    const double highest_freq = 20000.0; // Highest human-audible frequency
    const int bins_per_octave = 12;     // Typical value for music analysis
    // NOTE: Tolerances for energy checks might need adjustment after refactoring.
    const double double_tolerance_energy = 1.0; // Example: Allow +/- 1.0 energy unit initially
    const float float_tolerance_energy = 1.0f;

    // Helper function to generate a simple test signal (single frequency)
    template <typename T>
    std::vector<T> generate_test_signal(double frequency, double sample_rate, size_t length) {
        std::vector<T> signal(length);
        if (length == 0 || sample_rate <= 0) return signal; // Basic validation
        for (size_t i = 0; i < length; ++i) {
            double time = static_cast<double>(i) / sample_rate;
            signal[i] = static_cast<T>(std::cos(2.0 * M_PI * frequency * time));
        }
        return signal;
    }

    // Helper function to generate a two-frequency signal
     template <typename T>
    std::vector<T> generate_two_freq_signal(double freq1, double freq2, double sample_rate, size_t length) {
        std::vector<T> signal(length);
         if (length == 0 || sample_rate <= 0) return signal;
        for (size_t i = 0; i < length; ++i) {
            double time = static_cast<double>(i) / sample_rate;
            signal[i] = static_cast<T>(std::cos(2.0 * M_PI * freq1 * time) + std::cos(2.0 * M_PI * freq2 * time));
        }
        return signal;
    }

};

// --- Test Cases ---

TEST_F(FFT_CQT_Test, SingleFrequency_Double) {
    using T = double;
    using Complex = std::complex<T>;
    size_t signal_length = 4096; // Use a reasonable length
    double test_freq = 440.0; // A4 note

    std::vector<T> input_signal = generate_test_signal<T>(test_freq, sample_rate, signal_length);
    // **MODIFIED**: Output is now a single vector of complex coefficients
    std::vector<Complex> cqt_output;

    try {
        // Create CQTPlan with Hann window (using lambda adapter)
        auto hann_window_func = [](const std::vector<T>& input_win) {
            // Need to handle the input_win which might just be size 1 as placeholder
            // The actual window coeffs should be generated based on length needed internally
             return OmniDSP::Window::hann(input_win); // Assuming Window::hann handles this correctly or is adapted
        };
        OmniDSP::CQTPlan<T> cqt_plan(sample_rate, lowest_freq, highest_freq, bins_per_octave, hann_window_func);

        // **MODIFIED**: Call the execute method with the new signature
        cqt_plan.execute(input_signal, cqt_output);

        // --- Verification ---
        // 1. Output should have the correct number of bins
        size_t expected_bins = cqt_plan.getNumBins(); // Use getter if available, otherwise calculate again
        // size_t expected_bins_calc = static_cast<size_t>(std::ceil(bins_per_octave * std::log2(highest_freq / lowest_freq)));
        ASSERT_EQ(cqt_output.size(), expected_bins);

        // 2. Find the bin closest to the test frequency
        size_t expected_bin_index = 0;
        double min_diff = std::numeric_limits<double>::max();
        for (size_t i = 0; i < expected_bins; ++i) {
            // Calculate bin center frequency (consistent with CQTPlan internal logic)
            double bin_freq = lowest_freq * std::pow(2.0, static_cast<double>(i) / bins_per_octave);
            double diff = std::abs(bin_freq - test_freq);
            if (diff < min_diff) {
                min_diff = diff;
                expected_bin_index = i;
            }
        }
        // Add a check to ensure the found frequency is reasonably close
        EXPECT_LT(min_diff, lowest_freq * (std::pow(2.0, 1.0 / (2.0 * bins_per_octave)) - std::pow(2.0, -1.0 / (2.0 * bins_per_octave))))
            << "Could not find a CQT bin reasonably close to the target frequency " << test_freq << " Hz.";


        // 3. Check if the energy is concentrated in the expected bin
        for (size_t i = 0; i < expected_bins; ++i) {
            // **MODIFIED**: Calculate energy directly from the complex coefficient
            T energy = std::norm(cqt_output[i]); // Energy = |coefficient|^2

            if (i == expected_bin_index) {
                // **THRESHOLD NEEDS ADJUSTMENT**: Check if energy in the target bin is significant
                // The old threshold of 100.0 is likely incorrect due to normalization changes.
                 // EXPECT_GT(energy, 100.0) << "Energy in expected bin " << i << " (" << test_freq << " Hz) is lower than expected. Actual: " << energy;
                 // Example: Check if it's the maximum energy bin (or close to it)
                 T max_energy = 0.0;
                 for(const auto& coeff : cqt_output) max_energy = std::max(max_energy, std::norm(coeff));
                 EXPECT_NEAR(energy, max_energy, max_energy * 0.1) << "Energy in expected bin " << i << " not close to max energy."; // Check it's within 10% of max
            } else {
                // **THRESHOLD NEEDS ADJUSTMENT**: Check if energy in other bins is low
                // The old threshold of 10.0 is likely incorrect. Should be significantly lower than the peak.
                // EXPECT_LT(energy, 10.0) << "Energy in other bin " << i << " is higher than expected. Actual: " << energy;
                // Example: Check relative to the peak energy
                 T max_energy = 0.0;
                 for(const auto& coeff : cqt_output) max_energy = std::max(max_energy, std::norm(coeff));
                EXPECT_LT(energy, std::max(static_cast<T>(1e-9), max_energy * static_cast<T>(0.01))) // Allow very small noise floor, otherwise < 1% of peak
                     << "Energy in other bin " << i << " [" << energy << "] is too high relative to peak energy [" << max_energy << "]";

            }
        }

    } catch (const std::exception& e) {
        FAIL() << "CQT operation threw exception: " << e.what();
    }
}

TEST_F(FFT_CQT_Test, TwoFrequencies_Float) {
    using T = float;
    using Complex = std::complex<T>;
    size_t signal_length = 4096;
    double freq1 = 220.0; // A3
    double freq2 = 880.0; // A5 (two octaves higher)

    std::vector<T> input_signal = generate_two_freq_signal<T>(freq1, freq2, sample_rate, signal_length);
    // **MODIFIED**: Output is now a single vector of complex coefficients
    std::vector<Complex> cqt_output;

    try {
        // Create CQTPlan with Kaiser window (using lambda adapter)
        auto kaiser_window_func = [&](const std::vector<T>& input_win) {
             // Beta value for Kaiser window - might need tuning
            T beta = 5.0f;
            return OmniDSP::Window::kaiser(input_win, beta);
        };
        OmniDSP::CQTPlan<T> cqt_plan(sample_rate, lowest_freq, highest_freq, bins_per_octave, kaiser_window_func);

        // **MODIFIED**: Call the execute method with the new signature
        cqt_plan.execute(input_signal, cqt_output);

        // --- Verification ---
        size_t expected_bins = cqt_plan.getNumBins();
        ASSERT_EQ(cqt_output.size(), expected_bins);

        // Find bins closest to the test frequencies
        size_t bin_index1 = 0, bin_index2 = 0;
        double min_diff1 = std::numeric_limits<double>::max();
        double min_diff2 = std::numeric_limits<double>::max();
        for (size_t i = 0; i < expected_bins; ++i) {
            double bin_freq = lowest_freq * std::pow(2.0, static_cast<double>(i) / bins_per_octave);
            double diff1 = std::abs(bin_freq - freq1);
            double diff2 = std::abs(bin_freq - freq2);
            if (diff1 < min_diff1) {
                min_diff1 = diff1;
                bin_index1 = i;
            }
            if (diff2 < min_diff2) {
                min_diff2 = diff2;
                bin_index2 = i;
            }
        }
         // Add checks to ensure the found frequencies are reasonably close
        double bin_half_width = lowest_freq * (std::pow(2.0, 1.0 / (2.0 * bins_per_octave)) - std::pow(2.0, -1.0 / (2.0 * bins_per_octave)));
        EXPECT_LT(min_diff1, bin_half_width) << "Could not find a CQT bin reasonably close to the target frequency " << freq1 << " Hz.";
        EXPECT_LT(min_diff2, bin_half_width) << "Could not find a CQT bin reasonably close to the target frequency " << freq2 << " Hz.";
        ASSERT_NE(bin_index1, bin_index2) << "Target frequencies are too close for the given CQT resolution, mapping to the same bin.";


         //Check if energy is concentrated in the expected bins
         T max_energy = 0.0f;
         for(const auto& coeff : cqt_output) max_energy = std::max(max_energy, std::norm(coeff));

         for (size_t i = 0; i < expected_bins; ++i) {
             // **MODIFIED**: Calculate energy directly from the complex coefficient
             T energy = std::norm(cqt_output[i]);

            if (i == bin_index1 || i == bin_index2) {
                // **THRESHOLD NEEDS ADJUSTMENT**: Check if energy in the target bins is significant
                // The old threshold of 50.0f is likely incorrect.
                // EXPECT_GT(energy, 50.0f) << "Energy in expected bin " << i << " is lower than expected. Actual: " << energy;
                // Example: Check if energy is close to the max energy found
                 EXPECT_NEAR(energy, max_energy, max_energy * 0.2f) << "Energy in expected bin " << i << " not close to max energy."; // Within 20% of max
            } else {
                // **THRESHOLD NEEDS ADJUSTMENT**: Check if energy in other bins is low
                // The old threshold of 5.0f is likely incorrect.
                // EXPECT_LT(energy, 5.0f) << "Energy in other bin " << i << " is higher than expected. Actual: " << energy;
                // Example: Check relative to the peak energy
                EXPECT_LT(energy, std::max(static_cast<T>(1e-9), max_energy * static_cast<T>(0.05))) // Allow very small noise floor, otherwise < 5% of peak
                    << "Energy in other bin " << i << " [" << energy << "] is too high relative to peak energy [" << max_energy << "]";
            }
        }

    } catch (const std::exception& e) {
        FAIL() << "CQT operation threw exception: " << e.what();
    }
}

// TODO: Add more tests:
// - Test with different CQT parameters (bins_per_octave, frequency range)
// - Test edge cases (very low frequencies, very high frequencies near Nyquist)
// - Test with different input signal types (e.g., noise, square wave, impulse)
// - Error condition tests (e.g., invalid CQT parameters to constructor)
// - Test energy conservation properties if kernels are appropriately normalized (e.g., L2 norm)