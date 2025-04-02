#include <gtest/gtest.h>
#include <OmniDSP/omnidsp.h> // Use correct include path
#include <OmniDSP/cqt.h>     // Include CQT header
#include <vector>
#include <complex>
#include <limits>
#include <cmath>

// --- Test Fixture ---
class FFT_CQT_Test : public ::testing::Test {
protected:
    // Test parameters
    const double sample_rate = 44100.0; // Standard audio sample rate
    const double lowest_freq = 20.0;    // Lowest human-audible frequency
    const double highest_freq = 20000.0; // Highest human-audible frequency
    const int bins_per_octave = 12;     // Typical value for music analysis
    const double double_tolerance = 1.0; // CQT results can have larger tolerances
    const float float_tolerance = 1.0f;  // due to windowing and spectral leakage

    // Helper to compare vectors of complex vectors with tolerance
    template <typename T>
    void ExpectCQTOutputNear(const std::vector<std::vector<std::complex<T>>>& expected,
                            const std::vector<std::vector<std::complex<T>>>& actual,
                            T tolerance,
                            const std::string& msg = "") {
        ASSERT_EQ(expected.size(), actual.size()) << msg << " - Number of bins mismatch";
        for (size_t bin = 0; bin < expected.size(); ++bin) {
            ASSERT_EQ(expected[bin].size(), actual[bin].size()) << msg << " - Size mismatch for bin " << bin;
            for (size_t i = 0; i < expected[bin].size(); ++i) {
                EXPECT_NEAR(expected[bin][i].real(), actual[bin][i].real(), tolerance)
                    << msg << " - Mismatch at bin " << bin << ", index " << i << " (real)";
                EXPECT_NEAR(expected[bin][i].imag(), actual[bin][i].imag(), tolerance)
                    << msg << " - Mismatch at bin " << bin << ", index " << i << " (imag)";
            }
        }
    }

    // Helper function to generate a simple test signal (single frequency)
    template <typename T>
    std::vector<T> generate_test_signal(double frequency, double sample_rate, size_t length) {
        std::vector<T> signal(length);
        for (size_t i = 0; i < length; ++i) {
            double time = static_cast<double>(i) / sample_rate;
            signal[i] = static_cast<T>(std::cos(2.0 * M_PI * frequency * time));
        }
        return signal;
    }
};

// --- Test Cases ---

TEST_F(FFT_CQT_Test, SingleFrequency_Double) {
    using T = double;
    size_t signal_length = 4096;
    double test_freq = 440.0; // A4 note

    std::vector<T> input_signal = generate_test_signal<T>(test_freq, sample_rate, signal_length);
    std::vector<std::vector<std::complex<T>>> cqt_output;

    try {
        // Create CQTPlan with Hann window
        OmniDSP::CQTPlan<T> cqt_plan(sample_rate, lowest_freq, highest_freq, bins_per_octave, OmniDSP::Window::hann<T>);
        cqt_plan.execute(input_signal, cqt_output);

        // Basic verification:
        // 1. Output should have the correct number of bins
        size_t expected_bins = static_cast<size_t>(bins_per_octave * std::ceil(std::log2(highest_freq / lowest_freq)));
        ASSERT_EQ(cqt_output.size(), expected_bins);

        // 2. Find the bin closest to the test frequency
        double resolution = sample_rate / signal_length; // Frequency resolution of the FFT
        size_t expected_bin_index = 0;
        double min_diff = std::numeric_limits<double>::max();
        for (size_t i = 0; i < expected_bins; ++i) {
            double bin_freq = lowest_freq * std::pow(2.0, static_cast<double>(i) / bins_per_octave);
            double diff = std::abs(bin_freq - test_freq);
            if (diff < min_diff) {
                min_diff = diff;
                expected_bin_index = i;
            }
        }

        // 3. Check if the energy is concentrated in the expected bin
        for (size_t i = 0; i < expected_bins; ++i) {
            T energy = 0.0;
            for (const auto& val : cqt_output[i]) {
                energy += std::norm(val); // Sum of squares of complex values
            }
            if (i == expected_bin_index) {
                EXPECT_GT(energy, 100.0) << "Energy in expected bin is low"; // Adjust threshold as needed
            } else {
                EXPECT_LT(energy, 10.0) << "Energy in other bins is high";  // Adjust threshold as needed
            }
        }

    } catch (const std::exception& e) {
        FAIL() << "CQT operation threw exception: " << e.what();
    }
}

TEST_F(FFT_CQT_Test, TwoFrequencies_Float) {
    using T = float;
    size_t signal_length = 4096;
    double freq1 = 220.0;
    double freq2 = 880.0;
    std::vector<T> input_signal(signal_length);
    for (size_t i = 0; i < signal_length; ++i) {
        double time = static_cast<double>(i) / sample_rate;
        input_signal[i] = static_cast<T>(std::cos(2.0 * M_PI * freq1 * time) + std::cos(2.0 * M_PI * freq2 * time));
    }

    std::vector<std::vector<std::complex<T>>> cqt_output;
    try {
        // Create CQTPlan with Kaiser window
        auto kaiser_window = [](const std::vector<float>& input) {
            return OmniDSP::Window::kaiser(input, 5.0f); // Example beta value
        };
        OmniDSP::CQTPlan<T> cqt_plan(sample_rate, lowest_freq, highest_freq, bins_per_octave, kaiser_window);
        cqt_plan.execute(input_signal, cqt_output);

        // Basic verification:
        ASSERT_EQ(cqt_output.size(), static_cast<size_t>(bins_per_octave * std::ceil(std::log2(highest_freq / lowest_freq))));

        // Find bins closest to the test frequencies
        size_t bin_index1 = 0, bin_index2 = 0;
        double min_diff1 = std::numeric_limits<double>::max(), min_diff2 = std::numeric_limits<double>::max();
        for (size_t i = 0; i < cqt_output.size(); ++i) {
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

         //Check if energy is concentrated in the expected bins
         for (size_t i = 0; i < cqt_output.size(); ++i) {
            float energy = 0.0f;
            for (const auto& val : cqt_output[i]) {
                energy += std::norm(val); // Sum of squares of complex values
            }
            if (i == bin_index1 || i == bin_index2) {
                EXPECT_GT(energy, 50.0f) << "Energy in expected bin is low"; // Adjust threshold as needed
            } else {
                EXPECT_LT(energy, 5.0f) << "Energy in other bins is high";  // Adjust threshold as needed
            }
        }


    } catch (const std::exception& e) {
        FAIL() << "CQT operation threw exception: " << e.what();
    }
}

// TODO: Add more tests:
// - Test with different CQT parameters (bins_per_octave, frequency range)
// - Test edge cases (very low frequencies, very high frequencies)
// - Test with different input signal types (e.g., noise, square wave)
// - Error condition tests (e.g., invalid CQT parameters)