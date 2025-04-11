/**
 * @file tests/cpp/cqt.cpp
 * @brief Unit tests for the OmniDSP CQTPlan class.
 *
 * Tests CQTPlan creation and execution using the updated constructor
 * accepting std::function<vector<T>(size_t)>.
 */

 #include <gtest/gtest.h>
 #include <OmniDSP/omnidsp.h> // Include OmniDSP header
 #include <OmniDSP/cqt.h>     // Include CQT header
 #include <OmniDSP/windows.h> // Include Window header for test lambdas
 #include <vector>
 #include <complex>
 #include <limits>
 #include <cmath>
 #include <string>    // For std::to_string in error messages
 #include <functional> // For std::function type alias if needed
 
 #ifndef M_PI
 #define M_PI 3.14159265358979323846
 #endif
 
 // --- Test Fixture ---
 class FFT_CQT_Test : public ::testing::Test {
 protected:
     // Test parameters
     const double sample_rate = 44100.0;
     const double lowest_freq = 55.0;    // A1 note
     const double highest_freq = 1760.0; // A6 note (Adjusted for faster tests maybe)
     const int bins_per_octave = 12;     // Semitones
 
     // Helper function to generate a single frequency signal
     template <typename T>
     std::vector<T> generate_test_signal(double frequency, double sample_rate, size_t length) {
         std::vector<T> signal(length);
         if (length == 0 || sample_rate <= 0) return signal;
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
             // Reduce amplitude to avoid potential clipping/overflow if summing signals later
             signal[i] = static_cast<T>(0.5 * std::cos(2.0 * M_PI * freq1 * time) + 0.5 * std::cos(2.0 * M_PI * freq2 * time));
         }
         return signal;
     }
 
     // Helper to find index of bin closest to target frequency
     size_t find_closest_bin(double target_freq, double cqt_lowest_freq, int cqt_bpo, size_t num_bins) {
         size_t best_index = 0;
         double min_diff = std::numeric_limits<double>::max();
         for (size_t i = 0; i < num_bins; ++i) {
             double bin_freq = cqt_lowest_freq * std::pow(2.0, static_cast<double>(i) / cqt_bpo);
             double diff = std::abs(bin_freq - target_freq);
             if (diff < min_diff) {
                 min_diff = diff;
                 best_index = i;
             }
         }
         // Optional: Add an assertion here that min_diff is reasonably small
         // double bin_half_width = cqt_lowest_freq * (std::pow(2.0, 1.0 / (2.0 * cqt_bpo)) - std::pow(2.0, -1.0 / (2.0 * cqt_bpo)));
         // EXPECT_LT(min_diff, bin_half_width * 1.1); // Allow slight tolerance
         return best_index;
     }
 
 };
 
 // --- Test Cases ---
 
 TEST_F(FFT_CQT_Test, SingleFrequency_Double) {
     using T = double;
     using Complex = std::complex<T>;
     size_t signal_length = 8192; // Use a longer length for better CQT resolution
     double test_freq = 440.0; // A4 note
 
     std::vector<T> input_signal = generate_test_signal<T>(test_freq, sample_rate, signal_length);
     std::vector<Complex> cqt_output;
 
     try {
         // Create CQTPlan with Hann window using lambda compatible with std::function<vector<T>(size_t)>
         auto hann_window_generator = [](size_t length) -> std::vector<T> {
             // Generate a dummy input vector of the required length to pass to Window::hann
             if (length == 0) return {}; // Handle zero length case
             std::vector<T> dummy_input(length, static_cast<T>(1.0));
             return OmniDSP::Window::hann(dummy_input); // Apply Hann to the dummy vector
         };
 
         // Instantiate CQTPlan with the generator function
         OmniDSP::CQTPlan<T> cqt_plan(sample_rate, lowest_freq, highest_freq, bins_per_octave, hann_window_generator);
 
         // Execute CQT
         cqt_plan.execute(input_signal, cqt_output);
 
         // --- Verification ---
         size_t num_bins = cqt_plan.getNumBins();
         ASSERT_EQ(cqt_output.size(), num_bins);
 
         size_t expected_bin_index = find_closest_bin(test_freq, lowest_freq, bins_per_octave, num_bins);
 
         // Check energy concentration
         T max_energy = 0.0;
         for(const auto& coeff : cqt_output) {
             max_energy = std::max(max_energy, std::norm(coeff)); // norm = |coeff|^2
         }
         ASSERT_GT(max_energy, std::numeric_limits<T>::epsilon()) << "Max energy is near zero, CQT likely failed.";
 
         for (size_t i = 0; i < num_bins; ++i) {
             T energy = std::norm(cqt_output[i]);
             if (i == expected_bin_index) {
                 // Expect energy in the target bin to be close to the maximum energy
                 EXPECT_NEAR(energy, max_energy, max_energy * 0.1) << "Energy in expected bin " << i << " (" << test_freq << " Hz) not close to max energy.";
             } else {
                 // Expect energy in other bins to be significantly lower
                 EXPECT_LT(energy, std::max(static_cast<T>(1e-9), max_energy * static_cast<T>(0.01))) // Allow small noise floor, or < 1% of peak
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
     size_t signal_length = 8192;
     double freq1 = 220.0f; // A3
     double freq2 = 880.0f; // A5 (two octaves higher)
 
     std::vector<T> input_signal = generate_two_freq_signal<T>(freq1, freq2, sample_rate, signal_length);
     std::vector<Complex> cqt_output;
 
     try {
         // Create CQTPlan with Kaiser window
         auto kaiser_window_generator = [](size_t length) -> std::vector<T> {
             if (length == 0) return {};
             T beta = 5.0f; // Example beta
             std::vector<T> dummy_input(length, static_cast<T>(1.0));
             return OmniDSP::Window::kaiser(dummy_input, beta);
         };
 
         OmniDSP::CQTPlan<T> cqt_plan(sample_rate, lowest_freq, highest_freq, bins_per_octave, kaiser_window_generator);
 
         cqt_plan.execute(input_signal, cqt_output);
 
         // --- Verification ---
         size_t num_bins = cqt_plan.getNumBins();
         ASSERT_EQ(cqt_output.size(), num_bins);
 
         size_t bin_index1 = find_closest_bin(freq1, lowest_freq, bins_per_octave, num_bins);
         size_t bin_index2 = find_closest_bin(freq2, lowest_freq, bins_per_octave, num_bins);
         ASSERT_NE(bin_index1, bin_index2) << "Target frequencies are too close, mapping to the same bin.";
 
 
         // Check energy concentration
         T max_energy = 0.0f;
         for(const auto& coeff : cqt_output) {
              max_energy = std::max(max_energy, std::norm(coeff));
         }
         ASSERT_GT(max_energy, std::numeric_limits<T>::epsilon()) << "Max energy is near zero, CQT likely failed.";
 
          for (size_t i = 0; i < num_bins; ++i) {
              T energy = std::norm(cqt_output[i]);
 
             if (i == bin_index1 || i == bin_index2) {
                 // Expect energy in target bins to be close to max (might not be exactly max if freqs aren't perfectly centered)
                 EXPECT_NEAR(energy, max_energy, max_energy * 0.25f) << "Energy in expected bin " << i << " not close to max energy."; // Allow 25% variation
             } else {
                  // Expect energy in other bins to be significantly lower
                 EXPECT_LT(energy, std::max(static_cast<T>(1e-9), max_energy * static_cast<T>(0.05))) // Allow small noise floor, or < 5% of peak
                     << "Energy in other bin " << i << " [" << energy << "] is too high relative to peak energy [" << max_energy << "]";
             }
         }
 
     } catch (const std::exception& e) {
         FAIL() << "CQT operation threw exception: " << e.what();
     }
 }
 
 // --- Test for Invalid Constructor Arguments ---
 TEST_F(FFT_CQT_Test, InvalidParameters_ThrowsException) {
      using T = double;
       auto dummy_window_gen = [](size_t length) -> std::vector<T> {
             return std::vector<T>(length, 1.0); // Simple valid generator
       };
 
      // Test invalid sample rate
      ASSERT_THROW(OmniDSP::CQTPlan<T>(-1.0, lowest_freq, highest_freq, bins_per_octave, dummy_window_gen), std::invalid_argument);
      // Test invalid lowest frequency
      ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, 0.0, highest_freq, bins_per_octave, dummy_window_gen), std::invalid_argument);
      // Test invalid highest frequency (<= lowest)
      ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, lowest_freq, lowest_freq, bins_per_octave, dummy_window_gen), std::invalid_argument);
      // Test invalid highest frequency (> Nyquist)
      ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, lowest_freq, sample_rate, bins_per_octave, dummy_window_gen), std::invalid_argument);
      // Test invalid bins per octave
      ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, lowest_freq, highest_freq, 0, dummy_window_gen), std::invalid_argument);
      // Test null window function
       OmniDSP::CQTPlan<T>::WindowFuncType null_func = nullptr;
      ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, lowest_freq, highest_freq, bins_per_octave, null_func), std::invalid_argument);
 }
 
 
 // TODO: Add more tests:
 // - Test edge cases (very low/high frequencies if relevant)
 // - Test behavior with empty input signal (should return vector of zeros) -> Added basic check
 // - Test error if window function returns wrong size vector -> Added check in cqt.cpp