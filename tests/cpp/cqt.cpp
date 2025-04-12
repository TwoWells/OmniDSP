/**
 * @file tests/cpp/cqt.cpp
 * @brief Unit tests for the OmniDSP Recursive CQTPlan class
 * with precomputed sparse kernels.
 */

 #include <gtest/gtest.h>
 #include <OmniDSP/omnidsp.h> // For FFTPlan potentially
 #include <OmniDSP/cqt.h>     // Include Recursive CQT header
 #include <OmniDSP/windows.h> // For M_PI definition
 #include <vector>
 #include <complex>
 #include <limits>
 #include <cmath>
 #include <numeric> // For std::iota, std::accumulate
 #include <map>     // For sparse kernel checks
 #include <iostream>// For std::cout (used for warnings)
 
 #ifndef M_PI
 #define M_PI 3.14159265358979323846
 #endif
 
 // Define a constant for the default FIR order used in tests, matching the C++ implementation
 const int DEFAULT_FIR_FILTER_ORDER_TEST = 101;
 
 // --- Test Fixture ---
 class PrecomputedRecursiveCQTTest : public ::testing::Test
 {
 protected:
     // Common parameters
     const double sample_rate = 44100.0;
     const double lowest_freq = 55.0; // A1
     const double high_freq = 1760.0; // A6 (Covers 5 octaves from A1)
     const int bins_per_octave = 12;
     // Valid hop length: 5 octaves -> need hop % 16 == 0
     const size_t hop_length = 512;
     const float sparsity_threshold_float = 1e-5f;
     const double sparsity_threshold_double = 1e-5;
 
     // Default window generator helper (returns coefficients)
     template <typename T>
     static std::vector<T> hannWindowCoeffs(size_t length)
     {
         if (length == 0)
             return {};
         std::vector<T> window_coeffs(length);
         double denom = (length > 1) ? static_cast<double>(length - 1) : 1.0;
         for (size_t n = 0; n < length; ++n)
         {
             window_coeffs[n] = static_cast<T>(0.5 - 0.5 * cos(2.0 * M_PI * n / denom));
         }
         // No L1 norm here, assume kernel generation handles it if needed
         return window_coeffs;
     }
 
     // Helper to generate a sine wave
     template <typename T>
     std::vector<T> generateSineWave(double freq, double duration_sec, double sr)
     {
         size_t length = static_cast<size_t>(duration_sec * sr);
         if (length == 0)
             return {};
         std::vector<T> signal(length);
         for (size_t i = 0; i < length; ++i)
         {
             signal[i] = static_cast<T>(std::sin(2.0 * M_PI * freq * static_cast<double>(i) / sr));
         }
         return signal;
     }
 
     // Helper to compare complex matrices
     template <typename T>
     void ExpectComplexMatrixNear(const std::vector<std::vector<std::complex<T>>> &expected,
                                  const std::vector<std::vector<std::complex<T>>> &actual,
                                  T tolerance, const std::string &msg = "")
     {
         ASSERT_EQ(expected.size(), actual.size()) << msg << " - Bin count mismatch";
         if (expected.empty() && actual.empty())
             return;
         ASSERT_FALSE(expected.empty()) << msg << " - Expected matrix is empty";
         ASSERT_FALSE(actual.empty()) << msg << " - Actual matrix is empty";
 
         size_t num_bins = expected.size();
         ASSERT_GT(num_bins, 0) << msg << " - Zero bins in expected matrix";
         size_t num_frames = expected[0].size();
         // Check frame count consistency only if actual matrix is not empty
         if (!actual.empty() && !actual[0].empty()) {
              ASSERT_EQ(actual[0].size(), num_frames) << msg << " - Frame count mismatch for bin 0";
         } else if (num_frames > 0) {
              FAIL() << msg << " - Actual matrix has zero frames while expected has " << num_frames;
         }
 
 
         for (size_t i = 0; i < num_bins; ++i)
         {
             ASSERT_EQ(expected[i].size(), num_frames) << msg << " - Frame count mismatch in expected bin " << i;
             // Check actual only if it has enough rows
              if (i < actual.size()) {
                  ASSERT_EQ(actual[i].size(), num_frames) << msg << " - Frame count mismatch in actual bin " << i;
                  for (size_t j = 0; j < num_frames; ++j)
                  {
                      EXPECT_NEAR(expected[i][j].real(), actual[i][j].real(), tolerance)
                          << msg << " - Mismatch at bin " << i << ", frame " << j << " (real)";
                      EXPECT_NEAR(expected[i][j].imag(), actual[i][j].imag(), tolerance)
                          << msg << " - Mismatch at bin " << i << ", frame " << j << " (imag)";
                  }
              } else {
                  FAIL() << msg << " - Actual matrix has fewer bins than expected (" << actual.size() << " vs " << num_bins << ")";
              }
         }
     }
 
     // Helper to compare real vectors
     template <typename T>
     void ExpectRealVectorNear(const std::vector<T> &expected,
                               const std::vector<T> &actual,
                               T tolerance, const std::string &msg = "")
     {
         ASSERT_EQ(expected.size(), actual.size()) << msg << " - Size mismatch";
         for (size_t i = 0; i < expected.size(); ++i)
         {
             EXPECT_NEAR(expected[i], actual[i], tolerance)
                 << msg << " - Mismatch at index " << i;
         }
     }
 
     // Helper to get tolerance
     template <typename T>
     T get_tolerance(size_t N = 1024)
     {
         return std::numeric_limits<T>::epsilon() * N * 100;
     }
 
     // --- Accessors for Private/Protected CQTPlan Members (Requires Friend Declaration in cqt.h) ---
     // Friend declaration needed in cqt.h: friend class ::PrecomputedRecursiveCQTTest;
     template <typename T>
     static std::vector<T> callCalculateFirCoefficients(const OmniDSP::CQTPlan<T> &plan,
                                                        double current_sr, int N)
     {
         // Calls the private/protected helper method via friend access
         return plan.calculateFirCoefficients(current_sr, N);
     }
 
     template <typename T>
     static std::vector<T> callFilterAndDownsampleBy2(const OmniDSP::CQTPlan<T> &plan,
                                                      const std::vector<T> &signal,
                                                      double current_sr)
     {
         // Calls the private/protected helper method via friend access
         return plan.filterAndDownsampleBy2(signal, current_sr);
     }
 };
 
 // --- Test Cases ---
 
 // 1. Test Constructor and Precomputation Sanity Checks
 TEST_F(PrecomputedRecursiveCQTTest, ConstructorPrecomputation)
 {
     using T = double;
 
     // Test successful construction with valid parameters
     std::unique_ptr<OmniDSP::CQTPlan<T>> plan_ptr;
     ASSERT_NO_THROW(
         plan_ptr = std::make_unique<OmniDSP::CQTPlan<T>>(
             sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave,
             hannWindowCoeffs<T>, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST)) // Use test constant
         << "Constructor threw unexpected exception.";
 
     ASSERT_NE(plan_ptr, nullptr);
     OmniDSP::CQTPlan<T> &plan = *plan_ptr; // Dereference for easier access
 
     // Check basic parameter storage
     ASSERT_EQ(plan.getSampleRate(), sample_rate);
     ASSERT_EQ(plan.getHopLength(), hop_length);
     ASSERT_EQ(plan.getLowestFrequency(), lowest_freq);
     ASSERT_EQ(plan.getHighestFrequency(), high_freq);
     ASSERT_EQ(plan.getBinsPerOctave(), bins_per_octave);
     ASSERT_NEAR(plan.getSparsityThreshold(), sparsity_threshold_double, get_tolerance<T>());
     ASSERT_EQ(plan.getFirFilterOrder(), DEFAULT_FIR_FILTER_ORDER_TEST); // Check stored FIR order
 
     // Check calculated properties
     int expected_octaves = 5; // Based on fixture params A1 to A6
     size_t expected_bins = static_cast<size_t>(std::ceil(bins_per_octave * std::log2(high_freq / lowest_freq)));
     ASSERT_EQ(plan.getNumOctaves(), expected_octaves);
     ASSERT_EQ(plan.getNumBins(), expected_bins);
 
     // Print a reminder message instead of using a non-existent macro
     std::cout << "[ INFO ] Constructor test: Add checks for precomputed structure sizes/content (requires access)." << std::endl;
 }
 
 // 2. Test FIR Coefficient Calculation
 TEST_F(PrecomputedRecursiveCQTTest, CalculateFirCoefficients)
 {
     using T = double;
 
     // 1. Instantiate CQTPlan to access the helper
     OmniDSP::CQTPlan<T> plan(sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave,
                              hannWindowCoeffs<T>, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST);
 
     // 2. Call the coefficient calculation (using accessor)
     int N = DEFAULT_FIR_FILTER_ORDER_TEST; // Use the defined constant
     std::vector<T> coeffs;
     // Ensure the friend declaration works or the method is accessible
     ASSERT_NO_THROW(coeffs = callCalculateFirCoefficients(plan, sample_rate, N));
 
     // 3. Verify the output
     ASSERT_EQ(coeffs.size(), N);
     // Check for symmetry (h[k] == h[N-1-k])
     T symm_tolerance = get_tolerance<T>() * 10; // Allow slightly more tolerance
     for (int k = 0; k < N / 2; ++k)
     {
         ASSERT_NEAR(coeffs[k], coeffs[N - 1 - k], symm_tolerance) << "Symmetry check failed at k=" << k;
     }
     // Check sum is close to 1.0 (if normalized)
     T sum = std::accumulate(coeffs.begin(), coeffs.end(), T(0.0));
     ASSERT_NEAR(sum, T(1.0), get_tolerance<T>()) << "Coefficient sum not close to 1.0";
 }
 
 // 3. Test Combined Filter and Downsample Step
 TEST_F(PrecomputedRecursiveCQTTest, FilterAndDownsample)
 {
     using T = double;
 
     // 1. Create known input signal (e.g., low-freq sine that should pass filter)
     double sr = 4000.0;       // Lower SR for testing
     int num_octaves_test = 2; // Simpler case
     size_t valid_hop = 128;   // Divisible by 2^(2-1)=2
     size_t len = 2048;
     double freq_low = 500.0; // Below sr/4 cutoff (1000Hz)
     std::vector<T> test_signal = generateSineWave<T>(freq_low, static_cast<double>(len) / sr, sr);
 
     // 2. Instantiate CQTPlan
     OmniDSP::CQTPlan<T> plan(sr, valid_hop, 400, 1600, 12, hannWindowCoeffs<T>,
                              sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST); // Use test constant
 
     // 3. Call filterAndDownsampleBy2 (using accessor)
     std::vector<T> result;
     ASSERT_NO_THROW(result = callFilterAndDownsampleBy2(plan, test_signal, sr));
 
     // 4. Verify Output
     //    - Check Length: Output length depends on the backend's convolution 'valid' mode calculation
     //      and the downsampling factor. For vDSP_desamp or MKL(valid)+decimate:
     //      output_len = floor((input_len - filter_len) / factor) + 1
     size_t expected_len_approx = (len >= DEFAULT_FIR_FILTER_ORDER_TEST) ?
                                  ((len - DEFAULT_FIR_FILTER_ORDER_TEST) / 2 + 1) : 0;
     ASSERT_EQ(result.size(), expected_len_approx); // Check exact length based on formula
 }
 
 // 4. Test Full Recursive CQT Execute Method (Integration Test)
 TEST_F(PrecomputedRecursiveCQTTest, FullRecursiveCQT_Execute)
 {
     using T = double;
     using Complex = std::complex<T>;
 
     // 1. Create test signal with components in different octaves
     double freq1 = 65.4;                 // C2 (~octave 1 for A1 base)
     double freq2 = 261.6;                // C4 (~octave 3 for A1 base)
     double freq3 = 1046.5;               // C6 (~octave 5 for A1 base, within fmax=1760)
     size_t signal_len = static_cast<size_t>(sample_rate * 1.0); // 1 second
     std::vector<T> test_signal(signal_len);
     for (size_t i = 0; i < signal_len; ++i)
     {
         double time = static_cast<double>(i) / sample_rate;
         test_signal[i] = (T)0.3 * std::sin(2.0 * M_PI * freq1 * time) +
                          (T)0.3 * std::sin(2.0 * M_PI * freq2 * time) +
                          (T)0.3 * std::sin(2.0 * M_PI * freq3 * time);
     }
 
     // 2. Instantiate CQTPlan (using fixture parameters)
     OmniDSP::CQTPlan<T> plan(sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave,
                              hannWindowCoeffs<T>, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST);
     std::vector<std::vector<Complex>> cqt_output;
 
     // 3. Execute
     ASSERT_NO_THROW(plan.execute(test_signal, cqt_output));
 
     // --- Verification Steps ---
     // 4. Verify Output Shape
     size_t expected_bins = plan.getNumBins();
     ASSERT_EQ(cqt_output.size(), expected_bins);
     if (expected_bins > 0 && !cqt_output.empty()) {
         ASSERT_FALSE(cqt_output[0].empty()) << "First bin has zero frames.";
         size_t expected_frames = cqt_output[0].size();
         ASSERT_GT(expected_frames, 0);
         for(size_t b=1; b<expected_bins; ++b) {
              ASSERT_EQ(cqt_output[b].size(), expected_frames) << "Frame count mismatch at bin " << b;
         }
     } else if (expected_bins > 0) {
          FAIL() << "Expected bins > 0, but CQT output matrix is empty.";
     }
 
 
     // 5. Verify Energy Concentration (Basic Check)
     if (expected_bins > 0 && !cqt_output.empty() && !cqt_output[0].empty()) {
         size_t num_frames = cqt_output[0].size();
         std::vector<T> avg_magnitude(expected_bins, 0.0);
         for(size_t b=0; b<expected_bins; ++b) {
             T sum_mag = 0.0;
             for(size_t f=0; f<num_frames; ++f) {
                 sum_mag += std::abs(cqt_output[b][f]);
             }
              // Avoid division by zero if num_frames is somehow 0
             avg_magnitude[b] = (num_frames > 0) ? (sum_mag / num_frames) : 0.0;
         }
 
         // Find approximate expected bin indices (requires careful calculation)
         auto freq_to_bin = [&](double freq) {
             if (freq < lowest_freq) return -1.0;
             // Use log2 from cmath
             return bins_per_octave * std::log2(freq / lowest_freq);
         };
         int bin1_idx = static_cast<int>(std::round(freq_to_bin(freq1)));
         int bin2_idx = static_cast<int>(std::round(freq_to_bin(freq2)));
         int bin3_idx = static_cast<int>(std::round(freq_to_bin(freq3)));
 
         // Check if peaks exist near expected bins (simple check: max within +/- 1 bin)
         // *** FIX: Added '-> bool' return type and changed ASSERT_* to EXPECT_* ***
         auto check_peak = [&](int target_bin, double freq) -> bool {
             if (target_bin < 0 || target_bin >= static_cast<int>(expected_bins)) return false; // Return bool
             T max_val = 0.0;
             int max_idx = -1;
             for (int b_offset = -1; b_offset <= 1; ++b_offset) {
                 int current_bin = target_bin + b_offset;
                 if (current_bin >= 0 && current_bin < static_cast<int>(expected_bins)) {
                     if (avg_magnitude[current_bin] > max_val) {
                         max_val = avg_magnitude[current_bin];
                         max_idx = current_bin;
                     }
                 }
             }
              std::cout << "[ INFO ] Peak check for " << freq << " Hz: Target bin " << target_bin << ", Max found at bin " << max_idx << " (Mag: " << max_val << ")" << std::endl;
              // Use non-fatal checks inside the lambda
              EXPECT_GE(max_idx, target_bin - 1); // Line 372
              EXPECT_LE(max_idx, target_bin + 1); // Line 373
              EXPECT_GT(max_val, 0.0);            // Line 374
 
              // Return true if a peak was found nearby and is positive, otherwise false
              bool peak_found_nearby = (max_idx >= target_bin - 1) && (max_idx <= target_bin + 1);
              bool peak_is_positive = (max_val > 0.0);
              return peak_found_nearby && peak_is_positive; // Return bool
         };
 
         // Now call the lambda and use ASSERT_TRUE on its boolean result
         ASSERT_TRUE(check_peak(bin1_idx, freq1));
         ASSERT_TRUE(check_peak(bin2_idx, freq2));
         ASSERT_TRUE(check_peak(bin3_idx, freq3));
 
     } else {
          std::cout << "[ INFO ] Skipping energy concentration check due to empty CQT output." << std::endl;
     }
 }
 
 // 5. Test Parameter Validation in Constructor
 TEST_F(PrecomputedRecursiveCQTTest, ParameterValidation)
 {
     using T = double;
     auto dummy_gen = hannWindowCoeffs<T>;
 
     // Use fixture parameters which are valid
     ASSERT_NO_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave, dummy_gen, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST));
 
     // Invalid parameters (copied from previous test, should still apply)
     ASSERT_THROW(OmniDSP::CQTPlan<T>(0.0, hop_length, lowest_freq, high_freq, bins_per_octave, dummy_gen, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument);
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, 0, lowest_freq, high_freq, bins_per_octave, dummy_gen, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument);
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, 0.0, high_freq, bins_per_octave, dummy_gen, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument);
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, lowest_freq, lowest_freq, bins_per_octave, dummy_gen, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument);
     // Nyquist check might allow equality now due to clamping, but > sr should still fail
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, lowest_freq, sample_rate, bins_per_octave, dummy_gen, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument);
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, lowest_freq, high_freq, 0, dummy_gen, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument);
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave, nullptr, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument);
     // Invalid hop length (5 octaves need hop div by 16)
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, 511, lowest_freq, high_freq, bins_per_octave, dummy_gen, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument);
     // Invalid sparsity
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave, dummy_gen, -0.1, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument);
     // Invalid FIR order
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave, dummy_gen, sparsity_threshold_double, 0), std::invalid_argument);
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave, dummy_gen, sparsity_threshold_double, 100), std::invalid_argument); // Even order
 }
 