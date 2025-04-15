/**
 * @file tests/cpp/cqt.cpp
 * @brief Unit tests for the OmniDSP Recursive CQTPlan class
 * with precomputed sparse kernels. Includes comparison with Librosa reference.
 * Updated LoadReferenceData logic to use CMake-defined path.
 */

 #include <gtest/gtest.h>
 #include <OmniDSP/omnidsp.h> // For FFTPlan potentially and Enums
 #include <OmniDSP/cqt.h>     // Include Recursive CQT header
 #include <OmniDSP/window.h>  // Include renamed window header (needed for M_PI in helper)
 #include <vector>
 #include <complex>
 #include <limits>
 #include <cmath>
 #include <numeric>  // For std::iota, std::accumulate
 #include <map>      // For sparse kernel checks
 #include <iostream> // For std::cout/cerr
 #include <memory>   // For std::unique_ptr access in tests
 #include <fstream>  // For file input
 #include <sstream>  // For string parsing
 #include <string>   // For std::string, std::getline, std::stod, etc.
 #include <stdexcept> // Needed for EXPECT_THROW

 // Define M_PI if it's not already defined (e.g., by <cmath> in some environments/standards)
 #ifndef M_PI
 #define M_PI 3.14159265358979323846
 #endif

 // Define a constant for the default FIR order used in tests, matching the C++ implementation
 const int DEFAULT_FIR_FILTER_ORDER_TEST = 101;

 // Define aliases for vector/matrix types
 using VecD = std::vector<double>;
 using VecF = std::vector<float>;
 using ComplexVecD = std::vector<std::complex<double>>;
 using ComplexMatD = std::vector<ComplexVecD>; // Matrix type [bin][frame]

 // --- Test Fixture ---
 class PrecomputedRecursiveCQTTest : public ::testing::Test
 {
 protected:
     // Common parameters matching generate_references.py CQT section
     const double sample_rate = 44100.0;
     const double lowest_freq = 55.0; // A1
     const double high_freq = 1760.0; // A6
     const int bins_per_octave = 12;
     const size_t hop_length = 512; // Valid hop length: 5 octaves -> need hop % 16 == 0
     const float sparsity_threshold_float = 1e-5f;
     const double sparsity_threshold_double = 1e-5;
     const double cqt_signal_freq = 440.0;   // A4 - matches generate_references.py
     const double cqt_signal_duration = 1.0; // seconds - matches generate_references.py

     // Storage for loaded reference data
     static std::map<std::string, VecD> expected_data_d;
     static std::map<std::string, VecF> expected_data_f;
     static ComplexMatD expected_librosa_cqt_d;
     static bool references_loaded;

     // Default window generator helper (returns coefficients) - L1 Normalized
     // This is passed to the CQTPlan constructor in tests.
     template <typename T>
     static std::vector<T> hannWindowCoeffsGenerator(size_t length)
     {
         if (length == 0)
             return {};
         std::vector<T> window_coeffs(length);
         double denom = (length > 1) ? static_cast<double>(length - 1) : 1.0;
         for (size_t n = 0; n < length; ++n)
         {
             window_coeffs[n] = static_cast<T>(0.5 - 0.5 * std::cos(2.0 * M_PI * n / denom));
         }
         // CQTPlan internally normalizes the kernel, so no need to normalize here
         // for the generator function passed to the constructor.
         return window_coeffs;
     }

     // Helper to generate a sine wave - matches generate_references.py
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
     void ExpectComplexMatrixNear(const ComplexMatD &expected,
                                  const ComplexMatD &actual,
                                  T tolerance, const std::string &msg = "")
     {
         ASSERT_EQ(expected.size(), actual.size()) << msg << " - Bin count mismatch (Expected: " << expected.size() << ", Actual: " << actual.size() << ")";
         if (expected.empty())
             return;

         size_t num_bins = expected.size();
         ASSERT_GT(num_bins, 0) << msg << " - Zero bins in non-empty expected matrix";
         size_t num_frames = expected[0].size();

         if (actual.empty())
         {
             FAIL() << msg << " - Actual matrix is empty while expected has " << num_bins << " bins and " << num_frames << " frames.";
         }
         ASSERT_EQ(actual[0].size(), num_frames) << msg << " - Frame count mismatch for bin 0 (Expected: " << num_frames << ", Actual: " << actual[0].size() << ")";

         for (size_t i = 0; i < num_bins; ++i)
         {
             ASSERT_EQ(expected[i].size(), num_frames) << msg << " - Frame count mismatch in expected bin " << i;
             ASSERT_EQ(actual[i].size(), num_frames) << msg << " - Frame count mismatch in actual bin " << i;
             for (size_t j = 0; j < num_frames; ++j)
             {
                 EXPECT_NEAR(expected[i][j].real(), actual[i][j].real(), tolerance)
                     << msg << " - Mismatch at bin " << i << ", frame " << j << " (real)";
                 EXPECT_NEAR(expected[i][j].imag(), actual[i][j].imag(), tolerance)
                     << msg << " - Mismatch at bin " << i << ", frame " << j << " (imag)";
             }
         }
     }

     // Helper to compare real vectors
     template <typename T>
     void ExpectRealVectorNear(const std::vector<T> &expected,
                               const std::vector<T> &actual,
                               T tolerance, const std::string &msg = "")
     {
         ASSERT_EQ(expected.size(), actual.size()) << msg << " - Size mismatch (Expected: " << expected.size() << ", Actual: " << actual.size() << ")"; // Added size info
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
         // Tolerance might need adjustment based on Librosa vs OmniDSP differences
         return std::numeric_limits<T>::epsilon() * N * 1000;
     }

     // --- Helper to Load ALL Reference Data ---
     // Uses path defined by CMake preprocessor definition
     static void LoadReferenceData(const std::string &default_filename = "test_references.txt")
     {
         if (references_loaded)
             return;

         std::string filename_to_open;

         #ifdef OMNIDSP_TEST_REF_FILE_PATH
             filename_to_open = OMNIDSP_TEST_REF_FILE_PATH;
             std::cout << "Attempting to load reference data from CMake defined path: " << filename_to_open << std::endl;
         #else
             // Fallback if definition is missing (should not happen with CMake setup)
             filename_to_open = default_filename;
              std::cout << "Warning: OMNIDSP_TEST_REF_FILE_PATH not defined. Attempting relative path: " << filename_to_open << std::endl;
         #endif

         std::ifstream infile(filename_to_open);
         if (!infile.is_open()) {
             // Throw a specific error if open fails
             throw std::runtime_error("Failed to open reference data file. Path used: '" + filename_to_open +
                                      "'. Check CMake definition and file copy command.");
         }
         std::cout << "Loading reference data from: " << filename_to_open << std::endl;


         std::string line, current_key;
         bool reading_data = false;
         bool reading_complex_2d = false;
         size_t expected_rows = 0, expected_cols = 0;
         std::vector<double> flat_data_buffer; // Use one buffer, parse type later

         while (std::getline(infile, line))
         {
             // Trim whitespace
             line.erase(0, line.find_first_not_of(" \t\n\r"));
             line.erase(line.find_last_not_of(" \t\n\r") + 1);
             if (line.empty() || (line[0] == '#' && line.find("# START") != 0 && line.find("# END") != 0 && line.find("# SHAPE") != 0))
                 continue;

             if (line.rfind("# START ", 0) == 0)
             {
                 if (reading_data)
                     throw std::runtime_error("Found new START marker before END marker for key: " + current_key);
                 current_key = line.substr(8);
                 reading_data = true;
                 reading_complex_2d = (current_key.find("cqt_complex") != std::string::npos);
                 expected_rows = 0;
                 expected_cols = 0;
                 flat_data_buffer.clear();
             }
             else if (line.rfind("# END ", 0) == 0)
             {
                 if (!reading_data)
                     throw std::runtime_error("Found END marker without active START marker.");
                 std::string end_key = line.substr(6);
                 if (end_key != current_key)
                     throw std::runtime_error("Mismatched END marker. Expected: " + current_key + ", Got: " + end_key);

                 // Process collected data
                 if (reading_complex_2d)
                 {
                     if (expected_rows == 0 || expected_cols == 0)
                         throw std::runtime_error("Missing or invalid # SHAPE info for key: " + current_key);
                     size_t expected_parts = expected_rows * expected_cols * 2;
                     if (flat_data_buffer.size() != expected_parts)
                     {
                         throw std::runtime_error("Incorrect number of complex parts read for key '" + current_key + "'. Expected " + std::to_string(expected_parts) + ", Got " + std::to_string(flat_data_buffer.size()));
                     }
                     expected_librosa_cqt_d.assign(expected_rows, ComplexVecD(expected_cols));
                     for (size_t r = 0; r < expected_rows; ++r) {
                         for (size_t c = 0; c < expected_cols; ++c) {
                             size_t base_idx = (r * expected_cols + c) * 2;
                             expected_librosa_cqt_d[r][c] = {flat_data_buffer[base_idx], flat_data_buffer[base_idx + 1]};
                         }
                     }
                 }
                 else // Handle 1D float/double data
                 {
                     if (current_key.find("_f") != std::string::npos) { // Assume float if key ends with _f
                         VecF current_vec_f;
                         current_vec_f.reserve(flat_data_buffer.size());
                         for(double val : flat_data_buffer) {
                             current_vec_f.push_back(static_cast<float>(val));
                         }
                         expected_data_f[current_key] = current_vec_f;
                     } else { // Assume double otherwise
                         expected_data_d[current_key] = flat_data_buffer; // Direct move/copy
                     }
                 }

                 reading_data = false;
                 reading_complex_2d = false;
                 current_key = "";
             }
             else if (reading_data && line.rfind("# SHAPE: ", 0) == 0)
             {
                 if (!reading_complex_2d)
                     throw std::runtime_error("# SHAPE marker found for non-complex-2d key: " + current_key);
                 std::stringstream ss(line.substr(9));
                 if (!(ss >> expected_rows >> expected_cols)) {
                     throw std::runtime_error("Failed to parse # SHAPE info: " + line);
                 }
             }
             else if (reading_data)
             { // Read data values
                 try {
                     std::string value_str = line;
                     size_t comment_pos = value_str.find('#');
                     if (comment_pos != std::string::npos) {
                         value_str = value_str.substr(0, comment_pos);
                     }
                     value_str.erase(value_str.find_last_not_of(" \t\n\rf") + 1);
                     flat_data_buffer.push_back(std::stod(value_str));
                 } catch (const std::exception &e) {
                     throw std::runtime_error("Error parsing number '" + line + "' for key '" + current_key + "': " + e.what());
                 }
             }
         }
         if (reading_data)
             throw std::runtime_error("Reached end of file while reading data for key: " + current_key);
         references_loaded = true;
         infile.close();
     }


     // --- GTest Setup ---
     static void SetUpTestSuite()
     {
         try
         {
             LoadReferenceData(); // Load data once for all tests in the fixture
             std::cout << "PrecomputedRecursiveCQTTest: Reference data loaded successfully." << std::endl;
         }
         catch (const std::exception &e)
         {
              // Use ADD_FAILURE to report error without stopping other fixtures
             ADD_FAILURE() << "FATAL ERROR in PrecomputedRecursiveCQTTest::SetUpTestSuite: Loading reference data failed: " << e.what();
             // throw; // Optionally re-throw if tests are unusable
         }
     }

     // --- Helper to get loaded reference data ---
     const ComplexMatD &GetExpectedLibrosaCQT()
     {
         if (!references_loaded) throw std::runtime_error("Reference data accessed before SetUpTestSuite completed successfully.");
         if (expected_librosa_cqt_d.empty())
             throw std::runtime_error("Librosa CQT reference data not loaded or empty.");
         return expected_librosa_cqt_d;
     }
      // --- Helper to get loaded reference data (1D double) ---
     const VecD &GetExpectedD(const std::string &key) {
         if (!references_loaded) throw std::runtime_error("Reference data accessed before SetUpTestSuite completed successfully.");
         auto it = expected_data_d.find(key);
         if (it == expected_data_d.end()) throw std::runtime_error("Reference data key not found (double): " + key);
         return it->second;
     }
     // --- Helper to get loaded reference data (1D float) ---
     const VecF &GetExpectedF(const std::string &key) {
         if (!references_loaded) throw std::runtime_error("Reference data accessed before SetUpTestSuite completed successfully.");
         auto it = expected_data_f.find(key);
         if (it == expected_data_f.end()) throw std::runtime_error("Reference data key not found (float): " + key);
         return it->second;
     }


     // --- Accessors for Private CQTPlan Members (Requires Friend Declaration in cqt.h) ---
     template <typename T>
     static const std::vector<size_t> &getOctaveFFTLens(const OmniDSP::CQTPlan<T> &plan) { return plan.octave_fft_lens_; }
     template <typename T>
     static const std::vector<size_t> &getOctaveSpectrumLens(const OmniDSP::CQTPlan<T> &plan) { return plan.octave_spectrum_lens_; }
     template <typename T>
     static const std::vector<std::unique_ptr<OmniDSP::FFTPlan<T>>> &getOctaveSignalFFTPlans(const OmniDSP::CQTPlan<T> &plan) { return plan.octave_signal_fft_plans_; }
     template <typename T>
     static const std::vector<std::unique_ptr<OmniDSP::FFTPlan<T>>> &getOctaveKernelFFTPlans(const OmniDSP::CQTPlan<T> &plan) { return plan.octave_kernel_fft_plans_; }
     template <typename T>
     static const std::vector<typename OmniDSP::CQTPlan<T>::SparseKernelOctave> &getSparseKernels(const OmniDSP::CQTPlan<T> &plan) { return plan.precomputed_sparse_kernels_; }
     // Helper to call the private method calculateFirCoefficients
     template <typename T>
     static std::vector<T> callCalculateFirCoefficients(const OmniDSP::CQTPlan<T> &plan, double current_sr, int N) {
         return plan.calculateFirCoefficients(current_sr, N);
     }
     // Helper to call the private method filterAndDownsampleBy2
     template <typename T>
     static std::vector<T> callFilterAndDownsampleBy2(const OmniDSP::CQTPlan<T> &plan, const std::vector<T> &signal, double current_sr) {
         return plan.filterAndDownsampleBy2(signal, current_sr);
     }
 };

 // Initialize static members
 std::map<std::string, VecD> PrecomputedRecursiveCQTTest::expected_data_d;
 std::map<std::string, VecF> PrecomputedRecursiveCQTTest::expected_data_f;
 ComplexMatD PrecomputedRecursiveCQTTest::expected_librosa_cqt_d;
 bool PrecomputedRecursiveCQTTest::references_loaded = false;

 // --- Test Cases ---

 // 1. Test Constructor and Precomputation Sanity Checks
 TEST_F(PrecomputedRecursiveCQTTest, ConstructorPrecomputation)
 {
     using T = double;
     std::unique_ptr<OmniDSP::CQTPlan<T>> plan_ptr;
     ASSERT_NO_THROW(
         plan_ptr = std::make_unique<OmniDSP::CQTPlan<T>>(
             sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave,
             hannWindowCoeffsGenerator<T>, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST))
         << "Constructor threw unexpected exception.";
     ASSERT_NE(plan_ptr, nullptr);
     OmniDSP::CQTPlan<T> &plan = *plan_ptr;
     ASSERT_EQ(plan.getSampleRate(), sample_rate);
     ASSERT_EQ(plan.getHopLength(), hop_length);
     ASSERT_EQ(plan.getLowestFrequency(), lowest_freq);
     ASSERT_EQ(plan.getHighestFrequency(), high_freq);
     ASSERT_EQ(plan.getBinsPerOctave(), bins_per_octave);
     ASSERT_NEAR(plan.getSparsityThreshold(), sparsity_threshold_double, get_tolerance<T>());
     ASSERT_EQ(plan.getFirFilterOrder(), DEFAULT_FIR_FILTER_ORDER_TEST);
     int expected_octaves = 5;
     size_t expected_bins = static_cast<size_t>(std::ceil(bins_per_octave * std::log2(high_freq / lowest_freq)));
     ASSERT_EQ(plan.getNumOctaves(), expected_octaves);
     ASSERT_EQ(plan.getNumBins(), expected_bins);
     ASSERT_EQ(getOctaveFFTLens(plan).size(), expected_octaves);
     ASSERT_EQ(getOctaveSpectrumLens(plan).size(), expected_octaves);
     ASSERT_EQ(getOctaveSignalFFTPlans(plan).size(), expected_octaves);
     ASSERT_EQ(getOctaveKernelFFTPlans(plan).size(), expected_octaves);
     ASSERT_EQ(getSparseKernels(plan).size(), expected_octaves);
     for (int i = 0; i < expected_octaves; ++i) {
         ASSERT_GT(getOctaveFFTLens(plan)[i], 0);
         ASSERT_EQ(getOctaveSpectrumLens(plan)[i], getOctaveFFTLens(plan)[i] / 2 + 1);
         ASSERT_NE(getOctaveSignalFFTPlans(plan)[i], nullptr);
         ASSERT_NE(getOctaveKernelFFTPlans(plan)[i], nullptr);
         int bins_in_oct = (i == expected_octaves - 1) ? (static_cast<int>(expected_bins) - i * bins_per_octave) : bins_per_octave;
         ASSERT_EQ(getSparseKernels(plan)[i].size(), std::max(0, bins_in_oct));
         if (bins_in_oct > 0 && !getSparseKernels(plan)[i].empty() && !getSparseKernels(plan)[i][0].empty()) {
             ASSERT_FALSE(getSparseKernels(plan)[i][0].empty());
         }
     }
 }

 // 2. Test FIR Coefficient Calculation (Calls private helper via friend access)
 TEST_F(PrecomputedRecursiveCQTTest, CalculateFirCoefficients)
 {
     using T = double;
     OmniDSP::CQTPlan<T> plan(sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave,
                              hannWindowCoeffsGenerator<T>, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST);
     int N = DEFAULT_FIR_FILTER_ORDER_TEST;
     std::vector<T> coeffs;
     ASSERT_NO_THROW(coeffs = callCalculateFirCoefficients(plan, sample_rate, N));
     ASSERT_EQ(coeffs.size(), N);
     T symm_tolerance = get_tolerance<T>() * 10;
     for (int k = 0; k < N / 2; ++k) {
         ASSERT_NEAR(coeffs[k], coeffs[N - 1 - k], symm_tolerance);
     }
     T sum = std::accumulate(coeffs.begin(), coeffs.end(), T(0.0));
     ASSERT_NEAR(sum, T(1.0), get_tolerance<T>());
     int center_idx = (N - 1) / 2;
     ASSERT_GE(center_idx, 0);
     ASSERT_LT(center_idx, N);
     T center_val = coeffs[center_idx];
     ASSERT_GT(center_val, 0.0);
     for (int k = 0; k < N; ++k) {
         if (k != center_idx) {
             ASSERT_LE(coeffs[k], center_val + symm_tolerance);
         }
     }
 }

 // 3. Test Combined Filter and Downsample Step using Reference Data (Calls private helper)
 TEST_F(PrecomputedRecursiveCQTTest, FilterAndDownsample)
 {
     // Test Float precision
     {
         using T = float;
         double sr = 4000.0;
         size_t len = 2048;
         double freq_low = 500.0;
         double freq_high = 750.0;
         std::vector<T> test_signal_combined(len);
         for (size_t i = 0; i < len; ++i) {
             test_signal_combined[i] = T(0.5) * std::sin(2.0 * M_PI * freq_low * static_cast<double>(i) / sr) +
                                       T(0.5) * std::sin(2.0 * M_PI * freq_high * static_cast<double>(i) / sr);
         }
         OmniDSP::CQTPlan<T> plan(sr, 128, 400, 1600, 12, hannWindowCoeffsGenerator<T>,
                                  sparsity_threshold_float, DEFAULT_FIR_FILTER_ORDER_TEST);
         std::vector<T> result;
         ASSERT_NO_THROW(result = callFilterAndDownsampleBy2(plan, test_signal_combined, sr));

         size_t expected_len = 0;
         long long valid_len_signed = static_cast<long long>(len) - static_cast<long long>(DEFAULT_FIR_FILTER_ORDER_TEST) + 1;
         if (valid_len_signed > 0) {
             size_t valid_len = static_cast<size_t>(valid_len_signed);
             expected_len = (valid_len > 0) ? ((valid_len - 1) / 2 + 1) : 0; // Expect 974
         }

         // This assertion is EXPECTED TO FAIL if the backend (IPP) returns 1024 samples
         ASSERT_EQ(result.size(), expected_len) << "Mismatch between backend output size (" << result.size()
                                                << ") and reference calculation size (" << expected_len << ")";

         // Compare against loaded reference data (will also fail if sizes don't match)
         const VecF& expected_ref = GetExpectedF("expected_filter_downsample_f");
         T tolerance = get_tolerance<T>() * 10;
         ExpectRealVectorNear(expected_ref, result, tolerance, "Filter+Downsample Float vs Reference");
     }

     // Now test Double precision - **EXPECT IT TO THROW**
     {
         using T = double;
         double sr = 4000.0;
         size_t len = 2048;
         double freq_low = 500.0;
         double freq_high = 750.0;
         std::vector<T> test_signal_combined(len);
          for (size_t i = 0; i < len; ++i) {
             test_signal_combined[i] = T(0.5) * std::sin(2.0 * M_PI * freq_low * static_cast<double>(i) / sr) +
                                       T(0.5) * std::sin(2.0 * M_PI * freq_high * static_cast<double>(i) / sr);
         }
         OmniDSP::CQTPlan<T> plan(sr, 128, 400, 1600, 12, hannWindowCoeffsGenerator<T>,
                                  sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST);
         std::vector<T> result;
         ASSERT_THROW(result = callFilterAndDownsampleBy2(plan, test_signal_combined, sr), std::runtime_error);
     }
 }


 // 4. Test Full Recursive CQT Execute Method against Librosa Reference
 TEST_F(PrecomputedRecursiveCQTTest, FullRecursiveCQT_Execute_vs_Librosa)
 {
      // Test Float precision first
     {
         using T = float;
         using Complex = std::complex<T>;
         std::vector<T> test_signal = generateSineWave<T>(cqt_signal_freq, cqt_signal_duration, sample_rate);
         OmniDSP::CQTPlan<T> plan(sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave,
                                  hannWindowCoeffsGenerator<T>, sparsity_threshold_float, DEFAULT_FIR_FILTER_ORDER_TEST);
         std::vector<std::vector<Complex>> cqt_output;
         ASSERT_NO_THROW(plan.execute(test_signal, cqt_output));

         const ComplexMatD &expected_cqt = GetExpectedLibrosaCQT();
         ASSERT_EQ(cqt_output.size(), expected_cqt.size()) << "Float Bin count mismatch vs Librosa reference.";
         if (!expected_cqt.empty()) {
             size_t expected_frames = expected_cqt[0].size();
             ASSERT_GT(expected_frames, 0) << "Librosa reference has zero frames.";
             ASSERT_FALSE(cqt_output.empty()) << "OmniDSP Float CQT output is empty.";
             ASSERT_EQ(cqt_output[0].size(), expected_frames) << "Float Frame count mismatch vs Librosa reference.";
         }
     }

      // Now test Double precision - **EXPECT IT TO THROW** during execution
     {
         using T = double;
         using Complex = std::complex<T>;
         std::vector<T> test_signal = generateSineWave<T>(cqt_signal_freq, cqt_signal_duration, sample_rate);
         OmniDSP::CQTPlan<T> plan(sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave,
                                  hannWindowCoeffsGenerator<T>, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST);
         std::vector<std::vector<Complex>> cqt_output;
         ASSERT_THROW(plan.execute(test_signal, cqt_output), std::runtime_error);
     }
 }


 // 5. Test Parameter Validation in Constructor
 TEST_F(PrecomputedRecursiveCQTTest, ParameterValidation)
 {
     using T = double;
     auto dummy_gen = hannWindowCoeffsGenerator<T>;
     ASSERT_NO_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave, dummy_gen, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST));
     ASSERT_THROW(OmniDSP::CQTPlan<T>(0.0, hop_length, lowest_freq, high_freq, bins_per_octave, dummy_gen, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument);
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, 0, lowest_freq, high_freq, bins_per_octave, dummy_gen, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument);
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, 0.0, high_freq, bins_per_octave, dummy_gen, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument);
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, lowest_freq, lowest_freq, bins_per_octave, dummy_gen, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument);
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, lowest_freq, sample_rate / 1.9, bins_per_octave, dummy_gen, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument); // > Nyquist check
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, lowest_freq, high_freq, 0, dummy_gen, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument);
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave, nullptr, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument);
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, 511, lowest_freq, high_freq, bins_per_octave, dummy_gen, sparsity_threshold_double, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument); // Invalid hop
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave, dummy_gen, -0.1, DEFAULT_FIR_FILTER_ORDER_TEST), std::invalid_argument); // Negative sparsity
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave, dummy_gen, sparsity_threshold_double, 0), std::invalid_argument);  // Zero FIR order
     ASSERT_THROW(OmniDSP::CQTPlan<T>(sample_rate, hop_length, lowest_freq, high_freq, bins_per_octave, dummy_gen, sparsity_threshold_double, 100), std::invalid_argument); // Even FIR order
 }
