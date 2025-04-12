/**
 * @file tests/cpp/backend_conv_test.cpp
 * @brief Unit tests for backend implementations of Conv/Corr/Filter+Downsample.
 * Reads reference data from "test_references.txt".
 */

 #include <gtest/gtest.h>
 #include <vector>
 #include <string>
 #include <fstream>   // For file input (ifstream)
 #include <sstream>   // For string parsing (stringstream)
 #include <stdexcept> // For runtime_error
 #include <limits>    // For numeric_limits
 #include <map>       // To store loaded reference data
 #include <iostream>  // For cout
 
 // Include the internal header to get backend function declarations
 #include "../../src/omnidsp/backend/backend_impl.h"
 
 // Include public API for stub backend test
 #include <OmniDSP/convolution.h>
 
 // Define data types for clarity
 using VecD = std::vector<double>;
 using VecF = std::vector<float>;
 
 // --- Test Fixture ---
 class BackendConvTest : public ::testing::Test
 {
 protected:
     // --- Input Test Data ---
     VecD signal_d = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
     VecD kernel_d = {0.5, 1.0, 0.5};
     VecD kernel_d_asym = {1.0, 2.0, 0.5};
     VecF signal_f = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
     VecF kernel_f = {0.5f, 1.0f, 0.5f};
     VecF kernel_f_asym = {1.0f, 2.0f, 0.5f};
     VecD signal_edge = {1.0, 2.0, 3.0};
     VecD kernel_edge = {0.5, 1.0, 0.5};
     VecD empty_signal_d;
     VecD empty_kernel_d;
 
     // --- Storage for Loaded Reference Data ---
     static std::map<std::string, VecD> expected_data_d;
     static std::map<std::string, VecF> expected_data_f;
     static bool references_loaded;
 
     // --- Helper to Load ALL Reference Data ---
     static void LoadReferenceData(const std::string &filename = "test_references.txt")
     {
         if (references_loaded) return;
 
         std::ifstream infile(filename);
         if (!infile.is_open()) {
             throw std::runtime_error("Failed to open reference data file: " + filename);
         }
 
         std::string line, current_key;
         bool reading_data = false;
         VecD current_vec_d;
         VecF current_vec_f;
 
         while (std::getline(infile, line))
         {
             line.erase(0, line.find_first_not_of(" \t\n\r"));
             line.erase(line.find_last_not_of(" \t\n\r") + 1);
             if (line.empty()) continue;
 
             if (line.rfind("# START ", 0) == 0) {
                 if (reading_data) throw std::runtime_error("Found new START marker before END marker for key: " + current_key);
                 current_key = line.substr(8);
                 reading_data = true;
                 current_vec_d.clear();
                 current_vec_f.clear();
             } else if (line.rfind("# END ", 0) == 0) {
                  if (!reading_data) throw std::runtime_error("Found END marker without active START marker.");
                  std::string end_key = line.substr(6);
                  if (end_key != current_key) throw std::runtime_error("Mismatched END marker. Expected: " + current_key + ", Got: " + end_key);
 
                  if (!current_vec_f.empty()) { expected_data_f[current_key] = current_vec_f; }
                  else { expected_data_d[current_key] = current_vec_d; }
 
                  reading_data = false;
                  current_key = "";
             } else if (reading_data && line[0] != '#') {
                  try {
                     if (line.back() == 'f') { current_vec_f.push_back(std::stof(line)); }
                     else { current_vec_d.push_back(std::stod(line)); }
                  } catch (const std::exception& e) {
                      throw std::runtime_error("Error parsing number '" + line + "' for key '" + current_key + "': " + e.what());
                  }
             }
         }
          if (reading_data) throw std::runtime_error("Reached end of file while reading data for key: " + current_key);
         references_loaded = true;
         infile.close();
     }
 
     // --- GTest Setup ---
     static void SetUpTestSuite() {
         try {
              LoadReferenceData();
              std::cout << "Reference data loaded successfully." << std::endl;
              std::cout << "Loaded Double Keys:";
              for(const auto& pair : expected_data_d) std::cout << " " << pair.first;
              std::cout << std::endl;
              std::cout << "Loaded Float Keys:";
              for(const auto& pair : expected_data_f) std::cout << " " << pair.first;
              std::cout << std::endl;
         } catch (const std::exception& e) {
             FAIL() << "FATAL ERROR: Loading reference data failed: " << e.what();
         }
     }
 
     // --- Helper to get loaded reference data ---
     const VecD& GetExpectedD(const std::string& key) {
         auto it = expected_data_d.find(key);
         if (it == expected_data_d.end()) throw std::runtime_error("Reference data key not found in map (double): " + key);
         return it->second;
     }
     const VecF& GetExpectedF(const std::string& key) {
          auto it = expected_data_f.find(key);
         if (it == expected_data_f.end()) throw std::runtime_error("Reference data key not found in map (float): " + key);
         return it->second;
     }
 
     // --- Comparison Helper ---
     template <typename T>
     void ExpectRealVectorNear(const std::vector<T> &expected,
                               const std::vector<T> &actual,
                               T tolerance, const std::string &msg = "")
     {
         ASSERT_EQ(expected.size(), actual.size()) << msg << " - Size mismatch";
         for (size_t i = 0; i < expected.size(); ++i) {
             EXPECT_NEAR(expected[i], actual[i], tolerance)
                 << msg << " - Mismatch at index " << i
                 << " (Expected: " << expected[i] << ", Actual: " << actual[i] << ")";
         }
     }
 
     // --- Tolerance Helper ---
     template <typename T>
     T get_tolerance() { return std::numeric_limits<T>::epsilon() * 1000; }
 };
 
 // Initialize static members
 std::map<std::string, VecD> BackendConvTest::expected_data_d;
 std::map<std::string, VecF> BackendConvTest::expected_data_f;
 bool BackendConvTest::references_loaded = false;
 
 // --- Conditionally Compiled Tests for Specific Backends ---
 #if defined(USE_ONEMKL) || defined(USE_ACCELERATE)
 
 #if defined(USE_ONEMKL)
 #define CURRENT_BACKEND_STR "MKL"
 #elif defined(USE_ACCELERATE)
 #define CURRENT_BACKEND_STR "Accelerate"
 #endif
 
 // --- Correlation Tests ---
 TEST_F(BackendConvTest, Correlate1d_Valid_Double) {
     VecD result;
     ASSERT_NO_THROW(result = OmniDSP::Backend::convolve1d_impl(signal_d, kernel_d, true));
     ExpectRealVectorNear(GetExpectedD("expected_corr_d"), result, get_tolerance<double>(), CURRENT_BACKEND_STR " Correlation Double");
 }
 TEST_F(BackendConvTest, Correlate1d_Valid_Float) {
     VecF result;
     ASSERT_NO_THROW(result = OmniDSP::Backend::convolve1d_impl(signal_f, kernel_f, true));
     ExpectRealVectorNear(GetExpectedF("expected_corr_f"), result, get_tolerance<float>(), CURRENT_BACKEND_STR " Correlation Float");
 }
 TEST_F(BackendConvTest, Correlate1d_Valid_EdgeCase) {
     VecD result;
     ASSERT_NO_THROW(result = OmniDSP::Backend::convolve1d_impl(signal_edge, kernel_edge, true));
     ExpectRealVectorNear(GetExpectedD("expected_corr_edge_d"), result, get_tolerance<double>(), CURRENT_BACKEND_STR " Correlation Edge Double");
 }
 
 // --- Convolution Tests ---
 TEST_F(BackendConvTest, Convolve1d_Valid_Double) {
     VecD result;
     ASSERT_NO_THROW(result = OmniDSP::Backend::convolve1d_impl(signal_d, kernel_d_asym, false));
     ExpectRealVectorNear(GetExpectedD("expected_conv_d"), result, get_tolerance<double>(), CURRENT_BACKEND_STR " Convolution Double");
 }
 TEST_F(BackendConvTest, Convolve1d_Valid_Float) {
     VecF result;
     ASSERT_NO_THROW(result = OmniDSP::Backend::convolve1d_impl(signal_f, kernel_f_asym, false));
     ExpectRealVectorNear(GetExpectedF("expected_conv_f"), result, get_tolerance<float>(), CURRENT_BACKEND_STR " Convolution Float");
 }
 TEST_F(BackendConvTest, Convolve1d_Valid_EdgeCase) {
     VecD result;
     ASSERT_NO_THROW(result = OmniDSP::Backend::convolve1d_impl(signal_edge, kernel_edge, false));
     ExpectRealVectorNear(GetExpectedD("expected_conv_edge_d"), result, get_tolerance<double>(), CURRENT_BACKEND_STR " Convolution Edge Double");
 }
 
 // --- Filter and Downsample Tests ---
 TEST_F(BackendConvTest, FilterDownsampleBy2_Double) {
     int factor = 2;
     VecD result;
     ASSERT_NO_THROW(result = OmniDSP::Backend::filter_and_downsample_impl(signal_d, kernel_d, factor));
     ExpectRealVectorNear(GetExpectedD("expected_desamp_d"), result, get_tolerance<double>(), CURRENT_BACKEND_STR " Filter+Downsample(2) Double");
 }
 TEST_F(BackendConvTest, FilterDownsampleBy2_Float) {
     int factor = 2;
     VecF result;
     ASSERT_NO_THROW(result = OmniDSP::Backend::filter_and_downsample_impl(signal_f, kernel_f, factor));
     ExpectRealVectorNear(GetExpectedF("expected_desamp_f"), result, get_tolerance<float>(), CURRENT_BACKEND_STR " Filter+Downsample(2) Float");
 }
 TEST_F(BackendConvTest, FilterDownsampleBy2_EdgeCase) {
     int factor = 2;
     VecD result;
     ASSERT_NO_THROW(result = OmniDSP::Backend::filter_and_downsample_impl(signal_edge, kernel_edge, factor));
     ExpectRealVectorNear(GetExpectedD("expected_desamp_edge_d"), result, get_tolerance<double>(), CURRENT_BACKEND_STR " Filter+Downsample(2) Edge Double");
 }
 TEST_F(BackendConvTest, FilterDownsampleBy3_Double) {
     int factor = 3;
     VecD result;
     ASSERT_NO_THROW(result = OmniDSP::Backend::filter_and_downsample_impl(signal_d, kernel_d, factor));
     ExpectRealVectorNear(GetExpectedD("expected_desamp_d_f3"), result, get_tolerance<double>(), CURRENT_BACKEND_STR " Filter+Downsample(3) Double");
 }
 
 // --- Invalid Input Tests ---
 TEST_F(BackendConvTest, ConvCorr_InvalidInput_Throws) {
     // Kernel longer than signal for 'valid'/'direct' mode should throw invalid_argument
     EXPECT_THROW(OmniDSP::Backend::convolve1d_impl(signal_edge, signal_d, true), std::invalid_argument);
     EXPECT_THROW(OmniDSP::Backend::convolve1d_impl(signal_edge, signal_d, false), std::invalid_argument);
 
     // Empty inputs should return empty vector
     VecD res_empty1 = OmniDSP::Backend::convolve1d_impl(empty_signal_d, kernel_d, true);
     EXPECT_TRUE(res_empty1.empty());
     VecD res_empty2 = OmniDSP::Backend::convolve1d_impl(signal_d, empty_kernel_d, true);
     EXPECT_TRUE(res_empty2.empty());
 }
 
 TEST_F(BackendConvTest, FilterDownsample_InvalidInput_ThrowsOrEmpty) {
     // --- FIX: Expect throw when kernel is longer than signal ---
     // The backend convolve1d_impl throws invalid_argument in this case,
     // so filter_and_downsample_impl should also throw it.
     EXPECT_THROW(OmniDSP::Backend::filter_and_downsample_impl(signal_edge, signal_d, 2), std::invalid_argument);
     // --- End Fix ---
 
     // Invalid factor should throw
     EXPECT_THROW(OmniDSP::Backend::filter_and_downsample_impl(signal_d, kernel_d, 0), std::invalid_argument);
     EXPECT_THROW(OmniDSP::Backend::filter_and_downsample_impl(signal_d, kernel_d, -1), std::invalid_argument);
 
     // Empty inputs should return empty
     VecD res_empty1 = OmniDSP::Backend::filter_and_downsample_impl(empty_signal_d, kernel_d, 2);
     EXPECT_TRUE(res_empty1.empty());
     VecD res_empty2 = OmniDSP::Backend::filter_and_downsample_impl(signal_d, empty_kernel_d, 2);
     EXPECT_TRUE(res_empty2.empty());
 }
 
 #undef CURRENT_BACKEND_STR // Undefine macro
 
 #else // Stub Backend Tests
 
 TEST_F(BackendConvTest, StubBackend_PublicApiThrowsError) {
     std::vector<double> sig = {1.0, 2.0, 3.0};
     std::vector<double> ker = {1.0, 1.0};
     ASSERT_THROW(OmniDSP::correlate1d(sig, ker), std::runtime_error);
     ASSERT_THROW(OmniDSP::convolve1d(sig, ker), std::runtime_error);
     ASSERT_THROW(OmniDSP::Backend::filter_and_downsample_impl(sig, ker, 2), std::runtime_error);
 }
 
 #endif // Backend selection
 