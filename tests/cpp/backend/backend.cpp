/**
 * @file tests/cpp/backend/backend.cpp
 * @brief Unit tests for backend convolution/correlation implementations.
 * Updated LoadReferenceData logic to use CMake-defined path.
 * Updated tests to call public API (convolve1d/correlate1d) and include correct
 * header. Removed stub-specific test (moved to backend_conv_test_stub.cpp).
 */
#include <gtest/gtest.h>

#include <fstream>
#include <iostream>  // For setup confirmation/debug
#include <limits>    // For numeric_limits
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Include the public API header for convolution/correlation
#include <OmniDSP/convolution.h>

// Define aliases for vector types
using VecD = std::vector<double>;
using VecF = std::vector<float>;

// Test fixture for backend convolution/correlation tests
class BackendConvTest : public ::testing::Test {
 protected:
  // Static members to hold reference data loaded once
  static std::map<std::string, VecD> expected_data_d;
  static std::map<std::string, VecF> expected_data_f;
  static bool references_loaded;

  // Helper to compare vectors with tolerance
  template <typename T>
  void ExpectVectorNear(
      const std::vector<T>& expected,
      const std::vector<T>& actual,
      T tolerance,
      const std::string& msg = "")
  {
    ASSERT_EQ(expected.size(), actual.size()) << msg << " - Size mismatch";
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(expected[i], actual[i], tolerance)
          << msg << " - Mismatch at index " << i;
    }
  }

  // Helper to get tolerance based on type
  template <typename T>
  T get_tolerance()
  {
    if constexpr (std::is_same_v<T, float>) {
      return std::numeric_limits<float>::epsilon() * 100;
    }
    else {
      return std::numeric_limits<double>::epsilon() * 1000;
    }
  }

  // Helper to load reference data from file
  // Uses path defined by CMake preprocessor definition
  static void LoadReferenceData(
      const std::string& default_filename = "test_references.txt")
  {
    if (references_loaded) return;

    std::string filename_to_open;

#ifdef OMNIDSP_TEST_REF_FILE_PATH
    filename_to_open = OMNIDSP_TEST_REF_FILE_PATH;
    std::cout << "Attempting to load reference data from CMake defined path: "
              << filename_to_open << std::endl;
#else
    // Fallback if definition is missing (should not happen with CMake setup)
    filename_to_open = default_filename;
    std::cout << "Warning: OMNIDSP_TEST_REF_FILE_PATH not defined. Attempting "
                 "relative path: "
              << filename_to_open << std::endl;
#endif

    std::ifstream infile(filename_to_open);
    if (!infile.is_open()) {
      throw std::runtime_error(
          "Failed to open reference data file. Path used: '" + filename_to_open
          + "'. Check CMake definition and file copy command.");
    }
    std::cout << "Loading reference data from: " << filename_to_open
              << std::endl;

    std::string line, current_key;
    bool reading_data = false;
    std::vector<double> flat_data_buffer;

    while (std::getline(infile, line)) {
      line.erase(0, line.find_first_not_of(" \t\n\r"));
      line.erase(line.find_last_not_of(" \t\n\r") + 1);
      if (line.empty()
          || (line[0] == '#' && line.find("# START") != 0
              && line.find("# END") != 0))
        continue;

      if (line.rfind("# START ", 0) == 0) {
        if (reading_data)
          throw std::runtime_error(
              "Found new START marker before END marker for key: "
              + current_key);
        current_key = line.substr(8);
        reading_data = true;
        flat_data_buffer.clear();
      }
      else if (line.rfind("# END ", 0) == 0) {
        if (!reading_data)
          throw std::runtime_error(
              "Found END marker without active START marker.");
        std::string end_key = line.substr(6);
        if (end_key != current_key)
          throw std::runtime_error(
              "Mismatched END marker. Expected: " + current_key
              + ", Got: " + end_key);

        if (current_key.find("_f") != std::string::npos) {
          VecF current_vec_f;
          current_vec_f.reserve(flat_data_buffer.size());
          for (double val : flat_data_buffer) {
            current_vec_f.push_back(static_cast<float>(val));
          }
          expected_data_f[current_key] = current_vec_f;
        }
        else {
          expected_data_d[current_key] = flat_data_buffer;
        }

        reading_data = false;
        current_key = "";
      }
      else if (reading_data) {
        try {
          std::string value_str = line;
          size_t comment_pos = value_str.find('#');
          if (comment_pos != std::string::npos) {
            value_str = value_str.substr(0, comment_pos);
          }
          value_str.erase(value_str.find_last_not_of(" \t\n\rf") + 1);
          flat_data_buffer.push_back(std::stod(value_str));
        }
        catch (const std::exception& e) {
          throw std::runtime_error(
              "Error parsing number '" + line + "' for key '" + current_key
              + "': " + e.what());
        }
      }
    }
    if (reading_data)
      throw std::runtime_error(
          "Reached end of file while reading data for key: " + current_key);
    references_loaded = true;
    infile.close();
  }

  static void SetUpTestSuite()
  {
    try {
      LoadReferenceData();
      std::cout << "BackendConvTest: Reference data loaded successfully."
                << std::endl;
    }
    catch (const std::exception& e) {
      ADD_FAILURE() << "FATAL ERROR in BackendConvTest::SetUpTestSuite: "
                       "Loading reference data failed: "
                    << e.what();
    }
  }

  const VecD& GetExpectedD(const std::string& key)
  {
    if (!references_loaded)
      throw std::runtime_error(
          "Reference data accessed before SetUpTestSuite completed "
          "successfully.");
    auto it = expected_data_d.find(key);
    if (it == expected_data_d.end())
      throw std::runtime_error("Reference data key not found (double): " + key);
    return it->second;
  }
  const VecF& GetExpectedF(const std::string& key)
  {
    if (!references_loaded)
      throw std::runtime_error(
          "Reference data accessed before SetUpTestSuite completed "
          "successfully.");
    auto it = expected_data_f.find(key);
    if (it == expected_data_f.end())
      throw std::runtime_error("Reference data key not found (float): " + key);
    return it->second;
  }
};

// Initialize static members
std::map<std::string, VecD> BackendConvTest::expected_data_d;
std::map<std::string, VecF> BackendConvTest::expected_data_f;
bool BackendConvTest::references_loaded = false;

// --- Backend Agnostic Test Cases (These should pass regardless of backend) ---

TEST_F(BackendConvTest, Correlate1d_Valid_Double)
{
  using T = double;
  const VecD signal = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const VecD kernel = {0.5, 1.0, 0.5};  // Symmetric
  const VecD& expected = GetExpectedD("expected_corr_d");
  VecD result;
  ASSERT_NO_THROW(result = OmniDSP::correlate1d(signal, kernel));
  ExpectVectorNear(expected, result, get_tolerance<T>(), "Correlation Double");
}

TEST_F(BackendConvTest, Correlate1d_Valid_Float)
{
  using T = float;
  const VecF signal = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  const VecF kernel = {0.5f, 1.0f, 0.5f};  // Symmetric
  const VecF& expected = GetExpectedF("expected_corr_f");
  VecF result;
  ASSERT_NO_THROW(result = OmniDSP::correlate1d(signal, kernel));
  ExpectVectorNear(expected, result, get_tolerance<T>(), "Correlation Float");
}

TEST_F(BackendConvTest, Correlate1d_Valid_EdgeCase)
{
  using T = double;
  const VecD signal = {1.0, 2.0, 3.0};
  const VecD kernel = {0.5, 1.0, 0.5};
  const VecD& expected = GetExpectedD("expected_corr_edge_d");
  VecD result;
  ASSERT_NO_THROW(result = OmniDSP::correlate1d(signal, kernel));
  ExpectVectorNear(
      expected, result, get_tolerance<T>(), "Correlation Edge Double");
}

TEST_F(BackendConvTest, Convolve1d_Valid_Double)
{
  using T = double;
  const VecD signal = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  const VecD kernel = {1.0, 2.0, 0.5};  // Asymmetric
  const VecD& expected = GetExpectedD("expected_conv_d");
  VecD result;
  ASSERT_NO_THROW(result = OmniDSP::convolve1d(signal, kernel));
  ExpectVectorNear(expected, result, get_tolerance<T>(), "Convolution Double");
}

TEST_F(BackendConvTest, Convolve1d_Valid_Float)
{
  using T = float;
  const VecF signal = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  const VecF kernel = {1.0f, 2.0f, 0.5f};  // Asymmetric
  const VecF& expected = GetExpectedF("expected_conv_f");
  VecF result;
  ASSERT_NO_THROW(result = OmniDSP::convolve1d(signal, kernel));
  ExpectVectorNear(expected, result, get_tolerance<T>(), "Convolution Float");
}

TEST_F(BackendConvTest, Convolve1d_Valid_EdgeCase)
{
  using T = double;
  const VecD signal = {1.0, 2.0, 3.0};
  const VecD kernel = {0.5, 1.0, 0.5};
  const VecD& expected = GetExpectedD("expected_conv_edge_d");
  VecD result;
  ASSERT_NO_THROW(result = OmniDSP::convolve1d(signal, kernel));
  ExpectVectorNear(
      expected, result, get_tolerance<T>(), "Convolution Edge Double");
}

TEST_F(BackendConvTest, ConvCorr_InvalidInput_ThrowsOrEmpty)
{
  VecD signal_d = {1.0, 2.0, 3.0};
  VecD kernel_d = {1.0, 2.0, 3.0, 4.0};  // Kernel longer than signal
  VecD empty_d = {};

  EXPECT_THROW(OmniDSP::correlate1d(signal_d, kernel_d), std::invalid_argument);
  EXPECT_THROW(OmniDSP::convolve1d(signal_d, kernel_d), std::invalid_argument);
  EXPECT_TRUE(OmniDSP::correlate1d(empty_d, kernel_d).empty());
  EXPECT_TRUE(OmniDSP::convolve1d(empty_d, kernel_d).empty());
  EXPECT_TRUE(OmniDSP::correlate1d(signal_d, empty_d).empty());
  EXPECT_TRUE(OmniDSP::convolve1d(signal_d, empty_d).empty());
}

// ---- STUB TEST REMOVED FROM HERE ----
