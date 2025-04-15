/**
 * @file tests/cpp/backend_conv_test.cpp
 * @brief Unit tests for backend convolution/correlation implementations.
 * Updated LoadReferenceData logic to use CMake-defined path.
 * Updated tests to call public API (convolve1d/correlate1d) and include correct header.
 */
#include <gtest/gtest.h>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream> // For setup confirmation/debug
#include <limits>   // For numeric_limits

// Include the public API header for convolution/correlation
#include <OmniDSP/convolution.h>
// #include "../../src/omnidsp/backend/backend_impl.h" // No longer needed - test public API

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
    void ExpectVectorNear(const std::vector<T>& expected, const std::vector<T>& actual, T tolerance, const std::string& msg = "") {
        ASSERT_EQ(expected.size(), actual.size()) << msg << " - Size mismatch";
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(expected[i], actual[i], tolerance) << msg << " - Mismatch at index " << i;
        }
    }

    // Helper to get tolerance based on type
    template <typename T>
    T get_tolerance() {
        if constexpr (std::is_same_v<T, float>) {
            return std::numeric_limits<float>::epsilon() * 100;
        } else {
            return std::numeric_limits<double>::epsilon() * 1000;
        }
    }

    // Helper to load reference data from file
    // Uses path defined by CMake preprocessor definition
    static void LoadReferenceData(const std::string& default_filename = "test_references.txt") {
        if (references_loaded) return;

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
        std::vector<double> flat_data_buffer; // Use one buffer, parse type later

        while (std::getline(infile, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\n\r"));
            line.erase(line.find_last_not_of(" \t\n\r") + 1);
            // Skip empty lines and non-marker comments
            if (line.empty() || (line[0] == '#' && line.find("# START") != 0 && line.find("# END") != 0)) continue;

            if (line.rfind("# START ", 0) == 0) {
                if (reading_data) throw std::runtime_error("Found new START marker before END marker for key: " + current_key);
                current_key = line.substr(8);
                reading_data = true;
                flat_data_buffer.clear();
            } else if (line.rfind("# END ", 0) == 0) {
                if (!reading_data) throw std::runtime_error("Found END marker without active START marker.");
                std::string end_key = line.substr(6);
                if (end_key != current_key) throw std::runtime_error("Mismatched END marker. Expected: " + current_key + ", Got: " + end_key);

                // Process collected data (assuming only 1D float/double for conv tests)
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

                reading_data = false;
                current_key = "";
            } else if (reading_data) { // Read data values
                try {
                    std::string value_str = line;
                    size_t comment_pos = value_str.find('#');
                    if (comment_pos != std::string::npos) {
                        value_str = value_str.substr(0, comment_pos);
                    }
                    value_str.erase(value_str.find_last_not_of(" \t\n\rf") + 1); // Trim trailing space/f
                    flat_data_buffer.push_back(std::stod(value_str));
                } catch (const std::exception& e) {
                    throw std::runtime_error("Error parsing number '" + line + "' for key '" + current_key + "': " + e.what());
                }
            }
        }
        if (reading_data) throw std::runtime_error("Reached end of file while reading data for key: " + current_key);
        references_loaded = true;
        infile.close();
    }


    // GTest SetUpTestSuite: Called once before all tests in this fixture
    static void SetUpTestSuite() {
        try {
            LoadReferenceData(); // Load reference data
            std::cout << "BackendConvTest: Reference data loaded successfully." << std::endl;
        } catch (const std::exception& e) {
            // Use ADD_FAILURE to report error without stopping other fixtures
            ADD_FAILURE() << "FATAL ERROR in BackendConvTest::SetUpTestSuite: Loading reference data failed: " << e.what();
            // Optionally re-throw or exit if tests are unusable without data
            // throw; // Re-throwing will likely stop all tests
        }
    }

     // Helper to get loaded reference data (1D double)
    const VecD& GetExpectedD(const std::string& key) {
        if (!references_loaded) throw std::runtime_error("Reference data accessed before SetUpTestSuite completed successfully.");
        auto it = expected_data_d.find(key);
        if (it == expected_data_d.end()) throw std::runtime_error("Reference data key not found (double): " + key);
        return it->second;
    }
    // Helper to get loaded reference data (1D float)
    const VecF& GetExpectedF(const std::string& key) {
         if (!references_loaded) throw std::runtime_error("Reference data accessed before SetUpTestSuite completed successfully.");
        auto it = expected_data_f.find(key);
        if (it == expected_data_f.end()) throw std::runtime_error("Reference data key not found (float): " + key);
        return it->second;
    }
};

// Initialize static members
std::map<std::string, VecD> BackendConvTest::expected_data_d;
std::map<std::string, VecF> BackendConvTest::expected_data_f;
bool BackendConvTest::references_loaded = false;


// --- Test Cases ---

// Test Correlation (Double) - Uses Public API
TEST_F(BackendConvTest, Correlate1d_Valid_Double) {
    using T = double;
    const VecD signal = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const VecD kernel = {0.5, 1.0, 0.5}; // Symmetric
    const VecD& expected = GetExpectedD("expected_corr_d");
    VecD result;
    // Call public API
    ASSERT_NO_THROW(result = OmniDSP::correlate1d(signal, kernel));
    ExpectVectorNear(expected, result, get_tolerance<T>(), "Correlation Double");
}

// Test Correlation (Float) - Uses Public API
TEST_F(BackendConvTest, Correlate1d_Valid_Float) {
     using T = float;
    const VecF signal = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const VecF kernel = {0.5f, 1.0f, 0.5f}; // Symmetric
    const VecF& expected = GetExpectedF("expected_corr_f");
    VecF result;
     // Call public API
    ASSERT_NO_THROW(result = OmniDSP::correlate1d(signal, kernel));
    ExpectVectorNear(expected, result, get_tolerance<T>(), "Correlation Float");
}

// Test Correlation Edge Case (Double) - Uses Public API
TEST_F(BackendConvTest, Correlate1d_Valid_EdgeCase) {
    using T = double;
    const VecD signal = {1.0, 2.0, 3.0};
    const VecD kernel = {0.5, 1.0, 0.5};
    const VecD& expected = GetExpectedD("expected_corr_edge_d");
    VecD result;
     // Call public API
    ASSERT_NO_THROW(result = OmniDSP::correlate1d(signal, kernel));
    ExpectVectorNear(expected, result, get_tolerance<T>(), "Correlation Edge Double");
}


// Test Convolution (Double) - Uses Public API
TEST_F(BackendConvTest, Convolve1d_Valid_Double) {
     using T = double;
    const VecD signal = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    const VecD kernel = {1.0, 2.0, 0.5}; // Asymmetric
    const VecD& expected = GetExpectedD("expected_conv_d");
    VecD result;
     // Call public API
    ASSERT_NO_THROW(result = OmniDSP::convolve1d(signal, kernel));
    ExpectVectorNear(expected, result, get_tolerance<T>(), "Convolution Double");
}

// Test Convolution (Float) - Uses Public API
TEST_F(BackendConvTest, Convolve1d_Valid_Float) {
     using T = float;
    const VecF signal = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const VecF kernel = {1.0f, 2.0f, 0.5f}; // Asymmetric
    const VecF& expected = GetExpectedF("expected_conv_f");
    VecF result;
     // Call public API
    ASSERT_NO_THROW(result = OmniDSP::convolve1d(signal, kernel));
    ExpectVectorNear(expected, result, get_tolerance<T>(), "Convolution Float");
}

// Test Convolution Edge Case (Double) - Uses Public API
TEST_F(BackendConvTest, Convolve1d_Valid_EdgeCase) {
    using T = double;
    const VecD signal = {1.0, 2.0, 3.0};
    const VecD kernel = {0.5, 1.0, 0.5}; // Use symmetric here for edge case consistency
    const VecD& expected = GetExpectedD("expected_conv_edge_d");
    VecD result;
     // Call public API
    ASSERT_NO_THROW(result = OmniDSP::convolve1d(signal, kernel));
    ExpectVectorNear(expected, result, get_tolerance<T>(), "Convolution Edge Double");
}

// Test invalid inputs (using Public API)
TEST_F(BackendConvTest, ConvCorr_InvalidInput_ThrowsOrEmpty) {
    VecD signal_d = {1.0, 2.0, 3.0};
    VecD kernel_d = {1.0, 2.0, 3.0, 4.0}; // Kernel longer than signal
    VecD empty_d = {};

    // Expect invalid_argument if kernel > signal (check public API behavior)
    EXPECT_THROW(OmniDSP::correlate1d(signal_d, kernel_d), std::invalid_argument);
    EXPECT_THROW(OmniDSP::convolve1d(signal_d, kernel_d), std::invalid_argument);

    // Expect empty output if signal is empty
    EXPECT_TRUE(OmniDSP::correlate1d(empty_d, kernel_d).empty());
    EXPECT_TRUE(OmniDSP::convolve1d(empty_d, kernel_d).empty());

    // Expect empty output if kernel is empty
    EXPECT_TRUE(OmniDSP::correlate1d(signal_d, empty_d).empty());
    EXPECT_TRUE(OmniDSP::convolve1d(signal_d, empty_d).empty());
}


// Test specifically if stub backend throws (only runs if stub is compiled)
#if !defined(USE_ACCELERATE) && !defined(USE_ONEMKL)
TEST_F(BackendConvTest, StubBackend_PublicApiThrowsError) {
    VecD signal_d = {1.0, 2.0, 3.0};
    VecD kernel_d = {1.0, 0.5};
    // These call the public API which should dispatch to the throwing stub backend impl
    EXPECT_THROW(OmniDSP::convolve1d(signal_d, kernel_d), std::runtime_error);
    EXPECT_THROW(OmniDSP::correlate1d(signal_d, kernel_d), std::runtime_error);
}
#endif // Stub backend test guard
