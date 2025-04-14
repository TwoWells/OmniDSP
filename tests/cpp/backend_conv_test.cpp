/**
 * @file tests/cpp/backend_conv_test.cpp
 * @brief Unit tests for public convolution/correlation API functions.
 * Reads reference data from "test_references.txt".
 * Uses public API (convolve1d, correlate1d) instead of backend impl functions.
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

// Include the public API header for convolution/correlation
#include <OmniDSP/convolution.h>
// DO NOT include backend_impl.h here anymore

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
    VecF empty_signal_f; // Added for completeness
    VecF empty_kernel_f; // Added for completeness

    // --- Storage for Loaded Reference Data ---
    // Keep storage for expected conv/corr results
    static std::map<std::string, VecD> expected_data_d;
    static std::map<std::string, VecF> expected_data_f;
    static bool references_loaded;

    // --- Helper to Load ALL Reference Data ---
    // (Modified slightly to only load expected types)
    static void LoadReferenceData(const std::string &filename = "test_references.txt")
    {
        if (references_loaded)
            return;

        std::ifstream infile(filename);
        if (!infile.is_open())
        {
            throw std::runtime_error("Failed to open reference data file: " + filename);
        }

        std::string line, current_key;
        bool reading_data = false;
        VecD current_vec_d;
        VecF current_vec_f;

        while (std::getline(infile, line))
        {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\n\r"));
            line.erase(line.find_last_not_of(" \t\n\r") + 1);
            // Skip empty lines and comments (unless START/END)
            if (line.empty() || (line[0] == '#' && line.find("# START") != 0 && line.find("# END") != 0))
                continue;

            if (line.rfind("# START ", 0) == 0)
            {
                if (reading_data)
                    throw std::runtime_error("Found new START marker before END marker for key: " + current_key);
                current_key = line.substr(8);
                // Only process keys relevant to this test file
                if (current_key.find("expected_corr") == 0 || current_key.find("expected_conv") == 0)
                {
                    reading_data = true;
                    current_vec_d.clear();
                    current_vec_f.clear();
                }
                else
                {
                    reading_data = false; // Skip sections not needed here
                }
            }
            else if (line.rfind("# END ", 0) == 0)
            {
                if (!reading_data && (current_key.find("expected_corr") == 0 || current_key.find("expected_conv") == 0))
                {
                    // Only throw error if we *should* have been reading data for a relevant key
                    throw std::runtime_error("Found END marker without active START marker for relevant key.");
                }
                if (reading_data)
                { // Only process if we were reading relevant data
                    std::string end_key = line.substr(6);
                    if (end_key != current_key)
                        throw std::runtime_error("Mismatched END marker. Expected: " + current_key + ", Got: " + end_key);

                    // Store based on key suffix or collected type
                    if (!current_vec_f.empty())
                    {
                        expected_data_f[current_key] = current_vec_f;
                    }
                    else if (!current_vec_d.empty())
                    {
                        expected_data_d[current_key] = current_vec_d;
                    }
                    // else: maybe log warning if empty?
                }
                reading_data = false;
                current_key = "";
            }
            else if (reading_data)
            { // Read data values for relevant keys
                try
                {
                    if (line.back() == 'f')
                    {
                        current_vec_f.push_back(std::stof(line));
                    }
                    else
                    {
                        current_vec_d.push_back(std::stod(line));
                    }
                }
                catch (const std::exception &e)
                {
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
            LoadReferenceData();
            std::cout << "Reference data loaded successfully for BackendConvTest." << std::endl;
            // Optionally print loaded keys for verification
        }
        catch (const std::exception &e)
        {
            FAIL() << "FATAL ERROR: Loading reference data failed: " << e.what();
        }
    }

    // --- Helper to get loaded reference data ---
    const VecD &GetExpectedD(const std::string &key)
    {
        auto it = expected_data_d.find(key);
        if (it == expected_data_d.end())
            throw std::runtime_error("Reference data key not found in map (double): " + key);
        return it->second;
    }
    const VecF &GetExpectedF(const std::string &key)
    {
        auto it = expected_data_f.find(key);
        if (it == expected_data_f.end())
            throw std::runtime_error("Reference data key not found in map (float): " + key);
        return it->second;
    }

    // --- Comparison Helper ---
    template <typename T>
    void ExpectRealVectorNear(const std::vector<T> &expected,
                              const std::vector<T> &actual,
                              T tolerance, const std::string &msg = "")
    {
        ASSERT_EQ(expected.size(), actual.size()) << msg << " - Size mismatch";
        for (size_t i = 0; i < expected.size(); ++i)
        {
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

// --- Tests using Public API ---

// --- Correlation Tests ---
TEST_F(BackendConvTest, Correlate1d_Valid_Double)
{
    VecD result;
    // Use public API OmniDSP::correlate1d - ADDED <double>
    ASSERT_NO_THROW(result = OmniDSP::correlate1d<double>(signal_d, kernel_d));
    ExpectRealVectorNear(GetExpectedD("expected_corr_d"), result, get_tolerance<double>(), "Public API Correlation Double");
}
TEST_F(BackendConvTest, Correlate1d_Valid_Float)
{
    VecF result;
    // Use public API OmniDSP::correlate1d - ADDED <float>
    ASSERT_NO_THROW(result = OmniDSP::correlate1d<float>(signal_f, kernel_f));
    ExpectRealVectorNear(GetExpectedF("expected_corr_f"), result, get_tolerance<float>(), "Public API Correlation Float");
}
TEST_F(BackendConvTest, Correlate1d_Valid_EdgeCase)
{
    VecD result;
    // Use public API OmniDSP::correlate1d - ADDED <double>
    ASSERT_NO_THROW(result = OmniDSP::correlate1d<double>(signal_edge, kernel_edge));
    ExpectRealVectorNear(GetExpectedD("expected_corr_edge_d"), result, get_tolerance<double>(), "Public API Correlation Edge Double");
}

// --- Convolution Tests ---
TEST_F(BackendConvTest, Convolve1d_Valid_Double)
{
    VecD result;
    // Use public API OmniDSP::convolve1d - ADDED <double>
    ASSERT_NO_THROW(result = OmniDSP::convolve1d<double>(signal_d, kernel_d_asym));
    ExpectRealVectorNear(GetExpectedD("expected_conv_d"), result, get_tolerance<double>(), "Public API Convolution Double");
}
TEST_F(BackendConvTest, Convolve1d_Valid_Float)
{
    VecF result;
    // Use public API OmniDSP::convolve1d - ADDED <float>
    ASSERT_NO_THROW(result = OmniDSP::convolve1d<float>(signal_f, kernel_f_asym));
    ExpectRealVectorNear(GetExpectedF("expected_conv_f"), result, get_tolerance<float>(), "Public API Convolution Float");
}
TEST_F(BackendConvTest, Convolve1d_Valid_EdgeCase)
{
    VecD result;
    // Use public API OmniDSP::convolve1d - ADDED <double>
    ASSERT_NO_THROW(result = OmniDSP::convolve1d<double>(signal_edge, kernel_edge));
    ExpectRealVectorNear(GetExpectedD("expected_conv_edge_d"), result, get_tolerance<double>(), "Public API Convolution Edge Double");
}

// --- Filter and Downsample Tests ---
// REMOVED: These tests called internal backend functions directly which are not exported.
// To test filtering + downsampling, one would typically call the public correlate1d
// (for FIR filtering) and then manually downsample the result if needed, or create
// a dedicated public API function if this combined operation is frequently required.
// TEST_F(BackendConvTest, FilterDownsampleBy2_Double) { ... }
// TEST_F(BackendConvTest, FilterDownsampleBy2_Float) { ... }
// TEST_F(BackendConvTest, FilterDownsampleBy2_EdgeCase) { ... }
// TEST_F(BackendConvTest, FilterDownsampleBy3_Double) { ... }

// --- Invalid Input Tests (Using Public API) ---
TEST_F(BackendConvTest, ConvCorr_InvalidInput_ThrowsOrEmpty)
{
    // Kernel longer than signal for 'valid' mode should throw invalid_argument via public API
    // ADDED <double>
    EXPECT_THROW(OmniDSP::correlate1d<double>(signal_edge, signal_d), std::invalid_argument);
    EXPECT_THROW(OmniDSP::convolve1d<double>(signal_edge, signal_d), std::invalid_argument);

    // Empty inputs should return empty vector via public API
    VecD res_empty1, res_empty3;
    VecF res_empty2, res_empty4;
    // ADDED <double> and <float>
    ASSERT_NO_THROW(res_empty1 = OmniDSP::correlate1d<double>(empty_signal_d, kernel_d));
    EXPECT_TRUE(res_empty1.empty());
    ASSERT_NO_THROW(res_empty2 = OmniDSP::correlate1d<float>(signal_f, empty_kernel_f)); // Use float kernel here
    EXPECT_TRUE(res_empty2.empty());
    ASSERT_NO_THROW(res_empty3 = OmniDSP::convolve1d<double>(empty_signal_d, kernel_d));
    EXPECT_TRUE(res_empty3.empty());
    ASSERT_NO_THROW(res_empty4 = OmniDSP::convolve1d<float>(signal_f, empty_kernel_f)); // Use float kernel here
    EXPECT_TRUE(res_empty4.empty());
}

// Removed test for filter_and_downsample invalid factor as the function is no longer called directly.

// --- Stub Backend Test (If Applicable) ---
// This test checks if the public API throws when the stub backend is active.
// It requires the test executable to be linked against the stub implementation.
#if !defined(USE_ONEMKL) && !defined(USE_ACCELERATE)
TEST_F(BackendConvTest, StubBackend_PublicApiThrowsError)
{
    // Use public API functions - ADDED <double>
    ASSERT_THROW(OmniDSP::correlate1d<double>(signal_d, kernel_d), std::runtime_error);
    ASSERT_THROW(OmniDSP::convolve1d<double>(signal_d, kernel_d), std::runtime_error);
    // Cannot test filter_and_downsample directly via public API easily
}
#endif // Stub backend check