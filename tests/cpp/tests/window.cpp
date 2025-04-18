#include "OmniDSP/window.h"  // Include the OmniDSP window header

#include <cmath>      // For std::abs
#include <limits>     // For std::numeric_limits
#include <stdexcept>  // For std::invalid_argument
#include <string>     // For std::string
#include <vector>     // For std::vector

#include "../TestDataLoader.h"  // Include the new data loader utility
#include "gtest/gtest.h"        // Google Test framework

// Define the test fixture
class WindowTest : public ::testing::Test {
 protected:
  // Define constants for test parameters if needed
  const std::string suite_name =
      "window";  // Corresponds to data/window/ directory

  // Tolerance for floating point comparisons
  const double abs_error_d = 1e-9;
  const float abs_error_f = 1e-5f;

  // Window size used for generating reference data
  // Make sure this matches the size used in the Python generation script
  const int window_size = 1024;

  // Helper function to compare real vectors with tolerance
  template <typename T>
  void ExpectVectorNear(const std::vector<T> &actual,
                        const std::vector<T> &expected, T abs_error,
                        const std::string &message = "") {
    ASSERT_EQ(actual.size(), expected.size())
        << message << " - Vector sizes differ.";
    for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_NEAR(actual[i], expected[i], abs_error)
          << message << " - Mismatch at index " << i;
    }
  }
};

// --- Window Function Tests ---

TEST_F(WindowTest, Hann_D) {
  std::string test_case_name = "Hann_D";  // Base name for data file
  // Load expected data using the new loader
  auto expected_d = TestDataLoader::loadVectorData<double>(
      suite_name, test_case_name + "_expected_d.txt");
  ASSERT_EQ(expected_d.size(), window_size) << "Reference data size mismatch";

  // Generate the window using the library function
  std::vector<double> actual_d(window_size);
  OmniDSP::hannWindow(actual_d.data(), window_size);

  // Compare results
  ExpectVectorNear(actual_d, expected_d, abs_error_d, test_case_name);
}

TEST_F(WindowTest, Hann_F) {
  std::string test_case_name = "Hann_F";
  auto expected_f = TestDataLoader::loadVectorData<float>(
      suite_name, test_case_name + "_expected_f.txt");
  ASSERT_EQ(expected_f.size(), window_size) << "Reference data size mismatch";

  std::vector<float> actual_f(window_size);
  OmniDSP::hannWindow(actual_f.data(), window_size);

  ExpectVectorNear(actual_f, expected_f, abs_error_f, test_case_name);
}

TEST_F(WindowTest, Hamming_D) {
  std::string test_case_name = "Hamming_D";
  auto expected_d = TestDataLoader::loadVectorData<double>(
      suite_name, test_case_name + "_expected_d.txt");
  ASSERT_EQ(expected_d.size(), window_size) << "Reference data size mismatch";

  std::vector<double> actual_d(window_size);
  OmniDSP::hammingWindow(actual_d.data(), window_size);

  ExpectVectorNear(actual_d, expected_d, abs_error_d, test_case_name);
}

TEST_F(WindowTest, Hamming_F) {
  std::string test_case_name = "Hamming_F";
  auto expected_f = TestDataLoader::loadVectorData<float>(
      suite_name, test_case_name + "_expected_f.txt");
  ASSERT_EQ(expected_f.size(), window_size) << "Reference data size mismatch";

  std::vector<float> actual_f(window_size);
  OmniDSP::hammingWindow(actual_f.data(), window_size);

  ExpectVectorNear(actual_f, expected_f, abs_error_f, test_case_name);
}

TEST_F(WindowTest, Blackman_D) {
  std::string test_case_name = "Blackman_D";
  auto expected_d = TestDataLoader::loadVectorData<double>(
      suite_name, test_case_name + "_expected_d.txt");
  ASSERT_EQ(expected_d.size(), window_size) << "Reference data size mismatch";

  std::vector<double> actual_d(window_size);
  OmniDSP::blackmanWindow(actual_d.data(), window_size);

  ExpectVectorNear(actual_d, expected_d, abs_error_d, test_case_name);
}

TEST_F(WindowTest, Blackman_F) {
  std::string test_case_name = "Blackman_F";
  auto expected_f = TestDataLoader::loadVectorData<float>(
      suite_name, test_case_name + "_expected_f.txt");
  ASSERT_EQ(expected_f.size(), window_size) << "Reference data size mismatch";

  std::vector<float> actual_f(window_size);
  OmniDSP::blackmanWindow(actual_f.data(), window_size);

  ExpectVectorNear(actual_f, expected_f, abs_error_f, test_case_name);
}

// Note: Kaiser window needs a beta parameter.
// The reference data generation script must use the same beta value.
// We'll assume beta=8.0 for this example, adjust if needed.
const double kaiser_beta = 8.0;

TEST_F(WindowTest, Kaiser_D) {
  std::string test_case_name =
      "Kaiser_D";  // Assuming beta=8.0 in filename/generation
  auto expected_d = TestDataLoader::loadVectorData<double>(
      suite_name, test_case_name + "_expected_d.txt");
  ASSERT_EQ(expected_d.size(), window_size) << "Reference data size mismatch";

  std::vector<double> actual_d(window_size);
  OmniDSP::kaiserWindow(actual_d.data(), window_size, kaiser_beta);

  // May need slightly larger tolerance for Kaiser depending on calculation
  // method
  ExpectVectorNear(actual_d, expected_d, abs_error_d * 10, test_case_name);
}

TEST_F(WindowTest, Kaiser_F) {
  std::string test_case_name =
      "Kaiser_F";  // Assuming beta=8.0 in filename/generation
  auto expected_f = TestDataLoader::loadVectorData<float>(
      suite_name, test_case_name + "_expected_f.txt");
  ASSERT_EQ(expected_f.size(), window_size) << "Reference data size mismatch";

  std::vector<float> actual_f(window_size);
  OmniDSP::kaiserWindow(actual_f.data(), window_size,
                        static_cast<float>(kaiser_beta));

  // May need slightly larger tolerance for Kaiser depending on calculation
  // method
  ExpectVectorNear(actual_f, expected_f, abs_error_f * 10, test_case_name);
}

// --- Edge Case Tests ---

TEST_F(WindowTest, EdgeCases) {
  // Test with size 0 - should not crash, maybe throw? Or return empty?
  // Current API takes raw pointer, so size 0 might be tricky.
  // Let's test size 1.
  const int size_one = 1;
  std::vector<double> actual_d1(size_one);
  std::vector<float> actual_f1(size_one);

  // Expected output for size 1 is usually [1.0] for symmetric windows
  std::vector<double> expected_d1 = {1.0};
  std::vector<float> expected_f1 = {1.0f};

  OmniDSP::hannWindow(actual_d1.data(), size_one);
  ExpectVectorNear(actual_d1, expected_d1, abs_error_d, "Hann_D size 1");
  OmniDSP::hannWindow(actual_f1.data(), size_one);
  ExpectVectorNear(actual_f1, expected_f1, abs_error_f, "Hann_F size 1");

  OmniDSP::hammingWindow(actual_d1.data(), size_one);
  ExpectVectorNear(actual_d1, expected_d1, abs_error_d, "Hamming_D size 1");
  OmniDSP::hammingWindow(actual_f1.data(), size_one);
  ExpectVectorNear(actual_f1, expected_f1, abs_error_f, "Hamming_F size 1");

  OmniDSP::blackmanWindow(actual_d1.data(), size_one);
  ExpectVectorNear(actual_d1, expected_d1, abs_error_d, "Blackman_D size 1");
  OmniDSP::blackmanWindow(actual_f1.data(), size_one);
  ExpectVectorNear(actual_f1, expected_f1, abs_error_f, "Blackman_F size 1");

  OmniDSP::kaiserWindow(actual_d1.data(), size_one, kaiser_beta);
  ExpectVectorNear(actual_d1, expected_d1, abs_error_d, "Kaiser_D size 1");
  OmniDSP::kaiserWindow(actual_f1.data(), size_one,
                        static_cast<float>(kaiser_beta));
  ExpectVectorNear(actual_f1, expected_f1, abs_error_f, "Kaiser_F size 1");

  // Test invalid arguments if applicable (e.g., negative size - though API uses
  // int) The current API might not easily allow testing negative size with raw
  // pointers. Testing invalid beta for Kaiser:
  // EXPECT_THROW(OmniDSP::kaiserWindow(actual_d1.data(), size_one, -1.0),
  // std::invalid_argument);
}

// TODO: Add tests for Bartlett window if implemented
// TODO: Add tests for other windows if implemented (Flattop, etc.)
