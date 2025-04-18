// tests/window.cpp
// Tests for window application functions using reference data.

#include <OmniDSP/window.h>  // Header for OmniDSP::Window class
#include <gtest/gtest.h>

#include <cmath>
#include <limits>     // For numeric_limits
#include <numeric>    // For std::accumulate, std::vector initialization
#include <stdexcept>  // For exception checks
#include <vector>

#include "test_data_utils.h"  // Use the data loading utility

// --- Test Fixture ---
class WindowTest : public ::testing::Test {
 protected:
  // Window length matches the one used in generate_references.py
  const size_t N = 5;
  // Kaiser beta matches the one used in generate_references.py
  const double kaiser_beta = 8.0;

  // Tolerances might need adjustment based on backend precision
  const double double_tolerance =
      std::numeric_limits<double>::epsilon() * N * 10;
  const float float_tolerance =
      std::numeric_limits<float>::epsilon() * N * 10;  // Float tolerance

  // Helper to compare real vectors with tolerance
  template <typename T>
  void ExpectVectorNear(const std::vector<T> &expected,
                        const std::vector<T> &actual, T tolerance,
                        const std::string &msg = "") {
    ASSERT_EQ(expected.size(), actual.size()) << msg << " - Size mismatch";
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(expected[i], actual[i], tolerance)
          << msg << " - Mismatch at index " << i;
    }
  }
};

// --- Test Cases ---

TEST_F(WindowTest, Hann) {
  // Test Double Precision
  {
    std::vector<double> input_signal(N, 1.0);  // Dummy input of ones
    std::vector<double> actual_windowed_signal;
    // Correct API call: OmniDSP::Window class, static hann method
    ASSERT_NO_THROW(actual_windowed_signal =
                        OmniDSP::Window::hann<double>(input_signal));
    // Compare against reference coefficients (since input was 1.0)
    const auto &expected_coeffs =
        TestDataUtils::getExpectedDoubleVec("WINDOW_HANN_N5_D");
    ExpectVectorNear(expected_coeffs, actual_windowed_signal, double_tolerance,
                     "Hann Window (Double)");
  }
  // Test Float Precision
  {
    std::vector<float> input_signal(N, 1.0f);  // Dummy input of ones
    std::vector<float> actual_windowed_signal;
    // Correct API call: OmniDSP::Window class, static hann method
    ASSERT_NO_THROW(actual_windowed_signal =
                        OmniDSP::Window::hann<float>(input_signal));
    // Compare against reference coefficients (since input was 1.0)
    const auto &expected_coeffs =
        TestDataUtils::getExpectedFloatVec("WINDOW_HANN_N5_F");
    ExpectVectorNear(expected_coeffs, actual_windowed_signal, float_tolerance,
                     "Hann Window (Float)");
  }
}

TEST_F(WindowTest, Hamming) {
  // Test Double Precision
  {
    std::vector<double> input_signal(N, 1.0);
    std::vector<double> actual_windowed_signal;
    // Correct API call
    ASSERT_NO_THROW(actual_windowed_signal =
                        OmniDSP::Window::hamming<double>(input_signal));
    const auto &expected_coeffs =
        TestDataUtils::getExpectedDoubleVec("WINDOW_HAMMING_N5_D");
    ExpectVectorNear(expected_coeffs, actual_windowed_signal, double_tolerance,
                     "Hamming Window (Double)");
  }
  // Test Float Precision
  {
    std::vector<float> input_signal(N, 1.0f);
    std::vector<float> actual_windowed_signal;
    // Correct API call
    ASSERT_NO_THROW(actual_windowed_signal =
                        OmniDSP::Window::hamming<float>(input_signal));
    const auto &expected_coeffs =
        TestDataUtils::getExpectedFloatVec("WINDOW_HAMMING_N5_F");
    ExpectVectorNear(expected_coeffs, actual_windowed_signal, float_tolerance,
                     "Hamming Window (Float)");
  }
}

TEST_F(WindowTest, Kaiser) {
  // Test Double Precision
  {
    std::vector<double> input_signal(N, 1.0);
    std::vector<double> actual_windowed_signal;
    // Correct API call
    ASSERT_NO_THROW(actual_windowed_signal = OmniDSP::Window::kaiser<double>(
                        input_signal, kaiser_beta));
    const auto &expected_coeffs =
        TestDataUtils::getExpectedDoubleVec("WINDOW_KAISER_N5_B8_D");
    ExpectVectorNear(expected_coeffs, actual_windowed_signal, double_tolerance,
                     "Kaiser Window (Double)");
  }
  // Test Float Precision
  {
    std::vector<float> input_signal(N, 1.0f);
    std::vector<float> actual_windowed_signal;
    // Correct API call
    ASSERT_NO_THROW(actual_windowed_signal = OmniDSP::Window::kaiser<float>(
                        input_signal, static_cast<float>(kaiser_beta)));
    const auto &expected_coeffs =
        TestDataUtils::getExpectedFloatVec("WINDOW_KAISER_N5_B8_F");
    ExpectVectorNear(expected_coeffs, actual_windowed_signal, float_tolerance,
                     "Kaiser Window (Float)");
  }
}

TEST_F(WindowTest, FlatTop) {
  // Test Double Precision
  {
    std::vector<double> input_signal(N, 1.0);
    std::vector<double> actual_windowed_signal;
    // Correct API call
    ASSERT_NO_THROW(actual_windowed_signal =
                        OmniDSP::Window::flattop<double>(input_signal));
    const auto &expected_coeffs =
        TestDataUtils::getExpectedDoubleVec("WINDOW_FLATTOP_N5_D");
    ExpectVectorNear(expected_coeffs, actual_windowed_signal, double_tolerance,
                     "FlatTop Window (Double)");
  }
  // Test Float Precision
  {
    std::vector<float> input_signal(N, 1.0f);
    std::vector<float> actual_windowed_signal;
    // Correct API call
    ASSERT_NO_THROW(actual_windowed_signal =
                        OmniDSP::Window::flattop<float>(input_signal));
    const auto &expected_coeffs =
        TestDataUtils::getExpectedFloatVec("WINDOW_FLATTOP_N5_F");
    ExpectVectorNear(expected_coeffs, actual_windowed_signal, float_tolerance,
                     "FlatTop Window (Float)");
  }
}

// Test edge case N=1 (Input size 1)
TEST_F(WindowTest, SizeOne) {
  // Test Double Precision
  {
    std::vector<double> input_signal(1, 1.0);
    std::vector<double> actual_windowed_signal;
    ASSERT_NO_THROW(actual_windowed_signal =
                        OmniDSP::Window::hann<double>(input_signal));
    ASSERT_EQ(actual_windowed_signal.size(), 1);
    EXPECT_NEAR(actual_windowed_signal[0], 1.0, double_tolerance)
        << "Hann N=1 (Double)";

    ASSERT_NO_THROW(actual_windowed_signal =
                        OmniDSP::Window::hamming<double>(input_signal));
    ASSERT_EQ(actual_windowed_signal.size(), 1);
    EXPECT_NEAR(actual_windowed_signal[0], 1.0, double_tolerance)
        << "Hamming N=1 (Double)";

    ASSERT_NO_THROW(actual_windowed_signal = OmniDSP::Window::kaiser<double>(
                        input_signal, kaiser_beta));
    ASSERT_EQ(actual_windowed_signal.size(), 1);
    EXPECT_NEAR(actual_windowed_signal[0], 1.0, double_tolerance)
        << "Kaiser N=1 (Double)";

    ASSERT_NO_THROW(actual_windowed_signal =
                        OmniDSP::Window::flattop<double>(input_signal));
    ASSERT_EQ(actual_windowed_signal.size(), 1);
    EXPECT_NEAR(actual_windowed_signal[0], 1.0, double_tolerance)
        << "FlatTop N=1 (Double)";
  }
  // Test Float Precision
  {
    std::vector<float> input_signal(1, 1.0f);
    std::vector<float> actual_windowed_signal;
    ASSERT_NO_THROW(actual_windowed_signal =
                        OmniDSP::Window::hann<float>(input_signal));
    ASSERT_EQ(actual_windowed_signal.size(), 1);
    EXPECT_NEAR(actual_windowed_signal[0], 1.0f, float_tolerance)
        << "Hann N=1 (Float)";

    ASSERT_NO_THROW(actual_windowed_signal =
                        OmniDSP::Window::hamming<float>(input_signal));
    ASSERT_EQ(actual_windowed_signal.size(), 1);
    EXPECT_NEAR(actual_windowed_signal[0], 1.0f, float_tolerance)
        << "Hamming N=1 (Float)";

    ASSERT_NO_THROW(actual_windowed_signal = OmniDSP::Window::kaiser<float>(
                        input_signal, static_cast<float>(kaiser_beta)));
    ASSERT_EQ(actual_windowed_signal.size(), 1);
    EXPECT_NEAR(actual_windowed_signal[0], 1.0f, float_tolerance)
        << "Kaiser N=1 (Float)";

    ASSERT_NO_THROW(actual_windowed_signal =
                        OmniDSP::Window::flattop<float>(input_signal));
    ASSERT_EQ(actual_windowed_signal.size(), 1);
    EXPECT_NEAR(actual_windowed_signal[0], 1.0f, float_tolerance)
        << "FlatTop N=1 (Float)";
  }
}

// Test edge case N=0 (Input size 0) - Should throw invalid_argument
TEST_F(WindowTest, SizeZero) {
  // Test Double Precision
  {
    std::vector<double> input_signal;  // Empty vector
    ASSERT_THROW(OmniDSP::Window::hann<double>(input_signal),
                 std::invalid_argument)
        << "Hann N=0 (Double)";
    ASSERT_THROW(OmniDSP::Window::hamming<double>(input_signal),
                 std::invalid_argument)
        << "Hamming N=0 (Double)";
    ASSERT_THROW(OmniDSP::Window::kaiser<double>(input_signal, kaiser_beta),
                 std::invalid_argument)
        << "Kaiser N=0 (Double)";
    ASSERT_THROW(OmniDSP::Window::flattop<double>(input_signal),
                 std::invalid_argument)
        << "FlatTop N=0 (Double)";
  }
  // Test Float Precision
  {
    std::vector<float> input_signal;  // Empty vector
    ASSERT_THROW(OmniDSP::Window::hann<float>(input_signal),
                 std::invalid_argument)
        << "Hann N=0 (Float)";
    ASSERT_THROW(OmniDSP::Window::hamming<float>(input_signal),
                 std::invalid_argument)
        << "Hamming N=0 (Float)";
    ASSERT_THROW(OmniDSP::Window::kaiser<float>(
                     input_signal, static_cast<float>(kaiser_beta)),
                 std::invalid_argument)
        << "Kaiser N=0 (Float)";
    ASSERT_THROW(OmniDSP::Window::flattop<float>(input_signal),
                 std::invalid_argument)
        << "FlatTop N=0 (Float)";
  }
}

// Test Kaiser specific edge case: negative beta - Should throw invalid_argument
TEST_F(WindowTest, KaiserNegativeBeta) {
  // Test Double Precision
  {
    std::vector<double> input_signal(N, 1.0);
    ASSERT_THROW(OmniDSP::Window::kaiser<double>(input_signal, -1.0),
                 std::invalid_argument)
        << "Kaiser Negative Beta (Double)";
  }
  // Test Float Precision
  {
    std::vector<float> input_signal(N, 1.0f);
    ASSERT_THROW(OmniDSP::Window::kaiser<float>(input_signal, -1.0f),
                 std::invalid_argument)
        << "Kaiser Negative Beta (Float)";
  }
}
