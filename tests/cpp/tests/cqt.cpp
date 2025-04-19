#include "OmniDSP/cqt.h"  // Include the OmniDSP CQT header

#include <algorithm>  // For std::max, std::min_element etc. if needed
#include <cmath>      // For std::abs, std::log2, std::ceil, std::pow, etc.
#include <complex>    // For std::complex
#include <limits>     // For std::numeric_limits
#include <memory>     // For std::unique_ptr if accessing internals
#include <numeric>    // For std::iota, std::accumulate
#include <stdexcept>  // For std::invalid_argument, std::runtime_error
#include <string>     // For std::string
#include <vector>     // For std::vector

#include "../TestDataLoader.h"   // Include the new data loader utility
#include "OmniDSP/core_types.h"  // Include for Precision enum
#include "OmniDSP/fft.h"         // Include for FFTNorm enum
#include "OmniDSP/resample.h"    // <<< MOVED INCLUDE HERE
#include "OmniDSP/window.h"      // Needed for M_PI in helper if used locally
#include "gtest/gtest.h"         // Google Test framework

// Define M_PI if it's not already defined (e.g., by <cmath> in some
// environments/standards)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper function to compare complex vectors with tolerance
template <typename T>
void ExpectComplexVectorNear(const std::vector<std::complex<T>> &actual,
                             const std::vector<std::complex<T>> &expected,
                             T abs_error, const std::string &message = "") {
  ASSERT_EQ(actual.size(), expected.size())
      << message << " - Vector sizes differ.";
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i].real(), expected[i].real(), abs_error)
        << message << " - Mismatch at index " << i << " (real part)";
    EXPECT_NEAR(actual[i].imag(), expected[i].imag(), abs_error)
        << message << " - Mismatch at index " << i << " (imaginary part)";
  }
}

// Helper function to compare real vectors with tolerance
template <typename T>
void ExpectRealVectorNear(const std::vector<T> &actual,
                          const std::vector<T> &expected, T abs_error,
                          const std::string &message = "") {
  ASSERT_EQ(actual.size(), expected.size())
      << message << " - Vector sizes differ.";
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], abs_error)
        << message << " - Mismatch at index " << i;
  }
}

// Helper function to compare complex matrices (vector of vectors)
template <typename T>
void ExpectComplexMatrixNear(
    const std::vector<std::vector<std::complex<T>>> &actual,
    const std::vector<std::vector<std::complex<T>>> &expected, T abs_error,
    const std::string &message = "") {
  ASSERT_EQ(actual.size(), expected.size())
      << message << " - Matrix row counts differ.";
  if (expected.empty() && actual.empty()) return;  // Both empty is OK
  if (expected.empty() || actual.empty()) {
    FAIL() << message << " - One matrix is empty while the other is not.";
  }

  ASSERT_GT(expected.size(), 0) << message << " - Expected matrix has no rows.";
  ASSERT_GT(actual.size(), 0) << message << " - Actual matrix has no rows.";

  size_t num_rows = expected.size();
  size_t num_cols = expected[0].size();  // Assume consistent cols in expected

  for (size_t i = 0; i < num_rows; ++i) {
    ASSERT_EQ(actual[i].size(), num_cols)
        << message << " - Matrix column counts differ at row " << i
        << " (Expected: " << num_cols << ", Actual: " << actual[i].size()
        << ")";
    ASSERT_EQ(expected[i].size(), num_cols)  // Verify expected consistency
        << message << " - Expected matrix has inconsistent columns at row "
        << i;

    for (size_t j = 0; j < num_cols; ++j) {
      ASSERT_NEAR(actual[i][j].real(), expected[i][j].real(), abs_error)
          << message << " - Mismatch at (" << i << ", " << j << ") (real part)";
      ASSERT_NEAR(actual[i][j].imag(), expected[i][j].imag(), abs_error)
          << message << " - Mismatch at (" << i << ", " << j
          << ") (imaginary part)";
    }
  }
}

// --- Test Fixture ---
class CQT_Test : public ::testing::Test {
 protected:
  // Constants matching reference data generation (if applicable)
  // Using parameters from the failing tests for now
  const double sample_rate = 22050.0;
  const double fmin = 32.7;  // C1
  const int n_bins = 84;
  const int bins_per_octave = 12;
  // const int n_octaves = 7; // n_bins / bins_per_octave (derived)

  // Tolerances
  const double abs_error_d = 1e-7;  // CQT might need slightly larger tolerance
  const float abs_error_f = 1e-4f;

  const std::string suite_name = "cqt";  // Corresponds to data/cqt/ directory
};

// --- Recursive CQT Tests ---
// Tests using the recursive calculation approach

class RecursiveCQTTest : public CQT_Test {
  // No specific setup needed for recursive tests unless loading common data
};

TEST_F(RecursiveCQTTest, FullRecursiveCQT_Execute_Double) {
  std::string test_case_name = "FullRecursiveCQT_Execute_Double";
  auto input_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_input_cd.txt");
  // Expected output is complex double, likely flattened CQT matrix or similar
  auto expected_cqt_d = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_expected_cd.txt");

  // Create CQT plan (will calculate filters recursively) - CORRECTED
  // CONSTRUCTOR CALL
  OmniDSP::CQTPlan<double> plan(
      sample_rate,  // sr
      fmin,         // fmin
      n_bins,       // n_bins
      bins_per_octave,
      OmniDSP::Precision::DOUBLE,  // Added precision
      OmniDSP::FFTNorm::Ortho);    // Added norm (e.g., Ortho)

  // Prepare output vector
  // Size needs to match the expected output format from reference generation
  std::vector<std::complex<double>> output_cqt_d(expected_cqt_d.size());

  // Execute CQT
  plan.execute(input_cd, output_cqt_d);

  // Compare results
  ExpectComplexVectorNear(output_cqt_d, expected_cqt_d, abs_error_d,
                          test_case_name);
}

TEST_F(RecursiveCQTTest, FullRecursiveCQT_Execute_Float) {
  std::string test_case_name = "FullRecursiveCQT_Execute_Float";
  auto input_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_input_cf.txt");
  auto expected_cqt_f = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_expected_cf.txt");

  // CORRECTED CONSTRUCTOR CALL
  OmniDSP::CQTPlan<float> plan(
      static_cast<float>(sample_rate),  // sr
      static_cast<float>(fmin),         // fmin
      n_bins,                           // n_bins
      bins_per_octave,
      OmniDSP::Precision::SINGLE,  // Added precision
      OmniDSP::FFTNorm::Ortho);    // Added norm (e.g., Ortho)

  std::vector<std::complex<float>> output_cqt_f(expected_cqt_f.size());
  plan.execute(input_cf, output_cqt_f);

  ExpectComplexVectorNear(output_cqt_f, expected_cqt_f, abs_error_f,
                          test_case_name);
}

// --- Filter And Downsample Test ---
// This test focuses on the internal resampling helper function used by CQT
// Note: This test might need adjustment if filterAndDownsampleBy2 is private
// or if the public API OmniDSP::filter_and_downsample should be tested instead.

class FilterAndDownsampleTest : public CQT_Test {
  // Test fixture specific for filterAndDownsample
};

// Test combined Filter and Downsample step (using float precision)
TEST_F(FilterAndDownsampleTest,
       FilterAndDownsample_Float) {  // Renamed test slightly
  std::string test_case_name =
      "FilterAndDownsample";  // Base name for data files

  // Load float data
  auto input_f = TestDataLoader::loadVectorData<float>(
      suite_name, test_case_name + "_input_f.txt");
  auto filter_coeffs_f = TestDataLoader::loadVectorData<float>(
      suite_name, test_case_name + "_filter_coeffs_f.txt");
  auto expected_f = TestDataLoader::loadVectorData<float>(
      suite_name, test_case_name + "_expected_f.txt");

  // Use the public API function for testing the operation (float version)
  std::vector<float> output_f;
  int downsample_factor = 2;  // Assuming factor of 2 for this test case

  try {
    // Explicitly call the float version
    output_f = OmniDSP::filter_and_downsample<float>(input_f, filter_coeffs_f,
                                                     downsample_factor);
  } catch (const std::exception &e) {
    FAIL() << "OmniDSP::filter_and_downsample<float> threw an exception: "
           << e.what();
  }

  // Compare results using float tolerance
  ExpectRealVectorNear(output_f, expected_f, abs_error_f,
                       test_case_name + "_Float");
}

#if !defined(USE_ONEMKL)
TEST_F(FilterAndDownsampleTest, FilterAndDownsample) {
  // This test assumes the existence of reference data generated for a
  // filter+downsample operation, potentially within the 'cqt' suite.
  std::string test_case_name = "FilterAndDownsample";

  // Load data
  auto input_d = TestDataLoader::loadVectorData<double>(
      suite_name, test_case_name + "_input_d.txt");
  auto filter_coeffs_d = TestDataLoader::loadVectorData<double>(
      suite_name, test_case_name + "_filter_coeffs_d.txt");
  auto expected_d = TestDataLoader::loadVectorData<double>(
      suite_name, test_case_name + "_expected_d.txt");

  // Use the public API function for testing the operation
  std::vector<double> output_d;
  int downsample_factor = 2;  // Assuming factor of 2 for this test case

  try {
    // #include "OmniDSP/resample.h" // <<< INCLUDE MOVED TO TOP
    output_d = OmniDSP::filter_and_downsample(input_d, filter_coeffs_d,
                                              downsample_factor);
  } catch (const std::exception &e) {
    FAIL() << "OmniDSP::filter_and_downsample threw an exception: " << e.what();
  }

  // Compare results
  ExpectRealVectorNear(output_d, expected_d, abs_error_d, test_case_name);
}
#endif

// --- Precomputed CQT Tests ---
// Tests using the precomputed filter bank approach (Currently Disabled)

class PrecomputedCQTTest : public CQT_Test {
 protected:
  // Placeholder for potential setup involving loading filter banks
};

// This test is currently disabled because it relies on the 2D filter bank
// loading and potentially a different CQTPlan constructor/method.
TEST_F(PrecomputedCQTTest, DISABLED_ExecutePrecomputed) {
  GTEST_SKIP()
      << "Test disabled. Requires implementation of 2D complex data loading "
         "and potentially different CQTPlan API for precomputed filters.";
  /*
  std::string test_case_name = "Precomputed_Execute";
  auto input_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_input_cd.txt");
  auto expected_cqt_d = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_expected_cd.txt"); // Assuming 1D output
  for now

  // Placeholder: Load filter bank data (requires loadComplexMatrixData)
  // auto filter_bank_d =
  TestDataLoader::loadComplexMatrixData<double>(suite_name,
  "FilterBankFile.txt");

  // Placeholder: Create CQT plan using precomputed filters (requires
  appropriate constructor/method)
  // OmniDSP::CQTPlan<double> plan(sample_rate, n_bins, bins_per_octave, fmin,
  filter_bank_d); // Example signature

  // Prepare output vector
  std::vector<std::complex<double>> output_cqt_d(expected_cqt_d.size());

  // Execute CQT
  // plan.execute(input_cd, output_cqt_d); // Call the appropriate execute
  method

  // Compare results
  // ExpectComplexVectorNear(output_cqt_d, expected_cqt_d, abs_error_d,
  test_case_name);
  */
}

// TODO: Add tests for calculateSingleOctaveCQT if possible/needed
// TODO: Add tests for different CQT parameters (sparsity threshold, etc.)
