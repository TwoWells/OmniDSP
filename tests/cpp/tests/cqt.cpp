#include "OmniDSP/cqt.h"  // Include the OmniDSP CQT header

#include <cmath>    // For std::abs
#include <complex>  // For std::complex
#include <limits>   // For std::numeric_limits
#include <numeric>  // For std::iota
#include <string>   // For std::string
#include <vector>   // For std::vector

#include "../TestDataLoader.h"  // Include the new data loader utility
#include "OmniDSP/resample.h"   // For filterAndDownsample used in one test
#include "gtest/gtest.h"        // Google Test framework

// Helper function to compare complex vectors with tolerance
template <typename T>
void ASSERT_VECTORS_NEAR(const std::vector<std::complex<T>> &actual,
                         const std::vector<std::complex<T>> &expected,
                         T abs_error, const std::string &message = "") {
  ASSERT_EQ(actual.size(), expected.size())
      << message << " - Vector sizes differ.";
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_NEAR(actual[i].real(), expected[i].real(), abs_error)
        << message << " - Mismatch at index " << i << " (real part)";
    ASSERT_NEAR(actual[i].imag(), expected[i].imag(), abs_error)
        << message << " - Mismatch at index " << i << " (imaginary part)";
  }
}

// Helper function to compare real vectors with tolerance
template <typename T>
void ASSERT_REAL_VECTORS_NEAR(const std::vector<T> &actual,
                              const std::vector<T> &expected, T abs_error,
                              const std::string &message = "") {
  ASSERT_EQ(actual.size(), expected.size())
      << message << " - Vector sizes differ.";
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_NEAR(actual[i], expected[i], abs_error)
        << message << " - Mismatch at index " << i;
  }
}

// Helper function to compare complex matrices (vector of vectors)
template <typename T>
void ASSERT_MATRICES_NEAR(
    const std::vector<std::vector<std::complex<T>>> &actual,
    const std::vector<std::vector<std::complex<T>>> &expected, T abs_error,
    const std::string &message = "") {
  ASSERT_EQ(actual.size(), expected.size())
      << message << " - Matrix row counts differ.";
  for (size_t i = 0; i < actual.size(); ++i) {
    ASSERT_EQ(actual[i].size(), expected[i].size())
        << message << " - Matrix column counts differ at row " << i;
    for (size_t j = 0; j < actual[i].size(); ++j) {
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
  // Constants matching reference data generation
  const double sample_rate = 22050.0;
  const double fmin = 32.7;  // C1
  const int n_bins = 84;
  const int bins_per_octave = 12;
  const int n_octaves = 7;  // n_bins / bins_per_octave

  // Tolerances
  const double abs_error_d = 1e-7;  // CQT might need slightly larger tolerance
  const float abs_error_f = 1e-4f;

  const std::string suite_name = "cqt";  // Corresponds to data/cqt/ directory
};

// --- Precomputed CQT Tests ---
// Tests using the precomputed filter bank approach

class PrecomputedCQTTest : public CQT_Test {
 protected:
  // Load precomputed filter bank (needs 2D complex data loader)
  // TODO: Implement TestDataLoader::loadComplexMatrixData
  /*
  std::vector<std::vector<std::complex<double>>> filter_bank_d;

  void SetUp() override {
      std::string filename = "Precomputed_FilterBank_expected_cd.txt"; //
  Example filename try {
           // Placeholder for loading 2D complex data
           filter_bank_d =
  TestDataLoader::loadComplexMatrixData<double>(suite_name, filename); } catch
  (const std::exception& e) { GTEST_SKIP() << "Skipping test, failed to load
  required data: " << filename << " (" << e.what() << "). Implement
  TestDataLoader::loadComplexMatrixData.";
      }
       ASSERT_FALSE(filter_bank_d.empty()) << "Filter bank data is empty.";
  }
  */
};

// This test is currently disabled because it relies on the 2D filter bank
// loading
TEST_F(PrecomputedCQTTest, DISABLED_ExecutePrecomputed) {
  GTEST_SKIP() << "Test disabled until TestDataLoader::loadComplexMatrixData "
                  "is implemented.";
  /*
  std::string test_case_name = "Precomputed_Execute";
  auto input_cd = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_input_cd.txt");
  auto expected_cqt_d = TestDataLoader::loadVectorData<std::complex<double>>(
      suite_name, test_case_name + "_expected_cd.txt"); // Assuming 1D output
  for now

  // Create CQT plan using precomputed filters
  OmniDSP::CQTPlan<double> plan(sample_rate, n_bins, bins_per_octave, fmin);
  plan.setFilterBank(filter_bank_d); // Provide the loaded filter bank

  // Prepare output vector
  std::vector<std::complex<double>> output_cqt_d(n_bins); // Adjust size if
  necessary

  // Execute CQT
  plan.execute(input_cd, output_cqt_d);

  // Compare results
  ASSERT_VECTORS_NEAR(output_cqt_d, expected_cqt_d, abs_error_d,
  test_case_name);
  */
}

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

  // Create CQT plan (will calculate filters recursively)
  OmniDSP::CQTPlan<double> plan(sample_rate, n_bins, bins_per_octave, fmin);

  // Prepare output vector
  // Size needs to match the expected output format from reference generation
  std::vector<std::complex<double>> output_cqt_d(expected_cqt_d.size());

  // Execute CQT
  plan.execute(input_cd, output_cqt_d);

  // Compare results
  ASSERT_VECTORS_NEAR(output_cqt_d, expected_cqt_d, abs_error_d,
                      test_case_name);
}

TEST_F(RecursiveCQTTest, FullRecursiveCQT_Execute_Float) {
  std::string test_case_name = "FullRecursiveCQT_Execute_Float";
  auto input_cf = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_input_cf.txt");
  auto expected_cqt_f = TestDataLoader::loadVectorData<std::complex<float>>(
      suite_name, test_case_name + "_expected_cf.txt");

  OmniDSP::CQTPlan<float> plan(static_cast<float>(sample_rate), n_bins,
                               bins_per_octave, static_cast<float>(fmin));
  std::vector<std::complex<float>> output_cqt_f(expected_cqt_f.size());
  plan.execute(input_cf, output_cqt_f);

  ASSERT_VECTORS_NEAR(output_cqt_f, expected_cqt_f, abs_error_f,
                      test_case_name);
}

// --- Filter And Downsample Test ---
// This test focuses on the internal resampling helper function used by CQT

class FilterAndDownsampleTest : public CQT_Test {
  // Test fixture specific for filterAndDownsample
};

TEST_F(FilterAndDownsampleTest, FilterAndDownsample) {
  std::string test_case_name = "FilterAndDownsample";
  // Load data (assuming it's stored under 'cqt' suite for now, adjust if
  // needed)
  auto input_d = TestDataLoader::loadVectorData<double>(
      suite_name, test_case_name + "_input_d.txt");
  auto filter_coeffs_d = TestDataLoader::loadVectorData<double>(
      suite_name, test_case_name + "_filter_coeffs_d.txt");
  auto expected_d = TestDataLoader::loadVectorData<double>(
      suite_name, test_case_name + "_expected_d.txt");

  // Prepare output vector (size should match expected)
  std::vector<double> output_d(expected_d.size());

  // Call the function (assuming it's accessible, might need friend class or
  // public helper) If it's private within CQTPlan, this test needs rethinking
  // or CQTPlan instantiation. For now, assume a standalone or accessible
  // version exists. Let's use the public API version from resample.h for
  // demonstration:
  try {
    OmniDSP::filterAndDownsampleBy2(input_d, output_d, filter_coeffs_d);
  } catch (const std::exception &e) {
    FAIL() << "filterAndDownsampleBy2 threw an exception: " << e.what();
  }

  // Compare results
  ASSERT_REAL_VECTORS_NEAR(output_d, expected_d, abs_error_d, test_case_name);

  // TODO: Add float version of this test if needed
}

// TODO: Add tests for calculateSingleOctaveCQT if possible/needed
// TODO: Add tests for different CQT parameters (sparsity threshold, etc.)
