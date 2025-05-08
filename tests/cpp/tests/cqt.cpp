/**
 * @file tests/cpp/tests/cqt.cpp
 * @brief Unit tests for the OmniDSP CQTPlan class and related functions.
 * Uses TestDataLoader for loading reference data and tests the public API.
 */

#include <OmniDSP/core_types.hpp>  // Include for Precision enum
#include <OmniDSP/cqt.hpp>         // Include the OmniDSP CQT header
#include <OmniDSP/fft.hpp>         // Include for FFTNorm enum
#include <OmniDSP/resample.hpp>    // Include for filter_and_downsample
// #include <OmniDSP/window.h>
#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <limits>
#include <memory>  // For std::unique_ptr
#include <numbers>
#include <numeric>
#include <stdexcept>  // For std::invalid_argument, std::runtime_error
#include <string>
#include <vector>

#include "../TestDataLoader.hpp"  // Include the new data loader utility

// Define aliases for vector/matrix types
using VecD = std::vector<double>;
using VecF = std::vector<float>;
using ComplexVecD = std::vector<std::complex<double>>;
using ComplexVecF = std::vector<std::complex<float>>;
using ComplexMatD = std::vector<ComplexVecD>;  // Matrix type [bin][frame]
using ComplexMatF = std::vector<ComplexVecF>;  // Matrix type [bin][frame]

// Default FIR order used in reference data generation script
const int DEFAULT_FIR_FILTER_ORDER_TEST = 101;

// --- Test Helper Functions ---

// Helper function to compare complex matrices
template <typename T>
void ExpectComplexMatrixNear(
    const std::vector<std::vector<std::complex<T>>> &actual,
    const std::vector<std::vector<std::complex<T>>> &expected,
    T abs_error,
    const std::string &message = "")
{
  ASSERT_EQ(actual.size(), expected.size())
      << message
      << " - Matrix row counts (bins) differ. Expected: " << expected.size()
      << ", Got: " << actual.size();
  if (expected.empty()) return;  // Both empty is OK

  ASSERT_GT(expected.size(), 0) << message << " - Expected matrix has no rows.";
  size_t num_rows = expected.size();
  size_t num_cols = expected[0].size();  // Assume consistent cols in expected

  ASSERT_FALSE(actual.empty())
      << message << " - Actual matrix is empty while expected has " << num_rows
      << " bins.";
  ASSERT_EQ(actual[0].size(), num_cols)
      << message
      << " - Matrix column count (frames) differs for bin 0. Expected: "
      << num_cols << ", Got: " << actual[0].size();

  for (size_t i = 0; i < num_rows; ++i) {
    ASSERT_EQ(expected[i].size(), num_cols)
        << message << " - Expected matrix has inconsistent columns at bin "
        << i;
    ASSERT_EQ(actual[i].size(), num_cols)
        << message << " - Actual matrix has inconsistent columns at bin " << i;
    for (size_t j = 0; j < num_cols; ++j) {
      EXPECT_NEAR(actual[i][j].real(), expected[i][j].real(), abs_error)
          << message << " - Mismatch at bin " << i << ", frame " << j
          << " (real part)";
      EXPECT_NEAR(actual[i][j].imag(), expected[i][j].imag(), abs_error)
          << message << " - Mismatch at bin " << i << ", frame " << j
          << " (imaginary part)";
    }
  }
}

// Helper function to compare real vectors
template <typename T>
void ExpectRealVectorNear(
    const std::vector<T> &actual,
    const std::vector<T> &expected,
    T abs_error,
    const std::string &message = "")
{
  ASSERT_EQ(actual.size(), expected.size())
      << message << " - Vector sizes differ. Expected: " << expected.size()
      << ", Got: " << actual.size();
  for (size_t i = 0; i < actual.size(); ++i) {
    EXPECT_NEAR(actual[i], expected[i], abs_error)
        << message << " - Mismatch at index " << i;
  }
}

// Helper to generate a simple Hann window (L1 normalized is often used
// internally)
template <typename T>
std::vector<T> hannWindowGenerator(size_t length)
{
  if (length == 0) return {};
  std::vector<T> window_coeffs(length);
  double denom = (length > 1) ? static_cast<double>(length - 1) : 1.0;
  T sum = 0;
  for (size_t n = 0; n < length; ++n) {
    T val = static_cast<T>(
        0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * n / denom));
    window_coeffs[n] = val;
    sum += val;
  }
  // Normalize (optional, depends if CQTPlan expects normalized input)
  // CQTPlan usually normalizes internally, so we return unnormalized coeffs
  // here. if (sum > std::numeric_limits<T>::epsilon()) {
  //     for (size_t n = 0; n < length; ++n) {
  //         window_coeffs[n] /= sum;
  //     }
  // }
  return window_coeffs;
}

// Helper to generate a sine wave
template <typename T>
std::vector<T> generateSineWave(double freq, double duration_sec, double sr)
{
  size_t length = static_cast<size_t>(duration_sec * sr);
  if (length == 0) return {};
  std::vector<T> signal(length);
  for (size_t i = 0; i < length; ++i) {
    signal[i] = static_cast<T>(
        std::sin(2.0 * std::numbers::pi * freq * static_cast<double>(i) / sr));
  }
  return signal;
}

// --- Test Fixture ---
class CQT_Test : public ::testing::Test {
 protected:
  // Common parameters matching reference data generation script
  const double sample_rate = 22050.0;
  const double fmin = 32.7;  // C1
  const double high_freq
      = 1760.0;           // A6 (Example, ensure matches generation if used)
  const int n_bins = 84;  // Example, ensure matches generation
  const int bins_per_octave = 12;
  const size_t hop_length = 512;  // Must be compatible with CQT implementation
  const float sparsity_threshold_float = 1e-5f;
  const double sparsity_threshold_double = 1e-5;

  // Tolerances (may need adjustment based on backend/librosa differences)
  const double abs_error_d = 1e-5;  // Increased tolerance for CQT
  const float abs_error_f = 1e-3f;  // Increased tolerance for CQT

  const std::string suite_name = "cqt";  // Corresponds to data/cqt/ directory

  // Helper to get tolerance
  template <typename T>
  T get_tolerance()
  {
    return std::is_same_v<T, float> ? abs_error_f : abs_error_d;
  }
};

// --- CQTPlan Tests ---

// Test CQTPlan constructor parameter validation
TEST_F(CQT_Test, ConstructorParameterValidation)
{
  using T = double;
  auto window_gen = hannWindowGenerator<T>;  // Use helper

  // Valid case
  ASSERT_NO_THROW(OmniDSP::CQTPlan<T>(
      sample_rate,
      hop_length,
      fmin,
      high_freq,
      bins_per_octave,
      window_gen,
      sparsity_threshold_double,
      DEFAULT_FIR_FILTER_ORDER_TEST));

  // Invalid parameters
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          0.0,
          hop_length,
          fmin,
          high_freq,
          bins_per_octave,
          window_gen,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);  // Invalid SR
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          0,
          fmin,
          high_freq,
          bins_per_octave,
          window_gen,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);  // Invalid hop
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          hop_length,
          0.0,
          high_freq,
          bins_per_octave,
          window_gen,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);  // Invalid fmin
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          hop_length,
          fmin,
          fmin,
          bins_per_octave,
          window_gen,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);  // fmax <= fmin
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          hop_length,
          fmin,
          sample_rate,
          bins_per_octave,
          window_gen,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);  // fmax >= Nyquist
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          hop_length,
          fmin,
          high_freq,
          0,
          window_gen,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);  // Invalid BPO
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          hop_length,
          fmin,
          high_freq,
          bins_per_octave,
          nullptr,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);  // Null window gen
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          511,
          fmin,
          high_freq,
          bins_per_octave,
          window_gen,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);  // Invalid hop length for typical recursive CQT
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          hop_length,
          fmin,
          high_freq,
          bins_per_octave,
          window_gen,
          -0.1,
          DEFAULT_FIR_FILTER_ORDER_TEST),
      std::invalid_argument);  // Invalid sparsity
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          hop_length,
          fmin,
          high_freq,
          bins_per_octave,
          window_gen,
          sparsity_threshold_double,
          0),
      std::invalid_argument);  // Invalid FIR order (zero)
  ASSERT_THROW(
      OmniDSP::CQTPlan<T>(
          sample_rate,
          hop_length,
          fmin,
          high_freq,
          bins_per_octave,
          window_gen,
          sparsity_threshold_double,
          100),
      std::invalid_argument);  // Invalid FIR order (even)
}

// Test CQTPlan::execute method against reference data
TEST_F(CQT_Test, ExecuteCQT_Double)
{
  using T = double;
  std::string test_case_name = "FullRecursiveCQT_Execute_Double";
  std::vector<T> input_d;
  ComplexMatD expected_cqt_d;  // Expecting 2D complex matrix

  // Load input signal (1D real) and expected output (2D complex)
  ASSERT_NO_THROW(
      input_d = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_input_d.txt"))
      << "Failed to load input data";
  ASSERT_NO_THROW(
      expected_cqt_d = TestDataLoader::loadComplexMatrixData<T>(
          suite_name, test_case_name + "_expected_cd.txt"))
      << "Failed to load expected data";

  ASSERT_FALSE(input_d.empty()) << "Input data vector is empty";
  ASSERT_FALSE(expected_cqt_d.empty()) << "Expected data matrix is empty";

  // Create CQT plan
  std::unique_ptr<OmniDSP::CQTPlan<T>> plan_ptr;
  ASSERT_NO_THROW(
      plan_ptr = std::make_unique<OmniDSP::CQTPlan<T>>(
          sample_rate,
          hop_length,
          fmin,
          high_freq,
          bins_per_octave,
          hannWindowGenerator<T>,
          sparsity_threshold_double,
          DEFAULT_FIR_FILTER_ORDER_TEST))
      << "CQTPlan constructor threw unexpectedly.";
  ASSERT_NE(plan_ptr, nullptr);
  OmniDSP::CQTPlan<T> &plan = *plan_ptr;

  // Prepare output matrix
  ComplexMatD output_cqt_d;

  // Execute CQT
  ASSERT_NO_THROW(plan.execute(input_d, output_cqt_d))
      << "CQTPlan::execute threw unexpectedly.";

  // Compare results
  ExpectComplexMatrixNear(
      output_cqt_d, expected_cqt_d, get_tolerance<T>(), test_case_name);
}

TEST_F(CQT_Test, ExecuteCQT_Float)
{
  using T = float;
  std::string test_case_name = "FullRecursiveCQT_Execute_Float";
  std::vector<T> input_f;
  ComplexMatF expected_cqt_f;  // Expecting 2D complex matrix

  // Load input signal (1D real) and expected output (2D complex)
  ASSERT_NO_THROW(
      input_f = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_input_f.txt"))
      << "Failed to load input data";
  ASSERT_NO_THROW(
      expected_cqt_f = TestDataLoader::loadComplexMatrixData<T>(
          suite_name, test_case_name + "_expected_cf.txt"))
      << "Failed to load expected data";

  ASSERT_FALSE(input_f.empty()) << "Input data vector is empty";
  ASSERT_FALSE(expected_cqt_f.empty()) << "Expected data matrix is empty";

  // Create CQT plan
  std::unique_ptr<OmniDSP::CQTPlan<T>> plan_ptr;
  ASSERT_NO_THROW(
      plan_ptr = std::make_unique<OmniDSP::CQTPlan<T>>(
          sample_rate,
          hop_length,
          fmin,
          high_freq,
          bins_per_octave,
          hannWindowGenerator<T>,
          sparsity_threshold_float,
          DEFAULT_FIR_FILTER_ORDER_TEST))
      << "CQTPlan constructor threw unexpectedly.";
  ASSERT_NE(plan_ptr, nullptr);
  OmniDSP::CQTPlan<T> &plan = *plan_ptr;

  // Prepare output matrix
  ComplexMatF output_cqt_f;

  // Execute CQT
  ASSERT_NO_THROW(plan.execute(input_f, output_cqt_f))
      << "CQTPlan::execute threw unexpectedly.";

  // Compare results
  ExpectComplexMatrixNear(
      output_cqt_f, expected_cqt_f, get_tolerance<T>(), test_case_name);
}

// --- Filter And Downsample Test ---
// Test the public API function directly

class FilterAndDownsampleTest : public ::testing::Test {
 protected:
  const std::string suite_name = "cqt";  // Data is under cqt suite for now
  const double abs_error_d = 1e-9;
  const float abs_error_f = 1e-5f;
};

TEST_F(FilterAndDownsampleTest, PublicAPI_Float)
{
  using T = float;
  std::string test_case_name = "FilterAndDownsample";
  std::vector<T> input_f, filter_coeffs_f, expected_f;

  ASSERT_NO_THROW(
      input_f = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_input_f.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      filter_coeffs_f = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_filter_coeffs_f.txt"))
      << "Failed to load filter coeffs";
  ASSERT_NO_THROW(
      expected_f = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_expected_f.txt"))
      << "Failed to load expected output";

  ASSERT_FALSE(input_f.empty());
  ASSERT_FALSE(filter_coeffs_f.empty());
  ASSERT_FALSE(expected_f.empty());

  std::vector<T> output_f;
  int downsample_factor = 2;

  // Call the public API function
  ASSERT_NO_THROW(
      output_f = OmniDSP::filter_and_downsample<T>(
          input_f, filter_coeffs_f, downsample_factor))
      << "Public filter_and_downsample<float> threw unexpectedly.";

  // Compare results
  ExpectRealVectorNear(
      output_f, expected_f, abs_error_f, test_case_name + "_Float");
}

TEST_F(FilterAndDownsampleTest, PublicAPI_Double)
{
  using T = double;
  std::string test_case_name = "FilterAndDownsample";
  std::vector<T> input_d, filter_coeffs_d, expected_d;

  ASSERT_NO_THROW(
      input_d = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_input_d.txt"))
      << "Failed to load input";
  ASSERT_NO_THROW(
      filter_coeffs_d = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_filter_coeffs_d.txt"))
      << "Failed to load filter coeffs";
  ASSERT_NO_THROW(
      expected_d = TestDataLoader::loadVectorData<T>(
          suite_name, test_case_name + "_expected_d.txt"))
      << "Failed to load expected output";

  ASSERT_FALSE(input_d.empty());
  ASSERT_FALSE(filter_coeffs_d.empty());
  ASSERT_FALSE(expected_d.empty());

  std::vector<T> output_d;
  int downsample_factor = 2;

// Call the public API function
// Depending on the backend, this might throw if double is unsupported
#if defined(USE_ONEMKL)
  // MKL backend currently throws for double precision resampling
  ASSERT_THROW(
      OmniDSP::filter_and_downsample<T>(
          input_d, filter_coeffs_d, downsample_factor),
      std::runtime_error);
#else
  // Accelerate or Default backends might support double or throw differently
  ASSERT_NO_THROW(
      output_d = OmniDSP::filter_and_downsample<T>(
          input_d, filter_coeffs_d, downsample_factor))
      << "Public filter_and_downsample<double> threw unexpectedly.";
  // Compare results only if it didn't throw
  ExpectRealVectorNear(
      output_d, expected_d, abs_error_d, test_case_name + "_Double");
#endif
}
